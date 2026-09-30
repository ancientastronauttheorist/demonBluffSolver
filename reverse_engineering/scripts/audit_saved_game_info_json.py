"""Join native JSON fields over the pinned SavedGameInfo field inventory."""
import argparse
import hashlib
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_list_classifier import Machine as ClassifierMachine
from audit_unity_json_list_classifier import verify_native as verify_classifier
from audit_unity_json_lists import LIST_SOURCE
from audit_unity_json_strings import STRING_SOURCE
from audit_unityplayer_wait import ENGINE_SHA256


FIELDS = [('key', 'string', 0x10), ('completedTutorials', 'List<string>', 0x18),
          ('unlockedCharactersId', 'List<string>', 0x20)]


def verify_layout(game_root, dumper_root):
    root = Path(__file__).parents[1]
    manifest = json.loads((root / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((root / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    pinned = {}
    for name in ['game_assembly', 'global_metadata']:
        entry = manifest['inputs'][name]
        raw = (Path(game_root) / entry['path']).read_bytes()
        digest = hashlib.sha256(raw).hexdigest().upper()
        assert digest == entry['sha256'].upper()
        pinned[name] = digest
    raw = (Path(dumper_root) / 'dump.cs').read_bytes()
    digest = hashlib.sha256(raw).hexdigest().upper()
    assert digest == extraction['outputs']['dump_cs']['sha256'].upper()
    text = raw.decode('utf-8-sig')
    pattern = r'^public class SavedGameInfo // TypeDefIndex: 5944\r?\n\{'
    matches = list(re.finditer(pattern, text, flags=re.MULTILINE))
    assert len(matches) == 1
    block = text[matches[0].end():].split('// Methods', 1)[0]
    fields = [(name, typ, int(offset, 16)) for typ, name, offset in
              re.findall(r'public (string|List<string>) (\w+); // (0x[0-9A-Fa-f]+)', block)]
    assert fields == FIELDS
    return {'class': 'SavedGameInfo', 'type_def_index': 5944,
            'fields': [{'name': n, 'type': t, 'offset': hex(o)} for n, t, o in fields],
            'dump_cs_sha256': digest, 'pinned_inputs': pinned,
            'runtime_metadata': 'Authored services use these exact public fields; actual runtime class discovery is not supplied by Dumper.'}


class Machine(ClassifierMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.select_element(STRING_SOURCE, 8, 14)

    def saved_state(self):
        result, pointers = {}, []
        for name, typ, offset in FIELDS:
            token = self.rq(self.managed + offset)
            if token == 0:
                result[name] = None
            elif typ == 'string':
                result[name] = self.string_text(token) if token in self.strings else 'unset'
            elif token in self.lists:
                state = self.list_state(token)
                values = state['values_hex']
                result[name] = None if values is None else [self.string_text(int.from_bytes(bytes.fromhex(v), 'little')) for v in values]
            else:
                result[name] = 'unset'
            pointers.append(token)
        return {'values': result, 'lists_aliased': bool(pointers[1]) and pointers[1] == pointers[2]}

    def metadata_snapshot(self):
        result = super().metadata_snapshot()
        result['saved_game_info'] = self.saved_state()
        return result

    def prepare(self, state, alias=False):
        self.reset_strings()
        self.reset_arrays()
        self.reset_lists()
        definitions, values = [], []
        shared = None
        for name, typ, offset in FIELDS:
            value = state.get(name)
            if typ == 'string':
                token = self.make_string(value) if value is not None else 0
                source, enum = STRING_SOURCE, 14
            else:
                if alias and name == 'unlockedCharactersId':
                    assert value == state.get('completedTutorials')
                    token = shared
                elif value is None:
                    token = 0
                else:
                    raw = [(self.make_string(v) if v is not None else 0).to_bytes(8, 'little') for v in value]
                    token = self.make_list(raw, len(raw) + 2)
                shared = token
                source, enum = LIST_SOURCE, 0x15
            raw = token.to_bytes(8, 'little')
            definitions.append({'name': name, 'source': source, 'type_enum': enum,
                                'offset': offset, 'initial_hex': raw.hex()})
            values.append(raw)
        return definitions, values

    def read_saved(self, payload, old=None, options=None):
        definitions, _ = self.prepare(old or {}, (options or {}).get('alias', False))
        self.registry_build(False)
        result = self.run(definitions, dict(options or {}, joined=True, json=payload))
        result.update({'old_values': old or {}, 'loaded': self.saved_state()})
        return result

    def write_saved(self, state, options=None):
        definitions, values = self.prepare(state, (options or {}).get('alias', False))
        # Deep input storage is checked before constructing fresh reload fixtures.
        storage = [(a, bytes(self.u.mem_read(a, 0x40))) for a in self.lists]
        storage += [(a, bytes(self.u.mem_read(a, 0x40 + f['count'] * f['width']))) for a, f in self.arrays.items()]
        storage += [(f['chars'], bytes(self.u.mem_read(f['chars'], f['length'] * 2 + 2))) for f in self.strings.values()]
        result = self.serialize(definitions, values, options)
        assert all(bytes(self.u.mem_read(a, len(raw))) == raw for a, raw in storage)
        result.update({'input_values': state, 'deep_input_storage_retained': True})
        if result['returned']:
            rendered = bytes.fromhex(result['json_utf8_hex']).decode('utf-8')
            read = self.read_saved(rendered)
            assert read['returned'] and not read['final']['scope_linked']
            result['loaded'] = read['loaded']
        return result


def audit(game_root, dumper_root):
    layout = verify_layout(game_root, dumper_root)
    m = Machine(game_root)
    verified = verify_classifier(m)
    writes, reads, aliases, failures, baselines = [], [], [], [], {}
    states = [{}, {'key': 'offline-fixture'},
              {'key': 'offline-fixture', 'completedTutorials': [], 'unlockedCharactersId': []},
              {'key': 'offline-fixture', 'completedTutorials': ['tutorial_a', 'tutorial_b'],
               'unlockedCharactersId': ['character_a', 'character_a', 'character_b']},
              {'key': 'key\0suffix', 'completedTutorials': ['caf\u00e9 \U0001f608', None, 'a\0b'],
               'unlockedCharactersId': ['\u6f22\u5b57']}]
    for state in states:
        expected = {'key': (state.get('key') or '').split('\0', 1)[0]}
        expected.update({name: [(v or '').split('\0', 1)[0] for v in state.get(name) or []] for name, _, _ in FIELDS[1:]})
        for pretty in [False, True]:
            result = m.write_saved(state, {'pretty': pretty})
            assert result['returned'] and result['loaded']['values'] == expected
            assert not result['loaded']['lists_aliased']
            writes.append(result)
    old = {'key': 'old-key', 'completedTutorials': ['old-tutorial'],
           'unlockedCharactersId': ['old-character']}
    payloads = [('{}', old), ('{"key":"new-key"}', dict(old, key='new-key')),
                ('{"completedTutorials":["new-tutorial"]}', dict(old, completedTutorials=['new-tutorial'])),
                ('{"unlockedCharactersId":null}', dict(old, unlockedCharactersId=[])),
                ('{"key":null,"completedTutorials":null,"unlockedCharactersId":null}',
                 {'key': '', 'completedTutorials': [], 'unlockedCharactersId': []}),
                ('{"Key":"ignored","unknown":123}', old),
                ('{"key":"first","key":"second"}', dict(old, key='first')),
                ('{"completedTutorials":[null,true,12,1.5]}', dict(old, completedTutorials=['', 'true', '12', '1.500000']))]
    for payload, expected in payloads:
        result = m.read_saved(payload, old)
        assert result['returned'] and result['loaded']['values'] == expected
        assert not result['final']['scope_linked']
        reads.append(result)
    result = m.read_saved('{}')
    assert result['returned'] and result['loaded']['values'] == {'key': None, 'completedTutorials': [], 'unlockedCharactersId': []}
    reads.append(result)
    state = {'key': 'alias-fixture', 'completedTutorials': ['shared'], 'unlockedCharactersId': ['shared']}
    for pretty in [False, True]:
        result = m.write_saved(state, {'alias': True, 'pretty': pretty})
        assert result['returned'] and result['loaded']['values'] == state
        assert not result['loaded']['lists_aliased']
        result['input_lists_aliased'] = True
        aliases.append(result)
    for direction in ['read_new', 'read_existing', 'write']:
        operation = ((lambda options: m.write_saved(old, options)) if direction == 'write' else
                     (lambda options: m.read_saved('{"key":"new","completedTutorials":["new-t"],"unlockedCharactersId":["new-c"]}',
                                                   old if direction == 'read_existing' else None, options)))
        baseline = operation({})
        assert baseline['returned']
        baselines[direction] = baseline
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = operation({'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['error'] == kind
            assert result['events'] == baseline['events'][:index + 1]
            if direction != 'write':
                assert result['final'] == event['snapshot']
            failure = {'direction': direction, 'failure': [kind, counts[kind]],
                       'baseline_event_count': index + 1, 'returned': False, 'error': result['error']}
            if direction != 'write':
                failure.update({'final': result['final'], 'loaded': result['loaded']})
            failures.append(failure)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256, 'pinned_layout': layout,
            'classifier_verified': verified, 'write_cases': writes, 'write_case_count': len(writes),
            'read_cases': reads, 'read_case_count': len(reads), 'alias_cases': aliases, 'alias_case_count': len(aliases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'scope': 'All three pinned SavedGameInfo public fields composed through actual engine metadata/factory/classifier/string/List writing and reload. Runtime metadata/class discovery, managed allocation/constructors and GC/cache/array/allocator services remain supplied. GameData persistence callers, PlayerPrefs, SavedGameInfo mutation methods and actual constructor defaults are not executed by this audit.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['write_case_count'], result['read_case_count'], result['alias_case_count'], result['failure_case_count'])
