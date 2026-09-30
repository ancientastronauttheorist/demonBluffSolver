"""Execute pinned SavedGameData callers with explicit preference/JSON services."""
import argparse
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_info_methods import Machine as InfoMachine
from audit_saved_game_info_json import Machine as JsonMachine


METHODS = {'Load': 0x3EA940, 'Save': 0x3EAAA0, 'ResetTutorials': 0x3EAA10, '.ctor': 0x3EAAE0}
FIELDS = [('completedTutorials', 0x18), ('unlockedCharactersId', 0x20)]


class Machine(InfoMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        self.data = self.arena + 0x2000
        self.info_flags = self.flags.copy()
        self.data_ranges, self.data_targets = {}, []
        script = json.loads((Path(dumper_root) / 'script.json').read_text(encoding='utf-8-sig'))
        dump = (Path(dumper_root) / 'dump.cs').read_text(encoding='utf-8')
        block = re.search(r'^public class SavedGameData : ScriptableObject // TypeDefIndex: 5943\s*\{(.*?)^\}', dump, re.M | re.S)
        assert block and re.findall(r'^\s*public ([^\n]+); // (0x[0-9A-Fa-f]+)', block[1], re.M) == [('SavedGameInfo save', '0x18')]
        self.service_targets = []
        services = {
            0xF76390: ('System.String$$IsNullOrEmpty', 'bool System_String__IsNullOrEmpty (System_String_o* value, const MethodInfo* method);'),
            0x1C85F20: ('UnityEngine.PlayerPrefs$$GetString', 'System_String_o* UnityEngine_PlayerPrefs__GetString (System_String_o* key, const MethodInfo* method);'),
            0x1C86170: ('UnityEngine.PlayerPrefs$$SetString', 'void UnityEngine_PlayerPrefs__SetString (System_String_o* key, System_String_o* value, const MethodInfo* method);'),
            0x1CD6420: ('UnityEngine.JsonUtility$$ToJson', 'System_String_o* UnityEngine_JsonUtility__ToJson (Il2CppObject* obj, const MethodInfo* method);'),
            0x645DA0: ('UnityEngine.JsonUtility$$FromJson<object>', 'Il2CppObject* UnityEngine_JsonUtility__FromJson_object_ (System_String_o* json, const MethodInfo_645DA0* method);'),
            0x1C8A5C0: ('UnityEngine.ScriptableObject$$.ctor', 'void UnityEngine_ScriptableObject___ctor (UnityEngine_ScriptableObject_o* __this, const MethodInfo* method);'),
        }
        for rva, (name, signature) in services.items():
            rows = [r for r in script['ScriptMethod'] if r['Address'] == rva and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['Signature'] == signature
            self.service_targets.append(rows[0])
        for name, rva in METHODS.items():
            rows = [r for r in script['ScriptMethod'] if r['Name'] == 'SavedGameData$$' + name and r['Address'] == rva]
            assert len(rows) == 1
            expected = ('void SavedGameData___ctor' if name == '.ctor' else f'void SavedGameData__{name}')
            assert rows[0]['Signature'] == expected + ' (SavedGameData_o* __this, const MethodInfo* method);'
            self.data_targets.append(rows[0])
            entries = [e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if rva <= e.struct.BeginAddress < (0x3EAA10 if name == 'Load' else rva + 1)]
            assert entries and entries[0].struct.BeginAddress == rva
            for index, entry in enumerate(entries):
                assert bool(entry.unwindinfo.Flags & 4) == bool(index)
                a, b = entry.struct.BeginAddress, entry.struct.EndAddress
                decoded = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
                assert sum(i.size for i in decoded) == b - a
                self.instructions.update({i.address: i for i in decoded})
            self.data_ranges[name] = [[hex(e.struct.BeginAddress), hex(e.struct.EndAddress)] for e in entries]
            first = next((i for i in self.instructions.values() if rva <= i.address < entries[0].struct.EndAddress and i.mnemonic == 'cmp' and i.operands[0].type == self.capstone.x86.X86_OP_MEM and i.operands[0].mem.base == self.capstone.x86.X86_REG_RIP), None)
            if first:
                self.flags['SavedGameData.' + name] = first.address + first.size + first.operands[0].mem.disp
        slots = {r['Address']: ('metadata', r['Name']) for r in script['ScriptMetadata']}
        slots.update({r['Address']: ('method', r['Name']) for r in script['ScriptMetadataMethod']})
        self.data_bindings = {}
        for name, ranges in self.data_ranges.items():
            for a, b in ranges:
                for i in self.instructions.values():
                    if int(a, 16) <= i.address < int(b, 16):
                        for op in i.operands:
                            if op.type == self.capstone.x86.X86_OP_MEM and op.mem.base == self.capstone.x86.X86_REG_RIP:
                                slot = i.address + i.size + op.mem.disp
                                if slot in slots:
                                    self.data_bindings[slot] = slots[slot]
        assert len(self.data_bindings) == 3
        assert set(self.data_bindings.values()) == {('metadata', 'SavedGameInfo_TypeInfo'),
            ('method', 'Method$UnityEngine.JsonUtility.FromJson<SavedGameInfo>()'),
            ('method', 'Method$System.Collections.Generic.List<string>.Clear()')}
        self.bindings.update(self.data_bindings)
        self.saved_type = self.arena + 0x120000
        self.from_json_method = self.arena + 0x121000
        checks = {
            0x3EA984: ('call', '0x1c85f20'), 0x3EA98E: ('call', '0xf76390'),
            0x3EA9A3: ('call', '0x2b7d40'), 0x3EA9B0: ('call', '0x3eadd0'),
            0x3EA9B8: ('mov', 'qword ptr [rdi + 0x18], rbx'),
            0x3EA9DE: ('call', '0x1c85f20'), 0x3EA9ED: ('call', '0x645da0'),
            0x3EA9F5: ('mov', 'qword ptr [rdi + 0x18], rax'),
            0x3EAA4B: ('inc', 'dword ptr [rcx + 0x1c]'),
            0x3EAA4E: ('mov', 'dword ptr [rcx + 0x18], 0'),
            0x3EAA63: ('call', '0x112b9d0'), 0x3EAA6E: ('call', '0x1cd6420'),
            0x3EAA73: ('mov', 'rcx, qword ptr [rbx + 0x18]'),
            0x3EAA8B: ('jmp', '0x1c86170'), 0x3EAAAF: ('call', '0x1cd6420'),
            0x3EAAB4: ('mov', 'rcx, qword ptr [rbx + 0x18]'),
            0x3EAACC: ('jmp', '0x1c86170'), 0x3EAB1D: ('call', '0x3eadd0'),
            0x3EAB29: ('mov', 'qword ptr [rcx], rbx'),
            0x3EAB40: ('jmp', '0x1c8a5c0'),
        }
        for address, expected in checks.items():
            assert address in self.instructions
            i = self.instructions[address]
            assert (i.mnemonic, i.op_str) == expected
        null_empty = list(self.cs.disasm(self.pe.get_data(0xF76390, 17), 0xF76390))
        assert [(i.mnemonic, i.op_str) for i in null_empty] == [('test', 'rcx, rcx'), ('je', '0xf7639e'),
            ('cmp', 'dword ptr [rcx + 0x10], 0'), ('jbe', '0xf7639e'), ('xor', 'al, al'),
            ('ret', ''), ('mov', 'al, 1'), ('ret', '')]
        self.data_instruction_assertions = len(checks) + len(null_empty)
        self.engine = None

    def string(self, value):
        if value is None:
            return 0
        raw = value.encode('utf-16-le', errors='surrogatepass')
        token = self.string_cursor
        self.string_cursor += (0x20 + len(raw) + 15) & ~15
        assert self.string_cursor < self.arena + 0x20000
        self.strings[token] = value
        self.d(token + 0x10, len(raw) // 2)
        self.u.mem_write(token + 0x14, raw + b'\0\0')
        return token

    def new_info(self, values=None):
        token = self.saved + len(self.infos) * 0x80
        self.infos.append(token)
        self.u.mem_write(token, bytes(0x40))
        if values is not None:
            self.q(token + 0x10, self.string(values.get('key')))
            for name, offset in FIELDS:
                value = values.get(name)
                self.q(token + offset, self.list(value, version=0) if value is not None else 0)
        return token

    def info_state(self, token):
        if not token:
            return None
        assert token in self.infos
        return {'identity': self.infos.index(token), 'key': self.strings.get(self.rq(token + 0x10)),
                **{name: self.list_state(self.rq(token + offset)) for name, offset in FIELDS}}

    def values(self, token):
        state = self.info_state(token)
        if state is None:
            return None
        return {'key': state['key'], **{name: None if state[name] is None else state[name]['values'] for name, _ in FIELDS}}

    def snapshot(self):
        token = self.rq(self.data + 0x18)
        return {'save': self.info_state(token), 'allocated_infos': [self.info_state(p) for p in self.infos],
                'allocated_lists': [self.list_state(p) for p in self.lists], 'preference_writes': self.writes.copy(),
                'metadata_initialized': {name: bool(self.u.mem_read(self.base + rva, 1)[0]) for name, rva in self.flags.items()}}

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        cx, dx, r8 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        if rva == 0x2B7D40 and cx == self.saved_type:
            self.executed.add(rva)
            if self.event('saved_info_allocate_service', []):
                self.ret(self.new_info())
        elif rva == 0x1C85F20:
            self.executed.add(rva)
            assert dx == 0 and (not cx or cx in self.strings)
            index = self.counts.get('get_string_service', 0)
            result = self.options.get('get_values', ['', ''])[min(index, len(self.options.get('get_values', ['', ''])) - 1)]
            if self.event('get_string_service', [self.strings.get(cx), result]):
                if self.options.get('get_swap_at') == index + 1:
                    value = self.options.get('replacement')
                    self.q(self.data + 0x18, self.new_info(value) if value is not None else 0)
                self.ret(self.string(result))
        elif rva == 0x1CD6420:
            self.executed.add(rva)
            assert dx == 0 and (not cx or cx in self.infos)
            values = self.values(cx)
            if self.event('to_json_service', [values]):
                text = self.options.get('to_json_text', 'authored-json-token')
                if self.options.get('engine_join') and values is not None:
                    assert self.engine is not None
                    report = self.engine.write_saved(values)
                    assert report['returned']
                    text = bytes.fromhex(report['json_utf8_hex']).decode('utf-8')
                    self.joins.append({'direction': 'write', 'input_values': values, 'json': text,
                                       'loaded': report['loaded'], 'deep_input_storage_retained': report['deep_input_storage_retained']})
                if self.options.get('to_json_swap'):
                    value = self.options.get('replacement')
                    self.q(self.data + 0x18, self.new_info(value) if value is not None else 0)
                self.ret(self.string(text))
        elif rva == 0x645DA0:
            self.executed.add(rva)
            assert dx == self.from_json_method and (not cx or cx in self.strings)
            payload = self.strings.get(cx)
            if self.event('from_json_service', [payload]):
                values = self.options.get('loaded_values', {'key': 'loaded-key', 'completedTutorials': ['loaded-t'], 'unlockedCharactersId': ['loaded-c']})
                if self.options.get('engine_join'):
                    report = self.engine.read_saved(payload)
                    assert report['returned']
                    values = report['loaded']['values']
                    self.joins.append({'direction': 'read', 'json': payload, 'loaded': report['loaded']})
                self.ret(self.new_info(values) if values is not None else 0)
        elif rva == 0x1C86170:
            self.executed.add(rva)
            assert r8 == 0 and (not cx or cx in self.strings) and (not dx or dx in self.strings)
            args = [self.strings.get(cx), self.strings.get(dx)]
            if self.event('set_string_service', args):
                self.writes.append(args)
                self.ret()
        elif rva == 0x1C8A5C0:
            self.executed.add(rva)
            assert cx == self.data and dx == 0
            if self.event('scriptable_object_constructor_service', []):
                self.ret()
        else:
            super().hook(uc, address, size, data)

    def run_data(self, method, state=None, options=None):
        self.options = options or {}
        self.writes, self.infos, self.joins = [], [], []
        # Publish only base-method slots during the shared preparation step.
        all_bindings, self.bindings = self.bindings, {a: v for a, v in self.bindings.items() if a not in self.data_bindings}
        self.prepare(state, options=self.options)
        self.bindings = all_bindings
        self.infos = [self.saved]
        self.q(self.data + 0x18, self.saved if state is not None else 0)
        for slot, (kind, name) in self.data_bindings.items():
            token = {'SavedGameInfo_TypeInfo': self.saved_type,
                     'Method$UnityEngine.JsonUtility.FromJson<SavedGameInfo>()': self.from_json_method,
                     'Method$System.Collections.Generic.List<string>.Clear()': self.method_tokens['Clear']}[name]
            self.q(self.base + slot, token)
        returned = self.invoke(METHODS[method], self.data)
        return {'method': method, 'input': state, 'options': self.options, 'returned': returned, 'error': self.error,
                'events': self.events.copy(), 'final': self.snapshot(), 'joined_engine': self.joins.copy()}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, failures, joined, baselines = [], [], [], []
    old = {'key': 'old-key', 'completedTutorials': ['old-t', None], 'unlockedCharactersId': ['old-c']}
    replacement = {'key': 'replacement-key', 'completedTutorials': [], 'unlockedCharactersId': []}
    configurations = []
    for warm in [False, True]:
        configurations += [('.ctor', state, {'warm': warm}) for state in [None, old]]
        configurations += [('Load', old, {'warm': warm, 'get_values': values}) for values in [[None], [''], ['first', 'second'], ['first', ''], ['first', None]]]
        configurations += [('Load', old, {'warm': warm, 'get_values': ['first', 'second'], 'get_swap_at': 1, 'replacement': value}) for value in [None, replacement]]
        configurations += [('Load', old, {'warm': warm, 'get_values': ['first', 'second'], 'loaded_values': None})]
        configurations += [(method, state, {'warm': warm}) for method in ['Load', 'Save', 'ResetTutorials'] for state in [None, old]]
        configurations += [('ResetTutorials', dict(old, completedTutorials=value), {'warm': warm}) for value in [None, []]]
        configurations += [(method, old, {'warm': warm, 'to_json_swap': True, 'replacement': value}) for method in ['Save', 'ResetTutorials'] for value in [None, replacement]]
    for method, state, options in configurations:
        result = m.run_data(method, state, options)
        final = result['final']
        if state is None and method != '.ctor' or method == 'ResetTutorials' and state['completedTutorials'] is None or options.get('get_swap_at') and options.get('replacement') is None or options.get('to_json_swap') and options.get('replacement') is None:
            assert not result['returned'] and result['error'] == 'null_reference'
        else:
            assert result['returned']
        if method == '.ctor' or method == 'Load' and not options.get('get_values', [''])[0] and state is not None:
            assert final['save']['key'] == 'Tutorials'
            assert all(final['save'][name]['values'] == [] for name, _ in FIELDS)
        if method in ['Save', 'ResetTutorials'] and result['returned']:
            assert final['preference_writes'] == [[('replacement-key' if options.get('to_json_swap') else 'old-key'), 'authored-json-token']]
        if method == 'ResetTutorials' and state is not None and state['completedTutorials'] is not None:
            assert final['allocated_infos'][0]['completedTutorials']['values'] == []
            assert final['allocated_infos'][0]['completedTutorials']['version'] == 18
        if method == 'Load' and options.get('get_swap_at') and options.get('replacement') is not None:
            gets = [e for e in result['events'] if e['kind'] == 'get_string_service']
            assert [e['args'][0] for e in gets] == ['old-key', 'replacement-key']
        if method == 'Load' and 'loaded_values' in options:
            assert final['save'] is None
        cases.append(result)
    for method, state, options in [('.ctor', None, {}), ('Load', old, {'get_values': ['']}), ('Load', old, {'get_values': ['first', 'second']}), ('Save', old, {}), ('ResetTutorials', old, {})]:
        baseline = m.run_data(method, state, options)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run_data(method, state, dict(options, failure=[kind, counts[kind]]))
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1, 'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    m.engine = JsonMachine(game_root)
    for warm, method in itertools.product([False, True], ['Save', 'ResetTutorials', 'Load']):
        options = {'warm': warm, 'engine_join': True}
        if method == 'Load':
            options['get_values'] = ['first', '{"key":"from-json","completedTutorials":["a","a"],"unlockedCharactersId":["b"]}']
        result = m.run_data(method, old, options)
        assert result['returned'] and len(result['joined_engine']) == 1
        if method == 'Load':
            assert result['final']['save']['key'] == 'from-json'
        else:
            assert result['final']['preference_writes'][0][0] == 'old-key'
            assert result['joined_engine'][0]['input_values']['completedTutorials'] == ([] if method == 'ResetTutorials' else old['completedTutorials'])
        joined.append(result)
    return {'build_id': BUILD, 'pinned_layout': m.layout, 'saved_data_layout': {'type_def_index': 5943, 'base': 'ScriptableObject', 'fields': [{'name': 'save', 'type': 'SavedGameInfo', 'offset': '0x18'}]},
            'targets': m.data_targets, 'service_targets': m.service_targets,
            'native_ranges': m.data_ranges, 'metadata_bindings': [{'rva': hex(a), 'kind': k, 'name': n} for a, (k, n) in sorted(m.data_bindings.items())],
            'instruction_assertions': m.data_instruction_assertions, 'cases': cases, 'case_count': len(cases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'joined_cases': joined, 'joined_case_count': len(joined), 'executed_address_count': len(m.executed),
            'scope': 'All four SavedGameData native callers execute; SavedGameInfo construction and String.IsNullOrEmpty bodies run natively. Preference access, generic JSON gateway, ScriptableObject base construction and runtime helpers are explicit services. Six JSON joins transfer only exact public values into the independently audited engine pipeline; no live preferences, cross-emulator identity or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['failure_case_count'], result['joined_case_count'], result['executed_address_count'])
