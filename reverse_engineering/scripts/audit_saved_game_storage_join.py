"""Join native save callers, engine JSON and offline native preference storage."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_data import METHODS as DATA_METHODS
from audit_saved_game_info_json import Machine as JsonMachine
from audit_saved_game_info_methods import METHODS as INFO_METHODS
from audit_saved_game_preferences import Machine as PreferenceMachine
from audit_unity_preferences_getter import response
from audit_unity_preferences_provider import Machine as ProviderMachine, verify_native
from audit_unity_preferences_setter import formatted_key


class CallerBudget:
    """Keep the instruction cap and allow bounded nested-emulator wall time."""
    def __init__(self, emulator):
        self.emulator = emulator

    def __getattr__(self, name):
        return getattr(self.emulator, name)

    def emu_start(self, begin, until, timeout=0, count=0):
        assert count == 100000
        return self.emulator.emu_start(begin, until, timeout=120_000_000, count=count)


class RegistryMachine(ProviderMachine):
    """A supplied, isolated registry service shared between native entry calls."""
    def __init__(self, game_root):
        super().__init__(game_root)
        self.storage = {}

    def hook(self, uc, address, size, data):
        if address == self.registry_query and self.reg(self.x.UC_X86_REG_RDX):
            x = self.x
            name = self.cstring(self.reg(x.UC_X86_REG_RDX)).hex()
            sp = self.reg(x.UC_X86_REG_RSP)
            buffer, size_pointer = self.rq(sp + 0x28), self.rq(sp + 0x30)
            capacity = int.from_bytes(uc.mem_read(size_pointer, 4), 'little') if buffer else None
            stored = self.storage.get(name)
            if stored is None:
                # Authored missing-value policy retains the caller's type/capacity
                # so the actual helper's legacy retry gets the same output buffer.
                kind = int.from_bytes(uc.mem_read(self.reg(x.UC_X86_REG_R9), 4), 'little')
                answer = response(2, kind, capacity or 0)
            else:
                raw = bytes.fromhex(stored['data'])
                answer = response(234 if buffer and capacity < len(raw) else 0,
                                  stored['type'], len(raw),
                                  raw if buffer and capacity >= len(raw) else None)
            self.options['query_responses'].append(answer)
        previous = len(self.registry_writes)
        super().hook(uc, address, size, data)
        if address == self.registry_set and len(self.registry_writes) != previous:
            name, kind, raw, _ = self.registry_writes[-1]
            self.storage[name] = {'type': kind, 'data': raw}

    def entry(self, direction, key, value, options=None):
        return self.run_entry(direction, key, value, ['Studio', 'Game'],
                              {**(options or {}), 'query_responses': []})


class Machine(PreferenceMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        self.engine = JsonMachine(game_root)
        self.registry = RegistryMachine(game_root)
        self.storage_verified = verify_native(self.registry)
        self.storage_calls = []
        self.u = CallerBudget(self.u)

    def snapshot(self):
        return {**super().snapshot(), 'storage': self.registry.storage.copy(),
                'storage_calls': self.storage_calls.copy()}

    def invoke(self, rva, receiver, argument=0):
        if rva in DATA_METHODS.values():
            for method, value in self.options.get('info_actions', []):
                assert method in INFO_METHODS and method != '.ctor'
                if not super().invoke(INFO_METHODS[method][0], self.saved, self.string(value)):
                    return False
        return super().invoke(rva, receiver, argument)

    def hook(self, uc, address, size, data):
        if address not in [self.get_service, self.set_service]:
            return super().hook(uc, address, size, data)
        self.executed.add(address - self.base)
        x = self.x
        cx, dx = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX]]
        assert (not cx or cx in self.strings) and (not dx or dx in self.strings)
        direction = 'get' if address == self.get_service else 'set'
        key, value = self.strings.get(cx), self.strings.get(dx)
        if direction == 'get':
            assert dx == self.empty_literal and value == ''
        if not self.event('native_storage_' + direction + '_adapter', [key, value]):
            return
        index = len(self.storage_calls)
        replacements = self.options.get('storage_before_call', {})
        if str(index) in replacements:
            self.registry.storage = replacements[str(index)].copy()
        options = self.options.get('storage_options', {})
        report = self.registry.entry(direction, key, value, options)
        assert report['input_storage_retained'] and report['configuration_storage_retained']
        acquisitions = [c for c in report['final']['provider_api_calls']
                        if c['api'] in ['RegCreateKeyW', 'RegOpenKeyExW']]
        assert all(c['path'] == 'Software\\Studio\\Game' for c in acquisitions)
        self.storage_calls.append({'direction': direction, 'key': key, 'value': value,
                                   'returned': report['returned'], 'result': report['result'],
                                   'error': report['error'], 'events': report['events'],
                                   'final': report['final'],
                                   'input_storage_retained': report['input_storage_retained'],
                                   'configuration_storage_retained': report['configuration_storage_retained']})
        if not report['returned']:
            self.error = 'storage_' + str(report['error'])
            uc.emu_stop()
            return
        if direction == 'get':
            if self.options.get('get_swap_at') == index + 1:
                replacement = self.options.get('replacement')
                self.q(self.data + 0x18, self.new_info(replacement) if replacement is not None else 0)
            self.ret(self.string(report['result']))
        else:
            if report['result']:
                self.writes.append([key, value])
            self.ret(0xDEADBEEF00000000 | report['result'])

    def run_data(self, method, state=None, options=None, storage=None):
        self.storage_calls = []
        if storage is not None:
            self.registry.storage = storage.copy()
        return super().run_data(method, state, {'engine_join': True, **(options or {})})


def stored(key, text, kind=3, legacy=False):
    raw_key = (key or '').encode('utf-8')
    name = raw_key if legacy else formatted_key(raw_key)
    return {name.split(b'\0', 1)[0].hex(): {'type': kind, 'data': (text.encode('utf-8') + b'\0').hex()}}


def values(save):
    if save is None:
        return None
    return {'key': save['key'], **{name: None if save[name] is None else save[name]['values']
                                  for name in ['completedTutorials', 'unlockedCharactersId']}}


def compact_trace(value):
    """Omit repeated event snapshots after exact in-memory prefix checks pass."""
    if isinstance(value, list):
        return [compact_trace(item) for item in value]
    if isinstance(value, dict):
        return {key: compact_trace(item) for key, item in value.items()
                if not (key == 'snapshot' and 'kind' in value and 'args' in value)}
    return value


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, baselines, failures = [], [], []
    states = [
        {'key': key, 'completedTutorials': ['t', 't', None], 'unlockedCharactersId': ['c']}
        for key in [None, '', 'Tutorials', 'caf\u00e9', 'a\0tail', 'k' * 500]
    ]
    for method, state, warm in itertools.product(['Save', 'ResetTutorials'], states, [False, True]):
        result = m.run_data(method, state, {'cache_warm': warm}, storage={})
        assert result['returned'] and len(result['final']['storage_calls']) == 1
        expected = dict(state, completedTutorials=[] if method == 'ResetTutorials' else state['completedTutorials'])
        assert values(result['final']['save']) == expected
        assert len(result['joined_engine']) == 1
        text = result['joined_engine'][0]['json']
        assert result['final']['storage'] == stored(state['key'], text)
        loaded = m.run_data('Load', state, {'cache_warm': warm})
        assert loaded['returned'] and len(loaded['final']['storage_calls']) == 2
        assert values(loaded['final']['save']) == result['joined_engine'][0]['loaded']['values']
        assert loaded['joined_engine'][0]['json'] == text
        cases.append({'label': method + '_round_trip', 'write': result, 'read': loaded})
    old = {'key': 'Tutorials', 'completedTutorials': ['old'], 'unlockedCharactersId': ['c']}
    for actions in [[['AddTutorial', 'new'], ['AddTutorial', 'new'], ['AddCharacter', None]],
                    [['ClearTutorials', None], ['ClearUnlockedCharacters', None]]]:
        result = m.run_data('Save', old, {'info_actions': actions, 'capacity': 4}, storage={})
        assert result['returned']
        expected = ({'key': 'Tutorials', 'completedTutorials': ['old', 'new'], 'unlockedCharactersId': ['c', None]}
                    if actions[0][0] == 'AddTutorial' else
                    {'key': 'Tutorials', 'completedTutorials': [], 'unlockedCharactersId': []})
        assert values(result['final']['save']) == expected
        loaded = m.run_data('Load', old)
        assert loaded['returned'] and values(loaded['final']['save']) == result['joined_engine'][0]['loaded']['values']
        cases.append({'label': 'info_mutation_round_trip', 'write': result, 'read': loaded})
    payload = '{"key":"loaded","completedTutorials":["\u6f22"],"unlockedCharactersId":["c"]}'
    for legacy, kind in itertools.product([False, True], [1, 3]):
        result = m.run_data('Load', old, storage=stored('Tutorials', payload, kind, legacy))
        assert result['returned']
        expected = ({'key': 'Tutorials', 'completedTutorials': [], 'unlockedCharactersId': []}
                    if kind == 1 else {'key': 'loaded', 'completedTutorials': ['\u6f22'], 'unlockedCharactersId': ['c']})
        assert values(result['final']['save']) == expected
        assert len(result['final']['storage_calls']) == (1 if kind == 1 else 2)
        cases.append({'label': 'legacy_and_type_policy', 'legacy': legacy, 'kind': kind, 'read': result})
    for label, storage in [('missing', {}), ('empty', stored('Tutorials', ''))]:
        result = m.run_data('Load', old, storage=storage)
        assert result['returned'] and len(result['final']['storage_calls']) == 1
        assert values(result['final']['save']) == {'key': 'Tutorials', 'completedTutorials': [], 'unlockedCharactersId': []}
        cases.append({'label': label, 'read': result})
    result = m.run_data('Load', old, {'storage_options': {'open_status': 2}},
                        storage=stored('Tutorials', payload))
    assert result['returned'] and len(result['final']['storage_calls']) == 1
    assert result['final']['storage_calls'][0]['final']['query_requests'] == []
    assert values(result['final']['save']) == {
        'key': 'Tutorials', 'completedTutorials': [], 'unlockedCharactersId': []}
    cases.append({'label': 'read_handle_acquisition_failed', 'read': result})
    for method in ['Load', 'Save']:
        result = m.run_data(method, old, {'storage_options': {
            'cached_fields': ['Studio', 'Game'], 'initial_handles': [0xABCDEF, 0x123456]}},
            storage=stored('Tutorials', payload))
        assert result['returned']
        for call in result['final']['storage_calls']:
            assert not any(c['api'] in ['RegCreateKeyW', 'RegOpenKeyExW', 'RegCloseKey']
                           for c in call['final']['provider_api_calls'])
            assert call['final']['handles'] == [0xABCDEF, 0x123456]
        cases.append({'label': 'provider_cache_warm', 'method': method, 'result': result})
    replacement = dict(old, key='replacement')
    second = '{"key":"second","completedTutorials":[],"unlockedCharactersId":[]}'
    result = m.run_data('Load', old, {'get_swap_at': 1, 'replacement': replacement,
        'storage_before_call': {'1': stored('replacement', second)}}, storage=stored('Tutorials', payload))
    assert result['returned'] and values(result['final']['save'])['key'] == 'second'
    assert [c['key'] for c in result['final']['storage_calls']] == ['Tutorials', 'replacement']
    cases.append({'label': 'second_read_current_key', 'read': result})
    result = m.run_data('Save', old, {'to_json_swap': True, 'replacement': replacement}, storage={})
    assert result['returned'] and result['final']['storage'] == stored('replacement', result['joined_engine'][0]['json'])
    assert result['joined_engine'][0]['input_values'] == old
    cases.append({'label': 'serialize_then_current_key', 'write': result})
    for method, options in itertools.product(['Save', 'ResetTutorials'], [{'registry_status': 5}, {'create_status': 5}]):
        result = m.run_data(method, old, {'storage_options': options}, storage={})
        assert not result['returned'] and result['error'] == 'preference_exception'
        assert result['final']['preference_exception'] == {
            'allocated': True, 'message': 'Could not store preference value'}
        assert result['final']['storage'] == {} and result['final']['preference_writes'] == []
        assert values(result['final']['save'])['completedTutorials'] == ([] if method == 'ResetTutorials' else ['old'])
        cases.append({'label': 'write_failure', 'method': method, 'options': options, 'write': result})
    for method, kind, storage in [
        ('Save', 'RegSetValueExA_service', {}),
        ('Load', 'RegQueryValueExA_service', stored('Tutorials', payload)),
    ]:
        result = m.run_data(method, old, {'storage_options': {'failure': [kind, 1]}}, storage=storage)
        assert not result['returned'] and result['error'] == 'storage_' + kind
        assert result['final']['storage'] == storage
        assert values(result['final']['save']) == old
        assert result['final']['storage_calls'][-1]['events'][-1]['kind'] == kind
        cases.append({'label': 'storage_service_stop', 'method': method, 'result': result})
    for method, storage in [('Save', {}), ('ResetTutorials', {}), ('Load', stored('Tutorials', payload))]:
        baseline = m.run_data(method, old, storage=storage)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run_data(method, old, {'failure': [kind, counts[kind]]}, storage=storage)
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return compact_trace({'build_id': BUILD, 'storage_verification': m.storage_verified,
            'caller_instruction_limit': 100000, 'caller_wall_time_limit_seconds': 120,
            'event_snapshots_verified_in_memory_and_omitted_from_report': True,
            'cases': cases, 'case_count': len(cases), 'failure_baselines': baselines,
            'failure_cases': failures, 'failure_case_count': len(failures),
            'saved_caller_executed_address_count': len(m.executed),
            'storage_executed_address_count': len(m.registry.executed),
            'scope': 'Values-only adapters join actual SavedGameData and SavedGameInfo bodies, public PlayerPrefs wrappers, generic FromJson, native engine JSON pipeline and native preference entries/provider/getter/setter/hash/formatting. Registry contents and query/write/handle/conversion API outcomes are authored services in an isolated dictionary. Configuration fields and cached security token are authored inputs. Internal-call resolution, runtime exports, JSON non-generic gateways, metadata/object services, memory and allocator services remain explicit. Object identity, private List versions and native runtime construction do not transfer between emulators. Controlled stops do not simulate native managed exception unwinding. No actual OS registry or live game access.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['saved_caller_executed_address_count'],
          report['storage_executed_address_count'])
