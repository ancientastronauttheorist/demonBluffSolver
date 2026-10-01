"""Compose native runtime string construction with preference storage and Load."""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as StringMachine, verify_native as verify_strings
from audit_saved_game_storage_join import Machine as StorageMachine, RegistryMachine as StorageRegistry
from audit_saved_game_storage_join import compact_trace, values
from audit_unity_preferences_setter import formatted_key


class RegistryMachine(StorageRegistry):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.runtime = StringMachine(game_root)
        self.runtime_verified = verify_strings(self.runtime)
        self.runtime_calls = []

    def snapshot(self):
        return {**super().snapshot(), 'runtime_string_calls': self.runtime_calls.copy()}

    def hook(self, uc, address, size, data):
        if self.runtime_services.get(address) != 'il2cpp_string_new_len':
            return super().hook(uc, address, size, data)
        self.executed.add(address - self.base)
        length = self.reg(self.x.UC_X86_REG_RDX) & 0xFFFFFFFF
        assert length <= 2048
        raw = bytes(uc.mem_read(self.reg(self.x.UC_X86_REG_RCX), length))
        if not self.event('native_runtime_string_adapter', [raw.hex()]):
            return
        result = self.runtime.run_new_len(raw, self.options.get('runtime_options'))
        self.runtime_calls.append(result)
        if not result['native_trace']['returned']:
            self.error = 'runtime_' + result['native_trace']['error']
            uc.emu_stop()
            return
        self.ret(self.make_string(result['text']))  # Values only; identities stay local.

    def entry(self, direction, key, value, options=None):
        self.runtime_calls = []
        return super().entry(direction, key, value, options)


class Machine(StorageMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        self.registry = RegistryMachine(game_root)


def stored_raw(raw, kind=3):
    return {formatted_key(b'Tutorials').hex(): {'type': kind, 'data': (raw + b'\0').hex()}}


def expected(raw):
    try:
        return raw.decode('utf-8')
    except UnicodeDecodeError:
        return ''


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    direct, getters, loads, round_trips, failures, baselines = [], [], [], [], [], []
    valid = [b'', b'hello', 'caf\u00e9 \u6f22\U0001f608'.encode('utf-8'), b'a\0b', b'\0\xff',
             b'a\0\xff', b'abc' + b'\0' + 'caf\u00e9'.encode('utf-8'), b'x' * 2048]
    malformed = [bytes.fromhex(raw) for raw in ['80', 'ff', 'c0af', 'c2', 'eda080', 'e282', 'f4908080', 'f09f98']]
    payloads = valid + malformed + [b'A' + raw + b'Z' for raw in malformed]
    for raw in payloads:
        result = m.registry.runtime.run_new_len(raw)
        assert result['native_trace']['returned'] and result['text'] == expected(raw)
        assert result['utf16_unit_count'] == len(result['text'].encode('utf-16-le')) // 2
        direct.append(result)
        for kind in [1, 3]:
            m.registry.storage = stored_raw(raw, kind)
            report = m.registry.entry('get', 'Tutorials', 'default')
            native_input = b'default' if kind == 1 and any(b >= 128 for b in raw) else raw.split(b'\0', 1)[0]
            assert report['returned'] and report['result'] == expected(native_input)
            calls = report['final']['runtime_string_calls']
            assert len(calls) == 1 and calls[0]['native_trace']['input'] == native_input.hex()
            assert calls[0]['native_trace']['input_storage_retained']
            getters.append({'kind': kind, 'raw': raw.hex(), 'result': report})
    old = {'key': 'Tutorials', 'completedTutorials': ['old'], 'unlockedCharactersId': ['old-c']}
    for state in [old, dict(old, completedTutorials=['caf\u00e9', None], unlockedCharactersId=['\u6f22\U0001f608'])]:
        write = m.run_data('Save', state, storage={})
        assert write['returned'] and not write['final']['storage_calls'][0]['final']['runtime_string_calls']
        read = m.run_data('Load', old)
        assert read['returned'] and values(read['final']['save']) == write['joined_engine'][0]['loaded']['values']
        assert all(len(call['final']['runtime_string_calls']) == 1 for call in read['final']['storage_calls'])
        round_trips.append({'write': write, 'read': read})
    payload = '{"key":"loaded","completedTutorials":["\u6f22"],"unlockedCharactersId":["c"]}'.encode('utf-8')
    for raw, kind, fresh in [(payload, 3, False), (payload, 1, True),
                             (payload + b'\0\xff', 3, False), (payload + b'\0\xff', 1, True),
                             *[(raw, 3, True) for raw in malformed],
                             *[(payload + raw, 3, True) for raw in malformed]]:
        result = m.run_data('Load', old, storage=stored_raw(raw, kind))
        assert result['returned']
        final = values(result['final']['save'])
        assert final == ({'key': 'Tutorials', 'completedTutorials': [], 'unlockedCharactersId': []}
                         if fresh else {'key': 'loaded', 'completedTutorials': ['\u6f22'], 'unlockedCharactersId': ['c']})
        assert len(result['final']['storage_calls']) == (1 if fresh else 2)
        assert len(result['joined_engine']) == (0 if fresh else 1)
        if fresh:
            assert result['final']['save']['completedTutorials']['identity'] != result['final']['save']['unlockedCharactersId']['identity']
            runtime_call = result['final']['storage_calls'][0]['final']['runtime_string_calls'][0]
            assert runtime_call['text'] == '' and runtime_call['managed_storage']['uses_cached_empty']
            assert not any(e['kind'] == 'non_generic_from_json_service' for e in result['events'])
        loads.append({'kind': kind, 'raw': raw.hex(), 'fresh_constructor_branch': fresh, 'result': result})
    for raw in [b'abc', b'x' * 2048]:
        m.registry.storage = stored_raw(raw)
        baseline = m.registry.entry('get', 'Tutorials', '')
        baseline_id = len(baselines)
        baselines.append(baseline)
        occurrence = 0
        for index, event in enumerate(baseline['events']):
            if event['kind'] != 'native_runtime_string_adapter':
                continue
            occurrence += 1
            result = m.registry.entry('get', 'Tutorials', '', {'failure': [event['kind'], occurrence]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'failure': [event['kind'], occurrence],
                             'prefix_length': index + 1, 'exact_snapshot_verified': True})
        runtime_baseline = baseline['final']['runtime_string_calls'][0]['native_trace']
        counts = {}
        for index, event in enumerate(runtime_baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.registry.entry('get', 'Tutorials', '', {'runtime_options': {'failure': [kind, counts[kind]]}})
            assert not result['returned'] and result['error'] == 'runtime_' + kind
            trace = result['final']['runtime_string_calls'][0]['native_trace']
            assert trace['events'] == runtime_baseline['events'][:index + 1]
            assert trace['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'runtime_failure': [kind, counts[kind]],
                             'runtime_prefix_length': index + 1, 'exact_runtime_snapshot_verified': True})
    result = m.run_data('Load', old, {'storage_options': {'runtime_options': {
        'failure': ['gc_allocate_service', 1]}}}, storage=stored_raw(payload))
    assert not result['returned'] and result['error'] == 'storage_runtime_gc_allocate_service'
    assert values(result['final']['save']) == old and not result['joined_engine']
    failures.append({'label': 'runtime_stop_propagates_to_Load', 'result': result})
    return compact_trace({'build_id': BUILD, 'runtime_verification': m.registry.runtime_verified,
                          'backend_byte_bound': 2048, 'direct_runtime_cases': direct, 'direct_case_count': len(direct),
                          'getter_cases': getters, 'getter_case_count': len(getters),
                          'load_cases': loads, 'load_case_count': len(loads),
                          'round_trip_cases': round_trips, 'round_trip_case_count': len(round_trips),
                          'failure_baselines': baselines, 'failure_cases': failures,
                          'failure_case_count': len(failures),
                          'storage_executed_address_count': len(m.registry.executed),
                          'runtime_executed_address_count': len(m.registry.runtime.executed),
                          'caller_executed_address_count': len(m.executed),
                          'scope': 'Native explicit-length UTF-8 runtime string construction is composed by value with native preference provider/getter/setter, save Load, generic FromJson and engine JSON. Runtime-created text is projected into separate authored string tokens; object identities and allocation ownership never cross emulator boundaries. Registry and Windows API outcomes, runtime metadata/GC allocation/memory and remaining managed/JSON gateway services stay explicit. Fixture input is bounded to 2048 bytes. Controlled stops assert exact prefixes/snapshots without native exception unwinding; no actual OS registry or live process access.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['direct_case_count'], report['getter_case_count'], report['load_case_count'],
          report['failure_case_count'], report['runtime_executed_address_count'])
