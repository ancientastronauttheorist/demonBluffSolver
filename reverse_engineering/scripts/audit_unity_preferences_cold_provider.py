"""Join native cold token discovery into preference provider and entry execution."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_preferences_getter import response
from audit_unity_preferences_provider import Machine as ProviderMachine, compact_case, verify_native as verify_provider
from audit_unity_preferences_token import CACHE, TokenServices, verify_native as verify_token


class Machine(TokenServices, ProviderMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.bind_token_services()

    def initialize_provider(self, fields, options):
        super().initialize_provider(fields, options)
        self.d(self.base + CACHE, options.get('cached_token_value', 0xFFFFFFFF))
        self.initialize_token_services(options.get('token_options'))
        self.u.mem_write(self.token_data, bytes(32))

    def snapshot(self):
        return {**super().snapshot(), 'token': self.token_snapshot()}

    def hook(self, uc, address, size, data):
        if not self.hook_token_service(address):
            super().hook(uc, address, size, data)


def audit(game_root):
    m = Machine(game_root)
    verified = {**verify_provider(m), **verify_token(m)}
    cases, entry_cases, baselines, failures = [], [], [], []
    tokens = [
        ({'subauthority': 0x1000}, 0x1000),
        ({'subauthority': 0x2000}, 0x2000),
        ({'open_result': 0, 'open_handle': 0, 'last_errors': [5]}, 0xFFFFFFFF),
        ({'open_result': 0, 'open_handle': 0, 'last_errors': [0]}, 0),
        ({'allocation_success': False, 'last_errors': [122, 8]}, 0xFFFFFFFF),
        ({'data_result': 0, 'last_errors': [122, 5]}, 0xFFFFFFFF),
    ]
    for mode, (token_options, expected_cache) in itertools.product([0, 1, 255], tokens):
        result = m.run_provider(mode, ['Studio', 'Game'], {'token_options': token_options})
        assert result['returned'] and result['final']['token']['cached_value'] == expected_cache
        prefix = 'Software\\AppDataLow\\Software\\' if expected_cache == 0x1000 else 'Software\\'
        acquisitions = [c for c in result['final']['provider_api_calls'] if c['api'] in ['RegCreateKeyW', 'RegOpenKeyExW']]
        assert [c['path'] for c in acquisitions] == [prefix + 'Studio\\Game'] * 2
        assert result['final']['handles'] == [0xABCDEF, 0x123456]
        assert result['final']['token']['api_calls'][0]['api'] == 'GetCurrentProcess'
        cases.append(compact_case(result))
    for subauthority, probe_statuses in itertools.product([0x1000, 0x2000], [[0x3FA, 0], [0x3FA, 0x3FA, 0]]):
        result = m.run_provider(1, ['Studio', 'Game'], {'token_options': {'subauthority': subauthority},
                                                     'probe_statuses': probe_statuses})
        assert result['returned']
        assert len([c for c in result['final']['token']['api_calls'] if c['api'] == 'GetCurrentProcess']) == 1
        assert len([c for c in result['final']['provider_api_calls'] if c['api'] == 'RegCreateKeyW']) == len(probe_statuses)
        cases.append(compact_case(result))
    for last_error, expected_discoveries in [(0, 1), (5, 2)]:
        result = m.run_provider(1, ['Studio', 'Game'], {
            'token_options': {'open_result': 0, 'open_handle': 0, 'last_errors': [last_error] * 2},
            'probe_statuses': [0x3FA, 0]})
        assert result['returned']
        assert len([c for c in result['final']['token']['api_calls'] if c['api'] == 'GetCurrentProcess']) == expected_discoveries
        cases.append(compact_case(result))
    result = m.run_provider(1, ['Studio', 'Game'], {'cached_fields': ['Studio', 'Game'],
                                                  'initial_handles': [0xAAAA, 0xBBBB]})
    assert result['returned'] and not result['final']['token']['api_calls']
    assert result['final']['token']['cached_value'] == 0xFFFFFFFF
    cases.append({'label': 'matching_fields_skip_cold_discovery', **compact_case(result)})
    for direction, subauthority in itertools.product(['set', 'get'], [0x1000, 0x2000]):
        options = {'token_options': {'subauthority': subauthority}}
        if direction == 'get':
            options['query_responses'] = [response(size=6), response(size=6, data=b'value\0')]
        result = m.run_entry(direction, 'Tutorials', 'default' if direction == 'get' else 'value', ['Studio', 'Game'], options)
        assert result['returned'] and result['result'] == ('value' if direction == 'get' else 1)
        assert result['configuration_storage_retained'] and result['input_storage_retained']
        assert result['final']['token']['cached_value'] == subauthority
        entry_cases.append({'direction': direction, 'subauthority': subauthority, 'result': result['result'],
                            'final': result['final'], 'event_kinds': [e['kind'] for e in result['events']],
                            'normal_return_and_storage_verified': True})
    for direction in ['set', 'get']:
        fields = ['Studio', 'Game']
        options = {'token_options': {'subauthority': 0x1000}}
        if direction == 'get':
            options['query_responses'] = [response(size=6), response(size=6, data=b'value\0')]
        baseline = m.run_entry(direction, 'Tutorials', 'default' if direction == 'get' else 'value', fields, options)
        baseline_id = len(baselines)
        baselines.append({'entry_fields': fields, **baseline})
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run_entry(direction, 'Tutorials', baseline['value_or_default'], fields,
                                 {**options, 'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases),
            'entry_cases': entry_cases, 'entry_case_count': len(entry_cases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.executed),
            'scope': 'Actual cold token predicate, provider/path/acquisition/invalidation, native entries, getter/setter/hash/formatting and cleanup execute in one emulator. Windows security/registry/UTF-8 API results, native configuration fields, runtime exports, memory primitives, string assignment/copy and allocation remain authored services. No actual OS state or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['entry_case_count'], report['failure_case_count'], report['executed_address_count'])
