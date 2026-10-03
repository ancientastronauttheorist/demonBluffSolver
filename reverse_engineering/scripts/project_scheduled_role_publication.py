"""Project native post-click result/speech checkpoints into the offline Rust API.

The source report executes native bodies under its declared services. This
projection supplies typed initial storage/UI contracts and does not admit a
player observation, recreate acquisition, or include producer-only private data
in a live solver input. No Rust implementation is imported to derive outputs.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_report_snapshots import expand_snapshots


SOURCE_SHA256 = 'e8182a5a51353941b4d6e07d6564dc2c78f7c9e30c7e8de7366eb37aca62efcf'
ACQUISITION_SHA256 = '3603e404ea1a55ff481fe075bfef36bfe5d339b6c9738bba4632bfc7c4f5c9d8'
RULE = 'scheduled_role_publication_native_v1'


def signed32(value):
    assert 0 <= value < 2**32
    return value if value < 2**31 else value - 2**32


def text(identity, value):
    encoded = value.encode('utf-16-le')
    return {'identity': identity, 'units': [int.from_bytes(encoded[i:i+2], 'little')
                                          for i in range(0, len(encoded), 2)]}


def queue(value):
    return {'rule_version': 'unity_wait_queue_native_v1',
            'generation': value['generation'], 'next_id': value['next_id'],
            'entries': [{'logical_id': row['id'], 'release_present': True,
                         'timing': {'deadline': row['deadline'], 'frame_threshold': row['frame_threshold'],
                                    'phase_mask': row['phase_mask'], 'insertion_generation': row['generation']}}
                        for row in value['entries']]}


def project(case, name, acquired=False):
    initial = case['initial']
    history_prefix = [] if acquired else ['prior_info']
    initial_uses = 1 if acquired else 0
    assert len(initial['iterators']) == 1 and not initial['speech_iterators']
    assert initial['history'] == history_prefix and initial['uses_bits'] == initial_uses
    assert initial['actor_state'] == 10 and initial['actor_previous_state'] == 5
    assert initial['saved_text'] == 'old speech' and not initial['shown']
    assert initial['reveal_order'] == 1
    iterator = initial['iterators'][0]
    assert (iterator['state'], iterator['delay_bits'], iterator['owner'], iterator['info'], iterator['trigger']) == (
        1, 0, 'actor', 'info0', 30)
    assert initial['waits'][iterator['current']] == 0
    actor = next(row for row in initial['board'] if row['id'] == 'actor')
    assert actor['register_as'] is None and not actor['character_start_acted']
    bluff = actor['alignment'] == 20
    assert actor['data'] == ('baa_data' if bluff else 'data')
    assert actor['real_role'] == ('imp_role' if bluff else 'role')
    assert actor['display_bluff'] == ('data' if bluff else None)
    assert actor['bluff_role'] == ('role' if bluff else None)
    ids = {row['id']: (1 if row['id'] == 'actor' else 1000 + row['display_id'])
           for row in initial['board']}
    native_refs = initial['retained_lists'][initial['info']['references']]
    assert native_refs['kind'] == 'characters'
    assert initial['info']['description'] == initial['generated'][0]['description']
    refs = [ids[label] for label in native_refs['values']]
    assert [next(row['display_id'] for row in initial['board'] if row['id'] == label)
            for label in native_refs['values']] == initial['generated'][0]['ordered_reference_ids']
    statuses = {'active': [], 'version': 0, 'resistances': [], 'target': None}
    if acquired:
        acq = initial['acquisition']
        assert acq['active_status_count'] == 0 and acq['active_status_version'] == 24
        assert acq['resistance_count'] == len(acq['resistance_values']) == 1
        assert acq['resistance_values'] == [50] and acq['status_target'] == 'actor'
        assert acq['revealed_byte'] == acq['start_acted_byte'] == acq['killed_while_hidden_byte'] == acq['killed_by_demon_byte'] == 0
        assert acq['register_as'] is None and acq['raw_bluff'] is None
        assert acq['trailer'] is None and acq['runtime'] is None
        statuses = {'active': [], 'version': acq['active_status_version'],
                    'resistances': acq['resistance_values'], 'target': ids[acq['status_target']]}
    publication = {
        'version': 'character_role_publication_native_v1',
        'actor': {'identity': 1, 'data': 30 if bluff else 20, 'bluff': 20 if bluff else None,
                  'register_as': None, 'trailer': None, 'runtime': None, 'dead_prefab': None,
                  'revealed': False, 'uses': initial_uses, 'previous': 5, 'state': 10,
                  'killed_hidden': False, 'killed_demon': False, 'alignment': actor['alignment'],
                  'id': actor['display_id'], 'started': False,
                  'role': 41 if bluff else 40, 'bluff_role': 40 if bluff else None,
                  'saved_act': 103, 'infos': [] if acquired else [201], 'info_version': initial['history_version'],
                  'statuses': statuses,
                  'state_callback': None},
        'act': True, 'raw_bluff': {'identity': 20, 'live': True} if bluff else None,
        'data_assets': [{'identity': 20, 'picking': False}, {'identity': 30, 'picking': False}],
        'history': {'identity': 2, 'backing_array': 3, 'capacity': len(initial['history_slots'])},
        'strings': [text(100, initial['info']['description']), text(103, 'old speech'),
                    text(104, 'prior'), text(105, 'Fixture'), text(106, 'Character: Fixture')],
        'infos': [{'identity': 200, 'description': 100, 'references': 300},
                  {'identity': 201, 'description': 104, 'references': 301}],
        'reference_lists': [{'identity': 300, 'backing_array': 302, 'version': native_refs['version'],
                             'entries': refs},
                            {'identity': 301, 'backing_array': None, 'version': 0, 'entries': []}],
        'result_iterators': [{'identity': 500, 'actor': 1, 'info': 200, 'trigger_bits': 30,
                              'delay_bits': 0, 'state': 1, 'current': 501}],
        'result_waits': [{'identity': 501, 'seconds_bits': 0}],
        'allocations': [{'result_iterator': 500, 'speech_iterator': 600, 'speech_wait': 601}],
        'preappend_callback': None, 'info_revealed_callback': 700,
        'trailer_mode': False, 'trailer_text': None,
        'ui': {'acted_component': 400, 'acted_version': 401, 'blank_text': 402,
               'blank_text_value': 103, 'layout_array': 403, 'layouts': [404, 404],
               'first_game_object': 410, 'log_game_object': 410, 'show_game_object': 410,
               'name_text': 105, 'log_text': 106, 'pickable': 411,
               'objects': [{'identity': 410, 'active': False}, {'identity': 411, 'active': False}]},
        'result_resume_order': [], 'speech_resume_order': [],
        'services': {key: key != 'supplied_resume_order_verified' for key in [
            'runtime_and_metadata_verified', 'callback_captures_and_first_yield_verified',
            'list_storage_and_capacity_verified', 'preappend_callbacks_inert', 'global_callbacks_inert',
            'ui_and_other_services_inert', 'unity_liveness_verified', 'trailer_lookup_stable_verified',
            'supplied_resume_order_verified', 'normal_completion_verified']},
    }
    drains, expected = [], []
    for row in case['drains']:
        inp = row['input']
        owner = {'owner': 'matched', 'producer_time': inp['time'], 'producer_frame_counter': inp['frame']}
        if case.get('owner_outcome') in ('missing', 'null'):
            owner = {'owner': 'unavailable'}
        elif case.get('owner_outcome') == 'mismatch':
            owner = {'owner': 'mismatched'}
        drains.append({'dispatch': {'rule_version': 'unity_wait_eligibility_native_v1',
                                    'sampled_time': inp['time'], 'sampled_frame_counter': inp['frame'],
                                    'phase_mask': inp['phase'], 'generation_before': inp['generation_before']},
                       'owners': {str(entry['id']): owner for entry in row['before']['entries']}})
        state = row['managed']
        assert state['actor_state'] == 10 and state['actor_previous_state'] == 5
        assert state['history'] in (history_prefix, history_prefix + ['info0'])
        assert state['saved_text'] in ('old speech', initial['info']['description'])
        expected.append({'queue': queue(row['after']),
                         'callbacks': row['expected_callback_ids'],
                         'visits': [e['id'] for e in row['events'] if e['kind'] == 'native_wait_visit'],
                         'history': ([] if acquired else [201]) + ([200] if state['history'] == history_prefix + ['info0'] else []),
                         'history_version': state['history_version'], 'uses_bits': state['uses_bits'],
                         'saved_text': state['saved_text'], 'shown': state['shown'],
                         'result_states': [signed32(it['state']) for it in state['iterators']],
                         'speech_states': [signed32(it['state']) for it in state['speech_iterators']]})
    result_id = case['initial_queue']['entries'][0]['id']
    assert result_id == (1 if acquired else 0)
    return {'name': name, 'context': {'rule_version': RULE, 'publication': publication,
                                    'queue': queue(case['initial_queue']), 'result_bindings': {str(result_id): 500},
                                    'normal_lifetime_and_stable_services_verified': True, 'drains': drains},
            'expected': expected}


def project_report(source):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == SOURCE_SHA256, 'native source changed: re-review before projection'
    report = expand_snapshots(json.loads(source.read_text(encoding='utf-8')))
    assert report['build_id'] == 'f530404b0f3f_807de4a83df4'
    cases = []
    for family in ('cases', 'clock_cases', 'owner_cases'):
        for index, case in enumerate(report[family]):
            cases.append(project(case, f'{family}_{index}'))
    assert len(cases) == 29
    return {'schema_version': 1, 'native_report_sha256': digest,
            'build_id': report['build_id'], 'engine_sha256': report['engine_sha256'],
            'scope': '29 post-click native one-result Hunter publication checkpoints; typed metadata/UI service projection. Initial first-yield storage supplied; no acquisition, producer reconstruction, rendered availability, public-history admission, world prior or live caller.',
            'excluded': ['four double-Day stress compositions', 'double-Day same-generation composition',
                         '254 native failed-service prefixes: Rust rejects failed-service contracts'],
            'cases': cases}


def project_acquisition_report(source):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert digest == ACQUISITION_SHA256, 'acquisition report changed: re-review before projection'
    report = expand_snapshots(json.loads(source.read_text(encoding='utf-8')))
    assert report['schema'] == 'hunter_acquisition_publication_v1'
    assert report['build'] == 'f530404b0f3f_807de4a83df4'
    cases = []
    for index, case in enumerate(report['cases']):
        assert case['completed'] and case['click_invoked'] and case['error'] is None
        assert len(case['drains']) == 8 and len(case['after_acquisition_native_records']) == 1
        record = case['after_acquisition_native_records'][0]
        assert record['kind'] == 'acquisition' and record['iterator'] == 'acquisition0'
        assert record['reference_count'] == record['gc_handle'] == 0 and not record['owner_linked']
        assert record['cached_enumerator_present']
        assert case['drains'][3]['after']['entries'] == []
        assert case['click_admission'] == {'completed_acquisition_iterator': 'acquisition0',
            'queue_empty': True, 'native_record_released': True,
            'same_actor': 'actor', 'same_runtime_clone': 'role'}
        before = case['after_acquisition_drain']
        initial = case['after_click']
        assert before['history'] == [] and before['uses_bits'] == 1
        assert before['actor_state'] == 5 and before['actor_previous_state'] == 20
        assert before['reveal_card_init_reveal_byte'] == before['reveal_card_state_raw'] == before['gameplay_current_reveal'] == 0
        assert before['acquisition']['runtime_role'] == initial['acquisition']['runtime_role'] == 'role'
        assert before['acquisition']['data_source_role'] == initial['acquisition']['data_source_role'] == 'acquisition_source_role'
        assert before['acquisition']['current_data'] == initial['acquisition']['current_data'] == 'data'
        assert before['acquisition']['data_source_role'] != before['acquisition']['runtime_role']
        assert [p['trigger'] for p in before['acquisition']['phase_calls']] == [3, 7]
        assert [p['trigger'] for p in initial['acquisition']['phase_calls']] == [3, 7, 30]
        assert initial['uses_bits'] == 1 and initial['history'] == [] and initial['history_version'] == 10
        post_click = dict(case, initial=initial, initial_queue=case['drains'][4]['before'], drains=case['drains'][4:])
        assert post_click['initial_queue']['next_id'] == 2
        assert post_click['initial_queue']['entries'][0]['kind'] == 'result'
        projected = project(post_click, f'acquired_cases_{index}', acquired=True)
        projected['acquisition_proof'] = {'native_record_before_click': record,
            'click_admission': case['click_admission'],
            'source_role': before['acquisition']['data_source_role'],
            'runtime_role': before['acquisition']['runtime_role'],
            'native_history_version_after_clear': initial['history_version'],
            'scope': 'original conditional native Init/acquisition handoff; supplied old speech, clone/CLR/runtime/UI and other actors. No constructor, generated board or rendered public history.'}
        cases.append(projected)
    assert len(cases) == 6 and sum(len(c['expected']) for c in cases) == 24
    return {'schema_version': 1, 'native_report_sha256': digest, 'build_id': report['build'],
            'scope': 'Six actual acquired post-click result states, same actor/data/runtime clone after native acquisition callback/release; native use one and empty history. Rust executes result/speech publication only, not acquisition, pixel availability or player-history admission.',
            'excluded': ['three rejected acquisition owners: no click or result to project',
                         '192 native failed-service prefixes: Rust rejects failed-service contracts'],
            'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    parser.add_argument('--acquisition', action='store_true')
    args = parser.parse_args()
    projected = project_acquisition_report(args.source) if args.acquisition else project_report(args.source)
    assert args.output.parent.is_dir()
    args.output.write_text(json.dumps(projected, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': len(projected['cases']),
                      'drains': sum(len(case['expected']) for case in projected['cases']),
                      'native_report_sha256': projected['native_report_sha256']}))
