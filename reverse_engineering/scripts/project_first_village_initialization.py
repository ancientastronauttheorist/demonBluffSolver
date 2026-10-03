"""Project completed native N5 Init prefixes into the existing offline Rust batch.

Native expected actors/continuations are independent of Rust. Synthetic caller
boundary selectors stop before the next Init or publication effects. Physical
lists, CPU state, pools, act and wait duration remain native-only evidence.
"""
import argparse
import copy
import hashlib
import json
from pathlib import Path

from audit_report_snapshots import expand_snapshots


SOURCE_SHA256 = '855f8976081da9541161bcdf15e0a130cbd6f0115f2351c027708b6c3fb08eb1'
BUILD = 'f530404b0f3f_807de4a83df4'


def optional(value):
    return value or None


def signed(value):
    assert 0 <= value < 2**32
    return value if value < 2**31 else value - 2**32


def actor(row):
    storage = row['acted_infos_storage']
    statuses = row['status_storage']
    assert storage['count'] == 0
    assert statuses['count'] == len(statuses['active_values'])
    assert statuses['resistance_count'] == len(statuses['resistance_values'])
    assert row['dead_prefab'] == 0
    out = {k: optional(row[k]) for k in ('data', 'bluff', 'register_as', 'trailer',
           'runtime', 'role', 'bluff_role', 'saved_act', 'state_callback')}
    out.update(identity=row['identity'], dead_prefab=None, infos=[], info_version=storage['version'],
               statuses={'active': [signed(v) for v in statuses['active_values']],
                         'version': statuses['version'],
                         'resistances': [signed(v) for v in statuses['resistance_values']],
                         'target': optional(statuses['target'])})
    for k in ('revealed', 'killed_hidden', 'killed_demon', 'started'):
        assert row[k] in (0, 1)
        out[k] = bool(row[k])
    out.update({k: signed(row[k]) for k in ('uses', 'previous', 'state', 'alignment', 'id')})
    return out


def continuations(snapshot):
    out = []
    for row in snapshot['continuations']:
        assert row['wait_bits'] == 0x3E99999A and row['state'] == 1
        out.append({'identity': row['identity'], 'actor': row['owner'],
                    'state': signed(row['state']), 'current': optional(row['current'])})
    return out


def completed_prefix_cases(report):
    """Project native Init snapshots shared by retained, versioned joins."""
    initial = report['initial']
    rows = initial['actors']
    calls = report['initializers']
    assets = initial['asset_bindings']
    assert len(rows) == len(calls) == 5 and not initial['continuations']
    order = report['generation']['final']['returned_order']
    assert order == report['domain']['asset_order'] == [21596,21614,21626,21621,21618]
    actors = {str(r['identity']): actor(r) for r in rows}
    by_pointer = {a['identity']: 'data:'+key for key, a in assets.items()}
    object_ids = {r['label']: r['identity'] for r in rows}
    data_roles, source_roles, data = {}, {}, {}
    classes = {'Role': ['Role']}
    for key, a in assets.items():
        asset_alias, source_alias = 'data:'+key, 'source:'+key
        object_ids[asset_alias], object_ids[source_alias] = a['identity'], a['source_role']
        data_roles[asset_alias], source_roles[source_alias] = source_alias, a['role_type']
        classes[a['role_type']] = ['Role', a['role_type']]
        data[str(a['identity'])] = {'starting_alignment': a['startingAlignment'], 'source_role': a['source_role']}
    state = {'board': 'board', 'order': None, 'callback': None,
             'lists': {'board': {'items': [r['label'] for r in rows], 'count': 5}}, 'arrays': {},
             'identities': {r['label']: by_pointer.get(r['data']) for r in rows},
             'data_roles': data_roles, 'iterator_state': 0}
    for r in rows:
        assert not r['data'] or r['data'] in by_pointer
    caller = {'version': 'manage_setup_caller_native_v1',
              **{k: True for k in ('pinned_class_hierarchies', 'stable_occurrence_services',
                 'reference_equality', 'supplied_init_data_write_only', 'supplied_gateway_effects_only',
                 'caller_metadata_initialized', 'shuffle_metadata_initialized', 'math_initialized',
                 'gameplay_initialized', 'object_initialized')},
              'state': state, 'roster': ['data:'+str(v) for v in order],
              'roles': source_roles, 'classes': classes, 'callbacks': [], 'failure': None, 'after': []}
    context = {'version': 'setup_initialization_batch_native_v1', 'caller': caller,
               'object_identities': object_ids, 'actors': actors, 'raw_bluff_liveness': {},
               'positions': {str(r['identity']): i+1 for i, r in enumerate(rows)},
               'data': data, 'continuations': [], 'allocations': [],
               'required_objects_and_lists_valid': True, 'callbacks_and_ui_inert': True,
               'clone_results_verified': True, 'synchronous_first_yield_verified': True}
    cases = []
    for index, call in enumerate(calls):
        count = index+1
        assert call['completed'] and call['actor_identity'] == rows[index]['identity']
        assert call['asset_id'] == order[index] and call['display_id'] == 5-index
        snapshot = call['snapshot']
        assert len(snapshot['continuations']) == count
        current = snapshot['continuations'][-1]
        assert current['identity'] == call['iterator'] and current['owner'] == call['actor_identity']
        a = assets[str(call['asset_id'])]
        context['allocations'].append({'init_index': index,
            'clone': {'identity': call['clone'], 'source_role': a['source_role'], 'managed_class': a['role_type']},
            'continuation_identity': call['iterator'], 'wait_identity': current['current']})
        context['caller']['failure'] = {'gateway': 'publish' if count == 5 else 'init',
                                        'occurrence': 1 if count == 5 else count+1}
        cases.append({'completed_inits': count, 'context': copy.deepcopy(context),
                      'expected_actors': {str(r['identity']): actor(r) for r in snapshot['actors']},
                      'expected_continuations': continuations(snapshot)})
    return cases


def project_report(source):
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    assert SOURCE_SHA256 and digest == SOURCE_SHA256, 'native source needs final pin/review'
    report = expand_snapshots(json.loads(source.read_text(encoding='utf-8')))
    assert report['schema_version'] == 'first_village_initialization_v1' and report['build_id'] == BUILD
    return {'schema_version': 1, 'native_report_sha256': digest, 'build_id': BUILD,
            'scope': 'Five successful native Init-return prefixes projected to Actor/Continuation only. Caller failure settings select synthetic boundaries; they do not model original exceptions. Positions are declared native-slot labels, not certified UI positions. No constructor/List/CPU/pool/queue/role-action/pixel/PlayerHistory replay is asserted by Rust.',
            'excluded': ['native failed-service partial mutations', 'native-only act/list/backing/wait-duration/CPU/pool assertions'],
            'cases': completed_prefix_cases(report)}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('source', type=Path)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = project_report(args.source)
    assert args.output.parent.is_dir()
    args.output.write_text(json.dumps(result, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'cases': len(result['cases']), 'native_report_sha256': result['native_report_sha256']}))
