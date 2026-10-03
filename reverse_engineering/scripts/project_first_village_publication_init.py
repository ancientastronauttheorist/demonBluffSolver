"""Independent native-to-Rust semantic projection, ending before ordered Start.

No Rust is imported. UI/slot labels are declared adapter inputs. Callback object,
CPU, physical collection and publication-list identity remain native assertions.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_report_snapshots import expand_snapshots
from project_first_village_initialization import BUILD, completed_prefix_cases


SOURCE_SHA256 = '31929f003a87c6e0ae4242ea9dc3eb090fba00bb641eedad0526c1afc651b48e'
PUBLIC = {'Minion': 'minion', 'Confessor': 'confessor', 'Empath': 'lover',
          'Tracker': 'hunter', 'Shugenja': 'enlightened'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def project_report(source, asset_source):
    digest = sha(source)
    assert SOURCE_SHA256 and digest == SOURCE_SHA256
    report = expand_snapshots(json.loads(source.read_text(encoding='utf-8')))
    assert report['schema_version'] == 'first_village_publication_init_v1'
    assert report['build_id'] == BUILD
    assert report['joined']['boundary']['kind'] == 'before_ordered_start'
    assert report['joined']['boundary']['rva'] == '0x36d0f9'
    assert sha(asset_source) == report['prior_report_hashes']['character_assets_audit']
    asset_rows = json.loads(asset_source.read_text(encoding='utf-8'))['records']
    assets = {r['path_id']: r for r in asset_rows}
    final = report['joined']['final']
    initial = report['initial']
    pre = report['pre_publication']['snapshot']
    rows = final['actors']
    positions = {r['identity']: i+1 for i, r in enumerate(rows)}
    assert len(rows) == len(positions) == len(report['act_init_returns']) == 5
    assert final['publication']['actors'] == list(positions)
    assert final['publication']['identity'] != final['publication']['board_identity']
    initialization = completed_prefix_cases(report)[-1]['context']
    initialization['caller']['failure'] = None
    binding_by_data = {a['identity']: a for a in initial['asset_bindings'].values()}
    data_roles = {str(k): PUBLIC[a['role_type']] for k, a in binding_by_data.items()}
    action_classes = {str(c['clone']): binding_by_data[c['after']['data']]['role_type']
                      for c in report['initializers']}

    referenced = set()
    def names(ids):
        out = []
        for identity in ids:
            row = assets[identity]
            assert row['name'] == row['characterName'] and row['characterName']
            referenced.add(identity)
            out.append(row['characterName'])
        return out
    pools = final['pools']
    semantic_pools = {'unique': names(pools['unique_pool']['items']),
                      'duplicate': names(pools['duplicate_pool']['items']),
                      'must_include': names(pools['must_include']['items']),
                      'script': {k: names(v) for k, v in zip(
                          ('villagers', 'outcasts', 'minions', 'demons'), final['rosters'])}}
    assert len({assets[k]['characterName'] for k in referenced}) == len(referenced)
    pending = {str(r['identity']): positions[r['owner']] for r in final['continuations']}
    assert len(pending) == 5 and final['continuations'] == pre['continuations']
    context = {'version': 'setup_action_bridge_native_v2', 'initialization': initialization,
               'data_roles': data_roles, 'action_classes': action_classes,
               'action_classes_and_caches_verified': True, 'on_trigger_absent': True,
               'final_services_inert': True, 'post_initialization_ui_verified': True,
               'pools': semantic_pools, 'spy_caches': {},
               'ui': {str(i): {'pickable_active': False, 'rip_active': False,
                               'disguise_icon_active': None} for i in positions.values()},
               'next_logical_id': max(int(k) for k in pending)+1}
    expected_actors, bodies, calls, versions = [], {}, [], {}
    before_by_actor = {r['identity']: r for r in pre['actors']}
    for row, call in zip(rows, report['act_init_returns']):
        identity = row['identity']; position = positions[identity]
        before = before_by_actor[identity]
        assert call['completed'] and call['actor_identity'] == identity and call['trigger'] == 3
        assert row['bluff'] == row['bluff_role'] == row['register_as'] == row['runtime'] == 0
        assert row['revealed'] == row['started'] == row['killed_demon'] == row['dead_prefab'] == 0
        assert row['state_callback'] == 0 and row['acted_infos_storage']['count'] == 0
        klass = action_classes[str(row['role'])]
        native_calls = call['concrete_calls']
        methods = [c['method'] for c in native_calls]
        route = klass+'.BluffAct' if klass != 'Minion' else 'Role.BluffAct'
        lying = route in methods
        assert (klass+'.Act' in methods) or lying
        assert before['alignment'] == (20 if lying else 10)
        assert before['status_storage']['active_values'] == []
        role = PUBLIC[klass]
        status = row['status_storage']
        assert not status['target'] or status['target'] in positions
        application = None
        if klass == 'Confessor':
            assert 'Confessor.OnInit' in methods and 'CharacterStatuses.AddStatus' in methods
            assert status['active_values'] == [25] and status['target'] == 0
            application = {'status': 25, 'accepted': True, 'inserted': True, 'target_after': None}
        else:
            assert status['active_values'] == []
        versions[str(position)] = {'before': before['status_storage']['version'], 'after': status['version']}
        assert versions[str(position)] == {'before': 1, 'after': 2 if application else 1}
        callbacks = [{'trigger': 'init', 'slot': 'real', 'role': role,
                      'dispatch': 'bluff_act' if lying else 'act', 'status_application': application}]
        calls.append({'position': position, 'trigger': 'init', 'init_callbacks': callbacks,
                      'start_callbacks': [], 'initial_lying': lying})
        expected_actors.append({'position': position, 'data_role': data_roles[str(row['data'])],
            'action_role': role, 'runtime_evil': row['alignment'] == 20,
            'bluff': {'kind': 'null'}, 'bluff_role': None, 'register_as': None,
            'statuses': {'values': status['active_values'], 'resistance': status['resistance_values'],
                         'target_position': positions.get(status['target']) if status['target'] else None},
            'remaining_continuations': sum(r['owner'] == identity for r in final['continuations']),
            'on_trigger_subscribed': False, 'character_start_acted': bool(row['started'])})
        bodies[str(position)] = {'state': row['state'], 'previous_state': row['previous'],
            'revealed': bool(row['revealed']), 'killed_by_demon': bool(row['killed_demon']),
            'pickable_uses': row['uses'], 'acted_info_count': row['acted_infos_storage']['count'],
            'created_dead_presentation': False, 'on_state_change_subscribed': False}
    return {'schema_version': 1, 'build_id': BUILD, 'native_report_sha256': digest,
        'asset_report_sha256': sha(asset_source),
        'scope': 'One retained original N5 publication/ActInit witness. Rust compares semantic actors, bodies, dispatch/status effects, current data/order and retained pending identities. Slot labels/UI booleans are declared adapter inputs. Native-only: shallow-copy physical publication/list/CPU/closure/callback identities and status storage/version. No ordered Start, acquisition, Day, pixels, weights or PlayerHistory admission.',
        'context': context, 'expected': {'actors': expected_actors, 'bodies': bodies, 'calls': calls,
            'current_data': {str(positions[r['identity']]): r['data'] for r in rows},
            'current_order': [positions[a] for a in final['publication']['actors']],
            'pools': semantic_pools, 'pending': pending, 'status_versions': versions}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path); parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = project_report(args.source, args.assets)
    assert args.output.parent.is_dir()
    args.output.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'actors': len(value['expected']['actors']), 'native_report_sha256': value['native_report_sha256']}))
