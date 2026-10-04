"""Independent semantic projection of the retained N5 Start/first-wait boundary.

No Rust is imported. Native scene slots and inert UI labels are declared adapter
inputs. Native owner lists, lookup keys, payloads, callbacks and tree storage stay
separate from semantic actors and history-local continuation labels.
"""
import argparse
import copy
import hashlib
import json
import struct
from pathlib import Path

from audit_report_snapshots import expand_snapshots
from project_first_village_initialization import BUILD, completed_prefix_cases, signed


# Pinned after native producer success and source/report freeze.
REPORT_SHA256 = '7439fd1370909aa79bf2575b101ff8e943941fa148bb75e15254f822a14b3a60'
NATIVE_SOURCE_SHA256 = '35ca4b6b4df2c55eab5da437d0639245353a9e878f399c091f9cd6bf8db3e39a'
ASSET_SHA256 = '1a790f521ba8ec6983bb1634333accb7280f0fe93e49efbaec1af25914479d42'
NATIVE_SOURCE = 'reverse_engineering/scripts/audit_first_village_start_queue.py'
PUBLIC = {'Minion': 'minion', 'Confessor': 'confessor', 'Empath': 'lover',
          'Tracker': 'hunter', 'Shugenja': 'enlightened'}


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def initialization_context(report):
    """Use actual full data/source registries, including unmatched Start assets."""
    context = completed_prefix_cases(report)[-1]['context']
    context['caller']['failure'] = None
    bindings = report['initial']['asset_bindings']
    classes, class_identities = {}, {}
    for binding in bindings.values():
        chain = binding['ancestors']
        names = [row['managed_role'] for row in chain]
        assert names[0] == 'Role' and names[-1] == binding['role_type']
        assert chain[-1]['identity'] == binding['source_class']
        for index, row in enumerate(chain):
            name = row['managed_role']
            prefix = names[:index+1]
            assert name not in classes or classes[name] == prefix
            assert name not in class_identities or class_identities[name] == row['identity']
            classes[name] = prefix
            class_identities[name] = row['identity']
    context['caller']['classes'] = classes
    ordered = report['ordered_source_bindings']
    assert len(ordered) == 15
    ordered_ids = [row['asset_id'] for row in ordered]
    assert len(set(ordered_ids)) == 15
    assert not set(ordered_ids) & set(report['domain']['asset_order'])
    for row in ordered:
        assert row == {'asset_id': row['asset_id'], **bindings[str(row['asset_id'])]}
        alias = 'data:'+str(row['asset_id'])
        assert context['object_identities'][alias] == row['identity']
        source = 'source:'+str(row['asset_id'])
        assert context['object_identities'][source] == row['source_role']
        assert context['data'][str(row['identity'])]['source_role'] == row['source_role']
        assert context['caller']['roles'][source] == row['role_type']
    assert report['joined']['final']['ordered_start']['asset_ids'] == ordered_ids
    context['caller']['state']['order'] = 'original_start_order'
    context['caller']['state']['arrays']['original_start_order'] = {
        'items': ['data:'+str(identity) for identity in ordered_ids], 'length': 15}
    return context


def semantic_checkpoint(report, assets):
    """Derive gameplay fields from native storage and actual concrete call traces."""
    final = report['joined']['final']
    pre = report['pre_publication']
    rows = final['actors']
    positions = {row['identity']: index+1 for index, row in enumerate(rows)}
    assert len(rows) == len(positions) == len(report['act_init_returns']) == 5
    assert final['publication']['actors'] == list(positions)
    assert final['publication']['identity'] != final['publication']['board_identity']
    assert report['before_start'] == final
    initial = report['initial']['asset_bindings']
    binding_by_data = {row['identity']: row for row in initial.values()}
    current_data = {str(positions[row['identity']]): row['data'] for row in rows}
    data_roles = {str(row['data']): PUBLIC[binding_by_data[row['data']]['role_type']]
                  for row in rows}
    action_classes = {str(call['clone']): binding_by_data[call['after']['data']]['role_type']
                      for call in report['initializers']}
    assert set(action_classes) == {str(row['role']) for row in rows}
    referenced = set()

    def names(identities):
        result = []
        for identity in identities:
            row = assets[identity]
            assert row['name'] == row['characterName'] and row['characterName']
            referenced.add(identity)
            result.append(row['characterName'])
        return result

    pools = final['pools']
    semantic_pools = {'unique': names(pools['unique_pool']['items']),
                      'duplicate': names(pools['duplicate_pool']['items']),
                      'must_include': names(pools['must_include']['items']),
                      'script': {name: names(identities) for name, identities in zip(
                          ('villagers', 'outcasts', 'minions', 'demons'), final['rosters'])}}
    assert len({assets[identity]['characterName'] for identity in referenced}) == len(referenced)
    pending = {str(row['identity']): positions[row['owner']] for row in final['continuations']}
    assert len(pending) == 5 and final['continuations'] == pre['continuations']
    before_by_actor = {row['identity']: row for row in pre['actors']}
    expected_actors, bodies, calls, versions = [], {}, [], {}
    for row, call in zip(rows, report['act_init_returns']):
        identity = row['identity']; position = positions[identity]
        before = before_by_actor[identity]
        assert call['completed'] and call['actor_identity'] == identity and call['trigger'] == 3
        assert row['bluff'] == row['bluff_role'] == row['register_as'] == row['runtime'] == 0
        assert row['revealed'] == row['started'] == row['killed_demon'] == row['dead_prefab'] == 0
        assert row['state_callback'] == 0 and row['acted_infos_storage']['count'] == 0
        klass = action_classes[str(row['role'])]
        methods = [record['method'] for record in call['concrete_calls']]
        bluff_method = klass+'.BluffAct' if klass != 'Minion' else 'Role.BluffAct'
        lying = bluff_method in methods
        assert klass+'.Act' in methods or lying
        assert before['alignment'] == (20 if lying else 10)
        assert before['status_storage']['active_values'] == []
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
        role = PUBLIC[klass]
        calls.append({'position': position, 'trigger': 'init', 'initial_lying': lying,
                      'init_callbacks': [{'trigger': 'init', 'slot': 'real', 'role': role,
                          'dispatch': 'bluff_act' if lying else 'act', 'status_application': application}],
                      'start_callbacks': []})
        expected_actors.append({'position': position, 'data_role': data_roles[str(row['data'])],
            'action_role': role, 'runtime_evil': row['alignment'] == 20,
            'bluff': {'kind': 'null'}, 'bluff_role': None, 'register_as': None,
            'statuses': {'values': [signed(value) for value in status['active_values']],
                         'resistance': [signed(value) for value in status['resistance_values']],
                         'target_position': positions[status['target']] if status['target'] else None},
            'remaining_continuations': sum(item['owner'] == identity for item in final['continuations']),
            'on_trigger_subscribed': False, 'character_start_acted': bool(row['started'])})
        bodies[str(position)] = {'state': signed(row['state']), 'previous_state': signed(row['previous']),
            'revealed': bool(row['revealed']), 'killed_by_demon': bool(row['killed_demon']),
            'pickable_uses': signed(row['uses']), 'acted_info_count': row['acted_infos_storage']['count'],
            'created_dead_presentation': False, 'on_state_change_subscribed': False}
    ordered = [row['asset_id'] for row in report['ordered_source_bindings']]
    assert report['ordered_comparisons'] == [[left, right] for left in ordered
                                           for right in report['domain']['asset_order']]
    expected = {'actors': expected_actors, 'bodies': bodies, 'calls': calls,
                'current_data': current_data, 'current_order': list(positions.values()),
                'pools': semantic_pools, 'pending': pending, 'status_versions': versions,
                'ordered_comparisons': [{'left': 'data:'+str(left), 'right': 'data:'+str(right)}
                                        for left, right in report['ordered_comparisons']]}
    return positions, data_roles, action_classes, expected


def queue_admission_checkpoint(report, positions, pending, next_id):
    """Join actual first-yield instances to five admitted native queue records.

    This constructs a state, not a drain. Native queue IDs and owner keys are
    separately retained provenance; semantic queue IDs reuse registry labels.
    """
    final = report['joined']['final']
    queue = final['engine_queue']
    records = final['native_records']
    owners = final['physical_owners']
    admissions = report['engine_admissions']
    assert len(queue['entries']) == len(records) == len(owners) == len(admissions) == 5
    assert queue['actual_tree_order'] == [entry['id'] for entry in queue['entries']]
    assert queue['generation'] == report['domain']['generation']
    by_id = {row['id']: row for row in admissions}
    by_iterator = {row['iterator']: row for row in records}
    by_owner = {row['managed_actor']: row for row in owners}
    assert len(by_id) == len(by_iterator) == len(by_owner) == 5
    assert {row['iterator'] for row in admissions} == {int(identity) for identity in pending}
    assert [row['iterator'] for row in admissions] == [row['iterator'] for row in report['initializers']]
    assert {row['native_owner'] for row in owners} == {row['native_owner'] for row in records}
    assert len({row['native_owner'] for row in owners}) == len({row['key'] for row in owners}) == 5
    assert len({row['payload'] for row in records}) == 5
    entries, producers, native_links = [], [], []
    for index, admission in enumerate(admissions):
        identity = admission['iterator']
        actor = admission['actor']
        record = by_iterator[identity]
        owner = by_owner[actor]
        assert record['managed_actor'] == actor and pending[str(identity)] == positions[actor]
        assert record['payload'] == admission['payload'] and record['owner_linked']
        assert owner['payloads'] == [record['payload']]
        assert record['native_owner'] == owner['native_owner'] == admission['native_owner']
        assert record['owner_key'] == owner['key'] == admission['owner_key']
        assert record['reference_count'] == 1 and record['gc_handle']
        assert record['cached_enumerator'] == identity
        producer = admission['producer']
        assert producer['time'] == report['domain']['producer_time']
        assert producer['frame'] == report['domain']['producer_signed_frame']
        assert producer['duration_bits'] == 0x3E99999A
        duration = struct.unpack('<f', struct.pack('<I', producer['duration_bits']))[0]
        frame_after = (producer['frame']+1+(1 << 63)) % (1 << 64)-(1 << 63)
        timing = {'deadline': producer['time']+duration, 'frame_threshold': frame_after,
                  'phase_mask': 0xA, 'insertion_generation': admission['generation']}
        native = next(row for row in admission['queue']['entries'] if row['id'] == admission['id'])
        assert native['kind'] == 'acquisition' and native['iterator'] == admission['iterator_label']
        assert native['deadline'] == timing['deadline'] and native['frame_threshold'] == timing['frame_threshold']
        assert native['phase_mask'] == timing['phase_mask'] and native['generation'] == timing['insertion_generation']
        assert admission['generation'] == queue['generation']
        assert admission['queue']['actual_tree_order'] == [row['id'] for row in admissions[:index+1]]
        assert [row['id'] for row in admission['queue']['entries']] == admission['queue']['actual_tree_order']
        producers.append({'logical_id': identity, 'position': positions[actor],
            'producer': {'rule_version': 'unity_wait_eligibility_native_v1', 'duration': duration,
                         'producer_time': producer['time'], 'producer_frame_counter': producer['frame'],
                         'insertion_generation': admission['generation']}, 'expected_timing': timing})
        native_links.append({key: admission[key] for key in ('id', 'iterator', 'iterator_label', 'actor',
                                                           'payload', 'native_owner', 'owner_key')})
    for native in queue['entries']:
        admission = by_id[native['id']]
        identity = admission['iterator']
        producer = next(row for row in producers if row['logical_id'] == identity)
        timing = producer['expected_timing']
        assert native == {'id': admission['id'], 'kind': 'acquisition',
                          'iterator': admission['iterator_label'], 'deadline': timing['deadline'],
                          'frame_threshold': timing['frame_threshold'],
                          'generation': timing['insertion_generation'], 'phase_mask': timing['phase_mask']}
        entries.append({'logical_id': identity, 'timing': timing, 'release_present': True})
    assert {entry['logical_id'] for entry in entries} == {int(identity) for identity in pending}
    semantic_queue = {'rule_version': 'unity_wait_queue_native_v1', 'generation': queue['generation'],
                      'next_id': next_id, 'entries': entries}
    return semantic_queue, producers, {
        'native_queue_record_order': queue['actual_tree_order'], 'native_queue_next_id': queue['next_id'],
        'admission_links': native_links, 'physical_owners': copy.deepcopy(owners),
        'native_records': copy.deepcopy(records)}


def project_report(source, asset_source, native_source):
    assert REPORT_SHA256 and sha(source) == REPORT_SHA256
    assert NATIVE_SOURCE_SHA256 and sha(native_source) == NATIVE_SOURCE_SHA256
    assert ASSET_SHA256 and sha(asset_source) == ASSET_SHA256
    report = expand_snapshots(json.loads(source.read_text(encoding='utf-8')))
    assert report['schema_version'] == 'first_village_start_queue_v1' and report['build_id'] == BUILD
    assert report['source_hashes'][NATIVE_SOURCE] == NATIVE_SOURCE_SHA256
    assert report['prior_report_hashes']['character_assets_audit'] == ASSET_SHA256
    assert report['joined']['boundary']['kind'] == 'before_on_setup'
    assert report['joined']['boundary']['rva'] == '0x36d2db'
    assets = {row['path_id']: row for row in json.loads(asset_source.read_text(encoding='utf-8'))['records']}
    initialization = initialization_context(report)
    positions, data_roles, action_classes, expected = semantic_checkpoint(report, assets)
    next_id = max(int(identity) for identity in expected['pending'])+1
    context = {'version': 'setup_action_bridge_native_v2', 'initialization': initialization,
               'data_roles': data_roles, 'action_classes': action_classes,
               'action_classes_and_caches_verified': True, 'on_trigger_absent': True,
               'final_services_inert': True, 'post_initialization_ui_verified': True,
               'pools': expected['pools'], 'spy_caches': {},
               'ui': {str(position): {'pickable_active': False, 'rip_active': False,
                                     'disguise_icon_active': None} for position in positions.values()},
               'next_logical_id': next_id}
    queue, producers, native_only = queue_admission_checkpoint(report, positions, expected['pending'], next_id)
    native_only.update({'original_scene_order': copy.deepcopy(report['original_scene_order']),
                        'ordered_source_bindings': copy.deepcopy(report['ordered_source_bindings']),
                        'ordered_comparisons': copy.deepcopy(report['ordered_comparisons']),
                        'boundary': copy.deepcopy(report['joined']['boundary'])})
    assert report['counters']['start_calls'] == 0
    assert report['counters']['ordered_entries'] == 15 and report['counters']['ordered_comparisons'] == 75
    assert report['counters']['queue_records'] == report['counters']['physical_owners'] == 5
    return {'schema_version': 1, 'build_id': BUILD,
            'native_report_sha256': REPORT_SHA256, 'native_source_sha256': NATIVE_SOURCE_SHA256,
            'asset_report_sha256': ASSET_SHA256,
            'scope': 'One retained original N5 witness ending at 0x36d2db before onSetup is read. '
                'Compare semantic actors/bodies/Init dispatch/status effects/current data/order/pools/pending '
                'and actual 15-entry zero-Start matching. The additional unused data/source/class bindings '
                'are asset-derived supplied pre-entry hydration and Rust typechecking inputs; their concrete '
                'role bodies did not execute. Scene-slot labels and inert UI booleans are declared adapter '
                'inputs, not certified numbered player observations. Compare five actual first-wait admissions '
                'using exact producer arithmetic and native tree occurrence order. Native-record creation, '
                'CLR/runtime and scene gateways retain the native audit\'s supplied-provider contract. Queue logical IDs reuse '
                'registry labels, independently of native queue record IDs/owner keys. Physical ownership, '
                'callback/CPU/tree/storage identity and status list versions remain native-only. The underlying '
                'generic caller adapter callback=None and inert final services are synthetic bookkeeping inputs '
                'for its Init-prefix and ordered-match subset, not proof of runtime onSetup absence, later '
                'Shuffle, or full native caller return. Full setup-action replay ends after modeled Start. No onSetup, '
                'Shuffle, actual drain, acquisition, Day, pixels, prior weights or PlayerHistory admission.',
            'context': context, 'expected': expected, 'queue': queue,
            'admission_producers': producers, 'native_only': native_only}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--native-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    value = project_report(args.source, args.assets, args.native_source)
    assert args.output.parent.is_dir()
    args.output.write_text(json.dumps(value, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({'output': args.output.name, 'actors': len(value['expected']['actors'])}))
