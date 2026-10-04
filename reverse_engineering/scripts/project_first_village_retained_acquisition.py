"""Pure projection of one retained original N5 acquisition witness.

This privileged offline fixture compares the recorded Confessor selector path.
It is not a probability law, public observation, Day/readiness or policy input.
"""
import argparse
import copy
import json
import struct
from pathlib import Path

from project_first_village_shuffle_admission import decoded_report
from project_first_village_start_queue import (
    ASSET_SHA256, BUILD, initialization_context, semantic_checkpoint, sha,
)


REPORT_SHA256 = 'f74f10f6f9eccd5ae1af512871dc01e5c173e590290a414b59a11b74122ce9f3'
NATIVE_SOURCE_SHA256 = 'cdcc5ef15c2d36a1b961263c483b27c264e520a495f97985ad640d68df4c68cd'
NATIVE_SOURCE = 'reverse_engineering/scripts/audit_first_village_retained_acquisition.py'
V5 = 'bounded_setup_reveal_callbacks_native_v5'
SCHEDULED_V2 = 'scheduled_setup_reveal_native_v2'
QUEUE_V1 = 'unity_wait_queue_native_v1'
REGISTRY_V1 = 'logical_continuation_registry_native_v1'
WRITER_V2 = 'ordered_reveal_writer_view_native_v2'


def queue_projection(native):
    assert native['actual_tree_order'] == [row['id'] for row in native['entries']]
    assert len(set(native['actual_tree_order'])) == len(native['entries'])
    entries = []
    for row in native['entries']:
        assert row['phase_mask'] == 0xA and row['id'] < native['next_id']
        entries.append({'logical_id': row['id'], 'timing': {
            'deadline': row['deadline'], 'frame_threshold': row['frame_threshold'],
            'phase_mask': row['phase_mask'], 'insertion_generation': row['generation']},
            'release_present': True})
    return {'rule_version': QUEUE_V1, 'generation': native['generation'],
            'next_id': native['next_id'], 'entries': entries}


def check_storage(snapshot, queue):
    """Check live queue/payload/owner identities, retaining released storage."""
    storage = snapshot['engine_storage']
    assert storage['actual_tree_order'] == queue['actual_tree_order']
    records = snapshot['native_records']
    registrations = storage['nested_runtime']['registrations']
    owners = snapshot['physical_owners']
    by_label = {row['iterator_label']: row for row in registrations}
    by_iterator = {row['iterator']: row for row in records}
    by_actor = {row['managed_actor']: row for row in owners}
    assert len(by_label) == len(by_iterator) == len(registrations) == 8
    assert len(by_actor) == len(owners) == 7
    assert len({row['payload'] for row in records}) == 8
    assert len({row['key'] for row in owners}) == 7
    assert not storage['nested_runtime']['engine_frames']
    assert not storage['nested_runtime']['managed_step_stack']
    assert not storage['nested_runtime']['borrowed_payloads']
    assert storage['nested_runtime']['depth'] == 0
    linked = set()
    for owner in owners:
        raw = bytes.fromhex(storage['owner_bytes'][str(owner['native_owner'])])
        head = owner['native_owner'] + 0x70
        assert owner['list_head'] == head
        assert struct.unpack_from('<I', raw, 8)[0] == owner['key']
        payloads = owner['payloads']
        assert struct.unpack_from('<QQ', raw, 0x70) == (
            payloads[0] if payloads else head, payloads[-1] if payloads else head)
        for index, payload in enumerate(payloads):
            assert payload not in linked
            linked.add(payload)
            body = bytes.fromhex(storage['payload_bytes'][str(payload)])
            assert struct.unpack_from('<QQ', body) == (
                payloads[index+1] if index+1 < len(payloads) else head,
                payloads[index-1] if index else head)
    for record in records:
        registration = next(row for row in registrations if row['iterator'] == record['iterator'])
        owner = by_actor[record['managed_actor']]
        assert registration['payload'] == record['payload']
        assert registration['native_owner'] == record['native_owner'] == owner['native_owner']
        assert registration['owner_key'] == record['owner_key'] == owner['key']
        assert registration['managed_owner'] == record['managed_actor']
        raw = bytes.fromhex(storage['payload_bytes'][str(record['payload'])])
        assert struct.unpack_from('<Q', raw, 0x10)[0] == record['gc_handle']
        assert struct.unpack_from('<Q', raw, 0x20)[0] == record['cached_enumerator'] == record['iterator']
        assert struct.unpack_from('<Q', raw, 0x58)[0] == owner['native_owner']
        assert struct.unpack_from('<i', raw, 0x60)[0] == record['reference_count']
        assert record['owner_linked'] == (record['payload'] in linked)
        if record['owner_linked']:
            assert record['reference_count'] == 1 and record['gc_handle']
            assert storage['gc_targets'][str(record['gc_handle'])] == record['iterator']
        else:
            assert record['reference_count'] == record['gc_handle'] == 0
    callback_pairs = set()
    for row in queue['entries']:
        registration = by_label[row['iterator']]
        record = by_iterator[registration['iterator']]
        assert record['owner_linked']
        matches = [bytes.fromhex(raw)[0x20:0x60]
                   for raw in storage['tree_node_bytes'].values()
                   if struct.unpack_from('<Q', bytes.fromhex(raw), 0x38)[0] == record['payload']]
        assert len(matches) == 1
        raw = matches[0]
        assert struct.unpack_from('<dq', raw) == (row['deadline'], row['frame_threshold'])
        assert struct.unpack_from('<III', raw, 0x30) == (
            record['owner_key'], row['phase_mask'], row['generation'])
        callback_pairs.add(struct.unpack_from('<QQ', raw, 0x20))
    assert len(callback_pairs) == 1
    callback, release = next(iter(callback_pairs))
    assert callback and release - callback == 0xA0


def setup_prefix(report, assets):
    view = copy.deepcopy(report)
    # The final Init snapshot is the actual joint checkpoint before publication.
    view['pre_publication'] = report['initializers'][-1]['snapshot']
    view['joined']['final'] = report['before_start']
    initialization = initialization_context(view)
    positions, data_roles, classes, expected = semantic_checkpoint(view, assets)
    alias = 'onSetup:' + str(report['installed_delegate']['identity'])
    assert alias not in initialization['object_identities']
    initialization['caller']['state']['callback'] = alias
    initialization['caller']['callbacks'] = [alias]
    ui = {str(position): {'pickable_active': False, 'rip_active': False,
                         'disguise_icon_active': None} for position in positions.values()}
    context = {'version': 'setup_action_bridge_native_v2', 'initialization': initialization,
               'data_roles': data_roles, 'action_classes': classes,
               'action_classes_and_caches_verified': True, 'on_trigger_absent': True,
               'final_services_inert': True, 'post_initialization_ui_verified': True,
               'pools': expected['pools'], 'spy_caches': {}, 'ui': ui,
               'next_logical_id': max(int(identity) for identity in expected['pending']) + 1}
    return context, expected, positions


def chronology(report):
    drains = report['drains']
    assert len(drains) == 9
    compact = []
    for index, row in enumerate(drains):
        before, after = row['before'], row['after']
        assert row['failure'] is None
        assert row['input']['generation_before'] == before['generation']
        assert after['generation'] == (before['generation'] + 1) % (1 << 32)
        if index:
            assert before == drains[index-1]['after']
        callbacks = [event['id'] for event in row['events'] if event['kind'] == 'native_wait_callback']
        insertions = [event for event in row['events'] if event['kind'] == 'native_wait_inserted']
        visits = [event['id'] for event in row['events'] if event['kind'] == 'native_wait_visit']
        erases = [event['id'] for event in row['events'] if event['kind'] == 'native_wait_erase']
        responses, releases, managed_results, release_evidence = {}, [], {}, []
        callback_indices = [i for i, event in enumerate(row['events'])
                            if event['kind'] == 'native_wait_callback']
        for ordinal, event_index in enumerate(callback_indices):
            callback = row['events'][event_index]
            end = callback_indices[ordinal+1] if ordinal+1 < len(callback_indices) else len(row['events'])
            events = row['events'][event_index:end]
            identity, iterator = callback['id'], callback['iterator']
            lookup = [event for event in row['events'][:event_index]
                      if event['kind'] == 'owner_lookup_gateway' and event['id'] == identity]
            assert len(lookup) == 1 and lookup[0]['outcome'] == 'valid'
            returns = [event for event in events if event['kind'] == 'managed_move_next_return']
            assert len(returns) == 1 and returns[0]['iterator'] == iterator
            managed_result = returns[0]['result']
            reference_releases = [event for event in events if event['kind'] == 'native_reference_release']
            assert all(event['iterator'] == iterator for event in reference_releases)
            assert [event['before'] for event in reference_releases] == ([2, 2] if managed_result else [2, 2, 1])
            # The retained valid-owner dispatcher and consumer execute the native
            # release slot. The pinned one-shot consumer releases a dispatched
            # record only for ABI result exactly one. This is release-derived;
            # managed MoveNext's byte is retained separately and may be zero.
            release_evidence.append({'logical_id': identity, 'iterator': iterator,
                                     'reference_counts_before': [event['before'] for event in reference_releases],
                                     'dispatch_result_basis': 'valid_owner_native_post_dispatch_release_exact_one'})
            mutations = []
            for insertion in (event for event in events if event['kind'] == 'native_wait_inserted'):
                producer = insertion['producer']
                assert producer['duration_bits'] == 0x3D4CCCCD
                assert producer['time'] == row['input']['time'] and producer['frame'] == row['input']['frame']
                mutations.append({'operation': 'insert',
                                  'duration': struct.unpack('<f', struct.pack('<I', producer['duration_bits']))[0],
                                  'producer_time': producer['time'],
                                  'producer_frame_counter': producer['frame'], 'release_present': True})
            responses[str(identity)] = {'owner': 'resolved', 'callback_result': 1, 'mutations': mutations}
            managed_results[str(identity)] = managed_result
            releases.append(identity)
        assert erases == callbacks == releases
        assert after['next_id'] == before['next_id'] + len(insertions)
        if index < 3:
            assert not callbacks and not insertions
            assert before['entries'] == after['entries']
        elif index < 8:
            assert len(callbacks) == 1
            entry = next(entry for entry in before['entries'] if entry['id'] == callbacks[0])
            assert entry['kind'] == 'animation'
            assert len(insertions) == (1 if index < 7 else 0)
            for insertion in insertions:
                assert insertion['id'] not in visits
                added = next(entry for entry in after['entries'] if entry['id'] == insertion['id'])
                assert added['kind'] == 'animation'
                assert added['generation'] == after['generation']
                assert added['frame_threshold'] == row['input']['frame'] + 1
                assert added['deadline'] == row['input']['time'] + struct.unpack('<f', struct.pack('<I', 0x3D4CCCCD))[0]
        else:
            assert len(callbacks) == 5 and not insertions
            assert all(next(entry for entry in before['entries'] if entry['id'] == identity)['kind'] == 'acquisition'
                       for identity in callbacks)
        compact.append({'input': row['input'], 'before': queue_projection(before),
                        'after': queue_projection(after), 'callback_ids': callbacks,
                        'inserted_ids': [event['id'] for event in insertions], 'visited_ids': visits,
                        'erased_ids': erases, 'released_ids': releases, 'responses': responses,
                        'managed_move_next_results': managed_results,
                        'native_release_evidence': release_evidence})
    assert drains[0]['before']['generation'] == 0
    assert report['counters']['native_wait_insertions'] == drains[-1]['after']['next_id']
    resumes = report['resume_calls']
    assert [row['kind'] for row in resumes] == ['animation'] * 5 + ['acquisition'] * 5
    assert [row['result'] for row in resumes] == [1] * 4 + [0] * 6
    assert len({row['iterator'] for row in resumes[:5]}) == 1
    return compact


def final_semantics(report, setup, positions, mapping):
    final = report['completion']['managed']
    initial = report['before_consumption']
    binding_by_data = {row['identity']: (int(asset), row)
                       for asset, row in initial['asset_bindings'].items()}
    initial_by_actor = {row['identity']: row for row in initial['actors']}
    registrations = {row['iterator']: row for row in report['coroutine_registrations']}
    actors, versions, callbacks = [], {}, []
    for row, old, resume in zip(final['actors'], setup['actors'], report['resume_calls'][-5:]):
        actor = row['identity']; position = positions[actor]
        before = initial_by_actor[actor]
        assert resume['actor'] == actor and resume['result'] == 0
        assert row['data'] == before['data'] and row['role'] == before['role']
        assert row['alignment'] == before['alignment']
        for key in ('state', 'previous', 'revealed', 'started', 'runtime', 'uses', 'act',
                    'saved_act', 'register_as', 'killed_demon', 'dead_prefab', 'state_callback',
                    'acted_infos_storage', 'hover_infos_storage'):
            assert row[key] == before[key], key
        status, prior = row['status_storage'], before['status_storage']
        for key in ('status', 'active', 'active_backing', 'resistance', 'resistance_backing',
                    'resistance_values', 'resistance_version', 'resistance_count'):
            assert status[key] == prior[key], key
        assert status['count'] == len(status['active_values']) and status['target'] == 0
        result = copy.deepcopy(old)
        result['remaining_continuations'] = 0
        result['statuses'] = {'values': status['active_values'], 'resistance': status['resistance_values'],
                              'target_position': None}
        if row['bluff']:
            asset_id, binding = binding_by_data[row['bluff']]
            assert asset_id == 21614 and binding['role_type'] == 'Confessor'
            assert row['bluff_role'] == report['clone_calls'][0]['result']
            result['bluff'] = {'kind': 'live', 'role': 'confessor'}
            result['bluff_role'] = 'confessor'
        else:
            assert row['bluff_role'] == 0
        actors.append(result)
        versions[str(position)] = {'before': prior['version'], 'after': status['version']}
        native_calls = report['acquisition_calls']
        calls = []
        for trigger, name in ((3, 'init'), (7, 'after_round_start')):
            role_calls = [call for call in native_calls if call.get('role_method_rva') == '0x368790'
                          and call['receiver'] == actor and call['r8'] == trigger]
            assert len(role_calls) == (2 if row['bluff'] else 1)
            for call in role_calls:
                copied = call['rdx'] == row['bluff_role']
                assert copied or call['rdx'] == row['role']
                assert call['r9'] == (10 if copied else 0)
                role = 'confessor' if copied else result['action_role']
                application = None
                if trigger == 3 and role == 'confessor':
                    add = [entry for entry in native_calls if entry.get('role_method_rva') == '0x363aa0'
                           and entry['receiver'] == status['status']]
                    assert len(add) == 1 and add[0]['rdx'] == 25 and add[0]['r8'] == actor and add[0]['r9'] == 0
                    application = {'status': 25, 'accepted': True,
                                   'inserted': 25 not in prior['active_values'], 'target_after': None}
                calls.append({'trigger': name, 'slot': 'bluff' if copied else 'real', 'role': role,
                              'dispatch': 'bluff_act' if copied else 'act', 'status_application': application})
        inserted = sum(bool(call['status_application'] and call['status_application']['inserted']) for call in calls)
        assert status['version'] == prior['version'] + inserted
        assert status['active_values'] == ([25] if position in (1, 2) else [])
        mapped = next(item for item in mapping if item['position'] == position)
        assert registrations[resume['iterator']]['payload'] == mapped['payload']
        callbacks.append({'logical_id': mapped['logical_id'], 'position': position, 'callbacks': calls})
    assert final['pools'] == initial['pools'] and final['rosters'] == initial['rosters']
    assert final['current_script_fields'] == initial['current_script_fields']
    assert final['current_script_identity'] == initial['current_script_identity']
    assert final['source_role_saved_fields'] == initial['source_role_saved_fields']
    return {'actors': actors, 'bodies': setup['bodies'], 'pools': setup['pools'],
            'current_order': setup['current_order'], 'current_data': setup['current_data'],
            'pending': {}, 'status_versions': versions, 'callbacks': callbacks}


def project_report(source, assets_source, native_source):
    assert sha(source) == REPORT_SHA256 and sha(native_source) == NATIVE_SOURCE_SHA256
    assert sha(assets_source) == ASSET_SHA256
    report = decoded_report(source)
    assert report['schema_version'] == 'first_village_retained_acquisition_v1'
    assert report['build_id'] == BUILD
    assert report['source_hashes'][NATIVE_SOURCE] == NATIVE_SOURCE_SHA256
    assert report['prior_report_hashes']['character_assets_audit'] == ASSET_SHA256
    assets = {row['path_id']: row for row in json.loads(assets_source.read_text(encoding='utf-8'))['records']}
    setup_context, expected_setup, positions = setup_prefix(report, assets)
    history = chronology(report)
    drain = report['drains'][-1]
    before = report['drains'][-2]['managed']
    final = report['completion']['managed']
    assert drain['before'] == before['engine_queue']
    assert drain['after'] == report['completion']['queue'] == final['engine_queue']
    check_storage(before, drain['before'])
    check_storage(final, drain['after'])
    assert before['actors'] == report['before_consumption']['actors']
    assert before['pools'] == report['before_consumption']['pools']
    registrations = report['coroutine_registrations']
    labels = {row['iterator_label']: row for row in registrations}
    mapping = []
    pending, deferred = {}, {}
    for entry in drain['before']['entries']:
        registration = labels[entry['iterator']]
        if entry['kind'] == 'acquisition':
            position = positions[registration['managed_owner']]
            assert expected_setup['pending'][str(registration['iterator'])] == position
            pending[str(entry['id'])] = position
            mapping.append({'logical_id': entry['id'], 'native_iterator': registration['iterator'],
                            'native_actor': registration['managed_owner'], 'position': position,
                            'native_display_id': before['actors'][position-1]['id'],
                            'payload': registration['payload'], 'native_owner': registration['native_owner'],
                            'owner_key': registration['owner_key']})
        else:
            assert entry['kind'] in ('audio', 'shuffle')
            deferred[str(entry['id'])] = entry['kind']
    assert len(mapping) == len(pending) == 5 and len(deferred) == 2
    assert len({row['native_iterator'] for row in mapping}) == 5
    assert {row['position'] for row in mapping} == set(positions.values())
    clones = report['clone_calls']
    assert len(clones) == 1
    clone = clones[0]
    source_binding = before['asset_bindings']['21614']
    assert clone['actor'] == mapping[0]['native_actor'] and clone['source'] == source_binding['source_role']
    assert clone['result'] not in {row['role'] for row in before['actors']}
    assert clone['result'] != clone['source']
    for row in mapping:
        binding = before['asset_bindings'][str(report['domain']['asset_order'][row['position']-1])]
        calls = [call for call in report['selector_calls'] if call['actor'] == row['native_actor']]
        assert len(calls) == 2 and all(call['receiver'] == binding['source_role'] for call in calls)
        assert [call['caller_return_rva'] for call in calls] == ['0x368472', '0x368536']
        assert [call['rva'] for call in calls] == ['0x3712b0', '0x3e49f0' if row['position'] == 1 else '0x3712b0']
    assert report['selector_draws'] == [
        {'minimum': 1, 'maximum_exclusive': 11, 'width': 10, 'index': 1, 'caller_return_rva': '0x3e4a2b'},
        {'minimum': 0, 'maximum_exclusive': 4, 'width': 4, 'index': 0, 'caller_return_rva': '0x36c7e7'}]
    board = {'rule_version': 'twin_start_writer_native_v1', 'reveal': {
        'rule_version': V5, 'board_size': len(positions), 'trailer_mode': False,
        'pools': expected_setup['pools'], 'actors': expected_setup['actors'],
        'resumes': [], 'spy_caches': {}}, 'current_order': expected_setup['current_order'],
        'position': min(positions.values()), 'copied_slot': False, 'bodies': expected_setup['bodies']}
    initial = {'rule_version': SCHEDULED_V2, 'continuations': {'rule_version': REGISTRY_V1,
        'initial': {'rule_version': WRITER_V2, 'board': board, 'resumes': [], 'ui': setup_context['ui']},
        'pending': pending, 'next_id': drain['before']['next_id'], 'batch_ordinal': 0},
        'queue': queue_projection(drain['before']), 'deferred_waits': deferred}
    observed_order = [event['id'] for event in drain['events'] if event['kind'] == 'native_wait_callback']
    assert observed_order == [row['logical_id'] for row in mapping]
    callbacks = {}
    for row in mapping:
        lookup = [event for event in drain['events'] if event['kind'] == 'owner_lookup_gateway' and event['id'] == row['logical_id']]
        assert len(lookup) == 1 and lookup[0]['outcome'] == 'valid'
        assert lookup[0]['native_owner'] == row['native_owner'] and lookup[0]['key'] == row['owner_key']
        record = next(record for record in final['native_records'] if record['payload'] == row['payload'])
        assert record['gc_handle'] == record['reference_count'] == 0 and not record['owner_linked']
        assert row['payload'] in report['completion']['released_payloads']
        # The reviewed consumer releases only on dispatch result1. Managed
        # MoveNext's result0 is separately checked; it is not this ABI result.
        callbacks[str(row['logical_id'])] = {'same_live_owner': True, 'callback_result': 1,
            'producer_time': drain['input']['time'], 'producer_frame_counter': drain['input']['frame']}
    context = {'rule_version': SCHEDULED_V2, 'initial': initial, 'dispatch': {
        'rule_version': 'unity_wait_eligibility_native_v1', 'sampled_time': drain['input']['time'],
        'sampled_frame_counter': drain['input']['frame'], 'phase_mask': drain['input']['phase'],
        'generation_before': drain['input']['generation_before']}, 'callbacks': callbacks}
    expected = final_semantics(report, expected_setup, positions, mapping)
    expected.update({'ui': setup_context['ui'], 'queue': queue_projection(drain['after']),
                     'deferred_waits': deferred, 'next_id': drain['after']['next_id'],
                     'batch_ordinal': len(mapping), 'callback_order': observed_order,
                     'queue_visit_order': [event['id'] for event in drain['events'] if event['kind'] == 'native_wait_visit'],
                     'queue_erase_order': [event['id'] for event in drain['events'] if event['kind'] == 'native_wait_erase']})
    return {'schema_version': 1, 'build_id': BUILD, 'native_report_sha256': REPORT_SHA256,
        'native_source_sha256': NATIVE_SOURCE_SHA256, 'asset_report_sha256': ASSET_SHA256,
        'source_hashes': report['source_hashes'], 'prior_report_hashes': report['prior_report_hashes'],
        'scope': 'Privileged offline differential for one original generated N5 recorded Confessor acquisition. '
            'Preceding drain queue effects are explicit callback responses; actual Animate bodies/tweens remain native-only. '
            'Dispatch result1 is release-derived under the pinned exact-one consumer rule, separate from managed MoveNext. '
            'Native wait IDs label the complete registry/queue via an explicit bijection. '
            'Scene slots and pickable/rip/disguise UI flags are supplied adapter contracts, not captured pixels. '
            'Native name/sprite/color routing is excluded from the Rust view-tail comparison. '
            'Selected roll1/index0 has no certified probability; six Rust conditional selector outcomes remain distinct. '
            'The caller starts inside Characters, so Gameplay startup/Day/readiness provenance is not proved. '
            'Queue phase2 is an engine mask, not Gameplay phase. Audio/Shuffle remain unresumed. '
            'No PlayerHistory, legal observations, complete world sets, policy, rendering or S1 promotion.',
        'setup_context': setup_context, 'expected_setup': expected_setup,
        'context': context, 'expected': expected, 'native_logical_map': mapping,
        'native_only': {'preceding_drains': history, 'selector_draws': report['selector_draws'],
            'selector_calls': report['selector_calls'], 'clone_calls': clones,
            'source_confessor_binding': source_binding,
            'runtime_roles': final['runtime_roles'], 'released_payloads': report['completion']['released_payloads'],
            'physical_owners': final['physical_owners'], 'native_records': final['native_records'],
            'recorded_counter_contract': report['counters'], 'registered_coroutines': registrations}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('source', type=Path)
    parser.add_argument('--assets', type=Path, required=True)
    parser.add_argument('--native-source', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    assert args.output.parent.is_dir()
    result = project_report(args.source, args.assets, args.native_source)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'output': args.output.name, 'actors': len(result['expected']['actors']),
                      'generation': result['expected']['queue']['generation'],
                      'next_id': result['expected']['queue']['next_id']}))
