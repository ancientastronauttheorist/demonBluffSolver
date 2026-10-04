"""Independent projection of conditional installed-subscriber first waits.

The Rust actor bridge ends at modeled Start. Its generic callback is a request,
not animation execution; additional native owners/effects remain oracle-only.
"""
import argparse
import copy
import json
import struct
from pathlib import Path

from project_first_village_shuffle_admission import decoded_report
from project_first_village_start_queue import (
    BUILD, ASSET_SHA256, initialization_context, semantic_checkpoint, sha,
)


# Filled only after the native producer and independent review freeze.
REPORT_SHA256 = '70ade0e711d803735a8076d26e0030a9f61fc4458834a28884e375b6423b3f06'
NATIVE_SOURCE_SHA256 = '6076f9cd31eb6edd65a5c31c29b3b32e816cb9a50205a73ad52b2dd804165fbc'
NATIVE_SOURCE = 'reverse_engineering/scripts/audit_first_village_subscriber_admission.py'
BITS = {'acquisition': 0x3E99999A, 'animation': 0x3D4CCCCD,
        'audio': 0x3ECCCCCD, 'shuffle': 0x3F000000}


def semantic_prefix(report, assets):
    before, final = report['before_on_setup'], report['joined']['final']
    assert before == report['before_start']
    for key in ('actors', 'continuations', 'runtime_roles', 'source_role_saved_fields',
                'publication', 'pools', 'ordered_start', 'ordered_source_bindings'):
        assert before[key] == final[key], key
    binding = before['subscriber_binding']
    assert binding == final['subscriber_binding']
    delegate = report['installed_delegate']
    assert binding['scene_path_id'] == 137027
    assert binding['on_setup'] == delegate['identity'] != 0
    assert delegate['target'] == binding['component']
    assert delegate['method_name'] == 'Animates'
    assert delegate['fields'][:4] == [delegate['code'], delegate['code'],
                                     delegate['target'], delegate['method']]
    assert delegate['fields'][5] == delegate['target']
    assert binding['audio_on_play_sfx'] == 0
    assert len(report['callback_calls']) == 1
    call = report['callback_calls'][0]
    assert call['delegate'] == delegate['identity']
    assert all(call[key] == delegate[key] for key in ('target', 'method', 'code'))
    captures = [row for row in report['joined']['services']
                if row['service'] == 'subscriber_barrier' and row['caller_return_rva'] == '0x3753c8']
    assert len(captures) == 1
    assert captures[0]['value'] == 0
    assert captures[0]['stored_value'] == before['publication']['board_identity']
    assert captures[0]['stored_value'] != before['publication']['identity']
    view = copy.deepcopy(report)
    view['joined']['final'] = copy.deepcopy(before)
    initialization = initialization_context(view)
    positions, data_roles, action_classes, expected = semantic_checkpoint(view, assets)
    # This alias belongs only to the generic request registry, not the physical
    # actor/data/source object registry accepted by initialization.
    alias = 'onSetup:' + str(delegate['identity'])
    assert alias not in initialization['object_identities']
    initialization['caller']['state']['callback'] = alias
    initialization['caller']['callbacks'] = [alias]
    next_id = max(int(identity) for identity in expected['pending']) + 1
    context = {
        'version': 'setup_action_bridge_native_v2', 'initialization': initialization,
        'data_roles': data_roles, 'action_classes': action_classes,
        'action_classes_and_caches_verified': True, 'on_trigger_absent': True,
        'final_services_inert': True, 'post_initialization_ui_verified': True,
        'pools': expected['pools'], 'spy_caches': {},
        'ui': {str(position): {'pickable_active': False, 'rip_active': False,
                               'disguise_icon_active': None} for position in positions.values()},
        'next_logical_id': next_id,
    }
    return context, expected, positions


def mixed_admission(report, positions, pending, next_id):
    final = report['joined']['final']
    queue, storage = final['engine_queue'], final['engine_storage']
    records, owners = final['native_records'], final['physical_owners']
    admissions, registrations = report['engine_admissions'], report['coroutine_registrations']
    assert len(records) == len(admissions) == len(registrations) == len(queue['entries']) == 8
    assert len(owners) == 7
    assert queue['actual_tree_order'] == storage['actual_tree_order']
    assert queue['actual_tree_order'] == [row['id'] for row in queue['entries']]
    assert storage['nested_runtime']['registrations'] == registrations
    assert storage['nested_runtime']['depth'] == 0
    assert not storage['nested_runtime']['engine_frames']
    assert not storage['nested_runtime']['managed_step_stack']
    assert not storage['nested_runtime']['borrowed_payloads']
    assert [row['kind'] for row in registrations] == ['acquisition'] * 5 + ['animation', 'audio', 'shuffle']
    assert [row['kind'] for row in admissions] == ['acquisition'] * 5 + ['audio', 'animation', 'shuffle']
    assert [row['kind'] for row in queue['entries']] == ['animation'] + ['acquisition'] * 5 + ['audio', 'shuffle']
    assert [row['iterator'] for row in registrations[:5]] == [row['iterator'] for row in report['initializers']]
    assert {str(row['iterator']) for row in registrations[:5]} == set(pending)
    aux = registrations[5:]
    logical = {int(identity): int(identity) for identity in pending}
    logical.update({row['iterator']: next_id + index for index, row in enumerate(aux)})
    assert len(logical) == len(set(logical.values())) == 8
    assert [row['registration_ordinal'] for row in registrations] == list(range(1, 9))
    by_iterator = {row['iterator']: row for row in records}
    by_owner = {row['managed_actor']: row for row in owners}
    by_id = {row['id']: row for row in admissions}
    assert len(by_iterator) == len(by_id) == 8 and len(by_owner) == 7
    assert len({row['native_owner'] for row in owners}) == len({row['key'] for row in owners}) == 7
    assert len({row['payload'] for row in records}) == 8
    animation_owner = final['subscriber_binding']['component']
    for owner in owners:
        owned = [row for row in registrations if row['managed_owner'] == owner['managed_actor']]
        assert owner['payloads'] == [row['payload'] for row in owned]
        assert len(owned) == (2 if owner['managed_actor'] == animation_owner else 1)
        raw_owner = bytes.fromhex(storage['owner_bytes'][str(owner['native_owner'])])
        assert struct.unpack_from('<I', raw_owner, 8)[0] == owner['key']
        assert struct.unpack_from('<QQ', raw_owner, 0x70) == (owned[0]['payload'], owned[-1]['payload'])
        for index, row in enumerate(owned):
            raw = bytes.fromhex(storage['payload_bytes'][str(row['payload'])])
            head = owner['native_owner'] + 0x70
            forward = owned[index + 1]['payload'] if index + 1 < len(owned) else head
            backward = owned[index - 1]['payload'] if index else head
            assert struct.unpack_from('<QQ', raw) == (forward, backward)
    callback_pairs, entries, producers, links = set(), [], [], []
    for native in queue['entries']:
        admission = by_id[native['id']]
        iterator, actor, kind = admission['iterator'], admission['actor'], admission['kind']
        record, owner = by_iterator[iterator], by_owner[actor]
        registration = next(row for row in registrations if row['iterator'] == iterator)
        assert registration['managed_owner'] == record['managed_actor'] == actor
        assert registration['kind'] == kind and registration['iterator_label'] == admission['iterator_label']
        assert registration['payload'] == record['payload'] == admission['payload']
        assert registration['native_owner'] == record['native_owner'] == owner['native_owner'] == admission['native_owner']
        assert registration['owner_key'] == record['owner_key'] == owner['key'] == admission['owner_key']
        assert record['owner_linked'] and record['reference_count'] == 1
        assert record['gc_handle'] and record['cached_enumerator'] == iterator
        payload = bytes.fromhex(storage['payload_bytes'][str(record['payload'])])
        assert struct.unpack_from('<Q', payload, 0x10)[0] == record['gc_handle']
        assert struct.unpack_from('<Q', payload, 0x20)[0] == iterator
        assert struct.unpack_from('<Q', payload, 0x58)[0] == owner['native_owner']
        assert struct.unpack_from('<i', payload, 0x60)[0] == 1
        assert storage['gc_targets'][str(record['gc_handle'])] == iterator
        matching = [bytes.fromhex(raw)[0x20:0x60] for raw in storage['tree_node_bytes'].values()
                    if struct.unpack_from('<Q', bytes.fromhex(raw), 0x38)[0] == record['payload']]
        assert len(matching) == 1
        raw = matching[0]
        producer = admission['producer']
        assert producer == {'time': report['domain']['producer_time'],
                            'frame': report['domain']['producer_signed_frame'], 'duration_bits': BITS[kind]}
        duration = struct.unpack('<f', struct.pack('<I', BITS[kind]))[0]
        frame = (producer['frame'] + 1 + (1 << 63)) % (1 << 64) - (1 << 63)
        timing = {'deadline': producer['time'] + duration, 'frame_threshold': frame,
                  'phase_mask': 0xA, 'insertion_generation': queue['generation']}
        assert struct.unpack_from('<dq', raw) == (timing['deadline'], frame)
        assert struct.unpack_from('<III', raw, 0x30) == (owner['key'], 0xA, queue['generation'])
        callback_pairs.add(struct.unpack_from('<QQ', raw, 0x20))
        assert native == {'id': admission['id'], 'kind': kind, 'iterator': admission['iterator_label'],
                          'deadline': timing['deadline'], 'frame_threshold': frame,
                          'generation': queue['generation'], 'phase_mask': 0xA}
        assert admission['deadline'] == timing['deadline'] and admission['frame_threshold'] == frame
        assert admission['generation'] == queue['generation']
        assert admission['id'] in admission['queue']['actual_tree_order']
        assert len(admission['queue']['entries']) == admission['id'] + 1
        identity = logical[iterator]
        position = positions[actor] if kind == 'acquisition' else None
        if position is not None:
            assert pending[str(identity)] == position
        entries.append({'logical_id': identity, 'timing': timing, 'release_present': True})
        producers.append({'logical_id': identity, 'position': position, 'kind': kind,
                          'producer': {'rule_version': 'unity_wait_eligibility_native_v1',
                                       'duration': duration, 'producer_time': producer['time'],
                                       'producer_frame_counter': producer['frame'],
                                       'insertion_generation': queue['generation']}, 'expected_timing': timing})
        links.append({**{key: admission[key] for key in ('id', 'kind', 'iterator', 'actor',
                                                        'payload', 'native_owner', 'owner_key')},
                      'logical_id': identity, 'registration_ordinal': registration['registration_ordinal']})
    assert len(callback_pairs) == 1
    callback, release = next(iter(callback_pairs))
    assert callback and release - callback == 0xA0
    return {'rule_version': 'unity_wait_queue_native_v1', 'generation': queue['generation'],
            'next_id': next_id + 3, 'entries': entries}, producers, links


def project_report(source, assets_source, native_source):
    assert REPORT_SHA256 and sha(source) == REPORT_SHA256
    assert NATIVE_SOURCE_SHA256 and sha(native_source) == NATIVE_SOURCE_SHA256
    assert sha(assets_source) == ASSET_SHA256
    report = decoded_report(source)
    assert report['schema_version'] == 'first_village_subscriber_admission_v1'
    assert report['build_id'] == BUILD
    assert report['source_hashes'][NATIVE_SOURCE] == NATIVE_SOURCE_SHA256
    assert report['prior_report_hashes']['character_assets_audit'] == ASSET_SHA256
    assert report['joined']['boundary'] is None and report['joined']['failure'] is None
    completion = report['completion']
    assert completion['returned'] is True
    entry, before_return = completion['entry_cpu'], completion['before_return_cpu']
    returned = completion['final_cpu']['managed']
    assert before_return['RSP'] == entry['RSP']
    assert returned['RSP'] == entry['RSP'] + 8
    assert returned['RIP'] == completion['root_return_sentinel']
    preserved = ('RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15')
    preserved += tuple('XMM' + str(index) for index in range(6, 16))
    assert all(entry[key] == before_return[key] == returned[key] for key in preserved)
    assert report['counters']['on_setup_calls'] == len(report['callback_calls']) == 1
    assert report['counters']['queue_records'] == 8 and report['counters']['physical_owners'] == 7
    assets = {row['path_id']: row for row in json.loads(assets_source.read_text(encoding='utf-8'))['records']}
    context, expected, positions = semantic_prefix(report, assets)
    queue, producers, links = mixed_admission(report, positions, expected['pending'], context['next_logical_id'])
    return {
        'schema_version': 1, 'build_id': BUILD, 'native_report_sha256': REPORT_SHA256,
        'native_source_sha256': NATIVE_SOURCE_SHA256, 'asset_report_sha256': ASSET_SHA256,
        'scope': 'Original generated row0/poolrow0 N5 under supplied OnEnable lifecycle invocation, '
            'initially null delegate lists and AudioEvents=null. Actual installed onSetup handler and '
            'nested first steps reach normal Manage return; no original lifecycle or audio absence claim. '
            'Rust compares the five semantic actors and modeled Start prefix, one generic callback '
            'request/normal return and eight first-wait timings in native tree order. Rust does not '
            'execute Animates/audio/tween effects; inert flags cover modeled role/UI/service projection. '
            'The five DelayReveal labels are retained; three fresh queue-local labels follow actual '
            'registration order, distinct from wait-admission and tree order. Native identities, '
            'delegate/payload/GC/storage/CPU and visual requests remain oracle-only. Scene slots/UI '
            'booleans are supplied adapter inputs. No queue drain, resumed acquisition, rendering, '
            'public capture, PlayerHistory, world sets, probability or policy promotion.',
        'context': context, 'expected': expected, 'queue': queue, 'admission_producers': producers,
        'expected_caller': {'returned': True, 'callback_calls': 1,
                            'registration_state': 0, 'start_coroutine_calls': 1},
        'native_only': {'installed_delegate': report['installed_delegate'], 'admission_links': links,
                        'native_queue_record_order': report['joined']['final']['engine_queue']['actual_tree_order'],
                        'coroutine_registrations': report['coroutine_registrations'],
                        'physical_owners': report['joined']['final']['physical_owners']},
    }


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
                      'queue_records': len(result['queue']['entries'])}))
