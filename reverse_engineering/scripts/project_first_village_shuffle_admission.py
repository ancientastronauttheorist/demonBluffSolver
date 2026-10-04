"""No-Rust projection of conditional N5 Manage return and mixed wait admission.

Native ownership/storage and the conditional runtime callback stay oracle-only.
This does not produce an acquisition, Day or legitimate player history.
"""
import argparse
import copy
import json
import struct
from pathlib import Path

from audit_report_snapshots import expand_snapshots
from project_first_village_start_queue import (
    BUILD, ASSET_SHA256, REPORT_SHA256 as START_REPORT_SHA256,
    initialization_context, semantic_checkpoint, sha,
)


# Filled only after the native producer and independent review freeze.
REPORT_SHA256 = '512c221a8362d0fcb1f43cfb0f553d2dd3f63f86a9b7ae8105180b50aeac1565'
NATIVE_SOURCE_SHA256 = 'f1a4c5bf5b14da269f0dd55b62349bb16e078ecf6cf1a0dbe179550fc416c4c5'
NATIVE_SOURCE = 'reverse_engineering/scripts/audit_first_village_shuffle_admission.py'
ACQUISITION_BITS = 0x3E99999A
SHUFFLE_BITS = 0x3F000000


def decoded_report(path):
    value = json.loads(path.read_text(encoding='utf-8'))
    return expand_snapshots(value) if 'snapshot_encoding' in value else value


def semantic_prefix(report, assets):
    """Reuse the independent prefix projection at its actual native boundary."""
    final = report['joined']['final']
    before = report['before_on_setup']
    assert before == report['before_start']
    for key in ('actors', 'continuations', 'runtime_roles', 'source_role_saved_fields',
                'publication', 'pools', 'ordered_start', 'ordered_source_bindings'):
        assert final[key] == before[key], key
    assert final['conditional_on_setup'] == before['conditional_on_setup']
    assert final['conditional_on_setup']['identity'] == 0
    # The helper consumes the pre-tail snapshot, not a manufactured final state.
    view = copy.deepcopy(report)
    view['joined']['final'] = copy.deepcopy(before)
    initialization = initialization_context(view)
    positions, data_roles, action_classes, expected = semantic_checkpoint(view, assets)
    assert initialization['caller']['state']['callback'] is None
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
    queue = final['engine_queue']
    storage = final['engine_storage']
    owners = final['physical_owners']
    records = final['native_records']
    admissions = report['engine_admissions']
    assert len(owners) == len(records) == len(admissions) == len(queue['entries']) == 6
    assert queue['actual_tree_order'] == storage['actual_tree_order'] == list(range(6))
    assert [entry['id'] for entry in queue['entries']] == list(range(6))
    by_iterator = {row['iterator']: row for row in records}
    by_actor = {row['managed_actor']: row for row in owners}
    assert len(by_iterator) == len(by_actor) == 6
    assert len({row['native_owner'] for row in owners}) == len({row['key'] for row in owners}) == 6
    assert len({row['payload'] for row in records}) == 6
    manager = final['conditional_on_setup']['owner']
    shuffle = final['shuffle_continuation']
    assert shuffle['state'] == 1 and shuffle['wait_bits'] == SHUFFLE_BITS
    assert report['shuffle_first_yield']['iterator'] == shuffle['identity']
    assert report['shuffle_first_yield']['owner'] == manager
    assert [row['kind'] for row in admissions] == ['acquisition'] * 5 + ['shuffle']
    assert {str(row['iterator']) for row in admissions[:5]} == set(pending)
    assert [row['iterator'] for row in admissions[:5]] == [row['iterator'] for row in report['initializers']]
    assert admissions[-1]['iterator'] == shuffle['identity'] and admissions[-1]['actor'] == manager
    callback_pairs = set()
    entries, producers, links = [], [], []
    for index, (admission, native) in enumerate(zip(admissions, queue['entries'])):
        actor, iterator = admission['actor'], admission['iterator']
        record, owner = by_iterator[iterator], by_actor[actor]
        assert record['managed_actor'] == actor
        assert record['payload'] == admission['payload']
        assert record['native_owner'] == owner['native_owner'] == admission['native_owner']
        assert record['owner_key'] == owner['key'] == admission['owner_key']
        assert owner['payloads'] == [record['payload']]
        assert record['owner_linked'] and record['reference_count'] == 1
        assert record['gc_handle'] and record['cached_enumerator'] == iterator
        payload = bytes.fromhex(storage['payload_bytes'][str(record['payload'])])
        assert struct.unpack_from('<Q', payload, 0x10)[0] == record['gc_handle']
        assert struct.unpack_from('<Q', payload, 0x20)[0] == iterator
        assert struct.unpack_from('<Q', payload, 0x58)[0] == owner['native_owner']
        assert struct.unpack_from('<i', payload, 0x60)[0] == 1
        assert storage['gc_targets'][str(record['gc_handle'])] == iterator
        matches = []
        for raw_hex in storage['tree_node_bytes'].values():
            raw = bytes.fromhex(raw_hex)
            if struct.unpack_from('<Q', raw, 0x38)[0] == record['payload']:
                matches.append(raw[0x20:0x60])
        assert len(matches) == 1
        raw = matches[0]
        bits = SHUFFLE_BITS if index == 5 else ACQUISITION_BITS
        producer = admission['producer']
        assert producer == {'time': report['domain']['producer_time'],
                            'frame': report['domain']['producer_signed_frame'],
                            'duration_bits': bits}
        duration = struct.unpack('<f', struct.pack('<I', bits))[0]
        frame = (producer['frame'] + 1 + (1 << 63)) % (1 << 64) - (1 << 63)
        timing = {'deadline': producer['time'] + duration, 'frame_threshold': frame,
                  'phase_mask': 0xA, 'insertion_generation': queue['generation']}
        assert struct.unpack_from('<dq', raw, 0) == (timing['deadline'], frame)
        assert struct.unpack_from('<III', raw, 0x30) == (owner['key'], 0xA, queue['generation'])
        callback_pairs.add(struct.unpack_from('<QQ', raw, 0x20))
        assert native == {'id': admission['id'], 'kind': admission['kind'],
                          'iterator': admission['iterator_label'], 'deadline': timing['deadline'],
                          'frame_threshold': frame, 'generation': queue['generation'], 'phase_mask': 0xA}
        assert admission['deadline'] == timing['deadline'] and admission['frame_threshold'] == frame
        assert admission['generation'] == queue['generation']
        assert admission['queue']['actual_tree_order'] == list(range(index + 1))
        if index == 5:
            logical_id, position = next_id, None
        else:
            logical_id, position = iterator, positions[actor]
            assert pending[str(logical_id)] == position
        entries.append({'logical_id': logical_id, 'timing': timing, 'release_present': True})
        producers.append({'logical_id': logical_id, 'position': position, 'kind': admission['kind'],
                          'producer': {'rule_version': 'unity_wait_eligibility_native_v1',
                                       'duration': duration, 'producer_time': producer['time'],
                                       'producer_frame_counter': producer['frame'],
                                       'insertion_generation': queue['generation']},
                          'expected_timing': timing})
        links.append({**{key: admission[key] for key in ('id', 'kind', 'iterator', 'actor',
                                                        'payload', 'native_owner', 'owner_key')},
                      'logical_id': logical_id})
    assert len(callback_pairs) == 1
    callback, release = next(iter(callback_pairs))
    assert callback and release - callback == 0xA0
    assert entries[-2]['timing']['deadline'] < entries[-1]['timing']['deadline']
    return {'rule_version': 'unity_wait_queue_native_v1', 'generation': queue['generation'],
            'next_id': next_id + 1, 'entries': entries}, producers, links


def project_report(source, assets_source, native_source):
    assert REPORT_SHA256 and sha(source) == REPORT_SHA256
    assert NATIVE_SOURCE_SHA256 and sha(native_source) == NATIVE_SOURCE_SHA256
    assert sha(assets_source) == ASSET_SHA256
    report = decoded_report(source)
    assert report['schema_version'] == 'first_village_shuffle_admission_v1'
    assert report['build_id'] == BUILD
    assert report['source_hashes'][NATIVE_SOURCE] == NATIVE_SOURCE_SHA256
    assert report['prior_report_hashes']['first_village_start_queue'] == START_REPORT_SHA256
    assert report['prior_report_hashes']['character_assets_audit'] == ASSET_SHA256
    assert report['joined']['boundary'] is None and report['joined']['failure'] is None
    assert report['completion']['returned'] is True
    completion = report['completion']
    entry, before_return = completion['entry_cpu'], completion['before_return_cpu']
    returned = completion['final_cpu']['managed']
    assert before_return['RSP'] == entry['RSP']
    assert returned['RSP'] == entry['RSP'] + 8
    assert returned['RIP'] == completion['root_return_sentinel']
    preserved = ('RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15')
    preserved += tuple('XMM' + str(i) for i in range(6, 16))
    assert all(entry[key] == before_return[key] == returned[key] for key in preserved)
    assert report['counters']['on_setup_calls'] == report['counters']['start_calls'] == 0
    assert report['counters']['queue_records'] == report['counters']['physical_owners'] == 6
    assert report['shuffle_literal']['duration_bits'] == SHUFFLE_BITS
    assets = {row['path_id']: row for row in json.loads(assets_source.read_text(encoding='utf-8'))['records']}
    context, expected, positions = semantic_prefix(report, assets)
    queue, producers, links = mixed_admission(report, positions, expected['pending'], context['next_logical_id'])
    pin = next(row for row in report['selected_instruction_assertions'] if row['rva'] == '0x36d325')
    assert pin['mnemonic'] == 'mov' and pin['operands'] == 'dword ptr [rbx + 0x10], 0'
    return {
        'schema_version': 1, 'build_id': BUILD, 'native_report_sha256': REPORT_SHA256,
        'native_source_sha256': NATIVE_SOURCE_SHA256, 'asset_report_sha256': ASSET_SHA256,
        'scope': 'Conditional original row0/poolrow0 N5 retained Manage return with explicitly supplied '
            'runtime onSetup=null. This does not establish original subscriber absence or enabled-component '
            'lifetime. Compare five semantic actors and ordered Start matching, plus six heterogeneous '
            'first-wait admission timings/order. Full setup-action replay still ends after modeled Start; '
            'generic caller registration state0 is distinct from native Shuffle first-yield state1. '
            'A fresh queue-local label represents Shuffle, independently of native handles; the five '
            'DelayReveal registry labels are retained. Native owners/tree/payload/GC/CPU/storage remain '
            'oracle validation only. Scene slots/UI booleans are supplied adapter inputs. No acquisition '
            'resume, Shuffle state1/events, Day, rendered public capture, PlayerHistory, world sets or prior weights.',
        'context': context, 'expected': expected, 'queue': queue, 'admission_producers': producers,
        'expected_caller': {'returned': report['completion']['returned'], 'callback_calls': 0,
                            'registration_state': 0, 'start_coroutine_calls': 1},
        'native_only': {'conditional_on_setup': copy.deepcopy(report['joined']['final']['conditional_on_setup']),
                        'shuffle_first_yield': {'identity': report['joined']['final']['shuffle_continuation']['identity'],
                                                'state': 1, 'duration_bits': SHUFFLE_BITS},
                        'admission_links': links, 'native_queue_record_order': list(range(6)),
                        'physical_owners': copy.deepcopy(report['joined']['final']['physical_owners'])},
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
