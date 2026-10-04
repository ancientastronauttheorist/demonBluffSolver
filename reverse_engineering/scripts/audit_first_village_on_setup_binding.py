"""Pinned native-static/asset evidence for the original Characters.onSetup subscriber.

No lifecycle, delegate, tween or coroutine body is executed by this audit.
Complete original bytes/disassembly remain private; the report contains selected
operand assertions, metadata joins, hashes and authored scope statements only.
"""
import argparse
import bisect
import hashlib
import json
import re
import struct
from pathlib import Path

from audit_character_assets import ASSET_HASHES, BUILD, Cursor
from audit_first_village_profile_generation import load_inputs


def sha(data):
    return hashlib.sha256(data).hexdigest()


METHODS = {
    'CharacterShuffleAnimation$$Awake': 0x3630D0,
    'CharacterShuffleAnimation$$OnEnable': 0x363500,
    'CharacterShuffleAnimation$$OnDisable': 0x363160,
    'CharacterShuffleAnimation$$Animates': 0x363060,
    'CharacterShuffleAnimation.<Animate>d__6$$MoveNext': 0x375130,
    'CharacterShuffleAnimation.<PlayAudioDelay>d__9$$MoveNext': 0x375DB0,
    'Characters$$ManageCharacters': 0x36CE30,
}
PINS = [
    (0x36354E, 'mov', 'r14, qword ptr [rbp + 0x20]'),
    (0x363571, 'mov', 'rdi, qword ptr [r14 + 0x58]'),
    (0x363584, 'mov', 'rdx, rbp'),
    (0x36358D, 'call', '0x4d5170'),
    (0x363598, 'mov', 'rcx, rdi'),
    (0x36359B, 'call', '0x116bcc0'),
    (0x3635C6, 'mov', 'qword ptr [r14 + 0x58], rdx'),
    (0x3631AE, 'mov', 'r14, qword ptr [rbp + 0x20]'),
    (0x3631D1, 'mov', 'rdi, qword ptr [r14 + 0x58]'),
    (0x3631E4, 'mov', 'rdx, rbp'),
    (0x3631ED, 'call', '0x4d5170'),
    (0x3631FB, 'call', '0x116e070'),
    (0x363226, 'mov', 'qword ptr [r14 + 0x58], rdx'),
    (0x3630A9, 'mov', 'qword ptr [rcx], rdi'),
    (0x3630AC, 'mov', 'dword ptr [rbx + 0x10], 0'),
    (0x3630BB, 'mov', 'rdx, rbx'),
    (0x3630BE, 'mov', 'rcx, rdi'),
    (0x3630CB, 'jmp', '0x1c7f160'),
    (0x3751DC, 'jne', '0x3753d2'),
    (0x3751F2, 'mov', 'rax, qword ptr [rsi + 0x20]'),
    (0x3751FF, 'mov', 'rdx, qword ptr [rax + 0x20]'),
    (0x375281, 'call', '0x606fc0'),
    (0x37528F, 'mov', 'rdi, qword ptr [rax + 0x20]'),
    (0x3752E2, 'call', '0x1c92130'),
    (0x375359, 'mov', 'dword ptr [rbx + 0x10], 0'),
    (0x375363, 'mov', 'rdx, rbx'),
    (0x375366, 'mov', 'rcx, rsi'),
    (0x375369, 'call', '0x1c7f160'),
    (0x37544C, 'call', '0x50fb60'),
    (0x37546E, 'call', '0x1c961f0'),
    (0x37547B, 'mov', 'qword ptr [rax + 0x18], rbx'),
    (0x37549B, 'mov', 'dword ptr [rax + 0x10], 1'),
    (0x3754A2, 'mov', 'al, 1'),
    (0x375E0E, 'mov', 'edx, 0xc8'),
    (0x375E17, 'call', 'qword ptr [rax + 0x18]'),
    (0x375E37, 'call', '0x1c961f0'),
    (0x36D2DB, 'mov', 'rax, qword ptr [r12 + 0x58]'),
    (0x36D2E3, 'je', '0x36d2f0'),
    (0x36D2E5, 'mov', 'rdx, qword ptr [rax + 0x28]'),
    (0x36D2E9, 'mov', 'rcx, qword ptr [rax + 0x40]'),
    (0x36D2ED, 'call', 'qword ptr [rax + 0x18]'),
]


def audit(game_root, dumper_root):
    import UnityPy
    from capstone import CS_OP_MEM
    from capstone.x86_const import X86_REG_RIP
    raw, meta, dump, pe, cs, lock = load_inputs(game_root, dumper_root)
    starts = sorted({row['Address'] for row in meta['ScriptMethod']})
    instructions = {}
    bodies = []
    for name, start in METHODS.items():
        matches = [row for row in meta['ScriptMethod'] if row['Name'] == name]
        assert len(matches) == 1 and matches[0]['Address'] == start, name
        end = starts[bisect.bisect_right(starts, start)]
        section = next(s for s in pe.sections if s.VirtualAddress <= start < s.VirtualAddress + s.SizeOfRawData)
        assert end <= section.VirtualAddress + section.SizeOfRawData, name
        code = pe.get_data(start, end - start)
        assert len(code) == end - start
        decoded = list(cs.disasm(code, start))
        assert decoded and decoded[0].address == start
        assert all(a.address + a.size == b.address for a, b in zip(decoded, decoded[1:])), name
        decoded_end = decoded[-1].address + decoded[-1].size
        assert decoded_end <= end
        instructions.update({i.address: i for i in decoded})
        bodies.append({'name': name, 'entry_rva': hex(start),
                       'next_managed_entry_rva': hex(end), 'interval_bytes': len(code),
                       'interval_sha256': sha(code),
                       'raw_backed_section': section.Name.decode('ascii').rstrip('\0'),
                       'contiguous_decoded_prefix_end_rva': hex(decoded_end),
                       'decoded_prefix_bytes': decoded_end - start,
                       'decode_scope': 'linear instruction boundaries from verified entry, not complete control-flow coverage',
                       'fingerprint_scope': 'entry to next distinct managed entry, including padding'})
    pins = []
    for address, mnemonic, operands in PINS:
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == (mnemonic, operands), hex(address)
        pins.append({'rva': hex(address), 'mnemonic': mnemonic, 'operands': operands})

    def metadata_slot(address):
        i = instructions[address]
        op = next(o for o in i.operands if o.type == CS_OP_MEM and o.mem.base == X86_REG_RIP)
        slot = i.address + i.size + op.mem.disp
        rows = [row for key in ('ScriptMetadata', 'ScriptMetadataMethod')
                for row in meta[key] if row['Address'] == slot]
        assert len(rows) == 1, hex(slot)
        return {'instruction_rva': hex(address), 'slot_rva': hex(slot), **rows[0]}

    joins = [metadata_slot(a) for a in (0x36355B, 0x36357A, 0x3631DA,
             0x36360B, 0x3636C8, 0x363785, 0x37527A, 0x375340)]
    assert joins[0]['Name'] == 'System.Action_TypeInfo'
    for row in joins[1:3]:
        assert row['Name'] == 'Method$CharacterShuffleAnimation.Animates()'
        assert row['MethodAddress'] == 0x363060
    assert joins[-2]['Name'] == 'Method$UnityEngine.Component.GetComponent<SingleCharacterDrawAnimation>()'
    assert joins[-1]['Name'] == 'CharacterShuffleAnimation.<PlayAudioDelay>d__9_TypeInfo'
    events_join = metadata_slot(0x3635ED)
    assert events_join['Name'] == 'GameplayEvents_TypeInfo'
    joins.append(events_join)
    literals = []
    for address, expected_bits in [(0x375438, 0x3E99999A), (0x375440, 0x43C30000),
                                   (0x375463, 0x3D4CCCCD), (0x375E26, 0x3ECCCCCD)]:
        i = instructions[address]
        assert i.mnemonic == 'movss'
        slot = i.address + i.size + i.operands[1].mem.disp
        section = next(s for s in pe.sections if s.VirtualAddress <= slot < s.VirtualAddress + s.SizeOfRawData)
        assert slot + 4 <= section.VirtualAddress + section.SizeOfRawData
        value = pe.get_data(slot, 4)
        assert len(value) == 4 and struct.unpack('<I', value)[0] == expected_bits
        literals.append({'instruction_rva': hex(address), 'slot_rva': hex(slot),
                         'float32_bits': f'{expected_bits:08x}', 'exact_promoted_float': struct.unpack('<f', value)[0]})
    for text in ('public Characters characters; // 0x20', 'public AudioClip[] shuffleClips; // 0x28'):
        block = re.search(r'^public class CharacterShuffleAnimation : MonoBehaviour .*?^}', dump, re.M | re.S)
        assert block and text in block.group()
    for name, declaration in [('Characters', 'public Action onSetup; // 0x58'),
                               ('SingleCharacterDrawAnimation', 'public Transform pivot; // 0x20')]:
        block = re.search(r'^public class ' + name + r' : MonoBehaviour .*?^}', dump, re.M | re.S)
        assert block and declaration in block.group()
    events = re.search(r'^public static class GameplayEvents .*?^}', dump, re.M | re.S)
    assert events
    for declaration in ('public static Action OnNextChallenge; // 0x18',
                        'public static Action OnRestartCurrentLevel; // 0x30',
                        'public static Action OnRestartGame; // 0x38'):
        assert declaration in events.group()

    data_root = game_root / 'Demon Bluff_Data'
    asset_hashes = {}
    for filename in ('level0', 'globalgamemanagers.assets'):
        data = (data_root / filename).read_bytes()
        assert sha(data).upper() == ASSET_HASHES[filename]
        asset_hashes[filename] = sha(data)
    manager = UnityPy.load(str(data_root / 'globalgamemanagers.assets'))
    script = next(o for o in manager.objects if o.path_id == 920)
    script_fields = script.read_typetree()
    assert script.type.name == 'MonoScript'
    assert [script_fields[k] for k in ('m_ClassName', 'm_Namespace', 'm_AssemblyName')] == ['CharacterShuffleAnimation', '', 'Assembly-CSharp']
    env = UnityPy.load(str(data_root / 'level0'))
    source = next(iter(env.files.values()))
    assert [e.path for e in source.externals] == ['globalgamemanagers.assets', 'sharedassets0.assets', 'Library/unity default resources']
    objects = {o.path_id: o for o in env.objects}
    candidates = [o for o in env.objects if o.type.name == 'MonoBehaviour'
                  and o.read_typetree(check_read=False).get('m_Script') == {'m_FileID': 1, 'm_PathID': 920}]
    assert [o.path_id for o in candidates] == [137027]
    component = candidates[0]
    component_raw = component.get_raw_data()
    c = Cursor(component_raw)
    fields = {'name': c.string(), 'characters': c.pointer(), 'shuffle_clips': c.array(c.pointer)}
    assert c.offset == len(component_raw) == 84
    assert fields['characters'] == (0, 137026)
    assert sha(component_raw) == '2c919c0c0aebb29b0ceff8a44a20ab11df0fcfd1ddf9c3175314e14530753fc5'
    header = component.read_typetree(check_read=False)
    assert header['m_Enabled'] == 1 and header['m_GameObject'] == {'m_FileID': 0, 'm_PathID': 415}
    ancestors = []
    initial_go = objects[header['m_GameObject']['m_PathID']]
    assert initial_go.type.name == 'GameObject'
    initial_components = initial_go.read_typetree()['m_Component']
    assert all(row['component']['m_FileID'] == 0 for row in initial_components)
    transforms = [row['component']['m_PathID'] for row in initial_components
                  if objects[row['component']['m_PathID']].type.name in ('Transform', 'RectTransform')]
    assert transforms == [88510]
    transform_id = transforms[0]
    visited = set()
    while transform_id:
        assert transform_id not in visited, 'cyclic original scene transform hierarchy'
        visited.add(transform_id)
        transform = objects[transform_id]
        assert transform.type.name in ('Transform', 'RectTransform')
        td = transform.read_typetree()
        assert td['m_GameObject']['m_FileID'] == 0
        go_id = td['m_GameObject']['m_PathID']
        go = objects[go_id]
        assert go.type.name == 'GameObject'
        gd = go.read_typetree()
        assert {'component': {'m_FileID': 0, 'm_PathID': transform_id}} in gd['m_Component']
        ancestors.append({'transform_path_id': transform_id, 'transform_object_sha256': sha(transform.get_raw_data()),
                          'game_object_path_id': go_id, 'game_object_sha256': sha(go.get_raw_data()),
                          'name': gd['m_Name'], 'active_self': gd['m_IsActive'],
                          'parent_transform': td['m_Father'], 'components': gd['m_Component']})
        assert td['m_Father']['m_FileID'] == 0
        transform_id = td['m_Father']['m_PathID']
    assert [a['game_object_path_id'] for a in ancestors] == [415, 1529, 98, 1654, 7]
    assert all(a['active_self'] for a in ancestors)
    assert {'component': {'m_FileID': 0, 'm_PathID': 137027}} in ancestors[0]['components']
    assert {'component': {'m_FileID': 0, 'm_PathID': 137026}} in ancestors[0]['components']
    return {
        'schema_version': 'first_village_on_setup_binding_v1', 'build_id': BUILD,
        'evidence_status': 'native-static-and-authored-asset-configuration',
        'source_hashes': {**asset_hashes, 'GameAssembly.dll': sha(raw),
                          'script.json': sha((dumper_root / 'script.json').read_bytes()),
                          'dump.cs': sha((dumper_root / 'dump.cs').read_bytes()),
                          'global-metadata.dat': sha((data_root / 'il2cpp_data/Metadata/global-metadata.dat').read_bytes()),
                          'producer_source': sha(Path(__file__).read_bytes())},
        'component': {'path_id': 137027, 'object_sha256': sha(component_raw), 'object_size': len(component_raw),
                      'consumed_bytes': c.offset, 'header': header, 'serialized_fields': fields,
                      'script_path_id': 920, 'script_object_sha256': sha(script.get_raw_data()),
                      'script_binding': {k: script_fields[k] for k in ('m_ClassName', 'm_Namespace', 'm_AssemblyName')},
                      'ancestors': ancestors},
        'native_bodies': bodies, 'selected_operand_assertions': pins, 'metadata_joins': joins,
        'float_literals': literals,
        'binding': {'receiver': 'component.characters (original scene Characters137026)',
                    'on_enable': 'Combine(existing Action, new Action(component137027, Animates)); assign receiver.onSetup',
                    'on_disable': 'Remove(existing Action, new equivalent Action(component137027, Animates)); assign receiver.onSetup',
                    'identity': 'bound target component plus method identity; newly allocated delegates need not have equal object pointers',
                    'caller_abi': 'Manage loads method context delegate+0x28 into RDX, target delegate+0x40 into RCX; indirect call delegate+0x18',
                    'other_subscriptions': 'same lifecycle also Combine/Remove ShuffleCards on GameplayEvents.OnNextChallenge(+0x18), OnRestartCurrentLevel(+0x30), OnRestartGame(+0x38); future invocation excluded'},
        'first_state_trace': [
            'Animates allocates Animate iterator, captures component in +0x20, writes state0 and tail-calls StartCoroutine on that component.',
            'Animate state0 captures component.characters.characters; enumerates all current card occurrences; GetComponent<SingleCharacterDrawAnimation>, then pivot.localPosition=Vector3.zero.',
            'After first enumeration disposal, allocates PlayAudioDelay state0 and calls StartCoroutine on the animation component before beginning a fresh board enumeration.',
            'PlayAudioDelay state0 invokes AudioEvents action with integer200 if nonnull, then yields WaitForSeconds float32 bits3ecccccd (.4f). Audio callback effects are unestablished.',
            'Animate selects first current card, resolves its draw-animation pivot, requests DOLocalMoveY(end390,durationbits3e99999a,snappingfalse), then yields WaitForSeconds bits3d4ccccd (.05f), state1 and true.',
            'Under synchronous first-yield StartCoroutine services and nonempty stable N5 board, nested audio wait precedes outer animation wait; both use animation-component physical owner, distinct from manager and five card owners.',
        ],
        'scope': {
            'executed_native_bodies': 0,
            'static_first_state_writes': 'iterator state/current/enumerator storage and supplied transform/tween/audio effects; no direct role/status/gameplay-phase write observed in selected first-state trace',
            'no_visibility_admission': 'animation/tween effects are not a screenshot, reveal publication or legal-action readiness certificate',
            'runtime_lifecycle_provider': 'Unity scene hydration and actual Awake/OnEnable/OnDisable invocation/lifetime are supplied; serialized enabled/ancestor flags establish configuration, not invocation',
            'remaining_services': ['delegate allocation/Combine/Remove/invoke implementation and existing invocation list',
                                   'Unity object and GetComponent mapping for physical cards and pivots',
                                   'Vector3.zero initialization, transform setters and DOTween side effects',
                                   'AudioEvents subscriber identity/effects',
                                   'retained StartCoroutine first-yield, native owner keys and shared queue ordering',
                                   'later animation/audio resumes, Gameplay startup and player availability'],
            'next_exit': 'retained original component subscriber join at Manage onSetup through audio/animation first waits and subsequent manager ShuffleDeck admission; preserve all five card waits, actor/status/pool storage and distinct owner bijection; do not assume total count without native producer',
        },
        'counters': {'scene_components': len(candidates), 'ancestor_game_objects': len(ancestors),
                     'fingerprinted_native_intervals': len(bodies), 'selected_operand_assertions': len(pins),
                     'metadata_joins': len(joins), 'exact_float_literals': len(literals), 'executed_native_bodies': 0},
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--game-root', type=Path, required=True)
    p.add_argument('--dumper-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    assert args.output.parent.is_dir(), args.output.parent
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'schema': result['schema_version'], 'counters': result['counters'],
                      'output': str(args.output), 'sha256': sha(args.output.read_bytes())}))


if __name__ == '__main__':
    main()
