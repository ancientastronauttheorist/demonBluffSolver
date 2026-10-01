"""Execute complete Character.RefreshCharacter against authored runtime and UI services."""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD


def audit(game_root, dumper_root):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x

    assert unicorn.__version__ == '2.1.4'
    repo = Path(__file__).parents[1]
    manifest = json.loads((repo / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((repo / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))

    def pin(path, digest):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == digest.upper()
        return raw

    raw = pin(game_root / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    metadata = json.loads(pin(dumper_root / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump = pin(dumper_root / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    fields = {'Character': ['public CharacterData dataRef; // 0x50', 'public CharacterData bluff; // 0x58',
                            'public bool revealed; // 0xD8', 'private int pickableUses; // 0xDC',
                            'public ECharacterState state; // 0xE4', 'public GameObject[] pickeds; // 0x188',
                            'public GameObject pickable; // 0x1A8'],
              'CharacterData': ['public EAbilityUsage abilityUsage; // 0x138', 'public bool picking; // 0x13E'],
              'Gameplay': ['public static EGameplayState GameplayState; // 0x28',
                           'public static EGameplayState PrevState; // 0x2C']}
    for name, declarations in fields.items():
        body = re.search(r'^[^\n]*class ' + name + r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', dump, re.M | re.S)
        assert body and all(declaration in body[1] for declaration in declarations), name
    enums = {}
    for name in ['ECharacterState', 'EGameplayState', 'EAbilityUsage']:
        body = re.search(r'^public enum ' + name + r' // TypeDefIndex: \d+\s*\{(.*?)^\}', dump, re.M | re.S)
        assert body
        enums[name] = {n: int(v) for n, v in re.findall(r'public const ' + name + r' (\w+) = (-?\d+);', body[1])}
    assert enums['ECharacterState'] == {'None': 0, 'Hidden': 5, 'Alive': 10, 'Dead': 20, 'Revealed': 30}
    assert enums['EGameplayState']['Night'] == 20 and enums['EAbilityUsage']['ResetAfterNight'] == 10
    rows = [m for m in metadata['ScriptMethod'] if m['Name'] == 'Character$$RefreshCharacter']
    assert len(rows) == 1 and rows[0]['Address'] == 0x367970
    assert rows[0]['Signature'] == 'void Character__RefreshCharacter (Character_o* __this, const MethodInfo* method);'
    method = rows[0]
    end = min(m['Address'] for m in metadata['ScriptMethod'] if m['Address'] > method['Address'])
    assert end == 0x367B60
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    instructions = list(cs.disasm(pe.get_data(method['Address'], end - method['Address']), method['Address']))
    while instructions[-1].mnemonic == 'int3':
        instructions.pop()
    decoded = {i.address: i for i in instructions}
    assert instructions[-1].address + instructions[-1].size == 0x367B5A
    assert all(a.address + a.size == b.address for a, b in zip(instructions, instructions[1:]))
    checks = {0x3679A5: ('xor', 'edi, edi'), 0x3679D6: ('call', '0x1c7d810'),
              0x367A07: ('cmp', 'dword ptr [rax + 0x2c], 0x14'),
              0x367A33: ('cmp', 'eax, 0x14'), 0x367A38: ('cmp', 'eax, 0x1e'),
              0x367A3D: ('cmp', 'byte ptr [rbx + 0xd8], 0'),
              0x367A6C: ('test', 'al, al'), 0x367A70: ('mov', 'rax, qword ptr [rbx + 0x58]'),
              0x367A83: ('cmp', 'dword ptr [rax + 0x138], 0xa'),
              0x367A96: ('mov', 'dword ptr [rbx + 0xdc], 1'),
              0x367AA0: ('cmp', 'eax, 5'), 0x367AA9: ('cmp', 'eax, 0x14'),
              0x367AD4: ('cmp', 'eax, 0x14'), 0x367AD9: ('cmp', 'eax, 0x1e'),
              0x367ADE: ('cmp', 'byte ptr [rbx + 0xd8], 0'),
              0x367B0D: ('test', 'al, al'), 0x367B11: ('mov', 'rax, qword ptr [rbx + 0x58]'),
              0x367B20: ('cmp', 'byte ptr [rax + 0x13e], 0'),
              0x367B38: ('mov', 'dl, 1'), 0x367B3A: ('call', '0x1c7d810'), 0x367B4E: ('ret', '')}
    for address, expected in checks.items():
        assert address in decoded and (decoded[address].mnemonic, decoded[address].op_str) == expected
    def rip_slot(address):
        ins = decoded[address]
        operand = next(op for op in ins.operands if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP)
        return ins.address + ins.size + operand.mem.disp
    assert rip_slot(0x367A11) == rip_slot(0x367AB2)
    assert rip_slot(0x367A1A) == rip_slot(0x367ABB)
    assert rip_slot(0x367A46) == rip_slot(0x367AE7)
    slots, flags = set(), set()
    for i in instructions:
        for op in i.operands:
            if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                slot = i.address + i.size + op.mem.disp
                if i.mnemonic == 'cmp' and op.size == 1:
                    flags.add(slot)
                elif i.mnemonic in ['mov', 'lea'] and op.size == 8:
                    slots.add(slot)
    rows = [r for section in ['ScriptMetadata', 'ScriptMetadataMethod', 'ScriptString']
            for r in metadata[section] if r['Address'] in slots]
    assert {r['Address'] for r in rows} == slots
    assert {r['Name'] for r in rows} == {'Gameplay_TypeInfo', 'UnityEngine.Object_TypeInfo'}
    service_rvas = {0x2B7B40: 'metadata', 0x281D90: 'class_init', 0x1C822C0: 'unity_null',
                    0x1C7D810: 'set_active', 0x2B7D90: 'null', 0x2B7D80: 'bounds'}
    assert {i.op_str for i in instructions if i.mnemonic == 'call'} == {hex(n) for n in service_rvas}
    services_metadata = [m for m in metadata['ScriptMethod'] if
                         (m['Address'], m['Name']) in [(0x1C822C0, 'UnityEngine.Object$$op_Equality'),
                                                      (0x1C7D810, 'UnityEngine.GameObject$$SetActive')]]
    assert len(services_metadata) == 2
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    uc.mem_map(arena, 0x20000)
    uc.mem_map(stack, 0x20000)
    uc.mem_map(stop, 0x1000)
    actor, real, bluff, alternate, picked_array, gameplay = [arena + i * 0x1000 for i in range(1, 7)]
    pickable = arena + 0x7000
    picked_objects = [arena + n for n in [0x8000, 0x9000, 0xA000]]

    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def d(a, v): uc.mem_write(a, struct.pack('<I', v & 0xFFFFFFFF))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def rd(a): return struct.unpack('<I', uc.mem_read(a, 4))[0]
    def byte(a): return uc.mem_read(a, 1)[0]
    def reg(r): return uc.reg_read(r)
    def ret(value=0):
        sp = reg(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, sp + 8)
        uc.reg_write(x.UC_X86_REG_RIP, rq(sp))

    types = {}
    for index, row in enumerate(rows):
        pointer = arena + 0x10000 + index * 0x1000
        types[row['Name']] = pointer
        q(base + row['Address'], pointer)
    state, options, visited = {}, {}, set()

    def snapshot():
        return {'uses': rd(actor + 0xDC), 'state': rd(actor + 0xE4), 'revealed': byte(actor + 0xD8),
                'data': rq(actor + 0x50), 'bluff': rq(actor + 0x58),
                'pickable_active': state['pickable_active'], 'picked_active': state['picked_active'].copy()}

    def event(kind, **details):
        state['counts'][kind] = state['counts'].get(kind, 0) + 1
        state['events'].append({'kind': kind, **details, 'snapshot': snapshot()})
        if options.get('failure') == [kind, state['counts'][kind]]:
            state['error'] = kind
            uc.emu_stop()
            return False
        return True

    def hook(_, address, size, __):
        rva = address - base
        if address == stop:
            state['returned'] = True
            uc.emu_stop()
            return
        name = service_rvas.get(rva)
        if not name:
            assert rva in decoded and decoded[rva].size == size, hex(rva)
            visited.add(rva)
            return
        cx, dx, r8 = [reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8]]
        if name in ['null', 'bounds']:
            state['error'] = name
            event(name)
            uc.emu_stop()
        elif name == 'metadata':
            assert cx - base in slots
            if event(name, slot=hex(cx - base)):
                ret()
        elif name == 'class_init':
            assert cx in types.values()
            if event(name, type=next(n for n, p in types.items() if p == cx)):
                d(cx + 0xE0, 1)
                ret()
        elif name == 'unity_null':
            assert cx in [0, bluff, alternate] and dx == r8 == 0
            is_null = cx == 0 or (cx == bluff and options.get('bluff_liveness') == 'destroyed')
            if event(name, object=cx, authored_unity_null=is_null):
                effect = options.get('equality_effects', {}).get(str(state['counts'][name]), {})
                for field, value in effect.items():
                    if field == 'bluff':
                        q(actor + 0x58, {'alternate': alternate, 'absent': 0, 'original': bluff}[value])
                    elif field == 'data':
                        q(actor + 0x50, {'alternate': alternate, 'absent': 0, 'original': real}[value])
                    elif field == 'state':
                        d(actor + 0xE4, value)
                    else:
                        assert field == 'revealed'
                        uc.mem_write(actor + 0xD8, bytes([value]))
                ret(0xDEADBEEF00000100 | int(is_null))
        else:
            assert name == 'set_active' and r8 == 0
            value = dx & 0xFF
            assert value in [0, 1]
            assert cx in picked_objects + [pickable]
            if cx in picked_objects:
                assert value == 0 and dx == 0
            if event(name, object=cx, value=value):
                if cx == pickable:
                    state['pickable_active'] = bool(value)
                else:
                    state['picked_active'][picked_objects.index(cx)] = bool(value)
                ret()

    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]

    def run(authored):
        options.clear()
        options.update(authored)
        state.clear()
        state.update(returned=False, error=None, events=[], counts={},
                     picked_active=[True, True, True], pickable_active=options.get('pickable_active', False))
        uc.mem_write(actor, b'\xA5' * 0x1B8)
        for ptr in [real, bluff, alternate]:
            uc.mem_write(ptr, b'\xB6' * 0x148)
        q(actor + 0x50, 0 if options.get('null') == 'data' else real)
        q(actor + 0x58, 0 if options.get('bluff_liveness', 'live') == 'absent' else bluff)
        d(actor + 0xDC, options.get('uses', 0))
        d(actor + 0xE4, options.get('state', 10))
        uc.mem_write(actor + 0xD8, bytes([options.get('revealed', 0)]))
        q(actor + 0x188, 0 if options.get('null') == 'pickeds' else picked_array)
        q(actor + 0x1A8, 0 if options.get('null') == 'pickable' else pickable)
        d(picked_array + 0x18, options.get('picked_count', 2))
        for index, pointer in enumerate(picked_objects):
            q(picked_array + 0x20 + index * 8, 0 if options.get('null') == f'picked{index}' else pointer)
        for ptr, usage, picking in zip([real, bluff, alternate], options.get('usages', [0, 10, 10]),
                                      options.get('picking', [1, 1, 1])):
            d(ptr + 0x138, usage)
            uc.mem_write(ptr + 0x13E, bytes([picking]))
        for pointer in types.values():
            d(pointer + 0xE0, int(not options.get('cold', False)))
        q(types['Gameplay_TypeInfo'] + 0xB8, gameplay)
        d(gameplay + 0x28, options.get('current_phase', 20))
        d(gameplay + 0x2C, options.get('previous_phase', 20))
        for flag in flags:
            uc.mem_write(base + flag, bytes([int(not options.get('cold', False))]))
        actor_before = bytes(uc.mem_read(actor, 0x1B8))
        data_before = [bytes(uc.mem_read(ptr, 0x148)) for ptr in [real, bluff, alternate]]
        array_before = bytes(uc.mem_read(picked_array, 0x40))
        before = snapshot()
        sp = stack + 0x10008
        q(sp, stop)
        for index, register in enumerate(registers):
            uc.reg_write(register, 0xFAB00000 + index)
        uc.reg_write(x.UC_X86_REG_RSP, sp)
        uc.reg_write(x.UC_X86_REG_RCX, actor)
        uc.reg_write(x.UC_X86_REG_RDX, 0)
        uc.emu_start(base + 0x367970, stop + 0x100, timeout=10_000_000, count=10000)
        assert state['returned'] or state['error']
        if state['returned']:
            assert reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(reg(register) == 0xFAB00000 + index for index, register in enumerate(registers))
        allowed = set(range(0xDC, 0xE0))
        for effect in options.get('equality_effects', {}).values():
            for field in effect:
                start, count = {'bluff': (0x58, 8), 'data': (0x50, 8), 'state': (0xE4, 4), 'revealed': (0xD8, 1)}[field]
                allowed.update(range(start, start + count))
        actor_after = bytes(uc.mem_read(actor, 0x1B8))
        assert all(a == b for index, (a, b) in enumerate(zip(actor_before, actor_after)) if index not in allowed)
        assert data_before == [bytes(uc.mem_read(ptr, 0x148)) for ptr in [real, bluff, alternate]]
        assert array_before == bytes(uc.mem_read(picked_array, 0x40))
        return {'options': options.copy(), 'before': before, 'returned': state['returned'], 'error': state['error'],
                'events': state['events'].copy(), 'final': snapshot(), 'retained_storage_verified': True,
                'native_return_verified': state['returned']}

    def compact(result):
        return {k: v for k, v in result.items() if k != 'events'} | {'event_kinds': [e['kind'] for e in result['events']]}

    cases, baselines, failures, mutations = [], [], [], []
    for count, phase, actor_state, revealed, liveness, usages, picking in itertools.product(
            [0, 2], [0, 20], [0, 5, 10, 20, 30], [0, 1], ['absent', 'live', 'destroyed'],
            [[0, 0, 10], [10, 0, 10], [0, 10, 10], [10, 10, 10]],
            [[0, 0, 1], [1, 0, 1], [0, 1, 1], [1, 1, 1]]):
        options_case = {'picked_count': count, 'previous_phase': phase, 'state': actor_state,
                        'revealed': revealed, 'bluff_liveness': liveness, 'usages': usages, 'picking': picking}
        result = run(options_case)
        select_bluff = actor_state not in [20, 30] and not revealed and liveness == 'live'
        selected = int(select_bluff)
        resets = phase == 20 and usages[selected] == 10
        activates = resets and actor_state not in [5, 20] and picking[selected] != 0
        assert result['returned'] and result['final']['uses'] == int(resets)
        assert result['final']['pickable_active'] == bool(activates)
        assert result['final']['picked_active'] == ([False, False, True] if count == 2 else [True] * 3)
        cases.append(compact(result))
    for uses, previous_phase, current_phase, initial_active in itertools.product(
            [-1, 0, 1, 2, 0x80000000, 0xFFFFFFFF], [0, 20], [0, 20], [False, True]):
        result = run({'uses': uses, 'previous_phase': previous_phase, 'current_phase': current_phase,
                      'usages': [10, 10, 10], 'pickable_active': initial_active})
        assert result['returned'] and result['final']['uses'] == (1 if previous_phase == 20 else uses & 0xFFFFFFFF)
        assert result['final']['pickable_active'] == (True if previous_phase == 20 else initial_active)
        cases.append(compact(result))
    for actor_state in [-1, 0x80000000, 0xFFFFFFFF]:
        result = run({'state': actor_state})
        assert result['returned'] and result['final']['uses'] == 1 and result['final']['pickable_active']
        cases.append(compact(result))
    for count in [3, 0x80000000, 0xFFFFFFFF]:
        result = run({'picked_count': count, 'previous_phase': 0})
        assert result['returned'] and result['final']['picked_active'] == ([False] * 3 if count == 3 else [True] * 3)
        cases.append(compact(result))
    for null, phase, actor_state, revealed, liveness in itertools.product(
            ['pickeds', 'picked0', 'picked1', 'data', 'pickable'], [0, 20], [5, 10, 20], [0, 1], ['absent', 'live']):
        result = run({'null': null, 'previous_phase': phase, 'state': actor_state, 'revealed': revealed,
                      'bluff_liveness': liveness, 'usages': [10, 10, 10]})
        uses_real = actor_state in [20, 30] or revealed or liveness != 'live'
        expects_null = null in ['pickeds', 'picked0', 'picked1'] or (
            phase == 20 and ((null == 'data' and uses_real) or (null == 'pickable' and actor_state not in [5, 20])))
        assert result['returned'] == (not expects_null)
        assert result['error'] == ('null' if expects_null else None)
        cases.append(compact(result))
    for effect in [{'revealed': 1}, {'state': 20}, {'state': 30}, {'bluff': 'alternate'},
                   {'bluff': 'absent'}, {'data': 'alternate', 'revealed': 1}]:
        result = run({'equality_effects': {'1': effect}, 'usages': [0, 10, 10], 'picking': [0, 1, 1]})
        if effect.get('bluff') == 'absent':
            assert result['error'] == 'null' and result['final']['uses'] == 0
        else:
            assert result['returned'] and result['final']['uses'] == 1
            expected = effect == {'bluff': 'alternate'} or effect == {'data': 'alternate', 'revealed': 1}
            assert result['final']['pickable_active'] == expected
        mutations.append(result)
    for effect in [{'revealed': 1}, {'bluff': 'absent'}, {'data': 'absent', 'revealed': 1}, {'state': 20}]:
        result = run({'equality_effects': {'2': effect}, 'usages': [0, 10, 10], 'picking': [0, 1, 1]})
        assert result['final']['uses'] == 1
        if effect.get('bluff') == 'absent':
            assert result['error'] == 'null'
        else:
            assert result['returned'] and result['final']['pickable_active']
        mutations.append(result)
    for authored in [{'cold': True}, {'cold': True, 'state': 5, 'bluff_liveness': 'destroyed', 'usages': [10, 0, 10]},
                     {'cold': True, 'previous_phase': 0}]:
        baseline = run(authored)
        assert baseline['returned']
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event_row in enumerate(baseline['events']):
            kind = event_row['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = run({**authored, 'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event_row['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build': BUILD, 'method': method, 'fields': fields, 'enums': enums,
            'native_range': ['0x367970', '0x367b5a'], 'normal_return': '0x367b4e',
            'instruction_assertions': len(checks) + 3, 'service_metadata': services_metadata,
            'shared_selector_metadata_guard': hex(rip_slot(0x367A11)),
            'case_count': len(cases), 'cases': cases, 'mutation_case_count': len(mutations), 'mutation_cases': mutations,
            'failure_baselines': baselines, 'failure_case_count': len(failures), 'failure_cases': failures,
            'native_instructions_executed': len(visited), 'native_instruction_count': len(decoded),
            'unexecuted_native_addresses': [hex(n) for n in sorted(set(decoded) - visited)],
            'limits': ['Full caller body executes with explicit metadata/class-init, Unity null/equality and UI SetActive services.',
                       'Authored Unity equality distinguishes absent, live and destroyed pointers; no engine object lifetime implementation is claimed.',
                       'Mutations occur only at declared authored equality callbacks, preserving actual caller reload order.',
                       'Null failures and controlled service stops preserve exact native prefixes; no native exception unwinding is modeled.',
                       'Bounds helper is resolved but concurrent array-length mutation and its otherwise unreachable branch are not claimed.',
                       'Standalone RefreshCharacter is not a new initializer/scheduler composition or live-play rule.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'mutation_case_count', 'failure_case_count', 'native_instructions_executed']}))
