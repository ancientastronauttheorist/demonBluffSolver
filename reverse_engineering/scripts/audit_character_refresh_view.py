"""Execute complete Character.RefreshView with authored Unity presentation APIs."""
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
    fields = {'Character': ['public Transform icon; // 0x20', 'public CharacterData bluff; // 0x58',
                            'public GameObject ripView; // 0x78', 'public GameObject deadPrefab; // 0x80',
                            'public GameObject disguiseIcon; // 0x88', 'public GameObject createdDeadPrefab; // 0x98',
                            'public bool revealed; // 0xD8', 'private int pickableUses; // 0xDC',
                            'public ECharacterState state; // 0xE4', 'public bool killedByDemon; // 0xED',
                            'public GameObject pickable; // 0x1A8'],
              'Vector3': ['public float x; // 0x0', 'public float y; // 0x4', 'public float z; // 0x8',
                          'private static readonly Vector3 zeroVector; // 0x0']}
    type_definitions = {'Character': 5487, 'Vector3': 6699}
    for name, declarations in fields.items():
        kind = 'struct' if name == 'Vector3' else 'class'
        body = re.search(r'^[^\n]*' + kind + ' ' + name + r'(?: :[^\n]*)? // TypeDefIndex: '
                         + str(type_definitions[name]) + r'\s*\{(.*?)^\}', dump, re.M | re.S)
        assert body and all(declaration in body[1] for declaration in declarations), name
        namespace_start = dump.rfind('// Namespace:', 0, body.start())
        namespace_line = dump[namespace_start:dump.find('\n', namespace_start)].strip()
        assert namespace_line == ('// Namespace: UnityEngine' if name == 'Vector3' else '// Namespace:')
    rows = [m for m in metadata['ScriptMethod'] if m['Name'] == 'Character$$RefreshView']
    assert len(rows) == 1 and rows[0]['Address'] == 0x367B60
    assert rows[0]['Signature'] == 'void Character__RefreshView (Character_o* __this, const MethodInfo* method);'
    method = rows[0]
    end = min(m['Address'] for m in metadata['ScriptMethod'] if m['Address'] > method['Address'])
    assert end == 0x367E00
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    instructions = list(cs.disasm(pe.get_data(method['Address'], end - method['Address']), method['Address']))
    while instructions[-1].mnemonic == 'int3':
        instructions.pop()
    decoded = {i.address: i for i in instructions}
    assert instructions[-1].address + instructions[-1].size == 0x367DFB
    assert all(a.address + a.size == b.address for a, b in zip(instructions, instructions[1:]))
    checks = {0x367B91: ('cmp', 'dword ptr [rbx + 0xdc], 0'), 0x367BA2: ('jg', '0x367bbe'),
              0x367BBE: ('cmp', 'dword ptr [rbx + 0xe4], 0x14'),
              0x367BFC: ('mov', 'rsi, qword ptr [rbx + 0x80]'),
              0x367C37: ('call', '0x668010'), 0x367C3F: ('mov', 'qword ptr [rbx + 0x98], rax'),
              0x367C4D: ('call', '0x2b6ff0'), 0x367C52: ('mov', 'rcx, qword ptr [rbx + 0x98]'),
              0x367C69: ('call', '0x1c7ddf0'), 0x367C99: ('call', '0x1c91b80'),
              0x367C9E: ('test', 'rsi, rsi'), 0x367CA7: ('movsd', 'xmm0, qword ptr [rax]'),
              0x367CB0: ('mov', 'eax, dword ptr [rax + 8]'),
              0x367CC3: ('call', '0x1c923d0'), 0x367CC8: ('mov', 'rcx, qword ptr [rbx + 0x98]'),
              0x367CDA: ('call', '0x1c7ddf0'), 0x367D05: ('mov', 'rax, qword ptr [rcx + 0xb8]'),
              0x367D15: ('movsd', 'xmm0, qword ptr [rax]'), 0x367D1E: ('mov', 'eax, dword ptr [rax + 8]'),
              0x367D31: ('call', '0x1c91ec0'), 0x367D48: ('call', '0x1c7d810'),
              0x367D7A: ('cmp', 'byte ptr [rbx + 0xed], 0'),
              0x367D89: ('cmp', 'eax, 0x14'), 0x367D8E: ('cmp', 'eax, 0x1e'),
              0x367DB8: ('ret', ''), 0x367DDA: ('call', '0x1c822c0'),
              0x367DDF: ('mov', 'rcx, qword ptr [rbx + 0x88]'), 0x367DE6: ('test', 'al, al')}
    for address, expected in checks.items():
        assert address in decoded and (decoded[address].mnemonic, decoded[address].op_str) == expected
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
    assert {r['Name'] for r in rows} == {'UnityEngine.Object_TypeInfo', 'UnityEngine.Vector3_TypeInfo',
                                       'Method$UnityEngine.Object.Instantiate<GameObject>()'}
    names = {0x1C7A010: 'UnityEngine.Component$$get_transform', 0x1C7DDF0: 'UnityEngine.GameObject$$get_transform',
             0x1C7D810: 'UnityEngine.GameObject$$SetActive', 0x668010: 'UnityEngine.Object$$Instantiate<object>',
             0x1C822C0: 'UnityEngine.Object$$op_Equality', 0x1C82480: 'UnityEngine.Object$$op_Inequality',
             0x1C91B80: 'UnityEngine.Transform$$get_position', 0x1C923D0: 'UnityEngine.Transform$$set_position',
             0x1C91EC0: 'UnityEngine.Transform$$set_eulerAngles'}
    services_metadata = []
    for address, name in names.items():
        matches = [m for m in metadata['ScriptMethod'] if m['Address'] == address and m['Name'] == name]
        assert len(matches) == 1
        services_metadata.append(matches[0])
    services = {0x2B7B40: 'metadata', 0x281D90: 'class_init', 0x2B6FF0: 'barrier',
                0x2B7D90: 'null', 0x1C7A010: 'component_transform', 0x1C7DDF0: 'game_object_transform',
                0x1C7D810: 'set_active', 0x668010: 'instantiate', 0x1C822C0: 'unity_null',
                0x1C82480: 'unity_live', 0x1C91B80: 'get_position', 0x1C923D0: 'set_position',
                0x1C91EC0: 'set_euler_angles'}
    assert {i.op_str for i in instructions if i.mnemonic == 'call'} == {hex(n) for n in services}
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    uc.mem_map(arena, 0x20000)
    uc.mem_map(stack, 0x20000)
    uc.mem_map(stop, 0x1000)
    actor, prefab, old_created, new_created, alternate_created, icon, parent_transform, icon_transform = [
        arena + n * 0x1000 for n in range(1, 9)]
    transforms = [arena + 0x9000, arena + 0xA000]
    pickable, rip, disguise, alternate_disguise = [arena + n for n in [0xB000, 0xC000, 0xD000, 0xE000]]
    vector_static = arena + 0xF000

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
    def vector(bits):
        assert len(bits) == 3 and all(0 <= b <= 0xFFFFFFFF for b in bits)
        return struct.pack('<III', *bits)
    def vector_bits(pointer): return list(struct.unpack('<III', uc.mem_read(pointer, 12)))

    pointers = {}
    for index, row in enumerate(rows):
        pointer = arena + 0x10000 + index * 0x1000
        pointers[row['Name']] = pointer
        q(base + row['Address'], pointer)
    unity_type = pointers['UnityEngine.Object_TypeInfo']
    vector_type = pointers['UnityEngine.Vector3_TypeInfo']
    method_info = pointers['Method$UnityEngine.Object.Instantiate<GameObject>()']
    state, options, visited = {}, {}, set()

    def snapshot():
        return {'created': rq(actor + 0x98), 'uses': rd(actor + 0xDC), 'state': rd(actor + 0xE4),
                'revealed': byte(actor + 0xD8), 'killed_by_demon': byte(actor + 0xED),
                'prefab': rq(actor + 0x80), 'disguise': rq(actor + 0x88),
                'pickable_active': state['pickable_active'], 'rip_active': state['rip_active'],
                'disguise_active': state['disguise_active'], 'alternate_disguise_active': state['alternate_disguise_active'],
                'position_writes': state['position_writes'].copy(), 'euler_writes': state['euler_writes'].copy(),
                'instantiations': state['instantiations'].copy()}

    def event(kind, **details):
        state['counts'][kind] = state['counts'].get(kind, 0) + 1
        state['events'].append({'kind': kind, **details, 'snapshot': snapshot()})
        if options.get('failure') == [kind, state['counts'][kind]]:
            state['error'] = kind
            uc.emu_stop()
            return False
        return True

    def effects(kind):
        effect = options.get('effects', {}).get(f'{kind}:{state["counts"][kind]}', {})
        for field, value in effect.items():
            if field in ['created', 'prefab', 'disguise']:
                symbols = {'absent': 0, 'old': old_created, 'new': new_created, 'alternate': alternate_created,
                           'original_prefab': prefab, 'original_disguise': disguise, 'alternate_disguise': alternate_disguise}
                q(actor + {'created': 0x98, 'prefab': 0x80, 'disguise': 0x88}[field], symbols[value])
            elif field == 'state':
                d(actor + 0xE4, value)
            else:
                assert field in ['revealed', 'killed_by_demon']
                uc.mem_write(actor + {'revealed': 0xD8, 'killed_by_demon': 0xED}[field], bytes([value]))

    def authored_live(pointer):
        if pointer == 0:
            return False
        if pointer == old_created:
            return options.get('created_liveness', 'absent') == 'live'
        if pointer == disguise:
            return options.get('disguise_liveness', 'live') == 'live'
        if pointer == prefab:
            return options.get('bluff_liveness', 'live') == 'live'
        assert pointer in [new_created, alternate_created, alternate_disguise]
        return True

    def hook(_, address, size, __):
        rva = address - base
        if address == stop:
            state['returned'] = True
            uc.emu_stop()
            return
        name = services.get(rva)
        if not name:
            assert rva in decoded and decoded[rva].size == size, hex(rva)
            visited.add(rva)
            return
        cx, dx, r8 = [reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8]]
        details = {}
        if name == 'metadata':
            assert cx - base in slots
            details['slot'] = hex(cx - base)
        elif name == 'class_init':
            assert cx == unity_type
            details['type'] = 'UnityEngine.Object'
        elif name == 'barrier':
            assert cx == actor + 0x98 and rq(cx) == dx
            details['stored_created'] = dx
        elif name in ['unity_null', 'unity_live']:
            assert dx == r8 == 0
            live = authored_live(cx)
            details.update(object=cx, authored_live=live)
        elif name == 'component_transform':
            assert cx in [actor, icon] and dx == 0
            details['object'] = cx
        elif name == 'game_object_transform':
            assert cx in [new_created, old_created, alternate_created] and dx == 0
            details['object'] = cx
        elif name == 'instantiate':
            assert cx in [prefab, alternate_created, 0] and dx in [parent_transform, 0] and r8 == method_info
            details.update(prefab=cx, parent=dx, result=0 if options.get('clone_null') else new_created)
        elif name == 'get_position':
            assert dx == icon_transform and r8 == 0
            assert self_stack_start <= cx <= self_stack_end - 12
            details.update(transform=dx, bits=options.get('position_bits', [0x3F800000, 0xC0000000, 0x40400000]))
        elif name in ['set_position', 'set_euler_angles']:
            assert cx in transforms and r8 == 0 and self_stack_start <= dx <= self_stack_end - 12
            details.update(transform=cx, bits=vector_bits(dx))
            expected_bits = options.get('position_bits', [0x3F800000, 0xC0000000, 0x40400000]) if name == 'set_position' else options.get('zero_bits', [0, 0, 0])
            assert details['bits'] == expected_bits
        elif name == 'set_active':
            assert cx in [pickable, rip, disguise, alternate_disguise] and r8 == 0
            value = dx & 0xFF
            assert value in [0, 1]
            details.update(object=cx, value=value)
        else:
            assert name == 'null'
            state['error'] = name
        if not event(name, **details):
            return
        if name == 'null':
            uc.emu_stop()
            return
        effects(name)
        if name == 'metadata':
            ret()
        elif name == 'class_init':
            d(unity_type + 0xE0, 1)
            ret()
        elif name == 'barrier':
            ret()
        elif name in ['unity_null', 'unity_live']:
            ret(0xDEADBEEF00000100 | int(not live if name == 'unity_null' else live))
        elif name == 'component_transform':
            target = parent_transform if cx == actor else icon_transform
            ret(0 if options.get('null') == ('parent_transform' if cx == actor else 'icon_transform') else target)
        elif name == 'game_object_transform':
            occurrence = state['counts'][name]
            index = 1 if options.get('different_second_transform') and occurrence == 2 else 0
            ret(0 if options.get('null') == f'created_transform{occurrence}' else transforms[index])
        elif name == 'instantiate':
            state['instantiations'].append(details)
            ret(details['result'])
        elif name == 'get_position':
            uc.mem_write(cx, vector(details['bits']))
            ret(cx)
        elif name == 'set_position':
            state['position_writes'].append(details)
            ret()
        elif name == 'set_euler_angles':
            state['euler_writes'].append(details)
            ret()
        else:
            state[{pickable: 'pickable_active', rip: 'rip_active', disguise: 'disguise_active',
                   alternate_disguise: 'alternate_disguise_active'}[cx]] = bool(details['value'])
            ret()

    self_stack_start, self_stack_end = stack, stack + 0x20000
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    xmm_registers = [getattr(x, f'UC_X86_REG_XMM{n}') for n in range(6, 16)]

    def prepare(authored):
        options.clear()
        options.update(authored)
        state.clear()
        state.update(returned=False, error=None, events=[], counts={}, position_writes=[], euler_writes=[], instantiations=[],
                     pickable_active=options.get('initial_active', True), rip_active=options.get('initial_active', False),
                     disguise_active=options.get('initial_active', True), alternate_disguise_active=False)
        uc.mem_write(actor, b'\xA5' * 0x1B8)
        q(actor + 0x20, 0 if options.get('null') == 'icon' else icon)
        q(actor + 0x58, 0 if options.get('bluff_liveness', 'live') == 'absent' else prefab)
        q(actor + 0x78, 0 if options.get('null') == 'rip' else rip)
        q(actor + 0x80, 0 if options.get('null') == 'prefab' else prefab)
        q(actor + 0x88, 0 if options.get('disguise_liveness', 'live') == 'absent' else disguise)
        q(actor + 0x98, 0 if options.get('created_liveness', 'absent') == 'absent' else old_created)
        q(actor + 0x1A8, 0 if options.get('null') == 'pickable' else pickable)
        d(actor + 0xDC, options.get('uses', 1))
        d(actor + 0xE4, options.get('state', 20))
        uc.mem_write(actor + 0xD8, bytes([options.get('revealed', 0)]))
        uc.mem_write(actor + 0xED, bytes([options.get('killed_by_demon', 0)]))
        d(unity_type + 0xE0, int(not options.get('cold', False)))
        q(vector_type + 0xB8, vector_static)
        uc.mem_write(vector_static, vector(options.get('zero_bits', [0, 0, 0])))
        for flag in flags:
            uc.mem_write(base + flag, bytes([int(not options.get('cold', False))]))

    def invoke():
        actor_before = bytes(uc.mem_read(actor, 0x1B8))
        before = snapshot()
        sp = stack + 0x10008
        q(sp, stop)
        for index, register in enumerate(registers):
            uc.reg_write(register, 0xFAB00000 + index)
        for index, register in enumerate(xmm_registers):
            uc.reg_write(register, 0x123456789ABCDEF0123456789ABCDEF0 + index)
        uc.reg_write(x.UC_X86_REG_RSP, sp)
        uc.reg_write(x.UC_X86_REG_RCX, actor)
        uc.reg_write(x.UC_X86_REG_RDX, 0)
        uc.emu_start(base + 0x367B60, stop + 0x100, timeout=10_000_000, count=10000)
        assert state['returned'] or state['error']
        if state['returned']:
            assert reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(reg(register) == 0xFAB00000 + index for index, register in enumerate(registers))
            assert all(reg(register) == 0x123456789ABCDEF0123456789ABCDEF0 + index for index, register in enumerate(xmm_registers))
        allowed = set(range(0x98, 0xA0))
        for effect in options.get('effects', {}).values():
            for field in effect:
                start, count = {'created': (0x98, 8), 'prefab': (0x80, 8), 'disguise': (0x88, 8),
                                'state': (0xE4, 4), 'revealed': (0xD8, 1), 'killed_by_demon': (0xED, 1)}[field]
                allowed.update(range(start, start + count))
        actor_after = bytes(uc.mem_read(actor, 0x1B8))
        assert all(a == b for index, (a, b) in enumerate(zip(actor_before, actor_after)) if index not in allowed)
        return {'options': options.copy(), 'before': before, 'returned': state['returned'], 'error': state['error'],
                'events': state['events'].copy(), 'final': snapshot(), 'retained_actor_storage_verified': True,
                'native_return_verified': state['returned']}

    def run(authored):
        prepare(authored)
        return invoke()

    def compact(result):
        return {k: v for k, v in result.items() if k != 'events'} | {'event_kinds': [e['kind'] for e in result['events']]}

    cases, mutations, sequences, baselines, failures = [], [], [], [], []
    for actor_state, uses, created_live, disguise_live, bluff_live, killed, revealed in itertools.product(
            [0, 5, 10, 20, 30], [-1, 0, 1], ['absent', 'live', 'destroyed'],
            ['absent', 'live', 'destroyed'], ['absent', 'live', 'destroyed'], [0, 1], [0, 1]):
        authored = {'state': actor_state, 'uses': uses, 'created_liveness': created_live,
                    'disguise_liveness': disguise_live, 'bluff_liveness': bluff_live,
                    'killed_by_demon': killed, 'revealed': revealed}
        result = run(authored)
        creates = actor_state == 20 and created_live != 'live'
        expected_pointer = new_created if creates else (0 if created_live == 'absent' else old_created)
        final = result['final']
        assert result['returned'] and final['created'] == expected_pointer
        assert final['pickable_active'] == (uses > 0)
        assert final['rip_active'] == creates
        expected_icon = (actor_state in [20, 30] and bluff_live == 'live') if disguise_live == 'live' and not killed else True
        assert final['disguise_active'] == expected_icon
        assert len(final['instantiations']) == len(final['position_writes']) == len(final['euler_writes']) == int(creates)
        cases.append(compact(result))
    for uses, initial_active in itertools.product([-0x80000000, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], [False, True]):
        result = run({'uses': uses, 'initial_active': initial_active, 'state': 10})
        positive = (uses & 0xFFFFFFFF) < 0x80000000 and uses != 0
        assert result['returned'] and result['final']['pickable_active'] == (initial_active and positive)
        cases.append(compact(result))
    for bits, separate in itertools.product([[0, 0, 0], [0x80000000, 1, 0x7FC01234],
                                             [0x7F800000, 0xFF800000, 0x00800000]], [False, True]):
        result = run({'position_bits': bits, 'different_second_transform': separate})
        assert result['returned']
        assert result['final']['position_writes'][0]['bits'] == bits
        assert result['final']['position_writes'][0]['transform'] == transforms[0]
        assert result['final']['euler_writes'][0]['transform'] == transforms[int(separate)]
        cases.append(compact(result))
    for zero_bits in [[0x80000000, 1, 0x7FC01234], [0x3F800000, 0xC0000000, 0x40400000]]:
        result = run({'zero_bits': zero_bits})
        assert result['returned'] and result['final']['euler_writes'][0]['bits'] == zero_bits
        cases.append(compact(result))
    for null in ['pickable', 'icon', 'icon_transform', 'created_transform1', 'created_transform2', 'rip', 'prefab', 'parent_transform']:
        result = run({'null': null, 'uses': 0})
        assert result['returned'] == (null in ['prefab', 'parent_transform'])
        assert result['error'] == (None if result['returned'] else 'null')
        if null == 'created_transform1':
            assert any(e['kind'] == 'get_position' for e in result['events'])
        cases.append(compact(result))
    result = run({'clone_null': True})
    assert result['error'] == 'null' and result['final']['created'] == 0
    assert any(e['kind'] == 'barrier' and e['stored_created'] == 0 for e in result['events'])
    cases.append(compact(result))
    mutation_inputs = [({'effects': {'component_transform:1': {'prefab': 'alternate'}}}, None),
                       ({'effects': {'barrier:1': {'created': 'absent'}}}, 'null'),
                       ({'effects': {'barrier:1': {'created': 'alternate'}}}, None),
                       ({'effects': {'get_position:1': {'created': 'absent'}}}, 'null'),
                       ({'effects': {'unity_live:1': {'killed_by_demon': 1}}}, None),
                       ({'effects': {'unity_live:1': {'state': 5}}}, None),
                       ({'effects': {'unity_live:1': {'disguise': 'alternate_disguise'}}}, None),
                       ({'effects': {'unity_null:2': {'disguise': 'absent'}}}, 'null')]
    for authored, expected_error in mutation_inputs:
        result = run(authored)
        assert result['error'] == expected_error
        mutations.append(result)
    assert mutations[0]['final']['instantiations'][0]['prefab'] == prefab
    assert mutations[1]['final']['created'] == 0
    assert mutations[2]['final']['created'] == alternate_created
    assert any(e['kind'] == 'game_object_transform' and e['object'] == alternate_created for e in mutations[2]['events'])
    assert mutations[3]['final']['position_writes'] and not mutations[3]['final']['euler_writes']
    assert mutations[4]['final']['disguise_active']
    assert not any(e['kind'] == 'set_active' and e['object'] == disguise for e in mutations[4]['events'])
    assert not mutations[5]['final']['disguise_active']
    assert mutations[6]['final']['alternate_disguise_active']
    for authored in [{}, {'created_liveness': 'destroyed'}, {'uses': 0, 'initial_active': True}]:
        first = run(authored)
        assert first['returned']
        retained = bytes(uc.mem_read(actor, 0x1B8))
        state['returned'], state['error'], state['events'], state['counts'] = False, None, [], {}
        second = invoke()
        assert second['returned'] and bytes(uc.mem_read(actor, 0x1B8)) == retained
        assert not any(e['kind'] == 'instantiate' for e in second['events'])
        assert len(second['final']['instantiations']) == 1
        sequences.append({'first': first, 'second': second, 'retained_actor_verified': True})
    for authored in [{'cold': True, 'uses': 0}, {'cold': True, 'state': 30, 'created_liveness': 'live'},
                     {'cold': True, 'state': 5, 'disguise_liveness': 'destroyed'}]:
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
    return {'build': BUILD, 'method': method, 'fields': fields, 'type_definitions': type_definitions,
            'native_range': ['0x367b60', '0x367dfb'],
            'normal_return': '0x367db8', 'instruction_assertions': len(checks), 'service_metadata': services_metadata,
            'case_count': len(cases), 'cases': cases, 'mutation_case_count': len(mutations), 'mutation_cases': mutations,
            'sequence_count': len(sequences), 'sequences': sequences,
            'failure_baselines': baselines, 'failure_case_count': len(failures), 'failure_cases': failures,
            'native_instructions_executed': len(visited), 'native_instruction_count': len(decoded),
            'unexecuted_native_addresses': [hex(n) for n in sorted(set(decoded) - visited)],
            'limits': ['Complete RefreshView caller executes; Unity object/transform/Instantiate/UI/metadata/GC services are authored boundaries.',
                       'Vector3 position and Euler values are copied at exact 12-byte ABI width, with authored zeroVector static storage.',
                       'Destroyed/absent liveness, null API returns, callback mutations and failure stops are explicitly authored, not real engine behavior.',
                       'Authored Instantiate can accept a null prefab/parent; these caller stress inputs do not claim Unity accepts them.',
                       'Pixels, renderer implementation, asset localization, role/status/data selection, actual scene callbacks and native exception unwinding are outside scope.',
                       'No Rust projection or initializer/scheduler composition is broadened by this standalone report.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'mutation_case_count', 'sequence_count', 'failure_case_count', 'native_instructions_executed']}))
