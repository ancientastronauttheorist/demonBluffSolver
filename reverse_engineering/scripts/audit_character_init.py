"""Execute both Character initialization callers against authored actor fixtures.

UI, logging, refresh and coroutine scheduling are explicit service boundaries.
No game process is read and no proprietary executable bytes enter the report.
"""
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
    fields = {
        'Character': ['public TextMeshProUGUI number; // 0x48', 'public CharacterData dataRef; // 0x50',
                      'public CharacterData bluff; // 0x58', 'public CharacterData registerAs; // 0x60',
                      'private CharacterTrailerInfo trailerInfo; // 0x68', 'private RuntimeCharacterData runtimeData; // 0x70',
                      'public GameObject ripView; // 0x78', 'public GameObject createdDeadPrefab; // 0x98',
                      'public Acted acteds; // 0xA8', 'public bool revealed; // 0xD8', 'private int pickableUses; // 0xDC',
                      'public ECharacterState prevState; // 0xE0', 'public ECharacterState state; // 0xE4',
                      'public bool killedHidden; // 0xEC', 'public bool killedByDemon; // 0xED',
                      'public CharacterStatuses statuses; // 0xF0', 'public EAlignment alignment; // 0xF8',
                      'public int id; // 0x118', 'private bool characterStartActed; // 0x11C',
                      'public List<ActedInfo> actedInfos; // 0x148', 'public Role role; // 0x168',
                      'public Role bluffRole; // 0x170', 'public Action onStateChange; // 0x180',
                      'private string savedAct; // 0x198', 'public GameObject[] pickeds; // 0x188',
                      'public GameObject pickable; // 0x1A8'],
        'CharacterData': ['public string characterName; // 0x28', 'public EAlignment startingAlignment; // 0x134',
                          'public EAbilityUsage abilityUsage; // 0x138', 'public bool picking; // 0x13E', 'public Role role; // 0x140'],
        'Gameplay': ['public static EGameplayState PrevState; // 0x2C'],
        'CharacterStatuses': ['public List<ECharacterStatus> statuses; // 0x10',
                              'public List<ECharacterStatus> resistances; // 0x18', 'public Character targetCharacter; // 0x20'],
        'Character.<DelayReveal>d__84': ['private int <>1__state; // 0x10', 'private object <>2__current; // 0x18',
                                      'public Character <>4__this; // 0x20'],
    }
    for name, declarations in fields.items():
        body = re.search(r'^[^\n]*class ' + re.escape(name) + r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', dump, re.M | re.S)
        assert body, name
        assert all(declaration in body[1] for declaration in declarations), name
    entries = {'Init': 0x365a20, 'InitWithNoReset': 0x365720}
    joined_entries = {'Character$$RefreshCharacter': 0x367970, 'Character.<DelayReveal>d__84$$MoveNext': 0x3756b0}
    methods = []
    for name, start in entries.items():
        rows = [m for m in metadata['ScriptMethod'] if m['Name'] == 'Character$$' + name]
        assert len(rows) == 1 and rows[0]['Address'] == start
        assert rows[0]['Signature'] == f'void Character__{name} (Character_o* __this, CharacterData_o* character, int32_t id, const MethodInfo* method);'
        methods.append(rows[0])
    for name, start in joined_entries.items():
        rows = [m for m in metadata['ScriptMethod'] if m['Name'] == name]
        assert len(rows) == 1 and rows[0]['Address'] == start
        methods.append(rows[0])
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    decoded = {}
    for start in list(entries.values()) + list(joined_entries.values()):
        end = min(m['Address'] for m in metadata['ScriptMethod'] if m['Address'] > start)
        ins = list(cs.disasm(pe.get_data(start, end - start), start))
        while ins[-1].mnemonic == 'int3':
            ins.pop()
        assert ins[0].address == start and all(a.address + a.size == b.address for a, b in zip(ins, ins[1:]))
        decoded.update({i.address: i for i in ins})
    # The common call is a folded no-op, not the aliased managed name in Ghidra.
    noop = list(cs.disasm(pe.get_data(0x33ed50, 3), 0x33ed50))
    assert len(noop) == 1 and (noop[0].mnemonic, noop[0].op_str, noop[0].size) == ('ret', '0', 3)
    decoded[noop[0].address] = noop[0]
    checks = {
        0x365aad: ('mov', 'qword ptr [rcx], r15'), 0x365b16: ('mov', 'qword ptr [rcx], r15'),
        0x365c33: ('mov', 'eax, dword ptr [rsi + 0x134]'), 0x365c39: ('mov', 'dword ptr [rdi + 0xf8], eax'),
        0x365cbf: ('call', 'qword ptr [rax + 0x18]'), 0x365cc2: ('mov', 'rax, qword ptr [rdi + 0xf0]'),
        0x365cdf: ('inc', 'dword ptr [rax + 0x1c]'), 0x365ce2: ('mov', 'dword ptr [rax + 0x18], r15d'),
        0x36588e: ('mov', 'byte ptr [rdi + 0xd8], r15b'), 0x365986: ('call', 'qword ptr [rax + 0x18]'),
        0x3659dc: ('mov', 'dword ptr [rbx + 0x10], r15d'), 0x365d39: ('mov', 'dword ptr [rbx + 0x10], r15d'),
        0x365a05: ('jmp', '0x1c7f160'), 0x365d62: ('jmp', '0x1c7f160'),
        0x367a07: ('cmp', 'dword ptr [rax + 0x2c], 0x14'),
        0x367a83: ('cmp', 'dword ptr [rax + 0x138], 0xa'),
        0x3756f4: ('mov', 'dword ptr [rdi + 0x10], 0xffffffff'),
        0x37572e: ('mov', 'qword ptr [rcx], rax'),
        0x375769: ('mov', 'dword ptr [rdi + 0x10], 1'),
    }
    for rva, expected in checks.items():
        assert rva in decoded and (decoded[rva].mnemonic, decoded[rva].op_str) == expected
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    for pointer in (arena, stack, stop):
        uc.mem_map(pointer, 0x20000)
    obj, data, acted, info, items = [arena + k * 0x1000 for k in range(1, 6)]
    status, active, resistant, target = [arena + k * 0x1000 for k in range(6, 10)]
    number, number_class, callback, iterator = [arena + k * 0x1000 for k in range(10, 14)]
    alternate_status, alternate_active = arena + 0xe000, arena + 0xf000
    callback_code, text_code, yield_return = stop + 0x100, stop + 0x200, stop + 0x300
    pickeds, gameplay_static, wait_object, cloned_role, source_role = [arena + k for k in (0x1c000, 0x1c100, 0x1c200, 0x1c300, 0x1c400)]

    def q(a, value): uc.mem_write(a, struct.pack('<Q', value))
    def d(a, value): uc.mem_write(a, struct.pack('<I', value & 0xffffffff))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def rd(a): return struct.unpack('<I', uc.mem_read(a, 4))[0]
    def byte(a): return uc.mem_read(a, 1)[0]
    def reg(r): return uc.reg_read(r)
    def ret(value=0):
        sp = reg(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, sp + 8)
        uc.reg_write(x.UC_X86_REG_RIP, rq(sp))

    slots, flags = set(), set()
    for ins in decoded.values():
        refs = [ins.address + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        for slot in refs:
            if ins.mnemonic == 'cmp' and ins.operands[0].size == 1: flags.add(slot)
            elif ins.mnemonic in ('lea', 'mov') and ins.operands[-1].size == 8: slots.add(slot)
    rows = [r for section in ('ScriptMetadata', 'ScriptMetadataMethod', 'ScriptString') for r in metadata[section] if r['Address'] in slots]
    assert {r['Address'] for r in rows} == slots
    slot_values = {}
    for i, row in enumerate(rows):
        pointer = arena + 0x10000 + i * 0x400
        q(base + row['Address'], pointer)
        d(pointer + 0xe0, 1)
        slot_values[row['Address']] = pointer
    types = {r['Name']: slot_values[r['Address']] for r in rows if 'Name' in r}
    assert 'UnityEngine.Object_TypeInfo' in types and 'UnityEngine.Debug_TypeInfo' in types
    assert {r['Value'] for r in rows if 'Value' in r} == {'INIT: ', '# {0}'}
    seconds_ins = decoded[0x375742]
    seconds_rva = seconds_ins.address + seconds_ins.size + seconds_ins.operands[1].mem.disp
    seconds_raw = pe.get_data(seconds_rva, 4)
    assert len(seconds_raw) == 4 and struct.unpack('<I', seconds_raw)[0] == 0x3e99999a
    services = {0x1c79fd0: 'get_game_object', 0x1c7d810: 'set_inactive', 0x112b9d0: 'array_clear',
                0x1c82480: 'unity_live', 0x1c80520: 'destroy', 0xf71c60: 'concat',
                0x1c4b380: 'context_log', 0x1c4b450: 'log', 0x282580: 'box', 0xf74df0: 'format',
                0x367970: 'refresh_character', 0x367b60: 'refresh_view', 0x1c7f160: 'start_coroutine',
                0x2b7d40: 'allocate', 0x2b6ff0: 'barrier', 0x2b7b40: 'metadata', 0x281d90: 'class_init',
                0x1c822c0: 'unity_null', 0x603240: 'clone_role', 0x1c961f0: 'wait_constructor'}
    service_names = {0x1c79fd0: 'UnityEngine.Component$$get_gameObject', 0x1c7d810: 'UnityEngine.GameObject$$SetActive',
                     0x112b9d0: 'System.Array$$Clear', 0x1c82480: 'UnityEngine.Object$$op_Inequality',
                     0x1c80520: 'UnityEngine.Object$$Destroy', 0xf71c60: 'System.String$$Concat',
                     0x1c4b380: 'UnityEngine.Debug$$Log', 0x1c4b450: 'UnityEngine.Debug$$Log',
                     0xf74df0: 'System.String$$Format', 0x367b60: 'Character$$RefreshView',
                     0x1c7f160: 'UnityEngine.MonoBehaviour$$StartCoroutine', 0x1c822c0: 'UnityEngine.Object$$op_Equality',
                     0x603240: 'ClassConv$$CreateCopyNonGeneric<object>', 0x1c961f0: 'UnityEngine.WaitForSeconds$$.ctor'}
    service_metadata = []
    for rva, name in service_names.items():
        matches = [m for m in metadata['ScriptMethod'] if m['Address'] == rva and m['Name'] == name]
        assert len(matches) == 1
        service_metadata.append(matches[0])
    state, options, visited = {}, {}, set()

    def snapshot():
        return {'data': rq(obj + 0x50), 'bluff': rq(obj + 0x58), 'register_as': rq(obj + 0x60),
                'trailer': rq(obj + 0x68), 'runtime': rq(obj + 0x70), 'dead_prefab': rq(obj + 0x98),
                'revealed': byte(obj + 0xd8), 'uses': rd(obj + 0xdc), 'previous': rd(obj + 0xe0),
                'state': rd(obj + 0xe4), 'killed_hidden': byte(obj + 0xec), 'killed_demon': byte(obj + 0xed),
                'alignment': rd(obj + 0xf8), 'id': rd(obj + 0x118), 'started': byte(obj + 0x11c),
                'role': rq(obj + 0x168), 'bluff_role': rq(obj + 0x170), 'saved_act': rq(obj + 0x198),
                'info_count': rd(info + 0x18), 'info_version': rd(info + 0x1c),
                'status_count': rd(active + 0x18), 'status_version': rd(active + 0x1c),
                'alternate_status_count': rd(alternate_active + 0x18), 'alternate_status_version': rd(alternate_active + 0x1c),
                'iterator_state': rd(iterator + 0x10), 'iterator_current': rq(iterator + 0x18)}

    def emit(kind, **details):
        state['counts'][kind] = state['counts'].get(kind, 0) + 1
        state['events'].append({'kind': kind, **details, 'actor': snapshot()})
        state['prefixes'].append(bytes(uc.mem_read(obj, 0x1b8)))
        if options.get('fail') == [kind, state['counts'][kind]]:
            state['error'] = kind
            uc.emu_stop()
            return False
        return True

    def hook(_, address, size, __):
        rva = address - base
        c, dx, r8 = reg(x.UC_X86_REG_RCX), reg(x.UC_X86_REG_RDX), reg(x.UC_X86_REG_R8)
        if address == stop:
            state['returned'] = True
            uc.emu_stop()
        elif address == callback_code:
            assert c == target and dx == target + 0x80
            if emit('state_callback'):
                if options.get('swap_status'): q(obj + 0xf0, alternate_status)
                if options.get('callback_state') is not None: d(obj + 0xe4, options['callback_state'])
                ret()
        elif address == text_code:
            assert c == number and dx == arena + 0x1f100 and r8 == number_class + 0x800
            if emit('set_text'): ret()
        elif address == yield_return:
            assert reg(x.UC_X86_REG_RAX) & 255 == 1
            assert rd(iterator + 0x10) == 1 and rq(iterator + 0x18) == wait_object
            uc.reg_write(x.UC_X86_REG_RSP, state['scheduler_sp'])
            if emit('first_yield'): ret(arena + 0x1c500)
        elif rva in services:
            kind = services[rva]
            details = {}
            if kind == 'barrier':
                assert rq(c) == dx
                details['offset'] = c - (iterator if c in (iterator + 0x18, iterator + 0x20) else obj)
                details['owner'] = 'iterator' if c in (iterator + 0x18, iterator + 0x20) else 'character'
            if kind == 'get_game_object':
                assert c in (obj, acted) and dx == 0
                details['owner'] = 'character' if c == obj else 'acted'
            if kind == 'set_inactive':
                assert c in (acted + 0x800, arena + 0x1e000, pickeds + 0x100, pickeds + 0x180) and dx == 0 and r8 == 0
                details['object'] = c
            if kind == 'array_clear': assert c == items and dx == 0 and r8 == options['info_count']
            if kind == 'metadata': assert c - base in slots
            if kind == 'class_init': assert c in (types['UnityEngine.Object_TypeInfo'], types['UnityEngine.Debug_TypeInfo'], types['Gameplay_TypeInfo'])
            if kind == 'refresh_character' or kind == 'refresh_view': assert c == obj and dx == 0
            if kind == 'start_coroutine':
                assert c == obj and dx == iterator and r8 == 0
                assert rd(iterator + 0x10) == 0 and rq(iterator + 0x20) == obj and rq(iterator + 0x18) == 0
            if not emit(kind, **details): return
            if kind == 'refresh_character' and options['joined']:
                visited.add(rva)  # Continue executing this entry instead of returning from the service.
            elif kind == 'start_coroutine' and options['joined']:
                state['scheduler_sp'] = reg(x.UC_X86_REG_RSP)
                # An authored Windows x64 caller reserves home space and leaves
                # the nested callee's entry RSP congruent to eight modulo 16.
                sp = state['scheduler_sp'] - 0x30
                assert sp % 16 == 8
                q(sp, yield_return)
                uc.reg_write(x.UC_X86_REG_RSP, sp)
                uc.reg_write(x.UC_X86_REG_RCX, iterator)
                uc.reg_write(x.UC_X86_REG_RDX, 0)
                uc.reg_write(x.UC_X86_REG_RIP, base + 0x3756b0)
            elif kind == 'get_game_object': ret(0 if options.get('null_game_object') else c + 0x800)
            elif kind == 'unity_live': ret(options['dead'] == 'live')
            elif kind == 'unity_null': assert c == 0 and dx == 0; ret(1)
            elif kind == 'class_init': d(c + 0xe0, 1); ret()
            elif kind == 'array_clear': uc.mem_write(items + 0x20, bytes(r8 * 8)); ret()
            elif kind == 'allocate':
                assert c in (types['Character.<DelayReveal>d__84_TypeInfo'], types['UnityEngine.WaitForSeconds_TypeInfo'])
                allocated = iterator if c == types['Character.<DelayReveal>d__84_TypeInfo'] else wait_object
                uc.mem_write(allocated, bytes(0x40)); q(allocated, c); ret(allocated)
            elif kind == 'clone_role':
                assert c == source_role and dx == types['Method$ClassConv.CreateCopyNonGeneric<Role>()']
                ret(0 if options['clone_null'] else cloned_role)
            elif kind == 'wait_constructor':
                assert c == wait_object and r8 == 0 and reg(x.UC_X86_REG_XMM1) & 0xffffffff == 0x3e99999a
                ret()
            elif kind in ('concat', 'format', 'box'): ret(arena + 0x1f100)
            else: ret()
        elif rva == 0x2b7d90:
            state['error'] = 'null'
            uc.emu_stop()
        else:
            assert rva in decoded and decoded[rva].size == size, hex(rva)
            visited.add(rva)

    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]

    def run(name, *, id=-100, dead='absent', info_count=2, callback=True, cold=False, fail=None, null=None,
            null_game_object=False, swap_status=False, callback_state=None, joined=False, previous_phase=20, ability_usage=10, picked_count=2,
            clone_null=False):
        options.clear()
        options.update(id=id, dead=dead, info_count=info_count, callback=callback, cold=cold, fail=fail, null=null,
                       null_game_object=null_game_object, swap_status=swap_status, callback_state=callback_state,
                       joined=joined, previous_phase=previous_phase, ability_usage=ability_usage, picked_count=picked_count, clone_null=clone_null)
        state.clear()
        state.update(events=[], counts={}, prefixes=[], error=None, returned=False)
        uc.mem_write(obj, bytes([0xa5]) * 0x1b8)
        for offset, value in {0x48: number, 0x78: arena + 0x1e000, 0x98: 0 if dead == 'absent' else arena + 0x1d000,
                              0xa8: acted, 0xf0: status, 0x148: info, 0x180: globals_callback if callback else 0}.items(): q(obj + offset, value)
        d(data + 0x134, 20); q(data + 0x28, arena + 0x1f000); d(data + 0x138, ability_usage); q(data + 0x140, source_role)
        q(obj + 0x188, 0 if null == 'pickeds' else pickeds)
        d(pickeds + 0x18, picked_count)
        q(pickeds + 0x20, 0 if null == 'picked' else pickeds + 0x100); q(pickeds + 0x28, pickeds + 0x180)
        q(types['Gameplay_TypeInfo'] + 0xb8, gameplay_static); d(gameplay_static + 0x2c, previous_phase)
        uc.mem_write(iterator, bytes(0x40))
        q(info + 0x10, items); d(info + 0x18, info_count); d(info + 0x1c, 17)
        uc.mem_write(items + 0x20, bytes([0x66]) * 32)
        for st, al in [(status, active), (alternate_status, alternate_active)]:
            q(st + 0x10, al); q(st + 0x18, resistant); q(st + 0x20, target)
            d(al + 0x18, 3); d(al + 0x1c, 23)
            q(al + 0x10, al + 0x100)
            uc.mem_write(al + 0x120, struct.pack('<iii', 10, 30, 50))
        q(number, number_class); q(number_class + 0x558, text_code); q(number_class + 0x560, number_class + 0x800)
        q(globals_callback + 0x18, callback_code); q(globals_callback + 0x28, target + 0x80); q(globals_callback + 0x40, target)
        for offset in (0xd8, 0xec, 0xed, 0x11c): uc.mem_write(obj + offset, b'\1')
        d(obj + 0xdc, 7); d(obj + 0xe0, 10); d(obj + 0xe4, 20); d(obj + 0xf8, 10); d(obj + 0x118, 73)
        if null in {'number', 'acteds', 'infos', 'rip', 'statuses'}:
            q(obj + {'number': 0x48, 'acteds': 0xa8, 'infos': 0x148, 'rip': 0x78, 'statuses': 0xf0}[null], 0)
        if null == 'active_statuses': q(status + 0x10, 0)
        for flag in flags: uc.mem_write(base + flag, bytes([0 if cold else 1]))
        for typename in ('UnityEngine.Object_TypeInfo', 'UnityEngine.Debug_TypeInfo', 'Gameplay_TypeInfo'): d(types[typename] + 0xe0, 0 if cold else 1)
        before = bytes(uc.mem_read(obj, 0x1b8))
        status_before = bytes(uc.mem_read(status, 0x28))
        sp = stack + 0x10008; q(sp, stop)
        uc.reg_write(x.UC_X86_REG_RSP, sp); uc.reg_write(x.UC_X86_REG_RCX, obj)
        uc.reg_write(x.UC_X86_REG_RDX, 0 if null == 'data' else data); uc.reg_write(x.UC_X86_REG_R8, id & 0xffffffff)
        for i, register in enumerate(registers): uc.reg_write(register, 0xabc000 + i)
        uc.emu_start(base + entries[name], stop + 0x1000, count=10000)
        final = bytes(uc.mem_read(obj, 0x1b8))
        changed = {0x50: 8, 0x58: 8, 0x98: 8, 0xd8: 1, 0xdc: 4, 0xe0: 4, 0xe4: 4, 0xed: 1, 0x118: 4, 0x11c: 1}
        if name == 'Init': changed.update({0x60: 8, 0x68: 8, 0x70: 8, 0xf8: 4})
        if swap_status: changed[0xf0] = 8
        if joined: changed[0x168] = 8
        expected = bytearray(before)
        for offset, length in changed.items(): expected[offset:offset + length] = final[offset:offset + length]
        assert final == expected, name
        assert bytes(uc.mem_read(status, 0x28)) == status_before
        for al in (active, alternate_active):
            assert bytes(uc.mem_read(al + 0x120, 12)) == struct.pack('<iii', 10, 30, 50)
        if state['returned']:
            assert state['error'] is None and reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(reg(register) == 0xabc000 + i for i, register in enumerate(registers))
        else: assert state['error'] is not None
        result = {'method': name, 'input': dict(options), 'error': state['error'], 'events': list(state['events']), 'final': snapshot()}
        return result, final, list(state['prefixes'])

    globals_callback = callback
    cases = []
    for name, id, dead, count, cb, cold, joined in itertools.product(entries, [-100, 0, 19], ['absent', 'destroyed', 'live'], [0, 2], [False, True], [False, True], [False, True]):
        result, _, _ = run(name, id=id, dead=dead, info_count=count, callback=cb, cold=cold, joined=joined)
        assert result['error'] is None
        after = result['final']
        assert after['data'] == data and after['bluff'] == 0 and after['revealed'] == 0 and after['uses'] == 1
        assert after['previous'] == 20 and after['state'] == 5 and after['killed_demon'] == 0 and after['started'] == 0
        assert after['id'] == (73 if id == -100 else id) and after['info_count'] == 0 and after['info_version'] == 18
        assert after['dead_prefab'] == (arena + 0x1d000 if dead == 'destroyed' else 0)
        assert after['status_count'] == (0 if name == 'Init' else 3)
        assert after['status_version'] == (24 if name == 'Init' else 23)
        assert after['alignment'] == (20 if name == 'Init' else 10)
        for field in ('trailer', 'runtime', 'register_as'): assert after[field] == (0 if name == 'Init' else 0xa5a5a5a5a5a5a5a5)
        kinds = [e['kind'] for e in result['events']]
        if joined:
            assert kinds[-1] == 'first_yield' and after['role'] == cloned_role
        else:
            assert kinds[-3:] == ['allocate', 'barrier', 'start_coroutine']
        assert kinds.index('refresh_character') < kinds.index('refresh_view') < kinds.index('allocate')
        if cb:
            event = next(e for e in result['events'] if e['kind'] == 'state_callback')
            assert event['actor']['status_count'] == 3 and event['actor']['state'] == 5
        cases.append(result)
    for name, phase, usage, picked_count in itertools.product(entries, [0, 20], [0, 10], [0, 2]):
        result, _, _ = run(name, joined=True, previous_phase=phase, ability_usage=usage, picked_count=picked_count, cold=True)
        assert result['error'] is None and result['final']['role'] == cloned_role
        assert result['final']['iterator_state'] == 1 and result['final']['iterator_current'] == wait_object
        assert result['events'][-1]['kind'] == 'first_yield'
        cases.append(result)
    for name in entries:
        baseline, _, prefixes = run(name, joined=True, cold=True)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            result, final, _ = run(name, joined=True, cold=True, fail=[kind, counts[kind]])
            assert result['error'] == kind and final == prefixes[index] and result['final'] == event['actor']
            cases.append(result)
        for null in ('pickeds', 'picked'):
            result, _, _ = run(name, joined=True, null=null)
            assert result['error'] == 'null'; cases.append(result)
        result, _, _ = run(name, joined=True, clone_null=True)
        assert result['error'] is None and result['final']['role'] == 0 and result['final']['iterator_state'] == 1
        cases.append(result)
    for name in entries:
        baseline, _, prefixes = run(name, id=19, dead='live', cold=True)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            result, final, _ = run(name, id=19, dead='live', cold=True, fail=[kind, counts[kind]])
            assert result['error'] == kind and final == prefixes[index]
            assert result['final'] == event['actor'] and result['events'] == baseline['events'][:index + 1]
            cases.append(result)
        for null in ('acteds', 'infos', 'rip', 'number', 'data', 'statuses', 'active_statuses'):
            result, _, _ = run(name, id=19, dead='live', null=null)
            assert result['error'] == (None if name == 'InitWithNoReset' and null in ('statuses', 'active_statuses') else 'null')
            cases.append(result)
        result, _, _ = run(name, null_game_object=True)
        assert result['error'] == 'null'; cases.append(result)
        result, _, _ = run(name, swap_status=True, callback_state=30)
        assert result['error'] is None and result['final']['state'] == 30
        assert result['final']['status_count'] == 3
        assert result['final']['alternate_status_count'] == (0 if name == 'Init' else 3)
        cases.append(result)
        for null in ('number', 'rip'):
            result, _, _ = run(name, null=null)
            assert result['error'] is None; cases.append(result)
    result, _, _ = run('Init', null='statuses', swap_status=True)
    assert result['error'] is None and result['final']['alternate_status_count'] == 0
    cases.append(result)
    # Keep every authored case and ordered event, encoding repeated snapshots as deltas.
    for case in cases:
        previous = {}
        for event in case['events']:
            current = event.pop('actor')
            event['actor_changes'] = {key: value for key, value in current.items() if previous.get(key) != value}
            previous = current
    return {'build': BUILD, 'evidence': 'native-executable-bounded-caller', 'methods': methods, 'fields': fields,
            'instruction_assertions': len(checks) + 1, 'native_instruction_count_executed': len(visited),
            'case_count': len(cases), 'services': sorted(set(services.values()) | {'state_callback', 'set_text'}),
            'service_metadata': service_metadata, 'literal_metadata': [r for r in rows if 'Value' in r],
            'wait_seconds_f32': struct.unpack('<f', seconds_raw)[0],
            'limits': ['RefreshView is an inert service. RefreshCharacter executes natively only in joined cases.',
                       'Joined cases explicitly execute first MoveNext to its first yield; the Unity scheduler remains a service.',
                       'Role cloning and WaitForSeconds construction are services; their caller publication is native.',
                       'Engine object liveness, logging, formatting, array clearing and metadata services are authored.',
                       'Injected service failures stop at the service boundary without managed exception unwinding.',
                       'Callback mutations cover state and statuses-reference replacement; arbitrary reentrancy is outside this corpus.'],
            'cases': cases}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({'cases': report['case_count'], 'instructions': report['native_instruction_count_executed']}))
