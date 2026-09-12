"""Native Gameplay roster replacement and same-hand coroutine launch callers."""
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
    build = json.loads((repo / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((repo / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(path, digest):
        data = path.read_bytes()
        assert hashlib.sha256(data).hexdigest().upper() == digest.upper()
        return data
    raw = pinned(game_root / 'GameAssembly.dll', build['inputs']['game_assembly']['sha256'])
    script = json.loads(pinned(dumper_root / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump = pinned(dumper_root / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    body = re.search(r'^public class Gameplay : MonoBehaviour // TypeDefIndex: 5604\s*\{(.*?)^\}', dump, re.M | re.S)
    assert body
    field_names = ['savedTownsfolks', 'savedOutsiders', 'savedMinions', 'savedDemons']
    for name, offset in zip(field_names, [0x48, 0x50, 0x58, 0x60]):
        assert f'private List<CharacterData> {name}; // 0x{offset:X}' in body[1]
    exact = []
    for name, rva in [('ResetSavedCharacters', 0x37fde0), ('SameHandOut', 0x380260)]:
        rows = [m for m in script['ScriptMethod'] if m['Name'] == f'Gameplay$${name}']
        assert len(rows) == 1 and rows[0]['Address'] == rva
        assert rows[0]['Signature'] == f'void Gameplay__{name} (Gameplay_o* __this, const MethodInfo* method);'
        exact.append(rows[0])
    pe = pefile.PE(data=raw, fast_load=True); base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    decoded = {}
    for start, end in [(0x37fde0, 0x37ffbb), (0x380260, 0x3802d0), (0x33ed50, 0x33ed53)]:
        ins = list(cs.disasm(pe.get_data(start, end - start), start))
        assert ins[0].address == start and sum(i.size for i in ins) == end - start
        assert ins[-1].address + ins[-1].size == end
        decoded.update({i.address: i for i in ins})
    checks = {0x37fe5b: ('call', '0x1c822c0'), 0x37fe62: ('jne', '0x37ffab'),
              0x37fe7b: ('mov', 'rax, qword ptr [rcx]'), 0x37fe87: ('mov', 'rsi, qword ptr [rax + 0x20]'),
              0x37fea3: ('call', '0x3dc6f0'), 0x37fec7: ('call', '0xb610a0'),
              0x37fed3: ('mov', 'qword ptr [rcx], rdi'), 0x37ff99: ('mov', 'qword ptr [rcx], rdi'),
              0x38029d: ('call', '0x33ed50'), 0x3802a9: ('mov', 'qword ptr [rcx], rdi'),
              0x3802ac: ('mov', 'dword ptr [rbx + 0x10], 0'), 0x3802cb: ('jmp', '0x1c7f160'),
              0x33ed50: ('ret', '0')}
    for rva, expected in checks.items():
        assert (decoded[rva].mnemonic, decoded[rva].op_str) == expected
    def rip(rva):
        i = decoded[rva]
        refs = [i.address + i.size + o.mem.disp for o in i.operands if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP]
        assert len(refs) == 1
        return refs[0]
    slots = {rip(rva) for rva in [0x37fdf6, 0x37fe02, 0x37fe0e, 0x37fe1a, 0x380276]}
    metadata = [r for r in script['ScriptMetadata'] + script['ScriptMetadataMethod'] if r['Address'] in slots]
    by_name = {r['Name']: r['Address'] for r in metadata}
    assert set(by_name) == {'Method$System.Collections.Generic.List<CharacterData>..ctor()',
        'System.Collections.Generic.List<CharacterData>_TypeInfo', 'UnityEngine.Object_TypeInfo',
        'ProjectContext_TypeInfo', 'Gameplay.<SetupDelay>d__47_TypeInfo'}
    assert len(metadata) == 5
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095); uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    for a in (arena, stack, stop): uc.mem_map(a, 0x20000)
    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def d(a, v): uc.mem_write(a, struct.pack('<I', v & 0xffffffff))
    obj, static, project, game = arena + 0x100, arena + 0x2000, arena + 0x3000, arena + 0x4000
    identities = {name: arena + 0x5000 + index * 0x100 for index, name in enumerate(by_name)}
    state, opt, visited = {}, {}, set()
    def emit(kind, **args):
        state['counts'][kind] = state['counts'].get(kind, 0) + 1
        state['events'].append({'kind': kind, **args})
        if opt.get('fail') == [kind, state['counts'][kind]]:
            state['error'] = kind; uc.emu_stop(); return False
        return True
    def ret(value=0):
        sp = uc.reg_read(x.UC_X86_REG_RSP); uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, sp + 8); uc.reg_write(x.UC_X86_REG_RIP, rq(sp))
    def hook(_, address, size, __):
        rva = address - base; c = uc.reg_read(x.UC_X86_REG_RCX); v = uc.reg_read(x.UC_X86_REG_RDX)
        if address == stop: uc.emu_stop()
        elif rva == 0x2b7b40:
            assert c - base in slots
            if emit('metadata'): ret()
        elif rva == 0x281d90:
            assert c == identities['UnityEngine.Object_TypeInfo']
            if emit('class_init'): d(c + 0xe0, 1); ret()
        elif rva == 0x1c822c0:
            assert c == (project if opt['project_present'] else 0) and v == 0
            if emit('unity_null_check', project_present=bool(c)):
                if opt.get('replace_after_equality'): q(static, 0)
                ret(0xabc000 | int(opt['equality_null']))
        elif rva == 0x3dc6f0:
            assert c == game and uc.reg_read(x.UC_X86_REG_R8) == 0
            assert v in [10, 20, 30, 100]
            if emit('typed_pool', character_type=v):
                if opt.get('replace_after_first_pool'): q(static, 0)
                ret(0 if opt['null_pool'] == v else arena + 0x8000 + v * 8)
        elif rva == 0x2b7d40:
            expected_type = 'Gameplay.<SetupDelay>d__47_TypeInfo' if opt['same_hand'] else 'System.Collections.Generic.List<CharacterData>_TypeInfo'
            assert c == identities[expected_type]
            if emit('allocate'):
                state['allocated'] += 1; p = arena + 0x10000 + state['allocated'] * 0x100
                uc.mem_write(p, bytes(0x80)); q(p, c); ret(p)
        elif rva == 0xb610a0:
            assert c == arena + 0x10000 + state['allocated'] * 0x100
            assert uc.reg_read(x.UC_X86_REG_R8) == identities['Method$System.Collections.Generic.List<CharacterData>..ctor()']
            if emit('copy_ctor', null_source=v == 0):
                if v == 0: state['error'] = 'null_collection'; uc.emu_stop()
                else: state['copied'][c] = (v - arena - 0x8000) // 8; ret()
        elif rva == 0x2b6ff0:
            assert rq(c) == v
            if emit('barrier', offset=(0x20 if opt['same_hand'] else c - obj)): ret()
        elif rva == 0x1c7f160:
            assert c == (0 if opt['null_receiver'] else obj)
            assert v == arena + 0x10100 and rq(v + 0x20) == c and rq(v + 0x18) == 0
            assert bytes(uc.mem_read(v + 0x10, 4)) == bytes(4)
            if emit('start_coroutine', null_receiver=c == 0): ret(arena + 0x18000)
        elif rva == 0x2b7d90: state['error'] = 'null'; uc.emu_stop()
        else:
            assert rva in decoded and decoded[rva].size == size, hex(rva)
            visited.add(rva)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    regs = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    def run(same_hand=False, cold=False, fail=None, project_present=True, game_present=True,
            equality_null=False, null_pool=None, replace_after_equality=False, replace_after_first_pool=False, null_receiver=False):
        opt.clear(); opt.update(same_hand=same_hand, cold=cold, fail=fail,
            project_present=project_present, game_present=game_present, equality_null=equality_null,
            null_pool=null_pool, replace_after_equality=replace_after_equality,
            replace_after_first_pool=replace_after_first_pool, null_receiver=null_receiver)
        state.clear(); state.update(events=[], counts={}, error=None, allocated=0, copied={})
        uc.mem_write(obj, bytes([0xa5]) * 0x100)
        for name, slot in by_name.items(): q(base + slot, identities[name])
        q(identities['ProjectContext_TypeInfo'] + 0xb8, static); q(static, project if project_present else 0)
        q(project + 0x20, game if game_present else 0); d(identities['UnityEngine.Object_TypeInfo'] + 0xe0, 0 if cold else 1)
        for rva in [0x37fdea, 0x38026a]: uc.mem_write(base + rip(rva), bytes([0 if cold else 1]))
        before = bytes(uc.mem_read(obj, 0x100)); sp = stack + 0x10008; q(sp, stop)
        uc.reg_write(x.UC_X86_REG_RSP, sp); uc.reg_write(x.UC_X86_REG_RCX, 0 if null_receiver else obj)
        for i, reg in enumerate(regs): uc.reg_write(reg, 0xabc000 + i)
        uc.emu_start(base + (0x380260 if same_hand else 0x37fde0), stop, count=10000)
        final = bytes(uc.mem_read(obj, 0x100)); fields = {}; expected = bytearray(before)
        for name, offset in zip(field_names, [0x48, 0x50, 0x58, 0x60]):
            if final[offset:offset + 8] != before[offset:offset + 8]:
                fields[name] = state['copied'][rq(obj + offset)]; expected[offset:offset + 8] = final[offset:offset + 8]
        assert final == expected
        if same_hand: assert final == before
        if not state['error']:
            assert uc.reg_read(x.UC_X86_REG_RIP) == stop and uc.reg_read(x.UC_X86_REG_RSP) == sp + 8
            assert all(uc.reg_read(reg) == 0xabc000 + i for i, reg in enumerate(regs))
        return {'method': 'SameHandOut' if same_hand else 'ResetSavedCharacters',
                'input': {k: v for k, v in opt.items() if k not in ['opt', 'state']},
                'events': state['events'][:], 'replaced_fields': fields, 'allocation_count': state['allocated'],
                'project_is_null_after': rq(static) == 0, 'error': state['error']}
    cases = []
    failures = [None, ['class_init', 1], ['unity_null_check', 1], *[[k, i] for k in ['metadata', 'typed_pool', 'allocate', 'copy_ctor', 'barrier'] for i in range(1, 5)]]
    for cold, fail in itertools.product([False, True], failures):
        case = run(cold=cold, fail=fail)
        error = case['error']; n = fail[1] if fail else 0
        count = n if error == 'barrier' else n - 1 if error in ['typed_pool', 'allocate', 'copy_ctor'] else 0 if error else 4
        assert list(case['replaced_fields']) == field_names[:count]
        cases.append(case)
    for null_pool in [10, 20, 30, 100]:
        case = run(null_pool=null_pool); assert case['error'] == 'null_collection'
        assert len(case['replaced_fields']) == [10, 20, 30, 100].index(null_pool); cases.append(case)
    for project_present, game_present, equality_null in itertools.product([False, True], repeat=3):
        case = run(project_present=project_present, game_present=game_present, equality_null=equality_null)
        assert len(case['replaced_fields']) == (4 if project_present and game_present and not equality_null else 0)
        assert case['error'] == ('null' if not equality_null and not (project_present and game_present) else None)
        cases.append(case)
    case = run(replace_after_equality=True); assert case['error'] == 'null' and not case['replaced_fields']; cases.append(case)
    case = run(replace_after_first_pool=True); assert len(case['replaced_fields']) == 4 and case['error'] is None; cases.append(case)
    for cold, null_receiver, fail in itertools.product([False, True], [False, True], [None, ['metadata', 1], ['allocate', 1], ['barrier', 1], ['start_coroutine', 1]]):
        case = run(same_hand=True, cold=cold, fail=fail, null_receiver=null_receiver)
        assert case['allocation_count'] == (0 if (cold and fail == ['metadata', 1]) or fail == ['allocate', 1] else 1)
        cases.append(case)
    return {'build_id': BUILD, 'exact_declarations': exact, 'metadata_bindings': metadata, 'cases_passed': len(cases),
            'native_instructions_executed': len(visited), 'native_relationships': len(checks), 'cases': cases,
            'scope': 'Native caller instructions with explicit metadata/class initialization, Unity equality, typed-pool, allocation/copy, barrier and StartCoroutine gateways. Source pool contents and engine coroutine scheduling remain unmodeled. Controlled project replacement tests distinguish reread from captured GameData identity.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('game_root', type=Path); p.add_argument('dumper_root', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); report = audit(a.game_root, a.dumper_root)
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Passed {report['cases_passed']} native roster-reset/launch cases")
