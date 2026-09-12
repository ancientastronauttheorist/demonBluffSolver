"""Native Oracle-eye event forwarding, without executing UI subscribers."""
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
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == digest.upper()
        return raw
    raw = pinned(game_root / 'GameAssembly.dll', build['inputs']['game_assembly']['sha256'])
    script = json.loads(pinned(dumper_root / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump = pinned(dumper_root / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    body = re.search(r'^public static class GameplayEvents // TypeDefIndex: 5519\s*\{(.*?)^\}', dump, re.M | re.S)
    assert body and 'public static Action OnShowEyeOracleInfo; // 0xB8' in body[1]
    assert 'public static Action OnHideEyeOracleInfo; // 0xC0' in body[1]
    operations = [('HoverOverOracleEye', 0x37dde0, 0x37de2f, 0xb8), ('HideOracleEyeInfo', 0x37dd90, 0x37dddf, 0xc0)]
    exact = []
    for name, start, _, _ in operations:
        rows = [m for m in script['ScriptMethod'] if m['Name'] == f'Gameplay$${name}']
        assert len(rows) == 1 and rows[0]['Address'] == start
        assert rows[0]['Signature'] == f'void Gameplay__{name} (Gameplay_o* __this, const MethodInfo* method);'
        exact.append(rows[0])
    pe = pefile.PE(data=raw, fast_load=True); base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    decoded = {}
    for _, start, end, _ in operations:
        ins = list(cs.disasm(pe.get_data(start, end - start), start))
        assert ins[0].address == start and sum(i.size for i in ins) == end - start
        assert ins[-1].address + ins[-1].size == end
        decoded.update({i.address: i for i in ins})
    checks = {0x37ddbe: ('mov', 'rax, qword ptr [rcx + 0xc0]'), 0x37de0e: ('mov', 'rax, qword ptr [rcx + 0xb8]'),
              0x37ddca: ('mov', 'rdx, qword ptr [rax + 0x28]'), 0x37ddce: ('mov', 'rcx, qword ptr [rax + 0x40]'),
              0x37ddd6: ('jmp', 'qword ptr [rax + 0x18]'), 0x37de1a: ('mov', 'rdx, qword ptr [rax + 0x28]'),
              0x37de1e: ('mov', 'rcx, qword ptr [rax + 0x40]'), 0x37de26: ('jmp', 'qword ptr [rax + 0x18]')}
    for rva, expected in checks.items(): assert (decoded[rva].mnemonic, decoded[rva].op_str) == expected
    def rip(rva):
        i = decoded[rva]
        refs = [i.address + i.size + o.mem.disp for o in i.operands if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP]
        assert len(refs) == 1
        return refs[0]
    slot = rip(0x37ddb0)
    assert rip(0x37de00) == slot
    metadata = [row for row in script['ScriptMetadata'] if row['Address'] == slot]
    assert len(metadata) == 1 and metadata[0]['Name'] == 'GameplayEvents_TypeInfo'
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095); uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    for a in (arena, stack, stop): uc.mem_map(a, 0x10000)
    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    cls, static, delegate = arena + 0x1000, arena + 0x2000, arena + 0x3000
    state, options, visited = {}, {}, set()
    def ret():
        sp = uc.reg_read(x.UC_X86_REG_RSP); uc.reg_write(x.UC_X86_REG_RSP, sp + 8); uc.reg_write(x.UC_X86_REG_RIP, rq(sp))
    def hook(_, address, size, __):
        if address == stop: uc.emu_stop(); return
        if address == base + 0x2b7b40:
            assert uc.reg_read(x.UC_X86_REG_RCX) == base + slot
            state['events'].append('metadata')
            if options['fail'] == 'metadata': state['error'] = 'metadata'; uc.emu_stop()
            else: ret()
        elif address == stop + 0x100:
            state['events'].append('callback')
            assert uc.reg_read(x.UC_X86_REG_RCX) == options['arg0']
            assert uc.reg_read(x.UC_X86_REG_RDX) == options['method_arg']
            if options['fail'] == 'callback': state['error'] = 'callback'; uc.emu_stop()
            else: ret()
        else:
            rva = address - base
            assert rva in decoded and decoded[rva].size == size
            visited.add(rva)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    cases = []
    for operation, flag, present, receiver_null, forwarded_arg0_null, fail in itertools.product(
            operations, [0, 1, 255], [False, True], [False, True], [False, True], [None, 'metadata', 'callback']):
        name, entry, _, offset = operation
        uc.mem_write(arena, bytes([0xa5]) * 0x4000)
        q(base + slot, cls); q(cls + 0xb8, static)
        q(static + 0xb8, delegate if offset == 0xb8 and present else 0)
        q(static + 0xc0, delegate if offset == 0xc0 and present else 0)
        q(delegate + 0x18, stop + 0x100)
        arg0 = 0 if forwarded_arg0_null else arena + 0x800
        method_arg = 0 if forwarded_arg0_null else 0xabc123
        q(delegate + 0x40, arg0); q(delegate + 0x28, method_arg)
        for rva in [0x37dd94, 0x37dde4]: uc.mem_write(base + rip(rva), bytes([flag]))
        before = bytes(uc.mem_read(arena, 0x4000)); state.clear(); state.update(events=[], error=None)
        options.clear(); options.update(fail=fail, arg0=arg0, method_arg=method_arg)
        sp = stack + 0x8008; q(sp, stop)
        uc.reg_write(x.UC_X86_REG_RSP, sp); uc.reg_write(x.UC_X86_REG_RCX, 0 if receiver_null else arena)
        for index, register in enumerate(registers): uc.reg_write(register, 0xabc000 + index)
        uc.emu_start(base + entry, stop, count=1000)
        expected_events = ['metadata'] if flag == 0 else []
        expected_error = 'metadata' if flag == 0 and fail == 'metadata' else None
        if expected_error is None and present:
            expected_events.append('callback')
            if fail == 'callback': expected_error = 'callback'
        assert state['events'] == expected_events and state['error'] == expected_error
        assert bytes(uc.mem_read(arena, 0x4000)) == before
        if not expected_error:
            assert uc.reg_read(x.UC_X86_REG_RIP) == stop and uc.reg_read(x.UC_X86_REG_RSP) == sp + 8
            assert all(uc.reg_read(register) == 0xabc000 + index for index, register in enumerate(registers))
        cases.append({'method': name, 'metadata_flag': flag, 'subscribed': present, 'receiver_null': receiver_null,
                      'forwarded_arg0_null': forwarded_arg0_null, 'method_argument': method_arg, 'failure': fail,
                      'events': state['events'][:], 'error': state['error'], 'modeled_state_unchanged': True})
    return {'build_id': BUILD, 'exact_declarations': exact, 'metadata_binding': metadata[0],
            'cases_passed': len(cases), 'native_instructions_executed': len(visited), 'native_relationships': len(checks),
            'cases': cases, 'scope': 'Exact native event forwarding with warmed/cold metadata and explicit preserving callback service. Subscriber topology, callback bodies, actual UI appearance and managed unwinding remain outside scope.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('game_root', type=Path); p.add_argument('dumper_root', type=Path); p.add_argument('--output', type=Path, required=True)
    a = p.parse_args(); report = audit(a.game_root, a.dumper_root)
    a.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Passed {report['cases_passed']} native Oracle-eye caller cases")
