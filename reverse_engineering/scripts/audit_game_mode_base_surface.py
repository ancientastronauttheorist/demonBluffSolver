"""Audit the complete base GameMode declaration surface and remaining native bodies.

Abstract members have no native body. Shared no-op aliases stay supplemental;
no typed target membership or live game state is changed.
"""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD


ABSTRACT = {'GetGameMode': 4, 'Init': 5, 'LoadGame': 6, 'GetStartingLevel': 9,
            'GetResetLevel': 10, 'GetCurrentAscension': 11, 'GetPreviousAscension': 12,
            'CanResetLevel': 13, 'OnStageCompleted': 16, 'UpdateScore': 17,
            'GetScore': 18, 'GetMaxLevel': 19, 'GetCurrentLevel': 20, 'IsLocked': 21, 'GetScores': 22}
NOOPS = {'OnLoadGame': 7, 'DeInit': 8, 'AscensionComplete': 14, 'AbandonRun': 15}
SUMMARY = 0x3dd160
LITERAL = 0x26df1b8
INIT_FLAG = 0x288c64a


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
    match = re.search(r'^public abstract class GameMode // TypeDefIndex: 5932\s*\{(.*?)^\}', dump, re.M | re.S)
    assert match is not None
    declarations = re.findall(r'// RVA: (\S+) Offset: [^\n]*?\n\s*((?:public|protected) [^\n]+)', match[1])
    assert len(declarations) == 21
    for name, slot in ABSTRACT.items():
        pattern = rf'// RVA: -1 Offset: -1 Slot: {slot}\n\s*public abstract [^\n]+ {name}\([^\n]*\);'
        assert re.search(pattern, match[1]) is not None, name
        assert not any(row['Name'] == 'GameMode$$' + name for row in script['ScriptMethod'])
    supplemental = {}
    for name, address, signature in [
        *[(name, 0x33ed50, f'void GameMode__{name} (GameMode_o* __this, const MethodInfo* method);') for name in NOOPS],
        ('GetSummaryScores', SUMMARY, 'System_String_o* GameMode__GetSummaryScores (GameMode_o* __this, const MethodInfo* method);'),
        ('.ctor', 0x357920, 'void GameMode___ctor (GameMode_o* __this, const MethodInfo* method);')]:
        rows = [row for row in script['ScriptMethod'] if row['Name'] == 'GameMode$$' + name]
        assert len(rows) == 1 and rows[0]['Address'] == address and rows[0]['Signature'] == signature
        supplemental[name] = rows[0]
    for name, slot in NOOPS.items():
        assert re.search(rf'// RVA: 0x33ED50 [^\n]* Slot: {slot}\n\s*public virtual void {name}\(\)', match[1]) is not None
    assert re.search(r'// RVA: 0x3DD160 [^\n]* Slot: 23\n\s*public virtual string GetSummaryScores\(\)', match[1]) is not None
    literals = [row for row in script['ScriptString'] if row['Address'] == LITERAL]
    assert len(literals) == 1 and literals[0]['Value'] == ''
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    ranges = [(SUMMARY, 0x3dd18c), (0x33ed50, 0x33ed53), (0x357920, 0x357927)]
    decoded = {}
    for begin, end in ranges:
        instructions = list(cs.disasm(pe.get_data(begin, end - begin), begin))
        assert instructions[0].address == begin and sum(i.size for i in instructions) == end - begin
        assert instructions[-1].address + instructions[-1].size == end
        decoded.update({i.address: i for i in instructions})
    checks = {0x3dd160: ('sub', 'rsp, 0x28'), 0x3dd16b: ('jne', '0x3dd180'),
              0x3dd174: ('call', '0x2b7b40'), 0x3dd187: ('add', 'rsp, 0x28'),
              0x3dd18b: ('ret', ''), 0x33ed50: ('ret', '0'),
              0x357920: ('xor', 'edx, edx'), 0x357922: ('jmp', '0x33ed50')}
    for rva, expected in checks.items():
        assert rva in decoded and (decoded[rva].mnemonic, decoded[rva].op_str) == expected
    for rva, name, register, target in [(0x3dd164, 'cmp', None, INIT_FLAG), (0x3dd16d, 'lea', 'rcx', LITERAL),
                                        (0x3dd179, 'mov', None, INIT_FLAG), (0x3dd180, 'mov', 'rax', LITERAL)]:
        ins = decoded[rva]
        assert ins.mnemonic == name
        if register is not None:
            assert ins.reg_name(ins.operands[0].reg) == register
        refs = [ins.address + ins.size + o.mem.disp for o in ins.operands if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP]
        assert refs == [target]
    assert decoded[0x3dd164].operands[1].imm == 0 and decoded[0x3dd179].operands[1].imm == 1
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    for address in (arena, stack, stop):
        uc.mem_map(address, 0x10000)
    def q(address, value):
        uc.mem_write(address, struct.pack('<Q', value))
    visited, state, options = set(), {}, {}
    def hook(_, address, size, __):
        if address == base + 0x2b7b40:
            assert uc.reg_read(x.UC_X86_REG_RCX) == base + LITERAL
            state['init_calls'] += 1
            if options['fail_init']:
                state['failed'] = True
                uc.emu_stop()
                return
            q(base + LITERAL, options['resolved_pointer'])
            sp = uc.reg_read(x.UC_X86_REG_RSP)
            target = struct.unpack('<Q', uc.mem_read(sp, 8))[0]
            uc.reg_write(x.UC_X86_REG_RSP, sp + 8)
            uc.reg_write(x.UC_X86_REG_RIP, target)
            return
        rva = address - base
        assert rva in decoded and decoded[rva].size == size
        visited.add(rva)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    registers += [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
    seeds = {reg: (0xabcdef0123456000 if index < 8 else 0xabcdef0123456789abcdef0123456000) + index for index, reg in enumerate(registers)}
    cases = []
    for operation, receiver_null, flag, pointer, fail_init in itertools.product(
            ['GetSummaryScores', *NOOPS, '.ctor'], [False, True], [0, 1, 255], [0, arena + 0x1000], [False, True]):
        # Failure injection belongs only to cold metadata initialization.
        if fail_init and (operation != 'GetSummaryScores' or flag != 0):
            continue
        state.clear()
        state.update(init_calls=0, failed=False)
        options.clear()
        options.update(fail_init=fail_init, resolved_pointer=pointer)
        uc.mem_write(arena, bytes([0xa5]) * 512)
        uc.mem_write(base + INIT_FLAG, bytes([flag]))
        q(base + LITERAL, arena + 0x2000 if flag == 0 else pointer)
        source = bytes(uc.mem_read(arena, 512))
        sp = stack + 0x8008
        q(sp, stop)
        uc.reg_write(x.UC_X86_REG_RSP, sp)
        uc.reg_write(x.UC_X86_REG_RCX, 0 if receiver_null else arena)
        uc.reg_write(x.UC_X86_REG_RDX, 0xabcdef00)
        uc.reg_write(x.UC_X86_REG_RAX, 0x987654321)
        for reg, seed in seeds.items():
            uc.reg_write(reg, seed)
        uc.emu_start(base + supplemental[operation]['Address'], stop, count=100)
        assert bytes(uc.mem_read(arena, 512)) == source
        if not state['failed']:
            assert uc.reg_read(x.UC_X86_REG_RIP) == stop and uc.reg_read(x.UC_X86_REG_RSP) == sp + 8
            assert all(uc.reg_read(reg) == seed for reg, seed in seeds.items())
            assert uc.reg_read(x.UC_X86_REG_RAX) == (pointer if operation == 'GetSummaryScores' else 0x987654321)
        expected_calls = int(operation == 'GetSummaryScores' and flag == 0)
        assert state['init_calls'] == expected_calls and state['failed'] == fail_init
        actual_flag = bytes(uc.mem_read(base + INIT_FLAG, 1))[0]
        assert actual_flag == (1 if expected_calls and not fail_init else flag)
        expected_pointer = pointer if flag != 0 or (expected_calls and not fail_init) else arena + 0x2000
        assert struct.unpack('<Q', uc.mem_read(base + LITERAL, 8))[0] == expected_pointer
        if operation == '.ctor':
            assert uc.reg_read(x.UC_X86_REG_RDX) == 0
        elif operation in NOOPS:
            assert uc.reg_read(x.UC_X86_REG_RDX) == 0xabcdef00
        cases.append({'operation': operation, 'null_receiver': receiver_null, 'initial_flag': flag,
                      'null_literal_pointer': pointer == 0, 'init_failure': fail_init,
                      'init_calls': state['init_calls'], 'result_flag': actual_flag,
                      'object_bytes_preserved': 512})
    assert visited == set(decoded)
    return {'schema_version': 1, 'build_id': BUILD, 'base_declaration_count': len(declarations),
            'abstract_declarations_without_native_bodies': ABSTRACT, 'shared_noop_slots': NOOPS,
            'supplemental_native_declarations': supplemental, 'summary_virtual_slot': 23,
            'summary_literal': {'cell_rva': hex(LITERAL), 'value': '', 'metadata_flag_rva': hex(INIT_FLAG)},
            'native_ranges': [[hex(a), hex(b)] for a, b in ranges], 'instruction_assertions': len(checks) + 4,
            'native_case_count': len(cases), 'visited_instruction_count': len(visited),
            'unvisited_instruction_count': 0, 'cases': cases,
            'scope': 'All 21 base GameMode declarations classified; 15 abstract slots have no native ScriptMethod body. Native summary getter, four shared no-op declarations, and base constructor execute with an explicit metadata-literal initialization gateway. Shared no-op ret 0 preserves receiver bytes and return register; void methods have no return-value contract. Abstract/overridden derived-mode behavior and managed exception unwinding remain outside scope.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified all {result['base_declaration_count']} base GameMode declarations and {result['native_case_count']} native cases")
