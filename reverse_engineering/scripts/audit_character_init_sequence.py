"""Native repeated-body initialization with retained first-yield continuations.

Refresh/UI/runtime services are bounded as in audit_character_init. This distinct
fixture runs multiple actual Init bodies against the same actor memory, without
resetting it between calls. It never resumes an already yielded continuation.
"""
import argparse
import hashlib
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

    def pin(path, sha):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == sha.upper()
        return raw

    raw = pin(game_root / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    metadata = json.loads(pin(dumper_root / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump = pin(dumper_root / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    previous = json.loads((repo / f'reports/{BUILD}_character_init.json').read_text(encoding='utf-8'))
    assert previous['build'] == BUILD and previous['case_count'] == 475
    for name, declarations in previous['fields'].items():
        body = re.search(r'^[^\n]*class ' + re.escape(name) + r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', dump, re.M | re.S)
        assert body and all(declaration in body[1] for declaration in declarations)
    methods = [m for m in previous['methods'] if m['Address'] != 0x367970]
    assert {m['Name']: m['Address'] for m in methods} == {
        'Character$$Init': 0x365a20, 'Character$$InitWithNoReset': 0x365720,
        'Character.<DelayReveal>d__84$$MoveNext': 0x3756b0}
    assert all(m in metadata['ScriptMethod'] for m in methods)
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    decoded = {}
    for method in methods:
        start = method['Address']
        end = min(m['Address'] for m in metadata['ScriptMethod'] if m['Address'] > start)
        ins = list(cs.disasm(pe.get_data(start, end - start), start))
        while ins[-1].mnemonic == 'int3': ins.pop()
        assert ins[0].address == start and all(a.address + a.size == b.address for a, b in zip(ins, ins[1:]))
        decoded.update({i.address: i for i in ins})
    noop = list(cs.disasm(pe.get_data(0x33ed50, 3), 0x33ed50))
    assert len(noop) == 1 and (noop[0].mnemonic, noop[0].op_str) == ('ret', '0')
    decoded[0x33ed50] = noop[0]
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    uc.mem_map(arena, 0x80000); uc.mem_map(stack, 0x20000); uc.mem_map(stop, 0x1000)

    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def d(a, v): uc.mem_write(a, struct.pack('<I', v & 0xffffffff))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def rd(a): return struct.unpack('<I', uc.mem_read(a, 4))[0]
    def reg(r): return uc.reg_read(r)
    def ret(value=0):
        sp = reg(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value); uc.reg_write(x.UC_X86_REG_RSP, sp + 8)
        uc.reg_write(x.UC_X86_REG_RIP, rq(sp))

    flags, slots = set(), set()
    for ins in decoded.values():
        for op in ins.operands:
            if op.type != capstone.CS_OP_MEM or op.mem.base != capstone.x86.X86_REG_RIP: continue
            slot = ins.address + ins.size + op.mem.disp
            if ins.mnemonic == 'cmp' and op.size == 1: flags.add(slot)
            elif ins.mnemonic in ('lea', 'mov') and op.size == 8: slots.add(slot)
    rows = [m for section in ('ScriptMetadata', 'ScriptMetadataMethod', 'ScriptString') for m in metadata[section] if m['Address'] in slots]
    assert {r['Address'] for r in rows} == slots
    types = {}
    for i, row in enumerate(rows):
        pointer = arena + 0x20000 + i * 0x400
        q(base + row['Address'], pointer); d(pointer + 0xe0, 1)
        if 'Name' in row: types[row['Name']] = pointer
    for flag in flags: uc.mem_write(base + flag, b'\1')
    actors = [arena + i * 0x1000 for i in (1, 2)]
    data = [arena + i for i in (0x10000, 0x11000)]
    source_roles = [arena + i for i in (0x14000, 0x15000)]
    state, visited = {}, set()
    services = {0x2b6ff0, 0x1c79fd0, 0x1c7d810, 0x112b9d0, 0x1c82480, 0xf71c60,
                0x1c4b450, 0x1c4b380, 0x282580, 0xf74df0, 0x367970, 0x367b60,
                0x2b7d40, 0x1c7f160, 0x603240, 0x1c961f0}

    def hook(_, address, size, __):
        rva = address - base
        c, dx, r8 = reg(x.UC_X86_REG_RCX), reg(x.UC_X86_REG_RDX), reg(x.UC_X86_REG_R8)
        if address == stop:
            state['returned'] = True; uc.emu_stop()
        elif address == stop + 0x100:
            assert c == state['actor'] + 0x600
            ret()
        elif address == stop + 0x200:
            assert reg(x.UC_X86_REG_RAX) & 255 == 1
            assert rd(state['iterator'] + 0x10) == 1 and rq(state['iterator'] + 0x18) == state['wait']
            uc.reg_write(x.UC_X86_REG_RSP, state['outer_sp']); ret()
        elif rva in services:
            state['services'].append(hex(rva))
            if rva == 0x2b6ff0: assert rq(c) == dx; ret()
            elif rva == 0x1c79fd0: assert c in (state['actor'], state['actor'] + 0x500); ret(c + 0x80)
            elif rva == 0x1c7d810: assert dx == 0 and r8 == 0; ret()
            elif rva == 0x112b9d0: assert dx == 0; uc.mem_write(c + 0x20, bytes(8 * r8)); ret()
            elif rva == 0x1c82480: assert c == 0 and dx == 0; ret(0)
            elif rva in (0xf71c60, 0x282580, 0xf74df0): ret(arena + 0x60000)
            elif rva == 0x2b7d40:
                target = state['iterator'] if c == types['Character.<DelayReveal>d__84_TypeInfo'] else state['wait']
                assert c in (types['Character.<DelayReveal>d__84_TypeInfo'], types['UnityEngine.WaitForSeconds_TypeInfo'])
                uc.mem_write(target, bytes(0x40)); q(target, c); ret(target)
            elif rva == 0x1c7f160:
                assert c == state['actor'] and dx == state['iterator'] and r8 == 0
                state['outer_sp'] = reg(x.UC_X86_REG_RSP)
                sp = state['outer_sp'] - 0x30; assert sp % 16 == 8; q(sp, stop + 0x200)
                uc.reg_write(x.UC_X86_REG_RSP, sp); uc.reg_write(x.UC_X86_REG_RCX, dx)
                uc.reg_write(x.UC_X86_REG_RDX, 0); uc.reg_write(x.UC_X86_REG_RIP, base + 0x3756b0)
            elif rva == 0x603240:
                assert c == source_roles[state['data_index']] and dx == types['Method$ClassConv.CreateCopyNonGeneric<Role>()']
                state['clone_calls'].append({'source': c, 'result': state['clone']})
                ret(state['clone'])
            elif rva == 0x1c961f0:
                assert c == state['wait'] and r8 == 0 and reg(x.UC_X86_REG_XMM1) & 0xffffffff == 0x3e99999a
                ret()
            else: ret()
        else:
            assert rva in decoded and decoded[rva].size == size, hex(rva)
            visited.add(rva)

    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]

    def snapshot(actor):
        return {'actor': actor, 'data': rq(actor + 0x50), 'role': rq(actor + 0x168), 'bluff_role': rq(actor + 0x170),
                'bluff': rq(actor + 0x58), 'runtime': rq(actor + 0x70), 'register_as': rq(actor + 0x60),
                'alignment': rd(actor + 0xf8), 'id': rd(actor + 0x118), 'state': rd(actor + 0xe4),
                'previous': rd(actor + 0xe0), 'info_version': rd(actor + 0x41c),
                'status_count': rd(actor + 0x398), 'status_version': rd(actor + 0x39c),
                'resistance': rq(actor + 0x318), 'status_target': rq(actor + 0x320)}

    cases = []
    for methods_sequence in [('Init', 'Init', 'Init'), ('InitWithNoReset', 'InitWithNoReset', 'Init'),
                             ('Init', 'InitWithNoReset', 'InitWithNoReset'), ('InitWithNoReset', 'Init', 'Init')]:
        for i, actor in enumerate(actors):
            uc.mem_write(actor, bytes([0xa5]) * 0x1b8)
            for offset, pointer in {0x48: actor + 0x600, 0x50: data[i], 0x98: 0, 0xa8: actor + 0x500,
                                    0xf0: actor + 0x300, 0x148: actor + 0x400, 0x180: 0}.items(): q(actor + offset, pointer)
            q(actor + 0x310, actor + 0x380); q(actor + 0x318, actor + 0x3c0); q(actor + 0x320, actors[1])
            d(actor + 0x398, 3); d(actor + 0x39c, 23)
            q(actor + 0x410, actor + 0x480); d(actor + 0x418, 2); d(actor + 0x41c, 17)
            q(actor + 0x600, actor + 0x700); q(actor + 0xc58, stop + 0x100); q(actor + 0xc60, 0)
            d(actor + 0xe4, 20); d(actor + 0xe0, 10); d(actor + 0xf8, 10); d(actor + 0x118, 73)
        for i, pointer in enumerate(data):
            d(pointer + 0x134, [20, 10][i]); q(pointer + 0x140, source_roles[i]); q(pointer + 0x28, arena + 0x60000)
        old_iterator = arena + 0x30000
        d(old_iterator + 0x10, 1); q(old_iterator + 0x18, old_iterator + 0x100); q(old_iterator + 0x20, actors[0])
        pending = [(old_iterator, bytes(uc.mem_read(old_iterator, 0x28)))]
        initial = [snapshot(actor) for actor in actors]
        calls = []
        for index, (method, actor_index, data_index, display_id) in enumerate(zip(methods_sequence, [0, 0, 1], [0, 1, 0], [3, 2, 1])):
            state.clear(); state.update(actor=actors[actor_index], data_index=data_index, returned=False,
                iterator=arena + 0x40000 + index * 0x200, wait=arena + 0x40100 + index * 0x200,
                clone=arena + 0x50000 + index * 0x100, clone_calls=[], services=[])
            before = snapshot(state['actor'])
            other_actor = actors[1 - actor_index]
            other_before = bytes(uc.mem_read(other_actor, 0x1b8))
            sp = stack + 0x10008; q(sp, stop)
            uc.reg_write(x.UC_X86_REG_RSP, sp); uc.reg_write(x.UC_X86_REG_RCX, state['actor'])
            uc.reg_write(x.UC_X86_REG_RDX, data[data_index]); uc.reg_write(x.UC_X86_REG_R8, display_id)
            for j, register in enumerate(registers): uc.reg_write(register, 0xabc000 + j)
            uc.emu_start(base + (0x365a20 if method == 'Init' else 0x365720), stop + 0x300, count=10000)
            assert state['returned'] and reg(x.UC_X86_REG_RSP) == sp + 8
            assert bytes(uc.mem_read(other_actor, 0x1b8)) == other_before
            assert all(reg(register) == 0xabc000 + j for j, register in enumerate(registers))
            assert all(bytes(uc.mem_read(pointer, 0x28)) == stored for pointer, stored in pending)
            pending.append((state['iterator'], bytes(uc.mem_read(state['iterator'], 0x28))))
            after = snapshot(state['actor'])
            assert after['role'] == state['clone'] and after['data'] == data[data_index]
            assert after['previous'] == before['state'] and after['id'] == display_id
            assert after['info_version'] == before['info_version'] + 1
            assert after['status_count'] == (0 if method == 'Init' else before['status_count'])
            assert after['status_version'] == before['status_version'] + (1 if method == 'Init' else 0)
            assert after['alignment'] == ([20, 10][data_index] if method == 'Init' else before['alignment'])
            assert after['resistance'] == before['resistance'] and after['status_target'] == before['status_target']
            assert after['bluff_role'] == before['bluff_role'] and after['bluff'] == 0
            for field in ('runtime', 'register_as'):
                assert after[field] == (0 if method == 'Init' else before[field])
            calls.append({'method': method, 'actor_index': actor_index, 'data_index': data_index,
                'display_id': display_id, 'before': before, 'after': after,
                'iterator': state['iterator'], 'wait': state['wait'], 'clone_calls': list(state['clone_calls'])})
        assert snapshot(actors[0])['role'] == calls[1]['after']['role']
        cases.append({'initial': initial, 'calls': calls,
            'continuations': [{'identity': pointer, 'state': rd(pointer + 0x10), 'actor': rq(pointer + 0x20), 'current': rq(pointer + 0x18)} for pointer, _ in pending]})
    return {'build': BUILD, 'methods': methods, 'sequence_count': len(cases), 'initializer_calls': sum(len(c['calls']) for c in cases),
            'native_instructions_executed': len(visited), 'cases': cases,
            'limits': ['Explicit synchronous first-yield invocation; no readiness or resume ordering.',
                       'RefreshCharacter and RefreshView, cloning, Unity/runtime operations are service boundaries.',
                       'Only warm metadata, valid dependencies and inert callbacks are used in this sequence corpus.',
                       'Existing iterator bytes remain unchanged; no yielded iterator is resumed or cancelled.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ('sequence_count', 'initializer_calls', 'native_instructions_executed')}))
