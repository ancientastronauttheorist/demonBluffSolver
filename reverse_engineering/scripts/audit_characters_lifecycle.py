"""Execute Characters singleton publication, folded pool hiding and construction.

Only authored fixtures/reports are public. Engine objects, allocation and list
initialization are explicit services; no live game state is read or changed.
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
    def pinned(path, digest):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == digest.upper()
        return raw
    raw = pinned(game_root / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    script = json.loads(pinned(dumper_root / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump = pinned(dumper_root / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    body = re.search(r'^public class Characters : MonoBehaviour // TypeDefIndex: 5505\s*\{(.*?)^\}', dump, re.M | re.S)
    assert body
    for declaration in ['public List<Character> characters; // 0x20', 'public static Characters Instance; // 0x0',
                        'public CharactersPool[] characterPool; // 0x30', 'public List<CharacterData> UniquePool; // 0x40',
                        'public List<CharacterData> DuplicatesPool; // 0x48', 'public List<CharacterData> BluffMustInclude; // 0x50']:
        assert declaration in body[1]
    exact = []
    methods = {'Awake': 0x369d00, 'OnEnable': 0x36ca80, 'OnDisable': 0x36ca80, 'HideAll': 0x36ca80, '.ctor': 0x36e710}
    for name, rva in methods.items():
        rows = [m for m in script['ScriptMethod'] if m['Name'] == f'Characters$${name}']
        signature = f'void Characters__{"_ctor" if name == ".ctor" else name} (Characters_o* __this, const MethodInfo* method);'
        assert len(rows) == 1 and rows[0]['Address'] == rva and rows[0]['Signature'] == signature
        exact.append(rows[0])
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    ranges = [(0x369d00, 0x369d51), (0x36ca80, 0x36cae3), (0x36e710, 0x36e825)]
    decoded = {}
    for begin, end in ranges:
        instructions = list(cs.disasm(pe.get_data(begin, end - begin), begin))
        assert instructions[0].address == begin and sum(i.size for i in instructions) == end - begin
        assert instructions[-1].address + instructions[-1].size == end
        decoded.update({i.address: i for i in instructions})
    checks = {0x369d33: ('mov', 'qword ptr [rdx], rbx'), 0x369d4c: ('jmp', '0x2b6ff0'),
              0x36ca8a: ('mov', 'rdi, qword ptr [rcx + 0x30]'), 0x36ca9a: ('jge', '0x36cacd'),
              0x36caae: ('xor', 'edx, edx'), 0x36cab0: ('call', '0x1c79fd0'),
              0x36cabd: ('xor', 'edx, edx'), 0x36cac2: ('call', '0x1c7d810'),
              0x36cade: ('call', '0x2b7d90'), 0x36e77b: ('lea', 'rcx, [rdi + 0x20]'),
              0x36e7a8: ('lea', 'rcx, [rdi + 0x40]'), 0x36e7d5: ('lea', 'rcx, [rdi + 0x48]'),
              0x36e802: ('lea', 'rcx, [rdi + 0x50]'), 0x36e820: ('jmp', '0x1c79770')}
    for rva, expected in checks.items():
        assert rva in decoded and (decoded[rva].mnemonic, decoded[rva].op_str) == expected
    def rip(rva):
        i = decoded[rva]
        refs = [i.address + i.size + o.mem.disp for o in i.operands if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP]
        assert len(refs) == 1
        return refs[0]
    type_slot = rip(0x369d25)
    rows = [m for m in script['ScriptMetadata'] if m['Address'] == type_slot]
    assert len(rows) == 1 and rows[0]['Name'] == 'Characters_TypeInfo'
    flags = [rip(0x369d06), rip(0x36e71a)]
    slots = {rip(rva) for rva in [0x36e726, 0x36e732, 0x36e73e, 0x36e74a]}
    constructor_metadata = [row for row in script['ScriptMetadata'] + script['ScriptMetadataMethod'] if row['Address'] in slots]
    assert {row['Name'] for row in constructor_metadata} == {
        'System.Collections.Generic.List<Character>_TypeInfo',
        'System.Collections.Generic.List<CharacterData>_TypeInfo',
        'Method$System.Collections.Generic.List<Character>..ctor()',
        'Method$System.Collections.Generic.List<CharacterData>..ctor()'}
    assert len(constructor_metadata) == 4
    constructor_slots = {row['Name']: row['Address'] for row in constructor_metadata}
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    for address in (arena, stack, stop):
        uc.mem_map(address, 0x20000)
    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def d(a, v): uc.mem_write(a, struct.pack('<I', v & 0xffffffff))
    obj, cls, static, pool = arena + 0x100, arena + 0x1000, arena + 0x2000, arena + 0x3000
    state, options, visited = {}, {}, set()
    def ret(value=0):
        sp = uc.reg_read(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, sp + 8)
        uc.reg_write(x.UC_X86_REG_RIP, rq(sp))
    def event(name, **values):
        state['counts'][name] = state['counts'].get(name, 0) + 1
        state['events'].append({'kind': name, **values})
        if options.get('fail') == [name, state['counts'][name]]:
            state['error'] = name
            uc.emu_stop()
            return False
        return True
    def hook(_, address, size, __):
        rva = address - base
        c = uc.reg_read(x.UC_X86_REG_RCX)
        value = uc.reg_read(x.UC_X86_REG_RDX)
        if address == stop:
            uc.emu_stop()
        elif rva == 0x2b7b40:
            assert c - base in slots | {type_slot}
            if event('metadata'): ret()
        elif rva == 0x2b6ff0:
            offset = c - obj if c != static else 'Instance'
            assert rq(c) == value
            if event('barrier', field=offset): ret()
        elif rva == 0x2b7d40:
            element = 'Character' if state['counts'].get('allocate', 0) == 0 else 'CharacterData'
            assert c == rq(base + constructor_slots[f'System.Collections.Generic.List<{element}>_TypeInfo'])
            if event('allocate'):
                state['allocation'] += 1
                result = arena + 0x10000 + state['allocation'] * 0x100
                q(result, c)
                ret(result)
        elif rva == 0xb02160:
            element = 'Character' if state['counts'].get('list_ctor', 0) == 0 else 'CharacterData'
            assert c == arena + 0x10000 + state['allocation'] * 0x100
            assert value == rq(base + constructor_slots[f'Method$System.Collections.Generic.List<{element}>..ctor()'])
            if event('list_ctor'): ret()
        elif rva == 0x1c79770:
            assert c == obj and value == 0
            if event('base_ctor'): ret()
        elif rva == 0x1c79fd0:
            assert value == 0
            identity = (c - arena - 0x4000) // 0x100
            if event('get_game_object', pool_id=identity):
                ret(0 if identity in options.get('null_game_objects', []) else arena + 0x8000 + identity * 0x100)
        elif rva == 0x1c7d810:
            assert value == 0 and uc.reg_read(x.UC_X86_REG_R8) == 0
            identity = (c - arena - 0x8000) // 0x100
            if event('set_inactive', pool_id=identity):
                state['hidden'].append(identity)
                ret()
        elif rva in (0x2b7d80, 0x2b7d90):
            state['error'] = 'bounds' if rva == 0x2b7d80 else 'null'
            uc.emu_stop()
        else:
            assert rva in decoded and decoded[rva].size == size, hex(rva)
            visited.add(rva)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    def run(name, pools=None, cold=False, fail=None, null_game_objects=(), null_receiver=False):
        state.clear(); state.update(events=[], counts={}, error=None, allocation=0, hidden=[])
        options.clear(); options.update(fail=fail, null_game_objects=null_game_objects)
        uc.mem_write(obj, bytes([0xa5]) * 0x100)
        q(base + type_slot, cls); q(cls + 0xb8, static); q(static, arena + 0x600)
        for i, slot in enumerate(sorted(slots)): q(base + slot, arena + 0x6000 + i * 0x100)
        for flag in flags: uc.mem_write(base + flag, bytes([0 if cold else 1]))
        q(obj + 0x30, 0 if pools is None else pool)
        if pools is not None:
            d(pool + 0x18, len(pools))
            for i, identity in enumerate(pools): q(pool + 0x20 + i * 8, 0 if identity is None else arena + 0x4000 + identity * 0x100)
        before = bytes(uc.mem_read(obj, 0x100))
        sp = stack + 0x10008; q(sp, stop)
        uc.reg_write(x.UC_X86_REG_RSP, sp); uc.reg_write(x.UC_X86_REG_RCX, 0 if null_receiver else obj)
        for i, register in enumerate(registers): uc.reg_write(register, 0xabc000 + i)
        uc.emu_start(base + methods[name], stop, count=10000)
        final = bytes(uc.mem_read(obj, 0x100))
        initialized = [offset for offset in (0x20, 0x40, 0x48, 0x50) if final[offset:offset + 8] != before[offset:offset + 8]]
        expected = bytearray(before)
        for offset in initialized: expected[offset:offset + 8] = final[offset:offset + 8]
        assert final == expected
        if name != '.ctor': assert final == before
        if not state['error']:
            assert uc.reg_read(x.UC_X86_REG_RIP) == stop and uc.reg_read(x.UC_X86_REG_RSP) == sp + 8
            assert all(uc.reg_read(register) == 0xabc000 + i for i, register in enumerate(registers))
        return {'method': name, 'pools': pools, 'cold': cold, 'failure': fail, 'null_game_objects': list(null_game_objects),
                'null_receiver': null_receiver, 'events': state['events'][:], 'hidden': state['hidden'][:],
                'initialized_offsets': initialized, 'instance_is_receiver': rq(static) == (0 if null_receiver else obj), 'error': state['error']}
    cases = []
    for cold, null_receiver, fail in itertools.product([False, True], [False, True], [None, ['metadata', 1], ['barrier', 1]]):
        case = run('Awake', cold=cold, null_receiver=null_receiver, fail=fail)
        assert case['instance_is_receiver'] == (not (cold and fail == ['metadata', 1]))
        cases.append(case)
    failures = [None, *[[kind, index] for kind in ['metadata', 'allocate', 'list_ctor', 'barrier'] for index in range(1, 5)], ['base_ctor', 1]]
    for cold, fail in itertools.product([False, True], failures):
        case = run('.ctor', cold=cold, fail=fail)
        if case['error'] in ['allocate', 'list_ctor']: count = fail[1] - 1
        elif case['error'] == 'metadata': count = 0
        elif case['error'] == 'barrier': count = fail[1]
        else: count = 4
        assert case['initialized_offsets'] == [0x20, 0x40, 0x48, 0x50][:count]
        assert not case['instance_is_receiver']
        cases.append(case)
    for name, pools, nulls, fail in itertools.product(['OnEnable', 'OnDisable', 'HideAll'],
            [None, [], [0], [0, 1, 0], [None], [0, None, 2]], [(), (0,), (1,)],
            [None, ['get_game_object', 1], ['set_inactive', 1], ['set_inactive', 2]]):
        case = run(name, pools=pools, null_game_objects=nulls, fail=fail)
        expected_hidden = []
        error = None
        gets = sets = 0
        if pools is None: error = 'null'
        else:
            for identity in pools:
                if identity is None: error = 'null'; break
                gets += 1
                if fail == ['get_game_object', gets]: error = 'get_game_object'; break
                if identity in nulls: error = 'null'; break
                sets += 1
                if fail == ['set_inactive', sets]: error = 'set_inactive'; break
                expected_hidden.append(identity)
        assert case['hidden'] == expected_hidden and case['error'] == error
        cases.append(case)
    return {'build_id': BUILD, 'exact_declarations': exact, 'cases_passed': len(cases),
            'constructor_metadata': constructor_metadata,
            'native_relationships': len(checks), 'native_instructions_executed': len(visited), 'cases': cases,
            'scope': 'Authored native caller fixtures. Allocation/list construction, GC barriers, metadata and Unity get_gameObject/SetActive/base constructor are explicit stable gateways. Pool arrays remain stable during callbacks. No live state or complete engine service implementation.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Passed {report['cases_passed']} native Characters lifecycle cases")
