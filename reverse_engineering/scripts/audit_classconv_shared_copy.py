"""Execute fully-shared ClassConv callers with explicit runtime/JSON gateways.

No game code is loaded as a host library. Native instruction bytes stay private.
Synthetic generic layouts exercise caller ABI; engine serialization is a service.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from audit_character_assets import BUILD


def audit(game_root, dumper_root):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    assert unicorn.__version__ == '2.1.4'
    root = Path(__file__).parents[1]
    manifest = json.loads((root / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((root / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(path, digest):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == digest.upper(), path
        return raw
    raw = pinned(Path(game_root) / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    script = json.loads(pinned(Path(dumper_root) / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    header = pinned(Path(dumper_root) / 'il2cpp.h', extraction['outputs']['il2cpp_h']['sha256']).decode('utf-8-sig')
    contexts = {'602C60': ['_0_System_Collections_Generic_List_T_', '_1_System_Collections_Generic_List_T___ctor',
                           '_2_T__', '_3_T', '_4_ClassConv_CreateCopy_T_', '_5_System_Collections_Generic_List_T__Add'],
                '6032F0': ['_0_UnityEngine_JsonUtility_FromJson_T_', '_1_T']}
    for suffix, names in contexts.items():
        marker = f'struct MethodInfo_{suffix}_' + 'R' + 'GCTXs {'
        assert header.count(marker) == 1
        declaration = header.split(marker, 1)[1].split('};', 1)[0]
        assert [line.strip().split()[-1].rstrip(';') for line in declaration.splitlines() if line.strip()] == names
    for name, address in [('CopyArrayIntoList', 0x602C60), ('CreateCopy', 0x6032F0)]:
        matches = [r for r in script['ScriptMethod'] if r['Address'] == address and r['Name'] == f'ClassConv$${name}<__Il2CppFullySharedGenericType>']
        assert len(matches) == 1
    pe = pefile.PE(data=raw, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    instructions = {}
    for start, end in [(0x602C60, 0x602ED7), (0x6032F0, 0x6033A9),
                       (0x282580, 0x282585), (0x2BFD80, 0x2BFE40), (0x2F1430, 0x2F1437)]:
        decoded = list(cs.disasm(pe.get_data(start, end-start), start))
        assert decoded[0].address == start and decoded[-1].address + decoded[-1].size == end
        instructions.update({i.address: i for i in decoded})
    checks = [
        (0x602CB5, 'mov', 'r14d, dword ptr [rcx + 0xfc]'),
        (0x602D92, 'mov', 'ecx, dword ptr [rax + 0x104]'),
        (0x602D9B, 'imul', 'rcx, rax'),
        (0x602DD3, 'call', '0x282580'), (0x602E46, 'call', '0x282580'),
        (0x602E67, 'mov', 'rdx, r10'), (0x602E6D, 'call', 'qword ptr [r10 + 0x10]'),
        (0x602E7E, 'mov', 'ecx, dword ptr [rax + 0x28]'),
        (0x602E81, 'shr', 'ecx, 0x1f'), (0x602E8B, 'mov', 'rcx, qword ptr [rsi]'),
        (0x602EA8, 'call', 'qword ptr [rax + 0x10]'),
        (0x603304, 'mov', 'r14, rdx'), (0x60330B, 'mov', 'r15, rcx'),
        (0x603316, 'mov', 'rdi, r8'),
        (0x60332B, 'mov', 'esi, dword ptr [r9 + 0xfc]'),
        (0x60335B, 'call', '0x1cd6420'),
        (0x603381, 'call', 'qword ptr [r10 + 0x10]'),
        (0x60338B, 'mov', 'rcx, r14'), (0x60338E, 'call', '0x30cfe0'),
        (0x602ECA, 'ret', ''), (0x6033A8, 'ret', ''),
        (0x282580, 'jmp', '0x2bfd80'),
        (0x2BFD8A, 'cmp', 'dword ptr [rcx + 0x28], 0'),
        (0x2BFD96, 'mov', 'rax, qword ptr [rdx]'),
        (0x2BFDB5, 'test', 'byte ptr [rcx + 0x135], 8'),
        (0x2BFDBE, 'cmp', 'byte ptr [rdx], 0'),
        (0x2BFDE5, 'mov', 'edx, dword ptr [rcx + 0x38]'),
        (0x2BFE17, 'call', '0x30cfe0'), (0x2BFE23, 'call', '0x295d60'),
        (0x2F1430, 'mov', 'eax, dword ptr [rcx + 0xf8]'),
    ]
    for address, mnemonic, operand in checks:
        assert address in instructions
        assert (instructions[address].mnemonic, instructions[address].op_str) == (mnemonic, operand)
    base = pe.OPTIONAL_HEADER.ImageBase
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, service, stop = 0x200000000, 0x300000000, 0x400000000, 0x400001000
    uc.mem_map(arena, 0x100000); uc.mem_map(stack, 0x200000); uc.mem_map(service, 0x2000)
    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def d(a, v): uc.mem_write(a, struct.pack('<I', v))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def reg(r): return uc.reg_read(r)
    def ret(value=None):
        rsp = reg(x.UC_X86_REG_RSP)
        if value is not None: uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, rsp+8)
        uc.reg_write(x.UC_X86_REG_RIP, rq(rsp))
    array_method, array_context, copy_method, copy_context, from_method, add_method, ctor_method = [arena+n for n in range(0x100, 0x800, 0x100)]
    tclass, lclass, aclass, character_class, debug_class, ordinary_class = [arena+n for n in range(0x1000, 0x1C00, 0x200)]
    source, output, result_list = arena+0x2000, arena+0x4000, arena+0x6000
    q(array_context, lclass); q(array_context+8, ctor_method); q(array_context+0x18, tclass)
    q(array_context+0x20, copy_method); q(array_context+0x28, add_method)
    q(copy_context, from_method); q(copy_context+8, tclass)
    for method, target, invoker in [(ctor_method, service, 0), (copy_method, base+0x6032F0, service+0x10),
                                    (from_method, service+0x100, service+0x20), (add_method, service+0x110, service+0x30)]:
        q(method, target); q(method+0x10, invoker)
    q(source, aclass)
    slots = {'CharacterData_TypeInfo': character_class, 'UnityEngine.Debug_TypeInfo': debug_class}
    rows = {r['Name']: r['Address'] for r in script['ScriptMetadata'] if r['Name'] in slots}
    assert set(rows) == set(slots)
    for name, address in rows.items(): q(base+address, slots[name])
    uc.mem_write(character_class+0x130, b'\x01')
    q(character_class+0xC8, arena+0x1E00); q(arena+0x1E00, character_class)
    uc.mem_write(ordinary_class+0x130, b'\x00')
    state = {}
    def hook(_, address, instruction_size, __):
        rcx, rdx, r8, r9 = [reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        rva = address-base
        if rva == 0x30AEC0:
            # Windows stack probing is a service; preserve its size in RAX.
            state['stack_sizes'].append(reg(x.UC_X86_REG_RAX)); ret()
        elif rva == 0x30CFE0:
            assert r8 in (state['size'], state['size']-state['nullable_offset'])
            uc.mem_write(rcx, bytes(uc.mem_read(rdx, r8))); ret(rcx)
        elif rva == 0x30CE20:
            assert r8 == state['size'] and rdx == 0
            uc.mem_write(rcx, bytes(r8)); ret(rcx)
        elif rva == 0x2B7B40:
            assert rcx in [base+a for a in rows.values()]
            state['metadata_requests'] += 1; ret()
        elif rva == 0x29C910:
            assert rcx in (array_method, copy_method)
            q(rcx+0x38, array_context if rcx == array_method else copy_context)
            state['context_requests'] += 1; ret()
        elif rva == 0x29C890:
            assert rcx == lclass
            state['class_requests'] += 1; uc.mem_write(lclass+0x135, b'\x01'); ret(lclass)
        elif rva == 0x281D90:
            assert rcx == debug_class
            state['debug_init'] += 1; d(debug_class+0xE0, 1); ret()
        elif rva == 0x2B7D40:
            assert rcx == lclass
            state['allocated'] += 1; ret(result_list)
        elif address == service:
            assert rcx == result_list and rdx == ctor_method and r8 == ctor_method
            state['constructed'] += 1; ret()
        elif rva == 0x2BFD80:
            assert rcx == tclass
            value = bytes(uc.mem_read(rdx, state['size']))
            state['boxes'].append(value.hex())
        elif rva == 0x2C3B10:
            assert rcx == tclass and state['value_type']
            pointer = arena+0x10000+len(state['boxes'])*0x100
            q(pointer, ordinary_class); state['box_allocations'] += 1; ret(pointer)
        elif rva == 0x295D60:
            assert rdx == state['size']-state['nullable_offset']
            state['write_barriers'] += 1; ret()
        elif rva == 0x1C4B450:
            assert rdx == 0
            state['logs'].append(rcx); ret()
        elif address == service+0x10:
            assert (rcx, rdx, r8) == (base+0x6032F0, copy_method, 0)
            boxed, destination = rq(r9), rq(r9+8)
            assert rq(reg(x.UC_X86_REG_RSP)+0x28) == destination
            state['copy_invocations'] += 1
            # Tail-dispatch the synthetic invoker into the actual native body.
            uc.reg_write(x.UC_X86_REG_RCX, boxed)
            uc.reg_write(x.UC_X86_REG_RDX, destination)
            uc.reg_write(x.UC_X86_REG_R8, copy_method)
            uc.reg_write(x.UC_X86_REG_RIP, base+0x6032F0)
        elif rva == 0x1CD6420:
            assert rdx == 0
            state['json_sources'].append(rcx)
            if state['fail_at'] == len(state['json_sources']):
                state['error'] = 'json-service'; uc.emu_stop(); return
            ret(arena+0x30000+len(state['json_sources'])*0x100)
        elif address == service+0x20:
            assert (rcx, rdx, r8) == (service+0x100, from_method, 0)
            assert rq(r9) == arena+0x30000+len(state['json_sources'])*0x100
            destination = rq(r9+8)
            assert rq(reg(x.UC_X86_REG_RSP)+0x28) == destination
            payload = state['results'][len(state['json_sources'])-1]
            uc.mem_write(destination, payload); state['from_invocations'] += 1; ret(0xFEED)
        elif address == service+0x30:
            assert (rcx, rdx, r8) == (service+0x110, add_method, result_list)
            argument = rq(r9)
            assert rq(reg(x.UC_X86_REG_RSP)+0x28) == argument
            value = bytes(uc.mem_read(argument, state['size'])) if state['value_type'] else struct.pack('<Q', argument)
            state['added'].append(value); ret(0xCAFE)
        elif rva in (0x2B7D90, 0x2B7D80):
            state['error'] = 'null' if rva == 0x2B7D90 else 'index'; uc.emu_stop()
        elif rva not in instructions:
            raise AssertionError(f'left audited native bodies/services at {address:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    cases = []
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI, x.UC_X86_REG_RBP,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    shapes = [(False, 8, 0)]+[(True, n, 0) for n in (1, 4, 8, 12, 16, 24, 40, 129)]+[(True, 8, 4), (True, 24, 8)]
    for value_type, size, nullable_offset in shapes:
        for direct in (False, True):
            for cold in (False, True):
                for mode in (('normal', 'failure') if direct else ('null', 'empty', 'normal', 'failure')):
                    stride = size+7 if value_type else size
                    state.clear(); state.update(size=size, value_type=value_type, nullable_offset=nullable_offset, stack_sizes=[], boxes=[], added=[], json_sources=[], logs=[],
                        metadata_requests=0, context_requests=0, class_requests=0, debug_init=0, allocated=0, constructed=0,
                        copy_invocations=0, from_invocations=0, box_allocations=0, write_barriers=0,
                        fail_at=(1 if direct else 2) if mode == 'failure' else 0)
                    q(array_method+0x38, 0 if cold else array_context); q(copy_method+0x38, 0 if cold else copy_context)
                    d(tclass+0xFC, size); d(tclass+0x28, 0x80000000 if value_type else 0); d(aclass+0x104, stride)
                    d(tclass+0xF8, size+16)
                    q(tclass+0x60, arena+0x1F00 if nullable_offset else 0)
                    uc.mem_write(tclass+0x135, bytes([8 if nullable_offset else 0]))
                    q(tclass+0x80, arena+0x1F00); d(arena+0x1F38, nullable_offset+16)
                    uc.mem_write(lclass+0x135, bytes([0 if cold else 1])); d(debug_class+0xE0, 0 if cold else 1)
                    if value_type:
                        values = [bytes((j+i*31)%256 for j in range(size)) for i in range(3)]
                        if nullable_offset:
                            values = [bytes([i%2])+v[1:] for i, v in enumerate(values)]
                    else:
                        obj = arena+0x9000; q(obj, character_class); q(obj+0x28, 0x12345678)
                        values = [struct.pack('<Q', obj), bytes(8), struct.pack('<Q', obj)]
                    if mode in ('null', 'empty'): values = []
                    count = 1 if direct else len(values)
                    state['results'] = [bytes((j+i*13+99)%256 for j in range(size)) if value_type else struct.pack('<Q', 0 if i == 1 else arena+0xA000+i*0x100) for i in range(count)]
                    q(source+0x18, len(values))
                    for i, value in enumerate(values):
                        uc.mem_write(source+0x20+i*stride, value+bytes([0xCD])*(stride-size))
                    uc.mem_write(output-16, bytes([0xA5])*(size+32))
                    rsp = stack+0x180008; q(rsp, stop)
                    for i, r in enumerate(registers): uc.reg_write(r, 0xABCD0000+i)
                    for r, value in [(x.UC_X86_REG_RSP, rsp), (x.UC_X86_REG_RCX, arena+0x9000 if direct else (0 if mode == 'null' else source)),
                                     (x.UC_X86_REG_RDX, output if direct else array_method), (x.UC_X86_REG_R8, copy_method if direct else 0), (x.UC_X86_REG_R9, 0)]:
                        uc.reg_write(r, value)
                    uc.emu_start(base+(0x6032F0 if direct else 0x602C60), stop, timeout=1_000_000, count=20000)
                    expected_error = 'json-service' if mode == 'failure' else ('null' if mode == 'null' else None)
                    assert state.get('error') == expected_error
                    if not expected_error:
                        assert reg(x.UC_X86_REG_RIP) == stop and reg(x.UC_X86_REG_RSP) == rsp+8
                        for i, r in enumerate(registers): assert reg(r) == 0xABCD0000+i
                    if direct:
                        assert state['allocated'] == 0 and len(state['json_sources']) == 1
                        actual = bytes(uc.mem_read(output, size))
                        assert actual == (bytes([0xA5])*size if expected_error else state['results'][0])
                        assert bytes(uc.mem_read(output-16, 16)) == bytes([0xA5])*16
                        assert bytes(uc.mem_read(output+size, 16)) == bytes([0xA5])*16
                    else:
                        assert state['allocated'] == state['constructed'] == 1
                        completed = 1 if mode == 'failure' else len(values)
                        assert state['added'] == state['results'][:completed]
                        started = 2 if mode == 'failure' else len(values)
                        assert state['boxes'] == [v.hex() for v in values[:started] for _ in range(2)]
                        assert state['copy_invocations'] == started
                        assert state['logs'] == ([0x12345678 for v in values[:started] if int.from_bytes(v, 'little')] if not value_type else [])
                        expected_allocations = 2*sum(bool(v[0]) if nullable_offset else value_type for v in values[:started])
                        assert state['box_allocations'] == state['write_barriers'] == expected_allocations
                        if value_type:
                            for i, value in enumerate(values[:started]):
                                boxed = state['json_sources'][i]
                                if nullable_offset and not value[0]: assert boxed == 0
                                else: assert bytes(uc.mem_read(boxed+16, size-nullable_offset)) == value[nullable_offset:]
                        if not expected_error: assert reg(x.UC_X86_REG_RAX) == result_list
                    assert all(n == (size+15)&~15 for n in state['stack_sizes'])
                    cases.append({'value_type': value_type, 'size': size, 'nullable_value_offset': nullable_offset, 'array_stride': stride, 'direct': direct, 'cold_context': cold, 'mode': mode,
                                  'error': expected_error, 'json_calls': len(state['json_sources']), 'added': len(state['added']),
                                  'box_calls': len(state['boxes']), 'context_requests': state['context_requests']})
    return {'schema_version': 1, 'build_id': BUILD, 'game_assembly_sha256': manifest['inputs']['game_assembly']['sha256'],
            'instruction_checks': len(checks), 'context_metadata_checks': len(contexts), 'native_case_count': len(cases), 'cases': cases,
            'abi': {'copy_array': 'RCX array, RDX MethodInfo; RAX list', 'create_copy': 'RCX boxed object, RDX result buffer, R8 MethodInfo; result bytes written through RDX'},
            'scope': 'Two fully-shared native bodies, including nested native CreateCopy, runtime boxing branch/copy body and instance-size getter. Synthetic generic contexts, boxing allocation/GC barriers, invoker adapters, list allocation/constructor/Add, memory primitives, stack probing, initialization and JSON are explicit service boundaries. No UnityPlayer serialization or arbitrary real instantiation support is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"Verified {report['native_case_count']} fully-shared ClassConv cases")
