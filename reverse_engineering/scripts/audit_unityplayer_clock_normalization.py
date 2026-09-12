"""Audit two pinned clock virtuals with synthetic exact-byte state fixtures.

Native bodies remain in the local UnityPlayer image. This harness executes only
verified instruction ranges, including the native NaN classifier.
"""
import argparse
import itertools
import json
import random
import struct
from pathlib import Path

from audit_unityplayer_clock import read, put
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


def float_bits(value):
    try:
        return struct.unpack('<I', struct.pack('<f', value))[0]
    except OverflowError:
        return 0xff800000 if value < 0 else 0x7f800000


def as_float(bits):
    return struct.unpack('<f', struct.pack('<I', bits))[0]


def is_nan(bits):
    return bits & 0x7f800000 == 0x7f800000 and bits & 0x7fffff != 0


def project(source, operation):
    result = bytearray(source)
    fixed_bits = read(source, 0x48, 'I')
    fixed = as_float(fixed_bits)
    if operation == 'refresh':
        put(result, 0x50, 'I', fixed_bits)
        if is_nan(fixed_bits):
            reciprocal = fixed_bits | 0x400000
        elif fixed == 0:
            reciprocal = (fixed_bits & 0x80000000) | 0x7f800000
        else:
            reciprocal = float_bits(1.0 / fixed)
        put(result, 0x58, 'I', reciprocal)
    else:
        low, high = float_bits(0.0001), float_bits(10.0)
        selected = low if is_nan(fixed_bits) or fixed < as_float(low) else high if fixed > 10 else fixed_bits
        put(result, 0x48, 'I', selected)
        for offset in (0x100, 0x104):
            bits = read(source, offset, 'I')
            if is_nan(bits) or as_float(bits) < as_float(selected):
                bits = selected
            put(result, offset, 'I', bits)
    return bytes(result)


def audit(path):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    if unicorn.__version__ != '2.1.4':
        raise ValueError('Unicorn 2.1.4 required')
    raw = path.read_bytes()
    digest = verify_fingerprint(raw, ENGINE_SHA256)
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    public_names = {'UnityEngine.Time::get_maximumParticleDeltaTime': 0x10e320,
                    'UnityEngine.Time::set_maximumParticleDeltaTime': 0x10e330}
    registered = {}
    for index in range(0xd77):
        name_rva = read(pe.get_data(0x189bb80 + index * 8, 8), 0, 'Q') - base
        name = pe.get_string_at_rva(name_rva).decode('utf-8')
        if name not in public_names:
            continue
        target = read(pe.get_data(0x1894fc0 + index * 8, 8), 0, 'Q') - base
        assert name not in registered and target == public_names[name]
        registered[name] = hex(target)
    assert registered.keys() == public_names.keys()
    entries = {'refresh': (0x5520c0, 0x5520dc), 'normalize': (0x5520e0, 0x5521aa)}
    for slot, target in ((0x10, 0x5520c0), (0x20, 0x5520e0)):
        assert read(pe.get_data(0x1954e38 + slot, 8), 0, 'Q') == base + target
    for rva, expected in ((0x1a727d0, float_bits(0.0001)), (0x1a72dac, float_bits(10)), (0x1a72a68, float_bits(1))):
        assert read(pe.get_data(rva, 4), 0, 'I') == expected
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    getter = list(cs.disasm(pe.get_data(0x10e320, 0x10), 0x10e320))
    assert [(i.address, i.mnemonic, i.op_str) for i in getter] == [
        (0x10e320, 'mov', 'rax, qword ptr [rip + 0x1b603f1]'),
        (0x10e327, 'movss', 'xmm0, dword ptr [rax + 0x104]'),
        (0x10e32f, 'ret', '')]
    assert getter[-1].address + getter[-1].size == 0x10e330
    assert getter[0].address + getter[0].size + 0x1b603f1 == 0x1c6e718
    instructions = {}
    for begin, end in [*entries.values(), (0x17d6a84, 0x17d6aaf)]:
        decoded = list(cs.disasm(pe.get_data(begin, end - begin), begin))
        assert decoded and decoded[0].address == begin
        assert decoded[-1].address + decoded[-1].size == end and decoded[-1].mnemonic == 'ret'
        assert sum(i.size for i in decoded) == end - begin
        instructions.update({i.address: i for i in decoded})
    checks = {0x5520cd: ('divss', 'xmm0, xmm1'), 0x5520d1: ('movss', 'dword ptr [rcx + 0x50], xmm1'),
              0x5520d6: ('movss', 'dword ptr [rcx + 0x58], xmm0'), 0x552120: ('minss', 'xmm0, xmm6'),
              0x55212c: ('movss', 'dword ptr [rbx], xmm0'), 0x55215c: ('cmova', 'rax, rbx'),
              0x552165: ('mov', 'dword ptr [rdi + 0x100], eax'), 0x552193: ('cmova', 'rax, rbx'),
              0x55219e: ('mov', 'dword ptr [rdi + 0x104], eax')}
    for rva, expected in checks.items():
        assert rva in instructions
        assert (instructions[rva].mnemonic, instructions[rva].op_str) == expected
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 0xfff) & ~0xfff)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x40000000, 0x50000000, 0x60000000
    for address in (arena, stack, stop):
        uc.mem_map(address, 0x10000)
    visited = set()
    def hook(_uc, address, size, _data):
        rva = address - base
        if rva not in instructions or size != instructions[rva].size:
            raise ValueError(f'Unmodeled instruction {address:#x}')
        visited.add(rva)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    registers += [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
    seeds = {reg: (0x123456789abcdef if index < 8 else 0x123456789abcdef0123456789abcdef) + index for index, reg in enumerate(registers)}
    def run(operation, source):
        uc.mem_write(arena, source)
        rsp = stack + 0x8008
        uc.mem_write(rsp, struct.pack('<Q', stop))
        uc.reg_write(x.UC_X86_REG_RSP, rsp)
        uc.reg_write(x.UC_X86_REG_RCX, arena)
        uc.reg_write(x.UC_X86_REG_MXCSR, 0x1f80)
        for reg, value in seeds.items():
            uc.reg_write(reg, value)
        uc.emu_start(base + entries[operation][0], stop, count=500)
        assert uc.reg_read(x.UC_X86_REG_RIP) == stop and uc.reg_read(x.UC_X86_REG_RSP) == rsp + 8
        assert all(uc.reg_read(reg) == value for reg, value in seeds.items())
        actual = bytes(uc.mem_read(arena, len(source)))
        expected = project(source, operation)
        if actual != expected:
            changed = [hex(i) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
            raise ValueError(f'{operation} mismatch at {changed}; fixed={read(source, 0x48, "I"):#x}')
    low, high = float_bits(0.0001), float_bits(10)
    patterns = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff,
                0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x7f800001,
                0xff800001, low - 1, low, low + 1, high - 1, high, high + 1, float_bits(0.02), float_bits(-1)]
    rng = random.Random(0x5520e0)
    counts = {'refresh': 0, 'normalize': 0}
    examples = []
    def case(operation, values, record=False):
        source = bytearray(rng.randbytes(0x200))
        for offset, bits in zip((0x48, 0x100, 0x104), values):
            put(source, offset, 'I', bits)
        run(operation, bytes(source))
        counts[operation] += 1
        if record:
            result = project(source, operation)
            examples.append({'operation': operation, 'input_bits': [f'{v:08x}' for v in values],
                             'output_bits': {hex(o): f'{read(result, o, "I"):08x}' for o in (0x48, 0x50, 0x58, 0x100, 0x104)}})
    for bits in patterns:
        case('refresh', (bits, low, high), True)
    for values in itertools.product(patterns, repeat=3):
        case('normalize', values)
    for _ in range(2000):
        values = tuple(rng.getrandbits(32) for _ in range(3))
        for operation in entries:
            case(operation, values)
    for values in ((0x7fc12345, 0x7f800000, 0xff800000), (high + 1, 0x7f800001, high), (low - 1, low, high)):
        case('normalize', values, True)
    missing = sorted(set(instructions) - visited)
    assert not missing, f'Unvisited native instructions: {missing}'
    return {'schema_version': 1, 'engine_sha256': digest, 'entries': {key: [hex(v) for v in bounds] for key, bounds in entries.items()},
            'vtable_rva': '0x1954e38', 'vtable_slots': {'0x10': 'refresh', '0x20': 'normalize'},
            'registered_particle_delta_accessors': registered,
            'maximum_particle_delta_binding': {'field_offset': '0x104', 'clock_global_rva': '0x1c6e718',
                                               'getter_range': ['0x10e320', '0x10e330'],
                                               'getter_instruction_assertions': 3, 'setter_body_audited': False},
            'native_case_count': sum(counts.values()), 'operation_counts': counts,
            'verified_object_bytes_per_case': 512, 'visited_instruction_count': len(visited),
            'unvisited_instruction_count': 0, 'explicit_opcode_assertions': len(checks), 'examples': examples,
            'scope': 'Exact object-byte and Windows nonvolatile-register preservation, native NaN classifier, signed zeros, infinities, quiet/signaling NaN payloads, subnormals, clamp-adjacent values, seeded random bit patterns. Default MXCSR 0x1f80; exception flags are not asserted. Caller lifecycle/order and nondefault rounding/DAZ/FTZ remain separate.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unityplayer', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.unityplayer)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} native normalization/refresh cases")
