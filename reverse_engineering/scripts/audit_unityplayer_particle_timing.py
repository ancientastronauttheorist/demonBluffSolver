"""Audit maximumParticleDeltaTime and its immediate timing-writer interactions."""
import argparse
import itertools
import json
import random
import struct
from pathlib import Path

from audit_unityplayer_clock import read, put
from audit_unityplayer_clock_normalization import as_float, float_bits, is_nan, project as project_virtual
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


ENTRIES = {'fixed': (0x10e240, 0x10e2c9), 'maximum': (0x10e2e0, 0x10e30b),
           'particle': (0x10e330, 0x10e35b), 'normalize': (0x5520e0, 0x5521aa),
           'refresh': (0x5520c0, 0x5520dc)}
NAMES = {'fixed': 'fixedDeltaTime', 'maximum': 'maximumDeltaTime', 'particle': 'maximumParticleDeltaTime'}


def project(source, operation, value):
    if operation in ('normalize', 'refresh'):
        return project_virtual(source, operation)
    result = bytearray(source)
    fixed = read(source, 0x48, 'I')
    if operation == 'fixed':
        low, high = float_bits(0.0001), float_bits(10)
        selected = low if is_nan(value) or as_float(value) < as_float(low) else high if as_float(value) > 10 else value
        for offset in (0x48, 0x50):
            put(result, offset, 'I', selected)
        put(result, 0x58, 'I', float_bits(1.0 / as_float(selected)))
        prior = read(source, 0x100, 'I')
        put(result, 0x100, 'I', selected if as_float(selected) > as_float(prior) else prior)
    else:
        selected = fixed if as_float(fixed) > as_float(value) else value
        put(result, 0x100 if operation == 'maximum' else 0x104, 'I', selected)
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
    registered = {}
    expected_names = {f'UnityEngine.Time::set_{name}': ENTRIES[key][0] for key, name in NAMES.items()}
    for index in range(0xd77):
        name_rva = read(pe.get_data(0x189bb80 + 8 * index, 8), 0, 'Q') - base
        name = pe.get_string_at_rva(name_rva).decode('utf-8')
        if name not in expected_names:
            continue
        target = read(pe.get_data(0x1894fc0 + 8 * index, 8), 0, 'Q') - base
        assert name not in registered and target == expected_names[name]
        registered[name] = hex(target)
    assert registered.keys() == expected_names.keys()
    for slot, target in ((0x10, 0x5520c0), (0x20, 0x5520e0)):
        assert read(pe.get_data(0x1954e38 + slot, 8), 0, 'Q') == base + target
    for rva, value in ((0x1a727d0, 0.0001), (0x1a72dac, 10), (0x1a72a68, 1)):
        assert read(pe.get_data(rva, 4), 0, 'I') == float_bits(value)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    decoded = {}
    for begin, end in [*ENTRIES.values(), (0x17d6a84, 0x17d6aaf)]:
        items = list(cs.disasm(pe.get_data(begin, end - begin), begin))
        assert items and items[0].address == begin and sum(i.size for i in items) == end - begin
        assert items[-1].address + items[-1].size == end and items[-1].mnemonic == 'ret'
        decoded.update({i.address: i for i in items})
    particle_body = [
        (0x10e330, 'mov', 'rdx, qword ptr [rip + 0x1b603e1]'),
        (0x10e337, 'lea', 'rcx, [rsp + 8]'),
        (0x10e33c, 'movss', 'xmm1, dword ptr [rdx + 0x48]'),
        (0x10e341, 'lea', 'rax, [rdx + 0x48]'),
        (0x10e345, 'comiss', 'xmm1, xmm0'),
        (0x10e348, 'movss', 'dword ptr [rsp + 8], xmm0'),
        (0x10e34e, 'cmova', 'rcx, rax'),
        (0x10e352, 'mov', 'eax, dword ptr [rcx]'),
        (0x10e354, 'mov', 'dword ptr [rdx + 0x104], eax'),
        (0x10e35a, 'ret', '')]
    assert [(i.address, i.mnemonic, i.op_str) for r, i in decoded.items() if 0x10e330 <= r < 0x10e35b] == particle_body
    assert 0x10e337 + 0x1b603e1 == 0x1c6e718
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 0xfff) & ~0xfff)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x40000000, 0x50000000, 0x60000000
    for address in (arena, stack, stop):
        uc.mem_map(address, 0x10000)
    uc.mem_write(base + 0x1c6e718, struct.pack('<Q', arena))
    visited = set()
    def hook(_uc, address, size, _data):
        rva = address - base
        if rva not in decoded or size != decoded[rva].size:
            raise ValueError(f'Unexpected native instruction {address:#x}')
        visited.add(rva)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    registers += [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
    seeds = {reg: (0x123456789abcdef if index < 8 else 0x123456789abcdef0123456789abcdef) + index for index, reg in enumerate(registers)}
    counts = dict.fromkeys(ENTRIES, 0)
    def run(source, operation, value=0):
        uc.mem_write(arena, bytes(source))
        rsp = stack + 0x8008
        uc.mem_write(rsp, struct.pack('<Q', stop))
        uc.reg_write(x.UC_X86_REG_RSP, rsp)
        uc.reg_write(x.UC_X86_REG_RCX, arena)
        uc.reg_write(x.UC_X86_REG_XMM0, value)
        uc.reg_write(x.UC_X86_REG_MXCSR, 0x1f80)
        for reg, seed in seeds.items():
            uc.reg_write(reg, seed)
        uc.emu_start(base + ENTRIES[operation][0], stop, count=1000)
        assert uc.reg_read(x.UC_X86_REG_RIP) == stop and uc.reg_read(x.UC_X86_REG_RSP) == rsp + 8
        assert all(uc.reg_read(reg) == seed for reg, seed in seeds.items())
        actual = bytes(uc.mem_read(arena, len(source)))
        expected = project(source, operation, value)
        if actual != expected:
            raise ValueError(f'{operation} mismatch for {value:08x}')
        counts[operation] += 1
        return actual
    rng = random.Random(0x10e330)
    def fixture(fixed, maximum, particle):
        result = bytearray(rng.randbytes(512))
        for offset, bits in zip((0x48, 0x100, 0x104), (fixed, maximum, particle)):
            put(result, offset, 'I', bits)
        return bytes(result)
    low, high = float_bits(0.0001), float_bits(10)
    patterns = [0, 0x80000000, 1, 0x80000001, 0x007fffff, 0x00800000, 0x7f7fffff,
                0xff7fffff, 0x7f800000, 0xff800000, 0x7fc12345, 0xffc12345, 0x7f800001,
                0xff800001, low - 1, low, low + 1, high - 1, high, high + 1, float_bits(0.02), float_bits(-1)]
    for fixed, value in itertools.product(patterns, repeat=2):
        source = fixture(fixed, value, value)
        for operation in ('maximum', 'particle', 'fixed'):
            run(source, operation, value)
    for _ in range(2000):
        bits = tuple(rng.getrandbits(32) for _ in range(4))
        source = fixture(*bits[:3])
        for operation in ('fixed', 'maximum', 'particle'):
            run(source, operation, bits[3])
    sequences = []
    # All cases execute normalization and refresh separately after direct writes,
    # preserving the behavioral distinction and observing every intermediate.
    def describe(source):
        return {hex(offset): f'{read(source, offset, "I"):08x}' for offset in (0x48, 0x50, 0x58, 0x100, 0x104)}
    for name, initial, operations in [
        ('fixed_increase_leaves_particle_below_floor', (float_bits(0.02), float_bits(1 / 3), float_bits(0.03)), [('fixed', float_bits(1))]),
        ('particle_signaling_nan_survives_direct_store', (float_bits(0.02), float_bits(1 / 3), float_bits(0.03)), [('particle', 0x7f800001)]),
        ('fixed_setter_preserves_nan_maximum', (float_bits(0.02), 0xff800001, float_bits(0.03)), [('fixed', float_bits(1))]),
        ('particle_positive_infinity_survives_normalization', (float_bits(0.02), float_bits(1 / 3), float_bits(0.03)), [('particle', 0x7f800000)]),
        ('invalid_fixed_nan_does_not_floor_particle', (0x7f800001, 0x7f800001, 0x7f800001), [('particle', 0x80000000)]),
        ('invalid_fixed_infinity', (0x7f800000, 0xff800000, 0xff800001), [('particle', float_bits(0.03))]),
    ]:
        source = fixture(*initial)
        trace = [{'operation': 'initial', 'fields': describe(source)}]
        for operation, value in [*operations, ('normalize', 0), ('refresh', 0)]:
            source = run(source, operation, value)
            trace.append({'operation': operation, 'argument_bits': f'{value:08x}' if operation in NAMES else None, 'fields': describe(source)})
        sequences.append({'name': name, 'trace': trace})
    # Broaden virtual branch coverage while retaining arbitrary neighboring bytes.
    for values in itertools.product(patterns, repeat=2):
        source = fixture(values[0], values[1], values[1])
        source = run(source, 'normalize')
        run(source, 'refresh')
    assert set(decoded) == visited
    return {'schema_version': 1, 'engine_sha256': digest, 'registered_setters': registered,
            'native_ranges': {key: [hex(v) for v in bounds] for key, bounds in ENTRIES.items()},
            'native_case_count': sum(counts.values()), 'operation_counts': counts,
            'verified_object_bytes_per_case': 512, 'visited_instruction_count': len(visited),
            'unvisited_instruction_count': 0, 'complete_particle_opcode_assertions': len(particle_body),
            'sequences': sequences,
            'scope': 'Native three-setter interactions and separate normalization/refresh with native NaN classifier, exact full object-byte comparisons, nonvolatile-register/stack preservation, default MXCSR. No native service stubs. No startup order or full writer completeness inference; nondefault MXCSR and exception status flags remain outside scope.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unityplayer', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.unityplayer)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} native particle timing writer cases")
