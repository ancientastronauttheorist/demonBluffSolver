"""Audit native clock reset and QueryPerformanceCounter timestamp conversion.

OS counters, thread initialization and baseline allocation use explicit local
gateways. No game or host DLL execution is performed.
"""
import argparse
import itertools
import json
import math
import struct
from pathlib import Path

from audit_unityplayer_clock import f32, put, read, sample_state
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


def reset_projection(source, initial, timestamp):
    data = bytearray(source)
    for offset in [0x60, 0x68, 0x70, 0x80, 0x30, 0x38, 0x40, 0xC8, 0xF0]:
        put(data, offset, 'Q', 0)
    for offset in [0xD0, 0xD8]: put(data, offset, 'I', 0)
    put(data, 0x78, 'f', f32(0.02) if initial else 0.0)
    put(data, 0x88, 'f', 50.0 if initial else 0.0)
    if initial: put(data, 0x7C, 'f', f32(0.02))
    step = read(data, 0x48, 'f')
    put(data, 0x4C, 'f', step); put(data, 0x58, 'f', f32(1.0 / step))
    data[0xC0:0xC3] = b'\1\1\1'
    data[0x90:0xC0] = data[0x60:0x90]
    put(data, 0xE0, 'd', timestamp); put(data, 0xE8, 'd', timestamp)
    return data


def audit(path):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    if unicorn.__version__ != '2.1.4': raise ValueError('Unicorn 2.1.4 required')
    raw = path.read_bytes(); digest = verify_fingerprint(raw, ENGINE_SHA256)
    pe = pefile.PE(data=raw, fast_load=True)
    pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_IMPORT']])
    base = pe.OPTIONAL_HEADER.ImageBase
    imports = {i.address - base: (d.dll, i.name) for d in pe.DIRECTORY_ENTRY_IMPORT for i in d.imports}
    if imports[0x1825680] != (b'KERNEL32.dll', b'QueryPerformanceCounter'):
        raise ValueError('Unexpected timestamp OS import')
    if imports[0x1825678] != (b'KERNEL32.dll', b'QueryPerformanceFrequency'):
        raise ValueError('Unexpected frequency OS import')
    for slot, entry in [(0x182BBA0, 0x3FE70), (0x182BBA8, 0x3FEB0)]:
        if read(pe.get_data(slot, 8), 0, 'Q') != base + entry:
            raise ValueError('Calibration initializer entry mismatch')
    if read(pe.get_data(0x1A72DF0, 8), 0, 'd') != 1000000000.0:
        raise ValueError('Unexpected timestamp unit divisor')
    if read(pe.get_data(0x1954EF0, 8), 0, 'Q') != base + 0x551D70:
        raise ValueError('Clock updater virtual slot mismatch')
    routines = {'timestamp': (0x12F78D0, 0x12F798F), 'reset': (0x551F90, 0x5520B4),
                'construct': (0x551630, 0x5516F0), 'baseline': (0x5CE5B0, 0x5CE5CB),
                'frequency': (0x3FE70, 0x3FEA2), 'factor': (0x3FEB0, 0x3FF1E)}
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    decoded = {}
    for name, (begin, end) in routines.items():
        instructions = list(cs.disasm(pe.get_data(begin, end - begin), begin))
        if instructions[-1].address + instructions[-1].size != end:
            raise ValueError(f'Incomplete native boundary {name}')
        decoded.update({base + i.address: i for i in instructions})
    # Caller ABI checks are decoded from its verified entry, not byte windows.
    caller = {i.address: i for i in cs.disasm(pe.get_data(0x5B72C0, 0x150), 0x5B72C0)}
    relationships = [(0x5B739D, 'call', 'qword ptr [rdx + 0x690]'),
                     (0x5B73CE, 'call', '0x12f78d0'),
                     (0x5B73D3, 'subsd', 'xmm0, qword ptr [rbx]'),
                     (0x5B73DC, 'test', 'dil, dil'),
                     (0x5B73E4, 'jne', '0x5b73f9'),
                     (0x5B73ED, 'movaps', 'xmm1, xmm0'),
                     (0x5B73F3, 'call', 'qword ptr [rax + 0xb8]'),
                     (0x5B740B, 'jmp', '0x6fa8b0')]
    for address, mnemonic, operands in relationships:
        if (caller[address].mnemonic, caller[address].op_str) != (mnemonic, operands):
            raise ValueError('Frame caller relationship mismatch')
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 0xFFF) & ~0xFFF)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop, tls = 0x40000000, 0x50000000, 0x60000000, 0x70000000
    for address in [arena, stack, stop, tls]: uc.mem_map(address, 0x10000)
    uc.reg_write(x.UC_X86_REG_GS_BASE, tls)
    uc.mem_write(tls + 0x58, struct.pack('<Q', tls + 0x100))
    uc.mem_write(tls + 0x100, struct.pack('<Q', tls + 0x200))
    uc.mem_write(base + 0x1C4A0EC, struct.pack('<I', 0))
    uc.mem_write(base + 0x1825680, struct.pack('<Q', stop + 0x100))
    uc.mem_write(base + 0x1825678, struct.pack('<Q', stop + 0x200))
    state = {}
    def q(address, value): uc.mem_write(address, struct.pack('<Q', value))
    def d(address, value): uc.mem_write(address, struct.pack('<i', value))
    def ret(value=0):
        rsp = uc.reg_read(x.UC_X86_REG_RSP)
        target = read(uc.mem_read(rsp, 8), 0, 'Q')
        uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, rsp + 8); uc.reg_write(x.UC_X86_REG_RIP, target)
    visited = set()
    def hook(_uc, address, _size, _data):
        if address == stop + 0x200:
            q(uc.reg_read(x.UC_X86_REG_RCX), state['frequency'])
            state['frequency_reads'] += 1; ret(state['frequency_success']); return
        if address == stop + 0x100:
            counter = state['counters'].pop(0); state['reads'] += 1
            q(uc.reg_read(x.UC_X86_REG_RCX), counter); ret(1); return
        rva = address - base
        if rva == 0x17A7968:
            state['header'] += 1; d(base + 0x1CDC454, -1 if state['initialize'] else 0); ret(); return
        if rva == 0x17A7908:
            state['footer'] += 1; d(base + 0x1CDC454, 0); ret(); return
        if rva == 0x679460:
            if uc.reg_read(x.UC_X86_REG_RCX) != base + 0x1BDED40 or uc.reg_read(x.UC_X86_REG_RDX) != 8 or uc.reg_read(x.UC_X86_REG_R8) != base + 0x5CE5B0:
                raise ValueError('Baseline allocation forwarding mismatch')
            q(base + 0x1BDED40, arena + 0x2000)
            uc.mem_write(arena + 0x2000, struct.pack('<d', state['baseline']))
            state['baseline_allocations'] += 1; ret(); return
        if rva == 0x17C9930:
            dest, fill, size = [uc.reg_read(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8]]
            if (dest, fill, size) != (arena + 0x110, 0, 0x320): raise ValueError('Constructor memset mismatch')
            uc.mem_write(dest, bytes(size)); ret(dest); return
        if rva == 0x3A0BC0:
            if uc.reg_read(x.UC_X86_REG_RCX) != base + 0x1C706E8: raise ValueError('Reset tail gateway mismatch')
            state['tail_calls'] += 1; ret(); return
        if address not in decoded: raise ValueError(f'Execution escaped native source boundary {address:#x}')
        visited.add(rva)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI, x.UC_X86_REG_RBP,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    def prepare(counters, baseline_counter=0, factor=1.0, cold=False, initialize=False, baseline=0.0):
        state.clear(); state.update(counters=list(counters), reads=0, header=0, footer=0,
                                    initialize=initialize, baseline_allocations=0, baseline=baseline, tail_calls=0)
        d(base + 0x1CDC454, 1 if cold else 0); d(tls + 0x210, 0)
        q(base + 0x1CDC568, baseline_counter)
        uc.mem_write(base + 0x1CC58D8, struct.pack('<d', factor))
        q(base + 0x1BDED40, arena + 0x2000)
        uc.mem_write(arena + 0x2000, struct.pack('<d', baseline))
    def run(name, rcx=0, rdx=0):
        rsp = stack + 0x8008; q(rsp, stop)
        uc.reg_write(x.UC_X86_REG_RSP, rsp); uc.reg_write(x.UC_X86_REG_RCX, rcx); uc.reg_write(x.UC_X86_REG_RDX, rdx)
        uc.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        for index, register in enumerate(preserved): uc.reg_write(register, 0x12340000 + index)
        uc.emu_start(base + routines[name][0], stop, count=2000)
        if uc.reg_read(x.UC_X86_REG_RIP) != stop or uc.reg_read(x.UC_X86_REG_RSP) != rsp + 8:
            raise ValueError('Native source return/stack mismatch')
        if any(uc.reg_read(register) != 0x12340000 + index for index, register in enumerate(preserved)):
            raise ValueError('Native source nonvolatile register mismatch')
        return struct.unpack('<d', struct.pack('<Q', uc.reg_read(x.UC_X86_REG_XMM0) & (2**64 - 1)))[0]
    timestamp_cases = []
    for counter, origin, factor in itertools.product([0, 1, 2**53 + 1, 2**63 - 1, 2**63 + 17, 2**64 - 1], [0, 100], [1.0, 100.0, 0.25]):
        prepare([counter], origin, factor); actual = run('timestamp')
        expected = float((counter - origin) % 2**64) * factor / 1000000000.0
        if actual != expected or state['reads'] != 1 or state['header'] or state['footer']: raise ValueError('Timestamp conversion mismatch')
        timestamp_cases.append({'counter': counter, 'origin': origin, 'nanoseconds_per_tick': factor, 'seconds': actual})
    for initialize in [False, True]:
        prepare([100, 125] if initialize else [125], 50, 100.0, cold=True, initialize=initialize)
        actual = run('timestamp'); expected = (25 if initialize else 75) * 100.0 / 1000000000.0
        if actual != expected or state['header'] != 1 or state['footer'] != int(initialize) or state['reads'] != 1 + int(initialize):
            raise ValueError('Thread initialization gateway order mismatch')
    reset_cases = []
    for initial, present, step in itertools.product([False, True], [False, True], [0.01, 0.02, 0.1]):
        source = sample_state(); put(source, 0x48, 'f', step)
        prepare([200], 100, 100000.0, baseline=0.005)
        if not present: q(base + 0x1BDED40, 0)
        uc.mem_write(arena, bytes(source)); run('reset', arena, int(initial))
        expected = reset_projection(source, initial, 0.01 - 0.005)
        if bytes(uc.mem_read(arena, len(source))) != expected or state['tail_calls'] != 1 or state['baseline_allocations'] != int(not present):
            raise ValueError('Clock reset state/order mismatch')
        reset_cases.append({'initial': initial, 'baseline_present': present, 'fixed_delta': f32(step)})
    for flags in [0, 1, 0x12345678, 0xFFFFFFFF]:
        source = bytearray((i * 13 + 7) % 256 for i in range(0x430))
        expected = bytearray(source)
        for offset in [0x28, 0x18, 0x10, 0x30, 0x38, 0x50, 0x60, 0x68, 0x80, 0x90, 0x98, 0xB0, 0x108]: put(expected, offset, 'Q', 0)
        for offset in [0x20, 8, 0x58, 0x78, 0x88, 0xA8, 0xB8, 0xD4]: put(expected, offset, 'I', 0)
        put(expected, 0xC, 'I', flags & 0xFFF | 0xFFE00000)
        put(expected, 0, 'Q', base + 0x1954E38); put(expected, 0x48, 'f', f32(0.02))
        put(expected, 0xF8, 'H', 0); expected[0x110:0x430] = bytes(0x320)
        expected = reset_projection(expected, True, 0.005)
        prepare([200], 100, 100000.0, baseline=0.005); uc.mem_write(arena, bytes(source))
        run('construct', arena, flags)
        if bytes(uc.mem_read(arena, len(source))) != expected or uc.reg_read(x.UC_X86_REG_RAX) != arena:
            raise ValueError('Clock construction mismatch')
    prepare([200], 100, 100000.0); run('baseline', arena)
    if read(uc.mem_read(arena, 8), 0, 'd') != 0.01 or uc.reg_read(x.UC_X86_REG_RAX) != arena:
        raise ValueError('Baseline initializer mismatch')
    calibration_cases = []
    for frequency, success in [(1, True), (10000000, True), (2**53 + 1, True),
                               (2**63 + 17, True), (2**64 - 1, True), (0, False), (10000000, False)]:
        prepare([]); state.update(frequency=frequency, frequency_success=int(success), frequency_reads=0)
        q(base + 0x1CC5978, 123); q(base + 0x1CC5980, 456)
        run('frequency')
        if read(uc.mem_read(base + 0x1CC5978, 8), 0, 'Q') != 1000000000 or read(uc.mem_read(base + 0x1CC5980, 8), 0, 'Q') != frequency or state['frequency_reads'] != 1:
            raise ValueError('Frequency initializer mismatch')
        run('factor'); factor = read(uc.mem_read(base + 0x1CC58D8, 8), 0, 'd')
        expected = 1000000000.0 / float(frequency) if frequency else math.inf
        if factor != expected: raise ValueError('Unsigned frequency conversion mismatch')
        calibration_cases.append({'frequency': frequency, 'service_success': success,
                                  'nanoseconds_per_tick': factor if math.isfinite(factor) else 'positive_infinity'})
    return {'schema_version': 1, 'engine_sha256': digest, 'timestamp_cases': timestamp_cases,
            'thread_initialization_cases': 2, 'reset_cases': reset_cases, 'constructor_cases': 4,
            'baseline_initializer_cases': 1, 'calibration_cases': calibration_cases,
            'native_case_count': len(timestamp_cases) + 2 + len(reset_cases) + 5 + len(calibration_cases),
            'caller_instruction_relationships': len(relationships), 'visited_instruction_count': len(visited),
            'scope': 'Native timestamp conversion, frequency calibration, reset, constructor and baseline initializer. OS counters/frequency, thread initialization, baseline allocation, memset and reset tail service use explicit gateways. Outer frame-caller ABI is static only; provider callbacks, mode/pause policy and global initialization order remain open.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unityplayer', type=Path); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); result = audit(args.unityplayer)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} native clock-source/reset cases")
