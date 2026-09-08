"""Exercise the four public Unity timing setters with explicit clock objects."""
import argparse
import json
import math
import struct
from pathlib import Path

from audit_unityplayer_clock import f32, put, read, sample_state
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


def audit(path):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    if unicorn.__version__ != '2.1.4': raise ValueError('Unicorn 2.1.4 required')
    raw = path.read_bytes(); digest = verify_fingerprint(raw, ENGINE_SHA256)
    pe = pefile.PE(data=raw, fast_load=True); base = pe.OPTIONAL_HEADER.ImageBase
    entries = {'fixedDeltaTime': (0x10E240, 0x10E2C9), 'maximumDeltaTime': (0x10E2E0, 0x10E30B),
               'timeScale': (0x10E370, 0x10E501), 'captureDeltaTime': (0x10E600, 0x10E610)}
    registered = {}
    for index in range(0xD77):
        pointer = read(pe.get_data(0x189BB80 + index * 8, 8), 0, 'Q') - base
        name = pe.get_string_at_rva(pointer).decode('utf-8')
        if not name.startswith('UnityEngine.Time::set_'): continue
        key = name.split('::set_')[1]
        if key not in entries: continue
        target = read(pe.get_data(0x1894FC0 + index * 8, 8), 0, 'Q') - base
        if key in registered or target != entries[key][0]: raise ValueError('Setter registration mismatch')
        registered[key] = hex(target)
    if registered.keys() != entries.keys(): raise ValueError('Missing timing setter')
    minimum = read(pe.get_data(0x1A727D0, 4), 0, 'f')
    maximum = read(pe.get_data(0x1A72DAC, 4), 0, 'f')
    if minimum != f32(0.0001) or maximum != 10.0: raise ValueError('Fixed-step clamp constants changed')
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    instructions = {}
    for begin, end in [*entries.values(), (0x17D6A84, 0x17D6AAF)]:
        decoded = list(cs.disasm(pe.get_data(begin, end - begin), begin))
        if decoded[-1].address + decoded[-1].size != end or decoded[-1].mnemonic != 'ret':
            raise ValueError('Incomplete timing setter boundary')
        instructions.update({base + i.address: i for i in decoded})
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 0xFFF) & ~0xFFF)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x40000000, 0x50000000, 0x60000000
    for address in [arena, stack, stop]: uc.mem_map(address, 0x10000)
    uc.mem_write(base + 0x1C6E718, struct.pack('<Q', arena))
    # Object-notification subsystem disabled; it is a separate callee boundary.
    uc.mem_write(base + 0x1CD5920, b'\0')
    state = {}; visited = set()
    def ret(value=0):
        rsp = uc.reg_read(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RIP, read(uc.mem_read(rsp, 8), 0, 'Q'))
        uc.reg_write(x.UC_X86_REG_RSP, rsp + 8)
    def hook(_uc, address, _size, _data):
        if address == base + 0x670140:
            output = uc.reg_read(x.UC_X86_REG_RCX)
            uc.mem_write(output, bytes(0x28)); uc.mem_write(output + 0x20, b'\1')
            state['format_calls'] += 1; ret(output); return
        if address == base + 0x1049F20:
            state['log_calls'] += 1; ret(); return
        if address not in instructions: raise ValueError(f'Unmodeled timing setter service {address:#x}')
        visited.add(address - base)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15,
                 x.UC_X86_REG_XMM6]
    def run(name, source, value):
        state.clear(); state.update(format_calls=0, log_calls=0)
        uc.mem_write(arena, bytes(source)); rsp = stack + 0x8008
        uc.mem_write(rsp, struct.pack('<Q', stop)); uc.reg_write(x.UC_X86_REG_RSP, rsp)
        uc.reg_write(x.UC_X86_REG_XMM0, struct.unpack('<I', struct.pack('<f', value))[0])
        uc.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        for index, register in enumerate(preserved): uc.reg_write(register, 0x12340000 + index)
        uc.emu_start(base + entries[name][0], stop, count=2000)
        if uc.reg_read(x.UC_X86_REG_RIP) != stop or uc.reg_read(x.UC_X86_REG_RSP) != rsp + 8:
            raise ValueError('Setter return/stack mismatch')
        if any(uc.reg_read(register) != 0x12340000 + index for index, register in enumerate(preserved)):
            raise ValueError('Setter nonvolatile register mismatch')
        return bytes(uc.mem_read(arena, len(source)))
    def describe(value):
        if math.isnan(value): return 'nan'
        if math.isinf(value): return 'positive_infinity' if value > 0 else 'negative_infinity'
        return value
    values = [-math.inf, -10.0, -0.0, 0.0, f32(0.00001), minimum, f32(0.02), 1.0, 10.0, 100.0, math.inf, math.nan]
    cases = []
    for name in entries:
        for old_maximum in ([0.001, 0.333, 20.0] if name == 'fixedDeltaTime' else [0.333]):
            for value in values:
                source = sample_state(); put(source, 0x100, 'f', old_maximum)
                expected = bytearray(source); rejected = False
                if name == 'fixedDeltaTime':
                    selected = minimum if math.isnan(value) or value < minimum else min(maximum, value)
                    for offset in [0x48, 0x50]: put(expected, offset, 'f', selected)
                    put(expected, 0x58, 'f', f32(1.0 / selected))
                    put(expected, 0x100, 'f', max(selected, read(source, 0x100, 'f')))
                elif name == 'maximumDeltaTime':
                    selected = read(source, 0x48, 'f') if read(source, 0x48, 'f') > value else value
                    put(expected, 0x100, 'f', selected)
                elif name == 'captureDeltaTime':
                    put(expected, 0xD8, 'f', value)
                else:
                    rejected = math.isnan(value) or value < 0
                    if not rejected: put(expected, 0xFC, 'f', value)
                actual = run(name, source, value)
                if actual != expected or state['log_calls'] != int(rejected) or state['format_calls'] != int(rejected):
                    raise ValueError(f'Timing setter mismatch {name} {value}')
                cases.append({'setter': name, 'value': describe(value), 'prior_maximum': f32(old_maximum), 'rejected_with_log': rejected})
    return {'schema_version': 1, 'engine_sha256': digest, 'registered_setters': registered,
            'native_case_count': len(cases), 'cases': cases, 'visited_instruction_count': len(visited),
            'scope': 'Native setters and native NaN classifier, complete synthetic object-byte comparisons, default MXCSR. Error formatting/logging use explicit successful short-string gateways. The timeScale object-notification subsystem is disabled; its optional effects and other configuration writers remain open.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unityplayer', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.unityplayer)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} native public timing setter cases")
