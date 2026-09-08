"""Differentially audit the pinned Unity frame-clock update and fixed selector.

Only the two bounded native routines execute in Unicorn, with synthetic clock
objects and explicit timestamps. No host DLL loading or live game state.
"""
import argparse
import itertools
import json
import random
import struct
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


def f32(value):
    return struct.unpack('<f', struct.pack('<f', value))[0]


MIN_DELTA = f32(0.00001)
SCALE_EPSILON = f32(0.000001)
FIRST_DELTA = f32(0.02)


def read(data, offset, kind):
    return struct.unpack_from('<' + kind, data, offset)[0]


def put(data, offset, kind, value):
    struct.pack_into('<' + kind, data, offset, value)


def project_update(source, timestamp):
    """Finite-input oracle with explicit single-precision rounding points."""
    data = bytearray(source)
    get = lambda offset, kind='d': read(data, offset, kind)
    put(data, 0xC8, 'Q', (get(0xC8, 'Q') + 1) % 2**64)
    put(data, 0xD0, 'I', (get(0xD0, 'I') + 1) % 2**32)
    if data[0xF8]:
        return data, 'suppressed'
    unscaled = timestamp - get(0xE8)
    unscaled_delta = f32(unscaled - get(0x70))
    if unscaled_delta >= MIN_DELTA:
        put(data, 0x70, 'd', unscaled)
    else:
        unscaled_delta = MIN_DELTA
    put(data, 0x7C, 'f', unscaled_delta)
    capture, scale = get(0xD8, 'f'), get(0xFC, 'f')
    old = get(0x60)
    if capture > 0:
        updated = old + f32(capture * scale)
        path = 'capture'
    elif data[0xC0]:
        data[0xC0] = 0
        return data, 'skip_once'
    elif data[0xC1]:
        updated = old + f32(scale * FIRST_DELTA)
        path = 'first_step'
    else:
        candidate = timestamp - get(0xE0)
        elapsed = candidate - old
        maximum = get(0x100, 'f')
        if elapsed > maximum:
            updated = old + f32(maximum * scale)
            path = 'maximum'
        elif elapsed < MIN_DELTA:
            updated = old + f32(scale * MIN_DELTA)
            path = 'minimum'
        elif abs(f32(scale - 1.0)) <= SCALE_EPSILON:
            updated = candidate
            path = 'unit_scale'
        else:
            updated = old + f32(f32(elapsed) * scale)
            path = 'scaled'
    delta = f32(updated - old)
    put(data, 0x60, 'd', updated)
    put(data, 0x68, 'd', old)
    put(data, 0x78, 'f', delta)
    put(data, 0x88, 'f', f32(1.0 / delta) if delta > MIN_DELTA else 1.0)
    weight = f32(f32(get(0x84, 'f') * f32(0.8)) + f32(0.2))
    put(data, 0x84, 'f', weight)
    blend = f32(f32(0.2) / weight)
    smooth = f32(f32(f32(1.0 - blend) * get(0x80, 'f')) + f32(blend * delta))
    put(data, 0x80, 'f', smooth)
    data[0x90:0xC0] = data[0x60:0x90]
    put(data, 0xE0, 'd', timestamp - updated)
    if data[0xC1]:
        data[0xC1] = 0
        put(data, 0x84, 'f', 0.0)
    return data, path


def project_select(source):
    data = bytearray(source)
    get = lambda offset, kind='d': read(data, offset, kind)
    old, frame = get(0x30), get(0x60)
    candidate = old + get(0x48, 'f')
    if candidate > frame and not data[0xC2]:
        data[0x90:0xC0] = data[0x60:0x90]
        data[0xF9] = 0
        return data, False
    put(data, 0x38, 'd', old)
    if not data[0xC2]:
        put(data, 0x30, 'd', candidate)
    scale = get(0xFC, 'f')
    if scale != 0:
        unscaled = (get(0x30) - frame) / scale + get(0x70)
        put(data, 0x4C, 'f', f32(unscaled - get(0x40)))
        put(data, 0x40, 'd', unscaled)
    data[0xC2] = 0
    data[0x90:0xC0] = data[0x30:0x60]
    data[0xF9] = 1
    return data, True


def sample_state():
    data = bytearray((i * 37 + 11) % 256 for i in range(0x110))
    for offset, value in [(0x30, 9.98), (0x38, 9.96), (0x40, 19.98),
                          (0x60, 10.0), (0x68, 9.9), (0x70, 20.0),
                          (0xE0, 90.0), (0xE8, 80.0)]:
        put(data, offset, 'd', value)
    for offset, value in [(0x48, 0.02), (0x4C, 0.02), (0x78, 0.1), (0x7C, 0.1),
                          (0x80, 0.016), (0x84, 0.5), (0x88, 10.0),
                          (0xD8, 0.0), (0xFC, 1.0), (0x100, 1 / 3)]:
        put(data, offset, 'f', value)
    for offset in (0xC0, 0xC1, 0xC2, 0xF8, 0xF9): data[offset] = 0
    put(data, 0xC8, 'Q', 41); put(data, 0xD0, 'I', 9)
    data[0x90:0xC0] = data[0x60:0x90]
    put(data, 0x90, 'd', 42.0)
    return data


def describe_state(data):
    def block(offset):
        return {key: read(data, offset + relative, kind) for key, relative, kind in [
            ('time', 0, 'd'), ('previous_time', 8, 'd'), ('unscaled_time', 16, 'd'),
            ('delta', 24, 'f'), ('unscaled_delta', 28, 'f'), ('smooth_delta', 32, 'f'),
            ('smooth_weight', 36, 'f'), ('reciprocal_delta', 40, 'f'), ('opaque_tail', 44, 'I')]}
    return {'fixed': block(0x30), 'frame': block(0x60), 'public': block(0x90),
            'skip_once': bool(data[0xC0]), 'first_frame': bool(data[0xC1]), 'first_fixed': bool(data[0xC2]),
            'frame_counter': read(data, 0xC8, 'q'), 'rendered_counter': read(data, 0xD0, 'I'),
            'capture_delta': read(data, 0xD8, 'f'), 'scaled_offset': read(data, 0xE0, 'd'),
            'unscaled_offset': read(data, 0xE8, 'd'), 'suppress_update': bool(data[0xF8]),
            'in_fixed_step': bool(data[0xF9]), 'time_scale': read(data, 0xFC, 'f'),
            'maximum_delta': read(data, 0x100, 'f')}


def audit(path):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x

    if unicorn.__version__ != '2.1.4': raise ValueError('Unicorn 2.1.4 required')
    raw = path.read_bytes()
    digest = verify_fingerprint(raw, ENGINE_SHA256)
    pe = pefile.PE(data=raw, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    routines = {'update': (0x551D70, 0x551F90), 'select': (0x5A6690, 0x5A6775)}
    decoded = {}
    for name, (begin, end) in routines.items():
        instructions = list(cs.disasm(pe.get_data(begin, end - begin), begin))
        if not instructions or instructions[-1].address + instructions[-1].size != end or instructions[-1].mnemonic != 'ret':
            raise ValueError(f'Incomplete verified routine: {name}')
        if any(i.mnemonic == 'call' for i in instructions): raise ValueError('Unexpected external call')
        for ins in instructions:
            if ins.group(capstone.CS_GRP_JUMP) and not begin <= ins.operands[0].imm < end:
                raise ValueError('Branch escaped bounded routine')
        decoded[name] = instructions
    constants = [(0x1A727A8, 'f', MIN_DELTA), (0x1A72A68, 'f', 1.0),
                 (0x1A72950, 'd', MIN_DELTA), (0x1A74E70, 'I', 0x7FFFFFFF),
                 (0x1A72784, 'f', SCALE_EPSILON), (0x1A72858, 'f', FIRST_DELTA),
                 (0x1A728F0, 'f', f32(0.2)), (0x1A729F8, 'f', f32(0.8))]
    for rva, kind, expected in constants:
        if read(pe.get_data(rva, struct.calcsize(kind)), 0, kind) != expected:
            raise ValueError('Clock constant changed')
    getters = {'UnityEngine.Time::get_deltaTime': (0x10E170, 0xA8),
               'UnityEngine.Time::get_maximumDeltaTime': (0x10E2D0, 0x100),
               'UnityEngine.Time::get_smoothDeltaTime': (0x10E310, 0xB0),
               'UnityEngine.Time::get_captureDeltaTime': (0x10E5F0, 0xD8)}
    found = {}
    image_base = pe.OPTIONAL_HEADER.ImageBase
    for index in range(0xD77):
        name_rva = read(pe.get_data(0x189BB80 + index * 8, 8), 0, 'Q') - image_base
        name = pe.get_string_at_rva(name_rva).decode('utf-8')
        if name not in getters: continue
        rva = read(pe.get_data(0x1894FC0 + index * 8, 8), 0, 'Q') - image_base
        if rva != getters[name][0] or name in found: raise ValueError('Clock getter registration mismatch')
        instructions = list(cs.disasm(pe.get_data(rva, 16), rva))
        first, load, ret = instructions
        if (first.mnemonic != 'mov' or first.address + first.size + first.operands[1].mem.disp != 0x1C6E718
                or load.mnemonic != 'movss' or load.op_str != f'xmm0, dword ptr [rax + {getters[name][1]:#x}]'
                or ret.mnemonic != 'ret'):
            raise ValueError('Clock getter field mismatch')
        found[name] = {'rva': hex(rva), 'field_offset': hex(getters[name][1])}
    if found.keys() != getters.keys(): raise ValueError('Missing clock getter')
    base, arena, stack, stop = pe.OPTIONAL_HEADER.ImageBase, 0x40000000, 0x50000000, 0x60000000
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 0xFFF) & ~0xFFF)
    uc.mem_write(base, pe.get_memory_mapped_image())
    for address in [arena, stack, stop]: uc.mem_map(address, 0x10000)
    uc.mem_write(base + 0x1C6E718, struct.pack('<Q', arena))
    allowed = {base + i.address for instructions in decoded.values() for i in instructions}
    visited = set()
    def on_code(_uc, address, _size, _data):
        if address not in allowed: raise ValueError(f'Unexpected execution {address:#x}')
        visited.add(address - base)
    uc.hook_add(unicorn.UC_HOOK_CODE, on_code)
    preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15,
                 x.UC_X86_REG_XMM6, x.UC_X86_REG_XMM7, x.UC_X86_REG_XMM8]
    def run(name, source, timestamp=0.0):
        uc.mem_write(arena, bytes(source))
        rsp = stack + 0x8008
        uc.mem_write(rsp, struct.pack('<Q', stop))
        uc.reg_write(x.UC_X86_REG_RSP, rsp); uc.reg_write(x.UC_X86_REG_RCX, arena)
        uc.reg_write(x.UC_X86_REG_XMM1, struct.unpack('<Q', struct.pack('<d', timestamp))[0])
        uc.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        for index, register in enumerate(preserved): uc.reg_write(register, 0x12340000 + index)
        uc.emu_start(base + routines[name][0], stop, count=1000)
        if uc.reg_read(x.UC_X86_REG_RIP) != stop or uc.reg_read(x.UC_X86_REG_RSP) != rsp + 8:
            raise ValueError('Native return/stack mismatch')
        if any(uc.reg_read(register) != 0x12340000 + index for index, register in enumerate(preserved)):
            raise ValueError('Nonvolatile register changed')
        return bytes(uc.mem_read(arena, len(source))), bool(uc.reg_read(x.UC_X86_REG_RAX) & 0xFF)
    paths, update_cases, selected_fixtures = {}, 0, []
    def check_update(source, timestamp, label):
        nonlocal update_cases
        expected, path_name = project_update(source, timestamp)
        actual, _ = run('update', source, timestamp)
        if actual != expected:
            differences = [hex(i) for i, (a, b) in enumerate(zip(actual, expected)) if a != b]
            raise ValueError(f'Update mismatch {label} {path_name}: {differences}')
        paths[path_name] = paths.get(path_name, 0) + 1
        update_cases += 1
        if paths[path_name] == 1 or label != 'matrix' and not label.startswith('random'):
            selected_fixtures.append({'path': path_name, 'timestamp': timestamp,
                                      'input': describe_state(source), 'output': describe_state(actual)})
        return actual
    for suppressed, skip, first, capture, scale, elapsed in itertools.product(
            [0, 1], [0, 1], [0, 1], [-1.0, 0.0, 0.02],
            [-1.0, 0.0, 0.25, 1.0, f32(1 + 0.0000005), f32(1 + 0.000002), 2.0],
            [-1.0, 0.0, MIN_DELTA, 2 * MIN_DELTA, 1 / 60, f32(1 / 3), 1.0]):
        source = sample_state(); source[0xF8] = suppressed; source[0xC0] = skip; source[0xC1] = first
        put(source, 0xD8, 'f', f32(capture)); put(source, 0xFC, 'f', scale)
        check_update(source, 100.0 + elapsed, 'matrix')
    rng = random.Random(551)
    for index in range(256):
        source = sample_state()
        put(source, 0x84, 'f', f32(rng.uniform(0, 2)))
        put(source, 0x80, 'f', f32(rng.uniform(-1, 1)))
        put(source, 0xFC, 'f', f32(rng.uniform(-2, 4)))
        put(source, 0x100, 'f', f32(rng.uniform(-0.1, 1)))
        put(source, 0xE8, 'd', rng.uniform(75, 85))
        check_update(source, rng.uniform(99, 101), f'random {index}')
    for frame, rendered in [(2**64 - 1, 2**32 - 1), (2**63 - 1, 2**31 - 1)]:
        source = sample_state(); put(source, 0xC8, 'Q', frame); put(source, 0xD0, 'I', rendered)
        check_update(source, 100.02, 'counter wrap')
    for label, capture, skip, first in [('capture bypasses skip', 0.02, 1, 0),
                                      ('capture resets first', 0.02, 0, 1),
                                      ('skip retains first', 0.0, 1, 1)]:
        source = sample_state(); source[0xC0] = skip; source[0xC1] = first
        put(source, 0xD8, 'f', capture)
        check_update(source, 100.02, label)
    selector_cases, selector_fixtures = 0, []
    for first, scale, fixed, step, frame in itertools.product([0, 1], [-1.0, 0.0, 0.25, 1.0, 2.0],
                                                            [0.0, 10.0], [-0.02, 0.0, 0.02], [0.0, 10.0, 10.1]):
        source = sample_state(); source[0xC2] = first
        for offset, kind, value in [(0x30, 'd', fixed), (0x48, 'f', step), (0x60, 'd', frame), (0xFC, 'f', scale)]:
            put(source, offset, kind, value)
        expected, result = project_select(source); actual, returned = run('select', source)
        if actual != expected or returned != result: raise ValueError('Fixed selector mismatch')
        if fixed == 10.0 and step == 0.02:
            selector_fixtures.append({'input': describe_state(source), 'output': describe_state(actual), 'selected_fixed': returned})
        selector_cases += 1
    return {'schema_version': 1, 'engine_sha256': digest,
            'native_update_rva': '0x551D70', 'native_fixed_selector_rva': '0x5A6690',
            'constant_checks': len(constants), 'native_update_cases': update_cases,
            'registered_getters': found,
            'native_selector_cases': selector_cases, 'update_paths': paths,
            'visited_instruction_count': len(visited), 'selected_results': selected_fixtures,
            'selector_fixtures': selector_fixtures,
            'scope': 'Finite synthetic clock inputs, default round-to-nearest MXCSR; complete clock object byte comparison, normal return and nonvolatile register checks. Native timestamp producers, initialization, policy setters, nonfinite inputs and full PlayerLoop composition remain outside this audit.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unityplayer', type=Path); parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(); result = audit(args.unityplayer)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_update_cases']} clock updates and {result['native_selector_cases']} fixed selectors")
