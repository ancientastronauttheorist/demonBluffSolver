"""Execute pinned exported IL2CPP UTF-8/UTF-16 string constructors offline."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_parser import Machine as Helpers


EXPORTS = {'il2cpp_string_new': 0x2821F0, 'il2cpp_string_new_wrapper': 0x2821F0,
           'il2cpp_string_new_len': 0x282200, 'il2cpp_string_new_utf16': 0x282210}


class Machine(Helpers):
    def __init__(self, game_root):
        import capstone
        import pefile
        import unicorn
        from unicorn import x86_const as x
        manifest = json.loads((Path(__file__).parents[1] / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
        raw = (Path(game_root) / 'GameAssembly.dll').read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == manifest['inputs']['game_assembly']['sha256'].upper()
        self.x, self.unicorn = x, unicorn
        assert unicorn.__version__ == '2.1.4'
        self.pe = pefile.PE(data=raw, fast_load=True)
        self.pe.parse_data_directories(directories=[0, 3])
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        self.cs.detail = True
        self.u = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        self.u.mem_map(self.base, (self.pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
        self.u.mem_write(self.base, self.pe.get_memory_mapped_image())
        self.arena, self.stack, self.stop = 0x300000000, 0x400000000, 0x500000000
        self.u.mem_map(self.arena, 0x1000000)
        self.u.mem_map(self.stack, 0x20000)
        self.u.mem_map(self.stop, 0x1000)
        self.string_class, self.empty, self.input = [self.arena + n for n in [0x1000, 0x2000, 0x10000]]
        self.q(self.empty, self.string_class)
        self.q(self.base + 0x289DF48, self.empty)
        self.q(self.base + 0x289E640, self.string_class)
        self.q(self.base + 0x289D578, 0)
        self.q(self.base + 0x289D580, 0)
        self.executed = set()
        self.u.hook_add(unicorn.UC_HOOK_CODE, self.hook)

    def snapshot(self):
        return {'allocation_sizes': list(self.allocations.values()), 'gc_requests': self.gc_requests.copy(),
                'temporary_free_calls': self.temporary_frees.copy(), 'constructor_inputs': self.constructor_inputs.copy(),
                'native_allocation_counter': self.rq(self.base + 0x289D9D0)}

    def event(self, kind, args):
        self.events.append({'kind': kind, 'args': args, 'snapshot': self.snapshot()})
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.options.get('failure') == [kind, self.counts[kind]]:
            self.error = kind
            self.u.emu_stop()
            return False
        return True

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        self.executed.add(rva)
        cx, dx, r8 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8]]
        if rva == 0x29CE80:
            length = dx & 0xFFFFFFFF
            assert length <= 65536
            self.constructor_inputs.append(bytes(uc.mem_read(cx, length * 2)).hex() if length else '')
        elif rva in [0x30AF10, 0x3016B0]:
            assert cx <= 0x100000
            kind = 'gc_allocate_service' if rva == 0x3016B0 else 'temporary_allocate_service'
            if self.event(kind, [cx]):
                pointer = self.alloc(cx)
                if rva == 0x3016B0:
                    self.gc_requests.append([cx, pointer - self.arena])
                self.ret(pointer)
        elif rva == 0x30AF4C:
            assert cx in self.allocations
            if self.event('temporary_free_service', [self.allocations[cx]]):
                self.temporary_frees.append(self.allocations[cx])
                self.ret()
        elif rva == 0x30CFE0:
            assert r8 <= 0x100000
            raw = bytes(uc.mem_read(dx, r8)) if r8 else b''
            if self.event('memory_copy_service', [raw.hex()]):
                if raw:
                    uc.mem_write(cx, raw)
                self.ret(cx)

    def run(self, api, raw, options=None):
        self.options = options or {}
        self.events, self.counts, self.error = [], {}, None
        self.allocations, self.next_alloc = {}, self.arena + 0x30000
        self.gc_requests, self.temporary_frees, self.constructor_inputs = [], [], []
        self.q(self.base + 0x289D9D0, 0)
        self.d(self.base + 0x289D570, self.options.get('profiler_flags', 0))
        input_raw = raw + b'\0\0\0\0'
        self.u.mem_write(self.input, input_raw)
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                     x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(registers):
            self.u.reg_write(register, 0xFAB00000 + i)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        self.u.reg_write(x.UC_X86_REG_RCX, self.input)
        length = len(raw) // 2 if api == 'il2cpp_string_new_utf16' else len(raw)
        self.u.reg_write(x.UC_X86_REG_RDX, 0xFACE000000000000 | length)
        self.u.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        try:
            self.u.emu_start(self.base + EXPORTS[api], self.stop, timeout=10_000_000, count=1000000)
        except Exception as exc:
            raise AssertionError(f'{api}, {raw.hex()[:200]}, RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        result, storage_outcome = None, None
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(registers):
                assert self.reg(register) == 0xFAB00000 + i
            pointer = self.reg(x.UC_X86_REG_RAX)
            assert pointer == self.empty or pointer in self.allocations
            assert self.rq(pointer) == self.string_class and self.rq(pointer + 8) == 0
            count = self.rd(pointer + 0x10)
            assert count <= 65536 and bytes(self.u.mem_read(pointer + 0x14 + count * 2, 2)) == b'\0\0'
            result = bytes(self.u.mem_read(pointer + 0x14, count * 2)).hex()
            assert self.rq(self.base + 0x289D9D0) == int(pointer != self.empty)
            storage_outcome = {'utf16_unit_count': count, 'uses_cached_empty': pointer == self.empty,
                               'class_header_verified': True, 'terminator_verified': True,
                               'allocation_bytes': self.allocations.get(pointer)}
        assert bytes(self.u.mem_read(self.input, len(input_raw))) == input_raw
        return {'api': api, 'input': raw.hex(), 'options': self.options, 'returned': returned,
                'result_utf16': result, 'error': self.error, 'events': self.events.copy(),
                'final': self.snapshot(), 'input_storage_retained': True, 'managed_storage': storage_outcome}

    def run_new_len(self, raw, options=None):
        """Values-only adapter: execute arbitrary bounded bytes and decode actual UTF-16.

        Object identities belong to this emulator and must not be transplanted into
        another runtime. Controlled service stops return no text/storage outcome.
        """
        assert isinstance(raw, bytes) and len(raw) <= 2048
        result = self.run('il2cpp_string_new_len', raw, options)
        return {'text': bytes.fromhex(result['result_utf16']).decode('utf-16-le', errors='surrogatepass')
                if result['returned'] else None,
                'utf16_unit_count': result['managed_storage']['utf16_unit_count'] if result['returned'] else None,
                'managed_storage': result['managed_storage'], 'native_trace': result}


def verify_native(m):
    exports = {e.name.decode('ascii'): e.address for e in m.pe.DIRECTORY_ENTRY_EXPORT.symbols if e.name}
    assert all(exports[name] == rva for name, rva in EXPORTS.items())
    instructions, ranges = {}, {}
    for root in [0x29CCB0, 0x29CD50, 0x29CE80, 0x2435D0, 0x242790,
                 0x244010, 0x243E00, 0x2438E0, 0x2BFD50, 0x261390]:
        chunks = []
        for entry in m.pe.DIRECTORY_ENTRY_EXCEPTION:
            parent = entry
            while parent.unwindinfo.Flags & 4:
                parent = parent.unwindinfo._chained_entry
            if parent.struct.BeginAddress == root:
                a, b = entry.struct.BeginAddress, entry.struct.EndAddress
                chunks.append([hex(a), hex(b)])
                rows = list(m.cs.disasm(m.pe.get_data(a, b - a), a))
                assert sum(i.size for i in rows) == b - a
                instructions.update({i.address: i for i in rows})
        assert chunks
        ranges[hex(root)] = chunks
    for a, b in [(0x2821F0, 0x2821F5), (0x282200, 0x282205), (0x282210, 0x282215), (0x242D90, 0x242D95)]:
        rows = list(m.cs.disasm(m.pe.get_data(a, b - a), a))
        assert sum(i.size for i in rows) == b - a
        instructions.update({i.address: i for i in rows})
    checks = {0x2821F0: ('jmp', '0x29ccb0'), 0x282200: ('jmp', '0x29cd50'),
              0x282210: ('jmp', '0x29ce80'), 0x29CCD4: ('call', '0x2435d0'),
              0x29CD61: ('call', '0x2435d0'), 0x29CCEF: ('call', '0x29ce80'),
              0x29CD7C: ('call', '0x29ce80'), 0x29CEC5: ('mov', 'dword ptr [rax + 0x10], ebx'),
              0x29CECA: ('mov', 'word ptr [rdi + rbx*2 + 0x14], ax'),
              0x29CEBD: ('call', '0x2bfd50'), 0x29CEF1: ('call', '0x242d90'),
              0x2BFD59: ('call', '0x3016b0'), 0x2BFD5E: ('mov', 'qword ptr [rax], rbx'),
              0x2BFD61: ('mov', 'qword ptr [rax + 8], 0'),
              0x24390C: ('call', '0x30af10'), 0x243921: ('mov', 'qword ptr [rax - 8], rcx'),
              0x242D90: ('lea', 'rax, [rcx + 0x14]'), 0x242D94: ('ret', ''),
              0x243630: ('call', '0x242790'), 0x243659: ('call', '0x244010'),
              0x29CD3E: ('ret', ''), 0x29CDCB: ('ret', ''), 0x29CF1B: ('ret', '')}
    for a, expected in checks.items():
        assert a in instructions and (instructions[a].mnemonic, instructions[a].op_str) == expected
    slots = {0x29CE9E: 0x289DF48, 0x29CEB6: 0x289E640, 0x29CECF: 0x289D570}
    for address, slot in slots.items():
        i = instructions[address]
        assert i.address + i.size + i.operands[1].mem.disp == slot
    counter = instructions[0x2BFD69]
    assert counter.mnemonic == 'lock inc' and counter.address + counter.size + counter.operands[0].mem.disp == 0x289D9D0
    gc_leaf = list(m.cs.disasm(m.pe.get_data(0x3016B0, 7), 0x3016B0))
    assert [(i.mnemonic, i.op_str) for i in gc_leaf] == [('xor', 'edx, edx'), ('jmp', '0x3016c0')]
    return {'export_bindings': EXPORTS, 'native_ranges': ranges,
            'instruction_assertions': len(checks) + len(slots) + 3, 'authored_string_class_slot': '0x289e640',
            'authored_empty_string_slot': '0x289df48', 'managed_string_length_offset': '0x10',
            'managed_string_chars_offset': '0x14'}


def expected_utf16(api, raw):
    if api == 'il2cpp_string_new_utf16':
        return raw.hex()
    if api in ['il2cpp_string_new', 'il2cpp_string_new_wrapper']:
        raw = raw.split(b'\0', 1)[0]
    try:
        return raw.decode('utf-8').encode('utf-16-le').hex()
    except UnicodeDecodeError:
        return ''


def compact(result):
    return {k: v for k, v in result.items() if k != 'events'} | {
        'event_kinds': [e['kind'] for e in result['events']]}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases, baselines, failures = [], [], []
    payloads = [b'', b'a\0b', b'\0\xff', b'hello', 'caf\u00e9 \u6f22\U0001f608'.encode('utf-8'),
                b'a' * 7, b'a' * 8, b'a' * 2047, b'a' * 2048]
    payloads += [bytes.fromhex(v) for v in ['c280', 'dfbf', 'e0a080', 'ed9fbf',
                                           'ee8080', 'efbfbf', 'f0908080', 'f48fbfbf', 'efbbbf']]
    invalid = [bytes.fromhex(v) for v in ['80', 'bf', 'c0af', 'c1bf', 'c2', 'c241', 'e080af',
               'eda080', 'edbfbf', 'e2', 'e282', 'e228a1', 'f08080af', 'f4908080', 'f5808080',
               'f09f', 'f09f98', 'f09f4188', 'f8', 'ff']]
    payloads += [prefix + raw + suffix for raw, prefix, suffix in itertools.product(invalid, [b'', b'A'], [b'', b'Z'])]
    payloads += [bytes([b]) for b in range(256)]
    for api, raw in itertools.product(['il2cpp_string_new_len', 'il2cpp_string_new_wrapper'], payloads):
        result = m.run(api, raw)
        assert result['returned'] and result['result_utf16'] == expected_utf16(api, raw), (api, raw.hex(), result['result_utf16'])
        cases.append(compact(result))
    for raw, flags in itertools.product([b'', b'\0\0', b'a\0\0\0b\0', b'\0\xd8', b'\xff\xdf', b'a\0' * 8], [0, 0x80]):
        result = m.run('il2cpp_string_new_utf16', raw, {'profiler_flags': flags})
        assert result['returned'] and result['result_utf16'] == raw.hex()
        cases.append(compact(result))
    for api, raw in [('il2cpp_string_new_len', b'abc'), ('il2cpp_string_new_wrapper', b'a' * 2048)]:
        baseline = m.run(api, raw)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run(api, raw, {'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.executed),
            'scope': 'Export wrappers, native UTF-8 validation and conversion, std::wstring reserve/append/alignment/free callers, native UTF-16 managed construction and object header/allocation counting execute. CRT temporary allocation/free/copy and underlying GC allocation remain supplied services. String class/empty object globals and an empty profiler callback table are authored; no actual GC, invalid pointer handling, length overflow or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
