"""Execute native engine preference key formatting and registry setter offline."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_preferences_entries import Machine as EntryMachine, normalized, verify_native as verify_entries


class Machine(EntryMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.registry_set = self.services + 0x170
        self.q(self.base + 0x1825018, self.registry_set)
        self.q(self.backend, 0xABCDEF)

    def snapshot(self):
        return {**super().snapshot(), 'native_set_requests': self.native_set_requests.copy(),
                'registry_requests': self.registry_requests.copy(), 'registry_writes': self.registry_writes.copy()}

    def expected_registry_handle(self):
        return 0xABCDEF

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        cx, dx, r8, r9 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                                               x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        if rva == 0x7E6030:
            self.executed.add(rva)
            assert cx == self.backend and r8 & 0xFFFFFFFF == 3
            length = self.rq(self.reg(x.UC_X86_REG_RSP) + 0x28) & 0xFFFFFFFF
            assert 1 <= length <= 65537
            raw = bytes(uc.mem_read(r9, length))
            assert raw[-1] == 0
            self.native_set_requests.append([self.slice(dx).hex(), raw.hex(), length])
            return  # Execute the native helper, including hashing, formatting and cleanup.
        if address == self.registry_set:
            self.executed.add(rva)
            sp = self.reg(x.UC_X86_REG_RSP)
            pointer, count = self.rq(sp + 0x28), self.rq(sp + 0x30) & 0xFFFFFFFF
            assert cx == self.expected_registry_handle() and r8 & 0xFFFFFFFF == 0 and r9 & 0xFFFFFFFF == 3
            assert 1 <= count <= 65537
            request = [self.cstring(dx).hex(), r9 & 0xFFFFFFFF,
                       bytes(uc.mem_read(pointer, count)).hex(), count]
            if self.event('RegSetValueExA_service', request):
                self.registry_requests.append(request)
                status = self.options.get('registry_status', 0) & 0xFFFFFFFF
                if status == 0:
                    self.registry_writes.append(request)
                self.ret(0xDEADBEEF00000000 | status)
            return
        if rva == 0x17C9AF0:
            self.executed.add(rva)
            assert r8 <= 65536
            raw = bytes(uc.mem_read(dx, r8)) if r8 else b''
            if self.event('memory_copy_service', [raw.hex()]):
                if raw:
                    uc.mem_write(cx, raw)
                self.ret(cx)
            return
        if rva == 0x354BA0:
            self.executed.add(rva)
            assert cx == self.manager and (not dx or dx in self.allocations)
            old_size = self.allocations.get(dx, 0)
            if self.event('reallocate_service', [old_size, r8, r9 & 0xFFFFFFFF]):
                pointer = self.alloc(r8)
                if dx and min(old_size, r8):
                    uc.mem_write(pointer, bytes(uc.mem_read(dx, min(old_size, r8))))
                self.ret(pointer)
            return
        super().hook(uc, address, size, data)

    def run(self, direction, key, value, options=None):
        self.native_set_requests, self.registry_requests, self.registry_writes = [], [], []
        return super().run(direction, key, value, options)


def formatted_key(raw):
    """Independent arithmetic oracle; the emulator executes the real formatter."""
    value = 5381
    for byte in raw.split(b'\0', 1)[0]:
        value = (value * 33 ^ (byte if byte < 128 else byte - 256)) & 0xFFFFFFFF
    return raw + b'_h' + str(value).encode('ascii')


def verify_native(m):
    verified = verify_entries(m)
    m.pe.parse_data_directories(directories=[1])
    imports = [(lib.dll.decode('ascii'), item.name.decode('ascii'), item.address - m.base)
               for lib in m.pe.DIRECTORY_ENTRY_IMPORT for item in lib.imports
               if item.address - m.base == 0x1825018]
    assert imports == [('ADVAPI32.dll', 'RegSetValueExA', 0x1825018)]
    assert m.pe.get_data(0x197A558, 9) == b'{0}_h{1}\0'
    descriptor = [int.from_bytes(m.pe.get_data(0x1BE9740 + i * 8, 8), 'little') for i in range(6)]
    assert descriptor == [2, m.base + 0x4308B0, m.base + 0x1BCCBB0,
                          m.base + 0x430850, m.base + 0x1BCD210, 0]
    groups = {}
    for entry in m.pe.DIRECTORY_ENTRY_EXCEPTION:
        root = entry
        while root.unwindinfo.Flags & 4:
            root = root.unwindinfo._chained_entry
        groups.setdefault(root.struct.BeginAddress, []).append((entry.struct.BeginAddress, entry.struct.EndAddress))
    roots = [0x7E6030, 0x7E5F80, 0x4334C0, 0x433190, 0x1593B0, 0x433D00, 0x431610, 0x344B80]
    families, instructions = {}, {}
    for root in roots:
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            assert len(raw) == b - a
            instructions.update({i.address: i for i in m.cs.disasm(raw, a)})
    for a, b in [(0x430850, 0x430877), (0x4308B0, 0x4308BF)]:
        rows = list(m.cs.disasm(m.pe.get_data(a, b - a), a))
        assert sum(i.size for i in rows) == b - a
        instructions.update({i.address: i for i in rows})
    checks = {0x7E5F8C: ('mov', 'r8d, 0x1505'), 0x7E5FA0: ('imul', 'r8d, r8d, 0x21'),
              0x7E5FA8: ('movsx', 'eax, r10b'), 0x7E5FB1: ('xor', 'r8d, eax'),
              0x7E5FB4: ('test', 'r10b, r10b'), 0x7E6019: ('call', '0x4334c0'),
              0x7E6063: ('call', '0x7e5f80'), 0x7E6074: ('mov', 'eax, dword ptr [rsp + 0xa0]'),
              0x7E607B: ('mov', 'r9d, edi'), 0x7E6081: ('xor', 'r8d, r8d'),
              0x7E6098: ('mov', 'ebx, eax'), 0x7E60AD: ('call', '0x354ec0'),
              0x7E60B2: ('test', 'ebx, ebx'), 0x7E60C1: ('sete', 'al'), 0x7E60C8: ('ret', ''),
              0x43086D: ('call', '0x431610'), 0x430876: ('ret', ''),
              0x4308B0: ('mov', 'r8, qword ptr [r8]'), 0x4308BA: ('jmp', '0x344b80')}
    for address, expected in checks.items():
        assert address in instructions and (instructions[address].mnemonic, instructions[address].op_str) == expected
    call = instructions[0x7E608D]
    assert call.mnemonic == 'call' and call.address + call.size + call.operands[0].mem.disp == 0x1825018
    verified.update(setter_imports=imports, key_format='{0}_h{1}', formatter_descriptor_verified=True,
                    setter_native_ranges=families, setter_instruction_assertions=len(checks) + 1,
                    pointer_backed_leaf_ranges=[['0x430850', '0x430877'], ['0x4308b0', '0x4308bf']])
    return verified


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases, baselines, failures = [], [], []
    keys = [None, '', 'Tutorials', 'a\0b', '\0key', 'caf\u00e9', '\u6f22\U0001f608',
            'a' * 24, 'a' * 25, 'a' * 499, 'a' * 500, '\ud800', '\udc00', '\ud800Z']
    values = [None, '', 'value', 'a\0b', '\u6f22\U0001f608', 'v' * 24, 'v' * 25, 'v' * 500]
    for key, value in itertools.product(keys, values):
        result = m.run('set', key, value)
        raw_key, raw_value = normalized(key).encode('utf-8'), normalized(value).encode('utf-8')
        expected = [formatted_key(raw_key).split(b'\0', 1)[0].hex(), 3,
                    (raw_value + b'\0').hex(), len(raw_value) + 1]
        assert result['returned'] and result['result'] == 1
        assert result['final']['registry_requests'] == result['final']['registry_writes'] == [expected]
        assert result['final']['native_set_requests'] == [[raw_key.hex(), (raw_value + b'\0').hex(), len(raw_value) + 1]]
        assert not result['final']['backend_writes']
        cases.append(result)
    for key, blocked, status in itertools.product(['Tutorials', 'k' * 500], [False, True], [0, 5, 234, 0x80000000, 0xFFFFFFFF]):
        result = m.run('set', key, 'value', {'blocked': blocked, 'registry_status': status})
        assert result['returned'] and result['result'] == int(not blocked and status == 0)
        assert len(result['final']['registry_requests']) == int(not blocked)
        assert len(result['final']['registry_writes']) == int(not blocked and status == 0)
        cases.append(result)
    for key, value in [('Tutorials', 'value'), ('caf\u00e9' * 100, 'v' * 500)]:
        baseline = m.run('set', key, value)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run('set', key, value, {'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.executed),
            'scope': 'Native engine setter, signed-byte key hash, full key formatting, pointer-backed argument formatters and chained string cleanup execute. Windows registry calls, provider, runtime exports, memory copy, string copy/assign and allocation/ownership remain supplied services. No actual registry access or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
