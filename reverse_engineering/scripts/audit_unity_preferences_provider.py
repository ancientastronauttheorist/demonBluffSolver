"""Execute native preference handle acquisition and cache recovery offline."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_preferences_getter import Machine as GetterMachine, response, verify_native as verify_getter


class Machine(GetterMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.provider_services = {}
        for index, (slot, name) in enumerate([(0x1825000, 'RegCreateKeyW'), (0x1825008, 'RegCloseKey'),
                                             (0x18250A0, 'RegOpenKeyExW'), (0x18257F0, 'MultiByteToWideChar')]):
            address = self.services + 0x190 + index * 0x10
            self.q(self.base + slot, address)
            self.provider_services[address] = name
        self.config = self.arena + 0x13000

    def snapshot(self):
        return {**super().snapshot(), 'handles': [self.rq(self.base + n) for n in [0x1BE9340, 0x1BE9350]],
                'blocked_flags': [self.u.mem_read(self.base + n, 1)[0] for n in [0x1BE9348, 0x1BE9358]],
                'cached_fields': [self.get_string(self.base + n).hex() for n in [0x1BE9360, 0x1BE9388]],
                'provider_api_calls': self.provider_api_calls.copy()}

    def wstring(self, address):
        raw = bytearray()
        for i in range(65536):
            unit = bytes(self.u.mem_read(address + i * 2, 2))
            if unit == b'\0\0':
                return bytes(raw).decode('utf-16-le')
            raw.extend(unit)
        raise AssertionError('unbounded UTF-16 string')

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        cx, dx, r8, r9 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                                               x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        if rva == 0x7E58B0:
            self.executed.add(rva)
            return  # Native provider body; no replacement backend pointer.
        if rva == 0x7E6030:
            assert cx == self.base + 0x1BE9340
            self.backend = cx
        if address == self.registry_query and dx == 0:
            self.executed.add(rva)
            sp = self.reg(x.UC_X86_REG_RSP)
            assert r8 == r9 == self.rq(sp + 0x28) == self.rq(sp + 0x30) == 0
            probes = [c for c in self.provider_api_calls if c['api'] == 'provider_probe']
            statuses = self.options.get('probe_statuses', [0])
            assert len(probes) < len(statuses)
            status = statuses[len(probes)]
            call = {'api': 'provider_probe', 'handle': cx, 'status': status}
            if self.event('provider_probe_service', [call]):
                self.provider_api_calls.append(call)
                self.ret(0xDEADBEEF00000000 | (status & 0xFFFFFFFF))
            return
        name = self.provider_services.get(address)
        if name:
            self.executed.add(rva)
            sp = self.reg(x.UC_X86_REG_RSP)
            if name == 'RegCloseKey':
                call = {'api': name, 'handle': cx}
                if self.event(name + '_service', [call]):
                    self.provider_api_calls.append(call)
                    self.ret(0xDEADBEEF00000000 | self.options.get('close_status', 0))
            elif name in ['RegCreateKeyW', 'RegOpenKeyExW']:
                assert cx == 0xFFFFFFFF80000001
                if name == 'RegCreateKeyW':
                    pointer = r8
                else:
                    assert r8 & 0xFFFFFFFF == 0 and r9 & 0xFFFFFFFF == 0x20019
                    pointer = self.rq(sp + 0x28)
                assert pointer in [self.base + 0x1BE9340, self.base + 0x1BE9350]
                status = self.options.get('create_status' if name == 'RegCreateKeyW' else 'open_status', 0)
                handle = self.options.get('write_handle' if name == 'RegCreateKeyW' else 'read_handle',
                                          0xABCDEF if name == 'RegCreateKeyW' else 0x123456)
                call = {'api': name, 'path': self.wstring(dx), 'status': status, 'authored_handle': handle}
                if self.event(name + '_service', [call]):
                    self.provider_api_calls.append(call)
                    if status & 0xFFFFFFFF == 0:
                        self.q(pointer, handle)
                    self.ret(0xDEADBEEF00000000 | (status & 0xFFFFFFFF))
            else:
                assert name == 'MultiByteToWideChar' and cx & 0xFFFFFFFF == 65001 and dx & 0xFFFFFFFF == 0
                output, capacity = self.rq(sp + 0x28), self.rq(sp + 0x30) & 0xFFFFFFFF
                assert r9 & 0xFFFFFFFF <= 65536
                raw = bytes(uc.mem_read(r8, r9 & 0xFFFFFFFF))
                converted = raw.decode('utf-8').encode('utf-16-le')
                call = {'api': name, 'input': raw.hex(), 'phase': 'write' if output else 'size',
                        'capacity': capacity}
                if self.event(name + '_service', [call]):
                    self.provider_api_calls.append(call)
                    occurrence = self.counts[name + '_service']
                    if occurrence == self.options.get('conversion_failure'):
                        self.ret(0)
                    else:
                        if output:
                            assert len(converted) // 2 <= capacity
                            uc.mem_write(output, converted)
                        self.ret(len(converted) // 2)
            return
        super().hook(uc, address, size, data)

    def prepare_provider(self, fields, options):
        self.options = options
        self.events, self.counts, self.backend_writes = [], {}, []
        self.reference_count, self.owner_free_count, self.error = 0, 0, None
        self.native_set_requests, self.registry_requests, self.registry_writes = [], [], []
        self.native_get_requests, self.query_requests, self.provider_api_calls = [], [], []
        self.allocations, self.next_alloc = {}, self.arena + 0x20000
        self.strings, self.string_cursor = {}, self.arena + 0x600000
        self.initialize_provider(fields, options)

    def initialize_provider(self, fields, options):
        self.d(self.base + 0x1BD00A0, options.get('cached_token_value', 0))
        self.q(self.base + 0x1C6E6E0, self.config if fields is not None else 0)
        if fields is not None:
            assert len(fields) == 2
            for offset, value in zip([0xC0, 0xE8], fields):
                self.put_string(self.config + offset, value.encode('utf-8'))
        cached = options.get('cached_fields', ['', ''])
        for offset, value in zip([0x1BE9360, 0x1BE9388], cached):
            self.put_string(self.base + offset, value.encode('utf-8'))
        for i, offset in enumerate([0x1BE9340, 0x1BE9350]):
            self.q(self.base + offset, options.get('initial_handles', [0, 0])[i])
            self.u.mem_write(self.base + offset + 8, bytes([options.get('initial_blocked', [0, 0])[i]]))

    def prepare_run(self):
        self.backend = self.arena + 0x11000
        self.provider_api_calls = []
        self.initialize_provider(self.entry_fields, self.options)
        self.provider_input_storage = self.config_storage(self.entry_fields)

    def config_storage(self, fields):
        storage = []
        if fields is not None:
            for offset in [0xC0, 0xE8]:
                address = self.config + offset
                storage.append((address, bytes(self.u.mem_read(address, 40))))
                if self.u.mem_read(address + 32, 1)[0] == 0:
                    pointer, length = self.rq(address), self.rq(address + 16)
                    storage.append((pointer, bytes(self.u.mem_read(pointer, length + 1))))
        return storage

    def run_entry(self, direction, key, value, fields=None, options=None):
        self.entry_fields = fields
        self.entry_direction = direction
        result = super().run(direction, key, value, options)
        assert all(bytes(self.u.mem_read(a, len(raw))) == raw for a, raw in self.provider_input_storage)
        result['configuration_storage_retained'] = True
        return result

    def expected_registry_handle(self):
        return self.rq(self.base + (0x1BE9340 if self.entry_direction == 'set' else 0x1BE9350))

    def run_provider(self, mode, fields, options=None):
        self.prepare_provider(fields, options or {})
        input_storage = self.config_storage(fields)
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                     x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for index, register in enumerate(registers):
            self.u.reg_write(register, 0xFAB00000 + index)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        self.u.reg_write(x.UC_X86_REG_RCX, 0xFACE000000000000 | mode)
        self.u.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        try:
            self.u.emu_start(self.base + 0x7E58B0, self.stop, timeout=10_000_000, count=1000000)
        except Exception as exc:
            raise AssertionError(f'{fields}, {self.options}, RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RAX) == self.base + (0x1BE9340 if mode else 0x1BE9350)
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for index, register in enumerate(registers):
                assert self.reg(register) == 0xFAB00000 + index
        assert all(bytes(self.u.mem_read(a, len(raw))) == raw for a, raw in input_storage)
        return {'mode': mode, 'fields': fields, 'options': self.options, 'returned': returned,
                'error': self.error, 'events': self.events.copy(), 'final': self.snapshot(),
                'input_storage_retained': True}


def verify_native(m):
    verified = verify_getter(m)
    wanted = {0x1825000: 'RegCreateKeyW', 0x1825008: 'RegCloseKey',
              0x18250A0: 'RegOpenKeyExW', 0x18257F0: 'MultiByteToWideChar'}
    imports = [(lib.dll.decode('ascii'), item.name.decode('ascii'), item.address - m.base)
               for lib in m.pe.DIRECTORY_ENTRY_IMPORT for item in lib.imports
               if item.address - m.base in wanted]
    assert {slot: name for _, name, slot in imports} == wanted
    assert all(lib == ('KERNEL32.dll' if name == 'MultiByteToWideChar' else 'ADVAPI32.dll')
               for lib, name, _ in imports)
    for offset, expected in [(0x197A568, b'Software\\\0'),
                             (0x197A578, b'Software\\AppDataLow\\Software\\\0'),
                             (0x1955140, b'\\\0')]:
        assert m.pe.get_data(offset, len(expected)) == expected
    roots = [0x7E53D0, 0x7E5470, 0x7E57F0, 0x7E58B0, 0x14F3D0, 0x5B2650, 0x5B3750]
    instructions, ranges = {}, {}
    for root in roots:
        entries = []
        for entry in m.pe.DIRECTORY_ENTRY_EXCEPTION:
            parent = entry
            while parent.unwindinfo.Flags & 4:
                parent = parent.unwindinfo._chained_entry
            if parent.struct.BeginAddress == root:
                a, b = entry.struct.BeginAddress, entry.struct.EndAddress
                entries.append([hex(a), hex(b)])
                rows = list(m.cs.disasm(m.pe.get_data(a, b - a), a))
                assert sum(i.size for i in rows) == b - a
                instructions.update({i.address: i for i in rows})
        assert entries
        ranges[hex(root)] = entries
    # Leaf equality has three return paths and no unwind entry; include them all.
    for a, b in [(0x344180, 0x34423E), (0x9E1D00, 0x9E1D11), (0x9E1E40, 0x9E1E52)]:
        rows = list(m.cs.disasm(m.pe.get_data(a, b - a), a))
        assert sum(i.size for i in rows) == b - a
        instructions.update({i.address: i for i in rows})
    checks = {0x7E53E8: ('call', '0x9e1d00'), 0x7E541D: ('call', '0x159230'),
              0x7E542C: ('call', '0x344b80'), 0x7E5431: ('cmp', 'qword ptr [rdi + 8], 0'),
              0x7E5457: ('call', '0x344b80'), 0x7E547E: ('mov', 'byte ptr [rcx + 8], 0'),
              0x7E548B: ('movzx', 'esi, r8b'), 0x7E54D9: ('call', '0x5b2650'),
              0x7E54F7: ('test', 'rbx, rbx'), 0x7E5501: ('test', 'sil, sil'),
              0x7E550C: ('mov', 'rcx, 0xffffffff80000001'),
              0x7E5529: ('mov', 'r9d, 0x20019'), 0x7E554A: ('test', 'eax, eax'),
              0x7E554E: ('mov', 'byte ptr [rdi + 8], 1'),
              0x7E5914: ('call', '0x344180'), 0x7E5934: ('call', '0x344180'),
              0x7E59EB: ('call', '0x7e53d0'), 0x7E59FE: ('call', '0x7e5470'),
              0x7E5A0D: ('call', '0x7e5470'), 0x7E5A1C: ('call', '0x14f3d0'),
              0x7E5A2B: ('call', '0x14f3d0'), 0x7E5CAD: ('test', 'r13b, r13b'),
              0x7E5CB0: ('cmovne', 'r15, rax'), 0x7E5CD1: ('cmp', 'eax, 0x3fa'),
              0x7E5CDA: ('call', '0x7e57f0'), 0x7E5CE3: ('call', '0x7e58b0'),
              0x5B26C2: ('mov', 'ecx, 0xfde9'), 0x5B26D1: ('test', 'eax, eax'),
              0x5B272D: ('call', '0x5b3750'), 0x5B2753: ('mov', 'word ptr [rax + rbp*2], si'),
              0x3441C6: ('ret', ''), 0x34422C: ('ret', ''), 0x34423D: ('ret', ''),
              0x9E1E4A: ('sete', 'al'), 0x9E1E51: ('ret', '')}
    for address, expected in checks.items():
        assert address in instructions and (instructions[address].mnemonic, instructions[address].op_str) == expected
    slots = {0x7E5519: 0x1825000, 0x7E553F: 0x18250A0, 0x7E594D: 0x1825008,
             0x7E596D: 0x1825008, 0x7E5802: 0x1825008, 0x7E5823: 0x1825008,
             0x7E5CCB: 0x1825010, 0x5B26C7: 0x18257F0, 0x5B279E: 0x18257F0}
    for address, slot in slots.items():
        call = instructions[address]
        assert call.mnemonic == 'call' and call.address + call.size + call.operands[0].mem.disp == slot
    token_initial, token_cached = instructions[0x9E1D04], instructions[0x9E1E40]
    assert token_initial.mnemonic == token_cached.mnemonic == 'cmp'
    assert token_initial.op_str.endswith(', -1') and token_cached.op_str.endswith(', 0x1000')
    for i in [token_initial, token_cached]:
        assert i.address + i.size + i.operands[0].mem.disp == 0x1BD00A0
    verified.update(provider_imports=imports, provider_native_ranges=ranges,
                    provider_instruction_assertions=len(checks) + len(slots) + 2,
                    equality_leaf_range=['0x344180', '0x34423e'],
                    configuration_offsets=['0xc0', '0xe8'], cached_token_global='0x1bd00a0')
    return verified


def compact_case(result):
    return {k: v for k, v in result.items() if k != 'events'} | {
        'event_kinds': [e['kind'] for e in result['events']], 'native_return_verified': result['returned']}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases, baselines, failures = [], [], []
    fields_list = [None, ['', ''], ['Studio', 'Game'], ['Studio', ''], ['', 'Game'],
                   ['St\u00fadio', '\u6f22\U0001f608'], ['a\0tail', 'b\0tail'], ['a' * 25, 'b' * 50]]
    for mode, fields, token, matched in itertools.product([0, 1, 2, 255], fields_list, [0, 0x1000], [False, True]):
        actual = fields or ['', '']
        options = {'cached_fields': actual if matched else ['stale', 'cache'],
                   'cached_token_value': token, 'initial_handles': [0xAAAA, 0xBBBB], 'initial_blocked': [1, 0]}
        result = m.run_provider(mode, fields, options)
        assert result['returned'] and result['final']['cached_fields'] == [s.encode('utf-8').hex() for s in actual]
        calls = result['final']['provider_api_calls']
        assert len([c for c in calls if c['api'] == 'RegCloseKey']) == (0 if matched else 2)
        acquisitions = [c for c in calls if c['api'] in ['RegCreateKeyW', 'RegOpenKeyExW']]
        assert len(acquisitions) == (0 if matched else 2)
        if matched:
            assert result['final']['handles'] == [0xAAAA, 0xBBBB]
            assert result['final']['blocked_flags'] == [1, 0]
        else:
            prefix = 'Software\\AppDataLow\\Software\\' if token == 0x1000 else 'Software\\'
            path = prefix + actual[0] + ('\\' + actual[1] if actual[1] else '')
            assert [c['path'] for c in acquisitions] == [path.split('\0', 1)[0]] * 2
            assert result['final']['handles'] == [0xABCDEF, 0x123456]
            assert result['final']['blocked_flags'] == [0, 0]
            assert all(bytes.fromhex(c['input']) == path.encode('utf-8') for c in calls if c['api'] == 'MultiByteToWideChar')
        assert calls[-1] == {'api': 'provider_probe', 'handle': result['final']['handles'][0 if mode else 1], 'status': 0}
        cases.append(compact_case(result))
    for mode, create_status, open_status in itertools.product([0, 1], [0, 5, 0x80000000], [0, 2, 0xFFFFFFFF]):
        result = m.run_provider(mode, ['Studio', 'Game'], {'create_status': create_status, 'open_status': open_status})
        assert result['returned']
        assert result['final']['handles'] == [0 if create_status else 0xABCDEF, 0 if open_status else 0x123456]
        assert result['final']['blocked_flags'] == [int(create_status != 0), int(open_status != 0)]
        cases.append(compact_case(result))
    for mode, probe_statuses in itertools.product([0, 1, 255], [[5], [0x3FA, 0], [0x3FA, 0x3FA, 0]]):
        options = {'cached_fields': ['Studio', 'Game'], 'initial_handles': [0xAAAA, 0xBBBB],
                   'probe_statuses': probe_statuses, 'close_status': 5}
        result = m.run_provider(mode, ['Studio', 'Game'], options)
        assert result['returned']
        calls = result['final']['provider_api_calls']
        assert len([c for c in calls if c['api'] == 'RegCloseKey']) == 2 * (len(probe_statuses) - 1)
        assert result['final']['handles'] == ([0xAAAA, 0xBBBB] if len(probe_statuses) == 1 else [0xABCDEF, 0x123456])
        cases.append(compact_case(result))
    for mode, fields in itertools.product([0, 1], [None, ['', '']]):
        result = m.run_provider(mode, fields, {'probe_statuses': [6]})
        assert result['returned'] and result['final']['handles'] == [0, 0]
        assert result['final']['blocked_flags'] == [0, 0]
        assert result['final']['provider_api_calls'] == [{'api': 'provider_probe', 'handle': 0, 'status': 6}]
        cases.append({'label': 'empty_cache_zero_handles', **compact_case(result)})
    for failure in [1, 2, 3, 4]:
        result = m.run_provider(1, ['Studio', 'Game'], {'conversion_failure': failure})
        assert result['returned']
        calls = result['final']['provider_api_calls']
        assert any(c['api'] == 'MultiByteToWideChar' for c in calls)
        if failure == 1:
            assert result['final']['blocked_flags'] == [1, 0]
            assert not any(c['api'] == 'RegCreateKeyW' for c in calls)
        elif failure == 3:
            assert result['final']['blocked_flags'] == [0, 1]
            assert not any(c['api'] == 'RegOpenKeyExW' for c in calls)
        else:
            assert result['final']['blocked_flags'] == [0, 0]
        cases.append(compact_case(result))
    entry_cases = []
    for fields, direction, cached in itertools.product([['Studio', 'Game'], ['a' * 25, 'b' * 50],
                                                       ['St\u00fadio', '\u6f22\U0001f608']], ['set', 'get'], [False, True]):
        options = {'cached_fields': fields if cached else ['', ''],
                   'initial_handles': [0xAAAA, 0xBBBB] if cached else [0, 0]}
        if direction == 'get':
            options['query_responses'] = [response(size=6), response(size=6, data=b'value\0')]
        result = m.run_entry(direction, 'Tutorials', 'default' if direction == 'get' else 'value', fields, options)
        assert result['returned'] and result['result'] == ('value' if direction == 'get' else 1)
        assert result['configuration_storage_retained']
        assert result['final']['handles'] == ([0xAAAA, 0xBBBB] if cached else [0xABCDEF, 0x123456])
        entry_cases.append({'direction': direction, 'fields': fields, 'cached': cached,
                            'result': result['result'], 'final': result['final'],
                            'event_kinds': [e['kind'] for e in result['events']],
                            'normal_return_and_storage_verified': True})
    for direction in ['set', 'get']:
        options = {'create_status': 5, 'open_status': 5}
        result = m.run_entry(direction, 'Tutorials', 'default', ['Studio', 'Game'], options)
        assert result['returned'] and result['result'] == ('default' if direction == 'get' else 0)
        assert not result['final']['query_requests'] and not result['final']['registry_requests']
        entry_cases.append({'direction': direction, 'label': 'acquisition_failure',
                            'result': result['result'], 'final': result['final'],
                            'normal_return_and_storage_verified': True})
    for mode, fields, options in [(1, ['Studio', 'Game'], {'initial_handles': [0xAAAA, 0xBBBB]}),
                                  (0, ['Studio' * 20, '\u6f22\U0001f608' * 20], {'probe_statuses': [0x3FA, 0]})]:
        baseline = m.run_provider(mode, fields, options)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run_provider(mode, fields, {**options, 'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    for direction in ['set', 'get']:
        options = {'query_responses': [response(size=6), response(size=6, data=b'value\0')]} if direction == 'get' else {}
        fields = ['Studio', 'Game']
        baseline = m.run_entry(direction, 'Tutorials', 'default' if direction == 'get' else 'value', fields, options)
        baseline_id = len(baselines)
        baselines.append({'entry_fields': fields, **baseline})
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run_entry(direction, 'Tutorials', baseline['value_or_default'], fields,
                                 {**options, 'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases),
            'entry_cases': entry_cases, 'entry_case_count': len(entry_cases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.executed),
            'scope': 'Native provider, path construction, handle acquisition, cache comparison/copy, invalidation/retry, UTF-16 conversion orchestration/reserve and flag-zero cleanup execute. Windows registry and UTF-8 conversion APIs, runtime exports, memory primitives, assignment/copy and allocation/ownership remain supplied services. Configuration fields and cached security-token value are authored native inputs; cold token discovery and real configuration initialization remain open. No actual registry access or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
