"""Execute native preference registry reads, legacy fallback and type policy offline."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_preferences_entries import normalized
from audit_unity_preferences_setter import Machine as SetterMachine, formatted_key, verify_native as verify_setter


class Machine(SetterMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.registry_query = self.services + 0x180
        self.q(self.base + 0x1825010, self.registry_query)

    def snapshot(self):
        return {**super().snapshot(), 'native_get_requests': self.native_get_requests.copy(),
                'query_requests': self.query_requests.copy()}

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        cx, dx, r8, r9 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                                               x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        if rva == 0x7E61A0:
            self.executed.add(rva)
            self.native_get_requests.append([self.slice(dx).hex(), self.slice(r8).hex()])
            return  # Actual getter body, including its provider call and cleanup.
        if rva == 0x7E58B0:
            self.executed.add(rva)
            assert cx & 0xFF in [0, 1]
            if self.event('backend_provider_service', [cx & 0xFF]):
                self.ret(self.backend)
            return
        if address == self.registry_query:
            self.executed.add(rva)
            sp = self.reg(x.UC_X86_REG_RSP)
            buffer, size_pointer = self.rq(sp + 0x28), self.rq(sp + 0x30)
            assert cx == 0xABCDEF and r8 == 0 and r9 and size_pointer
            capacity = int.from_bytes(uc.mem_read(size_pointer, 4), 'little') if buffer else None
            index = len(self.query_requests)
            responses = self.options.get('query_responses', [])
            assert index < len(responses), (index, self.options, self.cstring(dx))
            response = responses[index]
            request = {'name': self.cstring(dx).hex(), 'phase': 'data' if buffer else 'size',
                       'capacity': capacity, 'response': response}
            if self.event('RegQueryValueExA_service', [request]):
                self.query_requests.append(request)
                self.d(r9, response['type'])
                self.d(size_pointer, response['size'])
                raw = bytes.fromhex(response.get('data', ''))
                assert not raw or (buffer and len(raw) <= capacity)
                if raw:
                    uc.mem_write(buffer, raw)
                self.ret(0xDEADBEEF00000000 | (response['status'] & 0xFFFFFFFF))
            return
        if rva == 0x17C9930:
            self.executed.add(rva)
            assert r8 <= 65536
            if self.event('memory_fill_service', [dx & 0xFF, r8]):
                if r8:
                    uc.mem_write(cx, bytes([dx & 0xFF]) * r8)
                self.ret(cx)
            return
        if rva == 0x159230:
            self.executed.add(rva)
            raw = bytes(uc.mem_read(dx, r8)) if r8 else b''
            tag = int.from_bytes(uc.mem_read(cx + 36, 4), 'little')
            if self.event('string_assign_service', [raw.hex(), tag]):
                self.put_string(cx, raw)
                self.d(cx + 36, tag)  # Assign retains the destination allocator tag.
                self.ret(cx)
            return
        super().hook(uc, address, size, data)

    def run(self, direction, key, value, options=None):
        self.native_get_requests, self.query_requests = [], []
        result = super().run(direction, key, value, options)
        if result['returned']:
            assert len(self.query_requests) == len(self.options.get('query_responses', []))
        return result


def response(status=0, kind=3, size=0, data=None):
    result = {'status': status, 'type': kind, 'size': size}
    if data is not None:
        result['data'] = data.hex()
    return result


def case_record(result):
    """Retain authored API inputs and checked outcomes without repeated snapshots."""
    return {'key': result['key'], 'default': result['value_or_default'],
            'options': {k: v for k, v in result['options'].items() if k != 'query_responses'},
            'result': result['result'], 'queries': result['final']['query_requests'],
            'native_get_requests': result['final']['native_get_requests'],
            'event_kinds': [e['kind'] for e in result['events']],
            'allocated_buffer_sizes': result['final']['allocated_buffer_sizes'],
            'owner_free_count': result['final']['owner_free_count'],
            'normal_return_and_input_storage_verified': result['returned'] and result['input_storage_retained']}


def verify_native(m):
    verified = verify_setter(m)
    imports = [(lib.dll.decode('ascii'), item.name.decode('ascii'), item.address - m.base)
               for lib in m.pe.DIRECTORY_ENTRY_IMPORT for item in lib.imports
               if item.address - m.base == 0x1825010]
    assert imports == [('ADVAPI32.dll', 'RegQueryValueExA', 0x1825010)]
    instructions, ranges = {}, {}
    for root in [0x7E60D0, 0x7E61A0]:
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
    checks = {0x7E610B: ('call', '0x7e5f80'), 0x7E6142: ('mov', 'ebx, eax'),
              0x7E615C: ('test', 'ebx, ebx'), 0x7E6164: ('mov', 'rdx, qword ptr [rsi]'),
              0x7E6199: ('ret', ''), 0x7E61C7: ('xor', 'ecx, ecx'),
              0x7E61D6: ('call', '0x7e58b0'), 0x7E61DE: ('cmp', 'byte ptr [rax + 8], 0'),
              0x7E61E4: ('mov', 'dword ptr [r15 + 0x24], 0x49'),
              0x7E6249: ('call', '0x7e5f80'), 0x7E62CB: ('mov', 'eax, dword ptr [rbp]'),
              0x7E62DF: ('add', 'eax, 1'), 0x7E62F0: ('cmp', 'r14, 0x7d0'),
              0x7E6310: ('call', '0x17a86c0'), 0x7E634D: ('call', '0x354970'),
              0x7E6365: ('call', '0x17c9930'), 0x7E6370: ('cmp', 'eax, 3'),
              0x7E63A3: ('call', '0x7e60d0'), 0x7E63A8: ('cmp', 'dword ptr [rbp + 0xc8], 3'),
              0x7E645B: ('cmp', 'eax, 1'), 0x7E648E: ('call', '0x7e60d0'),
              0x7E6493: ('cmp', 'dword ptr [rbp + 0xc8], 1'),
              0x7E64BB: ('cmp', 'al, 0x80'), 0x7E64D4: ('cmp', 'byte ptr [rdi + r8], 0'),
              0x7E64E2: ('call', '0x159230')}
    for address, expected in checks.items():
        assert address in instructions and (instructions[address].mnemonic, instructions[address].op_str) == expected
    for address in [0x7E6137, 0x7E617A, 0x7E6279, 0x7E62C3]:
        call = instructions[address]
        assert call.mnemonic == 'call' and call.address + call.size + call.operands[0].mem.disp == 0x1825010
    verified.update(getter_imports=imports, getter_native_ranges=ranges, getter_instruction_assertions=len(checks) + 4)
    return verified


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases, baselines, failures = [], [], []
    keys = [None, 'Tutorials', 'a\0b', 'caf\u00e9' * 100]
    defaults = [None, 'default', 'd\0tail', '\u6f22\U0001f608']
    payloads = [b'', b'hello', b'a\0tail', '\u6f22\U0001f608'.encode('utf-8'), b'x' * 1998, b'x' * 1999]
    for key, default, raw, kind, legacy in itertools.product(keys, defaults, payloads, [1, 3], [False, True]):
        stored = raw + b'\0'
        size, data = response(kind=kind, size=len(stored)), response(kind=kind, size=len(stored), data=stored)
        responses = ([response(2), size, response(2, size=len(stored)), data] if legacy else [size, data])
        result = m.run('get', key, default, {'query_responses': responses})
        expected = normalized(default) if kind == 1 and any(b >= 128 for b in stored) else raw.split(b'\0', 1)[0].decode('utf-8')
        assert result['returned'] and result['result'] == expected
        raw_key = normalized(key).encode('utf-8')
        hashed, bare = formatted_key(raw_key).split(b'\0', 1)[0].hex(), raw_key.split(b'\0', 1)[0].hex()
        assert [q['name'] for q in result['final']['query_requests']] == ([hashed, bare, hashed, bare] if legacy else [hashed, hashed])
        assert result['final']['native_get_requests'] == [[raw_key.hex(), normalized(default).encode('utf-8').hex()]]
        cases.append(case_record(result))
    scenarios = [
        ('blocked', {'blocked': True}, 'default'),
        ('missing', {'query_responses': [response(2), response(2)]}, 'default'),
        ('zero_size', {'query_responses': [response()]}, 'default'),
        ('wrong_initial_type', {'query_responses': [response(kind=4, size=4)]}, 'default'),
        ('changed_binary_type', {'query_responses': [response(size=4), response(kind=1, size=4, data=b'abc\0')]}, 'default'),
        ('changed_string_type', {'query_responses': [response(kind=1, size=4), response(kind=3, size=4, data=b'abc\0')]}, 'default'),
        ('failed_data', {'query_responses': [response(size=4), response(5, size=4), response(5, size=4)]}, 'default'),
        ('data_legacy_fallback', {'query_responses': [response(size=4), response(2, size=4), response(size=4, data=b'abc\0')]}, 'abc'),
        ('size_legacy_data_hashed', {'query_responses': [response(2), response(size=4), response(size=4, data=b'abc\0')]}, 'abc'),
        ('growth_race', {'query_responses': [response(size=4), response(234, size=8), response(234, size=8)]}, 'default'),
        ('shrunk_data', {'query_responses': [response(size=8), response(size=2, data=b'a\0')]}, 'a'),
        ('unterminated_binary', {'query_responses': [response(size=3), response(size=3, data=b'abc')]}, 'abc'),
        ('unterminated_ascii', {'query_responses': [response(kind=1, size=3), response(kind=1, size=3, data=b'abc')]}, 'abc'),
        ('non_ascii_after_nul', {'query_responses': [response(kind=1, size=3), response(kind=1, size=3, data=b'a\0\xff')]}, 'default'),
    ]
    for label, options, expected in scenarios:
        result = m.run('get', 'Tutorials', 'default', options)
        assert result['returned'] and result['result'] == expected, label
        cases.append({'label': label, **case_record(result)})
    for key, default, options in [
        ('Tutorials', 'default', {'query_responses': [response(size=4), response(size=4, data=b'abc\0')]}),
        ('caf\u00e9' * 100, 'd' * 500, {'query_responses': [response(2), response(size=2000), response(2, size=2000), response(size=2000, data=b'a' * 1999 + b'\0')]}),
    ]:
        baseline = m.run('get', key, default, options)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run('get', key, default, {**options, 'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.executed),
            'scope': 'Native entry, getter, hashed/raw fallback query helper, signed-byte hash, formatters, stack probing and flag-zero cleanup execute. Windows query responses, provider, runtime exports, memory primitives, string assign/copy and allocation/ownership remain supplied services. Query size/type/status/data are authored API outcomes. No actual registry access, arbitrary invalid UTF-8 managed conversion or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
