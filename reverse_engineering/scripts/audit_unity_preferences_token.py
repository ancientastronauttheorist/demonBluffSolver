"""Execute Unity's cold cached security-token predicate with authored Windows APIs."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_preferences_entries import Machine as EntryMachine, verify_native as verify_entries


IMPORTS = {
    0x1825298: ('KERNEL32.dll', 'LocalFree'),
    0x18253C8: ('KERNEL32.dll', 'LocalAlloc'),
    0x1825708: ('KERNEL32.dll', 'GetCurrentProcess'),
    0x1825798: ('KERNEL32.dll', 'CloseHandle'),
    0x1825818: ('KERNEL32.dll', 'GetLastError'),
    0x1825048: ('ADVAPI32.dll', 'OpenProcessToken'),
    0x1825050: ('ADVAPI32.dll', 'GetTokenInformation'),
    0x1825058: ('ADVAPI32.dll', 'GetSidSubAuthority'),
}
CACHE = 0x1BD00A0


class TokenServices:
    """Authored Windows callbacks reusable by a host with event/ret/register helpers."""

    def bind_token_services(self):
        self.token_services = {}
        for index, (slot, (_, name)) in enumerate(IMPORTS.items()):
            address = self.services + 0x300 + index * 0x10
            self.q(self.base + slot, address)
            self.token_services[address] = name
        self.token_data, self.sid, self.subauthority = [self.arena + n for n in [0x14000, 0x15000, 0x16000]]

    def initialize_token_services(self, options=None):
        self.token_options = options or {}
        self.token_api_calls, self.token_closed_handles, self.token_freed_buffers = [], [], []
        self.token_allocated = False

    def token_snapshot(self):
        # Release lists are call histories; allocated is the latest allocation
        # outcome, not a model of real Windows ownership after release.
        return {'cached_value': self.rq(self.base + CACHE) & 0xFFFFFFFF,
                'api_calls': self.token_api_calls.copy(), 'allocated': self.token_allocated,
                'closed_handles': self.token_closed_handles.copy(), 'freed_buffers': self.token_freed_buffers.copy()}

    def token_bool_result(self, value):
        # Upper RAX deliberately does not describe the Windows BOOL result.
        return 0xDEADBEEF00000000 | (value & 0xFFFFFFFF)

    def hook_token_service(self, address):
        name = self.token_services.get(address)
        if not name:
            return False
        self.executed.add(address - self.base)
        x = self.x
        cx, dx, r8, r9 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                                               x.UC_X86_REG_R8, x.UC_X86_REG_R9]]
        sp = self.reg(x.UC_X86_REG_RSP)
        opts = self.token_options
        call = {'api': name}
        if name == 'OpenProcessToken':
            assert cx == 0xFFFFFFFFFFFFFFFF and dx & 0xFFFFFFFF == 8
            call.update(access=8, result=opts.get('open_result', 1),
                        authored_handle=opts.get('open_handle', 0xABCDEF))
        elif name == 'GetTokenInformation':
            assert cx == opts.get('open_handle', 0xABCDEF) and dx & 0xFFFFFFFF == 25
            pointer = self.rq(sp + 0x28)
            phase = 'size' if r8 == 0 else 'data'
            assert r9 & 0xFFFFFFFF == (0 if phase == 'size' else opts.get('required_size', 16))
            assert phase == 'size' or r8 == self.token_data
            call.update(information_class=25, phase=phase, capacity=r9 & 0xFFFFFFFF,
                        result=opts.get(phase + '_result', 0 if phase == 'size' else 1),
                        authored_size=opts.get('required_size', 16))
        elif name == 'LocalAlloc':
            assert cx & 0xFFFFFFFF == 0x40 and dx == opts.get('required_size', 16)
            call.update(flags=0x40, size=dx, succeeds=opts.get('allocation_success', True))
        elif name == 'GetSidSubAuthority':
            assert cx == self.sid and dx & 0xFFFFFFFF == 0
            call.update(index=0, authored_value=opts.get('subauthority', 0x1000))
        elif name == 'CloseHandle':
            assert cx == opts.get('open_handle', 0xABCDEF) and cx != 0
            call.update(handle=cx, result=opts.get('close_result', 1))
        elif name == 'LocalFree':
            assert cx == self.token_data and self.token_allocated
            call.update(buffer='authored_token_buffer', result=opts.get('free_result', 0))
        elif name == 'GetLastError':
            errors = opts.get('last_errors', [122])
            occurrence = self.counts.get(name + '_service', 0)
            assert occurrence < len(errors), (opts, occurrence)
            call.update(value=errors[occurrence])
        else:
            assert name == 'GetCurrentProcess'
            call.update(result='current_process_pseudohandle')
        if not self.event(name + '_service', [call]):
            return True
        self.token_api_calls.append(call)
        if name == 'GetCurrentProcess':
            self.ret(0xFFFFFFFFFFFFFFFF)
        elif name == 'OpenProcessToken':
            # Explicitly authored output is permitted even on an authored API failure.
            self.q(r8, call['authored_handle'])
            self.ret(self.token_bool_result(call['result']))
        elif name == 'GetTokenInformation':
            self.d(pointer, call['authored_size'])
            if phase == 'data' and call['result'] & 0xFFFFFFFF:
                self.q(self.token_data, self.sid)
            self.ret(self.token_bool_result(call['result']))
        elif name == 'LocalAlloc':
            self.token_allocated = call['succeeds']
            self.ret(self.token_data if self.token_allocated else 0)
        elif name == 'GetSidSubAuthority':
            self.d(self.subauthority, call['authored_value'])
            self.ret(self.subauthority)
        elif name == 'CloseHandle':
            self.token_closed_handles.append(cx)
            self.ret(self.token_bool_result(call['result']))
        elif name == 'LocalFree':
            self.token_freed_buffers.append('authored_token_buffer')
            self.ret(call['result'])
        else:
            self.ret(self.token_bool_result(call['value']))
        return True


class Machine(TokenServices, EntryMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.bind_token_services()

    def snapshot(self):
        return self.token_snapshot()

    def hook(self, uc, address, size, data):
        self.executed.add(address - self.base)
        self.hook_token_service(address)

    def run_token(self, options=None, cached_value=0xFFFFFFFF):
        self.options = options or {}
        self.events, self.counts = [], {}
        self.error = None
        self.initialize_token_services(self.options)
        if cached_value is not None:
            self.d(self.base + CACHE, cached_value)
        initial_cache = self.rq(self.base + CACHE) & 0xFFFFFFFF
        self.u.mem_write(self.token_data, b'\0' * 65536)
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                     x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for index, register in enumerate(registers):
            self.u.reg_write(register, 0xFAB00000 + index)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        self.u.reg_write(x.UC_X86_REG_RAX, 0xA5A5A5A5A5A5A5A5)
        self.u.emu_start(self.base + 0x9E1D00, self.stop, timeout=10_000_000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        result = None
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for index, register in enumerate(registers):
                assert self.reg(register) == 0xFAB00000 + index
            result = self.reg(x.UC_X86_REG_RAX) & 0xFF
            assert result in [0, 1]
        return {'options': self.options, 'initial_cache': initial_cache, 'returned': returned,
                'result_al': result, 'error': self.error, 'events': self.events.copy(), 'final': self.snapshot()}


def verify_native(m):
    verified = verify_entries(m)
    m.pe.parse_data_directories(directories=[1])
    imports = [(lib.dll.decode('ascii'), item.name.decode('ascii'), item.address - m.base)
               for lib in m.pe.DIRECTORY_ENTRY_IMPORT for item in lib.imports
               if item.address - m.base in IMPORTS]
    assert {slot: (lib, name) for lib, name, slot in imports} == IMPORTS
    ranges, instructions = [], {}
    for entry in m.pe.DIRECTORY_ENTRY_EXCEPTION:
        root = entry
        while root.unwindinfo.Flags & 4:
            root = root.unwindinfo._chained_entry
        if root.struct.BeginAddress == 0x9E1D00:
            a, b = entry.struct.BeginAddress, entry.struct.EndAddress
            ranges.append([hex(a), hex(b)])
            rows = list(m.cs.disasm(m.pe.get_data(a, b - a), a))
            assert sum(i.size for i in rows) == b - a
            instructions.update({i.address: i for i in rows})
    assert ranges == [['0x9e1d00', '0x9e1d11'], ['0x9e1d11', '0x9e1d16'],
                      ['0x9e1d16', '0x9e1e18'], ['0x9e1e18', '0x9e1e2f'], ['0x9e1e2f', '0x9e1e52']]
    checks = {0x9E1D04: ('cmp', 'dword ptr [rip + 0x11ee395], -1'),
              0x9E1D24: ('mov', 'dword ptr [rip + 0x11ee376], esi'),
              0x9E1D43: ('lea', 'edx, [rsi + 8]'), 0x9E1D72: ('lea', 'edx, [r9 + 0x19]'),
              0x9E1D86: ('cmp', 'eax, 0x7a'), 0x9E1D99: ('mov', 'ecx, 0x40'),
              0x9E1DC8: ('mov', 'edx, 0x19'), 0x9E1DE6: ('mov', 'rcx, qword ptr [rdi]'),
              0x9E1DE9: ('xor', 'edx, edx'), 0x9E1DF1: ('mov', 'ecx, dword ptr [rax]'),
              0x9E1E26: ('test', 'ebx, ebx'), 0x9E1E2F: ('mov', 'dword ptr [rip + 0x11ee267], 0xffffffff'),
              0x9E1E39: ('xor', 'al, al'), 0x9E1E3F: ('ret', ''),
              0x9E1E40: ('cmp', 'dword ptr [rip + 0x11ee256], 0x1000'),
              0x9E1E4A: ('sete', 'al'), 0x9E1E51: ('ret', '')}
    for address, expected in checks.items():
        assert address in instructions and (instructions[address].mnemonic, instructions[address].op_str) == expected
    calls = {}
    for i in instructions.values():
        if i.mnemonic == 'call':
            slot = i.address + i.size + i.operands[0].mem.disp
            assert slot in IMPORTS
            calls[hex(i.address)] = IMPORTS[slot][1]
    assert set(calls) == {hex(n) for n in [0x9E1D35, 0x9E1D46, 0x9E1D50, 0x9E1D76,
                                         0x9E1D80, 0x9E1D8B, 0x9E1D9E, 0x9E1DAC,
                                         0x9E1DD2, 0x9E1DDC, 0x9E1DEB, 0x9E1E03, 0x9E1E1B]}
    verified.update(token_imports=imports, token_native_ranges=ranges, token_call_sites=calls,
                    token_instruction_assertions=len(checks), token_cached_global=hex(CACHE))
    return verified


def compact(result):
    return {k: v for k, v in result.items() if k != 'events'} | {
        'event_kinds': [e['kind'] for e in result['events']], 'native_return_verified': result['returned']}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases, baselines, failures, sequences = [], [], [], []
    for cached in [0, 1, 0xFFF, 0x1000, 0x1001, 0x2000, 0xFFFFFFFF - 1]:
        result = m.run_token(cached_value=cached)
        assert result['returned'] and not result['events']
        assert result['result_al'] == int(cached == 0x1000)
        cases.append(compact(result))
    for subauthority, size_result, close_result, free_result in itertools.product(
            [0, 0xFFF, 0x1000, 0x1001, 0x2000, 0xFFFFFFFF], [0, 1], [0, 1], [0, 0xABCD]):
        options = {'subauthority': subauthority, 'size_result': size_result,
                   'close_result': close_result, 'free_result': free_result}
        result = m.run_token(options)
        assert result['returned'] and result['final']['cached_value'] == subauthority
        assert result['result_al'] == int(subauthority == 0x1000)
        assert len(result['final']['closed_handles']) == len(result['final']['freed_buffers']) == 1
        assert len([e for e in result['events'] if e['kind'] == 'GetLastError_service']) == int(size_result == 0)
        cases.append(compact(result))
    failure_options = []
    for error in [0, 5, 122, 0xFFFFFFFF]:
        failure_options.extend([{'open_result': 0, 'open_handle': 0, 'last_errors': [error]},
                                {'open_result': 0, 'last_errors': [error]},
                                {'size_result': 1, 'allocation_success': False, 'last_errors': [error]},
                                {'size_result': 1, 'data_result': 0, 'last_errors': [error]}])
    for first, second in itertools.product([0, 5, 0xFFFFFFFF], [0, 5, 122, 0xFFFFFFFF]):
        failure_options.append({'last_errors': [first, second]})
    for options in failure_options:
        result = m.run_token(options)
        assert result['returned'] and result['result_al'] == 0
        error = options['last_errors'][-1]
        assert result['final']['cached_value'] == (0xFFFFFFFF if error else 0)
        assert bool(result['final']['freed_buffers']) == ('data_result' in options)
        assert bool(result['final']['closed_handles']) == (options.get('open_handle', 0xABCDEF) != 0)
        cases.append(compact(result))
    # DWORD/BOOL tests preserve poisoned upper RAX while only the actual operand width is consumed.
    for options in [{'open_result': 0x100000000, 'open_handle': 0, 'last_errors': [5]},
                    {'size_result': 0x80000000, 'data_result': 0xFFFFFFFF},
                    {'size_result': 1, 'required_size': 0},
                    {'size_result': 1, 'required_size': 4096}]:
        result = m.run_token(options)
        assert result['returned']
        assert result['result_al'] == (0 if options.get('open_result') == 0x100000000 else 1)
        cases.append(compact(result))
    for first in [{'subauthority': 0x1000}, {'subauthority': 0x2000},
                  {'open_result': 0, 'open_handle': 0, 'last_errors': [5]},
                  {'open_result': 0, 'open_handle': 0, 'last_errors': [0]},
                  {'subauthority': 0xFFFFFFFF}]:
        a = m.run_token(first)
        b = m.run_token({'subauthority': 0x1000}, cached_value=None)
        retry = a['final']['cached_value'] == 0xFFFFFFFF
        assert bool(b['events']) == retry
        assert b['result_al'] == (1 if retry else a['result_al'])
        sequences.append({'first': compact(a), 'second': compact(b), 'discovery_retried': retry})
    for options in [{}, {'size_result': 1, 'data_result': 0, 'last_errors': [5]}]:
        baseline = m.run_token(options)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run_token({**options, 'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases),
            'cache_sequences': sequences, 'cache_sequence_count': len(sequences),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.executed),
            'scope': 'The complete native cold/cached token predicate executes, including cleanup, cache mutation, operand widths and both normal returns. All security, SID, allocation, process-handle and last-error Windows APIs are supplied authored services. No real security token or Windows API access, invalid SID dereference, caller/provider join, or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['cache_sequence_count'], report['failure_case_count'], report['executed_address_count'])
