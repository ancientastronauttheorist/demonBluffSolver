"""Execute the pinned JSON reader registry with explicit runtime class tokens.

Class discovery, storage reservation and optional extension lookup are services.
The registry's writes, reset ordering and native handler pointers are executed.
"""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_parser import Machine as ParserMachine
from audit_unityplayer_wait import ENGINE_SHA256


class Machine(ParserMachine):
    def __init__(self, game_root):
        self.phase = None
        super().__init__(game_root)
        self.provider = self.arena + 0x100000
        self.core = self.arena + 0x101000
        self.runtime = self.arena + 0x102000
        self.rows = self.arena + 0x104000
        self.extension = self.arena + 0x106000
        self.sources = {}
        for origin, address, size in [('core', self.core, 0x400),
                                      ('runtime', self.runtime, 0xE78)]:
            for offset in range(0, size, 8):
                token = self.arena + 0x200000 + len(self.sources) * 0x100
                self.q(address + offset, token)
                self.sources[token] = f'{origin}+{offset:#x}'
        self.q(self.base + 0x1C6E708, self.core)
        self.q(self.base + 0x1CD6AF8, self.runtime)
        self.q(self.extension, self.extension + 0x100)
        self.q(self.extension + 0x110, self.services + 0x100)
        self.extension_token = self.arena + 0x300000
        self.sources[self.extension_token] = 'optional_extension'

    def snapshot(self):
        count = self.rq(self.provider + 0x10)
        assert count <= 40
        pointer = self.rq(self.provider)
        rows = []
        for index in range(count):
            a = pointer + index * 0x28
            rows.append({'source': self.sources.get(self.rq(a), 'unset'),
                         'handlers': [hex(self.rq(a + offset) - self.base)
                                      if self.rq(a + offset) else None
                                      for offset in (8, 0x10, 0x18)],
                         'flags_hex': bytes(self.u.mem_read(a + 0x20, 8)).hex()})
        return {'count': count, 'rows': rows,
                'storage_present': bool(pointer),
                'capacity': self.rq(self.provider + 0x18) >> 1,
                'runtime_initialized': self.rq(self.base + 0x1CD6AF8) == self.runtime}

    def event(self, kind):
        self.events.append({'kind': kind, 'snapshot': self.snapshot()})
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.failure == [kind, self.counts[kind]]:
            self.error = kind
            self.u.emu_stop()
            return False
        return True

    def hook(self, uc, address, size, data):
        if self.phase != 'registry':
            return super().hook(uc, address, size, data)
        self.executed.add(address - self.base)
        self.registry_executed.add(address - self.base)
        cx = self.reg(self.x.UC_X86_REG_RCX)
        rva = address - self.base
        if rva == 0x1520D0:
            assert cx == self.provider
            if self.event('reserve'):
                # Deliberately grow by a selected amount, retaining old rows.
                self.q(self.provider, self.rows)
                count = self.rq(self.provider + 0x10)
                self.q(self.provider + 0x18, (count + self.growth) * 2)
                self.ret()
        elif rva == 0x354EC0:
            assert cx == self.manager
            assert self.reg(self.x.UC_X86_REG_RDX) == self.rows
            if self.event('free_old_storage'):
                self.ret()
        elif rva == 0x81A880:
            if self.event('runtime_initialize'):
                self.q(self.base + 0x1CD6AF8, self.runtime)
                self.ret()
        elif rva == 0x75F6B0:
            assert cx == self.base + 0x81A880
            assert self.reg(self.x.UC_X86_REG_RDX) == self.base + 0x81A850
            if self.event('runtime_callback_register'):
                self.ret()
        elif address == self.services + 0x100:
            assert cx == self.extension
            out = self.reg(self.x.UC_X86_REG_RDX)
            if self.event('extension_class_lookup'):
                self.q(out, self.extension_token)
                self.ret(out)

    def call(self, rva):
        x = self.x
        sp = self.stack + 0x18008
        self.u.mem_write(self.stack, bytes([self.stack_seed]) * 0x18000)
        self.q(sp, self.stop)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI,
                     x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13,
                     x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(registers):
            self.u.reg_write(register, 0xABC00000 + i)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        self.u.reg_write(x.UC_X86_REG_RCX, self.provider)
        try:
            self.u.emu_start(self.base + rva, self.stop, timeout=2_000_000,
                            count=100000)
        except Exception as exc:
            raise AssertionError(f'RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(registers):
                assert self.reg(register) == 0xABC00000 + i
        return returned

    def build(self, options):
        self.phase = 'registry'
        self.u.mem_write(self.provider, bytes(0xD0))
        self.u.mem_write(self.rows, bytes(0x800))
        self.stack_seed = options.get('stack_seed', 0)
        self.events, self.counts = [], {}
        self.failure, self.error = options.get('failure'), None
        self.growth = options.get('growth', 40)
        assert self.call(0x783F40)
        if options.get('old_storage'):
            self.q(self.provider, self.rows)
            self.q(self.provider + 0x10, 1)
            self.q(self.provider + 0x18, 2 | options.get('borrowed', False))
            self.q(self.rows, self.extension_token)
        self.q(self.base + 0x1CD6AF8, 0 if options.get('cold_runtime') else self.runtime)
        self.q(self.base + 0x1CD5760, self.extension if options.get('extension') else 0)
        returned = self.call(0xA8F090)
        result = {'input': options.copy(), 'events': self.events.copy(),
                  'final': self.snapshot(), 'error': self.error,
                  'returned': returned}
        self.phase = None
        return result


def verify_native(m):
    ranges = [(0x783F40, 0x783FE0), (0xA8F090, 0xA902CE),
              (0xA8CC80, 0xA8CDF5)]
    instructions = {}
    for a, b in ranges:
        section = m.pe.get_section_by_rva(a)
        assert section and b - section.VirtualAddress <= section.SizeOfRawData
        raw = m.pe.get_data(a, b - a)
        assert len(raw) == b - a
        decoded = list(m.cs.disasm(raw, a))
        assert sum(i.size for i in decoded) == b - a
        assert decoded[-1].mnemonic == 'ret'
        instructions.update({i.address: i for i in decoded})
    # The constructor is a pointer-bound leaf without an unwind record.
    for a, b in ranges[1:]:
        assert any(e.struct.BeginAddress == a and e.struct.EndAddress == b
                   for e in m.pe.DIRECTORY_ENTRY_EXCEPTION)
    checks = {
        0x783F42: ('mov', 'dword ptr [rcx + 8], 0x2b'),
        0x783F50: ('mov', 'qword ptr [rcx + 0x18], 1'),
        0x783FDF: ('ret', ''),
        0xA8CCD5: ('call', '0x783f40'),
        0xA8CCE5: ('call', '0xa8f090'),
        0xA8CD5D: ('mov', 'qword ptr [rax + 0x48], rbp'),
        0xA8CD7D: ('call', '0xa902d0'),
        0xA8CDE1: ('mov', 'qword ptr [rax + 0x40], rdi'),
        0xA8F0CF: ('call', '0x354ec0'),
        0xA8F0DE: ('mov', 'qword ptr [rbx], r15'),
        0xA8F0E1: ('mov', 'qword ptr [rbx + 0x10], r15'),
        0xA8F0E5: ('mov', 'qword ptr [rbx + 0x18], 1'),
        0xA8F0F9: ('call', '0x81a880'),
        0xA8F10C: ('call', '0x75f6b0'),
        0xA8F118: ('mov', 'rax, qword ptr [r14 + 0x120]'),
        0xA8F12A: ('lea', 'rax, [rip + 0x269f]'),
        0xA8F156: ('mov', 'byte ptr [rbp - 0xc], r15b'),
        0xA8F162: ('call', '0x1520d0'),
        0xA8F172: ('mov', 'qword ptr [rbx + 0x10], rsi'),
        0xA8F193: ('movsd', 'qword ptr [rdx + 0x20], xmm0'),
        0xA90102: ('call', 'qword ptr [rax + 0x10]'),
        0xA902CD: ('ret', ''),
    }
    for address, expected in checks.items():
        assert address in instructions
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
    i = instructions[0xA8F12A]
    assert i.address + i.size + 0x269F == 0xA917D0
    for address, target in [(0xA8CCEA, 0x1CD66A8),
                            (0xA8F0D4, 0x1CD6AF8),
                            (0xA8F0ED, 0x1C6E708)]:
        i = instructions[address]
        assert i.mnemonic == 'mov' and 'rip' in i.op_str
        import struct
        assert i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0] == target
    return {'instruction_assertions': len(checks) + 4,
            'entry_ranges': [[hex(a), hex(b)] for a, b in ranges],
            'constructor_boundary': 'Leaf ending at its verified return; caller binds exact entry.',
            'publication_binding': 'Reader provider at registry slot +0x48; writer provider at +0x40.'}


def audit(game_root):
    m = Machine(game_root)
    m.registry_executed = set()
    verified = verify_native(m)
    cases = []
    for growth in [1, 4, 40]:
        for seed in [0, 0xA5]:
            for extension in [False, True]:
                options = {'growth': growth, 'stack_seed': seed,
                           'extension': extension}
                result = m.build(options)
                assert result['returned']
                assert result['final']['count'] == 33 + extension
                first = result['final']['rows'][0]
                assert first['source'] == 'core+0x120'
                assert first['handlers'] == ['0xa917d0', '0xa91810', '0xa91830']
                # Four metadata bytes and the feature byte are written; three
                # upper bytes retain the explicitly seeded stack contents.
                assert first['flags_hex'] == '0000000000' + f'{seed:02x}' * 3
                cases.append(result)
    for options in [{'old_storage': True}, {'old_storage': True, 'borrowed': True},
                    {'cold_runtime': True}, {'extension': True, 'growth': 4}]:
        baseline = m.build(options)
        assert baseline['returned']
        cases.append(baseline)
        counts = {}
        for i, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.build(dict(options, failure=[kind, counts[kind]]))
            assert not result['returned'] and result['error'] == kind
            assert result['events'] == baseline['events'][:i + 1]
            assert result['final'] == event['snapshot']
            cases.append(result)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'native_verified': verified,
            'cases': cases, 'case_count': len(cases),
            'executed_address_count': len(m.registry_executed),
            'scope': 'Actual reader-provider initialization and registry construction. Runtime class tokens, vector reservation, free and optional extension lookup are supplied. Stack bytes are explicit fixtures; no class-discovery or field-conversion claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['executed_address_count'])
