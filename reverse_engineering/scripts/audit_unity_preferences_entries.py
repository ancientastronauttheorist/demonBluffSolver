"""Execute engine preference entry bodies and native UTF-16 conversion offline."""
import argparse
import itertools
import json
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_parser import Machine as ParserMachine
from audit_unity_json_strings import EXPORTS, verify_native as verify_strings


class Machine(ParserMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.cs.detail = True
        self.strings, self.string_cursor = {}, self.arena + 0x600000
        self.runtime_services = {}
        for index, (slot, name, _, _) in enumerate(EXPORTS):
            address = self.services + 0x100 + index * 0x10
            self.q(self.base + slot, address)
            self.runtime_services[address] = name
        for index, (slot, name) in enumerate([(0x1CD6688, 'il2cpp_gc_wbarrier_set_field'),
                                               (0x1CD6068, 'il2cpp_string_new_len')]):
            address = self.services + 0x140 + index * 0x10
            self.q(self.base + slot, address)
            self.runtime_services[address] = name
        tib, slots, tls = [self.arena + n for n in (0x700000, 0x701000, 0x702000)]
        self.u.reg_write(self.x.UC_X86_REG_GS_BASE, tib)
        self.q(tib + 0x10, self.stack)
        self.q(tib + 0x58, slots)
        self.q(slots, tls)
        self.d(self.base + 0x1C4A0EC, 0)
        self.d(tls + 0x10, 0)
        self.d(self.base + 0x1CDA810, 0)
        self.d(self.base + 0x1CDA814, 0xFFFD)
        self.backend = self.arena + 0x11000
        self.owner, self.owner_vtable = self.arena + 0x12000, self.arena + 0x12100
        self.owner_free = self.services + 0x160
        self.q(self.owner, self.owner_vtable)
        self.q(self.owner_vtable + 0x18, self.owner_free)

    def make_string(self, text):
        if text is None:
            return 0
        raw = text.encode('utf-16-le', errors='surrogatepass')
        token = self.string_cursor
        self.string_cursor += (len(raw) + 0x40 + 15) & ~15
        chars = token + 0x20
        assert self.string_cursor < self.arena + 0xF00000
        self.u.mem_write(chars, raw + b'\0\0')
        self.strings[token] = {'text': text, 'chars': chars, 'length': len(raw) // 2}
        return token

    def slice(self, pointer):
        data, length = self.rq(pointer), self.rq(pointer + 8)
        assert length <= 65536
        return bytes(self.u.mem_read(data, length)) if length else b''

    def snapshot(self):
        return {'reference_store_count': self.reference_count,
                'backend_writes': self.backend_writes.copy(),
                'allocated_buffer_sizes': list(self.allocations.values()),
                'owner_free_count': self.owner_free_count}

    def event(self, kind, args):
        self.events.append({'kind': kind, 'args': args, 'snapshot': self.snapshot()})
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.options.get('failure') == [kind, self.counts[kind]]:
            self.error = kind
            self.u.emu_stop()
            return False
        return True

    def hook(self, uc, address, size, data):
        self.executed.add(address - self.base)
        x = self.x
        cx, dx, r8, r9 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        name = self.runtime_services.get(address)
        if name:
            if name in ['il2cpp_string_length', 'il2cpp_string_chars']:
                assert cx in self.strings
                value = self.strings[cx]
                if self.event(name, [value['text']]):
                    self.ret(value['length' if name.endswith('length') else 'chars'])
            elif name == 'il2cpp_gc_wbarrier_set_field':
                assert cx == 0 and (r8 == 0 or r8 in self.strings)
                if self.event(name, [self.strings[r8]['text'] if r8 else None]):
                    self.q(dx, r8)
                    self.reference_count += 1
                    self.ret()
            elif name == 'il2cpp_string_new_len':
                raw = bytes(self.u.mem_read(cx, dx & 0xFFFFFFFF))
                if self.event(name, [raw.hex()]):
                    self.ret(self.make_string(raw.decode('utf-8')))
            else:
                raise AssertionError(name)
            return
        rva = address - self.base
        if rva == 0x7E58B0:
            assert cx & 0xFF == 1
            if self.event('backend_provider_service', [cx & 0xFF]):
                self.ret(self.backend)
        elif rva == 0x7E6030:
            assert cx == self.backend and r8 & 0xFFFFFFFF == 3
            key, length = self.slice(dx), self.rq(self.reg(x.UC_X86_REG_RSP) + 0x28)
            assert 1 <= length <= 65537
            value = bytes(self.u.mem_read(r9, length))
            assert value[-1] == 0
            args = [key.hex(), value.hex(), length]
            if self.event('backend_set_service', args):
                if self.options.get('set_success', True):
                    self.backend_writes.append(args)
                self.ret(0xDEADBEEF00000000 | int(self.options.get('set_success', True)))
        elif rva == 0x7E61A0:
            key, default = self.slice(dx), self.slice(r8)
            result = self.options.get('backend_value')
            raw = default if result is None else bytes.fromhex(result)
            if self.event('backend_get_service', [key.hex(), default.hex(), raw.hex()]):
                self.put_string(cx, raw)
                self.ret(cx)
        elif rva == 0x354970:
            assert cx == self.manager
            if self.event('allocate_service', [dx, r8, r9]):
                self.ret(self.alloc(dx))
        elif rva == 0x354EC0:
            assert cx == self.manager and dx in self.allocations
            if self.event('free_service', [self.allocations[dx], r8 & 0xFFFFFFFF]):
                self.ret()
        elif rva == 0x355150:
            assert cx == self.manager and dx in self.allocations
            if self.event('allocation_owner_service', [self.allocations[dx]]):
                self.ret(self.owner)
        elif address == self.owner_free:
            assert cx == self.owner and dx in self.allocations
            if self.event('owner_free_service', [self.allocations[dx]]):
                self.owner_free_count += 1
                self.ret()
        elif rva == 0x159230:
            raw = bytes(self.u.mem_read(dx, r8)) if r8 else b''
            if self.event('string_assign_service', [raw.hex()]):
                self.put_string(cx, raw)
                self.ret(cx)
        elif rva == 0x14F740:
            raw = self.get_string(dx)
            if self.event('string_copy_service', [raw.hex()]):
                self.put_string(cx, raw)
                self.ret(cx)

    def run(self, direction, key, value, options=None):
        self.options = options or {}
        self.events, self.counts, self.backend_writes = [], {}, []
        self.reference_count, self.owner_free_count, self.error = 0, 0, None
        self.strings, self.string_cursor = {}, self.arena + 0x600000
        self.allocations, self.next_alloc = {}, self.arena + 0x20000
        self.u.mem_write(self.backend + 8, bytes([int(self.options.get('blocked', False))]))
        a, b = self.make_string(key), self.make_string(value)
        storage = [(v['chars'], bytes(self.u.mem_read(v['chars'], v['length'] * 2 + 2))) for v in self.strings.values()]
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                     x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(registers):
            self.u.reg_write(register, 0xFAB00000 + i)
        for register, v in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, a), (x.UC_X86_REG_RDX, b)]:
            self.u.reg_write(register, v)
        self.u.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        entry = 0xF22B0 if direction == 'set' else 0xF3150
        try:
            self.u.emu_start(self.base + entry, self.stop, timeout=10_000_000, count=1000000)
        except Exception as exc:
            raise AssertionError(f'{direction}, {self.options}, RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        result = None
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(registers):
                assert self.reg(register) == 0xFAB00000 + i
            result = self.reg(x.UC_X86_REG_RAX)
            if direction == 'get':
                assert result in self.strings
                result = self.strings[result]['text']
            else:
                assert result in [0, 1]
        assert all(bytes(self.u.mem_read(a, len(raw))) == raw for a, raw in storage)
        return {'direction': direction, 'key': key, 'value_or_default': value, 'options': self.options,
                'returned': returned, 'result': result, 'error': self.error, 'events': self.events.copy(),
                'final': self.snapshot(), 'input_storage_retained': True}


def verify_native(m):
    verified = verify_strings(m)
    groups = {}
    for entry in m.pe.DIRECTORY_ENTRY_EXCEPTION:
        root, visited = entry, set()
        while root.unwindinfo.Flags & 4:
            assert root.struct.BeginAddress not in visited
            visited.add(root.struct.BeginAddress)
            root = root.unwindinfo._chained_entry
        groups.setdefault(root.struct.BeginAddress, []).append((entry.struct.BeginAddress, entry.struct.EndAddress))
    instructions, families = {}, {}
    for root in [0xF22B0, 0xF3150, 0x4A86B0]:
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            assert len(raw) == b - a
            rows = list(m.cs.disasm(raw, a))
            assert sum(i.size for i in rows) == b - a
            instructions.update({i.address: i for i in rows})
    checks = {0xF2376: ('call', '0x4a86b0'), 0xF23BC: ('call', '0x4a86b0'),
              0xF2449: ('call', '0x7e58b0'), 0xF2451: ('cmp', 'byte ptr [rax + 8], r15b'),
              0xF246C: ('mov', 'r8d, 3'), 0xF247A: ('call', '0x7e6030'),
              0xF2967: ('movzx', 'eax, r13b'), 0xF298A: ('ret', ''),
              0xF321B: ('call', '0x4a86b0'), 0xF3261: ('call', '0x4a86b0'),
              0xF32F3: ('call', '0x7e61a0'), 0xF331B: ('call', 'rax'),
              0xF3889: ('ret', ''), 0x4A8718: ('cmp', 'r12, 0x18'),
              0x4A8829: ('cmp', 'rdx, 0x7d0'), 0x4A8849: ('call', '0x17a86c0'),
              0x4A8883: ('call', '0x354970'), 0x4A88A8: ('call', '0x4a8590'),
              0x4A88CD: ('call', '0x159230'), 0x4A8DBF: ('ret', '')}
    for a, expected in checks.items():
        assert a in instructions and (instructions[a].mnemonic, instructions[a].op_str) == expected
    exports = [(0x1CD6688, 'il2cpp_gc_wbarrier_set_field', 0x76DAE5, 0x76DAF1),
               (0x1CD6068, 'il2cpp_string_new_len', 0x76E4B1, 0x76E4BD)]
    for slot, name, lea, store in exports:
        a = next(m.cs.disasm(m.pe.get_data(lea, 7), lea))
        b = next(m.cs.disasm(m.pe.get_data(store, 7), store))
        assert a.mnemonic == 'lea' and b.mnemonic == 'mov'
        literal = a.address + a.size + a.operands[1].mem.disp
        assert m.pe.get_data(literal, 80).split(b'\0', 1)[0] == name.encode('ascii')
        assert b.address + b.size + b.operands[0].mem.disp == slot
    verified.update(preference_native_ranges=families, preference_instruction_assertions=len(checks) + 4)
    return verified


def normalized(text):
    if text in ['\ud800', '\udc00', '\ud800Z']:
        return '\ufffd'
    return text or ''


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases, baselines, failures = [], [], []
    texts = [None, '', 'hello', 'a\0b', 'caf\u00e9 \U0001f608', '\u6f22\u5b57',
             'a' * 24, 'a' * 25, 'a' * 499, 'a' * 500, '\ud800', '\udc00', '\ud800Z']
    for key, value in itertools.product(texts[:7], texts):
        result = m.run('set', key, value)
        assert result['returned'] and result['result'] == 1
        write = result['final']['backend_writes'][0]
        assert bytes.fromhex(write[0]) == normalized(key).encode('utf-8')
        assert bytes.fromhex(write[1]) == normalized(value).encode('utf-8') + b'\0'
        cases.append(result)
    for blocked, success in itertools.product([False, True], repeat=2):
        result = m.run('set', 'key', 'value', {'blocked': blocked, 'set_success': success})
        assert result['returned'] and result['result'] == int(not blocked and success)
        assert any(e['kind'] == 'backend_set_service' for e in result['events']) == (not blocked)
        cases.append(result)
    for key, direction in itertools.product(texts[7:], ['set', 'get']):
        result = m.run(direction, key, 'payload')
        assert result['returned']
        if direction == 'set':
            assert result['result'] == 1
            assert bytes.fromhex(result['final']['backend_writes'][0][0]) == normalized(key).encode('utf-8')
        else:
            assert result['result'] == 'payload'
            get = next(e for e in result['events'] if e['kind'] == 'backend_get_service')
            assert bytes.fromhex(get['args'][0]) == normalized(key).encode('utf-8')
        cases.append(result)
    for key, default, stored in itertools.product(texts[:7], texts, [None, 'stored\0value', '\u6f22\U0001f608']):
        options = {'backend_value': stored.encode('utf-8').hex()} if stored is not None else {}
        result = m.run('get', key, default, options)
        assert result['returned'] and result['result'] == (stored if stored is not None else normalized(default))
        cases.append(result)
    for direction, key, value in [('set', 'caf\u00e9', 'x' * 500), ('get', 'caf\u00e9', 'x' * 500)]:
        baseline = m.run(direction, key, value)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run(direction, key, value, {'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1, 'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.executed),
            'scope': 'Native engine preference entry bodies, chained cleanup ranges and UTF-16 conversion execute. Backend provider/get/set, runtime string/reference exports, string copy/assign and allocation/ownership/free remain explicit services. Fixtures use allocator-manager flag zero with supplied ownership. No actual Windows registry access or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
