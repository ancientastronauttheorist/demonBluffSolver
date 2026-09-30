"""Execute native numeric field readers over actual parsed JSON objects.

Field descriptors and object layout are explicit fixtures. Parser, registry
handler selection, key lookup and scalar conversion execute native code.
"""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_fields import verify_native as verify_adapter
from audit_unity_json_registry import Machine as RegistryMachine
from audit_unity_json_registry import verify_native as verify_registry
from audit_unityplayer_wait import ENGINE_SHA256


# Sources are exact runtime table offsets, not inferred managed class names.
SOURCES = [('core+0x120', 4), ('core+0x130', 1), ('core+0x188', 4),
           ('core+0x118', 2), ('core+0x128', 8), ('core+0x178', 2),
           ('core+0x168', 1), ('core+0x108', 4), ('core+0x110', 8)]


class Machine(RegistryMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.registry_executed = set()
        self.primitive_executed = set()
        self.registry_verified = verify_registry(self)
        self.adapter_verified = verify_adapter(self)
        registry = self.build({})
        self.handlers = {row['source']: int(row['handlers'][0], 16)
                         for row in registry['final']['rows']}
        self.descriptor = self.arena + 0x400000
        self.context = self.arena + 0x401000
        self.managed = self.arena + 0x402000
        self.key = self.arena + 0x403000
        self.cache_slot = self.arena + 0x404000
        self.cache = self.arena + 0x405000
        self.descriptors = self.arena + 0x406000
        self.klass = self.arena + 0x407000
        self.q(self.base + 0x1CD6688, self.services + 0x180)

    def hook(self, uc, address, size, data):
        if self.phase in ('primitive', 'batch'):
            self.primitive_executed.add(address - self.base)
        if self.phase == 'batch':
            if address == self.services + 0x180:
                assert self.reg(self.x.UC_X86_REG_RCX) == 0
                target = self.reg(self.x.UC_X86_REG_R8)
                assert target == self.managed
                self.q(self.reg(self.x.UC_X86_REG_RDX), target)
                self.batch_events.append('reference_store')
                return self.ret()
            if address - self.base == 0x14E2D0:
                self.batch_events.append('vector_cleanup_service')
                return self.ret()
        return super().hook(uc, address, size, data)

    def invoke(self, rva, arguments):
        x = self.x
        sp = self.stack + 0x18008
        self.q(sp, self.stop)
        preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI,
                     x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13,
                     x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(preserved):
            self.u.reg_write(register, 0xFED00000 + i)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        for register, value in zip([x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                                    x.UC_X86_REG_R8, x.UC_X86_REG_R9], arguments):
            self.u.reg_write(register, value)
        try:
            self.u.emu_start(self.base + rva, self.stop,
                            timeout=2_000_000, count=100000)
        except Exception as exc:
            raise AssertionError(f'entry={rva:x}, '
                                 f'RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        assert self.reg(x.UC_X86_REG_RIP) == self.stop
        assert self.reg(x.UC_X86_REG_RSP) == sp + 8
        for i, register in enumerate(preserved):
            assert self.reg(register) == 0xFED00000 + i

    def apply(self, source, payload, options=None):
        options = options or {}
        self.phase = None
        parsed = self.parse(payload)
        assert parsed['tree'] is not None
        assert self.next_alloc < self.provider
        self.phase = 'primitive'
        self.u.mem_write(self.descriptor, bytes(0x80))
        self.u.mem_write(self.context, bytes(0x80))
        self.u.mem_write(self.managed, bytes([0xCD]) * 0x100)
        key = options.get('key', 'score').encode('utf-8')
        self.u.mem_write(self.key, key + b'\0')
        self.q(self.descriptor + 0x10, self.key)
        self.d(self.descriptor + 0x2C, 0x20)
        self.d(self.descriptor + 0x34, options.get('field_flags', 0))
        reference = options.get('reference', True)
        self.u.mem_write(self.context, bytes([int(reference)]))
        self.q(self.context + 8, self.managed)
        self.d(self.context + 0x18, 0x18)
        self.q(self.context + 0x28, self.last_tree)
        self.u.mem_write(self.last_tree, bytes([options.get('parser_flags', 0)]))
        self.u.mem_write(self.last_tree + 0x70, b'\x01')
        original_node = self.rq(self.last_tree + 0x78)
        original_type = self.rq(self.last_tree + 0x40)
        original_depth = self.rq(self.last_tree + 0x90)
        self.invoke(self.handlers[source], [self.descriptor, self.context])
        assert self.rq(self.last_tree + 0x78) == original_node
        assert self.rq(self.last_tree + 0x40) == original_type
        assert self.rq(self.last_tree + 0x90) == original_depth
        width = dict(SOURCES)[source]
        offset = 0x20 if reference else 0x28
        raw = bytes(self.u.mem_read(self.managed, 0x100))
        assert raw[:offset] == bytes([0xCD]) * offset
        assert raw[offset + width:] == bytes([0xCD]) * (0x100 - offset - width)
        result = {'source': source, 'handler_rva': hex(self.handlers[source]),
                  'json_utf8_hex': payload.hex(), 'options': options.copy(),
                  'value_hex': raw[offset:offset + width].hex(),
                  'field_found': bool(self.u.mem_read(self.last_tree + 0x70, 1)[0]),
                  'field_error': bool(self.u.mem_read(self.last_tree + 0x30, 1)[0]),
                  'retained_native_context': True}
        self.phase = None
        return result

    def batch(self, payload, sources):
        assert len(sources) <= 9
        self.phase = None
        assert self.parse(payload)['tree'] is not None
        self.phase = 'batch'
        self.batch_events = []
        self.u.mem_write(self.managed, bytes([0xCD]) * 0x100)
        self.u.mem_write(self.cache, bytes(0x100))
        self.u.mem_write(self.descriptors, bytes(0x500))
        self.q(self.cache_slot, self.cache)
        self.d(self.cache + 0x10, 1)
        self.u.mem_write(self.cache + 0x18, b'\x09\x00')
        self.q(self.cache + 0x20, self.descriptors)
        self.q(self.cache + 0x30, len(sources))
        for i, source in enumerate(sources):
            descriptor = self.descriptors + i * 0x80
            key = self.key + i * 0x20
            self.u.mem_write(key, f'field{i}'.encode('ascii') + b'\0')
            self.q(descriptor + 8, self.base + self.handlers[source])
            self.q(descriptor + 0x10 + 0x10, key)
            self.d(descriptor + 0x10 + 0x2C, 0x20 + i * 0x10)
        self.invoke(0xA8E030, [self.last_tree, self.managed, self.klass, self.cache_slot])
        assert self.rq(self.last_tree + 0x10) == 0
        assert self.batch_events == ['reference_store'] + ['vector_cleanup_service'] * 3
        raw = bytes(self.u.mem_read(self.managed, 0x100))
        values = []
        for i, source in enumerate(sources):
            offset, width = 0x20 + i * 0x10, dict(SOURCES)[source]
            values.append(raw[offset:offset + width].hex())
        allowed = {0x20 + i * 0x10 + j for i, source in enumerate(sources)
                   for j in range(dict(SOURCES)[source])}
        assert all(byte == 0xCD for i, byte in enumerate(raw) if i not in allowed)
        self.phase = None
        return {'json_utf8_hex': payload.hex(), 'sources': sources,
                'values_hex': values, 'events': self.batch_events.copy(),
                'scope_links_cleared': True}


def verify_native(m):
    ranges = []
    assertion_count = 0
    converters = [0xA00E00, 0xA8EA00, 0xA9B910, 0xA9C810, 0xA9C8D0,
                  0xA8E930, 0xA8EAC0, 0xA9DBD0, 0xA9DC90]
    for (source, _), converter in zip(SOURCES, converters):
        a = m.handlers[source]
        entries = [e.struct for e in m.pe.DIRECTORY_ENTRY_EXCEPTION
                   if e.struct.BeginAddress == a]
        assert len(entries) == 1
        b = entries[0].EndAddress
        raw = m.pe.get_data(a, b - a)
        assert len(raw) == b - a
        decoded = list(m.cs.disasm(raw, a))
        assert sum(i.size for i in decoded) == b - a
        assert decoded[-1].mnemonic == 'ret'
        checks = {
            a + 4: ('cmp', 'byte ptr [rdx], 0'),
            a + 7: ('mov', 'r10, qword ptr [rdx + 0x28]'),
            a + 11: ('movsxd', 'r8, dword ptr [rcx + 0x2c]'),
            a + 26: ('movsxd', 'rax, dword ptr [rdx + 0x18]'),
            a + 34: ('lea', 'rdx, [r8 - 0x10]'),
            a + 41: ('mov', 'r9d, dword ptr [rcx + 0x34]'),
            a + 45: ('mov', 'r8, qword ptr [rcx + 0x10]'),
            a + 52: ('call', hex(converter)),
        }
        instructions = {i.address: i for i in decoded}
        for address, expected in checks.items():
            assert address in instructions
            i = instructions[address]
            assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
        assertion_count += len(checks)
        ranges.append([hex(a), hex(b)])
    return {'instruction_assertions': assertion_count,
            'complete_field_wrapper_ranges': ranges,
            'converter_entries': [hex(a) for a in converters]}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases = []
    batch_cases = []
    payloads = [b'{}', b'{"Score":9}', b'{"score":0}', b'{"score":123}',
                b'{"score":-1}', b'{"score":true}', b'{"score":false}',
                b'{"score":null}', b'{"score":"12"}', b'{"score":"-12xyz"}',
                b'{"score":"abc"}', b'{"score":{}}', b'{"score":[]}',
                b'{"score":[5]}', b'{"score":1.5}', b'{"score":-1.5}',
                b'{"score":2147483648}', b'{"score":-2147483649}',
                b'{"score":4294967295}', b'{"score":4294967296}',
                b'{"score":9223372036854775807}',
                b'{"score":18446744073709551615}',
                b'{"score":18446744073709551616}',
                b'{"score":1e100}', b'{"score":1e-100}',
                b'{"score":NaN}', b'{"score":Infinity}',
                b'{"score":-Infinity}', b'{"score":-0.0}',
                b'{"score":7,"score":11}', b'{"other":11,"score":7}',
                b'{"nested":{"score":11},"score":7}']
    for source, width in SOURCES:
        for payload in payloads:
            result = m.apply(source, payload)
            if payload in payloads[:2]:
                assert result['value_hex'] == 'cd' * width
                assert not result['field_found']
            if payload == b'{"score":7,"score":11}':
                assert result['value_hex'] == m.apply(source, b'{"score":7}')['value_hex']
            cases.append(result)
        baseline = m.apply(source, b'{"score":123}')
        for options in [{'reference': False}, {'parser_flags': 2},
                        {'field_flags': 1 << 19},
                        {'parser_flags': 2, 'field_flags': 1 << 19},
                        {'key': 'other'}]:
            result = m.apply(source, b'{"score":123}', options)
            skipped = options.get('key') == 'other' or (
                options.get('parser_flags') == 2 and
                options.get('field_flags') == 1 << 19)
            assert result['field_found'] == (not skipped)
            assert result['value_hex'] == ('cd' * width if skipped else baseline['value_hex'])
            cases.append(result)
    for sources in [[source for source, _ in SOURCES],
                    ['core+0x120', 'core+0x188', 'core+0x130'],
                    ['core+0x120'] * 3, []]:
        for members in [[], [('field0', 123)],
                        [(f'field{i}', i + 1.5) for i in range(len(sources))],
                        [('field0', 7), ('field0', 11), ('field1', -1)]]:
            payload = ('{' + ','.join(json.dumps(key) + ':' + json.dumps(value)
                                      for key, value in members) + '}').encode('utf-8')
            expected = [m.apply(source, payload, {'key': f'field{i}'})['value_hex']
                        for i, source in enumerate(sources)]
            result = m.batch(payload, sources)
            assert result['values_hex'] == expected
            batch_cases.append(result)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'registry_verified': m.registry_verified,
            'adapter_verified': m.adapter_verified,
            'native_verified': verified,
            'cases': cases, 'case_count': len(cases),
            'adapter_batch_cases': batch_cases, 'adapter_batch_case_count': len(batch_cases),
            'executed_address_count': len(m.primitive_executed),
            'scope': 'Nine actual registry-selected numeric field handlers over actual native parsed objects, including composed native adapter/descriptor traversal and reference-scope cleanup. Native key lookup, conversion, offset calculation and context restoration execute. GC reference store and vector storage cleanup are supplied services; descriptors and managed layout are explicit fixtures. No metadata inclusion, real managed type discovery, strings, arrays or arbitrary copy claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['executed_address_count'])
