"""Execute native numeric descriptor construction and application.

The runtime metadata exports return explicit fixture identities. Actual registry
lookup, descriptor construction and numeric field application execute natively.
"""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_primitives import Machine as PrimitiveMachine, SOURCES
from audit_unityplayer_wait import ENGINE_SHA256


EXPORTS = [
    (0x1CD60A0, 'il2cpp_field_get_name', 0x76D755, 0x76D761),
    (0x1CD61A0, 'il2cpp_field_get_type', 0x76D7C7, 0x76D7D3),
    (0x1CD6140, 'il2cpp_class_from_type', 0x76D379, 0x76D385),
    (0x1CD61B8, 'il2cpp_type_get_type', 0x76E7A9, 0x76E7B5),
    (0x1CD6098, 'il2cpp_field_get_offset', 0x76D7A1, 0x76D7AD),
    (0x1CD6320, 'il2cpp_class_get_name', 0x76D165, 0x76D171),
    (0x1CD6260, 'il2cpp_class_is_valuetype', 0x76D26F, 0x76D27B),
    (0x1CD6090, 'il2cpp_field_get_parent', 0x76D77B, 0x76D787),
    (0x1CD6318, 'il2cpp_class_is_enum', 0x76D437, 0x76D443),
]


class Machine(PrimitiveMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.field = self.arena + 0x500000
        self.field_type = self.arena + 0x501000
        self.description = self.arena + 0x502000
        self.build_context = self.arena + 0x503000
        self.vector = self.arena + 0x504000
        self.class_name = self.arena + 0x505000
        self.u.mem_write(self.class_name, b'FixtureNumeric\0')
        self.descriptor_executed = set()
        self.registry_rows = bytes(self.u.mem_read(self.rows, 33 * 0x28))
        self.export_services = {}
        for i, (slot, name, _, _) in enumerate(EXPORTS):
            service = self.services + 0x600 + i * 0x10
            self.q(self.base + slot, service)
            self.export_services[service] = name

    def snapshot(self):
        if self.phase != 'descriptor':
            return super().snapshot()
        count = self.rq(self.vector + 0x10)
        assert count <= 9
        records = []
        for i in range(count):
            a = self.descriptors + i * 0x80
            handler = self.rq(a + 8)
            records.append({'kind': self.rd(a),
                            'handler_rva': hex(handler - self.base) if handler else None,
                            'field_identity': self.rq(a + 0x18) == self.field,
                            'parent_identity': self.rq(a + 0x10) == self.klass,
                            'key_identity': self.rq(a + 0x20) == self.key,
                            'class_identity': self.rq(a + 0x30) == self.field_class,
                            'type_enum': self.rd(a + 0x38),
                            'field_offset': self.rd(a + 0x3C),
                            'flags': self.rd(a + 0x44),
                            'value_type_byte': self.u.mem_read(a + 0x6C, 1)[0],
                            'tails_zero': self.rq(a + 0x70) == self.rq(a + 0x78) == 0})
        return {'count': count, 'records': records}

    def event(self, kind):
        if self.phase != 'descriptor':
            return super().event(kind)
        self.descriptor_events.append({'kind': kind, 'snapshot': self.snapshot()})
        self.descriptor_counts[kind] = self.descriptor_counts.get(kind, 0) + 1
        if self.descriptor_failure == [kind, self.descriptor_counts[kind]]:
            self.descriptor_error = kind
            self.u.emu_stop()
            return False
        return True

    def hook(self, uc, address, size, data):
        if self.phase != 'descriptor':
            return super().hook(uc, address, size, data)
        self.descriptor_executed.add(address - self.base)
        self.executed.add(address - self.base)
        cx = self.reg(self.x.UC_X86_REG_RCX)
        name = self.export_services.get(address)
        if name:
            if name.startswith('il2cpp_field_'):
                assert cx == self.field
            elif name in ('il2cpp_class_from_type', 'il2cpp_type_get_type'):
                assert cx == self.field_type
            else:
                assert cx == self.field_class
            if self.event(name):
                values = {'il2cpp_field_get_name': self.key,
                          'il2cpp_field_get_type': self.field_type,
                          'il2cpp_class_from_type': self.field_class,
                          'il2cpp_type_get_type': self.type_enum,
                          'il2cpp_field_get_offset': self.field_offset,
                          'il2cpp_class_get_name': self.class_name,
                          'il2cpp_class_is_valuetype': self.value_type,
                          'il2cpp_field_get_parent': self.klass,
                          'il2cpp_class_is_enum': 0}
                self.ret(values[name])
        elif address - self.base == 0x75F2B0:
            assert cx == self.field_class
            if self.event('collection_predicate_service'):
                self.ret(0)
        elif address - self.base == 0x1C77A0:
            assert cx == self.vector
            if self.event('descriptor_reserve_service'):
                self.q(self.vector, self.descriptors)
                self.q(self.vector + 0x18, 20)
                self.ret()

    def construct(self, source, options=None):
        options = options or {}
        self.phase = 'descriptor'
        self.field_class = next(token for token, label in self.sources.items() if label == source)
        self.type_enum = options.get('type_enum', 8)
        assert self.type_enum not in (0x11, 0x12, 0x15, 0x1D)
        self.field_offset = options.get('offset', 0x20)
        self.value_type = options.get('value_type', True)
        self.descriptor_events, self.descriptor_counts = [], {}
        self.descriptor_failure = options.get('failure')
        self.descriptor_error = None
        self.u.mem_write(self.rows, self.registry_rows)
        self.q(self.provider + 0x10, 33)
        selected = next(self.rows + i * 0x28 for i in range(33)
                        if self.rq(self.rows + i * 0x28) == self.field_class)
        if options.get('feature_enabled'):
            self.u.mem_write(selected + 0x24, b'\x01')
        if options.get('disable_handler'):
            self.q(selected + 8, 0)
        if options.get('duplicate_later'):
            assert options.get('disable_handler')
            self.u.mem_write(self.rows + 33 * 0x28,
                             self.registry_rows[selected - self.rows:selected - self.rows + 0x28])
            self.q(self.provider + 0x10, 34)
        self.u.mem_write(self.description, bytes(0x80))
        self.u.mem_write(self.vector, bytes(0x20))
        self.u.mem_write(self.descriptors, bytes(0x500))
        self.u.mem_write(self.key, b'score\0')
        self.q(self.description, self.field)
        self.q(self.description + 8, self.field_class)
        self.d(self.description + 0x18, self.type_enum)
        self.u.mem_write(self.description + 0x29, bytes([options.get('require_feature', False)]))
        self.d(self.description + 0x2C, options.get('flags', 0))
        self.q(self.description + 0x38, self.runtime)
        self.q(self.build_context + 0x20, self.provider)
        self.q(self.vector + 0x18, 1)
        if options.get('reserved'):
            self.q(self.vector, self.descriptors)
            self.q(self.vector + 0x18, 20)
        x = self.x
        sp = self.stack + 0x18008
        self.u.mem_write(self.stack, bytes(0x18000))
        self.q(sp, self.stop)
        preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI,
                     x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13,
                     x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(preserved):
            self.u.reg_write(register, 0xBAD00000 + i)
        for register, value in [(x.UC_X86_REG_RSP, sp),
                                (x.UC_X86_REG_RCX, self.build_context),
                                (x.UC_X86_REG_RDX, self.description),
                                (x.UC_X86_REG_R8, self.vector),
                                (x.UC_X86_REG_R9, 0)]:
            self.u.reg_write(register, value)
        try:
            self.u.emu_start(self.base + 0x77FF80, self.stop,
                            timeout=2_000_000, count=100000)
        except Exception as exc:
            raise AssertionError(f'source={source}, options={options}, '
                                 f'RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.descriptor_error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(preserved):
                assert self.reg(register) == 0xBAD00000 + i
        result = {'source': source, 'options': options.copy(),
                  'events': self.descriptor_events.copy(), 'final': self.snapshot(),
                  'returned': returned, 'error': self.descriptor_error}
        self.phase = None
        return result

    def copy_numeric(self, source, payload, options=None):
        options = options or {}
        built = self.construct(source, options)
        assert built['returned'] and built['final']['count'] == 1
        self.phase = None
        assert self.parse(payload)['tree'] is not None
        self.phase = 'batch'
        self.batch_events = []
        self.u.mem_write(self.managed, bytes([0xCD]) * 0x100)
        self.u.mem_write(self.cache, bytes(0x100))
        self.q(self.cache_slot, self.cache)
        self.d(self.cache + 0x10, 1)
        self.u.mem_write(self.cache + 0x18, b'\x09\x00')
        self.q(self.cache + 0x20, self.descriptors)
        self.q(self.cache + 0x30, 1)
        self.invoke(0xA8E030, [self.last_tree, self.managed, self.klass, self.cache_slot])
        assert self.rq(self.last_tree + 0x10) == 0
        assert self.batch_events == ['reference_store'] + ['vector_cleanup_service'] * 3
        raw = bytes(self.u.mem_read(self.managed, 0x100))
        offset, width = options.get('offset', 0x20), dict(SOURCES)[source]
        assert raw[:offset] == bytes([0xCD]) * offset
        assert raw[offset + width:] == bytes([0xCD]) * (0x100 - offset - width)
        self.phase = None
        return {'source': source, 'json_utf8_hex': payload.hex(),
                'options': options.copy(), 'construction': built,
                'value_hex': raw[offset:offset + width].hex(),
                'scope_links_cleared': True, 'events': self.batch_events.copy()}


def verify_native(m):
    import struct
    groups = {}
    for entry in m.pe.DIRECTORY_ENTRY_EXCEPTION:
        root, visited = entry, set()
        while root.unwindinfo.Flags & 4:
            assert root.struct.BeginAddress not in visited
            visited.add(root.struct.BeginAddress)
            root = root.unwindinfo._chained_entry
        groups.setdefault(root.struct.BeginAddress, []).append(
            (entry.struct.BeginAddress, entry.struct.EndAddress))
    instructions, families = {}, {}
    for root in [0x77FF80, 0x77FD00, 0x77F620, 0x77EE20, 0x76CA70]:
        assert root in groups
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            assert len(raw) == b - a
            decoded = list(m.cs.disasm(raw, a))
            assert sum(i.size for i in decoded) == b - a
            instructions.update({i.address: i for i in decoded})
    names = {}
    for slot, name, lea, store in EXPORTS:
        assert lea in instructions and store in instructions
        i, s = instructions[lea], instructions[store]
        assert i.mnemonic == 'lea' and i.op_str.startswith('rdx, [rip')
        assert s.mnemonic == 'mov' and s.op_str.startswith('qword ptr [rip')
        literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
        section = m.pe.get_section_by_rva(literal)
        assert section and literal - section.VirtualAddress < section.SizeOfRawData
        available = section.SizeOfRawData - (literal - section.VirtualAddress)
        assert m.pe.get_data(literal, min(128, available)).split(b'\0', 1)[0] == name.encode('ascii')
        assert s.address + s.size + struct.unpack('<i', bytes(s.bytes[-4:]))[0] == slot
        names[name] = hex(slot)
    checks = {
        0x77FFB5: ('call', '0x75f2b0'),
        0x77FFCC: ('call', '0x77fd00'),
        0x77FFF1: ('call', '0x77ee20'),
        0x78027C: ('call', '0x77f620'),
        0x780334: ('call', '0x1c77a0'),
        0x78033C: ('mov', 'qword ptr [r14 + 0x10], rbx'),
        0x7801C0: ('ret', ''),
        0x77FF7D: ('ret', ''),
    }
    for address, expected in checks.items():
        assert address in instructions
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
    return {'instruction_assertions': len(checks) + len(EXPORTS) * 4,
            'chained_unwind_families': families, 'metadata_export_bindings': names}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases = []
    joined = []
    for source, _ in SOURCES:
        for options in [{}, {'reserved': True}, {'value_type': False},
                        {'require_feature': True}, {'offset': 0x38, 'flags': 0x80000}]:
            result = m.construct(source, options)
            assert result['returned']
            assert result['final']['count'] == (not options.get('require_feature', False))
            if result['final']['count']:
                record = result['final']['records'][0]
                assert record['handler_rva'] == hex(m.handlers[source])
                assert record['field_identity'] and record['parent_identity']
                assert record['key_identity'] and record['class_identity']
                assert record['field_offset'] == options.get('offset', 0x20)
                assert record['value_type_byte'] == options.get('value_type', True)
                assert record['tails_zero']
            cases.append(result)
        for options in [{'disable_handler': True},
                        {'disable_handler': True, 'duplicate_later': True},
                        {'require_feature': True, 'feature_enabled': True}]:
            result = m.construct(source, options)
            assert result['returned']
            assert result['final']['count'] == options.get('feature_enabled', False)
            cases.append(result)
        for payload in [b'{}', b'{"score":123}', b'{"score":7,"score":11}']:
            expected = m.apply(source, payload)['value_hex']
            result = m.copy_numeric(source, payload)
            assert result['value_hex'] == expected
            joined.append(result)
        expected = m.apply(source, b'{"score":123}')['value_hex']
        result = m.copy_numeric(source, b'{"score":123}', {'offset': 0x38})
        assert result['value_hex'] == expected
        joined.append(result)
    for options in [{}, {'reserved': True}, {'require_feature': True}]:
        baseline = m.construct('core+0x120', options)
        counts = {}
        for i, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.construct('core+0x120', dict(options, failure=[kind, counts[kind]]))
            assert not result['returned'] and result['error'] == kind
            assert result['events'] == baseline['events'][:i + 1]
            assert result['final'] == event['snapshot']
            cases.append(result)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'native_verified': verified,
            'cases': cases, 'case_count': len(cases),
            'joined_numeric_copies': joined, 'joined_numeric_copy_count': len(joined),
            'executed_address_count': len(m.descriptor_executed),
            'scope': 'Actual numeric descriptor factory, fixed-buffer/lazy predicates on supported noncollection fixture types and native registry lookup, joined to native parser/adapter/numeric field application. Runtime metadata exports, collection predicate, descriptor reservation, GC store and vector cleanup are explicit services. Real field inclusion/discovery, enums and compound descriptors are unclaimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['executed_address_count'])
