"""Execute native field enumeration, eligibility and numeric descriptor building.

Runtime class/field exports, type classifiers and allocation are fixture services.
The builder, inheritance traversal, field filters and descriptor factory execute.
"""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_descriptors import Machine as DescriptorMachine
from audit_unity_json_descriptors import EXPORTS as DESCRIPTOR_EXPORTS
from audit_unity_json_descriptors import verify_native as verify_descriptors
from audit_unityplayer_wait import ENGINE_SHA256


EXPORTS = DESCRIPTOR_EXPORTS + [
    (0x1CD6310, 'il2cpp_class_is_subclass_of', 0x76CF2B, 0x76CF37),
    (0x1CD61D8, 'il2cpp_class_get_fields', 0x76D035, 0x76D041),
    (0x1CD62E8, 'il2cpp_class_get_parent', 0x76D1D7, 0x76D1E3),
    (0x1CD60A8, 'il2cpp_field_get_flags', 0x76D72F, 0x76D73B),
    (0x1CD6160, 'il2cpp_field_has_attribute', 0x76D839, 0x76D845),
    (0x1CD61E0, 'il2cpp_type_get_class_or_element_class', 0x76E7CF, 0x76E7DB),
]


class Machine(DescriptorMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.metadata_executed = set()
        self.parent_class = self.arena + 0x510000
        self.registry = self.arena + 0x520000
        self.q(self.registry, self.registry + 0x100)
        self.q(self.registry + 0x148, self.provider)
        self.q(self.base + 0x1CD66A8, self.registry)
        for i, (slot, name, _, _) in enumerate(EXPORTS):
            service = self.services + 0x600 + i * 0x10
            self.q(self.base + slot, service)
            self.export_services[service] = name

    def metadata_snapshot(self):
        count = self.rq(self.output_vector + 0x10) if self.output_vector else 0
        assert count <= 9
        pointer = self.rq(self.output_vector) if self.output_vector else 0
        rows = []
        for i in range(count):
            a = pointer + i * 0x80
            token = self.rq(a + 0x18)
            rows.append({'field': self.fields[token]['name'] if token in self.fields else 'unset',
                         'handler_rva': hex(self.rq(a + 8) - self.base) if self.rq(a + 8) else None,
                         'offset': self.rd(a + 0x3C), 'flags': self.rd(a + 0x44)})
        return {'descriptor_count': count, 'descriptors': rows,
                'enumerated_fields': self.enumerated.copy(),
                'field_values_hex': bytes(self.u.mem_read(self.managed + 0x20, 0x60)).hex(),
                'scope_linked': self.last_tree is not None and self.rq(self.last_tree + 0x10) != 0}

    def metadata_event(self, kind, args):
        self.metadata_events.append({'kind': kind, 'args': args,
                                     'snapshot': self.metadata_snapshot()})
        self.metadata_counts[kind] = self.metadata_counts.get(kind, 0) + 1
        if self.metadata_failure == [kind, self.metadata_counts[kind]]:
            self.metadata_error = kind
            self.u.emu_stop()
            return False
        return True

    def hook(self, uc, address, size, data):
        if self.phase != 'metadata':
            return super().hook(uc, address, size, data)
        rva = address - self.base
        self.metadata_executed.add(rva)
        self.executed.add(rva)
        x = self.x
        cx, dx, r8 = [self.reg(r) for r in
                       (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        name = self.export_services.get(address)
        if rva == 0x784120:
            self.output_vector = dx
        if name:
            if name == 'il2cpp_class_get_fields':
                assert cx in self.class_fields
                index = self.rq(dx)
                fields = self.class_fields[cx]
                token = fields[index] if index < len(fields) else 0
                label = self.fields[token]['name'] if token else None
                if self.metadata_event(name, [self.class_labels[cx], index, label]):
                    self.q(dx, index + 1)
                    if token:
                        self.enumerated.append(label)
                    self.ret(token)
                return
            if name == 'il2cpp_class_get_parent':
                assert cx in self.class_fields
                if self.metadata_event(name, [self.class_labels[cx]]):
                    self.ret(self.parent_map[cx])
                return
            if name == 'il2cpp_class_is_subclass_of':
                assert r8 & 0xFF == 1
                if self.metadata_event(name, ['fixture_classes']):
                    self.ret(bool(self.run_options.get('subclass_parent') and
                                  cx == self.parent_class and
                                  dx == self.rq(self.core + 0x138)))
                return
            if name == 'il2cpp_field_has_attribute':
                assert cx in self.fields
                field = self.fields[cx]
                if dx == self.rq(self.runtime + 0xCD0):
                    attribute = 'serialize_field'
                else:
                    assert dx == self.rq(self.runtime + 0xCD8)
                    attribute = 'serialize_reference'
                if self.metadata_event(name, [field['name'], attribute]):
                    self.ret(field.get(attribute, False))
                return
            if name.startswith('il2cpp_field_'):
                assert cx in self.fields
                field = self.fields[cx]
                values = {'il2cpp_field_get_name': field['key'],
                          'il2cpp_field_get_type': field['type'],
                          'il2cpp_field_get_parent': field['parent'],
                          'il2cpp_field_get_offset': field['offset'],
                          'il2cpp_field_get_flags': field.get('flags', 6)}
                if self.metadata_event(name, [field['name']]):
                    self.ret(values[name])
                return
            if name in ('il2cpp_class_from_type', 'il2cpp_type_get_type',
                        'il2cpp_type_get_class_or_element_class'):
                field = self.fields[self.type_fields[cx]]
                if self.metadata_event(name, [field['name']]):
                    self.ret(field.get('type_enum', 8) if name == 'il2cpp_type_get_type' else field['class'])
                return
            assert cx in self.sources
            if self.metadata_event(name, [self.sources[cx]]):
                self.ret({'il2cpp_class_get_name': self.class_name,
                          'il2cpp_class_is_valuetype': 1,
                          'il2cpp_class_is_enum': 0}[name])
            return
        if rva in (0x75CC70, 0x75F2B0):
            kind = 'excluded_type_service' if rva == 0x75CC70 else 'collection_predicate_service'
            if self.metadata_event(kind, ['fixture_type']):
                self.ret(bool(rva == 0x75CC70 and
                              self.run_options.get('excluded_parent') and
                              cx == self.parent_class))
            return
        if rva == 0x1C77A0:
            assert cx == self.output_vector
            if self.metadata_event('descriptor_reserve_service', []):
                self.q(cx, self.descriptors)
                self.q(cx + 0x18, 20)
                self.ret()
            return
        if address == self.services + 0x180:
            assert cx == 0 and r8 == self.managed
            if self.metadata_event('reference_store', []):
                self.q(dx, r8)
                self.ret()
            return
        if rva == 0x14E2D0:
            if self.metadata_event('vector_cleanup_service', []):
                self.ret()
            return
        # Bounded allocator, reallocator and free-query services are inherited.
        return super().hook(uc, address, size, data)

    def run(self, definitions, options=None):
        options = options or {}
        self.run_options = options
        self.phase = None
        assert self.parse(options.get('json', '{"score":123}').encode('utf-8'))['tree'] is not None
        self.fields, self.type_fields = {}, {}
        self.class_fields = {self.klass: [], self.parent_class: []}
        self.class_labels = {self.klass: 'current', self.parent_class: 'parent'}
        self.parent_map = {self.klass: self.parent_class if options.get('parent') else 0,
                           self.parent_class: 0}
        if options.get('stop_parent_offset'):
            self.parent_map[self.klass] = self.rq(self.runtime + options['stop_parent_offset'])
        assert len(definitions) <= 6
        for i, definition in enumerate(definitions):
            field = dict(definition)
            token, type_token, key = [self.arena + offset + i * 0x100
                                      for offset in (0x530000, 0x531000, 0x532000)]
            self.u.mem_write(key, field['name'].encode('utf-8') + b'\0')
            source = field.get('source', 'core+0x120')
            field.update({'type': type_token, 'key': key,
                          'class': next(t for t, label in self.sources.items() if label == source),
                          'offset': field.get('offset', 0x20 + i * 0x10),
                          'parent': self.parent_class if field.get('in_parent') else self.klass})
            self.fields[token] = field
            self.type_fields[type_token] = token
            self.class_fields[field['parent']].append(token)
        self.u.mem_write(self.managed, bytes([0xCD]) * 0x100)
        for field in self.fields.values():
            if 'initial_hex' in field:
                initial = bytes.fromhex(field['initial_hex'])
                assert 0 <= field['offset'] <= 0x100 - len(initial)
                self.u.mem_write(self.managed + field['offset'], initial)
        self.u.mem_write(self.vector, bytes(0x28))
        self.u.mem_write(self.descriptors, bytes(0x500))
        self.q(self.vector + 0x18, 1)
        self.q(self.build_context, self.klass)
        self.q(self.build_context + 8, self.klass)
        self.q(self.build_context + 0x10, self.runtime)
        self.d(self.build_context + 0x18, options.get('depth', 0))
        self.u.mem_write(self.build_context + 0x1C, options.get('direction', 9).to_bytes(2, 'little'))
        self.q(self.build_context + 0x20, self.provider)
        self.output_vector = self.vector
        self.metadata_events, self.metadata_counts, self.enumerated = [], {}, []
        self.metadata_failure, self.metadata_error = options.get('failure'), None
        self.phase = 'metadata'
        x = self.x
        sp = self.stack + 0x18008
        self.u.mem_write(self.stack, bytes(0x18000))
        self.q(sp, self.stop)
        self.q(sp + 0x28, 0)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI,
                     x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13,
                     x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(registers):
            self.u.reg_write(register, 0xDAB00000 + i)
        if options.get('joined'):
            self.u.mem_write(self.cache, bytes(0x100))
            self.q(self.cache_slot, self.cache)
            arguments = [self.last_tree, self.managed, self.klass, self.cache_slot]
            entry = 0xA8E030
            self.output_vector = None
        else:
            arguments = [self.build_context, self.vector, 0, 0]
            entry = 0x784120
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        for register, value in zip([x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                                    x.UC_X86_REG_R8, x.UC_X86_REG_R9], arguments):
            self.u.reg_write(register, value)
        try:
            self.u.emu_start(self.base + entry, self.stop,
                            timeout=getattr(self, 'field_timeout', 2_000_000),
                            count=getattr(self, 'field_instruction_limit', 200000))
        except Exception as exc:
            raise AssertionError(f'options={options}, '
                                 f'RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.metadata_error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(registers):
                assert self.reg(register) == 0xDAB00000 + i
        result = {'fields': definitions, 'options': options.copy(),
                  'events': self.metadata_events.copy(), 'final': self.metadata_snapshot(),
                  'returned': returned, 'error': self.metadata_error}
        self.phase = None
        return result


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
    for root in [0x784120, 0x7827A0, 0x783FE0, 0x76CA70, 0x81A880]:
        assert root in groups
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            assert len(raw) == b - a
            decoded = list(m.cs.disasm(raw, a))
            assert sum(i.size for i in decoded) == b - a
            instructions.update({i.address: i for i in decoded})
    bindings = {}
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
        bindings[name] = hex(slot)
    checks = {
        0x784194: ('call', '0x783fe0'),
        0x78422F: ('call', '0x784120'),
        0x784269: ('call', '0x14e830'),
        0x784372: ('call', '0x7827a0'),
        0x7843D1: ('call', '0x77ff80'),
        0x784435: ('call', '0x783fe0'),
        0x784604: ('ret', ''),
        0x7827D8: ('test', 'bl, 0x10'),
        0x7827E1: ('test', 'bl, 0x20'),
        0x7828D1: ('call', '0x17c9038'),
        0x7828DE: ('cmp', 'bl, 6'),
        0x7828EA: ('mov', 'rdx, qword ptr [rdi + 0xcd0]'),
        0x782901: ('mov', 'rdx, qword ptr [rdi + 0xcd8]'),
        0x78411C: ('ret', ''),
        0x820426: ('call', '0x75fc00'),
        0x820446: ('mov', 'qword ptr [rcx + 0xcd0], rax'),
        0x820454: ('call', '0x75fc00'),
        0x82046D: ('mov', 'qword ptr [rcx + 0xcd8], rax'),
    }
    for address, expected in checks.items():
        assert address in instructions
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
    for lea, expected in [(0x820404, b'SerializeField'),
                           (0x820432, b'SerializeReference')]:
        i = instructions[lea]
        assert i.mnemonic == 'lea' and i.op_str.startswith('r9, [rip')
        literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
        assert m.pe.get_data(literal, 64).split(b'\0', 1)[0] == expected
    return {'instruction_assertions': len(checks) + len(EXPORTS) * 4 + 4,
            'chained_unwind_families': families, 'metadata_export_bindings': bindings,
            'attribute_class_slots': {'SerializeField': 'runtime+0xcd0',
                                      'SerializeReference': 'runtime+0xcd8'}}


def audit(game_root):
    m = Machine(game_root)
    descriptor_verified = verify_descriptors(m)
    verified = verify_native(m)
    cases = []
    for flags in [0, 1, 6, 0x16, 0x26, 0x86, 0x106]:
        for attribute in [None, 'serialize_field', 'serialize_reference']:
            definition = {'name': 'score', 'flags': flags}
            if attribute:
                definition[attribute] = True
            result = m.run([definition])
            assert result['returned']
            included = not (flags & 0xB0) and ((flags & 7) == 6 or attribute is not None)
            assert result['final']['descriptor_count'] == included
            cases.append(result)
    for name in ['score', 'a.b', '.score', 'score.', '', 'Score']:
        result = m.run([{'name': name}])
        assert result['returned']
        assert result['final']['descriptor_count'] == ('.' not in name)
        cases.append(result)
    definitions = [{'name': 'score'}, {'name': 'hidden', 'flags': 1},
                   {'name': 'baseScore', 'in_parent': True},
                   {'name': 'enabled', 'source': 'core+0x130'}]
    for options in [{}, {'parent': True}, {'direction': 7}, {'depth': 12}, {'joined': True},
                    {'joined': True, 'parent': True,
                     'json': '{"score":7,"baseScore":11,"enabled":1}'}]:
        result = m.run(definitions, options)
        assert result['returned']
        names = [r['field'] for r in result['final']['descriptors']]
        assert names == (['baseScore'] if options.get('parent') else []) + ['score', 'enabled']
        if options.get('joined'):
            assert not result['final']['scope_linked']
            assert result['final']['field_values_hex'][:8] == ('07000000' if options.get('parent') else '7b000000')
        cases.append(result)
    for options in [{'stop_parent_offset': 0x520}, {'stop_parent_offset': 0xCB0},
                    {'stop_parent_offset': 0x548}, {'parent': True, 'excluded_parent': True},
                    {'parent': True, 'subclass_parent': True}]:
        result = m.run(definitions, options)
        assert result['returned']
        assert [r['field'] for r in result['final']['descriptors']] == ['score', 'enabled']
        assert 'baseScore' not in result['final']['enumerated_fields']
        cases.append(result)
    for options in [{}, {'joined': True}]:
        result = m.run([], options)
        assert result['returned'] and result['final']['descriptor_count'] == 0
        assert result['final']['field_values_hex'] == 'cd' * 0x60
        cases.append(result)
    for options in [{}, {'parent': True}, {'joined': True}]:
        baseline = m.run(definitions, options)
        counts = {}
        for i, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run(definitions, dict(options, failure=[kind, counts[kind]]))
            assert not result['returned'] and result['error'] == kind
            assert result['events'] == baseline['events'][:i + 1]
            assert result['final'] == event['snapshot']
            cases.append(result)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'descriptor_verified': descriptor_verified,
            'native_verified': verified,
            'cases': cases, 'case_count': len(cases),
            'executed_address_count': len(m.metadata_executed),
            'scope': 'Native metadata builder with field enumeration, eligibility, supported parent traversal, registry-selected numeric descriptor construction and joined adapter application. Runtime metadata exports, exclusion/collection predicates and storage services are supplied; no real class discovery or compound-type serialization claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['executed_address_count'])
