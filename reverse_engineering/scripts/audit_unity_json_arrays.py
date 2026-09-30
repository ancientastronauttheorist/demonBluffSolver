"""Execute native one-dimensional scalar JSON arrays over supplied metadata."""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_strings import Machine as StringMachine
from audit_unityplayer_wait import ENGINE_SHA256


ARRAY_SOURCE = 'fixture_array'
EXPORTS = [
    (0x1CD6168, 'il2cpp_class_get_type', 0x76D39F, 0x76D3AB),
    (0x1CD6148, 'il2cpp_array_length', 0x76CD17, 0x76CD23),
    (0x1CD6088, 'il2cpp_array_new', 0x76CD63, 0x76CD6F),
    (0x1CD6150, 'il2cpp_class_array_element_size', 0x76D353, 0x76D35F),
]


class Machine(StringMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.array_class = self.arena + 0x810000
        self.element_type = self.arena + 0x820000
        self.sources[self.array_class] = ARRAY_SOURCE
        self.array_services = {}
        for i, (slot, name, _, _) in enumerate(EXPORTS):
            service = self.services + 0x7C0 + i * 0x10
            self.q(self.base + slot, service)
            self.array_services[service] = name
        self.array_executed = set()
        self.reset_arrays()
        self.select_element('core+0x120', 4, 8)

    def select_element(self, source, width, type_enum):
        self.element_source, self.element_width, self.element_enum = source, width, type_enum
        self.element_class = next(t for t, label in self.sources.items() if label == source)

    def reset_arrays(self):
        self.arrays = {}
        self.array_cursor = self.arena + 0x800000

    def make_array(self, values):
        token = self.array_cursor
        assert len(values) <= 256 and all(len(v) == self.element_width for v in values)
        raw = b''.join(values)
        self.array_cursor += (len(raw) + 0x60 + 15) & ~15
        assert self.array_cursor < self.array_class
        self.u.mem_write(token, bytes([0xA5]) * (0x40 + len(raw)))
        self.u.mem_write(token + 0x20, raw)
        self.arrays[token] = {'count': len(values), 'width': self.element_width}
        return token

    def array_values(self, token):
        if token == 0:
            return None
        fixture = self.arrays[token]
        return [bytes(self.u.mem_read(token + 0x20 + i * fixture['width'], fixture['width'])).hex()
                for i in range(fixture['count'])]

    def field_width(self, field):
        return 8 if field.get('source') == ARRAY_SOURCE else super().field_width(field)

    def metadata_snapshot(self):
        result = super().metadata_snapshot()
        token = self.rq(self.managed + 0x20)
        if token in self.arrays or token == 0:
            result['array_values_hex'] = self.array_values(token)
        else:
            result['array_values_hex'] = 'unset'
        result['array_storage'] = [{'count': fixture['count'], 'width': fixture['width'],
                                    'values_hex': self.array_values(token)}
                                   for token, fixture in self.arrays.items()]
        return result

    def hook(self, uc, address, size, data):
        if self.phase not in ('metadata', 'writer'):
            return super().hook(uc, address, size, data)
        self.array_executed.add(address - self.base)
        x = self.x
        cx, dx, r8 = [self.reg(r) for r in
                     (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        name = self.array_services.get(address)
        if name:
            if name == 'il2cpp_class_get_type':
                assert cx == self.element_class
                if self.metadata_event(name, [self.element_source]):
                    self.ret(self.element_type)
            elif name == 'il2cpp_array_length':
                assert cx in self.arrays
                if self.metadata_event(name, [self.arrays[cx]['count']]):
                    self.ret(self.arrays[cx]['count'])
            elif name == 'il2cpp_class_array_element_size':
                assert cx == self.element_class
                if self.metadata_event(name, [self.element_source]):
                    self.ret(self.element_width)
            elif name == 'il2cpp_array_new':
                assert cx == self.element_class and dx <= 256
                if self.metadata_event(name, [self.element_source, dx]):
                    self.ret(self.make_array([bytes(self.element_width)] * dx))
            return
        name = self.export_services.get(address)
        if cx == self.element_type and name == 'il2cpp_type_get_type':
            if self.metadata_event(name, ['array_element']):
                self.ret(self.element_enum)
            return
        if name == 'il2cpp_type_get_class_or_element_class':
            assert cx in self.type_fields
            if self.metadata_event(name, ['array_element']):
                self.ret(self.element_class)
            return
        if name == 'il2cpp_class_is_valuetype' and cx == self.array_class:
            if self.metadata_event(name, [ARRAY_SOURCE]):
                self.ret(0)
            return
        if address == self.services + 0x180:
            assert cx == 0 and (r8 == 0 or r8 == self.managed or r8 in self.arrays or r8 in self.strings)
            label = 'null' if not r8 else 'array' if r8 in self.arrays else 'string' if r8 in self.strings else 'object'
            if self.metadata_event('reference_store', [label]):
                self.q(dx, r8)
                self.ret()
            return
        return super().hook(uc, address, size, data)

    def read_array(self, payload, old=None, options=None):
        self.reset_arrays()
        token = self.make_array(old) if old is not None else 0
        field = {'name': 'score', 'source': ARRAY_SOURCE, 'type_enum': 0x1D,
                 'initial_hex': token.to_bytes(8, 'little').hex()}
        self.registry_build(False)
        result = self.run([field], dict(options or {}, joined=True, json=payload))
        final = self.rq(self.managed + 0x20)
        if result['returned']:
            for a, fixture in self.arrays.items():
                assert bytes(self.u.mem_read(a, 0x20)) == bytes([0xA5]) * 0x20
                end = a + 0x20 + fixture['count'] * fixture['width']
                assert bytes(self.u.mem_read(end, 0x20)) == bytes([0xA5]) * 0x20
        result.update({'element_source': self.element_source, 'element_width': self.element_width,
                       'element_type_enum': self.element_enum,
                       'old_values_hex': [v.hex() for v in old] if old is not None else None,
                       'loaded_values_hex': self.array_values(final),
                       'old_pointer_retained': final == token})
        return result

    def write_array(self, values, options=None):
        self.reset_arrays()
        token = self.make_array(values) if values is not None else 0
        before = bytes(self.u.mem_read(token, 0x40 + len(values) * self.element_width)) if token else None
        field = {'name': 'score', 'source': ARRAY_SOURCE, 'type_enum': 0x1D}
        result = self.serialize([field], [token.to_bytes(8, 'little')], options)
        assert not token or bytes(self.u.mem_read(token, len(before))) == before
        result['array_input_retained'] = True
        result.update({'element_source': self.element_source, 'element_width': self.element_width,
                       'element_type_enum': self.element_enum,
                       'input_elements_hex': [v.hex() for v in values] if values is not None else None})
        if result['returned']:
            rendered = bytes.fromhex(result['json_utf8_hex']).decode('utf-8')
            read = self.read_array(rendered)
            assert read['returned'] and not read['final']['scope_linked']
            result['loaded_values_hex'] = read['loaded_values_hex']
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
    for root in [0x77F870, 0x784860, 0x78F420, 0x7906A0, 0xA949E0,
                 0xA94AC0, 0xA98E50, 0xA92E80, 0xA9B9D0, 0xA9E530, 0x76CA70]:
        assert root in groups
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            assert len(raw) == b - a
            decoded = list(m.cs.disasm(raw, a))
            assert sum(i.size for i in decoded) == b - a
            instructions.update({i.address: i for i in decoded})
    leaves = {0xA91810: (0x14, [('mov', 'rax, rdx'), ('lea', 'r8, [rcx + 8]'),
                               ('mov', 'rdx, qword ptr [rdx + 0x30]'),
                               ('mov', 'rcx, qword ptr [rax + 0x28]'), ('jmp', '0xa949e0')]),
              0xA91B60: (9, [('add', 'rcx, 8'), ('jmp', '0xa9b9d0')]),
              0xA93250: (9, [('add', 'rcx, 8'), ('jmp', '0xa9e530')])}
    for a, (length, expected) in leaves.items():
        decoded = list(m.cs.disasm(m.pe.get_data(a, length), a))
        assert sum(i.size for i in decoded) == length
        assert [(i.mnemonic, i.op_str) for i in decoded] == expected
    checks = {
        0x77F8CA: ('call', '0x784860'),
        0x77FA50: ('call', '0x77f620'),
        0x77FCFD: ('ret', ''),
        0x7848E8: ('call', 'qword ptr [rip + 0x1551d9a]'),
        0x784908: ('call', 'rax'),
        0x784A50: ('ret', ''),
        0x78F446: ('call', '0x784860'),
        0x78F4ED: ('call', 'rbx'),
        0x78F53A: ('call', 'qword ptr [rsi + 0x50]'),
        0x78F5D4: ('ret', ''),
        0x7906BF: ('call', '0x784860'),
        0x79076A: ('call', 'rbx'),
        0x790817: ('call', 'qword ptr [rsi + 0x50]'),
        0x790825: ('ret', ''),
        0xA94A22: ('call', '0xa94ac0'),
        0xA94A3E: ('call', 'qword ptr [rip + 0x124170c]'),
        0xA94A52: ('call', 'rax'),
        0xA94A9F: ('call', '0x17c9af0'),
        0xA94ABD: ('ret', ''),
        0xA94B1C: ('call', '0xaac540'),
        0xA94B42: ('call', '0xa00ec0'),
        0xA94B6C: ('call', '0x3f2490'),
        0xA94BAE: ('call', '0x9e4150'),
        0xA94BEE: ('ret', ''),
        0xA92ECF: ('call', '0xa98e50'),
        0xA92ED9: ('call', '0x14e2d0'),
        0xA92EE2: ('ret', ''),
        0xA98E95: ('call', '0x20c5d0'),
        0xA98FBA: ('call', '0x1096320'),
        0xA99008: ('call', '0x1096320'),
        0xA9902B: ('ret', ''),
    }
    for address, expected in checks.items():
        assert address in instructions
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
    for slot, name, lea, store in EXPORTS:
        i, s = instructions[lea], instructions[store]
        assert i.mnemonic == 'lea' and i.op_str.startswith('rdx, [rip')
        literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
        assert m.pe.get_data(literal, 80).split(b'\0', 1)[0] == name.encode('ascii')
        assert s.mnemonic == 'mov' and s.op_str.startswith('qword ptr [rip')
        assert s.address + s.size + struct.unpack('<i', bytes(s.bytes[-4:]))[0] == slot
    for writer, expected in [(False, 0x78F420), (True, 0x7906A0)]:
        m.registry_build(writer)
        assert m.rq(m.provider + 0x28) == m.base + expected
    return {'instruction_assertions': len(checks) + 9 + 16 + 2,
            'chained_unwind_families': families,
            'array_export_bindings': {name: hex(slot) for slot, name, _, _ in EXPORTS},
            'native_collection_processors': {'reader': '0x78f420', 'writer': '0x7906a0'}}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    writes, reads, string_cases, aliases, failures = [], [], [], [], []
    patterns = [
        ('core+0x120', 4, 8, [0, 123, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF]),
        ('core+0x130', 1, 2, [0, 1, 0xFF]),
        ('core+0x188', 4, 12, [0, 0x80000000, 0x3FC00000, 1, 0x7FC00000, 0x7F800000, 0xFF800000]),
        ('core+0x118', 2, 6, [0, 0x7FFF, 0x8000, 0xFFFF]),
        ('core+0x128', 8, 10, [0, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF]),
        ('core+0x178', 2, 7, [0, 123, 0xFFFF]),
        ('core+0x168', 1, 4, [0, 0x7F, 0x80, 0xFF]),
        ('core+0x108', 4, 9, [0, 0x80000000, 0xFFFFFFFF]),
        ('core+0x110', 8, 11, [0, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF]),
        ('core+0x170', 1, 5, [0, 1, 0xFF]),
        ('core+0x100', 2, 3, [0, 123, 0xFFFF]),
        ('core+0x198', 8, 13, [0, 0x8000000000000000, 0x3FF8000000000000, 1,
                              0x7FF8000000000000, 0x7FF0000000000000, 0xFFF0000000000000]),
    ]
    for source, width, enum, values in patterns:
        m.select_element(source, width, enum)
        for pretty in [False, True]:
            for inventory in [None, [], [v.to_bytes(width, 'little') for v in values]]:
                result = m.write_array(inventory, {'pretty': pretty})
                expected = [v.hex() for v in inventory or []]
                if inventory:
                    if source == 'core+0x130':
                        expected[-1] = '01'
                    if source == 'core+0x188':
                        expected[1] = '00000000'
                    if source == 'core+0x198':
                        expected[1] = expected[3] = '0000000000000000'
                assert result['returned'] and result['loaded_values_hex'] == expected
                writes.append(result)
    m.select_element('core+0x120', 4, 8)
    payloads = [('{}', None), ('{"Score":[9]}', None), ('{"score":null}', []),
                ('{"score":[]}', []), ('{"score":123}', []), ('{"score":true}', []),
                ('{"score":{}}', []), ('{"score":"12"}', []),
                ('{"score":[1,2,3]}', ['01000000', '02000000', '03000000']),
                ('{"score":[true,null,"12",1.5,{},[]]}',
                 ['00000000', '00000000', '0c000000', '01000000', '00000000', '00000000']),
                ('{"score":[7],"score":[8]}', ['07000000'])]
    for payload, expected in payloads:
        for old in [None, [], [b'\x7b\0\0\0'], [b'\x7b\0\0\0'] * 3]:
            result = m.read_array(payload, old)
            expected_values = [v.hex() for v in old] if expected is None and old is not None else expected
            assert result['returned'] and result['loaded_values_hex'] == expected_values
            retained = expected is None or old is not None and len(old) == len(expected)
            assert result['old_pointer_retained'] == retained
            assert not result['final']['scope_linked']
            reads.append(result)
    m.select_element('core+0x180', 8, 14)
    for pretty in [False, True]:
        m.reset_strings()
        texts = ['hello', 'caf\u00e9 \U0001f608', 'a\0b', '', None]
        values = [(m.make_string(text) if text is not None else 0).to_bytes(8, 'little') for text in texts]
        result = m.write_array(values, {'pretty': pretty})
        loaded = [m.string_text(int.from_bytes(bytes.fromhex(h), 'little')) for h in result['loaded_values_hex']]
        assert loaded == ['hello', 'caf\u00e9 \U0001f608', 'a', '', '']
        result.update({'input_texts': texts, 'loaded_texts': loaded})
        string_cases.append(result)
    m.select_element('core+0x120', 4, 8)
    for pretty in [False, True]:
        m.reset_arrays()
        token = m.make_array([b'\x01\0\0\0', b'\x02\0\0\0'])
        fields = [{'name': name, 'source': ARRAY_SOURCE, 'type_enum': 0x1D,
                   'initial_hex': '00' * 8} for name in ['left', 'right']]
        values = [token.to_bytes(8, 'little')] * 2
        result = m.serialize(fields, values, {'pretty': pretty})
        assert result['returned']
        m.reset_arrays()
        m.registry_build(False)
        read = m.run(fields, {'joined': True,
                             'json': bytes.fromhex(result['json_utf8_hex']).decode('utf-8')})
        assert read['returned'] and not read['final']['scope_linked']
        left, right = m.rq(m.managed + 0x20), m.rq(m.managed + 0x30)
        assert left != right and m.array_values(left) == m.array_values(right) == ['01000000', '02000000']
        result.update({'input_array_aliased': True, 'loaded_arrays_aliased': False,
                       'loaded_left_hex': m.array_values(left), 'loaded_right_hex': m.array_values(right)})
        aliases.append(result)
    # Failure prefixes include writes into reused arrays and unpublished new arrays.
    m.select_element('core+0x120', 4, 8)
    for direction in ['read_new', 'read_reuse', 'write']:
        operation = ((lambda options: m.write_array([b'\x01\0\0\0', b'\x02\0\0\0'], options))
                     if direction == 'write' else
                     (lambda options: m.read_array('{"score":[1,2]}',
                                                  [b'\x7b\0\0\0'] * 2 if direction == 'read_reuse' else None,
                                                  options)))
        baseline = operation({})
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = operation({'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['error'] == kind
            assert result['events'] == baseline['events'][:index + 1]
            if direction != 'write':
                assert result['final'] == event['snapshot']
            result['direction'] = direction
            failures.append(result)
    required = [0x77F870, 0x784860, 0x78F420, 0x7906A0, 0xA949E0,
                0xA94AC0, 0xA98E50, 0xA91B60, 0xA93250, 0xA9B9D0, 0xA9E530]
    assert all(rva in m.array_executed for rva in required)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'native_verified': verified,
            'write_cases': writes, 'write_case_count': len(writes),
            'read_cases': reads, 'read_case_count': len(reads),
            'string_cases': string_cases, 'string_case_count': len(string_cases),
            'alias_cases': aliases, 'alias_case_count': len(aliases),
            'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.array_executed),
            'required_native_entries_executed': [hex(rva) for rva in required],
            'scope': 'Native one-dimensional array metadata/factory/reader/writer and scalar/string element conversion joined to native JSON save/load. Metadata, array allocation/length/element-size, GC, cache and allocator services are explicit fixtures. Lists, nested arrays, compound/reference element graphs and runtime metadata discovery remain unclaimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['write_case_count'], result['read_case_count'], result['string_case_count'], result['failure_case_count'], result['executed_address_count'])
