"""Execute native scalar List JSON copying with explicit runtime construction."""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_arrays import Machine as ArrayMachine
from audit_unityplayer_wait import ENGINE_SHA256


LIST_SOURCE = 'fixture_list'
EXPORTS = [
    (0x1CD5F48, 'il2cpp_object_new', 0x76E1DF, 0x76E1EB),
    (0x1CD62B0, 'il2cpp_runtime_object_init_exception', 0x76E3F3, 0x76E3FF),
]


class Machine(ArrayMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.list_class = self.arena + 0x830000
        self.items_field, self.size_field, self.items_type = [self.arena + a for a in
                                                            (0x831000, 0x832000, 0x833000)]
        self.sources[self.list_class] = LIST_SOURCE
        self.list_services = {}
        for i, (slot, name, _, _) in enumerate(EXPORTS):
            service = self.services + 0x800 + i * 0x10
            self.q(self.base + slot, service)
            self.list_services[service] = name
        self.list_executed = set()
        self.u.mem_write(self.base + 0x1CC6209, b'\0\0')
        self.reset_lists()

    def reset_lists(self):
        self.lists = {}
        self.list_cursor = self.arena + 0x840000
        self.backing_field_order = [self.size_field, self.items_field]

    def make_list(self, values, capacity=None, version=17):
        token = self.list_cursor
        self.list_cursor += 0x100
        assert self.list_cursor < self.arena + 0x850000
        self.u.mem_write(token, bytes([0xA5]) * 0x40)
        if values is None:
            backing, count = 0, 0
        else:
            count = len(values)
            capacity = count if capacity is None else capacity
            assert count <= capacity <= 256
            backing = self.make_array(values + [bytes([0x7E]) * self.element_width] * (capacity - count))
        self.q(token + 0x10, backing)
        self.d(token + 0x18, count)
        self.d(token + 0x1C, version)
        self.lists[token] = True
        return token

    def list_state(self, token):
        if token == 0:
            return None
        assert token in self.lists
        backing = self.rq(token + 0x10)
        count = self.rd(token + 0x18)
        raw = self.array_values(backing)
        assert count <= len(raw or [])
        return {'values_hex': raw[:count] if raw is not None else None,
                'count': count, 'capacity': self.arrays[backing]['count'] if backing else None,
                'version': self.rd(token + 0x1C), 'backing_values_hex': raw}

    def field_width(self, field):
        return 8 if field.get('source') == LIST_SOURCE else super().field_width(field)

    def metadata_snapshot(self):
        result = super().metadata_snapshot()
        token = self.rq(self.managed + 0x20)
        result['list_field'] = self.list_state(token) if token in self.lists or token == 0 else 'unset'
        result['list_storage'] = [self.list_state(token) for token in self.lists]
        return result

    def hook(self, uc, address, size, data):
        if self.phase not in ('metadata', 'writer'):
            return super().hook(uc, address, size, data)
        self.list_executed.add(address - self.base)
        x = self.x
        cx, dx, r8 = [self.reg(r) for r in
                     (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        name = self.list_services.get(address)
        if name == 'il2cpp_object_new':
            assert cx == self.list_class
            if self.metadata_event(name, [LIST_SOURCE]):
                self.ret(self.make_list(None, version=0))
            return
        if name == 'il2cpp_runtime_object_init_exception':
            assert cx in self.lists
            if self.metadata_event(name, [LIST_SOURCE]):
                self.q(cx + 0x10, self.make_array([]))
                self.d(cx + 0x18, 0)
                self.d(cx + 0x1C, 0)
                self.q(dx, 0)
                self.ret()
            return
        name = self.export_services.get(address)
        if name == 'il2cpp_class_get_fields' and cx == self.list_class:
            index = self.rq(dx)
            fields = self.backing_field_order
            token = fields[index] if index < len(fields) else 0
            label = '_items' if token == self.items_field else '_size' if token else None
            if self.metadata_event(name, [LIST_SOURCE, index, label]):
                self.q(dx, index + 1)
                self.ret(token)
            return
        if name == 'il2cpp_field_get_offset' and cx in (self.items_field, self.size_field):
            if self.metadata_event(name, ['_items' if cx == self.items_field else '_size']):
                self.ret(0x10 if cx == self.items_field else 0x18)
            return
        if name == 'il2cpp_field_get_type' and cx == self.items_field:
            if self.metadata_event(name, ['_items']):
                self.ret(self.items_type)
            return
        if name == 'il2cpp_type_get_class_or_element_class' and cx == self.items_type:
            if self.metadata_event(name, ['list_element']):
                self.ret(self.element_class)
            return
        if name == 'il2cpp_class_is_valuetype' and cx == self.list_class:
            if self.metadata_event(name, [LIST_SOURCE]):
                self.ret(0)
            return
        if address - self.base == 0x75F2B0 and cx == self.list_class:
            if self.metadata_event('collection_predicate_service', [LIST_SOURCE]):
                self.ret(1)
            return
        if address == self.services + 0x180 and r8 in self.lists:
            assert cx == 0
            if self.metadata_event('reference_store', ['list']):
                self.q(dx, r8)
                self.ret()
            return
        return super().hook(uc, address, size, data)

    def read_list(self, payload, old=None, capacity=None, options=None):
        options = options or {}
        self.reset_arrays()
        self.reset_lists()
        if options.get('items_first'):
            self.backing_field_order.reverse()
        token = self.make_list(old, capacity) if old is not None else 0
        old_backing = self.rq(token + 0x10) if token else 0
        field = {'name': 'score', 'source': LIST_SOURCE, 'type_enum': 0x15,
                 'initial_hex': token.to_bytes(8, 'little').hex()}
        self.registry_build(False)
        result = self.run([field], dict(options, joined=True, json=payload))
        final = self.rq(self.managed + 0x20)
        if result['returned']:
            for a in self.lists:
                assert bytes(self.u.mem_read(a, 0x10)) == bytes([0xA5]) * 0x10
                assert bytes(self.u.mem_read(a + 0x20, 0x20)) == bytes([0xA5]) * 0x20
        result.update({'old_values_hex': [v.hex() for v in old] if old is not None else None,
                       'old_capacity': capacity, 'loaded_list': self.list_state(final),
                       'old_list_retained': final == token,
                       'old_backing_retained': bool(final) and self.rq(final + 0x10) == old_backing,
                       'element_source': self.element_source})
        return result

    def write_list(self, values, capacity=None, options=None):
        self.reset_arrays()
        self.reset_lists()
        token = self.make_list(values, capacity) if values is not None else 0
        before = bytes(self.u.mem_read(token, 0x40)) if token else None
        backing = self.rq(token + 0x10) if token else 0
        backing_before = bytes(self.u.mem_read(backing, 0x40 + self.arrays[backing]['count'] * self.element_width)) if backing else None
        field = {'name': 'score', 'source': LIST_SOURCE, 'type_enum': 0x15}
        result = self.serialize([field], [token.to_bytes(8, 'little')], options)
        assert not token or bytes(self.u.mem_read(token, 0x40)) == before
        assert not backing or bytes(self.u.mem_read(backing, len(backing_before))) == backing_before
        result.update({'list_input_retained': True,
                       'input_elements_hex': [v.hex() for v in values] if values is not None else None,
                       'input_capacity': capacity, 'element_source': self.element_source})
        if result['returned']:
            read = self.read_list(bytes.fromhex(result['json_utf8_hex']).decode('utf-8'))
            assert read['returned'] and not read['final']['scope_linked']
            result['loaded_list'] = read['loaded_list']
        return result


def verify_native(m):
    import struct
    m.cs.detail = True
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
    for root in [0x75F310, 0x784680, 0x75F5E0, 0x784860, 0x78F420, 0x7906A0, 0x76CA70]:
        assert root in groups
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            assert len(raw) == b - a
            decoded = list(m.cs.disasm(raw, a))
            assert sum(i.size for i in decoded) == b - a
            instructions.update({i.address: i for i in decoded})
    checks = {
        0x75F322: ('call', '0x75f2b0'),
        0x75F38D: ('call', '0x14e830'),
        0x75F3EF: ('cmp', 'rax, 0x10'),
        0x75F5DA: ('ret', ''),
        0x784988: ('call', '0x75f310'),
        0x7849A1: ('call', '0x784680'),
        0x784717: ('mov', 'rax, qword ptr [rip + 0x155182a]'),
        0x784721: ('call', 'rax'),
        0x784780: ('call', '0x75f5e0'),
        0x784794: ('call', '0x75f5e0'),
        0x7847E0: ('ret', ''),
        0x75F600: ('call', 'qword ptr [rip + 0x1576caa]'),
        0x75F6A4: ('ret', ''),
    }
    for address, expected in checks.items():
        assert address in instructions
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
    slots = {0x784923: 0x1CC620A, 0x784748: 0x1CC6209,
             0x784717: 0x1CD5F48, 0x75F600: 0x1CD62B0}
    for address, expected in slots.items():
        i = instructions[address]
        assert i.address + i.size + i.disp == expected, (hex(address), hex(i.address + i.size + i.disp))
    for slot, name, lea, store in EXPORTS:
        i, s = instructions[lea], instructions[store]
        assert i.mnemonic == 'lea' and i.op_str.startswith('rdx, [rip')
        literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
        assert m.pe.get_data(literal, 80).split(b'\0', 1)[0] == name.encode('ascii')
        assert s.mnemonic == 'mov' and s.op_str.startswith('qword ptr [rip')
        assert s.address + s.size + struct.unpack('<i', bytes(s.bytes[-4:]))[0] == slot
    return {'instruction_assertions': len(checks) + len(slots) + len(EXPORTS) * 4,
            'chained_unwind_families': families,
            'runtime_construction_exports': {name: hex(slot) for slot, name, _, _ in EXPORTS},
            'backing_lookup_cache_flags': {'0x1cc6209': 0, '0x1cc620a': 0}}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    writes, reads, strings, failures, baselines = [], [], [], [], {}
    patterns = [('core+0x120', 4, 8, [0, 123, 0xFFFFFFFF]),
                ('core+0x130', 1, 2, [0, 1, 0xFF]),
                ('core+0x188', 4, 12, [0x3FC00000, 0x80000000]),
                ('core+0x118', 2, 6, [0x7FFF, 0x8000]),
                ('core+0x128', 8, 10, [0x7FFFFFFFFFFFFFFF, 0x8000000000000000]),
                ('core+0x178', 2, 7, [0, 0xFFFF]),
                ('core+0x168', 1, 4, [0x7F, 0x80]),
                ('core+0x108', 4, 9, [0, 0xFFFFFFFF]),
                ('core+0x110', 8, 11, [0, 0xFFFFFFFFFFFFFFFF]),
                ('core+0x170', 1, 5, [0, 0xFF]),
                ('core+0x100', 2, 3, [0, 0xFFFF]),
                ('core+0x198', 8, 13, [0x3FF8000000000000, 0x8000000000000000, 1])]
    for source, width, enum, values in patterns:
        m.select_element(source, width, enum)
        raw = [v.to_bytes(width, 'little') for v in values]
        expected = [v.hex() for v in raw]
        if source == 'core+0x130':
            expected[2] = '01'
        if source == 'core+0x188':
            expected[1] = '00000000'
        if source == 'core+0x198':
            expected[1] = expected[2] = '0000000000000000'
        for pretty in [False, True]:
            result = m.write_list(raw, len(raw) + 2, {'pretty': pretty})
            assert result['returned'] and result['loaded_list']['values_hex'] == expected
            assert result['loaded_list']['capacity'] == len(raw)
            writes.append(result)
    m.select_element('core+0x120', 4, 8)
    for pretty in [False, True]:
        for values in [None, []]:
            result = m.write_list(values, 4, {'pretty': pretty})
            assert result['returned'] and result['loaded_list']['values_hex'] == []
            writes.append(result)
    payloads = [('{}', None), ('{"Score":[9]}', None), ('{"score":null}', []),
                ('{"score":[]}', []), ('{"score":123}', []),
                ('{"score":[1,2]}', ['01000000', '02000000']),
                ('{"score":[7],"score":[8]}', ['07000000'])]
    inventories = [(None, None), ([], 4), ([b'\x7b\0\0\0'], 4),
                   ([b'\x7b\0\0\0'] * 2, 4)]
    for payload, expected in payloads:
        for old, capacity in inventories:
            for items_first in [False, True]:
                result = m.read_list(payload, old, capacity, {'items_first': items_first})
                loaded = result['loaded_list']
                expected_values = ([v.hex() for v in old] if old is not None else []) if expected is None else expected
                assert result['returned'] and loaded['values_hex'] == expected_values
                assert result['old_list_retained'] == (old is not None)
                retained = old is not None and (expected is None or len(old) == len(expected))
                assert result['old_backing_retained'] == retained
                assert loaded['version'] == (17 if old is not None else 0)
                assert loaded['capacity'] == (capacity if retained else len(expected_values))
                if retained and capacity > loaded['count']:
                    assert loaded['backing_values_hex'][loaded['count']:] == ['7e' * 4] * (capacity - loaded['count'])
                assert not result['final']['scope_linked']
                reads.append(result)
    m.select_element('core+0x180', 8, 14)
    for pretty in [False, True]:
        m.reset_strings()
        texts = ['caf\u00e9 \U0001f608', 'a\0b', None]
        values = [(m.make_string(text) if text is not None else 0).to_bytes(8, 'little') for text in texts]
        result = m.write_list(values, 5, {'pretty': pretty})
        loaded = [m.string_text(int.from_bytes(bytes.fromhex(h), 'little')) for h in result['loaded_list']['values_hex']]
        assert loaded == ['caf\u00e9 \U0001f608', 'a', '']
        result.update({'input_texts': texts, 'loaded_texts': loaded})
        strings.append(result)
    m.select_element('core+0x120', 4, 8)
    raw = [b'\x01\0\0\0', b'\x02\0\0\0']
    for direction in ['read_new', 'read_reuse', 'read_resize', 'write_null', 'write_existing']:
        if direction.startswith('write'):
            operation = lambda options: m.write_list(None if direction == 'write_null' else raw, 4, options)
        else:
            old = None if direction == 'read_new' else [b'\x7b\0\0\0'] * (2 if direction == 'read_reuse' else 1)
            operation = lambda options: m.read_list('{"score":[1,2]}', old, 4, options)
        baseline = operation({})
        assert baseline['returned']
        baselines[direction] = baseline
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = operation({'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['error'] == kind
            assert result['events'] == baseline['events'][:index + 1]
            if not direction.startswith('write'):
                assert result['final'] == event['snapshot']
            # Store one baseline stream per direction; the checked failure prefix
            # is its first N events, not a duplicated quadratic report payload.
            failure = {'direction': direction, 'failure': [kind, counts[kind]],
                       'baseline_event_count': index + 1, 'returned': False,
                       'error': result['error']}
            if not direction.startswith('write'):
                failure.update({'final': result['final'], 'loaded_list': result['loaded_list'],
                                'old_list_retained': result['old_list_retained'],
                                'old_backing_retained': result['old_backing_retained']})
            failures.append(failure)
    required = [0x75F310, 0x784680, 0x75F5E0, 0x784860, 0x78F420, 0x7906A0]
    assert all(rva in m.list_executed for rva in required)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256, 'native_verified': verified,
            'write_cases': writes, 'write_case_count': len(writes),
            'read_cases': reads, 'read_case_count': len(reads),
            'string_cases': strings, 'string_case_count': len(strings),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.list_executed),
            'required_native_entries_executed': [hex(rva) for rva in required],
            'scope': 'Native List backing-field discovery, collection classification paths, constructor wrapper, scalar/string processors and JSON save/reload. Runtime List classification, metadata field/type exports, object allocation/constructor execution, array allocation and GC/cache/allocator services are explicit fixtures. Backing lookup cache flags are disabled; actual managed constructors, compound elements, nested containers and real runtime metadata discovery remain unclaimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['write_case_count'], result['read_case_count'], result['string_case_count'], result['failure_case_count'], result['executed_address_count'])
