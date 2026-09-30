"""Execute the native List classifier and join it to JSON collection copying."""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_lists import Machine as ListMachine
from audit_unityplayer_wait import ENGINE_SHA256


EXPORTS = [
    (0x1CD5FF8, 'il2cpp_class_get_image', 0x76D45D, 0x76D469),
    (0x1CD5FB0, 'il2cpp_get_corlib', 0x76CC33, 0x76CC3F),
]


class Machine(ListMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.classifier_executed = set()
        self.classifier_name = self.arena + 0x860000
        self.corlib_fixture, self.other_image = self.arena + 0x861000, self.arena + 0x862000
        self.classifier_services = {}
        self.set_classifier('List`1', True)
        for i, (slot, name, _, _) in enumerate(EXPORTS):
            service = self.services + 0x820 + i * 0x10
            self.q(self.base + slot, service)
            self.classifier_services[service] = name

    def set_classifier(self, name, corlib):
        raw = name.encode('utf-8')
        assert len(raw) < 64
        self.u.mem_write(self.classifier_name, bytes(64))
        self.u.mem_write(self.classifier_name, raw + b'\0')
        self.classifier_text, self.classifier_corlib = name, corlib

    def hook(self, uc, address, size, data):
        if self.phase not in ('metadata', 'writer'):
            return super().hook(uc, address, size, data)
        rva = address - self.base
        self.classifier_executed.add(rva)
        if rva == 0x75F2B0:
            # Execute this entry instead of inherited classifier fixture hooks.
            return
        cx = self.reg(self.x.UC_X86_REG_RCX)
        name = self.classifier_services.get(address)
        if name:
            if name == 'il2cpp_class_get_image':
                assert cx == self.list_class
            if self.metadata_event(name, ['fixture_images']):
                self.ret(self.corlib_fixture if name == 'il2cpp_get_corlib' or self.classifier_corlib else self.other_image)
            return
        if self.export_services.get(address) == 'il2cpp_class_get_name' and cx == self.list_class:
            if self.metadata_event('il2cpp_class_get_name', ['fixture_list']):
                self.ret(self.classifier_name)
            return
        return super().hook(uc, address, size, data)

    def classify(self, name, corlib):
        self.set_classifier(name, corlib)
        self.reset_arrays()
        self.reset_lists()
        assert self.run([])['returned']
        self.metadata_events, self.metadata_counts = [], {}
        self.metadata_failure, self.metadata_error = None, None
        self.phase = 'metadata'
        self.invoke(0x75F2B0, [self.list_class])
        raw = self.reg(self.x.UC_X86_REG_RAX)
        self.phase = None
        return {'name': name, 'image_is_corlib': corlib, 'return_low_byte': raw & 0xFF,
                'nonzero_return_upper_bits': bool(raw >> 8), 'events': self.metadata_events.copy()}


def verify_native(m):
    import struct
    instructions, families = {}, {}
    groups = {}
    for entry in m.pe.DIRECTORY_ENTRY_EXCEPTION:
        root, visited = entry, set()
        while root.unwindinfo.Flags & 4:
            assert root.struct.BeginAddress not in visited
            visited.add(root.struct.BeginAddress)
            root = root.unwindinfo._chained_entry
        groups.setdefault(root.struct.BeginAddress, []).append((entry.struct.BeginAddress, entry.struct.EndAddress))
    for root in [0x75F2B0, 0x76CA70]:
        assert root in groups
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            decoded = list(m.cs.disasm(raw, a))
            assert sum(i.size for i in decoded) == b - a
            instructions.update({i.address: i for i in decoded})
    checks = {0x75F2C0: ('call', 'rax'), 0x75F2D7: ('cmp', 'dl, byte ptr [r8 + rcx - 1]'),
              0x75F2DE: ('cmp', 'rcx, 7'), 0x75F2EE: ('call', 'rax'),
              0x75F2F3: ('call', 'qword ptr [rip + 0x1576cb7]'),
              0x75F2F9: ('cmp', 'rbx, rax'), 0x75F2FE: ('mov', 'al, 1'),
              0x75F305: ('ret', ''), 0x75F306: ('xor', 'al, al'), 0x75F30D: ('ret', '')}
    for address, expected in checks.items():
        assert address in instructions
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
    i = instructions[0x75F2C4]
    literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
    assert literal == 0x1971F20 and m.pe.get_data(literal, 7) == b'List`1\0'
    for slot, name, lea, store in EXPORTS:
        i, s = instructions[lea], instructions[store]
        assert i.mnemonic == 'lea' and i.op_str.startswith('rdx, [rip')
        literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
        assert m.pe.get_data(literal, 80).split(b'\0', 1)[0] == name.encode('ascii')
        assert s.mnemonic == 'mov' and s.op_str.startswith('qword ptr [rip')
        assert s.address + s.size + struct.unpack('<i', bytes(s.bytes[-4:]))[0] == slot
    return {'instruction_assertions': len(checks) + 2 + len(EXPORTS) * 4,
            'chained_unwind_families': families,
            'classifier_export_bindings': {name: hex(slot) for slot, name, _, _ in EXPORTS}}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases, joins = [], []
    for name in ['List`1', 'List', 'List`2', 'List`10', 'list`1', 'List`1Extra', '', 'List\u00e9', 'List`1\0suffix']:
        for corlib in [False, True]:
            result = m.classify(name, corlib)
            matches = name.split('\0', 1)[0] == 'List`1'
            assert result['return_low_byte'] == bool(matches and corlib)
            kinds = [event['kind'] for event in result['events']]
            assert kinds == ['il2cpp_class_get_name'] + (['il2cpp_class_get_image', 'il2cpp_get_corlib'] if matches else [])
            cases.append(result)
    m.set_classifier('List`1', True)
    for old in [None, [], [b'\x7b\0\0\0']]:
        result = m.read_list('{"score":[1,2]}', old, 4)
        assert result['returned'] and result['loaded_list']['values_hex'] == ['01000000', '02000000']
        assert not any(e['kind'] == 'collection_predicate_service' and e['args'] == ['fixture_list'] for e in result['events'])
        joins.append(result)
    for values in [None, [], [b'\x01\0\0\0', b'\x02\0\0\0']]:
        result = m.write_list(values, 4)
        assert result['returned'] and result['loaded_list']['values_hex'] == [v.hex() for v in values or []]
        joins.append(result)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256, 'native_verified': verified,
            'cases': cases, 'case_count': len(cases), 'joined_cases': joins, 'joined_case_count': len(joins),
            'executed_address_count': len(m.classifier_executed),
            'scope': 'Native exact List`1 name-and-corlib classifier joined to List read/write and null construction. Class name/image/corlib metadata, managed allocation/constructor, GC/cache/array/allocator services remain supplied; namespace is not queried by this classifier and real managed discovery remains unclaimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['joined_case_count'], result['executed_address_count'])
