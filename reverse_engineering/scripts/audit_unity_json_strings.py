"""Execute native JSON string fields with explicit IL2CPP string services."""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_numeric_roundtrip import Machine as RoundtripMachine
from audit_unityplayer_wait import ENGINE_SHA256


STRING_SOURCE = 'core+0x180'
EXPORTS = [
    (0x1CD6358, 'il2cpp_string_new_wrapper', 0x76E4FD, 0x76E509),
    (0x1CD6378, 'il2cpp_string_length', 0x76E43F, 0x76E44B),
    (0x1CD6380, 'il2cpp_string_chars', 0x76E465, 0x76E471),
]


class Machine(RoundtripMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.strings = {}
        self.string_cursor = self.arena + 0x600000
        self.string_executed = set()
        self.field_timeout = 10_000_000
        self.field_instruction_limit = 1_000_000
        self.string_services = {}
        # Authored Windows TIB/TLS state. The conversion's replacement character
        # singleton is already initialized; no OS initialization is emulated.
        tib, slots, tls = [self.arena + a for a in (0x700000, 0x701000, 0x702000)]
        self.u.reg_write(self.x.UC_X86_REG_GS_BASE, tib)
        self.q(tib + 0x10, self.stack)
        self.q(tib + 0x58, slots)
        self.q(slots, tls)
        self.d(self.base + 0x1C4A0EC, 0)
        self.d(tls + 0x10, 0)
        self.d(self.base + 0x1CDA810, 0)
        self.d(self.base + 0x1CDA814, 0xFFFD)
        for i, (slot, name, _, _) in enumerate(EXPORTS):
            service = self.services + 0x790 + i * 0x10
            self.q(self.base + slot, service)
            self.string_services[service] = name

    def make_string(self, text):
        # Authored service objects, not a claim about IL2CPP object layout.
        raw = text.encode('utf-16-le', errors='surrogatepass')
        token = self.string_cursor
        chars = token + 0x20
        self.string_cursor += (len(raw) + 0x40 + 15) & ~15
        assert self.string_cursor < self.arena + 0xF00000
        self.u.mem_write(chars, raw + b'\0\0')
        self.strings[token] = {'text': text, 'chars': chars, 'length': len(raw) // 2}
        return token

    def string_text(self, token):
        return self.strings[token]['text'] if token else None

    def reset_strings(self):
        self.strings = {}
        self.string_cursor = self.arena + 0x600000

    def field_width(self, field):
        return 8 if field.get('source') == STRING_SOURCE else super().field_width(field)

    def node(self, address, depth=0):
        tag = self.rd(address + 16)
        if tag & 7 == 5 and tag & 0x400000:
            length = 15 - self.u.mem_read(address + 15, 1)[0]
            assert 0 <= length <= 15
            return {'kind': 'string', 'flags': tag, 'input_offset': None,
                    'utf8_hex': bytes(self.u.mem_read(address, length)).hex()}
        return super().node(address, depth)

    def hook(self, uc, address, size, data):
        if self.phase not in ('metadata', 'writer'):
            return super().hook(uc, address, size, data)
        self.string_executed.add(address - self.base)
        x = self.x
        cx, dx, r8 = [self.reg(r) for r in
                     (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        name = self.string_services.get(address)
        if name:
            if name == 'il2cpp_string_new_wrapper':
                raw = self.cstring(cx)
                if self.metadata_event(name, [raw.hex()]):
                    self.ret(self.make_string(raw.decode('utf-8')))
            else:
                assert cx in self.strings
                value = self.strings[cx]
                if self.metadata_event(name, [value['text']]):
                    self.ret(value['length' if name == 'il2cpp_string_length' else 'chars'])
            return
        if self.export_services.get(address) == 'il2cpp_class_is_valuetype' and self.sources.get(cx) == STRING_SOURCE:
            if self.metadata_event('il2cpp_class_is_valuetype', [STRING_SOURCE]):
                self.ret(0)
            return
        if address == self.services + 0x180:
            assert cx == 0 and (r8 == self.managed or r8 in self.strings)
            if self.metadata_event('reference_store', ['string' if r8 in self.strings else 'object']):
                self.q(dx, r8)
                self.ret()
            return
        return super().hook(uc, address, size, data)

    def read_string(self, payload, old='preserved', options=None):
        options = options or {}
        self.reset_strings()
        old_token = self.make_string(old) if old is not None else 0
        field = {'name': 'score', 'source': STRING_SOURCE, 'type_enum': 14,
                 'initial_hex': old_token.to_bytes(8, 'little').hex()}
        self.registry_build(False)
        result = self.run([field], dict(options, joined=True, json=payload))
        token = self.rq(self.managed + 0x20)
        result.update({'input_json': payload, 'old_text': old,
                       'loaded_text': self.string_text(token),
                       'old_pointer_retained': token == old_token})
        return result

    def write_string(self, text, options=None):
        self.reset_strings()
        token = self.make_string(text) if text is not None else 0
        field = {'name': 'score', 'source': STRING_SOURCE, 'type_enum': 14}
        result = self.serialize([field], [token.to_bytes(8, 'little')], options)
        result['input_text'] = text
        if result['returned']:
            rendered = bytes.fromhex(result['json_utf8_hex']).decode('utf-8')
            read = self.read_string(rendered)
            assert read['returned'] and not read['final']['scope_linked']
            result['loaded_text'] = read['loaded_text']
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
    for root in [0xA9BD70, 0xA9E7A0, 0x755EB0, 0x4A8590, 0xAA6450,
                 0x10962B0, 0xAAFC90, 0x17A86C0, 0x76CA70]:
        assert root in groups
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            assert len(raw) == b - a
            decoded = list(m.cs.disasm(raw, a))
            assert sum(i.size for i in decoded) == b - a
            instructions.update({i.address: i for i in decoded})
    for a in [0xA91B50, 0xA93240]:
        raw = m.pe.get_data(a, 9)
        decoded = list(m.cs.disasm(raw, a))
        assert sum(i.size for i in decoded) == 9
        assert [(i.mnemonic, i.op_str) for i in decoded] == [
            ('add', 'rcx, 8'), ('jmp', '0xa9bd70' if a == 0xA91B50 else '0xa9e7a0')]
    checks = {
        0xA9BDE3: ('call', '0xaac540'),
        0xA9BE05: ('call', '0xa00ec0'),
        0xA9BE12: ('call', '0xaa6450'),
        0xA9BE5D: ('call', 'rax'),
        0xA9BE6C: ('call', 'qword ptr [rip + 0x123a816]'),
        0xA9BE84: ('call', 'qword ptr [rip + 0x123a7fe]'),
        0xA9BEA0: ('call', 'qword ptr [rip + 0x123a7e2]'),
        0xA9BEC2: ('call', 'qword ptr [rip + 0x123a7c0]'),
        0xA9E7F8: ('call', '0x755eb0'),
        0xA9E85F: ('call', '0x159230'),
        0xA9E8DD: ('call', '0x10962b0'),
        0xA9E8F2: ('call', '0x1096320'),
        0x755EE3: ('call', 'rax'),
        0x755EF2: ('call', 'rax'),
        0x755F0B: ('cmp', 'rdx, 0x7d0'),
        0x755F2B: ('call', '0x17a86c0'),
        0x755F65: ('call', '0x354970'),
        0x755F8A: ('call', '0x4a8590'),
        0x755FAF: ('call', '0x159230'),
        0x4A85A3: ('mov', 'rax, qword ptr gs:[0x58]'),
        0x4A8610: ('lea', 'eax, [rcx - 0xdc00]'),
        0x4A862A: ('mov', 'edx, esi'),
        0x4A8644: ('call', '0x32ea90'),
        0xAAFCAD: ('cmp', 'ebp, 0xf'),
        0xAAFCB7: ('mov', 'dword ptr [rcx + 0x10], 0x700005'),
        0xAAFCC4: ('mov', 'byte ptr [rcx + 0xf], al'),
        0xAAFCCF: ('mov', 'dword ptr [rcx + 0x10], 0x300005'),
        0xAAFD03: ('call', '0x354970'),
        0x17A86DC: ('mov', 'r11, qword ptr gs:[0x10]'),
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
    slots = {0x755ED9: 0x1CD6378, 0x755EEB: 0x1CD6380,
             0xA9BE48: 0x1CD6358, 0x4A85B0: 0x1C4A0EC,
             0x4A85CE: 0x1CDA810, 0x4A85E4: 0x1CDA814}
    for address, expected in slots.items():
        i = instructions[address]
        assert i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0] == expected
    i = instructions[0xA9BDEC]
    literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
    assert m.pe.get_data(literal, 16).split(b'\0', 1)[0] == b'string'
    return {'instruction_assertions': len(checks) + 4 + 12 + len(slots) + 1,
            'chained_unwind_families': families,
            'string_export_bindings': {name: hex(slot) for slot, name, _, _ in EXPORTS}}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    writes, reads, mixed, failures = [], [], [], []
    texts = ['', None, 'hello', 'a' * 15, 'a' * 16, 'a' * 24, 'a' * 25,
             'a' * 499, 'a' * 500, 'a' * 600, 'caf\u00e9 \U0001f608',
             '\u6f22\u5b57', '"\\\n\r\t\b\f', '\0', 'a\0b',
             '\ud800', '\udc00', '\ud800Z', '\ud800\ud800\udc00']
    for text in texts:
        expected = (text or '').split('\0', 1)[0]
        if text in ('\ud800', '\udc00', '\ud800Z'):
            expected = '\ufffd'
        elif text == '\ud800\ud800\udc00':
            expected = '\ufffd\ufffd'
        for pretty in [False, True]:
            result = m.write_string(text, {'pretty': pretty})
            assert result['returned'] and result['loaded_text'] == expected
            writes.append(result)
    payloads = [('{}', 'preserved'), ('{"Score":"changed"}', 'preserved'),
                ('{"score":null}', ''), ('{"score":true}', 'true'),
                ('{"score":false}', 'false'), ('{"score":123}', '123'),
                ('{"score":-12}', '-12'), ('{"score":1.5}', '1.500000'),
                ('{"score":{}}', ''), ('{"score":[]}', ''),
                ('{"score":"a\\u0000b"}', 'a'),
                ('{"score":"first","score":"second"}', 'first'),
                ('{"score":"\\u00e9\\ud83d\\ude08"}', '\u00e9\U0001f608')]
    for payload, expected in payloads:
        result = m.read_string(payload)
        assert result['returned'] and result['loaded_text'] == expected
        assert not result['final']['scope_linked']
        reads.append(result)
    fields = [{'name': 'child', 'source': STRING_SOURCE, 'type_enum': 14},
              {'name': 'base', 'source': STRING_SOURCE, 'type_enum': 14,
               'in_parent': True}, {'name': 'score'}]
    for pretty in [False, True]:
        m.reset_strings()
        values = [m.make_string(text).to_bytes(8, 'little') for text in
                  ['child\U0001f608', 'parent\u00e9']] + [b'\x7b\0\0\0']
        result = m.serialize(fields, values, {'parent': True, 'pretty': pretty})
        assert result['returned']
        m.registry_build(False)
        rendered = bytes.fromhex(result['json_utf8_hex']).decode('utf-8')
        read = m.run(fields, {'parent': True, 'joined': True, 'json': rendered})
        assert read['returned'] and not read['final']['scope_linked']
        loaded = [m.string_text(m.rq(m.managed + offset)) for offset in [0x20, 0x30]]
        assert loaded == ['child\U0001f608', 'parent\u00e9'] and m.rd(m.managed + 0x40) == 123
        assert [d['field'] for d in read['final']['descriptors']] == ['base', 'child', 'score']
        result.update({'loaded_strings': loaded, 'loaded_number': 123,
                       'reader_descriptor_order': ['base', 'child', 'score']})
        mixed.append(result)
    # Every service boundary in a normal read/write is independently stopped.
    for direction in ['read', 'write']:
        operation = (lambda options: m.read_string('{"score":"hello"}', options=options)) if direction == 'read' else (lambda options: m.write_string('hello', options))
        baseline = operation({})
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = operation({'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['error'] == kind
            assert result['events'] == baseline['events'][:index + 1]
            if direction == 'read':
                assert result['final'] == event['snapshot']
            result['direction'] = direction
            failures.append(result)
    required = [0xA91B50, 0xA9BD70, 0xAA6450, 0xA93240, 0xA9E7A0,
                0x755EB0, 0x4A8590, 0x32EA90, 0x10962B0, 0xAAFC90]
    assert all(rva in m.string_executed for rva in required)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'native_verified': verified, 'write_cases': writes, 'read_cases': reads,
            'write_case_count': len(writes), 'read_case_count': len(reads),
            'mixed_cases': mixed, 'mixed_case_count': len(mixed),
            'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.string_executed),
            'required_native_entries_executed': [hex(rva) for rva in required],
            'scope': 'Native string metadata selection, read/write field handlers, UTF-16 to UTF-8 conversion, JSON construction/rendering and reload. IL2CPP string length/chars and valid UTF-8 string creation, authored TIB/TLS, initialized replacement-character state, metadata/cache/GC/allocator services are explicit fixtures. Runtime allocation and invalid UTF-8 creation policy, compound references and graphs remain unclaimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['write_case_count'], result['read_case_count'], result['failure_case_count'], result['executed_address_count'])
