"""Execute native numeric ToJson/FromJson round trips over supplied metadata."""
import argparse
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_metadata import Machine as MetadataMachine
from audit_unity_json_primitives import SOURCES
from audit_unityplayer_wait import ENGINE_SHA256


WIDTHS = dict(SOURCES + [('core+0x170', 1), ('core+0x100', 2), ('core+0x198', 8)])


class Machine(MetadataMachine):
    def __init__(self, game_root):
        super().__init__(game_root)
        self.output_string = self.arena + 0x540000
        self.writer_executed = set()
        self.q(self.base + 0x1CD62C8, self.services + 0x780)

    def registry_build(self, writer):
        self.phase = 'registry'
        self.events, self.counts = [], {}
        self.failure, self.error = None, None
        self.growth, self.stack_seed = 40, 0
        assert self.call(0xA902D0 if writer else 0xA8F090)
        self.q(self.registry + (0x140 if writer else 0x148), self.provider)
        self.phase = None

    def field_width(self, field):
        return WIDTHS[field.get('source', 'core+0x120')]

    def hook(self, uc, address, size, data):
        if self.phase != 'writer':
            return super().hook(uc, address, size, data)
        rva = address - self.base
        self.writer_executed.add(rva)
        if address == self.services + 0x780:
            assert self.reg(self.x.UC_X86_REG_RCX) == self.managed
            if self.metadata_event('il2cpp_object_get_class', ['fixture_object']):
                self.ret(self.klass)
            return
        if rva == 0x10964E0:
            self.last_tree = self.reg(self.x.UC_X86_REG_RCX)
        if rva == 0x7832B0:
            # The caller leaves RCX at the constructor's volatile scratch
            # value. This supplied cache service uses only RDX/R8.
            assert self.rq(self.reg(self.x.UC_X86_REG_R8)) == self.klass
            if self.metadata_event('cache_lookup_service', ['miss']):
                self.q(self.reg(self.x.UC_X86_REG_RDX), 0)
                self.ret()
            return
        if rva == 0x1096690:
            self.tree_before_render = self.node(self.last_tree + 0xB0)
            assert self.u.mem_read(self.last_tree + 0x30, 1)[0] == 0
        # Reuse the supplied metadata exports and allocator services, while the
        # native writer constructor, adapter, processors and renderer execute.
        self.phase = 'metadata'
        try:
            return super().hook(uc, address, size, data)
        finally:
            self.phase = 'writer'

    def serialize(self, definitions, values, options=None):
        options = options or {}
        self.registry_build(False)
        assert self.run(definitions, {'parent': options.get('parent', False)})['returned']
        self.registry_build(True)
        for i, value in enumerate(values):
            field = definitions[i]
            width = self.field_width(field)
            assert len(value) == width
            self.u.mem_write(self.managed + field.get('offset', 0x20 + i * 0x10), value)
        original = bytes(self.u.mem_read(self.managed, 0x100))
        self.put_string(self.output_string, b'')
        self.output_vector, self.tree_before_render = None, None
        self.metadata_events, self.metadata_counts = [], {}
        self.metadata_failure, self.metadata_error = options.get('failure'), None
        self.phase = 'writer'
        x = self.x
        sp = self.stack + 0x18008
        self.u.mem_write(self.stack, bytes(0x18000))
        self.q(sp, self.stop)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI,
                     x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13,
                     x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(registers):
            self.u.reg_write(register, 0xADD00000 + i)
        for register, value in [(x.UC_X86_REG_RSP, sp),
                                (x.UC_X86_REG_RCX, self.managed),
                                (x.UC_X86_REG_RDX, self.output_string),
                                (x.UC_X86_REG_R8, int(options.get('pretty', False)))]:
            self.u.reg_write(register, value)
        try:
            self.u.emu_start(self.base + 0xAACA50, self.stop,
                            timeout=getattr(self, 'field_timeout', 2_000_000),
                            count=getattr(self, 'field_instruction_limit', 200000))
        except Exception as exc:
            raise AssertionError(f'fields={definitions}, options={options}, '
                                 f'RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.metadata_error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(registers):
                assert self.reg(register) == 0xADD00000 + i
        assert bytes(self.u.mem_read(self.managed, 0x100)) == original
        rendered = self.get_string(self.output_string)
        result = {'fields': definitions, 'values_hex': [v.hex() for v in values],
                  'options': options.copy(), 'json_utf8_hex': rendered.hex(),
                  'tree_before_render': self.tree_before_render,
                  'events': self.metadata_events.copy(),
                  'returned': returned, 'error': self.metadata_error,
                  'managed_input_retained': True}
        self.phase = None
        return result

    def roundtrip(self, definitions, values, options=None):
        result = self.serialize(definitions, values, options)
        assert result['returned']
        self.registry_build(False)
        rendered = bytes.fromhex(result['json_utf8_hex']).decode('utf-8')
        read = self.run(definitions, {'joined': True, 'json': rendered,
                                     'parent': (options or {}).get('parent', False)})
        assert read['returned']
        loaded = []
        for i, field in enumerate(definitions):
            width = self.field_width(field)
            offset = field.get('offset', 0x20 + i * 0x10)
            loaded.append(bytes(self.u.mem_read(self.managed + offset, width)).hex())
        result.update({'loaded_values_hex': loaded,
                       'bit_exact_roundtrip': loaded == result['values_hex'],
                       'reader_scope_cleared': not read['final']['scope_linked']})
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
    for root in [0xA902D0, 0xAACA50, 0x10964E0, 0xA8E480, 0x9FC9B0,
                 0x7832B0, 0x76CA70]:
        assert root in groups
        families[hex(root)] = [[hex(a), hex(b)] for a, b in groups[root]]
        for a, b in groups[root]:
            raw = m.pe.get_data(a, b - a)
            assert len(raw) == b - a
            decoded = list(m.cs.disasm(raw, a))
            assert sum(i.size for i in decoded) == b - a
            instructions.update({i.address: i for i in decoded})
    checks = {
        0xA9030F: ('call', '0x354ec0'),
        0xA9031E: ('mov', 'qword ptr [rbx], r15'),
        0xA90321: ('mov', 'qword ptr [rbx + 0x10], r15'),
        0xA90325: ('mov', 'qword ptr [rbx + 0x18], 1'),
        0xA9036A: ('lea', 'rax, [rip + 0x2abf]'),
        0xA9150D: ('ret', ''),
        0xAACA85: ('mov', 'rax, qword ptr [rip + 0x122983c]'),
        0xAACA8F: ('call', 'rax'),
        0xAACA9D: ('call', '0x10964e0'),
        0xAACAC4: ('call', '0x7832b0'),
        0xAACAE5: ('cmp', 'byte ptr [rcx], 8'),
        0xAACB42: ('mov', 'rbx, qword ptr [rcx + 0x40]'),
        0xAACB9C: ('call', '0x784120'),
        0xAACBEC: ('call', '0xa8e480'),
        0xAACBF6: ('call', '0x14e2d0'),
        0xAACC06: ('call', '0x1096690'),
        0xAACC0F: ('call', '0x9fc9b0'),
        0xAACC37: ('ret', ''),
        0x1096605: ('call', '0xaad650'),
        0x109660D: ('call', '0x20bef0'),
        0x109663A: ('ret', ''),
        0xA8E4A2: ('call', '0x784bc0'),
        0xA8E4E7: ('call', '0x79bb60'),
        0xA8E678: ('call', '0x784c70'),
        0xA8E68A: ('ret', ''),
        0x9FC9FF: ('call', '0x20bef0'),
        0x9FCA2B: ('call', '0x14e2d0'),
    }
    for address, expected in checks.items():
        assert address in instructions
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
    for lea, store, slot in [(0x76E16D, 0x76E179, 0x1CD62C8)]:
        i, s = instructions[lea], instructions[store]
        assert i.mnemonic == 'lea' and i.op_str.startswith('rdx, [rip')
        literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
        assert m.pe.get_data(literal, 80).split(b'\0', 1)[0] == b'il2cpp_object_get_class'
        assert s.mnemonic == 'mov' and s.op_str.startswith('qword ptr [rip')
        assert s.address + s.size + struct.unpack('<i', bytes(s.bytes[-4:]))[0] == slot
    # Cache lookup homes incoming RCX, then replaces it with its lock address.
    # The writer call site sets only RDX/R8 after the constructor.
    cache_prefix = [instructions[a] for a in sorted(instructions)
                    if 0x7832B0 <= a < 0x7832E0]
    uses = [i for i in cache_prefix if 'rcx' in i.op_str]
    assert (uses[0].mnemonic, uses[0].op_str) == ('mov', 'qword ptr [rsp + 8], rcx')
    assert uses[1].mnemonic == 'lea' and uses[1].op_str.startswith('rcx, [rip')
    call_site = [instructions[a] for a in sorted(instructions)
                 if 0xAACAA2 <= a < 0xAACAC4]
    assert all(not i.op_str.startswith('rcx,') for i in call_site)
    assert instructions[0xA9036A].address + instructions[0xA9036A].size + 0x2ABF == 0xA92E30
    return {'instruction_assertions': len(checks) + 8,
            'chained_unwind_families': families,
            'object_class_export': {'name': 'il2cpp_object_get_class', 'slot': '0x1cd62c8'},
            'cache_first_register': 'Caller leaves constructor scratch in RCX; cache entry homes it before loading its lock address. Supplied lookup interprets only RDX/R8.'}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases = []
    patterns = {
        'core+0x120': [0, 123, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF],
        'core+0x130': [0, 1, 0xFF],
        'core+0x188': [0, 0x80000000, 0x3FC00000, 0x7F7FFFFF, 1,
                       0x7FC00000, 0x7F800000, 0xFF800000],
        'core+0x118': [0, 0x7FFF, 0x8000, 0xFFFF],
        'core+0x128': [0, 123, 0x7FFFFFFFFFFFFFFF, 0x8000000000000000,
                       0xFFFFFFFFFFFFFFFF],
        'core+0x178': [0, 123, 0xFFFF],
        'core+0x168': [0, 0x7F, 0x80, 0xFF],
        'core+0x108': [0, 123, 0x80000000, 0xFFFFFFFF],
        'core+0x110': [0, 123, 0x8000000000000000, 0xFFFFFFFFFFFFFFFF],
        'core+0x170': [0, 1, 0xFF],
        'core+0x100': [0, 123, 0xFFFF],
        'core+0x198': [0, 0x8000000000000000, 0x3FF8000000000000,
                       0x7FEFFFFFFFFFFFFF, 1, 0x7FF8000000000000,
                       0x7FF0000000000000, 0xFFF0000000000000],
    }
    for source, width in WIDTHS.items():
        for value in patterns[source]:
            for pretty in [False, True]:
                result = m.roundtrip([{'name': 'score', 'source': source}],
                                     [value.to_bytes(width, 'little')], {'pretty': pretty})
                expected_loss = (source == 'core+0x130' and value == 0xFF or
                                 source == 'core+0x188' and value == 0x80000000 or
                                 source == 'core+0x198' and value in (0x8000000000000000, 1))
                assert result['bit_exact_roundtrip'] == (not expected_loss)
                cases.append(result)
    fields = [{'name': 'score'}, {'name': 'ratio', 'source': 'core+0x188'},
              {'name': 'small', 'source': 'core+0x130'},
              {'name': 'base', 'in_parent': True}]
    values = [bytes.fromhex(v) for v in ['07000000', '0000c03f', '01', '0b000000']]
    for pretty in [False, True]:
        result = m.roundtrip(fields, values, {'pretty': pretty, 'parent': True})
        assert result['bit_exact_roundtrip']
        cases.append(result)
    failure_cases = []
    baseline = m.serialize([{'name': 'score'}], [b'\x7b\0\0\0'])
    counts = {}
    for i, event in enumerate(baseline['events']):
        kind = event['kind']
        counts[kind] = counts.get(kind, 0) + 1
        result = m.serialize([{'name': 'score'}], [b'\x7b\0\0\0'],
                             {'failure': [kind, counts[kind]]})
        assert not result['returned'] and result['error'] == kind
        assert result['events'] == baseline['events'][:i + 1]
        failure_cases.append(result)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'native_verified': verified,
            'cases': cases, 'case_count': len(cases),
            'failure_cases': failure_cases, 'failure_case_count': len(failure_cases),
            'writer_executed_address_count': len(m.writer_executed),
            'native_writer_address_count': sum(0 <= a < m.pe.OPTIONAL_HEADER.SizeOfImage
                                               for a in m.writer_executed),
            'scope': 'Native writer registry, ToJson serializer caller, writer construction, metadata building, adapter, numeric field processors, rendering and normal destruction joined to native FromJson numeric application. Runtime metadata/cache lookup, classifiers, allocation, GC and vector cleanup are supplied services; compound/reference serialization is unclaimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['failure_case_count'], result['writer_executed_address_count'])
