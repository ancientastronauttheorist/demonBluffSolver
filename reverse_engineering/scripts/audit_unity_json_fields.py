"""Execute the native JSON metadata adapter and descriptor traversal.

Metadata discovery and individual field bodies are explicit supplied services.
"""
import argparse
import json
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_parser import Machine as ParserMachine
from audit_unityplayer_wait import ENGINE_SHA256


class Machine(ParserMachine):
    def __init__(self, game_root):
        self.phase = None
        self.field_executed = set()
        super().__init__(game_root)
        self.managed = self.arena + 0x100000
        self.klass = self.arena + 0x101000
        self.cache_slot = self.arena + 0x102000
        self.cache = self.arena + 0x103000
        self.descriptors = self.arena + 0x104000
        self.runtime = self.arena + 0x108000
        self.q(self.base + 0x1CD6688, self.services + 0x100)
        self.q(self.base + 0x1CD6AF8, self.runtime)
        self.q(self.runtime + 0x548, self.services + 0x300)
        self.registry = self.arena + 0x109000
        self.q(self.registry, self.registry + 0x100)
        self.q(self.registry + 0x100 + 0x48, self.services + 0x400)
        self.q(self.base + 0x1CD66A8, self.registry)

    def snapshot(self):
        return {'managed_values': [self.rd(self.managed + 0x20 + i * 4)
                                   for i in range(4)],
                'cache_published': self.rq(self.cache_slot) == self.cache,
                'runtime_initialized': self.rq(self.base + 0x1CD6AF8) == self.runtime,
                'cursor_index': None if self.cursor is None else
                    (self.rq(self.cursor + 8) - self.descriptors) // 0x80,
                'remaining': None if self.cursor is None else self.rd(self.cursor + 0x18),
                'field_error': bool(self.u.mem_read(self.last_tree + 0x30, 1)[0]),
                'field_calls': self.field_calls.copy()}

    def event(self, kind, args):
        self.events.append({'kind': kind, 'args': args, 'snapshot': self.snapshot()})
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.failure == [kind, self.counts[kind]]:
            self.error = kind
            self.u.emu_stop()
            return False
        return True

    def hook(self, uc, address, size, data):
        if self.phase != 'fields':
            return super().hook(uc, address, size, data)
        x = self.x
        a = address - self.base
        self.executed.add(a)
        self.field_executed.add(a)
        cx, dx, r8, r9 = [self.reg(r) for r in
                        (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                         x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        if address == self.services + 0x100:
            assert cx == 0 and r8 == self.managed
            if self.event('reference_store', [0, 'managed']):
                self.q(dx, r8)
                self.ret()
        elif a == 0x81A880:
            if self.event('runtime_initialize', []):
                self.q(self.base + 0x1CD6AF8, self.runtime)
                self.ret()
        elif a == 0x75F6B0:
            assert cx == self.base + 0x81A880 and dx == self.base + 0x81A850
            if self.event('runtime_callback_register', ['initialize', 'cleanup']):
                self.ret()
        elif a == 0x7832B0:
            assert cx == self.last_tree and self.rq(r8) == self.klass
            if self.event('cache_lookup', ['tree', 'class']):
                self.q(dx, self.cache if self.lookup_cache else 0)
                self.ret()
        elif a == 0x784120:
            assert self.rq(cx) == self.klass and self.rq(cx + 8) == self.klass
            assert self.rq(cx + 0x10) == self.runtime
            direction = self.u.mem_read(cx + 0x1C, 2)
            assert bytes(direction) == b'\x09\x00'
            if self.event('metadata_build', [9, 'class']):
                self.q(dx, self.descriptors)
                self.q(dx + 0x10, self.field_count)
                self.ret()
        elif a == 0x79BB60:
            self.cursor = cx
        elif address in (self.services + 0x200, self.services + 0x280):
            index = (cx - self.descriptors - 0x10) // 0x80
            assert cx == self.descriptors + index * 0x80 + 0x10
            assert self.rq(dx + 8) == self.managed
            assert self.rq(dx + 0x10) == self.klass
            assert self.rq(dx + 0x20) == self.cursor
            assert self.rq(dx + 0x28) == self.last_tree
            assert self.rq(self.cursor + 8) == self.descriptors + (index + 1) * 0x80
            value = self.rd(cx)
            if self.event('field_body', [index, value,
                                      'alternate' if address == self.services + 0x280 else 'normal']):
                # Authored field-service effect, not recovered key lookup/type
                # conversion. The adapter's argument and cursor order is native.
                self.d(self.managed + 0x20 + index * 4, value)
                self.field_calls.append(index)
                if self.mutate_next and index == 0 and self.field_count > 1:
                    self.q(self.descriptors + 0x80 + 8, self.services + 0x280)
                    self.d(self.descriptors + 0x80 + 0x10, 777)
                if self.field_error and index == 0:
                    self.u.mem_write(self.last_tree + 0x30, b'\x01')
                if self.mutate_cache and index == 0:
                    self.q(self.cache + 0x30, 0)
                self.ret()
        elif a == 0x784C70:
            if self.event('reference_scope_cleanup', []):
                self.ret()
        elif a == 0x14E2D0:
            if self.event('metadata_storage_cleanup', []):
                self.ret()

    def apply(self, options):
        self.phase = None
        parsed = self.parse(b'{"score":1,"profile":{"enabled":true}}')
        assert parsed['tree'] is not None
        assert self.next_alloc < self.managed
        self.phase = 'fields'
        self.field_count = options.get('fields', 3)
        assert 0 <= self.field_count <= 4
        self.u.mem_write(self.managed, bytes(0x100))
        for i in range(4):
            self.d(self.managed + 0x20 + i * 4, 100 + i)
        self.u.mem_write(self.cache, bytes(0x200))
        directions = options.get('directions', [9])
        self.d(self.cache + 0x10, len(directions))
        for i, direction in enumerate(directions):
            self.u.mem_write(self.cache + 0x18 + i * 0x28,
                             direction.to_bytes(2, 'little'))
            self.q(self.cache + 0x20 + i * 0x28, self.descriptors)
            self.q(self.cache + 0x30 + i * 0x28,
                   options.get('counts', [self.field_count] * len(directions))[i])
        self.u.mem_write(self.descriptors, bytes(0x400))
        for i in range(self.field_count):
            self.q(self.descriptors + i * 0x80 + 8, self.services + 0x200)
            self.d(self.descriptors + i * 0x80 + 0x10, 10 + i)
        self.q(self.cache_slot, self.cache if options.get('preseed', True) else 0)
        self.q(self.base + 0x1CD6AF8, 0 if options.get('cold_runtime') else self.runtime)
        self.lookup_cache = options.get('lookup_cache', True)
        self.failure = options.get('failure')
        self.mutate_next = options.get('mutate_next', False)
        self.field_error = options.get('field_error', False)
        self.mutate_cache = options.get('mutate_cache', False)
        self.events, self.counts, self.field_calls = [], {}, []
        self.cursor, self.error = None, None
        x = self.x
        sp = self.stack + 0x18008
        self.q(sp, self.stop)
        preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI,
                     x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13,
                     x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(preserved):
            self.u.reg_write(register, 0xDEF00000 + i)
        for register, value in [(x.UC_X86_REG_RSP, sp),
                                (x.UC_X86_REG_RCX, self.last_tree),
                                (x.UC_X86_REG_RDX, self.managed),
                                (x.UC_X86_REG_R8, self.klass),
                                (x.UC_X86_REG_R9, self.cache_slot)]:
            self.u.reg_write(register, value)
        try:
            self.u.emu_start(self.base + 0xA8E030, self.stop,
                            timeout=2_000_000, count=100000)
        except Exception as exc:
            raise AssertionError(f'options={options}, RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(preserved):
                assert self.reg(register) == 0xDEF00000 + i
        result = {'input': options.copy(), 'events': self.events.copy(),
                  'final': self.snapshot(), 'error': self.error, 'returned': returned}
        self.phase = None
        return result


def verify_native(m):
    instructions = {}
    ranges = [(0xA8E030, 0xA8E2B5), (0x79BB60, 0x79BBFF), (0x784BC0, 0x784C6D),
              (0x76DAE5, 0x76DAF8)]
    for a, b in ranges:
        raw = m.pe.get_data(a, b - a)
        assert len(raw) == b - a
        decoded = list(m.cs.disasm(raw, a))
        assert sum(i.size for i in decoded) == b - a
        instructions.update({i.address: i for i in decoded})
    for a, b in ranges[:3]:
        assert any(e.struct.BeginAddress == a and e.struct.EndAddress == b
                   for e in m.pe.DIRECTORY_ENTRY_EXCEPTION)
    checks = {
        0xA8E07E: ('call', '0x7832b0'),
        0xA8E0A8: ('cmp', 'byte ptr [rax], 9'),
        0xA8E0AD: ('cmp', 'byte ptr [rax + 1], r15b'),
        0xA8E15A: ('call', '0x784120'),
        0xA8E186: ('call', '0x784bc0'),
        0xA8E1BE: ('call', '0x79bb60'),
        0xA8E255: ('call', '0x784c70'),
        0xA8E25F: ('call', '0x14e2d0'),
        0xA8E273: ('ret', ''),
        0xA8E2B0: ('jmp', '0xa8e164'),
        0x79BB99: ('call', 'qword ptr [rip + 0x153aae9]'),
        0x79BBCD: ('lea', 'rax, [r8 + 0x80]'),
        0x79BBD7: ('mov', 'qword ptr [rbx + 8], rax'),
        0x79BBDF: ('mov', 'dword ptr [rbx + 0x18], ecx'),
        0x79BBE2: ('lea', 'rcx, [r8 + 0x10]'),
        0x79BBE6: ('call', 'qword ptr [r8 + 8]'),
        0x79BBFE: ('ret', ''),
        0x784C6C: ('ret', ''),
        0x76DAF1: ('mov', 'qword ptr [rip + 0x1568b90], rax'),
    }
    for address, expected in checks.items():
        assert address in instructions
        i = instructions[address]
        assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
    # Resolve the previously bound registration literal without copying bodies.
    i = instructions[0x76DAE5]
    assert i.mnemonic == 'lea'
    literal = i.address + i.size + struct.unpack('<i', bytes(i.bytes[-4:]))[0]
    section = m.pe.get_section_by_rva(literal)
    assert section and literal - section.VirtualAddress < section.SizeOfRawData
    raw = m.pe.get_data(literal, min(128, section.SizeOfRawData - (literal - section.VirtualAddress)))
    assert raw.split(b'\0', 1)[0] == b'il2cpp_gc_wbarrier_set_field'
    assert 0x76DAF1 + 7 + 0x1568B90 == 0x1CD6688
    assert 0x79BB99 + 6 + 0x153AAE9 == 0x1CD6688
    return {'instruction_assertions': len(checks),
            'entry_ranges': [[hex(a), hex(b)] for a, b in ranges[:3]],
            'reference_store_export': 'il2cpp_gc_wbarrier_set_field'}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    cases = []
    for count in [0, 1, 3, 4]:
        for directions in [[], [9], [8], [0x109], [8, 9], [9, 9]]:
            for preseed in [False, True]:
                for lookup_cache in [False, True]:
                    options = {'fields': count, 'directions': directions,
                               'preseed': preseed, 'lookup_cache': lookup_cache}
                    result = m.apply(options)
                    assert result['returned'] and result['final']['field_calls'] == list(range(count))
                    kinds = [e['kind'] for e in result['events']]
                    assert ('cache_lookup' in kinds) == (not preseed)
                    hit = (preseed or lookup_cache) and 9 in directions
                    assert ('metadata_build' in kinds) == (not hit)
                    assert result['final']['cache_published'] == (preseed or lookup_cache)
                    assert result['final']['remaining'] == 0
                    cases.append(result)
    baselines = [{}, {'preseed': False}, {'preseed': False, 'lookup_cache': False},
                 {'directions': [8]}, {'fields': 0}, {'cold_runtime': True},
                 {'cold_runtime': True, 'preseed': False, 'lookup_cache': False}]
    for options in baselines:
        baseline = m.apply(options)
        if options.get('cold_runtime'):
            assert baseline['returned'] and baseline['final']['runtime_initialized']
            assert sum(e['kind'] == 'runtime_initialize' for e in baseline['events']) == 1
            cases.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.apply(dict(options, failure=[kind, counts[kind]]))
            assert result['error'] == kind and not result['returned']
            assert result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            cases.append(result)
    for options in [{'mutate_next': True}, {'mutate_cache': True}, {'field_error': True},
                    {'directions': [9, 9], 'counts': [0, 3]},
                    {'directions': [8, 9, 9], 'counts': [4, 1, 3]}]:
        result = m.apply(options)
        assert result['returned']
        if options.get('mutate_next'):
            assert result['final']['managed_values'] == [10, 777, 12, 103]
            assert result['events'][2]['args'] == [1, 777, 'alternate']
        if options.get('mutate_cache') or options.get('field_error'):
            assert result['final']['field_calls'] == [0, 1, 2]
        if options.get('counts'):
            first = options['directions'].index(9)
            assert result['final']['field_calls'] == list(range(options['counts'][first]))
        cases.append(result)
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'native_verified': verified,
            'cases': cases, 'case_count': len(cases),
            'executed_address_count': len(m.executed),
            'field_adapter_address_count': len(m.field_executed),
            'scope': 'Actual metadata adapter, native reference context and descriptor traversal. Cache discovery/build and individual field bodies remain supplied; reference-scope and metadata cleanup are inert gateways. No managed field inclusion/conversion or arbitrary serialization claim.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(len(result['cases']), result['executed_address_count'])
