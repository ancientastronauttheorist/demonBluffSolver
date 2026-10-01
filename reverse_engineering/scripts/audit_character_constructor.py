"""Execute the pinned Character constructor with authored allocation/base services."""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD


class Machine:
    def __init__(self, game_root, dumper_root):
        import capstone
        import pefile
        import unicorn
        from unicorn import x86_const as x
        assert unicorn.__version__ == '2.1.4'
        self.x, self.unicorn = x, unicorn
        root = Path(__file__).parents[1]
        manifest = json.loads((root / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
        extraction = json.loads((root / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(path, digest):
            raw = path.read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == digest.upper()
            return raw
        raw = pin(Path(game_root) / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
        metadata = json.loads(pin(Path(dumper_root) / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
        dump = pin(Path(dumper_root) / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
        body = re.search(r'^public class Character :[^\n]* // TypeDefIndex: 5487\s*\{(.*?)^\}', dump, re.M | re.S)
        assert body
        self.fields = {'uses': (0xDC, 4, 'private int pickableUses; // 0xDC'),
                       'acted_infos': (0x148, 8, 'public List<ActedInfo> actedInfos; // 0x148'),
                       'hover_infos': (0x150, 8, 'public List<ActedInfo> onHoverInfo; // 0x150'),
                       'saved_act': (0x198, 8, 'private string savedAct; // 0x198'),
                       'act': (0x1A1, 1, 'public bool act; // 0x1A1')}
        assert all(declaration in body[1] for _, _, declaration in self.fields.values())
        list_body = re.search(r'^public class List<T> :[^\n]* // TypeDefIndex: 1510\s*\{(.*?)^\}', dump, re.M | re.S)
        assert list_body and all(declaration in list_body[1] for declaration in
                                 ['private T[] _items;', 'private int _size;', 'private int _version;', 'private object _syncRoot;'])
        rows = [m for m in metadata['ScriptMethod'] if m['Name'] == 'Character$$.ctor']
        assert len(rows) == 1 and rows[0]['Address'] == 0x3697C0
        self.method = rows[0]
        assert self.method['Signature'] == 'void Character___ctor (Character_o* __this, const MethodInfo* method);'
        assert min(m['Address'] for m in metadata['ScriptMethod'] if m['Address'] > 0x3697C0) == 0x3698B0
        rows = [m for m in metadata['ScriptMethod'] if m['Name'] == 'UnityEngine.MonoBehaviour$$.ctor']
        assert len(rows) == 1 and rows[0]['Address'] == 0x1C79770
        self.base_method = rows[0]
        self.pe = pefile.PE(data=raw, fast_load=True)
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        cs.detail = True
        instructions = list(cs.disasm(self.pe.get_data(0x3697C0, 0xF0), 0x3697C0))
        while instructions[-1].mnemonic == 'int3': instructions.pop()
        assert instructions[-1].address + instructions[-1].size == 0x3698A3
        assert all(a.address + a.size == b.address for a, b in zip(instructions, instructions[1:]))
        self.decoded = {i.address: i for i in instructions}
        checks = {0x369801: ('mov', 'dword ptr [rdi + 0xdc], 1'),
                  0x36981E: ('mov', 'rcx, rax'), 0x369821: ('mov', 'rbx, rax'),
                  0x369824: ('call', '0xb02160'), 0x369829: ('lea', 'rcx, [rdi + 0x148]'),
                  0x369833: ('mov', 'qword ptr [rcx], rbx'), 0x369836: ('call', '0x2b6ff0'),
                  0x36984E: ('mov', 'rcx, rax'), 0x369851: ('mov', 'rbx, rax'),
                  0x369854: ('call', '0xb02160'), 0x369859: ('lea', 'rcx, [rdi + 0x150]'),
                  0x369863: ('mov', 'qword ptr [rcx], rbx'), 0x369866: ('call', '0x2b6ff0'),
                  0x369872: ('lea', 'rcx, [rdi + 0x198]'), 0x369879: ('mov', 'qword ptr [rcx], rax'),
                  0x369888: ('xor', 'edx, edx'), 0x36988A: ('mov', 'byte ptr [rdi + 0x1a1], 1'),
                  0x369891: ('mov', 'rcx, rdi'), 0x36989E: ('jmp', '0x1c79770')}
        for address, expected in checks.items():
            assert address in self.decoded and (self.decoded[address].mnemonic, self.decoded[address].op_str) == expected
        self.instruction_assertions = len(checks)
        self.services = {0x2B7B40: 'metadata', 0x2B7D40: 'allocate', 0xB02160: 'list_constructor',
                         0x2B6FF0: 'barrier', 0x1C79770: 'base_constructor'}
        assert {int(i.op_str, 16) for i in instructions if i.mnemonic == 'call'} == set(self.services) - {0x1C79770}
        self.flags = set()
        slots = set()
        for i in instructions:
            for operand in i.operands:
                if operand.type == capstone.CS_OP_MEM and operand.mem.base == capstone.x86.X86_REG_RIP:
                    address = i.address + i.size + operand.mem.disp
                    (self.flags if operand.size == 1 else slots).add(address)
        rows = [r for section in ['ScriptMetadata', 'ScriptMetadataMethod', 'ScriptString'] for r in metadata[section] if r['Address'] in slots]
        assert {r['Address'] for r in rows} == slots
        assert len(rows) == 3 and len(self.flags) == 1
        self.bindings = rows
        self.u = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        self.u.mem_map(self.base, (self.pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
        self.u.mem_write(self.base, self.pe.get_memory_mapped_image())
        self.arena, self.stack, self.stop = 0x200000000, 0x300000000, 0x400000000
        self.u.mem_map(self.arena, 0x20000)
        self.u.mem_map(self.stack, 0x20000)
        self.u.mem_map(self.stop, 0x1000)
        self.actor = self.arena + 0x1000
        self.old_lists = [self.arena + 0x2000, self.arena + 0x3000]
        self.fresh_lists = [self.arena + 0x4000, self.arena + 0x5000]
        self.empty_string, self.alternate_string = self.arena + 0x6000, self.arena + 0x7000
        self.type_info, self.method_info = self.arena + 0x8000, self.arena + 0x9000
        self.empty_array = self.arena + 0x10000
        self.slot_values = {}
        for row in rows:
            if row.get('Name') == 'System.Collections.Generic.List<ActedInfo>_TypeInfo': value = self.type_info
            elif row.get('Name') == 'Method$System.Collections.Generic.List<ActedInfo>..ctor()': value = self.method_info
            else:
                assert row.get('Value') == ''
                value = self.empty_string
                self.string_slot = row['Address']
            self.slot_values[row['Address']] = value
        self.visited = set()
        self.u.hook_add(unicorn.UC_HOOK_CODE, self.hook)

    def q(self, a, value): self.u.mem_write(a, struct.pack('<Q', value))
    def rq(self, a): return struct.unpack('<Q', self.u.mem_read(a, 8))[0]
    def reg(self, r): return self.u.reg_read(r)
    def ret(self, value=0):
        sp = self.reg(self.x.UC_X86_REG_RSP)
        for name in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']:
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_' + name), 0xBAD0000000000000)
        for index in range(6):
            self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{index}'), 0xBAD0000000000000)
        self.u.reg_write(self.x.UC_X86_REG_RAX, value)
        self.u.reg_write(self.x.UC_X86_REG_RSP, sp + 8)
        self.u.reg_write(self.x.UC_X86_REG_RIP, self.rq(sp))

    def snapshot(self):
        values = {name: int.from_bytes(self.u.mem_read(self.actor + offset, size), 'little')
                  for name, (offset, size, _) in self.fields.items()}
        lists = json.loads(json.dumps(self.lists))
        for record in lists:
            pointer = record['identity']
            record['backing'] = self.rq(pointer + 0x10)
            if record['constructed']:
                assert int.from_bytes(self.u.mem_read(pointer + 0x18, 4), 'little') == record['count']
                assert int.from_bytes(self.u.mem_read(pointer + 0x1C, 4), 'little') == record['version']
        return {'actor': values, 'lists': lists,
                'empty_literal_binding': self.rq(self.base + self.string_slot),
                'metadata_initialized': int.from_bytes(self.u.mem_read(self.base + next(iter(self.flags)), 1), 'little')}

    def event(self, kind, **details):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.events.append({'kind': kind, **details, 'snapshot': self.snapshot()})
        if self.options.get('failure') == [kind, self.counts[kind]]:
            self.error = kind
            self.u.emu_stop()
            return False
        for name, value in self.options.get('effects', {}).get(f'{kind}:{self.counts[kind]}', {}).items():
            if name == 'literal_binding': self.q(self.base + self.string_slot, self.alternate_string)
            else:
                offset, size, _ = self.fields[name]
                self.u.mem_write(self.actor + offset, int(value).to_bytes(size, 'little'))
                self.allowed.update(range(offset, offset + size))
        return True

    def hook(self, uc, address, size, data):
        if address == self.stop:
            self.returned = True
            uc.emu_stop()
            return
        rva = address - self.base
        if rva in self.decoded:
            self.visited.add(rva)
            return
        assert rva in self.services, hex(rva)
        x = self.x
        kind = self.services[rva]
        rcx, rdx = self.reg(x.UC_X86_REG_RCX), self.reg(x.UC_X86_REG_RDX)
        if kind == 'metadata':
            assert rcx - self.base in self.slot_values
            if self.event(kind, binding=next(r.get('Name', r.get('Value')) for r in self.bindings if r['Address'] == rcx - self.base)):
                self.ret(0xC0DE000000000001)
        elif kind == 'allocate':
            assert rcx == self.type_info
            index = self.counts.get(kind, 0)
            pointer = self.fresh_lists[index] if self.options.get('null_allocation') != index + 1 else 0
            if self.event(kind, result=pointer):
                if pointer:
                    self.u.mem_write(pointer, bytes(0x20))
                    self.lists.append({'identity': pointer, 'constructed': False, 'count': None, 'version': None, 'values': None})
                self.ret(pointer)
        elif kind == 'list_constructor':
            index = self.counts.get(kind, 0)
            expected = self.fresh_lists[index] if self.options.get('null_allocation') != index + 1 else 0
            assert rcx == expected and rdx == self.method_info
            if self.event(kind, receiver=rcx, generic_method=rdx):
                if not rcx:
                    self.error = 'authored_null_list_constructor'
                    uc.emu_stop()
                    return
                record = next(l for l in self.lists if l['identity'] == rcx)
                record.update(constructed=True, count=0, version=0, values=[])
                self.q(rcx + 0x10, self.empty_array)
                self.u.mem_write(rcx + 0x18, bytes(8))
                self.ret(self.options.get('constructor_return', 0xC0DE000000000002))
        elif kind == 'barrier':
            index = self.counts.get(kind, 0)
            assert rcx == self.actor + [0x148, 0x150, 0x198][index]
            expected = self.fresh_lists[index] if index < 2 else self.rq(self.base + self.string_slot)
            assert rdx == expected
            if self.event(kind, destination=rcx, value=rdx): self.ret(0xC0DE000000000003)
        else:
            assert rcx == self.actor and rdx == 0
            assert self.reg(x.UC_X86_REG_RSP) == self.initial_sp
            for register, value in self.nonvolatile.items(): assert self.reg(register) == value
            if self.event(kind, receiver=rcx, method=rdx): self.ret(0xC0DE000000000004)

    def run(self, options):
        self.options = options
        seed = options.get('seed', 'pattern')
        initial = bytes(0x1B8) if seed == 'zero' else bytes([0xA5] * 0x1B8) if seed == 'ones' else bytes((n * 37 + 11) & 255 for n in range(0x1B8))
        self.u.mem_write(self.actor, initial)
        for field, value in options.get('initial_fields', {}).items():
            offset, size, _ = self.fields[field]
            self.u.mem_write(self.actor + offset, value.to_bytes(size, 'little'))
        self.initial_bytes = bytes(self.u.mem_read(self.actor, 0x1B8))
        self.allowed = {n for offset, size, _ in self.fields.values() for n in range(offset, offset + size)}
        self.lists = [{'identity': p, 'constructed': True, 'count': 2, 'version': 17 + i,
                       'values': [self.arena + 0xA000 + i * 0x100, None]} for i, p in enumerate(self.old_lists)]
        for index, pointer in enumerate(self.old_lists):
            self.u.mem_write(pointer, bytes([0xA5] * 0x20))
            self.q(pointer + 0x10, self.arena + 0x11000 + index * 0x100)
            self.u.mem_write(pointer + 0x18, struct.pack('<II', 2, 17 + index))
        self.old_list_bytes = [bytes(self.u.mem_read(p, 0x20)) for p in self.old_lists]
        for slot, value in self.slot_values.items(): self.q(self.base + slot, value)
        for flag in self.flags: self.u.mem_write(self.base + flag, bytes([not options.get('cold', False)]))
        self.events, self.counts, self.error, self.returned = [], {}, None, False
        x = self.x
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                     x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15] + [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        self.nonvolatile = {r: 0x1234000000000000 + i for i, r in enumerate(registers)}
        for r, value in self.nonvolatile.items(): self.u.reg_write(r, value)
        self.initial_sp = self.stack + 0x1FF08
        self.q(self.initial_sp, self.stop)
        self.u.reg_write(x.UC_X86_REG_RSP, self.initial_sp)
        self.u.reg_write(x.UC_X86_REG_RCX, self.actor)
        self.u.reg_write(x.UC_X86_REG_RDX, 0xC0DE123400000005)
        self.u.emu_start(self.base + 0x3697C0, 0, count=10000)
        final_bytes = bytes(self.u.mem_read(self.actor, 0x1B8))
        assert all(a == b for n, (a, b) in enumerate(zip(self.initial_bytes, final_bytes)) if n not in self.allowed)
        assert self.lists[:2] == [{'identity': p, 'constructed': True, 'count': 2, 'version': 17 + i,
                                 'values': [self.arena + 0xA000 + i * 0x100, None]} for i, p in enumerate(self.old_lists)]
        assert all(bytes(self.u.mem_read(p, 0x20)) == old for p, old in zip(self.old_lists, self.old_list_bytes))
        if self.returned:
            assert self.reg(x.UC_X86_REG_RSP) == self.initial_sp + 8
            for r, value in self.nonvolatile.items(): assert self.reg(r) == value
        return {'input': options, 'initial': {name: int.from_bytes(self.initial_bytes[offset:offset + size], 'little')
                                            for name, (offset, size, _) in self.fields.items()},
                'returned': self.returned, 'error': self.error, 'events': self.events.copy(), 'final': self.snapshot(),
                'other_actor_bytes_retained': True, 'old_lists_retained': True}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, mutations, failures = [], [], []
    for seed, cold, constructor_return in itertools.product(['zero', 'ones', 'pattern'], [False, True], [0, 0xFFFFFFFFFFFFFFFF]):
        result = m.run({'seed': seed, 'cold': cold, 'constructor_return': constructor_return})
        assert result['returned'] and result['final']['actor'] == {'uses': 1, 'acted_infos': m.fresh_lists[0],
                                                                 'hover_infos': m.fresh_lists[1], 'saved_act': m.empty_string, 'act': 1}
        assert all(l['constructed'] and l['count'] == 0 for l in result['final']['lists'][2:])
        cases.append(result)
    for uses, act in itertools.product([0, 1, 0x80000000, 0xFFFFFFFF], [0, 1, 255]):
        result = m.run({'initial_fields': {'uses': uses, 'act': act, 'acted_infos': m.old_lists[0],
                                           'hover_infos': m.old_lists[1], 'saved_act': m.alternate_string}})
        assert result['returned'] and result['final']['actor']['uses'] == result['final']['actor']['act'] == 1
        cases.append(result)
    for index in [1, 2]:
        result = m.run({'null_allocation': index})
        assert not result['returned'] and result['error'] == 'authored_null_list_constructor'
        assert result['final']['actor']['uses'] == 1
        assert result['final']['actor']['acted_infos'] == (m.fresh_lists[0] if index == 2 else result['initial']['acted_infos'])
        cases.append(result)
    probes = [{'list_constructor:1': {'uses': 31, 'acted_infos': m.old_lists[0]}},
              {'barrier:1': {'acted_infos': m.old_lists[1]}},
              {'list_constructor:2': {'acted_infos': 0, 'hover_infos': m.old_lists[0]}},
              {'barrier:2': {'literal_binding': 1}},
              {'barrier:3': {'uses': 41, 'act': 0}},
              {'base_constructor:1': {'uses': 51, 'act': 0, 'saved_act': m.alternate_string}}]
    for effects in probes:
        result = m.run({'effects': effects})
        assert result['returned']
        mutations.append(result)
    assert mutations[0]['final']['actor']['uses'] == 31
    assert mutations[0]['final']['actor']['acted_infos'] == m.fresh_lists[0]
    assert mutations[1]['final']['actor']['acted_infos'] == m.old_lists[1]
    assert mutations[2]['final']['actor']['acted_infos'] == 0 and mutations[2]['final']['actor']['hover_infos'] == m.fresh_lists[1]
    assert mutations[3]['final']['actor']['saved_act'] == m.alternate_string
    assert mutations[4]['final']['actor']['uses'] == 41 and mutations[4]['final']['actor']['act'] == 1
    assert mutations[5]['final']['actor']['uses'] == 51 and mutations[5]['final']['actor']['act'] == 0
    baseline = m.run({'cold': True})
    counts = {}
    for index, event in enumerate(baseline['events']):
        kind = event['kind']
        counts[kind] = counts.get(kind, 0) + 1
        result = m.run({'cold': True, 'failure': [kind, counts[kind]]})
        assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
        assert result['final'] == event['snapshot']
        failures.append({'failure': [kind, counts[kind]], 'prefix_length': index + 1, 'exact_snapshot_verified': True})
    return {'build': BUILD, 'method': m.method, 'base_constructor_gateway': m.base_method,
            'native_range': ['0x3697c0', '0x3698a3'], 'next_managed_entry': '0x3698b0',
            'fields': {name: declaration for name, (_, _, declaration) in m.fields.items()},
            'metadata_bindings': m.bindings, 'instruction_assertions': m.instruction_assertions,
            'authored_list_storage': {'items': '0x10', 'count': '0x18', 'version': '0x1c',
                                     'empty_array_identity': m.empty_array,
                                     'scope': 'Explicit service storage, not generic dump.cs zero offsets or reconstructed List constructor internals.'},
            'case_count': len(cases), 'cases': cases, 'mutation_case_count': len(mutations), 'mutation_cases': mutations,
            'failure_baseline': baseline, 'failure_case_count': len(failures), 'failure_cases': failures,
            'native_instructions_executed': len(m.visited), 'native_instruction_count': len(m.decoded),
            'unexecuted_native_addresses': [hex(n) for n in sorted(set(m.decoded) - m.visited)],
            'limits': ['Complete Character constructor executes; allocation, generic List constructor, metadata, barriers and MonoBehaviour base body remain explicit authored services.',
                       'Allocated List records are physical identities with authored empty-list construction; their runtime backing allocation implementation is not reconstructed.',
                       'Constructor return registers are poisoned to establish that publication uses allocation identities, not void constructor return values.',
                       'Null allocation probes stop at the authored List constructor gateway, not a reconstructed runtime exception path.',
                       'Callbacks mutate only declared fields at supplied API boundaries; exact caller reload/write order and retained bytes are checked.',
                       'No scene load, constructor invocation provenance, initializer join or actual scheduler registration is inferred.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'mutation_case_count', 'failure_case_count', 'native_instructions_executed']}))
