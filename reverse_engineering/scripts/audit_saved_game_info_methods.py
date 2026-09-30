"""Execute all five pinned SavedGameInfo caller methods and join native JSON."""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_info_json import Machine as JsonMachine, verify_layout


METHODS = {'AddTutorial': (0x3EAC30, 0x18), 'AddCharacter': (0x3EAB50, 0x20),
           'ClearTutorials': (0x3EAD10, 0x18), 'ClearUnlockedCharacters': (0x3EAD70, 0x20),
           '.ctor': (0x3EADD0, None)}


class Machine:
    def __init__(self, game_root, dumper_root):
        import capstone
        import pefile
        import unicorn
        from unicorn import x86_const as x
        self.layout = verify_layout(game_root, dumper_root)
        self.x, self.capstone, self.unicorn = x, capstone, unicorn
        assert unicorn.__version__ == '2.1.4'
        root = Path(__file__).parents[1]
        extraction = json.loads((root / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        raw = (Path(dumper_root) / 'script.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs']['script_json']['sha256'].upper()
        script = json.loads(raw.decode('utf-8-sig'))
        self.targets = []
        for name, (rva, _) in METHODS.items():
            rows = [r for r in script['ScriptMethod'] if r['Name'] == 'SavedGameInfo$$' + name and r['Address'] == rva]
            assert len(rows) == 1
            signature = ('void SavedGameInfo___ctor (SavedGameInfo_o* __this, const MethodInfo* method);' if name == '.ctor' else
                         f'void SavedGameInfo__{name} (SavedGameInfo_o* __this, ' +
                         (f'System_String_o* {"newTutorial" if name == "AddTutorial" else "newCharacter"}, ' if name.startswith('Add') else '') +
                         'const MethodInfo* method);')
            assert rows[0]['Signature'] == signature
            self.targets.append(rows[0])
        self.pe = pefile.PE(data=(Path(game_root) / 'GameAssembly.dll').read_bytes(), fast_load=True)
        self.pe.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        self.cs.detail = True
        self.instructions, self.ranges, self.flags = {}, {}, {}
        for name, (rva, _) in METHODS.items():
            entry = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == rva)
            assert not entry.unwindinfo.Flags & 4
            a, b = entry.struct.BeginAddress, entry.struct.EndAddress
            raw = self.pe.get_data(a, b - a)
            decoded = list(self.cs.disasm(raw, a))
            assert sum(i.size for i in decoded) == b - a
            self.instructions.update({i.address: i for i in decoded})
            self.ranges[name] = [hex(a), hex(b)]
            first = next(i for i in decoded if i.mnemonic == 'cmp' and i.operands[0].type == capstone.x86.X86_OP_MEM and i.operands[0].mem.base == capstone.x86.X86_REG_RIP)
            self.flags[name] = first.address + first.size + first.operands[0].mem.disp
        checks = {
            0x3EAB9F: ('call', '0xb55950'),
            0x3EABB8: ('inc', 'dword ptr [rcx + 0x1c]'),
            0x3EABE0: ('call', '0xb54090'),
            0x3EABF4: ('mov', 'dword ptr [rcx + 0x18], eax'),
            0x3EAC08: ('mov', 'qword ptr [rcx], rbx'),
            0x3EAC7F: ('call', '0xb55950'),
            0x3EAC98: ('inc', 'dword ptr [rcx + 0x1c]'),
            0x3EACC0: ('call', '0xb54090'),
            0x3EACD4: ('mov', 'dword ptr [rcx + 0x18], eax'),
            0x3EACE8: ('mov', 'qword ptr [rcx], rbx'),
            0x3EAD42: ('inc', 'dword ptr [rcx + 0x1c]'),
            0x3EAD45: ('mov', 'dword ptr [rcx + 0x18], 0'),
            0x3EAD5F: ('jmp', '0x112b9d0'),
            0x3EADA2: ('inc', 'dword ptr [rcx + 0x1c]'),
            0x3EADA5: ('mov', 'dword ptr [rcx + 0x18], 0'),
            0x3EADBF: ('jmp', '0x112b9d0'),
            0x3EAE1C: ('mov', 'qword ptr [rcx], rax'),
            0x3EAE32: ('call', '0x2b7d40'),
            0x3EAE44: ('call', '0xb02160'),
            0x3EAE50: ('mov', 'qword ptr [rcx], rbx'),
            0x3EAE5F: ('call', '0x2b7d40'),
            0x3EAE71: ('call', '0xb02160'),
            0x3EAE7D: ('mov', 'qword ptr [rcx], rbx'),
            0x3EAE94: ('jmp', '0x33ed50'),
        }
        for address, expected in checks.items():
            assert address in self.instructions
            i = self.instructions[address]
            assert (i.mnemonic, i.op_str) == expected, (hex(address), i.op_str)
        leaf = list(self.cs.disasm(self.pe.get_data(0x33ED50, 3), 0x33ED50))
        assert [(i.mnemonic, i.op_str) for i in leaf] == [('ret', '0')]
        self.instruction_assertions = len(checks) + 1
        slots = {r['Address']: ('metadata', r['Name']) for r in script['ScriptMetadata']}
        slots.update({r['Address']: ('method', r['Name']) for r in script['ScriptMetadataMethod']})
        slots.update({r['Address']: ('string', r['Value']) for r in script['ScriptString']})
        used = set()
        for i in self.instructions.values():
            for operand in i.operands:
                if operand.type == capstone.x86.X86_OP_MEM and operand.mem.base == capstone.x86.X86_REG_RIP:
                    target = i.address + i.size + operand.mem.disp
                    if target in slots:
                        used.add(target)
        self.bindings = {a: slots[a] for a in sorted(used)}
        assert len(self.bindings) == 6
        self.key_literal = next(value for kind, value in self.bindings.values() if kind == 'string')
        assert self.key_literal == 'Tutorials'
        self.u = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        self.u.mem_map(self.base, (self.pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
        self.u.mem_write(self.base, self.pe.get_memory_mapped_image())
        self.arena, self.stack, self.stop = 0x300000000, 0x400000000, 0x500000000
        self.u.mem_map(self.arena, 0x200000)
        self.u.mem_map(self.stack, 0x20000)
        self.u.mem_map(self.stop, 0x1000)
        self.saved = self.arena + 0x1000
        self.executed = set()
        self.u.hook_add(unicorn.UC_HOOK_CODE, self.hook)

    def q(self, a, value):
        self.u.mem_write(a, struct.pack('<Q', value))

    def rq(self, a):
        return struct.unpack('<Q', self.u.mem_read(a, 8))[0]

    def d(self, a, value):
        self.u.mem_write(a, struct.pack('<I', value & 0xFFFFFFFF))

    def rd(self, a):
        return struct.unpack('<I', self.u.mem_read(a, 4))[0]

    def reg(self, register):
        return self.u.reg_read(register)

    def ret(self, value=0):
        x = self.x
        sp = self.reg(x.UC_X86_REG_RSP)
        self.u.reg_write(x.UC_X86_REG_RAX, value)
        self.u.reg_write(x.UC_X86_REG_RSP, sp + 8)
        self.u.reg_write(x.UC_X86_REG_RIP, self.rq(sp))

    def string(self, value):
        if value is None:
            return 0
        token = self.string_cursor
        self.string_cursor += 0x40
        self.strings[token] = value
        return token

    def array(self, values, capacity):
        assert len(values) <= capacity <= 256
        token = self.array_cursor
        self.array_cursor += (0x40 + capacity * 8 + 15) & ~15
        self.u.mem_write(token, bytes(0x40 + capacity * 8))
        self.d(token + 0x18, capacity)
        for i, value in enumerate(values):
            self.q(token + 0x20 + i * 8, value)
        self.arrays[token] = capacity
        return token

    def list(self, values=None, capacity=None, version=0):
        token = self.list_cursor
        self.list_cursor += 0x80
        self.u.mem_write(token, bytes(0x40))
        self.lists.append(token)
        if values is not None:
            raw = [self.string(value) for value in values]
            self.q(token + 0x10, self.array(raw, len(raw) if capacity is None else capacity))
            self.d(token + 0x18, len(raw))
        self.d(token + 0x1C, version)
        return token

    def list_state(self, token):
        if not token:
            return None
        assert token in self.lists
        backing, size, version = self.rq(token + 0x10), self.rd(token + 0x18), self.rd(token + 0x1C)
        capacity = self.arrays[backing] if backing else None
        values = [self.strings.get(self.rq(backing + 0x20 + i * 8)) for i in range(size)] if backing else None
        raw = [self.strings.get(self.rq(backing + 0x20 + i * 8)) for i in range(capacity)] if backing else None
        assert size <= (capacity or 0)
        return {'identity': self.lists.index(token), 'count': size, 'capacity': capacity,
                'version': version, 'values': values, 'backing_values': raw}

    def snapshot(self):
        return {'key': self.strings.get(self.rq(self.saved + 0x10)),
                'completedTutorials': self.list_state(self.rq(self.saved + 0x18)),
                'unlockedCharactersId': self.list_state(self.rq(self.saved + 0x20)),
                'allocated_lists': [self.list_state(token) for token in self.lists],
                'metadata_initialized': {name: bool(self.u.mem_read(self.base + rva, 1)[0]) for name, rva in self.flags.items()}}

    def event(self, kind, args):
        self.events.append({'kind': kind, 'args': args, 'snapshot': self.snapshot()})
        self.counts[kind] = self.counts.get(kind, 0) + 1
        if self.failure == [kind, self.counts[kind]]:
            self.error = kind
            self.u.emu_stop()
            return False
        return True

    def hook(self, uc, address, size, data):
        rva = address - self.base
        self.executed.add(rva)
        x = self.x
        cx, dx, r8, r9 = [self.reg(r) for r in
                         (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        if rva == 0x2B7B40:
            assert cx - self.base in self.bindings
            if self.event('metadata_initialize_service', [self.bindings[cx - self.base][1]]):
                self.ret()
        elif rva == 0xB55950:
            assert cx in self.lists and r8 == self.method_tokens['Contains']
            value = self.strings.get(dx)
            state = self.list_state(cx)
            if self.event('contains_service', [state['identity'], value]):
                self.ret(value in state['values'])
        elif rva == 0xB54090:
            assert cx in self.lists and r8 == self.resize_method
            if self.event('resize_append_service', [self.lists.index(cx), self.strings.get(dx)]):
                old, count = self.rq(cx + 0x10), self.rd(cx + 0x18)
                values = [self.rq(old + 0x20 + i * 8) for i in range(count)] + [dx]
                capacity = max(4, self.arrays[old] * 2, count + 1)
                self.q(cx + 0x10, self.array(values, capacity))
                self.d(cx + 0x18, count + 1)
                self.ret()
        elif rva == 0x2B6FF0:
            assert self.rq(cx) == dx
            if self.event('write_barrier_service', ['stored_reference']):
                self.ret()
        elif rva == 0x112B9D0:
            assert cx in self.arrays and dx == 0 and r9 == 0 and r8 <= self.arrays[cx]
            if self.event('array_clear_service', [r8]):
                if r8:
                    self.u.mem_write(cx + 0x20, bytes(r8 * 8))
                self.ret()
        elif rva == 0x2B7D40:
            assert cx == self.list_type
            if self.event('list_allocate_service', []):
                self.ret(self.list())
        elif rva == 0xB02160:
            assert cx in self.lists and dx == self.method_tokens['.ctor']
            if self.event('list_constructor_service', [self.lists.index(cx)]):
                self.q(cx + 0x10, self.empty_array)
                self.d(cx + 0x18, 0)
                self.d(cx + 0x1C, 0)
                self.ret()
        elif rva in (0x2B7D90, 0x2B7D80):
            kind = 'null_reference' if rva == 0x2B7D90 else 'bounds_failure'
            self.event(kind, [])
            self.error = kind
            self.u.emu_stop()

    def run(self, method, state=None, new_value=None, options=None):
        options = options or {}
        state = state or {}
        self.u.mem_write(self.arena, bytes(0x80000))
        self.strings, self.lists, self.arrays = {}, [], {}
        self.string_cursor, self.list_cursor, self.array_cursor = [self.arena + n for n in (0x10000, 0x20000, 0x30000)]
        self.list_type = self.arena + 0x100000
        self.method_tokens = {name: self.arena + 0x101000 + i * 0x1000 for i, name in enumerate(['Add', 'Contains', 'Clear', '.ctor'])}
        context, table, self.resize_method = [self.arena + a for a in (0x110000, 0x111000, 0x112000)]
        self.q(self.method_tokens['Add'] + 0x20, context)
        self.q(context + 0xC0, table)
        self.q(table + 0x70, self.resize_method)
        literal = self.string(self.key_literal)
        for rva, (kind, name) in self.bindings.items():
            if kind == 'string':
                token = literal
            elif kind == 'metadata':
                assert name == 'System.Collections.Generic.List<string>_TypeInfo'
                token = self.list_type
            else:
                assert name.startswith('Method$System.Collections.Generic.List<string>.')
                token = self.method_tokens[name.split('List<string>.', 1)[1].split('(', 1)[0]]
            self.q(self.base + rva, token)
        for rva in self.flags.values():
            self.u.mem_write(self.base + rva, bytes([int(options.get('warm', False))]))
        self.empty_array = self.array([], 0)
        self.q(self.saved + 0x10, self.string(state.get('key')))
        for name, offset in [('completedTutorials', 0x18), ('unlockedCharactersId', 0x20)]:
            values = state.get(name)
            token = self.list(values, options.get('capacity'), options.get('version', 17)) if values is not None else 0
            self.q(self.saved + offset, token)
        argument = self.string(new_value)
        self.events, self.counts = [], {}
        self.failure, self.error = options.get('failure'), None
        x = self.x
        sp = self.stack + 0x18008
        self.q(sp, self.stop)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                     x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(registers):
            self.u.reg_write(register, 0xFAB00000 + i)
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, self.saved),
                                (x.UC_X86_REG_RDX, argument), (x.UC_X86_REG_R8, 0)]:
            self.u.reg_write(register, value)
        try:
            self.u.emu_start(self.base + METHODS[method][0], self.stop, timeout=2_000_000, count=100000)
        except Exception as exc:
            raise AssertionError(f'{method}, {options}, RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(registers):
                assert self.reg(register) == 0xFAB00000 + i
        final = self.snapshot()
        return {'method': method, 'input_state': state, 'argument': new_value, 'options': options,
                'events': self.events.copy(), 'returned': returned, 'error': self.error, 'final': final}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, failures, joined = [], [], []
    for method in ['AddTutorial', 'AddCharacter']:
        field = 'completedTutorials' if method == 'AddTutorial' else 'unlockedCharactersId'
        for warm, initial, value, spare in itertools.product([False, True], [[], ['first'], [None]], [None, 'first', 'second'], [False, True]):
            state = {'key': 'fixture', 'completedTutorials': [], 'unlockedCharactersId': []}
            state[field] = initial
            result = m.run(method, state, value, {'warm': warm, 'capacity': max(len(initial), 0) + int(spare)})
            final = result['final'][field]
            duplicate = value in initial
            assert result['returned'] and final['values'] == initial + ([] if duplicate else [value])
            assert final['version'] == 17 + int(not duplicate)
            assert result['final']['key'] == 'fixture'
            cases.append(result)
    for method in ['ClearTutorials', 'ClearUnlockedCharacters']:
        field = 'completedTutorials' if method == 'ClearTutorials' else 'unlockedCharactersId'
        for warm, initial, version, spare in itertools.product([False, True], [[], ['first'], ['first', None]], [0, 0xFFFFFFFF], [False, True]):
            state = {'key': 'fixture', 'completedTutorials': [], 'unlockedCharactersId': []}
            state[field] = initial
            result = m.run(method, state, options={'warm': warm, 'capacity': len(initial) + int(spare), 'version': version})
            final = result['final'][field]
            assert result['returned'] and final['values'] == [] and final['version'] == (version + 1) & 0xFFFFFFFF
            assert final['backing_values'] == [None] * (len(initial) + int(spare))
            cases.append(result)
    for warm, existing in itertools.product([False, True], [False, True]):
        state = {'key': 'old', 'completedTutorials': ['old'], 'unlockedCharactersId': ['old']} if existing else {}
        result = m.run('.ctor', state, options={'warm': warm})
        assert result['returned'] and result['final']['key'] == 'Tutorials'
        assert result['final']['completedTutorials']['values'] == result['final']['unlockedCharactersId']['values'] == []
        assert result['final']['completedTutorials']['identity'] != result['final']['unlockedCharactersId']['identity']
        cases.append(result)
    for method, warm in itertools.product(list(METHODS)[:-1], [False, True]):
        result = m.run(method, {}, options={'warm': warm})
        assert not result['returned'] and result['error'] == 'null_reference'
        cases.append(result)
    configurations = [('AddTutorial', {'key': 'fixture', 'completedTutorials': [], 'unlockedCharactersId': []}, 'new', {'capacity': 1}),
                      ('AddCharacter', {'key': 'fixture', 'completedTutorials': [], 'unlockedCharactersId': []}, 'new', {'capacity': 0}),
                      ('ClearTutorials', {'key': 'fixture', 'completedTutorials': ['old'], 'unlockedCharactersId': []}, None, {'capacity': 1}),
                      ('.ctor', {}, None, {})]
    for method, state, value, options in configurations:
        baseline = m.run(method, state, value, options)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run(method, state, value, dict(options, failure=[kind, counts[kind]]))
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append(result)
    engine = JsonMachine(game_root)
    for method, warm in itertools.product(METHODS, [False, True]):
        state = {'key': 'fixture', 'completedTutorials': ['old-t'], 'unlockedCharactersId': ['old-c']}
        native = m.run(method, state, 'new', {'warm': warm, 'capacity': 3})
        assert native['returned']
        final = native['final']
        values = {'key': final['key'], 'completedTutorials': final['completedTutorials']['values'],
                  'unlockedCharactersId': final['unlockedCharactersId']['values']}
        serialized = engine.write_saved(values)
        assert serialized['returned'] and serialized['loaded']['values'] == values
        joined.append({'native_caller': native, 'engine_json': serialized,
                       'adapter': 'Exact three public values transferred once; runtime object identities and private List versions are not projected.'})
    return {'build_id': BUILD, 'pinned_layout': m.layout, 'targets': m.targets,
            'native_ranges': m.ranges, 'metadata_bindings': [{'rva': hex(a), 'kind': k, 'name': n} for a, (k, n) in m.bindings.items()],
            'instruction_assertions': m.instruction_assertions,
            'cases': cases, 'case_count': len(cases), 'failure_cases': failures, 'failure_case_count': len(failures),
            'joined_cases': joined, 'joined_case_count': len(joined), 'executed_address_count': len(m.executed),
            'scope': 'All five native SavedGameInfo caller bodies, including inline append/clear stores and constructor publication. Contains, resize growth, array clearing, List allocation/constructor, metadata initialization and write barriers are explicit services. Joined JSON transfers exact public values into the independently audited native engine pipeline. No persistence, arbitrary runtime graph copy or native managed exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['failure_case_count'], result['joined_case_count'], result['executed_address_count'])
