"""Execute eight exact CharacterData consumers; Unity/runtime services supplied."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine


# Complete bodies, including terminal native guard traps. Leaf bodies end at ret.
TARGETS = {
    'GetCharacterName': (0x3B4BE0, 0x3B4BE5, 0x3B4BF0, 0, 'System_String_o*', 160),
    'GetIWas': (0x33E8D0, 0x33E8D5, 0x33E8E0, 1, 'System_String_o*', 136),
    'GetGender': (0x3B4CF0, 0x3B4CF4, 0x3B4D00, 2, 'int32_t', 30),
    'GetTranslation': (0x3B4DA0, 0x3B4DA8, 0x3B4DB0, 6, 'CharacterLoc_o*', 8),
    'GetArt': (0x3B4AB0, 0x3B4B39, 0x3B4B40, 7, 'UnityEngine_Sprite_o*', 1),
    'GetAnimatedArt': (0x3B4990, 0x3B4A19, 0x3B4A20, 8, 'UnityEngine_Sprite_o*', 1),
    'GetArtType': (0x3B4A20, 0x3B4AA3, 0x3B4AB0, 9, 'int32_t', 1),
    'GetArtistName': (0x3B4B40, 0x3B4BD7, 0x3B4BE0, 18, 'System_String_o*', 1)}
ART_METHODS = ['GetArt', 'GetAnimatedArt', 'GetArtType', 'GetArtistName']
LEAF_FIELDS = {'GetCharacterName': 0x28, 'GetIWas': 0x30, 'GetGender': 0x38, 'GetTranslation': 0x148}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root)
        manifest = json.loads((Path(__file__).parents[1] /
            f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == manifest['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json'))
        dump = pin('dump.cs', 'dump_cs')
        def block(pattern):
            b = re.search(pattern, dump, re.M | re.S); assert b
            return b[1]
        data = block(r'^public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845\s*\{(.*?)\n\}')
        self.data_fields = {n: int(off, 16) for _, n, off in re.findall(r'public ([\w<>\[\]]+) (\w+); // (0x[\dA-Fa-f]+)', data.split('// Methods')[0])}
        for line in ['public string characterName; // 0x28', 'public string iWasName; // 0x30', 'public EGender gender; // 0x38',
                     'public Sprite art_cute; // 0x98', 'public Sprite art_animated; // 0xA8',
                     'public SkinData currentSkin; // 0xC0', 'public CharacterLoc translation; // 0x148']:
            assert line in data
        skin = block(r'^public class SkinData : ScriptableObject // TypeDefIndex: 5945\s*\{(.*?)\n\}')
        for line in ['public string artistName; // 0x20', 'public Sprite art; // 0x38',
                     'public Sprite animated_art; // 0x40', 'public EArtType type; // 0x50']:
            assert line in skin
        art_type = block(r'^public enum EArtType // TypeDefIndex: 5946\s*\{(.*?)\n\}')
        assert 'public const EArtType Default = 0;' in art_type and 'public const EArtType Clipping = 10;' in art_type
        gender = block(r'^public enum EGender // TypeDefIndex: 5956\s*\{(.*?)\n\}')
        assert all(f'public const EGender {n} = {v};' in gender for n, v in [('Female', 0), ('Male', 10), ('They', 20)])
        assert re.search(r'^public class CharacterLoc // TypeDefIndex: 5972$', dump, re.M)
        self.targets, self.instructions, self.bounds, self.ranges, self.flags, self.bindings = [], {}, {}, {}, {}, {}
        slots = {r['Address']: ('metadata', r['Name']) for r in self.metadata['ScriptMetadata']}
        slots.update({r['Address']: ('string', r['Value']) for r in self.metadata['ScriptString']})
        for name, (start, end, following, ordinal, ret, aliases) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start and r['Name'] == 'CharacterData$$' + name]
            assert len(rows) == 1 and rows[0]['TypeSignature'] == 'iii'
            assert rows[0]['Signature'] == f'{ret} CharacterData__{name} (CharacterData_o* __this, const MethodInfo* method);'
            assert len([r for r in self.metadata['ScriptMethod'] if r['Address'] == start]) == aliases
            self.targets.append(dict(rows[0], method_id=f'tdi5845.m{ordinal:04}', shared_rva_declaration_count=aliases))
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start) == following
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            if name in ART_METHODS: assert chunks == [(start, end)]
            else: assert not chunks
            self.ranges[name] = [[hex(a), hex(b)] for a, b in chunks]
            section = self.pe.get_section_by_rva(start)
            assert section and following <= section.VirtualAddress + section.SizeOfRawData
            raw = self.pe.get_data(start, following-start)
            assert len(raw) == following-start and raw[end-start:] == b'\xcc' * (following-end)
            ins = list(self.cs.disasm(raw[:end-start], start))
            assert sum(i.size for i in ins) == end-start
            assert ins[-1].mnemonic == ('int3' if name in ART_METHODS else 'ret')
            self.instructions.update({i.address: i for i in ins})
            self.bounds[name] = {'start': hex(start), 'end_exclusive': hex(end), 'next_managed': hex(following), 'padding_bytes': following-end}
            for i in ins:
                for op in i.operands:
                    if op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                        slot = i.address+i.size+op.mem.disp
                        if slot in slots: self.bindings[slot] = slots[slot]
                        elif i.mnemonic == 'cmp' and i.operands[0].size == 1: self.flags[name] = slot
        assert set(self.flags) == set(ART_METHODS)
        assert self.bindings == {0x2718BF0: ('metadata', 'UnityEngine.Object_TypeInfo'), 0x2714768: ('string', 'normandia')}
        self.supplied = []
        for address, name in [(0x1C822C0, 'UnityEngine.Object$$op_Equality'), (0x1C82480, 'UnityEngine.Object$$op_Inequality')]:
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['TypeSignature'] == 'iiii'
            symbol = 'op_Equality' if address == 0x1C822C0 else 'op_Inequality'
            assert rows[0]['Signature'] == f'bool UnityEngine_Object__{symbol} (UnityEngine_Object_o* x, UnityEngine_Object_o* y, const MethodInfo* method);'
            self.supplied += rows
        self.checks = {
            0x3B4BE0: ('mov', 'rax, qword ptr [rcx + 0x28]'), 0x33E8D0: ('mov', 'rax, qword ptr [rcx + 0x30]'),
            0x3B4CF0: ('mov', 'eax, dword ptr [rcx + 0x38]'), 0x3B4DA0: ('mov', 'rax, qword ptr [rcx + 0x148]'),
            0x3B4AE0: ('mov', 'rdi, qword ptr [rbx + 0xc0]'), 0x3B4AFD: ('call', '0x1c822c0'),
            0x3B4B02: ('test', 'al, al'), 0x3B4B06: ('mov', 'rax, qword ptr [rbx + 0xc0]'),
            0x3B4B12: ('mov', 'rax, qword ptr [rax + 0x38]'), 0x3B4B21: ('mov', 'rax, qword ptr [rbx + 0x98]'),
            0x3B49C0: ('mov', 'rdi, qword ptr [rbx + 0xc0]'), 0x3B49DD: ('call', '0x1c822c0'),
            0x3B49E2: ('test', 'al, al'), 0x3B49F2: ('mov', 'rax, qword ptr [rax + 0x40]'),
            0x3B4A01: ('mov', 'rax, qword ptr [rbx + 0xa8]'),
            0x3B4A50: ('mov', 'rdi, qword ptr [rbx + 0xc0]'), 0x3B4A6D: ('call', '0x1c822c0'),
            0x3B4A72: ('test', 'al, al'), 0x3B4A82: ('mov', 'eax, dword ptr [rax + 0x50]'),
            0x3B4A95: ('xor', 'eax, eax'),
            0x3B4B81: ('mov', 'rbx, qword ptr [rip + 0x235fbe0]'),
            0x3B4B88: ('mov', 'rsi, qword ptr [rdi + 0xc0]'), 0x3B4BA5: ('call', '0x1c82480'),
            0x3B4BAA: ('test', 'al, al'), 0x3B4BAE: ('mov', 'rax, qword ptr [rdi + 0xc0]'),
            0x3B4BBA: ('mov', 'rbx, qword ptr [rax + 0x20]')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        for name in ART_METHODS:
            a, b, *_ = TARGETS[name]
            calls = [i.op_str for i in self.instructions.values() if a <= i.address < b and i.mnemonic == 'call']
            assert calls.count('0x2b7b40') == (2 if name == 'GetArtistName' else 1)
            assert calls.count('0x281d90') == 1 and calls.count('0x2b7d90') == 1
            assert calls.count('0x1c82480' if name == 'GetArtistName' else '0x1c822c0') == 1
        names = ['data', 'skin', 'other_skin', 'object_class', 'name', 'other_name', 'i_was', 'translation',
                 'default_art', 'default_animated', 'skin_art', 'skin_animated', 'other_art', 'other_animated',
                 'artist', 'other_artist', 'normandia']
        self.p = {n: self.arena+0x90000+i*0x1000 for i, n in enumerate(names)}
        self.ids = {p: n for n, p in self.p.items()}
        self.sizes = {n: 0x180 if n == 'data' else 0x100 if n in ['skin', 'other_skin', 'object_class'] else 0x80 for n in names}
        self.entry_sp = self.stack+0x18008
        self.tracking_writes = False
        self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE, self.observe_write)

    def observe_write(self, uc, access, address, size, value, user_data):
        if not self.tracking_writes: return
        flag = next((n for n, a in self.flags.items() if address == self.base+a), None)
        if flag is not None:
            ins = self.instructions[self.reg(self.x.UC_X86_REG_RIP)-self.base]
            assert size == 1 and value == 1 and ins.mnemonic == 'mov'
            self.reached_flag_writes[flag] = value

    def oid(self, value):
        if not value: return None
        assert value in self.ids, hex(value)
        return self.ids[value]

    def snapshot(self):
        return {'current_skin': self.oid(self.rq(self.p['data']+0xC0)),
                'metadata_flags_u8': {n: self.u.mem_read(self.base+a, 1)[0] for n, a in self.flags.items()},
                'metadata_slots': {n: self.oid(self.rq(self.base+a)) for a, (_, n) in self.bindings.items()},
                'object_class_word_u32': self.rd(self.p['object_class']+0xE0),
                'comparisons': self.comparisons.copy(),
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        self.comparisons = []
        for n, p in self.p.items(): self.u.mem_write(p, bytes([0xA5])*self.sizes[n])
        for off, name in [(0x28, 'name'), (0x30, 'i_was'), (0x148, 'translation'), (0x98, 'default_art'), (0xA8, 'default_animated')]:
            self.q(self.p['data']+off, 0 if options.get('null_return_fields') else self.p[name])
        self.d(self.p['data']+0x38, options.get('gender_bits', 20))
        initial_skin = options.get('initial_skin', 'skin')
        assert initial_skin in [None, 'skin', 'other_skin']
        self.q(self.p['data']+0xC0, 0 if initial_skin is None else self.p[initial_skin])
        for skin, artist, art, animated, bits in [('skin', 'artist', 'skin_art', 'skin_animated', options.get('art_type_bits', 10)),
                                               ('other_skin', 'other_artist', 'other_art', 'other_animated', 0x8000000A)]:
            for off, name in [(0x20, artist), (0x38, art), (0x40, animated)]:
                self.q(self.p[skin]+off, 0 if options.get('null_return_fields') else self.p[name])
            self.d(self.p[skin]+0x50, bits)
        if options.get('alias_outputs'):
            self.q(self.p['data']+0x30, self.p['name'])
            for off in [0x98, 0xA8]: self.q(self.p['data']+off, self.p['skin_art'])
            for skin in ['skin', 'other_skin']:
                self.q(self.p[skin]+0x20, self.p['name'])
                for off in [0x38, 0x40]: self.q(self.p[skin]+off, self.p['skin_art'])
        self.d(self.p['object_class']+0xE0, options.get('class_word', 0))
        self.q(self.base+0x2718BF0, self.p['object_class']); self.q(self.base+0x2714768, self.p['normandia'])
        # Exact captured literal pointer; no native string decoder is supplied.
        self.d(self.p['normandia']+0x10, len('normandia'))
        self.u.mem_write(self.p['normandia']+0x14, 'normandia'.encode('utf-16-le')+b'\0\0')
        for a in self.flags.values(): self.u.mem_write(self.base+a, bytes([options.get('warm_flag', 0)]))
        assert 0 <= options.get('comparison_true_byte', 0xFE) <= 255

    def mutate(self, phase, captured=0):
        if self.options.get('mutation_phase') != phase: return
        action = self.options['mutation']
        def write(n, off, value):
            self.q(self.p[n]+off, value)
            self.completed_memory_writes.setdefault(n, set()).update(range(off, off+8))
        if action in ['replace_skin', 'clear_skin', 'same_skin']:
            value = self.p['other_skin'] if action == 'replace_skin' else 0 if action == 'clear_skin' else captured
            write('data', 0xC0, value)
        elif action in ['replace_default', 'clear_default']:
            for off in [0x98, 0xA8]: write('data', off, self.p['other_art'] if action == 'replace_default' else 0)
        elif action in ['replace_skin_output', 'clear_skin_output']:
            for off in [0x20, 0x38, 0x40]: write('skin', off, self.p['other_artist'] if off == 0x20 and action == 'replace_skin_output' else self.p['other_art'] if action == 'replace_skin_output' else 0)
        elif action == 'replace_literal':
            self.q(self.base+0x2714768, self.p['other_artist'])
            self.completed_slot_writes['normandia'] = 'other_artist'
        else: raise AssertionError(action)

    def ret(self, value=0):
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']:
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_'+n), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{i}'), (1 << 127) | i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        rva, x = address-self.base, self.x
        self.executed.add(rva)
        if rva in self.instructions: return
        cx, dx, r8 = [self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8']]
        if rva == 0x2B7B40:
            assert cx-self.base in self.bindings
            if self.event('metadata_service', [self.bindings[cx-self.base][1]]):
                self.mutate('metadata', self.rq(self.p['data']+0xC0)); self.ret(self.rq(cx))
        elif rva == 0x281D90:
            assert cx == self.p['object_class'] and self.rd(cx+0xE0) == 0
            if self.event('object_class_initialize_service', [self.oid(cx)]):
                self.d(cx+0xE0, 1)
                self.completed_memory_writes.setdefault('object_class', set()).update(range(0xE0, 0xE4))
                self.mutate('class_init', self.rq(self.p['data']+0xC0)); self.ret(cx)
        elif rva in [0x1C822C0, 0x1C82480]:
            assert cx in [0, self.p['skin'], self.p['other_skin']] and dx == r8 == 0
            live = bool(cx) and self.options.get('skin_live', True)
            truth = live if rva == 0x1C82480 else not live
            if 'forced_comparison_bits' in self.options: result = self.options['forced_comparison_bits']
            else: result = 0xFACE123456789000 | (self.options.get('comparison_true_byte', 0xFE) if truth else 0)
            kind = 'object_inequality_service' if rva == 0x1C82480 else 'object_equality_service'
            args = [self.oid(cx), None, r8, result]
            if self.event(kind, args):
                self.comparisons.append(args); self.mutate('comparison', cx); self.ret(result)
        elif rva == 0x2B7D90:
            self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def invoke(self, name):
        x, sp = self.x, self.entry_sp
        self.q(sp, self.stop); self.u.mem_write(sp+8, bytes([0xB6])*0x38)
        regs = [getattr(x, 'UC_X86_REG_'+n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, r in enumerate(regs): self.u.reg_write(r, 0xFAB00000+i)
        for i, r in enumerate(vectors): self.u.reg_write(r, (0xABCD0000+i)|((0xFEDC0000+i)<<64))
        for r, v in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, 0 if self.options.get('null_owner') else self.p['data']),
                     (x.UC_X86_REG_RDX, 0xDEADBEEF12345678)]: self.u.reg_write(r, v)
        try: self.u.emu_start(self.base+TARGETS[name][0], self.stop, timeout=10_000_000, count=2000)
        except self.unicorn.UcError as exc:
            assert self.options.get('null_owner') and exc.errno == self.unicorn.UC_ERR_READ_UNMAPPED
            pc = self.reg(x.UC_X86_REG_RIP)-self.base
            expected = TARGETS[name][0] if name in LEAF_FIELDS else {'GetArt': 0x3B4AE0, 'GetAnimatedArt': 0x3B49C0, 'GetArtType': 0x3B4A50, 'GetArtistName': 0x3B4B88}[name]
            assert pc == expected
            self.error = 'native_owner_access_fault'
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        result = self.reg(x.UC_X86_REG_RAX) if returned else None
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp+8
            assert all(self.reg(r) == 0xFAB00000+i for i, r in enumerate(regs))
            assert all(self.reg(r) == (0xABCD0000+i)|((0xFEDC0000+i)<<64) for i, r in enumerate(vectors))
            if name in ['GetGender', 'GetArtType']: assert result < 1 << 32
            else: self.oid(result)
        return returned, result

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.completed_memory_writes, self.completed_slot_writes, self.reached_flag_writes = {}, {}, {}
        initial, old = self.snapshot(), len(self.events)
        self.tracking_writes = True
        try: returned, result = self.invoke(name)
        finally: self.tracking_writes = False
        final, events = self.snapshot(), self.events[old:].copy()
        if name in LEAF_FIELDS: assert final == initial and events == []
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            allowed = self.completed_memory_writes.get(n, set())
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        assert final['metadata_slots'] == {**initial['metadata_slots'], **self.completed_slot_writes}
        assert final['metadata_flags_u8'] == {**initial['metadata_flags_u8'], **self.reached_flag_writes}
        if returned:
            if name in LEAF_FIELDS:
                off = LEAF_FIELDS[name]; raw = bytes.fromhex(final['memory']['data'])
                expected = int.from_bytes(raw[off:off+(4 if name == 'GetGender' else 8)], 'little')
            else:
                comparison = next(e for e in events if e['kind'] in ['object_equality_service', 'object_inequality_service'])
                default = comparison['args'][3] & 255 != 0 if name != 'GetArtistName' else comparison['args'][3] & 255 == 0
                if default:
                    if name == 'GetArtType': expected = 0
                    elif name == 'GetArtistName':
                        # Captured after metadata, before class initialization/comparison.
                        snap = next(e for e in events if e['kind'] in ['object_class_initialize_service', 'object_inequality_service'])['snapshot']
                        expected = self.p[snap['metadata_slots']['normandia']]
                    else:
                        raw = bytes.fromhex(final['memory']['data']); off = 0x98 if name == 'GetArt' else 0xA8
                        expected = int.from_bytes(raw[off:off+8], 'little')
                else:
                    skin = final['current_skin']; assert skin is not None
                    raw = bytes.fromhex(final['memory'][skin]); off = {'GetArt': 0x38, 'GetAnimatedArt': 0x40, 'GetArtType': 0x50, 'GetArtistName': 0x20}[name]
                    expected = int.from_bytes(raw[off:off+(4 if name == 'GetArtType' else 8)], 'little')
            assert result == expected
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'result_bits': result, 'result_identity': self.oid(result) if returned and name not in ['GetGender', 'GetArtType'] else None,
                'initial': initial, 'events': events, 'final': final, 'unconsumed_memory_retained': True,
                'completed_memory_write_offsets': {n: sorted(v) for n, v in self.completed_memory_writes.items()},
                'completed_metadata_slot_writes': self.completed_slot_writes.copy(),
                'reached_native_flag_writes': self.reached_flag_writes.copy(),
                'win64_nonvolatile_and_stack_preserved_on_return': returned}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for name in TARGETS:
        for options in [{}, {'null_return_fields': True}, {'null_owner': True}, {'alias_outputs': True}]: cases.append(m.run(name, options))
    for name in ['GetGender', 'GetArtType']:
        for bits in [0, 10, 20, 0x8000000A, 0xFFFFFFFF]: cases.append(m.run(name, {'gender_bits': bits, 'art_type_bits': bits}))
    for name, initial_skin, live, warm, class_word in itertools.product(ART_METHODS, [None, 'skin', 'other_skin'], [False, True], [0, 0xFE], [0, 0xFFFFFFFF]):
        r = m.run(name, {'initial_skin': initial_skin, 'skin_live': live, 'warm_flag': warm, 'class_word': class_word})
        assert r['returned']; cases.append(r)
    for name, raw in itertools.product(ART_METHODS, [0, 1, 0x80, 0xFF]):
        cases.append(m.run(name, {'forced_comparison_bits': 0xABCD123456789000 | raw}))
    for name in ART_METHODS:
        r = m.run(name, {'class_word': 1, 'mutation_phase': 'class_init', 'mutation': 'replace_skin'})
        assert r['completed_memory_write_offsets'] == {} and r['final']['current_skin'] == 'skin'
        cases.append(r)
    for name in LEAF_FIELDS:
        r = m.run(name, {'mutation_phase': 'metadata', 'mutation': 'replace_literal'})
        assert r['completed_metadata_slot_writes'] == {} and r['reached_native_flag_writes'] == {}
        cases.append(r)
    for name, phase, action, initial_skin in itertools.product(ART_METHODS, ['metadata', 'class_init', 'comparison'],
            ['replace_skin', 'clear_skin', 'same_skin', 'replace_default', 'clear_default', 'replace_skin_output', 'clear_skin_output', 'replace_literal'], ['skin', 'other_skin']):
        r = m.run(name, {'initial_skin': initial_skin, 'mutation_phase': phase, 'mutation': action})
        comparison = next(e for e in r['events'] if e['kind'] in ['object_equality_service', 'object_inequality_service'])
        captured = 'other_skin' if phase == 'metadata' and action == 'replace_skin' else None if phase == 'metadata' and action == 'clear_skin' else initial_skin
        assert comparison['args'][0] == captured
        if action == 'clear_skin' and phase in ['class_init', 'comparison']:
            assert not r['returned'] and r['error'] == 'native_null_guard'
        else: assert r['returned']
        cases.append(r)
    for initial_skin in ['skin', 'other_skin', None]:
        m.prepare({'initial_skin': initial_skin})
        rows = [m.run(name, retained=True) for name in ['GetCharacterName', 'GetArt', 'GetArtType', 'GetAnimatedArt', 'GetArtistName', 'GetIWas', 'GetGender', 'GetTranslation', 'GetArt']]
        assert all(r['returned'] for r in rows); sequences.append(rows)
    m.prepare({})
    rows = [m.run('GetArt', {'mutation_phase': 'class_init', 'mutation': 'replace_skin'}, retained=True),
            m.run('GetArtType', retained=True), m.run('GetAnimatedArt', retained=True), m.run('GetArtistName', retained=True)]
    assert all(r['returned'] for r in rows); sequences.append(rows)
    stop_profiles = [(n, {}) for n in ART_METHODS] + [(n, {'warm_flag': 0xFE, 'class_word': 1}) for n in ART_METHODS]
    stop_profiles += [(n, {'mutation_phase': phase, 'mutation': action})
                      for n in ART_METHODS for phase, action in [('class_init', 'replace_skin'), ('metadata', 'replace_literal'), ('comparison', 'replace_skin_output')]]
    for name, options in stop_profiles:
        baseline = m.run(name, options); assert baseline['returned']; bid = len(baselines); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0)+1
            stopped = m.run(name, {**options, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index+1] and stopped['final'] == e['snapshot']
            stops.append({'baseline': bid, 'prefix_length': index+1, 'result': stopped})
    missing = sorted(set(m.instructions)-m.executed)
    assert missing == [0x3B4A18, 0x3B4AA2, 0x3B4B38, 0x3B4BD6]
    assert all(m.instructions[a].mnemonic == 'int3' for a in missing)
    return {'schema': 'character_data_consumers_native_v1', 'build': BUILD, 'targets': m.targets,
            'scope': 'eight exact CharacterData consumers; Unity Object/runtime services supplied; folded aliases and localization/providers excluded',
            'supplied_targets': m.supplied, 'data_field_pins': m.data_fields, 'skin_type_def_index': 5945,
            'art_type_enum': {'type_def_index': 5946, 'Default': 0, 'Clipping': 10},
            'gender_enum': {'type_def_index': 5956, 'Female': 0, 'Male': 10, 'They': 20},
            'body_bounds': m.bounds, 'unwind_ranges': m.ranges,
            'metadata_bindings': [{'rva': hex(a), 'kind': k, 'name_or_value': n} for a, (k, n) in m.bindings.items()],
            'metadata_flag_rvas': {n: hex(a) for n, a in m.flags.items()},
            'diagnostic_windows_not_object_extents': m.sizes,
            'instruction_assertions': len(m.checks), 'decoded_instructions': len(m.instructions),
            'covered_instructions': len(set(m.instructions)&m.executed), 'unexecuted_terminal_traps': [hex(a) for a in missing],
            'cases': cases, 'retained_sequences': sequences, 'baselines': baselines, 'failure_stops': stops,
            'summary': {'cases': len(cases), 'sequences': len(sequences), 'baselines': len(baselines), 'stops': len(stops)}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--game-root', type=Path, required=True)
    p.add_argument('--dumper-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = audit(args.game_root, args.dumper_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], sort_keys=True))
