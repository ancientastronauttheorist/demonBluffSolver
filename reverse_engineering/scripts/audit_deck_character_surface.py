"""Execute five DeckCharacter surfaces and their actual HintInfo construction.

Callbacks, List.Contains, runtime services and RevealCard.RevealNoAct are supplied.
No UI renderer, scene lifecycle, preference storage or scheduling is executed.
"""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine


TARGETS = {
    'GetData': (0x36FA10, 0x36FA15, 0x36FA20, 'tdi5514.m0000'),
    'Click': (0x36F9B0, 0x36FA05, 0x36FA10, 'tdi5514.m0004'),
    'Reveal': (0x36FF70, 0x36FF8E, 0x36FF90, 'tdi5514.m0003'),
    'OnHoverExit': (0x36FDC0, 0x36FE0C, 0x36FE10, 'tdi5514.m0006'),
    'OnHover': (0x36FE10, 0x36FF65, 0x36FF70, 'tdi5514.m0005'),
}
HINT_CTOR = (0x3BC200, 0x3BC297, 0x3BC2A0)
OWNER_FIELDS = [('on_click', 0x20), ('character', 0x28), ('reveal', 0x30),
                ('interaction', 0x38), ('data', 0x40)]
HINT_FIELDS = [('title', 0x10), ('text', 0x18), ('hints', 0x20), ('flavor', 0x28), ('image', 0x30)]


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root)
        extraction = json.loads((Path(__file__).parents[1] /
            f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pinned(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pinned('script.json', 'script_json'))
        dump = pinned('dump.cs', 'dump_cs')
        def declaration(name, index, static=False):
            match = re.search(r'^public ' + ('static ' if static else '') + r'class ' + re.escape(name) +
                              r'(?: :[^\n]*)? // TypeDefIndex: ' + str(index) + r'\s*\{(.*?)\n\}', dump, re.M | re.S)
            assert match
            return match[1]
        owner = declaration('DeckCharacter', 5514)
        for field in ['public Action onClick; // 0x20', 'public Character character; // 0x28',
                      'public RevealCard revealCard; // 0x30', 'public CardInteraction interaction; // 0x38',
                      'private CharacterData data; // 0x40']:
            assert field in owner
        assert 'public static Action<DeckCharacter> OnDeckCharacterClicked; // 0x0' in declaration('CardsEvents', 5522, True)
        ui = declaration('UIEvents', 5523, True)
        assert 'public static Action<HintInfo, Transform> OnShowHint; // 0x10' in ui
        assert 'public static Action OnHideHint; // 0x40' in ui
        self.ui_fields = [(name, int(offset, 16)) for name, offset in re.findall(
            r'public static Action(?:<[^\n]+>)? (\w+); // (0x[\dA-Fa-f]+)', ui)]
        assert len(self.ui_fields) == 21
        assert 'public static List<CharacterData> ObscuredCharacters; // 0x0' in declaration('DeckView', 5744)
        assert 'public Transform hintPivot; // 0x38' in declaration('Character', 5487)
        hint = declaration('HintInfo', 5800)
        for field in ['public string title; // 0x10', 'public string text; // 0x18',
                      'public string hints; // 0x20', 'public string flavor; // 0x28',
                      'public Sprite img; // 0x30', 'public Color borderColor; // 0x38']:
            assert field in hint
        delegate = re.search(r'^public abstract class Delegate : ICloneable, ISerializable // TypeDefIndex: 419\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert delegate
        for field in ['private IntPtr invoke_impl; // 0x18', 'private object m_target; // 0x20',
                      'private IntPtr method; // 0x28', 'private IntPtr method_code; // 0x40']:
            assert field in delegate[1]
        self.targets, self.helper_targets, self.instructions, self.body_addresses, self.ranges = [], [], {}, set(), {}
        for name, (start, end, following, method) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == 'DeckCharacter$$' + name and r['Address'] == start]
            assert len(rows) == 1
            signature = ('CharacterData_o*' if name == 'GetData' else 'void') + f' DeckCharacter__{name} (DeckCharacter_o* __this, const MethodInfo* method);'
            assert rows[0]['Signature'] == signature
            self.targets.append(dict(rows[0], stable_method_id=method))
            self.decode(name, start, end, following, leaf=name == 'GetData')
        ctor_rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == 'HintInfo$$.ctor' and r['Address'] == HINT_CTOR[0]]
        assert len(ctor_rows) == 1
        assert ctor_rows[0]['Signature'] == 'void HintInfo___ctor (HintInfo_o* __this, System_String_o* txt, UnityEngine_Sprite_o* img, System_String_o* hints, System_String_o* flavor, System_String_o* title, UnityEngine_Color_o borderColor, const MethodInfo* method);'
        self.helper_targets = [dict(ctor_rows[0], stable_method_id='tdi5800.m0000')]
        self.decode('HintInfo.ctor', *HINT_CTOR)
        self.instructions.update({i.address: i for i in self.cs.disasm(self.pe.get_data(0x33ED50, 3), 0x33ED50)})
        assert (self.instructions[0x33ED50].mnemonic, self.instructions[0x33ED50].op_str) == ('ret', '0')
        self.service_targets = []
        for address, name in [(0xB55950, 'System.Collections.Generic.List<object>$$Contains'),
                              (0x386920, 'RevealCard$$RevealNoAct')]:
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == name and r['Address'] == address]
            assert len(rows) == 1
            self.service_targets += rows
        assert self.service_targets[1]['Signature'] == 'void RevealCard__RevealNoAct (RevealCard_o* __this, const MethodInfo* method);'
        references, self.flags = set(), {}
        for i in self.instructions.values():
            for operand in i.operands:
                if operand.type == capstone.CS_OP_MEM and operand.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + operand.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
                self.flags[i.address + i.size + i.operands[0].mem.disp] = 0
        self.bindings, self.metadata_slots = {}, {}
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                pointer = self.arena + 0x4000 + len(self.bindings) * 0x400
                self.bindings[row['Name']] = pointer
                self.metadata_slots[row['Address']] = pointer
                self.q(self.base + row['Address'], pointer)
        self.contains_method = 'Method$System.Collections.Generic.List<CharacterData>.Contains()'
        assert set(self.bindings) == {'UIEvents_TypeInfo', 'CardsEvents_TypeInfo', 'DeckView_TypeInfo', 'HintInfo_TypeInfo', self.contains_method}
        names = ['owner', 'owner_class', 'character0', 'character1', 'character_class', 'pivot0', 'pivot1',
                 'reveal0', 'reveal1', 'interaction', 'data0', 'data1', 'ui_static', 'other_ui_static',
                 'cards_static', 'other_cards_static', 'deck_static', 'other_deck_static', 'list', 'other_list',
                 'list_array', 'other_list_array', 'hint0', 'hint1', 'hint2', 'hint3', 'hint4',
                 'empty_string', 'hint_string', 'string_class', 'click_action', 'show_action', 'hide_action',
                 'other_click_action', 'other_show_action', 'other_hide_action', 'action_class_token',
                 'click_code', 'show_code', 'hide_code', 'other_code', 'uncalled_managed_target',
                 'click_method', 'show_method', 'hide_method', 'other_method']
        self.p = {name: self.arena + 0x10000 + index * 0x1000 for index, name in enumerate(names)}
        self.labels = {0: None, **{pointer: name for name, pointer in self.p.items()},
                       **{pointer: name for name, pointer in self.bindings.items()}}
        self.sizes = {name: 0xA8 if name.endswith('ui_static') or name == 'ui_static' else
                      0x100 if name.endswith('class') or name.endswith('_class_token') else 0x80 for name in self.p}
        self.literal_slots = {}
        for row in self.metadata['ScriptString']:
            if row['Address'] not in references:
                continue
            assert row['Value'] in ['', 'This card is Hidden,\nsomething is blocking it!']
            name = 'empty_string' if row['Value'] == '' else 'hint_string'
            self.literal_slots[row['Address']] = {'pointer': self.p[name], 'value': row['Value']}
            self.q(self.base + row['Address'], self.p[name])
        assert len(self.literal_slots) == 2 and len(self.flags) == 3
        self.callback = self.stop + 0x100
        self.checks = {
            0x36FA10: ('mov', 'rax, qword ptr [rcx + 0x40]'),
            0x36F9E3: ('mov', 'rax, qword ptr [rdx]'),
            0x36F9EB: ('mov', 'r8, qword ptr [rax + 0x28]'),
            0x36F9EF: ('mov', 'rdx, rbx'),
            0x36F9F2: ('mov', 'rcx, qword ptr [rax + 0x40]'),
            0x36F9FB: ('jmp', 'qword ptr [rax + 0x18]'),
            0x36FF74: ('mov', 'rcx, qword ptr [rcx + 0x30]'),
            0x36FF83: ('jmp', '0x386920'),
            0x36FDEE: ('mov', 'rax, qword ptr [rcx + 0x40]'),
            0x36FE03: ('jmp', 'qword ptr [rax + 0x18]'),
            0x36FE78: ('cmp', 'dword ptr [rax + 0xe0], 0'),
            0x36FEB4: ('mov', 'rdx, qword ptr [rdi + 0x40]'),
            0x36FEBD: ('test', 'al, al'),
            0x36FED3: ('mov', 'rbx, qword ptr [rcx + 0x10]'),
            0x36FF0F: ('xor', 'r8d, r8d'),
            0x36FF1A: ('mov', 'qword ptr [rsp + 0x28], r9'),
            0x36FF1F: ('mov', 'qword ptr [rsp + 0x20], r9'),
            0x36FF24: ('movdqa', 'xmmword ptr [rsp + 0x40], xmm6'),
            0x36FF2A: ('call', '0x3bc200'),
            0x36FF2F: ('mov', 'r8, qword ptr [rdi + 0x28]'),
            0x36FF44: ('mov', 'r8, qword ptr [r8 + 0x38]'),
            0x36FF4C: ('call', 'qword ptr [rbx + 0x18]'),
            0x3BC222: ('call', '0x33ed50'),
            0x3BC22E: ('mov', 'qword ptr [rcx], rbx'),
            0x3BC23F: ('mov', 'qword ptr [rcx], rdx'),
            0x3BC24E: ('mov', 'qword ptr [rcx], rdi'),
            0x3BC25D: ('mov', 'qword ptr [rcx], rsi'),
            0x3BC26E: ('mov', 'qword ptr [rcx], rdx'),
            0x3BC285: ('movups', 'xmm0, xmmword ptr [rax]'),
            0x3BC288: ('movups', 'xmmword ptr [rbp + 0x38], xmm0'),
        }
        assert all((self.instructions[address].mnemonic, self.instructions[address].op_str) == expected
                   for address, expected in self.checks.items())
        ctor_calls = [i.op_str for i in self.instructions.values() if HINT_CTOR[0] <= i.address < HINT_CTOR[1] and i.mnemonic == 'call']
        assert ctor_calls.count('0x2b6ff0') == len(HINT_FIELDS) and ctor_calls.count('0x33ed50') == 1

    def decode(self, name, start, end, following, leaf=False):
        assert min(row['Address'] for row in self.metadata['ScriptMethod'] if row['Address'] > start) == following
        chunks = []
        for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root = entry
            while root.unwindinfo.Flags & 4:
                root = root.unwindinfo._chained_entry
            if root.struct.BeginAddress == start:
                chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
        assert not chunks if leaf else chunks and min(a for a, _ in chunks) == start and max(b for _, b in chunks) == end
        section = self.pe.get_section_by_rva(start)
        assert section and following - section.VirtualAddress <= section.SizeOfRawData
        raw = self.pe.get_data(start, following - start)
        assert len(raw) == following - start and raw[end - start:] == b'\xCC' * (following - end)
        instructions = list(self.cs.disasm(raw[:end - start], start))
        assert sum(i.size for i in instructions) == end - start
        self.instructions.update({i.address: i for i in instructions})
        self.body_addresses.update(i.address for i in instructions)
        self.ranges[name] = {'body': [hex(start), hex(end)], 'unwind_chunks': [[hex(a), hex(b)] for a, b in chunks], 'next_entry': hex(following)}

    def oid(self, value):
        assert value in self.labels, hex(value)
        return self.labels[value]

    def snapshot(self):
        return {'owner_fields': {name: self.oid(self.rq(self.p['owner'] + offset)) for name, offset in OWNER_FIELDS},
                'class_state': {name: {'static': self.oid(self.rq(pointer + 0xB8)), 'initialized_word': self.rd(pointer + 0xE0)}
                                for name, pointer in self.bindings.items() if name in ['UIEvents_TypeInfo', 'CardsEvents_TypeInfo', 'DeckView_TypeInfo']},
                'metadata_flags': {hex(address): self.u.mem_read(self.base + address, 1)[0] for address in sorted(self.flags)},
                'metadata_slots': {hex(address): self.oid(self.rq(self.base + address)) for address in sorted(self.metadata_slots)},
                'literal_slots': {hex(address): self.oid(self.rq(self.base + address)) for address in sorted(self.literal_slots)},
                'hints': {name: {'phase': phase, 'fields': {field: self.oid(self.rq(self.p[name] + offset)) for field, offset in HINT_FIELDS},
                                  'color_bits': list(struct.unpack('<IIII', self.u.mem_read(self.p[name] + 0x38, 16)))}
                          for name, phase in self.hints.items()},
                'allocation_order': self.allocation_order.copy(), 'native_entries': [entry.copy() for entry in self.native_entries],
                'callbacks': [record.copy() for record in self.callbacks], 'reveal_requests': self.reveal_requests.copy(),
                'memory': {name: bytes(self.u.mem_read(pointer, self.sizes[name])).hex() for name, pointer in self.p.items()},
                'runtime_class_bytes': {name: bytes(self.u.mem_read(pointer, 0x100)).hex() for name, pointer in self.bindings.items() if name.endswith('_TypeInfo')}}

    def prepare(self, options):
        self.options = options.copy()
        self.events, self.counts, self.error, self.native_entries = [], {}, None, []
        self.hints, self.allocation_order, self.callbacks, self.reveal_requests, self.allowed = {}, [], [], [], {}
        for name, pointer in self.p.items():
            self.u.mem_write(pointer, bytes([0xA5]) * self.sizes[name])
        self.q(self.p['owner'], self.p['owner_class'])
        self.q(self.p['owner'] + 8, 0)
        for name, offset in OWNER_FIELDS:
            value = self.p[name + '0'] if name in ['character', 'reveal', 'data'] else self.p['interaction'] if name == 'interaction' else 0
            if options.get('null_' + name): value = 0
            self.q(self.p['owner'] + offset, value)
        if options.get('alias_instance_on_click'):
            self.q(self.p['owner'] + 0x20, self.p['hide_action'])
        for name in ['character0', 'character1']:
            self.q(self.p[name], self.p['character_class'])
            self.q(self.p[name] + 0x38, 0 if options.get('null_pivot') else self.p['pivot0' if options.get('alias_pivots') else 'pivot' + name[-1]])
        for name, static in [('CardsEvents_TypeInfo', 'cards_static'), ('UIEvents_TypeInfo', 'ui_static'), ('DeckView_TypeInfo', 'deck_static')]:
            self.u.mem_write(self.bindings[name], bytes([0xA5]) * 0x100)
            self.q(self.bindings[name] + 0xB8, self.p[static])
            self.d(self.bindings[name] + 0xE0, 0 if options.get('class_cold') else options.get('class_word', 1))
        self.u.mem_write(self.bindings['HintInfo_TypeInfo'], bytes([0xA5]) * 0x100)
        for name in ['ui_static', 'other_ui_static', 'cards_static', 'other_cards_static', 'deck_static', 'other_deck_static']:
            self.u.mem_write(self.p[name], bytes(self.sizes[name]))
        for prefix in ['', 'other_']:
            for kind, code, method in [('click', 'click_code', 'click_method'), ('show', 'show_code', 'show_method'), ('hide', 'hide_code', 'hide_method')]:
                pointer = self.p[prefix + kind + '_action']
                self.u.mem_write(pointer, bytes(0x80))
                self.q(pointer, self.p['action_class_token'])
                self.q(pointer + 0x18, self.callback)
                self.q(pointer + 0x20, self.p['uncalled_managed_target'])
                self.q(pointer + 0x28, self.p['other_method'] if prefix else self.p[method])
                self.q(pointer + 0x40, 0 if options.get('null_method_code') else self.p['other_code'] if prefix else self.p[code])
            self.q(self.p[prefix + 'cards_static'], 0 if options.get('no_click_callback') else self.p[prefix + 'click_action'])
            self.q(self.p[prefix + 'ui_static'] + 0x10, 0 if options.get('no_show_callback') else self.p[prefix + 'show_action'])
            self.q(self.p[prefix + 'ui_static'] + 0x40, 0 if options.get('no_hide_callback') else self.p[prefix + 'hide_action'])
            self.q(self.p[prefix + 'deck_static'], 0 if options.get('null_list') else self.p[prefix + 'list'])
            # These are opaque identity/retention diagnostics. Contains is wholly
            # supplied; no generic List/Array layout or membership is inferred.
            array, listing = self.p[prefix + 'list_array'], self.p[prefix + 'list']
            self.q(listing + 0x10, array); self.d(listing + 0x18, 2); self.d(listing + 0x1C, 0xF00D)
            self.q(array + 0x18, 0xFACE000000000002)
            self.q(array + 0x20, self.p['data0']); self.q(array + 0x28, self.p['data0'] if options.get('duplicate_list_entries') else self.p['data1'])
        for row in self.literal_slots.values():
            pointer, raw = row['pointer'], row['value'].encode('utf-16-le')
            self.u.mem_write(pointer, bytes(self.sizes[self.oid(pointer)]))
            self.q(pointer, self.p['string_class'])
            self.d(pointer + 0x10, len(raw) // 2)
            self.u.mem_write(pointer + 0x14, raw + b'\0\0')
        for address in self.flags:
            self.u.mem_write(self.base + address, bytes([0 if options.get('cold') else options.get('warm_flag', 1)]))

    def allow(self, name, offset, width):
        self.allowed.setdefault(name, set()).update(range(offset, offset + width))

    def mutate(self, kind, ordinal):
        action = self.options.get('mutations', {}).get(kind + ':' + str(ordinal))
        if action is None: return
        if action in ['replace_character', 'clear_character', 'replace_data', 'clear_data', 'replace_reveal', 'clear_reveal']:
            field = action.split('_')[1]
            offset = dict(OWNER_FIELDS)[field]
            value = 0 if action.startswith('clear_') else self.p[field + '1']
            self.q(self.p['owner'] + offset, value); self.allow('owner', offset, 8)
        elif action in ['replace_show', 'clear_show', 'replace_hide', 'clear_hide']:
            kind = action.split('_')[1]
            static = self.rq(self.bindings['UIEvents_TypeInfo'] + 0xB8)
            offset = 0x10 if kind == 'show' else 0x40
            self.q(static + offset, 0 if action.startswith('clear_') else self.p['other_' + kind + '_action'])
            self.allow(self.oid(static), offset, 8)
        elif action == 'swap_ui_static':
            self.q(self.bindings['UIEvents_TypeInfo'] + 0xB8, self.p['other_ui_static'])
            self.runtime_allowed.setdefault('UIEvents_TypeInfo', set()).update(range(0xB8, 0xC0))
        elif action == 'swap_deck_static':
            self.q(self.bindings['DeckView_TypeInfo'] + 0xB8, self.p['other_deck_static'])
            self.runtime_allowed.setdefault('DeckView_TypeInfo', set()).update(range(0xB8, 0xC0))
        elif action == 'swap_cards_static':
            self.q(self.bindings['CardsEvents_TypeInfo'] + 0xB8, self.p['other_cards_static'])
            self.runtime_allowed.setdefault('CardsEvents_TypeInfo', set()).update(range(0xB8, 0xC0))
        elif action == 'clear_pivot':
            character = self.rq(self.p['owner'] + 0x28)
            assert character
            self.q(character + 0x38, 0); self.allow(self.oid(character), 0x38, 8)
        else:
            raise AssertionError(action)

    def event(self, kind, args):
        ordinal = self.counts.get(kind, 0) + 1
        self.counts[kind] = ordinal
        self.events.append({'kind': kind, 'args': args, 'snapshot': self.snapshot()})
        if self.options.get('failure') == [kind, ordinal]:
            self.error = kind; self.u.emu_stop(); return False
        self.mutate(kind, ordinal)
        return True

    def ret(self, value=0xABCD123456789000):
        for index, name in enumerate(['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']):
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_' + name), 0xFACE123456789000 + index)
        for index in range(6):
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_XMM' + str(index)), (1 << 127) | index)
        super().ret(value)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        if address == self.stop: return
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + name)) for name in ['RCX', 'RDX', 'R8', 'R9']]
        if rva in [value[0] for value in TARGETS.values()]:
            assert cx == self.p['owner']
            self.native_entries.append({'method': self.method})
        if rva == HINT_CTOR[0]:
            assert self.oid(cx) in self.hints and dx == self.p['hint_string'] and r8 == 0 and r9 == self.p['empty_string']
            sp = self.reg(x.UC_X86_REG_RSP)
            assert self.rq(sp + 0x28) == self.p['empty_string'] and self.rq(sp + 0x30) == self.p['empty_string'] and self.rq(sp + 0x40) == 0
            color = self.rq(sp + 0x38)
            assert bytes(self.u.mem_read(color, 16)) == bytes(16)
            self.hints[self.oid(cx)] = 'constructor_entered'
            self.native_entries.append({'method': 'HintInfo.ctor', 'hint': self.oid(cx), 'color_bits': [0, 0, 0, 0]})
        elif rva == 0x33ED50:
            assert self.oid(cx) in self.hints and dx == 0
            self.folded_base_rax = self.reg(x.UC_X86_REG_RAX)
            self.hints[self.oid(cx)] = 'folded_base_entered'
            self.native_entries.append({'method': 'folded_Object_ret0', 'hint': self.oid(cx)})
        elif rva == 0x3BC227:
            assert self.reg(x.UC_X86_REG_RAX) == self.folded_base_rax
        elif rva == 0x3BC28C:
            hint = self.reg(x.UC_X86_REG_RBP)
            self.hints[self.oid(hint)] = 'constructed'
        if rva in self.instructions: return
        if rva == 0x2B7B40:
            assert cx - self.base in self.metadata_slots or cx - self.base in self.literal_slots
            if self.event('metadata', [hex(cx - self.base)]): self.ret(self.rq(cx))
        elif rva == 0x281D90:
            assert cx == self.bindings['DeckView_TypeInfo']
            if self.event('deck_class_initialization', [self.oid(cx)]):
                self.d(cx + 0xE0, 1)
                self.runtime_allowed.setdefault('DeckView_TypeInfo', set()).update(range(0xE0, 0xE4))
                self.ret()
        elif rva == 0xB55950:
            assert self.oid(cx) in ['list', 'other_list'] and r8 == self.bindings[self.contains_method]
            static = self.rq(self.bindings['DeckView_TypeInfo'] + 0xB8)
            assert cx == self.rq(static)
            assert dx == self.rq(self.p['owner'] + 0x40)
            result = self.options.get('contains_return_bits', 0xFACE000000000001)
            if self.event('contains', {'list': self.oid(cx), 'data': self.oid(dx), 'method': self.oid(r8), 'return_bits': result}): self.ret(result)
        elif rva == 0x2B7D40:
            assert cx == self.bindings['HintInfo_TypeInfo']
            static = self.rq(self.bindings['UIEvents_TypeInfo'] + 0xB8)
            self.expected_show_action = self.rq(static + 0x10)
            assert self.expected_show_action in [self.p['show_action'], self.p['other_show_action']]
            if self.event('allocate_hint', [self.oid(cx)]):
                name = 'hint' + str(len(self.hints)); assert name in self.p
                self.u.mem_write(self.p[name], bytes(self.sizes[name])); self.q(self.p[name], cx)
                self.allow(name, 0, self.sizes[name])
                self.hints[name] = 'allocated'; self.allocation_order.append(name)
                self.ret(self.p[name])
        elif rva == 0x2B6FF0:
            hint, offset = next(((self.p[name], cx - self.p[name]) for name in self.hints if self.p[name] <= cx < self.p[name] + 0x48), (None, None))
            assert hint and offset in dict(HINT_FIELDS).values() and self.rq(cx) == dx
            field = next(name for name, value in HINT_FIELDS if value == offset)
            assert dx == (0 if field == 'image' else self.p['hint_string'] if field == 'text' else self.p['empty_string'])
            self.hints[self.oid(hint)] = field + '_stored'
            if self.event('reference_barrier', {'hint': self.oid(hint), 'field': field, 'value': self.oid(dx)}): self.ret()
        elif rva == 0x386920:
            assert cx == self.rq(self.p['owner'] + 0x30) and self.oid(cx) in ['reveal0', 'reveal1'] and dx == 0
            if self.event('supplied_RevealCard_RevealNoAct', [self.oid(cx), dx]):
                self.reveal_requests.append(self.oid(cx)); self.ret()
        elif address == self.callback:
            if self.method == 'Click':
                assert dx == self.p['owner'] and r8 in [self.p['click_method'], self.p['other_method']]
                action = self.rq(self.rq(self.bindings['CardsEvents_TypeInfo'] + 0xB8))
                assert cx == self.rq(action + 0x40) and r8 == self.rq(action + 0x28)
                record = {'kind': 'deck_clicked', 'method_code': self.oid(cx), 'owner': self.oid(dx), 'method': self.oid(r8), 'r9_bits': r9}
            elif self.method == 'OnHoverExit':
                assert dx in [self.p['hide_method'], self.p['other_method']]
                action = self.rq(self.rq(self.bindings['UIEvents_TypeInfo'] + 0xB8) + 0x40)
                assert cx == self.rq(action + 0x40) and dx == self.rq(action + 0x28)
                record = {'kind': 'hide_hint', 'method_code': self.oid(cx), 'method': self.oid(dx), 'r8_bits': r8, 'r9_bits': r9}
            else:
                assert self.method == 'OnHover' and self.oid(dx) in self.hints and r9 in [self.p['show_method'], self.p['other_method']]
                assert cx == self.rq(self.expected_show_action + 0x40) and r9 == self.rq(self.expected_show_action + 0x28)
                assert self.hints[self.oid(dx)] == 'constructed'
                character = self.rq(self.p['owner'] + 0x28)
                assert character and r8 == self.rq(character + 0x38)
                record = {'kind': 'show_hint', 'method_code': self.oid(cx), 'hint': self.oid(dx), 'pivot': self.oid(r8), 'method': self.oid(r9)}
            assert cx == 0 or self.oid(cx) in ['click_code', 'show_code', 'hide_code', 'other_code']
            if self.event('callback', record): self.callbacks.append(record); self.ret()
        elif rva == 0x2B7D90:
            self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop()
        else:
            raise AssertionError(f'unclaimed native address {rva:x}')

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        elif options is not None: self.options.update(options)
        self.method, self.error, self.allowed, self.runtime_allowed = name, None, {}, {}
        initial, old = self.snapshot(), len(self.events)
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        for index, register in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']):
            self.u.reg_write(getattr(x, 'UC_X86_REG_' + register), 0xFAB0000000000000 + index)
        for index in range(6, 16):
            self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(index)), (0xABCDEF9876543210 << 64) | index)
        for register, value in [('RSP', sp), ('RCX', self.p['owner']), ('RDX', 0xDEAD123456789ABC), ('R8', 0xDEAD123400000008), ('R9', 0xDEAD123400000009)]:
            self.u.reg_write(getattr(x, 'UC_X86_REG_' + register), value)
        self.u.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        self.u.emu_start(self.base + TARGETS[name][0], self.stop, timeout=10_000_000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        result = None
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for index, register in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']):
                assert self.reg(getattr(x, 'UC_X86_REG_' + register)) == 0xFAB0000000000000 + index
            for index in range(6, 16):
                assert self.reg(getattr(x, 'UC_X86_REG_XMM' + str(index))) == (0xABCDEF9876543210 << 64) | index
            if name == 'GetData':
                result = self.oid(self.reg(x.UC_X86_REG_RAX))
                assert result == initial['owner_fields']['data']
        final = self.snapshot()
        for item, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][item])
            assert all(offset in self.allowed.get(item, set()) or value == after[offset] for offset, value in enumerate(before)), item
        for item, raw in initial['runtime_class_bytes'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['runtime_class_bytes'][item])
            assert all(offset in self.runtime_allowed.get(item, set()) or value == after[offset] for offset, value in enumerate(before)), item
        assert initial['metadata_slots'] == final['metadata_slots'] and initial['literal_slots'] == final['literal_slots']
        for address, byte in initial['metadata_flags'].items():
            assert final['metadata_flags'][address] == byte or (byte == 0 and final['metadata_flags'][address] == 1)
        completed = self.events[old:] if returned else self.events[old:-1]
        assert final['callbacks'] == initial['callbacks'] + [event['args'] for event in completed if event['kind'] == 'callback']
        assert final['reveal_requests'] == initial['reveal_requests'] + [event['args'][0] for event in completed if event['kind'] == 'supplied_RevealCard_RevealNoAct']
        row = {'method': name, 'options': self.options.copy(), 'returned': returned, 'result': result, 'error': self.error,
               'initial': initial, 'events': self.events[old:].copy(), 'final': final,
               'unrelated_diagnostic_storage_retained': True, 'win64_nonvolatile_verified': returned}
        verify_normal(row)
        return row


def verify_normal(row):
    options, method, events = row['options'], row['method'], row['events']
    if options.get('mutations') or options.get('failure'): return
    kinds = [event['kind'] for event in events if event['kind'] not in ['metadata', 'deck_class_initialization']]
    if method == 'GetData':
        assert row['returned'] and not kinds
    elif method == 'Click':
        assert row['returned'] and kinds == ([] if options.get('no_click_callback') else ['callback'])
    elif method == 'OnHoverExit':
        assert row['returned'] and kinds == ([] if options.get('no_hide_callback') else ['callback'])
    elif method == 'Reveal':
        assert row['returned'] == (not options.get('null_reveal'))
        assert kinds == (['native_null_guard'] if options.get('null_reveal') else ['supplied_RevealCard_RevealNoAct'])
    else:
        if options.get('null_list'):
            assert not row['returned'] and kinds == ['native_null_guard']
        elif not (options.get('contains_return_bits', 0xFACE000000000001) & 0xFF) or options.get('no_show_callback'):
            assert row['returned'] and kinds == ['contains']
        else:
            suffix = ['native_null_guard'] if options.get('null_character') else ['callback']
            assert row['returned'] == (not options.get('null_character'))
            assert kinds == ['contains', 'allocate_hint'] + ['reference_barrier'] * len(HINT_FIELDS) + suffix
            assert [event['args']['field'] for event in events if event['kind'] == 'reference_barrier'] == ['text', 'title', 'image', 'hints', 'flavor']
            hints = row['final']['hints']; name = row['final']['allocation_order'][-1]
            assert hints[name] == {'phase': 'constructed', 'fields': {'title': 'empty_string', 'text': 'hint_string',
                'hints': 'empty_string', 'flavor': 'empty_string', 'image': None}, 'color_bits': [0, 0, 0, 0]}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, sequences, baselines, stops = [], [], [], []
    for name, cold, class_cold in itertools.product(TARGETS, [False, True], [False, True]):
        cases.append(m.run(name, {'cold': cold, 'class_cold': class_cold}))
    for bits, list_null, character_null, callback in itertools.product(
            [0xFFFFFFFFFFFFFF00, 0xFACE000000000001, 0xFACE000000000080, 0xFACE0000000000FF],
            [False, True], [False, True], [False, True]):
        cases.append(m.run('OnHover', {'contains_return_bits': bits, 'null_list': list_null,
                                     'null_character': character_null, 'no_show_callback': not callback}))
    for name, options in [('GetData', {'null_data': True}), ('GetData', {'alias_instance_on_click': True}),
                          ('Click', {'no_click_callback': True}), ('Click', {'alias_instance_on_click': True}),
                          ('Reveal', {'null_reveal': True}), ('OnHoverExit', {'no_hide_callback': True}),
                          ('OnHover', {'null_data': True}), ('OnHover', {'null_pivot': True}),
                          ('OnHover', {'duplicate_list_entries': True}), ('OnHover', {'alias_pivots': True})]:
        cases.append(m.run(name, options))
    for name in TARGETS:
        cases.append(m.run(name, {'warm_flag': 0xFE, 'class_word': 0xFFFFFFFF, 'null_method_code': True}))
    mutation_specs = [
        ('metadata:1', 'replace_data'), ('deck_class_initialization:1', 'swap_deck_static'),
        ('contains:1', 'replace_data'), ('contains:1', 'replace_show'), ('contains:1', 'clear_show'),
        ('contains:1', 'swap_ui_static'), ('allocate_hint:1', 'replace_show'), ('allocate_hint:1', 'clear_show'),
        ('allocate_hint:1', 'swap_ui_static'), ('allocate_hint:1', 'replace_character'),
        ('reference_barrier:1', 'replace_character'), ('reference_barrier:5', 'clear_character'),
        ('reference_barrier:5', 'clear_pivot'), ('callback:1', 'replace_data'),
    ]
    for phase, action in mutation_specs:
        cases.append(m.run('OnHover', {'cold': True, 'class_cold': True, 'mutations': {phase: action}}))
    cases.append(m.run('OnHoverExit', {'cold': True, 'mutations': {'metadata:1': 'replace_hide'}}))
    cases.append(m.run('Click', {'cold': True, 'mutations': {'metadata:1': 'swap_cards_static'}}))
    for alias in [False, True]:
        m.prepare({'cold': True, 'class_cold': True, 'alias_pivots': alias, 'alias_instance_on_click': alias})
        calls = [m.run(name, retained=True) for name in ['GetData', 'Click', 'OnHover', 'OnHoverExit', 'Reveal', 'OnHover', 'GetData']]
        assert all(row['returned'] for row in calls)
        sequences.append({'alias_inputs': alias, 'calls': calls})
    specs = [(name, {}) for name in TARGETS] + [
        ('OnHover', {'mutations': {'allocate_hint:1': 'replace_show'}}),
        ('OnHover', {'mutations': {'reference_barrier:1': 'replace_character'}}),
    ]
    for name, options in specs:
        supplied = dict({'cold': True, 'class_cold': True}, **options)
        baseline = m.run(name, supplied); assert baseline['returned']
        baseline_id = len(baselines); baselines.append(baseline); counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(name, dict(supplied, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index + 1]
            assert stopped['final'] == event['snapshot']
            stops.append({'baseline': baseline_id, 'prefix_length': index + 1, 'stopped': stopped})
    missing = m.body_addresses - m.executed
    assert missing == {0x36FF8D, 0x36FF64}
    assert all(m.instructions[address].mnemonic == 'int3' for address in missing)
    return {'build': BUILD, 'targets': m.targets, 'actual_helper_targets': m.helper_targets,
            'ranges': m.ranges, 'instruction_assertions': len(m.checks), 'metadata_bindings': sorted(m.bindings),
            'literal_bindings': {hex(address): row['value'] for address, row in m.literal_slots.items()},
            'supplied_game_owned_targets': m.service_targets, 'ui_fields': self_fields(m.ui_fields),
            'cases': cases, 'case_count': len(cases), 'retained_sequences': sequences,
            'failure_baselines': baselines, 'failure_stops': stops, 'failure_case_count': len(stops),
            'body_instructions_decoded': len(m.body_addresses), 'body_instructions_executed': len(m.body_addresses & m.executed),
            'unexecuted_terminal_traps': [hex(address) for address in sorted(missing)],
            'actual_folded_base_executed': 0x33ED50 in m.executed, 'native_execution_addresses': len(m.executed),
            'scope': 'Five complete DeckCharacter surfaces and actual joined HintInfo constructor, with decoded folded Object ret0. List.Contains, callbacks, allocation/metadata/class initialization/GC and whole RevealCard.RevealNoAct remain explicitly supplied. Contains AL results are independently authored; raw List/Array records are opaque identity and retention diagnostics, not reconstructed membership or a pinned generic layout. No Init/OnDisable/constructor promotion, actual engine renderer, scene admission, callback implementation, scheduler or managed unwinding. Diagnostic sentinel storage is not an object-size or valid unused-field type claim.'}


def self_fields(fields):
    return [{'name': name, 'offset': hex(offset)} for name, offset in fields]


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
