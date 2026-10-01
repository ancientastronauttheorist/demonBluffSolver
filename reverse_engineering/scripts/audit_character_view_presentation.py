"""Execute CharacterView animation/init/art callers and joined disguise calls.

Image/TMP, data art, uppercase and DOTween implementations remain supplied.
No scheduler, animation completion or renderer implementation is synthesized.
"""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_presentation_helpers import Machine as CallerMachine, TARGETS as CALLER_TARGETS


TARGETS = {0x363D50: 'AnimateIn', 0x363DF0: 'AnimateOut', 0x363F10: 'Init', 0x3643F0: 'SetupArt'}
SUPPLIED_GAME = {0x3B4AB0: 'CharacterData$$GetArt', 0x3B4A20: 'CharacterData$$GetArtType'}
SIGNATURES = {
    'AnimateIn': 'void CharacterView__AnimateIn (CharacterView_o* __this, const MethodInfo* method);',
    'AnimateOut': 'void CharacterView__AnimateOut (CharacterView_o* __this, const MethodInfo* method);',
    'Init': 'void CharacterView__Init (CharacterView_o* __this, CharacterData_o* data, const MethodInfo* method);',
    'SetupArt': 'void CharacterView__SetupArt (CharacterView_o* __this, UnityEngine_Sprite_o* artSprite, int32_t type, const MethodInfo* method);'}


class Machine(CallerMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
        self.view_targets, self.view_addresses, self.view_ranges = [], set(), {}
        for address, name in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == 'CharacterView$$' + name]
            assert len(rows) == 1
            assert rows[0]['Signature'] == SIGNATURES[name]
            self.view_targets += rows
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == address:
                    chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks
            self.view_ranges[hex(address)] = [[hex(a), hex(b)] for a, b in chunks]
            for a, b in chunks:
                ins = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
                assert sum(i.size for i in ins) == b - a
                self.instructions.update({i.address: i for i in ins})
                self.view_addresses.update(i.address for i in ins)
        references = set()
        for address in self.view_addresses:
            i = self.instructions[address]
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + op.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
                self.flags.add(i.address + i.size + i.operands[0].mem.disp)
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                if row['Name'] not in self.bindings:
                    self.bindings[row['Name']] = self.arena + 0x4000 + len(self.bindings) * 0x200
                token = self.bindings[row['Name']]
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
                self.ids[token] = row['Name']
        self.set_id_method = 'Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<float, float, FloatOptions>>()'
        assert 'DG.Tweening.DOTween_TypeInfo' in self.bindings and self.set_id_method in self.bindings
        self.data_targets = []
        for address, name in SUPPLIED_GAME.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1; self.data_targets += rows
        for name, index, fields in [
                ('CharacterView', 5511, ['public Image bg; // 0x20', 'public CharacterData dataRef; // 0x28',
                                       'public TextMeshProUGUI chName; // 0x30', 'public Image art; // 0x38',
                                       'public Image clippingArt; // 0x40', 'public Image bgs; // 0x48',
                                       'public Image[] borders; // 0x50', 'public CanvasGroup canvasGroup; // 0x58',
                                       'private string animId; // 0x60']),
                ('CharacterData', 5845, ['public string characterName; // 0x28', 'public Sprite backgroundArt; // 0xB8',
                                       'public Color cardBgColor; // 0xF8', 'public Color cardBorderColor; // 0x108'])]:
            match = re.search(r'^public class ' + name + r'[^\n]*// TypeDefIndex: ' + str(index) +
                              r'\s*\{(.*?)// (?:Properties|Methods)', self.dump, re.M | re.S)
            assert match and all(f in match[1] for f in fields)
        for i, name in enumerate(['bg', 'bgs', 'border0', 'border1', 'art', 'clipping', 'text', 'canvas',
                                  'borders', 'anim_id', 'replacement_anim_id', 'background_sprite', 'art_sprite',
                                  'name', 'upper_name', 'image_class', 'text_class', 'color_method', 'text_method',
                                  'go_art', 'go_clipping']):
            self.p[name] = self.arena + 0x50000 + i * 0x1000
            self.ids[self.p[name]] = name
        self.color_service, self.text_service = self.stop + 0x210, self.stop + 0x220
        self.view_checks = {
            0x363D8C: ('mov', 'rdi, qword ptr [rbx + 0x60]'),
            0x363DA1: ('mov', 'dl, 1'), 0x363DBE: ('mov', 'rcx, qword ptr [rbx + 0x58]'),
            0x363DD1: ('mov', 'rdx, qword ptr [rbx + 0x60]'), 0x363DDF: ('jmp', '0x6bc9d0'),
            0x363E5A: ('xorps', 'xmm1, xmm1'),
            0x363F27: ('mov', 'qword ptr [rcx + 0x28], rdx'), 0x363F32: ('call', '0x2b6ff0'),
            0x363F55: ('movups', 'xmm0, xmmword ptr [rsi + 0xf8]'),
            0x363F6E: ('mov', 'rdi, qword ptr [rbp + 0x50]'),
            0x363F80: ('cmp', 'eax, dword ptr [rdi + 0x18]'),
            0x363F91: ('mov', 'rcx, qword ptr [rdi + rax*8 + 0x20]'),
            0x363FA7: ('movups', 'xmm0, xmmword ptr [rsi + 0x108]'),
            0x363FE6: ('mov', 'rbx, qword ptr [rbp + 0x30]'),
            0x364092: ('mov', 'r8d, eax'), 0x36409B: ('call', '0x3643f0'),
            0x364400: ('cmp', 'r8d, 0xa'), 0x364430: ('mov', 'rcx, qword ptr [rbx + 0x38]'),
            0x364492: ('mov', 'rcx, qword ptr [rbx + 0x40]'), 0x36446B: ('jmp', '0x1c7d810')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.view_checks.items())
        self.literals = {}
        for address in [0x363DAB, 0x363DB6, 0x363E4B, 0x36401E, 0x36404D, 0x3640A4]:
            i = self.instructions[address]; op = i.operands[-1]
            assert op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP
            slot = i.address + i.size + op.mem.disp
            section = self.pe.get_section_by_rva(slot)
            assert section and slot - section.VirtualAddress + op.size <= section.SizeOfRawData
            raw = self.pe.get_data(slot, op.size); assert len(raw) == op.size
            self.literals[hex(address)] = {'rva': hex(slot), 'bits': list(struct.unpack('<' + 'I' * (len(raw) // 4), raw))}
        assert self.literals['0x363dab']['bits'] == [0x3E4CCCCD]
        assert self.literals['0x363db6']['bits'] == [0x3F800000]
        assert all(self.literals[hex(a)]['bits'] == [0x3F800000] * 4 for a in [0x36401E, 0x36404D, 0x3640A4])

    def snapshot(self):
        out = super().snapshot()
        out.pop('supplied_view_state')
        view = self.p['view']
        out.update({'view': {name: self.oid(self.rq(view + offset)) for name, offset in
                            [('bg', 0x20), ('data', 0x28), ('text', 0x30), ('art', 0x38), ('clipping', 0x40),
                             ('bgs', 0x48), ('borders', 0x50), ('canvas', 0x58), ('anim_id', 0x60)]},
                    'border_length_bits': self.rq(self.p['borders'] + 0x18),
                    'border_slots': [self.oid(self.rq(self.p['borders'] + 0x20 + i * 8)) for i in range(3)],
                    'supplied_images': {name: dict(v, color_bits=v['color_bits'].copy()) for name, v in self.images.items()},
                    'supplied_text': self.text_value, 'supplied_game_objects': self.game_objects.copy(),
                    'supplied_tween_requests': self.tween_requests.copy(), 'supplied_kill_requests': self.kill_requests.copy(),
                    'supplied_set_id_requests': self.set_id_requests.copy(),
                    'dotween_class_initialized': self.rd(self.bindings['DG.Tweening.DOTween_TypeInfo'] + 0xE0),
                    'native_view_entries': self.native_view_entries.copy()})
        return out

    def prepare(self, options):
        super().prepare(options)
        self.images = {name: {'color_bits': [0xDEADBEEF] * 4, 'sprite': None} for name in ['bg', 'bgs', 'border0', 'border1', 'art', 'clipping']}
        self.game_objects = {'go_art': False, 'go_clipping': True}
        self.tween_requests, self.kill_requests, self.set_id_requests, self.native_view_entries = [], [], [], []
        self.text_value, self.fade_count = 'old supplied text', 0
        self.view_authored_offsets = set()
        self.data_inputs = []
        view = self.p['view']
        for offset, name in [(0x20, 'bg'), (0x28, 'data'), (0x30, 'text'), (0x38, 'art'), (0x40, 'clipping'),
                             (0x48, 'bgs'), (0x50, 'borders'), (0x58, 'canvas'), (0x60, 'anim_id')]:
            value = 0 if options.get('null_view_' + name) else self.p[name]
            if options.get('alias_images') and name in ['bg', 'bgs', 'art', 'clipping']: value = self.p['art']
            self.q(view + offset, value)
        self.q(self.p['image_class'] + 0x2A8, self.color_service)
        self.q(self.p['image_class'] + 0x2B0, self.p['color_method'])
        self.q(self.p['text_class'] + 0x558, self.text_service)
        self.q(self.p['text_class'] + 0x560, self.p['text_method'])
        for name in self.images: self.q(self.p[name], self.p['image_class'])
        self.q(self.p['text'], self.p['text_class'])
        self.q(self.p['borders'] + 0x18, options.get('border_length_bits', 0xFACE000000000002))
        slots = ['border0', 'border0' if options.get('alias_borders') else 'border1', 'art']
        for i, name in enumerate(slots): self.q(self.p['borders'] + 0x20 + i * 8, 0 if options.get('null_border_index') == i else self.p[name])
        for name, colors in [('data', [0x3F000001, 0x80000000, 0x7FC01234, 0x3F800000]),
                             ('bluff', [0x3E000001, 0x3F000002, 0x3F000003, 0x3F000004])]:
            for i, bits in enumerate(colors): self.d(self.p[name] + 0xF8 + i * 4, bits)
            for i, bits in enumerate(reversed(colors)): self.d(self.p[name] + 0x108 + i * 4, bits)
            self.q(self.p[name] + 0xB8, 0 if options.get('null_background_sprite') else self.p['background_sprite'])
            self.q(self.p[name] + 0x28, 0 if options.get('null_name') else self.p['name'])

    def mutate_view(self, phase):
        if self.options.get('view_mutation_phase') != phase: return
        action = self.options['view_mutation']; view = self.p['view']
        fields = {'clear_art': (0x38, 0), 'clear_clipping': (0x40, 0), 'clear_text': (0x30, 0),
                  'clear_borders_ref': (0x50, 0), 'replace_anim_id': (0x60, self.p['replacement_anim_id']),
                  'replace_data_ref': (0x28, self.p['bluff'])}
        if action in fields:
            offset, value = fields[action]; self.q(view + offset, value)
            self.view_authored_offsets.update(range(offset, offset + 8))
        elif action == 'clear_second_border': self.q(self.p['borders'] + 0x28, 0)
        elif action == 'shrink_borders': self.q(self.p['borders'] + 0x18, 1)
        else: raise AssertionError(action)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        self.executed.add(rva)
        if rva in TARGETS:
            assert cx == self.p['view']
            row = {'method': TARGETS[rva], 'view': self.oid(cx)}
            if rva == 0x363F10: row['data'] = self.oid(dx)
            if rva == 0x3643F0: row.update({'sprite': self.oid(dx), 'type_bits': r8 & 0xFFFFFFFF})
            self.native_view_entries.append(row)
        if rva in self.instructions: return
        if rva == 0x281D90 and cx == self.bindings['DG.Tweening.DOTween_TypeInfo']:
            if self.event('class_initialization_service', [self.oid(cx)]):
                self.d(cx + 0xE0, 1); self.mutate_view('class_init'); self.ret()
        elif rva == 0x5044D0:
            assert cx in [0, self.p['anim_id'], self.p['replacement_anim_id']] and dx & 0xFF == 1 and r8 == 0
            record = [self.oid(cx), dx, r8]
            if self.event('dotween_kill_service', record):
                self.kill_requests.append(record); self.mutate_view('kill'); self.ret(0xFACE000000000007)
        elif rva == 0x349E80:
            assert cx in [0, self.p['canvas']] and r9 == 0
            end, duration = [self.reg(getattr(x, 'UC_X86_REG_XMM' + str(i))) & 0xFFFFFFFF for i in [1, 2]]
            assert end in [0, 0x3F800000] and duration == 0x3E4CCCCD
            record = [self.oid(cx), end, duration]
            if self.event('dofade_service', record):
                self.tween_requests.append(record); self.mutate_view('dofade')
                p = 0 if self.options.get('null_tween') else self.arena + 0x90000 + self.fade_count * 0x100
                if p: self.ids[p] = 'tween' + str(self.fade_count)
                self.fade_count += 1; self.ret(p)
        elif rva == 0x6BC9D0:
            assert cx == 0 or self.ids.get(cx, '').startswith('tween')
            assert dx in [0, self.p['anim_id'], self.p['replacement_anim_id']] and r8 == self.bindings[self.set_id_method]
            record = [self.oid(cx), self.oid(dx), self.oid(r8)]
            if self.event('set_id_service', record): self.set_id_requests.append(record); self.ret(cx)
        elif rva == 0x2B6FF0:
            assert cx == self.p['view'] + 0x28 and self.rq(cx) == dx and dx in [0, self.p['data'], self.p['bluff']]
            if self.event('view_data_barrier_service', [cx - self.arena, self.oid(dx)]):
                self.mutate_view('barrier'); self.ret()
        elif address == self.color_service:
            assert cx in [self.p[n] for n in self.images] and r8 == self.p['color_method']
            bits = list(struct.unpack('<IIII', uc.mem_read(dx, 16)))
            return_rva = self.rq(self.reg(x.UC_X86_REG_RSP)) - self.base
            captured_data = self.reg(x.UC_X86_REG_RSI)
            assert captured_data in [self.p['data'], self.p['bluff']]
            if return_rva in [0x363F6E, 0x363FC0]:
                offset = 0xF8 if return_rva == 0x363F6E else 0x108
                assert bits == [self.rd(captured_data + offset + i * 4) for i in range(4)]
            else:
                assert return_rva in [0x364049, 0x364078, 0x3640CB] and bits == [0x3F800000] * 4
            name = self.oid(cx)
            if self.event('image_color_service', [name, bits, self.oid(r8)]):
                self.images[name]['color_bits'] = bits
                self.mutate_view('color:' + name); self.ret()
        elif rva == 0x1D49700:
            assert cx in [self.p[n] for n in self.images] and dx in [0, self.p['background_sprite'], self.p['art_sprite']] and r8 == 0
            if self.event('image_sprite_service', [self.oid(cx), self.oid(dx)]):
                self.images[self.oid(cx)]['sprite'] = self.oid(dx)
                self.mutate_view('sprite:' + self.oid(cx)); self.ret()
        elif rva == 0xF7B1B0:
            assert cx == self.p['name'] and dx == 0
            result = 0 if self.options.get('null_upper_result') else self.p['upper_name']
            if self.event('uppercase_service', [self.oid(cx), self.oid(result)]):
                self.mutate_view('uppercase'); self.ret(result)
        elif address == self.text_service:
            assert cx == self.p['text'] and dx in [0, self.p['upper_name']] and r8 == self.p['text_method']
            if self.event('text_setter_service', [self.oid(cx), self.oid(dx), self.oid(r8)]): self.text_value = self.oid(dx); self.ret()
        elif rva in SUPPLIED_GAME:
            assert cx in [self.p['data'], self.p['bluff']] and dx == 0
            assert cx == self.reg(x.UC_X86_REG_RSI)  # Captured argument, not reloaded view.dataRef.
            record = [self.oid(cx)]
            kind = 'get_art_service' if rva == 0x3B4AB0 else 'get_art_type_service'
            if self.event(kind, record):
                self.data_inputs.append(record); self.mutate_view(kind)
                self.ret((0 if self.options.get('null_art_sprite') else self.p['art_sprite']) if rva == 0x3B4AB0 else
                         (0xFACE000000000000 | self.options.get('art_type_bits', 0)))
        elif rva == 0x1C79FD0:
            assert cx in [self.p['art'], self.p['clipping']] and dx == 0
            name = self.oid(cx)
            result = 0 if self.options.get('null_game_for') == name else self.p['go_art' if name == 'art' else 'go_clipping']
            if self.event('image_game_object_service', [name, self.oid(result)]):
                self.mutate_view('game_object:' + name); self.ret(result)
        elif rva == 0x1C7D810 and cx in [self.p['go_art'], self.p['go_clipping']]:
            assert dx & 0xFF in [0, 1] and r8 == 0
            if self.event('image_set_active_service', [self.oid(cx), dx, dx & 0xFF]):
                self.game_objects[self.oid(cx)] = bool(dx & 0xFF)
                self.mutate_view('set_active:' + self.oid(cx)); self.ret()
        elif rva == 0x2B7D80:
            self.event('native_bounds_guard', []); self.error = 'native_bounds_guard'; uc.emu_stop()
        else:
            super().hook(uc, address, size, data)

    def invoke_view(self, address, owner, argument=0, type_bits=0):
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop); self.q(sp + 0x28, 0)
        regs = [getattr(x, 'UC_X86_REG_' + n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, reg in enumerate(regs): self.u.reg_write(reg, 0xFAB00000 + i)
        for i, reg in enumerate(vectors): self.u.reg_write(reg, (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64))
        for reg, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, owner), (x.UC_X86_REG_RDX, argument),
                           (x.UC_X86_REG_R8, 0xFACE000000000000 | type_bits)]: self.u.reg_write(reg, value)
        self.u.emu_start(self.base + address, self.stop, timeout=10_000_000, count=10000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(r) == 0xFAB00000 + i for i, r in enumerate(regs))
            assert all(self.reg(r) == (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64) for i, r in enumerate(vectors))
        return returned

    def run_view(self, name, options=None, joined=False, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.authored_offsets, self.view_authored_offsets = set(), set()
        actor, view = self.p['actor'], self.p['view']
        before_a, before_v = [bytes(self.u.mem_read(p, 0x200)) for p in [actor, view]]
        initial, old_count = self.snapshot(), len(self.events)
        if joined:
            address = next(a for a, n in CALLER_TARGETS.items() if n == name)
            returned = self.invoke_view(address, actor, 0xABCDEF1234567890)
        else:
            address = next(a for a, n in TARGETS.items() if n == name)
            argument = 0 if self.options.get('null_argument') else self.p['data'] if name == 'Init' else self.p['art_sprite'] if name == 'SetupArt' else 0xABCDEF1234567890
            returned = self.invoke_view(address, view, argument, self.options.get('art_type_bits', 0))
        after_a, after_v = [bytes(self.u.mem_read(p, 0x200)) for p in [actor, view]]
        actor_allowed = self.authored_offsets | ({0x1A0} if joined else set())
        view_allowed = self.view_authored_offsets | (set(range(0x28, 0x30)) if any(e['kind'] == 'view_data_barrier_service' for e in self.events[old_count:]) else set())
        assert all(i in actor_allowed or before_a[i] == after_a[i] for i in range(0x200))
        assert all(i in view_allowed or before_v[i] == after_v[i] for i in range(0x200))
        if returned and name in ['AnimateIn', 'AnimateOut']:
            expected_end = 0x3F800000 if name == 'AnimateIn' else 0
            assert self.tween_requests[-1] == [initial['view']['canvas'], expected_end, 0x3E4CCCCD]
            assert self.kill_requests[-1][0] == initial['view']['anim_id']
            assert self.set_id_requests[-1][1] == self.oid(self.rq(view + 0x60))
        if returned and name == 'Init' and not self.options.get('view_mutation_phase'):
            raw_count = initial['border_length_bits'] & 0xFFFFFFFF
            count = raw_count if raw_count < 0x80000000 else 0
            color_targets = [e['args'][0] for e in self.events[old_count:] if e['kind'] == 'image_color_service']
            assert color_targets == [initial['view']['bgs'], *initial['border_slots'][:count],
                                     initial['view']['art'], initial['view']['clipping'], initial['view']['bg']]
        return {'method': name, 'joined': joined, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'initial': initial, 'events': self.events[old_count:].copy(), 'final': self.snapshot(),
                'other_actor_and_view_bytes_retained': True}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, joins, sequences, baselines, failures = [], [], [], [], []
    for name, cold, class_cold, null_id, null_canvas, null_tween in itertools.product(
            ['AnimateIn', 'AnimateOut'], [False, True], [False, True], [False, True], [False, True], [False, True]):
        c = m.run_view(name, {'cold': cold, 'class_cold': class_cold, 'null_view_anim_id': null_id,
                              'null_view_canvas': null_canvas, 'null_tween': null_tween})
        assert c['returned']; cases.append(c)
    for name, phase in itertools.product(['AnimateIn', 'AnimateOut'], ['class_init', 'kill', 'dofade']):
        cases.append(m.run_view(name, {'class_cold': True, 'view_mutation_phase': phase, 'view_mutation': 'replace_anim_id'}))
    for name in ['AnimateIn', 'AnimateOut']:
        cases.append(m.run_view(name, {'warm_byte': 0x80, 'class_word': 0xDEADBEEF}))
    for art_type, count, alias, cold in itertools.product([0, 10, 20, 0x8000000A, 0xFFFFFFFF], [0, 1, 2, 3], [False, True], [False, True]):
        c = m.run_view('Init', {'art_type_bits': art_type, 'border_length_bits': 0xFACE000000000000 | count,
                              'alias_borders': alias, 'cold': cold})
        assert c['returned']; cases.append(c)
    for field in ['bg', 'text', 'art', 'clipping', 'bgs', 'borders']:
        c = m.run_view('Init', {'null_view_' + field: True}); assert not c['returned']; cases.append(c)
    for options in [{'null_argument': True}, {'null_name': True}, {'null_border_index': 0}, {'null_border_index': 1},
                    {'null_background_sprite': True}, {'null_upper_result': True}, {'null_art_sprite': True},
                    {'alias_images': True}, {'border_length_bits': 0xFACE000080000000},
                    {'null_game_for': 'art'}, {'null_game_for': 'clipping'}]:
        cases.append(m.run_view('Init', options))
    for phase, action in [('barrier', 'replace_data_ref'), ('color:bgs', 'clear_borders_ref'),
                          ('color:border0', 'clear_borders_ref'), ('color:border0', 'clear_second_border'),
                          ('color:border0', 'shrink_borders'), ('uppercase', 'clear_text'),
                          ('get_art_service', 'clear_art'), ('set_active:go_art', 'clear_art')]:
        cases.append(m.run_view('Init', {'view_mutation_phase': phase, 'view_mutation': action}))
    for art_type, alias, null_sprite in itertools.product([0, 10, 20, 0xFFFFFFFF], [False, True], [False, True]):
        c = m.run_view('SetupArt', {'art_type_bits': art_type, 'alias_images': alias, 'null_argument': null_sprite})
        assert c['returned']; cases.append(c)
    for name, cold, alias, art_type in itertools.product(['ShowDisguise', 'HideDisguise'], [False, True], [False, True], [0, 10]):
        c = m.run_view(name, {'cold': cold, 'class_cold': True, 'alias_images': alias, 'alias_borders': alias,
                             'art_type_bits': art_type}, joined=True)
        assert c['returned']; joins.append(c)
    for options in [{'null_view_bgs': True}, {'null_name': True}, {'null_border_index': 1},
                    {'view_mutation_phase': 'color:border0', 'view_mutation': 'clear_second_border'},
                    {'view_mutation_phase': 'set_active:go_art', 'view_mutation': 'clear_art'},
                    {'mutation_phase': 'callback', 'mutation': 'callback_actor_bytes'}]:
        joins.append(m.run_view('ShowDisguise', options, joined=True))
    for alias in [False, True]:
        m.prepare({'cold': True, 'class_cold': True, 'alias_images': alias, 'alias_borders': alias})
        calls = [m.run_view(name, joined=True, retained=True) for name in ['ShowDisguise', 'HideDisguise', 'ShowDisguise']]
        assert all(c['returned'] for c in calls); sequences.append({'aliases': alias, 'calls': calls})
    for name, joined, options in [('AnimateIn', False, {'cold': True, 'class_cold': True}),
                                  ('AnimateOut', False, {'cold': True, 'class_cold': True}),
                                  ('Init', False, {}), ('Init', False, {'art_type_bits': 10}),
                                  ('SetupArt', False, {}), ('SetupArt', False, {'art_type_bits': 10}),
                                  ('ShowDisguise', True, {'cold': True, 'class_cold': True}),
                                  ('HideDisguise', True, {'cold': True, 'class_cold': True})]:
        baseline = m.run_view(name, options, joined=joined); assert baseline['returned']
        ordinal = len(baselines); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run_view(name, dict(options, failure=[kind, counts[kind]]), joined=joined)
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index + 1]
            assert stopped['final'] == e['snapshot']
            failures.append({'baseline': ordinal, 'failure': [kind, counts[kind]], 'prefix_length': index + 1,
                             'exact_snapshot_verified': True})
    missing = set(m.view_addresses) - m.executed
    assert all(m.instructions[a].mnemonic in ['int3', 'call'] and
               (m.instructions[a].mnemonic == 'int3' or m.instructions[a].op_str == '0x2b7d80') for a in missing)
    return {'build': BUILD, 'targets': m.view_targets, 'caller_ranges': m.view_ranges,
            'instruction_assertions': len(m.view_checks), 'literal_bindings': m.literals,
            'case_count': len(cases), 'cases': cases, 'joined_case_count': len(joins), 'joined_cases': joins,
            'retained_sequences': sequences, 'failure_baselines': baselines,
            'failure_case_count': len(failures), 'failure_cases': failures,
            'caller_instructions_decoded': len(m.view_addresses),
            'caller_instructions_executed': len(m.executed & m.view_addresses),
            'unexecuted_bounds_gateway_and_terminal_traps': len(missing),
            'native_execution_addresses': len(m.executed), 'supplied_game_owned_metadata': m.data_targets,
            'scope': 'Actual CharacterView AnimateIn/AnimateOut/Init/SetupArt and joined Character disguise callers execute. DOTween, Image/TMP/Unity setters/getters, CharacterData art/type, uppercase, metadata/class services and hint callback implementation remain supplied. Captured/reloaded pointers, widths, aliases and stopped partial effects are retained. Scheduler, renderer, real animation completion/lifetimes, exception unwinding and concurrent array-size races remain unclaimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'joined_case_count', 'failure_case_count', 'caller_instructions_executed', 'native_execution_addresses']}))
