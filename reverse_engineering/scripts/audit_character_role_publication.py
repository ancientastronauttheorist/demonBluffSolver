"""Join installed role callbacks to native history and explicit speech resumes."""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_role_callback import Machine as CallbackMachine, verify_native


TARGETS = {0x376240: 'Character.<ShowInfoDelayed>d__134$$MoveNext',
           0x364C40: 'Character$$GetCharacterBluffIfAble', 0x35DD10: 'Acted$$Act',
           0xF76390: 'System.String$$IsNullOrEmpty'}


class Machine(CallbackMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
        for address, name in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1
            self.targets += rows
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4:
                    root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == address:
                    chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            if address == 0xF76390:
                assert not chunks
                chunks = [(address, 0xF763A1)]  # Both verified leaf returns, no padding.
            assert chunks
            self.ranges[hex(address)] = [[hex(a), hex(b)] for a, b in chunks]
            for a, b in chunks:
                ins = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
                assert sum(i.size for i in ins) == b - a
                self.instructions.update({i.address: i for i in ins})
        chunks = [(0x2EB0, 0x2F0F)]
        assert any(e.struct.BeginAddress == 0x2EB0 and e.struct.EndAddress == 0x2F0F
                   for e in self.pe.DIRECTORY_ENTRY_EXCEPTION)
        self.ranges['0x2eb0'] = [[hex(a), hex(b)] for a, b in chunks]
        for a, b in chunks:
            ins = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
            assert sum(i.size for i in ins) == b - a
            self.instructions.update({i.address: i for i in ins})
        references = set()
        for i in self.instructions.values():
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + op.mem.disp)
                    if i.mnemonic == 'cmp' and op.size == 1:
                        self.flags.add(i.address + i.size + op.mem.disp)
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                if row['Name'] not in self.bindings:
                    self.bindings[row['Name']] = self.arena + 0x4000 + len(self.bindings) * 0x200
                token = self.bindings[row['Name']]
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
        self.strings, self.string_labels = {}, {}
        for row in self.metadata['ScriptString']:
            if row['Address'] in references:
                token = self.arena + 0x80000 + len(self.strings) * 0x100
                self.make_string(token, row['Value'], 'literal:' + row['Value'])
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
        assert 'literal:Character: ' in self.string_labels.values()
        names = ['history', 'history_array', 'references', 'replacement_refs', 'prior_info',
                 'acted', 'version', 'blank', 'text_class', 'layout_array', 'rect',
                 'game', 'pickable', 'data', 'bluff', 'about_delegate', 'event_delegate',
                 'gameplay_static', 'trailer_static', 'game_data_static', 'trailer_controller',
                 'trailer_info', 'generic_context', 'generic_method', 'generic_types']
        self.fixtures = {name: self.arena + 0x60000 + i * 0x1000 for i, name in enumerate(names)}
        self.static_ids = {p: name for name, p in self.fixtures.items()}
        for i, (name, value) in enumerate([('original', 'authored clue'), ('replacement', 'authored replacement'),
                                           ('empty', ''), ('old', 'old speech'), ('prior', 'prior clue'),
                                           ('object_name', 'Fixture'), ('log', 'Character: Fixture'),
                                           ('trailer', 'authored trailer')]):
            self.make_string(self.arena + 0xA0000 + i * 0x100, value, name)
        self.string_pointers = {name: p for p, name in self.string_labels.items()}
        self.about_service, self.info_event_service, self.text_service = [self.stop + n for n in [0x300, 0x310, 0x320]]
        wait_load = self.instructions[0x376486]
        slot = wait_load.address + wait_load.size + wait_load.operands[1].mem.disp
        self.speech_wait_bits = struct.unpack('<I', self.pe.get_data(slot, 4))[0]
        self.service_rvas = {0x2B7B40, 0x2B7D40, 0x2B6FF0, 0x4D5B60, 0x1C7F160, 0x1C961F0,
                             0x2B7D90, 0x2B7D80, 0x281D90, 0x38AD80, 0x1C79FD0, 0x1C82250,
                             0xF71C60, 0x1C4B380, 0x1C822C0, 0x1C7D810, 0x35D920, 0x1EC1010, 0xB54090}

    def make_string(self, pointer, text, label):
        raw = text.encode('utf-16-le')
        self.d(pointer + 0x10, len(raw) // 2)
        self.u.mem_write(pointer + 0x14, raw + b'\0\0')
        self.strings[pointer], self.string_labels[pointer] = text, label

    def object_id(self, pointer):
        if pointer in self.static_ids:
            return self.static_ids[pointer]
        if pointer in self.string_labels:
            return self.string_labels[pointer]
        return super().object_id(pointer)

    def speech_iterator(self, pointer):
        return {'id': self.object_id(pointer), 'state': self.rd(pointer + 0x10),
                'current': self.object_id(self.rq(pointer + 0x18)),
                'owner': self.object_id(self.rq(pointer + 0x20)),
                'description': self.object_id(self.rq(pointer + 0x28))}

    def snapshot(self):
        result = super().snapshot()
        f = self.fixtures
        size = self.rd(f['history'] + 0x18)
        assert size <= 16
        history = [self.object_id(self.rq(f['history_array'] + 0x20 + i * 8)) for i in range(size)]
        records = []
        for i in range(size):
            p = self.rq(f['history_array'] + 0x20 + i * 8)
            records.append({'id': self.object_id(p), 'description': self.object_id(self.rq(p + 0x10)),
                            'references': self.object_id(self.rq(p + 0x18))})
        result.update(history=history, history_version=self.rd(f['history'] + 0x1C),
                      history_records=records,
                      history_slots=[self.object_id(self.rq(f['history_array'] + 0x20 + i * 8)) for i in range(4)],
                      info={'description': self.object_id(self.rq(self.info + 0x10)),
                            'references': self.object_id(self.rq(self.info + 0x18))},
                      uses_bits=self.rd(self.actor + 0xDC), saved_speech=self.object_id(self.rq(self.actor + 0x198)),
                      speech_iterators=[self.speech_iterator(p) for p, name in self.objects.items() if name.startswith('speech')],
                      speech_registered=self.speech_registered.copy(), speech_resumes=self.speech_resumes.copy(),
                      text=self.display_text, shown=self.shown.copy(), active=self.active.copy())
        return result

    def prepare(self, options):
        super().prepare(options)
        f, o = self.fixtures, self.options
        self.speech_registered, self.speech_resumes, self.shown, self.active = [], [], [], {}
        self.display_text = 'old'
        for pointer in f.values():
            self.u.mem_write(pointer, bytes(0x800))
        for type_name in ['UnityEngine.Debug_TypeInfo', 'UnityEngine.UI.LayoutRebuilder_TypeInfo',
                          'UnityEngine.Object_TypeInfo', 'GameData_TypeInfo']:
            assert type_name in self.bindings
            self.d(self.bindings[type_name] + 0xE0, 0 if o.get('cold') else 1)
        self.q(self.bindings['GameplayEvents_TypeInfo'] + 0xB8, f['gameplay_static'])
        self.q(f['gameplay_static'] + 0x58, 0 if o.get('no_event') else f['event_delegate'])
        self.q(self.bindings['GameData_TypeInfo'] + 0xB8, f['game_data_static'])
        self.u.mem_write(f['game_data_static'] + 0x1E, bytes([int(o.get('trailer', False))]))
        self.q(self.bindings['TrailerCharacters_TypeInfo'] + 0xB8, f['trailer_static'])
        self.q(f['trailer_static'], 0 if o.get('null_trailer_controller') else f['trailer_controller'])
        self.q(f['trailer_info'] + 0x18, 0 if o.get('null_trailer_text') else self.string_pointers['empty' if o.get('empty_trailer') else 'trailer'])
        self.q(self.actor + 0x50, 0 if o.get('null_data') else f['data'])
        self.q(self.actor + 0x58, f['bluff'] if o.get('bluff') else 0)
        self.u.mem_write(self.actor + 0xD8, bytes([int(o.get('revealed', False))]))
        self.d(self.actor + 0xDC, o.get('uses', 1) & 0xFFFFFFFF)
        self.d(self.actor + 0xE4, o.get('state', 10))
        self.d(self.actor + 0x118, 2)
        self.q(self.actor + 0x148, 0 if o.get('null_history') else f['history'])
        self.q(self.actor + 0x198, self.string_pointers['old'])
        self.u.mem_write(self.actor + 0x1A1, bytes([int(o.get('act', True))]))
        self.q(self.actor + 0x1A8, 0 if o.get('null_pickable') else f['pickable'])
        self.q(self.actor + 0x1B0, f['about_delegate'] if o.get('about') else 0)
        self.q(self.actor + 0xA8, 0 if o.get('null_acteds') else f['acted'])
        self.q(self.info + 0x10, 0 if o.get('null_description') else self.string_pointers['empty' if o.get('empty_description') else 'original'])
        self.q(self.info + 0x18, f['references'])
        self.q(f['prior_info'] + 0x10, self.string_pointers['prior'])
        self.q(f['prior_info'] + 0x18, f['references'])
        self.d(f['history'] + 0x18, 1)
        self.d(f['history'] + 0x1C, 9)
        self.q(f['history'] + 0x10, 0 if o.get('null_history_array') else f['history_array'])
        self.d(f['history_array'] + 0x18, o.get('capacity', 8))
        self.q(f['history_array'] + 0x20, f['prior_info'])
        self.d(f['references'] + 0x18, 0)
        self.d(f['replacement_refs'] + 0x18, 1)
        self.q(f['replacement_refs'] + 0x10, f['layout_array'])
        self.q(f['about_delegate'] + 0x18, self.about_service)
        self.q(f['about_delegate'] + 0x28, 0xBA01)
        self.q(f['about_delegate'] + 0x40, f['about_delegate'])
        self.q(f['event_delegate'] + 0x18, self.info_event_service)
        self.q(f['event_delegate'] + 0x28, 0xBA02)
        self.q(f['event_delegate'] + 0x40, f['event_delegate'])
        self.q(f['acted'] + 0x20, 0 if o.get('null_version') else f['version'])
        self.q(f['version'] + 0x28, 0 if o.get('null_blank') else f['blank'])
        self.q(f['blank'], f['text_class'])
        self.q(f['text_class'] + 0x558, self.text_service)
        self.q(f['text_class'] + 0x560, 0xBA03)
        self.q(f['acted'] + 0x28, 0 if o.get('null_layouts') else f['layout_array'])
        self.d(f['layout_array'] + 0x18, 2)
        self.q(f['layout_array'] + 0x20, f['rect'])
        self.q(f['layout_array'] + 0x28, f['rect'])
        self.u.mem_write(f['data'] + 0x13E, bytes([int(o.get('picking', False))]))
        self.u.mem_write(f['bluff'] + 0x13E, bytes([int(o.get('bluff_picking', True))]))
        method = self.bindings['Method$System.Collections.Generic.List<ActedInfo>.Add()']
        self.q(method + 0x20, f['generic_context'])
        self.q(f['generic_context'] + 0xC0, f['generic_types'])
        self.q(f['generic_types'] + 0x70, f['generic_method'])

    def hook(self, uc, address, size, data):
        rva, x, f = address - self.base, self.x, self.fixtures
        assert rva in self.instructions or rva in self.service_rvas or rva == 0x33ED50 or address in {
            self.role_act, self.role_bluff, self.after_callback, self.after_wait,
            self.about_service, self.info_event_service, self.text_service}
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]]
        if rva == 0x2B7D40 and cx == self.bindings['Character.<ShowInfoDelayed>d__134_TypeInfo']:
            if self.event('allocate_service', ['speech']):
                p = self.alloc(0x100)
                self.q(p, cx)
                self.objects[p] = 'speech' + str(len(self.objects))
                self.ret(p)
        elif rva == 0x1C7F160 and self.objects.get(dx, '').startswith('speech'):
            assert cx == self.actor_arg and r8 == 0
            item = self.speech_iterator(dx)
            assert item['state'] == 0 and item['current'] is None and item['owner'] == 'actor'
            if self.event('speech_registration_service', [item]):
                self.speech_registered.append(item)
                self.ret(self.arena + 0xC0000 + len(self.speech_registered) * 0x100)
        elif address == self.about_service:
            assert cx == f['about_delegate'] and dx == self.info and r9 == 0xBA01
            if self.event('preappend_delegate_service', [r8 & 0xFFFFFFFF]):
                if self.options.get('about_mutation', True):
                    self.q(self.info + 0x10, 0 if self.options.get('about_null_desc') else self.string_pointers['empty' if self.options.get('about_empty_desc') else 'replacement'])
                    self.q(self.info + 0x18, f['replacement_refs'])
                if self.options.get('about_null_history'):
                    self.q(self.actor + 0x148, 0)
                self.ret()
        elif address == self.info_event_service:
            assert cx == f['event_delegate'] and dx == self.actor and r8 == 0xBA02
            if self.event('info_revealed_delegate_service', []):
                if self.options.get('event_mutation'):
                    self.q(self.info + 0x10, self.string_pointers['replacement'])
                self.ret()
        elif rva == 0xB54090:
            assert cx == f['history'] and dx == self.info and r8 == f['generic_method']
            if self.event('list_grow_service', []):
                # Capacity growth is explicit; preceding native version increment persists.
                n = self.rd(f['history'] + 0x18)
                self.d(f['history_array'] + 0x18, 8)
                self.q(f['history_array'] + 0x20 + n * 8, dx)
                self.d(f['history'] + 0x18, n + 1)
                self.ret()
        elif rva == 0x281D90:
            name = next(n for n, p in self.bindings.items() if p == cx)
            if self.event('class_initialization_service', [name]):
                self.d(cx + 0xE0, 1)
                self.ret()
        elif rva == 0x38AD80:
            assert cx == f['trailer_controller'] and dx == 2 and r8 == 0
            if self.event('trailer_lookup_service', []):
                self.ret(0 if self.options.get('null_trailer') else f['trailer_info'])
        elif rva == 0x1C79FD0:
            assert cx == f['acted'] and dx == 0
            if self.event('game_object_service', []):
                self.ret(0 if self.options.get('null_game_call') == self.counts['game_object_service'] else f['game'])
        elif rva == 0x1C82250:
            assert cx == f['game'] and dx == 0
            if self.event('name_service', []):
                self.ret(self.string_pointers['object_name'])
        elif rva == 0xF71C60:
            assert self.strings[cx] == 'Character: ' and self.strings[dx] == 'Fixture' and r8 == 0
            if self.event('concat_service', []):
                self.ret(self.string_pointers['log'])
        elif rva == 0x1C4B380:
            assert cx == self.string_pointers['log'] and dx in [0, f['game']] and r8 == 0
            if self.event('log_service', [self.object_id(dx)]):
                self.ret()
        elif address == self.text_service:
            assert cx == f['blank'] and r8 == 0xBA03
            if self.event('text_setter_service', [self.object_id(dx)]):
                self.display_text = self.object_id(dx)
                self.ret()
        elif rva == 0x1C822C0:
            assert cx in [0, f['bluff']] and dx == r8 == 0
            if self.event('unity_null_service', [self.object_id(cx)]):
                self.ret(0xABC000 | int(cx == 0 or self.options.get('destroyed_bluff', False)))
        elif rva == 0x1C7D810:
            assert cx in [f['pickable'], f['game']] and r8 == 0
            value = bool(dx & 0xFF)
            if self.event('set_active_service', [self.object_id(cx), value]):
                self.active[self.object_id(cx)] = value
                self.ret()
        elif rva == 0x35D920:
            assert cx == f['version'] and r8 == 0
            if self.event('show_version_service', [self.object_id(dx)]):
                self.shown.append(self.object_id(dx))
                self.ret()
        elif rva == 0x1EC1010:
            assert cx == f['rect'] and dx == 0
            if self.event('layout_service', [self.object_id(cx)]):
                self.ret()
        elif rva == 0x1C961F0:
            assert self.objects[cx].startswith('wait') and r8 == 0
            bits = self.reg(x.UC_X86_REG_XMM1) & 0xFFFFFFFF
            assert bits in [0, self.speech_wait_bits]
            if self.event('wait_constructor_service', [self.object_id(cx), bits]):
                self.d(cx + 0x10, bits)
                self.ret()
        elif rva == 0x2B7D80:
            self.error = 'native_bounds_guard'
            uc.emu_stop()
        else:
            super().hook(uc, address, size, data)

    def invoke(self, address, cx, dx=0, r8=0, r9=0):
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        self.q(sp + 0x28, 0)
        keep = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(keep):
            self.u.reg_write(register, 0xFAB00000 + i)
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, cx), (x.UC_X86_REG_RDX, dx),
                                (x.UC_X86_REG_R8, r8), (x.UC_X86_REG_R9, r9)]:
            self.u.reg_write(register, value & 0xFFFFFFFFFFFFFFFF)
        saved_xmm6 = 0xFEDCBA98765432100123456789ABCDEF
        self.u.reg_write(x.UC_X86_REG_XMM6, saved_xmm6)
        before = bytearray(self.u.mem_read(self.actor, 0x200))
        self.u.emu_start(self.base + address, self.stop, timeout=10_000_000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(register) == 0xFAB00000 + i for i, register in enumerate(keep))
            assert self.reg(x.UC_X86_REG_XMM6) == saved_xmm6 and not self.frames and not self.native_calls
        after = bytearray(self.u.mem_read(self.actor, 0x200))
        for offset, width in [(0xDC, 4), (0x148, 8), (0x198, 8)]:
            before[offset:offset + width] = after[offset:offset + width] = bytes(width)
        assert before == after
        return returned

    def run(self, options=None):
        self.prepare(options or {})
        returned = self.invoke(0x368790, self.actor_arg, 0 if self.options.get('null_role') else self.role,
                               self.options.get('trigger', 30), self.options.get('route', 0))
        return self.finish_publication(returned)

    def finish_publication(self, returned):
        order = list(self.registered)
        if self.options.get('reverse_results'):
            order.reverse()
        resumes = []
        for item in order:
            if not returned:
                break
            pointer = next(p for p, name in self.objects.items() if name == item['id'])
            returned = self.invoke(0x375FE0, pointer)
            if returned:
                result = self.reg(self.x.UC_X86_REG_RAX) & 0xFF
                assert result == 0
                resumes.append({'iterator': item['id'], 'result': result})
        for item in list(self.speech_registered):
            if not returned:
                break
            pointer = next(p for p, name in self.objects.items() if name == item['id'])
            for _ in range(3):
                returned = self.invoke(0x376240, pointer)
                if not returned:
                    break
                result = self.reg(self.x.UC_X86_REG_RAX) & 0xFF
                self.speech_resumes.append({'iterator': item['id'], 'result': result, 'state': self.rd(pointer + 0x10)})
                if result == 0:
                    assert self.invoke(0x376240, pointer) and self.reg(self.x.UC_X86_REG_RAX) & 0xFF == 0
                    break
        return {'options': self.options, 'returned': returned, 'error': self.error,
                'events': self.events.copy(), 'result_resumes': resumes, 'final': self.snapshot()}


def verify_publication(m):
    base = verify_native(m)
    fields = {
        ('Character', 5487): ['private int pickableUses; // 0xDC', 'public ECharacterState state; // 0xE4',
                             'public CharacterData dataRef; // 0x50', 'public CharacterData bluff; // 0x58',
                             'public bool revealed; // 0xD8', 'public Acted acteds; // 0xA8',
                             'public List<ActedInfo> actedInfos; // 0x148', 'private string savedAct; // 0x198',
                             'public bool act; // 0x1A1', 'public GameObject pickable; // 0x1A8',
                             'public Action<ActedInfo, ETriggerPhase> onAboutToAct; // 0x1B0'],
        ('ActedInfo', 5498): ['public string desc; // 0x10', 'public List<Character> characters; // 0x18'],
        ('GameData', 5928): ['public static bool TrailerCharacters; // 0x1E'],
        ('TrailerCharacters', 5591): ['public static TrailerCharacters Instance; // 0x0'],
        ('CharacterTrailerInfo', 5592): ['public string text; // 0x18'],
        ('Acted', 5477): ['public ActedVersion acted; // 0x20', 'public RectTransform[] layoutsToRebuild; // 0x28'],
        ('ActedVersion', 5478): ['public TextMeshProUGUI blankText; // 0x28'],
        ('CharacterData', 5845): ['public bool picking; // 0x13E'],
        ('GameplayEvents', 5519): ['public static Action<Character> OnCharacterInfoRevealed; // 0x58'],
        ('Character.<ShowInfoDelayed>d__134', 5486): ['private int <>1__state; // 0x10',
                                                   'private object <>2__current; // 0x18',
                                                   'public Character <>4__this; // 0x20', 'public string info; // 0x28']}
    for (name, index), declarations in fields.items():
        match = re.search(r'^[^\n]*class ' + re.escape(name) + r'(?: :[^\n]*)? // TypeDefIndex: '
                          + str(index) + r'\s*\{(.*?)^\}', m.dump, re.M | re.S)
        assert match and all(s in match[1] for s in declarations)
    checks = {0x376095: ('mov', 'dword ptr [rdi + 0x10], 0xffffffff'),
              0x3760EE: ('call', 'qword ptr [rax + 0x18]'), 0x37610C: ('call', '0x2eb0'),
              0x376117: ('dec', 'dword ptr [rsi + 0xdc]'), 0x37613F: ('call', 'qword ptr [rax + 0x18]'),
              0x2EB4: ('inc', 'dword ptr [rcx + 0x1c]'), 0x2EE6: ('mov', 'dword ptr [rcx + 0x18], eax'),
              0x2EF7: ('mov', 'qword ptr [rcx], rdx'), 0x2EFE: ('jmp', '0x2b6ff0'),
              0x37644D: ('mov', 'qword ptr [rcx], rdx'), 0x376450: ('call', '0x2b6ff0'),
              0x376442: ('mov', 'rdx, qword ptr [rdi + 0x28]'), 0x37643C: ('call', 'qword ptr [rax + 0x558]'),
              0x364CAE: ('test', 'al, al'), 0x364CB2: ('mov', 'rax, qword ptr [rbx + 0x58]')}
    for address, expected in checks.items():
        assert address in m.instructions and (m.instructions[address].mnemonic, m.instructions[address].op_str) == expected
    base['instruction_assertions'] += len(checks)
    gateway_rows = []
    for address, name in {0x38AD80: 'TrailerCharacters$$GetCharacterInfoOfId',
                          0x1C79FD0: 'UnityEngine.Component$$get_gameObject',
                          0x1C82250: 'UnityEngine.Object$$get_name',
                          0xF71C60: 'System.String$$Concat', 0x1C4B380: 'UnityEngine.Debug$$Log',
                          0x1C822C0: 'UnityEngine.Object$$op_Equality',
                          0x1C7D810: 'UnityEngine.GameObject$$SetActive',
                          0x35D920: 'ActedVersion$$Show',
                          0x1EC1010: 'UnityEngine.UI.LayoutRebuilder$$ForceRebuildLayoutImmediate'}.items():
        rows = [r for r in m.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
        assert len(rows) == 1
        gateway_rows += rows
    base['verified_gateways'] = gateway_rows
    base['speech_wait_bits'] = m.speech_wait_bits
    base['supplied_service_rvas'] = [hex(rva) for rva in sorted(m.service_rvas)]
    base['scope'] = 'Actual installed callback, both ShowActedDelayed resumes, String.IsNullOrEmpty leaf, List.Add fast path, ShowInfoDelayed explicit resumes, GetCharacterBluffIfAble and Acted.Act(string) execute. Metadata/class/GC/delegate/preappend/event, capacity growth, trailer lookup, Unity/string/log/text/Wait/ActedVersion/layout and coroutine adapters remain supplied. Resume order is authored, not scheduler readiness or time; native exception unwinding is unclaimed.'
    return base


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    verified = verify_publication(m)
    cases, baselines, failures = [], [], []
    def run(options):
        result = m.run(options)
        cases.append({k: v for k, v in result.items() if k != 'events'} | {'event_kinds': [e['kind'] for e in result['events']]})
        return result
    for trigger, uses, picking, state, about in itertools.product([3, 5, 30, 0xFFFFFFFF], [0, 1, 2, 0x80000000], [False, True], [10, 20], [False, True]):
        result = run({'trigger': trigger, 'uses': uses, 'picking': picking, 'state': state, 'about': about})
        assert result['returned']
        final = result['final']
        assert final['history'] == ['prior_info', 'info'] and final['history_version'] == 10
        assert final['uses_bits'] == (uses - int(trigger == 30)) & 0xFFFFFFFF
        expected = 'replacement' if about else 'original'
        assert final['saved_speech'] == final['text'] == expected and final['shown'] == [expected]
        assert [r['result'] for r in final['speech_resumes']] == ([0] if picking or state == 20 else [1, 0])
    for options in [dict(act=False), dict(empty_description=True), dict(null_description=True),
                    dict(null_info=True), dict(null_history=True), dict(null_history_array=True),
                    dict(null_pickable=True), dict(null_acteds=True), dict(null_version=True),
                    dict(null_blank=True), dict(null_layouts=True), dict(null_data=True),
                    dict(null_game_call=1), dict(null_game_call=2), dict(trailer=True),
                    dict(trailer=True, empty_trailer=True), dict(trailer=True, null_trailer_text=True),
                    dict(trailer=True, null_trailer_controller=True), dict(trailer=True, null_trailer=True),
                    dict(about=True, about_empty_desc=True), dict(about=True, about_null_desc=True),
                    dict(about=True, about_null_history=True), dict(event_mutation=True),
                    dict(capacity=1), dict(uses=-1), dict(bluff=True), dict(bluff=True, destroyed_bluff=True),
                    dict(bluff=True, revealed=True), dict(bluff=True, state=30), dict(no_event=True)]:
        result = run(options)
        if options.get('act') is False or options.get('empty_description') or options.get('null_description'):
            assert result['returned'] and result['final']['history'] == ['prior_info'] and not result['final']['speech_registered']
        if options.get('about_empty_desc') or options.get('about_null_desc'):
            assert result['returned'] and result['final']['history'] == ['prior_info', 'info']
            assert result['final']['saved_speech'] == ('empty' if options.get('about_empty_desc') else None)
        if options.get('event_mutation'):
            assert result['returned'] and result['final']['saved_speech'] == 'replacement'
            assert result['final']['history_records'][1]['description'] == 'replacement'
        if options.get('trailer') and not any(options.get(k) for k in ['null_trailer_controller', 'null_trailer']):
            assert result['returned'] and result['final']['history_records'][1]['description'] == 'original'
            assert result['final']['saved_speech'] == ('original' if options.get('empty_trailer') or options.get('null_trailer_text') else 'trailer')
        expected_error = any(options.get(k) for k in ['null_info', 'null_history', 'null_history_array', 'null_pickable',
                                                      'null_acteds', 'null_version', 'null_blank', 'null_layouts',
                                                      'null_data', 'null_trailer_controller', 'null_trailer', 'about_null_history']) or options.get('null_game_call') == 1
        assert result['returned'] != expected_error
    for options in [{'cold': True}, {'cold': True, 'callback_repeats': 2, 'about': True},
                    {'cold': True, 'trailer': True, 'bluff': True, 'capacity': 1}]:
        baseline = m.run(options)
        assert baseline['returned']
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run({**options, 'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    sequences = []
    for reverse in [False, True]:
        result = m.run({'callback_repeats': 2, 'about': True, 'reverse_results': reverse})
        assert result['returned'] and result['final']['history'] == ['prior_info', 'info', 'info']
        assert result['final']['history_version'] == 11 and result['final']['uses_bits'] == 0xFFFFFFFF
        assert len(result['final']['speech_registered']) == 2
        sequences.append(result)
    ordered_sequences = []
    for reverse in [False, True]:
        m.prepare({'callback_repeats': 0, 'uses': 2, 'reverse_results': reverse})
        assert m.invoke(0x368790, m.actor, m.role, 3, 0)
        old = m.rq(m.role + 0x28)
        assert m.invoke(0x368790, m.actor, m.role, 30, 1)
        new = m.rq(m.role + 0x28)
        assert m.invoke(0x377120, m.delegates[old], m.info)
        assert m.invoke(0x377120, m.delegates[new], m.fixtures['prior_info'])
        result = m.finish_publication(True)
        assert result['returned'] and result['final']['history'] == (['prior_info', 'prior_info', 'info'] if reverse else ['prior_info', 'info', 'prior_info'])
        assert result['final']['uses_bits'] == 1
        assert result['final']['saved_speech'] == ('original' if reverse else 'prior')
        assert m.event('postpublication_mutation_service', ['info'])
        m.q(m.info + 0x10, m.string_pointers['replacement'])
        mutated = m.snapshot()
        assert next(r for r in mutated['history_records'] if r['id'] == 'info')['description'] == 'replacement'
        assert mutated['saved_speech'] == result['final']['saved_speech']
        ordered_sequences.append({'publication': result, 'postpublication_mutation': mutated})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases),
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'alias_sequences': sequences, 'explicit_order_sequences': ordered_sequences,
            'executed_address_count': len(m.executed)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
