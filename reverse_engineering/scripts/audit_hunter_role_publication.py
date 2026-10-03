"""Pinned Hunter Day producers joined to retained callbacks and publication.

Finished setup, collection/runtime services and the inherited explicit coroutine
resume schedule are supplied. No live generation or engine chronology is claimed.
"""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_role_publication import Machine as PublicationMachine, verify_publication


TARGETS = {
    0x3645C0: 'Character$$Act', 0x397750: 'CharacterHelper$$CheckLying',
    0x3B09F0: 'Tracker$$Act', 0x3B33E0: 'Tracker$$BluffAct',
    0x3EF150: 'Tracker$$GetInfo', 0x3EE960: 'Tracker$$GetBluffInfo',
    0x3EEBB0: 'Tracker$$GetDistanceToEvil', 0x3EE8C0: 'Tracker$$ConjourInfo',
    0x36C590: 'Characters$$GetCharactersAtRange',
    0x398AF0: 'CharactersHelper$$GetSortedListWithCharacterFirst',
    0x365030: 'Character$$GetRegisterAlignment', 0x35D5D0: 'ActedInfo$$.ctor',
    0x3DD410: 'Imp$$Act',
}


class Machine(PublicationMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
        self.body_hashes = {}
        for start, name in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start and r['Name'] == name]
            assert len(rows) == 1
            self.targets += rows
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4:
                    root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start:
                    chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks
            self.ranges[hex(start)] = [[hex(a), hex(b)] for a, b in chunks]
            self.body_hashes[name] = hashlib.sha256(b''.join(self.pe.get_data(a, b-a) for a, b in chunks)).hexdigest()
            for a, b in chunks:
                ins = list(self.cs.disasm(self.pe.get_data(a, b-a), a))
                assert sum(i.size for i in ins) == b-a
                self.instructions.update({i.address: i for i in ins})
        # Folded Dispose/Object constructor is a native ret 0, preserving RAX.
        no_op = list(self.cs.disasm(self.pe.get_data(0x33ED50, 3), 0x33ED50))
        assert len(no_op) == 1 and (no_op[0].mnemonic, no_op[0].op_str) == ('ret', '0')
        self.instructions[0x33ED50] = no_op[0]
        references = set()
        for ins in self.instructions.values():
            for op in ins.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(ins.address + ins.size + op.mem.disp)
                    if ins.mnemonic == 'cmp' and op.size == 1:
                        self.flags.add(ins.address + ins.size + op.mem.disp)
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                if row['Name'] not in self.bindings:
                    self.bindings[row['Name']] = self.arena + 0x4000 + len(self.bindings)*0x200
                token = self.bindings[row['Name']]
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
        for row in self.metadata['ScriptString']:
            if row['Address'] in references and self.base + row['Address'] not in self.metadata_slots:
                token = self.arena + 0x80000 + len(self.strings)*0x200
                self.make_string(token, row['Value'], 'literal:' + row['Value'])
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
        names = ['board', 'board_static', 'statuses', 'baa_data', 'imp_role', 'imp_class',
                 'other0', 'other1', 'other2', 'characters_owner', 'characters_static',
                 'other_role0', 'other_role1', 'other_role2']
        self.native_fixtures = {name: self.arena + 0xD0000 + i*0x1000 for i, name in enumerate(names)}
        self.static_ids.update({p: name for name, p in self.native_fixtures.items()})
        self.hunter_services = {0x1C82480, 0x363C40, 0xB02160, 0xB610A0,
                                0xB16640, 0x9693D0, 0xB59E70, 0xB59CE0,
                                0xB5A600, 0xB22150, 0xB4A240, 0xB31810,
                                0x1C86600, 0x282580, 0xF74DF0, 0x1C4B450}
        self.fixture_strings, self.fixture_string_labels = self.strings.copy(), self.string_labels.copy()

    def values(self, pointer):
        n = self.rd(pointer + 0x18)
        assert n <= 16
        width = 4 if self.lists[pointer] == 'integers' else 8
        read = self.rd if width == 4 else self.rq
        return [read(self.rq(pointer + 0x10) + 0x20 + i*width) for i in range(n)]

    def event(self, kind, args):
        self.counts[kind] = self.counts.get(kind, 0)+1
        stopping = self.options.get('failure') == [kind, self.counts[kind]]
        event = {'kind': kind, 'args': args}
        # Every prefix's snapshot is independently checked at its own stop.
        # Rebuilding all preceding snapshots in each stop run is redundant.
        if not self.options.get('failure') or stopping:
            event['snapshot'] = self.snapshot()
        self.events.append(event)
        if stopping:
            self.error = kind
            self.u.emu_stop()
            return False
        return True

    def fill(self, pointer, values, kind=None):
        if kind is not None:
            self.lists[pointer] = kind
        width = 4 if self.lists[pointer] == 'integers' else 8
        write = self.d if width == 4 else self.q
        self.q(pointer + 0x10, pointer + 0x200)
        self.d(pointer + 0x18, len(values))
        self.d(pointer + 0x1C, self.rd(pointer + 0x1C) + 1)
        self.q(pointer + 0x218, 16)
        for i, value in enumerate(values):
            write(pointer + 0x220 + i*width, value)

    def snapshot(self):
        result = super().snapshot()
        result.update(board=[{'id': self.object_id(p), 'display_id': self.rd(p + 0x118),
                             'alignment': self.rd(p + 0xF8), 'data': self.object_id(self.rq(p + 0x50)),
                             'register_as': self.object_id(self.rq(p + 0x60)),
                             'real_role': self.object_id(self.rq(p+0x168)),
                             'bluff_role': self.object_id(self.rq(p+0x170)),
                             'display_bluff': self.object_id(self.rq(p+0x58)),
                             'character_start_acted': bool(self.u.mem_read(p+0x11C, 1)[0])} for p in self.board],
                      provider_calls=self.provider_calls.copy(), captured_callbacks=self.captured_callbacks.copy(),
                      rng=self.rng.copy(),
                      retained_role_delegates={self.object_id(p): self.object_id(self.rq(p+0x28)) for p in
                                              [self.role] + [self.native_fixtures[name] for name in
                                              ['imp_role', 'other_role0', 'other_role1', 'other_role2']]},
                      generated=[{'id': self.object_id(p), 'description': self.strings.get(self.rq(p + 0x10)),
                                  'reference_identity': self.object_id(self.rq(p + 0x18)),
                                  'ordered_reference_ids': [self.rd(q + 0x118) for q in self.values(self.rq(p + 0x18))]
                                  if self.rq(p + 0x18) in self.lists else []} for p in self.generated],
                      retained_lists={self.object_id(p): {'kind': kind, 'version': self.rd(p+0x1C),
                                      'values': self.values(p) if kind == 'integers' else
                                      [self.object_id(q) for q in self.values(p)]} for p, kind in self.lists.items()})
        return result

    def prepare(self, options):
        self.lists, self.generated, self.provider_calls, self.rng, self.boxes, self.board = {}, [], [], [], {}, []
        self.captured_callbacks = []
        self.strings, self.string_labels = self.fixture_strings.copy(), self.fixture_string_labels.copy()
        super().prepare({**options, 'bluff': options['actor_seat'] == options['baa_seat'],
                         'bluff_picking': False, 'uses': 0})
        f, n = self.fixtures, self.native_fixtures
        for name, pointer in self.bindings.items():
            if name.endswith('_TypeInfo'):
                self.d(pointer+0xE0, 1)
        for p in n.values():
            self.u.mem_write(p, bytes(0x800))
        others = iter([n['other0'], n['other1'], n['other2']])
        self.board = [self.actor if seat == options['actor_seat'] else next(others) for seat in range(4)]
        self.fill(n['board'], self.board, 'characters')
        self.q(self.bindings['Gameplay_TypeInfo'] + 0xB8, n['board_static'])
        self.d(self.bindings['Gameplay_TypeInfo'] + 0xE0, 1)
        self.q(n['board_static'] + 0x18, n['board'])
        self.q(self.bindings['Characters_TypeInfo']+0xB8, n['characters_static'])
        self.q(n['characters_static'], n['characters_owner'])
        self.q(n['characters_owner']+0x20, n['board'])
        for seat, p in enumerate(self.board):
            if p != self.actor:
                self.u.mem_write(p, bytes(0x200))
            self.d(p + 0x118, 4-seat)
            self.d(p + 0xF8, 20 if seat == options['baa_seat'] else 10)
            self.q(p + 0x50, n['baa_data'] if seat == options['baa_seat'] else f['data'])
            self.q(p + 0x60, 0)
            self.q(p + 0xF0, n['statuses'])
            role = self.role if p == self.actor else n['other_role'+str([n['other0'], n['other1'], n['other2']].index(p))]
            self.q(role, self.role_class)
            self.q(p+0x168, n['imp_role'] if seat == options['baa_seat'] else role)
            self.q(p+0x170, role if seat == options['baa_seat'] else 0)
            self.q(p+0x58, f['data'] if seat == options['baa_seat'] else 0)
        self.q(self.actor + 0x168, n['imp_role'] if self.options['bluff'] else self.role)
        self.q(self.actor + 0x170, self.role if self.options['bluff'] else 0)
        self.q(self.actor + 0x58, f['data'] if self.options['bluff'] else 0)
        self.q(n['imp_role'], n['imp_class'])
        for cls in [self.role_class, n['imp_class']]:
            self.q(cls + 0x208, self.base + (0x3DD410 if cls == n['imp_class'] else 0x3B09F0))
            self.q(cls + 0x210, cls + 0x500)
            self.q(cls + 0x258, self.base + 0x3B33E0)
            self.q(cls + 0x260, cls + 0x580)
        self.q(self.role_class + 0x1D8, self.base + 0x3EF150)
        self.q(self.role_class + 0x1E0, self.role_class + 0x600)
        self.q(self.role_class + 0x1E8, self.base + 0x3EE960)
        self.q(self.role_class + 0x1F0, self.role_class + 0x680)
        self.q(self.info + 0x10, 0)
        self.q(self.info + 0x18, 0)
        self.lists[f['history']] = 'infos'
        for name, method in self.bindings.items():
            if name.startswith('Method$System.Collections.Generic.List<') and '.Add()' in name:
                self.q(method + 0x20, f['generic_context'])
        # The current data pool has three occurrences of one Hunter asset; physical
        # actors remain distinct. This is a supplied finished setup, not generation.
        self.initial_other_bytes = {p: bytes(self.u.mem_read(p, 0x200)) for p in self.board if p != self.actor}

    def hook(self, uc, address, size, data):
        rva, x = address-self.base, self.x
        if rva in self.instructions:
            self.executed.add(rva)
            if rva not in [0x3EF150, 0x3EE960, 0x377120]:
                return  # All other decoded instructions execute without service effects.
        cx, dx, r8, r9 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]]
        if rva in self.hunter_services or rva in self.instructions:
            self.executed.add(rva)
        if rva in [0x3EF150, 0x3EE960]:
            assert cx == self.role and dx == self.actor
            assert r8 == self.role_class + (0x600 if rva == 0x3EF150 else 0x680)
            self.provider_calls.append('truth' if rva == 0x3EF150 else 'bluff')
            return
        if rva == 0x377120:
            delegate = self.rq(self.role+0x28)
            assert cx == self.delegates[delegate] and dx == self.info
            assert r8 == self.bindings['Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()']
            assert self.rq(cx+0x10) == self.actor and self.rd(cx+0x18) == 30
            self.captured_callbacks.append({'delegate': self.object_id(delegate), 'closure': self.object_id(cx),
                                            'info': self.object_id(dx), 'owner': 'actor', 'trigger': 30})
            return
        if rva == 0x4D5B60:
            assert self.objects[cx].startswith('delegate') and self.objects[dx].startswith('closure')
            assert r8 == self.bindings['Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()'] and r9 == 0
            if self.event('delegate_constructor_service', [self.object_id(cx), self.object_id(dx)]):
                self.delegates[cx] = dx
                self.q(cx+0x18, self.base+0x377120)
                self.q(cx+0x28, r8)
                self.q(cx+0x40, dx)
                self.ret()
        elif rva == 0x2B7D40 and cx in [self.bindings[name] for name in
                ['System.Collections.Generic.List<Character>_TypeInfo', 'System.Collections.Generic.List<int>_TypeInfo', 'ActedInfo_TypeInfo']]:
            kind = next(name for name, p in self.bindings.items() if p == cx)
            if self.event('producer_allocate_service', [kind]):
                if kind == 'ActedInfo_TypeInfo':
                    assert not self.generated
                    p = self.info
                    self.generated.append(p)
                else:
                    p = self.alloc(0x500)
                    self.objects[p] = ('integers' if '<int>' in kind else 'characters') + str(len(self.objects))
                    self.lists[p] = 'integers' if '<int>' in kind else 'characters'
                self.q(p, cx)
                self.ret(p)
        elif rva in [0xB02160, 0xB610A0]:
            assert cx in self.lists
            source = self.values(dx) if rva == 0xB610A0 else []
            if self.event('list_constructor_service', [self.object_id(cx), self.object_id(dx) if source else None,
                                                         [self.object_id(p) for p in source]]):
                self.fill(cx, source)
                self.d(cx+0x1C, 0)
                self.ret()
        elif rva == 0xB16640:
            assert dx in self.lists
            if self.event('enumerator_constructor_service', [self.object_id(dx)]):
                self.q(cx, dx); self.d(cx+8, 0); self.d(cx+0xC, self.rd(dx+0x1C)); self.q(cx+0x10, 0)
                self.ret(cx)
        elif rva == 0x9693D0:
            owner, index = self.rq(cx), self.rd(cx+8)
            assert owner in self.lists and self.rd(cx+0xC) == self.rd(owner+0x1C)
            values = self.values(owner)
            if self.event('enumerator_move_service', [self.object_id(owner), index]):
                present = index < len(values)
                self.d(cx+8, index+1 if present else len(values)+1)
                self.q(cx+0x10, values[index] if present else 0)
                self.ret(int(present))
        elif rva in [0xB59E70, 0xB59CE0, 0xB5A600, 0xB4A240]:
            values = self.values(cx)
            if self.event('list_mutation_service', [hex(rva), self.object_id(cx),
                    self.object_id(dx) if rva == 0xB59E70 else dx & 0xFFFFFFFF]):
                removed = False
                if rva == 0xB59E70 or rva == 0xB4A240:
                    if dx in values:
                        values.remove(dx); removed = True
                elif rva == 0xB59CE0:
                    assert dx < len(values)
                    values.pop(dx)
                else:
                    values.reverse()
                if rva not in [0xB59E70, 0xB4A240] or removed:
                    self.fill(cx, values)
                self.ret(int(removed))
        elif rva in [0xB22150, 0xB31810]:
            values = self.values(cx)
            assert dx < len(values)
            if self.event('list_index_service', [self.object_id(cx), dx]):
                self.ret(values[dx])
        elif rva == 0x1C86600:
            assert cx == r8 == 0 and dx in [1, 2]
            chosen = self.options.get('draw', 0)
            assert chosen < dx
            if self.event('rng_integer_service', [0, dx, chosen]):
                self.rng.append({'min': 0, 'max': dx, 'index': chosen})
                self.ret(chosen)
        elif rva == 0x282580:
            value = self.rd(dx)
            typename = next(name for name, p in self.bindings.items() if p == cx)
            assert typename in ['ETriggerPhase_TypeInfo', 'int_TypeInfo']
            if self.event('integer_box_service', [typename, value]):
                p = self.alloc(0x100); self.boxes[p] = (typename, value)
                self.objects[p] = 'box' + str(len(self.objects)); self.ret(p)
        elif rva == 0xF74DF0:
            assert cx in self.strings and dx in self.boxes and r8 == 0
            fmt, (typename, value) = self.strings[cx], self.boxes[dx]
            assert fmt.count('{0}') == 1
            if self.event('integer_format_service', [fmt, typename, value]):
                assert typename != 'ETriggerPhase_TypeInfo' or value == 30
                text = fmt.replace('{0}', 'Day' if typename == 'ETriggerPhase_TypeInfo' else str(value))
                p = self.alloc(0x200)
                self.make_string(p, text, text)
                self.ret(p)
        elif rva == 0x1C4B450:
            assert cx in self.strings and dx == 0
            if self.event('trigger_log_service', [self.strings[cx]]):
                self.ret()
        elif rva in [0x1C82480, 0x1C822C0]:
            assert r8 == 0
            assert cx == 0 or cx in self.static_ids or cx in self.board
            assert dx == 0 or dx in self.static_ids or dx in self.board
            if self.event('unity_object_service', [hex(rva), self.object_id(cx), self.object_id(dx)]):
                self.ret(int(cx != 0) if rva == 0x1C82480 else int(cx == dx))
        elif rva == 0x363C40:
            assert cx == self.native_fixtures['statuses'] and dx in [30, 10] and r8 == 0, (hex(cx), dx, r8)
            if self.event('status_contains_service', [dx]):
                self.ret(0)
        elif rva == 0x33ED50:
            pass  # Actual folded instruction; never the base synthetic gateway.
        else:
            assert rva in self.instructions or rva in self.service_rvas or address in {
                self.role_act, self.role_bluff, self.after_callback, self.after_wait,
                self.about_service, self.info_event_service, self.text_service}, hex(rva)
            super().hook(uc, address, size, data)

    def invoke(self, address, cx, dx=0, r8=0, r9=0):
        x, sp = self.x, self.stack+0x18008
        self.q(sp, self.stop); self.q(sp+0x28, 0)
        keep = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(keep):
            self.u.reg_write(register, 0xFAB00000+i)
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, cx), (x.UC_X86_REG_RDX, dx),
                                (x.UC_X86_REG_R8, r8), (x.UC_X86_REG_R9, r9)]:
            self.u.reg_write(register, value & 0xFFFFFFFFFFFFFFFF)
        for register in [x.UC_X86_REG_RAX, x.UC_X86_REG_R10, x.UC_X86_REG_R11]:
            self.u.reg_write(register, 0)
        saved_xmm6 = 0xFEDCBA98765432100123456789ABCDEF
        self.u.reg_write(x.UC_X86_REG_XMM6, saved_xmm6)
        before = bytearray(self.u.mem_read(self.actor, 0x200))
        self.u.emu_start(self.base+address, self.stop, timeout=10_000_000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp+8
            assert all(self.reg(r) == 0xFAB00000+i for i, r in enumerate(keep))
            assert self.reg(x.UC_X86_REG_XMM6) == saved_xmm6 and not self.frames and not self.native_calls
        after = bytearray(self.u.mem_read(self.actor, 0x200))
        for offset, width in [(0xDC, 4), (0x11C, 1), (0x148, 8), (0x198, 8)]:
            before[offset:offset+width] = after[offset:offset+width] = bytes(width)
        assert before == after
        assert all(bytes(self.u.mem_read(p, 0x200)) == raw for p, raw in self.initial_other_bytes.items())
        return returned

    def run(self, options):
        self.prepare(options)
        returned = self.invoke(0x3645C0, self.actor, 30)
        return self.finish_publication(returned)


def verify_native(m):
    result = verify_publication(m)
    asset_path = Path(__file__).parents[1] / 'reports' / f'{BUILD}_character_assets_audit.json'
    assets = json.loads(asset_path.read_text(encoding='utf-8'))
    assert assets['build_id'] == BUILD
    hunter = [r for r in assets['records'] if r['name'] == 'Hunter']
    assert len(hunter) == 1
    asset = {key: hunter[0][key] for key in ['name', 'path_id', 'object_sha256', 'role_type',
                                           'role_type_def_index', 'abilityUsage', 'picking']}
    assert asset == {'name': 'Hunter', 'path_id': 21621,
                     'object_sha256': 'E180A1B912A9F604F15F8206AF64705E4BAD7FF6857C753EB8C49AD89A613419',
                     'role_type': 'Tracker', 'role_type_def_index': 5891, 'abilityUsage': 0, 'picking': False}
    for name, index, declaration in [('ETriggerPhase', 5605, 'public const ETriggerPhase Day = 30;'),
                                    ('ECharacterStatus', 5491, 'public const ECharacterStatus Corrupted = 10;'),
                                    ('ECharacterStatus', 5491, 'public const ECharacterStatus HealthyBluff = 30;')]:
        match = re.search(r'^public enum ' + name + r' // TypeDefIndex: ' + str(index) + r'\s*\{(.*?)^\}',
                          m.dump, re.M|re.S)
        assert match and declaration in match[1]
    fields = {
        'Character': ['public CharacterData registerAs; // 0x60', 'public EAlignment alignment; // 0xF8',
                      'public Role role; // 0x168', 'public Role bluffRole; // 0x170',
                      'private bool characterStartActed; // 0x11C', 'public CharacterStatuses statuses; // 0xF0'],
        'Gameplay': ['public static List<Character> CurrentCharacters; // 0x18'],
    }
    for name, declarations in fields.items():
        match = re.search(r'^[^\n]*class ' + re.escape(name) + r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', m.dump, re.M|re.S)
        assert match and all(s in match[1] for s in declarations), name
    checks = {0x364727: ('mov', 'byte ptr [rbx + 0x11c], 1'),
              0x3B0A10: ('mov', 'rax, qword ptr [r8 + 0x1d8]'),
              0x3B0A17: ('mov', 'r8, qword ptr [r8 + 0x1e0]'),
              0x3B0A20: ('mov', 'r8, qword ptr [rbx + 0x28]'),
              0x3B0A27: ('mov', 'rcx, qword ptr [rbx + 0x40]'),
              0x3B0A2B: ('mov', 'rax, qword ptr [rbx + 0x18]'),
              0x3B3400: ('mov', 'rax, qword ptr [r8 + 0x1e8]'),
              0x3B3407: ('mov', 'r8, qword ptr [r8 + 0x1f0]'),
              0x365060: ('mov', 'rdi, qword ptr [rbx + 0x60]'),
              0x3B0A1E: ('call', 'rax'), 0x3B0A38: ('jmp', 'rax'),
              0x3B340E: ('call', 'rax'), 0x3B3428: ('jmp', 'rax'),
              0x3EF19B: ('call', '0x3eebb0'), 0x3EF1D0: ('call', '0x36c590'),
              0x3EEB14: ('call', '0x1c86600'), 0x3EEB4A: ('call', '0x36c590'),
              0x36C68E: ('call', '0x398af0'), 0x3EEFDB: ('call', '0x365030'),
              0x3EF0A4: ('call', '0x365030'), 0x33ED50: ('ret', '0')}
    for address, expected in checks.items():
        assert (m.instructions[address].mnemonic, m.instructions[address].op_str) == expected
    result.update(instruction_assertions=result['instruction_assertions']+len(checks),
                  hunter_body_hashes=m.body_hashes, hunter_fields=fields,
                  hunter_asset_evidence={'source': asset_path.relative_to(Path(__file__).parents[2]).as_posix(),
                                         **asset},
                  hunter_supplied_service_rvas=[hex(a) for a in sorted(m.hunter_services)])
    return result


def expected(options):
    actor, baa = options['actor_seat'], options['baa_seat']
    truth = 3 if actor == baa else min((actor-baa)%4, (baa-actor)%4)
    support = [d for d in [1, 2] if d != truth]
    distance = support[options.get('draw', 0)] if actor == baa else truth
    return {'distance': distance,
            'text': f'I am {distance} ' + ('card' if distance == 1 else 'cards') + ' away from closest Evil',
            'ordered_reference_ids': [4-(actor+distance)%4, 4-(actor-distance)%4]}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    verified = verify_native(m)
    cases, failures = [], []
    for baa, actor in itertools.product(range(4), repeat=2):
        for draw in (range(2) if actor == baa else range(1)):
            options = {'baa_seat': baa, 'actor_seat': actor, 'draw': draw}
            result = m.run(options)
            assert result['returned'], result['error']
            final, want = result['final'], expected(options)
            assert len(final['generated']) == 1
            got = final['generated'][0]
            assert got['description'] == want['text'] and got['ordered_reference_ids'] == want['ordered_reference_ids']
            assert final['history'] == ['prior_info', 'info'] and final['history_version'] == 10
            assert final['history_records'][0] == {'id': 'prior_info', 'description': 'prior', 'references': 'references'}
            assert final['history_records'][1] == {'id': 'info', 'description': want['text'],
                                                  'references': got['reference_identity']}
            assert final['info'] == {'description': want['text'], 'references': got['reference_identity']}
            assert len(final['captured_callbacks']) == 1 and final['captured_callbacks'][0]['info'] == 'info'
            assert all(final['retained_role_delegates'][name] is None for name in
                       ['other_role0', 'other_role1', 'other_role2'])
            assert (final['retained_role_delegates']['imp_role'] is not None) == (actor == baa)
            assert final['uses_bits'] == 0xFFFFFFFF and final['saved_speech'] == final['text'] == want['text']
            assert final['shown'] == [want['text']]
            assert final['provider_calls'] == ['bluff' if actor == baa else 'truth']
            assert len(final['rng']) == int(actor == baa)
            baseline_id = len(cases)
            cases.append({'expected': want, **result})
            counts = {}
            for index, event in enumerate(result['events']):
                kind = event['kind']; counts[kind] = counts.get(kind, 0)+1
                stopped = m.run({**options, 'failure': [kind, counts[kind]]})
                assert not stopped['returned']
                assert [{k: v for k, v in e.items() if k != 'snapshot'} for e in stopped['events']] == [
                    {k: v for k, v in e.items() if k != 'snapshot'} for e in result['events'][:index+1]]
                assert stopped['events'][-1] == event
                assert stopped['final'] == event['snapshot']
                failures.append({'baseline': baseline_id, 'failure': [kind, counts[kind]],
                                 'prefix_length': index+1, 'exact_snapshot_verified': True})
    return {'schema_version': 1, 'build_id': BUILD, **verified,
            'scope': 'Supplied N4 finished setup, three repeated Hunter current-data occurrences and one Baa; all physical actor/Baa seats and all Baa Hunter bluff draws. Actual Character.Act(Day), RoleAct, Imp Day no-op, Tracker direct delegate real/bluff producers, distance/register alignment/range/text selection, ActedInfo ctor, callback/result/history/speech consumers execute. CLR collections, allocation, boxing/String.Format, delegate, metadata/class/status/Unity/UI and inherited explicit result/speech resume schedule are supplied. Native ordered references are separate from speech capture. Generation, actual engine/coroutine chronology, capture admission and mixed-role bodies are unclaimed.',
            'display_id_by_native_seat': [4, 3, 2, 1], 'case_count': len(cases),
            'supplied_root_entry_abi': {'trigger': 30, 'rcx': 'actor', 'rdx': 30, 'r8': 0, 'r9': 0,
                                        'rax': 0, 'r10': 0, 'r11': 0, 'stack_method_info': 0,
                                        'initial_history_version': 9, 'initial_history_count': 1,
                                        'initial_uses': 0},
            'cases': cases, 'failure_case_count': len(failures), 'failure_cases': failures,
            'executed_address_count': len(m.executed)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
