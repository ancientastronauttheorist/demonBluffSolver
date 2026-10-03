"""Conditional original Hunter acquisition and publication on one native queue.

Actual Init/DelayReveal/Reveal/click/producer/consumer bodies retain the actor.
CLR, clone, owner lookup, native record creation and presentation are providers.
"""
import argparse
import hashlib
import inspect
import json
import re
import struct
from collections import Counter
from pathlib import Path

from audit_character_assets import BUILD
from audit_hunter_role_publication import expected, verify_native as verify_hunter
from audit_hunter_scheduled_publication import ScheduledHunter, ScheduledEngine, NativePrefixStopped
from audit_report_snapshots import pool_snapshots, expand_snapshots
from audit_unityplayer_coroutines import audit as verify_bridge
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


TARGETS = {
    0x365A20: 'Character$$Init', 0x367970: 'Character$$RefreshCharacter',
    0x3756B0: 'Character.<DelayReveal>d__84$$MoveNext', 0x368410: 'Character$$Reveal',
    0x365160: 'Character$$GiveBluff', 0x3682A0: 'Character$$RevealReal',
    0x3694D0: 'Character$$UpdateViewReal', 0x3695A0: 'Character$$UpdateView',
    0x3688B0: 'Character$$SetupArt', 0x3B4AB0: 'CharacterData$$GetArt',
    0x3B4A20: 'CharacterData$$GetArtType',
}


class AcquiredHunter(ScheduledHunter):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
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
        defaults = list(self.cs.disasm(self.pe.get_data(0x3712B0, 3), 0x3712B0))
        assert [(i.mnemonic, i.op_str) for i in defaults] == [('xor', 'eax, eax'), ('ret', '')]
        self.instructions.update({i.address: i for i in defaults})
        self.body_hashes['Role.GetRegisterAsRole/GetBluffIfAble folded null'] = hashlib.sha256(self.pe.get_data(0x3712B0, 3)).hexdigest()
        for name in ['Role$$GetRegisterAsRole', 'Role$$GetBluffIfAble']:
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == name and r['Address'] == 0x3712B0]
            assert len(rows) == 1
            self.targets += rows
        refs = set()
        for i in self.instructions.values():
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    refs.add(i.address+i.size+op.mem.disp)
                    if i.mnemonic == 'cmp' and op.size == 1:
                        self.flags.add(i.address+i.size+op.mem.disp)
        for row in self.metadata['ScriptMetadata']+self.metadata['ScriptMetadataMethod']:
            if row['Address'] in refs and self.base+row['Address'] not in self.metadata_slots:
                self.bindings.setdefault(row['Name'], self.arena+0x4000+len(self.bindings)*0x200)
                self.metadata_slots[self.base+row['Address']] = self.bindings[row['Name']]
                self.q(self.base+row['Address'], self.bindings[row['Name']])
        for row in self.metadata['ScriptString']:
            if row['Address'] in refs and self.base+row['Address'] not in self.metadata_slots:
                p = self.arena+0x90000+len(self.strings)*0x200
                self.make_string(p, row['Value'], 'literal:'+row['Value'])
                self.metadata_slots[self.base+row['Address']] = p
                self.q(self.base+row['Address'], p)
        self.fixture_strings, self.fixture_string_labels = self.strings.copy(), self.string_labels.copy()
        self.acq = {name: self.arena+0x150000+i*0x1000 for i, name in enumerate(
            ['source_role', 'source_class', 'number', 'view_class', 'view', 'empty_views',
             'empty_picked', 'active_statuses', 'resistances', 'status_backing', 'resistance_backing',
             'state_delegate', 'view_backside', 'view_art', 'other_status0', 'other_status1', 'other_status2'])}
        self.static_ids.update({p: 'acquisition_'+name for name, p in self.acq.items()})
        self.number_setter, self.view_color, self.state_callback = [self.stop+n for n in [0xF00, 0xF10, 0xF20]]
        self.u.mem_write(self.number_setter, b'\xc3'*0x30)
        load = self.instructions[0x375742]
        slot = load.address+load.size+load.operands[1].mem.disp
        self.acquisition_wait_bits = struct.unpack('<I', self.pe.get_data(slot, 4))[0]
        assert self.acquisition_wait_bits == 0x3E99999A

    def prepare(self, options):
        super().prepare(dict(options, route='click'))
        assert options['actor_seat'] != options['baa_seat']
        a, f, n = self.acq, self.fixtures, self.native_fixtures
        for p in a.values():
            self.u.mem_write(p, bytes(0x800))
        self.acquisition_iterators, self.phase_calls = [], []
        self.acquisition_registered = []
        self.q(f['data']+0x28, self.string_pointers['object_name'])
        self.d(f['data']+0x134, 10)
        self.d(f['data']+0x138, 0)
        self.q(f['data']+0x140, a['source_role'])
        self.q(a['source_role'], a['source_class'])
        for cls in [a['source_class'], self.role_class]:
            for offset in [0x278, 0x288]:
                self.q(cls+offset, self.base+0x3712B0)
                self.q(cls+offset+8, cls+offset+0x400)
        # Before acquisition the runtime clone is not installed on the actor.
        self.q(self.actor+0x168, 0)
        for index, actor in enumerate(p for p in self.board if p != self.actor):
            status = a['other_status'+str(index)]
            self.q(actor+0xF0, status)
            self.q(status+0x10, status+0x100); self.q(status+0x18, status+0x200)
            self.q(status+0x20, actor)
        for offset, value in [(0x40, a['view']), (0x48, a['number']), (0x28, a['view_backside']),
                              (0x30, a['view_art']), (0x120, a['view']), (0x128, a['empty_views']),
                              (0x188, a['empty_picked']), (0x180, a['state_delegate'])]:
            self.q(self.actor+offset, value)
        self.q(a['number'], a['view_class']); self.q(a['view'], a['view_class'])
        self.q(a['view_class']+0x558, self.number_setter)
        self.q(a['view_class']+0x560, a['view_class']+0x600)
        self.q(a['view_class']+0x2A8, self.view_color)
        self.q(a['view_class']+0x2B0, a['view_class']+0x680)
        self.q(a['state_delegate']+0x18, self.state_callback)
        self.q(a['state_delegate']+0x40, self.actor)
        self.q(a['state_delegate']+0x28, a['state_delegate']+0x500)
        self.q(n['statuses']+0x10, a['active_statuses'])
        self.q(n['statuses']+0x18, a['resistances'])
        self.q(n['statuses']+0x20, self.actor)
        self.q(a['active_statuses']+0x10, a['status_backing'])
        self.d(a['active_statuses']+0x18, 3); self.d(a['active_statuses']+0x1C, 23)
        self.d(a['status_backing']+0x18, 3)
        for i, status in enumerate([10, 30, 50]): self.d(a['status_backing']+0x20+4*i, status)
        self.q(a['resistances']+0x10, a['resistance_backing'])
        self.d(a['resistances']+0x18, 1); self.d(a['resistances']+0x1C, 7)
        self.d(a['resistance_backing']+0x18, 1); self.d(a['resistance_backing']+0x20, 50)
        self.q(self.actor+0x68, self.string_pointers['old'])
        self.q(self.actor+0x70, self.fixtures['prior_info'])
        self.q(self.actor+0x60, f['data'])
        self.u.mem_write(self.actor+0xD8, b'\1'); self.u.mem_write(self.actor+0x11C, b'\1')
        self.u.mem_write(self.actor+0xED, b'\1'); self.d(self.actor+0xDC, 7)
        self.d(self.actor+0xE0, 10); self.d(self.actor+0xE4, 20)
        self.initial_other_bytes = {p: bytes(self.u.mem_read(p, 0x200)) for p in self.board if p != self.actor}
        self.initial_other_roles = {p: bytes(self.u.mem_read(p, 0x100)) for p in
            [n['imp_role'], n['other_role0'], n['other_role1'], n['other_role2']]}
        self.initial_other_statuses = {a['other_status'+str(i)]: bytes(self.u.mem_read(a['other_status'+str(i)], 0x800)) for i in range(3)}
        self.initial_resistance = bytes(self.u.mem_read(a['resistances'], 0x30))+bytes(self.u.mem_read(a['resistance_backing'], 0x30))

    def snapshot(self):
        s = super().snapshot()
        a = self.acq
        s['acquisition'] = {
            'actor': self.object_id(self.actor), 'current_data': self.object_id(self.rq(self.actor+0x50)),
            'data_source_role': self.object_id(self.rq(self.fixtures['data']+0x140)),
            'runtime_role': self.object_id(self.rq(self.actor+0x168)),
            'register_as': self.object_id(self.rq(self.actor+0x60)),
            'raw_bluff': self.object_id(self.rq(self.actor+0x58)),
            'trailer': self.object_id(self.rq(self.actor+0x68)), 'runtime': self.object_id(self.rq(self.actor+0x70)),
            'revealed_byte': self.u.mem_read(self.actor+0xD8, 1)[0],
            'start_acted_byte': self.u.mem_read(self.actor+0x11C, 1)[0],
            'killed_while_hidden_byte': self.u.mem_read(self.actor+0xEC, 1)[0],
            'killed_by_demon_byte': self.u.mem_read(self.actor+0xED, 1)[0],
            'active_status_count': self.rd(a['active_statuses']+0x18),
            'active_status_version': self.rd(a['active_statuses']+0x1C),
            'active_status_backing': [self.rd(a['status_backing']+0x20+4*i) for i in range(3)],
            'resistance_count': self.rd(a['resistances']+0x18),
            'resistance_version': self.rd(a['resistances']+0x1C),
            'resistance_values': [self.rd(a['resistance_backing']+0x20)],
            'status_target': self.object_id(self.rq(self.native_fixtures['statuses']+0x20)),
            'iterators': [self.speech_iterator(p) for p in self.acquisition_iterators],
            'phase_calls': self.phase_calls.copy(),
            'other_status_components': [self.object_id(self.rq(p+0xF0)) for p in self.board if p != self.actor],
        }
        return s

    def hook(self, uc, address, size, data):
        x, rva = self.x, address-self.base
        cx, dx, r8, r9 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]]
        caller = self.rq(self.reg(x.UC_X86_REG_RSP))-self.base
        if rva in self.instructions:
            self.executed.add(rva)
        if rva == 0x3645C0:
            self.phase_calls.append({'trigger': dx&0xFFFFFFFF, 'caller': hex(caller), 'actor_state': self.rd(cx+0xE4)})
        if rva == 0x1C7F160 and self.objects.get(dx, '').startswith('acquisition'):
            assert cx == self.actor and r8 == 0
            item = self.speech_iterator(dx)
            assert item['state'] == 0 and item['owner'] == 'actor' and item['current'] is None
            if self.event('synchronous_start_gateway', ['acquisition', item]):
                self.acquisition_registered.append(item); self.pending_start = (dx, 'acquisition'); uc.emu_stop()
            return
        if rva == 0x4060 and r8 in self.acquisition_iterators:
            assert cx == 0 and dx == self.arena+0x121000
            if self.event('ienumerator_slot_zero_gateway', [self.object_id(r8), '0x3756b0']):
                uc.reg_write(x.UC_X86_REG_RCX, r8)
                for reg in [x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]: uc.reg_write(reg, 0)
                uc.reg_write(x.UC_X86_REG_RIP, self.base+0x3756B0)
            return
        if rva == 0x2B7D40 and cx == self.bindings['Character.<DelayReveal>d__84_TypeInfo']:
            if self.event('acquisition_allocate_gateway', [self.service_argument(cx), hex(caller)]):
                p = self.alloc(0x100); self.q(p, cx); self.objects[p] = 'acquisition'+str(len(self.acquisition_iterators))
                self.acquisition_iterators.append(p); self.ret(p)
            return
        if rva == 0x603240:
            assert cx == self.acq['source_role'] and dx == self.bindings['Method$ClassConv.CreateCopyNonGeneric<Role>()']
            assert r8 == 0
            if self.event('clone_role_gateway', [self.object_id(cx), self.object_id(self.role), hex(caller)]): self.ret(self.role)
            return
        if rva == 0xB45070:
            assert caller == 0x3685C5 and cx == self.acq['active_statuses'] and dx == 30
            assert self.rd(cx+0x18) == 0
            assert r8 == self.bindings['Method$System.Collections.Generic.List<ECharacterStatus>.Contains()']
            if self.event('empty_active_status_contains_gateway', [self.object_id(cx), dx, self.service_argument(r8), hex(caller)]):
                self.ret(0)
            return
        if rva == 0x1C961F0 and self.reg(x.UC_X86_REG_XMM1)&0xFFFFFFFF == self.acquisition_wait_bits:
            assert self.objects[cx].startswith('wait') and r8 == 0
            if self.event('wait_constructor_service', [self.object_id(cx), self.acquisition_wait_bits]):
                self.d(cx+0x10, self.acquisition_wait_bits); self.ret()
            return
        if rva in [0x112B9D0, 0x367B60, 0xF7B1B0, 0x1D49700] or address in [self.number_setter, self.view_color, self.state_callback]:
            args = [hex(rva), hex(caller)]+[self.service_argument(v) for v in [cx, dx, r8, r9]]
            kind = 'acquisition_presentation_gateway'
            if address == self.state_callback:
                assert cx == self.actor and dx == self.acq['state_delegate']+0x500
                kind = 'init_state_callback_gateway'
            elif rva == 0x112B9D0:
                assert dx == 0 and cx == self.fixtures['history_array'] and r8 == 1
                kind = 'init_history_array_clear_gateway'
            elif address == self.number_setter:
                assert cx in [self.acq['number'], self.acq['view']] and r8 == self.acq['view_class']+0x600
            elif address == self.view_color:
                assert cx == self.acq['view'] and r8 == self.acq['view_class']+0x680
                args += [bytes(self.u.mem_read(dx, 16)).hex()]
            if self.event(kind, args):
                if rva == 0x112B9D0: self.u.mem_write(cx+0x20, bytes(r8*8))
                self.ret(cx if rva == 0xF7B1B0 else 0)
            return
        if rva == 0xF71C60 and self.strings.get(cx) == 'INIT: ':
            assert dx == self.string_pointers['object_name']
            if self.event('init_log_concat_gateway', [self.strings[cx], self.strings[dx]]): self.ret(dx)
            return
        if rva == 0xF74DF0 and cx in self.strings and dx in self.boxes:
            fmt, (typename, value) = self.strings[cx], self.boxes[dx]
            if typename == 'ETriggerPhase_TypeInfo' and value in [3, 7]:
                if self.event('integer_format_service', [fmt, typename, value]):
                    p = self.alloc(0x200); text = fmt.replace('{0}', {3: 'Init', 7: 'AfterRoundStart'}[value])
                    self.make_string(p, text, text); self.ret(p)
                return
        if rva == 0x1C4B450:
            assert cx in self.strings and dx == 0
            if self.event('trigger_log_service', [self.strings[cx]]): self.ret()
            return
        if rva == 0x1C79FD0 and cx in [self.actor, self.fixtures['acted'], self.acq['view_backside'], self.acq['view_art']]:
            assert dx == 0
            if self.event('acquisition_game_object_gateway', [self.object_id(cx), hex(caller)]): self.ret(self.fixtures['game'])
            return
        super().hook(uc, address, size, data)

    def invoke(self, address, cx, dx=0, r8=0, r9=0):
        x = self.x
        sp = self.stack+0x18008-self.depth*0x4000
        assert sp >= self.stack+0x4008
        self.depth += 1
        self.q(sp, self.stop); self.q(sp+0x28, 0)
        keep = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(keep): self.u.reg_write(register, 0xFAB00000+i)
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, cx), (x.UC_X86_REG_RDX, dx),
                                (x.UC_X86_REG_R8, r8), (x.UC_X86_REG_R9, r9)]:
            self.u.reg_write(register, value&0xFFFFFFFFFFFFFFFF)
        for register in [x.UC_X86_REG_RAX, x.UC_X86_REG_R10, x.UC_X86_REG_R11]+[getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6)]:
            self.u.reg_write(register, 0)
        xmm6 = 0xFEDCBA98765432100123456789ABCDEF
        self.u.reg_write(x.UC_X86_REG_XMM6, xmm6)
        before = bytearray(self.u.mem_read(self.actor, 0x200))
        pc = self.base+address
        try:
            for _ in range(8):
                self.pending_start = None
                self.u.emu_start(pc, self.stop, timeout=10_000_000, count=100000)
                if self.pending_start is None: break
                iterator, kind = self.pending_start; self.pending_start = None
                context = self.u.context_save()
                handle = self.engine_join.start(iterator, kind)
                self.u.context_restore(context); self.ret(handle); pc = self.reg(x.UC_X86_REG_RIP)
            else: raise AssertionError('bounded synchronous start recursion exceeded')
            returned = self.reg(x.UC_X86_REG_RIP) == self.stop
            assert returned or self.error
            if returned:
                assert self.reg(x.UC_X86_REG_RSP) == sp+8
                assert all(self.reg(register) == 0xFAB00000+i for i, register in enumerate(keep))
                assert self.reg(x.UC_X86_REG_XMM6) == xmm6
            after = bytearray(self.u.mem_read(self.actor, 0x200))
            # Only fields with actual writes in Init, DelayReveal, click/publication.
            for offset, width in [(0x50, 8), (0x58, 8), (0x60, 8), (0x68, 8), (0x70, 8), (0x98, 8),
                                  (0xD8, 1), (0xDC, 4), (0xE0, 4), (0xE4, 4), (0xED, 1), (0xF8, 4),
                                  (0x118, 4), (0x11C, 1), (0x148, 8), (0x160, 4), (0x168, 8), (0x198, 8)]:
                before[offset:offset+width] = after[offset:offset+width] = bytes(width)
            assert before == after
            assert all(bytes(self.u.mem_read(p, 0x200)) == raw for p, raw in self.initial_other_bytes.items())
            assert all(bytes(self.u.mem_read(p, 0x100)) == raw for p, raw in self.initial_other_roles.items())
            assert all(bytes(self.u.mem_read(p, 0x800)) == raw for p, raw in self.initial_other_statuses.items())
            assert bytes(self.u.mem_read(self.acq['resistances'], 0x30))+bytes(self.u.mem_read(self.acq['resistance_backing'], 0x30)) == self.initial_resistance
            return returned
        finally:
            self.depth -= 1


def audit(game_root, dumper_root):
    root = Path(__file__).parents[2]
    old = root/'reverse_engineering/reports'/f'{BUILD}_hunter_scheduled_publication.json'
    old_hash = hashlib.sha256(old.read_bytes()).hexdigest()
    m = AcquiredHunter(game_root, dumper_root)
    verified = verify_hunter(m)
    bridge = verify_bridge(game_root)
    data = (game_root/'UnityPlayer.dll').read_bytes()
    verify_fingerprint(data, ENGINE_SHA256)
    e = ScheduledEngine(data, m)
    init_report = root/'reverse_engineering/reports'/f'{BUILD}_character_init.json'
    init_raw = init_report.read_bytes()
    init_report_hash = hashlib.sha256(init_raw).hexdigest()
    fields = json.loads(init_raw.decode('utf-8'))['fields']
    for name, declarations in fields.items():
        block = re.search(r'^[^\n]*class '+re.escape(name)+r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', m.dump, re.M|re.S)
        assert block and all(d in block[1] for d in declarations), name
    duration = struct.unpack('<f', struct.pack('<I', m.acquisition_wait_bits))[0]
    speech_duration = struct.unpack('<f', struct.pack('<I', m.speech_wait_bits))[0]
    assert duration == 0.30000001192092896
    pins = {0x375780: ('mov', 'dword ptr [rdi + 0x10], 0xffffffff'),
            0x375791: ('call', '0x368410'), 0x368470: ('call', 'rax'),
            0x368534: ('call', 'rax'), 0x3685E2: ('call', '0x3645c0'),
            0x3685F1: ('call', '0x3645c0'), 0x375769: ('mov', 'dword ptr [rdi + 0x10], 1'),
            0x37572E: ('mov', 'qword ptr [rcx], rax'),
            0x365B19: ('mov', 'byte ptr [rdi + 0x11c], r15b'),
            0x365C20: ('mov', 'dword ptr [rdi + 0xdc], 1'),
            0x365CA8: ('mov', 'dword ptr [rdi + 0xe4], 5'),
            0x365CDF: ('inc', 'dword ptr [rax + 0x1c]'),
            0x365CE2: ('mov', 'dword ptr [rax + 0x18], r15d'),
            0x3685BB: ('mov', 'edx, 0x1e'), 0x3685C0: ('call', '0xb45070'),
            0x3B09F7: ('cmp', 'edx, 0x1e'), 0x3B09FA: ('jne', '0x3b0a40')}
    for address, wanted in pins.items():
        assert (m.instructions[address].mnemonic, m.instructions[address].op_str) == wanted
    managed_seen, engine_seen = set(), set()

    def run(frame=7, generation=0, failure=None, owner='valid'):
        options = {'actor_seat': 1, 'baa_seat': 0, 'draw': 0, 'route': 'click'}
        if failure: options['failure'] = failure
        m.prepare(options); e.reset_join(generation); e.set_producer(1.0, frame)
        c = {'input': dict(options, producer_frame=frame, generation=generation, acquisition_owner=owner),
             'before_init': m.snapshot(), 'drains': [], 'click_invoked': False}
        def drain(time, f, phase, callbacks, remaining):
            d = e.drain_join(time, f, phase)
            got = [v['id'] for v in d['events'] if v['kind'] == 'native_wait_callback']
            assert got == callbacks, (d['input'], got, callbacks)
            assert [v['id'] for v in d['after']['entries']] == remaining
            d['expected_callback_ids'], d['expected_remaining_ids'] = callbacks, remaining
            c['drains'].append(d)
        try:
            if not m.invoke(0x365A20, m.actor, m.fixtures['data'], 3): raise NativePrefixStopped(m.error)
            c['after_init_first_yield'] = m.snapshot()
            assert m.snapshot()['history'] == [] and m.snapshot()['history_version'] == 10
            assert m.snapshot()['uses_bits'] == 1 and m.snapshot()['actor_state'] == 5
            assert m.snapshot()['actor_previous_state'] == 20
            assert m.snapshot()['acquisition']['start_acted_byte'] == 0
            assert m.snapshot()['saved_speech'] == 'old' and m.snapshot()['saved_text'] == 'old speech'
            assert m.snapshot()['reveal_card_init_reveal_byte'] == m.snapshot()['reveal_card_state_raw'] == m.snapshot()['gameplay_current_reveal'] == 0
            assert len(m.acquisition_iterators) == 1 and m.rq(m.actor+0x168) == m.role
            assert e.queue_state()['entries'] == [{'id': 0, 'kind': 'acquisition', 'iterator': 'acquisition0',
                'deadline': 1.0+duration, 'frame_threshold': frame+1, 'generation': generation, 'phase_mask': 10}]
            c['acquisition_initial_queue'] = e.queue_state()
            deadline = 1.0+duration
            drain(1.3, frame+1, 2, [], [0])
            drain(deadline, frame+1, 0, [], [0])
            drain(deadline, frame, 2, [], [0])
            if owner != 'valid': e.owner_outcomes[0] = owner
            drain(deadline, frame+1, 2, [0] if owner in ['valid', 'mismatch'] else [], [])
            c['after_acquisition_drain'] = m.snapshot()
            assert all(not r['owner_linked'] and r['reference_count'] == 0 and r['gc_handle'] == 0 for r in e.retained_native_records())
            c['after_acquisition_native_records'] = e.retained_native_records()
            if owner == 'valid':
                assert m.rd(m.acquisition_iterators[0]+0x10) == 0xFFFFFFFF
                assert m.snapshot()['uses_bits'] == 1 and m.snapshot()['history'] == []
                assert [v['trigger'] for v in m.phase_calls] == [3, 7]
                assert not m.generated and not m.captured_callbacks
                assert m.rq(m.actor+0x168) == m.role and m.rq(m.actor+0x50) == m.fixtures['data']
                assert m.snapshot()['acquisition']['start_acted_byte'] == 0
                assert m.snapshot()['actor_state'] == 5 and m.snapshot()['actor_previous_state'] == 20
                assert m.snapshot()['reveal_card_init_reveal_byte'] == m.snapshot()['reveal_card_state_raw'] == m.snapshot()['gameplay_current_reveal'] == 0
                assert m.snapshot()['acquisition']['register_as'] is None and m.snapshot()['acquisition']['raw_bluff'] is None
                c['click_admission'] = {'completed_acquisition_iterator': 'acquisition0',
                    'queue_empty': True, 'native_record_released': True, 'same_actor': 'actor', 'same_runtime_clone': 'role'}
                e.set_producer(deadline, frame+1)
                c['click_invoked'] = True
                if not m.invoke(0x366270, m.actor): raise NativePrefixStopped(m.error)
                c['after_click'] = m.snapshot()
                assert m.snapshot()['uses_bits'] == 1 and m.snapshot()['actor_state'] == 10
                assert m.snapshot()['history'] == [] and m.snapshot()['history_version'] == 10
                assert m.snapshot()['actor_previous_state'] == 5
                assert m.snapshot()['reveal_card_init_reveal_byte'] == 1 and m.snapshot()['reveal_card_state_raw'] == 20 and m.snapshot()['gameplay_current_reveal'] == 1
                assert [v['trigger'] for v in m.phase_calls] == [3, 7, 30]
                assert [v['actor_state'] for v in m.phase_calls] == [5, 5, 5]
                drain(deadline, frame+1, 2, [], [1])
                drain(deadline, frame+2, 2, [1], [2])
                assert m.snapshot()['uses_bits'] == 0 and m.snapshot()['history'] == ['info0']
                assert m.snapshot()['history_version'] == 11
                speech = e.queue_state()['entries'][0]
                assert speech['deadline'] == deadline+speech_duration and speech['frame_threshold'] == frame+3
                drain(deadline+speech_duration, frame+2, 2, [], [2])
                drain(deadline+speech_duration, frame+3, 2, [2], [])
                want = expected(options)
                assert m.snapshot()['saved_text'] == want['text'] and m.snapshot()['shown'] == [want['text']]
                assert m.snapshot()['generated'][0]['description'] == want['text']
                assert m.snapshot()['generated'][0]['ordered_reference_ids'] == want['ordered_reference_ids']
                assert [v['id'] for v in e.join_trace if v['kind'] == 'native_wait_callback'] == [0, 1, 2]
                assert m.snapshot()['acquisition']['start_acted_byte'] == 0
                assert m.rq(m.actor+0x168) == m.role and m.rq(m.actor+0x50) == m.fixtures['data']
                # Supplied free retains storage bytes: native release clears the
                # handle/refcount and unlinks, not the cached raw pointer.
                assert all(not r['owner_linked'] and r['reference_count'] == 0 and r['gc_handle'] == 0 and r['cached_enumerator_present'] for r in e.retained_native_records())
                c['expected'] = want
            else:
                assert not c['click_invoked'] and not m.phase_calls and not m.generated
                assert m.rd(m.acquisition_iterators[0]+0x10) == 1
            c['completed'] = True
        except NativePrefixStopped:
            assert m.error and failure
            c['completed'] = False
        c.update(error=m.error, final=m.snapshot(), managed_events=m.events.copy(),
                 engine_events=e.join_trace.copy(), final_queue=e.queue_state(),
                 final_native_records=e.retained_native_records())
        managed_seen.update(m.executed); engine_seen.update(e.executed)
        return c

    cases = [run(frame, generation) for frame in [7, -2, 2**32+7] for generation in [0, 0xFFFFFFFF]]
    owners = [run(owner=outcome) for outcome in ['missing', 'null', 'mismatch']]
    baseline, failures, counts = cases[0], [], Counter()
    for index, event in enumerate(baseline['managed_events']):
        counts[event['kind']] += 1
        stopped = run(failure=[event['kind'], counts[event['kind']]])
        assert not stopped['completed']
        assert [(v['kind'], v['args']) for v in stopped['managed_events']] == [(v['kind'], v['args']) for v in baseline['managed_events'][:index+1]]
        assert stopped['managed_events'][-1] == event
        assert stopped['final'] == event['snapshot']
        assert stopped['engine_events'] == baseline['engine_events'][:len(stopped['engine_events'])]
        failures.append(stopped)
    sources = {Path(inspect.getfile(cls)) for cls in AcquiredHunter.__mro__[:-1]+ScheduledEngine.__mro__[:-1]}
    sources.update([Path(inspect.getfile(verify_bridge)), Path(inspect.getfile(pool_snapshots))])
    hashes = {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(sources)}
    assert hashlib.sha256(old.read_bytes()).hexdigest() == old_hash
    assert hashlib.sha256(init_report.read_bytes()).hexdigest() == init_report_hash
    all_cases = cases+owners
    event_counts = Counter(v['kind'] for c in all_cases for v in c['engine_events'])
    metadata = {'composition_count': len(all_cases), 'publication_completed_count': len(cases),
                'owner_rejected_count': len(owners), 'managed_prefix_stop_count': len(failures),
                'drain_count': sum(len(c['drains']) for c in all_cases),
                'native_wait_inserted_count': event_counts['native_wait_inserted'],
                'native_wait_callback_count': event_counts['native_wait_callback'],
                'managed_native_address_count': len(managed_seen), 'engine_native_address_count': len(engine_seen),
                'acquisition_instruction_assertion_count': len(pins), 'managed_body_hash_count': len(m.body_hashes),
                'source_hash_count': len(hashes), 'engine_event_counts': dict(sorted(event_counts.items()))}
    return {'schema': 'hunter_acquisition_publication_v1', 'build': BUILD, 'metadata_counts': metadata,
            'domain': {'board': 'conditional N4: original Hunter at seat1, Baa at seat0, other actors supplied acquired',
                       'display_ids_in_board_order': [4, 3, 2, 1], 'uses': 'actual Init1 -> acquisition1 -> result resume0',
                       'acquisition_seconds_f32_bits': m.acquisition_wait_bits, 'acquisition_seconds_promoted': duration,
                       'click_after': 'actual acquisition callback completion, native release, owner unlink and empty queue'},
            'source_sha256': hashes, 'unchanged_prior_report_sha256': {
                old.relative_to(root).as_posix(): old_hash,
                init_report.relative_to(root).as_posix(): init_report_hash},
            'native_verification': verified, 'bridge_verification': bridge, 'acquisition_pins': {hex(a): list(v) for a, v in pins.items()},
            'acquisition_fields': fields,
            'supplied_boundaries': {
                'native_coroutine_record_creation': 'authored existing 88-byte records; dispatcher/callback/release execute',
                'clr': 'metadata/class/GC/delegate/list/string adapters; RuntimeInvoke calls actual managed MoveNext bridge',
                'clone_api': 'supplied distinct preconfigured Tracker clone; actual DelayReveal installs returned identity',
                'presentation': 'GameObject, text/color/sprite, tween services and RefreshView supplied; closure completion uninvoked',
                'prior_reset_fixture': 'stress history/status/runtime/latches; Init clears actual fields, preserves old savedAct and act',
                'other_actors': 'finished acquisition supplied with distinct roles and status components; bytes preserved',
                'readiness': 'explicit producer/consumer clocks, full signed frames, phases/generations and owner responses',
                'exclusions': 'no constructor, selectable deck generation, original init UI, live PlayerLoop timing, rendered capture or complete initial-Day PlayerHistory',
            },
            'managed_body_sha256': m.body_hashes,
            'engine_body_sha256': {name: hashlib.sha256(e.pe.get_data(a, b-a)).hexdigest()
                                   for name, (a, b) in e.routines.items()},
            'cases': cases, 'owner_cases': owners, 'failure_cases': failures}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    pooled = pool_snapshots(result)
    assert expand_snapshots(pooled) == result
    assert expand_snapshots(json.loads(json.dumps(pooled))) == result
    original = json.dumps(result, ensure_ascii=True, indent=2)
    candidate = json.dumps(pooled, ensure_ascii=True, indent=2)
    chosen = candidate if len(candidate.encode('utf-8')) < len(original.encode('utf-8')) else original
    args.output.write_text(chosen+'\n', encoding='utf-8')
    print(json.dumps({'metadata_counts': result['metadata_counts'],
                      'expanded_utf8_bytes': len(original.encode('utf-8')),
                      'output_utf8_bytes': len(chosen.encode('utf-8')),
                      'pooled_output': chosen is candidate,
                      'snapshot_blob_count': len(pooled['snapshot_blobs']) if chosen is candidate else 0}, sort_keys=True))
