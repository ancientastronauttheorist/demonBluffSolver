"""Conditional retained Hunter publication through actual native wait drains.

Record creation, runtime invocation, cache/GC/owner and UI gateways are supplied.
Managed producer/publication and engine dispatch/type/wait/tree bodies execute.
"""
import argparse
import hashlib
import itertools
import inspect
import json
import math
import struct
from collections import Counter
from pathlib import Path

from audit_character_assets import BUILD
from audit_hunter_role_publication import Machine as HunterMachine, expected, verify_native as verify_hunter
from audit_report_snapshots import pool_snapshots, expand_snapshots
from audit_unityplayer_completion import NativeCompletion
from audit_unityplayer_coroutines import audit as verify_bridge
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint
from audit_unityplayer_wait_tree import NativeTree, validate_tree


class NativePrefixStopped(Exception):
    """A supplied managed service stopped before its authored effects."""


class ScheduledHunter(HunterMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
        self.seed_info, self.depth = self.info, 0
        self.bridge_out = self.arena+0x120000
        self.pending_start = None
        self.bridge_rva = 0x1C8A780
        self.click_targets = {0x366270: 'Character$$OnClick', 0x386A60: 'RevealCard$$Reveal',
                              0x385FF0: 'RevealCard$$CheckIfCanRevealCard', 0x367500: 'Character$$OnReveal',
                              0x364EC0: 'Character$$GetHiddenCardsAmount', 0x37ED60: 'Gameplay$$OnCharacterReveal'}
        for start, name in self.click_targets.items():
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
            self.body_hashes[name] = hashlib.sha256(b''.join(self.pe.get_data(a, b-a) for a, b in chunks)).hexdigest()
            for a, b in chunks:
                decoded = list(self.cs.disasm(self.pe.get_data(a, b-a), a))
                assert sum(i.size for i in decoded) == b-a
                self.instructions.update({i.address: i for i in decoded})
        rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == self.bridge_rva
                and r['Name'] == 'UnityEngine.SetupCoroutine$$InvokeMoveNext']
        assert len(rows) == 1
        self.targets += rows
        chunks = []
        for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root = e
            while root.unwindinfo.Flags & 4:
                root = root.unwindinfo._chained_entry
            if root.struct.BeginAddress == self.bridge_rva:
                chunks.append((e.struct.BeginAddress, e.struct.EndAddress))
        assert chunks
        self.body_hashes['UnityEngine.SetupCoroutine$$InvokeMoveNext'] = hashlib.sha256(
            b''.join(self.pe.get_data(a, b-a) for a, b in chunks)).hexdigest()
        for a, b in chunks:
            ins = list(self.cs.disasm(self.pe.get_data(a, b-a), a))
            assert sum(i.size for i in ins) == b-a
            self.instructions.update({i.address: i for i in ins})
        self.q(self.base+0x26FE930, self.arena+0x121000)
        self.metadata_slots[self.base+0x26FE930] = self.arena+0x121000
        for start, length, wanted in [(0x3E8F30, 4, [('mov', 'rax, rcx'), ('ret', '')]),
                                      (0x4A0210, 7, [('cmp', 'rcx, rdx'), ('sete', 'al'), ('ret', '')])]:
            decoded = list(self.cs.disasm(self.pe.get_data(start, length), start))
            assert [(i.mnemonic, i.op_str) for i in decoded] == wanted
            self.instructions.update({i.address: i for i in decoded})
        for instruction in self.instructions.values():
            for operand in instruction.operands:
                if operand.type == capstone.CS_OP_MEM and operand.mem.base == capstone.x86.X86_REG_RIP and instruction.mnemonic == 'cmp' and operand.size == 1:
                    self.flags.add(instruction.address+instruction.size+operand.mem.disp)
        references = {i.address+i.size+o.mem.disp for i in self.instructions.values()
                      for o in i.operands if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP}
        for row in self.metadata['ScriptMetadata']+self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references and self.base+row['Address'] not in self.metadata_slots:
                self.bindings.setdefault(row['Name'], self.arena+0x4000+len(self.bindings)*0x200)
                self.metadata_slots[self.base+row['Address']] = self.bindings[row['Name']]
                self.q(self.base+row['Address'], self.bindings[row['Name']])
        self.click_fixtures = {name: self.arena+0x130000+i*0x1000 for i, name in enumerate(
            ['reveal_card', 'click_delegate', 'player_static', 'player_info', 'blocks', 'mana',
             'block_value', 'mana_value', 'value_class', 'reveal_delegate', 'event_static', 'tween', 'tween_callback',
             'click_config', 'click_bool', 'ui_event_static'])}
        self.static_ids.update({p: name for name, p in self.click_fixtures.items()})
        self.value_getter = self.stop+0xE00

    def prepare(self, options):
        self.info = self.seed_info
        self.pending_start, self.depth = None, 0
        super().prepare(options)
        for pointer in self.click_fixtures.values():
            self.u.mem_write(pointer, bytes(0x800))
        if options.get('route') == 'click':
            c = self.click_fixtures
            for pointer in self.board:
                self.d(pointer+0xE4, 5)
            self.q(self.bindings['Gameplay_TypeInfo']+0xB8, self.native_fixtures['board_static'])
            self.d(self.native_fixtures['board_static']+0x28, 10)
            self.d(self.native_fixtures['board_static']+0x38, 0)
            self.q(self.native_fixtures['board_static']+0x10, c['click_config'])
            self.q(c['click_config']+0x70, c['click_bool'])
            self.q(self.bindings['UIEvents_TypeInfo']+0xB8, c['ui_event_static'])
            self.q(self.bindings['PlayerController_TypeInfo']+0xB8, c['player_static'])
            self.q(c['player_static'], c['player_info'])
            self.q(c['player_info']+0x18, c['mana']); self.q(c['player_info']+0x20, c['blocks'])
            for resource, value, amount in [('blocks', 'block_value', 0), ('mana', 'mana_value', 1)]:
                self.q(c[resource]+0x10, c[value]); self.q(c[value], c['value_class'])
                self.d(c[value]+0x10, amount)
            self.q(c['value_class']+0x1A8, self.value_getter)
            self.q(c['value_class']+0x1B0, c['value_class']+0x500)
            self.q(self.actor+0x100, c['click_delegate'])
            self.q(c['click_delegate']+0x18, self.base+0x386A60)
            self.q(c['click_delegate']+0x40, c['reveal_card'])
            self.q(c['reveal_card']+0x20, self.actor)
            self.q(c['reveal_card']+0x28, self.fixtures['game'])
            self.q(c['reveal_card']+0x38, self.fixtures['rect'])
            self.q(c['reveal_card']+0x48, self.fixtures['rect'])
            self.q(self.bindings['GameplayEvents_TypeInfo']+0xB8, c['event_static'])
            self.q(c['event_static']+0x50, c['reveal_delegate'])
            self.q(c['event_static']+0x58, self.fixtures['event_delegate'])
            self.q(c['reveal_delegate']+0x18, self.base+0x37ED60)
            self.q(c['reveal_delegate']+0x40, c['click_config'])
            self.initial_other_bytes = {p: bytes(self.u.mem_read(p, 0x200)) for p in self.board if p != self.actor}

    def snapshot(self):
        result = super().snapshot()
        result.update(actor_state=self.rd(self.actor+0xE4),
                      actor_previous_state=self.rd(self.actor+0xE0), reveal_order=self.rd(self.actor+0x160),
                      reveal_card_init_reveal_byte=self.u.mem_read(self.click_fixtures['reveal_card']+0x50, 1)[0],
                      reveal_card_state_raw=self.rd(self.click_fixtures['reveal_card']+0x54),
                      gameplay_current_reveal=self.rd(self.rq(self.bindings['Gameplay_TypeInfo']+0xB8)+0x38),
                      saved_text=self.strings.get(self.rq(self.actor+0x198)),
                      saved_source_infos=[self.object_id(p) for p in self.generated
                                          if self.rq(p+0x10) == self.rq(self.actor+0x198)])
        return result

    def service_argument(self, value):
        names = [name for name, pointer in self.bindings.items() if pointer == value]
        if names:
            return sorted(names)[0]
        if value == 0 or value in self.objects or value in self.static_ids or value in self.string_labels or value in [self.actor, self.role, self.info]:
            return self.object_id(value)
        return hex(value)

    def hook(self, uc, address, size, data):
        rva, x = address-self.base, self.x
        if address == self.value_getter:
            cx, dx = self.reg(x.UC_X86_REG_RCX), self.reg(x.UC_X86_REG_RDX)
            assert cx in [self.click_fixtures['block_value'], self.click_fixtures['mana_value']]
            assert dx == self.click_fixtures['value_class']+0x500
            if self.event('resource_value_gateway', [self.object_id(cx), self.rd(cx+0x10)]):
                self.ret(self.rd(cx+0x10))
            return
        if rva in [0x1C7DC50, 0x5131D0, 0x6BC7A0, 0x6BC9D0, 0x6BBF10, 0x4D5170]:
            caller = self.rq(self.reg(x.UC_X86_REG_RSP))-self.base
            assert caller in [0x386B42, 0x3664D3, 0x386C28, 0x386C3C, 0x386C4F, 0x386C76, 0x386C88], hex(caller)
            arguments = [hex(rva), hex(caller)]+[self.service_argument(self.reg(register)) for register in
                [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]]
            arguments += [hex(self.reg(register)&((1<<64)-1)) for register in
                [x.UC_X86_REG_XMM0, x.UC_X86_REG_XMM1, x.UC_X86_REG_XMM2]]
            if self.event('reveal_ui_gateway', arguments):
                self.ret(1 if rva == 0x1C7DC50 else self.click_fixtures['tween'])
            return
        if rva == 0x2B7D40 and self.rq(self.reg(x.UC_X86_REG_RSP)) == self.base+0x386C5E:
            token = self.reg(x.UC_X86_REG_RCX)
            if self.event('reveal_tween_callback_allocation_gateway', [self.service_argument(token), '0x386c5e']):
                self.q(self.click_fixtures['tween_callback'], token)
                self.ret(self.click_fixtures['tween_callback'])
            return
        if rva == 0x1C7F160:
            cx, dx, r8 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8]]
            assert cx == self.actor and r8 == 0
            kind = 'speech' if self.objects[dx].startswith('speech') else 'result'
            item = self.speech_iterator(dx) if kind == 'speech' else self.iterator(dx)
            assert item['state'] == 0 and item['current'] is None
            if self.event('synchronous_start_gateway', [kind, item]):
                (self.speech_registered if kind == 'speech' else self.registered).append(item)
                self.pending_start = (dx, kind)
                uc.emu_stop()
            return
        if rva == 0x4060:
            slot, typename, iterator = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8]]
            assert slot == 0 and typename == self.arena+0x121000 and iterator in self.objects
            method = 0x376240 if self.objects[iterator].startswith('speech') else 0x375FE0
            if self.event('ienumerator_slot_zero_gateway', [self.object_id(iterator), hex(method)]):
                self.u.reg_write(x.UC_X86_REG_RCX, iterator)
                for register in [x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]:
                    self.u.reg_write(register, 0)
                self.u.reg_write(x.UC_X86_REG_RIP, self.base+method)
            return
        if rva == 0x2B7D40 and self.reg(x.UC_X86_REG_RCX) == self.bindings['ActedInfo_TypeInfo']:
            if self.event('producer_allocate_service', ['ActedInfo_TypeInfo']):
                p = self.seed_info if not self.generated else self.alloc(0x100)
                self.objects[p] = 'info'+str(len(self.generated))
                self.generated.append(p)
                self.info = p
                self.q(p, self.bindings['ActedInfo_TypeInfo'])
                self.ret(p)
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
        for i, register in enumerate(keep):
            self.u.reg_write(register, 0xFAB00000+i)
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, cx), (x.UC_X86_REG_RDX, dx),
                                (x.UC_X86_REG_R8, r8), (x.UC_X86_REG_R9, r9)]:
            self.u.reg_write(register, value & 0xFFFFFFFFFFFFFFFF)
        for register in [x.UC_X86_REG_RAX, x.UC_X86_REG_R10, x.UC_X86_REG_R11]:
            self.u.reg_write(register, 0)
        for register in [x.UC_X86_REG_XMM0, x.UC_X86_REG_XMM1, x.UC_X86_REG_XMM2,
                         x.UC_X86_REG_XMM3, x.UC_X86_REG_XMM4, x.UC_X86_REG_XMM5]:
            self.u.reg_write(register, 0)
        xmm6 = 0xFEDCBA98765432100123456789ABCDEF
        self.u.reg_write(x.UC_X86_REG_XMM6, xmm6)
        before = bytearray(self.u.mem_read(self.actor, 0x200))
        pc = self.base+address
        try:
            for _ in range(8):
                self.pending_start = None
                self.u.emu_start(pc, self.stop, timeout=10_000_000, count=100000)
                if self.pending_start is None:
                    break
                iterator, kind = self.pending_start
                self.pending_start = None
                context = self.u.context_save()
                handle = self.engine_join.start(iterator, kind)
                self.u.context_restore(context)
                self.ret(handle)
                pc = self.reg(x.UC_X86_REG_RIP)
            else:
                raise AssertionError('bounded synchronous start recursion exceeded')
            returned = self.reg(x.UC_X86_REG_RIP) == self.stop
            assert returned or self.error
            if returned:
                assert self.reg(x.UC_X86_REG_RSP) == sp+8
                assert all(self.reg(register) == 0xFAB00000+i for i, register in enumerate(keep))
                assert self.reg(x.UC_X86_REG_XMM6) == xmm6
            after = bytearray(self.u.mem_read(self.actor, 0x200))
            for offset, width in [(0xDC, 4), (0x11C, 1), (0x148, 8), (0x198, 8),
                                  (0xE0, 4), (0xE4, 4), (0x160, 4)]:
                before[offset:offset+width] = after[offset:offset+width] = bytes(width)
            assert before == after
            assert all(bytes(self.u.mem_read(p, 0x200)) == raw for p, raw in self.initial_other_bytes.items())
            return returned
        finally:
            self.depth -= 1

    def step(self, iterator):
        out = self.bridge_out+self.depth*0x100
        self.u.mem_write(out, b'\x7f')
        if not self.invoke(self.bridge_rva, iterator, out):
            raise NativePrefixStopped(self.error)
        result = self.u.mem_read(out, 1)[0]
        assert result in [0, 1]
        return result


class ScheduledEngine(NativeCompletion):
    routines = {**NativeCompletion.routines, 'wait_type': (0x779370, 0x7798D2)}

    def __init__(self, data, managed):
        import capstone
        import pefile
        super().__init__(data)
        self.managed = managed
        managed.engine_join = self
        self.pe = pefile.PE(data=data, fast_load=True)
        self.cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        self.cs.detail = True
        self.wait_class = self.arena+0x126000
        self.object_class, self.subclass = self.stop+0x900, self.stop+0xA00
        self.reset_join()

    def write_q(self, address, value):
        self.uc.mem_write(address, struct.pack('<Q', value))

    def write_d(self, address, value):
        self.uc.mem_write(address, struct.pack('<I', value & 0xFFFFFFFF))

    def generation(self):
        return struct.unpack('<I', self.uc.mem_read(self.owner+0x48, 4))[0]

    def reset_join(self, generation=0):
        self.reset_stop()
        self.write_d(self.owner+0x48, generation)
        head = self.native_owner+0x70
        self.write_q(head, head); self.write_q(head+8, head)
        self.write_q(self.native_owner+0x40, self.owner_vtable)
        self.write_q(self.owner_vtable+0x10, self.owner_context)
        self.uc.mem_write(self.cache, bytes(0x1000))
        self.write_q(self.cache+0xCF0, self.method)
        self.write_q(self.cache+0xDC8, self.wait_class)
        for slot, target in [(0x1CD6AF8, self.cache), (0x1CD6250, self.param_count),
                             (0x1CD62A0, self.invoke), (0x1CD6038, self.free_handle),
                             (0x1CD62C8, self.object_class), (0x1CD6310, self.subclass),
                             (0x1C6E708, self.engine)]:
            self.write_q(self.base+slot, target)
        self.depth, self.registry, self.payload_registry = 0, {}, {}
        self.wait_mirrors, self.wait_info, self.next_identity = {}, {}, 0
        self.pending_invoke, self.pending_insert, self.visit_identity = None, None, None
        self.join_trace, self.owner_outcomes = [], {}
        self.producer_time, self.producer_frame = 1.0, 7
        self.producer_override = None
        self.executed = set()

    def queue_state(self):
        entries = []
        for identity in sorted(self.deadlines, key=self.deadlines.__getitem__):
            record = self.records[self.nodes[identity]]
            entries.append({'id': identity, 'kind': self.wait_info[identity]['kind'],
                            'iterator': self.wait_info[identity]['iterator'],
                            'deadline': struct.unpack_from('<d', record, 0)[0],
                            'frame_threshold': struct.unpack_from('<q', record, 8)[0],
                            'generation': struct.unpack_from('<I', record, 0x38)[0],
                            'phase_mask': struct.unpack_from('<I', record, 0x34)[0]})
        return {'generation': self.generation(), 'entries': entries, 'next_id': self.next_identity}

    def retained_native_records(self):
        head, linked = self.native_owner+0x70, []
        pointer = self.qword(head)
        while pointer != head:
            assert pointer in self.payload_registry and pointer not in linked
            linked.append(pointer); pointer = self.qword(pointer)
        return [{'iterator': row['label'], 'kind': row['kind'],
                 'reference_count': struct.unpack('<i', self.uc.mem_read(pointer+0x60, 4))[0],
                 'gc_handle': struct.unpack('<I', self.uc.mem_read(pointer+0x10, 4))[0],
                 'cached_enumerator_present': bool(self.qword(pointer+0x20)),
                 'owner_linked': pointer in linked} for pointer, row in self.payload_registry.items()]

    def validate(self):
        order = sorted(self.deadlines, key=self.deadlines.__getitem__)
        validate_tree(self.uc.mem_read, self.container, self.head, self.records, [0]*len(order))
        observed = []
        reverse = {node: identity for identity, node in self.nodes.items()}
        def visit(node):
            if node == self.head:
                return
            visit(self.qword(node)); observed.append(reverse[node]); visit(self.qword(node+0x10))
        visit(self.qword(self.head+8))
        assert observed == order

    def _on_code(self, uc, address, size, data):
        rva, x = address-self.base, self.x86
        self.executed.add(rva if address >= self.base else 'gateway:'+hex(address-self.stop))
        if address == getattr(self, 'param_count', None):
            assert uc.reg_read(x.UC_X86_REG_RCX) == self.method
            self._return(2); return
        if address == getattr(self, 'owner_context', None):
            assert uc.reg_read(x.UC_X86_REG_RCX) == self.native_owner+0x40
            result = uc.reg_read(x.UC_X86_REG_RDX)
            self.write_q(result, 0); self._return(result); return
        if address == getattr(self, 'invoke', None):
            assert uc.reg_read(x.UC_X86_REG_RCX) == self.method and uc.reg_read(x.UC_X86_REG_RDX) == 0
            args = uc.reg_read(x.UC_X86_REG_R8)
            iterator = self.qword(args)
            assert iterator in self.registry
            result_address = self.qword(self.qword(args+8))
            self.pending_invoke = (iterator, result_address, uc.reg_read(x.UC_X86_REG_R9))
            self.join_trace.append({'kind': 'runtime_invoke_gateway', 'iterator': self.registry[iterator]['label'],
                                    'method': 'UnityEngine.SetupCoroutine.InvokeMoveNext', 'static_instance': None,
                                    'argument_count': 2, 'result_storage': 'native dispatcher byte'})
            uc.emu_stop(); return
        if address == getattr(self, 'gc_write', None):
            self.write_q(uc.reg_read(x.UC_X86_REG_RDX), uc.reg_read(x.UC_X86_REG_R8))
            self._return(); return
        if address == getattr(self, 'gc_target', None):
            self._return(self.gc_targets[uc.reg_read(x.UC_X86_REG_ECX)]); return
        if address == getattr(self, 'free_handle', None):
            self.join_trace.append({'kind': 'gc_handle_free_gateway', 'handle': uc.reg_read(x.UC_X86_REG_ECX)})
            self._return(); return
        if address == getattr(self, 'object_class', None):
            assert uc.reg_read(x.UC_X86_REG_RCX) in self.wait_mirrors
            self._return(self.wait_class); return
        if address == getattr(self, 'subclass', None):
            assert uc.reg_read(x.UC_X86_REG_RCX) == uc.reg_read(x.UC_X86_REG_RDX) == self.wait_class
            assert uc.reg_read(x.UC_X86_REG_R8) & 0xFF == 1
            self._return(1); return
        if rva == 0x17D6A84:
            value = struct.unpack('<d', struct.pack('<Q', uc.reg_read(x.UC_X86_REG_XMM0) & ((1<<64)-1)))[0]
            assert math.isfinite(value)
            self._return(0); return
        if rva == 0x6E78E0:
            self._return(self.profiler); return
        if rva == 0x355F00:
            self._return(); return
        if rva == 0x17A7808:
            payload = uc.reg_read(x.UC_X86_REG_RCX)
            assert payload in self.payload_registry and uc.reg_read(x.UC_X86_REG_RDX) == 0x88
            self.join_trace.append({'kind': 'payload_free_gateway', 'iterator': self.payload_registry[payload]['label']})
            self._return(); return
        if rva == 0x1049F20:
            self.join_trace.append({'kind': 'owner_mismatch_diagnostic_gateway'})
            self._return(); return
        if rva == 0x151EF0:
            key = struct.unpack('<I', uc.mem_read(uc.reg_read(x.UC_X86_REG_RDX), 4))[0]
            assert key == 11
            outcome = self.owner_outcomes.get(self.visit_identity, 'valid')
            self.join_trace.append({'kind': 'owner_lookup_gateway', 'id': self.visit_identity, 'outcome': outcome})
            if outcome == 'missing':
                self._return(self.owner_entry+24)
            else:
                value = 0 if outcome == 'null' else self.native_owner+(0x100 if outcome == 'mismatch' else 0)
                uc.mem_write(self.owner_entry, struct.pack('<QQQ', key, 0, value))
                self._return(self.owner_entry)
            return
        if rva == 0x779070:
            payload = uc.reg_read(x.UC_X86_REG_RCX)
            row = self.payload_registry[payload]
            original = self.managed.rq(row['pointer']+0x18)
            assert self.managed.objects[original].startswith('wait')
            mirror = self.arena+0x1A0000+len(self.wait_mirrors)*0x100
            self.wait_mirrors[mirror] = original
            self.uc.mem_write(mirror, bytes(self.managed.u.mem_read(original, 0x20)))
            self.write_q(mirror, self.wait_class)
            self.join_trace.append({'kind': 'current_yield_gateway', 'iterator': row['label'],
                                    'wait': self.managed.object_id(original),
                                    'duration_bits': self.managed.rd(original+0x10)})
            self.uc.reg_write(x.UC_X86_REG_RCX, payload)
            self.uc.reg_write(x.UC_X86_REG_RDX, mirror)
            self.uc.reg_write(x.UC_X86_REG_RIP, self.base+0x779370)
            return
        if rva == 0x440F00:
            record = bytes(uc.mem_read(uc.reg_read(x.UC_X86_REG_R8), 0x40))
            payload = struct.unpack_from('<Q', record, 0x18)[0]
            row = self.payload_registry[payload]
            assert struct.unpack_from('<Q', record, 0x10)[0] == 0
            assert struct.unpack_from('<I', record, 0x30)[0] == 11
            assert struct.unpack_from('<QQ', record, 0x20) == (self.base+0x778B30, self.base+0x778BD0)
            if row['kind'] == 'speech':
                desc = self.managed.rq(row['pointer']+0x28)
                assert self.managed.rq(self.managed.actor+0x198) == desc
                assert self.managed.display_text == self.managed.object_id(desc)
            self.pending_insert = {'id': self.next_identity, 'record': record, 'node': self.next_node,
                                   'kind': row['kind'], 'iterator': row['label'],
                                   'producer': {'time': struct.unpack('<d', uc.mem_read(self.engine+0x60, 8))[0],
                                                'frame': struct.unpack('<q', uc.mem_read(self.engine+0xC8, 8))[0],
                                                'duration_bits': self.managed.rd(self.managed.rq(row['pointer']+0x18)+0x10)}}
            self.next_identity += 1
        if rva == 0x7794F1:
            item = self.pending_insert
            assert item is not None and bytes(uc.mem_read(item['node']+0x20, 0x40)) == item['record']
            identity = item['id']
            self.nodes[identity], self.records[item['node']] = item['node'], item['record']
            self.deadlines[identity] = struct.unpack_from('<d', item['record'], 0)[0]
            self.wait_info[identity] = {'kind': item['kind'], 'iterator': item['iterator']}
            self.pending_insert = None
            self.validate()
            self.join_trace.append({'kind': 'native_wait_inserted', 'id': identity,
                                    'producer': item['producer'],
                                    'queue': self.queue_state(), 'managed': self.managed.snapshot()})
        if rva in [0x43BC60, 0x43BB00]:
            node = uc.reg_read(x.UC_X86_REG_R8)
            identity = next(i for i, p in self.nodes.items() if p == node)
            self.join_trace.append({'kind': 'native_wait_erase', 'id': identity})
            del self.nodes[identity]; del self.records[node]; del self.deadlines[identity]
        if rva == 0x43BE23:
            node = uc.reg_read(x.UC_X86_REG_RBX)
            self.visit_identity = next(i for i, p in self.nodes.items() if p == node)
            self.join_trace.append({'kind': 'native_wait_visit', 'id': self.visit_identity})
        if rva == 0x778B30:
            payload = uc.reg_read(x.UC_X86_REG_RDX)
            self.validate()
            self.join_trace.append({'kind': 'native_wait_callback', 'id': self.visit_identity,
                                    'iterator': self.payload_registry[payload]['label']})
        if rva == 0x778BD0:
            payload = uc.reg_read(x.UC_X86_REG_RCX)
            if payload in self.payload_registry:
                self.join_trace.append({'kind': 'native_reference_release', 'iterator': self.payload_registry[payload]['label'],
                                        'before': struct.unpack('<i', uc.mem_read(payload+0x60, 4))[0]})
        NativeTree._on_code(self, uc, address, size, data)

    def run_native(self, name, *args):
        x, depth = self.x86, self.depth
        sp = self.stack+0xF008-depth*0x4000
        assert sp >= self.stack+0x3008
        self.depth += 1
        self.uc.mem_write(sp, struct.pack('<Q', self.stop)+bytes(0x28))
        for register in [x.UC_X86_REG_RAX, x.UC_X86_REG_R10, x.UC_X86_REG_R11,
                         x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]:
            self.uc.reg_write(register, 0)
        self.uc.reg_write(x.UC_X86_REG_RSP, sp)
        keep = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for index, register in enumerate(keep):
            self.uc.reg_write(register, 0xFEA00000+index)
        for register in [x.UC_X86_REG_XMM0, x.UC_X86_REG_XMM1, x.UC_X86_REG_XMM2,
                         x.UC_X86_REG_XMM3, x.UC_X86_REG_XMM4, x.UC_X86_REG_XMM5]:
            self.uc.reg_write(register, 0)
        xmm6 = 0xFEDCBA98765432100123456789ABCDEF
        self.uc.reg_write(x.UC_X86_REG_XMM6, xmm6)
        self.uc.reg_write(x.UC_X86_REG_EFLAGS, 2); self.uc.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        for register, value in zip([x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9], args):
            self.uc.reg_write(register, value)
        pc = self.base+self.routines[name][0]
        try:
            for _ in range(32):
                self.pending_invoke = None
                self.uc.emu_start(pc, self.stop, timeout=10_000_000, count=100000)
                if self.pending_invoke is None:
                    break
                iterator, out, error = self.pending_invoke
                self.pending_invoke = None
                context = self.uc.context_save()
                if self.producer_override is not None:
                    self.set_producer(*self.producer_override)
                result = self.managed.step(iterator)
                self.uc.context_restore(context)
                self.uc.mem_write(out, bytes([result])); self.write_q(error, 0)
                self.join_trace.append({'kind': 'managed_move_next_return', 'iterator': self.registry[iterator]['label'],
                                        'result': result, 'managed': self.managed.snapshot()})
                self._return()
                pc = self.uc.reg_read(x.UC_X86_REG_RIP)
            else:
                raise AssertionError('bounded managed invoke count exceeded')
            assert self.uc.reg_read(x.UC_X86_REG_RIP) == self.stop
            assert self.uc.reg_read(x.UC_X86_REG_RSP) == sp+8
            assert all(self.uc.reg_read(register) == 0xFEA00000+i for i, register in enumerate(keep))
            assert self.uc.reg_read(x.UC_X86_REG_XMM6) == xmm6
            return self.uc.reg_read(x.UC_X86_REG_RAX)
        finally:
            self.depth -= 1

    def set_producer(self, time, frame):
        assert math.isfinite(time) and -(1<<63) <= frame < (1<<63)
        self.producer_time, self.producer_frame = time, frame
        self.uc.mem_write(self.engine+0x60, struct.pack('<d', time))
        self.uc.mem_write(self.engine+0xC8, struct.pack('<q', frame))

    def start(self, iterator, kind):
        payload = self.arena+0x1B0000+len(self.registry)*0x100
        self.uc.mem_write(payload, bytes(0x88))
        row = {'pointer': iterator, 'payload': payload, 'kind': kind, 'label': self.managed.object_id(iterator)}
        self.registry[iterator], self.payload_registry[payload] = row, row
        self.payload_labels[payload] = row['label']
        handle = 17+len(self.registry)
        self.gc_targets[handle] = iterator
        for offset, value in [(0x10, handle), (0x20, iterator), (0x58, self.native_owner)]:
            self.write_q(payload+offset, value)
        self.write_d(payload+0x18, 2); self.write_d(payload+0x60, 1)
        head = self.native_owner+0x70
        last = self.qword(head+8)
        self.write_q(payload, head); self.write_q(payload+8, last)
        self.write_q(last, payload); self.write_q(head+8, payload)
        self.join_trace.append({'kind': 'record_creation_gateway', 'iterator': row['label'], 'coroutine_kind': kind,
                                'producer_time': struct.unpack('<d', self.uc.mem_read(self.engine+0x60, 8))[0],
                                'producer_frame': struct.unpack('<q', self.uc.mem_read(self.engine+0xC8, 8))[0]})
        self.run_native('dispatch', payload, 0)
        assert struct.unpack('<i', self.uc.mem_read(payload+0x60, 4))[0] == 2
        self.run_native('release_native', payload)
        handle_pointer = self.managed.arena+0x1E0000+len(self.registry)*0x100
        self.managed.objects[handle_pointer] = 'handle'+str(len(self.registry))
        self.managed.q(handle_pointer+0x10, payload)
        return handle_pointer

    def drain_join(self, time, frame, phase=2):
        before = self.queue_state()
        start = len(self.join_trace)
        self.set_clock(time, frame)
        self.run_native('drain', self.owner, phase)
        self.validate()
        return {'input': {'time': time, 'frame': frame, 'phase': phase, 'generation_before': before['generation']},
                'before': before, 'after': self.queue_state(), 'events': self.join_trace[start:],
                'managed': self.managed.snapshot()}


def audit(game_root, dumper_root):
    repository = Path(__file__).parents[2]
    old_paths = [repository/'reverse_engineering/reports'/f'{BUILD}_{suffix}.json'
                 for suffix in ['hunter_role_publication', 'unity_completion']]
    old_hashes = {path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in old_paths}
    m = ScheduledHunter(game_root, dumper_root)
    verified = verify_hunter(m)
    bridge = verify_bridge(game_root)
    data = (game_root/'UnityPlayer.dll').read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    e = ScheduledEngine(data, m)
    click_pins = {0x3662F2: ('mov', 'rax, qword ptr [rdi + 0x100]'),
                  0x366306: ('call', 'qword ptr [rax + 0x18]'),
                  0x386B6A: ('mov', 'byte ptr [rsi + 0x50], 1'),
                  0x386BCB: ('mov', 'dword ptr [rsi + 0x54], 0x14'),
                  0x37EDA6: ('inc', 'dword ptr [rax + 0x38]'),
                  0x386BA4: ('call', '0x3645c0'), 0x386BC2: ('call', '0x367500'),
                  0x36645A: ('mov', 'dword ptr [rdi + 0xe0], 5'),
                  0x366464: ('mov', 'dword ptr [rdi + 0xe4], 0xa'),
                  0x1C8A7DB: ('call', '0x4060'), 0x1C8A7E0: ('mov', 'byte ptr [rbx], al')}
    for address, want in click_pins.items():
        assert address in m.instructions
        assert (m.instructions[address].mnemonic, m.instructions[address].op_str) == want
    source_paths = {Path(inspect.getfile(cls)) for cls in ScheduledHunter.__mro__[:-1]+ScheduledEngine.__mro__[:-1]}
    source_paths.update([Path(inspect.getfile(verify_bridge)), Path(inspect.getfile(validate_tree)),
                         Path(inspect.getfile(pool_snapshots))])
    source_hashes = {path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest()
                     for path in sorted(source_paths)}
    engine_seen, managed_seen = set(), set()
    cases, clock_cases, stress_cases, owner_cases, failure_cases = [], [], [], [], []
    duration = struct.unpack('<f', struct.pack('<I', m.speech_wait_bits))[0]
    assert duration == 0.4000000059604645

    def begin(options, frame=7, generation=0):
        m.prepare(options); e.reset_join(generation); e.set_producer(1.0, frame)
        if options.get('route') == 'click':
            returned = m.invoke(0x366270, m.actor)
        else:
            returned = m.invoke(0x3645C0, m.actor, 30)
        assert returned and not m.error
        assert e.queue_state()['entries'] == [{'id': 0, 'kind': 'result', 'iterator': m.registered[0]['id'],
            'deadline': 1.0, 'frame_threshold': frame+1, 'generation': generation, 'phase_mask': 10}]
        if options.get('route') == 'click':
            assert m.snapshot()['actor_state'] == 10 and m.snapshot()['actor_previous_state'] == 5
            assert m.snapshot()['reveal_order'] == 1
            assert m.snapshot()['reveal_card_init_reveal_byte'] == 1
            assert m.snapshot()['reveal_card_state_raw'] == 20
            assert m.snapshot()['gameplay_current_reveal'] == 1
            inserted = [v for v in e.join_trace if v['kind'] == 'native_wait_inserted']
            assert inserted[0]['managed']['actor_state'] == 5 and inserted[0]['managed']['reveal_order'] == 0
            assert inserted[0]['managed']['reveal_card_init_reveal_byte'] == 1
            assert inserted[0]['managed']['reveal_card_state_raw'] == 0
            assert inserted[0]['managed']['gameplay_current_reveal'] == 0
        return {'input': dict(options, producer_time=1.0, producer_frame=frame, generation=generation),
                'initial_queue': e.queue_state(), 'initial': m.snapshot(),
                'initial_engine_events': e.join_trace.copy(), 'drains': []}

    def drain(case, time, frame, callbacks, remaining, phase=2):
        row = e.drain_join(time, frame, phase)
        got = [v['id'] for v in row['events'] if v['kind'] == 'native_wait_callback']
        assert got == callbacks, (case['input'], row['input'], got, callbacks)
        assert [v['id'] for v in row['after']['entries']] == remaining
        assert row['after']['generation'] == (row['before']['generation']+1)&0xFFFFFFFF
        row['expected_callback_ids'], row['expected_remaining_ids'] = callbacks, remaining
        case['drains'].append(row)
        return row

    def finish(case, wants, count=1):
        final = m.snapshot()
        if case['input']['route'] == 'click':
            assert final['reveal_card_init_reveal_byte'] == 1
            assert final['reveal_card_state_raw'] == 20
            assert final['gameplay_current_reveal'] == 1
        assert not e.queue_state()['entries']
        assert final['uses_bits'] == (-count)&0xFFFFFFFF
        assert final['history'] == ['prior_info']+['info'+str(i) for i in range(count)]
        assert final['history_version'] == 9+count
        assert final['saved_text'] == wants[-1]['text']
        assert final['shown'] == [want['text'] for want in wants]
        assert len(final['captured_callbacks']) == count
        assert len(final['generated']) == count
        for info, want in zip(final['generated'], wants):
            assert info['description'] == want['text'] and info['ordered_reference_ids'] == want['ordered_reference_ids']
        assert all(struct.unpack('<i', e.uc.mem_read(p+0x60, 4))[0] == 0 for p in e.payload_registry)
        assert e.qword(e.native_owner+0x70) == e.qword(e.native_owner+0x78) == e.native_owner+0x70
        case.update(expected=wants, final=final, managed_events=m.events.copy(),
                    final_queue=e.queue_state(), engine_events=e.join_trace.copy(),
                    final_native_records=e.retained_native_records())
        engine_seen.update(e.executed); managed_seen.update(m.executed)
        return case

    for baa, actor in itertools.product(range(4), repeat=2):
        for draw in (range(2) if actor == baa else range(1)):
            options = {'baa_seat': baa, 'actor_seat': actor, 'draw': draw, 'route': 'click'}
            c = begin(options)
            drain(c, 0.5, 8, [], [0]); drain(c, 1.0, 8, [], [0], phase=0)
            drain(c, 1.0, 7, [], [0]); drain(c, 1.0, 8, [0], [1])
            assert m.snapshot()['history'] == ['prior_info', 'info0']
            speech = e.queue_state()['entries'][0]
            assert speech['deadline'] == 1.0+duration and speech['frame_threshold'] == 9
            assert m.snapshot()['saved_text'] == expected(options)['text']
            drain(c, 1.3, 9, [], [1]); drain(c, 1.5, 8, [], [1]); drain(c, 1.5, 9, [1], [])
            cases.append(finish(c, [expected(options)]))

    for frame, generation in itertools.product([7, -2, 2**32+7], [0, 0xFFFFFFFF]):
        options = {'baa_seat': 0, 'actor_seat': 1, 'draw': 0, 'route': 'click'}
        c = begin(options, frame, generation)
        drain(c, 1.0, frame, [], [0]); drain(c, 1.0, frame+1, [0], [1])
        assert e.queue_state()['entries'][0]['frame_threshold'] == frame+2
        drain(c, 1.5, frame+2, [1], [])
        clock_cases.append(finish(c, [expected(options)]))

    for bluff, schedule in itertools.product([False, True], ['equal_zero', 'equal_mixed']):
        options = {'baa_seat': 0, 'actor_seat': 0 if bluff else 1, 'draw': 0, 'route': 'supplied_day'}
        c = begin(options)
        first = expected(options)
        if schedule == 'equal_mixed':
            drain(c, 1.0, 8, [0], [1])
            e.set_producer(1.0+duration, 8)
        m.options['draw'] = int(bluff)
        second = expected({**options, 'draw': int(bluff)})
        assert m.invoke(0x3645C0, m.actor, 30)
        c['schedule'], c['second_request_input'] = schedule, {'draw': int(bluff), 'producer_time': e.producer_time,
                                                              'producer_frame': e.producer_frame}
        c['second_request_queue'], c['second_request_snapshot'] = e.queue_state(), m.snapshot()
        assert m.captured_callbacks[0]['info'] == 'info0' and m.captured_callbacks[1]['info'] == 'info1'
        if schedule == 'equal_zero':
            drain(c, 1.0, 8, [0, 1], [2, 3])
            drain(c, 1.5, 9, [2, 3], [])
        else:
            assert e.queue_state()['entries'][0]['deadline'] == e.queue_state()['entries'][1]['deadline']
            drain(c, 1.0+duration, 9, [1, 2], [3])
            drain(c, 2.0, 10, [3], [])
        stress_cases.append(finish(c, [first, second], 2))

    options = {'baa_seat': 0, 'actor_seat': 1, 'draw': 0, 'route': 'supplied_day'}
    c = begin(options)
    assert m.invoke(0x3645C0, m.actor, 30)
    c['second_request_queue'], c['second_request_snapshot'] = e.queue_state(), m.snapshot()
    e.producer_override = (1.0, 6)
    c['resume_producer_override'] = {'time': 1.0, 'frame': 6}
    first = drain(c, 2.0, 10, [0, 1], [2, 3])
    assert [v['id'] for v in first['events'] if v['kind'] == 'native_wait_visit'] == [0, 1, 2, 3]
    assert first['after']['entries'][0]['generation'] == first['after']['generation']
    assert first['after']['entries'][0]['deadline'] <= 2.0 and first['after']['entries'][0]['frame_threshold'] <= 10
    e.producer_override = None
    drain(c, 2.0, 10, [2, 3], [])
    same_generation_case = finish(c, [expected(options)]*2, 2)

    options = {'baa_seat': 0, 'actor_seat': 1, 'draw': 0, 'route': 'click'}
    for outcome in ['missing', 'null', 'mismatch']:
        c = begin(options)
        e.owner_outcomes[0] = outcome
        drain(c, 1.0, 8, [0] if outcome == 'mismatch' else [], [])
        final = m.snapshot()
        assert final['history'] == ['prior_info'] and final['saved_text'] == 'old speech'
        assert not final['shown'] and len(final['generated']) == 1
        payload = e.registry[next(iter(e.registry))]['payload']
        assert struct.unpack('<i', e.uc.mem_read(payload+0x60, 4))[0] == 0
        c.update(owner_outcome=outcome, final=final, final_queue=e.queue_state(), engine_events=e.join_trace.copy())
        owner_cases.append(c); engine_seen.update(e.executed); managed_seen.update(m.executed)

    for baseline in [cases[2], cases[0]]:
        counts = {}
        for index, event in enumerate(baseline['managed_events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0)+1
            options = {k: baseline['input'][k] for k in ['baa_seat', 'actor_seat', 'draw', 'route']}
            m.prepare({**options, 'failure': [kind, counts[kind]]})
            e.reset_join(); e.set_producer(1.0, 7)
            try:
                returned = m.invoke(0x366270, m.actor)
                if returned:
                    e.drain_join(0.5, 8, 2); e.drain_join(1.0, 8, 0); e.drain_join(1.0, 7, 2)
                    e.drain_join(1.0, 8, 2); e.drain_join(1.3, 9, 2); e.drain_join(1.5, 8, 2)
                    e.drain_join(1.5, 9, 2)
            except NativePrefixStopped:
                pass
            assert m.error
            assert [{k: v for k, v in row.items() if k != 'snapshot'} for row in m.events] == [
                {k: v for k, v in row.items() if k != 'snapshot'} for row in baseline['managed_events'][:index+1]]
            assert m.events[-1] == event
            assert m.snapshot() == event['snapshot']
            failure_cases.append({'baseline_input': options, 'service': [kind, counts[kind]],
                                  'prefix_length': index+1, 'exact_managed_prefix_and_snapshot': True,
                                  'queue_at_stop': e.queue_state(), 'native_records_at_stop': e.retained_native_records()})

    assert old_hashes == {path.relative_to(repository).as_posix(): hashlib.sha256(path.read_bytes()).hexdigest() for path in old_paths}
    compositions = cases+clock_cases+stress_cases+[same_generation_case]+owner_cases
    engine_event_counts = Counter(event['kind'] for case in compositions for event in case['engine_events'])
    metadata_counts = {'composition_count': len(compositions),
                       'completed_publication_composition_count': len(compositions)-len(owner_cases),
                       'day_request_count': sum(len(case['final']['captured_callbacks']) for case in compositions),
                       'drain_count': sum(len(case['drains']) for case in compositions),
                       'managed_prefix_stop_count': len(failure_cases),
                       'managed_body_fingerprint_count': len(verified['hunter_body_hashes']),
                       'engine_body_fingerprint_count': len(e.routines),
                       'source_hash_count': len(source_hashes),
                       'engine_event_counts': dict(sorted(engine_event_counts.items()))}
    return {'schema_version': 1, 'build_id': BUILD, 'engine_sha256': digest,
            'scope': 'Conditional supplied N4 finished acquisition. Actual Hidden Character.OnClick -> installed RevealCard.Reveal -> native quota/mana predicate -> Character.Act(Day30), Hunter real/bluff producer and publication; synchronous actual managed MoveNext under supplied runtime invocation, actual engine dispatcher/type-wait/red-black queue/drain/callback/release. Allocation/native coroutine record creation, runtime/cache/GC/owner lookup/resource/CLR/UI gateways are supplied. No selectable N4 generation, CLR interpreter, live engine phase legality, pixel readiness or complete initial-Day PlayerHistory certificate.',
            'old_audit_sources': ['reverse_engineering/scripts/audit_hunter_role_publication.py',
                                  'reverse_engineering/scripts/audit_unityplayer_completion.py'],
            'producer_consumer_bridge_semantic_checks': bridge['semantic_checks_passed'],
            'metadata_counts': metadata_counts,
            'selected_click_bridge_instruction_assertions': len(click_pins)+5,
            'source_sha256': source_hashes, 'unchanged_old_report_sha256': old_hashes,
            'engine_body_sha256': {name: hashlib.sha256(e.pe.get_data(a, b-a)).hexdigest()
                                   for name, (a, b) in e.routines.items()},
            'hunter_verified': verified, 'display_id_by_native_seat': [4, 3, 2, 1],
            'supplied_runtime_fields': {'gameplay_state': 10, 'trigger': 30, 'hidden_state': 5, 'alive_state': 10,
                'mana': 1, 'blocks': 0, 'read_real_role': False, 'initial_uses': 0,
                'backside_active_self': True, 'init_reveal': False, 'killed_by_demon': False,
                'initial_reveal_card_state_raw': 0, 'reveal_tween_completion_invoked': False,
                'prior_history': 'supplied stress record; not initial-Day public observation',
                'unused_entry_volatiles': {'rax': 0, 'r10': 0, 'r11': 0, 'xmm0_to_xmm5': 0},
                'producer_clock_offset': '0x60', 'consumer_clock_offset': '0x90', 'signed_frame_offset': '0xc8'},
            'case_count': len(cases), 'cases': cases, 'clock_case_count': len(clock_cases), 'clock_cases': clock_cases,
            'stress_case_count': len(stress_cases), 'stress_cases': stress_cases,
            'same_generation_case': same_generation_case,
            'owner_case_count': len(owner_cases), 'owner_cases': owner_cases,
            'failure_case_count': len(failure_cases), 'failure_cases': failure_cases,
            'managed_executed_address_count': len(managed_seen), 'engine_executed_address_count': len(engine_seen)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    expanded_text = json.dumps(report, indent=2)+'\n'
    pooled = pool_snapshots(report)
    pooled_text = json.dumps(pooled, indent=2)+'\n'
    assert expand_snapshots(pooled) == report
    assert expand_snapshots(json.loads(pooled_text)) == report
    output = pooled_text if len(pooled_text.encode('utf-8')) < len(expanded_text.encode('utf-8')) else expanded_text
    args.output.write_text(output, encoding='utf-8')
    print(json.dumps({'metadata_counts': report['metadata_counts'],
                      'expanded_utf8_bytes': len(expanded_text.encode('utf-8')),
                      'output_utf8_bytes': len(output.encode('utf-8')),
                      'pooled_output': output is pooled_text,
                      'snapshot_blob_count': len(pooled['snapshot_blobs']) if output is pooled_text else 0}))
