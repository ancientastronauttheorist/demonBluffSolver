"""Exact CardInteraction lifecycle callers with supplied visual/runtime/delegate services."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

from audit_character_assets import BUILD
from audit_card_interaction_awake import Machine as AwakeMachine, POISON, INCOMING
from audit_character_oracle_reveal_join import expand_memory, pool_memory
from audit_report_snapshots import expand_snapshots, pool_snapshots

TARGETS = {'OnEnable': (0x35FE70, 0x35FF8D, 0x35FF90, 'tdi5468.m0001'),
           'OnDisable': (0x35FD20, 0x35FE6F, 0x35FE70, 'tdi5468.m0002')}
SERVICES = {0x35F810: 'CardInteraction$$MouseExit', 0x5044D0: 'DG.Tweening.DOTween$$Kill',
            0x1C7F4A0: 'UnityEngine.MonoBehaviour$$StopAllCoroutines',
            0x1C82480: 'UnityEngine.Object$$op_Inequality', 0x4D5170: 'System.Action$$.ctor',
            0x116BCC0: 'System.Delegate$$Combine', 0x116E070: 'System.Delegate$$Remove'}
SITES = {'OnEnable': {'constructor': 0x35FF26, 'allocate': 0x35FF0E, 'delegate': 0x35FF34,
                     'barrier': 0x35FF66, 'null': 0x35FF7B, 'cast': 0x35FF87},
         'OnDisable': {'constructor': 0x35FE08, 'allocate': 0x35FDF0, 'delegate': 0x35FE16,
                      'barrier': 0x35FE48, 'null': 0x35FE5D, 'cast': 0x35FE69}}


class Machine(AwakeMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        import capstone
        raw = (Path(dumper_root) / 'dump.cs').read_bytes()
        extraction = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs']['dump_cs']['sha256'].upper()
        dump = raw.decode('utf-8-sig')
        character = re.search(r'^public class Character :[^\n]* // TypeDefIndex: 5487\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert character and 'private Action <onClick>k__BackingField; // 0x100' in character[1]
        action = re.search(r'^public sealed class Action : MulticastDelegate // TypeDefIndex: 153\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert action and 'public virtual void Invoke()' in action[1]
        self.instructions, self.body_addresses, self.ranges, self.targets, self.flags = {}, set(), {}, [], {}
        self.eh = {}
        for name, (start, end, following, stable) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start]
            assert len(rows) == 1 and rows[0]['Name'] == 'CardInteraction$$' + name
            assert rows[0]['Signature'] == f'void CardInteraction__{name} (CardInteraction_o* __this, const MethodInfo* method);' and rows[0]['TypeSignature'] == 'vii'
            self.targets.append(dict(rows[0], stable_method_id=stable)); self.decode(name, start, end, following)
            entry = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == start)
            assert entry.unwindinfo.Flags == 0
            self.eh[name] = {'root_unwind_flags': 0, 'method_local_exception_handler': False}
            assert len(self.ranges[name]['unwind_chunks']) == 6
        self.checks = {0x35FEB6: ('call', '0x35f810'), 0x35FEC2: ('mov', 'rbx, qword ptr [rsi + 0x20]'),
            0x35FEEE: ('mov', 'r14, qword ptr [rsi + 0x20]'), 0x35FF07: ('mov', 'rdi, qword ptr [r14 + 0x100]'),
            0x35FF13: ('mov', 'r8, qword ptr [rip + 0x23ae5f6]'), 0x35FF1A: ('xor', 'r9d, r9d'),
            0x35FF23: ('mov', 'rbx, rax'), 0x35FF2B: ('xor', 'r8d, r8d'),
            0x35FF40: ('xor', 'edx, edx'), 0x35FF4C: ('cmp', 'qword ptr [rax], rcx'),
            0x35FF5F: ('mov', 'qword ptr [r14 + 0x100], rdx'),
            0x35FD74: ('mov', 'rbx, qword ptr [rsi + 0x40]'), 0x35FD89: ('xor', 'edx, edx'),
            0x35FD8E: ('call', '0x5044d0'), 0x35FD98: ('call', '0x1c7f4a0'),
            0x35FDA4: ('mov', 'rbx, qword ptr [rsi + 0x20]'), 0x35FDD0: ('mov', 'r14, qword ptr [rsi + 0x20]'),
            0x35FDE9: ('mov', 'rdi, qword ptr [r14 + 0x100]'), 0x35FDFC: ('xor', 'r9d, r9d'),
            0x35FE05: ('mov', 'rbx, rax'), 0x35FE0D: ('xor', 'r8d, r8d'),
            0x35FE22: ('xor', 'edx, edx'), 0x35FE2E: ('cmp', 'qword ptr [rax], rcx'),
            0x35FE41: ('mov', 'qword ptr [r14 + 0x100], rdx')}
        for method, sites in SITES.items():
            for kind, target in [('allocate', 0x2B7D40), ('constructor', 0x4D5170),
                ('delegate', 0x116BCC0 if method == 'OnEnable' else 0x116E070), ('barrier', 0x2B6FF0), ('null', 0x2B7D90), ('cast', 0x2B7040)]:
                self.checks[sites[kind]] = ('call', hex(target))
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == value for a, value in self.checks.items())
        refs = set()
        for i in self.instructions.values():
            for o in i.operands:
                if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP:
                    slot = i.address + i.size + o.mem.disp; refs.add(slot)
                    if i.mnemonic == 'cmp' and o.size == 1: self.flags[slot] = 0
        assert set(self.flags) == {0x288C138, 0x288C139}
        self.method_flags = {'OnEnable': 0x288C138, 'OnDisable': 0x288C139}
        names = ['owner', 'owner_class', 'character0', 'character1', 'character_class', 'animation0', 'animation1',
                 'action_class', 'foreign_class', 'object_class', 'tween_class', 'click_method', 'prior_owner', 'prior_method', 'old_click', 'other_click']
        self.fixed_names = names.copy(); names += ['action' + str(i) for i in range(24)]
        self.p = {n: self.arena + 0x210000 + i * 0x1000 for i, n in enumerate(names)}
        self.labels = {0: None, **{p: n for n, p in self.p.items()}}
        self.sizes = {n: 0x200 if n in ['character0', 'character1'] else 0x100 if n.endswith('_class') else 0x80 for n in names}
        keys = {'System.Action_TypeInfo': 'action_class', 'UnityEngine.Object_TypeInfo': 'object_class',
                'DG.Tweening.DOTween_TypeInfo': 'tween_class', 'Method$CardInteraction.Click()': 'click_method'}
        self.bindings, self.metadata_slots, self.literal_slots = {}, {}, {}
        for r in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if r['Address'] not in refs: continue
            if r['Name'].startswith('Method$'): assert r['MethodAddress'] == 0x35F4B0
            key = keys[r['Name']]; self.bindings[r['Name']] = self.p[key]; self.metadata_slots[r['Address']] = self.p[key]
            self.q(self.base + r['Address'], self.p[key])
        assert set(self.bindings) == set(keys)
        self.metadata_order = {}
        for name, (start, end, _, _) in TARGETS.items():
            ins = [i for a, i in sorted(self.instructions.items()) if start <= a < end]; order = []
            for previous, i in zip(ins, ins[1:]):
                if i.mnemonic == 'call' and i.op_str == '0x2b7b40':
                    assert previous.mnemonic == 'lea'
                    slot = previous.address + previous.size + previous.operands[1].mem.disp
                    assert slot in self.metadata_slots; order.append((slot, i.address + i.size))
            assert len(order) == (3 if name == 'OnEnable' else 4); self.metadata_order[name] = order
        self.service_targets = []
        for address, name in SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1; self.service_targets += rows
        signatures = ['void CardInteraction__MouseExit (CardInteraction_o* __this, const MethodInfo* method);',
            'int32_t DG_Tweening_DOTween__Kill (Il2CppObject* targetOrId, bool complete, const MethodInfo* method);',
            'void UnityEngine_MonoBehaviour__StopAllCoroutines (UnityEngine_MonoBehaviour_o* __this, const MethodInfo* method);',
            'bool UnityEngine_Object__op_Inequality (UnityEngine_Object_o* x, UnityEngine_Object_o* y, const MethodInfo* method);',
            'void System_Action___ctor (System_Action_o* __this, Il2CppObject* object, intptr_t method, const MethodInfo* method);',
            'System_Delegate_o* System_Delegate__Combine (System_Delegate_o* a, System_Delegate_o* b, const MethodInfo* method);',
            'System_Delegate_o* System_Delegate__Remove (System_Delegate_o* source, System_Delegate_o* value, const MethodInfo* method);']
        assert [r['Signature'] for r in self.service_targets] == signatures

    def snapshot(self):
        return {'fields': {'character': self.oid(self.rq(self.p['owner'] + 0x20)), 'animation': self.oid(self.rq(self.p['owner'] + 0x40))},
                'click_channels': {n: self.oid(self.rq(self.p[n] + 0x100)) for n in ['character0', 'character1']},
                'initialized_words': {n: self.rd(self.p[n] + 0xE0) for n in ['object_class', 'tween_class']},
                'metadata_flags': {hex(a): self.u.mem_read(self.base + a, 1)[0] for a in sorted(self.flags)},
                'metadata_slots': {hex(a): self.oid(self.rq(self.base + a)) for a in sorted(self.metadata_slots)},
                'delegates': {n: {'phase': self.phases[n], 'entries': deepcopy(e)} for n, e in self.delegates.items()},
                'allocation_order': self.allocations.copy(), 'operations': deepcopy(self.operations),
                'mouse_exit_requests': self.mouse_exits.copy(), 'kill_requests': deepcopy(self.kills), 'coroutine_stop_requests': self.coro_stops.copy(),
                'native_entries': deepcopy(self.entries), 'memory': {n: bytes(self.u.mem_read(self.p[n], self.sizes[n])).hex() for n in self.fixed_names + self.allocations}}

    def prepare(self, options):
        self.options = deepcopy(options); self.events, self.counts, self.error = [], {}, None
        self.entries, self.mouse_exits, self.kills, self.coro_stops = [], [], [], []
        self.delegates, self.phases, self.allocations, self.operations, self.allowed = {}, {}, [], [], {}
        for n in self.p: self.u.mem_write(self.p[n], b'\xA5' * self.sizes[n])
        for n in self.fixed_names:
            if n.endswith('_class'): self.u.mem_write(self.p[n], bytes(self.sizes[n]))
        self.q(self.p['owner'], self.p['owner_class']); self.q(self.p['owner'] + 8, 0)
        self.q(self.p['owner'] + 0x20, 0 if options.get('null_character') else self.p['character0'])
        self.q(self.p['owner'] + 0x40, 0 if options.get('null_animation') else self.p['animation0'])
        for n in ['character0', 'character1']: self.q(self.p[n], self.p['character_class'])
        for n in ['object_class', 'tween_class']: self.d(self.p[n] + 0xE0, options.get(n + '_word', 1))
        own = ['owner', 'click_method']; mode = options.get('old_entries', 'prior')
        entries = [] if mode == 'empty' else [own] if mode == 'own' else [own, own] if mode == 'duplicate' else [['prior_owner', 'prior_method'], own] if mode == 'mixed' else [['prior_owner', 'prior_method']]
        assert mode in ['empty', 'own', 'duplicate', 'mixed', 'prior']
        for n in ['old_click', 'other_click']: self.set_delegate(n, entries)
        for n, key in [('character0', 'old_click'), ('character1', 'other_click')]: self.q(self.p[n] + 0x100, 0 if options.get('null_channel') else self.p['old_click' if options.get('alias_channels') else key])
        for a in self.flags: self.u.mem_write(self.base + a, bytes([0 if options.get('cold') else options.get('warm_flag', 1)]))

    def set_delegate(self, name, entries, foreign=False, allocated=False):
        p = self.p[name]; self.u.mem_write(p, bytes(self.sizes[name])); self.q(p, self.p['foreign_class' if foreign else 'action_class'])
        self.delegates[name] = deepcopy(entries); self.phases[name] = 'allocated' if allocated else 'supplied_ready'
        if entries:
            owner, token = entries[-1]
            for offset, value in [(0x20, self.p[owner]), (0x28, self.p[token]), (0x40, self.p[owner])]: self.q(p + offset, value)
        self.allow(name, 0, self.sizes[name]); return p

    def fresh(self, entries=None, foreign=False, allocated=False):
        n = 'action' + str(len(self.allocations)); assert n in self.p
        self.allocations.append(n); return self.set_delegate(n, entries or [], foreign, allocated)

    def mutate(self, kind, relative):
        action = self.options.get('mutations', {}).get(kind + ':' + str(relative))
        if action is None: return
        if action in ['replace_character', 'clear_character', 'replace_animation', 'clear_animation']:
            field = action.split('_')[1]; off = 0x20 if field == 'character' else 0x40
            self.q(self.p['owner'] + off, self.p[field + '1'] if action.startswith('replace') else 0); self.allow('owner', off, 8)
        elif action in ['replace_channel', 'clear_channel']:
            target = self.rq(self.p['owner'] + 0x20); assert target
            self.q(target + 0x100, self.p['other_click'] if action.startswith('replace') else 0); self.allow(self.oid(target), 0x100, 8)
        else: raise AssertionError(action)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        if address == self.stop: return
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == TARGETS[self.method][0]: self.entries.append({'method': self.method, 'raw_args': [cx, dx, r8, r9]})
        if rva == 0x35FD74: self.captured_animation = self.rq(self.p['owner'] + 0x40)
        if rva in [0x35FEC2, 0x35FDA4]: self.captured_validity = self.rq(self.p['owner'] + 0x20)
        if rva in [0x35FF5F, 0x35FE41]: self.allow(self.oid(self.reg(x.UC_X86_REG_R14)), 0x100, 8)
        if rva in self.instructions: return
        if rva == 0x2B7B40:
            assert cx - self.base in self.metadata_slots
            if self.event('metadata', [hex(cx - self.base), self.oid(self.rq(cx))]): self.ret(self.rq(cx))
        elif rva == 0x281D90:
            n = self.oid(cx); assert n in ['object_class', 'tween_class'] and self.rd(cx + 0xE0) == 0
            if self.event('class_init', [n]): self.d(cx + 0xE0, 1); self.allow(n, 0xE0, 4); self.ret()
        elif rva == 0x35F810:
            assert self.method == 'OnEnable' and cx == self.p['owner'] and dx == 0
            if self.event('supplied_MouseExit', ['owner', dx]): self.mouse_exits.append('owner'); self.ret()
        elif rva == 0x5044D0:
            assert self.method == 'OnDisable' and cx == self.captured_animation and dx == 0 and r8 == 0
            bits = self.options.get('kill_return_bits', 0xCAFEBABEFFFFFFFF)
            args = [self.oid(cx), dx, r8, bits]
            if self.event('kill_tween', args): self.kills.append(args); self.ret(bits)
        elif rva == 0x1C7F4A0:
            assert self.method == 'OnDisable' and cx == self.p['owner'] and dx == 0
            if self.event('stop_coroutines', ['owner', dx]): self.coro_stops.append('owner'); self.ret()
        elif rva == 0x1C82480:
            assert cx == self.captured_validity and dx == 0 and r8 == 0
            bits = 0xFEDCBA9876543200 | self.options.get('predicate_byte', 0 if cx == 0 else 1)
            if self.event('predicate', [self.oid(cx), dx, r8, bits]): self.ret(bits)
        elif rva == 0x2B7D40:
            assert cx == self.p['action_class']
            character = self.reg(x.UC_X86_REG_R14); assert character == self.rq(self.p['owner'] + 0x20) and character
            self.captured_target, self.captured_old = character, self.rq(character + 0x100)
            if self.event('allocate', ['action_class']): self.last_new = self.fresh(allocated=True); self.ret(self.last_new)
        elif rva == 0x4D5170:
            assert cx == self.last_new and dx == self.p['owner'] and r8 == self.p['click_method'] and r9 == 0
            assert self.phases[self.oid(cx)] == 'allocated'
            if self.event('constructor', [self.oid(cx), 'owner', 'click_method', r9]):
                n = self.oid(cx); self.delegates[n] = [['owner', 'click_method']]; self.phases[n] = 'supplied_constructed'
                for off, value in [(0x20, dx), (0x28, r8), (0x40, dx)]: self.q(cx + off, value); self.allow(n, off, 8)
                self.ret()
        elif rva in [0x116BCC0, 0x116E070]:
            assert cx == self.captured_old and dx == self.last_new and r8 == 0 and r9 == POISON[3]
            operation = 'Combine' if self.method == 'OnEnable' else 'Remove'
            assert rva == (0x116BCC0 if operation == 'Combine' else 0x116E070)
            if self.event('delegate', [operation, self.oid(cx), self.oid(dx), r8, r9]):
                before = deepcopy(self.delegates[self.oid(cx)]) if cx else []; own = deepcopy(self.delegates[self.oid(dx)])
                if operation == 'Combine': result = self.fresh(before + own) if cx else dx
                else:
                    last = next((i for i in range(len(before) - len(own), -1, -1) if before[i:i + len(own)] == own), None)
                    rest = before if last is None else before[:last] + before[last + len(own):]
                    result = cx if last is None else self.fresh(rest) if rest else 0
                mode = self.options.get('result_mode', 'normal')
                if mode == 'null': result = 0
                elif mode == 'source': result = cx
                elif mode == 'new': result = dx
                elif mode == 'other': result = self.p['other_click']
                elif mode == 'foreign': result = self.fresh(foreign=True)
                assert mode in ['normal', 'null', 'source', 'new', 'other', 'foreign']
                self.operations.append({'kind': operation, 'old': self.oid(cx), 'new': self.oid(dx), 'result': self.oid(result)}); self.ret(result)
        elif rva == 0x2B6FF0:
            result = self.p.get(self.operations[-1]['result'], 0)
            assert cx == self.captured_target + 0x100 and dx == result and self.rq(cx) == result and r8 == POISON[2] and r9 == POISON[3]
            if self.event('barrier', [self.oid(self.captured_target), self.oid(dx)]): self.ret()
        elif rva in [0x2B7D90, 0x2B7040]:
            kind = 'native_null_guard' if rva == 0x2B7D90 else 'native_cast_failure'
            args = [] if kind == 'native_null_guard' else [self.oid(cx), self.oid(dx)]
            if kind == 'native_cast_failure': assert self.rq(cx) == self.p['foreign_class'] and dx == self.p['action_class']
            self.event(kind, args); self.error = kind; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    # Awake's event wrapper already records exact ABI/caller/cumulative ordinals
    # and runs only reached mutations after an entry snapshot and before effects.
    def run(self, method, options=None, retained=False):
        if not retained: self.prepare(options or {})
        elif options is not None: self.options.update(deepcopy(options))
        self.method, self.error, self.allowed = method, None, {}
        self.entry_counts = self.counts.copy(); initial, old = self.snapshot(), len(self.events)
        x, sp = self.x, self.stack + 0x18008; self.q(sp, self.stop)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFAB0000000000000 + i)
        for i in range(6, 16): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), (0xABCDEF9876543210 << 64) | i)
        incoming = [self.p['owner'], *INCOMING[1:]]
        for n, value in zip(['RCX', 'RDX', 'R8', 'R9'], incoming): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), value)
        self.u.reg_write(x.UC_X86_REG_RSP, sp); self.u.emu_start(self.base + TARGETS[method][0], self.stop, timeout=10_000_000, count=10000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): assert self.reg(getattr(x, 'UC_X86_REG_' + n)) == 0xFAB0000000000000 + i
            for i in range(6, 16): assert self.reg(getattr(x, 'UC_X86_REG_XMM' + str(i))) == (0xABCDEF9876543210 << 64) | i
        final = self.snapshot()
        for n, before in initial['memory'].items():
            before, after = bytes.fromhex(before), bytes.fromhex(final['memory'][n])
            assert all(i in self.allowed.get(n, set()) or byte == after[i] for i, byte in enumerate(before)), n
        for i in range(len(self.allocations), 24):
            n = 'action' + str(i); assert bytes(self.u.mem_read(self.p[n], self.sizes[n])) == b'\xA5' * self.sizes[n]
        row = {'method': method, 'options': deepcopy(self.options), 'entry_raw_args': incoming,
               'returned': returned, 'error': self.error, 'initial': initial, 'events': deepcopy(self.events[old:]), 'final': final,
               'service_counts_before': self.entry_counts.copy(), 'service_counts_after': self.counts.copy(),
               'unrelated_diagnostic_storage_retained': True, 'unallocated_arena_retained': True, 'nonvolatile_abi_verified': returned}
        self.verify(row); row['independent_ordered_model_verified'] = True; return row

    def verify(self, row):
        model, options, counts = deepcopy(row['initial']), row['options'], row['service_counts_before'].copy()
        raw = row['entry_raw_args'].copy(); index, returned, error = 0, False, None
        method = row['method']; model['native_entries'].append({'method': method, 'raw_args': raw.copy()})
        class Stopped(Exception): pass
        def pointer(n): return self.p[n] if n else 0
        def write(n, off, width, value):
            data = bytearray.fromhex(model['memory'][n]); data[off:off + width] = value.to_bytes(width, 'little'); model['memory'][n] = data.hex()
        def mutation(kind, relative):
            action = options.get('mutations', {}).get(kind + ':' + str(relative))
            if action is None: return
            if action in ['replace_character', 'clear_character', 'replace_animation', 'clear_animation']:
                field = action.split('_')[1]; value = field + '1' if action.startswith('replace') else None
                model['fields'][field] = value; write('owner', 0x20 if field == 'character' else 0x40, 8, pointer(value))
            else:
                target = model['fields']['character']; assert target
                value = 'other_click' if action.startswith('replace') else None
                model['click_channels'][target] = value; write(target, 0x100, 8, pointer(value))
        def emit(kind, args, registers, caller, terminal=False):
            nonlocal index, raw, error
            ordinal = counts.get(kind, 0) + 1; counts[kind] = ordinal
            abi = {n + '_bits': value for n, value in zip(['rcx', 'rdx', 'r8', 'r9'], registers)}
            abi['caller_return'] = self.caller_return(self.base + caller)
            assert row['events'][index] == {'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': abi, 'snapshot': model}, (method, kind, index)
            index += 1; relative = ordinal - row['service_counts_before'].get(kind, 0)
            if terminal or options.get('failure') == [kind, relative]: error = kind; raise Stopped()
            mutation(kind, relative); raw = POISON.copy()
        def init(n, caller):
            if model['initialized_words'][n] == 0:
                emit('class_init', [n], [self.p[n], *raw[1:]], caller)
                model['initialized_words'][n] = 1; write(n, 0xE0, 4, 1)
        def fresh(entries=None, foreign=False, allocated=False):
            n = 'action' + str(len(model['allocation_order'])); assert n in self.p
            model['allocation_order'].append(n); model['memory'][n] = bytes(self.sizes[n]).hex()
            write(n, 0, 8, self.p['foreign_class' if foreign else 'action_class']); entries = entries or []
            model['delegates'][n] = {'phase': 'allocated' if allocated else 'supplied_ready', 'entries': deepcopy(entries)}
            if entries:
                owner, token = entries[-1]
                for off, value in [(0x20, pointer(owner)), (0x28, pointer(token)), (0x40, pointer(owner))]: write(n, off, 8, value)
            return n
        try:
            flag = hex(self.method_flags[method])
            if model['metadata_flags'][flag] == 0:
                for slot, caller in self.metadata_order[method]: emit('metadata', [hex(slot), model['metadata_slots'][hex(slot)]], [self.base + slot, *raw[1:]], caller)
                model['metadata_flags'][flag] = 1
            if method == 'OnEnable':
                emit('supplied_MouseExit', ['owner', 0], [self.p['owner'], 0, *raw[2:]], 0x35FEBB)
                model['mouse_exit_requests'].append('owner')
            else:
                animation = model['fields']['animation']; init('tween_class', 0x35FD86)
                bits = options.get('kill_return_bits', 0xCAFEBABEFFFFFFFF); args = [animation, 0, 0, bits]
                emit('kill_tween', args, [pointer(animation), 0, 0, raw[3]], 0x35FD93); model['kill_requests'].append(args)
                emit('stop_coroutines', ['owner', 0], [self.p['owner'], 0, *raw[2:]], 0x35FD9D); model['coroutine_stop_requests'].append('owner')
            captured = model['fields']['character']; init('object_class', 0x35FED4 if method == 'OnEnable' else 0x35FDB6)
            low = options.get('predicate_byte', 0 if captured is None else 1); bits = 0xFEDCBA9876543200 | low
            emit('predicate', [captured, 0, 0, bits], [pointer(captured), 0, 0, raw[3]], 0x35FEE1 if method == 'OnEnable' else 0x35FDC3)
            if low:
                target = model['fields']['character']
                if target is None: emit('native_null_guard', [], raw, SITES[method]['null'] + 5, True)
                old = model['click_channels'][target]
                emit('allocate', ['action_class'], [self.p['action_class'], *raw[1:]], SITES[method]['allocate'] + 5); new = fresh(allocated=True)
                emit('constructor', [new, 'owner', 'click_method', 0], [pointer(new), self.p['owner'], self.p['click_method'], 0], SITES[method]['constructor'] + 5)
                model['delegates'][new] = {'phase': 'supplied_constructed', 'entries': [['owner', 'click_method']]}
                for off, value in [(0x20, self.p['owner']), (0x28, self.p['click_method']), (0x40, self.p['owner'])]: write(new, off, 8, value)
                operation = 'Combine' if method == 'OnEnable' else 'Remove'
                emit('delegate', [operation, old, new, 0, POISON[3]], [pointer(old), pointer(new), 0, POISON[3]], SITES[method]['delegate'] + 5)
                before = deepcopy(model['delegates'][old]['entries']) if old else []; own = deepcopy(model['delegates'][new]['entries'])
                if operation == 'Combine': result = fresh(before + own) if old else new
                else:
                    last = next((i for i in range(len(before) - len(own), -1, -1) if before[i:i + len(own)] == own), None)
                    rest = before if last is None else before[:last] + before[last + len(own):]
                    result = old if last is None else fresh(rest) if rest else None
                mode = options.get('result_mode', 'normal')
                if mode == 'null': result = None
                elif mode == 'source': result = old
                elif mode == 'new': result = new
                elif mode == 'other': result = 'other_click'
                elif mode == 'foreign': result = fresh(foreign=True)
                model['operations'].append({'kind': operation, 'old': old, 'new': new, 'result': result})
                if result and int.from_bytes(bytes.fromhex(model['memory'][result])[:8], 'little') != self.p['action_class']:
                    emit('native_cast_failure', [result, 'action_class'], [pointer(result), self.p['action_class'], POISON[2], POISON[3]], SITES[method]['cast'] + 5, True)
                model['click_channels'][target] = result; write(target, 0x100, 8, pointer(result))
                emit('barrier', [target, result], [pointer(target) + 0x100, pointer(result), POISON[2], POISON[3]], SITES[method]['barrier'] + 5)
            returned = True
        except Stopped: pass
        assert index == len(row['events']) and (returned, error) == (row['returned'], row['error'])
        assert model == row['final'] and counts == row['service_counts_after']


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for method in TARGETS:
        for cold, object_word, tween_word, low in itertools.product([False, True], [0, 1], [0, 1], [0, 1, 0x80, 0xFF]):
            cases.append(m.run(method, {'cold': cold, 'object_class_word': object_word, 'tween_class_word': tween_word, 'predicate_byte': low}))
        for mode in ['empty', 'own', 'duplicate', 'mixed', 'prior']: cases.append(m.run(method, {'old_entries': mode}))
        for mode in ['null', 'source', 'new', 'other', 'foreign']: cases.append(m.run(method, {'result_mode': mode}))
        for option in ['null_character', 'null_animation', 'null_channel', 'alias_channels']: cases.append(m.run(method, {option: True}))
        cases.append(m.run(method, {'null_character': True, 'predicate_byte': 1}))
        for flag in [0x80, 0xFF]: cases.append(m.run(method, {'warm_flag': flag, 'object_class_word': 0x80000000, 'tween_class_word': 0x80000000}))
        if method == 'OnDisable':
            for bits in [0, 1, 0x1234567880000000]: cases.append(m.run(method, {'kill_return_bits': bits}))
        mutations = [('metadata:1', 'replace_character'), ('predicate:1', 'replace_character'), ('predicate:1', 'clear_character'),
                     ('allocate:1', 'replace_character'), ('constructor:1', 'replace_channel'), ('delegate:1', 'replace_character'),
                     ('delegate:1', 'clear_character'), ('barrier:1', 'replace_character'), ('class_init:1', 'replace_character'),
                     ('class_init:1', 'clear_character')]
        if method == 'OnEnable': mutations += [('supplied_MouseExit:1', 'replace_character'), ('supplied_MouseExit:1', 'clear_character')]
        else: mutations += [('class_init:1', 'replace_animation'), ('class_init:1', 'clear_animation'),
                            ('kill_tween:1', 'replace_character'), ('stop_coroutines:1', 'replace_character'),
                            ('stop_coroutines:1', 'clear_character'), ('class_init:2', 'replace_character'), ('class_init:2', 'clear_character')]
        for phase, action in mutations: cases.append(m.run(method, {'cold': True, 'object_class_word': 0, 'tween_class_word': 0, 'mutations': {phase: action}}))
    for options in [{'cold': True, 'object_class_word': 0, 'tween_class_word': 0}, {'null_channel': True},
                    {'old_entries': 'duplicate'}, {'mutations': {'allocate:1': 'replace_character', 'kill_tween:1': 'replace_animation'}}]:
        m.prepare(options); rows = [m.run(method, retained=True) for method in ['OnEnable', 'OnDisable', 'OnEnable', 'OnDisable']]
        assert all(r['returned'] for r in rows) and all(b['initial'] == a['final'] for a, b in zip(rows, rows[1:])); sequences.append(rows)
    for method, options in itertools.product(TARGETS, [{'cold': True, 'object_class_word': 0, 'tween_class_word': 0},
                                                      {'predicate_byte': 0}, {'result_mode': 'foreign'},
                                                      {'mutations': {'predicate:1': 'clear_character'}},
                                                      {'mutations': {'allocate:1': 'replace_character'}}]):
        baseline = m.run(method, options); bid = len(baselines); baselines.append(baseline); counts = {}
        for i, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(method, dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:i + 1] and stopped['final'] == event['snapshot']
            stops.append({'baseline': bid, 'prefix_length': i + 1, 'stopped': stopped})
    missing = m.body_addresses - m.executed; traps = {a for a, i in m.instructions.items() if i.mnemonic == 'int3'}
    assert missing == traps and len(traps) == 4
    return {'build': BUILD, 'targets': m.targets, 'ranges': m.ranges, 'exception_bounds': m.eh, 'supplied_targets': m.service_targets,
            'instruction_assertions': len(m.checks), 'metadata_bindings': {n: m.oid(p) for n, p in m.bindings.items()},
            'case_count': len(cases), 'cases': cases, 'retained_sequences': sequences, 'failure_baselines': baselines,
            'failure_case_count': len(stops), 'failure_stops': stops, 'body_instructions_decoded': len(m.body_addresses),
            'body_instructions_executed': len(m.body_addresses & m.executed), 'native_execution_addresses': len(m.executed),
            'unexecuted_terminal_traps': [hex(a) for a in sorted(traps)],
            'scope': 'Complete CardInteraction OnEnable/OnDisable native callers, including six chained chunks each. Whole MouseExit, DOTween.Kill, StopAllCoroutines, Object validity/initialization, metadata, allocation, Action construction, Combine/Remove and barriers remain supplied; callback/CLR/scene/runtime admission, scheduler and exceptions/unwinding excluded. Folded constructor unpromoted.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('game_root'); p.add_argument('dumper_root'); p.add_argument('--output', required=True)
    args = p.parse_args(); full = audit(args.game_root, args.dumper_root)
    memory = pool_memory(full); assert expand_memory(memory) == full
    report = pool_snapshots(memory); assert expand_snapshots(report) == memory and expand_memory(expand_snapshots(report)) == full
    Path(args.output).write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
