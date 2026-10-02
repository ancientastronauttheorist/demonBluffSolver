"""Execute DeckCharacter.Init/OnDisable with explicit supplied delegate services."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_deck_character_surface import Machine as SurfaceMachine, OWNER_FIELDS
from audit_report_snapshots import expand_snapshots, pool_snapshots


TARGETS = {'Init': (0x36FA20, 0x36FBF8, 0x36FC00, 'tdi5514.m0001'),
           'OnDisable': (0x36FC00, 0x36FDB5, 0x36FDC0, 'tdi5514.m0002')}
SERVICES = {0x4D5170: 'System.Action$$.ctor', 0x116BCC0: 'System.Delegate$$Combine',
            0x116E070: 'System.Delegate$$Remove', 0x365640: 'Character$$InitReward'}


class Machine(SurfaceMachine):
    def __init__(self, game_root, dumper_root):
        # Reuse immutable build/declaration/signature pins, complete-range decode,
        # explicit memory runtime and volatile return poisoning from the frozen audit.
        super().__init__(game_root, dumper_root)
        import capstone
        import hashlib
        import re
        raw = (Path(dumper_root) / 'dump.cs').read_bytes()
        manifest = json.loads((Path(__file__).parents[1] /
            f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(raw).hexdigest().upper() == manifest['outputs']['dump_cs']['sha256'].upper()
        dump = raw.decode('utf-8-sig')
        def block(pattern):
            found = re.search(pattern, dump, re.M | re.S); assert found
            return found[1]
        interaction = block(r'^public class CardInteraction : MonoBehaviour // TypeDefIndex: 5468\s*\{(.*?)\n\}')
        assert 'public Action onHover; // 0x60' in interaction and 'public Action onHoverExit; // 0x68' in interaction
        action = block(r'^public sealed class Action : MulticastDelegate // TypeDefIndex: 153\s*\{(.*?)\n\}')
        assert 'public virtual void Invoke()' in action
        multicast = block(r'^public abstract class MulticastDelegate : Delegate // TypeDefIndex: 440\s*\{(.*?)\n\}')
        assert 'private Delegate[] delegates; // 0x78' in multicast
        self.targets, self.instructions, self.body_addresses, self.ranges, self.flags = [], {}, set(), {}, {}
        references = set()
        for name, (start, end, following, stable) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start and r['Name'] == 'DeckCharacter$$' + name]
            assert len(rows) == 1
            middle = 'CharacterData_o* data, ' if name == 'Init' else ''
            assert rows[0]['Signature'] == f'void DeckCharacter__{name} (DeckCharacter_o* __this, {middle}const MethodInfo* method);'
            assert rows[0]['TypeSignature'] == ('viii' if name == 'Init' else 'vii')
            self.targets.append(dict(rows[0], stable_method_id=stable))
            self.decode(name, start, end, following)
        for i in self.instructions.values():
            for o in i.operands:
                if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP:
                    slot = i.address + i.size + o.mem.disp; references.add(slot)
                    if i.mnemonic == 'cmp' and o.size == 1: self.flags[slot] = 0
        names = ['owner', 'owner_class', 'interaction0', 'interaction1', 'interaction_class',
                 'character0', 'character1', 'character_class', 'data0', 'data1', 'input_data',
                 'reveal', 'instance_action', 'action_class', 'foreign_class',
                 'hover_method', 'exit_method', 'prior_method', 'prior_owner',
                 'old_hover', 'old_exit', 'other_hover', 'other_exit'] + ['action' + str(i) for i in range(32)]
        self.p = {name: self.arena + 0x90000 + index * 0x1000 for index, name in enumerate(names)}
        self.labels = {0: None, **{p: n for n, p in self.p.items()}}
        self.sizes = {name: 0x100 if name.endswith('_class') else 0x80 for name in names}
        self.bindings, self.metadata_slots, self.literal_slots = {}, {}, {}
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] not in references: continue
            name = row['Name']
            key = {'System.Action_TypeInfo': 'action_class', 'Method$DeckCharacter.OnHover()': 'hover_method',
                   'Method$DeckCharacter.OnHoverExit()': 'exit_method'}[name]
            if name.startswith('Method$'):
                assert row['MethodAddress'] == (0x36FE10 if key == 'hover_method' else 0x36FDC0)
            self.bindings[name] = self.p[key]; self.metadata_slots[row['Address']] = self.p[key]
            self.q(self.base + row['Address'], self.p[key])
        assert len(self.bindings) == 3 and len(self.flags) == 2
        assert set(self.flags) == {0x288C1E2, 0x288C1E3}
        self.method_flags = {'Init': 0x288C1E2, 'OnDisable': 0x288C1E3}
        self.service_targets = []
        for address, name in SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1; self.service_targets += rows
        signatures = {
            'System.Action$$.ctor': 'void System_Action___ctor (System_Action_o* __this, Il2CppObject* object, intptr_t method, const MethodInfo* method);',
            'System.Delegate$$Combine': 'System_Delegate_o* System_Delegate__Combine (System_Delegate_o* a, System_Delegate_o* b, const MethodInfo* method);',
            'System.Delegate$$Remove': 'System_Delegate_o* System_Delegate__Remove (System_Delegate_o* source, System_Delegate_o* value, const MethodInfo* method);',
            'Character$$InitReward': 'void Character__InitReward (Character_o* __this, CharacterData_o* character, const MethodInfo* method);',
        }
        assert all(r['Signature'] == signatures[r['Name']] for r in self.service_targets)
        self.stores = {0x36FAC6: ('interaction', 0x60), 0x36FADC: ('interaction', 0x60),
                       0x36FB52: ('interaction', 0x68), 0x36FB67: ('interaction', 0x68),
                       0x36FB91: ('owner', 0x40), 0x36FCA4: ('interaction', 0x60),
                       0x36FCBA: ('interaction', 0x60), 0x36FD30: ('interaction', 0x68),
                       0x36FD45: ('interaction', 0x68)}
        self.checks = {
            0x36FA31: ('mov', 'r15, rdx'), 0x36FA64: ('mov', 'r14, qword ptr [rbp + 0x38]'),
            0x36FA87: ('mov', 'rdi, qword ptr [r14 + 0x60]'), 0x36FA97: ('xor', 'r9d, r9d'),
            0x36FAAB: ('mov', 'rdx, rbx'), 0x36FACC: ('cmp', 'qword ptr [rax], rcx'),
            0x36FAC6: ('mov', 'qword ptr [r14 + 0x60], rsi'),
            0x36FADC: ('mov', 'qword ptr [r14 + 0x60], rdx'),
            0x36FB03: ('mov', 'r14, qword ptr [rbp + 0x38]'),
            0x36FB17: ('mov', 'rdi, qword ptr [r14 + 0x68]'),
            0x36FB27: ('xor', 'r9d, r9d'), 0x36FB38: ('xor', 'r8d, r8d'),
            0x36FB52: ('mov', 'qword ptr [r14 + 0x68], rsi'),
            0x36FB67: ('mov', 'qword ptr [r14 + 0x68], rdx'),
            0x36FB91: ('mov', 'qword ptr [rcx], r15'),
            0x36FB99: ('mov', 'rcx, qword ptr [rbp + 0x28]'),
            0x36FBA2: ('xor', 'r8d, r8d'), 0x36FBC0: ('jmp', '0x365640'),
            0x36FC42: ('mov', 'r14, qword ptr [rbp + 0x38]'),
            0x36FC65: ('mov', 'rdi, qword ptr [r14 + 0x60]'),
            0x36FC75: ('xor', 'r9d, r9d'), 0x36FC86: ('xor', 'r8d, r8d'),
            0x36FCA4: ('mov', 'qword ptr [r14 + 0x60], rsi'),
            0x36FCBA: ('mov', 'qword ptr [r14 + 0x60], rdx'),
            0x36FCE1: ('mov', 'r14, qword ptr [rbp + 0x38]'),
            0x36FCF5: ('mov', 'rdi, qword ptr [r14 + 0x68]'),
            0x36FD05: ('xor', 'r9d, r9d'), 0x36FD16: ('xor', 'r8d, r8d'),
            0x36FD30: ('mov', 'qword ptr [r14 + 0x68], rsi'),
            0x36FD45: ('mov', 'qword ptr [r14 + 0x68], rdx'),
            0x36FD7D: ('jmp', '0x2b6ff0'),
        }
        for address, expected in self.checks.items():
            assert (self.instructions[address].mnemonic, self.instructions[address].op_str) == expected
        for name, (start, end, _, _) in TARGETS.items():
            calls = [i.op_str for i in self.instructions.values() if start <= i.address < end and i.mnemonic == 'call']
            assert calls.count('0x2b7b40') == 3 and calls.count('0x2b7d40') == 2 and calls.count('0x4d5170') == 2
            assert calls.count('0x116bcc0' if name == 'Init' else '0x116e070') == 2
            assert calls.count('0x2b6ff0') == (3 if name == 'Init' else 1)

    def snapshot(self):
        return {'owner_fields': {n: self.oid(self.rq(self.p['owner'] + o)) for n, o in OWNER_FIELDS},
                'channels': {n: {field: self.oid(self.rq(self.p[n] + o)) for field, o in [('hover', 0x60), ('exit', 0x68)]}
                             for n in ['interaction0', 'interaction1']},
                'delegates': {n: {'phase': self.phases[n], 'supplied_entries': [entry.copy() for entry in entries]}
                              for n, entries in self.delegates.items()},
                'allocation_order': self.allocations.copy(), 'operations': self.operations.copy(),
                'reward_requests': self.rewards.copy(),
                'metadata_flags': {hex(a): self.u.mem_read(self.base + a, 1)[0] for a in sorted(self.flags)},
                'metadata_slots': {hex(a): self.oid(self.rq(self.base + a)) for a in sorted(self.metadata_slots)},
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options = options.copy()
        self.events, self.counts, self.error, self.delegates, self.phases = [], {}, None, {}, {}
        self.allocations, self.operations, self.rewards, self.allowed = [], [], [], {}
        for n, p in self.p.items(): self.u.mem_write(p, bytes([0xA5]) * self.sizes[n])
        self.q(self.p['owner'], self.p['owner_class']); self.q(self.p['owner'] + 8, 0)
        for n, o in OWNER_FIELDS:
            key = {'on_click': 'instance_action', 'character': 'character0', 'reveal': 'reveal',
                   'interaction': 'interaction0', 'data': 'data0'}[n]
            self.q(self.p['owner'] + o, 0 if options.get('null_' + n) else self.p[key])
        self.input_data = 0 if options.get('null_input_data') else self.p['input_data']
        for name in ['interaction0', 'interaction1']:
            self.q(self.p[name], self.p['interaction_class'])
        for key, channel in [('old_hover', 'hover_method'), ('old_exit', 'exit_method'),
                             ('other_hover', 'hover_method'), ('other_exit', 'exit_method')]:
            mode = options.get('old_entries', 'prior')
            own = ['owner', channel]
            entries = [] if mode == 'empty' else [own] if mode == 'own' else [own, own] if mode == 'duplicate' else [['prior_owner', 'prior_method'], own] if mode == 'mixed' else [['prior_owner', 'prior_method']]
            self.set_delegate(key, entries)
        for prefix, interaction in [('', 'interaction0'), ('other_', 'interaction1')]:
            for field, o in [('hover', 0x60), ('exit', 0x68)]:
                key = prefix + field if prefix else 'old_' + field
                if options.get('alias_channels'): key = 'old_hover'
                self.q(self.p[interaction] + o, 0 if options.get('null_channels') else self.p[key])
        for a in self.flags:
            self.u.mem_write(self.base + a, bytes([0 if options.get('cold') else options.get('warm_flag', 1)]))

    def set_delegate(self, name, entries, foreign=False, allocated=False):
        p = self.p[name]
        self.u.mem_write(p, bytes(self.sizes[name])); self.q(p, self.p['foreign_class' if foreign else 'action_class'])
        self.delegates[name] = [entry.copy() for entry in entries]
        self.phases[name] = 'allocated' if allocated else 'supplied_ready'
        if entries:
            owner, method = entries[-1]
            self.q(p + 0x20, self.p[owner]); self.q(p + 0x28, self.p[method]); self.q(p + 0x40, self.p[owner])
        if hasattr(self, 'allowed'): self.allow(name, 0, self.sizes[name])
        return p

    def fresh(self, entries=None, foreign=False, allocated=False):
        name = 'action' + str(len(self.allocations)); assert name in self.p
        self.allocations.append(name)
        return self.set_delegate(name, entries or [], foreign, allocated)

    def mutate(self, kind, ordinal):
        action = self.options.get('mutations', {}).get(kind + ':' + str(ordinal))
        if action is None: return
        if action in ['replace_interaction', 'clear_interaction', 'replace_character', 'clear_character', 'replace_data', 'clear_data']:
            field = action.split('_')[1]; offset = dict(OWNER_FIELDS)[field]
            self.q(self.p['owner'] + offset, 0 if action.startswith('clear') else self.p[field + '1'])
            self.allow('owner', offset, 8)
        elif action in ['replace_hover', 'clear_hover', 'replace_exit', 'clear_exit']:
            field = action.split('_')[1]; offset = 0x60 if field == 'hover' else 0x68
            pointer = self.rq(self.p['owner'] + 0x38); assert pointer
            self.q(pointer + offset, 0 if action.startswith('clear') else self.p['other_' + field])
            self.allow(self.oid(pointer), offset, 8)
        else: raise AssertionError(action)

    def next_ordinal(self, kind):
        return self.counts.get(kind, 0) - self.entry_counts.get(kind, 0) + 1

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        if address == self.stop: return
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva in self.stores:
            kind, offset = self.stores[rva]
            pointer = self.p['owner'] if kind == 'owner' else self.reg(x.UC_X86_REG_R14)
            self.allow(self.oid(pointer), offset, 8)
        if rva in self.instructions: return
        if rva == 0x2B7B40:
            assert cx - self.base in self.metadata_slots
            if self.event('metadata', [self.oid(self.rq(cx))]): self.ret(self.rq(cx))
        elif rva == 0x2B7D40:
            assert cx == self.p['action_class']
            ordinal = self.next_ordinal('allocate')
            assert ordinal in [1, 2]
            channel = 'hover' if ordinal % 2 else 'exit'
            # Independent capture oracle observes fields before any service mutation.
            interaction = self.rq(self.p['owner'] + 0x38); assert interaction
            self.captured[channel] = (interaction, self.rq(interaction + (0x60 if channel == 'hover' else 0x68)))
            if self.event('allocate', [self.oid(cx), channel]):
                self.last_new = self.fresh(allocated=True); self.ret(self.last_new)
        elif rva == 0x4D5170:
            channel = 'hover' if self.next_ordinal('constructor') == 1 else 'exit'
            assert cx == self.last_new and self.phases[self.oid(cx)] == 'allocated'
            assert dx == self.p['owner'] and r8 == self.p[channel + '_method'] and r9 == 0
            if self.event('constructor', [self.oid(cx), self.oid(dx), self.oid(r8), r9]):
                name = self.oid(cx); self.delegates[name] = [['owner', channel + '_method']]; self.phases[name] = 'supplied_constructed'
                for offset, value in [(0x20, dx), (0x28, r8), (0x40, dx)]: self.q(cx + offset, value); self.allow(name, offset, 8)
                self.ret()
        elif rva in [0x116BCC0, 0x116E070]:
            channel = 'hover' if self.next_ordinal('delegate') == 1 else 'exit'
            assert dx == self.last_new and cx == self.captured[channel][1] and r8 == 0
            assert r9 == 0xFACE123456789003
            kind = 'Combine' if rva == 0x116BCC0 else 'Remove'
            assert kind == ('Combine' if self.method == 'Init' else 'Remove')
            if self.event('delegate', [kind, channel, self.oid(cx), self.oid(dx), r8, r9]):
                before = self.delegates[self.oid(cx)].copy() if cx else []
                own = self.delegates[self.oid(dx)].copy()
                if kind == 'Combine': result = dx if not cx else self.fresh(before + own)
                else:
                    last = next((i for i in range(len(before) - len(own), -1, -1) if before[i:i + len(own)] == own), None)
                    rest = before if last is None else before[:last] + before[last + len(own):]
                    result = cx if last is None else self.fresh(rest) if rest else 0
                mode = self.options.get('result_' + channel, 'normal')
                if mode == 'null': result = 0
                elif mode == 'source': result = cx
                elif mode == 'new': result = dx
                elif mode == 'foreign': result = self.fresh([], foreign=True)
                elif mode == 'other': result = self.p['other_' + channel]
                assert mode in ['normal', 'null', 'source', 'new', 'foreign', 'other']
                self.operations.append({'kind': kind, 'channel': channel, 'old': self.oid(cx), 'new': self.oid(dx), 'result': self.oid(result)})
                self.ret(result)
        elif rva == 0x2B6FF0:
            if cx == self.p['owner'] + 0x40:
                assert self.method == 'Init' and dx == self.input_data and self.rq(cx) == dx
                args = ['owner', 'data', self.oid(dx)]
            else:
                channel = 'hover' if self.next_ordinal('barrier') == 1 else 'exit'
                pointer = self.captured[channel][0]; offset = 0x60 if channel == 'hover' else 0x68
                assert cx == pointer + offset and self.rq(cx) == dx
                assert dx == self.p.get(self.operations[-1]['result'], 0)
                args = [self.oid(pointer), channel, self.oid(dx)]
            if self.event('barrier', args): self.ret()
        elif rva == 0x365640:
            assert self.method == 'Init' and cx == self.rq(self.p['owner'] + 0x28) and cx in [self.p['character0'], self.p['character1']]
            assert dx == self.input_data and r8 == 0
            args = [self.oid(cx), self.oid(dx), r8, r9]
            if self.event('supplied_InitReward', args): self.rewards.append(args); self.ret()
        elif rva in [0x2B7D90, 0x2B7040]:
            kind = 'native_null_guard' if rva == 0x2B7D90 else 'native_cast_failure'
            args = [] if kind == 'native_null_guard' else [self.oid(cx), self.oid(dx)]
            if kind == 'native_cast_failure': assert self.rq(cx) == self.p['foreign_class'] and dx == self.p['action_class']
            self.event(kind, args); self.error = kind; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def run(self, method, options=None, retained=False):
        if not retained: self.prepare(options or {})
        elif options is not None: self.options.update(options)
        self.method, self.error, self.allowed, self.retained, self.captured = method, None, {}, retained, {}
        self.entry_counts = self.counts.copy()
        initial, old = self.snapshot(), len(self.events)
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFAB0000000000000 + i)
        for i in range(6, 16): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), (0xABCDEF9876543210 << 64) | i)
        for n, value in [('RSP', sp), ('RCX', self.p['owner']), ('RDX', self.input_data if method == 'Init' else 0xDEAD123456789ABC),
                         ('R8', 0xDEAD123400000008), ('R9', 0xDEAD123400000009)]: self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), value)
        self.u.emu_start(self.base + TARGETS[method][0], self.stop, timeout=10_000_000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): assert self.reg(getattr(x, 'UC_X86_REG_' + n)) == 0xFAB0000000000000 + i
            for i in range(6, 16): assert self.reg(getattr(x, 'UC_X86_REG_XMM' + str(i))) == (0xABCDEF9876543210 << 64) | i
        final = self.snapshot()
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            assert all(i in self.allowed.get(n, set()) or byte == after[i] for i, byte in enumerate(before)), n
        assert initial['metadata_slots'] == final['metadata_slots']
        completed = self.events[old:] if returned else self.events[old:-1]
        expected_flags = initial['metadata_flags'].copy()
        if sum(e['kind'] == 'metadata' for e in completed) == 3:
            expected_flags[hex(self.method_flags[method])] = 1
        assert final['metadata_flags'] == expected_flags
        assert final['reward_requests'] == initial['reward_requests'] + [e['args'] for e in completed if e['kind'] == 'supplied_InitReward']
        row = {'method': method, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
               'initial': initial, 'events': self.events[old:].copy(), 'final': final,
               'unrelated_diagnostic_storage_retained': True, 'win64_nonvolatile_verified': returned}
        self.verify(row)
        return row

    def verify(self, row):
        if row['options'].get('mutations') or row['options'].get('failure'): return
        kinds = [e['kind'] for e in row['events'] if e['kind'] != 'metadata']
        if row['options'].get('null_interaction'): assert kinds == ['native_null_guard']; return
        for channel in ['hover', 'exit']:
            if row['options'].get('result_' + channel) == 'foreign':
                prefix = [] if channel == 'hover' else ['allocate', 'constructor', 'delegate', 'barrier']
                assert kinds == prefix + ['allocate', 'constructor', 'delegate', 'native_cast_failure']
                return
        prefix = ['allocate', 'constructor', 'delegate', 'barrier'] * 2
        if row['method'] == 'OnDisable': assert row['returned'] and kinds == prefix
        else:
            assert kinds == prefix + ['barrier', 'native_null_guard' if row['options'].get('null_character') else 'supplied_InitReward']
            assert row['returned'] == (not row['options'].get('null_character'))
            assert row['final']['owner_fields']['data'] == self.oid(self.input_data)


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, sequences, baselines, stops = [], [], [], []
    for method, cold, old_entries, null_channels in itertools.product(TARGETS, [False, True], ['prior', 'empty', 'own', 'duplicate', 'mixed'], [False, True]):
        cases.append(m.run(method, {'cold': cold, 'old_entries': old_entries, 'null_channels': null_channels}))
    for method, channel, mode in itertools.product(TARGETS, ['hover', 'exit'], ['null', 'source', 'new', 'foreign', 'other']):
        cases.append(m.run(method, {'cold': True, 'result_' + channel: mode}))
    for method, option in itertools.product(TARGETS, ['null_interaction', 'null_character', 'null_input_data', 'alias_channels']):
        cases.append(m.run(method, {'cold': True, option: True}))
    for method in TARGETS: cases.append(m.run(method, {'warm_flag': 0xFE}))
    mutations = [('metadata:1', 'replace_interaction'), ('allocate:1', 'replace_interaction'),
                 ('constructor:1', 'replace_hover'), ('delegate:1', 'clear_hover'),
                 ('barrier:1', 'replace_interaction'), ('barrier:1', 'clear_interaction'),
                 ('allocate:2', 'replace_interaction'), ('constructor:2', 'replace_exit'),
                 ('delegate:2', 'clear_exit'), ('barrier:2', 'replace_character'),
                 ('barrier:2', 'replace_data')]
    for method, (phase, action) in itertools.product(TARGETS, mutations):
        cases.append(m.run(method, {'cold': True, 'mutations': {phase: action}}))
    for phase, action in [('barrier:3', 'replace_character'), ('barrier:3', 'clear_character'),
                          ('barrier:3', 'replace_data'), ('supplied_InitReward:1', 'replace_data')]:
        cases.append(m.run('Init', {'cold': True, 'mutations': {phase: action}}))
    for alias in [False, True]:
        m.prepare({'cold': True, 'old_entries': 'mixed', 'alias_channels': alias})
        calls = [m.run(method, retained=True) for method in ['Init', 'Init', 'OnDisable', 'OnDisable', 'Init', 'OnDisable']]
        assert all(row['returned'] for row in calls)
        sequences.append({'alias_inputs': alias, 'calls': calls})
    for method, mutation in itertools.product(TARGETS, [None, {'allocate:1': 'replace_interaction'}, {'barrier:1': 'replace_interaction'}]):
        options = {'cold': True}
        if mutation: options['mutations'] = mutation
        baseline = m.run(method, options); assert baseline['returned']
        baseline_id = len(baselines); baselines.append(baseline); counts = {}
        for i, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(method, dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:i + 1]
            assert stopped['final'] == event['snapshot']
            stops.append({'baseline': baseline_id, 'prefix_length': i + 1, 'stopped': stopped})
    missing = m.body_addresses - m.executed
    expected_unexecuted = {
        0x36FBCA: ('int3', ''), 0x36FBCB: ('mov', 'rcx, rax'),
        0x36FBCE: ('call', '0x2b7040'), 0x36FBD3: ('int3', ''),
        0x36FBDF: ('int3', ''), 0x36FBE0: ('mov', 'rdx, rcx'),
        0x36FBE3: ('mov', 'rcx, rax'), 0x36FBE6: ('call', '0x2b7040'),
        0x36FBEB: ('int3', ''), 0x36FBF7: ('int3', ''),
        0x36FD87: ('int3', ''), 0x36FD88: ('mov', 'rcx, rax'),
        0x36FD8B: ('call', '0x2b7040'), 0x36FD90: ('int3', ''),
        0x36FD9C: ('int3', ''), 0x36FD9D: ('mov', 'rdx, rcx'),
        0x36FDA0: ('mov', 'rcx, rax'), 0x36FDA3: ('call', '0x2b7040'),
        0x36FDA8: ('int3', ''), 0x36FDB4: ('int3', ''),
    }
    assert missing == set(expected_unexecuted)
    assert all((m.instructions[a].mnemonic, m.instructions[a].op_str) == expected
               for a, expected in expected_unexecuted.items())
    return {'build': BUILD, 'targets': m.targets, 'ranges': m.ranges, 'supplied_targets': m.service_targets,
            'instruction_assertions': len(m.checks), 'metadata_bindings': sorted(m.bindings),
            'cases': cases, 'case_count': len(cases), 'retained_sequences': sequences,
            'failure_baselines': baselines, 'failure_stops': stops, 'failure_case_count': len(stops),
            'body_instructions_decoded': len(m.body_addresses), 'body_instructions_executed': len(m.body_addresses & m.executed),
            'unexecuted_instructions': [{'rva': hex(a), 'instruction': m.instructions[a].mnemonic + ' ' + m.instructions[a].op_str} for a in sorted(missing)],
            'native_execution_addresses': len(m.executed),
            'scope': 'Two complete native DeckCharacter registration callers. Action allocation/constructor, Combine/Remove, metadata/GC and whole Character.InitReward are explicitly supplied. Nominal Action identity and ordered delegate entries are diagnostic service state, not CLR/event admission or callback implementation evidence. No shared constructor or OnHover/OnHoverExit callback promotion; no scene, scheduler or exception unwinding.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); expanded = audit(args.game_root, args.dumper_root)
    report = pool_snapshots(expanded)
    assert expand_snapshots(report) == expanded
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
