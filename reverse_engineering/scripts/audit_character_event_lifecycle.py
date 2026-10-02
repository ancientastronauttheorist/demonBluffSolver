"""Execute Character.OnEnable/OnDisable with explicitly supplied event services."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

from audit_character_assets import BUILD
from audit_deck_character_surface import Machine as SurfaceMachine
from audit_character_init_reward import Machine as RewardMachine
from audit_character_oracle_reveal_join import expand_memory, pool_memory
from audit_report_snapshots import expand_snapshots, pool_snapshots


TARGETS = {'OnEnable': (0x366E40, 0x3674DE, 0x3674E0, 'tdi5487.m0017'),
           'OnDisable': (0x3667A0, 0x366E3E, 0x366E40, 'tdi5487.m0018')}
# Exact OnEnable instruction boundaries; the paired body is independently decoded.
CHANNELS = [
    ('hover', 'interaction', 0x60, 'ShowAnimatedArt', 0x366F5A, 0x366F72, 0x366F80, 0x366FCC, 0x367469,
     (0x366F94, 0x366FAA), (0x367457, 0x367462)),
    ('exit', 'interaction', 0x68, 'HideAnimatedArt', 0x366FEC, 0x367004, 0x367012, 0x36705E, 0x367451,
     (0x367026, 0x36703C), (0x36743F, 0x36744A)),
    ('game', 'game', 0, 'RefreshCharacter', 0x36707B, 0x367093, 0x3670A1, 0x367114, 0x367439,
     (0x3670C6, 0x3670E9), (0x3674D2, 0x3674DD)),
    ('oracle_show', 'gameplay', 0xB8, 'OracleEyeActive', 0x367135, 0x36714D, 0x36715B, 0x3671DD, 0x3674CC,
     (0x367180, 0x3671A7), (0x3674BA, 0x3674C5)),
    ('oracle_hide', 'gameplay', 0xC0, 'HideOracleInfo', 0x3671FE, 0x367216, 0x367224, 0x3672A6, 0x3674B4,
     (0x367249, 0x367270), (0x3674A2, 0x3674AD)),
    ('ui_pref', 'ui', 8, 'ReInitPreferences', 0x3672C4, 0x3672DC, 0x3672EA, 0x367363, 0x36749C,
     (0x36730F, 0x367333), (0x36748A, 0x367495)),
    ('ui_update', 'ui', 0, 'RefreshView', 0x367380, 0x367398, 0x3673A6, 0x36742E, 0x367484,
     (0x3673C8, 0x3673EE), (0x367475, 0x36747D)),
]
STATIC_TYPES = {'game': ('GameEvents', 5518), 'gameplay': ('GameplayEvents', 5519), 'ui': ('UIEvents', 5523)}
SERVICES = {0x4D5170: 'System.Action$$.ctor', 0x116BCC0: 'System.Delegate$$Combine',
            0x116E070: 'System.Delegate$$Remove', 0x1C82480: 'UnityEngine.Object$$op_Inequality'}
POISON = [0xFACE123456789000 + i for i in range(4)]
INCOMING = [0, 0xDEAD123400000002, 0xDEAD123400000008, 0xDEAD123400000009]


class Machine(SurfaceMachine):
    caller_return = RewardMachine.caller_return

    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        import capstone
        raw = (Path(dumper_root) / 'dump.cs').read_bytes()
        extraction = json.loads((Path(__file__).parents[1] /
            f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs']['dump_cs']['sha256'].upper()
        dump = raw.decode('utf-8-sig')
        def declaration(name, index, static=False):
            pattern = r'^public ' + ('static ' if static else '') + r'class ' + name + r'(?: :[^\n]*)? // TypeDefIndex: ' + str(index) + r'\s*\{(.*?)\n\}'
            found = re.search(pattern, dump, re.M | re.S); assert found
            return found[1]
        character = declaration('Character', 5487)
        assert 'public CardInteraction cardInteraction; // 0xA0' in character
        assert 'public bool disableAnimated; // 0x164' in character
        interaction = declaration('CardInteraction', 5468)
        assert 'public Action onHover; // 0x60' in interaction and 'public Action onHoverExit; // 0x68' in interaction
        action = re.search(r'^public sealed class Action : MulticastDelegate // TypeDefIndex: 153\s*\{(.*?)\n\}', dump, re.M | re.S)
        multicast = re.search(r'^public abstract class MulticastDelegate : Delegate // TypeDefIndex: 440\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert action and 'public virtual void Invoke()' in action[1]
        assert multicast and 'private Delegate[] delegates; // 0x78' in multicast[1]
        self.static_fields = {}
        for key, (name, index) in STATIC_TYPES.items():
            body = declaration(name, index, True)
            self.static_fields[key] = [(n, int(o, 16)) for n, o in re.findall(r'public static Action(?:<[^\n]+>)? (\w+); // (0x[\dA-Fa-f]+)', body)]
        assert [len(self.static_fields[k]) for k in STATIC_TYPES] == [6, 29, 21]
        expected_fields = [('game', 'OnGameplayStateChange', 0), ('gameplay', 'OnShowEyeOracleInfo', 0xB8),
                           ('gameplay', 'OnHideEyeOracleInfo', 0xC0), ('ui', 'OnCharacterPrefChanges', 8), ('ui', 'OnUIUpdate', 0)]
        assert all((name, offset) in self.static_fields[k] for k, name, offset in expected_fields)
        self.instructions, self.body_addresses, self.ranges, self.targets, self.flags = {}, set(), {}, [], {}
        self.eh = {}
        for name, (start, end, following, stable) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start and r['Name'] == 'Character$$' + name]
            assert len(rows) == 1 and rows[0]['Signature'] == f'void Character__{name} (Character_o* __this, const MethodInfo* method);'
            assert rows[0]['TypeSignature'] == 'vii'
            self.targets.append(dict(rows[0], stable_method_id=stable)); self.decode(name, start, end, following)
            unwind = [e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == start]
            assert len(unwind) == 1 and unwind[0].struct.EndAddress == end and unwind[0].unwindinfo.Flags == 0
            self.eh[name] = {'unwind_data_rva': hex(unwind[0].struct.UnwindData), 'unwind_flags': 0,
                             'method_local_exception_handler': False}
        assert self.eh['OnEnable']['unwind_data_rva'] == self.eh['OnDisable']['unwind_data_rva'] == '0x2520c38'
        refs = set()
        for i in self.instructions.values():
            for operand in i.operands:
                if operand.type == capstone.CS_OP_MEM and operand.mem.base == capstone.x86.X86_REG_RIP:
                    slot = i.address + i.size + operand.mem.disp; refs.add(slot)
                    if i.mnemonic == 'cmp' and operand.size == 1: self.flags[slot] = 0
        assert set(self.flags) == {0x288C167, 0x288C168}
        self.method_flags = {'OnEnable': 0x288C167, 'OnDisable': 0x288C168}
        names = ['owner', 'owner_class', 'interaction0', 'interaction1', 'interaction_class',
                 'action_class', 'foreign_class', 'object_class', 'game_class', 'gameplay_class', 'ui_class',
                 'game_static', 'other_game_static', 'gameplay_static', 'other_gameplay_static', 'ui_static', 'other_ui_static',
                 'prior_owner', 'prior_method']
        names += [c[0] + '_method' for c in CHANNELS]
        names += [prefix + c[0] for prefix in ['old_', 'other_'] for c in CHANNELS]
        self.fixed_names = names.copy()
        names += ['action' + str(i) for i in range(64)]
        self.p = {n: self.arena + 0x110000 + i * 0x1000 for i, n in enumerate(names)}
        self.labels = {0: None, **{p: n for n, p in self.p.items()}}
        self.sizes = {n: 0x200 if n == 'owner' else 0x100 if n.endswith('_class') or n.endswith('_static') else 0x80 for n in names}
        keys = {'System.Action_TypeInfo': 'action_class', 'UnityEngine.Object_TypeInfo': 'object_class',
                **{name + '_TypeInfo': key + '_class' for key, (name, _) in STATIC_TYPES.items()},
                **{'Method$Character.' + c[3] + '()': c[0] + '_method' for c in CHANNELS}}
        self.bindings, self.metadata_slots, self.literal_slots = {}, {}, {}
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] not in refs: continue
            key = keys[row['Name']]
            if row['Name'].startswith('Method$'):
                rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == 'Character$$' + row['Name'].split('.')[1][:-2]]
                assert len(rows) == 1 and row['MethodAddress'] == rows[0]['Address']
                assert rows[0]['Signature'] == f"void Character__{row['Name'].split('.')[1][:-2]} (Character_o* __this, const MethodInfo* method);"
            self.bindings[row['Name']] = self.p[key]; self.metadata_slots[row['Address']] = self.p[key]
            self.q(self.base + row['Address'], self.p[key])
        assert set(self.bindings) == set(keys) and len(self.metadata_slots) == 12
        self.metadata_order = {}
        self.calls, self.stores, self.excluded_stubs, self.checks = {}, {}, set(), {}
        for method, (start, end, _, _) in TARGETS.items():
            delta = 0 if method == 'OnEnable' else -0x6A0
            ins = [i for a, i in sorted(self.instructions.items()) if start <= a < end]
            order = []
            for previous, i in zip(ins, ins[1:]):
                if i.mnemonic == 'call' and i.op_str == '0x2b7b40':
                    assert previous.mnemonic == 'lea' and previous.operands[1].mem.base == capstone.x86.X86_REG_RIP
                    slot = previous.address + previous.size + previous.operands[1].mem.disp
                    assert slot in self.metadata_slots; order.append((slot, i.address + i.size))
            assert len(order) == 12; self.metadata_order[method] = order
            for c in CHANNELS:
                name, kind, offset, _, alloc, ctor, delegate, barrier, failure, stores, stub = c
                for pc, service, target in [(alloc, 'allocate', 0x2B7D40), (ctor, 'constructor', 0x4D5170),
                    (delegate, 'delegate', 0x116BCC0 if method == 'OnEnable' else 0x116E070),
                    (barrier, 'barrier', 0x2B6FF0), (failure, 'native_cast_failure', 0x2B7040)]:
                    i = self.instructions[pc + delta]
                    assert i.mnemonic == ('jmp' if name == 'ui_update' and service == 'barrier' else 'call') and i.op_str == hex(target)
                    self.calls[(method, service, self.stop if i.mnemonic == 'jmp' else self.base + i.address + i.size)] = c
                    self.checks[i.address] = (i.mnemonic, i.op_str)
                for pc in stores:
                    i = self.instructions[pc + delta]; assert i.mnemonic == 'mov' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 8
                    assert i.operands[0].mem.disp == offset
                    self.stores[pc + delta] = (kind, offset)
                self.excluded_stubs.update(i.address for i in ins if stub[0] + delta <= i.address < stub[1] + delta)
            for pc, mnemonic, operands in [(0x366F02, 'cmp', 'byte ptr [rbp + 0x164], sil'),
                (0x366F16, 'mov', 'rbx, qword ptr [rbp + 0xa0]'),
                (0x366F3F, 'mov', 'r14, qword ptr [rbp + 0xa0]'),
                (0x366FD1, 'mov', 'r14, qword ptr [rbp + 0xa0]')]:
                i = self.instructions[pc + delta]; assert (i.mnemonic, i.op_str) == (mnemonic, operands)
                self.checks[i.address] = (mnemonic, operands)
            assert self.instructions[0x366F16 + delta].mnemonic == 'mov'
            assert (self.instructions[0x366F25 + delta].mnemonic, self.instructions[0x366F25 + delta].op_str) == ('call', '0x281d90')
            assert (self.instructions[0x366F32 + delta].mnemonic, self.instructions[0x366F32 + delta].op_str) == ('call', '0x1c82480')
            assert (self.instructions[0x36746F + delta].mnemonic, self.instructions[0x36746F + delta].op_str) == ('call', '0x2b7d90')
            assert sum(i.mnemonic == 'call' and i.op_str == '0x2b7040' for i in ins) == 14
        assert len(self.excluded_stubs) == 40
        self.service_targets = []
        for address, name in SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1; self.service_targets += rows
        assert self.service_targets[0]['Signature'] == 'void System_Action___ctor (System_Action_o* __this, Il2CppObject* object, intptr_t method, const MethodInfo* method);'
        assert self.service_targets[1]['Signature'] == 'System_Delegate_o* System_Delegate__Combine (System_Delegate_o* a, System_Delegate_o* b, const MethodInfo* method);'
        assert self.service_targets[2]['Signature'] == 'System_Delegate_o* System_Delegate__Remove (System_Delegate_o* source, System_Delegate_o* value, const MethodInfo* method);'
        assert self.service_targets[3]['Signature'] == 'bool UnityEngine_Object__op_Inequality (UnityEngine_Object_o* x, UnityEngine_Object_o* y, const MethodInfo* method);'

    def snapshot(self):
        return {'interaction': self.oid(self.rq(self.p['owner'] + 0xA0)),
                'disable_animated_byte': self.u.mem_read(self.p['owner'] + 0x164, 1)[0],
                'object_initialized_word': self.rd(self.p['object_class'] + 0xE0),
                'static_storage': {k: self.oid(self.rq(self.p[k + '_class'] + 0xB8)) for k in STATIC_TYPES},
                'channels': {n: {str(offset): self.oid(self.rq(self.p[n] + offset)) for offset in [0x60, 0x68]} for n in ['interaction0', 'interaction1']},
                'static_channels': {n: {field: self.oid(self.rq(self.p[n] + offset)) for field, offset in self.static_fields[n.removeprefix('other_').removesuffix('_static')]} for n in self.fixed_names if n.endswith('_static')},
                'delegates': {n: {'phase': self.phases[n], 'supplied_entries': deepcopy(e)} for n, e in self.delegates.items()},
                'allocation_order': self.allocations.copy(), 'operations': deepcopy(self.operations),
                'metadata_flags': {hex(a): self.u.mem_read(self.base + a, 1)[0] for a in sorted(self.flags)},
                'metadata_slots': {hex(a): self.oid(self.rq(self.base + a)) for a in sorted(self.metadata_slots)},
                'memory': {n: bytes(self.u.mem_read(self.p[n], self.sizes[n])).hex() for n in self.fixed_names + self.allocations}}

    def prepare(self, options):
        self.options = deepcopy(options)
        self.events, self.counts, self.error, self.delegates, self.phases = [], {}, None, {}, {}
        self.allocations, self.operations, self.allowed = [], [], {}
        for n in self.p: self.u.mem_write(self.p[n], bytes([0xA5]) * self.sizes[n])
        for n in self.fixed_names:
            if n.endswith('_class'): self.u.mem_write(self.p[n], bytes(self.sizes[n]))
        self.q(self.p['owner'], self.p['owner_class']); self.q(self.p['owner'] + 8, 0)
        self.q(self.p['owner'] + 0xA0, 0 if options.get('null_interaction') else self.p['interaction0'])
        self.u.mem_write(self.p['owner'] + 0x164, bytes([options.get('disable_animated', 0)]))
        self.d(self.p['object_class'] + 0xE0, options.get('object_initialized', 1))
        for n in ['interaction0', 'interaction1']:
            self.q(self.p[n], self.p['interaction_class']); self.q(self.p[n] + 8, 0)
        for k in STATIC_TYPES:
            for n in [k + '_static', 'other_' + k + '_static']:
                self.u.mem_write(self.p[n], bytes(self.sizes[n]))
            self.q(self.p[k + '_class'] + 0xB8, self.p[k + '_static'])
        for c in CHANNELS:
            channel, kind, offset, *_ = c
            own = ['owner', channel + '_method']
            mode = options.get('old_entries', 'prior')
            entries = [] if mode == 'empty' else [own] if mode == 'own' else [own, own] if mode == 'duplicate' else [['prior_owner', 'prior_method'], own] if mode == 'mixed' else [['prior_owner', 'prior_method']]
            assert mode in ['empty', 'own', 'duplicate', 'mixed', 'prior']
            for prefix in ['old_', 'other_']: self.set_delegate(prefix + channel, entries)
            for other in [False, True]:
                n = ('interaction1' if other else 'interaction0') if kind == 'interaction' else ('other_' if other else '') + kind + '_static'
                key = ('other_' if other else 'old_') + channel
                if options.get('alias_channels') and kind == 'interaction': key = 'old_hover'
                self.q(self.p[n] + offset, 0 if options.get('null_channels') else self.p[key])
        for a in self.flags: self.u.mem_write(self.base + a, bytes([0 if options.get('cold') else options.get('warm_flag', 1)]))

    def set_delegate(self, name, entries, foreign=False, allocated=False):
        p = self.p[name]; self.u.mem_write(p, bytes(self.sizes[name]))
        self.q(p, self.p['foreign_class' if foreign else 'action_class'])
        self.delegates[name], self.phases[name] = deepcopy(entries), 'allocated' if allocated else 'supplied_ready'
        if entries:
            owner, method = entries[-1]
            for offset, value in [(0x20, self.p[owner]), (0x28, self.p[method]), (0x40, self.p[owner])]: self.q(p + offset, value)
        self.allow(name, 0, self.sizes[name]); return p

    def fresh(self, entries=None, foreign=False, allocated=False):
        name = 'action' + str(len(self.allocations)); assert name in self.p
        self.allocations.append(name); return self.set_delegate(name, entries or [], foreign, allocated)

    def mutate(self, kind, relative):
        action = self.options.get('mutations', {}).get(kind + ':' + str(relative))
        if action is None: return
        if action in ['replace_interaction', 'clear_interaction']:
            self.q(self.p['owner'] + 0xA0, self.p['interaction1'] if action.startswith('replace') else 0); self.allow('owner', 0xA0, 8)
        elif action == 'disable_animation':
            self.u.mem_write(self.p['owner'] + 0x164, b'\xFF'); self.allow('owner', 0x164, 1)
        elif action.startswith('swap_'):
            key = action[5:]; assert key in STATIC_TYPES
            self.q(self.p[key + '_class'] + 0xB8, self.p['other_' + key + '_static']); self.allow(key + '_class', 0xB8, 8)
        elif action.startswith(('replace_', 'clear_')):
            channel = action.split('_', 1)[1]; c = next(c for c in CHANNELS if c[0] == channel)
            key = self.oid(self.rq(self.p['owner'] + 0xA0)) if c[1] == 'interaction' else self.oid(self.rq(self.p[c[1] + '_class'] + 0xB8))
            assert key
            self.q(self.p[key] + c[2], self.p['other_' + channel] if action.startswith('replace') else 0); self.allow(key, c[2], 8)
        else: raise AssertionError(action)

    def event(self, kind, args):
        ordinal = self.counts.get(kind, 0) + 1; self.counts[kind] = ordinal
        abi = {n.lower() + '_bits': self.reg(getattr(self.x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']}
        abi['caller_return'] = self.caller_return(self.rq(self.reg(self.x.UC_X86_REG_RSP)))
        self.events.append({'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': abi, 'snapshot': self.snapshot()})
        relative = ordinal - self.entry_counts.get(kind, 0)
        if self.options.get('failure') == [kind, relative]:
            self.error = kind; self.u.emu_stop(); return False
        self.mutate(kind, relative); return True

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        if address == self.stop: return
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == (0x366F16 if self.method == 'OnEnable' else 0x366876):
            self.entry_interaction = self.rq(self.p['owner'] + 0xA0)
        if rva in self.stores:
            kind, offset = self.stores[rva]
            pointer = self.reg(x.UC_X86_REG_R14) if kind == 'interaction' else cx
            self.allow(self.oid(pointer), offset, 8)
        if rva in self.instructions: return
        caller = self.rq(self.reg(x.UC_X86_REG_RSP))
        if rva == 0x2B7B40:
            assert cx - self.base in self.metadata_slots
            if self.event('metadata', [hex(cx - self.base), self.oid(self.rq(cx))]): self.ret(self.rq(cx))
        elif rva == 0x281D90:
            assert cx == self.p['object_class'] and self.rd(cx + 0xE0) == 0
            if self.event('class_init', ['object_class']): self.d(cx + 0xE0, 1); self.allow('object_class', 0xE0, 4); self.ret()
        elif rva == 0x1C82480:
            assert cx == self.entry_interaction and dx == 0 and r8 == 0
            bits = 0xFEDCBA9876543200 | self.options.get('predicate_byte', 0 if cx == 0 else 1)
            if self.event('predicate', [self.oid(cx), dx, r8, bits]): self.ret(bits)
        elif rva == 0x2B7D40:
            assert cx == self.p['action_class']
            c = self.calls[(self.method, 'allocate', caller)]; channel, kind, offset, *_ = c
            pointer = self.rq(self.p['owner'] + 0xA0) if kind == 'interaction' else self.rq(self.p[kind + '_class'] + 0xB8)
            assert pointer; self.captured[channel] = (pointer, self.rq(pointer + offset))
            if self.event('allocate', ['action_class', channel]): self.last_new = self.fresh(allocated=True); self.ret(self.last_new)
        elif rva == 0x4D5170:
            c = self.calls[(self.method, 'constructor', caller)]; channel = c[0]
            assert cx == self.last_new and dx == self.p['owner'] and r8 == self.p[channel + '_method'] and r9 == 0
            assert self.phases[self.oid(cx)] == 'allocated'
            if self.event('constructor', [self.oid(cx), 'owner', channel + '_method', r9]):
                name = self.oid(cx); self.delegates[name] = [['owner', channel + '_method']]; self.phases[name] = 'supplied_constructed'
                for offset, value in [(0x20, dx), (0x28, r8), (0x40, dx)]: self.q(cx + offset, value); self.allow(name, offset, 8)
                self.ret()
        elif rva in [0x116BCC0, 0x116E070]:
            c = self.calls[(self.method, 'delegate', caller)]; channel = c[0]
            assert cx == self.captured[channel][1] and dx == self.last_new and r8 == 0 and r9 == POISON[3]
            kind = 'Combine' if self.method == 'OnEnable' else 'Remove'
            assert rva == (0x116BCC0 if kind == 'Combine' else 0x116E070)
            if self.event('delegate', [kind, channel, self.oid(cx), self.oid(dx), r8, r9]):
                before = deepcopy(self.delegates[self.oid(cx)]) if cx else []; own = deepcopy(self.delegates[self.oid(dx)])
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
            c = self.calls[(self.method, 'barrier', caller)]; channel, kind, offset, *_ = c
            pointer = self.captured[channel][0] if kind == 'interaction' else self.rq(self.p[kind + '_class'] + 0xB8)
            result = self.p.get(self.operations[-1]['result'], 0)
            assert cx == pointer + offset and dx == result and self.rq(cx) == result
            assert r8 == (POISON[2] if kind == 'interaction' else result) and r9 == POISON[3]
            if self.event('barrier', [self.oid(pointer), channel, self.oid(result)]): self.ret()
        elif rva in [0x2B7D90, 0x2B7040]:
            kind = 'native_null_guard' if rva == 0x2B7D90 else 'native_cast_failure'
            args = [] if kind == 'native_null_guard' else [self.oid(cx), self.oid(dx)]
            if kind == 'native_cast_failure':
                assert self.rq(cx) == self.p['foreign_class'] and dx == self.p['action_class']
                self.calls[(self.method, kind, caller)]
            self.event(kind, args); self.error = kind; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def run(self, method, options=None, retained=False):
        if not retained: self.prepare(options or {})
        elif options is not None: self.options.update(deepcopy(options))
        self.method, self.error, self.allowed, self.captured = method, None, {}, {}
        self.entry_counts = self.counts.copy()
        initial, old = self.snapshot(), len(self.events)
        self.entry_interaction = self.rq(self.p['owner'] + 0xA0)
        # This early pointer is captured by actual code before the class helper.
        # If metadata mutates it, update the oracle exactly when RBX loads it.
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFAB0000000000000 + i)
        for i in range(6, 16): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), (0xABCDEF9876543210 << 64) | i)
        for n, value in [('RSP', sp), ('RCX', self.p['owner']), ('RDX', INCOMING[1]), ('R8', INCOMING[2]), ('R9', INCOMING[3])]: self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), value)
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
        # Unallocated arena slots are outside the logical object graph, but still
        # retained exactly until an explicitly reached allocator/helper owns them.
        for i in range(len(self.allocations), 64):
            n = 'action' + str(i)
            assert bytes(self.u.mem_read(self.p[n], self.sizes[n])) == b'\xA5' * self.sizes[n]
        row = {'method': method, 'options': deepcopy(self.options), 'returned': returned, 'error': self.error,
               'service_counts_before': self.entry_counts.copy(), 'service_counts_after': self.counts.copy(),
               'initial': initial, 'events': deepcopy(self.events[old:]), 'final': final,
               'unrelated_diagnostic_storage_retained': True, 'unallocated_arena_retained': True,
               'win64_nonvolatile_verified': returned}
        self.verify_model(row); row['independent_ordered_model_verified'] = True
        return row

    def verify_model(self, row):
        """Derive every service entry and retained effect from authored initial bytes."""
        model, options = deepcopy(row['initial']), row['options']
        counts = row['service_counts_before'].copy(); index, returned, expected_error = 0, False, None
        method = row['method']; delta = 0 if method == 'OnEnable' else -0x6A0
        registers = [self.p['owner'], *INCOMING[1:]]
        class Stopped(Exception): pass
        def pointer(name): return self.p[name] if name else 0
        def write(name, offset, width, value):
            raw = bytearray.fromhex(model['memory'][name]); raw[offset:offset + width] = value.to_bytes(width, 'little'); model['memory'][name] = raw.hex()
        def channel_write(name, c, value):
            write(name, c[2], 8, pointer(value))
            if c[1] == 'interaction': model['channels'][name][str(c[2])] = value
            else:
                field = next(n for n, offset in self.static_fields[c[1]] if offset == c[2]); model['static_channels'][name][field] = value
        def channel_read(name, c):
            if c[1] == 'interaction': return model['channels'][name][str(c[2])]
            field = next(n for n, offset in self.static_fields[c[1]] if offset == c[2]); return model['static_channels'][name][field]
        def mutation(kind, relative):
            action = options.get('mutations', {}).get(kind + ':' + str(relative))
            if action is None: return
            if action in ['replace_interaction', 'clear_interaction']:
                model['interaction'] = 'interaction1' if action.startswith('replace') else None; write('owner', 0xA0, 8, pointer(model['interaction']))
            elif action == 'disable_animation': model['disable_animated_byte'] = 255; write('owner', 0x164, 1, 255)
            elif action.startswith('swap_'):
                key = action[5:]; model['static_storage'][key] = 'other_' + key + '_static'; write(key + '_class', 0xB8, 8, pointer(model['static_storage'][key]))
            else:
                channel = action.split('_', 1)[1]; c = next(c for c in CHANNELS if c[0] == channel)
                target = model['interaction'] if c[1] == 'interaction' else model['static_storage'][c[1]]; assert target
                channel_write(target, c, 'other_' + channel if action.startswith('replace') else None)
        def emit(kind, args, raw, caller, terminal=False):
            nonlocal index, registers, expected_error
            ordinal = counts.get(kind, 0) + 1; counts[kind] = ordinal
            abi = {n + '_bits': v for n, v in zip(['rcx', 'rdx', 'r8', 'r9'], raw)}
            abi['caller_return'] = self.caller_return(self.stop if caller is None else self.base + caller)
            expected = {'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': abi, 'snapshot': model}
            assert index < len(row['events']) and row['events'][index] == expected, (method, kind, args, index)
            index += 1
            relative = ordinal - row['service_counts_before'].get(kind, 0)
            if terminal or options.get('failure') == [kind, relative]:
                expected_error = kind; raise Stopped()
            mutation(kind, relative); registers = POISON.copy()
        def fresh(entries=None, foreign=False, allocated=False):
            name = 'action' + str(len(model['allocation_order'])); assert name in self.p
            model['allocation_order'].append(name)
            model['memory'][name] = bytes(self.sizes[name]).hex()
            write(name, 0, 8, self.p['foreign_class' if foreign else 'action_class'])
            entries = entries or []
            model['delegates'][name] = {'phase': 'allocated' if allocated else 'supplied_ready', 'supplied_entries': deepcopy(entries)}
            if entries:
                owner, token = entries[-1]
                for offset, value in [(0x20, pointer(owner)), (0x28, pointer(token)), (0x40, pointer(owner))]: write(name, offset, 8, value)
            return name
        try:
            flag = hex(self.method_flags[method])
            if model['metadata_flags'][flag] == 0:
                for slot, caller in self.metadata_order[method]:
                    name = model['metadata_slots'][hex(slot)]
                    emit('metadata', [hex(slot), name], [self.base + slot, *registers[1:]], caller)
                model['metadata_flags'][flag] = 1
            active = False
            if model['disable_animated_byte'] == 0:
                captured_interaction = model['interaction']
                if model['object_initialized_word'] == 0:
                    emit('class_init', ['object_class'], [self.p['object_class'], *registers[1:]], 0x366F2A + delta)
                    model['object_initialized_word'] = 1; write('object_class', 0xE0, 4, 1)
                low = options.get('predicate_byte', 0 if captured_interaction is None else 1)
                bits = 0xFEDCBA9876543200 | low
                emit('predicate', [captured_interaction, 0, 0, bits], [pointer(captured_interaction), 0, 0, registers[3]], 0x366F37 + delta)
                active = low != 0
            for c in CHANNELS:
                channel, kind, offset, _, alloc, ctor, delegate, barrier, failure, *_ = c
                if kind == 'interaction' and not active: continue
                target = model['interaction'] if kind == 'interaction' else model['static_storage'][kind]
                if target is None: emit('native_null_guard', [], registers, 0x367474 + delta, True)
                old = channel_read(target, c)
                emit('allocate', ['action_class', channel], [self.p['action_class'], *registers[1:]], alloc + delta + 5)
                new = fresh(allocated=True)
                emit('constructor', [new, 'owner', channel + '_method', 0], [pointer(new), self.p['owner'], self.p[channel + '_method'], 0], ctor + delta + 5)
                model['delegates'][new] = {'phase': 'supplied_constructed', 'supplied_entries': [['owner', channel + '_method']]}
                for off, val in [(0x20, self.p['owner']), (0x28, self.p[channel + '_method']), (0x40, self.p['owner'])]: write(new, off, 8, val)
                operation = 'Combine' if method == 'OnEnable' else 'Remove'
                emit('delegate', [operation, channel, old, new, 0, POISON[3]], [pointer(old), pointer(new), 0, POISON[3]], delegate + delta + 5)
                before = deepcopy(model['delegates'][old]['supplied_entries']) if old else []
                own = deepcopy(model['delegates'][new]['supplied_entries'])
                if operation == 'Combine': result = fresh(before + own) if old else new
                else:
                    last = next((i for i in range(len(before) - len(own), -1, -1) if before[i:i + len(own)] == own), None)
                    rest = before if last is None else before[:last] + before[last + len(own):]
                    result = old if last is None else fresh(rest) if rest else None
                mode = options.get('result_' + channel, 'normal')
                if mode == 'null': result = None
                elif mode == 'source': result = old
                elif mode == 'new': result = new
                elif mode == 'foreign': result = fresh(foreign=True)
                elif mode == 'other': result = 'other_' + channel
                model['operations'].append({'kind': operation, 'channel': channel, 'old': old, 'new': new, 'result': result})
                raw_r8 = POISON[2] if kind == 'interaction' else pointer(result)
                if result and int.from_bytes(bytes.fromhex(model['memory'][result])[:8], 'little') != self.p['action_class']:
                    emit('native_cast_failure', [result, 'action_class'], [pointer(result), self.p['action_class'], raw_r8, POISON[3]], failure + delta + 5, True)
                # Static class storage is reloaded after all three supplied calls.
                destination = target if kind == 'interaction' else model['static_storage'][kind]
                channel_write(destination, c, result)
                emit('barrier', [destination, channel, result], [pointer(destination) + offset, pointer(result), raw_r8, POISON[3]], None if channel == 'ui_update' else barrier + delta + 5)
            returned = True
        except Stopped: pass
        assert index == len(row['events']) and returned == row['returned'] and expected_error == row['error']
        assert counts == row['service_counts_after']
        assert model == row['final'], (method, 'final model mismatch')


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, sequences, baselines, stops = [], [], [], []
    for method in TARGETS:
        for cold, initialized, disabled, low in itertools.product([False, True], [0, 1], [0, 1, 0x80, 0xFF], [0, 1, 0x80, 0xFF]):
            cases.append(m.run(method, {'cold': cold, 'object_initialized': initialized, 'disable_animated': disabled, 'predicate_byte': low}))
        for flag in [0x80, 0xFF]: cases.append(m.run(method, {'warm_flag': flag}))
        cases.append(m.run(method, {'object_initialized': 0x80000000}))
        for mode in ['empty', 'prior', 'own', 'duplicate', 'mixed']:
            cases.append(m.run(method, {'old_entries': mode}))
        for option in ['null_interaction', 'null_channels', 'alias_channels']:
            cases.append(m.run(method, {option: True}))
        cases.append(m.run(method, {'null_interaction': True, 'predicate_byte': 1}))
        for channel in [c[0] for c in CHANNELS]:
            for mode in ['null', 'source', 'new', 'foreign', 'other']:
                cases.append(m.run(method, {'result_' + channel: mode}))
        mutations = [('class_init:1', 'replace_interaction'), ('class_init:1', 'clear_interaction'),
                     ('predicate:1', 'replace_interaction'), ('predicate:1', 'clear_interaction'),
                     ('allocate:1', 'replace_interaction'), ('constructor:1', 'replace_hover'),
                     ('delegate:1', 'clear_interaction'), ('barrier:1', 'replace_interaction'),
                     ('barrier:1', 'clear_interaction'), ('barrier:1', 'disable_animation'),
                     ('allocate:2', 'replace_interaction'), ('constructor:2', 'replace_exit')]
        for phase, action in [('metadata:1', 'replace_interaction'), ('metadata:12', 'disable_animation')]:
            cases.append(m.run(method, {'cold': True, 'object_initialized': 0, 'mutations': {phase: action}}))
        for kind, ordinal, key in [('allocate', 3, 'game'), ('constructor', 3, 'game'), ('delegate', 3, 'game'),
                                   ('barrier', 3, 'gameplay'), ('allocate', 4, 'gameplay'), ('delegate', 5, 'gameplay'),
                                   ('barrier', 5, 'ui'), ('allocate', 6, 'ui'), ('delegate', 7, 'ui')]:
            mutations.append((kind + ':' + str(ordinal), 'swap_' + key))
        for phase, action in mutations:
            cases.append(m.run(method, {'object_initialized': 0, 'mutations': {phase: action}}))
        for phase, channel in [('allocate:1', 'hover'), ('delegate:1', 'hover'),
                               ('allocate:3', 'game'), ('constructor:4', 'oracle_show'),
                               ('delegate:5', 'oracle_hide'), ('allocate:6', 'ui_pref'),
                               ('constructor:7', 'ui_update')]:
            cases.append(m.run(method, {'mutations': {phase: 'replace_' + channel}}))
    for options in [{'cold': True}, {'cold': True, 'null_channels': True}, {'alias_channels': True},
                    {'mutations': {'barrier:1': 'replace_interaction', 'allocate:3': 'swap_game'}}]:
        m.prepare(options)
        calls = [m.run(method, retained=True) for method in ['OnEnable', 'OnDisable', 'OnEnable', 'OnDisable']]
        assert all(r['returned'] for r in calls) and all(b['initial'] == a['final'] for a, b in zip(calls, calls[1:]))
        sequences.append({'options': options, 'calls': calls})
    for method, options in itertools.product(TARGETS, [{'cold': True, 'object_initialized': 0}, {'disable_animated': 255},
                                                      {'mutations': {'allocate:3': 'swap_game', 'barrier:1': 'replace_interaction'}},
                                                      {'mutations': {'barrier:1': 'clear_interaction'}},
                                                      {'result_ui_update': 'foreign'}]):
        baseline = m.run(method, options)
        baseline_id = len(baselines); baselines.append(baseline); counts = {}
        for i, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(method, dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:i + 1]
            assert stopped['final'] == event['snapshot']
            stops.append({'baseline': baseline_id, 'prefix_length': i + 1, 'stopped': stopped})
    missing = m.body_addresses - m.executed
    traps = {a for a, i in m.instructions.items() if i.mnemonic == 'int3'}
    assert len(traps) == 30 and len(m.excluded_stubs) == 40 and not traps & m.excluded_stubs
    assert missing == traps | m.excluded_stubs, [hex(a) for a in sorted(missing - traps - m.excluded_stubs)]
    return {'build': BUILD, 'targets': m.targets, 'ranges': m.ranges, 'exception_bounds': m.eh,
            'supplied_targets': m.service_targets, 'static_field_declarations': m.static_fields,
            'metadata_bindings': {n: m.oid(p) for n, p in m.bindings.items()},
            'instruction_assertions': len(m.checks), 'case_count': len(cases), 'cases': cases,
            'retained_sequences': sequences, 'failure_baselines': baselines, 'failure_stops': stops, 'failure_case_count': len(stops),
            'body_instructions_decoded': len(m.body_addresses), 'body_instructions_executed': len(m.body_addresses & m.executed),
            'native_execution_addresses': len(m.executed), 'unexecuted_terminal_traps': [hex(a) for a in sorted(traps)],
            'unexecuted_stable_class_cast_stubs': [hex(a) for a in sorted(m.excluded_stubs)],
            'scope': 'Complete actual Character OnEnable/OnDisable callers. Whole metadata/class initialization, Object inequality, allocation, Action construction, Combine/Remove, GC barriers and null/cast exceptions are supplied services. Ordered delegate entries, object validity and event identities are authored diagnostics, not CLR multicast behavior, managed admission, subscription dispatch or Unity lifecycle. No callback bodies or runtime helpers are reconstructed. Both exact unwind records have Flags=0 and no method-local EH handler; guard/service stops occur before effects, without runtime exception unwinding.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); expanded = audit(args.game_root, args.dumper_root)
    memory_pooled = pool_memory(expanded)
    assert expand_memory(memory_pooled) == expanded
    report = pool_snapshots(memory_pooled)
    assert expand_snapshots(report) == memory_pooled
    assert expand_memory(expand_snapshots(report)) == expanded
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
