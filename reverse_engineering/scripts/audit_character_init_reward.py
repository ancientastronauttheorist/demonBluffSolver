"""Execute complete Character.InitReward with supplied Unity/callback/RevealReal services."""
import argparse
from copy import deepcopy
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_deck_character_surface import Machine as SurfaceMachine
from audit_report_snapshots import expand_snapshots, pool_snapshots


ENTRY, END, FOLLOWING = 0x365640, 0x365712, 0x365720
POINTER_FIELDS = [('data_ref', 0x50), ('bluff', 0x58), ('register_as', 0x60),
                  ('acteds', 0xA8), ('state_action', 0x180)]
WORD_FIELDS = [('pickable_uses', 0xDC), ('prev_state', 0xE0), ('state', 0xE4), ('alignment', 0xF8)]


class Machine(SurfaceMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        # SurfaceMachine already pins the exact dump bytes before this read.
        dump = (Path(dumper_root) / 'dump.cs').read_text(encoding='utf-8-sig')
        def block(pattern):
            found = re.search(pattern, dump, re.M | re.S); assert found
            return found[1]
        character = block(r'^public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487\s*\{(.*?)\n\}')
        for field in ['public CharacterData dataRef; // 0x50', 'public CharacterData bluff; // 0x58',
                      'public CharacterData registerAs; // 0x60', 'public Acted acteds; // 0xA8',
                      'private int pickableUses; // 0xDC', 'public ECharacterState prevState; // 0xE0',
                      'public ECharacterState state; // 0xE4', 'public EAlignment alignment; // 0xF8',
                      'public Action onStateChange; // 0x180']:
            assert field in character
        data = block(r'^public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845\s*\{(.*?)\n\}')
        assert 'public EAlignment startingAlignment; // 0x134' in data
        states = block(r'^public enum ECharacterState // TypeDefIndex: 5489\s*\{(.*?)\n\}')
        assert 'public const ECharacterState Hidden = 5;' in states
        self.state_values = {name: int(value) for name, value in re.findall(r'public const ECharacterState (\w+) = (\d+);', states)}
        alignment = block(r'^public enum EAlignment // TypeDefIndex: 5492\s*\{(.*?)\n\}')
        self.alignment_values = {name: int(value) for name, value in re.findall(r'public const EAlignment (\w+) = (\d+);', alignment)}
        assert self.state_values == {'None': 0, 'Hidden': 5, 'Alive': 10, 'Dead': 20, 'Revealed': 30}
        assert self.alignment_values == {'None': 0, 'Good': 10, 'Evil': 20}
        rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == ENTRY and r['Name'] == 'Character$$InitReward']
        assert len(rows) == 1 and rows[0]['Signature'] == 'void Character__InitReward (Character_o* __this, CharacterData_o* character, const MethodInfo* method);'
        assert rows[0]['TypeSignature'] == 'viii'
        self.target = dict(rows[0], stable_method_id='tdi5487.m0027')
        self.instructions, self.body_addresses, self.ranges = {}, set(), {}
        self.decode('InitReward', ENTRY, END, FOLLOWING)
        self.services = []
        expected = {
            0x1C79FD0: ('UnityEngine.Component$$get_gameObject', 'UnityEngine_GameObject_o* UnityEngine_Component__get_gameObject (UnityEngine_Component_o* __this, const MethodInfo* method);'),
            0x1C7D810: ('UnityEngine.GameObject$$SetActive', 'void UnityEngine_GameObject__SetActive (UnityEngine_GameObject_o* __this, bool value, const MethodInfo* method);'),
            0x3682A0: ('Character$$RevealReal', 'void Character__RevealReal (Character_o* __this, const MethodInfo* method);'),
        }
        for address, (name, signature) in expected.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['Signature'] == signature; self.services += rows
        names = ['owner', 'character_class', 'data0', 'data1', 'data_class', 'bluff', 'register_as',
                 'acteds0', 'acteds1', 'acteds_class', 'gameobject0', 'gameobject1', 'gameobject_class',
                 'state_action', 'other_state_action', 'action_class_token', 'state_code', 'other_code',
                 'state_method', 'other_method', 'unused_managed_target']
        self.p = {name: self.arena + 0xD0000 + i * 0x1000 for i, name in enumerate(names)}
        self.labels = {0: None, **{p: n for n, p in self.p.items()}}
        self.sizes = {n: 0x200 if n in ['owner', 'data0', 'data1'] else 0x100 if 'class' in n else 0x80 for n in names}
        self.callback = self.stop + 0x100
        self.stores = {0x365683: (0x58, 8), 0x365696: (0x50, 8), 0x3656A4: (0x60, 8),
                       0x3656B0: (0xDC, 4), 0x3656C5: (0xF8, 4),
                       0x3656D1: (0xE0, 4), 0x3656DE: (0xE4, 4)}
        self.checks = {
            0x36564D: ('mov', 'rdi, rdx'), 0x365650: ('mov', 'rcx, qword ptr [rcx + 0xa8]'),
            0x365660: ('xor', 'edx, edx'), 0x365670: ('xor', 'r8d, r8d'),
            0x365673: ('xor', 'edx, edx'), 0x365683: ('mov', 'qword ptr [rcx], 0'),
            0x365696: ('mov', 'qword ptr [rcx], rdi'), 0x3656A4: ('mov', 'qword ptr [rcx], 0'),
            0x3656B0: ('mov', 'dword ptr [rbx + 0xdc], 1'),
            0x3656BF: ('mov', 'eax, dword ptr [rdi + 0x134]'),
            0x3656C5: ('mov', 'dword ptr [rbx + 0xf8], eax'),
            0x3656CB: ('mov', 'eax, dword ptr [rbx + 0xe4]'),
            0x3656D1: ('mov', 'dword ptr [rbx + 0xe0], eax'),
            0x3656D7: ('mov', 'rax, qword ptr [rbx + 0x180]'),
            0x3656DE: ('mov', 'dword ptr [rbx + 0xe4], 5'),
            0x3656ED: ('mov', 'rdx, qword ptr [rax + 0x28]'),
            0x3656F1: ('mov', 'rcx, qword ptr [rax + 0x40]'),
            0x3656F5: ('call', 'qword ptr [rax + 0x18]'),
            0x3656F8: ('xor', 'edx, edx'), 0x365707: ('jmp', '0x3682a0'),
        }
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == expected for a, expected in self.checks.items())
        calls = [i.op_str for i in self.instructions.values() if i.mnemonic == 'call']
        assert calls.count('0x2b6ff0') == 3 and calls.count('0x1c79fd0') == 1 and calls.count('0x1c7d810') == 1 and calls.count('qword ptr [rax + 0x18]') == 1

    def snapshot(self):
        return {'pointers': {n: self.oid(self.rq(self.p['owner'] + o)) for n, o in POINTER_FIELDS},
                'words': {n: self.rd(self.p['owner'] + o) for n, o in WORD_FIELDS},
                'data_alignment_bits': {n: self.rd(self.p[n] + 0x134) for n in ['data0', 'data1']},
                'gameobject_active': self.active.copy(), 'activation_requests': self.activation_requests.copy(),
                'callbacks': self.callbacks.copy(), 'reveal_requests': self.reveals.copy(),
                'native_phases': self.phases.copy(),
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options = options.copy()
        self.events, self.counts, self.error, self.allowed = [], {}, None, {}
        self.callbacks, self.reveals, self.activation_requests, self.phases = [], [], [], []
        self.active = {'gameobject0': True, 'gameobject1': True}
        for n, p in self.p.items(): self.u.mem_write(p, bytes([0xA5]) * self.sizes[n])
        self.q(self.p['owner'], self.p['character_class']); self.q(self.p['owner'] + 8, 0)
        for n, o in POINTER_FIELDS:
            key = {'data_ref': 'data0', 'bluff': 'bluff', 'register_as': 'register_as',
                   'acteds': 'acteds0', 'state_action': 'state_action'}[n]
            self.q(self.p['owner'] + o, 0 if options.get('null_' + n) else self.p[key])
        for n, o in WORD_FIELDS:
            self.d(self.p['owner'] + o, options.get(n, {'pickable_uses': 0xF0000007, 'prev_state': 30, 'state': 10, 'alignment': 20}[n]))
        self.input_data = 0 if options.get('null_input_data') else self.p['data0' if options.get('alias_input_data') else 'data1']
        for n in ['data0', 'data1']:
            self.q(self.p[n], self.p['data_class']); self.d(self.p[n] + 0x134, options.get('input_alignment', 10))
        for n in ['acteds0', 'acteds1']: self.q(self.p[n], self.p['acteds_class'])
        for n in ['gameobject0', 'gameobject1']: self.q(self.p[n], self.p['gameobject_class'])
        for name, code, method in [('state_action', 'state_code', 'state_method'), ('other_state_action', 'other_code', 'other_method')]:
            self.u.mem_write(self.p[name], bytes(self.sizes[name])); self.q(self.p[name], self.p['action_class_token'])
            self.q(self.p[name] + 0x18, self.callback); self.q(self.p[name] + 0x20, self.p['unused_managed_target'])
            self.q(self.p[name] + 0x28, self.p[method]); self.q(self.p[name] + 0x40, 0 if options.get('null_method_code') else self.p[code])

    def mutate(self, kind, ordinal):
        action = self.options.get('mutations', {}).get(kind + ':' + str(ordinal))
        if action is None: return
        fields = {'replace_acteds': (0xA8, 'acteds1'), 'clear_acteds': (0xA8, None),
                  'replace_data_ref': (0x50, 'data0'), 'clear_data_ref': (0x50, None),
                  'replace_state_action': (0x180, 'other_state_action'), 'clear_state_action': (0x180, None)}
        if action in fields:
            offset, name = fields[action]; self.q(self.p['owner'] + offset, self.p[name] if name else 0); self.allow('owner', offset, 8)
        elif action == 'change_input_alignment':
            assert self.input_data; self.d(self.input_data + 0x134, 0xF1234567); self.allow(self.oid(self.input_data), 0x134, 4)
        elif action in ['change_state', 'change_prev_state', 'change_alignment', 'change_pickable_uses']:
            field = action[7:]; offset = dict(WORD_FIELDS)[field]
            self.d(self.p['owner'] + offset, 0xE1234567); self.allow('owner', offset, 4)
        else: raise AssertionError(action)

    def caller_return(self, value):
        rva = value - self.base
        if value == self.stop:
            return {'address_bits': value, 'native_rva': None, 'decoded': 'authored caller stop'}
        assert rva in self.instructions
        instruction = self.instructions[rva]
        return {'address_bits': value, 'native_rva': hex(rva),
                'decoded': instruction.mnemonic + (' ' + instruction.op_str if instruction.op_str else '')}

    def event(self, kind, args):
        ordinal = self.counts.get(kind, 0) + 1
        self.counts[kind] = ordinal
        abi = {n.lower() + '_bits': self.reg(getattr(self.x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']}
        abi['caller_return'] = self.caller_return(self.rq(self.reg(self.x.UC_X86_REG_RSP)))
        self.events.append({'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': abi, 'snapshot': self.snapshot()})
        if self.options.get('failure') == [kind, ordinal]:
            self.error = kind; self.u.emu_stop(); return False
        self.mutate(kind, ordinal)
        return True

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        if address == self.stop: return
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == ENTRY:
            assert cx == self.p['owner'] and dx == self.input_data
            self.phases.append({'phase': 'entry', 'captured_input': self.oid(dx)})
        elif rva == 0x3656BF:
            self.expected_alignment = self.rd(self.input_data + 0x134)
            self.phases.append({'phase': 'alignment_load', 'input': self.oid(self.input_data), 'bits': self.expected_alignment})
        elif rva == 0x3656CB:
            assert self.rd(self.p['owner'] + 0xF8) == self.expected_alignment
            self.expected_prev_state = self.rd(self.p['owner'] + 0xE4)
            self.phases.append({'phase': 'current_state_load', 'bits': self.expected_prev_state})
        elif rva == 0x3656D7:
            assert self.rd(self.p['owner'] + 0xE0) == self.expected_prev_state
        if rva in self.stores:
            offset, width = self.stores[rva]; self.allow('owner', offset, width)
        if rva in self.instructions: return
        if rva == 0x1C79FD0:
            assert cx == self.entry_acted and cx in [self.p['acteds0'], self.p['acteds1']] and dx == 0
            mode = self.options.get('gameobject_result', 'normal')
            result = 0 if mode == 'null' else self.p['gameobject1'] if mode == 'other' or cx == self.p['acteds1'] else self.p['gameobject0']
            assert mode in ['normal', 'null', 'other']
            if self.event('get_gameobject', [self.oid(cx), dx, self.oid(result)]): self.last_gameobject = result; self.ret(result)
        elif rva == 0x1C7D810:
            assert cx == self.last_gameobject and cx in [self.p['gameobject0'], self.p['gameobject1']] and dx == 0 and r8 == 0
            if self.event('set_active', [self.oid(cx), dx, r8, r9]):
                self.active[self.oid(cx)] = False; self.activation_requests.append(self.oid(cx)); self.ret()
        elif rva == 0x2B6FF0:
            ordinal = self.counts.get('barrier', 0) - self.entry_counts.get('barrier', 0) + 1
            field, offset = [('bluff', 0x58), ('data_ref', 0x50), ('register_as', 0x60)][ordinal - 1]
            assert cx == self.p['owner'] + offset and dx == (self.input_data if field == 'data_ref' else 0) and self.rq(cx) == dx
            if self.event('barrier', [field, self.oid(dx)]): self.ret()
        elif address == self.callback:
            action = self.rq(self.p['owner'] + 0x180)
            assert action in [self.p['state_action'], self.p['other_state_action']]
            assert cx == self.rq(action + 0x40) and dx == self.rq(action + 0x28)
            assert r8 == 0xFACE123456789002 and r9 == 0xFACE123456789003
            assert self.rd(self.p['owner'] + 0xF8) == self.expected_alignment
            assert self.rd(self.p['owner'] + 0xE0) == self.expected_prev_state and self.rd(self.p['owner'] + 0xE4) == 5
            assert self.rd(self.p['owner'] + 0xDC) == 1
            record = [self.oid(cx), self.oid(dx), r8, r9]
            if self.event('callback', record): self.callbacks.append(record); self.ret()
        elif rva == 0x3682A0:
            assert cx == self.p['owner'] and dx == 0 and r8 == 0xFACE123456789002 and r9 == 0xFACE123456789003
            record = [self.oid(cx), dx, r8, r9]
            if self.event('supplied_RevealReal', record): self.reveals.append(record); self.ret()
        elif rva == 0x2B7D90:
            self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def run(self, options=None, retained=False):
        if not retained: self.prepare(options or {})
        elif options is not None: self.options.update(options)
        self.error, self.allowed = None, {}
        self.entry_counts = self.counts.copy()
        initial, old = self.snapshot(), len(self.events)
        self.entry_acted = self.rq(self.p['owner'] + 0xA8)
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFAB0000000000000 + i)
        for i in range(6, 16): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), (0xABCDEF9876543210 << 64) | i)
        for n, value in [('RSP', sp), ('RCX', self.p['owner']), ('RDX', self.input_data), ('R8', 0xDEAD123400000008), ('R9', 0xDEAD123400000009)]: self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), value)
        self.u.emu_start(self.base + ENTRY, self.stop, timeout=10_000_000, count=100000)
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
        completed = self.events[old:] if returned else self.events[old:-1]
        assert final['callbacks'] == initial['callbacks'] + [e['args'] for e in completed if e['kind'] == 'callback']
        assert final['reveal_requests'] == initial['reveal_requests'] + [e['args'] for e in completed if e['kind'] == 'supplied_RevealReal']
        assert final['activation_requests'] == initial['activation_requests'] + [e['args'][0] for e in completed if e['kind'] == 'set_active']
        expected_active = initial['gameobject_active'].copy()
        for e in completed:
            if e['kind'] == 'set_active': expected_active[e['args'][0]] = False
        assert final['gameobject_active'] == expected_active
        row = {'options': self.options.copy(), 'returned': returned, 'error': self.error,
               'service_counts_before': self.entry_counts.copy(),
               'initial': initial, 'events': self.events[old:].copy(), 'final': final,
               'unrelated_diagnostic_storage_retained': True, 'win64_nonvolatile_verified': returned}
        if not self.options.get('mutations') and not self.options.get('failure'):
            kinds = [e['kind'] for e in row['events']]
            if self.options.get('null_acteds'): assert kinds == ['native_null_guard']
            elif self.options.get('gameobject_result') == 'null': assert kinds == ['get_gameobject', 'native_null_guard']
            elif self.options.get('null_input_data'):
                assert kinds == ['get_gameobject', 'set_active', 'barrier', 'barrier', 'barrier', 'native_null_guard']
                assert final['words'] == dict(initial['words'], pickable_uses=1)
            else:
                assert returned and kinds == ['get_gameobject', 'set_active', 'barrier', 'barrier', 'barrier'] + ([] if self.options.get('null_state_action') else ['callback']) + ['supplied_RevealReal']
                assert final['words'] == dict(initial['words'], pickable_uses=1, prev_state=initial['words']['state'], state=5, alignment=initial['data_alignment_bits'][self.oid(self.input_data)])
            if 'barrier' in kinds:
                assert final['pointers'] == dict(initial['pointers'], bluff=None, register_as=None, data_ref=self.oid(self.input_data))
        self.verify_model(row)
        row['independent_ordered_model_verified'] = True
        return row

    def verify_model(self, row):
        """Model all state/effect chronology from authored input, without native reads."""
        options = row['options']
        model = deepcopy(row['initial'])
        counts = row['service_counts_before'].copy()
        index, expected_returned = 0, False
        input_name = None if options.get('null_input_data') else 'data0' if options.get('alias_input_data') else 'data1'
        input_pointer = self.p[input_name] if input_name else 0
        initial_acted = model['pointers']['acteds']
        model['native_phases'].append({'phase': 'entry', 'captured_input': input_name})

        class Stopped(Exception): pass

        def raw_write(name, offset, width, bits):
            data = bytearray.fromhex(model['memory'][name])
            data[offset:offset + width] = bits.to_bytes(width, 'little')
            model['memory'][name] = data.hex()

        def pointer_write(field, name):
            model['pointers'][field] = name
            raw_write('owner', dict(POINTER_FIELDS)[field], 8, self.p[name] if name else 0)

        def word_write(field, bits):
            model['words'][field] = bits
            raw_write('owner', dict(WORD_FIELDS)[field], 4, bits)

        def authored_mutation(kind, ordinal):
            action = options.get('mutations', {}).get(kind + ':' + str(ordinal))
            if action is None: return
            pointer_effects = {'replace_acteds': ('acteds', 'acteds1'), 'clear_acteds': ('acteds', None),
                               'replace_data_ref': ('data_ref', 'data0'), 'clear_data_ref': ('data_ref', None),
                               'replace_state_action': ('state_action', 'other_state_action'), 'clear_state_action': ('state_action', None)}
            if action in pointer_effects: pointer_write(*pointer_effects[action])
            elif action == 'change_input_alignment':
                assert input_name
                model['data_alignment_bits'][input_name] = 0xF1234567
                raw_write(input_name, 0x134, 4, 0xF1234567)
            elif action.startswith('change_'): word_write(action[7:], 0xE1234567)
            else: raise AssertionError(action)

        def emit(kind, args, registers, caller, terminal=False):
            nonlocal index
            ordinal = counts.get(kind, 0) + 1; counts[kind] = ordinal
            expected_abi = {name + '_bits': value for name, value in zip(['rcx', 'rdx', 'r8', 'r9'], registers)}
            expected_abi['caller_return'] = self.caller_return(self.stop if caller is None else self.base + caller)
            actual = row['events'][index]; index += 1
            assert actual == {'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': expected_abi, 'snapshot': model}, kind
            if terminal or options.get('failure') == [kind, ordinal]: raise Stopped()
            authored_mutation(kind, ordinal)

        poison = [0xFACE123456789000 + i for i in range(4)]
        incoming = [self.p['owner'], input_pointer, 0xDEAD123400000008, 0xDEAD123400000009]
        try:
            if initial_acted is None:
                emit('native_null_guard', [], [0, incoming[1], incoming[2], incoming[3]], 0x365711, terminal=True)
            mode = options.get('gameobject_result', 'normal')
            gameobject = None if mode == 'null' else 'gameobject1' if mode == 'other' or initial_acted == 'acteds1' else 'gameobject0'
            emit('get_gameobject', [initial_acted, 0, gameobject], [self.p[initial_acted], 0, incoming[2], incoming[3]], 0x365667)
            if gameobject is None: emit('native_null_guard', [], poison, 0x365711, terminal=True)
            emit('set_active', [gameobject, 0, 0, poison[3]], [self.p[gameobject], 0, 0, poison[3]], 0x36567D)
            model['gameobject_active'][gameobject] = False
            model['activation_requests'].append(gameobject)
            for field, value, caller in [('bluff', None, 0x36568F), ('data_ref', input_name, 0x36569E), ('register_as', None, 0x3656B0)]:
                pointer_write(field, value)
                emit('barrier', [field, value], [self.p['owner'] + dict(POINTER_FIELDS)[field], self.p[value] if value else 0, poison[2], poison[3]], caller)
            word_write('pickable_uses', 1)
            if input_name is None: emit('native_null_guard', [], poison, 0x365711, terminal=True)
            alignment = model['data_alignment_bits'][input_name]
            model['native_phases'].append({'phase': 'alignment_load', 'input': input_name, 'bits': alignment})
            word_write('alignment', alignment)
            state = model['words']['state']
            model['native_phases'].append({'phase': 'current_state_load', 'bits': state})
            word_write('prev_state', state)
            action = model['pointers']['state_action']
            word_write('state', 5)
            if action:
                action_bytes = bytes.fromhex(model['memory'][action])
                code = int.from_bytes(action_bytes[0x40:0x48], 'little')
                method = int.from_bytes(action_bytes[0x28:0x30], 'little')
                record = [self.labels[code], self.labels[method], poison[2], poison[3]]
                emit('callback', record, [code, method, poison[2], poison[3]], 0x3656F8)
                model['callbacks'].append(record)
            record = ['owner', 0, poison[2], poison[3]]
            emit('supplied_RevealReal', record, [self.p['owner'], 0, poison[2], poison[3]], None)
            model['reveal_requests'].append(record)
            expected_returned = True
        except Stopped:
            pass
        assert index == len(row['events']) and row['returned'] == expected_returned
        assert row['final'] == model


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, sequences, baselines, stops = [], [], [], []
    for acted_null, input_null, callback_null, result in itertools.product([False, True], [False, True], [False, True], ['normal', 'null', 'other']):
        cases.append(m.run({'null_acteds': acted_null, 'null_input_data': input_null, 'null_state_action': callback_null, 'gameobject_result': result}))
    for state, alignment in itertools.product([0, 5, 10, 20, 30, 0xF1234567], [0, 10, 20, 0xF2345678]):
        cases.append(m.run({'state': state, 'input_alignment': alignment}))
    for option in ['null_data_ref', 'null_bluff', 'null_register_as', 'alias_input_data', 'null_method_code']:
        cases.append(m.run({option: True}))
    mutations = [('get_gameobject:1', 'replace_acteds'), ('get_gameobject:1', 'clear_acteds'),
                 ('get_gameobject:1', 'replace_data_ref'), ('set_active:1', 'replace_acteds'),
                 ('set_active:1', 'replace_data_ref'), ('barrier:1', 'replace_data_ref'),
                 ('barrier:2', 'replace_data_ref'), ('barrier:3', 'change_input_alignment'),
                 ('barrier:3', 'change_state'), ('barrier:3', 'replace_state_action'),
                 ('barrier:3', 'clear_state_action'), ('callback:1', 'replace_data_ref'),
                 ('callback:1', 'clear_data_ref'), ('callback:1', 'change_state'),
                 ('callback:1', 'change_prev_state'), ('callback:1', 'change_alignment'),
                 ('callback:1', 'change_pickable_uses'), ('supplied_RevealReal:1', 'change_state')]
    for phase, action in mutations: cases.append(m.run({'mutations': {phase: action}}))
    cases.append(m.run({'alias_input_data': True, 'mutations': {'barrier:3': 'change_input_alignment'}}))
    retained_inputs = [{'alias_input_data': alias} for alias in [False, True]] + [
        {'mutations': {'get_gameobject:1': 'replace_acteds', 'set_active:1': 'replace_data_ref'}}]
    for supplied in retained_inputs:
        m.prepare(supplied)
        calls = [m.run(retained=True) for _ in range(3)]
        assert all(row['returned'] for row in calls)
        assert calls[1]['initial'] == calls[0]['final'] and calls[2]['initial'] == calls[1]['final']
        sequences.append({'options': supplied, 'calls': calls})
    for mutations in [None, {'get_gameobject:1': 'replace_acteds'}, {'barrier:3': 'change_state'}, {'callback:1': 'change_alignment'}]:
        options = {'mutations': mutations} if mutations else {}
        baseline = m.run(options); assert baseline['returned']
        baseline_id = len(baselines); baselines.append(baseline); counts = {}
        for i, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:i + 1]
            assert stopped['final'] == e['snapshot']
            stops.append({'baseline': baseline_id, 'prefix_length': i + 1, 'stopped': stopped})
    missing = m.body_addresses - m.executed
    assert missing == {0x365711} and (m.instructions[0x365711].mnemonic, m.instructions[0x365711].op_str) == ('int3', '')
    return {'build': BUILD, 'target': m.target, 'ranges': m.ranges, 'supplied_targets': m.services,
            'enums': {'ECharacterState': m.state_values, 'EAlignment': m.alignment_values},
            'instruction_assertions': len(m.checks), 'cases': cases, 'case_count': len(cases),
            'retained_sequences': sequences, 'failure_baselines': baselines, 'failure_stops': stops, 'failure_case_count': len(stops),
            'body_instructions_decoded': len(m.body_addresses), 'body_instructions_executed': len(m.body_addresses & m.executed),
            'unexecuted_terminal_traps': [hex(a) for a in sorted(missing)], 'native_execution_addresses': len(m.executed),
            'scope': 'Complete actual Character.InitReward caller. Whole Component.get_gameObject, GameObject.SetActive, state Action callback, reference barriers and Character.RevealReal are supplied explicit services. Acted/GameObject/Action identity and active state are authored diagnostics, not Unity/CLR scene admission. Native pointer stores and alignment/previous/current-state DWORD chronology execute. No RevealReal/constructor/callback implementation promotion, runtime dispatch/scheduler or exception unwinding.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); expanded = audit(args.game_root, args.dumper_root)
    report = pool_snapshots(expanded); assert expand_snapshots(report) == expanded
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
