"""Complete CardInteraction Awake/hover-gate callers with supplied engine/runtime services."""
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

TARGETS = {'Awake': (0x35F3E0, 0x35F496, 0x35F4A0, 'tdi5468.m0000'),
           'BlockHover': (0x35F4A0, 0x35F4A5, 0x35F4B0, 'tdi5468.m0003'),
           'UnblockHover': (0x360070, 0x360075, 0x360080, 'tdi5468.m0004')}
SERVICES = {0x606FC0: 'UnityEngine.Component$$GetComponent<object>',
            0x1C79FD0: 'UnityEngine.Component$$get_gameObject',
            0x1C81060: 'UnityEngine.Object$$GetInstanceID', 0xF74DF0: 'System.String$$Format'}
POISON = [0xFACE123456789000 + i for i in range(4)]
INCOMING = [0, 0xDEAD123400000002, 0xDEAD123400000008, 0xDEAD123400000009]


class Machine(SurfaceMachine):
    caller_return = RewardMachine.caller_return

    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        import capstone
        raw = (Path(dumper_root) / 'dump.cs').read_bytes()
        extraction = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs']['dump_cs']['sha256'].upper()
        body = re.search(r'^public class CardInteraction : MonoBehaviour // TypeDefIndex: 5468\s*\{(.*?)\n\}', raw.decode('utf-8-sig'), re.M | re.S)
        assert body
        for field in ['private Character character; // 0x20', 'private string animationId; // 0x40', 'private bool blockHover; // 0x70']:
            assert field in body[1]
        self.instructions, self.body_addresses, self.ranges, self.targets, self.flags, self.same_rva = {}, set(), {}, [], {}, {}
        for name, (start, end, following, stable) in TARGETS.items():
            aliases = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start]
            rows = [r for r in aliases if r['Name'] == 'CardInteraction$$' + name]
            assert len(rows) == 1
            self.same_rva[name] = [r for r in aliases if r['Name'] != rows[0]['Name']]
            assert rows[0]['Signature'] == f'void CardInteraction__{name} (CardInteraction_o* __this, const MethodInfo* method);' and rows[0]['TypeSignature'] == 'vii'
            self.targets.append(dict(rows[0], stable_method_id=stable)); self.decode(name, start, end, following, leaf=name != 'Awake')
        self.checks = {0x35F3E6: ('cmp', 'byte ptr [rip + 0x252cd4a], 0'),
            0x35F424: ('mov', 'rcx, rbx'), 0x35F427: ('call', '0x606fc0'),
            0x35F42C: ('lea', 'rcx, [rbx + 0x20]'), 0x35F430: ('mov', 'rdx, rax'),
            0x35F433: ('mov', 'qword ptr [rcx], rax'), 0x35F436: ('call', '0x2b6ff0'),
            0x35F440: ('call', '0x1c79fd0'), 0x35F445: ('test', 'rax, rax'),
            0x35F44F: ('call', '0x1c81060'), 0x35F45B: ('lea', 'rdx, [rsp + 0x30]'),
            0x35F460: ('mov', 'dword ptr [rsp + 0x30], eax'), 0x35F464: ('call', '0x282580'),
            0x35F470: ('xor', 'r8d, r8d'), 0x35F473: ('mov', 'rdx, rax'),
            0x35F476: ('call', '0xf74df0'), 0x35F47B: ('lea', 'rcx, [rbx + 0x40]'),
            0x35F482: ('mov', 'qword ptr [rcx], rax'), 0x35F485: ('call', '0x2b6ff0'),
            0x35F490: ('call', '0x2b7d90'), 0x35F4A0: ('mov', 'byte ptr [rcx + 0x70], 1'),
            0x360070: ('mov', 'byte ptr [rcx + 0x70], 0')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        refs = set()
        for i in self.instructions.values():
            for o in i.operands:
                if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP:
                    slot = i.address + i.size + o.mem.disp; refs.add(slot)
                    if i.mnemonic == 'cmp' and o.size == 1: self.flags[slot] = 0
        assert set(self.flags) == {0x288C137}; self.flag = 0x288C137
        names = ['owner', 'owner_class', 'character0', 'character1', 'character_class', 'gameobject0', 'gameobject1',
                 'gameobject_class', 'component_method', 'int_class', 'boxed0', 'boxed1', 'string_class',
                 'format_literal', 'other_literal', 'name0', 'name1', 'replacement']
        self.p = {n: self.arena + 0x1C0000 + i * 0x1000 for i, n in enumerate(names)}
        self.labels = {0: None, **{p: n for n, p in self.p.items()}}
        self.sizes = {n: 0x100 if n.endswith('_class') else 0x80 for n in names}
        self.bindings, self.metadata_slots, self.literal_slots = {}, {}, {}
        for r in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if r['Address'] not in refs: continue
            key = {'int_TypeInfo': 'int_class', 'Method$UnityEngine.Component.GetComponent<Character>()': 'component_method'}[r['Name']]
            if key == 'component_method': assert r['MethodAddress'] == 0
            self.bindings[r['Name']] = self.p[key]; self.metadata_slots[r['Address']] = self.p[key]
        strings = [r for r in self.metadata['ScriptString'] if r['Address'] in refs]
        assert len(strings) == 1 and strings[0]['Value'] == 'cardAnim_{0}'
        self.literal_slot = strings[0]['Address']; self.metadata_slots[self.literal_slot] = self.p['format_literal']
        assert len(self.bindings) == 2 and len(self.metadata_slots) == 3
        self.metadata_order = []
        ordered = [i for _, i in sorted(self.instructions.items())]
        for previous, i in zip(ordered, ordered[1:]):
            if i.mnemonic == 'call' and i.op_str == '0x2b7b40':
                assert previous.mnemonic == 'lea'
                slot = previous.address + previous.size + previous.operands[1].mem.disp
                assert slot in self.metadata_slots; self.metadata_order.append((slot, i.address + i.size))
        assert len(self.metadata_order) == 3
        self.service_targets = []
        for address, name in SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1; self.service_targets += rows
        signatures = ['Il2CppObject* UnityEngine_Component__GetComponent_object_ (UnityEngine_Component_o* __this, const MethodInfo_606FC0* method);',
            'UnityEngine_GameObject_o* UnityEngine_Component__get_gameObject (UnityEngine_Component_o* __this, const MethodInfo* method);',
            'int32_t UnityEngine_Object__GetInstanceID (UnityEngine_Object_o* __this, const MethodInfo* method);',
            'System_String_o* System_String__Format (System_String_o* format, Il2CppObject* arg0, const MethodInfo* method);']
        assert [r['Signature'] for r in self.service_targets] == signatures
        self.entry_sp = self.stack + 0x18008

    def snapshot(self):
        return {'fields': {'character': self.oid(self.rq(self.p['owner'] + 0x20)), 'animation_id': self.oid(self.rq(self.p['owner'] + 0x40)),
                           'block_hover_byte': self.u.mem_read(self.p['owner'] + 0x70, 1)[0]},
                'metadata_flag_byte': self.u.mem_read(self.base + self.flag, 1)[0],
                'metadata_slots': {hex(a): self.oid(self.rq(self.base + a)) for a in sorted(self.metadata_slots)},
                'boxing_slot_qword': self.rq(self.entry_sp + 8), 'boxed_values': self.boxed.copy(),
                'native_entries': deepcopy(self.entries), 'service_history': deepcopy(self.history),
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options = deepcopy(options); self.events, self.counts, self.error = [], {}, None
        self.entries, self.history, self.boxed, self.allowed = [], [], {}, {}
        for n in self.p: self.u.mem_write(self.p[n], b'\xA5' * self.sizes[n])
        self.q(self.p['owner'], self.p['owner_class']); self.q(self.p['owner'] + 8, 0)
        self.q(self.p['owner'] + 0x20, self.p['character0']); self.q(self.p['owner'] + 0x40, self.p['name0'])
        self.u.mem_write(self.p['owner'] + 0x70, bytes([options.get('initial_hover_byte', 0x80)]))
        for n, text in [('format_literal', 'cardAnim_{0}'), ('other_literal', 'alternate_{0}'), ('name0', 'old id'), ('name1', 'supplied formatted id'), ('replacement', 'replacement id')]:
            self.u.mem_write(self.p[n], bytes(self.sizes[n])); self.q(self.p[n], self.p['string_class'])
            self.d(self.p[n] + 0x10, len(text)); self.u.mem_write(self.p[n] + 0x14, text.encode('utf-16-le') + b'\0\0')
        for a, value in self.metadata_slots.items(): self.q(self.base + a, value)
        self.u.mem_write(self.base + self.flag, bytes([0 if options.get('cold') else options.get('warm_flag', 1)]))
        self.q(self.entry_sp + 8, options.get('initial_boxing_qword', 0xAABBCCDD11223344))

    def mutate(self, kind, relative):
        action = self.options.get('mutations', {}).get(kind + ':' + str(relative))
        if action is None: return
        if action in ['replace_character', 'clear_character']:
            self.q(self.p['owner'] + 0x20, self.p['character1'] if action.startswith('replace') else 0); self.allow('owner', 0x20, 8)
        elif action in ['replace_animation', 'clear_animation']:
            self.q(self.p['owner'] + 0x40, self.p['replacement'] if action.startswith('replace') else 0); self.allow('owner', 0x40, 8)
        elif action == 'replace_literal': self.q(self.base + self.literal_slot, self.p['other_literal'])
        else: raise AssertionError(action)

    def event(self, kind, args):
        ordinal = self.counts.get(kind, 0) + 1; self.counts[kind] = ordinal
        abi = {n.lower() + '_bits': self.reg(getattr(self.x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']}
        abi['caller_return'] = self.caller_return(self.rq(self.reg(self.x.UC_X86_REG_RSP)))
        self.events.append({'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': abi, 'snapshot': self.snapshot()})
        relative = ordinal - self.entry_counts.get(kind, 0)
        if self.options.get('failure') == [kind, relative]: self.error = kind; self.u.emu_stop(); return False
        self.mutate(kind, relative); return True

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        if address == self.stop: return
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == TARGETS[self.method][0]: self.entries.append({'method': self.method, 'raw_args': [cx, dx, r8, r9]})
        if rva in [0x35F433, 0x35F482] and self.owner_arg: self.allow('owner', 0x20 if rva == 0x35F433 else 0x40, 8)
        if rva in [0x35F4A0, 0x360070] and self.owner_arg: self.allow('owner', 0x70, 1)
        if rva in self.instructions: return
        if rva == 0x2B7B40:
            assert cx - self.base in self.metadata_slots
            if self.event('metadata', [hex(cx - self.base), self.oid(self.rq(cx))]): self.ret(self.rq(cx))
        elif rva == 0x606FC0:
            assert cx == self.owner_arg and dx == self.p['component_method']
            result = 0 if self.options.get('null_component') else self.p[self.options.get('component_result', 'character1')]
            if self.event('get_component', [self.oid(cx), self.oid(dx), self.oid(result)]): self.last_component = result; self.history.append(['get_component', self.oid(result)]); self.ret(result)
        elif rva == 0x1C79FD0:
            assert cx == self.owner_arg and dx == 0
            result = 0 if self.options.get('null_gameobject') else self.p[self.options.get('gameobject_result', 'gameobject0')]
            if self.event('get_gameobject', [self.oid(cx), dx, self.oid(result)]): self.last_gameobject = result; self.history.append(['get_gameobject', self.oid(result)]); self.ret(result)
        elif rva == 0x1C81060:
            assert cx == self.last_gameobject and dx == 0
            result = self.options.get('instance_id_bits', 0xFEDCBA9880000001)
            if self.event('instance_id', [self.oid(cx), dx, result]): self.last_instance = result; self.history.append(['instance_id', result]); self.ret(result)
        elif rva == 0x282580:
            assert cx == self.p['int_class'] and dx == self.entry_sp + 8
            low = self.rd(dx); full = self.rq(dx); assert low == self.last_instance & 0xFFFFFFFF
            result = 0 if self.options.get('null_box_result') else self.p[self.options.get('box_result', 'boxed0')]
            if self.event('box_int32', ['int_class', dx, low, full, self.oid(result)]):
                self.history.append(['box_int32', low, self.oid(result)])
                if result:
                    name = self.oid(result); self.u.mem_write(result, bytes(self.sizes[name])); self.q(result, self.p['int_class']); self.d(result + 0x10, low)
                    self.boxed[name] = low; self.allow(name, 0, self.sizes[name])
                self.last_box = result; self.ret(result)
        elif rva == 0xF74DF0:
            assert cx == self.rq(self.base + self.literal_slot) and dx == self.last_box and r8 == 0
            result = 0 if self.options.get('null_format_result') else self.p[self.options.get('format_result', 'name1')]
            if self.event('format', [self.oid(cx), self.oid(dx), r8, self.oid(result)]): self.last_format = result; self.history.append(['format', self.oid(cx), self.oid(dx), self.oid(result)]); self.ret(result)
        elif rva == 0x2B6FF0:
            relative = self.counts.get('barrier', 0) - self.entry_counts.get('barrier', 0) + 1
            offset, result = (0x20, self.last_component) if relative == 1 else (0x40, self.last_format)
            assert relative in [1, 2] and cx == self.owner_arg + offset and dx == result and self.rq(cx) == result
            if self.event('barrier', [hex(offset), self.oid(result)]): self.history.append(['barrier', offset, self.oid(result)]); self.ret()
        elif rva == 0x2B7D90: self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def run(self, method, options=None, retained=False):
        if not retained: self.prepare(options or {})
        elif options is not None: self.options.update(deepcopy(options))
        self.method, self.error, self.fault, self.allowed = method, None, None, {}
        self.owner_arg = 0 if self.options.get('null_owner') else self.p['owner']
        self.entry_counts = self.counts.copy(); initial, old = self.snapshot(), len(self.events)
        x, sp = self.x, self.entry_sp; self.q(sp, self.stop)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFAB0000000000000 + i)
        for i in range(6, 16): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), (0xABCDEF9876543210 << 64) | i)
        raw = [self.owner_arg, *INCOMING[1:]]
        for n, value in zip(['RCX', 'RDX', 'R8', 'R9'], raw): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), value)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        try: self.u.emu_start(self.base + TARGETS[method][0], self.stop, timeout=10_000_000, count=10000)
        except self.unicorn.UcError as exc:
            assert self.options.get('null_owner') and exc.errno == self.unicorn.UC_ERR_WRITE_UNMAPPED
            pc = self.reg(x.UC_X86_REG_RIP) - self.base
            assert pc == {'Awake': 0x35F433, 'BlockHover': 0x35F4A0, 'UnblockHover': 0x360070}[method]
            self.error, self.fault = 'native_owner_access_fault', hex(pc)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): assert self.reg(getattr(x, 'UC_X86_REG_' + n)) == 0xFAB0000000000000 + i
            for i in range(6, 16): assert self.reg(getattr(x, 'UC_X86_REG_XMM' + str(i))) == (0xABCDEF9876543210 << 64) | i
        final = self.snapshot()
        for n, before in initial['memory'].items():
            before, after = bytes.fromhex(before), bytes.fromhex(final['memory'][n])
            assert all(i in self.allowed.get(n, set()) or byte == after[i] for i, byte in enumerate(before)), n
        assert final['boxing_slot_qword'] >> 32 == initial['boxing_slot_qword'] >> 32
        row = {'method': method, 'options': deepcopy(self.options), 'entry_raw_args': raw, 'returned': returned,
               'error': self.error, 'fault_rva': self.fault, 'initial': initial, 'events': deepcopy(self.events[old:]), 'final': final,
               'service_counts_before': self.entry_counts.copy(), 'service_counts_after': self.counts.copy(),
               'unrelated_diagnostic_storage_retained': True, 'nonvolatile_abi_verified': returned}
        self.verify(row); row['independent_ordered_model_verified'] = True; return row

    def verify(self, row):
        model, options, counts = deepcopy(row['initial']), row['options'], row['service_counts_before'].copy()
        raw = row['entry_raw_args'].copy(); index, returned, error, fault = 0, False, None, None
        model['native_entries'].append({'method': row['method'], 'raw_args': raw.copy()})
        class Stopped(Exception): pass
        def pointer(name): return self.p[name] if name else 0
        def write(name, offset, width, value):
            data = bytearray.fromhex(model['memory'][name]); data[offset:offset + width] = value.to_bytes(width, 'little'); model['memory'][name] = data.hex()
        def field(name, target):
            offset = {'character': 0x20, 'animation_id': 0x40}[name]; model['fields'][name] = target; write('owner', offset, 8, pointer(target))
        def mutation(kind, relative):
            action = options.get('mutations', {}).get(kind + ':' + str(relative))
            if action is None: return
            if action in ['replace_character', 'clear_character']: field('character', 'character1' if action.startswith('replace') else None)
            elif action in ['replace_animation', 'clear_animation']: field('animation_id', 'replacement' if action.startswith('replace') else None)
            elif action == 'replace_literal': model['metadata_slots'][hex(self.literal_slot)] = 'other_literal'
            else: raise AssertionError(action)
        def emit(kind, args, registers, caller, terminal=False):
            nonlocal index, raw, error
            ordinal = counts.get(kind, 0) + 1; counts[kind] = ordinal
            abi = {n + '_bits': v for n, v in zip(['rcx', 'rdx', 'r8', 'r9'], registers)}
            abi['caller_return'] = self.caller_return(self.base + caller)
            assert row['events'][index] == {'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': abi, 'snapshot': model}, (row['method'], kind, index)
            index += 1; relative = ordinal - row['service_counts_before'].get(kind, 0)
            if terminal or options.get('failure') == [kind, relative]: error = kind; raise Stopped()
            mutation(kind, relative); raw = POISON.copy()
        try:
            method = row['method']
            if method != 'Awake':
                if row['entry_raw_args'][0] == 0: error, fault = 'native_owner_access_fault', hex(TARGETS[method][0])
                else:
                    value = int(method == 'BlockHover'); model['fields']['block_hover_byte'] = value; write('owner', 0x70, 1, value); returned = True
            else:
                if model['metadata_flag_byte'] == 0:
                    for slot, caller in self.metadata_order:
                        token = model['metadata_slots'][hex(slot)]
                        emit('metadata', [hex(slot), token], [self.base + slot, *raw[1:]], caller)
                    model['metadata_flag_byte'] = 1
                component = None if options.get('null_component') else options.get('component_result', 'character1')
                owner = 'owner' if row['entry_raw_args'][0] else None
                emit('get_component', [owner, 'component_method', component], [pointer(owner), self.p['component_method'], *raw[2:]], 0x35F42C)
                model['service_history'].append(['get_component', component])
                if owner is None: error, fault = 'native_owner_access_fault', '0x35f433'
                else:
                    field('character', component)
                    emit('barrier', ['0x20', component], [self.p['owner'] + 0x20, pointer(component), *raw[2:]], 0x35F43B); model['service_history'].append(['barrier', 0x20, component])
                    game = None if options.get('null_gameobject') else options.get('gameobject_result', 'gameobject0')
                    emit('get_gameobject', ['owner', 0, game], [self.p['owner'], 0, *raw[2:]], 0x35F445); model['service_history'].append(['get_gameobject', game])
                    if game is None: emit('native_null_guard', [], raw, 0x35F495, True)
                    bits = options.get('instance_id_bits', 0xFEDCBA9880000001)
                    emit('instance_id', [game, 0, bits], [pointer(game), 0, *raw[2:]], 0x35F454); model['service_history'].append(['instance_id', bits])
                    low = bits & 0xFFFFFFFF; model['boxing_slot_qword'] = (model['boxing_slot_qword'] & 0xFFFFFFFF00000000) | low
                    box = None if options.get('null_box_result') else options.get('box_result', 'boxed0')
                    emit('box_int32', ['int_class', self.entry_sp + 8, low, model['boxing_slot_qword'], box], [self.p['int_class'], self.entry_sp + 8, *raw[2:]], 0x35F469)
                    model['service_history'].append(['box_int32', low, box])
                    if box:
                        model['memory'][box] = bytes(self.sizes[box]).hex(); write(box, 0, 8, self.p['int_class']); write(box, 0x10, 4, low); model['boxed_values'][box] = low
                    literal = model['metadata_slots'][hex(self.literal_slot)]
                    formatted = None if options.get('null_format_result') else options.get('format_result', 'name1')
                    emit('format', [literal, box, 0, formatted], [pointer(literal), pointer(box), 0, raw[3]], 0x35F47B)
                    model['service_history'].append(['format', literal, box, formatted]); field('animation_id', formatted)
                    emit('barrier', ['0x40', formatted], [self.p['owner'] + 0x40, pointer(formatted), *raw[2:]], 0x35F48A)
                    model['service_history'].append(['barrier', 0x40, formatted]); returned = True
        except Stopped: pass
        assert index == len(row['events']) and (returned, error, fault) == (row['returned'], row['error'], row['fault_rva'])
        assert model == row['final'] and counts == row['service_counts_after']


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for cold, component_null, game_null, box_null, format_null in itertools.product([False, True], repeat=5):
        cases.append(m.run('Awake', {'cold': cold, 'null_component': component_null, 'null_gameobject': game_null, 'null_box_result': box_null, 'null_format_result': format_null}))
    for bits in [0, 1, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF, 0xABCDEF1200000001, 0xFEDCBA9880000001]: cases.append(m.run('Awake', {'instance_id_bits': bits}))
    for flag in [0x80, 0xFF]: cases.append(m.run('Awake', {'warm_flag': flag}))
    for options in [{'component_result': 'character0'}, {'gameobject_result': 'gameobject1'}, {'box_result': 'boxed1'}, {'format_result': 'name0'}, {'format_result': 'format_literal'}, {'null_owner': True}, {'cold': True, 'null_owner': True}]: cases.append(m.run('Awake', options))
    for phase, action in [('metadata:1', 'replace_character'), ('metadata:3', 'replace_literal'), ('get_component:1', 'clear_character'),
        ('barrier:1', 'replace_character'), ('get_gameobject:1', 'replace_animation'), ('instance_id:1', 'clear_animation'),
        ('box_int32:1', 'replace_literal'), ('format:1', 'replace_animation'), ('barrier:2', 'clear_animation')]:
        cases.append(m.run('Awake', {'cold': True, 'mutations': {phase: action}}))
    for method in ['BlockHover', 'UnblockHover']:
        for value in [0, 1, 0x80, 0xFF]: cases.append(m.run(method, {'initial_hover_byte': value}))
        cases.append(m.run(method, {'null_owner': True}))
    for options in [{'cold': True}, {'mutations': {'barrier:1': 'replace_character', 'box_int32:1': 'replace_literal'}}, {'null_component': True, 'null_format_result': True}]:
        m.prepare(options); rows = [m.run(name, retained=True) for name in ['Awake', 'BlockHover', 'UnblockHover', 'Awake']]
        assert all(r['returned'] for r in rows) and all(b['initial'] == a['final'] for a, b in zip(rows, rows[1:])); sequences.append(rows)
    m.prepare({'instance_id_bits': 0xFEDCBA9800000001})
    rows = [m.run('Awake', retained=True),
            m.run('Awake', {'instance_id_bits': 0x12345678FFFFFFFF, 'box_result': 'boxed1', 'format_result': 'name0'}, True),
            m.run('UnblockHover', retained=True),
            m.run('Awake', {'instance_id_bits': 0xABCDEF1280000000, 'box_result': 'boxed0', 'format_result': 'name1'}, True)]
    assert all(r['returned'] for r in rows) and all(b['initial'] == a['final'] for a, b in zip(rows, rows[1:])); sequences.append(rows)
    for options in [{'cold': True}, {'mutations': {'box_int32:1': 'replace_literal'}}, {'null_gameobject': True}]:
        baseline = m.run('Awake', options); bid = len(baselines); baselines.append(baseline); counts = {}
        for i, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run('Awake', dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:i + 1] and stopped['final'] == event['snapshot']
            stops.append({'baseline': bid, 'prefix_length': i + 1, 'stopped': stopped})
    missing = m.body_addresses - m.executed; assert missing == {0x35F495} and m.instructions[0x35F495].mnemonic == 'int3'
    return {'build': BUILD, 'targets': m.targets, 'ranges': m.ranges, 'supplied_targets': m.service_targets,
            'unpromoted_same_rva_declarations': m.same_rva,
            'metadata_bindings': {k: m.oid(v) for k, v in m.bindings.items()}, 'literal': {'slot_rva': hex(m.literal_slot), 'value': 'cardAnim_{0}'},
            'case_count': len(cases), 'cases': cases, 'retained_sequences': sequences, 'failure_baselines': baselines,
            'failure_case_count': len(stops), 'failure_stops': stops, 'instruction_assertions': len(m.checks),
            'body_instructions_decoded': len(m.body_addresses), 'body_instructions_executed': len(m.body_addresses & m.executed),
            'native_execution_addresses': len(m.executed), 'unexecuted_terminal_traps': [hex(a) for a in sorted(missing)],
            'scope': 'Complete actual CardInteraction Awake/BlockHover/UnblockHover callers. Metadata resolution, GetComponent<Character> generic entry, GameObject getter, GetInstanceID, Int32 boxing, String.Format, reference barriers and null exception are supplied. String/boxed layouts and scene identities are authored diagnostics; no generic runtime, formatter, Unity object admission or exception unwinding. Folded constructor and lifecycle registration pair excluded.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('game_root'); p.add_argument('dumper_root'); p.add_argument('--output', required=True)
    args = p.parse_args(); full = audit(args.game_root, args.dumper_root)
    memory = pool_memory(full); assert expand_memory(memory) == full
    report = pool_snapshots(memory); assert expand_snapshots(report) == memory and expand_memory(expand_snapshots(report)) == full
    Path(args.output).write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
