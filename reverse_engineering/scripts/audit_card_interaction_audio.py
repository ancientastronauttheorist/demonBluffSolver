"""Exact RevealInteraction/TapCard callers; float RNG and AudioEvents delegates supplied."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct

from audit_character_assets import BUILD
from audit_deck_character_surface import Machine as SurfaceMachine
from audit_character_init_reward import Machine as RewardMachine
from audit_character_oracle_reveal_join import expand_memory, pool_memory
from audit_report_snapshots import expand_snapshots, pool_snapshots

TARGETS = {'RevealInteraction': (0x35FF90, 0x35FFF8, 0x360000, 'tdi5468.m0010'),
           'TapCard': (0x360000, 0x360068, 0x360070, 'tdi5468.m0009')}
FLAGS = {'RevealInteraction': 0x288C13E, 'TapCard': 0x288C13D}
FLOATS = {'RevealInteraction': [(0x1F34C60, 0x3F19999A), (0x1F34B18, 0x3F800000)],
          'TapCard': [(0x1F34C64, 0x3F4CCCCD), (0x1F34C74, 0x3FB33333)]}
POISON = [0xFACE123456789000 + i for i in range(4)]
XPOISON = [(1 << 127) | i for i in range(6)]
INCOMING = [0, 0xDEAD123400000002, 0xDEAD123400000008, 0xDEAD123400000009]
XINCOMING = [(0xABCDEF9876543210 << 64) | (0x1234000000000000 + i) for i in range(6)]


class Machine(SurfaceMachine):
    caller_return = RewardMachine.caller_return

    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        import capstone
        raw = (Path(dumper_root) / 'dump.cs').read_bytes()
        manifest = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(raw).hexdigest().upper() == manifest['outputs']['dump_cs']['sha256'].upper()
        dump = raw.decode('utf-8-sig')
        card = re.search(r'^public class CardInteraction : MonoBehaviour // TypeDefIndex: 5468\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert card and 'private void TapCard() { }' in card[1] and 'public void RevealInteraction() { }' in card[1]
        audio = re.search(r'^public static class AudioEvents // TypeDefIndex: 5524\s*\{(.*?)\n\}', dump, re.M | re.S)
        enum = re.search(r'^public enum ESFX // TypeDefIndex: 5467\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert audio and enum and 'public static Action<ESFX> OnPlaySfxOneShot; // 0x0' in audio[1]
        assert 'public int value__; // 0x0' in enum[1] and 'public const ESFX CardClick = 110;' in enum[1] and 'public const ESFX CardTap = 120;' in enum[1]
        header_raw = (Path(dumper_root) / 'il2cpp.h').read_bytes()
        assert hashlib.sha256(header_raw).hexdigest().upper() == manifest['outputs']['il2cpp_h']['sha256'].upper()
        header = header_raw.decode('utf-8-sig')
        delegate = re.search(r'struct __declspec\(align\(8\)\) System_Delegate_Fields \{(.*?)\n\};', header, re.S)
        assert delegate and [line.strip() for line in delegate[1].strip().splitlines()][:7] == ['intptr_t method_ptr;', 'intptr_t invoke_impl;', 'Il2CppObject* m_target;', 'intptr_t method;', 'intptr_t delegate_trampoline;', 'intptr_t extra_arg;', 'intptr_t method_code;']
        assert 'struct System_Action_ESFX__Fields : System_MulticastDelegate_Fields' in header
        self.instructions, self.body_addresses, self.ranges, self.targets = {}, set(), {}, []
        self.checks, self.flag_sites, self.float_pins = {}, {}, []
        for name, (start, end, following, stable) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start]
            assert len(rows) == 1 and rows[0]['Name'] == 'CardInteraction$$' + name
            assert rows[0]['Signature'] == f'void CardInteraction__{name} (CardInteraction_o* __this, const MethodInfo* method);' and rows[0]['TypeSignature'] == 'vii'
            self.targets.append(dict(rows[0], stable_method_id=stable)); self.decode(name, start, end, following)
            off = start - 0x35FF90
            selected = {0x35FFB8: ('xor', 'r8d, r8d'), 0x35FFC3: ('call', '0x1c86640'),
                0x35FFCF: ('mov', 'rcx, qword ptr [rax + 0xb8]'), 0x35FFD6: ('mov', 'rax, qword ptr [rcx]'),
                0x35FFDE: ('mov', 'r8, qword ptr [rax + 0x28]'), 0x35FFE2: ('mov', 'edx, 0x6e' if off == 0 else 'edx, 0x78'),
                0x35FFE7: ('mov', 'rcx, qword ptr [rax + 0x40]'), 0x35FFEF: ('jmp', 'qword ptr [rax + 0x18]')}
            self.checks.update({a + off: v for a, v in selected.items()})
            assert self.instructions[start + 0xB].mnemonic == 'jne' and self.instructions[start + 0xB].op_str == hex(start + 0x20)
            assert self.instructions[start + 0x14].mnemonic == 'call' and self.instructions[start + 0x14].op_str == '0x2b7b40'
            store = self.instructions[start + 0x19]
            assert store.mnemonic == 'mov' and store.operands[0].size == 1 and store.operands[1].imm == 1
            assert store.address + store.size + store.operands[0].mem.disp == FLAGS[name]
            assert self.instructions[start + 0x4C].mnemonic == 'je' and self.instructions[start + 0x4C].op_str == hex(start + 0x63)
            flag_ins = self.instructions[start + 4]
            assert flag_ins.mnemonic == 'cmp' and flag_ins.operands[0].size == 1 and flag_ins.operands[0].mem.base == capstone.x86.X86_REG_RIP
            assert flag_ins.address + flag_ins.size + flag_ins.operands[0].mem.disp == FLAGS[name]
            for pc, expected in zip([start + 0x2B, start + 0x20], FLOATS[name]):
                ins = self.instructions[pc]; operand = ins.operands[1]
                assert ins.mnemonic == 'movss' and operand.mem.base == capstone.x86.X86_REG_RIP
                slot = ins.address + ins.size + operand.mem.disp; assert slot == expected[0]
                section = self.pe.get_section_by_rva(slot); assert section and slot + 4 - section.VirtualAddress <= section.SizeOfRawData
                value = self.pe.get_data(slot, 4); assert len(value) == 4 and struct.unpack('<I', value)[0] == expected[1]
                self.float_pins.append({'method': name, 'instruction_rva': hex(pc), 'slot_rva': hex(slot), 'bits': expected[1], 'float_value': struct.unpack('<f', value)[0]})
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        assert all(e.unwindinfo.Flags == 0 for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress in [v[0] for v in TARGETS.values()])
        rows = [r for r in self.metadata['ScriptMetadata'] if r['Name'] == 'AudioEvents_TypeInfo']; assert len(rows) == 1
        self.slot = rows[0]['Address']; assert rows[0]['Signature'] == 'AudioEvents_c*'
        for name, (start, _, _, _) in TARGETS.items():
            ins = self.instructions[start + 0xD]; assert ins.mnemonic == 'lea' and ins.address + ins.size + ins.operands[1].mem.disp == self.slot
            ins = self.instructions[start + 0x38]; assert ins.mnemonic == 'mov' and ins.address + ins.size + ins.operands[1].mem.disp == self.slot
        rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == 0x1C86640]; assert len(rows) == 1
        self.service_targets = rows
        assert rows[0]['Signature'] == 'float UnityEngine_Random__Range (float minInclusive, float maxInclusive, const MethodInfo* method);' and rows[0]['TypeSignature'] == 'fffi'
        names = ['owner', 'owner_class', 'audio_class', 'other_class', 'statics', 'other_statics', 'action0', 'action1', 'action_class', 'target0', 'target1', 'method0', 'method1']
        self.p = {n: self.arena + 0x240000 + i * 0x1000 for i, n in enumerate(names)}
        self.callbacks = [self.stop + 0x100, self.stop + 0x200]
        self.labels = {0: None, **{p: n for n, p in self.p.items()}, **{v: 'callback' + str(i) for i, v in enumerate(self.callbacks)}}
        self.sizes = {n: 256 if n.endswith('_class') else 128 for n in names}
        self.entry_sp = self.stack + 0x18008

    def snapshot(self):
        return {'metadata_flags': {n: self.u.mem_read(self.base + a, 1)[0] for n, a in FLAGS.items()},
                'metadata_slot': self.oid(self.rq(self.base + self.slot)), 'native_entries': deepcopy(self.entries),
                'service_history': deepcopy(self.history), 'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options = deepcopy(options); self.entries, self.history, self.events, self.counts = [], [], [], {}
        for n, p in self.p.items(): self.u.mem_write(p, b'\xA5' * self.sizes[n])
        for n in ['owner', 'target0', 'target1', 'method0', 'method1']: self.q(self.p[n], self.p['owner_class']); self.q(self.p[n] + 8, 0)
        for n, block in [('audio_class', 'statics'), ('other_class', 'other_statics')]: self.q(self.p[n] + 0xB8, self.p[block])
        for block, action in [('statics', 'action0'), ('other_statics', 'action1')]: self.q(self.p[block], 0 if options.get('null_callback') else self.p[action])
        for i in range(2):
            p = self.p['action' + str(i)]; self.q(p, self.p['action_class']); self.q(p + 8, 0)
            for off, value in [(0x18, self.callbacks[i]), (0x20, self.p['target' + str(1 - i)]), (0x28, self.p['method' + str(i)]), (0x40, self.p['target' + str(i)])]: self.q(p + off, value)
        self.q(self.base + self.slot, 0 if options.get('null_class') else self.p['audio_class'])
        if options.get('null_statics'): self.q(self.p['audio_class'] + 0xB8, 0)
        for n, a in FLAGS.items(): self.u.mem_write(self.base + a, bytes([0 if options.get('cold') else options.get('warm_flag', 1)]))

    def mutate(self, kind, relative):
        actions = self.options.get('mutations', {}).get(kind + ':' + str(relative), [])
        for name, offset, target in actions:
            if name == 'metadata_slot': self.q(self.base + self.slot, self.p[target] if target else 0)
            else:
                self.q(self.p[name] + offset, self.labels_to_pointer(target)); self.allow(name, offset, 8)

    def labels_to_pointer(self, label):
        return 0 if label is None else (self.callbacks[int(label[-1])] if label.startswith('callback') else self.p[label])

    def abi(self):
        return {'raw_args': [self.reg(getattr(self.x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']],
                'raw_xmm': [self.reg(getattr(self.x, 'UC_X86_REG_XMM' + str(i))) for i in range(6)],
                'caller_return': self.caller_return(self.rq(self.reg(self.x.UC_X86_REG_RSP)))}

    def event(self, kind, args):
        ordinal = self.counts.get(kind, 0) + 1; self.counts[kind] = ordinal
        self.events.append({'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': self.abi(), 'snapshot': self.snapshot()})
        relative = ordinal - self.entry_counts.get(kind, 0)
        if self.options.get('failure') == [kind, relative]: self.error = kind; self.u.emu_stop(); return False
        self.mutate(kind, relative); return True

    def hook(self, uc, address, size, data):
        if address == self.stop: return
        rva = address - self.base; self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(self.x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == TARGETS[self.method][0]: self.entries.append({'method': self.method, **self.abi()})
        if rva in self.instructions: return
        if rva == 0x2B7B40:
            assert cx == self.base + self.slot
            if self.event('metadata', [hex(self.slot), self.oid(self.rq(cx))]): self.history.append(['metadata', self.oid(self.rq(cx))]); self.ret(self.rq(cx))
        elif rva == 0x1C86640:
            xmm = [self.reg(getattr(self.x, 'UC_X86_REG_XMM' + str(i))) for i in range(2)]
            assert xmm == [v[1] for v in FLOATS[self.method]] and r8 == 0
            bits = self.options.get('random_return_xmm0', 0x8877665544332211000000003F400000)
            if self.event('random_float', [*xmm, bits]):
                self.history.append(['random_float', *xmm, bits]); self.ret(); self.u.reg_write(self.x.UC_X86_REG_XMM0, bits)
        elif address in self.callbacks:
            action = self.reg(self.x.UC_X86_REG_RAX)
            assert cx == self.rq(action + 0x40) and dx == (110 if self.method == 'RevealInteraction' else 120) and r8 == self.rq(action + 0x28)
            args = [self.oid(action), self.oid(address), self.oid(cx), dx, self.oid(r8)]
            if self.event('audio_callback', args): self.history.append(['audio_callback', *args]); self.ret()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def run(self, method, options=None, retained=False):
        if not retained: self.prepare(options or {})
        elif options is not None: self.options.update(deepcopy(options))
        self.method, self.error, self.fault, self.allowed = method, None, None, {}
        self.entry_counts = self.counts.copy(); initial, old = self.snapshot(), len(self.events)
        x, sp = self.x, self.entry_sp; self.q(sp, self.stop)
        raw = [0 if self.options.get('null_owner') else self.p['owner'], *INCOMING[1:]]
        for n, v in zip(['RCX', 'RDX', 'R8', 'R9'], raw): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), v)
        for i, v in enumerate(XINCOMING): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), v)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFAB0000000000000 + i)
        for i in range(6, 16): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), (0xABCDEF9876543210 << 64) | i)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        try: self.u.emu_start(self.base + TARGETS[method][0], self.stop, timeout=10_000_000, count=10000)
        except self.unicorn.UcError as exc:
            assert exc.errno == self.unicorn.UC_ERR_READ_UNMAPPED
            pc = self.reg(x.UC_X86_REG_RIP) - self.base
            assert pc in [TARGETS[method][0] + 0x3F, TARGETS[method][0] + 0x46]
            self.error, self.fault = 'native_audio_storage_fault', hex(pc)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): assert self.reg(getattr(x, 'UC_X86_REG_' + n)) == 0xFAB0000000000000 + i
            for i in range(6, 16): assert self.reg(getattr(x, 'UC_X86_REG_XMM' + str(i))) == (0xABCDEF9876543210 << 64) | i
        final = self.snapshot()
        for n, before in initial['memory'].items():
            before, after = bytes.fromhex(before), bytes.fromhex(final['memory'][n])
            assert all(i in self.allowed.get(n, set()) or b == after[i] for i, b in enumerate(before)), n
        for pin in self.float_pins:
            assert self.rd(self.base + int(pin['slot_rva'], 16)) == pin['bits']
        row = {'method': method, 'options': deepcopy(self.options), 'entry_raw_args': raw, 'entry_raw_xmm': XINCOMING.copy(),
               'returned': returned, 'error': self.error, 'fault_rva': self.fault, 'initial': initial, 'events': deepcopy(self.events[old:]), 'final': final,
               'service_counts_before': self.entry_counts.copy(), 'service_counts_after': self.counts.copy(),
               'nonvolatile_abi_verified': returned, 'reached_only_storage_retention_verified': True}
        self.verify(row); row['independent_ordered_model_verified'] = True; return row

    def verify(self, row):
        model, options, counts = deepcopy(row['initial']), row['options'], row['service_counts_before'].copy()
        raw, xmm = row['entry_raw_args'].copy(), row['entry_raw_xmm'].copy()
        model['native_entries'].append({'method': row['method'], 'raw_args': raw.copy(), 'raw_xmm': xmm.copy(), 'caller_return': self.caller_return(self.stop)})
        index, error, fault, returned = 0, None, None, False
        class Stopped(Exception): pass
        def read(n, off): return int.from_bytes(bytes.fromhex(model['memory'][n])[off:off + 8], 'little')
        def oid(v): return self.oid(v)
        def mutate(kind, relative):
            for n, off, target in options.get('mutations', {}).get(kind + ':' + str(relative), []):
                if n == 'metadata_slot': model['metadata_slot'] = target
                else:
                    data = bytearray.fromhex(model['memory'][n]); data[off:off + 8] = self.labels_to_pointer(target).to_bytes(8, 'little'); model['memory'][n] = data.hex()
        def emit(kind, args, caller):
            nonlocal index, raw, xmm, error
            ordinal = counts.get(kind, 0) + 1; counts[kind] = ordinal
            expected = {'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': {'raw_args': raw.copy(), 'raw_xmm': xmm.copy(), 'caller_return': self.caller_return(caller)}, 'snapshot': model}
            assert row['events'][index] == expected, (row['method'], kind, index)
            index += 1; relative = ordinal - row['service_counts_before'].get(kind, 0)
            if options.get('failure') == [kind, relative]: error = kind; raise Stopped()
            mutate(kind, relative); raw, xmm = POISON.copy(), XPOISON.copy()
        method, start = row['method'], TARGETS[row['method']][0]
        try:
            if model['metadata_flags'][method] == 0:
                raw[0] = self.base + self.slot
                emit('metadata', [hex(self.slot), model['metadata_slot']], self.base + start + 0x19)
                model['service_history'].append(['metadata', model['metadata_slot']]); model['metadata_flags'][method] = 1
            xmm[0], xmm[1] = [v[1] for v in FLOATS[method]]; raw[2] = 0
            bits = options.get('random_return_xmm0', 0x8877665544332211000000003F400000)
            emit('random_float', [*xmm[:2], bits], self.base + start + 0x38)
            model['service_history'].append(['random_float', *[v[1] for v in FLOATS[method]], bits]); xmm[0] = bits
            cls = model['metadata_slot']
            if cls is None: error, fault = 'native_audio_storage_fault', hex(start + 0x3F)
            else:
                static = oid(read(cls, 0xB8))
                if static is None: error, fault = 'native_audio_storage_fault', hex(start + 0x46)
                else:
                    action = oid(read(static, 0))
                    if action is not None:
                        callback, code, info = oid(read(action, 0x18)), oid(read(action, 0x40)), oid(read(action, 0x28))
                        value = 110 if method == 'RevealInteraction' else 120
                        raw = [self.labels_to_pointer(code), value, self.labels_to_pointer(info), raw[3]]
                        args = [action, callback, code, value, info]; emit('audio_callback', args, self.stop)
                        model['service_history'].append(['audio_callback', *args])
                    returned = True
        except Stopped: pass
        assert index == len(row['events']) and (returned, error, fault) == (row['returned'], row['error'], row['fault_rva'])
        assert model == row['final'] and counts == row['service_counts_after']


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    actions = [[['metadata_slot', 0, 'other_class']], [['audio_class', 0xB8, 'other_statics']], [['statics', 0, 'action1']],
               [['statics', 0, None]], [['statics', 0, 'action1'], ['action1', 0x40, 'target0'], ['action1', 0x28, 'method0'], ['action1', 0x18, 'callback0']],
               [['metadata_slot', 0, None]], [['audio_class', 0xB8, None]],
               [['action0', 0x40, None], ['action0', 0x28, None]],
               [['action0', 0x40, 'method1'], ['action0', 0x28, 'target1'], ['action0', 0x20, 'target0']]]
    for method in TARGETS:
        for cold, null in itertools.product([False, True], repeat=2): cases.append(m.run(method, {'cold': cold, 'null_callback': null}))
        for flag in [0x80, 0xFF]: cases.append(m.run(method, {'warm_flag': flag}))
        for bits in [0, 0x80000000, 0x7FC01234, 0x7F800000, 0xFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF]: cases.append(m.run(method, {'random_return_xmm0': bits}))
        for option in ['null_owner', 'null_class', 'null_statics']: cases.append(m.run(method, {option: True}))
        for kind in ['metadata', 'random_float', 'audio_callback']:
            for mutation in actions: cases.append(m.run(method, {'cold': True, 'mutations': {kind + ':1': mutation}}))
    for options in [{'cold': True}, {'cold': True, 'null_owner': True},
                    {'cold': True, 'mutations': {'audio_callback:1': [['metadata_slot', 0, 'other_class'], ['other_statics', 0, 'action0']]}},
                    {'mutations': {'random_float:1': [['statics', 0, None], ['other_statics', 0, 'action0'], ['metadata_slot', 0, 'other_class']]}}]:
        m.prepare(options); rows = [m.run(name, retained=True) for name in ['RevealInteraction', 'TapCard', 'RevealInteraction', 'TapCard']]
        assert all(r['returned'] for r in rows) and all(b['initial'] == a['final'] for a, b in zip(rows, rows[1:])); sequences.append(rows)
    for kind in ['metadata', 'random_float', 'audio_callback']:
        m.prepare({'cold': True, 'failure': [kind, 1], 'mutations': {'audio_callback:1': [['statics', 0, 'action1']]}})
        rows = [m.run('RevealInteraction', retained=True), m.run('RevealInteraction', {'failure': None}, True), m.run('TapCard', retained=True)]
        assert not rows[0]['returned'] and all(r['returned'] for r in rows[1:])
        assert all(b['initial'] == a['final'] for a, b in zip(rows, rows[1:])); sequences.append(rows)
    for method in TARGETS:
        for options in [{'cold': True}, {'cold': True, 'null_callback': True}, {'cold': True, 'mutations': {'random_float:1': actions[4]}}, {'cold': True, 'null_class': True}, {'cold': True, 'null_statics': True}]:
            baseline = m.run(method, options); bid = len(baselines); baselines.append(baseline); counts = {}
            for i, event in enumerate(baseline['events']):
                kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
                stopped = m.run(method, dict(options, failure=[kind, counts[kind]]))
                assert not stopped['returned'] and stopped['events'] == baseline['events'][:i + 1] and stopped['final'] == event['snapshot']
                stops.append({'baseline': bid, 'prefix_length': i + 1, 'stopped': stopped})
    assert m.body_addresses <= m.executed
    return {'build': BUILD, 'targets': m.targets, 'ranges': m.ranges, 'supplied_targets': m.service_targets,
            'float_literals': m.float_pins, 'metadata_slot_rva': hex(m.slot), 'metadata_flag_rvas': {k: hex(v) for k, v in FLAGS.items()},
            'delegate_fields': {'invoke_impl': 0x18, 'm_target_unused': 0x20, 'method': 0x28, 'method_code': 0x40},
            'case_count': len(cases), 'cases': cases, 'retained_sequences': sequences, 'failure_baselines': baselines,
            'failure_case_count': len(stops), 'failure_stops': stops, 'instruction_assertions': len(m.checks),
            'body_instructions_decoded': len(m.body_addresses), 'body_instructions_executed': len(m.body_addresses & m.executed), 'native_execution_addresses': len(m.executed),
            'scope': 'Complete exact CardInteraction RevealInteraction/TapCard callers. Metadata, whole float Random.Range and whole Action<ESFX> callback are supplied. RNG return XMM0 is discarded semantically but retained as full raw callback-entry bits; no callback float argument. No delegate/runtime/audio implementation or Unity object admission. Both single unwind entries have Flags=0; invalid authored class/static accesses stop at native read faults without unwinding. No owner dereference. Constructor/other interaction bodies excluded.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('game_root'); p.add_argument('dumper_root'); p.add_argument('--output', required=True)
    args = p.parse_args(); full = audit(args.game_root, args.dumper_root)
    memory = pool_memory(full); assert expand_memory(memory) == full
    report = pool_snapshots(memory); assert expand_snapshots(report) == memory and expand_memory(expand_snapshots(report)) == full
    Path(args.output).write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
