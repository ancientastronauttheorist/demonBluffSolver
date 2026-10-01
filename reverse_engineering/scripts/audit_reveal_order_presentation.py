"""Offline execution of two RevealOrder callers; Unity/TMP/formatting supplied.

Private native bytes are read from the pinned installed PE, never emitted.
No rendering, scheduler, DOTween or constructor implementation is inferred.
"""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine

TARGETS = {0x3A7190: ('Hide', 0x3A71B7, 0x3A71C0),
           0x3A71C0: ('Init', 0x3A721D, 0x3A7220)}
SIGNATURES = {
    'Hide': 'void RevealOrder__Hide (RevealOrder_o* __this, const MethodInfo* method);',
    'Init': 'void RevealOrder__Init (RevealOrder_o* __this, int32_t order, const MethodInfo* method);'}
SERVICES = {0x1C79FD0: 'UnityEngine.Component$$get_gameObject',
            0x1C7D810: 'UnityEngine.GameObject$$SetActive',
            0x1117320: 'System.Int32$$ToString'}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root)
        manifest = json.loads((Path(__file__).parents[1] /
            f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == manifest['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json'))
        dump = pin('dump.cs', 'dump_cs')
        declaration = re.search(r'^public class RevealOrder : MonoBehaviour // TypeDefIndex: 5735\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert declaration and 'public TextMeshProUGUI text; // 0x20' in declaration[1]
        assert declaration[1].split('// Methods')[0].count('// 0x') == 1
        tmp = re.search(r'^public abstract class TMP_Text : MaskableGraphic // TypeDefIndex: 9110\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert tmp and '// RVA: 0x1BE7620 Offset: 0x1BE6220 VA: 0x181BE7620 Slot: 66\n\tpublic virtual void set_text(string value)' in tmp[1]
        assert re.search(r'^public class TextMeshProUGUI : TMP_Text, ILayoutElement // TypeDefIndex: 8974$', dump, re.M)
        self.targets, self.service_targets, self.instructions, self.ranges = [], [], {}, {}
        for start, (name, end, following) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start and r['Name'] == 'RevealOrder$$' + name]
            assert len(rows) == 1 and rows[0]['Signature'] == SIGNATURES[name]
            assert rows[0]['TypeSignature'] == ('viii' if name == 'Init' else 'vii')
            self.targets += rows
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start) == following
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks and min(a for a, _ in chunks) == start and max(b for _, b in chunks) >= end
            self.ranges[hex(start)] = [[hex(a), hex(b)] for a, b in chunks]
            ins = list(self.cs.disasm(self.pe.get_data(start, end - start), start))
            assert sum(i.size for i in ins) == end - start and ins[-1].mnemonic == 'int3'
            assert self.pe.get_data(end, following - end) == bytes([0xCC]) * (following - end)
            self.instructions.update({i.address: i for i in ins})
        for address, name in SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1
            self.service_targets += rows
        virtual = [r for r in self.metadata['ScriptMethod'] if r['Address'] == 0x1BE7620 and r['Name'] == 'TMPro.TMP_Text$$set_text']
        assert len(virtual) == 1 and virtual[0]['Signature'] == 'void TMPro_TMP_Text__set_text (TMPro_TMP_Text_o* __this, System_String_o* value, const MethodInfo* method);'
        self.service_targets += virtual
        self.checks = {
            0x3A7194: ('xor', 'edx, edx'), 0x3A71A0: ('xor', 'r8d, r8d'),
            0x3A71A3: ('xor', 'edx, edx'), 0x3A71AC: ('jmp', '0x1c7d810'),
            0x3A71C0: ('mov', 'dword ptr [rsp + 0x10], edx'),
            0x3A71DB: ('mov', 'dl, 1'), 0x3A71E5: ('mov', 'rbx, qword ptr [rbx + 0x20]'),
            0x3A71E9: ('lea', 'rcx, [rsp + 0x38]'), 0x3A71EE: ('xor', 'edx, edx'),
            0x3A71F0: ('call', '0x1117320'), 0x3A71F5: ('test', 'rbx, rbx'),
            0x3A7203: ('mov', 'r8, qword ptr [r9 + 0x560]'),
            0x3A720A: ('call', 'qword ptr [r9 + 0x558]'), 0x3A7216: ('ret', '')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        self.p = {n: self.arena + 0x50000 + i * 0x1000 for i, n in enumerate(
            ['reveal', 'text', 'other_text', 'text_class', 'other_class', 'game', 'formatted', 'other_formatted', 'method', 'other_method'])}
        self.ids = {p: n for n, p in self.p.items()}
        self.setter, self.other_setter = self.stop + 0x100, self.stop + 0x110
        self.entry_sp = self.stack + 0x18008
        self.sizes = {n: 0x600 if n.endswith('class') else 0x80 for n in self.p}
        self.ready = False

    def oid(self, value):
        if not value: return None
        assert value in self.ids, hex(value)
        return self.ids[value]

    def snapshot(self):
        return {'text_ref': self.oid(self.rq(self.p['reveal'] + 0x20)),
                'caller_order_slot_bits': self.rq(self.entry_sp + 0x10),
                'game_active': self.game_active, 'text_values': self.text_values.copy(),
                'formatted_values': self.formatted_values.copy(),
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        self.game_active = options.get('initial_active', False)
        self.text_values = {'text': 'old supplied text', 'other_text': 'other supplied text'}
        self.formatted_values, self.allowed = {}, {}
        for name, p in self.p.items(): self.u.mem_write(p, bytes([0xA5]) * self.sizes[name])
        self.q(self.p['reveal'] + 0x20, 0 if options.get('null_text') else self.p['text'])
        for n, cls in [('text', 'text_class'), ('other_text', 'other_class')]: self.q(self.p[n], self.p[cls])
        for cls, fn, method in [('text_class', self.setter, 'method'), ('other_class', self.other_setter, 'other_method')]:
            self.q(self.p[cls] + 0x558, fn); self.q(self.p[cls] + 0x560, self.p[method])
        if options.get('alias_text_classes'): self.q(self.p['other_text'], self.p['text_class'])
        self.ready = True

    def mutate(self, phase):
        if self.options.get('mutation_phase') != phase: return
        action = self.options['mutation']
        if action in ['replace_text', 'clear_text']:
            self.q(self.p['reveal'] + 0x20, self.p['other_text'] if action == 'replace_text' else 0)
            self.allowed.setdefault('reveal', set()).update(range(0x20, 0x28))
        elif action == 'replace_text_class':
            self.q(self.p['text'], self.p['other_class']); self.allowed.setdefault('text', set()).update(range(8))
        elif action == 'replace_saved_order':
            self.d(self.entry_sp + 0x10, self.options.get('replacement_order_bits', 0xFFFFFFFF))
        else: raise AssertionError(action)

    def ret(self, value=0):
        x = self.x
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']:
            self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(x, f'UC_X86_REG_XMM{i}'), (1 << 127) | i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        self.executed.add(rva)
        if rva in self.instructions: return
        cx, dx, r8 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8']]
        if rva == 0x1C79FD0:
            assert cx in [0, self.p['reveal']] and dx == 0
            result = 0 if self.options.get('null_game') else self.p['game']
            if self.event('component_game_object_service', [self.oid(cx), self.oid(result), dx]):
                self.mutate('game_object'); self.ret(result)
        elif rva == 0x1C7D810:
            assert cx == self.p['game'] and r8 == 0
            assert dx == (0 if self.method == 'Hide' else 0xFACE123456789001)
            if self.event('set_active_service', [self.oid(cx), dx, dx & 255, r8]):
                self.game_active = bool(dx & 255); self.mutate('active'); self.ret()
        elif rva == 0x1117320:
            assert cx == self.entry_sp + 0x10 and dx == 0
            bits = self.rd(cx); signed = bits if bits < 0x80000000 else bits - 0x100000000
            result = 0 if self.options.get('null_formatted') else self.p['other_formatted' if self.options.get('other_formatted') else 'formatted']
            if self.event('int32_to_string_service', [cx - self.stack, bits, signed, dx, self.oid(result)]):
                if result: self.formatted_values[self.oid(result)] = str(signed)
                self.mutate('format'); self.ret(result)
        elif address in [self.setter, self.other_setter]:
            assert cx in [self.p['text'], self.p['other_text']] and dx in [0, self.p['formatted'], self.p['other_formatted']]
            cls = self.rq(cx)
            assert address == self.rq(cls + 0x558) and r8 == self.rq(cls + 0x560)
            if self.event('tmp_text_setter_service', [self.oid(cx), self.oid(dx), self.oid(r8)]):
                self.text_values[self.oid(cx)] = self.formatted_values.get(self.oid(dx))
                self.mutate('text'); self.ret()
        elif rva == 0x2B7D90:
            self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def invoke(self, name):
        x, sp = self.x, self.stack + 0x18008
        self.entry_sp, self.method = sp, name
        self.q(sp, self.stop)
        registers = [getattr(x, 'UC_X86_REG_' + n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, r in enumerate(registers): self.u.reg_write(r, 0xFAB00000 + i)
        for i, r in enumerate(vectors): self.u.reg_write(r, (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64))
        for r, v in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, 0 if self.options.get('null_owner') else self.p['reveal']),
                     (x.UC_X86_REG_RDX, 0xABCDEF1200000000 | self.options.get('order_bits', 0x80000003)),
                     (x.UC_X86_REG_R8, 0xDEADBEEF11111111), (x.UC_X86_REG_R9, 0xDEADBEEF22222222)]:
            self.u.reg_write(r, v)
        address = next(a for a, (n, _, _) in TARGETS.items() if n == name)
        try: self.u.emu_start(self.base + address, self.stop, timeout=10_000_000, count=1000)
        except self.unicorn.UcError as exc:
            assert self.options.get('null_owner') and name == 'Init' and not self.options.get('null_game')
            assert exc.errno == self.unicorn.UC_ERR_READ_UNMAPPED
            assert self.reg(x.UC_X86_REG_RIP) == self.base + 0x3A71E5
            self.error = 'native_null_owner_dereference'
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(r) == 0xFAB00000 + i for i, r in enumerate(registers))
            assert all(self.reg(r) == (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64) for i, r in enumerate(vectors))
        return returned

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.q(self.entry_sp + 0x10, 0x11223344AABBCCDD)
        self.allowed = {}; initial = self.snapshot(); old = len(self.events)
        returned = self.invoke(name); final = self.snapshot(); events = self.events[old:].copy()
        assert final['caller_order_slot_bits'] >> 32 == 0x11223344
        if name == 'Hide': assert final['caller_order_slot_bits'] == initial['caller_order_slot_bits']
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            assert all(i in self.allowed.get(n, set()) or b == after[i] for i, b in enumerate(before)), n
        if not self.options.get('failure') and not self.options.get('mutation_phase'):
            kinds = [e['kind'] for e in events]
            expected = ['component_game_object_service']
            if self.options.get('null_game'): expected += ['native_null_guard']
            else:
                expected += ['set_active_service']
                if name == 'Init' and not self.options.get('null_owner'):
                    expected += ['int32_to_string_service', 'native_null_guard' if self.options.get('null_text') else 'tmp_text_setter_service']
            assert kinds == expected, (name, self.options, kinds, expected)
            assert returned == (not self.options.get('null_game') and (name == 'Hide' or not self.options.get('null_owner') and not self.options.get('null_text')))
            if 'set_active_service' in kinds: assert final['game_active'] == (name == 'Init')
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'initial': initial, 'events': events, 'final': final, 'unconsumed_memory_retained': True,
                'win64_nonvolatile_and_stack_preserved_on_return': returned}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, stops, baselines = [], [], [], []
    for bits in [0, 1, 0x7FFFFFFF, 0x80000000, 0x80000003, 0xFFFFFFFF]:
        for active in [False, True]: cases.append(m.run('Init', {'order_bits': bits, 'initial_active': active}))
    for name in ['Init', 'Hide']:
        for options in [{}, {'null_text': True}, {'null_game': True}, {'null_text': True, 'null_game': True},
                        {'null_owner': True}, {'null_owner': True, 'null_game': True}, {'alias_text_classes': True},
                        {'null_formatted': True}, {'other_formatted': True}]: cases.append(m.run(name, options))
    for phase in ['game_object', 'active', 'format', 'text']:
        for action in ['replace_text', 'clear_text', 'replace_text_class', 'replace_saved_order']:
            r = m.run('Init', {'mutation_phase': phase, 'mutation': action}); cases.append(r)
            assert r['returned'] == (action != 'clear_text' or phase in ['format', 'text'])
            selected = 'other_text' if action == 'replace_text' and phase in ['game_object', 'active'] else 'text'
            if r['returned']: assert r['events'][-1]['args'][0] == selected
            if action == 'replace_saved_order':
                fmt = next(e for e in r['events'] if e['kind'] == 'int32_to_string_service')
                assert fmt['args'][1] == (0xFFFFFFFF if phase in ['game_object', 'active'] else 0x80000003)
            if action == 'replace_text_class' and phase != 'text': assert r['events'][-1]['args'][2] == 'other_method'
    for phase in ['game_object', 'active']:
        for action in ['replace_text', 'clear_text']: cases.append(m.run('Hide', {'mutation_phase': phase, 'mutation': action}))
    for alias in [False, True]:
        m.prepare({'alias_text_classes': alias}); sequences.append([m.run(n, {'order_bits': bits}, retained=True)
            for n, bits in [('Init', 0xFFFFFFFF), ('Hide', 0), ('Init', 0), ('Hide', 1)]])
    for name in ['Init', 'Hide']:
        baseline = m.run(name); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0) + 1
            r = m.run(name, {'failure': [kind, counts[kind]]}); assert not r['returned'] and r['error'] == kind
            assert r['final'] == e['snapshot']; assert r['events'] == baseline['events'][:index + 1]
            stops.append(r)
    covered = sorted(set(m.instructions) & m.executed)
    terminal = [a for a, i in m.instructions.items() if i.mnemonic == 'int3']
    assert set(m.instructions) - set(covered) == set(terminal)
    return {'schema': 'reveal_order_presentation_native_v1', 'build': BUILD,
            'scope': 'complete two RevealOrder caller bodies; services supplied; no constructor or renderer',
            'targets': m.targets, 'supplied_targets': m.service_targets, 'unwind_ranges': m.ranges,
            'field_pin': {'type': 'RevealOrder', 'type_def_index': 5735, 'text_offset': '0x20'},
            'virtual_pin': {'type': 'TMPro.TMP_Text', 'type_def_index': 9110, 'slot': 66,
                            'function_offset': '0x558', 'method_info_offset': '0x560',
                            'receiver_type': 'TMPro.TextMeshProUGUI', 'receiver_type_def_index': 8974},
            'instruction_assertions': len(m.checks), 'decoded_instructions': len(m.instructions),
            'covered_instructions': len(covered), 'terminal_int3_not_executed': [hex(a) for a in terminal],
            'literal_slots': [], 'metadata_or_class_gates': [],
            'normal_and_edge_cases': cases, 'retained_sequences': sequences, 'baselines': baselines,
            'failure_stops': stops,
            'summary': {'cases': len(cases), 'sequences': len(sequences), 'stops': len(stops)}}


def serialize_report(report):
    # Each complete row remains compact and independently parseable.
    return json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n'


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--game-root', type=Path, required=True)
    p.add_argument('--dumper-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); result = audit(args.game_root, args.dumper_root)
    args.output.parent.mkdir(parents=True, exist_ok=True); args.output.write_text(serialize_report(result), encoding='utf-8')
    print(json.dumps(result['summary'], sort_keys=True))
