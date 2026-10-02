"""Offline InGameSettings callers; input and GameObject services supplied."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine


TARGETS = {'Update': (0x3A22A0, 0x3A22EE, 0x3A22F0, 'tdi5769.m0000'),
           'OnEnable': (0x3A2270, 0x3A2291, 0x3A22A0, 'tdi5769.m0001'),
           'ManageShowSettings': (0x3A2220, 0x3A226F, 0x3A2270, 'tdi5769.m0002')}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root)
        extraction = json.loads((Path(__file__).parents[1] /
            f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json'))
        dump = pin('dump.cs', 'dump_cs')
        block = re.search(r'^public class InGameSettings : MonoBehaviour // TypeDefIndex: 5769\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert block and block[1].split('// Methods')[0].strip() == '// Fields\n\tpublic GameObject settings; // 0x20'
        keycode = re.search(r'^public enum KeyCode // TypeDefIndex: 6685\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert keycode and 'public const KeyCode Escape = 27;' in keycode[1]
        self.targets, self.instructions, self.bounds, self.ranges = [], {}, {}, {}
        for name, (start, end, following, mid) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start and r['Name'] == 'InGameSettings$$' + name]
            assert len(rows) == 1 and rows[0]['TypeSignature'] == 'vii'
            assert rows[0]['Signature'] == f'void InGameSettings__{name} (InGameSettings_o* __this, const MethodInfo* method);'
            self.targets.append(dict(rows[0], method_id=mid))
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start) == following
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks == [(start, end)]
            self.ranges[name] = [[hex(a), hex(b)] for a, b in chunks]
            section = self.pe.get_section_by_rva(start)
            assert section and following <= section.VirtualAddress + section.SizeOfRawData
            raw = self.pe.get_data(start, following - start)
            assert len(raw) == following - start and raw[end-start:] == bytes([0xCC]) * (following-end)
            ins = list(self.cs.disasm(raw[:end-start], start))
            assert sum(i.size for i in ins) == end-start and ins[-1].mnemonic == 'int3'
            self.instructions.update({i.address: i for i in ins})
            self.bounds[name] = {'start': hex(start), 'end_exclusive': hex(end), 'next_managed': hex(following),
                                 'padding_bytes': following-end}
        self.enable_aliases = [r for r in self.metadata['ScriptMethod'] if r['Address'] == TARGETS['OnEnable'][0]]
        assert len(self.enable_aliases) == 2 and {r['Name'] for r in self.enable_aliases} == {'InGameSettings$$OnEnable', 'NightStep$$Disable'}
        self.supplied = []
        signatures = {
            'UnityEngine.Input$$GetKeyDown': (0x1CD3C80, 'bool UnityEngine_Input__GetKeyDown (int32_t key, const MethodInfo* method);', 'iii'),
            'UnityEngine.GameObject$$get_activeSelf': (0x1C7DC50, 'bool UnityEngine_GameObject__get_activeSelf (UnityEngine_GameObject_o* __this, const MethodInfo* method);', 'iii'),
            'UnityEngine.GameObject$$SetActive': (0x1C7D810, 'void UnityEngine_GameObject__SetActive (UnityEngine_GameObject_o* __this, bool value, const MethodInfo* method);', 'viii')}
        for name, (rva, signature, typesig) in signatures.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == rva and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['Signature'] == signature and rows[0]['TypeSignature'] == typesig
            self.supplied += rows
        self.input_aliases = [r for r in self.metadata['ScriptMethod'] if r['Address'] == 0x1CD3C80]
        assert {r['Name'] for r in self.input_aliases} == {'UnityEngine.Input$$GetKeyDown', 'UnityEngine.Input$$GetKeyDownInt'}
        self.checks = {
            0x3A2274: ('mov', 'rcx, qword ptr [rcx + 0x20]'),
            0x3A227D: ('xor', 'r8d, r8d'), 0x3A2280: ('xor', 'edx, edx'),
            0x3A2286: ('jmp', '0x1c7d810'),
            0x3A2234: ('call', '0x1c7dc50'), 0x3A2239: ('mov', 'rcx, qword ptr [rbx + 0x20]'),
            0x3A223D: ('test', 'al, al'), 0x3A2249: ('xor', 'edx, edx'),
            0x3A225D: ('mov', 'dl, 1'),
            0x3A22AB: ('lea', 'ecx, [rdx + 0x1b]'), 0x3A22AE: ('call', '0x1cd3c80'),
            0x3A22B3: ('test', 'al, al'), 0x3A22B7: ('mov', 'rcx, qword ptr [rbx + 0x20]'),
            0x3A22C2: ('call', '0x1c7dc50'), 0x3A22C7: ('mov', 'rcx, qword ptr [rbx + 0x20]'),
            0x3A22D0: ('test', 'al, al'), 0x3A22D2: ('sete', 'dl'),
            0x3A22D5: ('xor', 'r8d, r8d'), 0x3A22DD: ('jmp', '0x1c7d810'),
            0x3A22E7: ('ret', '')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        for name, (start, end, _, _) in TARGETS.items():
            calls = [i.op_str for i in self.instructions.values() if start <= i.address < end and i.mnemonic == 'call']
            assert calls.count('0x1cd3c80') == int(name == 'Update')
            assert calls.count('0x1c7dc50') == int(name != 'OnEnable')
            assert calls.count('0x2b7d90') == 1
        self.p = {n: self.arena + 0x60000 + i * 0x1000 for i, n in enumerate(['owner', 'settings', 'other', 'class'])}
        self.ids = {p: n for n, p in self.p.items()}
        self.entry_sp = self.stack + 0x18008

    def oid(self, value):
        if not value: return None
        assert value in self.ids, hex(value)
        return self.ids[value]

    def snapshot(self):
        return {'settings_ref': self.oid(self.rq(self.p['owner'] + 0x20)), 'games': self.games.copy(),
                'key_requests': self.key_requests.copy(), 'active_reads': self.active_reads.copy(),
                'active_writes': self.active_writes.copy(),
                'memory': {n: bytes(self.u.mem_read(p, 0x80)).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        for n, p in self.p.items():
            self.u.mem_write(p, bytes([0xA5]) * 0x80)
            if n != 'class': self.q(p, self.p['class']); self.q(p + 8, 0)
        setting = options.get('initial_receiver', 'settings')
        assert setting in ['settings', 'other']
        self.q(self.p['owner'] + 0x20, 0 if options.get('null_settings') else self.p[setting])
        self.games = {'settings': options.get('settings_active', False), 'other': options.get('other_active', True)}
        self.key_requests, self.active_reads, self.active_writes = [], [], []
        assert all(0 <= options.get(n, default) <= 255 for n, default in [('key_true_byte', 0x80), ('active_true_byte', 0xFE)])

    def mutate(self, phase, captured=0):
        if self.options.get('mutation_phase') != phase: return
        action = self.options['mutation']
        if action == 'replace_settings': value = self.p['other']
        elif action == 'restore_settings': value = self.p['settings']
        elif action == 'same_receiver': value = captured
        elif action == 'clear_settings': value = 0
        else: raise AssertionError(action)
        assert value in [0, self.p['settings'], self.p['other']]
        self.q(self.p['owner'] + 0x20, value)

    def ret(self, value=0):
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']:
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_' + n), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{i}'), (1 << 127) | i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        self.executed.add(rva)
        if rva in self.instructions: return
        cx, dx, r8 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8']]
        if rva == 0x1CD3C80:
            assert cx == 27 and dx == 0
            low = self.options.get('key_true_byte', 0x80) if self.options.get('key_pressed', True) else 0
            result = 0xFACE123456789000 | low
            if self.event('key_down_service', [cx, result, dx]):
                self.key_requests.append([cx, result]); self.mutate('key', self.rq(self.p['owner'] + 0x20)); self.ret(result)
        elif rva == 0x1C7DC50:
            assert cx in [self.p['settings'], self.p['other']] and dx == 0
            low = self.options.get('active_true_byte', 0xFE) if self.games[self.oid(cx)] else 0
            result = 0xFACE123456789000 | low
            if self.event('active_self_service', [self.oid(cx), result, dx]):
                self.active_reads.append([self.oid(cx), result]); self.mutate('read', cx); self.ret(result)
        elif rva == 0x1C7D810:
            assert cx in [self.p['settings'], self.p['other']] and r8 == 0
            if self.method == 'OnEnable': expected = 0
            else:
                value = int(self.active_reads[-1][1] & 255 == 0)
                expected = 0xFACE123456789000 | value if self.method == 'Update' or value else 0
            assert dx == expected, (self.method, hex(dx), hex(expected))
            if self.event('set_active_service', [self.oid(cx), dx, dx & 255, r8]):
                self.games[self.oid(cx)] = bool(dx & 255); self.active_writes.append([self.oid(cx), dx, dx & 255])
                self.mutate('write', cx); self.ret()
        elif rva == 0x2B7D90:
            self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def invoke(self, name):
        self.method = name
        x, sp = self.x, self.entry_sp
        self.q(sp, self.stop); self.u.mem_write(sp + 8, bytes([0xB6]) * 0x38)
        regs = [getattr(x, 'UC_X86_REG_' + n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, r in enumerate(regs): self.u.reg_write(r, 0xFAB00000 + i)
        for i, r in enumerate(vectors): self.u.reg_write(r, (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64))
        for r, v in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, 0 if self.options.get('null_owner') else self.p['owner']),
                     (x.UC_X86_REG_RDX, 0xDEADBEEF12345678), (x.UC_X86_REG_R8, 0xDEADBEEF11111111),
                     (x.UC_X86_REG_R9, 0xDEADBEEF22222222)]: self.u.reg_write(r, v)
        try: self.u.emu_start(self.base + TARGETS[name][0], self.stop, timeout=10_000_000, count=1000)
        except self.unicorn.UcError as exc:
            assert self.options.get('null_owner') and exc.errno == self.unicorn.UC_ERR_READ_UNMAPPED
            assert self.reg(x.UC_X86_REG_RIP) == self.base + {'OnEnable': 0x3A2274, 'ManageShowSettings': 0x3A2229, 'Update': 0x3A22B7}[name]
            self.error = 'native_owner_access_fault'
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(r) == 0xFAB00000 + i for i, r in enumerate(regs))
            assert all(self.reg(r) == (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64) for i, r in enumerate(vectors))
        return returned

    def expected_normal(self, name, initial):
        games, events = initial['games'].copy(), []
        target = initial['settings_ref']
        if name == 'Update':
            key_result = 0xFACE123456789000 | (self.options.get('key_true_byte', 0x80) if self.options.get('key_pressed', True) else 0)
            events.append({'kind': 'key_down_service', 'args': [27, key_result, 0]})
            if key_result & 255 == 0: return events, games
        if name == 'OnEnable': value, raw = 0, 0
        else:
            result = 0xFACE123456789000 | (self.options.get('active_true_byte', 0xFE) if games[target] else 0)
            events.append({'kind': 'active_self_service', 'args': [target, result, 0]})
            value = int(result & 255 == 0)
            raw = 0xFACE123456789000 | value if name == 'Update' or value else 0
        events.append({'kind': 'set_active_service', 'args': [target, raw, value, 0]})
        games[target] = bool(value)
        return events, games

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        initial, old = self.snapshot(), len(self.events)
        returned = self.invoke(name)
        final, events = self.snapshot(), self.events[old:].copy()
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            allowed = range(0x20, 0x28) if n == 'owner' and self.options.get('mutation_phase') else []
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        if returned and not self.options.get('mutation_phase'):
            expected, games = self.expected_normal(name, initial)
            assert [{'kind': e['kind'], 'args': e['args']} for e in events] == expected
            assert final['games'] == games
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'initial': initial, 'events': events, 'final': final,
                'win64_nonvolatile_and_stack_preserved_on_return': returned, 'unconsumed_memory_retained': True}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, sequences, baselines, stops = [], [], [], []
    for name, settings_active, other_active, receiver in itertools.product(TARGETS, [False, True], [False, True], ['settings', 'other']):
        cases.append(m.run(name, {'settings_active': settings_active, 'other_active': other_active, 'initial_receiver': receiver}))
    for name in ['Update', 'ManageShowSettings']:
        for key_byte, active_byte, active in itertools.product([0, 1, 0x80, 0xFF], [0, 1, 0x80, 0xFF], [False, True]):
            cases.append(m.run(name, {'key_true_byte': key_byte, 'active_true_byte': active_byte, 'settings_active': active}))
    for name in TARGETS:
        for options in [{'null_owner': True}, {'null_owner': True, 'key_pressed': False},
                        {'null_settings': True}, {'null_settings': True, 'key_pressed': False},
                        {'key_pressed': False}]: cases.append(m.run(name, options))
    for name, phases in [('Update', ['key', 'read', 'write']), ('ManageShowSettings', ['read', 'write']), ('OnEnable', ['write'])]:
        for phase, action, receiver in itertools.product(phases, ['replace_settings', 'restore_settings', 'clear_settings', 'same_receiver'], ['settings', 'other']):
            r = m.run(name, {'mutation_phase': phase, 'mutation': action, 'initial_receiver': receiver, 'settings_active': False, 'other_active': True})
            kinds = [e['kind'] for e in r['events']]
            if action == 'clear_settings' and phase in ['key', 'read']:
                assert not r['returned'] and r['error'] == 'native_null_guard'
            else: assert r['returned']
            if phase == 'read' and action != 'clear_settings':
                read = next(e for e in r['events'] if e['kind'] == 'active_self_service')
                write = next(e for e in r['events'] if e['kind'] == 'set_active_service')
                expected_receiver = 'other' if action == 'replace_settings' else 'settings' if action == 'restore_settings' else receiver
                assert read['args'][0] == receiver and write['args'][0] == expected_receiver
                assert write['args'][2] == int(read['args'][1] & 255 == 0)
            if phase == 'key' and action != 'clear_settings':
                expected_receiver = 'other' if action == 'replace_settings' else 'settings' if action == 'restore_settings' else receiver
                assert next(e for e in r['events'] if e['kind'] == 'active_self_service')['args'][0] == expected_receiver
            cases.append(r)
    for receiver in ['settings', 'other']:
        m.prepare({'initial_receiver': receiver, 'settings_active': True, 'other_active': False})
        sequences.append([m.run(n, o, retained=True) for n, o in [
            ('OnEnable', {}), ('Update', {}), ('ManageShowSettings', {}), ('Update', {'key_pressed': False}),
            ('ManageShowSettings', {}), ('OnEnable', {}), ('Update', {})]])
        m.prepare({'initial_receiver': receiver, 'settings_active': False, 'other_active': True})
        sequences.append([m.run(n, o, retained=True) for n, o in [
            ('ManageShowSettings', {'mutation_phase': 'read', 'mutation': 'replace_settings'}),
            ('Update', {}), ('ManageShowSettings', {'mutation_phase': 'read', 'mutation': 'restore_settings'}),
            ('OnEnable', {})]])
    assert all(r['returned'] for sequence in sequences for r in sequence)
    baseline_profiles = [(n, {'settings_active': active}) for n, active in itertools.product(TARGETS, [False, True])]
    baseline_profiles += [('Update', {'key_pressed': False}),
                          ('Update', {'mutation_phase': 'read', 'mutation': 'replace_settings'}),
                          ('ManageShowSettings', {'mutation_phase': 'read', 'mutation': 'replace_settings'})]
    for name, options in baseline_profiles:
        baseline = m.run(name, options); assert baseline['returned']
        baseline_id = len(baselines); baselines.append(baseline); counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(name, {**options, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index+1]
            assert stopped['final'] == event['snapshot']
            stops.append({'baseline': baseline_id, 'prefix_length': index+1, 'result': stopped})
    missing = sorted(set(m.instructions) - m.executed)
    assert missing == [0x3A226E, 0x3A2290, 0x3A22ED]
    assert all(m.instructions[a].mnemonic == 'int3' for a in missing)
    return {'schema': 'in_game_settings_native_v1', 'build': BUILD, 'targets': m.targets,
            'scope': 'three complete exact InGameSettings callers; Input and GameObject services supplied; no NightStep alias promotion',
            'supplied_targets': m.supplied, 'input_gateway_aliases': m.input_aliases,
            'on_enable_shared_aliases': m.enable_aliases, 'field_pin': {'type': 'InGameSettings', 'type_def_index': 5769, 'settings': '0x20'},
            'key_pin': {'type': 'UnityEngine.KeyCode', 'type_def_index': 6685, 'member': 'Escape', 'value': 27},
            'body_bounds': m.bounds, 'unwind_ranges': m.ranges,
            'instruction_assertions': len(m.checks), 'decoded_instructions': len(m.instructions),
            'covered_instructions': len(set(m.instructions) & m.executed), 'unexecuted_terminal_traps': [hex(a) for a in missing],
            'diagnostic_window_bytes_not_object_extent': 0x80,
            'cases': cases, 'retained_sequences': sequences, 'baselines': baselines, 'failure_stops': stops,
            'summary': {'cases': len(cases), 'sequences': len(sequences), 'baselines': len(baselines), 'stops': len(stops)}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--game-root', type=Path, required=True)
    p.add_argument('--dumper-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = audit(args.game_root, args.dumper_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], sort_keys=True))
