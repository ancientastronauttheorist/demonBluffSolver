"""Execute exact CardTokens callers offline; keyboard/Unity effects supplied."""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_oracle_reveal_join import pool_memory

TARGETS = {0x3971A0: ('OnEnable', 'tdi5734.m0000', 0x3971FF),
           0x397200: ('Update', 'tdi5734.m0001', 0x397436)}
KEYS = [53, 49, 51, 50, 52]
FIELDS = {'character': 0x20, 'good': 0x28, 'excl': 0x30, 'unsure': 0x38, 'bad': 0x40}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root)
        extraction = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pinned(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pinned('script.json', 'script_json'))
        dump = pinned('dump.cs', 'dump_cs')
        self.targets, self.instructions = [], {}
        self.body_bounds = {}
        for start, (name, mid, end) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == 'CardTokens$$' + name and r['Address'] == start]
            assert len(rows) == 1 and rows[0]['TypeSignature'] == 'vii'
            assert rows[0]['Signature'] == f'void CardTokens__{name} (CardTokens_o* __this, const MethodInfo* method);'
            self.targets.append(dict(rows[0], method_id=mid))
            following = min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start)
            section = self.pe.get_section_by_rva(start)
            assert section and following <= section.VirtualAddress + section.SizeOfRawData
            raw = self.pe.get_data(start, following - start)
            assert len(raw) == following - start and raw[end-start:] == bytes([0xCC]) * (following-end)
            ins = list(self.cs.disasm(raw[:end-start], start))
            assert sum(i.size for i in ins) == end-start
            self.instructions.update({i.address: i for i in ins})
            self.body_bounds[hex(start)] = {'end_exclusive': hex(end), 'next_managed': hex(following), 'padding_bytes': following-end}
        m = re.search(r'^public class CardTokens : MonoBehaviour // TypeDefIndex: 5734\s*\{(.*?)// Methods', dump, re.M | re.S)
        assert m and all(s in m[1] for s in ['public Character character; // 0x20', 'public GameObject tagGood; // 0x28', 'public GameObject tagExcl; // 0x30', 'public GameObject tagUnsure; // 0x38', 'public GameObject tagBad; // 0x40'])
        m = re.search(r'^public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487\s*\{(.*?)// Properties', dump, re.M | re.S)
        assert m and 'public ECharacterPlacement placement; // 0xE8' in m[1] and 'public bool hover; // 0x190' in m[1]
        m = re.search(r'^public enum ECharacterPlacement // TypeDefIndex: 5490\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert m and all(f'public const ECharacterPlacement {n} = {v};' in m[1] for n, v in [('None', 0), ('Gameplay', 10), ('Browsing', 20)])
        self.supplied = []
        for rva, name in [(0x1CD3C80, 'UnityEngine.Input$$GetKeyDown'), (0x1C7DC50, 'UnityEngine.GameObject$$get_activeSelf'), (0x1C7D810, 'UnityEngine.GameObject$$SetActive')]:
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == rva and r['Name'] == name]
            assert len(rows) == 1
            self.supplied += rows
        self.input_aliases = [r for r in self.metadata['ScriptMethod'] if r['Address'] == 0x1CD3C80]
        assert {r['Name'] for r in self.input_aliases} == {'UnityEngine.Input$$GetKeyDown', 'UnityEngine.Input$$GetKeyDownInt'}
        self.checks = {0x3971A9: ('mov', 'rcx, qword ptr [rcx + 0x28]'),
            0x3971F5: ('jmp', '0x1c7d810'), 0x397216: ('cmp', 'dword ptr [rax + 0xe8], 0xa'),
            0x397223: ('cmp', 'byte ptr [rax + 0x190], 0'), 0x397252: ('mov', 'rcx, qword ptr [rbx + 0x30]'),
            0x3972A9: ('test', 'al, al'), 0x3972AB: ('sete', 'dl'),
            0x397334: ('mov', 'rcx, qword ptr [rbx + 0x40]'), 0x3973AC: ('mov', 'rcx, qword ptr [rbx + 0x38]'),
            0x397426: ('jmp', '0x1c7d810'), 0x397430: ('ret', ''), 0x397431: ('call', '0x2b7d90')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        self.p = {n: self.arena + 0x10000 + i * 0x1000 for i, n in enumerate(['owner', 'character', 'good', 'excl', 'unsure', 'bad', 'other', 'class'])}
        self.ids = {p: n for n, p in self.p.items()}
        self.sizes = {n: 0x200 if n == 'character' else 0x80 for n in self.p}
        self.stop_pc = None

    def oid(self, p):
        if not p: return None
        assert p in self.ids, hex(p)
        return self.ids[p]

    def snapshot(self):
        owner, ch = self.p['owner'], self.p['character']
        return {'fields': {n: self.oid(self.rq(owner + off)) for n, off in FIELDS.items()},
                'placement_bits': self.rd(ch + 0xE8), 'hover_bits': self.u.mem_read(ch + 0x190, 1)[0],
                'games': self.games.copy(), 'keys': self.key_requests.copy(),
                'active_reads': self.active_reads.copy(), 'active_writes': self.active_writes.copy(),
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        assert all(0 <= options.get(n, default) <= 255 for n, default in [('key_true_byte', 0x80), ('active_true_byte', 0xFE)])
        self.stop_pc = None
        self.games = {n: bool(options.get('active_mask', 5) & (1 << i)) for i, n in enumerate(['good', 'excl', 'unsure', 'bad', 'other'])}
        self.key_requests, self.active_reads, self.active_writes = [], [], []
        for n, p in self.p.items():
            self.u.mem_write(p, bytes([0xA5]) * self.sizes[n]); self.q(p, self.p['class']); self.q(p + 8, 0)
        owner = self.p['owner']
        for n, off in FIELDS.items():
            value = 0 if options.get('null_field') == n else self.p['good'] if options.get('alias_tags') and n != 'character' else self.p[n]
            self.q(owner + off, value)
        self.d(self.p['character'] + 0xE8, options.get('placement_bits', 10))
        self.u.mem_write(self.p['character'] + 0x190, bytes([options.get('hover_bits', 0x80)]))

    def mutation(self, phase):
        if self.options.get('mutation_phase') != phase: return
        action = self.options['mutation']
        if action.startswith('clear_'):
            self.q(self.p['owner'] + FIELDS[action[6:]], 0)
        elif action.startswith('replace_'):
            self.q(self.p['owner'] + FIELDS[action[8:]], self.p['other'])
        elif action == 'placement_and_hover':
            self.d(self.p['character'] + 0xE8, 20); self.u.mem_write(self.p['character'] + 0x190, b'\x00')
        else: raise AssertionError(action)

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
            assert cx in KEYS and dx == 0
            result = 0xFACE123456789000 | (self.options.get('key_true_byte', 0x80) if self.options.get('key_mask', 31) & (1 << KEYS.index(cx)) else 0)
            if self.event('key_down_service', [cx, result]):
                self.key_requests.append([cx, result]); self.mutation('key:' + str(cx)); self.ret(result)
        elif rva == 0x1C7DC50:
            assert cx in [self.p[n] for n in self.games] and dx == 0
            result = 0xFACE123456789000 | (self.options.get('active_true_byte', 0xFE) if self.games[self.oid(cx)] else 0)
            if self.event('active_self_service', [self.oid(cx), result]):
                self.active_reads.append([self.oid(cx), result]); self.mutation('read:' + self.oid(cx)); self.ret(result)
        elif rva == 0x1C7D810:
            assert cx in [self.p[n] for n in self.games] and r8 == 0 and dx & 255 in [0, 1]
            return_rva = self.rq(self.reg(x.UC_X86_REG_RSP)) - self.base
            toggles = {0x39727A: False, 0x3972B6: True, 0x39735C: False, 0x3973CC: False}
            if return_rva in toggles:
                active = bool(self.active_reads[-1][1] & 255)
                expected_dx = (0xFACE123456789000 | int(not active)) if toggles[return_rva] or not active else 0
            else: expected_dx = 0
            assert dx == expected_dx, (hex(return_rva), hex(dx), hex(expected_dx))
            args = [self.oid(cx), dx, dx & 255]
            if self.event('set_active_service', args):
                self.games[self.oid(cx)] = bool(dx & 255); self.active_writes.append(args)
                self.mutation('write:' + self.oid(cx)); self.ret()
        elif rva == 0x2B7D90:
            self.event('native_null_guard', []); self.error = 'native_null_guard'; self.stop_pc = hex(rva); uc.emu_stop()
        else: raise AssertionError(f'unclaimed address {rva:x}')

    def invoke(self, address):
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop); self.u.mem_write(sp + 8, bytes([0xB6]) * 0x38)
        ints = [getattr(x, 'UC_X86_REG_' + n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, r in enumerate(ints): self.u.reg_write(r, 0xFAB00000 + i)
        for i, r in enumerate(vectors): self.u.reg_write(r, (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64))
        for r, v in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, 0 if self.options.get('null_owner') else self.p['owner']),
                     (x.UC_X86_REG_RDX, 0xDEADBEEF12345678)]: self.u.reg_write(r, v)
        try: self.u.emu_start(self.base + address, self.stop, timeout=10_000_000, count=10000)
        except self.unicorn.UcError as exc:
            assert self.options.get('null_owner') and exc.errno == self.unicorn.UC_ERR_READ_UNMAPPED
            assert self.reg(x.UC_X86_REG_RIP) - self.base == (0x3971A9 if address == 0x3971A0 else 0x397206)
            self.error, self.stop_pc = 'native_owner_access_fault', hex(self.reg(x.UC_X86_REG_RIP) - self.base)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(r) == 0xFAB00000 + i for i, r in enumerate(ints))
            assert all(self.reg(r) == (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64) for i, r in enumerate(vectors))
        return returned

    def expected_normal(self, name, initial):
        """Independent values-only contract for successful unmutated callers."""
        fields, games, expected = initial['fields'], initial['games'].copy(), []
        def write(field, value, raw=None):
            target = fields[field]
            assert target is not None
            expected.append({'kind': 'set_active_service', 'args': [target, value if raw is None else raw, value]})
            games[target] = bool(value)
        def toggle(field, always_byte_write=False):
            target = fields[field]
            active = games[target] and self.options.get('active_true_byte', 0xFE) != 0
            result = 0xFACE123456789000 | (self.options.get('active_true_byte', 0xFE) if active else 0)
            expected.append({'kind': 'active_self_service', 'args': [target, result]})
            value = int(not active)
            raw = (0xFACE123456789000 | value) if always_byte_write or not active else 0
            write(field, value, raw)
        if name == 'OnEnable':
            for field in ['good', 'unsure', 'bad', 'excl']: write(field, 0)
        elif initial['placement_bits'] == 10 and initial['hover_bits'] != 0:
            for i, key in enumerate(KEYS):
                selected = bool(self.options.get('key_mask', 31) & (1 << i))
                result = 0xFACE123456789000 | (self.options.get('key_true_byte', 0x80) if selected else 0)
                pressed = bool(result & 255)
                expected.append({'kind': 'key_down_service', 'args': [key, result]})
                if not pressed: continue
                if key == 53: toggle('excl')
                elif key == 49:
                    toggle('good', always_byte_write=True); write('unsure', 0); write('bad', 0)
                elif key == 51:
                    write('good', 0); write('unsure', 0); toggle('bad')
                elif key == 50:
                    write('good', 0); write('bad', 0); toggle('unsure')
                else:
                    for field in ['good', 'unsure', 'bad', 'excl']: write(field, 0)
        return expected, games

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error, self.stop_pc = options or {}, None, None
        initial, old = self.snapshot(), len(self.events)
        returned = self.invoke(next(a for a, (n, _, _) in TARGETS.items() if n == name))
        final, events = self.snapshot(), self.events[old:].copy()
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            allowed = set()
            action = self.options.get('mutation', '')
            if self.options.get('mutation_phase'):
                if n == 'owner' and action.startswith(('clear_', 'replace_')):
                    field = action.split('_', 1)[1]; off = FIELDS[field]; allowed = set(range(off, off+8))
                elif n == 'character' and action == 'placement_and_hover': allowed = set(range(0xE8, 0xEC)) | {0x190}
            assert all(i in allowed or byte == after[i] for i, byte in enumerate(before)), n
        if returned and not self.options.get('mutation_phase'):
            expected, games = self.expected_normal(name, initial)
            assert [{'kind': e['kind'], 'args': e['args']} for e in events] == expected
            assert final['games'] == games
            eligible = initial['placement_bits'] == 10 and initial['hover_bits'] != 0
            assert [e['args'][0] for e in events if e['kind'] == 'key_down_service'] == (KEYS if name == 'Update' and eligible else [])
            if name == 'OnEnable':
                assert [e['args'][0] for e in events] == [initial['fields'][n] for n in ['good', 'unsure', 'bad', 'excl']]
                assert all(e['args'][1] == 0 for e in events)
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'stop_pc': self.stop_pc, 'initial': initial, 'events': events, 'final': final}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, baselines, stops, seqs = [], [], [], []
    for mask, active, alias in itertools.product(range(32), [0, 5, 15], [False, True]):
        cases.append(m.run('Update', {'key_mask': mask, 'active_mask': active, 'alias_tags': alias}))
    for active in range(16):
        cases.append(m.run('Update', {'active_mask': active})); cases.append(m.run('OnEnable', {'active_mask': active}))
    for key_byte, active_byte in itertools.product([0, 1, 0x80, 0xFF], repeat=2):
        cases.append(m.run('Update', {'key_true_byte': key_byte, 'active_true_byte': active_byte}))
    for placement, hover in itertools.product([0, 10, 20, 0x8000000A, 0xFFFFFFFF], [0, 1, 0x80, 0xFF]):
        cases.append(m.run('Update', {'placement_bits': placement, 'hover_bits': hover}))
    for name in ['OnEnable', 'Update']:
        cases.append(m.run(name, {'null_owner': True}))
        for field in FIELDS: cases.append(m.run(name, {'null_field': field}))
    for key in KEYS:
        for field in ['good', 'excl', 'unsure', 'bad']:
            cases.append(m.run('Update', {'key_mask': 1 << KEYS.index(key), 'null_field': field}))
    mutation_profiles = [('key:53', 'placement_and_hover'), ('key:53', 'clear_character'),
        ('read:excl', 'replace_excl'), ('read:excl', 'clear_excl'), ('read:good', 'replace_good'),
        ('read:bad', 'replace_bad'), ('read:unsure', 'replace_unsure'), ('write:good', 'clear_unsure'),
        ('write:unsure', 'replace_bad'), ('write:bad', 'replace_excl')]
    for phase, action in mutation_profiles:
        row = m.run('Update', {'mutation_phase': phase, 'mutation': action})
        if phase.startswith('key:'):
            assert row['returned'] and [e['args'][0] for e in row['events'] if e['kind'] == 'key_down_service'] == KEYS
        elif phase.startswith('read:'):
            field = phase.split(':')[1]
            index = next(i for i, e in enumerate(row['events']) if e['kind'] == 'active_self_service' and e['args'][0] == field)
            if action.startswith('replace_'):
                event = row['events'][index+1]
                assert event['kind'] == 'set_active_service' and event['args'][0] == 'other'
                assert event['args'][2] == int(not (row['events'][index]['args'][1] & 255))
            else:
                assert row['error'] == 'native_null_guard' and row['events'][index+1]['kind'] == 'native_null_guard'
        elif action == 'clear_unsure':
            index = next(i for i,e in enumerate(row['events']) if e['kind']=='set_active_service' and e['args'][0]=='good')
            assert row['error'] == 'native_null_guard' and row['events'][index+1]['kind'] == 'native_null_guard'
        else:
            assert row['returned'] and any(e['kind']=='set_active_service' and e['args'][0]=='other' for e in row['events'])
        cases.append(row)
    for name, options in [('OnEnable', {}), ('Update', {}), ('Update', {'active_mask': 15}), ('Update', {'alias_tags': True})]:
        base = m.run(name, options); baselines.append(base)
        seen = {}
        for index, event in enumerate(base['events']):
            kind = event['kind']; seen[kind] = seen.get(kind, 0) + 1
            row = m.run(name, {**options, 'failure': [kind, seen[kind]]})
            assert not row['returned'] and row['events'] == base['events'][:index+1]
            assert row['final'] == event['snapshot']
            stops.append(row)
    for alias in [False, True]:
        m.prepare({'alias_tags': alias})
        rows = [m.run('Update', retained=True), m.run('OnEnable', retained=True),
                m.run('Update', {'key_mask': 3}, retained=True), m.run('Update', {'key_mask': 16}, retained=True)]
        assert all(r['returned'] for r in rows)
        seqs.append(rows)
    assert set(m.instructions) <= m.executed
    return {'schema': 'card_tokens_native_v1', 'build': BUILD,
            'scope': 'two complete CardTokens callers; input and Unity services supplied; no live keyboard or rendering',
            'targets': m.targets, 'body_bounds': m.body_bounds, 'supplied_targets': m.supplied,
            'input_entry_aliases': m.input_aliases, 'field_offsets': FIELDS,
            'placement_gate': {'type_def_index': 5490, 'field_offset': 0xE8, 'Gameplay': 10},
            'key_query_order': KEYS, 'instruction_assertions': len(m.checks),
            'caller_instructions': len(m.instructions), 'caller_instructions_executed': len(set(m.instructions) & m.executed),
            'diagnostic_windows_not_object_extents': m.sizes, 'cases': cases, 'baselines': baselines,
            'failure_stops': stops, 'retained_sequences': seqs,
            'summary': {'cases': len(cases), 'stops': len(stops), 'sequences': len(seqs), 'instructions': len(m.instructions)}}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--game-root', type=Path, required=True)
    parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.game_root, args.dumper_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(pool_memory(result), sort_keys=True, separators=(',', ':')) + '\n', encoding='utf-8')
    print(json.dumps(result['summary'], sort_keys=True))
