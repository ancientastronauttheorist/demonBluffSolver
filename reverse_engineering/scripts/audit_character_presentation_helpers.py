"""Execute six pinned Character presentation callers with explicit services.

Game-owned CharacterView/CardHighlight callees are supplied caller boundaries.
This audit does not execute their bodies or claim Unity rendering/input effects.
"""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine


TARGETS = {0x369240: 'ShowDisguise', 0x365460: 'HideDisguise',
           0x369320: 'ShowHighlight', 0x364B20: 'DisableHighlight',
           0x367E00: 'ResetRotation', 0x3674E0: 'OnHoverOff'}
GAME_SERVICES = {0x363D50: 'CharacterView.AnimateIn', 0x363DF0: 'CharacterView.AnimateOut',
                 0x363F10: 'CharacterView.Init', 0x397090: 'CardHighlight.ShowHighlight',
                 0x396F40: 'CardHighlight.DisableHighlight'}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root)
        ext = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == ext['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json'))
        self.dump = pin('dump.cs', 'dump_cs')
        self.targets, self.instructions, self.ranges = [], {}, {}
        self.supplied_game_targets = []
        for address, name in GAME_SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name.replace('.', '$$')]
            assert len(rows) == 1
            self.supplied_game_targets += rows
        for address, name in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == 'Character$$' + name]
            assert len(rows) == 1
            assert rows[0]['Signature'] == f'void Character__{name} (Character_o* __this, const MethodInfo* method);'
            self.targets += rows
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4:
                    root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == address:
                    chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            if name == 'OnHoverOff':
                assert not chunks
                chunks = [(address, address + 8)]  # Complete verified leaf; no padding.
            assert chunks
            self.ranges[hex(address)] = [[hex(a), hex(b)] for a, b in chunks]
            for a, b in chunks:
                rows = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
                assert sum(i.size for i in rows) == b - a
                self.instructions.update({i.address: i for i in rows})
        references, self.flags = set(), set()
        for i in self.instructions.values():
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + op.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
                self.flags.add(i.address + i.size + i.operands[0].mem.disp)
        self.bindings, self.metadata_slots = {}, {}
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                token = self.arena + 0x4000 + len(self.bindings) * 0x200
                self.bindings[row['Name']] = token
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
        assert set(self.bindings) == {'UIEvents_TypeInfo', 'UnityEngine.Object_TypeInfo', 'UnityEngine.Vector3_TypeInfo'}
        for name, index, fields in [
                ('Character', 5487, ['public Transform icon; // 0x20', 'public Transform hintPivot; // 0x38',
                                     'public CharacterData dataRef; // 0x50', 'public CharacterData bluff; // 0x58',
                                     'public CardHighlight highlight; // 0x90', 'public bool killedByDemon; // 0xED',
                                     'public CharacterView charBluff; // 0x140', 'public bool hover; // 0x190',
                                     'public bool showDisguise; // 0x1A0']),
                ('UIEvents', 5523, ['public static Action<CharacterData, Transform> OnShowCharacterDataHint; // 0x20']),
                ('Vector3', 6699, ['private static readonly Vector3 zeroVector; // 0x0'])]:
            declaration = re.search(r'^public (?:static )?(?:class|struct) ' + re.escape(name) +
                                    r'(?: :[^\n]*)? // TypeDefIndex: ' + str(index) +
                                    r'\s*\{(.*?)// (?:Properties|Methods)', self.dump, re.M | re.S)
            assert declaration and all(f in declaration[1] for f in fields)
        self.checks = {
            0x3674E0: ('mov', 'byte ptr [rcx + 0x190], 0'), 0x3674E7: ('ret', ''),
            0x36929B: ('test', 'al, al'), 0x36929F: ('cmp', 'byte ptr [rbx + 0xed], al'),
            0x3692AE: ('mov', 'byte ptr [rbx + 0x1a0], 1'),
            0x3692C1: ('mov', 'rcx, qword ptr [rbx + 0x140]'),
            0x3692CD: ('mov', 'rdx, qword ptr [rbx + 0x58]'),
            0x3692F4: ('mov', 'r8, qword ptr [rbx + 0x38]'),
            0x369300: ('call', 'qword ptr [rax + 0x18]'),
            0x36548C: ('mov', 'byte ptr [rbx + 0x1a0], 0'),
            0x36549F: ('cmp', 'byte ptr [rbx + 0x190], 0'),
            0x3654C7: ('mov', 'rdx, qword ptr [rbx + 0x50]'),
            0x3654D4: ('jmp', 'qword ptr [rax + 0x18]'),
            0x369336: ('jmp', '0x397090'), 0x364B36: ('jmp', '0x396f40'),
            0x367E48: ('movsd', 'xmm0, qword ptr [rax]'),
            0x367E51: ('mov', 'eax, dword ptr [rax + 8]'),
            0x367E60: ('mov', 'dword ptr [rsp + 0x28], eax')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        self.p = {name: self.arena + 0x10000 + i * 0x1000 for i, name in enumerate([
            'actor', 'icon', 'transform', 'pivot', 'view', 'highlight', 'data', 'bluff',
            'ui_static', 'vector_static', 'callback', 'callback_target', 'callback_method', 'replacement_callback'])}
        self.ids = {v: k for k, v in self.p.items()}
        self.ids.update({v: k for k, v in self.bindings.items()})
        self.callback_service = self.stop + 0x100
        self.services = {0x2B7B40, 0x281D90, 0x1C822C0, 0x1C7A010, 0x1C91EC0, 0x2B7D90, *GAME_SERVICES}

    def oid(self, p):
        if not p: return None
        assert p in self.ids, hex(p)
        return self.ids[p]

    def byte(self, p):
        return self.u.mem_read(p, 1)[0]

    def snapshot(self):
        a = self.p['actor']
        return {'actor': {'hover_bits': self.byte(a + 0x190), 'show_disguise_bits': self.byte(a + 0x1A0),
                          'killed_by_demon_bits': self.byte(a + 0xED),
                          **{name: self.oid(self.rq(a + offset)) for name, offset in
                             [('icon', 0x20), ('pivot', 0x38), ('data', 0x50), ('bluff', 0x58),
                              ('highlight', 0x90), ('view', 0x140)]}},
                'metadata_flags': {hex(f): self.byte(self.base + f) for f in sorted(self.flags)},
                'object_class_initialized': self.rd(self.bindings['UnityEngine.Object_TypeInfo'] + 0xE0),
                'ui_class_initialized': self.rd(self.bindings['UIEvents_TypeInfo'] + 0xE0),
                'vector_class_initialized': self.rd(self.bindings['UnityEngine.Vector3_TypeInfo'] + 0xE0),
                'ui_static_slots': [self.oid(self.rq(self.p['ui_static'] + i * 8)) if i == 4 else
                                    self.rq(self.p['ui_static'] + i * 8) for i in range(21)],
                'zero_vector_bits': [self.rd(self.p['vector_static'] + i * 4) for i in range(3)],
                'supplied_view_state': self.view_state.copy(), 'supplied_highlight_active': self.highlight_active,
                'supplied_rotations': self.rotations.copy(), 'callback_observations': self.callback_observations.copy()}

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        self.view_state = {'active': False, 'data': 'data'}
        self.highlight_active, self.rotations, self.callback_observations = False, [], []
        self.authored_offsets = set()
        for p in self.p.values(): self.u.mem_write(p, bytes(0x400))
        a = self.p['actor']; self.u.mem_write(a, bytes([0xA5]) * 0x200)
        for field, offset, name in [('icon', 0x20, 'icon'), ('pivot', 0x38, 'pivot'), ('data', 0x50, 'data'),
                                  ('bluff', 0x58, 'bluff'), ('highlight', 0x90, 'highlight'), ('view', 0x140, 'view')]:
            p = 0 if options.get('null_' + field) else self.p[name]
            if field == 'bluff' and options.get('bluff') == 'absent': p = 0
            if field == 'bluff' and options.get('same_data_bluff'): p = self.p['data']
            if field == 'pivot' and options.get('alias_pivot_icon'): p = self.p['icon']
            self.q(a + offset, p)
        for offset, value in [(0xED, options.get('killed', 0)), (0x190, options.get('hover', 1)),
                              (0x1A0, options.get('show', 0xFE))]:
            self.u.mem_write(a + offset, bytes([value]))
        for flag in self.flags: self.u.mem_write(self.base + flag, bytes([options.get('warm_byte', 1) if not options.get('cold') else 0]))
        for n in self.bindings:
            self.d(self.bindings[n] + 0xE0, options.get('class_word', 1) if not options.get('class_cold') else 0)
        self.q(self.bindings['UIEvents_TypeInfo'] + 0xB8, self.p['ui_static'])
        self.q(self.bindings['UnityEngine.Vector3_TypeInfo'] + 0xB8, self.p['vector_static'])
        for i in range(21): self.q(self.p['ui_static'] + i * 8, 0xFACE0000 + i * 8)
        self.q(self.p['ui_static'] + 0x20, self.p['callback'] if options.get('callback', True) else 0)
        for name in ['callback', 'replacement_callback']:
            cb = self.p[name]
            self.q(cb + 0x18, self.callback_service)
            self.q(cb + 0x28, self.p['callback_method'])
            self.q(cb + 0x40, 0 if options.get('null_callback_target') else self.p['view'] if options.get('alias_callback_view') else self.p['callback_target'])
        for i, value in enumerate(options.get('vector_bits', [0, 0, 0])): self.d(self.p['vector_static'] + i * 4, value)

    def mutate(self, phase):
        if self.options.get('mutation_phase') != phase: return
        action = self.options['mutation']
        a = self.p['actor']
        if action in ['clear_view', 'clear_bluff', 'bluff_to_data', 'clear_pivot']:
            offset = {'clear_view': 0x140, 'clear_bluff': 0x58, 'bluff_to_data': 0x58, 'clear_pivot': 0x38}[action]
            self.q(a + offset, self.p['data'] if action == 'bluff_to_data' else 0)
            self.authored_offsets.update(range(offset, offset + 8))
        elif action in ['clear_hover', 'noncanonical_hover', 'callback_actor_bytes']:
            self.u.mem_write(a + 0x190, bytes([0 if action == 'clear_hover' else 0x80]))
            self.authored_offsets.add(0x190)
            if action == 'callback_actor_bytes':
                self.u.mem_write(a + 0x1A0, b'\xff'); self.authored_offsets.add(0x1A0)
        elif action in ['clear_callback', 'replace_callback']:
            self.q(self.p['ui_static'] + 0x20, self.p['replacement_callback'] if action == 'replace_callback' else 0)
        else: raise AssertionError(action)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        self.executed.add(rva)
        if rva in self.instructions: return
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == 0x2B7B40:
            assert cx in self.metadata_slots
            if self.event('metadata_service', [cx - self.base]): self.ret(self.metadata_slots[cx])
        elif rva == 0x281D90:
            assert cx == self.bindings['UnityEngine.Object_TypeInfo']
            if self.event('class_initialization_service', [self.oid(cx)]):
                self.d(cx + 0xE0, 1); self.ret()
        elif rva == 0x1C822C0:
            assert cx in [0, self.p['bluff'], self.p['data']] and dx == r8 == 0
            result = self.options.get('null_return_bits', 0xABC000 | int(cx == 0 or self.options.get('bluff') == 'destroyed'))
            assert bool(result & 0xFF) == (cx == 0 or self.options.get('bluff') == 'destroyed')
            if self.event('unity_null_service', [self.oid(cx), result]):
                self.mutate('unity_null'); self.ret(result)
        elif rva in GAME_SERVICES:
            name = GAME_SERVICES[rva]; expected = 'view' if name.startswith('CharacterView') else 'highlight'
            assert cx == self.p[expected]
            if name == 'CharacterView.Init':
                assert dx in [0, self.p['data'], self.p['bluff']] and r8 == 0
                args = [self.oid(cx), self.oid(dx)]
            else:
                assert dx == 0
                args = [self.oid(cx)]
            if self.event('supplied_' + name.replace('.', '_'), args):
                if name == 'CharacterView.Init': self.view_state['data'] = self.oid(dx)
                elif name == 'CharacterView.AnimateIn': self.view_state['active'] = True
                elif name == 'CharacterView.AnimateOut': self.view_state['active'] = False
                else: self.highlight_active = name.endswith('ShowHighlight')
                self.mutate(name); self.ret()
        elif rva == 0x1C7A010:
            assert cx == self.p['icon'] and dx == 0
            result = 0 if self.options.get('null_transform') else self.p['icon'] if self.options.get('same_transform_icon', True) else self.p['transform']
            if self.event('transform_getter_service', [self.oid(cx), self.oid(result)]): self.ret(result)
        elif rva == 0x1C91EC0:
            assert cx in [self.p['icon'], self.p['transform']] and r8 == 0
            bits = list(struct.unpack('<III', uc.mem_read(dx, 12)))
            assert bits == [self.rd(self.p['vector_static'] + i * 4) for i in range(3)]
            if self.event('rotation_setter_service', [self.oid(cx), bits]):
                self.rotations.append([self.oid(cx), bits]); self.ret()
        elif address == self.callback_service:
            cb = self.rq(self.p['ui_static'] + 0x20)
            assert cb in [self.p['callback'], self.p['replacement_callback']]
            assert cx == self.rq(cb + 0x40) and r9 == self.rq(cb + 0x28)
            assert dx in [0, self.p['data'], self.p['bluff']] and r8 in [0, self.p['pivot'], self.p['icon']]
            observed = {'delegate': self.oid(cb), 'target': self.oid(cx), 'data': self.oid(dx),
                        'pivot': self.oid(r8), 'method': self.oid(r9)}
            if self.event('hint_callback_service', [observed]):
                self.callback_observations.append(observed); self.mutate('callback'); self.ret()
        elif rva == 0x2B7D90:
            self.event('native_null_guard', [])
            self.error = 'native_null_guard'; uc.emu_stop()
        else:
            raise AssertionError(f'unclaimed native address {rva:x}')

    def invoke(self, address):
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop); self.q(sp + 0x28, 0)
        regs = [getattr(x, 'UC_X86_REG_' + n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, reg in enumerate(regs): self.u.reg_write(reg, 0xFAB00000 + i)
        for i, reg in enumerate(vectors): self.u.reg_write(reg, (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64))
        for reg, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, self.p['actor']),
                           (x.UC_X86_REG_RDX, 0xABCDEF1234567890)]: self.u.reg_write(reg, value)
        self.u.emu_start(self.base + address, self.stop, timeout=10_000_000, count=10000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(r) == 0xFAB00000 + i for i, r in enumerate(regs))
            assert all(self.reg(r) == (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64) for i, r in enumerate(vectors))
        return returned

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.authored_offsets = set()
        a = self.p['actor']; before = bytes(self.u.mem_read(a, 0x200)); initial = self.snapshot()
        old_event_count = len(self.events)
        returned = self.invoke(next(a for a, n in TARGETS.items() if n == name))
        after = bytes(self.u.mem_read(a, 0x200))
        allowed = {0x190} if name == 'OnHoverOff' else {0x1A0} if name in ['ShowDisguise', 'HideDisguise'] else set()
        allowed |= self.authored_offsets
        assert all(i in allowed or before[i] == after[i] for i in range(0x200))
        if returned and not self.options.get('mutation_phase'):
            if name == 'HideDisguise': assert self.byte(a + 0x1A0) == 0
            elif name == 'OnHoverOff': assert self.byte(a + 0x190) == 0
            elif name == 'ShowDisguise':
                changed = self.options.get('bluff', 'live') == 'live' and initial['actor']['killed_by_demon_bits'] == 0
                assert self.byte(a + 0x1A0) == (1 if changed else initial['actor']['show_disguise_bits'])
        if not retained and not self.options.get('mutation_phase') and not self.options.get('failure'):
            reached = [e['kind'] for e in self.events[old_event_count:] if e['kind'].startswith('supplied_') or e['kind'] == 'hint_callback_service']
            if name == 'ShowDisguise':
                eligible = initial['actor']['bluff'] is not None and self.options.get('bluff') != 'destroyed' and initial['actor']['killed_by_demon_bits'] == 0
                valid_view = initial['actor']['view'] is not None
                assert returned == (not eligible or valid_view)
                expected = ['supplied_CharacterView_AnimateIn', 'supplied_CharacterView_Init'] if eligible and valid_view else []
                if expected and self.options.get('callback', True): expected.append('hint_callback_service')
                assert reached == expected
            elif name == 'HideDisguise':
                assert returned == (initial['actor']['view'] is not None)
                expected = ['supplied_CharacterView_AnimateOut'] if returned else []
                if returned and initial['actor']['hover_bits'] != 0 and self.options.get('callback', True): expected.append('hint_callback_service')
                assert reached == expected
            elif name in ['ShowHighlight', 'DisableHighlight']:
                assert returned == (initial['actor']['highlight'] is not None)
                assert reached == (['supplied_CardHighlight_' + name] if returned else [])
            elif name == 'ResetRotation':
                assert returned == (initial['actor']['icon'] is not None and not self.options.get('null_transform'))
            else: assert returned and not reached
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'initial': initial, 'events': self.events[old_event_count:].copy(), 'final': self.snapshot(),
                'other_actor_bytes_retained': True}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, sequences, baselines, failures = [], [], [], []
    for bluff, killed, callback, null_view, cold, class_cold in itertools.product(
            ['live', 'destroyed', 'absent'], [0, 1, 0x80, 0xFF], [False, True], [False, True], [False, True], [False, True]):
        cases.append(m.run('ShowDisguise', {'bluff': bluff, 'killed': killed, 'callback': callback,
                                         'null_view': null_view, 'cold': cold, 'class_cold': class_cold}))
    for hover, callback, null_view, cold in itertools.product([0, 1, 0x80, 0xFF], [False, True], [False, True], [False, True]):
        cases.append(m.run('HideDisguise', {'hover': hover, 'callback': callback, 'null_view': null_view, 'cold': cold}))
    for name, null, cold in itertools.product(['ShowHighlight', 'DisableHighlight'], [False, True], [False, True]):
        cases.append(m.run(name, {'null_highlight': null, 'cold': cold}))
    for null_icon, null_transform, cold, same, bits in itertools.product([False, True], [False, True], [False, True], [False, True],
                                                                       [[0, 0, 0], [0x80000000, 0x7F800000, 0x7FC01234], [0x3F800000, 0xBF800000, 0x41200000]]):
        cases.append(m.run('ResetRotation', {'null_icon': null_icon, 'null_transform': null_transform, 'cold': cold,
                                           'same_transform_icon': same, 'vector_bits': bits}))
    for hover in [0, 1, 0x80, 0xFF]: cases.append(m.run('OnHoverOff', {'hover': hover}))
    for options in [{'null_pivot': True}, {'null_data': True}, {'alias_pivot_icon': True, 'alias_callback_view': True},
                    {'same_data_bluff': True}, {'null_callback_target': True}, {'warm_byte': 0x80, 'class_word': 0xDEADBEEF},
                    {'null_return_bits': 0xFFFFFFFFFFFFFF00},
                    {'bluff': 'destroyed', 'null_return_bits': 0xFFFFFFFFFFFFFF80}]:
        for name in ['ShowDisguise', 'HideDisguise']: cases.append(m.run(name, options))
    for name, phase, action in [('ShowDisguise', 'unity_null', 'clear_bluff'),
                              ('ShowDisguise', 'CharacterView.AnimateIn', 'clear_view'),
                              ('ShowDisguise', 'CharacterView.AnimateIn', 'bluff_to_data'),
                              ('ShowDisguise', 'CharacterView.Init', 'clear_pivot'),
                              ('ShowDisguise', 'CharacterView.Init', 'clear_callback'),
                              ('ShowDisguise', 'CharacterView.Init', 'replace_callback'),
                              ('ShowDisguise', 'callback', 'callback_actor_bytes'),
                              ('HideDisguise', 'CharacterView.AnimateOut', 'clear_hover'),
                              ('HideDisguise', 'CharacterView.AnimateOut', 'noncanonical_hover'),
                              ('HideDisguise', 'CharacterView.AnimateOut', 'replace_callback'),
                              ('HideDisguise', 'callback', 'callback_actor_bytes')]:
        cases.append(m.run(name, {'mutation_phase': phase, 'mutation': action}))
    for alias in [False, True]:
        m.prepare({'cold': True, 'class_cold': True, 'alias_pivot_icon': alias, 'alias_callback_view': alias})
        calls = [m.run(name, retained=True) for name in ['ShowDisguise', 'ShowHighlight', 'HideDisguise', 'OnHoverOff',
                                                       'HideDisguise', 'DisableHighlight', 'ResetRotation']]
        assert all(c['returned'] for c in calls)
        sequences.append({'alias_ui': alias, 'calls': calls})
    for name in TARGETS.values():
        baseline = m.run(name, {'cold': True, 'class_cold': True}); assert baseline['returned']
        ordinal = len(baselines); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(name, {'cold': True, 'class_cold': True, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index + 1]
            assert stopped['final'] == e['snapshot']
            failures.append({'baseline': ordinal, 'failure': [kind, counts[kind]], 'prefix_length': index + 1,
                             'exact_snapshot_verified': True})
    missing = set(m.instructions) - m.executed
    assert len(missing) == 5 and all(m.instructions[a].mnemonic == 'int3' for a in missing)
    return {'build': BUILD, 'targets': m.targets, 'caller_ranges': m.ranges,
            'instruction_assertions': len(m.checks), 'metadata_bindings': sorted(m.bindings),
            'case_count': len(cases), 'cases': cases, 'retained_sequences': sequences,
            'failure_baselines': baselines, 'failure_case_count': len(failures), 'failure_cases': failures,
            'caller_instructions_decoded': len(m.instructions),
            'caller_instructions_executed': len(m.executed & m.instructions.keys()),
            'unexecuted_terminal_traps': len(missing),
            'native_execution_addresses': len(m.executed),
            'supplied_game_owned_callees': GAME_SERVICES,
            'supplied_game_owned_metadata': m.supplied_game_targets,
            'scope': 'Six complete native Character callers execute; CharacterView/CardHighlight bodies, Unity rendering/input/liveness, metadata/class initialization and callback implementation remain explicit supplied services. Partial stopped state is retained; real exception unwinding, animation completion and other presentation entries are not claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'failure_case_count', 'caller_instructions_executed', 'native_execution_addresses']}))
