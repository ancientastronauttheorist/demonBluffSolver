"""Execute tutorial presentation callers through native save/JSON/storage bodies."""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_runtime_strings import Machine as StorageMachine
from audit_saved_game_storage_join import compact_trace, values

TARGETS = {
    'controller_reset': (0x38DF80, 0x38E0B3, 'TutorialsController$$ResetTutorials', 'tdi5650.m0002'),
    'note_reset': (0x38BD80, 0x38BE16, 'TutorialNote$$ResetTutorial', 'tdi5633.m0000'),
    'enable': (0x38C950, 0x38C958, 'TutorialsController$$EnableTutorial', 'tdi5650.m0009'),
    'show': (0x38E1A0, 0x38E3CA, 'TutorialsController$$ShowTutorial', 'tdi5650.m0023'),
    'closed': (0x38C430, 0x38C4A4, 'TutorialsController$$CheckIfAllTutorialsClosed', 'tdi5650.m0026'),
    'note_show': (0x38BE20, 0x38C099, 'TutorialNote$$Show', 'tdi5633.m0001'),
    'queue_ctor': (0x3A88D0, 0x3A8922, 'TutorialQueue$$.ctor', 'tdi5651.m0000'),
}


class Machine(StorageMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        self.controller, self.project_type, self.project_static, self.project, self.game_data = [
            self.arena + n for n in [0x160000, 0x161000, 0x162000, 0x163000, 0x164000]]
        self.ui_type, self.iterator_type, self.queue_type, self.delegate_type = [
            self.arena + n for n in [0x165000, 0x166000, 0x167000, 0x168000]]
        self.callback = self.stop + 0x200
        self.tutorial_ready = False
        s = json.loads((Path(dumper_root) / 'script.json').read_text(encoding='utf-8-sig'))
        dump = (Path(dumper_root) / 'dump.cs').read_text(encoding='utf-8')
        fields = {
            'TutorialsController': (5650, 'MonoBehaviour', ['TutorialNote[] allTutorials; // 0x20',
                'List<TutorialQueue> queuedTutorials; // 0x28', 'List<ETutorialType> showedTutorials; // 0x30',
                'Action<ETutorialType> onTutorialShow; // 0x38']),
            'TutorialNote': (5633, 'MonoBehaviour', ['string tutorialId; // 0x20', 'ETutorialState startingState; // 0x28',
                'ETutorialState state; // 0x2C', 'ETutorialType type; // 0x30', 'bool stopTime; // 0x34',
                'Action<TutorialNote> onHide; // 0x38', 'GameObject[] toturialStages; // 0x40',
                'int currentTutStage; // 0x48', 'bool clickable; // 0x4C']),
            'ProjectContext': (5546, 'MonoBehaviour', ['GameData gameData; // 0x20', 'static ProjectContext Instance; // 0x0']),
            'GameData': (5928, 'ScriptableObject', ['SavedGameData saveData; // 0x30']),
        }
        for name, (tdi, base, declarations) in fields.items():
            block = re.search(r'^public class ' + name + r' : ' + base + r' // TypeDefIndex: ' + str(tdi) + r'\s*\{(.*?)// Methods', dump, re.M | re.S)
            assert block and all(d in block[1] for d in declarations), name
        for declaration, required in [
            ('public class TutorialQueue // TypeDefIndex: 5651', ['ETutorialType type; // 0x10', 'Transform pivot; // 0x18', 'bool restriction; // 0x20']),
            ('private sealed class TutorialNote.<CloseCooldown>d__12 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5632',
             ['int <>1__state; // 0x10', 'object <>2__current; // 0x18', 'TutorialNote <>4__this; // 0x20']),
        ]:
            block = re.search(r'^' + re.escape(declaration) + r'\s*\{(.*?)// (?:Methods|Properties)', dump, re.M | re.S)
            assert block and all(field in block[1] for field in required)
        self.tutorial_fields = fields
        slots = {r['Address']: ('metadata', r['Name']) for r in s['ScriptMetadata']}
        slots.update({r['Address']: ('method', r['Name']) for r in s['ScriptMetadataMethod']})
        self.tutorial_bindings, self.tutorial_flags, self.tutorial_targets = {}, {}, []
        self.tutorial_instructions = {}
        for label, (a, b, name, method_id) in TARGETS.items():
            rows = [r for r in s['ScriptMethod'] if r['Address'] == a and r['Name'] == name]
            assert len(rows) == 1
            self.tutorial_targets.append(dict(rows[0], method_id=method_id))
            # Some methods span several unwind chunks. Decode to the verified
            # next managed entry / complete final trap, never first-chunk end.
            decoded = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
            assert sum(i.size for i in decoded) == b-a, (label, hex(a), hex(b), hex(decoded[-1].address + decoded[-1].size))
            self.tutorial_instructions.update({i.address: i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type == self.capstone.x86.X86_OP_MEM and op.mem.base == self.capstone.x86.X86_REG_RIP:
                        slot = i.address + i.size + op.mem.disp
                        if slot in slots:
                            self.tutorial_bindings[slot] = slots[slot]
                        elif i.mnemonic == 'cmp' and i.operands[0].size == 1:
                            self.tutorial_flags[slot] = label
        checks = {0x38C953: ('jmp', '0x38e1a0'), 0x38E25A: ('call', '0x38c430'),
            0x38E270: ('call', '0x38be20'), 0x38BF37: ('call', '0x3eac30'),
            0x38BF72: ('call', '0x3eaaa0'), 0x38BF7D: ('mov', 'dword ptr [rdi + 0x2c], 0xa'),
            0x38DFCB: ('inc', 'dword ptr [rax + 0x1c]'), 0x38DFD1: ('mov', 'dword ptr [rax + 0x18], r15d'),
            0x38E019: ('mov', 'dword ptr [rsi + 0x48], r15d'),
            0x38C015: ('call', '0x1c7d810'), 0x38C025: ('call', '0x1c8e540'),
            0x38C069: ('mov', 'dword ptr [rbx + 0x10], 0'), 0x38C07E: ('call', '0x1c7f160')}
        for a, expected in checks.items():
            i = self.tutorial_instructions[a]
            assert (i.mnemonic, i.op_str) == expected
        self.tutorial_assertions = len(checks)

    def alloc(self, size=0x80):
        token = self.tutorial_cursor
        self.tutorial_cursor += size
        assert self.tutorial_cursor < self.arena + 0x1F0000
        self.u.mem_write(token, bytes(size))
        return token

    def raw_array(self, entries):
        assert len(entries) <= 32
        token = self.alloc(0x20 + 8*len(entries) + 0x20)
        self.q(token+0x18, len(entries))
        for i, entry in enumerate(entries):
            self.q(token+0x20+8*i, entry)
        return token

    def raw_list(self, entries, width):
        token, backing = self.alloc(), self.alloc(0x120)
        assert len(entries) <= 32
        self.q(token+0x10, backing)
        self.q(backing+0x18, 32)
        for i, entry in enumerate(entries):
            (self.q if width == 8 else self.d)(backing+0x20+i*width, entry)
        self.d(token+0x18, len(entries)); self.d(token+0x1C, 0xFFFFFFFF)
        self.raw_lists[token] = width
        return token

    def raw_list_state(self, token):
        if not token:
            return None
        width = self.raw_lists[token]
        backing, count = self.rq(token+0x10), self.rd(token+0x18)
        assert count <= 32
        reader = self.rq if width == 8 else self.rd
        return {'identity': token, 'count': count, 'version': self.rd(token+0x1C),
                'values': [reader(backing+0x20+i*width) for i in range(count)],
                'backing_values': [reader(backing+0x20+i*width) for i in range(32)]}

    def setup_tutorials(self):
        o = self.options
        self.tutorial_cursor = self.arena + 0x170000
        self.notes, self.note_objects, self.note_transforms = [], {}, {}
        self.active, self.positions, self.raw_lists, self.queues, self.delegates, self.coroutines = {}, {}, {}, [], {}, []
        self.time_scale_bits = 0x3F800000
        self.q(self.project_type+0xB8, self.project_static)
        self.q(self.project_static, 0 if o.get('null_chain') == 'instance' else self.project)
        self.q(self.project+0x20, 0 if o.get('null_chain') == 'game_data' else self.game_data)
        self.q(self.game_data+0x30, 0 if o.get('null_chain') == 'save_data' else self.data)
        self.d(self.ui_type+0xE0, 1)
        for row in o.get('notes', [{'id': 'new-t', 'state': 0, 'type': 10, 'stages': [True, False]}]):
            note, obj, transform = self.alloc(), self.alloc(), self.alloc()
            self.notes.append(note); self.note_objects[note] = obj; self.note_transforms[note] = transform
            self.active[obj] = row.get('active', False); self.positions[transform] = [0x7FC12345, 0x80000000, 0x3F800000]
            self.q(note+0x20, self.string(row.get('id')))
            for off, value in [(0x28, row.get('starting', 100)), (0x2C, row.get('state', 0)),
                               (0x30, row.get('type', 10)), (0x48, row.get('stage', 7))]:
                self.d(note+off, value)
            self.u.mem_write(note+0x34, bytes([int(row.get('stop_time', False))]))
            self.u.mem_write(note+0x4C, b'\x01')
            stages = []
            for active in row.get('stages', []):
                stage = self.alloc(); self.active[stage] = active; stages.append(stage)
            if row.get('stage_aliases') is not None:
                stages = [stages[i] for i in row['stage_aliases']]
            if row.get('null_stage'):
                assert stages
                stages[-1] = 0
            self.q(note+0x40, self.raw_array(stages) if not row.get('null_stages') else 0)
        array_notes = self.notes.copy()
        if o.get('null_note'):
            array_notes[-1] = 0
        if o.get('duplicate_note'):
            array_notes *= 2
        self.q(self.controller+0x20, self.raw_array(array_notes) if not o.get('null_notes') else 0)
        self.q(self.controller+0x28, self.raw_list([0xABC001], 8))
        self.q(self.controller+0x30, self.raw_list([20, 10, 20], 4) if not o.get('null_showed') else 0)
        self.q(self.controller+0x38, 0)
        self.callback_observations = []
        if o.get('show_callback'):
            delegate = self.alloc()
            self.q(delegate+0x18, self.callback)
            self.q(delegate+0x28, self.controller+0x100)
            self.q(delegate+0x40, self.controller)
            self.q(self.controller+0x38, delegate)
        self.pivot = self.alloc() if o.get('pivot_live') else 0
        if self.pivot:
            self.positions[self.pivot] = [0x80000000, 0x7F800000, 0x3F800001]
        token_map = {'ProjectContext_TypeInfo': self.project_type, 'UnityEngine.Object_TypeInfo': self.ui_type,
                     'System.Action<TutorialNote>_TypeInfo': self.delegate_type,
                     'TutorialQueue_TypeInfo': self.queue_type,
                     'TutorialNote.<CloseCooldown>d__12_TypeInfo': self.iterator_type}
        self.tutorial_tokens = {}
        for slot, (kind, name) in self.tutorial_bindings.items():
            if name in token_map:
                token = token_map[name]
            elif name == 'Method$System.Collections.Generic.List<string>.Contains()':
                token = self.method_tokens['Contains']
            else:
                assert kind == 'method', name
                token = self.alloc()
            self.tutorial_tokens[slot] = token
            self.q(self.base+slot, token)
        for slot in self.tutorial_flags:
            self.u.mem_write(self.base+slot, bytes([int(o.get('warm', False))]))
        self.retained_notes = {n: bytes(self.u.mem_read(n, 0x80)) for n in self.notes}
        self.retained_controller = bytes(self.u.mem_read(self.controller, 0x40))
        stage_arrays = {self.rq(n+0x40) for n in self.notes} - {0}
        self.retained_arrays = {a: bytes(self.u.mem_read(a, 0x20+8*self.rq(a+0x18))) for a in stage_arrays}
        self.tutorial_ready = True

    def snapshot(self):
        result = super().snapshot()
        if not self.tutorial_ready:
            return result
        result['tutorials'] = {'showed': self.raw_list_state(self.rq(self.controller+0x30)),
            'queued': self.raw_list_state(self.rq(self.controller+0x28)),
            'notes': [{'identity': n, 'id': self.strings.get(self.rq(n+0x20)),
                'starting': self.rd(n+0x28), 'state': self.rd(n+0x2C), 'type': self.rd(n+0x30),
                'stage': self.rd(n+0x48), 'on_hide': self.rq(n+0x38),
                'clickable': self.u.mem_read(n+0x4C, 1)[0], 'stages': self.rq(n+0x40)} for n in self.notes],
            'active': self.active.copy(), 'positions': self.positions.copy(), 'time_scale_bits': self.time_scale_bits,
            'queue_records': [{'identity': q, 'type': self.rd(q+0x10), 'pivot': self.rq(q+0x18),
                               'restriction': self.u.mem_read(q+0x20, 1)[0]} for q in self.queues],
            'delegates': self.delegates.copy(), 'coroutines': self.coroutines.copy(),
            'callback_observations': self.callback_observations.copy()}
        return result

    def invoke(self, rva, receiver, argument=0):
        if rva == 0x3EAAA0 and self.options.get('tutorial_entry'):
            self.setup_tutorials()
            label = self.options['tutorial_entry']
            receiver = self.notes[0] if label.startswith('note_') else self.controller
            argument = self.pivot if label == 'note_show' else self.options.get('type', 10)
            self.entry_pending = True
            return super().invoke(TARGETS[label][0], receiver, argument)
        return super().invoke(rva, receiver, argument)

    def service(self, kind, args, effect):
        if self.event(kind, args):
            effect()

    def hook(self, uc, address, size, data):
        rva, x = address-self.base, self.x
        cx, dx, r8, r9 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        if self.tutorial_ready and self.entry_pending and rva == TARGETS[self.options['tutorial_entry']][0]:
            self.entry_pending = False
            if self.options['tutorial_entry'] in ['enable', 'show']:
                uc.reg_write(x.UC_X86_REG_R8, self.pivot)
        if not self.tutorial_ready:
            return super().hook(uc, address, size, data)
        self.executed.add(rva)
        if rva == 0x2B7B40 and cx-self.base in self.tutorial_bindings:
            self.service('tutorial_metadata_service', [self.tutorial_bindings[cx-self.base][1]], lambda: self.ret(self.rq(cx)))
        elif rva == 0x1C79FD0:
            assert cx in self.note_objects and dx == 0
            result = 0 if self.options.get('null_game_object') else self.note_objects[cx]
            self.service('component_game_object_service', [cx, result], lambda: self.ret(result))
        elif rva == 0x1C7DC50:
            assert cx in self.active and dx == 0
            self.service('active_self_service', [cx, self.active[cx]], lambda: self.ret(0xAABB000000000000 | int(self.active[cx])))
        elif rva == 0x1C7D810:
            assert cx in self.active and r8 == 0 and dx & 255 in [0, 1]
            def set_active():
                self.active[cx] = bool(dx & 255); self.ret()
            self.service('set_active_service', [cx, bool(dx & 255)], set_active)
        elif rva == 0x1C82480:
            assert dx == r8 == 0 and cx in [0, self.pivot]
            self.service('unity_live_service', [cx, bool(cx)], lambda: self.ret(0xAABB000000000000 | int(bool(cx))))
        elif rva == 0x1C7A010:
            assert cx in self.note_transforms and dx == 0
            result = 0 if self.options.get('null_note_transform') else self.note_transforms[cx]
            self.service('component_transform_service', [cx, result], lambda: self.ret(result))
        elif rva == 0x1C91B80:
            assert dx in self.positions and r8 == 0 and self.stack <= cx <= self.stack+0x20000-12
            def get_position():
                uc.mem_write(cx, struct.pack('<III', *self.positions[dx])); self.ret(cx)
            self.service('get_position_service', [dx, self.positions[dx]], get_position)
        elif rva == 0x1C923D0:
            assert cx in self.positions and r8 == 0 and self.stack <= dx <= self.stack+0x20000-12
            bits = list(struct.unpack('<III', uc.mem_read(dx, 12)))
            def position():
                self.positions[cx] = bits; self.ret()
            self.service('set_position_service', [cx, bits], position)
        elif rva == 0x1C8E540:
            assert dx == 0 and self.reg(x.UC_X86_REG_XMM0) & 0xFFFFFFFF == 0
            def time_scale():
                self.time_scale_bits = 0; self.ret()
            self.service('time_scale_service', [0], time_scale)
        elif rva == 0x2B7D40 and cx in [self.iterator_type, self.queue_type, self.delegate_type]:
            def allocation():
                token = self.alloc()
                if cx == self.queue_type:
                    self.queues.append(token)
                elif cx == self.delegate_type:
                    self.delegates[token] = None
                self.ret(token)
            self.service('tutorial_allocate_service', [cx], allocation)
        elif rva == 0x4D5B60:
            method = next(self.tutorial_tokens[a] for a, (_, n) in self.tutorial_bindings.items()
                          if n == 'Method$TutorialsController.OnTutHidden()')
            assert cx in self.delegates and dx == self.controller and r8 == method and r9 == 0
            def construct():
                self.delegates[cx] = {'target': dx, 'method': r8}; self.ret()
            self.service('tutorial_delegate_constructor_service', [cx, dx, r8], construct)
        elif rva == 0x116BCC0:
            assert r8 == 0 and dx in self.delegates and (cx == 0 or cx in self.delegates)
            def combine():
                if not cx:
                    self.ret(dx)
                else:
                    token = self.alloc()
                    self.delegates[token] = {'combined': [cx, dx]}
                    self.ret(token)
            self.service('tutorial_delegate_combine_service', [cx, dx], combine)
        elif rva == 0x2B7010:
            assert cx in self.delegates and dx == self.delegate_type
            self.service('tutorial_delegate_cast_service', [cx, dx], lambda: self.ret(cx))
        elif rva in [0x41A0, 0x2EB0]:
            assert cx in self.raw_lists and r8 in self.tutorial_tokens.values()
            width = self.raw_lists[cx]
            assert width == (4 if rva == 0x41A0 else 8)
            def append():
                count = self.rd(cx+0x18); assert count < 32
                (self.q if width == 8 else self.d)(self.rq(cx+0x10)+0x20+width*count, dx)
                self.d(cx+0x18, count+1); self.d(cx+0x1C, self.rd(cx+0x1C)+1); self.ret()
            self.service('tutorial_list_add_service', [cx, dx, width], append)
        elif rva == 0x1C7F160:
            assert cx in self.notes and r8 == 0 and self.rd(dx+0x10) == 0 and self.rq(dx+0x20) == cx
            def coroutine():
                self.coroutines.append({'identity': dx, 'actor': cx, 'state': self.rd(dx+0x10), 'current': self.rq(dx+0x18)}); self.ret(self.alloc())
            self.service('tutorial_start_coroutine_service', [cx, dx], coroutine)
        elif address == self.callback:
            assert cx == self.controller and r8 == self.controller+0x100
            def callback():
                self.callback_observations.append({'type': dx, 'showed': self.raw_list_state(self.rq(self.controller+0x30))}); self.ret()
            self.service('on_tutorial_show_service', [cx, dx, r8], callback)
        else:
            return super().hook(uc, address, size, data)

    def run_tutorial(self, label, state, options=None, storage=None):
        self.tutorial_ready = False
        result = self.run_data('Save', state, {'tutorial_entry': label, **(options or {})}, storage=storage)
        result['method'] = label
        result['boolean_return'] = bool(self.reg(self.x.UC_X86_REG_RAX) & 255) if label == 'closed' and result['returned'] else None
        allowed = set(range(0x28, 0x30)) | set(range(0x38, 0x40)) | set(range(0x48, 0x4C))
        assert all(all(a == b for i, (a, b) in enumerate(zip(before, self.u.mem_read(n, 0x80))) if i not in allowed)
                   for n, before in self.retained_notes.items())
        assert bytes(self.u.mem_read(self.controller, 0x40)) == self.retained_controller
        assert all(bytes(self.u.mem_read(a, len(before))) == before for a, before in self.retained_arrays.items())
        result['retained_unconsumed_note_fields_controller_references_stage_arrays_verified'] = True
        return result


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    old = {'key': 'Tutorials', 'completedTutorials': ['old-t'], 'unlockedCharactersId': ['c']}
    cases, baselines, failures = [], [], []
    for label, warm, starting, stages in itertools.product(['controller_reset', 'note_reset'], [False, True], [0, 10, 20, 100, -1], [[], [True], [True, True]]):
        result = m.run_tutorial(label, old, {'warm': warm, 'notes': [{'id': 'old-t', 'state': 10, 'starting': starting, 'stages': stages}]}, {})
        assert result['returned'] and values(result['final']['save']) == old and result['final']['storage'] == {}
        note = result['final']['tutorials']['notes'][0]
        assert note['stage'] == 0 and note['state'] == (10 if starting == 100 else starting & 0xFFFFFFFF)
        showed = result['final']['tutorials']['showed']
        assert showed['values'] == ([] if label == 'controller_reset' else [20, 10, 20])
        assert showed['version'] == (0 if label == 'controller_reset' else 0xFFFFFFFF)
        assert result['final']['tutorials']['queued']['values'] == [0xABC001]
        cases.append(result)
    for label, warm, state, completed, active, pivot, stop_time in itertools.product(['enable', 'show', 'note_show'], [False, True], [0, 10, 20, 100], [False, True], [False, True], [False, True], [False, True]):
        saved = dict(old, completedTutorials=['new-t'] if completed else ['old-t'])
        opts = {'warm': warm, 'pivot_live': pivot, 'notes': [{'id': 'new-t', 'state': state, 'active': active, 'stop_time': stop_time}]}
        result = m.run_tutorial(label, saved, opts, {})
        assert result['returned'], result['error']
        final = result['final']; presentations = final['tutorials']
        queued = label != 'note_show' and active
        reached = not queued and state != 10 and not completed
        writes = reached and state == 0
        assert len(final['storage_calls']) == int(writes)
        assert values(final['save'])['completedTutorials'] == saved['completedTutorials'] + (['new-t'] if writes else [])
        assert len(presentations['coroutines']) == int(reached)
        assert presentations['time_scale_bits'] == (0 if reached and stop_time else 0x3F800000)
        assert presentations['notes'][0]['state'] == (10 if writes else state)
        assert presentations['notes'][0]['starting'] == (state if state != 100 and not queued else 100)
        assert presentations['queued']['count'] == 1 + int(queued)
        # Controller publishes the requested type even when Note.Show early exits.
        assert presentations['showed']['values'] == [20, 10, 20] + ([10] if label != 'note_show' and not queued else [])
        cases.append(result)
    for label, options in [('controller_reset', {'null_notes': True}), ('controller_reset', {'null_showed': True}),
                            ('controller_reset', {'null_note': True}), ('note_reset', {'notes': [{'null_stages': True}]}),
                            ('note_reset', {'notes': [{'stages': [True, True], 'null_stage': True}]}),
                            ('controller_reset', {'null_note': True, 'notes': [{'stages': [True, True]}, {}]}),
                            ('note_show', {'null_chain': 'instance'}), ('note_show', {'null_chain': 'game_data'}),
                            ('note_show', {'null_chain': 'save_data'}), ('closed', {'null_notes': True}),
                            ('closed', {'null_note': True})]:
        result = m.run_tutorial(label, old, options, {})
        assert not result['returned'] and result['error'] == 'null_reference'
        cases.append(result)
    for label, opts in [('note_show', {'null_game_object': True}),
                        ('note_show', {'null_note_transform': True, 'pivot_live': True}),
                        ('enable', {'null_game_object': True})]:
        result = m.run_tutorial(label, old, opts, {})
        assert not result['returned'] and result['error'] == 'null_reference'
        assert len(result['final']['storage_calls']) == int(label == 'note_show')
        assert result['final']['tutorials']['coroutines'] == []
        cases.append(result)
    for completed in [False, True]:
        saved = dict(old, completedTutorials=['new-t'] if completed else ['old-t'])
        result = m.run_tutorial('enable', saved, {'show_callback': True}, {})
        assert result['returned']
        observation = result['final']['tutorials']['callback_observations']
        assert len(observation) == 1 and observation[0]['type'] == 10
        assert observation[0]['showed']['values'] == [20, 10, 20, 10]
        cases.append(result)
    for active in [False, True]:
        result = m.run_tutorial('closed', old, {'notes': [{'active': active}]}, {})
        assert result['returned'] and result['boolean_return'] == (not active)
        cases.append(result)
    for label in ['controller_reset', 'note_reset']:
        result = m.run_tutorial(label, old, {'duplicate_note': True, 'notes': [
            {'stages': [True, False], 'stage_aliases': [0, 0], 'starting': 20}]}, {})
        assert result['returned']
        calls = [e for e in result['events'] if e['kind'] == 'set_active_service']
        assert len(calls) == (6 if label == 'controller_reset' else 3)
        assert all(e['args'][0] == calls[0]['args'][0] for e in calls)
        assert result['final']['tutorials']['active'][calls[0]['args'][0]]
        cases.append(result)
    for notes, requested, expected_shown, expected_queued, completed, duplicate in [
        ([], 10, [], 0, ['old-t'], False),
        ([{'type': 20}], 10, [], 0, ['old-t'], False),
        ([{'id': 'a', 'type': 10}, {'id': 'b', 'type': 10}], 10, [10], 1, ['old-t', 'a'], False),
        ([{'id': 'old-t', 'type': 10, 'state': 0}], 10, [10, 10], 0, ['old-t'], True),
    ]:
        result = m.run_tutorial('enable', old, {'notes': notes, 'type': requested, 'duplicate_note': duplicate}, {})
        assert result['returned']
        assert result['final']['tutorials']['showed']['values'] == [20, 10, 20] + expected_shown
        assert result['final']['tutorials']['queued']['count'] == 1+expected_queued
        assert values(result['final']['save'])['completedTutorials'] == completed
        cases.append(result)
    for storage_options in [{'registry_status': 5}, {'create_status': 5}]:
        result = m.run_tutorial('enable', old, {'storage_options': storage_options}, {})
        assert not result['returned'] and result['error'] == 'preference_exception'
        assert values(result['final']['save'])['completedTutorials'] == ['old-t', 'new-t']
        assert result['final']['tutorials']['notes'][0]['state'] == 0
        assert result['final']['tutorials']['showed']['values'] == [20, 10, 20]
        assert result['final']['tutorials']['coroutines'] == [] and result['final']['storage'] == {}
        cases.append(result)
    for tutorial_id in ['new-t', None, '', 'caf\u00e9']:
        result = m.run_tutorial('enable', old, {'notes': [{'id': tutorial_id, 'state': 0}]}, {})
        assert result['returned']
        m.tutorial_ready = False
        loaded = m.run_data('Load', old, storage=result['final']['storage'])
        assert loaded['returned'] and values(loaded['final']['save']) == result['joined_engine'][0]['loaded']['values']
        cases.append({'label': 'persisted_completion_round_trip', 'write': result, 'read': loaded})
    for label, opts in [('controller_reset', {'notes': [{'starting': 20, 'stages': [False, True]}, {'starting': 0, 'stages': [True]}]}),
                        ('enable', {'pivot_live': True, 'show_callback': True, 'notes': [{'id': 'new-t', 'state': 0, 'stop_time': True}]}),
                        ('show', {'notes': [{'active': True}]}), ('note_show', {'notes': [{'state': 20}]})]:
        baseline = m.run_tutorial(label, old, opts, {})
        assert baseline['returned']
        baseline_id = len(baselines); baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0)+1
            result = m.run_tutorial(label, old, dict(opts, failure=[kind, counts[kind]]), {})
            assert not result['returned'] and result['events'] == baseline['events'][:index+1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index+1, 'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, 'targets': m.tutorial_targets, 'fields': m.tutorial_fields,
        'metadata_bindings': [{'rva': hex(a), 'kind': kind, 'name': name} for a, (kind, name) in sorted(m.tutorial_bindings.items())],
        'instruction_assertions': m.tutorial_assertions, 'cases': cases, 'case_count': len(cases),
        'native_ranges': {label: [hex(row[0]), hex(row[1])] for label, row in TARGETS.items()},
        'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
        'executed_address_count': len(m.executed),
        'scope': 'Actual seven native tutorial bodies plus SavedGameInfo.AddTutorial/SavedGameData.Save/JSON/native preference/runtime strings execute. Explicit inert Unity, List append, delegate allocation/constructor/combine/cast, coroutine publication and runtime metadata services; no Unity scheduling, actual registry, or cross-emulator object identity.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(compact_trace(result), indent=2)+'\n', encoding='utf-8')
    print(result['case_count'], result['failure_case_count'], result['executed_address_count'])
