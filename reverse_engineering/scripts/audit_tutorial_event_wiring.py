"""Execute tutorial event registration with explicit inert delegate services."""
import argparse
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_storage_join import compact_trace
from audit_tutorial_persistence_join import Machine as TutorialMachine, StorageMachine

ENTRIES = {'enable': (0x38D4A0, 0x38DB71, 'OnEnable', 'tdi5650.m0000'),
           'disable': (0x38CDC0, 0x38D491, 'OnDisable', 'tdi5650.m0001')}
BINDINGS = [
    ('OnGameStart', 0, 'System.Action_TypeInfo', 'StartTutorials', 0x38E3D0, 'tdi5650.m0011'),
    ('OnCharacterRevealed', 0x50, 'System.Action<Character>_TypeInfo', 'OnCharacterReveal', 0x38CC30, 'tdi5650.m0010'),
    ('OnCharacterInfoRevealed', 0x58, 'System.Action<Character>_TypeInfo', 'CharacterInfoNote', 0x38C170, 'tdi5650.m0012'),
    ('OnCharacterKilled', 0x48, 'System.Action<Character>_TypeInfo', 'CharacterKilledTutorial', 0x38C330, 'tdi5650.m0006'),
    ('OnShowTutorial', 0x90, 'System.Action<ETutorialType, Transform>_TypeInfo', 'EnableTutorial', 0x38C950, 'tdi5650.m0009'),
    ('OnCloseTutorial', 0x98, 'System.Action<ETutorialType>_TypeInfo', 'CloseTutorialIfAble', 0x38C8C0, 'tdi5650.m0013'),
    ('OnStartNewLevel', 0x28, 'System.Action_TypeInfo', 'LevelIdTutorial', 0x38CAD0, 'tdi5650.m0003'),
]


class Machine(TutorialMachine):
    def __init__(self, game_root, dumper_root):
        self.wiring_ready = False
        super().__init__(game_root, dumper_root)
        script = json.loads((Path(dumper_root)/'script.json').read_text(encoding='utf-8-sig'))
        dump = (Path(dumper_root)/'dump.cs').read_text(encoding='utf-8')
        block = re.search(r'^public static class GameplayEvents // TypeDefIndex: 5519\s*\{(.*?)^\}', dump, re.M|re.S)
        assert block
        self.event_fields = [(name, int(offset, 16)) for name, offset in
            re.findall(r'public static Action(?:<[^\n]+>)? (\w+); // (0x[0-9A-Fa-f]+)', block[1])]
        assert len(self.event_fields) == 29
        assert all((name, off) in self.event_fields for name, off, *_ in BINDINGS)
        slots = {r['Address']: ('metadata', r['Name']) for r in script['ScriptMetadata']}
        slots.update({r['Address']: ('method', r['Name']) for r in script['ScriptMetadataMethod']})
        self.wiring_bindings, self.wiring_flags, self.wiring_instructions, self.wiring_targets = {}, {}, {}, []
        for label, (a, b, name, method_id) in ENTRIES.items():
            rows = [r for r in script['ScriptMethod'] if r['Address'] == a and r['Name'] == 'TutorialsController$$'+name]
            assert len(rows) == 1 and rows[0]['Signature'] == f'void TutorialsController__{name} (TutorialsController_o* __this, const MethodInfo* method);'
            self.wiring_targets.append(dict(rows[0], method_id=method_id))
            entry = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == a)
            assert entry.struct.EndAddress == b and not entry.unwindinfo.Flags & 4
            decoded = list(self.cs.disasm(self.pe.get_data(a, b-a), a))
            assert sum(i.size for i in decoded) == b-a
            self.wiring_instructions.update({i.address: i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type == self.capstone.x86.X86_OP_MEM and op.mem.base == self.capstone.x86.X86_REG_RIP:
                        address = i.address+i.size+op.mem.disp
                        if address in slots:
                            self.wiring_bindings[address] = slots[address]
                        elif i.mnemonic == 'cmp' and i.operands[0].size == 1:
                            self.wiring_flags[label] = address
        assert len(self.wiring_bindings) == 12
        assert {name for _, name in self.wiring_bindings.values()} == {
            'GameplayEvents_TypeInfo', *[row[2] for row in BINDINGS],
            *['Method$TutorialsController.'+row[3]+'()' for row in BINDINGS]}
        self.handler_targets = []
        for event, off, type_name, method, rva, method_id in BINDINGS:
            rows = [r for r in script['ScriptMethod'] if r['Address'] == rva and r['Name'] == 'TutorialsController$$'+method]
            assert len(rows) == 1
            self.handler_targets.append(dict(rows[0], method_id=method_id, event=event, static_offset=hex(off), delegate_type=type_name))
        self.gameplay_type, self.gameplay_static, self.foreign_type = [self.arena+n for n in [0x180000, 0x181000, 0x182000]]
        self.registration_types = {name: self.arena+0x183000+i*0x1000 for i, name in enumerate(dict.fromkeys(row[2] for row in BINDINGS))}
        self.wiring_assertions = 0
        for label, combine, ctor_calls in [('enable', 0x116BCC0, [0x38D58B, 0x38D643, 0x38D708, 0x38D7CD, 0x38D895, 0x38D966, 0x38DA34]),
                                           ('disable', 0x116E070, [0x38CEAB, 0x38CF63, 0x38D028, 0x38D0ED, 0x38D1B5, 0x38D286, 0x38D354])]:
            for call, row in zip(ctor_calls, BINDINGS):
                i = self.wiring_instructions[call]
                assert i.mnemonic == 'call' and i.op_str == hex({'System.Action_TypeInfo': 0x4D5170,
                    'System.Action<Character>_TypeInfo': 0x4D5B60,
                    'System.Action<ETutorialType, Transform>_TypeInfo': 0x4D5D10,
                    'System.Action<ETutorialType>_TypeInfo': 0x4D5E50}[row[2]])
                self.wiring_assertions += 1
            calls = [i for i in self.wiring_instructions.values() if ENTRIES[label][0] <= i.address < ENTRIES[label][1]
                     and i.mnemonic == 'call' and i.op_str == hex(combine)]
            assert len(calls) == 7
            self.wiring_assertions += len(calls)

    def new_delegate(self, type_name, invocations=None):
        token = self.wiring_cursor
        self.wiring_cursor += 0x80
        assert self.wiring_cursor < self.arena+0x1F0000
        self.u.mem_write(token, bytes(0x80))
        foreign = (self.options.get('force_cast') and type_name != 'System.Action_TypeInfo') or (
            self.options.get('wrong_plain_type') and type_name == 'System.Action_TypeInfo')
        self.q(token, self.foreign_type if foreign else self.registration_types[type_name])
        self.registration_delegates[token] = {'type': type_name, 'invocations': invocations or []}
        return token

    def setup_wiring(self):
        self.registration_delegates, self.registration_methods = {}, {}
        self.wiring_cursor = self.arena+0x190000
        self.q(self.gameplay_type+0xB8, self.gameplay_static)
        self.d(self.gameplay_type+0xE0, int(not self.options.get('class_cold')))
        for name, off in self.event_fields:
            self.q(self.gameplay_static+off, 0xFACE0000+off)
        type_for_event = {name: type_name for name, _, type_name, *_ in BINDINGS}
        for name, off, type_name, method, *_ in BINDINGS:
            handler_slot = next(a for a, (_, n) in self.wiring_bindings.items() if n == 'Method$TutorialsController.'+method+'()')
            token = self.wiring_cursor; self.wiring_cursor += 0x80
            self.registration_methods[token] = {'method': method, 'type': type_name, 'slot': handler_slot}
            self.q(self.base+handler_slot, token)
        for slot, (kind, name) in self.wiring_bindings.items():
            if kind == 'metadata':
                self.q(self.base+slot, self.gameplay_type if name == 'GameplayEvents_TypeInfo' else self.registration_types[name])
        for name, off, type_name, method, *_ in BINDINGS:
            initial = []
            if self.options.get('prior'):
                initial.append({'target': 0xAFFE0001, 'method': 'authored_prior_handler'})
            if self.options.get('preexisting_own'):
                initial.append({'target': self.controller, 'method': method})
            if self.options.get('trailing_prior'):
                initial.append({'target': 0xAFFE0002, 'method': 'authored_trailing_handler'})
            self.q(self.gameplay_static+off, self.new_delegate(type_name, initial) if initial else 0)
        for slot in self.wiring_flags.values():
            self.u.mem_write(self.base+slot, bytes([int(self.options.get('warm', False))]))
        self.registration_allocations = []
        self.wiring_ready = True

    def snapshot(self):
        # Registration reads no tutorial note or persisted-save field.
        result = StorageMachine.snapshot(self)
        if self.wiring_ready:
            result['registration'] = {
                'fields': {name: self.rq(self.gameplay_static+off) for name, off in self.event_fields},
                'delegates': {k: {'type': v['type'], 'invocations': [dict(i) for i in v['invocations']]} for k, v in self.registration_delegates.items()},
                'metadata_initialized': {name: bool(self.u.mem_read(self.base+slot, 1)[0]) for name, slot in self.wiring_flags.items()},
                'class_initialized': bool(self.rd(self.gameplay_type+0xE0)),
                'allocation_order': self.registration_allocations.copy()}
        return result

    def invoke(self, rva, receiver, argument=0):
        if rva == 0x3EAAA0 and self.options.get('wiring_sequence'):
            self.setup_wiring()
            self.step_reports = []
            for label in self.options['wiring_sequence']:
                returned = StorageMachine.invoke(self, ENTRIES[label][0], self.controller)
                self.step_reports.append({'entry': label, 'returned': returned, 'state': self.snapshot()['registration']})
                if not returned:
                    return False
            return True
        return StorageMachine.invoke(self, rva, receiver, argument)

    def hook(self, uc, address, size, data):
        if not self.wiring_ready:
            return StorageMachine.hook(self, uc, address, size, data)
        rva, x = address-self.base, self.x
        cx, dx, r8, r9 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        self.executed.add(rva)
        if rva == 0x2B7B40 and cx-self.base in self.wiring_bindings:
            self.service('registration_metadata_service', [self.wiring_bindings[cx-self.base][1]], lambda: self.ret(self.rq(cx)))
        elif rva == 0x2B7D40 and cx in self.registration_types.values():
            type_name = next(n for n, t in self.registration_types.items() if t == cx)
            def allocation():
                token = self.new_delegate(type_name)
                self.registration_allocations.append(token); self.ret(token)
            self.service('registration_allocate_service', [type_name], allocation)
        elif rva in [0x4D5170, 0x4D5B60, 0x4D5D10, 0x4D5E50]:
            assert cx in self.registration_delegates and dx == self.controller and r8 in self.registration_methods and r9 == 0
            method = self.registration_methods[r8]
            assert self.registration_delegates[cx]['type'] == method['type']
            def constructor():
                self.registration_delegates[cx]['invocations'] = [{'target': dx, 'method': method['method']}]; self.ret()
            self.service('registration_delegate_constructor_service', [cx, dx, r8, method], constructor)
        elif rva in [0x116BCC0, 0x116E070]:
            assert (cx == 0 or cx in self.registration_delegates) and dx in self.registration_delegates and r8 == 0
            right = self.registration_delegates[dx]
            assert not cx or self.registration_delegates[cx]['type'] == right['type']
            def combination():
                before = self.registration_delegates[cx]['invocations'].copy() if cx else []
                if rva == 0x116BCC0:
                    result = dx if not cx else self.new_delegate(right['type'], before+right['invocations'])
                else:
                    own = right['invocations']
                    last = next((i for i in range(len(before)-len(own), -1, -1) if before[i:i+len(own)] == own), None)
                    if last is None:
                        result = cx
                    else:
                        remaining = before[:last]+before[last+len(own):]
                        result = self.new_delegate(right['type'], remaining) if remaining else 0
                self.ret(result)
            self.service('registration_'+('combine' if rva == 0x116BCC0 else 'remove')+'_service', [cx, dx], combination)
        elif rva == 0x2B7010:
            assert cx in self.registration_delegates and dx in self.registration_types.values()
            assert self.registration_delegates[cx]['type'] == next(n for n, t in self.registration_types.items() if t == dx)
            self.service('registration_cast_service', [cx, dx], lambda: self.ret(0 if self.options.get('cast_fail') or
                self.options.get('cast_fail_at') == self.counts.get('registration_cast_service') else cx))
        elif rva == 0x2B7040:
            assert cx in self.registration_delegates and dx in self.registration_types.values()
            self.event('registration_cast_failure', [cx, dx]); self.error = 'registration_cast_failure'; uc.emu_stop()
        elif rva == 0x2B6FF0:
            assert self.rq(cx) == dx and self.gameplay_static <= cx < self.gameplay_static+0xE8
            offset = cx-self.gameplay_static
            name = next(n for n, off in self.event_fields if off == offset)
            self.service('registration_barrier_service', [name, dx], lambda: self.ret())
        elif rva == 0x281D90:
            assert cx == self.gameplay_type
            def initialization():
                self.d(cx+0xE0, 1); self.ret(cx)
            self.service('registration_class_initialize_service', ['GameplayEvents'], initialization)
        else:
            return StorageMachine.hook(self, uc, address, size, data)

    def run_wiring(self, sequence, options=None):
        self.wiring_ready = False
        result = self.run_data('Save', {'key': 'untouched', 'completedTutorials': ['t'], 'unlockedCharactersId': ['c']},
                               {'wiring_sequence': sequence, **(options or {})}, storage={})
        result['method'] = 'event_wiring'
        result['steps'] = self.step_reports.copy()
        expected_offsets = {row[1] for row in BINDINGS}
        assert all(self.rq(self.gameplay_static+off) == 0xFACE0000+off for _, off in self.event_fields if off not in expected_offsets)
        assert result['final']['storage_calls'] == [] and result['final']['preference_writes'] == []
        result['unrelated_static_fields_and_persistence_retained_verified'] = True
        return result


def invocations(result, event):
    r = result['final']['registration']; pointer = r['fields'][event]
    return r['delegates'][pointer]['invocations'] if pointer else []


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, baselines, failures = [], [], []
    sequences = [['enable'], ['disable'], ['enable', 'disable'], ['enable', 'enable'],
                 ['enable', 'enable', 'disable'], ['enable', 'disable', 'disable'], ['disable', 'enable']]
    for seq, warm, prior, own, trailing, cast in itertools.product(sequences, [False, True], [False, True], [False, True], [False, True], [False, True]):
        options = {'warm': warm, 'prior': prior, 'preexisting_own': own, 'trailing_prior': trailing, 'force_cast': cast, 'class_cold': not warm}
        result = m.run_wiring(seq, options)
        assert result['returned'], (seq, options, result['error'])
        for event, _, _, method, *_ in BINDINGS:
            expected = ([{'target': 0xAFFE0001, 'method': 'authored_prior_handler'}] if prior else [])
            own_handler = {'target': m.controller, 'method': method}
            if own:
                expected.append(own_handler)
            if trailing:
                expected.append({'target': 0xAFFE0002, 'method': 'authored_trailing_handler'})
            for entry in seq:
                if entry == 'enable':
                    expected.append(own_handler)
                else:
                    last = next((i for i in range(len(expected)-1, -1, -1) if expected[i] == own_handler), None)
                    if last is not None:
                        expected.pop(last)
            assert invocations(result, event) == expected
        cases.append(result)
    for sequence, options in [(['enable'], {'force_cast': True, 'cast_fail': True}),
                              (['enable'], {'wrong_plain_type': True}),
                              (['enable'], {'force_cast': True, 'cast_fail_at': 2}),
                              (['disable'], {'force_cast': True, 'cast_fail': True, 'preexisting_own': True, 'prior': True})]:
        result = m.run_wiring(sequence, options)
        assert not result['returned'] and result['error'] == 'registration_cast_failure'
        if options.get('cast_fail_at') == 2:
            assert invocations(result, 'OnCharacterRevealed') == [{'target': m.controller, 'method': 'OnCharacterReveal'}]
            assert not any(e['kind'] == 'registration_barrier_service' and e['args'][0] == 'OnCharacterRevealed' for e in result['events'])
        cases.append(result)
    for sequence, options in [(['enable'], {'force_cast': True, 'class_cold': True, 'prior': True}),
                              (['disable'], {'force_cast': True, 'preexisting_own': True, 'prior': True})]:
        baseline = m.run_wiring(sequence, options); assert baseline['returned']
        baseline_id = len(baselines); baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0)+1
            result = m.run_wiring(sequence, dict(options, failure=[kind, counts[kind]]))
            assert not result['returned'] and result['events'] == baseline['events'][:index+1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index+1, 'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, 'targets': m.wiring_targets, 'handler_targets': m.handler_targets,
            'event_fields': [{'name': n, 'offset': hex(off)} for n, off in m.event_fields],
            'metadata_bindings': [{'rva': hex(a), 'kind': kind, 'name': name} for a, (kind, name) in sorted(m.wiring_bindings.items())],
            'native_ranges': {label: [hex(row[0]), hex(row[1])] for label, row in ENTRIES.items()},
            'instruction_assertions': m.wiring_assertions, 'case_count': len(cases), 'cases': cases,
            'failure_baselines': baselines, 'failure_case_count': len(failures), 'failure_cases': failures,
            'executed_address_count': len(m.executed),
            'scope': 'Both complete native OnEnable/OnDisable registration callers execute. Runtime metadata/class initialization, allocation, folded delegate construction, Combine/Remove, casts and write barriers are explicit bounded services. Retained ordered invocation tokens are authored; no managed multicast implementation or event dispatch executes. All unrelated GameplayEvents statics and save/persistence fields remain retained.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(compact_trace(result), indent=2)+'\n', encoding='utf-8')
    print(result['case_count'], result['failure_case_count'], result['executed_address_count'])
