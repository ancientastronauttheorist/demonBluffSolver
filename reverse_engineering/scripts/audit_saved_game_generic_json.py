"""Execute the shared generic FromJson wrapper inside native SavedGameData.Load."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_data import Machine as DataMachine
from audit_saved_game_info_json import Machine as JsonMachine


class Machine(DataMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        self.context, self.handle, self.generic_class, self.system_type, self.type_object = [self.arena + n for n in (0x130000, 0x131000, 0x132000, 0x133000, 0x134000)]
        entry = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == 0x645DA0)
        assert entry.struct.EndAddress == 0x645E5D and not entry.unwindinfo.Flags & 4
        decoded = list(self.cs.disasm(self.pe.get_data(0x645DA0, 0xBD), 0x645DA0))
        assert sum(i.size for i in decoded) == 0xBD
        self.generic_instructions = {i.address: i for i in decoded}
        checks = {
            0x645DAF: ('cmp', 'qword ptr [rdx + 0x38], 0'),
            0x645DC3: ('call', '0x2b7b40'), 0x645DC8: ('cmp', 'qword ptr [rdi + 0x38], 0'),
            0x645DD2: ('call', '0x29c910'), 0x645DDE: ('mov', 'rbx, qword ptr [rdi + 0x38]'),
            0x645DE2: ('cmp', 'dword ptr [rcx + 0xe0], 0'),
            0x645DE9: ('mov', 'rbx, qword ptr [rbx]'), 0x645DEE: ('call', '0x281d90'),
            0x645DF8: ('call', '0x113fca0'), 0x645E06: ('call', '0x1cd61d0'),
            0x645E0B: ('mov', 'rcx, qword ptr [rdi + 0x38]'),
            0x645E12: ('mov', 'rbx, qword ptr [rcx + 8]'),
            0x645E16: ('test', 'byte ptr [rbx + 0x135], 1'),
            0x645E22: ('call', '0x29c890'), 0x645E2A: ('test', 'rsi, rsi'),
            0x645E2F: ('xor', 'eax, eax'), 0x645E40: ('ret', ''),
            0x645E47: ('call', '0x2b7010'), 0x645E4C: ('test', 'rax, rax'),
            0x645E57: ('call', '0x2b7040'), 0x645E5C: ('int3', ''),
        }
        for a, expected in checks.items():
            assert a in self.generic_instructions
            i = self.generic_instructions[a]
            assert (i.mnemonic, i.op_str) == expected
        i = self.generic_instructions[0x645DD7]
        assert i.mnemonic == 'mov' and i.operands[1].mem.base == self.capstone.x86.X86_REG_RIP
        self.type_slot = i.address + i.size + i.operands[1].mem.disp
        assert self.type_slot == 0x26E39E0
        script = json.loads((Path(dumper_root) / 'script.json').read_text(encoding='utf-8-sig'))
        rows = [r for r in script['ScriptMetadata'] if r['Address'] == self.type_slot]
        assert len(rows) == 1 and rows[0]['Name'] == 'System.Type_TypeInfo'
        self.bindings[self.type_slot] = ('metadata', rows[0]['Name'])
        self.runtime_targets = []
        for rva, name, signature in [
            (0x113FCA0, 'System.Type$$GetTypeFromHandle', 'System_Type_o* System_Type__GetTypeFromHandle (System_RuntimeTypeHandle_o handle, const MethodInfo* method);'),
            (0x1CD61D0, 'UnityEngine.JsonUtility$$FromJson', 'Il2CppObject* UnityEngine_JsonUtility__FromJson (System_String_o* json, System_Type_o* type, const MethodInfo* method);')]:
            rows = [r for r in script['ScriptMethod'] if r['Address'] == rva and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['Signature'] == signature
            self.runtime_targets.append(rows[0])
        self.generic_instruction_assertions = len(checks) + 1

    def invoke(self, rva, receiver, argument=0):
        self.bindings[self.type_slot] = self.runtime_binding
        self.q(self.base + self.type_slot, self.system_type)
        self.q(self.context, self.handle)
        self.q(self.context + 8, self.generic_class)
        self.q(self.from_json_method + 0x38, self.context if self.options.get('context_warm') else 0)
        self.d(self.system_type + 0xE0, int(self.options.get('type_initialized', False)))
        self.u.mem_write(self.generic_class + 0x135, bytes([int(self.options.get('class_resolved', False))]))
        return super().invoke(rva, receiver, argument)

    def snapshot(self):
        state = super().snapshot()
        state['generic_context_published'] = bool(self.rq(self.from_json_method + 0x38))
        state['system_type_initialized'] = bool(self.rd(self.system_type + 0xE0))
        state['generic_class_resolved'] = bool(self.u.mem_read(self.generic_class + 0x135, 1)[0] & 1)
        return state

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        cx, dx, r8 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        if rva == 0x645DA0:
            assert dx == self.from_json_method and cx in self.strings
            self.executed.add(rva)
            return  # Execute the actual wrapper rather than the parent service.
        if rva == 0x29C910:
            assert cx == self.from_json_method and self.rq(cx + 0x38) == 0
            self.executed.add(rva)
            if self.event('generic_context_initialize_service', []):
                self.q(cx + 0x38, self.context)
                self.ret()
        elif rva == 0x281D90:
            assert cx == self.system_type and self.rd(cx + 0xE0) == 0
            self.executed.add(rva)
            if self.event('system_type_initialize_service', []):
                self.d(cx + 0xE0, 1)
                self.ret()
        elif rva == 0x113FCA0:
            assert cx == self.handle and dx == 0
            self.executed.add(rva)
            if self.event('type_from_handle_service', []):
                self.ret(self.type_object)
        elif rva == 0x1CD61D0:
            assert cx in self.strings and dx == self.type_object and r8 == 0
            self.executed.add(rva)
            payload = self.strings[cx]
            if self.event('non_generic_from_json_service', [payload]):
                values = self.options.get('loaded_values', {'key': 'loaded-key', 'completedTutorials': ['loaded-t'], 'unlockedCharactersId': ['loaded-c']})
                if self.options.get('engine_join'):
                    report = self.engine.read_saved(payload)
                    assert report['returned']
                    values = report['loaded']['values']
                    self.joins.append({'direction': 'read', 'json': payload, 'loaded': report['loaded']})
                self.ret(self.new_info(values) if values is not None else 0)
        elif rva == 0x29C890:
            assert cx == self.generic_class and not self.u.mem_read(cx + 0x135, 1)[0] & 1
            self.executed.add(rva)
            if self.event('generic_class_resolve_service', []):
                self.u.mem_write(cx + 0x135, b'\1')
                self.ret(self.generic_class)
        elif rva == 0x2B7010:
            assert cx in self.infos and dx == self.generic_class
            self.executed.add(rva)
            if self.event('cast_service', [self.infos.index(cx), bool(self.options.get('cast_success', True))]):
                self.ret(cx if self.options.get('cast_success', True) else 0)
        elif rva == 0x2B7040:
            assert cx in self.infos and dx == self.generic_class
            self.executed.add(rva)
            self.event('cast_failure', [self.infos.index(cx)])
            self.error = 'cast_failure'
            self.u.emu_stop()
        else:
            super().hook(uc, address, size, data)

    def run_data(self, method, state=None, options=None):
        # The new runtime slot is initialized at invoke, after shared setup.
        binding = self.bindings.pop(self.type_slot)
        try:
            # Preserve the slot for any metadata initializer reached by native code.
            self.runtime_binding = binding
            return super().run_data(method, state, options)
        finally:
            self.bindings[self.type_slot] = binding


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    old = {'key': 'old-key', 'completedTutorials': ['old-t'], 'unlockedCharactersId': ['old-c']}
    cases, failures, baselines, joined = [], [], [], []
    for warm, context_warm, initialized, resolved, result in itertools.product([False, True], [False, True], [False, True], [False, True], ['success', 'null', 'cast_failure']):
        options = {'warm': warm, 'context_warm': context_warm, 'type_initialized': initialized,
                   'class_resolved': resolved, 'get_values': ['first', 'second'], 'cast_success': result != 'cast_failure'}
        if result == 'null':
            options['loaded_values'] = None
        case = m.run_data('Load', old, options)
        assert case['returned'] == (result != 'cast_failure')
        kinds = [e['kind'] for e in case['events']]
        assert ('generic_context_initialize_service' in kinds) == (not context_warm)
        assert ('system_type_initialize_service' in kinds) == (not initialized)
        assert ('generic_class_resolve_service' in kinds) == (not resolved)
        assert ('cast_service' in kinds) == (result != 'null')
        if result == 'null':
            assert case['final']['save'] is None
        elif result == 'cast_failure':
            assert case['error'] == 'cast_failure' and case['final']['save']['key'] == 'old-key'
        else:
            assert case['final']['save']['key'] == 'loaded-key'
        cases.append(case)
    for warm, result in itertools.product([False, True], ['success', 'null', 'cast_failure']):
        options = {'warm': warm, 'get_values': ['first', 'second'], 'cast_success': result != 'cast_failure'}
        if result == 'null':
            options['loaded_values'] = None
        baseline = m.run_data('Load', old, options)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            case = m.run_data('Load', old, dict(options, failure=[kind, counts[kind]]))
            assert not case['returned'] and case['events'] == baseline['events'][:index + 1]
            assert case['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1, 'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    m.engine = JsonMachine(game_root)
    for warm, context_warm in itertools.product([False, True], repeat=2):
        payload = '{"key":"native-wrapper","completedTutorials":["a","a"],"unlockedCharactersId":["b"]}'
        case = m.run_data('Load', old, {'warm': warm, 'context_warm': context_warm, 'get_values': ['first', payload], 'engine_join': True})
        assert case['returned'] and case['final']['save']['key'] == 'native-wrapper'
        assert len(case['joined_engine']) == 1
        joined.append(case)
    return {'build_id': BUILD, 'generic_native_range': ['0x645da0', '0x645e5d'],
            'instruction_assertions': m.generic_instruction_assertions, 'runtime_targets': m.runtime_targets,
            'cases': cases, 'case_count': len(cases), 'failure_baselines': baselines,
            'failure_cases': failures, 'failure_case_count': len(failures), 'joined_cases': joined,
            'joined_case_count': len(joined), 'executed_address_count': len(m.executed),
            'scope': 'Actual shared generic FromJson body executes inside native SavedGameData.Load. Generic context publication, System.Type initialization/conversion, non-generic FromJson gateway, class resolution and casts remain explicit runtime services. Four joins execute native engine field loading over authored zero-initialized destinations. No live preferences, actual runtime constructor or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['failure_case_count'], result['joined_case_count'], result['executed_address_count'])
