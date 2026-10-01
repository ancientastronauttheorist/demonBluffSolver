"""Execute the native tutorial-reset UI caller through JSON and storage."""
import argparse
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_runtime_strings import Machine as RuntimeStorageMachine
from audit_saved_game_storage_join import compact_trace, values


class Machine(RuntimeStorageMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        self.button_entry, self.project_slot, self.button_flag = 0x3A6600, 0x271F268, 0x288C3B9
        self.project_type, self.project_static, self.project, self.game_data, self.button = [
            self.arena + n for n in [0x150000, 0x151000, 0x152000, 0x153000, 0x154000]]
        self.button_storage = []
        script = json.loads((Path(dumper_root) / 'script.json').read_text(encoding='utf-8-sig'))
        rows = [r for r in script['ScriptMethod'] if r['Address'] == self.button_entry
                and r['Name'] == 'ResetTutorialButton$$ResetTutorials']
        assert len(rows) == 1 and rows[0]['Signature'] == (
            'void ResetTutorialButton__ResetTutorials (ResetTutorialButton_o* __this, const MethodInfo* method);')
        self.button_target = rows[0]
        metadata = [r for r in script['ScriptMetadata'] if r['Address'] == self.project_slot]
        assert len(metadata) == 1 and metadata[0]['Name'] == 'ProjectContext_TypeInfo'
        dump = (Path(dumper_root) / 'dump.cs').read_text(encoding='utf-8')
        self.button_fields = {}
        for name, declaration, required in [
            ('ProjectContext', 'public class ProjectContext : MonoBehaviour // TypeDefIndex: 5546',
             ['public GameData gameData; // 0x20', 'public static ProjectContext Instance; // 0x0']),
            ('GameData', 'public class GameData : ScriptableObject // TypeDefIndex: 5928',
             ['public SavedGameData saveData; // 0x30']),
            ('ResetTutorialButton', 'public class ResetTutorialButton : MonoBehaviour // TypeDefIndex: 5726', []),
        ]:
            match = re.search(r'^' + re.escape(declaration) + r'\s*\{(.*?)// Methods', dump, re.M | re.S)
            assert match and all(field in match[1] for field in required)
            if name == 'ResetTutorialButton':
                assert not re.search(r'; // 0x', match[1])
            self.button_fields[name] = required
        entry = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == self.button_entry)
        assert entry.struct.EndAddress == 0x3A6659 and not entry.unwindinfo.Flags & 4
        rows = list(self.cs.disasm(self.pe.get_data(self.button_entry, 0x59), self.button_entry))
        assert sum(i.size for i in rows) == 0x59
        instructions = {i.address: i for i in rows}
        checks = {0x3A6614: ('call', '0x2b7b40'), 0x3A6627: ('mov', 'rcx, qword ptr [rax + 0xb8]'),
                  0x3A662E: ('mov', 'rax, qword ptr [rcx]'), 0x3A6636: ('mov', 'rax, qword ptr [rax + 0x20]'),
                  0x3A663F: ('mov', 'rcx, qword ptr [rax + 0x30]'), 0x3A6648: ('xor', 'edx, edx'),
                  0x3A664E: ('jmp', '0x3eaa10'), 0x3A6653: ('call', '0x2b7d90'),
                  0x3A6658: ('int3', '')}
        for address, expected in checks.items():
            assert (instructions[address].mnemonic, instructions[address].op_str) == expected
        for address, operand, expected in [(0x3A6604, 0, self.button_flag),
                                           (0x3A660D, 1, self.project_slot),
                                           (0x3A6619, 0, self.button_flag),
                                           (0x3A6620, 1, self.project_slot)]:
            i = instructions[address]
            assert i.operands[operand].mem.base == self.capstone.x86.X86_REG_RIP
            assert i.address + i.size + i.operands[operand].mem.disp == expected
        self.button_assertions = len(checks) + 4

    def snapshot(self):
        return {**super().snapshot(),
                'button_metadata_initialized': bool(self.u.mem_read(self.base + self.button_flag, 1)[0]),
                'button_chain': {'instance_nonnull': bool(self.rq(self.project_static)),
                                 'game_data_nonnull': bool(self.rq(self.project + 0x20)),
                                 'save_data_nonnull': bool(self.rq(self.game_data + 0x30))}}

    def invoke(self, rva, receiver, argument=0):
        if rva == 0x3EAA10 and self.options.get('button_caller'):
            null = self.options.get('null_chain')
            self.q(self.base + self.project_slot, self.project_type)
            self.q(self.project_type + 0xB8, self.project_static)
            self.q(self.project_static, 0 if null == 'instance' else self.project)
            self.u.mem_write(self.project, bytes([0xA5]) * 0x40)
            self.u.mem_write(self.game_data, bytes([0x5A]) * 0x90)
            self.q(self.project + 0x20, 0 if null == 'game_data' else self.game_data)
            self.q(self.game_data + 0x30, 0 if null == 'save_data' else self.data)
            self.u.mem_write(self.base + self.button_flag, bytes([int(self.options.get('button_warm', False))]))
            self.button_storage = [(self.project_static, bytes(self.u.mem_read(self.project_static, 8))),
                                   (self.project, bytes(self.u.mem_read(self.project, 0x40))),
                                   (self.game_data, bytes(self.u.mem_read(self.game_data, 0x90)))]
            return super().invoke(self.button_entry, 0 if self.options.get('null_button') else self.button)
        return super().invoke(rva, receiver, argument)

    def hook(self, uc, address, size, data):
        if address - self.base == 0x2B7B40 and self.reg(self.x.UC_X86_REG_RCX) == self.base + self.project_slot:
            self.executed.add(address - self.base)
            if self.event('button_metadata_initialize_service', ['ProjectContext_TypeInfo']):
                self.ret(self.project_type)
            return
        return super().hook(uc, address, size, data)

    def run_button(self, state, options=None, storage=None):
        report = self.run_data('ResetTutorials', state, {'button_caller': True, **(options or {})}, storage)
        assert all(bytes(self.u.mem_read(address, len(raw))) == raw for address, raw in self.button_storage)
        report['project_chain_storage_retained'] = True
        return report


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, baselines, failures = [], [], []
    for tutorials, warm, null_button, version in itertools.product([[], ['t', None]], [False, True],
                                                                   [False, True], [17, 0xFFFFFFFF]):
        state = {'key': 'Tutorials', 'completedTutorials': tutorials, 'unlockedCharactersId': ['c']}
        result = m.run_button(state, {'button_warm': warm, 'cache_warm': warm,
                                     'null_button': null_button, 'version': version}, storage={})
        assert result['returned'] and result['project_chain_storage_retained']
        assert values(result['final']['save']) == dict(state, completedTutorials=[])
        assert result['final']['save']['completedTutorials']['version'] == (version + 1) & 0xFFFFFFFF
        assert result['final']['save']['unlockedCharactersId']['version'] == version
        assert len(result['joined_engine']) == 1 and len(result['final']['storage_calls']) == 1
        loaded = m.run_data('Load', state)
        assert loaded['returned'] and values(loaded['final']['save']) == dict(state, completedTutorials=[])
        assert all(c['final']['runtime_string_calls'][0]['native_trace']['returned']
                   for c in loaded['final']['storage_calls'])
        cases.append({'label': 'button_reset_round_trip', 'button': result, 'read': loaded})
    old = {'key': 'Tutorials', 'completedTutorials': ['t'], 'unlockedCharactersId': ['c']}
    for warm, chain in itertools.product([False, True], ['instance', 'game_data', 'save_data']):
        result = m.run_button(old, {'button_warm': warm, 'null_chain': chain}, storage={})
        assert not result['returned'] and result['error'] == 'null_reference'
        assert values(result['final']['save']) == old
        assert not result['joined_engine'] and not result['final']['storage_calls']
        assert result['final']['button_metadata_initialized']
        cases.append({'label': 'null_project_chain', 'result': result})
    for state in [None, dict(old, completedTutorials=None)]:
        result = m.run_button(state, storage={})
        assert not result['returned'] and result['error'] == 'null_reference'
        assert not result['joined_engine'] and not result['final']['storage_calls']
        cases.append({'label': 'null_callee_state', 'result': result})
    for options in [{'registry_status': 5}, {'create_status': 5}]:
        result = m.run_button(old, {'storage_options': options}, storage={})
        assert not result['returned'] and result['error'] == 'preference_exception'
        assert values(result['final']['save']) == dict(old, completedTutorials=[])
        assert result['final']['save']['completedTutorials']['version'] == 18
        assert result['final']['storage'] == {}
        cases.append({'label': 'persistence_failure_after_clear', 'result': result})
    for warm in [False, True]:
        baseline = m.run_button(old, {'button_warm': warm, 'cache_warm': warm}, storage={})
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run_button(old, {'button_warm': warm, 'cache_warm': warm,
                                       'failure': [kind, counts[kind]]}, storage={})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1,
                             'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return compact_trace({'build_id': BUILD, 'target': m.button_target,
                          'coverage_method': 'tdi5726.m0000', 'prior_classification': 'unclassified/not-reviewed',
                          'field_bindings': m.button_fields, 'native_range': ['0x3a6600', '0x3a6659'],
                          'instruction_assertions': m.button_assertions,
                          'metadata_slot': hex(m.project_slot), 'metadata_flag': hex(m.button_flag),
                          'cases': cases, 'case_count': len(cases), 'failure_baselines': baselines,
                          'failure_cases': failures, 'failure_case_count': len(failures),
                          'executed_address_count': len(m.executed),
                          'scope': 'Actual ResetTutorialButton.ResetTutorials follows authored ProjectContext.Instance/gameData/saveData references and tailcalls native SavedGameData.ResetTutorials. Native List mutation, JSON writing, public preference wrapper, provider/key formatter/setter execute; value-level reload also executes native getter/runtime string construction/JSON reader. Project metadata, singleton/reference objects, runtime/allocator/collection/JSON gateway and Windows API services remain explicit. No Unity Button event routing, TutorialsController UI reset, cross-emulator object identity, native exception unwinding or actual OS registry/live process access is claimed.'})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
