"""Execute native PlayerPrefs wrappers inside SavedGameData persistence calls."""
import argparse
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_generic_json import Machine as GenericMachine
from audit_saved_game_info_json import Machine as JsonMachine


class Machine(GenericMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        self.get_service, self.set_service = self.stop + 0x100, self.stop + 0x110
        self.exception_type, self.exception, self.set_method = [self.arena + n for n in (0x140000, 0x141000, 0x142000)]
        self.preference_ranges = {}
        instructions = {}
        for a, b in [(0x1C85F20, 0x1C85F82), (0x1C86170, 0x1C861FF)]:
            entry = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == a)
            assert entry.struct.EndAddress == b and not entry.unwindinfo.Flags & 4
            decoded = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
            assert sum(i.size for i in decoded) == b - a
            instructions.update({i.address: i for i in decoded})
            self.preference_ranges[hex(a)] = [hex(a), hex(b)]
        checks = {0x1C85F3D: ('call', '0x2b7b40'), 0x1C85F63: ('call', '0x2b7df0'),
                  0x1C85F6F: ('mov', 'rdx, rbx'), 0x1C85F7F: ('jmp', 'rax'),
                  0x1C86193: ('call', '0x2b7df0'), 0x1C861A5: ('call', 'rax'),
                  0x1C861A7: ('test', 'al, al'), 0x1C861B5: ('ret', ''),
                  0x1C861BD: ('call', '0x2b7b40'), 0x1C861C5: ('call', '0x2b7d40'),
                  0x1C861D4: ('call', '0x2b7b40'), 0x1C861E2: ('call', '0x1c85d80'),
                  0x1C861EE: ('call', '0x2b7b40'), 0x1C861F9: ('call', '0x2b7d50')}
        for a, expected in checks.items():
            assert a in instructions and (instructions[a].mnemonic, instructions[a].op_str) == expected
        def rip(address, operand=1):
            i = instructions[address]
            o = i.operands[operand]
            assert o.type == self.capstone.x86.X86_OP_MEM and o.mem.base == self.capstone.x86.X86_REG_RIP
            return i.address + i.size + o.mem.disp
        self.get_cache, self.set_cache, self.get_flag = rip(0x1C85F49), rip(0x1C8617A), rip(0x1C85F2A, 0)
        self.requests = {rip(0x1C85F5C): 'UnityEngine.PlayerPrefs::GetString(System.String,System.String)',
                         rip(0x1C8618C): 'UnityEngine.PlayerPrefs::TrySetSetString(System.String,System.String)'}
        for rva, expected in self.requests.items():
            raw = self.pe.get_data(rva, 160)
            assert len(raw) == 160 and raw.split(b'\0', 1)[0].decode('ascii') == expected
        self.pref_bindings = {rip(0x1C85F36): ('string', ''),
                              rip(0x1C861B6): ('metadata', 'UnityEngine.PlayerPrefsException_TypeInfo'),
                              rip(0x1C861CA): ('string', 'Could not store preference value'),
                              rip(0x1C861E7): ('method', 'Method$UnityEngine.PlayerPrefs.SetString()')}
        script = json.loads((Path(dumper_root) / 'script.json').read_text(encoding='utf-8-sig'))
        slots = {r['Address']: ('metadata', r['Name']) for r in script['ScriptMetadata']}
        slots.update({r['Address']: ('method', r['Name']) for r in script['ScriptMetadataMethod']})
        slots.update({r['Address']: ('string', r['Value']) for r in script['ScriptString']})
        assert all(slots[a] == value for a, value in self.pref_bindings.items())
        rows = [r for r in script['ScriptMethod'] if r['Address'] == 0x1C85D80 and r['Name'] == 'UnityEngine.PlayerPrefsException$$.ctor']
        assert len(rows) == 1 and rows[0]['Signature'] == 'void UnityEngine_PlayerPrefsException___ctor (UnityEngine_PlayerPrefsException_o* __this, System_String_o* error, const MethodInfo* method);'
        self.exception_target = rows[0]
        self.preference_instruction_assertions = len(checks) + 9

    def invoke(self, rva, receiver, argument=0):
        self.bindings.update(self.pref_bindings)
        self.empty_literal = self.string('')
        self.error_literal = self.string('Could not store preference value')
        tokens = {'': self.empty_literal, 'Could not store preference value': self.error_literal,
                  'UnityEngine.PlayerPrefsException_TypeInfo': self.exception_type,
                  'Method$UnityEngine.PlayerPrefs.SetString()': self.set_method}
        for slot, (_, name) in self.pref_bindings.items():
            self.q(self.base + slot, tokens[name])
        warm = self.options.get('cache_warm', False)
        self.q(self.base + self.get_cache, self.get_service if warm else 0)
        self.q(self.base + self.set_cache, self.set_service if warm else 0)
        self.u.mem_write(self.base + self.get_flag, bytes([int(self.options.get('literal_warm', False))]))
        self.exception_state = {'allocated': False, 'message': None}
        return super().invoke(rva, receiver, argument)

    def snapshot(self):
        state = super().snapshot()
        state.update({'get_cache_published': bool(self.rq(self.base + self.get_cache)),
                      'set_cache_published': bool(self.rq(self.base + self.set_cache)),
                      'get_literal_initialized': bool(self.u.mem_read(self.base + self.get_flag, 1)[0]),
                      'preference_exception': self.exception_state.copy()})
        return state

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        cx, dx, r8 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        if rva in [0x1C85F20, 0x1C86170]:
            self.executed.add(rva)
            return  # Run the native wrapper and its indirect-call cache logic.
        if rva == 0x2B7DF0:
            assert cx - self.base in self.requests
            request = self.requests[cx - self.base]
            self.executed.add(rva)
            if self.event('internal_call_resolve_service', [request]):
                self.ret(self.get_service if '::GetString(' in request else self.set_service)
        elif address == self.get_service:
            assert dx == self.empty_literal and (not cx or cx in self.strings)
            index = self.counts.get('get_string_service', 0)
            values = self.options.get('get_values', ['', ''])
            result = values[min(index, len(values) - 1)]
            if self.event('get_string_service', [self.strings.get(cx), self.strings[dx], result]):
                if self.options.get('get_swap_at') == index + 1:
                    value = self.options.get('replacement')
                    self.q(self.data + 0x18, self.new_info(value) if value is not None else 0)
                self.ret(self.string(result))
        elif address == self.set_service:
            assert (not cx or cx in self.strings) and (not dx or dx in self.strings)
            args = [self.strings.get(cx), self.strings.get(dx)]
            success = self.options.get('set_success', True)
            if self.event('try_set_string_service', args + [success]):
                if success:
                    self.writes.append(args)
                # Native wrapper must consume AL rather than the full RAX.
                self.ret(0xDEADBEEF00000000 | int(success))
        elif rva == 0x2B7B40 and cx - self.base in self.pref_bindings:
            self.executed.add(rva)
            if self.event('preference_metadata_initialize_service', [self.pref_bindings[cx - self.base][1]]):
                self.ret(self.rq(cx))
        elif rva == 0x2B7D40 and cx == self.exception_type:
            self.executed.add(rva)
            if self.event('preference_exception_allocate_service', []):
                self.exception_state['allocated'] = True
                self.ret(self.exception)
        elif rva == 0x1C85D80:
            assert cx == self.exception and dx == self.error_literal and r8 == 0
            assert self.exception_state['allocated']
            self.executed.add(rva)
            if self.event('preference_exception_constructor_service', [self.strings[dx]]):
                self.exception_state['message'] = self.strings[dx]
                self.ret()
        elif rva == 0x2B7D50:
            assert cx == self.exception and dx == self.set_method
            self.executed.add(rva)
            self.event('preference_throw_gateway', [self.exception_state['message']])
            self.error = 'preference_exception'
            self.u.emu_stop()
        else:
            super().hook(uc, address, size, data)

    def run_data(self, method, state=None, options=None):
        for slot in self.pref_bindings:
            self.bindings.pop(slot, None)
        return super().run_data(method, state, options)


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    old = {'key': 'offline-key', 'completedTutorials': ['t'], 'unlockedCharactersId': ['c']}
    cases, failures, baselines, joined = [], [], [], []
    for cache, literal, key, first in itertools.product([False, True], [False, True], [None, '', 'offline-key'], [None, '', 'first']):
        state = dict(old, key=key)
        case = m.run_data('Load', state, {'cache_warm': cache, 'literal_warm': literal, 'get_values': [first, 'second']})
        assert case['returned']
        gets = [e for e in case['events'] if e['kind'] == 'get_string_service']
        assert len(gets) == (2 if first else 1) and all(e['args'][0:2] == [key, ''] for e in gets)
        resolutions = [e for e in case['events'] if e['kind'] == 'internal_call_resolve_service']
        assert len(resolutions) == int(not cache)
        assert case['final']['save']['key'] == ('loaded-key' if first else 'Tutorials')
        cases.append(case)
    for method, cache, success, key in itertools.product(['Save', 'ResetTutorials'], [False, True], [False, True], [None, 'offline-key']):
        state = dict(old, key=key)
        case = m.run_data(method, state, {'cache_warm': cache, 'set_success': success})
        assert case['returned'] == success
        if success:
            assert case['final']['preference_writes'] == [[key, 'authored-json-token']]
        else:
            assert case['error'] == 'preference_exception'
            assert case['final']['preference_exception'] == {'allocated': True, 'message': 'Could not store preference value'}
            assert case['final']['preference_writes'] == []
        if method == 'ResetTutorials':
            assert case['final']['save']['completedTutorials']['values'] == []
        cases.append(case)
    for method, options in [('Load', {'get_values': ['first', 'second']}), ('Save', {}), ('ResetTutorials', {'set_success': False})]:
        baseline = m.run_data(method, old, options)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            case = m.run_data(method, old, dict(options, failure=[kind, counts[kind]]))
            assert not case['returned'] and case['events'] == baseline['events'][:index + 1]
            assert case['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1, 'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    m.engine = JsonMachine(game_root)
    for method, cache in itertools.product(['Load', 'Save', 'ResetTutorials'], [False, True]):
        options = {'cache_warm': cache, 'engine_join': True}
        if method == 'Load':
            options['get_values'] = ['first', '{"key":"joined","completedTutorials":["t"],"unlockedCharactersId":["c"]}']
        case = m.run_data(method, old, options)
        assert case['returned'] and len(case['joined_engine']) == 1
        if method == 'Load':
            assert case['final']['save']['key'] == 'joined'
        else:
            assert case['final']['preference_writes'][0][0] == 'offline-key'
        joined.append(case)
    return {'build_id': BUILD, 'native_ranges': m.preference_ranges,
            'instruction_assertions': m.preference_instruction_assertions,
            'internal_call_requests': list(m.requests.values()), 'exception_target': m.exception_target,
            'cases': cases, 'case_count': len(cases), 'failure_baselines': baselines,
            'failure_cases': failures, 'failure_case_count': len(failures), 'joined_cases': joined,
            'joined_case_count': len(joined), 'executed_address_count': len(m.executed),
            'scope': 'Actual PlayerPrefs GetString/SetString wrappers execute inside native SavedGameData callers; Load also executes generic FromJson. Internal-call resolution, preference storage, exception allocation/construction/throw policy and remaining runtime services are explicit. Six value-level engine JSON joins run offline. Exact requests are pinned without claiming registration-name equality or fallback behavior. No live preferences or native exception unwinding is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(result['case_count'], result['failure_case_count'], result['joined_case_count'], result['executed_address_count'])
