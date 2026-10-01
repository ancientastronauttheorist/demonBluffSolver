"""Execute Character history leaves and register-as type selection offline."""
import argparse
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_publication_entries import Machine as EntryMachine


TARGETS = {0x3647D0: 'AddOnHoverInfo', 0x3649A0: 'ClearRecentMemory',
           0x364E60: 'GetCurrentActedInfo', 0x364DE0: 'GetCharacterType'}
HELPERS = [0xB22150, 0xB59CE0]
SIGNATURES = {
    'AddOnHoverInfo': 'void Character__AddOnHoverInfo (Character_o* __this, ActedInfo_o* info, const MethodInfo* method);',
    'ClearRecentMemory': 'void Character__ClearRecentMemory (Character_o* __this, const MethodInfo* method);',
    'GetCurrentActedInfo': 'ActedInfo_o* Character__GetCurrentActedInfo (Character_o* __this, const MethodInfo* method);',
    'GetCharacterType': 'int32_t Character__GetCharacterType (Character_o* __this, const MethodInfo* method);'}


class Machine(EntryMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
        self.history_targets, self.history_addresses, self.helper_addresses = [], set(), set()
        for address in [*TARGETS, *HELPERS]:
            if address in TARGETS:
                rows = [r for r in self.metadata['ScriptMethod']
                        if r['Address'] == address and r['Name'] == 'Character$$' + TARGETS[address]]
                assert len(rows) == 1
                assert rows[0]['Signature'] == SIGNATURES[TARGETS[address]]
                self.history_targets.extend(rows)
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4:
                    root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == address:
                    chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks
            self.ranges[hex(address)] = [[hex(a), hex(b)] for a, b in chunks]
            for a, b in chunks:
                instructions = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
                assert sum(i.size for i in instructions) == b - a
                self.instructions.update({i.address: i for i in instructions})
                destination = self.history_addresses if address in TARGETS else self.helper_addresses
                destination.update(i.address for i in instructions)
        references = set()
        for address in self.history_addresses | self.helper_addresses:
            i = self.instructions[address]
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + op.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1:
                assert i.operands[0].mem.base == capstone.x86.X86_REG_RIP
                self.flags.add(i.address + i.size + i.operands[0].mem.disp)
        self.history_bindings = {}
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                if row['Name'] not in self.bindings:
                    self.bindings[row['Name']] = self.arena + 0x4000 + len(self.bindings) * 0x200
                token = self.bindings[row['Name']]
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
                self.history_bindings[row['Name']] = token
        required = ['Method$System.Collections.Generic.List<ActedInfo>.Add()',
                    'Method$System.Collections.Generic.List<ActedInfo>.get_Count()',
                    'Method$System.Collections.Generic.List<ActedInfo>.get_Item()',
                    'Method$System.Collections.Generic.List<ActedInfo>.RemoveAt()', 'UnityEngine.Object_TypeInfo']
        assert all(name in self.history_bindings for name in required)
        for name, index, fields in [('Character', 5487, ['public List<ActedInfo> actedInfos; // 0x148',
                                                       'public List<ActedInfo> onHoverInfo; // 0x150',
                                                       'public CharacterData registerAs; // 0x60',
                                                       'public CharacterData dataRef; // 0x50']),
                                    ('CharacterData', 5845, ['public ECharacterType type; // 0x130'])]:
            declaration = re.search(r'^public class ' + name + r'[^\n]*// TypeDefIndex: ' + str(index) +
                                    r'\s*\{(.*?)// Properties', self.dump, re.M | re.S)
            assert declaration and all(field in declaration[1] for field in fields)
        self.hover, self.hover_array = self.arena + 0xD0000, self.arena + 0xD1000
        self.register_as = self.arena + 0xD2000
        self.static_ids.update({self.hover: 'hover', self.hover_array: 'hover_array', self.register_as: 'register_as'})
        self.checks = {
            0x36480F: ('inc', 'dword ptr [rcx + 0x1c]'),
            0x36484A: ('mov', 'dword ptr [rcx + 0x18], eax'),
            0x36485E: ('mov', 'qword ptr [rcx], rbx'),
            0x3649DD: ('cmp', 'dword ptr [rcx + 0x18], 0'),
            0x3649ED: ('dec', 'edx'),
            0x3649F4: ('jmp', '0xb59ce0'),
            0x364EA7: ('dec', 'edx'),
            0x364EAE: ('jmp', '0xb22150'),
            0x364E10: ('mov', 'rdi, qword ptr [rbx + 0x60]'),
            0x364E33: ('mov', 'rax, qword ptr [rbx + 0x60]'),
            0x364E3C: ('mov', 'eax, dword ptr [rax + 0x130]'),
            0xB59CFB: ('dec', 'dword ptr [rbx + 0x18]'),
            0xB59D42: ('mov', 'qword ptr [rcx], 0'),
            0xB59D49: ('call', '0x2b6ff0'),
            0xB59D4E: ('inc', 'dword ptr [rbx + 0x1c]'),
            0xB22160: ('cmp', 'ebx, dword ptr [rcx + 0x18]')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == expected
                   for a, expected in self.checks.items())
        self.service_rvas.add(0x113B2C0)

    def list_snapshot(self, pointer):
        backing = self.rq(pointer + 0x10)
        return {'identity': self.object_id(pointer), 'backing': self.object_id(backing),
                'count_bits': self.rd(pointer + 0x18), 'version': self.rd(pointer + 0x1C),
                'capacity': self.rd(backing + 0x18) if backing else None,
                'slots': [self.object_id(self.rq(backing + 0x20 + i * 8)) for i in range(8)] if backing else None}

    def snapshot(self):
        f = self.fixtures
        return {'history_ref': self.object_id(self.rq(self.actor + 0x148)),
                'hover_ref': self.object_id(self.rq(self.actor + 0x150)),
                'history': self.list_snapshot(f['history']), 'hover': self.list_snapshot(self.hover),
                'data_ref': self.object_id(self.rq(self.actor + 0x50)),
                'register_as_ref': self.object_id(self.rq(self.actor + 0x60)),
                'data_type_bits': self.rd(f['data'] + 0x130), 'register_type_bits': self.rd(self.register_as + 0x130),
                'object_class_initialized': self.rd(self.bindings['UnityEngine.Object_TypeInfo'] + 0xE0),
                'saved_speech': self.object_id(self.rq(self.actor + 0x198)),
                'metadata_flags': {hex(p): self.u.mem_read(self.base + p, 1)[0] for p in sorted(self.flags)}}

    def prepare(self, options):
        super().prepare(options)
        f = self.fixtures
        for p in [self.hover, self.hover_array, self.register_as]:
            self.u.mem_write(p, bytes(0x800))
        self.q(self.actor + 0x150, 0 if options.get('null_hover') else f['history'] if options.get('alias_lists') else self.hover)
        self.q(self.actor + 0x60, 0 if options.get('register', 'live') == 'absent' else self.register_as)
        self.d(f['data'] + 0x130, options.get('data_type', 10))
        self.d(self.register_as + 0x130, options.get('register_type', 20))
        for pointer, array in [(f['history'], f['history_array']), (self.hover, self.hover_array)]:
            count = options.get('count', 1)
            self.q(pointer + 0x10, 0 if options.get('null_backing') else array)
            self.d(pointer + 0x18, count & 0xFFFFFFFF)
            self.d(pointer + 0x1C, options.get('version', 9))
            self.d(array + 0x18, options.get('capacity', 8))
            for i in range(8):
                self.q(array + 0x20 + i * 8, 0 if options.get('null_last') and i == count - 1 else
                       self.info if i & 1 else f['prior_info'])

    def hook(self, uc, address, size, data):
        rva, x, f = address - self.base, self.x, self.fixtures
        cx, dx, r8 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8]]
        if rva == 0x1C822C0:
            self.executed.add(rva)
            assert cx in [0, self.register_as] and dx == r8 == 0
            if self.event('register_unity_null_service', [self.object_id(cx)]):
                result = int(cx == 0 or self.options.get('register') == 'destroyed')
                if self.options.get('clear_register_during_check'):
                    self.q(self.actor + 0x60, 0)
                self.ret(0xABC000 | result)
        elif rva == 0xB54090:
            self.executed.add(rva)
            assert cx in [self.hover, f['history']] and dx in [0, self.info] and r8 == f['generic_method']
            if self.event('hover_list_growth_service', [self.object_id(cx), self.object_id(dx)]):
                backing = self.rq(cx + 0x10); count = self.rd(cx + 0x18)
                assert backing and count <= 7
                self.d(backing + 0x18, 8)
                self.q(backing + 0x20 + count * 8, dx)
                self.d(cx + 0x18, count + 1)
                self.ret(0xDEADBEEF)
        elif rva == 0x113B2C0:
            self.executed.add(rva)
            assert cx & 0xFFFFFFFF == 0
            self.event('native_index_guard', [])
            self.error = 'native_index_guard'; uc.emu_stop()
        else:
            if rva in [0xB22150, 0xB59CE0]:
                assert cx == f['history']
                assert dx & 0xFFFFFFFF == (self.rd(cx + 0x18) - 1) & 0xFFFFFFFF
                key = 'get_Item' if rva == 0xB22150 else 'RemoveAt'
                assert r8 == self.bindings[f'Method$System.Collections.Generic.List<ActedInfo>.{key}()']
            super().hook(uc, address, size, data)

    def run_history(self, name, options=None, retained=False):
        if not retained:
            self.prepare(options or {})
        else:
            self.options = options or {}; self.error = None
        before = bytes(self.u.mem_read(self.actor, 0x200))
        initial = self.snapshot()
        address = next(a for a, n in TARGETS.items() if n == name)
        argument = self.info if name == 'AddOnHoverInfo' and not self.options.get('null_info') else 0
        returned = self.invoke(address, self.actor, argument)
        after = bytes(self.u.mem_read(self.actor, 0x200))
        if self.options.get('clear_register_during_check'):
            assert before[:0x60] == after[:0x60] and before[0x68:] == after[0x68:]
        else:
            assert before == after
        result = self.reg(self.x.UC_X86_REG_RAX) if returned and name in ['GetCurrentActedInfo', 'GetCharacterType'] else None
        final = self.snapshot()
        expected = {key: dict(initial[key], slots=initial[key]['slots'].copy() if initial[key]['slots'] is not None else None)
                    for key in ['history', 'hover']}
        if returned and name == 'AddOnHoverInfo':
            key = 'history' if initial['hover_ref'] == 'history' else 'hover'
            record = expected[key]; count = record['count_bits']
            assert count < 8
            record['count_bits'] = count + 1
            record['version'] = (record['version'] + 1) & 0xFFFFFFFF
            record['slots'][count] = None if self.options.get('null_info') else 'info'
            if count >= record['capacity']: record['capacity'] = 8
        elif returned and name == 'ClearRecentMemory' and 0 < initial['history']['count_bits'] < 0x80000000:
            record = expected['history']; count = record['count_bits']
            record['count_bits'] = count - 1
            record['version'] = (record['version'] + 1) & 0xFFFFFFFF
            record['slots'][count - 1] = None
        if returned:
            assert all(final[key] == expected[key] for key in expected)
            if name == 'GetCurrentActedInfo':
                assert self.object_id(result) == initial['history']['slots'][initial['history']['count_bits'] - 1]
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'result': self.object_id(result) if result is not None and name == 'GetCurrentActedInfo' else result,
                'initial': initial, 'events': self.events.copy(), 'final': final}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, sequences, baselines, failures = [], [], [], []
    for name, count, cold, version in itertools.product(['AddOnHoverInfo', 'ClearRecentMemory', 'GetCurrentActedInfo'],
                                                      [0, 1, 3], [False, True], [9, 0xFFFFFFFF]):
        result = m.run_history(name, {'count': count, 'cold': cold, 'version': version})
        assert result['returned'] == (name != 'GetCurrentActedInfo' or count > 0)
        cases.append(result)
    for name, null, count in itertools.product(['AddOnHoverInfo', 'ClearRecentMemory', 'GetCurrentActedInfo'],
                                              ['null_history', 'null_hover', 'null_backing'], [0, 2]):
        result = m.run_history(name, {null: True, 'count': count})
        expected = null == ('null_history' if name == 'AddOnHoverInfo' else 'null_hover')
        expected |= name == 'ClearRecentMemory' and null == 'null_backing' and count == 0
        if name == 'GetCurrentActedInfo' and count == 0: expected = False
        assert result['returned'] == expected, (name, null, count, result)
        cases.append(result)
    for name in ['AddOnHoverInfo', 'ClearRecentMemory', 'GetCurrentActedInfo']:
        for options in [{'count': 2, 'alias_lists': True}, {'count': 2, 'null_info': True, 'null_last': True},
                        {'count': 2, 'capacity': 2}, {'count': 0xFFFFFFFF}]:
            if options['count'] == 0xFFFFFFFF and name == 'AddOnHoverInfo': continue
            result = m.run_history(name, options)
            cases.append(result)
    for register, cold, real_type, registered_type in itertools.product(['absent', 'destroyed', 'live'], [False, True],
                                                                        [0, 10, 0xFFFFFFFF, 0x80000000], [20, 0xFFFFFFFF]):
        result = m.run_history('GetCharacterType', {'register': register, 'cold': cold,
                                                  'data_type': real_type, 'register_type': registered_type})
        assert result['returned'] and result['result'] == (registered_type if register == 'live' else real_type)
        cases.append(result)
    for register in ['absent', 'destroyed', 'live']:
        result = m.run_history('GetCharacterType', {'register': register, 'null_data': True})
        assert result['returned'] == (register == 'live')
        cases.append(result)
    mutation = m.run_history('GetCharacterType', {'register': 'live', 'clear_register_during_check': True})
    assert not mutation['returned'] and mutation['error'] == 'native_null_guard'
    for alias in [False, True]:
        m.prepare({'count': 1, 'alias_lists': alias})
        calls = [m.run_history(name, {'alias_lists': alias}, retained=True) for name in
                 ['AddOnHoverInfo', 'GetCurrentActedInfo', 'ClearRecentMemory', 'ClearRecentMemory', 'AddOnHoverInfo']]
        assert all(c['returned'] for c in calls)
        sequences.append({'alias_lists': alias, 'calls': calls})
    for name, options in [('AddOnHoverInfo', {'cold': True}), ('AddOnHoverInfo', {'cold': True, 'capacity': 1}),
                          ('ClearRecentMemory', {'cold': True, 'count': 2}),
                          ('GetCurrentActedInfo', {'cold': True, 'count': 2}),
                          ('GetCharacterType', {'cold': True, 'register': 'live'})]:
        baseline = m.run_history(name, options); assert baseline['returned']
        ordinal = len(baselines); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run_history(name, dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index + 1]
            assert stopped['final'] == e['snapshot']
            failures.append({'baseline': ordinal, 'failure': [kind, counts[kind]], 'prefix_length': index + 1,
                             'exact_snapshot_verified': True})
    return {'build': BUILD, 'targets': m.history_targets, 'instruction_assertions': len(m.checks),
            'metadata_bindings': sorted(m.history_bindings), 'case_count': len(cases), 'cases': cases,
            'retained_sequences': sequences, 'mutation_cases': [mutation], 'failure_baselines': baselines,
            'failure_case_count': len(failures), 'failure_cases': failures,
            'caller_instructions_executed': len(m.executed & m.history_addresses),
            'caller_instructions_decoded': len(m.history_addresses),
            'helper_instructions_executed': len(m.executed & m.helper_addresses),
            'native_execution_addresses': len(m.executed),
            'scope': 'Four complete Character history/type callers and actual get_Item/last-element RemoveAt helper slices execute. Growth, metadata/runtime, liveness, GC and exception gateways remain supplied. Non-last removal/Array.Copy, real callback/lifetime semantics and managed exception unwinding are not claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'failure_case_count', 'caller_instructions_executed', 'native_execution_addresses']}))
