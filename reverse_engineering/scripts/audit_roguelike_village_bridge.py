"""Execute RoguelikeStandard stage completion through native village advancement.

The stage receiver and GameData.GameMode are separate explicit objects. Native
IncreaseVillage and concrete RoguelikeStandard.MaxLevel execute without stubs.
"""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path
from audit_character_assets import BUILD
from audit_roguelike_standard_progression import FIELDS, i32, project


def audit(game_root, dumper_root):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    assert unicorn.__version__ == '2.1.4'
    repo = Path(__file__).parents[1]
    build = json.loads((repo / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((repo / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(path, digest):
        raw = path.read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == digest.upper()
        return raw
    raw = pinned(game_root / 'GameAssembly.dll', build['inputs']['game_assembly']['sha256'])
    script = json.loads(pinned(dumper_root / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump = pinned(dumper_root / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    for declaration in ('public static GameMode GameMode; // 0x10',
                        'public static int CurrentVillage; // 0x18',
                        'public AscensionsList[] roguelikeStandardAscensions; // 0x50',
                        'public AscensionsData[] standardAscensions; // 0x60',
                        'public AscensionsData[] ascensions; // 0x10'):
        assert declaration in dump
    methods = {}
    for name, address in [('RoguelikeStandard$$OnStageCompleted', 0x3ea320), ('GameData$$IncreaseVillage', 0x3dc820),
                          ('RoguelikeStandard$$MaxLevel', 0x3ea1c0), ('SavesGame$$set_RoguelikeStandard', 0x387eb0)]:
        rows = [row for row in script['ScriptMethod'] if row['Name'] == name]
        assert len(rows) == 1 and rows[0]['Address'] == address
        methods[name] = rows[0]
    expected_signatures = [
        'void RoguelikeStandard__OnStageCompleted (RoguelikeStandard_o* __this, const MethodInfo* method);',
        'void GameData__IncreaseVillage (const MethodInfo* method);',
        'int32_t RoguelikeStandard__MaxLevel (RoguelikeStandard_o* __this, const MethodInfo* method);',
        'void SavesGame__set_RoguelikeStandard (RoguelikeStandard_o* value, const MethodInfo* method);']
    assert [row['Signature'] for row in methods.values()] == expected_signatures
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    # Null/bounds/cast throw gateways have following trap padding. Separate
    # verified branch targets are decoded through their complete final calls.
    ranges = [(0x3ea320, 0x3ea3a9), (0x387eb0, 0x387ef6), (0x3dc820, 0x3dca4b),
              (0x3dca4c, 0x3dca57), (0x3ea1c0, 0x3ea245), (0x3ea246, 0x3ea24b)]
    decoded = {}
    for a, b in ranges:
        items = list(cs.disasm(pe.get_data(a, b - a), a))
        assert items[0].address == a and sum(i.size for i in items) == b - a
        assert items[-1].address + items[-1].size == b and all(i.mnemonic != 'int3' for i in items)
        decoded.update({i.address: i for i in items})
    checks = {0x3ea395: ('call', '0x3dc820'), 0x3dc890: ('mov', 'rcx, qword ptr [rax + 0x10]'),
              0x3dc8c1: ('cmp', 'qword ptr [rax + rcx*8 - 8], rdx'),
              0x3dc925: ('cmp', 'dword ptr [rcx + 0x18], eax'), 0x3dc94a: ('inc', 'dword ptr [rax + 0x18]'),
              0x3dc974: ('mov', 'rcx, qword ptr [rax + 0x10]'),
              0x3dc9a6: ('cmp', 'qword ptr [rax + rcx*8 - 8], r9'),
              0x3dc9d8: ('mov', 'r8, qword ptr [rax + 0x10]'),
              0x3dc9dc: ('mov', 'ebx, dword ptr [rax + 0x18]'), 0x3dca0a: ('call', '0x3ea1c0'),
              0x3dca11: ('jge', '0x3dca3c'), 0x3dca39: ('inc', 'dword ptr [rax + 0x18]'),
              0x3ea20d: ('mov', 'ecx, dword ptr [rbx + 0x14]'), 0x3ea213: ('jl', '0x3ea21d'),
              0x3ea218: ('dec', 'ecx'), 0x3ea21d: ('jae', '0x3ea246'),
              0x3ea22c: ('mov', 'rax, qword ptr [rax + 0x10]'), 0x3ea238: ('dec', 'eax'),
              0x3dc9f5: ('jb', '0x3dca4c'), 0x3dca03: ('jne', '0x3dca4c')}
    for address, expected in checks.items():
        assert (decoded[address].mnemonic, decoded[address].op_str) == expected
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    for address in (arena, stack, stop): uc.mem_map(address, 0x10000)
    def q(a, value): uc.mem_write(a, struct.pack('<Q', value))
    def d(a, value): uc.mem_write(a, struct.pack('<I', value & 0xffffffff))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def ri(a): return struct.unpack('<i', uc.mem_read(a, 4))[0]
    type_names = ['GameData_TypeInfo', 'ProjectContext_TypeInfo', 'StandardMode_TypeInfo', 'RoguelikeStandard_TypeInfo']
    types = {name: arena + 0x1000 + index * 0x400 for index, name in enumerate(type_names)}
    for name, pointer in types.items():
        rows = [row for row in script['ScriptMetadata'] if row['Name'] == name]
        assert len(rows) == 1
        q(base + rows[0]['Address'], pointer)
    caller, current_mode, game_static, project_static, project_obj, game = [arena + off for off in (0, 0x200, 0x3000, 0x3100, 0x3200, 0x3400)]
    standard_array, outer = arena + 0x4000, arena + 0x5000
    key, json_text = arena + 0x7000, arena + 0x7100
    q(types['GameData_TypeInfo'] + 0xb8, game_static)
    q(types['ProjectContext_TypeInfo'] + 0xb8, project_static)
    literal = [row for row in script['ScriptString'] if row['Address'] == 0x2711c08]
    assert len(literal) == 1 and literal[0]['Value'] == 'SavedRoguelikeStandard'
    q(base + literal[0]['Address'], key)
    classes = {'standard': types['StandardMode_TypeInfo'], 'rogue': types['RoguelikeStandard_TypeInfo'],
               'standard_subclass': arena + 0x8000, 'rogue_subclass': arena + 0x8400,
               'other': arena + 0x8800, 'short_other': arena + 0x8c00}
    for index, (name, klass) in enumerate(classes.items()):
        hierarchy = arena + 0x9000 + index * 0x100
        ancestor = classes['standard'] if name.startswith('standard') else classes['rogue'] if name.startswith('rogue') else klass
        depth = 1 if name == 'short_other' else 3 if name.endswith('subclass') else 2
        uc.mem_write(klass + 0x130, bytes([depth])); q(klass + 0xc8, hierarchy)
        q(hierarchy, arena + 0xa000); q(hierarchy + 8, ancestor); q(hierarchy + 16, klass)
    for ins in decoded.values():
        if ins.mnemonic == 'cmp' and ins.operands[0].type == capstone.CS_OP_MEM and ins.operands[0].size == 1 and ins.operands[0].mem.base == capstone.x86.X86_REG_RIP:
            uc.mem_write(base + ins.address + ins.size + ins.operands[0].mem.disp, b'\1')
    visited, state, options = set(), {}, {}
    def fields():
        return {name: int(uc.mem_read(caller + off, 1)[0]) if name == 'showedNewCharacters' else ri(caller + off) for name, off in FIELDS.items()}
    def ret(value=0):
        sp = uc.reg_read(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value); uc.reg_write(x.UC_X86_REG_RSP, sp + 8); uc.reg_write(x.UC_X86_REG_RIP, rq(sp))
    def event(name):
        state['events'].append(name)
        if options['failure'] == name:
            state['error'] = name; uc.emu_stop(); return False
        return True
    def hook(_, address, size, __):
        rva = address - base
        rcx, rdx, r8 = [uc.reg_read(reg) for reg in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        if rva == 0x281d90:
            assert rcx == types['GameData_TypeInfo']
            if event('class_init'): d(rcx + 0xe0, 1); ret()
        elif rva == 0x1cd6420:
            assert rcx == caller and rdx == 0
            state['save_snapshot'] = {'caller_fields': fields(), 'global_village': ri(game_static + 0x18)}
            if event('json'): ret(json_text)
        elif rva == 0x1c86170:
            assert (rcx, rdx, r8) == (key, json_text, 0)
            if event('set_string'): ret()
        elif rva in (0x2b7d90, 0x2b7d80, 0x2b7040):
            state['error'] = {0x2b7d90: 'null', 0x2b7d80: 'bounds', 0x2b7040: 'cast'}[rva]
            state['events'].append(state['error']); uc.emu_stop()
        else:
            assert rva in decoded and decoded[rva].size == size, hex(rva)
            visited.add(rva)
            if rva == 0x3dc820:
                assert rcx == 0
                state['events'].append('increase_village')
                state['helper_entry_snapshot'] = {'caller_fields': fields(), 'global_village': ri(game_static + 0x18)}
            if rva == 0x3ea1c0:
                assert rcx == rq(game_static + 0x10) and rdx == 0
                state['events'].append('max_level')
                state['max_level_receiver'] = 'caller' if rcx == caller else 'distinct_global_mode'
                state['max_level_ascension'] = ri(rcx + 0x14)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    defaults = dict(zip(FIELDS, (7, 6, 2, 4, 3, 1000, 80, 500, 200, 1)))
    cases = []
    def run(kind, global_village=0, global_ascension=0, lengths=(1, 3, 7), standard_count=3,
            changes=None, missing=None, cold=False, failure=None):
        initial = defaults | (changes or {})
        options.clear(); options.update(failure=failure)
        state.clear(); state.update(events=[], error=None)
        uc.mem_write(caller, bytes([0xa5]) * 128); q(caller, classes['rogue'])
        for name, value in initial.items():
            if name == 'showedNewCharacters': uc.mem_write(caller + FIELDS[name], bytes([value]))
            else: d(caller + FIELDS[name], value)
        uc.mem_write(current_mode, bytes([0x5a]) * 128)
        q(current_mode, classes.get(kind, classes['other'])); d(current_mode + 0x14, global_ascension)
        global_mode = 0 if kind == 'null' else caller if kind == 'rogue_same' else current_mode
        q(game_static + 0x10, global_mode); d(game_static + 0x18, global_village)
        d(types['GameData_TypeInfo'] + 0xe0, 0 if cold else 1)
        q(project_static, 0 if missing == 'project' else project_obj)
        q(project_obj + 0x20, 0 if missing == 'game' else game)
        q(game + 0x60, 0 if missing == 'standard_array' else standard_array)
        d(standard_array + 0x18, standard_count)
        q(game + 0x50, 0 if missing == 'rogue_outer' else outer)
        d(outer + 0x18, len(lengths))
        for index, length in enumerate(lengths):
            entry, inner = arena + 0x6000 + index * 0x100, arena + 0x6400 + index * 0x100
            q(outer + 0x20 + index * 8, 0 if missing == 'rogue_entry' else entry)
            q(entry + 0x10, 0 if missing == 'rogue_inner' else inner)
            d(inner + 0x18, length)
        before = bytes(uc.mem_read(caller, 128)); distinct_before = bytes(uc.mem_read(current_mode, 128))
        sp = stack + 0x8008; q(sp, stop)
        uc.reg_write(x.UC_X86_REG_RSP, sp); uc.reg_write(x.UC_X86_REG_RCX, caller); uc.reg_write(x.UC_X86_REG_RDX, 0)
        for index, reg in enumerate(preserved): uc.reg_write(reg, 0x12340000 + index)
        uc.emu_start(base + 0x3ea320, stop, count=3000)
        expected_fields = project('OnStageCompleted', initial, False, ('increase_village', 1))[0]
        expected_bytes = bytearray(before)
        for name, value in expected_fields.items():
            if name == 'showedNewCharacters': expected_bytes[FIELDS[name]] = value
            else: struct.pack_into('<I', expected_bytes, FIELDS[name], value & 0xffffffff)
        assert bytes(uc.mem_read(caller, 128)) == bytes(expected_bytes)
        assert bytes(uc.mem_read(current_mode, 128)) == distinct_before
        assert rq(game_static + 0x10) == global_mode
        expected_events = ['class_init'] if cold else []
        expected_error = 'class_init' if cold and failure == 'class_init' else None
        expected_global = global_village
        selected = None
        if expected_error is None:
            expected_events.append('increase_village')
            assert state['helper_entry_snapshot'] == {'caller_fields': expected_fields, 'global_village': global_village}
            if kind.startswith('standard'):
                if missing in ('project', 'game', 'standard_array'): expected_error = 'null'
                elif global_village < i32(standard_count - 1): expected_global = i32(global_village + 1)
            elif kind.startswith('rogue'):
                expected_events.append('max_level')
                index = initial['currentAscension'] if kind == 'rogue_same' else global_ascension
                assert state['max_level_ascension'] == index
                assert state['max_level_receiver'] == ('caller' if kind == 'rogue_same' else 'distinct_global_mode')
                if missing in ('project', 'game', 'rogue_outer'): expected_error = 'null'
                else:
                    selected = min(index, len(lengths) - 1)
                    if selected < 0: expected_error = 'bounds'
                    elif missing in ('rogue_entry', 'rogue_inner'): expected_error = 'null'
                    elif global_village < i32(lengths[selected] - 1): expected_global = i32(global_village + 1)
            if expected_error is not None: expected_events.append(expected_error)
            else:
                expected_events.append('json')
                assert state['save_snapshot'] == {'caller_fields': expected_fields, 'global_village': expected_global}
                if failure == 'json': expected_error = 'json'
                else:
                    expected_events.append('set_string')
                    if failure == 'set_string': expected_error = 'set_string'
        assert state['events'] == expected_events and state['error'] == expected_error, (kind, missing, failure, state, expected_events, expected_error)
        assert ri(game_static + 0x18) == expected_global
        if expected_error is None:
            assert uc.reg_read(x.UC_X86_REG_RIP) == stop and uc.reg_read(x.UC_X86_REG_RSP) == sp + 8
            assert all(uc.reg_read(reg) == 0x12340000 + index for index, reg in enumerate(preserved))
        cases.append({'mode_kind': kind, 'caller_initial': initial, 'initial_global_village': global_village,
                      'global_mode_ascension': global_ascension, 'profile_lengths': list(lengths), 'standard_count': standard_count,
                      'missing': missing, 'cold_class': cold, 'gateway_failure': failure, 'selected_outer_index': selected,
                      'caller_result': expected_fields, 'result_global_village': expected_global,
                      'events': expected_events, 'error': expected_error,
                      'helper_entry_snapshot': state.get('helper_entry_snapshot'), 'save_snapshot': state.get('save_snapshot')})
    kinds = ['standard', 'standard_subclass', 'rogue', 'rogue_subclass', 'rogue_same', 'other', 'short_other', 'null']
    for kind, village, ascension, lengths in itertools.product(kinds, (-2, 0, 2, 6, 2**31 - 1), (-1, 0, 1, 10), ((), (0,), (1, 3, 7))):
        run(kind, village, ascension, lengths)
    for kind, missing in itertools.product(kinds, ('project', 'game', 'standard_array', 'rogue_outer', 'rogue_entry', 'rogue_inner')):
        run(kind, missing=missing)
    for kind, cold, failure in itertools.product(kinds, (False, True), (None, 'class_init', 'json', 'set_string')):
        run(kind, cold=cold, failure=failure)
    for field, value, kind in itertools.product(('currentVillage', 'currentAscension', 'roundScore', 'ascensionScore'), (-2**31, 2**31 - 1), ('standard', 'rogue_same')):
        run(kind, changes={field: value})
    for kind, count, village in itertools.product(('standard', 'rogue'), (0, 1, 3, 2**31 - 1, -2**31), (-2**31, -1, 0, 2**31 - 2, 2**31 - 1)):
        run(kind, village, lengths=(count,), standard_count=count)
    run('rogue', changes={'bestScore': 0})
    run('rogue', changes={'bestScore': 0}, cold=True, failure='class_init')
    return {'schema_version': 1, 'build_id': BUILD, 'metadata_declarations': methods, 'native_case_count': len(cases),
            'native_ranges': [[hex(a), hex(b)] for a, b in ranges], 'instruction_assertions': len(checks),
            'visited_instruction_count': len(visited), 'unvisited_instruction_rvas': [hex(v) for v in decoded if v not in visited],
            'cases': cases,
            'scope': 'Native RoguelikeStandard.OnStageCompleted -> GameData.IncreaseVillage -> concrete RoguelikeStandard.MaxLevel and save setter. Runtime assignability uses explicit synthetic hierarchy data. Stage receiver and global current mode are independently supplied; caller and distinct mode bytes checked. Metadata warmed; class initialization preserves supplied mode/global fields. JSON and SetString remain gateways. No Rust contract changes, live saves, whole class initialization, concurrent mode replacement or arbitrary subclass overrides.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} native stage/village bridge cases")
