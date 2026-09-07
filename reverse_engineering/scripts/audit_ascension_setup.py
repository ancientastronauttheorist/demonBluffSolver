"""Execute pinned ascension selectors with explicit runtime service gateways.

No host DLL loading. Native branches, field reads/writes and index checks run
in Unicorn; RNG, GC barriers, allocator, collection and ClassConv services are
authored gateways. This does not recover a Unity random state or game mode UI.
"""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path
from audit_character_assets import BUILD


def audit(game_root, dumper_root):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    if unicorn.__version__ != '2.1.4':
        raise ValueError('requires Unicorn 2.1.4')
    repo = Path(__file__).parents[1]
    manifest = json.loads((repo / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((repo / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))

    def pinned(path, digest):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest().upper() != digest.upper():
            raise ValueError(f'fingerprint changed: {path.name}')
        return data

    raw = pinned(Path(game_root) / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    script = json.loads(pinned(Path(dumper_root) / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    target = json.loads((repo / 'targets/ascension_setup.json').read_text(encoding='utf-8'))
    methods = {d['Name']: d for d in script['ScriptMethod']}
    for f in target['functions']:
        if methods[f['metadata_name']]['Address'] != int(f['rva'], 16) or methods[f['metadata_name']]['Signature'] != f['signature']:
            raise ValueError('target metadata mismatch')
    pe = pefile.PE(data=raw, fast_load=False)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    names = ['AdvancedMode.GetCurrentAscension', 'AdvancedMode.GetPreviousAscension',
             'StandardMode.GetCurrentAscension', 'StandardMode.GetPreviousAscension',
             'RoguelikeMode.GetCurrentAscension', 'RoguelikeMode.GetPreviousAscension',
             'RoguelikeStandard.GetCurrentAscension', 'RoguelikeStandard.GetPreviousAscension',
             'AscensionsData.SetupCharactersCount', 'AscensionsData.GetStartingtCharactersOfType',
             'AscensionsData.SetupStartingCharacters', 'AscensionsData.ClearCurrentPickedScript',
             'AscensionsData.GetCharactersCount', 'GameData.GetCharactersOfType',
             'AscensionsData.CopyData', 'GameData.SetupCurrentAscension']
    entries = {f['name']: int(f['rva'], 16) for f in target['functions'] if f['name'] in names}
    instructions = {}
    for name, start in entries.items():
        unwind = next((e.struct for e in pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress <= start < e.struct.EndAddress), None)
        # Small leaf methods have no unwind record. End at the next managed
        # entry, always decoding from this verified entry and capping at 64.
        next_entry = min(d['Address'] for d in script['ScriptMethod'] if d['Address'] > start)
        end = unwind.EndAddress if unwind else min(start + 64, next_entry)
        if name == 'AscensionsData.CopyData':
            # This verified body has adjacent unwind chunks: decoding stops at
            # the next managed method, not the first chunk's epilogue boundary.
            end = next_entry
        instructions.update({i.address: i for i in cs.disasm(pe.get_data(start, end - start), start)})
    base = pe.OPTIONAL_HEADER.ImageBase
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    uc.mem_map(arena, 0x100000)
    uc.mem_map(stack, 0x10000)
    uc.mem_map(stop, 0x1000)
    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def d(a, v): uc.mem_write(a, struct.pack('<I', v & 0xFFFFFFFF))
    def readq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def reg(r): return uc.reg_read(r)
    project_type, game_type = arena + 0x100, arena + 0x300
    project_static, game_static, context, game, actor, profile = [arena + n for n in (0x500, 0x600, 0x700, 0x800, 0x1000, 0x1200)]
    for row in script['ScriptMetadata']:
        if row['Name'] in ('ProjectContext_TypeInfo', 'GameData_TypeInfo'):
            q(base + row['Address'], project_type if row['Name'] == 'ProjectContext_TypeInfo' else game_type)
    method_slots = {row['Address']: row['Name'] for row in script['ScriptMetadataMethod']}
    for i in instructions.values():
        if i.mnemonic == 'mov' and len(i.operands) == 2 and i.operands[1].type == capstone.x86.X86_OP_MEM and i.operands[1].mem.base == capstone.x86.X86_REG_RIP:
            slot = i.address + i.size + i.operands[1].mem.disp
            if slot in method_slots: q(base + slot, slot)
    q(project_type + 0xB8, project_static)
    q(game_type + 0xB8, game_static)
    d(project_type + 0xE0, 1)
    d(game_type + 0xE0, 1)
    q(project_static, context)
    q(context + 0x20, game)
    q(game + 0x78, profile)
    for i in instructions.values():
        if i.mnemonic == 'cmp' and i.operands[0].type == capstone.x86.X86_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
            uc.mem_write(base + i.address + i.size + i.operands[0].mem.disp, b'\x01')
    state = {}
    def ret(value=0):
        rsp = reg(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, rsp + 8)
        uc.reg_write(x.UC_X86_REG_RIP, readq(rsp))
    def hook(_, address, size, __):
        rva = address - base
        if rva == 0x2B6FF0:
            state['barriers'].append(reg(x.UC_X86_REG_RCX)); ret()
        elif rva == 0x1C86600:
            lower, upper = reg(x.UC_X86_REG_RCX), reg(x.UC_X86_REG_RDX)
            index = len(state['draws'])
            if lower != 0 or index >= len(state['choices']) or not 0 <= state['choices'][index] < upper:
                raise ValueError('invalid explicit RNG gateway request')
            state['draws'].append({'width': upper, 'index': state['choices'][index]})
            ret(state['choices'][index])
        elif rva == 0xB01F50:
            source = reg(x.UC_X86_REG_RCX)
            state['to_arrays'].append(source)
            ret(source + 0x1000000)
        elif rva in (0x602EE0, 0x6033B0):
            source, method = reg(x.UC_X86_REG_RCX), reg(x.UC_X86_REG_RDX)
            state['copies'].append({'helper': 'array_elements' if rva == 0x602EE0 else 'object', 'source': source, 'method': method_slots[method]})
            ret(source + 0x2000000)
        elif rva == 0x2B7D40:
            state['allocations'] += 1
            ret(arena + 0x30000 + state['allocations'] * 0x100)
        elif rva == 0xB610A0:
            state['list_constructors'].append({'destination': reg(x.UC_X86_REG_RCX), 'source': reg(x.UC_X86_REG_RDX), 'method': method_slots[reg(x.UC_X86_REG_R8)]})
            ret()
        elif rva in (0x2B7D80, 0x2B7D90):
            state['error'] = 'index' if rva == 0x2B7D80 else 'null'
            uc.emu_stop()
        elif rva not in instructions:
            raise ValueError(f'execution left selected native instructions/gateways: {rva:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    def run(name, this=actor, argument=0, choices=()):
        state.clear()
        state.update(choices=list(choices), draws=[], barriers=[], to_arrays=[], copies=[], allocations=0, list_constructors=[])
        rsp = stack + 0x8008
        q(rsp, stop)
        for r, value in [(x.UC_X86_REG_RSP, rsp), (x.UC_X86_REG_RCX, this), (x.UC_X86_REG_RDX, argument & 0xFFFFFFFFFFFFFFFF),
                         (x.UC_X86_REG_R8, 0), (x.UC_X86_REG_R9, 0), (x.UC_X86_REG_RBX, 0x11223344),
                         (x.UC_X86_REG_RSI, 0x22334455), (x.UC_X86_REG_RDI, 0x33445566), (x.UC_X86_REG_RBP, 0x44556677)]:
            uc.reg_write(r, value)
        uc.emu_start(base + entries[name], stop, timeout=1_000_000, count=500)
        if 'error' not in state:
            if reg(x.UC_X86_REG_RIP) != stop or reg(x.UC_X86_REG_RSP) != rsp + 8:
                raise ValueError('incomplete native return')
            for r, value in [(x.UC_X86_REG_RBX, 0x11223344), (x.UC_X86_REG_RSI, 0x22334455),
                             (x.UC_X86_REG_RDI, 0x33445566), (x.UC_X86_REG_RBP, 0x44556677)]:
                if reg(r) != value: raise ValueError('nonvolatile register changed')
        return {'error': state.get('error'), 'returned': reg(x.UC_X86_REG_RAX),
                'draws': list(state['draws']), 'to_arrays': list(state['to_arrays']), 'barrier_count': len(state['barriers'])}
    def array(address, values):
        q(address + 0x18, len(values))
        for index, value in enumerate(values): q(address + 0x20 + index * 8, value)
        return address

    mode_cases = []
    one = arena + 0x2000
    group_array = arena + 0x3000
    for prefix, offset in [('StandardMode', 0x60), ('RoguelikeMode', 0x58)]:
        for values in [[], [21658], [21658, 21662, 21658]]:
            q(game + offset, array(one, values))
            for index in [-2147483648, -1, 0, 1, 2, 3, 7, 100, 2147483647]:
                d(game_static + 0x18, index)
                for previous in (False, True):
                    selected = index - 1 if previous else min(index, len(values) - 1)
                    expected = values[selected] if 0 <= selected < len(values) else None
                    name = prefix + ('.GetPreviousAscension' if previous else '.GetCurrentAscension')
                    result = run(name)
                    if result['error'] != ('index' if expected is None else None) or (expected is not None and result['returned'] != expected):
                        raise ValueError(f'mode selection mismatch: {name} {index} {values}')
                    mode_cases.append({'method': name, 'profiles': values, 'index': index, 'expected_profile': expected, 'error': result['error']})
    groups = [[21674, 21675], [21684], [21688, 21689, 21690], [21695, 21696]]
    records = []
    for index, values in enumerate(groups):
        record = arena + 0x4000 + index * 0x200
        q(record + 0x10, array(record + 0x80, values)); records.append(record)
    q(game + 0x50, array(group_array, records))
    for ascension, village, previous in itertools.product([-1, 0, 1, 3, 4, 84, 2147483647], [-1, 0, 1, 2, 7, 2147483647], (False, True)):
        d(actor + 0x14, ascension); d(actor + 0x20, village)
        values = groups[min(ascension, len(groups) - 1)] if ascension >= 0 else []
        selected = village - 1 if previous else min(village, len(values) - 1)
        expected = values[selected] if 0 <= selected < len(values) else None
        name = 'RoguelikeStandard' + ('.GetPreviousAscension' if previous else '.GetCurrentAscension')
        result = run(name)
        if result['error'] != ('index' if expected is None else None) or (expected is not None and result['returned'] != expected):
            raise ValueError('two-level mode selection mismatch')
        mode_cases.append({'method': name, 'ascension': ascension, 'village': village, 'expected_profile': expected, 'error': result['error']})
    q(game + 0x48, 21657)
    for name in ['AdvancedMode.GetCurrentAscension', 'AdvancedMode.GetPreviousAscension']:
        result = run(name)
        if result['error'] or result['returned'] != 21657: raise ValueError('advanced profile mismatch')
        mode_cases.append({'method': name, 'expected_profile': 21657, 'error': None})

    scripts = [arena + 0x10000 + i * 0x200 for i in range(4)]
    customs = [arena + 0x12000 + i * 0x100 for i in range(2)]
    for index, custom in enumerate(customs): q(custom + 0x18, scripts[index + 2])
    for index, script_ptr in enumerate(scripts):
        for field in (0x10, 0x18, 0x20, 0x28, 0x30, 0x38): q(script_ptr + field, arena + 0x20000 + index * 0x100 + field)
    plain = {10: 0x40, 20: 0x48, 30: 0x50, 100: 0x58}
    picked = {10: 0x10, 20: 0x18, 30: 0x20, 100: 0x28}
    for kind, offset in plain.items(): q(profile + offset, 90000 + kind)
    selection_cases = []
    for inline_count, custom_count, cached in itertools.product(range(3), range(3), (False, True)):
        inline = scripts[:inline_count]; custom = customs[:custom_count]
        q(profile + 0x20, array(arena + 0x14000, inline)); q(profile + 0x18, array(arena + 0x15000, custom))
        widths = [] if cached else ([inline_count] if inline_count else []) + ([custom_count] if custom_count else [])
        for choices in itertools.product(*(range(width) for width in widths)):
            expected_script = scripts[0] if cached else 0
            if not cached:
                if inline_count: expected_script = inline[choices[0]]
                if custom_count: expected_script = readq(custom[choices[-1]] + 0x18)
            for kind in (10, 20, 30, 100, 0, 99):
                q(profile + 0x60, scripts[0] if cached else 0)
                result = run('AscensionsData.GetStartingtCharactersOfType', profile, kind, choices)
                expected_return = (readq(expected_script + picked[kind]) + 0x1000000 if expected_script else 90000 + kind) if kind in plain else 0
                if result['error'] or result['returned'] != expected_return or readq(profile + 0x60) != expected_script or [r['width'] for r in result['draws']] != widths:
                    raise ValueError('native cached script selection mismatch')
                selection_cases.append({'inline_count': inline_count, 'custom_count': custom_count, 'cached': cached, 'choices': choices,
                                        'type': kind, 'selected_script_index': scripts.index(expected_script) if expected_script else None,
                                        'rng_draws': result['draws'], 'uses_script': bool(expected_script), 'returns_null': kind not in plain})
            q(profile + 0x60, scripts[0] if cached else 0)
            result = run('AscensionsData.SetupCharactersCount', profile, choices=choices)
            if result['error'] or readq(profile + 0x60) != expected_script or [r['width'] for r in result['draws']] != widths:
                raise ValueError('count setup script selection mismatch')
    failure_cases = []
    for missing in ('inline_array', 'custom_array', 'custom_record'):
        q(profile + 0x60, 0)
        q(profile + 0x20, 0 if missing == 'inline_array' else array(arena + 0x14000, [scripts[0]]))
        q(profile + 0x18, 0 if missing == 'custom_array' else array(arena + 0x15000, [0 if missing == 'custom_record' else customs[0]]))
        result = run('AscensionsData.SetupCharactersCount', profile, choices=[0, 0])
        expected_cache = 0 if missing == 'inline_array' else scripts[0]
        if result['error'] != 'null' or readq(profile + 0x60) != expected_cache:
            raise ValueError('partial script selection failure mismatch')
        failure_cases.append({'missing': missing, 'error': result['error'], 'inline_selection_retained': bool(expected_cache), 'rng_draws': result['draws']})
    setup_cases = []
    script_fields = [0x30, 0x10, 0x18, 0x20, 0x28]
    profile_fields = [0x30, 0x40, 0x48, 0x50, 0x58]
    for missing_index in [None, 0, 1, 2, 3, 4]:
        q(profile + 0x20, array(arena + 0x14000, [scripts[0]]))
        q(profile + 0x18, array(arena + 0x15000, []))
        q(profile + 0x60, scripts[0])
        for index, field in enumerate(script_fields): q(scripts[0] + field, 0 if index == missing_index else arena + 0x40000 + index * 0x100)
        for field in profile_fields: q(profile + field, 0xAABBCC)
        result = run('AscensionsData.SetupStartingCharacters', profile)
        copied = len(profile_fields) if missing_index is None else missing_index
        expected_fields = [(arena + 0x40000 + index * 0x100 + 0x1000000) if index < copied else 0xAABBCC for index in range(5)]
        if result['error'] != (None if missing_index is None else 'null') or [readq(profile + field) for field in profile_fields] != expected_fields or result['barrier_count'] != copied:
            raise ValueError('partial starting-field materialization mismatch')
        setup_cases.append({'missing_script_field': hex(script_fields[missing_index]) if missing_index is not None else None,
                            'copied_profile_fields': [hex(f) for f in profile_fields[:copied]], 'error': result['error']})
    q(profile + 0x20, array(arena + 0x14000, []))
    q(profile + 0x18, array(arena + 0x15000, []))
    result = run('AscensionsData.SetupStartingCharacters', profile)
    if result['error'] or result['barrier_count'] or result['to_arrays']: raise ValueError('empty script sources should skip materialization')
    setup_cases.append({'empty_script_sources': True, 'copied_profile_fields': [], 'error': None})
    run('AscensionsData.ClearCurrentPickedScript', profile)
    if readq(profile + 0x60) != 0 or state['barriers'] != [profile + 0x60]: raise ValueError('cache clear mismatch')

    copy_cases = []
    source = arena + 0x50000
    fields = list(range(0x18, 0x98, 8))
    shared_fields = {0x18, 0x30, 0x40, 0x48, 0x50, 0x58, 0x68, 0x70, 0x78, 0x80}
    for missing in [None, 0, 0x88, 0x90]:
        for field in fields:
            q(source + field, 0 if field == missing else arena + 0x60000 + field * 0x100)
            q(profile + field, 0xAABBCC)
        result = run('AscensionsData.CopyData', profile, 0 if missing == 0 else source)
        copied = 16 if missing is None else (0 if missing == 0 else (14 if missing == 0x88 else 15))
        if result['error'] != (None if missing is None else 'null') or state['barriers'] != [profile + field for field in fields[:copied]]:
            raise ValueError('copy field ordering or failure boundary changed')
        for field in fields[:copied]:
            original = readq(source + field)
            if field in shared_fields: expected = original
            elif field in (0x20, 0x88, 0x90): expected = original + 0x3000000
            elif field == 0x60: expected = original + 0x2000000
            else: expected = arena + 0x30000 + (1 if field == 0x28 else 2) * 0x100
            if readq(profile + field) != expected: raise ValueError(f'copy field source mismatch: {field:x}')
        if any(readq(profile + field) != 0xAABBCC for field in fields[copied:]): raise ValueError('copy modified fields beyond failure')
        copy_cases.append({'missing_source_field': hex(missing) if missing else ('source' if missing == 0 else None),
                           'copied_fields': [hex(field) for field in fields[:copied]],
                           'shared_fields': [hex(field) for field in fields[:copied] if field in shared_fields],
                           'copy_helper_instantiations': [c['method'] for c in state['copies']],
                           'list_constructor_count': len(state['list_constructors']), 'error': result['error']})
    null_script_cases = []
    for null_inline, null_custom_script in [(True, False), (False, True), (True, True)]:
        q(profile + 0x20, array(arena + 0x14000, [0 if null_inline else scripts[0]]))
        q(profile + 0x18, array(arena + 0x15000, [customs[0]]))
        q(customs[0] + 0x18, 0 if null_custom_script else scripts[2])
        q(profile + 0x60, 0)
        result = run('AscensionsData.SetupCharactersCount', profile, choices=[0, 0])
        expected = 0 if null_custom_script else scripts[2]
        if result['error'] or readq(profile + 0x60) != expected or len(result['draws']) != 2:
            raise ValueError('null ScriptInfo replacement mismatch')
        second = run('AscensionsData.SetupCharactersCount', profile, choices=[0, 0])
        if len(second['draws']) != (2 if expected == 0 else 0): raise ValueError('null script should permit later reselection')
        null_script_cases.append({'null_inline_script': null_inline, 'null_custom_script': null_custom_script,
                                  'first_draw_count': 2, 'next_draw_count': len(second['draws']), 'error': None})
    setup_current_cases = []
    mode_class = arena + 0x90000
    q(actor, mode_class)
    q(mode_class + 0x1E8, base + entries['StandardMode.GetCurrentAscension'])
    q(mode_class + 0x1F0, 0)
    q(game_static + 0x10, actor)
    d(game_static + 0x18, 1)
    normal_source, debug_source = source, source + 0x1000
    q(game + 0x60, array(one, [debug_source, normal_source]))
    q(game + 0x70, debug_source)
    for source_ptr in (normal_source, debug_source):
        for field in fields: q(source_ptr + field, arena + 0x60000 + field * 0x100 + (0x100 if source_ptr == debug_source else 0))
    for debug, missing_temporary in [(False, False), (True, False), (False, True), (True, True)]:
        uc.mem_write(game_static + 0x1D, bytes([debug]))
        q(game + 0x78, 0 if missing_temporary else profile)
        result = run('GameData.SetupCurrentAscension', game)
        expected_source = debug_source if debug else normal_source
        if result['error'] != ('null' if missing_temporary else None): raise ValueError('temporary setup failure mismatch')
        if not missing_temporary and readq(profile + 0x18) != readq(expected_source + 0x18): raise ValueError('temporary setup selected wrong profile')
        setup_current_cases.append({'debug_ascension': debug, 'missing_temporary': missing_temporary,
                                    'source': 'debug' if debug else 'mode_current', 'copied_field_count': result['barrier_count'], 'error': result['error']})
    accessor_cases = []
    for kind, missing_temporary in itertools.product([10, 20, 30, 100, -1, 0, 99], (False, True)):
        q(game + 0x78, 0 if missing_temporary else profile)
        for index, field in enumerate([0x68, 0x70, 0x78, 0x80]): q(profile + field, 91000 + index)
        result = run('GameData.GetCharactersOfType', game, kind)
        valid_type = kind in plain
        expected_error = 'null' if valid_type and missing_temporary else None
        expected_return = 91000 + list(plain).index(kind) if valid_type else 0
        if result['error'] != expected_error or (not expected_error and result['returned'] != expected_return): raise ValueError('temporary pool accessor mismatch')
        accessor_cases.append({'method': 'GameData.GetCharactersOfType', 'type': kind, 'missing_temporary': missing_temporary,
                               'returned': expected_return if not expected_error else None, 'error': expected_error})
    for cached, null_counts in itertools.product((False, True), (False, True)):
        q(profile + 0x60, scripts[0] if cached else 0)
        q(profile + 0x88, 0 if null_counts else 92000)
        q(scripts[0] + 0x38, 0 if null_counts else 93000)
        result = run('AscensionsData.GetCharactersCount', profile)
        expected = 0 if null_counts else (93000 if cached else 92000)
        if result['error'] or result['returned'] != expected or result['draws']: raise ValueError('count accessor unexpectedly selects or dereferences null list')
        accessor_cases.append({'method': 'AscensionsData.GetCharactersCount', 'cached': cached, 'null_counts': null_counts, 'returned': expected, 'error': None})
    return {'schema_version': 1, 'build_id': BUILD, 'game_assembly_sha256': manifest['inputs']['game_assembly']['sha256'],
            'target_count': len(target['functions']), 'mode_case_count': len(mode_cases), 'mode_cases': mode_cases,
            'script_selection_case_count': len(selection_cases), 'script_selection_cases': selection_cases,
            'count_setup_cases': len(selection_cases) // 6, 'partial_failure_cases': failure_cases,
            'starting_materialization_cases': setup_cases, 'copy_caller_cases': copy_cases, 'cache_clear_cases': 1,
            'null_script_cases': null_script_cases, 'temporary_setup_cases': setup_current_cases,
            'accessor_cases': accessor_cases,
            'native_case_count': len(mode_cases) + len(selection_cases) + len(selection_cases) // 6 + len(failure_cases) + len(setup_cases) + len(copy_cases) + 1 + 2 * len(null_script_cases) + len(setup_current_cases) + len(accessor_cases),
            'scope': 'Native selected caller/control flow with initialized metadata/static state; RNG, GC barriers, allocator, list conversion/constructor and ClassConv services are gateways. Deep-copy internals, mode UI/init and broader lifecycle remain separate.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} native ascension cases")
