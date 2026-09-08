"""Audit ascension copy and unlock callers without loading the game DLL.

Unicorn executes pinned callers; collections, JSON services and allocation are
explicit gateways. JSON serialization internals are not emulated or inferred.
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
    if unicorn.__version__ != '2.1.4': raise ValueError('requires Unicorn 2.1.4')
    repo = Path(__file__).parents[1]
    manifest = json.loads((repo / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((repo / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(path, digest):
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest().upper() != digest.upper(): raise ValueError(f'fingerprint changed: {path.name}')
        return raw
    raw = pinned(Path(game_root) / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    script = json.loads(pinned(Path(dumper_root) / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    targets = json.loads((repo / 'targets/ascension_helpers.json').read_text(encoding='utf-8'))['functions']
    for f in targets:
        if not any(d['Name'] == f['metadata_name'] and d['Address'] == int(f['rva'], 16) and d['Signature'] == f['signature'] for d in script['ScriptMethod']):
            raise ValueError('target signature mismatch')
    entries = {f['name']: int(f['rva'], 16) for f in targets}
    pe = pefile.PE(data=raw, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    instructions = {}
    # These spans begin at verified entry points. The private helper is an
    # actual call target of CopyArrayIntoList and has the same 66-byte shape
    # as the registered CreateCopy body, with separately resolved rel32 calls.
    spans = [(0x602EE0, 0x603030), (0x6033B0, 0x6033F2), (0x3B590, 0x3B5D2),
             (0x399010, 0x3992E0), (0x3992E0, 0x399500)]
    for start, end in spans:
        instructions.update({i.address: i for i in cs.disasm(pe.get_data(start, end - start), start)})
    json_instructions = {}
    for f in targets:
        if not f['name'].startswith('JsonUtility.'): continue
        start = int(f['rva'], 16)
        end = min(d['Address'] for d in script['ScriptMethod'] if d['Address'] > start)
        json_instructions.update({i.address: i for i in cs.disasm(pe.get_data(start, end - start), start)})
    checks = [(0x602FE6, 'call', '0x3b590'), (0x6033D4, 'call', '0x1cd6420'),
              (0x6033ED, 'jmp', '0x645da0'), (0x3B5B4, 'call', '0x1cd6420'),
              (0x3B5CD, 'jmp', '0x645da0'), (0x399100, 'call', '0x3b1be0')]
    for address, mnemonic, operands in checks:
        if address not in instructions or (instructions[address].mnemonic, instructions[address].op_str) != (mnemonic, operands):
            raise ValueError('native helper relationship changed')
    json_checks = [(0x645DF8, 'call', '0x113fca0'), (0x645E06, 'call', '0x1cd61d0'),
                   (0x645E47, 'call', '0x2b7010'), (0x645E57, 'call', '0x2b7040'),
                   (0x1CD6212, 'call', '0xf76390'), (0x1CD623C, 'call', '0x4a0210'),
                   (0x1CD6257, 'call', '0x1141800'), (0x1CD62A1, 'call', 'r9'),
                   (0x1CD62CA, 'mov', 'r8, rdi'), (0x1CD62E1, 'jmp', 'rax'),
                   (0x1CD63F3, 'call', '0x2b7df0'), (0x1CD640F, 'jmp', 'rax'),
                   (0x1CD5FEB, 'call', '0x2b7df0'), (0x1CD600F, 'jmp', 'rax')]
    for address, mnemonic, operands in json_checks:
        if address not in json_instructions or (json_instructions[address].mnemonic, json_instructions[address].op_str) != (mnemonic, operands):
            raise ValueError('JSON wrapper relationship changed')
    strings = {row['Address']: row['Value'] for row in script['ScriptString']}
    for address in (0x1CD6529, 0x1CD6697):
        i = json_instructions[address]
        if strings.get(i.address + i.size + i.operands[1].mem.disp) != '': raise ValueError('null ToJson result is not pinned empty string')
    base = pe.OPTIONAL_HEADER.ImageBase
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    uc.mem_map(arena, 0x100000); uc.mem_map(stack, 0x10000); uc.mem_map(stop, 0x1000)
    def q(a, value): uc.mem_write(a, struct.pack('<Q', value))
    def d(a, value): uc.mem_write(a, struct.pack('<I', value))
    def readq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def readd(a): return struct.unpack('<I', uc.mem_read(a, 4))[0]
    def reg(r): return uc.reg_read(r)
    project_type, character_type, debug_type, ordinary_type, list_type = [arena + n for n in (0x100, 0x300, 0x500, 0x700, 0x900)]
    project_static, context, game, all_profile, temporary, saved, save_info = [arena + n for n in (0xB00, 0xC00, 0xD00, 0xE00, 0x1000, 0x1100, 0x1200)]
    catalog, starting, baseline, saved_ids = [arena + n for n in (0x2000, 0x2100, 0x2200, 0x2300)]
    type_values = {'ProjectContext_TypeInfo': project_type, 'CharacterData_TypeInfo': character_type,
                   'UnityEngine.Debug_TypeInfo': debug_type, 'System.Collections.Generic.List<CharacterData>_TypeInfo': list_type}
    if not set(type_values) <= {row['Name'] for row in script['ScriptMetadata']}:
        raise ValueError('required initialized metadata slot not found')
    for row in script['ScriptMetadata']:
        if row['Name'] in type_values: q(base + row['Address'], type_values[row['Name']])
    q(project_type + 0xB8, project_static); q(project_static, context); q(context + 0x20, game)
    q(game + 0x38, catalog); q(game + 0x68, all_profile); q(all_profile + 0x28, baseline)
    q(game + 0x78, temporary); q(game + 0x30, saved); q(saved + 0x18, save_info); q(save_info + 0x20, saved_ids)
    d(debug_type + 0xE0, 1)
    uc.mem_write(character_type + 0x130, b'\x01'); uc.mem_write(ordinary_type + 0x130, b'\x00')
    uc.mem_write(list_type + 0x135, b'\x01')
    hierarchy = arena + 0x3000; q(character_type + 0xC8, hierarchy); q(hierarchy, character_type)
    array_method, array_rgctx, copy_method, copy_rgctx, from_json_method = [arena + n for n in (0x4000, 0x4100, 0x4200, 0x4300, 0x4400)]
    q(array_method + 0x38, array_rgctx); q(array_rgctx, list_type); q(array_rgctx + 8, 1)
    q(array_rgctx + 0x20, copy_method); q(array_rgctx + 0x28, 2)
    q(copy_method + 0x38, copy_rgctx); q(copy_rgctx, from_json_method)
    objects = {i: arena + 0x10000 + i * 0x200 for i in range(1, 5)}
    for i, pointer in objects.items():
        q(pointer, character_type if i == 4 else ordinary_type)
        q(pointer + 0x18, 1000 + i); q(pointer + 0x28, 2000 + i)
    for i in instructions.values():
        if i.mnemonic == 'cmp' and i.operands[0].type == capstone.x86.X86_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
            uc.mem_write(base + i.address + i.size + i.operands[0].mem.disp, b'\x01')
    state = {}; collections = {}
    def ret(value=0):
        rsp = reg(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value); uc.reg_write(x.UC_X86_REG_RSP, rsp + 8); uc.reg_write(x.UC_X86_REG_RIP, readq(rsp))
    def hook(_, address, size, __):
        rva = address - base
        rcx, rdx, r8 = reg(x.UC_X86_REG_RCX), reg(x.UC_X86_REG_RDX), reg(x.UC_X86_REG_R8)
        if rva == 0x2B7D40:
            state['allocations'] += 1
            pointer = arena + 0x50000 + state['allocations'] * 0x100
            collections[pointer] = []; state['output_list'] = pointer; ret(pointer)
        elif rva in (0xB02160, 0x33ED50): ret()
        elif rva == 0x3B1BE0:
            if rcx != temporary: raise ValueError('wrong temporary starting source')
            state['starting_requests'] += 1; ret(starting)
        elif rva == 0xB16640:
            if rdx not in collections: raise ValueError('unknown enumerated list')
            q(rcx, rdx); d(rcx + 8, 0); d(rcx + 12, 0); q(rcx + 16, 0); ret(rcx)
        elif rva == 0x9693D0:
            values = collections[readq(rcx)]; index = readd(rcx + 8)
            if index < len(values):
                q(rcx + 16, values[index]); d(rcx + 8, index + 1); ret(1)
            else: q(rcx + 16, 0); ret()
        elif rva == 0xB55950:
            state['contains'].append({'list': 'saved_ids' if rcx == saved_ids else 'baseline', 'value': rdx})
            ret(int(rdx in collections[rcx]))
        elif rva == 0x2EB0:
            collections[rcx].append(rdx); ret()
        elif rva == 0x1C4B450:
            state['logs'].append(rcx); ret()
        elif rva == 0x1CD6420:
            if rdx != 0: raise ValueError('copy should request compact JSON overload')
            state['json_sources'].append(rcx); ret(0 if rcx == 0 else rcx + 0x1000000)
        elif rva == 0x645DA0:
            if rdx != from_json_method: raise ValueError('wrong generic FromJson context')
            state['json_results'].append(rcx)
            if rcx == 0: ret()
            else:
                state['clones'] += 1; ret(arena + 0x60000 + state['clones'] * 0x100)
        elif rva in (0x2B7D90, 0x2B7D80):
            state['error'] = 'null' if rva == 0x2B7D90 else 'index'; uc.emu_stop()
        elif rva not in instructions:
            raise ValueError(f'execution left audited caller/gateways: {rva:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    def run(name, this=0, argument=0):
        state.clear(); state.update(allocations=0, starting_requests=0, contains=[], logs=[], json_sources=[], json_results=[], clones=0)
        rsp = stack + 0x8008; q(rsp, stop)
        for register, value in [(x.UC_X86_REG_RSP, rsp), (x.UC_X86_REG_RCX, this), (x.UC_X86_REG_RDX, argument), (x.UC_X86_REG_R8, 0), (x.UC_X86_REG_R9, 0),
                                (x.UC_X86_REG_RBX, 0x11223344), (x.UC_X86_REG_RDI, 0x22334455), (x.UC_X86_REG_RSI, 0x33445566), (x.UC_X86_REG_RBP, 0x44556677)]:
            uc.reg_write(register, value)
        try:
            uc.emu_start(base + entries[name], stop, timeout=1_000_000, count=3000)
        except unicorn.UcError as error:
            raise ValueError(f'{name} native fixture fault at RVA {reg(x.UC_X86_REG_RIP) - base:x}; state={state}') from error
        if 'error' not in state:
            if reg(x.UC_X86_REG_RIP) != stop or reg(x.UC_X86_REG_RSP) != rsp + 8: raise ValueError('incomplete native return')
            for register, value in [(x.UC_X86_REG_RBX, 0x11223344), (x.UC_X86_REG_RDI, 0x22334455), (x.UC_X86_REG_RSI, 0x33445566), (x.UC_X86_REG_RBP, 0x44556677)]:
                if reg(register) != value: raise ValueError('nonvolatile register changed')
        return reg(x.UC_X86_REG_RAX)
    membership_cases = []
    for base_mask, saved_mask in itertools.product(range(8), repeat=2):
        collections[catalog] = [objects[i] for i in (1, 2, 1, 3)]
        collections[starting] = [objects[i] for i in (3, 1, 2, 3)]
        base_ids = [i for i in (1, 2, 3) if base_mask & (1 << (i - 1))]
        saved_values = [i for i in (1, 2, 3) if saved_mask & (1 << (i - 1))]
        collections[baseline] = [objects[i] for i in base_ids]; collections[saved_ids] = [1000 + i for i in saved_values]
        for locked in (False, True):
            name = 'Compendium.GetAllLockedCharacters' if locked else 'Compendium.GetAllUnlockedCharacters'
            returned = run(name)
            expected_ids = [i for i in (3, 1, 2, 3) if i not in base_ids and i not in saved_values] if locked else [i for i in (1, 2, 1, 3) if i in base_ids]
            if state.get('error') or collections[returned] != [objects[i] for i in expected_ids]: raise ValueError('unlock source/membership mismatch')
            if state['starting_requests'] != int(locked): raise ValueError('wrong candidate universe')
            if not locked and any(c['list'] == 'saved_ids' for c in state['contains']): raise ValueError('unlocked getter unexpectedly reads saved IDs')
            membership_cases.append({'getter': 'locked' if locked else 'unlocked', 'baseline_assets': base_ids, 'saved_asset_ids': saved_values, 'result_assets': expected_ids})
    null_cases = []
    for baseline_contains_null, locked in itertools.product((False, True), repeat=2):
        collections[catalog] = [objects[1], 0]; collections[starting] = [objects[1], 0]
        collections[baseline] = [0] if baseline_contains_null else []
        collections[saved_ids] = []
        returned = run('Compendium.GetAllLockedCharacters' if locked else 'Compendium.GetAllUnlockedCharacters')
        expected_error = 'null' if locked and not baseline_contains_null else None
        if state.get('error') != expected_error: raise ValueError('null asset short-circuit mismatch')
        values = collections[state['output_list']]
        expected = [objects[1]] if locked else ([0] if baseline_contains_null else [])
        if values != expected: raise ValueError('partial locked-list output mismatch')
        null_cases.append({'getter': 'locked' if locked else 'unlocked', 'baseline_contains_null': baseline_contains_null,
                           'error': expected_error, 'partial_or_final_result': [None if v == 0 else 1 for v in values]})
    copy_cases = []
    native_array = arena + 0x70000
    for values in [None, [], [1], [1, 1], [1, None, 2], [4]]:
        actual = [] if values is None else [0 if value is None else objects[value] for value in values]
        q(native_array + 0x18, len(actual))
        for index, value in enumerate(actual): q(native_array + 0x20 + index * 8, value)
        returned = run('ClassConv.CopyArrayIntoList_Object', 0 if values is None else native_array, array_method)
        if state.get('error') != ('null' if values is None else None) or state['json_sources'] != actual or len(state['json_results']) != len(actual):
            raise ValueError('copy-array JSON chronology mismatch')
        output = collections[state['output_list']]
        if len(output) != len(actual) or [v == 0 for v in output] != [v == 0 for v in actual]: raise ValueError('copy-array returned values not preserved')
        if state['logs'] != ([2004] if values == [4] else []): raise ValueError('CharacterData debug log branch mismatch')
        copy_cases.append({'input_assets': values, 'json_round_trips': len(actual), 'null_result_indices': [i for i, v in enumerate(output) if v == 0],
                           'character_data_log_count': len(state['logs']), 'error': state.get('error'), 'allocated_list_count': state['allocations']})
    for value in [None, 1]:
        returned = run('ClassConv.CreateCopy_Object', 0 if value is None else objects[value], copy_method)
        if state.get('error') or len(state['json_sources']) != 1 or len(state['json_results']) != 1 or (returned == 0) != (value is None):
            raise ValueError('direct JSON copy caller mismatch')
        copy_cases.append({'direct_copy': value, 'json_round_trips': 1, 'returns_null': returned == 0})
    return {'schema_version': 1, 'build_id': BUILD, 'game_assembly_sha256': manifest['inputs']['game_assembly']['sha256'],
            'target_count': len(targets), 'native_relationships_verified': len(checks) + len(json_checks) + 2,
            'membership_cases': membership_cases, 'null_asset_cases': null_cases, 'copy_cases': copy_cases,
            'native_case_count': len(membership_cases) + len(null_cases) + len(copy_cases),
            'candidate_universes': {'unlocked': 'GameData.allCharacterData filtered by allCharactersAscension.unlockedCharacters only',
                                   'locked': 'temporary stored starting arrays excluding baseline assets and saved unlockedCharactersId'},
            'json_wrapper_scope': 'Static native checks cover null/empty behavior, runtime type resolution, abstract/subclass guards, generic cast and cached internal-call forwarding; UnityPlayer serialization bodies remain open.',
            'scope': 'Pinned native callers and nested private CreateCopy body. Collections, allocator, Debug.Log, starting-array concatenation and JSON services are gateways; Unity JSON engine internals, generic cast and string-equality internals are not executed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} native ascension helper cases")
