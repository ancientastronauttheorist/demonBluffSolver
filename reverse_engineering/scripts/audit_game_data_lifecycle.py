"""Pinned GameData lifecycle caller audit with explicit service gateways."""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path
from audit_character_assets import BUILD


def audit(game_root, dumper_root, target_manifest):
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
    targets = json.loads(Path(target_manifest).read_text(encoding='utf-8'))['functions']
    for f in targets:
        if not any(d['Name'] == f['metadata_name'] and d['Address'] == int(f['rva'], 16) and d['Signature'] == f['signature'] for d in script['ScriptMethod']):
            raise ValueError('target metadata mismatch')
    entries = {f['name']: int(f['rva'], 16) for f in targets}
    pe = pefile.PE(data=raw, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    instructions = {}
    for start in entries.values():
        end = min(d['Address'] for d in script['ScriptMethod'] if d['Address'] > start)
        instructions.update({i.address: i for i in cs.disasm(pe.get_data(start, end - start), start)})
    base = pe.OPTIONAL_HEADER.ImageBase
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095); uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    uc.mem_map(arena, 0x100000); uc.mem_map(stack, 0x10000); uc.mem_map(stop, 0x1000)
    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def d(a, v): uc.mem_write(a, struct.pack('<I', v & 0xFFFFFFFF))
    def readq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def readd(a): return struct.unpack('<I', uc.mem_read(a, 4))[0]
    def reg(r): return uc.reg_read(r)
    game_type, events_type, loc_type, standard_type = [arena + n for n in (0x100, 0x300, 0x500, 0x700)]
    game_static, events_static, loc_static, game, saved, info, catalogue = [arena + n for n in (0x1000, 0x1100, 0x1200, 0x2000, 0x2200, 0x2300, 0x2400)]
    project_type, rogue_type = arena + 0x900, arena + 0xB00
    types = {'GameData_TypeInfo': game_type, 'GameEvents_TypeInfo': events_type,
             'CharacterLocProvider_TypeInfo': loc_type, 'StandardMode_TypeInfo': standard_type,
             'ProjectContext_TypeInfo': project_type, 'RoguelikeStandard_TypeInfo': rogue_type}
    if not set(types) <= {r['Name'] for r in script['ScriptMetadata']}: raise ValueError('required metadata slot absent')
    for row in script['ScriptMetadata']:
        if row['Name'] in types: q(base + row['Address'], types[row['Name']])
    for type_ptr, static_ptr in [(game_type, game_static), (events_type, events_static), (loc_type, loc_static)]:
        q(type_ptr + 0xB8, static_ptr); d(type_ptr + 0xE0, 1)
    for i in instructions.values():
        if i.mnemonic == 'cmp' and i.operands[0].type == capstone.x86.X86_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
            uc.mem_write(base + i.address + i.size + i.operands[0].mem.disp, b'\x01')
    new_mode, old_mode, loaded_mode = [arena + n for n in (0x3000, 0x3100, 0x3200)]
    callbacks = {stop + 0x100 + i * 0x10: name for i, name in enumerate(['load', 'deinit', 'init', 'mode_changed', 'game_init', 'state_changed'])}
    callback_address = {name: address for address, name in callbacks.items()}
    for index, mode in enumerate([new_mode, old_mode, loaded_mode]):
        klass = arena + 0x4000 + index * 0x400; q(mode, klass)
        for offset, name in [(0x198, 'load'), (0x1B8, 'deinit'), (0x188, 'init')]:
            q(klass + offset, callback_address[name]); q(klass + offset + 8, 0)
    delegates = {}
    for index, name in enumerate(['mode_changed', 'game_init', 'state_changed']):
        pointer = arena + 0x5000 + index * 0x100; delegates[name] = pointer
        q(pointer + 0x18, callback_address[name]); q(pointer + 0x40, 100 + index); q(pointer + 0x28, 0)
    objects = {i: arena + 0x10000 + i * 0x100 for i in range(1, 4)}
    for i, pointer in objects.items():
        q(pointer + 0x28, 1000 + (i % 2)); q(pointer + 0x18, 1500 + (i % 2))
    collections = {}; options = {}; state = {}
    def ret(value=0):
        rsp = reg(x.UC_X86_REG_RSP); uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, rsp + 8); uc.reg_write(x.UC_X86_REG_RIP, readq(rsp))
    def hook(_, address, size, __):
        rva = address - base
        rcx, rdx = reg(x.UC_X86_REG_RCX), reg(x.UC_X86_REG_RDX)
        if address in callbacks:
            name = callbacks[address]
            state['events'].append(name); state['callback_modes'].append(readq(game_static + 0x10))
            if options.get('fail_callback') == name:
                state['error'] = 'callback:' + name; uc.emu_stop(); return
            ret(options.get('loaded_mode', loaded_mode) if name == 'load' else 0)
        elif rva == 0x2B6FF0:
            state['barriers'].append(rcx); ret()
        elif rva == 0xB16640:
            q(rcx, rdx); d(rcx + 8, 0); d(rcx + 12, 0); q(rcx + 16, 0); ret(rcx)
        elif rva == 0x9693D0:
            values = collections[readq(rcx)]; index = readd(rcx + 8)
            if index < len(values): q(rcx + 16, values[index]); d(rcx + 8, index + 1); ret(1)
            else: ret()
        elif rva == 0xF73E00: ret(int(rcx == rdx))
        elif rva == 0x1C85F20:
            state['preference_reads'].append(rcx); ret(options.get('json', 0))
        elif rva == 0xF76390: ret(int(rcx == 0))
        elif rva == 0x645DA0:
            state['json_reads'] += 1; ret(arena + 0x70000)
        elif rva == 0x2B7D40:
            state['allocations'] += 1; pointer = arena + 0x60000 + state['allocations'] * 0x100
            state['allocation_types'].append(rcx)
            collections[pointer] = []; ret(pointer)
        elif rva in (0x3EADD0, 0xB02160, 0x33ED50): ret()
        elif rva == 0x3B4DB0:
            state['preference_loads'].append(rcx); ret()
        elif rva == 0x3EA1C0:
            state['max_level_reads'] += 1; ret(options['max_level'] & 0xFFFFFFFF)
        elif rva == 0x387BA0:
            state['achievement_state_reads'] += 1; ret(arena + 0x90000)
        elif rva == 0xB55950: ret(int(rdx in collections[rcx]))
        elif rva == 0x2EB0: collections[rcx].append(rdx); ret()
        elif rva == 0x3AFC90: state['achievement_unlocks'].append(rcx); ret()
        elif rva == 0x2B7D90:
            state['error'] = 'null'; uc.emu_stop()
        elif rva not in instructions:
            raise ValueError(f'execution left selected native caller/gateways: {rva:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    def run(name, this=0, argument=0):
        state.clear(); state.update(events=[], callback_modes=[], barriers=[], preference_reads=[], preference_loads=[], json_reads=0, allocations=0, max_level_reads=0,
                                    allocation_types=[], achievement_state_reads=0, achievement_unlocks=[])
        rsp = stack + 0x8008; q(rsp, stop)
        for register, value in [(x.UC_X86_REG_RSP, rsp), (x.UC_X86_REG_RCX, this & 0xFFFFFFFFFFFFFFFF), (x.UC_X86_REG_RDX, argument),
                                (x.UC_X86_REG_R8, 0), (x.UC_X86_REG_R9, 0), (x.UC_X86_REG_RBX, 0x11223344),
                                (x.UC_X86_REG_RDI, 0x22334455), (x.UC_X86_REG_RSI, 0x33445566), (x.UC_X86_REG_RBP, 0x44556677)]: uc.reg_write(register, value)
        uc.emu_start(base + entries[name], stop, timeout=1_000_000, count=2000)
        if 'error' not in state:
            if reg(x.UC_X86_REG_RIP) != stop or reg(x.UC_X86_REG_RSP) != rsp + 8: raise ValueError('native return incomplete')
            for register, value in [(x.UC_X86_REG_RBX, 0x11223344), (x.UC_X86_REG_RDI, 0x22334455), (x.UC_X86_REG_RSI, 0x33445566), (x.UC_X86_REG_RBP, 0x44556677)]:
                if reg(register) != value: raise ValueError('nonvolatile register changed')
        return reg(x.UC_X86_REG_RAX)
    mode_cases = []
    for new_present, old_present, loaded_present, event_mask in itertools.product((False, True), (False, True), (False, True), range(4)):
        old = old_mode if old_present else 0; loaded = loaded_mode if loaded_present else 0
        q(game_static + 0x10, old); options.clear(); options['loaded_mode'] = loaded
        q(events_static + 0x10, delegates['mode_changed'] if event_mask & 1 else 0)
        q(events_static + 0x18, delegates['game_init'] if event_mask & 2 else 0)
        run('GameData.ChangeGameMode', new_mode if new_present else 0)
        events = [] if not new_present else ['load'] + (['deinit'] if old_present else [])
        failed = not new_present or (old_present and not loaded_present)
        if not failed:
            if old_present: events.append('init')
            if event_mask & 1: events.append('mode_changed')
            if event_mask & 2: events.append('game_init')
        if state['events'] != events or state.get('error') != ('null' if failed else None) or readq(game_static + 0x10) != (old if failed else loaded):
            raise ValueError('mode-change ordering mismatch')
        for event, observed in zip(state['events'], state['callback_modes']):
            if observed != (loaded if event in ('mode_changed', 'game_init') else old): raise ValueError('mode publication happened at wrong callback boundary')
        mode_cases.append({'new_present': new_present, 'old_present': old_present, 'loaded_present': loaded_present,
                           'event_mask': event_mask, 'events': events, 'error': state.get('error'), 'publishes_loaded': not failed})
    callback_failures = []
    for failure in ['load', 'deinit', 'init', 'mode_changed', 'game_init']:
        q(game_static + 0x10, old_mode); options.clear(); options.update(loaded_mode=loaded_mode, fail_callback=failure)
        q(events_static + 0x10, delegates['mode_changed']); q(events_static + 0x18, delegates['game_init'])
        run('GameData.ChangeGameMode', new_mode)
        published = failure in ('mode_changed', 'game_init')
        if state.get('error') != 'callback:' + failure or readq(game_static + 0x10) != (loaded_mode if published else old_mode): raise ValueError('callback failure publication boundary mismatch')
        callback_failures.append({'failure_gateway': failure, 'events_before_stop': state['events'], 'loaded_mode_published': published})
    options.clear(); state_cases = []
    for previous, current, callback in itertools.product([-1, 0, 5], [-1, 0, 6], (False, True)):
        d(game_static, 99); d(game_static + 4, previous)
        q(events_static + 8, delegates['state_changed'] if callback else 0)
        run('GameData.ChangeGameState', current)
        if readd(game_static) != previous & 0xFFFFFFFF or readd(game_static + 4) != current & 0xFFFFFFFF or state['events'] != (['state_changed'] if callback else []): raise ValueError('state publication mismatch')
        state_cases.append({'old_state': previous, 'new_state': current, 'callback': callback})
    for value in [-2147483648, -1, 0, 100, 2147483647]:
        run('GameData.UpdateCurrentVillage', value)
        if readd(game_static + 0x18) != value & 0xFFFFFFFF: raise ValueError('village setter clamps or transforms')
    lookup_cases = []
    q(game + 0x38, catalogue)
    for lookup_kind, values, query_index in itertools.product(['name', 'id'], [[], [1, 2, 3], [2, 1, 3], [1, None, 2]], range(4)):
        query = [0, 1000, 1001, 9999][query_index] if lookup_kind == 'name' else [0, 1500, 1501, 9999][query_index]
        field = 0x28 if lookup_kind == 'name' else 0x18
        collections[catalogue] = [objects[v] if v is not None else 0 for v in values]
        expected = 0; failed = False
        for v in values:
            if v is None: failed = True; break
            if readq(objects[v] + field) == query: expected = objects[v]; break
        returned = run('GameData.GetCharacterDataOfName' if lookup_kind == 'name' else 'GameData.GetCharacterDataOfId', game, query)
        if state.get('error') != ('null' if failed else None) or (not failed and returned != expected): raise ValueError('first catalogue match mismatch')
        lookup_cases.append({'lookup_kind': lookup_kind, 'catalogue': values, 'query_token': query, 'result_asset': next((i for i, p in objects.items() if p == expected), None), 'error': state.get('error')})
    init_cases = []
    for has_json, failure in itertools.product((False, True), [None, 'saved_data', 'saved_info', 'catalogue', 'null_character']):
        options.clear(); options['json'] = 5000 if has_json else 0
        q(game + 0x28, 123456); q(game + 0x30, 0 if failure == 'saved_data' else saved)
        q(saved + 0x18, 0 if failure == 'saved_info' else info); q(info + 0x10, 456789)
        q(game + 0x38, 0 if failure == 'catalogue' else catalogue)
        collections[catalogue] = [objects[1], 0 if failure == 'null_character' else objects[2]]
        run('GameData.Init', game)
        if readq(game_static + 8) != 123456 or readq(loc_static + 8) != 123456: raise ValueError('localization publication missing')
        save_reached = failure not in ('saved_data', 'saved_info')
        if len(state['preference_reads']) != ((2 if has_json else 1) if save_reached else 0): raise ValueError('PlayerPrefs read chronology mismatch')
        expected_loads = [] if failure in ('saved_data', 'saved_info', 'catalogue') else [objects[1]] if failure == 'null_character' else [objects[1], objects[2]]
        if state.get('error') != ('null' if failure else None) or state['preference_loads'] != expected_loads: raise ValueError('initialization partial failure mismatch')
        init_cases.append({'has_saved_json': has_json, 'failure_input': failure, 'preference_read_count': len(state['preference_reads']),
                           'json_decode_count': state['json_reads'], 'character_preference_load_count': len(state['preference_loads']), 'error': state.get('error')})
    growth_cases = []
    project_static, project_context, standard_array = arena + 0x80000, arena + 0x80100, arena + 0x80200
    q(project_type + 0xB8, project_static); q(project_static, project_context); q(project_context + 0x20, game)
    q(game + 0x60, standard_array)
    other_type, growth_mode = arena + 0x80300, arena + 0x80600
    for index, type_ptr in enumerate([standard_type, rogue_type]):
        hierarchy = arena + 0x81000 + index * 0x100
        uc.mem_write(type_ptr + 0x130, b'\x01'); q(type_ptr + 0xC8, hierarchy); q(hierarchy, type_ptr)
    uc.mem_write(other_type + 0x130, b'\x00')
    for mode, length, village in itertools.product(['standard', 'roguelike_standard', 'other', 'null'], [0, 1, 3], [-1, 0, 1, 2, 3, 2147483647]):
        options.clear(); options['max_level'] = length - 1
        q(standard_array + 0x18, length)
        q(growth_mode, {'standard': standard_type, 'roguelike_standard': rogue_type, 'other': other_type, 'null': other_type}[mode])
        q(game_static + 0x10, 0 if mode == 'null' else growth_mode); d(game_static + 0x18, village)
        run('GameData.IncreaseVillage')
        expected = village + 1 if mode in ('standard', 'roguelike_standard') and village < length - 1 else village
        if state.get('error') or readd(game_static + 0x18) != expected & 0xFFFFFFFF or state['max_level_reads'] != int(mode == 'roguelike_standard'):
            raise ValueError('mode-specific village increment mismatch')
        growth_cases.append({'mode': mode, 'standard_length_or_rogue_max_plus_one': length, 'old_village': village, 'new_village': expected})
    options.clear(); d(game_static, 101); d(game_static + 4, 102)
    run('GameData.cctor')
    default_mode = readq(game_static + 0x10)
    if state['allocation_types'] != [standard_type] or readd(default_mode + 0x38) != 20 or readd(game_static + 0x18) != 0 or bytes(uc.mem_read(game_static + 0x1C, 3)) != b'\0\0\0':
        raise ValueError('static default initialization mismatch')
    if readd(game_static) != 101 or readd(game_static + 4) != 102: raise ValueError('cctor unexpectedly resets game state fields')
    for value in [0, 123456]:
        q(game + 0x40, value)
        if run('GameData.GetAllRelics', game) != value: raise ValueError('relic getter does not preserve reference')
    achievement_cases = []
    achievement_array, unlocked_ids = arena + 0x91000, arena + 0x92000
    q(arena + 0x90000 + 0x10, unlocked_ids)
    for values, unlocked in itertools.product([[], [1, 2, 3], [1, None, 2]], [[], [1501]]):
        q(game + 0x80, achievement_array); q(achievement_array + 0x18, len(values))
        for index, value in enumerate(values): q(achievement_array + 0x20 + index * 8, 0 if value is None else objects[value])
        collections[unlocked_ids] = unlocked
        returned = run('GameData.GetUnlockedAchieves', game)
        expected = []
        for value in values:
            if value is None: break
            if readq(objects[value] + 0x18) in unlocked: expected.append(objects[value])
        failed = None in values
        output = collections[arena + 0x60000 + 0x100]
        if state.get('error') != ('null' if failed else None) or output != expected: raise ValueError('achievement enumeration mismatch')
        achievement_cases.append({'operation': 'get_unlocked', 'assets': values, 'unlocked_id_tokens': unlocked, 'result_count': len(expected), 'error': state.get('error')})
        run('GameData.UnlockAchievement', game, 1501)
        prefix = values[:values.index(None)] if failed else values
        expected_calls = [objects[v] for v in prefix if readq(objects[v] + 0x18) == 1501]
        if state.get('error') != ('null' if failed else None) or state['achievement_unlocks'] != expected_calls: raise ValueError('achievement unlock must visit every matching occurrence')
        achievement_cases.append({'operation': 'unlock', 'assets': values, 'requested_id_token': 1501, 'unlock_call_count': len(expected_calls), 'error': state.get('error')})
    return {'schema_version': 1, 'build_id': BUILD, 'game_assembly_sha256': manifest['inputs']['game_assembly']['sha256'],
            'target_count': len(targets), 'mode_cases': mode_cases, 'callback_failure_cases': callback_failures,
            'state_cases': state_cases, 'village_setter_cases': 5, 'catalogue_lookup_cases': lookup_cases, 'initialization_cases': init_cases,
            'village_growth_cases': growth_cases,
            'static_constructor_cases': 1, 'relic_reference_cases': 2, 'achievement_cases': achievement_cases,
            'native_case_count': len(mode_cases) + len(callback_failures) + len(state_cases) + 5 + len(lookup_cases) + len(init_cases) + len(growth_cases) + 3 + len(achievement_cases),
            'scope': 'Native GameData callers with initialized metadata; mode virtual methods, delegates, collections, strings, PlayerPrefs, JSON and character preference loading are explicit gateways. Callback-failure fixtures stop at the failing gateway, not native exception unwinding. Other declared methods are separate static findings.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--target-manifest', type=Path, default=Path(__file__).parents[1] / 'targets/game_data_lifecycle.json')
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.game_root, args.dumper_root, args.target_manifest)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} native GameData lifecycle cases")
