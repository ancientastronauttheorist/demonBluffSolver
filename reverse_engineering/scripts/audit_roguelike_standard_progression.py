"""Pinned RoguelikeStandard progression, reset and save interactions.

Native mode callers and nested ResetScores/save setters execute; external
class initialization, village advancement, JSON and preferences are gateways.
"""
import argparse
import hashlib
import itertools
import json
import random
import re
import struct
from pathlib import Path
from audit_character_assets import BUILD

FIELDS = {'bestAscension': 0x10, 'currentAscension': 0x14, 'currentDiedTimes': 0x18,
          'bestVillage': 0x1c, 'currentVillage': 0x20, 'bestScore': 0x24,
          'roundScore': 0x28, 'ascensionScore': 0x2c, 'prevAscensionScore': 0x30,
          'showedNewCharacters': 0x34}
ENTRIES = {'OnStageCompleted': (0x3ea320, 0x3ea3a9), 'AscensionComplete': (0x3e97b0, 0x3e9867),
           'OnFailed': (0x3ea2e0, 0x3ea311), 'ResetScores': (0x3ea3b0, 0x3ea447),
           'AbandonRun': (0x3e97a0, 0x3e97a7), 'Save': (0x3ea450, 0x3ea457),
           'set_RoguelikeStandard': (0x387eb0, 0x387ef6),
           'GetCurrentLevel': (0x3e9c80, 0x3e9c84), 'GetCurrentRunScore': (0x3e9c90, 0x3e9c97),
           'CanResetLevel': (0x3e9870, 0x3e9882), 'GetGameMode': (0x3712b0, 0x3712b3),
           'IsLocked': (0x3bcc90, 0x3bcc93), 'UpdateScore': (0x33ed50, 0x33ed53)}
ALIASES = {'GetStartingLevel': 'GetCurrentLevel', 'GetResetLevel': 'GetCurrentLevel',
           'GetMaxLevel': 'GetGameMode', 'GetScore': 'GetGameMode'}


def i32(value):
    return (value + 2**31) % 2**32 - 2**31


def project(method, initial, cold, failure):
    fields = dict(initial)
    events, snapshots, counts = [], [], {}
    global_village = 123
    class Stop(Exception):
        pass
    def event(name):
        events.append(name)
        counts[name] = counts.get(name, 0) + 1
        if failure == (name, counts[name]):
            raise Stop
    def initialize():
        nonlocal cold
        if cold:
            event('class_init')
            cold = False
    def save():
        snapshots.append(dict(fields))
        event('json')
        event('set_string')
    def reset():
        nonlocal global_village
        initialize()
        global_village = 0
        for key in ('currentDiedTimes', 'currentVillage', 'roundScore', 'ascensionScore'):
            fields[key] = 0
        save()
    failed = False
    try:
        if method == 'OnStageCompleted':
            fields['currentVillage'] = i32(fields['currentVillage'] + 1)
            amount = i32(fields['roundScore'] + 50 + fields['currentAscension'] * 50)
            fields['roundScore'] = 0
            fields['ascensionScore'] = i32(fields['ascensionScore'] + amount)
            if fields['currentAscension'] >= fields['bestAscension'] and fields['currentVillage'] >= fields['bestVillage']:
                fields['bestVillage'] = fields['currentVillage']
            fields['bestScore'] = max(fields['bestScore'], fields['ascensionScore'])
            initialize()
            event('increase_village')
            save()
        elif method == 'AscensionComplete':
            fields['currentAscension'] = i32(fields['currentAscension'] + 1)
            fields['bestVillage'] = 0
            fields['bestAscension'] = max(fields['bestAscension'], fields['currentAscension'])
            fields['prevAscensionScore'] = i32(fields['ascensionScore'] + fields['roundScore'])
            initialize()
            global_village = 0
            reset()
            save()
        elif method == 'OnFailed':
            fields['currentDiedTimes'] = i32(fields['currentDiedTimes'] + 1)
            if fields['currentDiedTimes'] >= 4:
                event('abandon_dispatch')
                reset()
            save()
        elif method in ('ResetScores', 'AbandonRun'):
            reset()
        elif method in ('Save', 'set_RoguelikeStandard'):
            save()
    except Stop:
        failed = True
    return fields, events, snapshots, global_village, failed


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
    declaration = re.search(r'^public class RoguelikeStandard : GameMode .*?^}', dump, re.M | re.S).group()
    for name, offset in FIELDS.items():
        kind = 'bool' if name == 'showedNewCharacters' else 'int'
        assert f'public {kind} {name}; // 0x{offset:X}' in declaration
    methods = {}
    for name in [*ENTRIES, *ALIASES]:
        owner = 'SavesGame' if name == 'set_RoguelikeStandard' else 'RoguelikeStandard'
        address = ENTRIES[ALIASES.get(name, name)][0]
        rows = [row for row in script['ScriptMethod'] if row['Name'] == owner + '$$' + name]
        assert len(rows) == 1 and rows[0]['Address'] == address
        result_type = 'bool' if name in ('CanResetLevel', 'IsLocked') else 'int32_t' if name.startswith('Get') else 'void'
        arguments = 'RoguelikeStandard_o* value, const MethodInfo* method' if owner == 'SavesGame' else 'RoguelikeStandard_o* __this, ' + ('int32_t score, int32_t level, ' if name == 'UpdateScore' else '') + 'const MethodInfo* method'
        assert rows[0]['Signature'] == f'{result_type} {owner}__{name} ({arguments});'
        methods[name] = rows[0]
    assert re.search(r'// RVA: 0x3E97A0 [^\n]* Slot: 15\n\s*public override void AbandonRun\(\)', declaration)
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    decoded = {}
    for a, b in ENTRIES.values():
        items = list(cs.disasm(pe.get_data(a, b - a), a))
        assert items[0].address == a and sum(i.size for i in items) == b - a
        assert items[-1].address + items[-1].size == b and all(i.mnemonic != 'int3' for i in items)
        decoded.update({i.address: i for i in items})
    checks = {0x3ea34b: ('inc', 'dword ptr [rbx + 0x20]'), 0x3ea355: ('imul', 'ecx, edx, 0x32'),
              0x3ea361: ('add', 'dword ptr [rbx + 0x2c], ecx'), 0x3ea367: ('jl', '0x3ea373'),
              0x3ea36d: ('jl', '0x3ea373'), 0x3ea379: ('jle', '0x3ea37e'),
              0x3ea2e6: ('inc', 'dword ptr [rcx + 0x18]'), 0x3ea2ec: ('cmp', 'dword ptr [rcx + 0x18], 4'),
              0x3ea2fc: ('call', 'qword ptr [rax + 0x228]'), 0x3e97a2: ('jmp', '0x3ea3b0'),
              0x3e97d5: ('inc', 'dword ptr [rbx + 0x14]'), 0x3e97f0: ('mov', 'dword ptr [rbx + 0x30], eax'),
              0x3e9853: ('call', '0x3ea3b0'), 0x3e9862: ('jmp', '0x387eb0'),
              0x3ea439: ('mov', 'qword ptr [rbx + 0x28], rax'), 0x3ea442: ('jmp', '0x387eb0'),
              0x3bcc90: ('xor', 'al, al'), 0x33ed50: ('ret', '0')}
    for address, expected in checks.items():
        assert (decoded[address].mnemonic, decoded[address].op_str) == expected
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    for address in (arena, stack, stop): uc.mem_map(address, 0x10000)
    mode, klass, game_type, game_static, key, json_text = [arena + off for off in (0, 0x1000, 0x2000, 0x3000, 0x4000, 0x5000)]
    def q(address, value): uc.mem_write(address, struct.pack('<Q', value))
    def d(address, value): uc.mem_write(address, struct.pack('<I', value & 0xffffffff))
    def rq(address): return struct.unpack('<Q', uc.mem_read(address, 8))[0]
    def ri(address): return struct.unpack('<i', uc.mem_read(address, 4))[0]
    rows = [row for row in script['ScriptMetadata'] if row['Name'] == 'GameData_TypeInfo']
    assert len(rows) == 1
    q(base + rows[0]['Address'], game_type)
    q(game_type + 0xb8, game_static)
    q(mode, klass)
    q(klass + 0x228, base + ENTRIES['AbandonRun'][0])
    q(klass + 0x230, arena + 0x6000)
    literal = [row for row in script['ScriptString'] if row['Address'] == 0x2711c08]
    assert len(literal) == 1 and literal[0]['Value'] == 'SavedRoguelikeStandard'
    q(base + literal[0]['Address'], key)
    for ins in decoded.values():
        if ins.mnemonic == 'cmp' and ins.operands[0].type == capstone.CS_OP_MEM and ins.operands[0].size == 1 and ins.operands[0].mem.base == capstone.x86.X86_REG_RIP:
            uc.mem_write(base + ins.address + ins.size + ins.operands[0].mem.disp, b'\1')
    state, options, visited = {}, {}, set()
    def event(name):
        state['events'].append(name)
        state['counts'][name] = state['counts'].get(name, 0) + 1
        if options['failure'] == (name, state['counts'][name]):
            state['failed'] = True
            uc.emu_stop()
            return False
        return True
    def ret(value=0):
        sp = uc.reg_read(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, value)
        uc.reg_write(x.UC_X86_REG_RSP, sp + 8)
        uc.reg_write(x.UC_X86_REG_RIP, rq(sp))
    def fields():
        return {name: int(uc.mem_read(mode + off, 1)[0]) if name == 'showedNewCharacters' else ri(mode + off) for name, off in FIELDS.items()}
    def hook(_, address, size, __):
        rva = address - base
        rcx, rdx, r8 = [uc.reg_read(reg) for reg in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        if rva == 0x281d90:
            assert rcx == game_type
            if event('class_init'): d(game_type + 0xe0, 1); ret()
        elif rva == 0x3dc820:
            assert rcx == 0
            if event('increase_village'): ret()
        elif rva == 0x1cd6420:
            assert rcx == mode and rdx == 0
            state['snapshots'].append(fields())
            if event('json'): ret(json_text + state['counts']['json'] * 0x100)
        elif rva == 0x1c86170:
            assert (rcx, rdx, r8) == (key, json_text + state['counts']['json'] * 0x100, 0)
            if event('set_string'): ret()
        else:
            assert rva in decoded and decoded[rva].size == size, hex(rva)
            visited.add(rva)
            if rva == 0x3ea2fc:
                assert rcx == mode and rdx == arena + 0x6000
                event('abandon_dispatch')
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    defaults = dict(zip(FIELDS, (7, 6, 2, 4, 3, 1000, 80, 500, 200, 1)))
    cases = []
    def run(method, changes=None, cold=False, failure=None):
        initial = defaults | (changes or {})
        options.clear(); options.update(failure=failure)
        state.clear(); state.update(events=[], counts={}, snapshots=[], failed=False)
        uc.mem_write(mode, bytes([0xa5]) * 128); q(mode, klass)
        for name, value in initial.items():
            if name == 'showedNewCharacters': uc.mem_write(mode + FIELDS[name], bytes([value]))
            else: d(mode + FIELDS[name], value)
        source = bytes(uc.mem_read(mode, 128))
        d(game_type + 0xe0, 0 if cold else 1); d(game_static + 0x18, 123)
        sp = stack + 0x8008; q(sp, stop)
        uc.reg_write(x.UC_X86_REG_RSP, sp)
        uc.reg_write(x.UC_X86_REG_RCX, mode); uc.reg_write(x.UC_X86_REG_RDX, 0)
        uc.reg_write(x.UC_X86_REG_RAX, 0xabcdef1234567890)
        for index, reg in enumerate(preserved): uc.reg_write(reg, 0xabcd0000 + index)
        uc.emu_start(base + methods[method]['Address'], stop, count=3000)
        expected, events, snapshots, village, failed = project(method, initial, cold, failure)
        expected_bytes = bytearray(source)
        for name, value in expected.items():
            if name == 'showedNewCharacters': expected_bytes[FIELDS[name]] = value
            else: struct.pack_into('<I', expected_bytes, FIELDS[name], value & 0xffffffff)
        assert bytes(uc.mem_read(mode, 128)) == bytes(expected_bytes), (method, initial, failure, fields(), expected)
        assert (state['events'], state['snapshots'], ri(game_static + 0x18), state['failed']) == (events, snapshots, village, failed), (method, failure, state, events)
        if not failed:
            assert uc.reg_read(x.UC_X86_REG_RIP) == stop and uc.reg_read(x.UC_X86_REG_RSP) == sp + 8
            assert all(uc.reg_read(reg) == 0xabcd0000 + index for index, reg in enumerate(preserved))
        result = uc.reg_read(x.UC_X86_REG_RAX)
        if method in ('GetCurrentLevel', 'GetStartingLevel', 'GetResetLevel'): assert i32(result) == initial['currentVillage']
        if method == 'GetCurrentRunScore': assert i32(result) == i32(initial['ascensionScore'] + initial['roundScore'])
        if method in ('GetGameMode', 'GetMaxLevel', 'GetScore'): assert result == 0
        if method == 'CanResetLevel': assert bool(result & 0xff) == (initial['currentVillage'] > 0 or initial['currentDiedTimes'] > 0)
        if method == 'IsLocked': assert result & 0xff == 0 and result >> 8 == 0xabcdef12345678
        if method == 'UpdateScore': assert result == 0xabcdef1234567890
        cases.append({'method': method, 'initial': initial, 'cold_class': cold, 'failure': failure,
                      'events': events, 'saved_snapshots': snapshots, 'result_fields': expected,
                      'result_game_data_initialized': ri(game_type + 0xe0) != 0,
                      'result_global_village': village, 'stopped_at_gateway': failed})
    failures = [None, ('class_init', 1), ('increase_village', 1), ('abandon_dispatch', 1),
                ('json', 1), ('set_string', 1), ('json', 2), ('set_string', 2)]
    for method, cold, failure in itertools.product(list(ENTRIES)[:7], (False, True), failures):
        run(method, cold=cold, failure=failure)
    for deaths in (-2**31, -1, 0, 2, 3, 4, 2**31 - 1):
        for failure in failures:
            run('OnFailed', {'currentDiedTimes': deaths}, cold=True, failure=failure)
    boundaries = [-2**31, -1, 0, 1, 2**31 - 1]
    for method in ('OnStageCompleted', 'AscensionComplete'):
        for field in ('bestAscension', 'currentAscension', 'bestVillage', 'currentVillage', 'bestScore', 'roundScore', 'ascensionScore'):
            for value in boundaries: run(method, {field: value})
    rng = random.Random(0x3ea320)
    for _ in range(300):
        changes = {key: i32(rng.getrandbits(32)) for key in FIELDS if key != 'showedNewCharacters'}
        for method in ('OnStageCompleted', 'AscensionComplete', 'OnFailed'): run(method, changes)
    for method in [*list(ENTRIES)[7:], *ALIASES]:
        for village, deaths, score, round_score in itertools.product((-1, 0, 1), (-1, 0, 1), (-2**31, 2**31 - 1), (-1, 1)):
            run(method, {'currentVillage': village, 'currentDiedTimes': deaths, 'ascensionScore': score, 'roundScore': round_score})
    unvisited = []
    for ins in decoded.values():
        if ins.address not in visited: unvisited.append(hex(ins.address))
    metadata_excluded = [0x3ea332, 0x3ea339, 0x3ea33e, 0x3e97c2, 0x3e97c9, 0x3e97ce,
                         0x3e9811, 0x3e9818, 0x3e981d, 0x3ea3c2, 0x3ea3c9, 0x3ea3ce,
                         0x3ea3f3, 0x3ea3fa, 0x3ea3ff, 0x387ec2, 0x387ec9, 0x387ece]
    redundant_class_calls = [0x3e9834, 0x3ea416]
    assert {int(rva, 16) for rva in unvisited} == set(metadata_excluded + redundant_class_calls)
    return {'schema_version': 1, 'build_id': BUILD, 'native_case_count': len(cases), 'metadata_declarations': methods,
            'field_offsets': {k: hex(v) for k, v in FIELDS.items()}, 'save_key': literal[0]['Value'],
            'abandon_virtual_slot': 15, 'native_ranges': {name: [hex(v) for v in bounds] for name, bounds in ENTRIES.items()},
            'instruction_assertions': len(checks), 'visited_instruction_count': len(visited),
            'unvisited_instruction_rvas': unvisited,
            'unvisited_warmed_metadata_instruction_rvas': [hex(v) for v in metadata_excluded],
            'redundant_class_init_calls_skipped_after_successful_initialization': [hex(v) for v in redundant_class_calls],
            'cases': cases,
            'scope': 'Native progression/reset/save callers and nested actual RoguelikeStandard AbandonRun slot15; exact 128-byte object snapshots and partial gateway failures, signed int32 wrap. Metadata warmed. Class initialization, GameData.IncreaseVillage, JSON and PlayerPrefs.SetString are explicit gateways; village helper side effects are not projected. No live save or state mutation, exception unwinding, full mode overrides or showcase/selector audit.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path); parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_case_count']} RoguelikeStandard progression cases")
