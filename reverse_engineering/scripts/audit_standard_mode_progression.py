"""Execute pinned StandardMode progression/scoring callers with explicit services.

Native instructions remain private. No live saves, achievements or UI are changed.
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
    assert unicorn.__version__ == '2.1.4'
    repo = Path(__file__).parents[1]
    manifest = json.loads((repo / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((repo / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(path, digest):
        raw = Path(path).read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == digest.upper()
        return raw
    raw = pinned(Path(game_root) / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    script = json.loads(pinned(Path(dumper_root) / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    targets = json.loads((repo / 'targets/standard_mode_progression.json').read_text(encoding='utf-8'))['functions']
    for f in targets:
        assert any(r['Name'] == f['metadata_name'] and r['Signature'] == f['signature'] and r['Address'] == int(f['rva'], 16) for r in script['ScriptMethod'])
    supplemental = {}
    for name, rva in [('GetGameMode', 0x3712B0), ('GetCurrentLevel', 0x3E9250), ('GetScore', 0x358300), ('IsLocked', 0x3BCC90)]:
        row = next(r for r in script['ScriptMethod'] if r['Name'] == 'StandardMode$$' + name)
        assert row['Address'] == rva
        supplemental[name] = row
    pe = pefile.PE(data=raw, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    ends = {0x387AC0: 0x387AF5, 0x387F00: 0x387F52, 0x387F60: 0x387FA6,
            0x3EDD40: 0x3EDD47, 0x3EE4E0: 0x3EE5CE, 0x3EE4B0: 0x3EE4D6,
            0x3EDE00: 0x3EDE87, 0x3ED880: 0x3ED892, 0x3EDF90: 0x3EE017,
            0x3EE5E0: 0x3EE6D8, 0x3ED8B0: 0x3ED920, 0x3EE3B0: 0x3EE407,
            0x3EE410: 0x3EE4A2, 0x3ED8A0: 0x3ED8A5, 0x3ED800: 0x3ED880,
            0x3EE5D0: 0x3EE5D7, 0x3EDE90: 0x3EDF8E}
    decoded = {}
    for start, end in ends.items():
        data = pe.get_data(start, end-start)
        assert len(data) == end-start
        ins = list(cs.disasm(data, start))
        assert ins[0].address == start and ins[-1].address + ins[-1].size == end
        assert all(i.mnemonic != 'int3' for i in ins)
        decoded.update({i.address: i for i in ins})
    # Supplementary folded declarations are checked at their exact entries.
    for row in supplemental.values():
        ins = []
        for i in cs.disasm(pe.get_data(row['Address'], 16), row['Address']):
            ins.append(i)
            if i.mnemonic == 'ret': break
        assert ins[-1].mnemonic == 'ret'
        decoded.update({i.address: i for i in ins})
    checks = [(0x3EE509, 'inc', 'dword ptr [rbx + 0x18]'),
              (0x3EE55C, 'mov', 'dword ptr [rbx + 0x28], 0'),
              (0x3EE5A5, 'jle', '0x3ee5aa'), (0x3EE5B0, 'jle', '0x3ee5b5'),
              (0x3EE4B9, 'sub', 'dword ptr [rcx + 0x2c], eax'),
              (0x3EE66C, 'mov', 'dword ptr [rdi + 0x14], ebp'),
              (0x3EE6AD, 'mov', 'ecx, dword ptr [rax + 0x18]'),
              (0x3ED840, 'mov', 'qword ptr [rbx + 0x10], rax'),
              (0x3ED847, 'mov', 'qword ptr [rbx + 0x28], rax'),
              (0x3EE44A, 'cmp', 'dword ptr [rax + 0x130], 0x64'),
              (0x3EE453, 'cmp', 'dword ptr [rax + 0x130], 0x1e'),
              (0x3EE46B, 'add', 'dword ptr [rdi + 0x28], ecx'),
              (0x3EE46E, 'add', 'dword ptr [rdi + 0x2c], ecx'),
              (0x387F34, 'jle', '0x387f4c'), (0x3ED910, 'cmp', 'edi, ecx')]
    for address, mnemonic, operands in checks:
        assert (decoded[address].mnemonic, decoded[address].op_str) == (mnemonic, operands)
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, service, stop = 0x200000000, 0x300000000, 0x400000000, 0x400001000
    uc.mem_map(arena, 0x100000); uc.mem_map(stack, 0x10000); uc.mem_map(service, 0x2000)
    def q(a, v): uc.mem_write(a, struct.pack('<Q', v))
    def d(a, v): uc.mem_write(a, struct.pack('<I', v & 0xffffffff))
    def rq(a): return struct.unpack('<Q', uc.mem_read(a, 8))[0]
    def ri(a): return struct.unpack('<i', uc.mem_read(a, 4))[0]
    def reg(r): return uc.reg_read(r)
    def i32(v): return (v + 2**31) % 2**32 - 2**31
    def ret(v=0):
        sp = reg(x.UC_X86_REG_RSP)
        uc.reg_write(x.UC_X86_REG_RAX, v & 0xffffffffffffffff)
        uc.reg_write(x.UC_X86_REG_RSP, sp+8); uc.reg_write(x.UC_X86_REG_RIP, rq(sp))
    type_names = ['ProjectContext_TypeInfo', 'GameData_TypeInfo', 'UIEvents_TypeInfo', 'UnityEngine.Object_TypeInfo', 'int_TypeInfo']
    types = {name: arena+0x1000+n*0x400 for n, name in enumerate(type_names)}
    found = set()
    for r in script['ScriptMetadata']:
        if r['Name'] in types:
            q(base+r['Address'], types[r['Name']]); found.add(r['Name'])
    assert found == set(types), set(types)-found
    strings = {}
    for load in (0x387AE0, 0x387F8F, 0x3EDF63, 0x3EDEF3, 0x3EDF20):
        instruction = decoded[load]
        assert instruction.operands[1].mem.base == capstone.x86.X86_REG_RIP
        address = instruction.address + instruction.size + instruction.operands[1].mem.disp
        rows = [r for r in script['ScriptString'] if r['Address'] == address]
        assert len(rows) == 1, hex(address)
        pointer = arena+0x3000+len(strings)*0x100
        strings[pointer] = rows[0]['Value']; q(base+address, pointer)
    project_static, game_static, ui_static = arena+0x4000, arena+0x5000, arena+0x6000
    project, game, array, mode, klass, char, data, delegate = [arena+n for n in range(0x7000, 0xF000, 0x1000)]
    q(types['ProjectContext_TypeInfo']+0xB8, project_static)
    q(types['GameData_TypeInfo']+0xB8, game_static)
    q(types['UIEvents_TypeInfo']+0xB8, ui_static)
    q(mode, klass); q(klass+0x1B8, service); q(klass+0x1C0, arena+0xF100)
    q(klass+0x268, service+0x20); q(klass+0x270, arena+0xF200)
    q(delegate+0x18, service+0x10); q(delegate+0x28, arena+0xF300); q(delegate+0x40, mode)
    for i in decoded.values():
        if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
            uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp, b'\1')
    fields = {'score': 0x10, 'currentLevel': 0x14, 'savedVillages': 0x18, 'currentDiedTimes': 0x1C,
              'bestDiedTimes': 0x20, 'completed': 0x24, 'roundScore': 0x28, 'currentScore': 0x2C,
              'bestScore': 0x30, 'currentCompleted': 0x34, 'failScoreDecrease': 0x38}
    defaults = dict(zip(fields, [100, 2, 8, 3, 7, 0, 60, 400, 800, 0, 20]))
    opt, state, visited = {}, {}, set()
    def halt(error): state['error'] = error; uc.emu_stop()
    def event(name):
        state['events'].append(name)
        if opt.get('fail') == name: halt(name); return False
        return True
    def hook(_, address, size, __):
        rva = address-base
        rcx, rdx, r8 = (reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8))
        if rva in (0x1C85EE0, 0x1C85EA0):
            assert strings[rcx] == 'SavedStandardMode' and rdx == 0
            if rva == 0x1C85EE0: assert r8 == 0
            if event('get_int'):
                values = opt.get('prefs', [0])
                index = state['reads']; state['reads'] += 1
                ret(values[min(index, len(values)-1)])
        elif rva == 0x1C860E0:
            assert strings[rcx] == 'SavedStandardMode' and r8 == 0
            state['set_int'].append(i32(rdx))
            if event('set_int'): ret()
        elif rva == 0x1CD6420:
            assert rcx == (0 if opt.get('null_mode') else mode) and rdx == 0
            state['saved_state'] = bytes(uc.mem_read(mode+0x10, 0x30))
            if event('json'): ret(arena+0xF400)
        elif rva == 0x1C86170:
            assert strings[rcx] == 'SavedStandard' and rdx == arena+0xF400 and r8 == 0
            if event('set_string'): ret()
        elif rva == 0x281D90:
            assert rcx in (types['GameData_TypeInfo'], types['UnityEngine.Object_TypeInfo'])
            if event('class_init'): d(rcx+0xE0, 1); ret()
        elif rva == 0x3DC820:
            assert rcx == 0
            if event('increase_village'): ret()
        elif rva == 0x398D10:
            assert rcx == 0
            if event('unrevealed'): ret(opt.get('hidden', 3))
        elif rva == 0x1C822C0:
            assert rcx == rq(project_static) and rdx == r8 == 0
            if event('unity_null'): ret(opt.get('unity_null', False))
        elif address == service:
            assert rcx == mode and rdx == arena+0xF100
            if event('virtual_deinit'): ret()
        elif address == service+0x10:
            assert rcx == mode and rdx == arena+0xF300
            if event('ui'): ret()
        elif address == service+0x20:
            assert rcx == mode and rdx == arena+0xF200
            if event('virtual_max'): ret(opt.get('virtual_max', 6))
        elif rva == 0x282580:
            assert rcx == types['int_TypeInfo']
            p = arena+0x20000+len(state['boxes'])*0x100
            state['boxes'][p] = ri(rdx); ret(p)
        elif rva == 0xF74DF0:
            assert r8 == 0 and rcx in strings and rdx in state['boxes']
            state['formats'].append([strings[rcx], state['boxes'][rdx]])
            if event('format'): ret(arena+0x30000+len(state['formats'])*0x100)
        elif rva == 0xF71940:
            assert (rcx, rdx, r8) == (arena+0x30100, arena+0x30200, arena+0x30300)
            if event('concat'): ret(arena+0xF500)
        elif rva == 0x2B7D90: halt('null')
        elif rva not in decoded: raise AssertionError(f'Unexpected native instruction/service {rva:x}')
        else: visited.add(rva)
    uc.hook_add(unicorn.UC_HOOK_CODE, hook)
    preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                 x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
    cases = []
    def run(name, changes=None, options=None, args=()):
        opt.clear(); opt.update(options or {})
        state.clear(); state.update(events=[], reads=0, set_int=[], boxes={}, formats=[])
        initial = defaults | (changes or {})
        uc.mem_write(mode+0x10, bytes([0xA5])*0x30)
        for key, value in initial.items():
            if key in ('completed', 'currentCompleted'): uc.mem_write(mode+fields[key], bytes([value]))
            else: d(mode+fields[key], value)
        before = bytes(uc.mem_read(mode+0x10, 0x30))
        for pointer in types.values(): d(pointer+0xE0, 0 if opt.get('cold') else 1)
        missing = opt.get('missing')
        q(project_static, 0 if missing == 'project' else project)
        q(project+0x20, 0 if missing == 'game' else game)
        q(game+0x60, 0 if missing == 'array' else array)
        d(array+0x18, opt.get('count', 7)); d(game_static+0x18, opt.get('village', 4))
        q(char+0x50, 0 if missing == 'data' else data); d(data+0x130, opt.get('type', 100))
        d(char+0xF8, opt.get('alignment', 10)); d(data+0x134, opt.get('starting_alignment', 10))
        q(char+0x58, arena+0x40000); q(char+0x60, arena+0x41000)
        d(arena+0x40000+0x130, opt.get('apparent_type', 10))
        d(arena+0x41000+0x130, opt.get('apparent_type', 10))
        q(ui_static, delegate if opt.get('ui') else 0)
        sp = stack+0x8008; q(sp, stop)
        for index, register in enumerate(preserved): uc.reg_write(register, 0xABCD0000+index)
        entry = next((int(t['rva'], 16) for t in targets if t['name'] == name), None)
        if entry is None: entry = supplemental[name.split('.')[-1]]['Address']
        native_args = [mode, 0, 0, 0]
        if name.startswith('SavesGame.'):
            native_args[0] = args[0] if args else (0 if opt.get('null_mode') else mode)
        elif name == 'StandardMode.OnCharacterKilled': native_args[1] = 0 if missing == 'character' else char
        else:
            for index, value in enumerate(args, 1): native_args[index] = value
        for register, value in zip([x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9], native_args):
            uc.reg_write(register, value & 0xffffffffffffffff)
        uc.reg_write(x.UC_X86_REG_RAX, 0xABCD1234567800FE)
        uc.reg_write(x.UC_X86_REG_RSP, sp)
        uc.emu_start(base+entry, stop, timeout=1_000_000, count=20000)
        if 'error' not in state:
            assert reg(x.UC_X86_REG_RIP) == stop and reg(x.UC_X86_REG_RSP) == sp+8
            for index, register in enumerate(preserved): assert reg(register) == 0xABCD0000+index
        after = bytes(uc.mem_read(mode+0x10, 0x30))
        result = {'method': name, 'initial': initial, 'options': dict(opt), 'args': list(args),
                  'events': list(state['events']), 'error': state.get('error'), 'set_int': list(state['set_int']),
                  'result_i32': i32(reg(x.UC_X86_REG_RAX)), 'result_bool': bool(reg(x.UC_X86_REG_RAX)&0xff),
                  'final': {key: (after[offset-0x10] if key in ('completed', 'currentCompleted') else struct.unpack_from('<i', after, offset-0x10)[0]) for key, offset in fields.items()}}
        cases.append(result)
        return initial, bytearray(before), after, result
    def expected_bytes(before, updates):
        for key, value in updates.items():
            offset = fields[key]-0x10
            if key in ('completed', 'currentCompleted'): before[offset] = value
            else: struct.pack_into('<I', before, offset, value & 0xffffffff)
        return bytes(before)
    for completed, deaths, score, round_score, fail_score, failure in itertools.product(
            (0, 1), (3, 2**31-1), (-100, 400), (0, 60), (-20, 20), (None, 'json', 'set_string')):
        initial, before, after, result = run('StandardMode.OnFailed',
            {'completed': completed, 'currentDiedTimes': deaths, 'currentScore': score, 'roundScore': round_score, 'failScoreDecrease': fail_score}, {'fail': failure})
        updates = {'currentDiedTimes': i32(deaths+1), 'currentScore': i32(score-fail_score-round_score), 'roundScore': 0}
        if not completed: updates['bestDiedTimes'] = i32(deaths+1)
        assert after == expected_bytes(before, updates)
        assert result['error'] == failure and state['saved_state'] == after
    for score, round_score, deduction in ((-2**31, 60, 20), (2**31-1, 2**31-1, 20), (0, -2**31, -20)):
        initial, before, after, result = run('StandardMode.OnFailed', {'currentScore': score, 'roundScore': round_score, 'failScoreDecrease': deduction})
        assert after == expected_bytes(before, {'currentScore': i32(score-round_score-deduction), 'roundScore': 0, 'currentDiedTimes': 4, 'bestDiedTimes': 4})
    for level, count, completed, deaths, best_deaths in itertools.product((-1, 5, 6, 8), (0, 7), (0, 1), (2, 9), (3, 10)):
        initial, before, after, result = run('StandardMode.OnStageCompleted',
            {'currentLevel': level, 'completed': completed, 'currentDiedTimes': deaths, 'bestDiedTimes': best_deaths}, {'count': count})
        updates = {'savedVillages': initial['savedVillages']+1, 'roundScore': 0}
        if level >= count-1:
            updates.update(completed=1, currentCompleted=1, bestDiedTimes=max(deaths, best_deaths), bestScore=max(initial['currentScore'], initial['bestScore']))
            assert 'increase_village' not in result['events']
        else:
            if not completed: updates['bestScore'] = initial['currentScore']
            assert result['events'][0] == 'increase_village'
        assert after == expected_bytes(before, updates) and result['error'] is None
    for missing in ('project', 'game', 'array'):
        initial, before, after, result = run('StandardMode.OnStageCompleted', options={'missing': missing})
        assert after == expected_bytes(before, {'savedVillages': initial['savedVillages']+1}) and result['error'] == 'null'
    for score, best in itertools.product((-100, 400, 1000), repeat=2):
        initial, before, after, result = run('StandardMode.OnStageCompleted', {'currentLevel': 6, 'currentScore': score, 'bestScore': best, 'savedVillages': 2**31-1})
        assert after == expected_bytes(before, {'savedVillages': -2**31, 'roundScore': 0, 'currentCompleted': 1, 'completed': 1, 'bestScore': max(score, best)})
    for failure in ('class_init', 'increase_village', 'json', 'set_string'):
        initial, before, after, result = run('StandardMode.OnStageCompleted', options={'fail': failure, 'cold': True})
        assert after == expected_bytes(before, {'savedVillages': initial['savedVillages']+1, 'roundScore': 0, 'bestScore': initial['currentScore']})
        assert result['error'] == failure
    for old, requested, score, saved, village in itertools.product((2, 6), (-3, 3, 20), (-1, 200), (0, 10), (-1, 7)):
        initial, before, after, result = run('StandardMode.UpdateScore', {'currentLevel': old}, {'prefs': [saved], 'village': village}, (score, requested))
        level = requested if old < 6 else old
        assert after == expected_bytes(before, {'currentLevel': level, 'score': max(initial['score'], score)})
        assert result['set_int'] == ([village] if level > saved and village > saved else [])
        assert state['reads'] == (2 if level > saved else 1)
    for failure in ('get_int', 'class_init', 'set_int', 'json', 'set_string'):
        initial, before, after, result = run('StandardMode.UpdateScore', options={'fail': failure, 'cold': True, 'prefs': [0], 'village': 7}, args=(200, 3))
        assert after == expected_bytes(before, {'currentLevel': 3, 'score': 200})
        assert result['error'] == failure
    for failure in ('json', 'set_string'):
        initial, before, after, result = run('StandardMode.OnStageCompleted', {'currentLevel': 6, 'currentScore': 1000, 'currentDiedTimes': 10}, {'fail': failure})
        assert after == expected_bytes(before, {'savedVillages': 9, 'roundScore': 0, 'currentCompleted': 1, 'completed': 1, 'bestDiedTimes': 10, 'bestScore': 1000})
        assert result['error'] == failure
    # The second preference read can change the monotone setter's decision.
    _, before, after, result = run('StandardMode.UpdateScore', options={'prefs': [0, 100], 'village': 7}, args=(200, 3))
    assert result['set_int'] == [] and state['reads'] == 2
    for saved, requested in itertools.product((-2**31, -1, 0, 9, 2**31-1), repeat=2):
        _, before, after, result = run('SavesGame.set_SavedMaxStandardAscension', options={'prefs': [saved]}, args=(requested,))
        assert after == before and result['set_int'] == ([requested] if requested > saved else [])
    for saved in (-2**31, -1, 0, 2**31-1):
        for name in ('SavesGame.get_SavedMaxStandardAscension', 'StandardMode.GetMaxLevel'):
            _, before, after, result = run(name, options={'prefs': [saved]})
            assert after == before and result['result_i32'] == saved and state['reads'] == 1
    for kind, hidden, ui in itertools.product((0, 10, 20, 30, 100), (-5, 0, 8, 2**31-1), (False, True)):
        initial, before, after, result = run('StandardMode.OnCharacterKilled', options={'type': kind, 'hidden': hidden, 'ui': ui})
        updates = {}
        if kind in (30, 100):
            amount = i32((hidden+5)*10)
            updates = {'roundScore': i32(initial['roundScore']+amount), 'currentScore': i32(initial['currentScore']+amount)}
            assert result['events'] == ['unrevealed']+(['ui'] if ui else [])
        else: assert result['events'] == []
        assert after == expected_bytes(before, updates)
    for kind, alignment, apparent in itertools.product((10, 30, 100), (10, 100), (10, 100)):
        initial, before, after, result = run('StandardMode.OnCharacterKilled', options={'type': kind, 'alignment': alignment, 'starting_alignment': alignment, 'apparent_type': apparent})
        updates = {} if kind == 10 else {'roundScore': initial['roundScore']+80, 'currentScore': initial['currentScore']+80}
        assert after == expected_bytes(before, updates)
    for missing in ('character', 'data'):
        _, before, after, result = run('StandardMode.OnCharacterKilled', options={'missing': missing})
        assert before == after and result['error'] == 'null'
    for failure in ('unrevealed', 'ui'):
        initial, before, after, result = run('StandardMode.OnCharacterKilled', options={'fail': failure, 'ui': True})
        updates = {} if failure == 'unrevealed' else {'roundScore': initial['roundScore']+80, 'currentScore': initial['currentScore']+80}
        assert after == expected_bytes(before, updates) and result['error'] == failure
    for failure, ui in itertools.product((None, 'virtual_deinit', 'json', 'set_string', 'ui'), (False, True)):
        initial, before, after, result = run('StandardMode.AbandonRun', {'completed': 1, 'currentCompleted': 1}, {'fail': failure, 'ui': ui})
        updates = {} if failure == 'virtual_deinit' else dict(score=0, currentLevel=0, roundScore=0, currentScore=0, currentDiedTimes=0, currentCompleted=0)
        assert after == expected_bytes(before, updates)
        assert result['error'] == (None if failure == 'ui' and not ui else failure)
    for name in ('StandardMode.GetStartingLevel', 'StandardMode.GetResetLevel'):
        for level, is_null, missing, cold in itertools.product((-2, 0, 9), (False, True), (None, 'project'), (False, True)):
            _, before, after, result = run(name, {'currentLevel': level}, {'unity_null': is_null, 'missing': missing, 'cold': cold})
            assert after == before and result['result_i32'] == (0 if is_null else level)
    for level, deaths in itertools.product((-1, 0, 1), repeat=2):
        _, before, after, result = run('StandardMode.CanResetLevel', {'currentLevel': level, 'currentDiedTimes': deaths})
        assert after == before and result['result_bool'] == (level > 0 or deaths > 0)
    for count, level, mod in itertools.product((0, 1, 7, 2**31-1), (-2, 0, 6, 2**31-1), (-1, 0, 2)):
        _, before, after, result = run('StandardMode.CheckIfLastLevel', {'currentLevel': level}, {'count': count}, (mod,))
        assert after == before and result['result_bool'] == (level >= i32(count-1+mod))
    for count in (0, 1, 7, 2**31-1):
        _, before, after, result = run('StandardMode.MaxLevel', options={'count': count})
        assert after == before and result['result_i32'] == count-1
    for name in ('StandardMode.UpdateScore', 'StandardMode.CheckIfLastLevel', 'StandardMode.MaxLevel'):
        for missing in ('project', 'game', 'array'):
            _, before, after, result = run(name, options={'missing': missing}, args=(200, 3) if name.endswith('UpdateScore') else ())
            assert after == before and result['error'] == 'null'
    for current in (0, 1):
        _, before, after, result = run('StandardMode.CheckIfCompleted', {'currentCompleted': current, 'completed': 1-current})
        assert after == before and result['result_bool'] == bool(current)
    for name, key, expected in [('GetGameMode', None, 0), ('GetCurrentLevel', 'currentLevel', 2), ('GetScore', 'score', 100), ('IsLocked', None, 0)]:
        _, before, after, result = run('StandardMode.'+name)
        assert after == before
        if name == 'IsLocked': assert result['result_bool'] is False
        else: assert result['result_i32'] == expected
    for name in ('SavesGame.set_StandardMode', 'StandardMode.Save'):
        _, before, after, result = run(name)
        assert after == before and result['events'] == ['json', 'set_string']
    for value in (-1, 6, 2**31-1):
        _, before, after, result = run('StandardMode.GetScores', options={'virtual_max': value})
        assert after == before and [item[1] for item in state['formats']] == [800, 7, i32(value+1)]
        assert result['events'] == ['format', 'format', 'virtual_max', 'format', 'concat']
    return {'schema_version': 1, 'build_id': BUILD, 'native_case_count': len(cases),
            'target_count': len(targets), 'supplemental_declarations': supplemental,
            'instruction_assertions': len(checks), 'visited_instruction_count': len(visited),
            'native_ranges': [[hex(a), hex(b)] for a, b in ends.items()],
            'literal_values': list(strings.values()), 'cases': cases,
            'scope': 'Native StandardMode progression/scoring/save bodies and three nested SavesGame callers. Metadata is warmed. GameData.IncreaseVillage, mode/UI callbacks, Unity object equality, class initialization, PlayerPrefs, JSON, boxing and string formatting are explicit services. Failures stop at gateways without claiming exception unwinding. No live state was read or modified.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2)+'\n', encoding='utf-8')
    print(f"Verified {report['native_case_count']} StandardMode progression cases")
