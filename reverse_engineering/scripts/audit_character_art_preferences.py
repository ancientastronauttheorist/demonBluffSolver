"""Execute four exact Character art callers; data getters and Unity supplied."""
import argparse
import hashlib
import itertools
import json
import re
from copy import deepcopy
from pathlib import Path

import capstone
from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_oracle_reveal_join import pool_memory

TARGETS = {0x367890: ('ReInitPreferences', 'tdi5487.m0020', 0x367963),
           0x3688B0: ('SetupArt', 'tdi5487.m0021', 0x3689C0),
           0x368B40: ('ShowAnimatedArt', 'tdi5487.m0022', 0x368C13),
           0x365260: ('HideAnimatedArt', 'tdi5487.m0023', 0x365333)}
SERVICES = {0x364C40: 'Character$$GetCharacterBluffIfAble', 0x3B4AB0: 'CharacterData$$GetArt',
            0x3B4990: 'CharacterData$$GetAnimatedArt', 0x3B4A20: 'CharacterData$$GetArtType',
            0x1C822C0: 'UnityEngine.Object$$op_Equality', 0x1C79FD0: 'UnityEngine.Component$$get_gameObject',
            0x1C7D810: 'UnityEngine.GameObject$$SetActive', 0x1D49700: 'UnityEngine.UI.Image$$set_sprite'}
FIELDS = {'art': 0x28, 'clipping': 0x30, 'data': 0x50, 'bluff': 0x58}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root)
        ext = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == ext['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json')); dump = pin('dump.cs', 'dump_cs')
        self.targets, self.instructions, self.bounds = [], {}, {}
        for start, (name, mid, end) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start and r['Name'] == 'Character$$' + name]
            args = 'UnityEngine_Sprite_o* artSprite, int32_t type, ' if name == 'SetupArt' else ''
            assert len(rows) == 1 and rows[0]['Signature'] == f'void Character__{name} (Character_o* __this, {args}const MethodInfo* method);'
            assert rows[0]['TypeSignature'] == ('viiii' if args else 'vii')
            self.targets.append(dict(rows[0], method_id=mid))
            following = min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start)
            section = self.pe.get_section_by_rva(start)
            assert section and following <= section.VirtualAddress + section.SizeOfRawData
            raw = self.pe.get_data(start, following - start)
            assert len(raw) == following - start and raw[end-start:] == bytes([0xCC]) * (following-end)
            ins = list(self.cs.disasm(raw[:end-start], start)); assert sum(i.size for i in ins) == end-start
            self.instructions.update({i.address: i for i in ins})
            self.bounds[hex(start)] = {'end_exclusive': hex(end), 'next_managed': hex(following), 'terminal_trap': hex(end)}
        block = re.search(r'^public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487\s*\{(.*?)// Properties', dump, re.M | re.S)
        assert block and all(f in block[1] for f in ['public Image art; // 0x28', 'public Image clippingArt; // 0x30', 'public CharacterData dataRef; // 0x50', 'public CharacterData bluff; // 0x58'])
        enum = re.search(r'^public enum EArtType // TypeDefIndex: 5946\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert enum and 'public const EArtType Default = 0;' in enum[1] and 'public const EArtType Clipping = 10;' in enum[1]
        self.supplied = []
        service_types = {'Character$$GetCharacterBluffIfAble': 'iii', 'CharacterData$$GetArt': 'iii',
                         'CharacterData$$GetAnimatedArt': 'iii', 'CharacterData$$GetArtType': 'iii',
                         'UnityEngine.Object$$op_Equality': 'iiii', 'UnityEngine.Component$$get_gameObject': 'iii',
                         'UnityEngine.GameObject$$SetActive': 'viii', 'UnityEngine.UI.Image$$set_sprite': 'viii'}
        for a, n in SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == a and r['Name'] == n]
            assert len(rows) == 1 and rows[0]['TypeSignature'] == service_types[n]; self.supplied += rows
        self.flags, self.entry_flags, refs = set(), {}, set()
        for i in self.instructions.values():
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    refs.add(i.address + i.size + op.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
                self.flags.add(i.address + i.size + i.operands[0].mem.disp)
                start = max(a for a in TARGETS if a <= i.address)
                self.entry_flags[TARGETS[start][0]] = i.address + i.size + i.operands[0].mem.disp
        rows = [r for r in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod'] if r['Address'] in refs]
        assert len(rows) == 1 and rows[0]['Name'] == 'UnityEngine.Object_TypeInfo'
        self.slot = self.base + rows[0]['Address']
        names = ['actor', 'art', 'clipping', 'other', 'game0', 'game1', 'game2', 'data', 'bluff', 'other_data', 'normal', 'animated', 'other_sprite', 'object_class', 'component_class', 'data_class', 'sprite_class']
        self.p = {n: self.arena + 0x10000 + i * 0x1000 for i, n in enumerate(names)}; self.ids = {v: k for k, v in self.p.items()}
        self.sizes = {n: 0x200 if n in ['actor', 'object_class'] else 0x180 if n in ['data', 'bluff', 'other_data'] else 0x80 for n in names}
        self.q(self.slot, self.p['object_class'])
        self.checks = {0x3688C6: ('mov', 'esi, r8d'), 0x3688C9: ('mov', 'rdi, rdx'),
            0x368906: ('test', 'al, al'), 0x36890A: ('cmp', 'esi, 0xa'),
            0x36892F: ('mov', 'dl, 1'), 0x368939: ('mov', 'rcx, qword ptr [rbx + 0x28]'),
            0x368945: ('mov', 'rdx, rdi'), 0x368965: ('xor', 'edx, edx'),
            0x368997: ('mov', 'dl, 1'), 0x3689A1: ('mov', 'rcx, qword ptr [rbx + 0x30]'),
            0x3689AD: ('mov', 'rdx, rdi'), 0x3689B9: ('jmp', '0x368951'),
            0x3689BB: ('call', '0x2b7d90')}
        for a, p in [(0x367890, 0x3678DF), (0x365260, 0x3652AF), (0x368B40, 0x368B8F)]:
            self.checks[p] = ('test', 'al, al'); self.checks[p+0x2A] = ('test', 'al, al')
        self.checks.update({0x367948: ('mov', 'rdx, rdi'), 0x367945: ('mov', 'r8d, eax'),
                            0x365318: ('mov', 'rdx, rdi'), 0x365315: ('mov', 'r8d, eax'),
                            0x368BF8: ('mov', 'rdx, rdi'), 0x368BF5: ('mov', 'r8d, eax')})
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())

    def oid(self, p):
        if not p: return None
        assert p in self.ids, hex(p)
        return self.ids[p]

    def snapshot(self):
        return {'fields': {n: self.oid(self.rq(self.p['actor'] + off)) for n, off in FIELDS.items()},
                'component_games': self.component_games.copy(), 'games': self.games.copy(), 'sprites': self.sprites.copy(),
                'liveness': self.live.copy(), 'selected_data': self.selected, 'data_types': self.types.copy(),
                'requests': [r.copy() for r in self.requests],
                'native_entries': [r.copy() for r in self.native_entries],
                'metadata_slot': self.oid(self.rq(self.slot)),
                'metadata_flags': {hex(f): self.u.mem_read(self.base + f, 1).hex() for f in sorted(self.flags)},
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        self.requests, self.native_entries, self.allowed, self.stop_pc = [], [], {}, None
        self.games = {'game0': False, 'game1': True, 'game2': False}; self.sprites = {n: 'other_sprite' for n in ['art', 'clipping', 'other']}
        self.component_games = {'art': 'game0', 'clipping': 'game0' if options.get('alias_games') else 'game1', 'other': 'game2'}
        self.live = {n: True for n in self.p}; self.live.update(options.get('liveness', {}))
        self.selected = options.get('selected_data', 'data'); self.types = {'data': options.get('type_bits', 0), 'bluff': 10, 'other_data': 0xFFFFFFFF}
        for n, p in self.p.items():
            self.u.mem_write(p, bytes([0xA5]) * self.sizes[n]); self.q(p, self.p['component_class']); self.q(p + 8, 0)
        for n, off in FIELDS.items():
            value = None if options.get('null_field') == n else 'art' if n == 'clipping' and options.get('alias_images') else n
            if n == 'bluff' and options.get('alias_data') and value is not None: value = 'data'
            self.q(self.p['actor'] + off, 0 if value is None else self.p[value])
        self.d(self.p['object_class'] + 0xE0, int(options.get('warm', False)))
        for f in self.flags: self.u.mem_write(self.base + f, bytes([int(options.get('warm', False))]))
        self.q(self.slot, self.p['object_class'])

    def mutate(self, phase):
        if self.options.get('mutation_phase') != phase: return
        action = self.options['mutation']
        if action.startswith(('clear_', 'replace_')):
            field = action.partition('_')[2]; off = FIELDS[field]
            target = 'other' if field in ['art', 'clipping'] else 'other_data'
            self.q(self.p['actor'] + off, 0 if action.startswith('clear_') else self.p[target])
            self.allowed.setdefault('actor', set()).update(range(off, off + 8))
        elif action == 'selected_other': self.selected = 'other_data'
        elif action == 'sprite_dead': self.live['normal'] = False
        elif action == 'swap_games': self.component_games['art'], self.component_games['clipping'] = self.component_games['clipping'], self.component_games['art']
        elif action == 'class_cold':
            self.d(self.p['object_class']+0xE0, 0); self.allowed.setdefault('object_class', set()).update(range(0xE0, 0xE4))
        else: raise AssertionError(action)

    def event(self, kind, args):
        result = super().event(kind, args)
        self.events[-1]['raw_args'] = [self.reg(getattr(self.x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        self.events[-1]['caller'] = hex(self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base)
        return result

    def ret(self, value=0):
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']: self.u.reg_write(getattr(self.x, 'UC_X86_REG_' + n), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{i}'), (1 << 127) | i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x; self.executed.add(rva)
        if rva in TARGETS:
            args = [self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
            assert args[0] == self.p['actor']
            if rva == 0x3688B0:
                assert args[1] in [0, self.p['normal'], self.p['animated'], self.p['other_sprite']] and args[3] == 0
                assert args[2] >> 32 in [0, 0xFACE0000]
            else: assert args[1] == 0
            self.native_entries.append({'entry': TARGETS[rva][0], 'raw_args': args})
        if rva in self.instructions: return
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        caller = self.rq(self.reg(x.UC_X86_REG_RSP)) - self.base
        if rva == 0x2B7B40:
            assert cx == self.slot and caller in [0x3678B2, 0x365282, 0x368B62, 0x3688DD]
            if self.event('metadata_service', [cx-self.base, self.oid(self.rq(cx))]): self.mutate('metadata'); self.ret(self.rq(cx))
            return
        if rva == 0x281D90:
            assert cx == self.p['object_class'] and caller in [0x3688F9, 0x3678D2, 0x3678FC, 0x3652A2, 0x3652CC, 0x368B82, 0x368BAC]
            if self.event('class_initialization_service', [self.oid(cx)]): self.d(cx+0xE0, 1); self.allowed.setdefault('object_class', set()).update(range(0xE0, 0xE4)); self.mutate('class_init'); self.ret()
            return
        if rva == 0x2B7D90:
            self.stop_pc = caller - 5
            assert self.stop_pc in [0x36795E, 0x36532E, 0x368C0E, 0x3689BB]
            self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop(); return
        assert rva in SERVICES, hex(rva)
        kind = SERVICES[rva]; phase = None; value = 0
        if rva == 0x1C822C0:
            assert dx == 0 and r8 == 0
            result = 0x1234567800000000 | (self.options.get('equal_true_byte', 0xFE) if not cx or not self.live[self.oid(cx)] else 0)
            args = [self.oid(cx), None, r8, result]; value = result
            phase = 'equality:' + str(self.counts.get(kind, 0) + 1)
        elif rva == 0x364C40:
            assert cx == self.p['actor'] and dx == 0
            args = ['actor', dx, self.selected]; value = 0 if self.selected is None else self.p[self.selected]
            phase = 'appearance:' + str(self.counts.get(kind, 0) + 1)
        elif rva in [0x3B4AB0, 0x3B4990, 0x3B4A20]:
            assert self.oid(cx) in self.types and dx == 0
            value = 0xFACE123400000000 | self.types[self.oid(cx)] if rva == 0x3B4A20 else 0 if self.options.get('null_sprite') else self.p['animated' if rva == 0x3B4990 else 'normal']
            args = [self.oid(cx), dx, value & 0xFFFFFFFF, value] if rva == 0x3B4A20 else [self.oid(cx), dx, self.oid(value)]
            phase = 'type' if rva == 0x3B4A20 else 'sprite'
        elif rva == 0x1C79FD0:
            assert self.oid(cx) in self.component_games and dx == 0
            n = self.component_games[self.oid(cx)]; value = 0 if self.options.get('null_game') == self.oid(cx) else self.p[n]
            args = [self.oid(cx), dx, self.oid(value)]; phase = 'getter:' + str(self.counts.get(kind, 0) + 1)
        elif rva == 0x1C7D810:
            assert self.oid(cx) in self.games and r8 == 0
            assert dx == (0xFACE123456789001 if dx & 255 else 0)
            args = [self.oid(cx), dx, r8]; phase = 'active:' + str(self.counts.get(kind, 0) + 1)
        else:
            assert self.oid(cx) in self.sprites and dx in [0, self.p['normal'], self.p['animated'], self.p['other_sprite']] and r8 == 0
            args = [self.oid(cx), self.oid(dx), r8]; phase = 'sprite_set'
        if self.event(kind, args):
            self.requests.append({'kind': kind, 'args': args, 'caller': hex(caller), 'raw_args': [cx, dx, r8, r9]})
            if rva == 0x1C7D810: self.games[self.oid(cx)] = bool(dx & 255)
            if rva == 0x1D49700: self.sprites[self.oid(cx)] = self.oid(dx)
            self.mutate(phase); self.ret(value)

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error, self.counts, self.allowed, self.stop_pc = options or {}, None, {}, {}, None
        initial, old = self.snapshot(), len(self.events)
        x, sp = self.x, self.stack + 0x18008; self.q(sp, self.stop)
        regs = [getattr(x, 'UC_X86_REG_' + n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        for i, r in enumerate(regs): self.u.reg_write(r, 0xFAB00000+i)
        for i in range(6, 16): self.u.reg_write(getattr(x, f'UC_X86_REG_XMM{i}'), (1 << 125) | i)
        for n, v in [('RSP', sp), ('RCX', self.p['actor']), ('RDX', (0 if self.options.get('null_sprite') else self.p[self.options.get('direct_sprite', 'normal')]) if name == 'SetupArt' else 0), ('R8', 0xFACE000000000000 | self.options.get('type_bits', 0) if name == 'SetupArt' else 0), ('R9', 0)]: self.u.reg_write(getattr(x, 'UC_X86_REG_'+n), v)
        entry_args = [self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        self.u.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        start = next(a for a, (n, _, _) in TARGETS.items() if n == name)
        self.u.emu_start(self.base+start, self.stop, timeout=10000000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp+8
            assert all(self.reg(r) == 0xFAB00000+i for i, r in enumerate(regs))
            assert all(self.reg(getattr(x, f'UC_X86_REG_XMM{i}')) == (1 << 125) | i for i in range(6, 16))
        final = self.snapshot(); events = self.events[old:].copy()
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            assert all(i in self.allowed.get(n, set()) or b == after[i] for i, b in enumerate(before)), n
        assert final['metadata_slot'] == initial['metadata_slot']
        for flag in self.flags:
            before, after = initial['metadata_flags'][hex(flag)], final['metadata_flags'][hex(flag)]
            completed_metadata = [e for i, e in enumerate(events) if e['kind'] == 'metadata_service' and (i+1 < len(events) or self.error != 'metadata_service')]
            reached = any(self.entry_flags[TARGETS[max(a for a in TARGETS if a < int(e['caller'], 16))][0]] == flag for e in completed_metadata)
            assert after == before or (before == '00' and after == '01' and reached)
        row = {'entry': name, 'entry_raw_args': entry_args, 'options': self.options.copy(), 'returned': returned, 'error': self.error, 'stop_pc': None if self.stop_pc is None else hex(self.stop_pc), 'events': events, 'initial': initial, 'final': final, 'normal_abi_verified': returned}
        verify_semantics(row, self.entry_flags)
        return row


def verify_semantics(row, entry_flags):
    """Independent ordered model of supplied services and physical UI writes."""
    if row['options'].get('failure'): return
    s, o, name = deepcopy(row['initial']), row['options'], row['entry']; expected = []; games, sprites = s['games'].copy(), s['sprites'].copy()
    cls = int.from_bytes(bytes.fromhex(s['memory']['object_class'])[0xE0:0xE4], 'little')
    counters = {}
    def mutate(phase):
        nonlocal cls
        if o.get('mutation_phase') != phase: return
        action = o['mutation']
        if action.startswith(('clear_', 'replace_')):
            field = action.partition('_')[2]
            s['fields'][field] = None if action.startswith('clear_') else 'other' if field in ['art', 'clipping'] else 'other_data'
        elif action == 'selected_other': s['selected_data'] = 'other_data'
        elif action == 'sprite_dead': s['liveness']['normal'] = False
        elif action == 'swap_games': s['component_games']['art'], s['component_games']['clipping'] = s['component_games']['clipping'], s['component_games']['art']
        elif action == 'class_cold': cls = 0
        else: raise AssertionError(action)
    def init(body):
        flag = hex(entry_flags[body])
        if s['metadata_flags'][flag] == '00': mutate('metadata'); s['metadata_flags'][flag] = '01'
    def class_init():
        nonlocal cls
        if not cls: cls = 1; mutate('class_init')
    def service(kind, args, phase=None):
        expected.append([kind, args])
        if phase and phase.endswith(':'):
            counters[phase] = counters.get(phase, 0)+1; phase += str(counters[phase])
        if phase: mutate(phase)
    guard = False
    def eq(n):
        result = 0x1234567800000000 | (o.get('equal_true_byte', 0xFE) if n is None or not s['liveness'][n] else 0)
        service('UnityEngine.Object$$op_Equality', [n, None, 0, result], 'equality:')
        return bool(result & 255)
    sprite, typ = (None if o.get('null_sprite') else o.get('direct_sprite', 'normal')), o.get('type_bits', 0) & 0xFFFFFFFF
    init(name)
    if name != 'SetupArt':
        data_captured = s['fields']['data']
        class_init()
        suppressed = eq(data_captured)
        if suppressed:
            bluff_captured = s['fields']['bluff']; class_init(); suppressed = eq(bluff_captured)
        if suppressed: sprite = 'suppressed'
        else:
            data = s['selected_data']; service('Character$$GetCharacterBluffIfAble', ['actor', 0, data], 'appearance:')
            if data is None: guard = True
            else:
                sprite = None if o.get('null_sprite') else 'animated' if name == 'ShowAnimatedArt' else 'normal'
                service('CharacterData$$GetAnimatedArt' if name == 'ShowAnimatedArt' else 'CharacterData$$GetArt', [data, 0, sprite], 'sprite')
                data = s['selected_data']; service('Character$$GetCharacterBluffIfAble', ['actor', 0, data], 'appearance:')
                if data is None: guard = True
                else:
                    typ = s['data_types'][data]; service('CharacterData$$GetArtType', [data, 0, typ, 0xFACE123400000000 | typ], 'type'); init('SetupArt'); class_init()
    else: class_init()
    if not guard and sprite != 'suppressed' and not eq(sprite):
        primary, secondary = ('clipping', 'art') if typ == 10 else ('art', 'clipping')
        for field, on in [(primary, True), (secondary, False)]:
            component = s['fields'][field]
            if component is None: guard = True; break
            game = None if o.get('null_game') == component else s['component_games'][component]
            service('UnityEngine.Component$$get_gameObject', [component, 0, game], 'getter:')
            if game is None: guard = True; break
            # Supplied SetActive applies its effect before the callback.
            games[game] = on
            service('UnityEngine.GameObject$$SetActive', [game, 0xFACE123456789001 if on else 0, 0], 'active:')
            if on:
                component = s['fields'][field]
                if component is None: guard = True; break
                sprites[component] = sprite
                service('UnityEngine.UI.Image$$set_sprite', [component, sprite, 0], 'sprite_set')
    actual = [[e['kind'], e['args']] for e in row['events'] if e['kind'] not in ['metadata_service', 'class_initialization_service', 'native_null_guard']]
    assert actual == expected, (name, o, actual, expected)
    assert row['returned'] == (not guard) and row['final']['games'] == games and row['final']['sprites'] == sprites
    assert row['final']['fields'] == s['fields'] and row['final']['component_games'] == s['component_games']
    assert row['final']['selected_data'] == s['selected_data'] and row['final']['liveness'] == s['liveness']
    assert row['final']['metadata_flags'] == s['metadata_flags']
    assert int.from_bytes(bytes.fromhex(row['final']['memory']['object_class'])[0xE0:0xE4], 'little') == cls
    entries = row['final']['native_entries'][len(row['initial']['native_entries']):]
    assert [e['entry'] for e in entries] == ([name] if name == 'SetupArt' or guard and not any(e[0] == 'CharacterData$$GetArtType' for e in expected) or sprite == 'suppressed' else [name, 'SetupArt'])
    if len(entries) == 2:
        assert entries[1]['raw_args'][1:] == [0 if sprite is None else row['entry_raw_args'][0] + (['actor', 'art', 'clipping', 'other', 'game0', 'game1', 'game2', 'data', 'bluff', 'other_data', 'normal', 'animated', 'other_sprite'].index(sprite))*0x1000, typ, 0]


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); rows = []
    for name in [r[0] for r in TARGETS.values()]:
        for warm, typ, data_live, bluff_live, sprite_live, alias in itertools.product([False, True], [0, 10, 0xFFFFFFFF], [False, True], [False, True], [False, True], [False, True]):
            rows.append(m.run(name, {'warm': warm, 'type_bits': typ, 'liveness': {'data': data_live, 'bluff': bluff_live, 'normal': sprite_live, 'animated': sprite_live}, 'alias_games': alias}))
        for opt in [{'alias_images': True}, {'alias_data': True, 'liveness': {'data': False}}, {'null_sprite': True}, {'selected_data': None}, {'equal_true_byte': 0}, {'equal_true_byte': 0x80}, {'type_bits': 0x8000000A}]: rows.append(m.run(name, opt))
        for field in FIELDS: rows.append(m.run(name, {'null_field': field}))
        for component in ['art', 'clipping']: rows.append(m.run(name, {'null_game': component}))
    for name in ['SetupArt', 'ShowAnimatedArt']:
        for typ in [0, 10]:
            for phase, action in [('metadata', 'replace_art'), ('class_init', 'replace_art'), ('equality:1', 'replace_data'), ('appearance:1', 'selected_other'), ('sprite', 'selected_other'), ('type', 'clear_data'), ('getter:1', 'replace_art'), ('getter:1', 'clear_art'), ('active:1', 'replace_art'), ('active:1', 'clear_art'), ('active:1', 'replace_clipping'), ('sprite_set', 'clear_clipping'), ('sprite_set', 'swap_games')]:
                rows.append(m.run(name, {'type_bits': typ, 'mutation_phase': phase, 'mutation': action}))
    for name in ['ReInitPreferences', 'HideAnimatedArt', 'ShowAnimatedArt']:
        rows.append(m.run(name, {'liveness': {'data': False}, 'mutation_phase': 'equality:1', 'mutation': 'class_cold'}))
        for phase, action in [('class_init', 'clear_data'), ('metadata', 'clear_data'), ('equality:1', 'clear_bluff'), ('appearance:1', 'selected_other'), ('sprite', 'selected_other'), ('getter:1', 'replace_art'), ('active:1', 'replace_art'), ('active:1', 'clear_art'), ('sprite_set', 'clear_clipping')]:
            rows.append(m.run(name, {'liveness': {'data': False} if action == 'clear_bluff' else {}, 'mutation_phase': phase, 'mutation': action}))
    sequences = []
    for options in [{}, {'alias_images': True}, {'alias_games': True}, {'type_bits': 10}, {'alias_data': True, 'liveness': {'data': False}}]:
        sequences.append([m.run('ShowAnimatedArt', options), m.run('HideAnimatedArt', {}, True), m.run('ReInitPreferences', {}, True)])
    baselines, stops = [], []
    for name, options in [('SetupArt', {}), ('SetupArt', {'type_bits': 10}), ('ShowAnimatedArt', {}), ('ShowAnimatedArt', {'type_bits': 10}), ('HideAnimatedArt', {}), ('ReInitPreferences', {}),
                          ('ShowAnimatedArt', {'mutation_phase': 'sprite', 'mutation': 'selected_other'}),
                          ('SetupArt', {'mutation_phase': 'active:1', 'mutation': 'replace_art'}),
                          ('ShowAnimatedArt', {'liveness': {'data': False}, 'mutation_phase': 'equality:1', 'mutation': 'class_cold'})]:
        baseline = m.run(name, options); baselines.append(baseline); counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0)+1
            stopped = m.run(name, dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['error'] == kind
            assert stopped['events'] == baseline['events'][:index+1] and stopped['final'] == event['snapshot']
            stops.append(stopped)
    missing = set(m.instructions) - m.executed
    assert not missing, [hex(a) for a in sorted(missing)]
    result = {'build': BUILD, 'boundary': 'Four actual Character art callers with actual SetupArt join; Character.GetCharacterBluffIfAble, CharacterData getters, runtime and Unity services supplied', 'exact_declarations': m.targets, 'supplied_declarations': m.supplied,
              'bounds': m.bounds, 'operand_checks': {hex(a): list(v) for a, v in m.checks.items()},
              'metadata_slot_rva': hex(m.slot-m.base), 'metadata_flags': [hex(f) for f in sorted(m.flags)],
              'cases_passed': len(rows), 'normal_returns': sum(r['returned'] for r in rows), 'native_guards': sum(not r['returned'] for r in rows),
              'retained_sequences': len(sequences), 'baseline_count': len(baselines), 'stopped_prefixes': len(stops),
              'native_instructions': len(m.instructions), 'observed_addresses': len(m.executed), 'terminal_traps_excluded': [b['terminal_trap'] for b in m.bounds.values()],
              'cases': rows, 'sequences': sequences, 'baselines': baselines, 'stops': stops}
    return pool_memory(result)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('--game-root', required=True); parser.add_argument('--dumper-root', required=True); parser.add_argument('--output', required=True)
    args = parser.parse_args(); result = audit(args.game_root, args.dumper_root)
    Path(args.output).write_text(json.dumps(result, indent=2, sort_keys=True) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['cases_passed', 'native_instructions', 'stopped_prefixes', 'retained_sequences']}))
