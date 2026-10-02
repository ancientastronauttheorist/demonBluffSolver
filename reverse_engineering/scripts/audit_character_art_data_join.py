"""Actual Character art callers and CharacterData getters in one physical graph."""
import argparse
import itertools
import json
from copy import deepcopy
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_art_preferences import Machine as ArtMachine, TARGETS, FIELDS, SERVICES
from audit_character_data_consumers import Machine as DataVerifier, TARGETS as DATA_TARGETS
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots

DATA_NAMES = ['GetArt', 'GetAnimatedArt', 'GetArtType']
DATA_RECORDS = ['data', 'bluff', 'other_data']
SKINS = ['skin', 'other_skin', 'third_skin']
SPRITES = ['normal', 'animated', 'other_sprite', 'skin_normal', 'skin_animated', 'other_animated']


class Machine(ArtMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        verifier = DataVerifier(game_root, dumper_root)
        self.data_targets, self.data_bounds, self.data_flag_names = [], {}, {}
        for name in DATA_NAMES:
            start, end, *_ = DATA_TARGETS[name]
            self.data_targets.append(next(r for r in verifier.targets if r['Name'] == 'CharacterData$$'+name))
            self.data_bounds[name] = verifier.bounds[name]
            self.instructions.update({a: i for a, i in verifier.instructions.items() if start <= a < end})
            self.flags.add(verifier.flags[name]); self.entry_flags[name] = verifier.flags[name]
            self.data_flag_names[name] = hex(verifier.flags[name])
        assert self.slot-self.base == 0x2718BF0 and verifier.bindings[0x2718BF0] == ('metadata', 'UnityEngine.Object_TypeInfo')
        self.checks.update({a: v for a, v in verifier.checks.items() if a in self.instructions})
        self.unwind_ranges = {}
        for start, (name, _, end) in TARGETS.items():
            chunks = []
            for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = e
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append((e.struct.BeginAddress, e.struct.EndAddress))
            assert chunks == [(start, end+1)]
            assert self.pe.get_data(end, 1) == b'\xcc'
            trap = list(self.cs.disasm(self.pe.get_data(end, 1), end))
            assert len(trap) == 1 and trap[0].mnemonic == 'int3' and trap[0].size == 1
            self.instructions[end] = trap[0]
            following = int(self.bounds[hex(start)]['next_managed'], 16)
            self.bounds[hex(start)].update(start=hex(start), end_exclusive=hex(end+1),
                                          padding_bytes=following-end-1)
            self.unwind_ranges[name] = [[hex(a), hex(b)] for a, b in chunks]
            assert len([r for r in self.metadata['ScriptMethod'] if r['Address'] == start]) == 1
        for i, n in enumerate([*SKINS, 'skin_normal', 'skin_animated', 'other_animated']):
            self.p[n] = self.arena+0x90000+i*0x1000; self.ids[self.p[n]] = n
            self.sizes[n] = 0x100 if n in SKINS else 0x80
        self.data_field_pins = verifier.data_fields
        self.capture_instructions = {0x3B4AE0: 'GetArt', 0x3B49C0: 'GetAnimatedArt', 0x3B4A50: 'GetArtType'}
        self.tracking = False; self.ready = True
        self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE, self.observe_native_write)

    def observe_native_write(self, uc, access, address, size, value, user_data):
        if not self.tracking: return
        if address-self.base in self.flags:
            pc = self.reg(self.x.UC_X86_REG_RIP)-self.base
            assert pc in self.instructions and self.instructions[pc].mnemonic == 'mov' and size == value == 1
            self.flag_writes[hex(address-self.base)] = '01'

    def snapshot(self):
        out = super().snapshot()
        if getattr(self, 'ready', False):
            out.update(data_fields={n: {'skin': self.oid(self.rq(self.p[n]+0xC0)),
                                       'normal': self.oid(self.rq(self.p[n]+0x98)),
                                       'animated': self.oid(self.rq(self.p[n]+0xA8))} for n in DATA_RECORDS},
                       skin_fields={n: {'normal': self.oid(self.rq(self.p[n]+0x38)),
                                       'animated': self.oid(self.rq(self.p[n]+0x40)),
                                       'type_bits': self.rd(self.p[n]+0x50)} for n in SKINS},
                       native_data_entries=[dict(v) for v in self.data_entries],
                       native_data_results=[dict(v) for v in self.data_results])
        return out

    def prepare(self, options):
        super().prepare(options)
        for n, skin, normal, animated in [('data', options.get('initial_skin', 'skin'), 'normal', 'animated'),
                                        ('bluff', 'other_skin', 'other_sprite', 'other_animated'),
                                        ('other_data', 'third_skin', 'other_sprite', 'other_animated')]:
            assert skin in [None, *SKINS]
            self.q(self.p[n]+0xC0, self.p[skin] if skin else 0)
            self.q(self.p[n]+0x98, 0 if options.get('null_sprite') else self.p[normal])
            self.q(self.p[n]+0xA8, 0 if options.get('null_sprite') else self.p[animated])
            self.q(self.p[n]+0x90, self.p['other_sprite'])
        for n, normal, animated, bits in [('skin', 'skin_normal', 'skin_animated', options.get('type_bits', 0)),
                                         ('other_skin', 'other_sprite', 'other_animated', options.get('other_type_bits', 10)),
                                         ('third_skin', 'normal', 'animated', 0xFFFFFFFF)]:
            self.q(self.p[n]+0x38, 0 if options.get('null_sprite') else self.p[normal])
            self.q(self.p[n]+0x40, 0 if options.get('null_sprite') else self.p[animated])
            self.d(self.p[n]+0x50, bits)
        if options.get('alias_skins'):
            for n in DATA_RECORDS: self.q(self.p[n]+0xC0, self.p['skin'])
        if options.get('alias_sprites'):
            for n in DATA_RECORDS:
                for off in [0x98, 0xA8]: self.q(self.p[n]+off, self.p['normal'])
            for n in SKINS:
                for off in [0x38, 0x40]: self.q(self.p[n]+off, self.p['normal'])
        if options.get('class_word') is not None: self.d(self.p['object_class']+0xE0, options['class_word'])
        if options.get('warm_byte') is not None:
            for f in self.flags: self.u.mem_write(self.base+f, bytes([options['warm_byte']]))
        self.data_entries, self.data_results, self.data_frames = [], [], []

    def mutate(self, phase):
        self.phase_counts[phase] = self.phase_counts.get(phase, 0)+1
        plans = [(self.options.get('mutation_phase'), self.options.get('mutation'), self.options.get('mutation_occurrence', 1))]
        extra = self.options.get('extra_mutation')
        if extra: plans.append((extra['phase'], extra['action'], extra.get('occurrence', 1)))
        for requested, action, occurrence in plans:
            if requested == phase and occurrence == self.phase_counts[phase]: self.apply_mutation(phase, action)

    def apply_mutation(self, phase, action):
        if not action.startswith('data_'):
            options = self.options
            try:
                self.options = {**options, 'mutation_phase': phase, 'mutation': action}
                super().mutate(phase)
            finally: self.options = options
            return
        n = self.options.get('mutation_data', 'data'); skin = self.options.get('mutation_skin', 'skin')
        assert n in DATA_RECORDS and skin in SKINS
        def write(record, off, value, size=8):
            if size == 8: self.q(self.p[record]+off, value)
            else: self.d(self.p[record]+off, value)
            self.allowed.setdefault(record, set()).update(range(off, off+size))
        if action in ['data_replace_skin', 'data_clear_skin', 'data_same_skin']:
            value = 0 if action == 'data_clear_skin' else self.p['other_skin'] if action == 'data_replace_skin' else self.rq(self.p[n]+0xC0)
            write(n, 0xC0, value)
        elif action in ['data_replace_default', 'data_clear_default']:
            for off in [0x98, 0xA8]: write(n, off, 0 if action == 'data_clear_default' else self.p['other_sprite'])
        elif action in ['data_replace_sprite', 'data_clear_sprite']:
            for off in [0x38, 0x40]: write(skin, off, 0 if action == 'data_clear_sprite' else self.p['other_sprite'])
        elif action == 'data_replace_type': write(skin, 0x50, self.options.get('replacement_type_bits', 10), 4)
        else: raise AssertionError(action)

    def event(self, kind, args):
        result = super().event(kind, args)
        assert self.phase_frames
        self.events[-1]['native_phase'] = self.phase_frames[-1]['method']
        return result

    def hook(self, uc, address, size, data):
        rva, x = address-self.base, self.x; self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        while self.phase_frames and address == self.phase_frames[-1]['return_address']: self.phase_frames.pop()
        dname = next((n for n in DATA_NAMES if DATA_TARGETS[n][0] == rva), None)
        name = TARGETS[rva][0] if rva in TARGETS else dname
        if name:
            self.phase_frames.append(dict(method=name, return_address=self.rq(self.reg(x.UC_X86_REG_RSP))))
        if rva in TARGETS:
            assert cx == self.p['actor']
            if name == 'SetupArt':
                assert dx in [0, *[self.p[n] for n in SPRITES]] and r9 == 0 and r8>>32 in [0, 0xFACE0000]
                if len(self.phase_frames) > 1 and self.data_results:
                    assert dx == self.data_results[-2]['result_bits'] and r8 == self.data_results[-1]['result_bits']
            else: assert dx == 0
            self.native_entries.append(dict(entry=name, raw_args=[cx, dx, r8, r9]))
        if dname:
            assert self.oid(cx) in DATA_RECORDS and dx == 0
            row = dict(method=dname, data=self.oid(cx), entry_skin_ref=self.oid(self.rq(cx+0xC0)), raw_args=[cx, dx, r8, r9])
            self.data_entries.append(row)
            saved = {n: self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in
                     ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15', *[f'XMM{i}' for i in range(6, 16)]]}
            self.data_frames.append(dict(method=dname, data=cx, sp=self.reg(x.UC_X86_REG_RSP), saved=saved, row=row))
        if rva in self.capture_instructions:
            frame = self.data_frames[-1]
            assert frame['method'] == self.capture_instructions[rva] and self.reg(x.UC_X86_REG_RBX) == frame['data']
            frame['skin'] = self.rq(frame['data']+0xC0); frame['row']['captured_skin'] = self.oid(frame['skin'])
        if rva in self.instructions:
            if self.instructions[rva].mnemonic == 'ret' and self.data_frames:
                frame = self.data_frames[-1]; name = frame['method']
                if DATA_TARGETS[name][0] <= rva < DATA_TARGETS[name][1]:
                    assert self.reg(x.UC_X86_REG_RSP) == frame['sp']
                    assert all(self.reg(getattr(x, 'UC_X86_REG_'+n)) == v for n, v in frame['saved'].items())
                    result = self.reg(x.UC_X86_REG_RAX)
                    if name == 'GetArtType': assert result < 1<<32
                    else: assert result in [0, *[self.p[n] for n in SPRITES]]
                    self.data_results.append(dict(method=name, data=self.oid(frame['data']), result_bits=result,
                                                  result_identity=self.oid(result) if name != 'GetArtType' else None))
                    self.data_frames.pop()
            return
        caller = self.rq(self.reg(x.UC_X86_REG_RSP))-self.base
        if rva == 0x2B7B40:
            assert cx == self.slot
            if self.event('metadata_service', [cx-self.base, self.oid(self.rq(cx))]): self.mutate('metadata'); self.ret(self.rq(cx))
        elif rva == 0x281D90:
            assert cx == self.p['object_class'] and self.rd(cx+0xE0) == 0
            if self.event('class_initialization_service', [self.oid(cx)]):
                self.d(cx+0xE0, 1); self.allowed.setdefault('object_class', set()).update(range(0xE0, 0xE4))
                self.mutate('class_init'); self.ret()
        elif rva == 0x1C822C0 and self.data_frames:
            frame = self.data_frames[-1]
            assert cx == frame['skin'] and cx in [0, *[self.p[n] for n in SKINS]] and dx == r8 == 0
            supplied = self.options.get('skin_comparison_bits'); index = self.skin_comparison_count
            result = supplied[index % len(supplied)] if supplied else 0xFACE123456789000 | (self.options.get('skin_true_byte', 0xFE) if not cx or not self.live[self.oid(cx)] else 0)
            phase = 'skin_equality:'+str(index+1)
            if self.event('skin_object_equality_service', [self.oid(cx), None, r8, result]):
                self.skin_comparison_count += 1; self.mutate(phase); self.ret(result)
        elif rva == 0x1D49700:
            assert self.oid(cx) in self.sprites and dx in [0, *[self.p[n] for n in SPRITES]] and r8 == 0
            args = [self.oid(cx), self.oid(dx), r8]
            if self.event(SERVICES[rva], args):
                self.requests.append(dict(kind=SERVICES[rva], args=args, caller=hex(caller), raw_args=[cx, dx, r8, r9]))
                self.sprites[self.oid(cx)] = self.oid(dx); self.mutate('sprite_set'); self.ret()
        elif rva == 0x2B7D90:
            self.stop_pc = caller-5
            assert self.stop_pc in [0x36795E, 0x36532E, 0x368C0E, 0x3689BB, 0x3B4B33, 0x3B4A13, 0x3B4A9D]
            self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop()
        else: super().hook(uc, address, size, data)

    def run_join(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error, self.counts, self.allowed, self.stop_pc = options or {}, None, {}, {}, None
        self.phase_counts, self.phase_frames, self.flag_writes, self.skin_comparison_count = {}, [], {}, 0
        initial, old, old_results = self.snapshot(), len(self.events), len(self.data_results)
        x, sp = self.x, self.stack+0x18008; self.q(sp, self.stop)
        regs = [getattr(x, 'UC_X86_REG_'+n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        for i, r in enumerate(regs): self.u.reg_write(r, 0xFAB00000+i)
        for i in range(6, 16): self.u.reg_write(getattr(x, f'UC_X86_REG_XMM{i}'), (1<<125)|i)
        args = [self.p['actor'], (0 if self.options.get('null_sprite') else self.p[self.options.get('direct_sprite', 'normal')]) if name == 'SetupArt' else 0,
                0xFACE000000000000|self.options.get('type_bits', 0) if name == 'SetupArt' else 0, 0]
        for n, v in zip(['RCX', 'RDX', 'R8', 'R9'], args): self.u.reg_write(getattr(x, 'UC_X86_REG_'+n), v)
        for n, v in [('RSP', sp), ('R10', 0xABCD0010), ('R11', 0xABCD0011), ('MXCSR', 0x1F80)]: self.u.reg_write(getattr(x, 'UC_X86_REG_'+n), v)
        start = next(a for a, (n, _, _) in TARGETS.items() if n == name)
        self.tracking = True
        try: self.u.emu_start(self.base+start, self.stop, timeout=10000000, count=100000)
        finally: self.tracking = False
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp+8
            assert all(self.reg(r) == 0xFAB00000+i for i, r in enumerate(regs))
            assert all(self.reg(getattr(x, f'UC_X86_REG_XMM{i}')) == (1<<125)|i for i in range(6, 16))
        final, events = self.snapshot(), self.events[old:].copy()
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            assert all(i in self.allowed.get(n, set()) or b == after[i] for i, b in enumerate(before)), n
        assert final['metadata_slot'] == initial['metadata_slot']
        assert final['metadata_flags'] == {**initial['metadata_flags'], **self.flag_writes}
        row = dict(entry=name, entry_raw_args=args, options=self.options.copy(), returned=returned, error=self.error,
                   stop_pc=hex(self.stop_pc) if self.stop_pc is not None else None, initial=initial, events=events, final=final,
                   joined_data_results=self.data_results[old_results:].copy(), completed_memory_write_offsets={n: sorted(v) for n, v in self.allowed.items()},
                   reached_native_flag_writes=self.flag_writes.copy(), normal_abi_verified=returned,
                   unrelated_memory_and_metadata_slot_retained=True)
        verify_semantics(row, self.entry_flags, self.p)
        return row


def verify_semantics(row, entry_flags, pointers):
    """Independent complete ordered service model, including actual Data selections."""
    if row['options'].get('failure'): return
    s, o, name = deepcopy(row['initial']), row['options'], row['entry']; expected, results = [], []
    raw = {n: bytearray.fromhex(v) for n, v in s['memory'].items()}; inverse = {v: k for k, v in pointers.items()}
    phase_counts, service_counts = {}, {}; skin_count = 0
    def pointer(n): return pointers[n] if n is not None else 0
    def word(n, off): return int.from_bytes(raw[n][off:off+8], 'little')
    def ref(n, off): return inverse[word(n, off)] if word(n, off) else None
    def cls(): return int.from_bytes(raw['object_class'][0xE0:0xE4], 'little')
    def write(n, off, value, size=8): raw[n][off:off+size] = value.to_bytes(size, 'little')
    def mutate(phase):
        phase_counts[phase] = phase_counts.get(phase, 0)+1
        plans = [(o.get('mutation_phase'), o.get('mutation'), o.get('mutation_occurrence', 1))]
        extra = o.get('extra_mutation')
        if extra: plans.append((extra['phase'], extra['action'], extra.get('occurrence', 1)))
        for requested, action, occurrence in plans:
            if requested == phase and occurrence == phase_counts[phase]: apply_mutation(action)
    def apply_mutation(action):
        n = o.get('mutation_data', 'data'); skin = o.get('mutation_skin', 'skin')
        if action.startswith('data_'):
            if action in ['data_replace_skin', 'data_clear_skin', 'data_same_skin']:
                write(n, 0xC0, 0 if action == 'data_clear_skin' else pointers['other_skin'] if action == 'data_replace_skin' else word(n, 0xC0))
            elif action in ['data_replace_default', 'data_clear_default']:
                for off in [0x98, 0xA8]: write(n, off, 0 if action == 'data_clear_default' else pointers['other_sprite'])
            elif action in ['data_replace_sprite', 'data_clear_sprite']:
                for off in [0x38, 0x40]: write(skin, off, 0 if action == 'data_clear_sprite' else pointers['other_sprite'])
            elif action == 'data_replace_type': write(skin, 0x50, o.get('replacement_type_bits', 10), 4)
            else: raise AssertionError(action)
        elif action.startswith(('clear_', 'replace_')):
            field = action.partition('_')[2]
            target = None if action.startswith('clear_') else 'other' if field in ['art', 'clipping'] else 'other_data'
            s['fields'][field] = target; write('actor', FIELDS[field], pointer(target))
        elif action == 'selected_other': s['selected_data'] = 'other_data'
        elif action == 'sprite_dead': s['liveness']['normal'] = False
        elif action == 'swap_games': s['component_games']['art'], s['component_games']['clipping'] = s['component_games']['clipping'], s['component_games']['art']
        elif action == 'class_cold': write('object_class', 0xE0, 0, 4)
        else: raise AssertionError(action)
    def service(kind, args, phase=None):
        expected.append([kind, args])
        if phase: mutate(phase)
    def init(body):
        flag = hex(entry_flags[body])
        if s['metadata_flags'][flag] == '00':
            service('metadata_service', [0x2718BF0, 'object_class'], 'metadata'); s['metadata_flags'][flag] = '01'
    def class_init():
        if cls() == 0:
            expected.append(['class_initialization_service', ['object_class']]); write('object_class', 0xE0, 1, 4); mutate('class_init')
    def equality(n, skin=False):
        nonlocal skin_count
        if skin:
            supplied = o.get('skin_comparison_bits')
            result = supplied[skin_count % len(supplied)] if supplied else 0xFACE123456789000 | (o.get('skin_true_byte', 0xFE) if n is None or not s['liveness'][n] else 0)
            skin_count += 1; service('skin_object_equality_service', [n, None, 0, result], 'skin_equality:'+str(skin_count))
        else:
            kind = 'UnityEngine.Object$$op_Equality'; service_counts[kind] = service_counts.get(kind, 0)+1
            result = 0x1234567800000000 | (o.get('equal_true_byte', 0xFE) if n is None or not s['liveness'][n] else 0)
            service(kind, [n, None, 0, result], 'equality:'+str(service_counts[kind]))
        return result & 255 != 0
    guard = False
    def appearance():
        kind = 'Character$$GetCharacterBluffIfAble'; service_counts[kind] = service_counts.get(kind, 0)+1
        captured = s['selected_data']; service(kind, ['actor', 0, captured], 'appearance:'+str(service_counts[kind])); return captured
    def getter(body, data):
        nonlocal guard
        init(body); captured_skin = ref(data, 0xC0); class_init(); default = equality(captured_skin, True)
        if default: value = 0 if body == 'GetArtType' else word(data, 0x98 if body == 'GetArt' else 0xA8)
        else:
            skin = ref(data, 0xC0)
            if skin is None: guard = True; return None
            off, size = (0x50, 4) if body == 'GetArtType' else (0x38 if body == 'GetArt' else 0x40, 8)
            value = int.from_bytes(raw[skin][off:off+size], 'little')
        results.append(dict(method=body, data=data, result_bits=value,
                            result_identity=inverse[value] if value and body != 'GetArtType' else None))
        return value
    sprite = None if o.get('null_sprite') else o.get('direct_sprite', 'normal'); typ = o.get('type_bits', 0)&0xFFFFFFFF
    init(name); suppressed = False
    if name != 'SetupArt':
        captured = s['fields']['data']; class_init(); suppressed = equality(captured)
        if suppressed:
            captured = s['fields']['bluff']; class_init(); suppressed = equality(captured)
        if not suppressed:
            data = appearance()
            if data is None: guard = True
            else:
                sprite_bits = getter('GetAnimatedArt' if name == 'ShowAnimatedArt' else 'GetArt', data)
                if not guard:
                    sprite = inverse[sprite_bits] if sprite_bits else None; data = appearance()
                    if data is None: guard = True
                    else:
                        typ = getter('GetArtType', data)
                        if not guard: init('SetupArt'); class_init()
    else: class_init()
    if not suppressed and not guard and not equality(sprite):
        primary, secondary = ('clipping', 'art') if typ == 10 else ('art', 'clipping')
        for field, on in [(primary, True), (secondary, False)]:
            component = s['fields'][field]
            if component is None: guard = True; break
            kind = 'UnityEngine.Component$$get_gameObject'; service_counts[kind] = service_counts.get(kind, 0)+1
            game = None if o.get('null_game') == component else s['component_games'][component]
            service(kind, [component, 0, game], 'getter:'+str(service_counts[kind]))
            if game is None: guard = True; break
            kind = 'UnityEngine.GameObject$$SetActive'; service_counts[kind] = service_counts.get(kind, 0)+1
            s['games'][game] = on
            service(kind, [game, 0xFACE123456789001 if on else 0, 0], 'active:'+str(service_counts[kind]))
            if on:
                component = s['fields'][field]
                if component is None: guard = True; break
                s['sprites'][component] = sprite
                service('UnityEngine.UI.Image$$set_sprite', [component, sprite, 0], 'sprite_set')
    if guard: expected.append(['native_null_guard', []])
    actual = [[e['kind'], e['args']] for e in row['events']]
    assert actual == expected, (name, o, actual, expected)
    assert row['returned'] == (not guard) and row['joined_data_results'] == results
    assert row['final']['memory'] == {n: bytes(v).hex() for n, v in raw.items()}
    for key in ['fields', 'games', 'sprites', 'liveness', 'selected_data', 'component_games', 'metadata_flags']:
        assert row['final'][key] == s[key], key
    new_entries = row['final']['native_entries'][len(row['initial']['native_entries']):]
    joined_setup = name != 'SetupArt' and not suppressed and len(results) == 2
    assert [e['entry'] for e in new_entries] == ([name, 'SetupArt'] if joined_setup else [name])
    if joined_setup: assert new_entries[1]['raw_args'][1:] == [pointer(sprite), typ, 0]


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    names = [r[0] for r in TARGETS.values()]
    for name, warm, skin, live, typ, alias in itertools.product(names, [False, True], [None, 'skin'],
            [False, True], [0, 10, 0x8000000A, 0xFFFFFFFF], [False, True]):
        cases.append(m.run_join(name, dict(warm=warm, initial_skin=skin, type_bits=typ,
                                         liveness={'skin': live}, alias_games=alias)))
    for name in names:
        for byte in [0, 1, 0x80, 0xFF]:
            cases.append(m.run_join(name, dict(equal_true_byte=byte, liveness={'data': False, 'normal': False, 'animated': False,
                                                                            'skin_normal': False, 'skin_animated': False})))
        for options in [dict(alias_skins=True), dict(alias_sprites=True), dict(alias_images=True),
                        dict(alias_data=True, liveness={'data': False}), dict(null_sprite=True), dict(selected_data=None),
                        dict(selected_data='bluff'), dict(selected_data='other_data'), dict(warm_byte=0xFE, class_word=0xFFFFFFFF),
                        dict(liveness={'data': False, 'bluff': False}), dict(liveness={'skin_normal': False, 'skin_animated': False}),
                        *[dict(null_field=n) for n in FIELDS], *[dict(null_game=n) for n in ['art', 'clipping']]]:
            cases.append(m.run_join(name, options))
    for name in ['ReInitPreferences', 'ShowAnimatedArt', 'HideAnimatedArt']:
        cases.append(m.run_join(name, dict(liveness={'data': False}, mutation_phase='equality:1', mutation='class_cold')))
        for phase, action in itertools.product(['equality:1', 'skin_equality:1'], ['data_replace_skin', 'data_clear_skin']):
            cases.append(m.run_join(name, dict(mutation_phase=phase, mutation='class_cold',
                extra_mutation=dict(phase='class_init', action=action, occurrence=2))))
        for byte0, byte1 in itertools.product([0, 1, 0x80, 0xFF], repeat=2):
            cases.append(m.run_join(name, dict(skin_comparison_bits=[0xABCD123456789000|byte0, 0xFFFF123456789000|byte1])))
        for phase, action in itertools.product(['metadata', 'class_init', 'skin_equality:1', 'skin_equality:2'],
                ['data_replace_skin', 'data_clear_skin', 'data_same_skin', 'data_replace_default', 'data_clear_default',
                 'data_replace_sprite', 'data_clear_sprite', 'data_replace_type']):
            cases.append(m.run_join(name, dict(mutation_phase=phase, mutation=action)))
        for phase, action in [('skin_equality:1', 'selected_other'), ('skin_equality:2', 'selected_other'),
                             ('appearance:1', 'selected_other'), ('equality:1', 'class_cold'),
                             ('skin_equality:1', 'class_cold'), ('class_init', 'clear_data'),
                             ('metadata', 'clear_data'), ('getter:1', 'replace_art'), ('active:1', 'clear_art'),
                             ('active:1', 'replace_clipping'), ('sprite_set', 'swap_games')]:
            cases.append(m.run_join(name, dict(mutation_phase=phase, mutation=action)))
        cases.append(m.run_join(name, dict(mutation_phase='metadata', mutation='data_replace_skin', mutation_occurrence=2)))
    for options in [{}, dict(alias_images=True), dict(alias_games=True), dict(alias_sprites=True), dict(initial_skin=None),
                    dict(mutation_phase='skin_equality:1', mutation='selected_other')]:
        rows = [m.run_join('ShowAnimatedArt', options), m.run_join('HideAnimatedArt', retained=True), m.run_join('ReInitPreferences', retained=True)]
        assert all(r['returned'] for r in rows); sequences.append(rows)
    profiles = [(n, {}) for n in names] + [(n, dict(type_bits=10)) for n in names]
    profiles += [('ShowAnimatedArt', dict(mutation_phase='skin_equality:1', mutation='selected_other')),
                 ('ShowAnimatedArt', dict(mutation_phase='skin_equality:1', mutation='class_cold')),
                 ('ShowAnimatedArt', dict(liveness={'data': False}, mutation_phase='equality:1', mutation='class_cold')),
                 ('ShowAnimatedArt', dict(mutation_phase='skin_equality:1', mutation='data_clear_skin')),
                 ('ReInitPreferences', dict(mutation_phase='skin_equality:2', mutation='data_clear_skin')),
                 ('ShowAnimatedArt', dict(mutation_phase='equality:1', mutation='class_cold',
                     extra_mutation=dict(phase='class_init', action='data_replace_skin', occurrence=2))),
                 ('ReInitPreferences', dict(mutation_phase='skin_equality:1', mutation='class_cold',
                     extra_mutation=dict(phase='class_init', action='data_clear_skin', occurrence=2))),
                 ('SetupArt', dict(mutation_phase='active:1', mutation='replace_art'))]
    for name, options in profiles:
        baseline = m.run_join(name, options); bid = len(baselines); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0)+1
            stopped = m.run_join(name, {**options, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index+1] and stopped['final'] == e['snapshot']
            stops.append(dict(baseline=bid, prefix_length=index+1, result=stopped))
    missing = sorted(set(m.instructions)-m.executed)
    assert all(m.instructions[a].mnemonic == 'int3' for a in missing), [hex(a) for a in missing]
    return dict(schema='character_art_data_join_native_v1', build=BUILD,
                scope='four exact Character art bodies and three actual CharacterData art consumers; appearance/runtime/Unity services supplied',
                targets=m.targets+m.data_targets, actor_bounds=m.bounds, actor_unwind_ranges=m.unwind_ranges, data_bounds=m.data_bounds,
                supplied_targets=[r for r in m.supplied if r['Name'] not in ['CharacterData$$'+n for n in DATA_NAMES]],
                data_field_pins=m.data_field_pins, diagnostic_windows_not_object_extents=m.sizes,
                metadata_slot_rva=hex(m.slot-m.base), metadata_flag_rvas={n: hex(a) for n, a in m.entry_flags.items()},
                instruction_assertions=len(m.checks), decoded_instructions=len(m.instructions),
                covered_instructions=len(set(m.instructions)&m.executed), unexecuted_terminal_traps=[hex(a) for a in missing],
                cases=cases, retained_sequences=sequences, baselines=baselines, failure_stops=stops,
                summary=dict(cases=len(cases), sequences=len(sequences), baselines=len(baselines), stops=len(stops)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--game-root', required=True); p.add_argument('--dumper-root', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); result = pool_snapshots(pool_memory(audit(args.game_root, args.dumper_root)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(result, sort_keys=True, separators=(',', ':'), ensure_ascii=True)+'\n', encoding='utf-8')
    print(json.dumps(result['summary'], sort_keys=True))
