"""Join actual Oracle callers to actual CharacterView bodies, offline.

RevealOrder, Acted, appearance, List, runtime, Unity, DOTween, data art,
uppercase, Image and TMP implementations remain explicitly supplied.
"""
import argparse
import capstone
import itertools
import json
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_oracle_presentation import Machine as OracleMachine, TARGETS as ORACLE_TARGETS
from audit_character_view_presentation import Machine as ViewVerifier, TARGETS as VIEW_TARGETS
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots


class Machine(OracleMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        verifier = ViewVerifier(game_root, dumper_root)
        self.view_targets, self.view_ranges, self.view_checks = verifier.view_targets, verifier.view_ranges, verifier.view_checks
        self.view_literals, self.view_data_targets = verifier.literals, verifier.data_targets
        self.view_addresses = verifier.view_addresses
        self.instructions.update({a: verifier.instructions[a] for a in self.view_addresses})
        self.view_active_returns = {i.address + i.size for a, i in self.instructions.items()
                                    if a in self.view_addresses and i.mnemonic == 'call' and i.op_str == '0x1c7d810'}
        self.setup_return = next(i.address + i.size for a, i in self.instructions.items()
                                 if a in self.view_addresses and i.mnemonic == 'call' and i.op_str == '0x3643f0')
        self.animation_flags = {}
        for start in [0x363D50, 0x363DF0]:
            end = int(self.view_ranges[hex(start)][0][1], 16)
            gates = [i for a, i in self.instructions.items() if start <= a < end and i.mnemonic == 'cmp'
                     and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1
                     and i.operands[0].mem.base == capstone.x86.X86_REG_RIP]
            assert len(gates) == 1
            i = gates[0]; self.animation_flags[start] = i.address + i.size + i.operands[0].mem.disp
        self.flags.update(verifier.flags)
        for name in verifier.bindings:
            if name not in self.bindings:
                self.bindings[name] = self.arena + 0x200000 + len(self.bindings) * 0x1000
                self.ids[self.bindings[name]] = name
        for slot, old in verifier.metadata_slots.items():
            name = verifier.oid(old); self.metadata_slots[slot] = self.bindings[name]; self.q(slot, self.bindings[name])
        self.set_id_method = verifier.set_id_method
        for i, name in enumerate(['bg', 'bgs', 'border0', 'border1', 'art', 'clipping', 'text', 'canvas', 'borders',
                                  'anim_id', 'replacement_anim_id', 'background_sprite', 'art_sprite', 'name',
                                  'upper_name', 'image_class', 'text_class', 'color_method', 'text_method', 'go_art', 'go_clipping']):
            key = 'view_' + name
            self.p[key] = self.arena + 0x100000 + i * 0x1000; self.ids[self.p[key]] = key
        self.view_color_gateway, self.view_text_gateway = self.stop + 0x400, self.stop + 0x410
        self.view_ready = False

    def snapshot(self):
        result = super().snapshot()
        if not getattr(self, 'view_ready', False): return result
        result.pop('supplied_view_state')
        view = self.p['view']
        result['view_join'] = {
            'fields': {n: self.oid(self.rq(view + off)) for n, off in
                       [('bg', 0x20), ('data', 0x28), ('text', 0x30), ('art', 0x38), ('clipping', 0x40),
                        ('bgs', 0x48), ('borders', 0x50), ('canvas', 0x58), ('anim_id', 0x60)]},
            'border_length_bits': self.rq(self.p['view_borders'] + 0x18),
            'border_slots': [self.oid(self.rq(self.p['view_borders'] + 0x20 + i * 8)) for i in range(3)],
            'images': {n: dict(v, color_bits=v['color_bits'].copy()) for n, v in self.view_images.items()},
            'text_value': self.view_text_value, 'games': self.games.copy(),
            'tween_requests': [r.copy() for r in self.tween_requests],
            'kill_requests': [r.copy() for r in self.kill_requests],
            'set_id_requests': [r.copy() for r in self.set_id_requests],
            'entries': [r.copy() for r in self.native_view_entries],
            'data_requests': [r.copy() for r in self.data_requests],
            'dotween_initialized': self.rd(self.bindings['DG.Tweening.DOTween_TypeInfo'] + 0xE0),
            'memory': {n: bytes(self.u.mem_read(p, self.extra_sizes[n])).hex() for n, p in self.extra_storage.items()}}
        return result

    def prepare(self, options):
        self.view_ready = False
        super().prepare(options)
        # Parent's Oracle colors are lists. View images are records with colors/sprites.
        self.oracle_images = self.images
        self.view_images = {'view_' + n: {'color_bits': [0xDEADBEEF] * 4, 'sprite': None}
                            for n in ['bg', 'bgs', 'border0', 'border1', 'art', 'clipping']}
        self.extra_storage = {n: p for n, p in self.p.items() if n.startswith('view_') and n != 'view_game'}
        self.extra_storage.update({'oracle_record:' + n: self.p[n] for n in
                                   ['description_game', 'pick_game', 'view_game', 'image_class', 'color_get_method', 'color_set_method']})
        self.extra_storage.update({'metadata_record:' + n: p for n, p in self.bindings.items()})
        self.extra_sizes = {n: 0x600 if n in ['view_image_class', 'view_text_class', 'oracle_record:image_class'] else
                           0x200 if n.startswith('metadata_record:') else 0x80 for n in self.extra_storage}
        for n, p in self.extra_storage.items():
            if n.startswith('view_'): self.u.mem_write(p, bytes([0xA5]) * self.extra_sizes[n])
        view = self.p['view']
        for off, name in [(0x20, 'bg'), (0x28, 'data'), (0x30, 'text'), (0x38, 'art'), (0x40, 'clipping'),
                          (0x48, 'bgs'), (0x50, 'borders'), (0x58, 'canvas'), (0x60, 'anim_id')]:
            key = name if name == 'data' else 'view_' + name
            value = 0 if options.get('null_view_' + name) else self.p[key]
            if options.get('alias_view_images') and name in ['bg', 'bgs', 'art', 'clipping']: value = self.p['view_art']
            self.q(view + off, value)
        self.q(self.p['view_image_class'] + 0x2A8, self.view_color_gateway)
        self.q(self.p['view_image_class'] + 0x2B0, self.p['view_color_method'])
        self.q(self.p['view_text_class'] + 0x558, self.view_text_gateway)
        self.q(self.p['view_text_class'] + 0x560, self.p['view_text_method'])
        for n in self.view_images: self.q(self.p[n], self.p['view_image_class'])
        self.q(self.p['view_text'], self.p['view_text_class'])
        self.q(self.p['view_borders'] + 0x18, options.get('border_length_bits', 0xFACE000000000002))
        slots = ['view_border0', 'view_border0' if options.get('alias_view_borders') else 'view_border1', 'view_art']
        for i, n in enumerate(slots): self.q(self.p['view_borders'] + 0x20 + i * 8, 0 if options.get('null_border_index') == i else self.p[n])
        for n, colors in [('data', [0x3F000001, 0x80000000, 0x7FC01234, 0x3F800000]),
                          ('bluff', [0x3E000001, 0x3F000002, 0x3F000003, 0x3F000004])]:
            for i, bits in enumerate(colors): self.d(self.p[n] + 0xF8 + i * 4, bits)
            for i, bits in enumerate(reversed(colors)): self.d(self.p[n] + 0x108 + i * 4, bits)
            self.q(self.p[n] + 0xB8, 0 if options.get('null_background_sprite') else self.p['view_background_sprite'])
            self.q(self.p[n] + 0x28, 0 if options.get('null_view_name') else self.p['view_name'])
        self.games.update({'view_go_art': False, 'view_go_clipping': True})
        self.view_game_map = {'view_art': 'view_go_art', 'view_clipping': 'view_go_clipping'}
        if options.get('alias_view_games'): self.view_game_map['view_clipping'] = 'view_go_art'
        for n in self.view_game_map:
            if options.get('view_game_alias') in ['pick_game', 'description_game', 'view_game']:
                self.view_game_map[n] = options['view_game_alias']
        self.tween_requests, self.kill_requests, self.set_id_requests, self.native_view_entries, self.data_requests = [], [], [], [], []
        self.view_text_value, self.fade_count = 'old supplied text', 0
        self.view_allowed, self.extra_allowed = set(), {}
        self.view_ready = True

    def mutate_view(self, phase):
        if self.options.get('view_mutation_phase') != phase: return
        action = self.options['view_mutation']
        fields = {'clear_art': (0x38, 0), 'clear_clipping': (0x40, 0), 'clear_text': (0x30, 0),
                  'clear_borders_ref': (0x50, 0), 'replace_anim_id': (0x60, self.p['view_replacement_anim_id']),
                  'replace_data_ref': (0x28, self.p['data'] if self.rq(self.p['view'] + 0x28) == self.p['bluff'] else self.p['bluff'])}
        if action in fields:
            off, value = fields[action]; self.q(self.p['view'] + off, value); self.view_allowed.update(range(off, off + 8))
        elif action == 'clear_second_border':
            self.q(self.p['view_borders'] + 0x28, 0); self.extra_allowed.setdefault('view_borders', set()).update(range(0x28, 0x30))
        elif action == 'shrink_borders':
            self.q(self.p['view_borders'] + 0x18, 1); self.extra_allowed.setdefault('view_borders', set()).update(range(0x18, 0x20))
        elif action in ['clear_actor_view', 'actor_bluff_to_data']:
            off = 0x140 if action == 'clear_actor_view' else 0x58
            self.q(self.p['actor'] + off, 0 if off == 0x140 else self.p['data']); self.authored_offsets.update(range(off, off + 8))
        else: raise AssertionError(action)

    def ret(self, value=0):
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']:
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_' + n), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{i}'), (1 << 127) | i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        if not getattr(self, 'view_ready', False): return super().hook(uc, address, size, data)
        rva, x = address - self.base, self.x; self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva in VIEW_TARGETS:
            assert cx == self.p['view']; row = {'method': VIEW_TARGETS[rva], 'receiver': self.oid(cx)}
            if rva in self.animation_flags:
                row.update({'entry_rdx_bits': dx, 'metadata_flag_bits': self.byte(self.base + self.animation_flags[rva]),
                            'dotween_initialized_bits': self.rd(self.bindings['DG.Tweening.DOTween_TypeInfo'] + 0xE0)})
                cold = row['metadata_flag_bits'] == 0 or row['dotween_initialized_bits'] == 0
                self.expected_kill_rdx = ((0xFACE123456789090 if cold else dx) & ~255) | 1
            if rva == 0x363F10: row['data'] = self.oid(dx)
            if rva == 0x3643F0: row.update({'sprite': self.oid(dx), 'type_bits': r8 & 0xFFFFFFFF})
            self.native_view_entries.append(row)
        if rva in self.instructions: return
        if rva == 0x281D90 and cx == self.bindings['DG.Tweening.DOTween_TypeInfo']:
            if self.event('class_initialization_service', [self.oid(cx)]): self.d(cx + 0xE0, 1); self.mutate_view('class_init'); self.ret()
        elif rva == 0x5044D0:
            assert cx in [0, self.p['view_anim_id'], self.p['view_replacement_anim_id']] and dx == self.expected_kill_rdx and r8 == 0
            if self.event('view_dotween_kill_service', [self.oid(cx), dx, r8]):
                self.kill_requests.append([self.oid(cx), dx, r8]); self.mutate_view('kill'); self.ret(0xFACE000000000007)
        elif rva == 0x349E80:
            assert cx in [0, self.p['view_canvas']] and r9 == 0
            end, duration = [self.reg(getattr(x, 'UC_X86_REG_XMM' + str(i))) & 0xFFFFFFFF for i in [1, 2]]
            assert end in [0, 0x3F800000] and duration == 0x3E4CCCCD
            if self.event('view_dofade_service', [self.oid(cx), end, duration]):
                self.tween_requests.append([self.oid(cx), end, duration]); self.mutate_view('dofade')
                p = 0 if self.options.get('null_view_tween') else self.arena + 0x180000 + self.fade_count * 0x100
                if p: self.ids[p] = 'view_tween' + str(self.fade_count)
                self.fade_count += 1; self.ret(p)
        elif rva == 0x6BC9D0:
            assert cx == 0 or self.ids.get(cx, '').startswith('view_tween')
            assert dx in [0, self.p['view_anim_id'], self.p['view_replacement_anim_id']] and r8 == self.bindings[self.set_id_method]
            if self.event('view_set_id_service', [self.oid(cx), self.oid(dx), self.oid(r8)]):
                self.set_id_requests.append([self.oid(cx), self.oid(dx), self.oid(r8)])
                self.mutation('view_' + self.native_view_entries[-1]['method']); self.ret(cx)
        elif rva == 0x2B6FF0:
            assert cx == self.p['view'] + 0x28 and self.rq(cx) == dx and dx in [0, self.p['data'], self.p['bluff']]
            if self.event('view_data_barrier_service', [cx - self.arena, self.oid(dx)]): self.mutate_view('barrier'); self.ret()
        elif address == self.view_color_gateway:
            assert cx in [self.p[n] for n in self.view_images] and r8 == self.p['view_color_method']
            bits = list(struct.unpack('<IIII', uc.mem_read(dx, 16)))
            ret = self.rq(self.reg(x.UC_X86_REG_RSP)) - self.base
            captured = self.reg(x.UC_X86_REG_RSI); assert captured in [self.p['data'], self.p['bluff']]
            if ret in [0x363F6E, 0x363FC0]:
                off = 0xF8 if ret == 0x363F6E else 0x108
                assert bits == [self.rd(captured + off + i * 4) for i in range(4)]
            else: assert ret in [0x364049, 0x364078, 0x3640CB] and bits == [0x3F800000] * 4
            if self.event('view_image_color_service', [self.oid(cx), bits, self.oid(r8), self.oid(captured)]):
                self.view_images[self.oid(cx)]['color_bits'] = bits; self.mutate_view('color:' + self.oid(cx))
                if ret == 0x3640CB: self.mutation('view_Init')
                self.ret()
        elif rva == 0x1D49700:
            assert cx in [self.p[n] for n in self.view_images] and dx in [0, self.p['view_background_sprite'], self.p['view_art_sprite']] and r8 == 0
            if self.event('view_image_sprite_service', [self.oid(cx), self.oid(dx)]):
                self.view_images[self.oid(cx)]['sprite'] = self.oid(dx); self.mutate_view('sprite:' + self.oid(cx)); self.ret()
        elif rva == 0xF7B1B0:
            assert cx == self.p['view_name'] and dx == 0
            result = 0 if self.options.get('null_view_upper_result') else self.p['view_upper_name']
            if self.event('view_uppercase_service', [self.oid(cx), self.oid(result)]): self.mutate_view('uppercase'); self.ret(result)
        elif address == self.view_text_gateway:
            assert cx == self.p['view_text'] and dx in [0, self.p['view_upper_name']] and r8 == self.p['view_text_method']
            if self.event('view_text_setter_service', [self.oid(cx), self.oid(dx), self.oid(r8)]): self.view_text_value = self.oid(dx); self.ret()
        elif rva in [0x3B4AB0, 0x3B4A20]:
            assert cx in [self.p['data'], self.p['bluff']] and dx == 0 and cx == self.reg(x.UC_X86_REG_RSI)
            kind = 'view_get_art_service' if rva == 0x3B4AB0 else 'view_get_art_type_service'
            if self.event(kind, [self.oid(cx)]):
                self.data_requests.append([kind, self.oid(cx)]); self.mutate_view(kind)
                self.ret((0 if self.options.get('null_view_art_sprite') else self.p['view_art_sprite']) if rva == 0x3B4AB0 else
                         (0xFACE000000000000 | self.options.get('art_type_bits', 0)))
        elif rva == 0x1C79FD0 and cx in [self.p['view_art'], self.p['view_clipping']]:
            assert dx == 0; name = self.oid(cx)
            result = 0 if self.options.get('null_view_game_for') == name else self.p[self.view_game_map[name]]
            if self.event('view_image_game_object_service', [name, self.oid(result)]): self.mutate_view('game_object:' + name); self.ret(result)
        elif rva == 0x1C7D810:
            ret = self.rq(self.reg(x.UC_X86_REG_RSP)) - self.base
            # SetupArt's tail calls retain Init's return location; other calls have
            # their own decoded return location. This also handles physical GO aliases.
            if ret not in self.view_active_returns and ret != self.setup_return: return super().hook(uc, address, size, data)
            assert cx in [self.p[n] for n in self.games] and dx in [0, 0xFACE123456789001] and r8 == 0
            if self.event('view_image_set_active_service', [self.oid(cx), dx, dx & 255]):
                self.games[self.oid(cx)] = bool(dx & 255); self.mutate_view('set_active:' + self.oid(cx)); self.ret()
        elif rva == 0x2B7D80:
            self.event('native_bounds_guard', []); self.error = 'native_bounds_guard'; uc.emu_stop()
        else: return super().hook(uc, address, size, data)

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.authored_offsets, self.memory_mutations, self.view_allowed, self.extra_allowed = set(), {}, set(), {}
        initial, old = self.snapshot(), len(self.events)
        returned = self.invoke(next(a for a, (n, _, _) in ORACLE_TARGETS.items() if n == name))
        final, events = self.snapshot(), self.events[old:].copy()
        completed_initializers = {e['args'][0] for i, e in enumerate(events)
                                  if e['kind'] == 'class_initialization_service'
                                  and (i + 1 < len(events) or self.error != 'class_initialization_service')}
        for n, raw in initial['oracle']['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['oracle']['memory'][n])
            allowed = self.authored_offsets if n == 'actor' else self.memory_mutations.get(n, set()) | (set(range(0x40, 0x50)) if name == 'OracleEyeActive' and n in ['acted', 'other_acted'] else set())
            if n == 'view':
                allowed |= self.view_allowed
                if any(e['kind'] == 'view_data_barrier_service' for e in events): allowed |= set(range(0x28, 0x30))
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        for n, raw in initial['view_join']['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['view_join']['memory'][n])
            allowed = self.extra_allowed.get(n, set())
            if n.startswith('metadata_record:') and n.partition(':')[2] in completed_initializers:
                allowed |= set(range(0xE0, 0xE4))
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        if returned and not self.options.get('mutation_phase') and not self.options.get('view_mutation_phase'):
            entries = final['view_join']['entries'][len(initial['view_join']['entries']):]
            if name == 'OracleEyeActive':
                want = initial['oracle']['state_bits'] == 20 and initial['actor']['killed_by_demon_bits'] == 0 and self.options.get('inequality_return_bits', 1) & 255 != 0 and initial['actor']['bluff'] is not None and not self.options.get('destroyed_bluff')
                assert [r['method'] for r in entries] == (['AnimateIn', 'Init', 'SetupArt'] if want else [])
                if want:
                    data = initial['actor']['bluff']; assert entries[1]['data'] == data
                    colors = [e['args'][0] for e in events if e['kind'] == 'view_image_color_service']
                    raw_count = initial['view_join']['border_length_bits'] & 0xFFFFFFFF
                    count = raw_count if raw_count < 0x80000000 else 0
                    assert colors == [initial['view_join']['fields']['bgs'], *initial['view_join']['border_slots'][:count], initial['view_join']['fields']['art'], initial['view_join']['fields']['clipping'], initial['view_join']['fields']['bg']]
                    assert self.tween_requests[-1][1:] == [0x3F800000, 0x3E4CCCCD]
            else:
                active = self.options.get('active_return_bits', int(initial['oracle']['games']['view_game'])) & 255
                assert [r['method'] for r in entries] == (['AnimateOut'] if active else [])
                if active: assert self.tween_requests[-1][1:] == [0, 0x3E4CCCCD]
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'initial': initial, 'events': events, 'final': final, 'retained_unconsumed_memory': True,
                'win64_nonvolatile_and_stack_preserved_on_return': returned}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for state, count, killed, active in itertools.product([5, 10, 20, 30], [0, 1, 2, 3], [0, 0x80], [False, True]):
        for name in ['OracleEyeActive', 'HideOracleInfo']:
            r = m.run(name, {'state': state, 'count_bits': count, 'killed': killed, 'view_active': active}); assert r['returned']; cases.append(r)
    for art_type, count, alias in itertools.product([0, 10, 20, 0x8000000A, 0xFFFFFFFF], [0, 1, 2, 3], [False, True]):
        r = m.run('OracleEyeActive', {'art_type_bits': art_type, 'border_length_bits': 0xFACE000000000000 | count, 'alias_view_images': alias, 'alias_view_borders': alias}); assert r['returned']; cases.append(r)
    for uses, picking in itertools.product([0, 1, 0xFFFFFFFF], [0, 0x80]):
        r = m.run('OracleEyeActive', {'uses_bits': uses, 'picking_bits': picking}); assert r['returned']; cases.append(r)
    for options in [{'cold': True, 'class_cold': True}, {'warm_byte': 0x80, 'class_word': 0xDEADBEEF}, {'bluff': 'absent'}, {'destroyed_bluff': True}, {'same_data_bluff': True}, {'null_view_anim_id': True}, {'null_view_canvas': True}, {'null_view_tween': True}, {'null_background_sprite': True}, {'null_view_upper_result': True}, {'null_view_art_sprite': True}, {'alias_view_games': True}, {'view_game_alias': 'pick_game'}, {'view_game_alias': 'description_game'}, {'view_game_alias': 'view_game'}, {'border_length_bits': 0xFACE000080000000}, {'active_return_bits': 0xFACE000000000080}, {'inequality_return_bits': 0xFACE000000000080}]:
        for name in ['OracleEyeActive', 'HideOracleInfo']:
            r = m.run(name, options); assert r['returned']; cases.append(r)
    for field in ['bg', 'text', 'art', 'clipping', 'bgs', 'borders']:
        r = m.run('OracleEyeActive', {'null_view_' + field: True}); assert not r['returned']; cases.append(r)
    for options in [{'null_view_name': True}, {'null_border_index': 0}, {'null_border_index': 1}, {'null_view_game_for': 'view_art'}, {'null_view_game_for': 'view_clipping'}, {'null_view': True}]:
        r = m.run('OracleEyeActive', options); assert not r['returned']; cases.append(r)
    for name in ['OracleEyeActive', 'HideOracleInfo']:
        for field in ['reveal', 'history', 'acted', 'info', 'description_game', 'arrow', 'pick_game']:
            r = m.run(name, {'null_' + field: True}); assert not r['returned']; cases.append(r)
    r = m.run('HideOracleInfo', {'null_view_game': True}); assert not r['returned']; cases.append(r)
    for phase, action in [('class_init', 'replace_anim_id'), ('kill', 'replace_anim_id'), ('dofade', 'replace_anim_id'), ('kill', 'clear_actor_view'), ('kill', 'actor_bluff_to_data'), ('barrier', 'replace_data_ref'), ('color:view_bgs', 'clear_borders_ref'), ('color:view_border0', 'clear_borders_ref'), ('color:view_border0', 'clear_second_border'), ('color:view_border0', 'shrink_borders'), ('uppercase', 'clear_text'), ('view_get_art_service', 'clear_art'), ('set_active:view_go_art', 'clear_art')]:
        r = m.run('OracleEyeActive', {'class_cold': True, 'view_mutation_phase': phase, 'view_mutation': action})
        expected_return = action in ['replace_anim_id', 'actor_bluff_to_data', 'replace_data_ref', 'shrink_borders'] or (phase == 'color:view_border0' and action == 'clear_borders_ref') or phase == 'uppercase'
        assert r['returned'] == expected_return, (phase, action, r['error'])
        if action == 'replace_anim_id':
            assert r['final']['view_join']['kill_requests'][0][0] == 'view_anim_id'
            assert r['final']['view_join']['set_id_requests'][0][1] == 'view_replacement_anim_id'
        if action == 'clear_actor_view':
            assert [e['method'] for e in r['final']['view_join']['entries']] == ['AnimateIn']
            assert r['error'] == 'native_null_guard' and r['final']['view_join']['set_id_requests']
        if action == 'actor_bluff_to_data':
            assert r['final']['view_join']['entries'][1]['data'] == 'data'
            assert r['final']['view_join']['data_requests'] == [['view_get_art_service', 'data'], ['view_get_art_type_service', 'data']]
        if action == 'replace_data_ref':
            assert r['final']['view_join']['fields']['data'] == 'data'
            assert r['final']['view_join']['data_requests'] == [['view_get_art_service', 'bluff'], ['view_get_art_type_service', 'bluff']]
        if phase == 'uppercase':
            assert r['final']['view_join']['fields']['text'] is None
            assert [e['args'][0] for e in r['events'] if e['kind'] == 'view_text_setter_service'] == ['view_text']
        if action == 'shrink_borders':
            assert [e['args'][0] for e in r['events'] if e['kind'] == 'view_image_color_service'] == ['view_bgs', 'view_border0', 'view_art', 'view_clipping', 'view_bg']
        if not expected_return: assert r['error'] == 'native_null_guard'
        cases.append(r)
    for phase in ['class_init', 'kill', 'dofade']:
        r = m.run('HideOracleInfo', {'class_cold': True, 'view_mutation_phase': phase, 'view_mutation': 'replace_anim_id'})
        assert r['returned'] and r['final']['view_join']['kill_requests'][0][0] == 'view_anim_id'
        assert r['final']['view_join']['set_id_requests'][0][1] == 'view_replacement_anim_id'; cases.append(r)
    for alias in [False, True]:
        for game_alias in [None, 'pick_game', 'view_game']:
            m.prepare({'cold': True, 'class_cold': True, 'alias_view_images': alias, 'alias_view_borders': alias, 'alias_view_games': alias, **({'view_game_alias': game_alias} if game_alias else {})})
            sequences.append([m.run(n, retained=True) for n in ['OracleEyeActive', 'HideOracleInfo', 'OracleEyeActive']])
    for name, options in [('OracleEyeActive', {'cold': True, 'class_cold': True}), ('HideOracleInfo', {'cold': True, 'class_cold': True}), ('OracleEyeActive', {'art_type_bits': 10}), ('OracleEyeActive', {'alias_view_images': True, 'alias_view_borders': True, 'alias_view_games': True}), ('OracleEyeActive', {'view_mutation_phase': 'uppercase', 'view_mutation': 'clear_text'}), ('OracleEyeActive', {'view_game_alias': 'view_game'})]:
        base = m.run(name, options); baselines.append(base); counts = {}
        for index, e in enumerate(base['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(name, {**options, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == base['events'][:index + 1] and stopped['final'] == e['snapshot']; stops.append(stopped)
    missing = m.view_addresses - m.executed
    assert not set(m.oracle_instructions) - m.executed, [hex(a) for a in sorted(set(m.oracle_instructions) - m.executed)]
    assert all(m.instructions[a].mnemonic in ['int3', 'call'] and (m.instructions[a].mnemonic == 'int3' or m.instructions[a].op_str == '0x2b7d80') for a in missing)
    return {'schema': 'character_oracle_view_join_native_v1', 'build': BUILD,
            'scope': 'Actual Oracle callers and actual CharacterView AnimateIn/Out/Init/SetupArt. RevealOrder, Acted, appearance, List and runtime/Unity/DOTween/data art/uppercase/Image/TMP effects supplied. No renderer, scheduler, real animation completion or native unwinding. Memory windows are authored diagnostics, not complete typed objects.',
            'targets': m.oracle_targets + m.view_targets, 'ranges': {**m.oracle_ranges, **m.view_ranges},
            'instruction_assertions': len(m.checks_oracle) + len(m.view_checks), 'color_literal': m.color_literal,
            'view_literals': m.view_literals, 'supplied_targets': [r for r in m.supplied_metadata if r['Address'] not in VIEW_TARGETS] + m.view_data_targets,
            'oracle_instructions_executed': len(m.oracle_instructions), 'view_instructions_decoded': len(m.view_addresses), 'view_instructions_executed': len(m.view_addresses & m.executed), 'unexecuted_bounds_and_traps': [hex(a) for a in sorted(missing)], 'addresses': len(m.executed),
            'cases': cases, 'sequences': sequences, 'baselines': baselines, 'stops': stops,
            'summary': {'cases': len(cases), 'sequences': len(sequences), 'stops': len(stops), 'oracle_instructions': len(m.oracle_instructions), 'view_instructions': len(m.view_addresses & m.executed), 'addresses': len(m.executed)}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('game_root', type=Path); p.add_argument('dumper_root', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(pool_snapshots(pool_memory(report)), sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(report['summary']))
