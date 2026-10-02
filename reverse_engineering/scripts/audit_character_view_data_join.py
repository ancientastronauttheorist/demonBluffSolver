"""One-state native CharacterView.Init -> CharacterData art -> SetupArt join."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_view_presentation import Machine as ViewMachine
from audit_character_data_consumers import Machine as DataVerifier, TARGETS as DATA_TARGETS
from audit_report_snapshots import pool_snapshots


VIEW_TARGETS = {'Init': (0x363F10, 0x3640EC, 0x3640F0, 3),
                'SetupArt': (0x3643F0, 0x3644B2, 0x3644C0, 6)}
DATA_NAMES = ['GetArt', 'GetArtType']


class Machine(ViewMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        verifier = DataVerifier(game_root, dumper_root)
        self.join_targets, self.join_bounds, self.join_instructions = [], {}, {}
        for name, (start, end, following, ordinal) in VIEW_TARGETS.items():
            row = next(r for r in self.view_targets if r['Name'] == 'CharacterView$$'+name)
            assert self.view_ranges[hex(start)] == [[hex(start), hex(end)]]
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start) == following
            raw = self.pe.get_data(start, following-start)
            assert len(raw) == following-start and raw[end-start:] == b'\xcc'*(following-end)
            assert len([r for r in self.metadata['ScriptMethod'] if r['Address'] == start]) == 1
            self.join_targets.append(dict(row, method_id=f'tdi5511.m{ordinal:04}'))
            self.join_bounds[name] = dict(start=hex(start), end_exclusive=hex(end), next_managed=hex(following), padding_bytes=following-end)
            self.join_instructions.update({a: i for a, i in self.instructions.items() if start <= a < end})
        for name in DATA_NAMES:
            start, end, *_ = DATA_TARGETS[name]
            self.join_targets.append(next(r for r in verifier.targets if r['Name'] == 'CharacterData$$'+name))
            self.join_bounds[name] = verifier.bounds[name]
            self.join_instructions.update({a: i for a, i in verifier.instructions.items() if start <= a < end})
            self.flags.add(verifier.flags[name])
        self.instructions.update(self.join_instructions)
        assert self.metadata_slots[self.base+0x2718BF0] == self.bindings['UnityEngine.Object_TypeInfo']
        self.join_checks = {a: v for a, v in verifier.checks.items() if a in self.join_instructions}
        self.join_checks.update({a: v for a, v in self.view_checks.items() if a in self.join_instructions})
        for a, text in {
            0x36407D: ('call', '0x3b4ab0'), 0x364087: ('mov', 'rbx, rax'),
            0x36408A: ('call', '0x3b4a20'), 0x364095: ('mov', 'rdx, rbx'),
            0x36408F: ('xor', 'r9d, r9d'), 0x3643FA: ('mov', 'rdi, rdx')}.items():
            assert (self.instructions[a].mnemonic, self.instructions[a].op_str) == text
            self.join_checks[a] = text
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == text for a, text in self.join_checks.items())
        calls = [i.op_str for i in self.join_instructions.values() if 0x363F10 <= i.address < 0x3640EC and i.mnemonic == 'call']
        assert calls.count('0x3b4ab0') == calls.count('0x3b4a20') == calls.count('0x3643f0') == 1
        assert not any('0x3b4be0' in c or '0x3b4990' in c for c in calls)
        block = re.search(r'^public class CharacterView : MonoBehaviour // TypeDefIndex: 5511\s*\{(.*?)\n\}', self.dump, re.M|re.S)[1]
        methods = re.findall(r'// RVA: (0x[0-9A-F]+).*?\n\s*([^\n]+)', block)
        assert methods[3] == ('0x363F10', 'public void Init(CharacterData data) { }')
        assert methods[6] == ('0x3643F0', 'private void SetupArt(Sprite artSprite, EArtType type) { }')
        self.data_field_pins = verifier.data_fields
        self.supplied_object_target = next(r for r in verifier.supplied if r['Name'] == 'UnityEngine.Object$$op_Equality')
        for i, n in enumerate(['skin', 'other_skin', 'default_sprite', 'skin_sprite', 'other_sprite']):
            self.p[n] = self.arena+0xB0000+i*0x1000
            self.ids[self.p[n]] = n
        self.storage_sizes = {n: 0x100 for n in self.p}
        self.storage_sizes.update(actor=0x200, view=0x200, data=0x180, bluff=0x180,
                                  image_class=0x400, text_class=0x600)
        self.storage_sizes.update({n: 0x100 for n in self.bindings})
        self.storage_pointers = {**self.p, **self.bindings}
        self.tracking = False
        self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE, self.observe_native_write)
        self.join_ready = True

    def observe_native_write(self, uc, access, address, size, value, user_data):
        if not self.tracking: return
        pc = self.reg(self.x.UC_X86_REG_RIP)-self.base
        if address == self.p['view']+0x28:
            assert pc == 0x363F27 and size == 8 and value in [0, self.p['data'], self.p['bluff']]
            self.completed_writes.setdefault('view', set()).update(range(0x28, 0x30))
        elif address-self.base in self.flags:
            assert pc in self.join_instructions and size == 1 and value == 1
            self.flag_writes[hex(address-self.base)] = 1

    def storage(self):
        return {n: bytes(self.u.mem_read(p, self.storage_sizes[n])).hex() for n, p in self.storage_pointers.items()}

    def snapshot(self):
        out = super().snapshot()
        if getattr(self, 'join_ready', False):
            out.update(physical_storage=self.storage(),
                       metadata_slots={hex(a-self.base): self.oid(self.rq(a)) for a in self.metadata_slots},
                       data_skin_refs={n: self.oid(self.rq(self.p[n]+0xC0)) for n in ['data', 'bluff']},
                       native_data_entries=[dict(v) for v in self.data_entries],
                       native_data_results=[dict(v) for v in self.data_results],
                       supplied_skin_comparisons=[list(v) for v in self.skin_comparisons])
        return out

    def prepare(self, options):
        super().prepare(options)
        for n in ['data', 'bluff', 'skin', 'other_skin', 'default_sprite', 'skin_sprite', 'other_sprite']:
            # Re-author consumed fields after filling diagnostic record windows.
            if n in ['data', 'bluff']:
                self.u.mem_write(self.p[n]+0x118, bytes([0xA5])*0x68)
            else: self.u.mem_write(self.p[n], bytes([0xA5])*self.storage_sizes[n])
        for n in ['data', 'bluff']:
            skin = options.get('initial_skin', 'skin') if n == 'data' else 'other_skin'
            assert skin in [None, 'skin', 'other_skin']
            self.q(self.p[n]+0xC0, self.p[skin] if skin else 0)
            self.q(self.p[n]+0x98, 0 if options.get('null_default_sprite') else self.p['default_sprite'])
            self.q(self.p[n]+0x90, self.p['other_sprite'])
        for n, sprite, bits in [('skin', 'skin_sprite', options.get('art_type_bits', 10)),
                                ('other_skin', 'other_sprite', options.get('other_type_bits', 0))]:
            self.q(self.p[n]+0x38, 0 if options.get('null_skin_sprite') else self.p[sprite])
            self.d(self.p[n]+0x50, bits)
        if options.get('alias_skins'): self.q(self.p['bluff']+0xC0, self.rq(self.p['data']+0xC0))
        if options.get('alias_sprites'):
            for n, off in [('data', 0x98), ('bluff', 0x98), ('skin', 0x38), ('other_skin', 0x38)]:
                self.q(self.p[n]+off, self.p['background_sprite'])
        self.data_entries, self.data_results, self.skin_comparisons, self.data_frames = [], [], [], []

    def capture_changes(self, before):
        for n, raw in before.items():
            after = bytes(self.u.mem_read(self.storage_pointers[n], self.storage_sizes[n]))
            changed = {i for i, b in enumerate(bytes.fromhex(raw)) if b != after[i]}
            if changed: self.completed_writes.setdefault(n, set()).update(changed)

    def mutate_view(self, phase):
        before = self.storage() if self.tracking else None
        super().mutate_view(phase)
        if before is not None: self.capture_changes(before)
        self.mutate_data(phase)

    def mutate_data(self, phase):
        if self.options.get('data_mutation_phase') != phase: return
        if self.options.get('data_mutation_occurrence', 1) != self.phase_counts.get(phase, 0): return
        action = self.options['data_mutation']
        n = self.options.get('mutation_data', 'data'); assert n in ['data', 'bluff']
        skin = self.options.get('mutation_skin', 'skin'); assert skin in ['skin', 'other_skin']
        before = self.storage()
        if action in ['replace_skin', 'clear_skin', 'same_skin']:
            value = 0 if action == 'clear_skin' else self.p['other_skin'] if action == 'replace_skin' else self.rq(self.p[n]+0xC0)
            self.q(self.p[n]+0xC0, value)
        elif action in ['replace_default', 'clear_default']:
            self.q(self.p[n]+0x98, 0 if action == 'clear_default' else self.p['other_sprite'])
        elif action in ['replace_skin_sprite', 'clear_skin_sprite']:
            self.q(self.p[skin]+0x38, 0 if action == 'clear_skin_sprite' else self.p['other_sprite'])
        elif action == 'replace_type': self.d(self.p[skin]+0x50, self.options.get('replacement_type_bits', 0))
        elif action == 'swap_view_data': self.q(self.p['view']+0x28, self.p['bluff'])
        elif action == 'reset_object_class': self.d(self.bindings['UnityEngine.Object_TypeInfo']+0xE0, 0)
        else: raise AssertionError(action)
        self.capture_changes(before)

    def phase(self, name):
        self.phase_counts[name] = self.phase_counts.get(name, 0)+1
        self.mutate_view(name)

    def ret(self, value=0):
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']:
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_'+n), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{i}'), (1<<127)|i)
        super().ret(value)

    def event(self, kind, args):
        result = super().event(kind, args)
        assert self.phase_frames
        entry = self.events[-1]
        entry['raw_args'] = [self.reg(getattr(self.x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        entry['caller'] = hex(self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base)
        entry['native_phase'] = self.phase_frames[-1]['method']
        return result

    def invoke_view(self, address, owner, argument=0, type_bits=0):
        # Full diagnostic entry registers must not inherit the previous fixture's
        # unused volatile bits. Consumed argument setup remains native/parent-owned.
        for n, value in [('R9', self.options.get('unused_entry_r9_bits', 0xCAFEBABE12345678)),
                         ('R10', 0xDEAD123456789010), ('R11', 0xDEAD123456789011)]:
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_'+n), value)
        return super().invoke_view(address, owner, argument, type_bits)

    def hook(self, uc, address, size, data):
        rva, x = address-self.base, self.x
        cx, dx, r8 = [self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8']]
        self.executed.add(rva)
        name = next((n for n in DATA_NAMES if DATA_TARGETS[n][0] == rva), None)
        while self.phase_frames and address == self.phase_frames[-1]['return_address']:
            self.phase_frames.pop()
        phase = next((n for n, (start, *_) in VIEW_TARGETS.items() if start == rva), name)
        if phase:
            self.phase_frames.append(dict(method=phase, return_address=self.rq(self.reg(x.UC_X86_REG_RSP))))
        if name:
            assert cx in [self.p['data'], self.p['bluff']] and dx == 0 and cx == self.reg(x.UC_X86_REG_RSI)
            self.data_entries.append(dict(method=name, data=self.oid(cx), entry_skin_ref=self.oid(self.rq(cx+0xC0))))
            saved_regs = {n: self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in
                          ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15', *[f'XMM{i}' for i in range(6, 16)]]}
            self.data_frames.append(dict(name=name, owner=cx, sp=self.reg(x.UC_X86_REG_RSP),
                                         saved_regs=saved_regs, entry=self.data_entries[-1]))
        if rva in [0x3B4AE0, 0x3B4A50]:
            frame = self.data_frames[-1]
            assert self.reg(x.UC_X86_REG_RBX) == frame['owner']
            frame['captured_skin'] = self.rq(frame['owner']+0xC0)
            frame['entry']['captured_skin'] = self.oid(frame['captured_skin'])
        if rva in self.join_instructions and self.join_instructions[rva].mnemonic == 'ret' and self.data_frames:
            frame = self.data_frames[-1]
            name, owner, sp = [frame[n] for n in ['name', 'owner', 'sp']]
            if DATA_TARGETS[name][0] <= rva < DATA_TARGETS[name][1]:
                assert self.reg(x.UC_X86_REG_RSP) == sp
                assert all(self.reg(getattr(x, 'UC_X86_REG_'+n)) == v for n, v in frame['saved_regs'].items())
                result = self.reg(x.UC_X86_REG_RAX)
                default = self.skin_comparisons[-1][3] & 255 != 0
                skin = self.rq(owner+0xC0)
                expected = (0 if name == 'GetArtType' else self.rq(owner+0x98)) if default else self.rd(skin+0x50) if name == 'GetArtType' else self.rq(skin+0x38)
                assert result == expected
                self.data_results.append(dict(method=name, data=self.oid(owner), result_bits=result,
                                              result_identity=self.oid(result) if name == 'GetArt' else None))
                self.data_frames.pop()
        if rva in self.instructions:
            if rva == 0x3643F0:
                assert self.reg(x.UC_X86_REG_R9) == 0 and r8 < 1<<32
                assert dx == self.data_results[-2]['result_bits'] and r8 == self.data_results[-1]['result_bits']
            super().hook(uc, address, size, data); return
        if rva == 0x1C822C0:
            assert cx in [0, self.p['skin'], self.p['other_skin']] and dx == r8 == 0
            assert cx == self.data_frames[-1]['captured_skin']
            index = len(self.skin_comparisons)
            supplied = self.options.get('comparison_bits')
            result = supplied[index % len(supplied)] if supplied else 0xFACE123456789000 | (0xFE if not cx or not self.options.get('skin_live', True) else 0)
            args = [self.oid(cx), None, r8, result]
            if self.event('skin_object_equality_service', args):
                self.skin_comparisons.append(args); self.phase('comparison'); self.ret(result)
        elif rva == 0x281D90 and cx == self.bindings['UnityEngine.Object_TypeInfo']:
            assert self.rd(cx+0xE0) == 0
            if self.event('object_class_initialize_service', [self.oid(cx)]):
                self.d(cx+0xE0, 1)
                self.completed_writes.setdefault('UnityEngine.Object_TypeInfo', set()).update(range(0xE0, 0xE4))
                self.phase('class_init'); self.ret()
        elif rva == 0x2B7B40:
            assert cx in self.metadata_slots
            if self.event('metadata_service', [cx-self.base]):
                self.phase('metadata'); self.ret(self.metadata_slots[cx])
        elif rva == 0x1D49700:
            assert cx in [self.p[n] for n in self.images] and r8 == 0
            assert dx in [0, *[self.p[n] for n in ['background_sprite', 'default_sprite', 'skin_sprite', 'other_sprite']]]
            if self.event('image_sprite_service', [self.oid(cx), self.oid(dx)]):
                self.images[self.oid(cx)]['sprite'] = self.oid(dx)
                self.phase('sprite:'+self.oid(cx)); self.ret()
        else: super().hook(uc, address, size, data)

    def run_join(self, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.completed_writes, self.flag_writes, self.phase_counts, self.phase_frames = {}, {}, {}, []
        self.view_authored_offsets, self.authored_offsets = set(), set()
        initial, old = self.snapshot(), len(self.events)
        old_results, old_entries, old_comparisons = len(self.data_results), len(self.data_entries), len(self.skin_comparisons)
        self.tracking = True
        try:
            argument = 0 if self.options.get('null_argument') else self.p[self.options.get('argument', 'data')]
            returned = self.invoke_view(0x363F10, self.p['view'], argument)
        finally: self.tracking = False
        final, events = self.snapshot(), self.events[old:].copy()
        for n, raw in initial['physical_storage'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['physical_storage'][n])
            allowed = self.completed_writes.get(n, set())
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        assert final['metadata_slots'] == initial['metadata_slots']
        assert final['metadata_flags'] == {**initial['metadata_flags'], **self.flag_writes}
        if returned:
            entries, results = self.data_entries[old_entries:], self.data_results[old_results:]
            assert [r['method'] for r in entries] == [r['method'] for r in results] == DATA_NAMES
            assert all(r['data'] == self.oid(argument) for r in entries+results)
            setup = self.native_view_entries[-1]
            assert setup['method'] == 'SetupArt' and setup['sprite'] == results[0]['result_identity'] and setup['type_bits'] == results[1]['result_bits']
            selected = 'clipping' if results[1]['result_bits'] == 10 else 'art'
            sprite_event = next(e for e in events if e['kind'] == 'image_sprite_service' and e['args'][1] == results[0]['result_identity'] and e['snapshot']['native_view_entries'][-1]['method'] == 'SetupArt')
            assert sprite_event['args'][0] == self.oid(self.rq(self.p['view']+(0x40 if selected == 'clipping' else 0x38)))
        return dict(method='Init', options=self.options.copy(), returned=returned, error=self.error,
                    initial=initial, events=events, final=final,
                    completed_memory_write_offsets={n: sorted(v) for n, v in self.completed_writes.items()},
                    reached_native_flag_writes=self.flag_writes.copy(),
                    joined_data_entries=self.data_entries[old_entries:].copy(), joined_data_results=self.data_results[old_results:].copy(),
                    joined_comparisons=self.skin_comparisons[old_comparisons:].copy(),
                    unrelated_physical_bytes_retained=True, metadata_slots_retained=True,
                    win64_nonvolatile_and_stack_preserved_on_return=returned)


def expand_memory(report):
    assert report['memory_encoding'] == 'sha256-authored-physical-storage-v1'
    blobs = report['memory_blobs']
    for digest, raw in blobs.items(): assert hashlib.sha256(bytes.fromhex(raw)).hexdigest() == digest
    def decode(v):
        if isinstance(v, list): return [decode(i) for i in v]
        if not isinstance(v, dict): return v
        if set(v) == {'memory_sha256'}: return blobs[v['memory_sha256']]
        return {k: decode(i) for k, i in v.items()}
    return decode({k: v for k, v in report.items() if k not in ['memory_encoding', 'memory_blobs']})


def pool_memory(report):
    blobs = {}
    def encode(v):
        if isinstance(v, list): return [encode(i) for i in v]
        if not isinstance(v, dict): return v
        result = {}
        for k, item in v.items():
            if k == 'physical_storage':
                refs = {}
                for n, raw in item.items():
                    digest = hashlib.sha256(bytes.fromhex(raw)).hexdigest()
                    assert digest not in blobs or blobs[digest] == raw
                    blobs[digest] = raw; refs[n] = {'memory_sha256': digest}
                result[k] = refs
            else: result[k] = encode(item)
        return result
    pooled = encode(report)
    pooled.update(memory_encoding='sha256-authored-physical-storage-v1', memory_blobs=blobs)
    assert expand_memory(pooled) == report
    return pooled


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for skin, live, bits, cold, class_cold in itertools.product([None, 'skin', 'other_skin'], [False, True],
            [0, 10, 20, 0x8000000A, 0xFFFFFFFF], [False, True], [False, True]):
        r = m.run_join(dict(initial_skin=skin, skin_live=live, art_type_bits=bits, cold=cold, class_cold=class_cold))
        assert r['returned']; cases.append(r)
    for options in [dict(null_default_sprite=True), dict(null_skin_sprite=True), dict(alias_sprites=True),
                    dict(alias_skins=True), dict(alias_images=True), dict(alias_borders=True),
                    dict(null_background_sprite=True), dict(null_upper_result=True),
                    dict(border_length_bits=0xFACE000080000000), dict(argument='bluff'),
                    dict(class_cold=True, data_mutation_phase='comparison', data_mutation='reset_object_class'),
                    dict(warm_byte=0xFE, class_word=0xFFFFFFFF, unused_entry_r9_bits=0xFFFFFFFFABCDEF01),
                    dict(comparison_bits=[0xABCD123456789080, 0xABCD123456789000]),
                    dict(comparison_bits=[0xABCD123456789000, 0xABCD123456789001])]:
        cases.append(m.run_join(options))
    for art_byte, type_byte in itertools.product([0, 1, 0x80, 0xFF], repeat=2):
        r = m.run_join(dict(comparison_bits=[0xABCD123456789000|art_byte, 0xFFFF123456789000|type_byte]))
        assert r['returned']; cases.append(r)
    for phase, action, occurrence in itertools.product(['metadata', 'class_init', 'comparison'],
            ['replace_skin', 'clear_skin', 'same_skin', 'replace_default', 'clear_default',
             'replace_skin_sprite', 'clear_skin_sprite', 'replace_type', 'swap_view_data'], [1, 2]):
        cases.append(m.run_join(dict(cold=True, class_cold=True, data_mutation_phase=phase,
                                    data_mutation=action, data_mutation_occurrence=occurrence)))
    for phase, action in [('barrier', 'replace_data_ref'), ('color:border0', 'clear_borders_ref'),
                         ('color:border0', 'clear_second_border'), ('color:border0', 'shrink_borders'),
                         ('uppercase', 'clear_text'), ('set_active:go_clipping', 'clear_clipping')]:
        cases.append(m.run_join(dict(view_mutation_phase=phase, view_mutation=action)))
    for options in [dict(null_argument=True), dict(null_name=True), dict(null_border_index=1),
                    *[{'null_view_'+n: True} for n in ['bg', 'text', 'art', 'clipping', 'bgs', 'borders']],
                    dict(null_game_for='art'), dict(null_game_for='clipping')]:
        cases.append(m.run_join(options))
    for skin in [None, 'skin', 'other_skin']:
        m.prepare(dict(initial_skin=skin, cold=True, class_cold=True))
        rows = [m.run_join(retained=True), m.run_join(dict(argument='bluff'), retained=True), m.run_join(retained=True)]
        assert all(r['returned'] for r in rows); sequences.append(rows)
    m.prepare(dict(cold=True, class_cold=True))
    rows = [m.run_join(dict(data_mutation_phase='class_init', data_mutation='replace_skin'), retained=True),
            m.run_join(dict(data_mutation_phase='comparison', data_mutation='replace_type', replacement_type_bits=10,
                            mutation_skin='other_skin'), retained=True),
            m.run_join(retained=True)]
    assert all(r['returned'] for r in rows); sequences.append(rows)
    profiles = [{}, dict(initial_skin=None, cold=True, class_cold=True), dict(cold=True, class_cold=True),
                dict(cold=True, class_cold=True, data_mutation_phase='class_init', data_mutation='replace_skin'),
                dict(data_mutation_phase='comparison', data_mutation='replace_skin_sprite', data_mutation_occurrence=2),
                dict(alias_images=True, alias_sprites=True), dict(view_mutation_phase='barrier', view_mutation='replace_data_ref'),
                dict(class_cold=True, data_mutation_phase='comparison', data_mutation='reset_object_class'),
                dict(class_cold=True, data_mutation_phase='class_init', data_mutation='clear_skin'),
                dict(view_mutation_phase='set_active:go_clipping', view_mutation='clear_clipping')]
    for options in profiles:
        baseline = m.run_join(options)
        assert baseline['returned'] or baseline['error'] == 'native_null_guard'
        bid = len(baselines); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0)+1
            r = m.run_join({**options, 'failure': [kind, counts[kind]]})
            assert not r['returned'] and r['events'] == baseline['events'][:index+1] and r['final'] == e['snapshot']
            stops.append(dict(baseline=bid, prefix_length=index+1, result=r))
    missing = sorted(set(m.join_instructions)-m.executed)
    nontraps = [a for a in missing if m.join_instructions[a].mnemonic != 'int3']
    # Signed and unsigned array checks are adjacent with no supplied service
    # between them; valid nonnegative iterations cannot reach the second guard.
    assert nontraps == [0x3640E6], [hex(a) for a in missing]
    return dict(schema='character_view_data_join_native_v1', build=BUILD, targets=m.join_targets,
                scope='actual CharacterView.Init/GetArt/GetArtType/SetupArt in one state; engine/TMP/uppercase/runtime services supplied; Oracle/animations excluded',
                body_bounds=m.join_bounds, supplied_object_target=m.supplied_object_target,
                data_field_pins=m.data_field_pins, diagnostic_windows_not_object_extents=m.storage_sizes,
                instruction_assertions=len(m.join_checks), decoded_instructions=len(m.join_instructions),
                covered_instructions=len(set(m.join_instructions)&m.executed),
                unexecuted_terminal_traps=[hex(a) for a in missing if m.join_instructions[a].mnemonic == 'int3'],
                unexecuted_nontrap_guard_instructions=[hex(a) for a in nontraps],
                cases=cases, retained_sequences=sequences, baselines=baselines, failure_stops=stops,
                summary=dict(cases=len(cases), sequences=len(sequences), baselines=len(baselines), stops=len(stops)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--game-root', type=Path, required=True)
    p.add_argument('--dumper-root', type=Path, required=True); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = pool_snapshots(pool_memory(audit(args.game_root, args.dumper_root)))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], sort_keys=True))
