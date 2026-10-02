"""Actual Oracle callers joined to immediate Acted.Act; other services supplied."""
import argparse
import capstone
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_oracle_presentation import Machine as OracleMachine, TARGETS as ORACLE_TARGETS
from audit_acted_surface import audit as verify_acted_surface
from audit_character_oracle_reveal_join import pool_memory

ACT_START, ACT_END = 0x35DD10, 0x35DDB4


class Machine(OracleMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        verified = verify_acted_surface(game_root, dumper_root)
        self.frozen_verifier_counts = {'cases': verified['cases_passed'], 'instructions': verified['distinct_native_instructions']}
        rows = [r for r in verified['exact_declarations'] if r['Name'] == 'Acted$$Act' and r['Address'] == ACT_START]
        assert len(rows) == 1 and rows[0]['TypeSignature'] == 'viii'; self.acted_target = rows[0]
        following = min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > ACT_START)
        section = self.pe.get_section_by_rva(ACT_START)
        assert section and following <= section.VirtualAddress + section.SizeOfRawData
        raw = self.pe.get_data(ACT_START, following - ACT_START)
        assert len(raw) == following - ACT_START and raw[ACT_END - ACT_START:] == bytes([0xCC]) * (following - ACT_END)
        rows = list(self.cs.disasm(raw[:ACT_END - ACT_START], ACT_START)); assert sum(i.size for i in rows) == ACT_END - ACT_START
        self.act_instructions = {i.address: i for i in rows}; self.instructions.update(self.act_instructions)
        self.act_checks = {0x35DD21: ('mov', 'rbx, rdx'), 0x35DD3C: ('mov', 'rcx, qword ptr [rdi + 0x20]'),
                           0x35DD4B: ('call', '0x35d920'), 0x35DD50: ('mov', 'rdi, qword ptr [rdi + 0x28]'),
                           0x35DD62: ('cmp', 'eax, dword ptr [rdi + 0x18]'), 0x35DD65: ('jge', '0x35dd99'),
                           0x35DD67: ('cmp', 'ebx, dword ptr [rdi + 0x18]'), 0x35DD7D: ('mov', 'rsi, qword ptr [rdi + rax*8 + 0x20]'),
                           0x35DD84: ('call', '0x281d90'), 0x35DD89: ('xor', 'edx, edx'),
                           0x35DD8E: ('call', '0x1ec1010'), 0x35DDA8: ('ret', ''),
                           0x35DDA9: ('call', '0x2b7d80'), 0x35DDAF: ('call', '0x2b7d90')}
        assert all((self.act_instructions[a].mnemonic, self.act_instructions[a].op_str) == expected for a, expected in self.act_checks.items())
        references = set()
        for i in rows:
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + op.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
                self.flags.add(i.address + i.size + i.operands[0].mem.disp)
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                name = row['Name']
                if name not in self.bindings: self.bindings[name] = self.arena + 0x200000 + len(self.bindings) * 0x1000
                token = self.bindings[name]; self.ids[token] = name
                self.metadata_slots[self.base + row['Address']] = token; self.q(self.base + row['Address'], token)
        assert 'UnityEngine.UI.LayoutRebuilder_TypeInfo' in self.bindings
        for name, tdi, fields in [('Acted', 5477, ['public ActedVersion acted; // 0x20', 'public RectTransform[] layoutsToRebuild; // 0x28']),
                                   ('ActedVersion', 5478, ['public EActedSizeVersion size; // 0x20', 'public TextMeshProUGUI blankText; // 0x28', 'public TextMeshProUGUI text; // 0x30', 'private string animationId; // 0x38', 'private Vector3 savedScale; // 0x40'])]:
            m = re.search(r'^public class ' + name + r' : MonoBehaviour // TypeDefIndex: ' + str(tdi) + r'\s*\{(.*?)// Methods', self.dump, re.M | re.S)
            assert m and all(f in m[1] for f in fields)
        self.services = [r for r in verified['verified_gateways'] if r['Address'] in [0x35D920, 0x1EC1010]]
        assert len(self.services) == 2
        for i, name in enumerate(['version0', 'version1', 'layouts0', 'layouts1', 'rect0', 'rect1', 'rect2',
                                  'blank0', 'blank1', 'text0', 'text1', 'animation0', 'animation1', 'component_class', 'version_class', 'rect_class']):
            key = 'act_' + name; self.p[key] = self.arena + 0x100000 + i * 0x1000; self.ids[self.p[key]] = key
        self.acted_ready = False

    def snapshot(self):
        result = super().snapshot()
        if not getattr(self, 'acted_ready', False): return result
        result['acted_join'] = {
            'fields': {n: {'version': self.oid(self.rq(self.p[n] + 0x20)), 'layouts': self.oid(self.rq(self.p[n] + 0x28))} for n in ['acted', 'other_acted']},
            'versions': {n: {'size_bits': self.rd(self.p[n] + 0x20), 'blank': self.oid(self.rq(self.p[n] + 0x28)),
                             'text': self.oid(self.rq(self.p[n] + 0x30)), 'animation': self.oid(self.rq(self.p[n] + 0x38)),
                             'scale_bits': [self.rd(self.p[n] + 0x40 + 4 * i) for i in range(3)]} for n in ['act_version0', 'act_version1']},
            'layouts': {n: {'length_bits': self.rq(self.p[n] + 0x18), 'slots': [self.oid(self.rq(self.p[n] + 0x20 + 8 * i)) for i in range(3)]} for n in ['act_layouts0', 'act_layouts1']},
            'active_owner': self.oid(self.active_owner), 'captured_layouts': self.oid(self.captured_layouts),
            'native_entries': [r.copy() for r in self.act_entries], 'show_requests': [r.copy() for r in self.show_requests],
            'rebuild_requests': [r.copy() for r in self.rebuild_requests], 'shown_descriptions': self.shown_descriptions.copy(),
            'rebuilt_counts': self.rebuilt_counts.copy(),
            'layout_class_initialized_bits': self.rd(self.bindings['UnityEngine.UI.LayoutRebuilder_TypeInfo'] + 0xE0),
            'memory': {n: bytes(self.u.mem_read(p, self.extra_sizes[n])).hex() for n, p in self.extra_storage.items()}}
        return result

    def prepare(self, options):
        self.acted_ready = False; super().prepare(options)
        self.extra_storage = {n: p for n, p in self.p.items() if n.startswith('act_')}
        self.extra_storage.update({'metadata_record:' + n: p for n, p in self.bindings.items()})
        self.extra_storage.update({'oracle_record:' + n: self.p[n] for n in ['description_game', 'pick_game', 'view_game', 'image_class', 'color_get_method', 'color_set_method']})
        self.extra_sizes = {n: 0x600 if n == 'oracle_record:image_class' else 0x200 if n.startswith('metadata_record:') else 0x80 for n in self.extra_storage}
        for n, p in self.extra_storage.items():
            if n.startswith('act_'): self.u.mem_write(p, bytes([0xA5]) * self.extra_sizes[n])
        for i, n in enumerate(['acted', 'other_acted']):
            self.q(self.p[n] + 0x20, 0 if options.get('null_acted_version') else self.p['act_version0' if i == 0 or options.get('alias_acted_versions') else 'act_version1'])
            self.q(self.p[n] + 0x28, 0 if options.get('null_acted_layouts') else self.p['act_layouts0' if i == 0 or options.get('alias_acted_layouts') else 'act_layouts1'])
        for i, n in enumerate(['act_version0', 'act_version1']):
            p = self.p[n]; self.q(p, self.p['act_version_class']); self.d(p + 0x20, 10 + i)
            self.q(p + 0x28, self.p['act_blank' + str(i)]); self.q(p + 0x30, self.p['act_text' + str(i)])
            self.q(p + 0x38, self.p['act_animation' + str(i)])
            for j, bits in enumerate([0x3F800000, 0x80000000, 0x7FC01234]): self.d(p + 0x40 + 4 * j, bits)
        for n in ['act_blank0', 'act_blank1', 'act_text0', 'act_text1']: self.q(self.p[n], self.p['act_component_class'])
        for n in ['act_rect0', 'act_rect1', 'act_rect2']: self.q(self.p[n], self.p['act_rect_class'])
        self.q(self.p['act_layouts0'] + 0x18, options.get('layout_length_bits', 0xFACE000000000002))
        self.q(self.p['act_layouts1'] + 0x18, 0xFACE000000000001)
        slots = ['act_rect0', 'act_rect0' if options.get('alias_rects') else 'act_rect1', 'act_rect2']
        for i, n in enumerate(slots):
            self.q(self.p['act_layouts0'] + 0x20 + i * 8, 0 if options.get('null_rect_index') == i else self.p[n])
            self.q(self.p['act_layouts1'] + 0x20 + i * 8, self.p[['act_rect2', 'act_rect0', 'act_rect1'][i]])
        self.active_owner, self.captured_layouts = 0, 0
        self.act_entries, self.show_requests, self.rebuild_requests = [], [], []
        self.shown_descriptions = {'act_version0': None, 'act_version1': None}; self.rebuilt_counts = {'act_rect0': 0, 'act_rect1': 0, 'act_rect2': 0, 'null': 0}
        self.extra_allowed = {}; self.acted_ready = True

    def mutate_acted(self, phase):
        if self.options.get('acted_mutation_phase') != phase: return
        action = self.options['acted_mutation']; owner = self.active_owner
        if action in ['replace_version', 'clear_version', 'replace_layouts', 'clear_layouts']:
            off = 0x20 if action.endswith('version') else 0x28
            value = 0 if action.startswith('clear_') else self.p['act_version1'] if off == 0x20 else self.p['act_layouts1']
            self.q(owner + off, value); self.memory_mutations.setdefault(self.oid(owner), set()).update(range(off, off + 8))
        elif action in ['shrink_layouts', 'replace_second_rect', 'replace_first_rect']:
            n = self.oid(self.captured_layouts); assert n in ['act_layouts0', 'act_layouts1']
            off, value = (0x18, 1) if action == 'shrink_layouts' else (0x20 if action == 'replace_first_rect' else 0x28, self.p['act_rect2'])
            self.q(self.captured_layouts + off, value); self.extra_allowed.setdefault(n, set()).update(range(off, off + 8))
        elif action in ['clear_actor_acted', 'replace_actor_acted']:
            self.q(self.p['actor'] + 0xA8, 0 if action.startswith('clear_') else self.p['other_acted']); self.authored_offsets.update(range(0xA8, 0xB0))
        else: raise AssertionError(action)

    def ret(self, value=0):
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']: self.u.reg_write(getattr(self.x, 'UC_X86_REG_' + n), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{i}'), (1 << 127) | i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        if not getattr(self, 'acted_ready', False): return super().hook(uc, address, size, data)
        rva, x = address - self.base, self.x; self.executed.add(rva)
        cx, dx, r8 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8']]
        if rva == ACT_START:
            assert cx in [self.p['acted'], self.p['other_acted']] and dx in [0, self.p['speech0'], self.p['speech1'], self.p['speech2']] and r8 == 0
            self.active_owner, self.captured_layouts = cx, 0; self.act_entries.append({'owner': self.oid(cx), 'description': self.oid(dx), 'method_info': r8})
        if rva == 0x35DD54: self.captured_layouts = self.reg(x.UC_X86_REG_RDI)
        if rva == 0x35DDA8: self.mutation('act')
        if rva in self.instructions: return
        if rva == 0x2B7B40:
            assert cx in self.metadata_slots
            caller = self.rq(self.reg(x.UC_X86_REG_RSP)) - self.base
            if self.event('metadata_service', [cx - self.base]):
                if caller == 0x35DD35: self.mutate_acted('metadata')
                self.ret(self.metadata_slots[cx])
        elif rva == 0x35D920:
            assert cx in [self.p['act_version0'], self.p['act_version1']] and dx in [0, self.p['speech0'], self.p['speech1'], self.p['speech2']] and r8 == 0
            args = [self.oid(cx), self.oid(dx), r8]
            if self.event('acted_version_show_service', args): self.show_requests.append(args); self.shown_descriptions[self.oid(cx)] = self.oid(dx); self.mutate_acted('show'); self.ret()
        elif rva == 0x281D90 and cx == self.bindings['UnityEngine.UI.LayoutRebuilder_TypeInfo']:
            if self.event('class_initialization_service', [self.oid(cx)]): self.d(cx + 0xE0, 1); self.mutate_acted('class_init'); self.ret()
        elif rva == 0x1EC1010:
            assert cx in [0, self.p['act_rect0'], self.p['act_rect1'], self.p['act_rect2']] and dx == 0
            assert self.captured_layouts == self.reg(x.UC_X86_REG_RDI)
            index = self.reg(x.UC_X86_REG_RBX) & 0xFFFFFFFF; assert index < 3
            args = [self.oid(cx), dx, self.oid(self.captured_layouts), index]
            if self.event('layout_rebuild_service', args):
                self.rebuild_requests.append(args); self.rebuilt_counts[self.oid(cx) or 'null'] += 1; self.mutate_acted('rebuild:' + str(index)); self.ret()
        elif rva == 0x2B7D80:
            self.event('native_bounds_guard', []); self.error = 'native_bounds_guard'; uc.emu_stop()
        else: return super().hook(uc, address, size, data)

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.authored_offsets, self.memory_mutations, self.extra_allowed = set(), {}, {}
        initial, old = self.snapshot(), len(self.events)
        returned = self.invoke(next(a for a, (n, _, _) in ORACLE_TARGETS.items() if n == name))
        final, events = self.snapshot(), self.events[old:].copy()
        completed = {e['args'][0] for i, e in enumerate(events) if e['kind'] == 'class_initialization_service' and (i + 1 < len(events) or self.error != 'class_initialization_service')}
        for n, raw in initial['oracle']['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['oracle']['memory'][n])
            allowed = self.authored_offsets if n == 'actor' else self.memory_mutations.get(n, set()) | (set(range(0x40, 0x50)) if name == 'OracleEyeActive' and n in ['acted', 'other_acted'] else set())
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        for n, raw in initial['acted_join']['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['acted_join']['memory'][n])
            allowed = self.extra_allowed.get(n, set())
            if n.startswith('metadata_record:') and n.partition(':')[2] in completed: allowed |= set(range(0xE0, 0xE4))
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        if returned and not self.options.get('acted_mutation_phase') and not self.options.get('mutation_phase'):
            entries = final['acted_join']['native_entries'][len(initial['acted_join']['native_entries']):]
            count = initial['oracle']['count_bits']; signed = count if count < 0x80000000 else count - 0x100000000
            assert len(entries) == int(signed > 1)
            if entries:
                index = 0 if name == 'OracleEyeActive' else signed - 1
                desc = None if self.options.get('null_text') else 'speech' + str(index)
                assert entries == [{'owner': initial['oracle']['acted'], 'description': desc, 'method_info': 0}]
                version = initial['acted_join']['fields'][initial['oracle']['acted']]['version']
                assert [e['args'] for e in events if e['kind'] == 'acted_version_show_service'] == [[version, desc, 0]]
                array = initial['acted_join']['fields'][initial['oracle']['acted']]['layouts']
                low = initial['acted_join']['layouts'][array]['length_bits'] & 0xFFFFFFFF
                size = low if low < 0x80000000 else 0
                assert [e['args'][0] for e in events if e['kind'] == 'layout_rebuild_service'] == initial['acted_join']['layouts'][array]['slots'][:size]
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error, 'initial': initial, 'events': events, 'final': final, 'unconsumed_memory_retained': True, 'win64_nonvolatile_and_stack_preserved_on_return': returned}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for name, state, count, size, alias in itertools.product(['OracleEyeActive', 'HideOracleInfo'], [5, 10, 20, 30], [0, 1, 2, 3], [0, 1, 2, 3], [False, True]):
        r = m.run(name, {'state': state, 'count_bits': count, 'layout_length_bits': 0xFACE000000000000 | size, 'alias_rects': alias}); assert r['returned']; cases.append(r)
    for name in ['OracleEyeActive', 'HideOracleInfo']:
        for options in [{'cold': True, 'class_cold': True}, {'warm_byte': 0x80, 'class_word': 0xDEADBEEF}, {'null_text': True}, {'null_rect_index': 0}, {'null_rect_index': 1}, {'alias_acted_versions': True, 'alias_acted_layouts': True}, {'layout_length_bits': 0xFACE000080000000}, {'null_acted_version': True}, {'null_acted_layouts': True}]:
            r = m.run(name, options)
            assert r['returned'] == (not options.get('null_acted_version') and not options.get('null_acted_layouts'))
            if not r['returned']: assert r['error'] == 'native_null_guard'
            if options.get('null_acted_layouts'): assert len(r['final']['acted_join']['show_requests']) == 1
            cases.append(r)
        for field in ['reveal', 'history', 'acted', 'info', 'description_game', 'arrow', 'pick_game', 'view']:
            r = m.run(name, {'null_' + field: True}); assert not r['returned']; cases.append(r)
    for uses, picking in itertools.product([0, 1, 0xFFFFFFFF], [0, 0x80]):
        r = m.run('OracleEyeActive', {'uses_bits': uses, 'picking_bits': picking}); assert r['returned']; cases.append(r)
    for phase, action in [('metadata', 'replace_version'), ('metadata', 'clear_version'), ('show', 'replace_version'), ('show', 'clear_version'), ('show', 'replace_layouts'), ('show', 'clear_layouts'), ('class_init', 'replace_layouts'), ('rebuild:0', 'replace_layouts'), ('rebuild:0', 'clear_layouts'), ('rebuild:0', 'shrink_layouts'), ('rebuild:0', 'replace_second_rect'), ('class_init', 'replace_first_rect'), ('class_init', 'shrink_layouts'), ('show', 'clear_actor_acted'), ('show', 'replace_actor_acted')]:
        for name in ['OracleEyeActive', 'HideOracleInfo']:
            r = m.run(name, {'cold': phase == 'metadata', 'class_cold': phase == 'class_init', 'acted_mutation_phase': phase, 'acted_mutation': action}); cases.append(r)
            assert r['returned'] == (action not in ['clear_actor_acted'] and not (phase == 'metadata' and action == 'clear_version') and not (phase == 'show' and action == 'clear_layouts'))
            rebuilds = [e['args'][0] for e in r['events'] if e['kind'] == 'layout_rebuild_service']
            if action == 'replace_layouts': assert rebuilds == (['act_rect2'] if phase == 'show' else ['act_rect0', 'act_rect1'])
            if action == 'clear_layouts' and phase == 'rebuild:0': assert rebuilds == ['act_rect0', 'act_rect1']
            if action == 'shrink_layouts': assert rebuilds == ['act_rect0']
            if action == 'replace_second_rect': assert rebuilds == ['act_rect0', 'act_rect2']
            if action == 'replace_first_rect':
                assert r['final']['acted_join']['layouts']['act_layouts0']['slots'][0] == 'act_rect2'
                assert rebuilds == ['act_rect0', 'act_rect1']
            if action == 'replace_version':
                assert r['final']['acted_join']['show_requests'][0][0] == ('act_version1' if phase == 'metadata' else 'act_version0')
                assert r['final']['acted_join']['fields']['acted']['version'] == 'act_version1'
            if action == 'clear_actor_acted':
                assert r['final']['oracle']['acted'] is None and rebuilds == ['act_rect0', 'act_rect1']
            if action == 'replace_actor_acted': assert r['final']['oracle']['acted'] == 'other_acted' and r['final']['acted_join']['native_entries'][0]['owner'] == 'acted'
    for name, phase, action in [('OracleEyeActive', 'item', 'other_acted'), ('OracleEyeActive', 'item', 'clear_acted'), ('OracleEyeActive', 'act', 'other_acted'), ('HideOracleInfo', 'item', 'other_acted'), ('HideOracleInfo', 'act', 'clear_acted')]:
        cases.append(m.run(name, {'mutation_phase': phase, 'mutation': action}))
    for alias in [False, True]:
        m.prepare({'cold': True, 'class_cold': True, 'alias_acted_versions': alias, 'alias_acted_layouts': alias, 'alias_rects': alias})
        sequences.append([m.run(n, retained=True) for n in ['OracleEyeActive', 'HideOracleInfo', 'OracleEyeActive']])
    for name, options in [('OracleEyeActive', {'cold': True, 'class_cold': True}), ('HideOracleInfo', {'cold': True, 'class_cold': True}), ('OracleEyeActive', {'alias_rects': True}), ('OracleEyeActive', {'null_text': True, 'null_rect_index': 1}), ('OracleEyeActive', {'acted_mutation_phase': 'show', 'acted_mutation': 'replace_layouts'}), ('OracleEyeActive', {'acted_mutation_phase': 'rebuild:0', 'acted_mutation': 'replace_second_rect'})]:
        base = m.run(name, options); baselines.append(base); counts = {}
        for index, e in enumerate(base['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0) + 1
            row = m.run(name, {**options, 'failure': [kind, counts[kind]]})
            assert not row['returned'] and row['events'] == base['events'][:index + 1] and row['final'] == e['snapshot']; stops.append(row)
    missing = set(m.act_instructions) - m.executed
    assert missing == {0x35DDA9, 0x35DDAE}
    assert not set(m.oracle_instructions) - m.executed, [hex(a) for a in sorted(set(m.oracle_instructions) - m.executed)]
    return {'schema': 'character_oracle_acted_join_native_v1', 'build': BUILD, 'scope': 'Actual Oracle callers and immediate Acted.Act only. ActedVersion.Show/runtime/layout rebuild supplied. RevealOrder/View/appearance/List/Unity remain named supplied. Diagnostic windows are authored storage, not complete valid managed objects; no renderer/scheduler/unwinding.',
            'targets': m.oracle_targets + [m.acted_target], 'acted_body': {'start': hex(ACT_START), 'end_exclusive': hex(ACT_END)}, 'frozen_verifier_counts': m.frozen_verifier_counts,
            'supplied_targets': [r for r in m.supplied_metadata if r['Address'] != ACT_START] + m.services,
            'instruction_assertions': len(m.checks_oracle) + len(m.act_checks), 'oracle_instructions': len(m.oracle_instructions), 'acted_decoded': len(m.act_instructions), 'acted_executed': len(m.act_instructions) - len(missing), 'unexecuted_bounds_and_trap': [hex(a) for a in sorted(missing)], 'addresses': len(m.executed),
            'cases': cases, 'sequences': sequences, 'baselines': baselines, 'stops': stops,
            'summary': {'cases': len(cases), 'sequences': len(sequences), 'stops': len(stops), 'oracle_instructions': len(m.oracle_instructions), 'acted_instructions': len(m.act_instructions) - len(missing), 'addresses': len(m.executed)}}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('game_root', type=Path); p.add_argument('dumper_root', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(pool_memory(report), sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(report['summary']))
