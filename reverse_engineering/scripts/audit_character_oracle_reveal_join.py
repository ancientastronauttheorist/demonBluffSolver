"""Execute the Oracle callers joined to their actual RevealOrder callees.

All other game-owned callees and runtime/Unity/TMP/formatting services are
explicitly supplied. No native bytes, rendering or scheduler are reconstructed.
"""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_oracle_presentation import Machine as OracleMachine, TARGETS as ORACLE_TARGETS
from audit_reveal_order_presentation import Machine as RevealVerifier, TARGETS as REVEAL_TARGETS


class Machine(OracleMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        verifier = RevealVerifier(game_root, dumper_root)
        self.reveal_targets = verifier.targets
        self.reveal_services = verifier.service_targets
        self.reveal_ranges = verifier.ranges
        self.reveal_checks = verifier.checks
        self.reveal_instructions = verifier.instructions
        self.instructions.update(self.reveal_instructions)
        self.terminal = {a for a, i in self.reveal_instructions.items() if i.mnemonic == 'int3'}
        self.reveal_active_return = next(i.address + i.size for i in self.reveal_instructions.values()
                                        if i.mnemonic == 'call' and i.op_str == '0x1c7d810')
        self.hide_return = next(i.address + i.size for i in self.oracle_instructions.values()
                                if i.mnemonic == 'call' and i.op_str == '0x3a7190')
        for i, name in enumerate(['reveal_text', 'other_reveal_text', 'reveal_text_class', 'other_reveal_class',
                                  'reveal_game', 'formatted_order', 'other_formatted_order',
                                  'reveal_text_method', 'other_reveal_method']):
            self.p[name] = self.arena + 0x80000 + i * 0x1000
            self.ids[self.p[name]] = name
        self.reveal_setter, self.other_reveal_setter = self.stop + 0x300, self.stop + 0x310
        self.join_ready = False

    def snapshot(self):
        result = super().snapshot()
        if not getattr(self, 'join_ready', False): return result
        result['reveal_join'] = {
            'text_ref': self.oid(self.rq(self.p['reveal'] + 0x20)),
            'current_method': self.reveal_method,
            'callee_entry_sp_offset': None if self.reveal_sp is None else self.reveal_sp - self.stack,
            'caller_order_slot_bits': None if self.reveal_sp is None else self.rq(self.reveal_sp + 0x10),
            'game_ref': self.oid(self.reveal_game),
            'games': self.games.copy(), 'text_values': self.text_values.copy(),
            'formatted_values': self.formatted_values.copy(),
            'entries': [row.copy() for row in self.reveal_entries],
            'memory': {name: bytes(self.u.mem_read(pointer, self.join_sizes[name])).hex()
                       for name, pointer in self.join_storage.items()}}
        return result

    def prepare(self, options):
        self.join_ready = False
        super().prepare(options)
        names = ['reveal', 'reveal_text', 'other_reveal_text', 'reveal_text_class', 'other_reveal_class',
                 'reveal_game', 'formatted_order', 'other_formatted_order', 'reveal_text_method', 'other_reveal_method',
                 'image_class', 'color_get_method', 'color_set_method']
        self.join_storage = {name: self.p[name] for name in names}
        self.join_storage.update({'metadata_class:' + name: pointer for name, pointer in self.bindings.items()})
        self.join_sizes = {name: (0x600 if 'class' in name and not name.startswith('metadata') else
                                 0x200 if name.startswith('metadata_class:') else 0x80)
                           for name in self.join_storage}
        for name in names[:10]: self.u.mem_write(self.p[name], bytes([0xA5]) * self.join_sizes[name])
        self.q(self.p['reveal'] + 0x20, 0 if options.get('null_reveal_text') else self.p['reveal_text'])
        for name, cls in [('reveal_text', 'reveal_text_class'), ('other_reveal_text', 'other_reveal_class')]:
            self.q(self.p[name], self.p[cls])
        for cls, fn, method in [('reveal_text_class', self.reveal_setter, 'reveal_text_method'),
                                ('other_reveal_class', self.other_reveal_setter, 'other_reveal_method')]:
            self.q(self.p[cls] + 0x558, fn); self.q(self.p[cls] + 0x560, self.p[method])
        if options.get('alias_reveal_text_classes'):
            self.q(self.p['other_reveal_text'], self.p['reveal_text_class'])
        self.reveal_game = self.p[options.get('reveal_game_alias', 'reveal_game')]
        if self.reveal_game == self.p['reveal_game']: self.games['reveal_game'] = options.get('reveal_active', False)
        self.text_values = {'reveal_text': 'old supplied text', 'other_reveal_text': 'other supplied text'}
        self.formatted_values = {}
        self.reveal_sp, self.reveal_method = None, None
        self.reveal_entries = []
        self.join_allowed = {}
        self.join_ready = True

    def join_mutation(self, phase):
        if self.options.get('join_mutation_phase') != phase: return
        action = self.options['join_mutation']
        if action in ['replace_text', 'clear_text']:
            self.q(self.p['reveal'] + 0x20, self.p['other_reveal_text'] if action == 'replace_text' else 0)
            self.join_allowed.setdefault('reveal', set()).update(range(0x20, 0x28))
        elif action == 'replace_text_class':
            self.q(self.p['reveal_text'], self.p['other_reveal_class'])
            self.join_allowed.setdefault('reveal_text', set()).update(range(8))
        elif action == 'replace_saved_order':
            self.d(self.reveal_sp + 0x10, self.options.get('replacement_order_bits', 0xFFFFFFFF))
        elif action == 'actor_order':
            self.d(self.p['actor'] + 0x160, self.options.get('replacement_order_bits', 0xFFFFFFFF))
            self.authored_offsets.update(range(0x160, 0x164))
        elif action == 'count_one':
            self.d(self.p['history'] + 0x18, 1)
            self.memory_mutations.setdefault('history', set()).update(range(0x18, 0x1C))
        else: raise AssertionError(action)

    def ret(self, value=0):
        for name in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']:
            self.u.reg_write(getattr(self.x, 'UC_X86_REG_' + name), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{i}'), (1 << 127) | i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        if not getattr(self, 'join_ready', False): return super().hook(uc, address, size, data)
        rva, x = address - self.base, self.x
        self.executed.add(rva)
        cx, dx, r8 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8']]
        if rva in REVEAL_TARGETS:
            assert cx == self.p['reveal']
            self.reveal_sp = self.reg(x.UC_X86_REG_RSP)
            self.reveal_method = REVEAL_TARGETS[rva][0]
            assert (dx == self.rd(self.p['actor'] + 0x160) and r8 == 0) if self.reveal_method == 'Init' else dx == 0
            self.q(self.reveal_sp + 0x10, 0x11223344AABBCCDD)
            self.reveal_entries.append({'method': self.reveal_method, 'receiver': self.oid(cx),
                                        'order_argument_bits': dx, 'entry_sp_offset': self.reveal_sp - self.stack})
            return
        if rva in self.instructions: return
        if rva == 0x1C79FD0 and cx == self.p['reveal']:
            assert dx == 0
            result = 0 if self.options.get('null_reveal_game') else self.reveal_game
            if self.event('reveal_component_game_object_service', [self.oid(cx), self.oid(result), dx]):
                self.join_mutation('game_object'); self.ret(result)
        elif rva == 0x1C7D810 and cx == self.reveal_game:
            # The return address distinguishes the joined callee's shared Unity call
            # even when its GameObject aliases a later Oracle-owned GameObject.
            ret = self.rq(self.reg(x.UC_X86_REG_RSP)) - self.base
            reveal_call = ret == self.reveal_active_return or (self.reveal_method == 'Hide' and ret == self.hide_return)
            if not reveal_call: return super().hook(uc, address, size, data)
            assert r8 == 0 and dx == (0xFACE123456789001 if self.reveal_method == 'Init' else 0)
            if self.event('reveal_set_active_service', [self.oid(cx), dx, dx & 255, r8]):
                self.games[self.oid(cx)] = bool(dx & 255)
                self.join_mutation('active')
                if self.reveal_method == 'Hide': self.mutation('reveal')
                self.ret()
        elif rva == 0x1117320:
            assert cx == self.reveal_sp + 0x10 and dx == 0
            bits = self.rd(cx); signed = bits if bits < 0x80000000 else bits - 0x100000000
            result = 0 if self.options.get('null_reveal_formatted') else self.p['other_formatted_order' if self.options.get('other_reveal_formatted') else 'formatted_order']
            if self.event('reveal_int32_to_string_service', [cx - self.stack, self.rq(cx), bits, signed, dx, self.oid(result)]):
                if result: self.formatted_values[self.oid(result)] = str(signed)
                self.join_mutation('format'); self.ret(result)
        elif address in [self.reveal_setter, self.other_reveal_setter]:
            assert cx in [self.p['reveal_text'], self.p['other_reveal_text']] and dx in [0, self.p['formatted_order'], self.p['other_formatted_order']]
            cls = self.rq(cx)
            assert address == self.rq(cls + 0x558) and r8 == self.rq(cls + 0x560)
            if self.event('reveal_tmp_text_setter_service', [self.oid(cx), self.oid(dx), self.oid(r8)]):
                self.text_values[self.oid(cx)] = self.formatted_values.get(self.oid(dx))
                self.join_mutation('text'); self.mutation('reveal'); self.ret()
        else: return super().hook(uc, address, size, data)

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.authored_offsets, self.memory_mutations, self.join_allowed = set(), {}, {}
        initial = self.snapshot(); old = len(self.events)
        returned = self.invoke(next(a for a, (n, _, _) in ORACLE_TARGETS.items() if n == name))
        final = self.snapshot(); events = self.events[old:].copy()
        for n, raw in initial['oracle']['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['oracle']['memory'][n])
            allowed = self.authored_offsets if n == 'actor' else self.memory_mutations.get(n, set()) | (set(range(0x40, 0x50)) if name == 'OracleEyeActive' and n in ['acted', 'other_acted'] else set())
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        for n, raw in initial['reveal_join']['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['reveal_join']['memory'][n])
            allowed = self.join_allowed.get(n, set())
            if n.startswith('metadata_class:'): allowed |= set(range(0xE0, 0xE4))
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        if final['reveal_join']['caller_order_slot_bits'] is not None:
            assert final['reveal_join']['caller_order_slot_bits'] >> 32 == 0x11223344
        if not self.options.get('failure') and not self.options.get('join_mutation_phase') and not self.options.get('mutation_phase'):
            suppressed = name == 'OracleEyeActive' and initial['oracle']['state_bits'] == 5
            entries = final['reveal_join']['entries'][len(initial['reveal_join']['entries']):]
            assert len(entries) == int(not suppressed and not self.options.get('null_reveal'))
            if returned and not suppressed:
                if self.reveal_game == self.p['reveal_game']:
                    assert final['oracle']['games'][self.oid(self.reveal_game)] == (name == 'OracleEyeActive')
                if name == 'OracleEyeActive':
                    assert [e['args'][2] for e in events if e['kind'] == 'reveal_int32_to_string_service'] == [initial['oracle']['order_bits']]
            if self.options.get('null_reveal_game') and not suppressed: assert not returned and self.error == 'native_null_guard'
            if self.options.get('null_reveal_text') and name == 'OracleEyeActive' and not suppressed: assert not returned and self.error == 'native_null_guard'
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'initial': initial, 'events': events, 'final': final, 'unconsumed_memory_retained': True,
                'win64_nonvolatile_and_stack_preserved_on_return': returned}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for state, count, uses, picking in itertools.product([5, 10, 20, 30], [0, 1, 2], [0, 1, 0xFFFFFFFF], [0, 0x80]):
        r = m.run('OracleEyeActive', {'state': state, 'count_bits': count, 'uses_bits': uses, 'picking_bits': picking})
        assert r['returned']; cases.append(r)
    for bits in [0, 1, 0x7FFFFFFF, 0x80000000, 0x80000003, 0xFFFFFFFF]:
        for alias in [False, True]:
            r = m.run('OracleEyeActive', {'order_bits': bits, 'alias_reveal_text_classes': alias}); assert r['returned']; cases.append(r)
    for count, active in itertools.product([0, 1, 2, 3], [False, True]):
        r = m.run('HideOracleInfo', {'count_bits': count, 'view_active': active}); assert r['returned']; cases.append(r)
    for name in ['OracleEyeActive', 'HideOracleInfo']:
        for options in [{'cold': True, 'class_cold': True}, {'warm_byte': 0x80, 'class_word': 0xDEADBEEF},
                        {'alias_games': True}, {'reveal_game_alias': 'pick_game'}, {'reveal_game_alias': 'description_game'},
                        {'reveal_game_alias': 'view_game'}, {'null_reveal_formatted': True}, {'other_reveal_formatted': True},
                        {'null_text': True}, {'count_bits': 0x80000000}, {'killed': 0x80}, {'same_data_bluff': True},
                        {'active_return_bits': 0xFACE000000000080}, {'inequality_return_bits': 0xFACE000000000080}]:
            r = m.run(name, options); assert r['returned']; cases.append(r)
        if name == 'OracleEyeActive':
            for field in ['reveal', 'reveal_game', 'reveal_text']:
                r = m.run(name, {'state': 5, 'null_' + field: True})
                assert r['returned'] and not r['final']['reveal_join']['entries']; cases.append(r)
        for options in [{'null_reveal_game': True}, {'null_reveal_text': True}, {'null_reveal': True},
                        {'null_history': True}, {'null_acted': True}, {'null_info': True}, {'null_arrow': True},
                        {'null_pick_game': True}, {'null_view': True}]:
            r = m.run(name, options); assert r['returned'] == (name == 'HideOracleInfo' and options.get('null_reveal_text', False)); cases.append(r)
    for phase in ['game_object', 'active', 'format', 'text']:
        for action in ['replace_text', 'clear_text', 'replace_text_class', 'replace_saved_order', 'actor_order', 'count_one']:
            r = m.run('OracleEyeActive', {'join_mutation_phase': phase, 'join_mutation': action}); cases.append(r)
            assert r['returned'] == (action != 'clear_text' or phase in ['format', 'text'])
            setters = [e for e in r['events'] if e['kind'] == 'reveal_tmp_text_setter_service']
            if setters:
                selected = 'other_reveal_text' if action == 'replace_text' and phase in ['game_object', 'active'] else 'reveal_text'
                assert setters[0]['args'][0] == selected
                if action == 'replace_text_class' and phase != 'text': assert setters[0]['args'][2] == 'other_reveal_method'
            if action == 'replace_saved_order':
                fmt = next(e for e in r['events'] if e['kind'] == 'reveal_int32_to_string_service')
                assert fmt['args'][2] == (0xFFFFFFFF if phase in ['game_object', 'active'] else 0x80000003)
            if action == 'actor_order':
                fmt = next(e for e in r['events'] if e['kind'] == 'reveal_int32_to_string_service')
                assert fmt['args'][2] == 0x80000003
            if action == 'count_one':
                assert not [e for e in r['events'] if e['kind'] == 'supplied_acted_act']
    for phase in ['game_object', 'active']:
        for action in ['replace_text', 'clear_text', 'actor_order', 'count_one']:
            r = m.run('HideOracleInfo', {'join_mutation_phase': phase, 'join_mutation': action})
            assert r['returned']; cases.append(r)
            assert not [e for e in r['events'] if e['kind'].startswith('reveal_int32') or e['kind'].startswith('reveal_tmp')]
    for name, phase, action in [('OracleEyeActive', 'item', 'other_acted'), ('OracleEyeActive', 'act', 'clear_acted'),
                                ('OracleEyeActive', 'active', 'other_acted'), ('OracleEyeActive', 'color_get', 'clear_arrow'),
                                ('OracleEyeActive', 'view_AnimateIn', 'clear_view'), ('OracleEyeActive', 'reveal', 'count_one'),
                                ('HideOracleInfo', 'item', 'other_acted'), ('HideOracleInfo', 'act', 'clear_acted'),
                                ('HideOracleInfo', 'active_self', 'clear_view')]:
        cases.append(m.run(name, {'mutation_phase': phase, 'mutation': action}))
    for alias in [False, True]:
        for game_alias in ['reveal_game', 'pick_game', 'description_game', 'view_game']:
            m.prepare({'cold': True, 'class_cold': True, 'alias_reveal_text_classes': alias, 'reveal_game_alias': game_alias})
            sequences.append([m.run(n, retained=True) for n in ['OracleEyeActive', 'HideOracleInfo', 'OracleEyeActive']])
    profiles = [('OracleEyeActive', {'cold': True, 'class_cold': True}),
                ('HideOracleInfo', {'cold': True, 'class_cold': True}),
                ('OracleEyeActive', {'reveal_game_alias': 'pick_game'}),
                ('HideOracleInfo', {'reveal_game_alias': 'description_game'}),
                ('OracleEyeActive', {'join_mutation_phase': 'format', 'join_mutation': 'replace_text_class'}),
                ('OracleEyeActive', {'null_reveal_formatted': True}),
                ('OracleEyeActive', {'null_reveal_text': True}),
                ('HideOracleInfo', {'null_reveal_game': True})]
    for name, options in profiles:
        baseline = m.run(name, options); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(name, {**options, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index + 1] and stopped['final'] == e['snapshot']
            stops.append(stopped)
    assert not set(m.oracle_instructions) - m.executed
    assert set(m.reveal_instructions) - m.executed == m.terminal
    return {'schema': 'character_oracle_reveal_join_native_v1', 'build': BUILD,
            'scope': 'Actual OracleEyeActive/HideOracleInfo joined to actual RevealOrder.Init/Hide; other game-owned/runtime/Unity/TMP/formatting services supplied; no renderer/scheduler/unwinding. Sentinel memory windows are diagnostic authored storage, not reconstructed complete typed objects.',
            'targets': m.oracle_targets + m.reveal_targets, 'ranges': {**m.oracle_ranges, **m.reveal_ranges},
            'field_pins': m.field_pins + [('RevealOrder', 5735, ['public TextMeshProUGUI text; // 0x20'])],
            'virtual_pin': {'type': 'TMPro.TMP_Text', 'type_def_index': 9110, 'receiver_type_def_index': 8974, 'slot': 66, 'function_offset': '0x558', 'method_info_offset': '0x560'},
            'supplied_metadata': [r for r in m.supplied_metadata if r['Address'] not in REVEAL_TARGETS] + m.reveal_services,
            'instruction_assertions': len(m.checks_oracle) + len(m.reveal_checks), 'color_literal': m.color_literal,
            'oracle_instructions_executed': len(m.oracle_instructions),
            'reveal_nontrap_instructions_executed': len(m.reveal_instructions) - len(m.terminal),
            'terminal_int3_not_executed': [hex(a) for a in sorted(m.terminal)], 'native_execution_addresses': len(m.executed),
            'case_count': len(cases), 'cases': cases, 'retained_sequences': sequences,
            'failure_baselines': baselines, 'failure_case_count': len(stops), 'failures': stops}


def pool_memory(report):
    """Losslessly deduplicate only diagnostic memory windows after all assertions."""
    blobs = {}
    def encode(value):
        if isinstance(value, list): return [encode(item) for item in value]
        if not isinstance(value, dict): return value
        result = {}
        for key, item in value.items():
            if key == 'memory':
                references = {}
                for name, raw in item.items():
                    assert isinstance(raw, str)
                    digest = hashlib.sha256(bytes.fromhex(raw)).hexdigest()
                    assert digest not in blobs or blobs[digest] == raw
                    blobs[digest] = raw
                    references[name] = {'memory_sha256': digest}
                result[key] = references
            else: result[key] = encode(item)
        return result
    result = encode(report)
    result['memory_encoding'] = 'sha256-authored-memory-hex-v1'
    result['memory_blobs'] = blobs
    assert expand_memory(result) == report
    return result


def expand_memory(report):
    """Restore every original byte window and full snapshot from a saved report."""
    assert report['memory_encoding'] == 'sha256-authored-memory-hex-v1'
    blobs = report['memory_blobs']
    for digest, raw in blobs.items(): assert hashlib.sha256(bytes.fromhex(raw)).hexdigest() == digest
    def decode(value):
        if isinstance(value, list): return [decode(item) for item in value]
        if not isinstance(value, dict): return value
        if set(value) == {'memory_sha256'}: return blobs[value['memory_sha256']]
        return {key: decode(item) for key, item in value.items()}
    return decode({key: value for key, value in report.items() if key not in ['memory_encoding', 'memory_blobs']})


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('game_root', type=Path); p.add_argument('dumper_root', type=Path); p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(pool_memory(report), sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['case_count', 'failure_case_count', 'oracle_instructions_executed', 'reveal_nontrap_instructions_executed', 'native_execution_addresses']}))
