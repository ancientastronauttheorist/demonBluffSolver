"""Retained actual Manage/pools/Init/publication/Act setup composition.

Concrete role virtual bodies and runtime services are supplied. This distinct
versioned corpus preserves the original pre-publication initializer audit.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_manage_initialization_join import InitializationJoin
from audit_manage_pool_composition import audit as pool_audit


class PublicationActionJoin(InitializationJoin):
    def decode(self, metadata, dump, pe, cs, decoded):
        super().decode(metadata, dump, pe, cs, decoded)
        self.action_methods = []
        self.action_hashes = {}
        for start, name in ((0x3811b0, 'Gameplay$$UpdateCharacters'),
                            (0x3645c0, 'Character$$Act'),
                            (0x368790, 'Character$$RoleAct'),
                            (0x397750, 'CharacterHelper$$CheckLying')):
            rows = [m for m in metadata['ScriptMethod'] if m['Address'] == start and m['Name'] == name]
            assert len(rows) == 1
            self.action_methods.extend(rows)
            end = min(m['Address'] for m in metadata['ScriptMethod'] if m['Address'] > start)
            instructions = list(cs.disasm(pe.get_data(start, end - start), start))
            while instructions[-1].mnemonic == 'int3': instructions.pop()
            assert instructions[0].address == start
            assert all(a.address + a.size == b.address for a, b in zip(instructions, instructions[1:]))
            end = instructions[-1].address + instructions[-1].size
            self.action_hashes[name] = hashlib.sha256(pe.get_data(start, end - start)).hexdigest()
            decoded.update({i.address: i for i in instructions})
        self.action_fields = {
            'Gameplay': ['public static List<Character> CurrentCharacters; // 0x18'],
            'Characters': ['public CharacterData[] startGameActOrder; // 0x28',
                           'public Action onSetup; // 0x58'],
            'Character': ['public Action<Character, ETriggerPhase> onTrigger; // 0x110'],
            'Role': ['public Action<ActedInfo> onActed; // 0x28'],
        }
        for name, fields in self.action_fields.items():
            match = re.search(r'^[^\n]*class ' + re.escape(name)
                              + r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', dump, re.M | re.S)
            assert match and all(field in match[1] for field in fields), name
        self.action_checks = {
            0x36d03d: ('call', '0x3811b0'),
            0x36d0ba: ('call', '0x3645c0'),
            0x36d1dc: ('call', '0x3645c0'),
            0x38120d: ('call', '0xb610a0'),
            0x38123b: ('mov', 'qword ptr [rax + 0x18], rbx'),
            0x38125b: ('jmp', '0x2b6ff0'),
            0x364727: ('mov', 'byte ptr [rbx + 0x11c], 1'),
            0x364733: ('call', '0x397750'),
            0x36476d: ('mov', 'rdx, qword ptr [rbx + 0x170]'),
            0x36884e: ('mov', 'qword ptr [rcx], rbp'),
            0x36886e: ('call', 'qword ptr [rax + 0x258]'),
            0x368896: ('call', 'qword ptr [rax + 0x208]'),
            0x3977dd: ('call', '0x363c40'),
            0x3977fb: ('call', '0x363c40'),
        }
        for address, expected in self.action_checks.items():
            assert (decoded[address].mnemonic, decoded[address].op_str) == expected
        self.action_returns = {decoded[a].address + decoded[a].size for a in (0x36d0ba, 0x36d1dc)}
        self.publication_return = decoded[0x36d03d].address + decoded[0x36d03d].size
        ctor_load = decoded[0x3811fd]
        ctor_slot = ctor_load.address + ctor_load.size + ctor_load.operands[1].mem.disp
        ctor_rows = [m for m in metadata['ScriptMetadataMethod'] if m['Address'] == ctor_slot]
        assert len(ctor_rows) == 1
        self.publication_ctor = ctor_rows[0]['Name']
        assert self.publication_ctor == 'Method$System.Collections.Generic.List<Character>..ctor()'
        self.decoded = decoded
        self.action_seen = set()

    def bind(self, **runtime):
        super().bind(**runtime)
        self.publication = self.arena + 0x1c0000
        self.order = self.arena + 0x1c6000
        self.role_class = self.arena + 0x1c7000
        self.real_code, self.bluff_code = self.stop + 0x600, self.stop + 0x700
        required = ['System.Collections.Generic.List<Character>_TypeInfo',
                    self.publication_ctor,
                    'System.Action<ActedInfo>_TypeInfo',
                    'Character.<>c__DisplayClass125_0_TypeInfo',
                    'Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()']
        assert all(n in self.bindings for n in required)

    def values(self, pointer):
        return [self.rq(self.rq(pointer + 0x10) + 0x20 + i * 8)
                for i in range(self.rd(pointer + 0x18))]

    def fill(self, pointer, values):
        self.q(pointer + 0x10, pointer + 0x1000)
        self.d(pointer + 0x18, len(values)); self.d(pointer + 0x1c, 0)
        self.q(pointer + 0x1018, 128)
        for i, value in enumerate(values): self.q(pointer + 0x1020 + i * 8, value)

    def role(self, pointer):
        self.q(pointer, self.role_class)
        self.q(self.role_class + 0x208, self.real_code)
        self.q(self.role_class + 0x210, self.role_class + 0x500)
        self.q(self.role_class + 0x258, self.bluff_code)
        self.q(self.role_class + 0x260, self.role_class + 0x580)
        self.uc.mem_write(self.role_class + 0x130, b'\x01')
        self.q(self.role_class + 0xc8, self.role_class + 0x600)
        self.q(self.role_class + 0x600, self.role_class)

    def prepare(self):
        super().prepare()
        self.post_events, self.actions, self.role_calls, self.allocations = [], [], [], []
        self.phase, self.action = 'initialization', None
        self.q(self.gs + 0x18, 0)
        self.uc.mem_write(self.publication, bytes(0x3000))
        self.uc.mem_write(self.order, bytes(0x100))
        order = self.opt.get('order', [1, 1])
        self.q(self.order + 0x18, len(order))
        for i, identity in enumerate(order): self.q(self.order + 0x20 + i * 8, self.data[identity])
        self.q(self.owner + 0x28, self.order); self.q(self.owner + 0x58, 0)
        for source in self.sources.values(): self.role(source)
        # Replacement board B has an authored pre-existing valid ordinary role.
        self.q(self.actors['other_character'] + 0x168, self.sources[7])
        self.initial_actors = self.snapshot()

    def post_snapshot(self):
        published = self.rq(self.gs + 0x18)
        return {**self.snapshot(), 'board': self.values(self.rq(self.owner + 0x20)),
                'published_identity': published,
                'published': self.values(published) if published else None,
                'allocations': self.allocations[:], 'role_calls': self.role_calls[:],
                'on_acted': {str(pointer): self.rq(pointer + 0x28)
                             for pointer in self.sources.values()
                             if self.rq(pointer + 0x28)} |
                            {str(call['clone']): self.rq(call['clone'] + 0x28)
                             for call in self.calls if call['clone']}}

    def post_emit(self, kind, **details):
        event = {'kind': kind, 'phase': self.phase, **details, 'snapshot': self.post_snapshot()}
        self.post_events.append(event)
        if self.opt.get('stop_post') == len(self.post_events):
            self.halt('post_' + kind)
            return False
        return True

    def hook(self, address, rva, c, dx, r8):
        x = self.x
        if rva in self.decoded and rva != 0x36d2db:
            self.action_seen.add(rva)
        if rva == 0x36d01e:
            assert self.active is None
            self.phase = 'publication'
            self.post_emit('before_publication')
            return True
        if rva == 0x36d2db:
            assert self.active is None and self.action is None
            self.state['boundary'] = 'before_on_setup'
            self.uc.emu_stop(); return True
        if self.phase == 'initialization':
            handled = super().hook(address, rva, c, dx, r8)
            if address == self.yield_return:
                self.role(self.clone)
            return handled
        if rva == 0x3811b0:
            assert dx == 0
            self.publication_source = c
            self.post_emit('publication_entry', source=c)
            self.action_seen.add(rva); return True
        if rva == self.publication_return:
            assert self.rq(self.gs + 0x18) == self.publication
            if self.post_emit('publication_return'):
                self.phase = 'actions'
            return True
        if rva == 0x3645c0:
            assert self.action is None and c in self.actors.values() and dx in (3, 5) and r8 == 0
            self.action = {'actor': c, 'trigger': dx, 'completed': False,
                           'before': self.actor_snapshot(c), 'role_start': len(self.role_calls)}
            self.actions.append(self.action)
            self.action_sp = self.reg(x.UC_X86_REG_RSP)
            self.action_saved = [self.reg(r) for r in self.nonvolatile]
            self.post_emit('action_entry', actor=c, trigger=dx)
            self.action_seen.add(rva); return True
        if rva in self.action_returns:
            assert self.action is not None
            assert self.reg(x.UC_X86_REG_RSP) == self.action_sp + 8
            assert [self.reg(r) for r in self.nonvolatile] == self.action_saved
            self.action['completed'] = True
            self.action['after'] = self.actor_snapshot(self.action['actor'])
            self.post_emit('action_return', actor=self.action['actor'], trigger=self.action['trigger'])
            self.action = None
            return True
        if address in (self.real_code, self.bluff_code):
            assert self.action is not None and dx == self.action['trigger'] and r8 == self.action['actor']
            assert self.reg(x.UC_X86_REG_R9) == self.role_class + (0x500 if address == self.real_code else 0x580)
            call = {'role': c, 'actor': r8, 'trigger': dx,
                    'route': 'act' if address == self.real_code else 'bluff_act'}
            if self.post_emit('role_gateway', **call):
                self.role_calls.append(call); self.ret()
            return True
        if rva == 0x2b7d40:
            if c == self.bindings['System.Collections.Generic.List<Character>_TypeInfo']:
                pointer, kind = self.publication, 'publication_list'
            else:
                assert c in (self.bindings['Character.<>c__DisplayClass125_0_TypeInfo'],
                             self.bindings['System.Action<ActedInfo>_TypeInfo'])
                kind = 'closure' if c == self.bindings['Character.<>c__DisplayClass125_0_TypeInfo'] else 'delegate'
                pointer = self.arena + 0x1d0000 + len(self.allocations) * 0x100
            if self.post_emit('allocate', object=kind, identity=pointer):
                self.allocations.append({'kind': kind, 'identity': pointer})
                self.uc.mem_write(pointer, bytes(0x100)); self.q(pointer, c); self.ret(pointer)
            return True
        if rva == 0xb610a0:
            assert c == self.publication and dx == self.publication_source and r8 == self.bindings[self.publication_ctor]
            if self.post_emit('copy_board', source=dx):
                self.fill(c, self.values(dx))
                if self.opt.get('replace_publication'): self.q(self.owner + 0x20, self.alternate)
                self.ret()
            return True
        if rva == 0x2b6ff0:
            assert self.rq(c) == dx
            if self.post_emit('barrier', target=c, value=dx): self.ret()
            return True
        if rva == 0x363c40:
            assert self.action and c == self.action['actor'] + 0x300 and dx in (10, 30) and r8 == 0
            status_list = self.rq(c + 0x10)
            statuses = [self.rd(self.rq(status_list + 0x10) + 0x20 + i * 4)
                        for i in range(self.rd(status_list + 0x18))]
            if self.post_emit('has_status', status=dx): self.ret(int(dx in statuses))
            return True
        if rva == 0x1c82480:
            assert c == 0 and dx == 0 and r8 == 0
            if self.post_emit('unity_live'): self.ret(0)
            return True
        if rva == 0x1c822c0:
            assert r8 == 0
            if self.post_emit('unity_equal', left=c, right=dx): self.ret(int(c == dx))
            return True
        if rva == 0x4d5b60:
            assert r8 == self.bindings['Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()']
            assert self.reg(x.UC_X86_REG_R9) == 0 and self.rq(dx + 0x10) == self.action['actor']
            assert self.rd(dx + 0x18) == self.action['trigger']
            if self.post_emit('delegate_ctor', identity=c, closure=dx): self.ret()
            return True
        if rva in (0x282580, 0xf74df0, 0x1c4b450):
            kind = {0x282580: 'box', 0xf74df0: 'format', 0x1c4b450: 'log'}[rva]
            if self.post_emit(kind): self.ret(self.arena + 0x190000 if kind != 'log' else 0)
            return True
        if rva == 0x33ed50:
            # Native ret is deliberately retained; base synthetic dispose is bypassed.
            self.action_seen.add(rva); return True
        if rva in self.decoded: self.action_seen.add(rva)
        return False

    def report(self, run, starts, rosters, fallback, **evidence):
        cases = []
        base_options = {'board': ['character', 'character'], 'manage_roster': [7, 1],
                        'alternate_board': ['other_character'], 'first_yield': True}
        def execute(options, family):
            result = run(starts, rosters, fallback, {**base_options, **options})
            assert len(self.calls) == 2
            assert all(bytes(self.uc.mem_read(p, 0x28)) == raw for p, raw in self.retained)
            result.update(family=family, initializer_calls=self.calls, initializer_events=self.events,
                          initial_actors=self.initial_actors, final_actors=self.post_snapshot(),
                          action_calls=self.actions, post_events=self.post_events)
            cases.append(result)
            return result
        for family, options in (
                ('alias', {}), ('replace_init', {'replace_callback': 1}),
                ('replace_publication', {'replace_publication': True}),
                ('ordered_repeat', {'order': [1, 7, 1]}),
                ('stop_second_init', {'stop_callback': 2})):
            baseline = execute(options, family)
            if 'stop_callback' in options:
                assert baseline['error'] == 'state_callback' and not baseline['post_events']
                continue
            assert baseline['error'] is None and baseline['boundary'] == 'before_on_setup'
            published = baseline['final_actors']['published']
            a, b = self.actors['character'], self.actors['other_character']
            assert published == ([b] if family == 'replace_init' else [a, a])
            actions = baseline['action_calls']
            expected = [(b, 3)] if family in ('replace_init', 'replace_publication') else [
                (a, 3), (a, 3), (a, 5), (a, 5)]
            assert [(act['actor'], act['trigger']) for act in actions] == expected
            assert all(act['completed'] for act in actions)
            expected_roles = [] if family in ('replace_init', 'replace_publication') else [3, 3, 5]
            actual_roles = baseline['final_actors']['role_calls']
            if family in ('replace_init', 'replace_publication'):
                assert [call['trigger'] for call in actual_roles] == [3]
                assert [call['route'] for call in actual_roles] == ['bluff_act']  # B keeps its authored corruption.
            else:
                assert [call['trigger'] for call in actual_roles] == expected_roles
                assert all(call['route'] == 'act' for call in actual_roles)
                assert all(call['role'] == self.calls[-1]['clone'] for call in actual_roles)
            # Inject an attempted-service stop at every post-Init recorded event.
            for index, event in enumerate(baseline['post_events']):
                stopped = execute({**options, 'stop_post': index + 1}, family + '_stop')
                assert stopped['error'] == 'post_' + event['kind']
                assert stopped['post_events'] == baseline['post_events'][:index + 1]
                assert stopped['final_actors'] == event['snapshot']
        snapshots, indices = [], {}
        for case in cases:
            for event in case['events'] + case['initializer_events'] + case['post_events']:
                value = event.pop('snapshot'); key = json.dumps(value, sort_keys=True)
                if key not in indices: indices[key] = len(snapshots); snapshots.append(value)
                event['snapshot_index'] = indices[key]
        verified_fields = {}
        for family in (evidence['fields'], self.fields, self.action_fields):
            for name, declarations in family.items():
                verified_fields.setdefault(name, [])
                verified_fields[name].extend(d for d in declarations if d not in verified_fields[name])
        return {'build_id': BUILD, 'schema': 'manage_publication_action_join_v1',
                'case_count': len(cases),
                'initializer_calls': sum(len(c['initializer_calls']) for c in cases),
                'completed_initializers': sum(call['completed'] for c in cases for call in c['initializer_calls']),
                'action_calls': sum(len(c['action_calls']) for c in cases),
                'completed_actions': sum(call['completed'] for c in cases for call in c['action_calls']),
                'metadata_verified': evidence['metadata'] + self.methods + self.action_methods,
                'native_assertions': len(evidence['checks']) + len(self.checks) + len(self.action_checks),
                'native_instructions_executed': len(evidence['visited'] | self.native_seen | self.action_seen),
                'initializer_body_sha256': self.fingerprints, 'publication_action_body_sha256': self.action_hashes,
                'field_declarations': verified_fields,
                'snapshot_table': snapshots, 'cases': cases,
                'limits': [
                    'Development corpus, not held-out or original live observations.',
                    'Actual Manage/pools/sources/filters/predicate/Init/Hidden Refresh/first MoveNext/UpdateCharacters/Act/RoleAct/CheckLying share memory.',
                    'Ordinary authored virtual role gateways are inert; no concrete role behavior, clues or player observation claim.',
                    'Clone identities, collection shallow-copy service, status membership, equality, metadata, allocation, delegates, UI and first-step scheduling are supplied.',
                    'Stops before onSetup; no ShuffleDeck, resumed Reveal, native engine queue admission or solver integration.',
                    'Stable list occurrences and controlled board replacement; no List version checking, arbitrary reentrancy or unwind.',
                    'Every post-Init event has an exact stopped-prefix development regression; initial second-callback failure remains separate.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    result = pool_audit(args.game_root, args.dumper_root, initialization=PublicationActionJoin())
    with args.output.open('w', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(result, indent=2) + '\n')
    print(json.dumps({key: result[key] for key in (
        'case_count', 'initializer_calls', 'completed_initializers', 'action_calls',
        'completed_actions', 'native_assertions', 'native_instructions_executed')}))
