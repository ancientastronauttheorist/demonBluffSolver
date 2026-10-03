"""Concrete N5 Init/Start role dispatch from explicitly supplied post-Init actors.

This is not the retained Manage/Init join. Proprietary instruction bytes stay
private; original role/status bodies execute under named runtime services.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_first_village_bluff_generation import BluffJoin
from audit_first_village_profile_generation import ROOT, load_inputs
from audit_report_snapshots import expand_snapshots, pool_snapshots


ORDER = [21596, 21614, 21626, 21621, 21618]
ROLES = {21596: 'Minion', 21614: 'Confessor', 21626: 'Empath',
         21621: 'Tracker', 21618: 'Shugenja'}
METHODS = ['Character.Act', 'Character.RoleAct', 'CharacterHelper.CheckLying',
           'CharacterHelper.CheckLyingAppearance', 'CharacterStatuses.AddStatus',
           'CharacterStatuses.Contains', 'Confessor.Act', 'Confessor.BluffAct',
           'Confessor.OnInit', 'Empath.Act', 'Empath.BluffAct', 'Tracker.Act',
           'Tracker.BluffAct', 'Shugenja.Act', 'Shugenja.BluffAct', 'Minion.Act',
           'Role.BluffAct']


class RoleSetupJoin(BluffJoin):
    def __init__(self, inputs, profiles, characters):
        super().__init__(inputs, profiles, characters)
        import capstone
        self.role_method_names = {}
        self.role_bodies = []
        for name in METHODS:
            rows = [m for m in self.meta['ScriptMethod'] if m['Name'] == name.replace('.', '$$', 1)]
            assert len(rows) == 1, name
            row = rows[0]
            start = row['Address']
            end = min(m['Address'] for m in self.meta['ScriptMethod'] if m['Address'] > start)
            section = self.pe.get_section_by_rva(start)
            assert section and end <= section.VirtualAddress+section.SizeOfRawData
            body = self.pe.get_data(start, end-start)
            assert len(body) == end-start
            instructions = list(self.cs.disasm(body, start))
            while instructions and instructions[-1].mnemonic == 'int3':
                instructions.pop()
            assert instructions and instructions[0].address == start
            assert all(a.address+a.size == b.address for a, b in zip(instructions, instructions[1:]))
            self.instructions.update({i.address: i for i in instructions})
            self.entries[name] = start
            self.role_method_names.setdefault(start, []).append(name)
            self.role_bodies.append({'name': name, 'rva': hex(start),
                                     'signature': row['Signature'],
                                     'end_rva': hex(instructions[-1].address+instructions[-1].size),
                                     'instruction_count': len(instructions),
                                     'body_sha256': hashlib.sha256(body[:instructions[-1].address+instructions[-1].size-start]).hexdigest()})
        self.role_assertions = {
            0x33ED50: ('ret', '0'),
            0x363AF1: ('call', '0xb45070'), 0x363B0C: ('call', '0xb45070'),
            0x363B27: ('call', '0x41a0'), 0x363B36: ('call', '0x2b6ff0'),
            0x364733: ('call', '0x397750'),
            0x36886E: ('call', 'qword ptr [rax + 0x258]'),
            0x368896: ('call', 'qword ptr [rax + 0x208]'),
            0x3D65D7: ('cmp', 'edx, 3'),
            0x3D65ED: ('jmp', 'qword ptr [rax + 0x1c8]'),
            0x3D65F4: ('cmp', 'edx, 0x1e'),
            0x3D6657: ('cmp', 'edx, 3'),
            0x3D666D: ('jmp', 'qword ptr [rax + 0x1c8]'),
            0x3D6674: ('cmp', 'edx, 0x1e'),
            0x3D6A65: ('xor', 'r9d, r9d'),
            0x3D6A68: ('mov', 'qword ptr [rsp + 0x20], 0'),
            0x3D6A78: ('call', '0x363aa0'),
            0x3B09F7: ('cmp', 'edx, 0x1e'),
            0x3B33E7: ('cmp', 'edx, 0x1e'),
            0x3C4CAA: ('jmp', 'qword ptr [rax + 0x208]'),
        }
        for address, expected in self.role_assertions.items():
            assert (self.instructions[address].mnemonic, self.instructions[address].op_str) == expected
        # Exact target membership/signatures, including folded aliases, remain
        # separate from whichever canonical native address happens to be shared.
        declarations = []
        for target in ('gameplay_role_confessor', 'gameplay_role_lover',
                       'gameplay_role_enlightened', 'gameplay_roles_scout_hunter',
                       'gameplay_status_corruption_truth'):
            declarations.extend(json.loads((ROOT/f'targets/{target}.json').read_text(encoding='utf-8'))['functions'])
        for body in self.role_bodies:
            rows = [r for r in declarations if r['metadata_name'] == body['name'].replace('.', '$$', 1)]
            if rows:
                assert any(int(r['rva'], 16) == int(body['rva'], 16) and r['signature'] == body['signature'] for r in rows)
        slots = {r['Address']: r['Name'] for key in ('ScriptMetadata', 'ScriptMetadataMethod') for r in self.meta[key]}
        strings = {r['Address']: r['Value'] for r in self.meta['ScriptString']}
        for instruction in self.instructions.values():
            for operand in instruction.operands:
                if operand.type != capstone.CS_OP_MEM or operand.mem.base != capstone.x86.X86_REG_RIP:
                    continue
                address = instruction.address+instruction.size+operand.mem.disp
                if address in slots or address in strings:
                    name = slots.get(address, 'literal:'+strings.get(address, ''))
                    if name not in self.names:
                        p = self.allocate(name)
                        self.names[name], self.metadata_names[p] = p, name
                        self.d(p+0xE0, 1)
                    self.q(self.base+address, self.names[name])
                elif instruction.mnemonic == 'cmp' and operand.size == 1:
                    self.uc.mem_write(self.base+address, b'\1')
        fields = {
            'CharacterStatuses': ['public List<ECharacterStatus> statuses; // 0x10',
                                  'public List<ECharacterStatus> resistances; // 0x18',
                                  'public Character targetCharacter; // 0x20'],
            'Character': ['public Role role; // 0x168', 'public Role bluffRole; // 0x170',
                          'private bool characterStartActed; // 0x11C',
                          'public CharacterStatuses statuses; // 0xF0',
                          'public EAlignment alignment; // 0xF8'],
            'Role': ['public Action<ActedInfo> onActed; // 0x28',
                     'public Action<ActedInfo> savedOnActed; // 0x30',
                     'public Dictionary<ETriggerPhase, Action<ActedInfo>> savedTriggerActs; // 0x38',
                     'public ActedInfo savedActInfo; // 0x40'],
        }
        lines = self.dump.splitlines()
        type_indices = {'Character': 5487, 'CharacterStatuses': 5488, 'Role': 5853}
        for cls, wanted in fields.items():
            matches = [i for i, line in enumerate(lines)
                       if (line.startswith('public class '+cls+' ') or line.startswith('public abstract class '+cls+' '))
                       and line.endswith('// TypeDefIndex: '+str(type_indices[cls]))]
            assert len(matches) == 1, (cls, type_indices[cls], matches)
            stop = next(i for i in range(matches[0]+1, len(lines)) if lines[i] == '}')
            assert all(field in '\n'.join(lines[matches[0]:stop]) for field in wanted)
        self.fields = fields
        self.enum_lists = {}
        self.actor_storage = {}
        self.action_roles = {}
        self.role_fields_sentinels = {}
        for index, identity in enumerate(ORDER):
            actor = self.actors[index]
            status = self.allocate('statuses:'+str(index))
            active = self.enum_list('active:'+str(index))
            resistance = self.enum_list('resistance:'+str(index))
            self.q(status+0x10, active); self.q(status+0x18, resistance)
            real = self.make_role(ROLES[identity], 'runtime_role:'+str(index))
            source = self.make_role(ROLES[identity], 'source_role:'+str(identity))
            self.q(self.assets[identity]+0x140, source)
            self.actor_storage[actor] = (status, active, resistance, real, identity)
        self.copied_hunter = self.make_role('Tracker', 'supplied_copied_Hunter')
        self.copied_confessor = self.make_role('Confessor', 'supplied_copied_Confessor')
        self.role_ready = False
        self.role_calls = []
        self.role_mode = False

    def enum_list(self, label):
        pointer = self.allocate(label)
        backing = self.allocate(label+':backing')
        self.q(pointer+0x10, backing); self.q(backing+0x18, 64)
        self.enum_lists[pointer] = backing
        return pointer

    def enum_values(self, pointer):
        return [self.rd(self.enum_lists[pointer]+0x20+i*4) for i in range(self.rd(pointer+0x18))]

    def enum_write(self, pointer, values, version=0):
        assert len(values) <= 64
        self.d(pointer+0x18, len(values)); self.d(pointer+0x1C, version)
        for index, value in enumerate(values):
            self.d(self.enum_lists[pointer]+0x20+index*4, value)

    def make_role(self, role, label):
        pointer, klass = self.allocate(label), self.allocate(label+':class')
        self.q(pointer, klass)
        self.action_roles[pointer] = role
        for offset, method in ((0x208, role+'.Act'),
                               (0x258, role+'.BluffAct' if role != 'Minion' else 'Role.BluffAct')):
            self.q(klass+offset, self.base+self.entries[method])
            self.q(klass+offset+8, self.allocate(label+':'+method+':method_info'))
        if role == 'Confessor':
            self.q(klass+0x1C8, self.base+self.entries['Confessor.OnInit'])
            self.q(klass+0x1D0, self.allocate(label+':OnInit:method_info'))
        for offset, name in ((0x30, 'saved_callback'), (0x38, 'saved_trigger_acts'), (0x40, 'saved_info')):
            value = self.allocate(label+':'+name)
            self.q(pointer+offset, value)
        self.role_fields_sentinels[pointer] = [self.rq(pointer+offset) for offset in (0x30, 0x38, 0x40)]
        return pointer

    def prepare_post_init(self):
        # Supplied state only: this script never executes Character.Init or
        # pretends to resume A's independent retained initialization fixture.
        for actor, (status, active, resistance, real, identity) in self.actor_storage.items():
            self.uc.mem_write(actor, bytes(0x200))
            self.q(actor+0x50, self.assets[identity]); self.q(actor+0xF0, status)
            self.d(actor+0xF8, 20 if identity == 21596 else 10)
            self.q(actor+0x168, real)
            self.enum_write(active, []); self.enum_write(resistance, [])
            self.q(status+0x20, 0)
            self.uc.mem_write(actor+0xE4, struct.pack('<I', 5))
            self.uc.mem_write(actor+0x11C, b'\0')
        self.d(self.gameplay_static+0x28, 10)
        for role in self.action_roles:
            self.q(role+0x28, 0)
        self.role_ready = True

    def role_snapshot(self):
        return {
            'actors': {self.labels[actor]: {
                'asset': identity, 'alignment': self.rd(actor+0xF8),
                'real_role': self.labels.get(self.rq(actor+0x168)),
                'copied_role': self.labels.get(self.rq(actor+0x170)),
                'raw_bluff': self.asset_ids.get(self.rq(actor+0x58), self.labels.get(self.rq(actor+0x58))),
                'start_acted': self.uc.mem_read(actor+0x11C, 1)[0],
                'active_identity': self.labels[active], 'active': self.enum_values(active),
                'active_version': self.rd(active+0x1C),
                'resistance_identity': self.labels[resistance], 'resistance': self.enum_values(resistance),
                'resistance_version': self.rd(resistance+0x1C),
                'target': self.labels.get(self.rq(status+0x20)),
            } for actor, (status, active, resistance, real, identity) in self.actor_storage.items()},
            'roles': {self.labels[p]: {'class': role, 'on_acted': self.labels.get(self.rq(p+0x28)),
                                      'saved_fields': [self.labels.get(self.rq(p+offset)) for offset in (0x30, 0x38, 0x40)]}
                      for p, role in self.action_roles.items()},
        }

    def snapshot(self):
        result = super().snapshot()
        if getattr(self, 'role_ready', False):
            result['concrete_role_storage'] = self.role_snapshot()
        return result

    def hook(self, uc, address, size, user):
        if not getattr(self, 'role_mode', False):
            return super().hook(uc, address, size, user)
        x = self.x
        rva = address-self.base
        args = [uc.reg_read(reg) for reg in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        c, t, m, n = args
        if rva in self.role_method_names and (rva != 0x33ED50 or c in self.action_roles):
            role = self.action_roles.get(c)
            names = self.role_method_names[rva]
            concrete = next((name for name in names if role and name.startswith(role+'.')), names[0])
            row = {'method': concrete, 'receiver': self.labels.get(c),
                   'argument': self.labels.get(t, t), 'r8': self.labels.get(m, m), 'r9': self.labels.get(n, n)}
            if rva == self.entries['CharacterStatuses.AddStatus']:
                assert t == 25 and m in self.actor_storage and n == 0
                assert c == self.actor_storage[m][0]
                assert self.rq(uc.reg_read(x.UC_X86_REG_RSP)+0x28) == 0
                row['fifth_stack_argument'] = 0
            self.role_calls.append(row)
        if rva in self.instructions:
            self.visited.add(rva)
            return
        if address == self.stop+0x400:
            raise AssertionError('Init/Start unexpectedly invoked onActed callback')
        if rva == 0xB45070:
            assert c in self.enum_lists
            if self.service('enum_contains_supplied', list=self.labels[c], status=t & 0xFFFFFFFF,
                            method=self.metadata_names.get(m, self.labels.get(m))):
                self.ret(int((t & 0xFFFFFFFF) in self.enum_values(c)))
            return
        if rva == 0x41A0:
            assert c in self.enum_lists
            if self.service('enum_add_supplied', list=self.labels[c], status=t & 0xFFFFFFFF,
                            method=self.metadata_names.get(m, self.labels.get(m))):
                values = self.enum_values(c)
                self.enum_write(c, values+[t & 0xFFFFFFFF], self.rd(c+0x1C)+1)
                self.ret()
            return
        if rva == 0x282580:
            if self.service('enum_box_supplied', type=self.metadata_names.get(c), value=self.rd(t)):
                p = self.allocate('boxed_trigger:'+str(self.cursor)); self.d(p+0x10, self.rd(t)); self.ret(p)
            return
        if rva == 0xF74DF0:
            if self.service('log_format_supplied', literal=self.metadata_names.get(c), boxed=self.labels.get(t), method=m):
                self.ret(self.allocate('formatted_log:'+str(self.cursor)))
            return
        if rva == 0x1C4B450:
            if self.service('log_supplied', text=self.labels.get(c), context=t):
                self.ret()
            return
        if rva == 0x2B7D40:
            name = self.metadata_names.get(c)
            assert name in ('Character.<>c__DisplayClass125_0_TypeInfo', 'System.Action<ActedInfo>_TypeInfo'), name
            if self.service('role_allocation_supplied', type=name):
                self.ret(self.allocate(name+':instance:'+str(self.cursor)))
            return
        if rva == 0x4D5B60:
            assert self.metadata_names.get(m) == 'Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()' and n == 0
            assert self.rq(t+0x10) in self.actor_storage
            if self.service('role_delegate_ctor_supplied', allocated=self.labels[c], target=self.labels[t],
                            actor=self.labels[self.rq(t+0x10)], trigger=self.rd(t+0x18), method=self.metadata_names[m]):
                self.q(c+0x18, self.stop+0x400); self.q(c+0x20, t)
                self.q(c+0x28, m); self.q(c+0x40, t)
                self.ret()
            return
        return super().hook(uc, address, size, user)

    def invoke(self, name, actor, trigger=None, stop_service=None):
        self.role_mode = True
        self.role_calls = []
        result = self.run(name, actor, 0 if trigger is None else trigger, stop_service=stop_service)
        self.role_mode = False
        result['concrete_calls'] = list(self.role_calls)
        result['native_boolean'] = self.uc.reg_read(self.x.UC_X86_REG_RAX) & 255 if trigger is None else None
        assert all([self.rq(p+offset) for offset in (0x30, 0x38, 0x40)] == values
                   for p, values in self.role_fields_sentinels.items())
        return result


def audit(game_root, dumper_root, probe=False):
    profiles = json.loads((ROOT/f'reports/{BUILD}_ascension_assets_audit.json').read_text(encoding='utf-8'))
    characters = json.loads((ROOT/f'reports/{BUILD}_character_assets_audit.json').read_text(encoding='utf-8'))
    generation_bytes = (ROOT/f'reports/{BUILD}_first_village_bluff_generation.json').read_bytes()
    prior = expand_snapshots(json.loads(generation_bytes.decode('utf-8')))
    records = {row['path_id']: row for row in characters['records']}
    assert {identity: records[identity]['role_type'] for identity in ORDER} == ROLES
    assert all(records[identity]['startingAlignment'] == (20 if identity == 21596 else 10)
               for identity in ORDER)
    assert prior['schema_version'] == 1 and prior['build_id'] == BUILD
    assert prior['generation_index_factor']['cases'][0]['returned_order'] == ORDER
    if not probe:
        from audit_ascension_assets import audit as audit_profiles
        from audit_character_assets import audit as audit_characters
        assert profiles == json.loads(json.dumps(audit_profiles(game_root, dumper_root)))
        assert characters == json.loads(json.dumps(audit_characters(game_root, dumper_root)))
    inputs = load_inputs(game_root, dumper_root)
    runner = RoleSetupJoin(inputs, profiles, characters)
    setup = []
    for name, receiver, argument, choices in (
        ('GameData.SetupCurrentAscension', runner.game, 0, []),
        ('AscensionsData.ClearCurrentPickedScript', runner.temporary, 0, []),
        ('AscensionsData.SetupCharactersCount', runner.temporary, 0, [0]),
        ('AscensionsData.SetupStartingCharacters', runner.temporary, 0, []),
        ('Gameplay.GetCurrentScript', runner.gameplay, 0, [])):
        row = runner.run(name, receiver, argument, choices)
        assert not row['failure'] and not row['boundary']; setup.append(row)
    runner.q(runner.gameplay_static+0x30, runner.uc.reg_read(runner.x.UC_X86_REG_RAX))
    generation = runner.run('Gameplay.GetRandomCharacters', runner.gameplay, 5, [0]*10)
    returned = runner.uc.reg_read(runner.x.UC_X86_REG_RAX)
    pool = runner.run('Characters.ManageCharacters', runner.owner, returned, [0]*6)
    assert generation['final']['returned_order'] == ORDER
    assert generation['draws'] == prior['generation']['draws']
    assert pool['final']['pools'] == prior['pool']['final']['pools']
    assert pool['boundary'] == prior['pool']['boundary']
    runner.prepare_post_init()
    fresh = runner.save()
    initial = runner.snapshot()
    rows, stopped = [], []

    def record(label, name, actor, trigger=None):
        before = runner.save()
        full = runner.invoke(name, actor, trigger)
        assert not full['failure'] and not full['boundary']
        after = runner.save()
        for ordinal in range(1, len(full['services'])+1):
            runner.restore(before)
            partial = runner.invoke(name, actor, trigger, ordinal)
            assert partial['services'] == full['services'][:ordinal]
            assert partial['final'] == full['services'][ordinal-1]['snapshot']
            assert partial['failure'] == 'service:'+full['services'][ordinal-1]['service']
            stopped.append({'case': label, 'stop_service_ordinal': ordinal,
                            'failure': partial['failure'], 'final': partial['final'],
                            'prefix_sha256': hashlib.sha256(json.dumps(partial['services'], sort_keys=True).encode('utf-8')).hexdigest()})
        runner.restore(after)
        rows.append({'case': label, 'actor': runner.labels[actor], 'trigger': trigger, 'result': full})
        return full

    for index, actor in enumerate(runner.actors):
        row = record('post_init_pass:'+str(index), 'Character.Act', actor, 3)
        state = row['final']['concrete_role_storage']['actors'][runner.labels[actor]]
        assert state['active'] == ([25] if ORDER[index] == 21614 else [])
        assert state['start_acted'] == 0
        assert len([e for e in row['services'] if e['service'] == 'role_delegate_ctor_supplied']) == 1
        assert len([e for e in row['concrete_calls'] if e['method'] == 'Confessor.OnInit']) == int(ORDER[index] == 21614)
    after_init = runner.snapshot()
    # Separate trigger probes, not a claim that the original ordered Start
    # array schedules these five roles or that Manage has completed.
    for index, actor in enumerate(runner.actors):
        row = record('start_probe:'+str(index), 'Character.Act', actor, 5)
        assert row['final']['concrete_role_storage']['actors'][runner.labels[actor]]['start_acted'] == 1
        assert not any(e['method'] == 'Confessor.OnInit' for e in row['concrete_calls'])
        repeat = record('start_repeat_probe:'+str(index), 'Character.Act', actor, 5)
        assert not any(e['service'] == 'role_delegate_ctor_supplied' for e in repeat['services'])
        assert repeat['final'] == row['final']
    confessor = runner.actors[1]
    for label, active, resistance, alignment, copied in (
        ('real_fresh', [], [], 10, False), ('real_repeat', [25], [], 10, False),
        ('real_corrupted', [10], [], 10, False), ('real_evil', [], [], 20, False),
        ('resisted_fresh', [], [25], 10, False), ('resisted_duplicate', [25], [25], 10, False),
        ('copied_on_ordinary_minion', [], [], 20, True)):
        runner.restore(fresh)
        actor = runner.actors[0] if copied else confessor
        status, active_list, resistance_list, _, _ = runner.actor_storage[actor]
        runner.enum_write(active_list, active, 7); runner.enum_write(resistance_list, resistance, 9)
        runner.q(status+0x20, runner.actors[4]); runner.d(actor+0xF8, alignment)
        if copied:
            runner.q(actor+0x170, runner.copied_confessor); runner.q(actor+0x58, runner.assets[21614])
        row = record(label, 'Character.Act', actor, 3)
        state = row['final']['concrete_role_storage']['actors'][runner.labels[actor]]
        expected = active if 25 in resistance or 25 in active else active+[25]
        assert state['active'] == expected
        assert state['active_version'] == 7+int(25 not in resistance and 25 not in active)
        assert state['resistance'] == resistance and state['resistance_version'] == 9
        assert state['target'] == ('actor:4' if 25 in resistance else None)
        actual = record(label+':actual_truth', 'CharacterHelper.CheckLying', actor)
        appearance = record(label+':appearance_truth', 'CharacterHelper.CheckLyingAppearance', actor)
        assert actual['native_boolean'] == int(10 in active or alignment == 20)
        assert appearance['native_boolean'] == int(25 not in expected and (10 in active or alignment == 20))
    for trigger in (3, 5):
        runner.restore(fresh)
        actor = runner.actors[0]
        runner.q(actor+0x170, runner.copied_hunter); runner.q(actor+0x58, runner.assets[21621])
        row = record('ordinary_minion_copied_hunter:'+str(trigger), 'Character.Act', actor, trigger)
        assert row['final']['concrete_role_storage']['actors']['actor:0']['active'] == []
        assert any(e['method'] == 'Tracker.BluffAct' for e in row['concrete_calls'])
    report = {
        'schema_version': 'first_village_role_setup_v1', 'build_id': BUILD,
        'setup': setup, 'generation': generation, 'pool': pool,
        'supplied_post_init': initial, 'after_init_pass': after_init,
        'cases': rows, 'stopped_prefixes': stopped,
        'concrete_bodies': runner.role_bodies, 'fields_verified': runner.fields,
        'instruction_assertions': [{'rva': hex(a), 'mnemonic': e[0], 'operands': e[1]} for a, e in runner.role_assertions.items()],
        'counters': {'completed_cases': len(rows), 'stopped_prefixes': len(stopped),
                     'concrete_body_memberships': len(runner.role_bodies),
                     'distinct_concrete_body_rvas': len({r['rva'] for r in runner.role_bodies}),
                     'distinct_executed_instructions': len(runner.visited)},
        'source_hashes': {'game_assembly': inputs[-1]['inputs']['game_assembly']['sha256'],
                          'script_json': hashlib.sha256((dumper_root/'script.json').read_bytes()).hexdigest(),
                          'dump_cs': hashlib.sha256((dumper_root/'dump.cs').read_bytes()).hexdigest(),
                          'assets': profiles['source_hashes'],
                          'script': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          'generation_report': hashlib.sha256(generation_bytes).hexdigest(),
                          'targets': {name: hashlib.sha256((ROOT/'targets'/name).read_bytes()).hexdigest()
                                      for name in ('gameplay_role_confessor.json', 'gameplay_role_lover.json',
                                                   'gameplay_role_enlightened.json', 'gameplay_roles_scout_hunter.json',
                                                   'gameplay_status_corruption_truth.json')},
                          'dependencies': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                           for name in ('audit_first_village_bluff_generation.py', 'audit_first_village_profile_generation.py',
                                                        'audit_manage_pool_composition.py', 'audit_character_assets.py',
                                                        'audit_ascension_assets.py', 'audit_report_snapshots.py')}},
        'scope': {'profile': 21674, 'generation_row': 0, 'pool_row': 0,
                  'post_init': 'Explicit supplied sparse actors/source-role bindings, clear statuses and resistance, no raw/copy bluff; Init/Manage continuation not executed.',
                  'start': 'Separate actual trigger/latch probes; actual ordered-Start scheduling not asserted.',
                  'sensitivity': 'Exact 25 resistance/corruption/Evil/copy states are supplied sensitivities, not generated worlds.',
                  'copied_roles': 'Hunter and Confessor probes only; no claim to close all 24 generated bluff candidates.',
                  'services': 'Warm metadata, enum-list membership/add with capacity 64, actor/role allocation, delegate construction, logging, reference barriers and Unity liveness supplied.',
                  'excluded': 'Day results, Reveal/acquisition/queue/UI rendering/admitted history, world weights and full generated setup.'},
    }
    assert (ROOT/f'reports/{BUILD}_first_village_bluff_generation.json').read_bytes() == generation_bytes
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--probe', action='store_true')
    arguments = parser.parse_args()
    report = audit(arguments.game_root, arguments.dumper_root, arguments.probe)
    if arguments.probe:
        print(json.dumps(report['counters'], sort_keys=True))
    else:
        assert arguments.output
        packed = pool_snapshots(report)
        assert expand_snapshots(packed) == report
        arguments.output.write_text(json.dumps(packed, sort_keys=True, separators=(',', ':'), ensure_ascii=True)+'\n', encoding='utf-8')
        print(json.dumps(report['counters'], sort_keys=True))
