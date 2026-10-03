"""Original N5 roster/pool/Minion join; all engine services are explicit.

No live process or save access. Native bytes stay in the private workspace.
"""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_first_village_profile_generation import NativeJoin, load_inputs, ROOT, COUNT_FIELDS
from audit_manage_pool_composition import ENTRIES as POOL_ENTRIES
from audit_report_snapshots import pool_snapshots, expand_snapshots


class BluffJoin(NativeJoin):
    """Retain the original profile graph and its actual caller-side mutations."""

    def __init__(self, inputs, profiles, characters):
        super().__init__(inputs, profiles, characters)
        import capstone
        self.aliases = {}
        for row in self.meta['ScriptMethod']:
            self.aliases.setdefault(row['Address'], []).append(row)
        specs = {address: ('Characters.<>c__DisplayClass22_0$$<PickRoundBluffs>b__0'
                          if address == 0x377170 else
                          ('Gameplay$$' if address in (0x37C3F0, 0x37C1A0, 0x37DC00)
                           else 'AscensionsData$$' if address == 0x3B1E10 else 'Characters$$') + name)
                 for address, name in POOL_ENTRIES.items()}
        specs.update({0x37CE10: 'Gameplay$$GetRandomCharacters',
                      0x37E760: 'Gameplay$$ManageAlwaysInDeck',
                      0x37B080: 'Gameplay$$AddCharacterToTheRoster',
                      0x392D50: 'Gameplay.<>c$$<GetRandomCharacters>b__65_0',
                      0x3E49F0: 'Minion$$GetBluffIfAble',
                      0x396840: 'Helpers$$RollDice',
                      0x36C7A0: 'Characters$$GetRandomDuplicateBluff',
                      0x36C810: 'Characters$$GetRandomUniqueBluff',
                      0x36BF80: 'Characters$$GetARandomBluffMustInclude',
                      0x37B370: 'Gameplay$$AddScriptCharacterIfAble'})
        self.extra_names = {}
        for start, name in specs.items():
            rows = [r for r in self.aliases[start] if r['Name'] == name]
            assert len(rows) == 1, name
            row = rows[0]
            end = min(a for a in self.aliases if a > start)
            body = self.pe.get_data(start, end-start)
            ins = list(self.cs.disasm(body, start))
            while ins and ins[-1].mnemonic == 'int3':
                ins.pop()
            assert ins and ins[0].address == start
            assert all(a.address+a.size == b.address for a, b in zip(ins, ins[1:]))
            if start not in (v for v in self.entries.values()):
                self.body_evidence.append({'name': name, 'rva': hex(start),
                                          'end_rva': hex(ins[-1].address+ins[-1].size),
                                          'instruction_count': len(ins), 'signature': row['Signature'],
                                          'body_sha256': hashlib.sha256(body[:ins[-1].address+ins[-1].size-start]).hexdigest()})
            self.instructions.update({i.address: i for i in ins})
            key = name.replace('$$', '.', 1)
            self.entries[key] = start
            self.extra_names[start] = key
        self.assertions = {
            0x37D024: ('call', '0x3802d0'),
            0x37D702: ('call', '0x628d70'), 0x37D711: ('call', '0x62e890'),
            0x392D52: ('jmp', '0x1c86710'),
            0x36CF0A: ('call', '0x36d3a0'), 0x36CF14: ('call', '0x36d720'),
            0x37C24B: ('mov', 'rdx, qword ptr [rdx + 0x68]'),
            0x37C308: ('mov', 'rdx, qword ptr [rdx + 0x68]'),
            0x3E4A26: ('call', '0x396840'), 0x3E4A48: ('call', '0x36c810'),
            0x3E4A8D: ('call', '0x37b370'), 0x3E4AAF: ('jmp', '0x36c7a0')}
        for address, expected in self.assertions.items():
            assert (self.instructions[address].mnemonic, self.instructions[address].op_str) == expected, (
                hex(address), expected, self.instructions[address].mnemonic, self.instructions[address].op_str)
        slots = {r['Address']: r['Name'] for k in ('ScriptMetadata', 'ScriptMetadataMethod') for r in self.meta[k]}
        for ins in self.instructions.values():
            for op in ins.operands:
                if op.type != capstone.CS_OP_MEM or op.mem.base != capstone.x86.X86_REG_RIP:
                    continue
                address = ins.address+ins.size+op.mem.disp
                if address in slots:
                    name = slots[address]
                    if name not in self.names:
                        p = self.allocate(name)
                        self.names[name], self.metadata_names[p] = p, name
                        self.d(p+0xE0, 1)
                    self.q(self.base+address, self.names[name])
                elif ins.mnemonic == 'cmp' and op.size == 1:
                    self.uc.mem_write(self.base+address, b'\1')
        for name in ('Characters', 'Gameplay', 'CharacterData'):
            rows = [s for s in self.dump.splitlines() if s.startswith('public class '+name+' ')]
            assert len(rows) == 1
            block = self.dump.split(rows[0], 1)[1].split('\n}', 1)[0]
            fields = {'Characters': ['public List<CharacterData> BluffMustInclude; // 0x50',
                                     'public static Characters Instance; // 0x0'],
                      'Gameplay': ['public static Gameplay Instance; // 0x10'],
                      'CharacterData': ['public bool bluffable; // 0x13C']}[name]
            assert all(field in block for field in fields), (name, fields)
        self.owner = self.allocate('Characters')
        self.owner_static = self.allocate('Characters.static')
        self.q(self.names['Characters_TypeInfo']+0xB8, self.owner_static)
        self.q(self.owner_static, self.owner)
        self.q(self.gameplay_static+0x10, self.gameplay)
        self.pools = [self.list([], name) for name in ('unique_pool', 'duplicate_pool', 'must_include')]
        for offset, p in zip((0x40, 0x48, 0x50), self.pools):
            self.q(self.owner+offset, p)
        self.actors = [self.allocate(f'actor:{i}') for i in range(5)]
        self.board = self.list(self.actors, 'physical_board')
        self.q(self.owner+0x20, self.board)
        self.minion_role = self.allocate('ordinary_Minion_role')
        self.q(self.actors[0]+0x50, self.assets[21596])
        self.q(self.actors[0]+0x90, 0)
        for row in characters['records']:
            self.uc.mem_write(self.assets[row['path_id']]+0x13C, bytes([row['bluffable']]))
        # The native duplicate Add fast path reads this supplied RGCTX storage.
        method = self.names['Method$System.Collections.Generic.List<CharacterData>.Add()']
        klass, context, slot = [self.allocate(n) for n in ('add.class', 'add.rgctx', 'add.slot')]
        self.q(method+0x20, klass)
        self.q(klass+0xC0, context)
        self.q(context+0x70, slot)
        closure_static = self.allocate('Gameplay.closure.static')
        self.closure_receiver = self.allocate('Gameplay.closure.receiver')
        self.q(self.names['Gameplay.<>c_TypeInfo']+0xB8, closure_static)
        self.q(closure_static, self.closure_receiver)
        self.boundary = None
        self.pending = None
        self.key_plan = []
        self.keys = []
        self.calls = []
        self.result_list = 0
        self.tranche_start = self.cursor

    def values(self, p):
        assert p in self.collections, self.labels.get(p, hex(p))
        if self.kinds[p] == 'array':
            return list(self.collections[p])
        return [self.rq(self.rq(p+0x10)+0x20+i*8) for i in range(self.rd(p+0x18))]

    def snapshot(self):
        out = super().snapshot()
        if not hasattr(self, 'pools'):
            return out
        out.update(pools={self.labels[p]: {'identity': self.labels[p], 'items': self.norm(self.values(p)),
                                          'version': self.rd(p+0x1C)} for p in self.pools},
                   roster_identities=[self.labels[self.rq(self.gameplay+0x28+i*8)] for i in range(4)],
                   board_order=[self.labels[p] for p in self.values(self.board)],
                   returned_list=self.labels.get(self.result_list) if self.result_list else None,
                   returned_order=self.norm(self.values(self.result_list)) if self.result_list else [],
                   allocated_collection_views={self.labels[p]: {
                       'kind': self.kinds[p], 'items': self.norm(self.values(p)),
                       'version': self.rd(p+0x1C) if self.kinds[p] == 'list' else None}
                       for p in self.collections if p >= self.tranche_start},
                   pending_service=None if not self.pending else {
                       'kind': self.pending['kind'], 'index': self.pending['index'],
                       'items': self.norm(self.pending['items']), 'results': list(self.pending['results'])})
        counts = self.rq(self.gameplay_static+0x30)
        out['current_script_identity'] = self.labels.get(counts) if counts else None
        out['current_script_fields'] = {field: self.rd(counts+0x10+i*4) for i, field in enumerate(COUNT_FIELDS)} if counts else None
        return out

    def service(self, name, **details):
        self.events.append({'service': name, 'caller_return_rva': hex(self.rq(self.uc.reg_read(self.x.UC_X86_REG_RSP))-self.base),
                            **details, 'snapshot': self.snapshot() if self.record_events else None})
        if self.stop_service == len(self.events):
            self.failure = 'service:'+name
            self.uc.emu_stop()
            return False
        return True

    def callback_step(self):
        p = self.pending
        if p['index'] == len(p['items']):
            self.uc.reg_write(self.x.UC_X86_REG_RSP, p['sp'])
            if p['kind'] == 'predicate':
                before = p['items']
                after = [item for item, found in zip(before, p['results']) if not found]
                self.write_list(p['source'], after, self.rd(p['source']+0x1C)+int(after != before))
                self.pending = None
                self.ret(len(before)-len(after))
            else:
                order = sorted(range(len(p['items'])), key=lambda i: (p['results'][i], i))
                self.result_list = self.list([p['items'][i] for i in order], 'roster_return')
                self.pending = None
                self.ret(self.result_list)
            return
        sp = p['sp']-0x100
        self.q(sp, self.stop+0x200)
        self.uc.reg_write(self.x.UC_X86_REG_RSP, sp)
        for reg, value in ((self.x.UC_X86_REG_RCX, p['receiver']),
                           (self.x.UC_X86_REG_RDX, p['items'][p['index']]),
                           (self.x.UC_X86_REG_R8, 0), (self.x.UC_X86_REG_R9, 0)):
            self.uc.reg_write(reg, value)
        self.uc.reg_write(self.x.UC_X86_REG_RIP, self.base+p['entry'])

    def hook(self, uc, address, size, user):
        if not hasattr(self, 'pools'):
            return super().hook(uc, address, size, user)
        x = self.x
        r = address-self.base
        if r in self.instructions:
            self.visited.add(r)
            if r in self.extra_names or r == self.entries['Gameplay.SetupCurrentVillageForStandard']:
                c = self.uc.reg_read(x.UC_X86_REG_RCX)
                t = self.uc.reg_read(x.UC_X86_REG_RDX)
                self.calls.append({'method': self.extra_names.get(r, 'Gameplay.SetupCurrentVillageForStandard'),
                                   'receiver': self.labels.get(c), 'argument': self.labels.get(t, t)})
            return
        c, t, m = [self.uc.reg_read(reg) for reg in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        if address == self.stop+0x200:
            p = self.pending
            assert self.uc.reg_read(x.UC_X86_REG_RSP) == p['sp']-0x100+8
            if p['kind'] == 'predicate':
                p['results'].append(self.uc.reg_read(x.UC_X86_REG_RAX)&255)
            else:
                bits = self.uc.reg_read(x.UC_X86_REG_XMM0)&0xffffffff
                p['results'].append(struct.unpack('<f', struct.pack('<I', bits))[0])
            p['index'] += 1
            self.callback_step()
            return
        if r == 0x365A20:
            self.boundary = {'kind': 'before_first_init', 'actor': self.labels[c],
                             'data': self.asset_ids[t], 'display_id': m & 0xffffffff}
            self.uc.emu_stop()
            return
        if r == 0x36E4E0:
            assert c == self.owner
            if self.service('positions_supplied'):
                self.ret()
            return
        if r in (0x281D90, 0x2B7B40):
            if self.service('class_init' if r == 0x281D90 else 'metadata_init'):
                if r == 0x281D90:
                    self.d(c+0xE0, 1)
                self.ret()
            return
        if r == 0x2B7D40:
            name = self.metadata_names.get(c)
            assert name is not None, hex(c)
            if self.service('allocation', object_type=name):
                if name == 'System.Collections.Generic.List<CharacterData>_TypeInfo':
                    p = self.list([], 'allocated:'+str(self.cursor-self.arena))
                else:
                    p = self.allocate(name+':object:'+str(self.cursor-self.arena))
                self.q(p, c)
                self.ret(p)
            return
        if r == 0xA651A0:
            assert self.metadata_names.get(m) == 'Method$Gameplay.<>c.<GetRandomCharacters>b__65_0()', self.metadata_names.get(m)
            if self.service('sort_delegate_ctor', receiver=self.labels.get(t), metadata=self.metadata_names.get(m)):
                self.q(c+0x10, t)
                self.q(c+0x18, self.base+0x392D50)
                self.ret()
            return
        if r == 0x628D70:
            assert c in self.collections and self.rq(t+0x18) == self.base+0x392D50
            if self.service('linq_orderby_supplied', source=self.norm(self.values(c))):
                p = self.allocate('ordered_roster')
                self.q(p+0x10, c)
                self.q(p+0x18, t)
                self.ret(p)
            return
        if r == 0x62E890:
            source, delegate = self.rq(c+0x10), self.rq(c+0x18)
            if self.service('linq_tolist_supplied', source=self.norm(self.values(source))):
                self.pending = {'kind': 'sort', 'source': source, 'items': self.values(source),
                                'receiver': self.rq(delegate+0x10), 'entry': 0x392D50,
                                'index': 0, 'results': [], 'sp': self.uc.reg_read(x.UC_X86_REG_RSP)}
                self.callback_step()
            return
        if r == 0x1C86710:
            assert self.pending and self.pending['kind'] == 'sort'
            index = self.pending['index']
            value = self.key_plan[index]
            bits = struct.unpack('<I', struct.pack('<f', value))[0]
            if self.service('random_value_supplied', source_ordinal=index, bits=bits):
                self.keys.append({'source_ordinal': index, 'asset_id': self.asset_ids[self.pending['items'][index]], 'bits': bits})
                self.uc.reg_write(x.UC_X86_REG_XMM0, bits)
                self.ret()
            return
        if r == 0xC8B620:
            if self.service('predicate_ctor'):
                self.q(c+0x10, t)
                self.ret()
            return
        if r == 0xB59980:
            if self.service('remove_all_deferred_commit'):
                self.pending = {'kind': 'predicate', 'source': c, 'items': self.values(c),
                                'receiver': self.rq(t+0x10), 'entry': 0x377170,
                                'index': 0, 'results': [], 'sp': self.uc.reg_read(x.UC_X86_REG_RSP)}
                self.callback_step()
            return
        if r == 0x112B9D0:
            if self.service('array_clear', count=m):
                self.uc.mem_write(c+0x20+t*8, bytes(m*8))
                self.ret()
            return
        if r == 0x1C82480:
            if self.service('unity_liveness_supplied', left=self.labels.get(c), right=self.labels.get(t)):
                self.ret(int(c != t))
            return
        if r == 0x1C86600:
            choice = self.choices.pop(0) if self.choices else 0
            # Native RollDice supplies [1,11), while list draws supply [0,n).
            assert c <= choice < t, (c, t, choice)
            draw = {'minimum': c, 'maximum_exclusive': t, 'width': t-c, 'index': choice,
                    'caller_return_rva': hex(self.rq(self.uc.reg_read(x.UC_X86_REG_RSP))-self.base)}
            self.draws.append(draw)
            if self.service('rng', **draw):
                self.ret(choice)
            return
        if r in (0xB610A0, 0xB53F50, 0xB55950, 0x2EB0, 0xB01F50, 0xB59E70, 0xB22150):
            for p in (c, t):
                if p in self.collections:
                    self.collections[p] = self.values(p)
        if r == 0x9693D0 and self.rq(c) in self.collections:
            p = self.rq(c)
            assert self.rd(c+12) == self.rd(p+0x1C), 'enumerated list changed'
            self.collections[p] = self.values(p)
        return super().hook(uc, address, size, user)

    def run(self, name, this, argument=0, choices=(), stop_service=None, record_events=True, keys=None):
        self.boundary, self.pending, self.result_list = None, None, 0
        self.calls, self.keys = [], []
        self.key_plan = list(keys if keys is not None else [i/8 for i in range(5)])
        # Base native run performs return ABI checks; a before-Init boundary is
        # an intentional nonreturning stop, so use the same frame explicitly.
        self.events, self.draws, self.choices, self.failure = [], [], list(choices), None
        self.native_calls = []
        self.stop_service, self.record_events = stop_service, record_events
        x = self.x
        sp = self.stack+0x10008
        self.q(sp, self.stop)
        nonvolatiles = (x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                        x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15)
        for index, reg in enumerate(nonvolatiles):
            self.uc.reg_write(reg, 0xABC000+index)
        for reg, value in ((x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, this), (x.UC_X86_REG_RDX, argument),
                           (x.UC_X86_REG_R8, 0), (x.UC_X86_REG_R9, 0), (x.UC_X86_REG_RAX, 0),
                           (x.UC_X86_REG_R10, 0), (x.UC_X86_REG_R11, 0)):
            self.uc.reg_write(reg, value)
        self.uc.emu_start(self.base+self.entries[name], self.stop, timeout=5_000_000, count=500000)
        if not self.failure and not self.boundary:
            assert self.uc.reg_read(x.UC_X86_REG_RIP) == self.stop
            assert self.uc.reg_read(x.UC_X86_REG_RSP) == sp+8
            assert all(self.uc.reg_read(reg) == 0xABC000+i for i, reg in enumerate(nonvolatiles))
        if not self.failure:
            assert not self.choices, self.choices
        returned = self.uc.reg_read(x.UC_X86_REG_RAX)
        return {'method': name, 'failure': self.failure, 'boundary': self.boundary,
                'returned': self.asset_ids.get(returned, self.labels.get(returned, returned)),
                'draws': list(self.draws), 'sort_keys': list(self.keys), 'native_calls': list(self.calls),
                'services': list(self.events), 'final': self.snapshot()}


def inspect(inputs):
    raw, meta, dump, pe, cs, lock = inputs
    starts = [0x37CE10, 0x3E49F0, 0x396840, 0x36C7A0, 0x36C810,
              0x36BF80, 0x37B370]
    methods = [r for r in meta['ScriptMethod']
               if 'GetRandomCharacters' in r['Name']]
    starts += [r['Address'] for r in methods]
    rows = []
    for start in sorted(set(starts)):
        end = min(r['Address'] for r in meta['ScriptMethod'] if r['Address'] > start)
        ins = list(cs.disasm(pe.get_data(start, end-start), start))
        calls = []
        for i in ins:
            if i.mnemonic not in ('call', 'jmp'):
                continue
            if i.op_str.startswith('0x'):
                target = int(i.op_str, 16)
                aliases = [r['Name'] for r in meta['ScriptMethod'] if r['Address'] == target]
                calls.append({'site': hex(i.address), 'kind': i.mnemonic,
                              'target': hex(target), 'aliases': aliases[:3]})
        rows.append({'rva': hex(start), 'methods': [r for r in meta['ScriptMethod'] if r['Address'] == start], 'calls': calls})
    return rows


def audit(game_root, dumper_root, probe=False):
    inputs = load_inputs(game_root, dumper_root)
    profiles = json.loads((ROOT/f'reports/{BUILD}_ascension_assets_audit.json').read_text(encoding='utf-8'))
    characters = json.loads((ROOT/f'reports/{BUILD}_character_assets_audit.json').read_text(encoding='utf-8'))
    from audit_ascension_assets import audit as profile_audit
    from audit_character_assets import audit as character_audit
    if not probe:
        assert profiles == json.loads(json.dumps(profile_audit(game_root, dumper_root)))
        assert characters == json.loads(json.dumps(character_audit(game_root, dumper_root)))
    runner = BluffJoin(inputs, profiles, characters)
    steps = []
    for name, this, argument, choices in (
        ('GameData.SetupCurrentAscension', runner.game, 0, []),
        ('AscensionsData.ClearCurrentPickedScript', runner.temporary, 0, []),
        ('AscensionsData.SetupCharactersCount', runner.temporary, 0, [0]),
        ('AscensionsData.SetupStartingCharacters', runner.temporary, 0, []),
        ('Gameplay.GetCurrentScript', runner.gameplay, 0, [])):
        result = runner.run(name, this, argument, choices)
        assert result['failure'] is None and result['boundary'] is None
        steps.append(result)
    counts = runner.uc.reg_read(runner.x.UC_X86_REG_RAX)
    runner.q(runner.gameplay_static+0x30, counts)
    runner.tranche_start = runner.cursor
    materialized = runner.save()
    generation = runner.run('Gameplay.GetRandomCharacters', runner.gameplay, 5)
    assert not generation['failure']
    returned = runner.uc.reg_read(runner.x.UC_X86_REG_RAX)
    pool = runner.run('Characters.ManageCharacters', runner.owner, returned)
    assert not pool['failure'] and pool['boundary']['kind'] == 'before_first_init'
    pool_saved = runner.save()
    duplicate = runner.run('Minion.GetBluffIfAble', runner.minion_role, runner.actors[0], choices=[1, 0])
    runner.restore(pool_saved)
    unique = runner.run('Minion.GetBluffIfAble', runner.minion_role, runner.actors[0], choices=[5, 0])
    assert not duplicate['failure'] and not unique['failure'], (duplicate['failure'], unique['failure'], duplicate['draws'], unique['draws'])
    report = {'schema_version': 1, 'build_id': BUILD, 'setup': steps,
            'generation': generation, 'pool': pool, 'duplicate': duplicate, 'unique': unique,
            'bodies': runner.body_evidence}
    if probe:
        report['executed_instruction_count'] = len(runner.visited)
        return report

    def digest(value):
        return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'), ensure_ascii=True).encode('utf-8')).hexdigest()

    def random_service_order(result):
        allowed = ('rng', 'sort_delegate_ctor', 'linq_orderby_supplied', 'linq_tolist_supplied', 'random_value_supplied')
        return [{'service_ordinal': i+1, **{key: value for key, value in event.items()
                 if key in ('service', 'caller_return_rva', 'minimum', 'maximum_exclusive', 'width', 'source_ordinal', 'metadata')}}
                for i, event in enumerate(result['services']) if event['service'] in allowed]

    # These are separate exhaustive factors, not an exhaustive Cartesian
    # product of all roster, key, pool and selector histories.
    generation_rows, pool_rows, sort_rows, selector_rows, stops = [], [], [], [], []
    generation_sites = [{k: v for k, v in d.items() if k != 'index'} for d in generation['draws']]
    generation_random_order = random_service_order(generation)
    choice_domains = [range(d['minimum'], d['maximum_exclusive']) for d in generation['draws']]
    ordered_rosters = {}
    subset_representatives = {}
    for choices in itertools.product(*choice_domains):
        runner.restore(materialized)
        result = runner.run('Gameplay.GetRandomCharacters', runner.gameplay, 5, choices, record_events=False)
        assert result['failure'] is None
        assert [{k: v for k, v in d.items() if k != 'index'} for d in result['draws']] == generation_sites
        assert sum(c['method'] == 'Gameplay.SetupCurrentVillageForStandard' for c in result['native_calls']) == 1
        rosters = result['final']['rosters']
        assert rosters[1:] == [[], [21596], []] and len(rosters[0]) == len(set(rosters[0])) == 4
        assert set(result['final']['returned_order']) == set(rosters[0]+[21596])
        assert len(result['sort_keys']) == 5
        assert random_service_order(result) == generation_random_order
        row = {'choices': list(choices), 'current_villagers': rosters[0],
               'returned_order': result['final']['returned_order'], 'sort_keys': result['sort_keys'],
               'service_count': len(result['services']), 'native_call_trace_sha256': digest(result['native_calls'])}
        generation_rows.append(row)
        key = tuple(rosters[0])
        if key not in ordered_rosters:
            ordered_rosters[key] = {'saved': runner.save(), 'returned': runner.uc.reg_read(runner.x.UC_X86_REG_RAX),
                                    'row_index': len(generation_rows)-1, 'choices': list(choices)}
        subset = tuple(sorted(key))
        if subset not in subset_representatives:
            subset_representatives[subset] = key
    print('Completed exhaustive observed roster-index factor: '+str(len(generation_rows)), flush=True)

    pool_sites = [{k: v for k, v in d.items() if k != 'index'} for d in pool['draws']]
    assert pool_sites[0]['width'] == 1
    fallback_width = pool_sites[1]['width']
    duplicate_domains = [range(d['minimum'], d['maximum_exclusive']) for d in pool['draws'][2:]]
    duplicate_choices = list(itertools.product(*duplicate_domains))
    catalogue = set()
    witnesses = {}

    def acquire(saved, actor, die, index, pool_row_index, factor, full=False):
        runner.restore(saved)
        result = runner.run('Minion.GetBluffIfAble', runner.minion_role, actor,
                            choices=[die, index], record_events=full)
        assert not result['failure'] and not result['boundary']
        assert len(result['draws']) == 2
        assert result['draws'][0]['minimum'] == 1 and result['draws'][0]['maximum_exclusive'] == 11
        before = result['final']['rosters'][0]
        selected = result['returned']
        expected_pool = result['final']['pools']['duplicate_pool' if die <= 4 else 'unique_pool']['items']
        assert selected == expected_pool[index]
        original_roster = pool_rows[pool_row_index]['current_villagers']
        expected_roster = original_roster if die <= 4 or selected in original_roster else original_roster+[selected]
        assert before == expected_roster
        assert result['final']['rosters'][1:] == [[], [21596], []]
        catalogue.add(selected)
        row = {'pool_row': pool_row_index, 'factor': factor, 'die': die, 'pool_index': index,
               'returned_asset': selected, 'final_current_villagers': before,
               'draws': result['draws'], 'native_call_trace_sha256': digest(result['native_calls'])}
        selector_rows.append(row)
        if selected not in witnesses:
            # Complete acquisition chronology is retained once per catalogue
            # identity and linked to its actually generated/pool factor row.
            runner.restore(saved)
            complete = runner.run('Minion.GetBluffIfAble', runner.minion_role, actor,
                                  choices=[die, index])
            assert complete['returned'] == selected and not complete['failure']
            witnesses[selected] = {'pool_row': pool_row_index, 'selector': complete}
        return result

    def execute_pool(info, choices, factor, full=False):
        runner.restore(info['saved'])
        returned = info['returned']
        returned_order = runner.norm(runner.values(returned))
        result = runner.run('Characters.ManageCharacters', runner.owner, returned,
                            choices=choices, record_events=full)
        assert not result['failure'] and result['boundary'] == {
            'kind': 'before_first_init', 'actor': 'actor:0', 'data': returned_order[0], 'display_id': len(returned_order)}
        assert [{k: v for k, v in d.items() if k != 'index'} for d in result['draws']] == pool_sites
        rosters = result['final']['rosters']
        assert tuple(rosters[0]) in ordered_rosters
        pools = result['final']['pools']
        assert len(pools['unique_pool']['items']) == 2
        assert sorted(pools['duplicate_pool']['items']) == sorted(rosters[0])
        assert pools['must_include']['items'] == []
        pool_rows.append({'generation_row': info['row_index'], 'factor': factor, 'choices': list(choices),
                          'returned_order': returned_order, 'current_villagers': rosters[0],
                          'pool_identities_and_contents': pools, 'boundary': result['boundary'],
                          'native_call_trace_sha256': digest(result['native_calls'])})
        saved = runner.save()
        actor = runner.actors[returned_order.index(21596)]
        return result, saved, actor, len(pool_rows)-1

    for ordinal, (ordered, info) in enumerate(ordered_rosters.items()):
        canonical = subset_representatives[tuple(sorted(ordered))] == ordered
        if not canonical:
            result, saved, actor, index = execute_pool(info, [0]*len(pool_sites), 'ordered_roster_default_pool')
            acquire(saved, actor, 1, 0, index, 'ordered_roster_default_duplicate')
            acquire(saved, actor, 5, 1, index, 'ordered_roster_default_unique')
            continue
        for fallback_index in range(fallback_width):
            choices = [0, fallback_index]+[0]*len(duplicate_domains)
            result, saved, actor, index = execute_pool(info, choices, 'fallback_indices')
            acquire(saved, actor, 5, 1, index, 'fallback_catalogue')
            for die in range(1, 11):
                width = len(result['final']['pools']['duplicate_pool' if die <= 4 else 'unique_pool']['items'])
                for selected in range(width):
                    acquire(saved, actor, die, selected, index, 'canonical_subset_all_minion_indices')
        for selected in duplicate_choices:
            if not any(selected):
                continue  # Already executed at fallback index zero.
            result, saved, actor, index = execute_pool(info, [0, 0, *selected], 'duplicate_orders')
            for selected_index in range(len(result['final']['pools']['duplicate_pool']['items'])):
                acquire(saved, actor, 1, selected_index, index, 'duplicate_order_indices')
        print('Completed one four-Villager subset pool/index factor', flush=True)

    # All distinct sort-key rank orders, plus stable all-equal and each single
    # pair tie, are run for one actual generation history per four-role subset.
    sort_plans = [('distinct', tuple(p)) for p in itertools.permutations(range(5))]
    sort_plans.append(('all_equal', (0, 0, 0, 0, 0)))
    for left, right in itertools.combinations(range(5), 2):
        ranks = list(range(5))
        ranks[right] = ranks[left]
        sort_plans.append(('pair_tie:'+str(left)+':'+str(right), tuple(ranks)))
    for subset, ordered in subset_representatives.items():
        source = ordered_rosters[ordered]
        for family, ranks in sort_plans:
            runner.restore(materialized)
            result = runner.run('Gameplay.GetRandomCharacters', runner.gameplay, 5,
                                source['choices'], keys=[rank/8 for rank in ranks], record_events=False)
            assert not result['failure']
            assert random_service_order(result) == generation_random_order
            keys = result['sort_keys']
            source_items = [r['asset_id'] for r in keys]
            expected = [source_items[i] for i in sorted(range(5), key=lambda i: (ranks[i], i))]
            assert result['final']['returned_order'] == expected
            info = {'saved': runner.save(), 'returned': runner.uc.reg_read(runner.x.UC_X86_REG_RAX),
                    'row_index': source['row_index']}
            pool_result, saved, actor, index = execute_pool(info, [0]*len(pool_sites), 'sorted_placement_default_pool')
            acquire(saved, actor, 1, 0, index, 'sorted_placement_duplicate')
            acquire(saved, actor, 5, 1, index, 'sorted_placement_unique')
            sort_rows.append({'generation_row': source['row_index'], 'family': family, 'key_ranks': list(ranks),
                              'sort_keys': keys, 'returned_order': expected, 'pool_row': index})
    print('Completed actual sorted placements and supplied Minion invocations: '+str(len(sort_rows)), flush=True)

    # Stop every reached service of full original baseline generation, pool and
    # each selector branch. Compare the actual recorded chronology and modeled
    # collection/roster/cache/pool state at the interrupted service entry.
    runner.restore(materialized)
    generation_full = runner.run('Gameplay.GetRandomCharacters', runner.gameplay, 5)
    generation_saved = runner.save()
    returned = runner.uc.reg_read(runner.x.UC_X86_REG_RAX)
    pool_full = runner.run('Characters.ManageCharacters', runner.owner, returned)
    pool_saved = runner.save()
    baselines = [(materialized, 'Gameplay.GetRandomCharacters', runner.gameplay, 5, [], generation_full),
                 (generation_saved, 'Characters.ManageCharacters', runner.owner, returned, [], pool_full)]
    for die in (1, 5):
        runner.restore(pool_saved)
        full = runner.run('Minion.GetBluffIfAble', runner.minion_role, runner.actors[0], choices=[die, 0])
        baselines.append((pool_saved, 'Minion.GetBluffIfAble', runner.minion_role, runner.actors[0], [die, 0], full))
    for saved, name, this, argument, choices, full in baselines:
        for ordinal in range(1, len(full['services'])+1):
            runner.restore(saved)
            partial = runner.run(name, this, argument, choices, stop_service=ordinal)
            assert partial['failure'] == 'service:'+full['services'][ordinal-1]['service']
            assert partial['services'] == full['services'][:ordinal]
            assert partial['final'] == full['services'][ordinal-1]['snapshot']
            stops.append({'method': name, 'choices': choices, 'stop_service_ordinal': ordinal,
                          'failure': partial['failure'], 'final': partial['final'],
                          'full_service_prefix_sha256': digest(partial['services'])})
    record_by_id = {r['path_id']: r for r in characters['records']}
    source = next(r['data'] for r in profiles['ascensions'] if r['path_id'] == 21674)
    fallback_ids = [r[1] for r in source['townsfolks']+source['outsiders']+source['minions']+source['townsfolks']]
    eligible = [i for i in fallback_ids if record_by_id[i]['bluffable'] and record_by_id[i]['startingAlignment'] == 10 and record_by_id[i]['type'] == 10]
    assert len(eligible) == fallback_width
    assert catalogue == set(eligible)
    report.update(generation_index_factor={'draw_sites_and_bounds': generation_sites,
                                          'random_service_order': generation_random_order, 'cases': generation_rows},
                  pool_index_factors={'draw_sites_and_bounds': pool_sites, 'cases': pool_rows},
                  sort_order_factor=sort_rows, supplied_minion_invocations=selector_rows,
                  native_catalogue_witnesses={str(i): witnesses[i] for i in sorted(witnesses)},
                  candidate_catalogue=[{'path_id': i, 'object_name': record_by_id[i]['name'],
                                        'public_name': record_by_id[i]['characterName'],
                                        'managed_role': record_by_id[i]['role_type'],
                                        'type': record_by_id[i]['type'], 'starting_alignment': record_by_id[i]['startingAlignment'],
                                        'fallback_occurrences': eligible.count(i)} for i in sorted(catalogue)],
                  fallback_source_occurrences=fallback_ids, filtered_fallback_occurrences=eligible,
                  stopped_prefixes=stops,
                  source_hashes={'game_assembly': inputs[-1]['inputs']['game_assembly']['sha256'],
                                 'script_json': hashlib.sha256((dumper_root/'script.json').read_bytes()).hexdigest(),
                                 'dump_cs': hashlib.sha256((dumper_root/'dump.cs').read_bytes()).hexdigest(),
                                 'assets': profiles['source_hashes'],
                                 'script': hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                                 'dependencies': {name: hashlib.sha256((Path(__file__).parent/name).read_bytes()).hexdigest()
                                                  for name in ('audit_first_village_profile_generation.py', 'audit_manage_pool_composition.py',
                                                               'audit_character_assets.py', 'audit_ascension_assets.py', 'audit_report_snapshots.py')}},
                  counters={'generation_index_cases': len(generation_rows), 'ordered_standard_rosters': len(ordered_rosters),
                            'four_villager_subsets': len(subset_representatives), 'pool_factor_cases': len(pool_rows),
                            'sorted_placement_cases': len(sort_rows), 'minion_invocations': len(selector_rows),
                            'stopped_prefixes': len(stops), 'candidate_assets': len(catalogue),
                            'filtered_fallback_occurrences': len(eligible), 'distinct_executed_instructions': len(runner.visited)},
                  instruction_assertions=[{'rva': hex(a), 'mnemonic': v[0], 'operands': v[1]} for a, v in runner.assertions.items()],
                  scope={'profile': 21674, 'group': 0, 'village': 0, 'mode': 'supplied RoguelikeStandard returning Standard0',
                         'accumulation_invocations': 0, 'copy_service': 'field_faithful JSON-result output',
                         'sort_service': 'deferred key evaluation in source occurrence order; stable ascending float keys; normal materialization',
                         'pool_stop': 'before first Character.Init', 'minion_dispatch': 'separately supplied invocation, no intervening pool/script writers',
                         'factoring': 'Exhaustive factors only. Noncanonical final-selection/sort/pool Cartesian combinations and intervening Init/Start/queue/Reveal are unverified.',
                         'probability': 'No world prior, Unity RNG state, seed reachability or joint weights recovered.'})
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inspect', action='store_true')
    parser.add_argument('--probe', action='store_true')
    args = parser.parse_args()
    result = inspect(load_inputs(args.game_root, args.dumper_root)) if args.inspect else audit(args.game_root, args.dumper_root, args.probe)
    if args.inspect:
        print(json.dumps(result, indent=2, ensure_ascii=True))
    elif args.probe:
        print(json.dumps({key: {'failure': result[key]['failure'], 'draws': result[key]['draws'],
                               'final': result[key]['final']} for key in ('generation', 'pool', 'duplicate', 'unique')}, ensure_ascii=True))
    else:
        assert args.output
        packed = pool_snapshots(result)
        assert expand_snapshots(packed) == result
        args.output.write_text(json.dumps(packed, sort_keys=True, separators=(',', ':'), ensure_ascii=True)+'\n', encoding='utf-8')
        print('Verified original N5 generation, actual pools and supplied ordinary Minion invocation')
