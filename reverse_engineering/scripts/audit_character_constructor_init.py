"""Join native Character construction, initialization, Hidden refresh and first yield."""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_constructor import Machine as ConstructorMachine


class Machine(ConstructorMachine):
    ENTRIES = {'Init': 0x365A20, 'InitWithNoReset': 0x365720, 'RefreshCharacter': 0x367970,
               'RefreshView': 0x367B60, 'FirstYield': 0x3756B0}

    def __init__(self, game_root, dumper_root):
        import capstone
        self.phase = 'Construct'
        super().__init__(game_root, dumper_root)
        self.constructor_decoded = set(self.decoded)
        root = Path(__file__).parents[1]
        extraction = json.loads((root / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        raw = (Path(dumper_root) / 'script.json').read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs']['script_json']['sha256'].upper()
        metadata = json.loads(raw.decode('utf-8-sig'))
        previous = json.loads((root / f'reports/{BUILD}_character_init.json').read_text(encoding='utf-8'))
        assert previous['build'] == BUILD and previous['case_count'] == 475
        dump = (Path(dumper_root) / 'dump.cs').read_bytes()
        assert hashlib.sha256(dump).hexdigest().upper() == extraction['outputs']['dump_cs']['sha256'].upper()
        dump = dump.decode('utf-8-sig')
        for name, declarations in previous['fields'].items():
            body = re.search(r'^[^\n]*class ' + re.escape(name) + r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', dump, re.M | re.S)
            assert body and all(declaration in body[1] for declaration in declarations)
        self.methods, self.address_phase = [], {}
        cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        cs.detail = True
        for name, entry in self.ENTRIES.items():
            symbol = 'Character.<DelayReveal>d__84$$MoveNext' if name == 'FirstYield' else 'Character$$' + name
            rows = [m for m in metadata['ScriptMethod'] if m['Name'] == symbol and m['Address'] == entry]
            assert len(rows) == 1
            self.methods.append(rows[0])
            end = min(m['Address'] for m in metadata['ScriptMethod'] if m['Address'] > entry)
            instructions = list(cs.disasm(self.pe.get_data(entry, end - entry), entry))
            while instructions[-1].mnemonic == 'int3': instructions.pop()
            assert all(a.address + a.size == b.address for a, b in zip(instructions, instructions[1:]))
            for i in instructions:
                self.decoded[i.address] = i
                self.address_phase[i.address] = name
        noop = list(cs.disasm(self.pe.get_data(0x33ED50, 3), 0x33ED50))
        assert [(i.mnemonic, i.op_str) for i in noop] == [('ret', '0')]
        self.decoded[0x33ED50] = noop[0]
        self.join_checks = {0x365CDF: ('inc', 'dword ptr [rax + 0x1c]'),
                            0x365CE2: ('mov', 'dword ptr [rax + 0x18], r15d'),
                            0x365CC2: ('mov', 'rax, qword ptr [rdi + 0xf0]'),
                            0x367A07: ('cmp', 'dword ptr [rax + 0x2c], 0x14'),
                            0x3756F4: ('mov', 'dword ptr [rdi + 0x10], 0xffffffff'),
                            0x37572E: ('mov', 'qword ptr [rcx], rax'),
                            0x375769: ('mov', 'dword ptr [rdi + 0x10], 1'),
                            0x367B91: ('cmp', 'dword ptr [rbx + 0xdc], 0'),
                            0x367BBE: ('cmp', 'dword ptr [rbx + 0xe4], 0x14'),
                            0x367D7A: ('cmp', 'byte ptr [rbx + 0xed], 0')}
        for address, expected in self.join_checks.items():
            assert address in self.decoded and (self.decoded[address].mnemonic, self.decoded[address].op_str) == expected
        slots = set(self.slot_values)
        self.all_flags = set(self.flags)
        for i in self.decoded.values():
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    address = i.address + i.size + op.mem.disp
                    if i.mnemonic == 'cmp' and op.size == 1: self.all_flags.add(address)
                    elif i.mnemonic in ['lea', 'mov'] and op.size == 8: slots.add(address)
        rows = [r for section in ['ScriptMetadata', 'ScriptMetadataMethod', 'ScriptString'] for r in metadata[section] if r['Address'] in slots]
        assert {r['Address'] for r in rows} == slots
        self.all_bindings = rows
        self.u.mem_map(self.arena + 0x20000, 0x60000)
        for index, row in enumerate(rows):
            if row['Address'] not in self.slot_values:
                self.slot_values[row['Address']] = self.arena + 0x40000 + index * 0x400
        self.types = {r['Name']: self.slot_values[r['Address']] for r in rows if 'Name' in r}
        self.literals = {r['Value']: self.slot_values[r['Address']] for r in rows if 'Value' in r}
        assert {'INIT: ', '# {0}', ''} <= set(self.literals)
        assert self.literals[''] == self.empty_string
        seconds = self.decoded[0x375742]
        address = seconds.address + seconds.size + seconds.operands[1].mem.disp
        assert struct.unpack('<I', self.pe.get_data(address, 4))[0] == 0x3E99999A
        self.data = [self.arena + 0x12000, self.arena + 0x13000]
        self.source_roles = [self.arena + 0x14000, self.arena + 0x15000]
        self.acted, self.number, self.number_class = self.arena + 0x16000, self.arena + 0x17000, self.arena + 0x18000
        self.status, self.active, self.resistant = self.arena + 0x19000, self.arena + 0x1A000, self.arena + 0x1B000
        self.alternate_status, self.alternate_active = self.arena + 0x20000, self.arena + 0x21000
        self.callback, self.callback_target = self.arena + 0x22000, self.arena + 0x23000
        self.pickeds, self.gameplay_static = self.arena + 0x24000, self.arena + 0x25000
        self.controls = [self.arena + 0x26000 + i * 0x100 for i in range(6)]
        self.picked_controls = self.controls[:2]
        self.pickable, self.rip, self.disguise, self.old_death = self.controls[2:]
        self.text, self.name, self.boxed = self.arena + 0x27000, self.arena + 0x28000, self.arena + 0x29000
        self.join_services = {0x2B7B40: 'metadata', 0x281D90: 'class_init', 0x2B6FF0: 'barrier',
                              0x1C79FD0: 'get_game_object', 0x1C7D810: 'set_active', 0x112B9D0: 'array_clear',
                              0x1C82480: 'unity_live', 0x1C822C0: 'unity_null', 0x1C80520: 'destroy',
                              0xF71C60: 'concat', 0x1C4B380: 'context_log', 0x1C4B450: 'log',
                              0x282580: 'box', 0xF74DF0: 'format', 0x1C7F160: 'start_coroutine',
                              0x2B7D40: 'allocate', 0x603240: 'clone_role', 0x1C961F0: 'wait_constructor'}
        self.actor_fields = {'data': (0x50, 8), 'bluff': (0x58, 8), 'register_as': (0x60, 8),
                             'trailer': (0x68, 8), 'runtime': (0x70, 8), 'dead_prefab': (0x98, 8),
                             'revealed': (0xD8, 1), 'uses': (0xDC, 4), 'previous': (0xE0, 4), 'state': (0xE4, 4),
                             'killed_hidden': (0xEC, 1), 'killed_demon': (0xED, 1), 'alignment': (0xF8, 4),
                             'id': (0x118, 4), 'started': (0x11C, 1), 'acted_infos': (0x148, 8), 'hover_infos': (0x150, 8),
                             'role': (0x168, 8), 'bluff_role': (0x170, 8), 'saved_act': (0x198, 8), 'act': (0x1A1, 1),
                             'statuses': (0xF0, 8)}

    def d(self, a, value): self.u.mem_write(a, struct.pack('<I', value & 0xFFFFFFFF))
    def rd(self, a): return int.from_bytes(self.u.mem_read(a, 4), 'little')

    def snapshot(self):
        fields = {name: int.from_bytes(self.u.mem_read(self.actor + offset, size), 'little')
                  for name, (offset, size) in self.actor_fields.items()}
        lists = []
        for record in self.lists:
            p = record['identity']
            lists.append({'identity': p, 'constructed': record['constructed'], 'backing': self.rq(p + 0x10),
                          'count': self.rd(p + 0x18) if record['constructed'] else None,
                          'version': self.rd(p + 0x1C) if record['constructed'] else None})
        return {'actor': fields, 'lists': lists,
                'statuses': [{'identity': p, 'count': self.rd(p + 0x18), 'version': self.rd(p + 0x1C),
                              'backing_values': list(struct.unpack('<iii', self.u.mem_read(p + 0x120, 12)))}
                             for p in [self.active, self.alternate_active]],
                'controls': [{'identity': p, 'active': active} for p, active in sorted(self.ui.items())],
                'continuations': [{'identity': p, 'actor': self.rq(p + 0x20), 'state': self.rd(p + 0x10),
                                   'current': self.rq(p + 0x18)} for p in self.continuations],
                'metadata_flags': [{'rva': hex(p), 'initialized': int.from_bytes(self.u.mem_read(self.base + p, 1), 'little')}
                                   for p in sorted(self.all_flags)]}

    def event(self, kind, **details):
        key = (self.phase, kind)
        self.phase_counts[key] = self.phase_counts.get(key, 0) + 1
        self.counts[kind] = self.counts.get(kind, 0) + 1
        self.events.append({'phase': self.phase, 'kind': kind, **details, 'snapshot': self.snapshot()})
        self.byte_prefixes.append(bytes(self.u.mem_read(self.actor, 0x1B8)))
        if self.options.get('failure') == [self.phase, kind, self.phase_counts[key]]:
            self.error = kind
            self.u.emu_stop()
            return False
        effect = self.options.get('effects', {}).get(f'{self.phase}:{kind}:{self.phase_counts[key]}', {})
        for field, value in effect.items():
            offset, size = self.actor_fields[field]
            self.u.mem_write(self.actor + offset, int(value).to_bytes(size, 'little'))
        return True

    def hook(self, uc, address, size, data):
        rva = address - self.base
        if address == self.stop:
            self.returned = True
            uc.emu_stop()
            return
        if rva in self.constructor_decoded or self.phase == 'Construct' and rva in self.services:
            self.phase = 'Construct'
            return super().hook(uc, address, size, data)
        if rva in self.decoded:
            self.visited.add(rva)
            if rva in self.address_phase:
                self.phase = f'{self.address_phase[rva]}#{self.occurrence}'
            if rva in [0x367970, 0x367B60]:
                assert self.rd(self.actor + 0xE4) == 5
                self.event('native_refresh_entry', receiver=self.actor)
            return
        x = self.x
        c, dx, r8 = self.reg(x.UC_X86_REG_RCX), self.reg(x.UC_X86_REG_RDX), self.reg(x.UC_X86_REG_R8)
        if address == self.stop + 0x100:
            assert c == self.callback_target and dx == self.callback_target + 0x80
            if self.event('state_callback'): self.ret()
            return
        if address == self.stop + 0x200:
            assert c == self.number and dx == self.text and r8 == self.number_class + 0x800
            if self.event('set_text'): self.ret()
            return
        if address == self.stop + 0x300:
            assert self.reg(x.UC_X86_REG_RAX) & 255 == 1
            assert self.rd(self.iterator + 0x10) == 1 and self.rq(self.iterator + 0x18) == self.wait
            assert self.reg(x.UC_X86_REG_RSP) == self.scheduler_sp - 0x28
            self.u.reg_write(x.UC_X86_REG_RSP, self.scheduler_sp)
            if self.event('first_yield'): self.ret(self.arena + 0x3F000 + self.occurrence * 0x100)
            return
        if rva == 0x2B7D90:
            self.error = 'null'
            uc.emu_stop()
            return
        assert rva in self.join_services, (self.phase, hex(rva))
        kind = self.join_services[rva]
        details, result = {}, 0
        if kind == 'metadata':
            assert c - self.base in self.slot_values
            details['binding'] = next(r.get('Name', r.get('Value')) for r in self.all_bindings if r['Address'] == c - self.base)
        elif kind == 'class_init':
            assert c in self.types.values()
            details['type'] = next(name for name, p in self.types.items() if p == c)
        elif kind == 'barrier':
            assert self.rq(c) == dx
            details.update(destination=c, value=dx)
        elif kind == 'get_game_object':
            assert c in [self.actor, self.acted] and dx == 0
            result = c + 0x800
            details.update(component=c, result=result)
        elif kind == 'set_active':
            assert c in self.ui and r8 == 0
            assert dx & 255 in [0, 1]
            details.update(object=c, active=bool(dx & 255))
        elif kind in ['unity_live', 'unity_null']:
            assert dx == 0 and c in [0, self.old_death, self.disguise]
            live = c == self.disguise and self.options['disguise'] == 'live' or c == self.old_death and self.options['death'] == 'live'
            result = 0xC0DE000000000000 | int(live if kind == 'unity_live' else not live)
            details.update(object=c, result=bool(result & 255))
        elif kind == 'destroy':
            assert c == self.old_death and dx == 0
        elif kind == 'concat':
            assert c == self.literals['INIT: '] and dx == self.name
            result = self.text
        elif kind == 'log':
            assert c == self.text and dx == 0
        elif kind == 'context_log':
            assert c == self.text and dx == self.actor + 0x800 and r8 == 0
        elif kind == 'box':
            assert c == self.types['int_TypeInfo']
            assert self.rd(dx) == self.current_id & 0xFFFFFFFF
            result = self.boxed
        elif kind == 'format':
            assert c == self.literals['# {0}'] and dx == self.boxed and r8 == 0
            result = self.text
        elif kind == 'array_clear':
            raise AssertionError('Constructor-produced empty actedInfos must not call Array.Clear')
        elif kind == 'allocate':
            if c == self.types['Character.<DelayReveal>d__84_TypeInfo']: result = self.iterator
            else:
                assert c == self.types['UnityEngine.WaitForSeconds_TypeInfo']
                result = self.wait
            details.update(type=next(name for name, p in self.types.items() if p == c), result=result)
        elif kind == 'start_coroutine':
            assert c == self.actor and dx == self.iterator and r8 == 0
            assert self.rd(self.iterator + 0x10) == 0 and self.rq(self.iterator + 0x20) == self.actor
            details.update(actor=c, iterator=dx)
        elif kind == 'clone_role':
            assert c == self.source_roles[self.data_index] and dx == self.types['Method$ClassConv.CreateCopyNonGeneric<Role>()']
            result = 0 if self.options['clone_null'] else self.clone
            details.update(source=c, result=result)
        elif kind == 'wait_constructor':
            assert c == self.wait and r8 == 0 and self.reg(x.UC_X86_REG_XMM1) & 0xFFFFFFFF == 0x3E99999A
            details.update(receiver=c, seconds_f32_bits=0x3E99999A)
        if not self.event(kind, **details): return
        if kind == 'class_init': self.d(c + 0xE0, 1)
        elif kind == 'set_active': self.ui[c] = bool(dx & 255)
        elif kind == 'allocate':
            self.u.mem_write(result, bytes(0x40))
            self.q(result, c)
            if result == self.iterator: self.continuations.append(result)
        elif kind == 'start_coroutine':
            self.scheduler_sp = self.reg(x.UC_X86_REG_RSP)
            sp = self.scheduler_sp - 0x30
            assert sp % 16 == 8
            self.q(sp, self.stop + 0x300)
            self.u.reg_write(x.UC_X86_REG_RSP, sp)
            self.u.reg_write(x.UC_X86_REG_RCX, self.iterator)
            self.u.reg_write(x.UC_X86_REG_RDX, 0)
            self.u.reg_write(x.UC_X86_REG_RIP, self.base + 0x3756B0)
            return
        self.ret(result)

    def invoke(self, entry, *, data=None, id=-100):
        x = self.x
        self.initial_sp = self.stack + 0x1FF08
        self.q(self.initial_sp, self.stop)
        self.u.reg_write(x.UC_X86_REG_RSP, self.initial_sp)
        self.u.reg_write(x.UC_X86_REG_RCX, self.actor)
        self.u.reg_write(x.UC_X86_REG_RDX, 0xC0DE123400000005 if data is None else data)
        self.u.reg_write(x.UC_X86_REG_R8, id & 0xFFFFFFFF)
        self.u.reg_write(x.UC_X86_REG_R9, 0xC0DE123400000006)
        registers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                     x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15] + [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        self.nonvolatile = {r: 0x1234000000000000 + i for i, r in enumerate(registers)}
        for r, value in self.nonvolatile.items(): self.u.reg_write(r, value)
        self.returned = False
        self.u.emu_start(self.base + entry, 0, count=10000)
        if self.returned:
            assert self.reg(x.UC_X86_REG_RSP) == self.initial_sp + 8
            for r, value in self.nonvolatile.items(): assert self.reg(r) == value

    def run_join(self, options):
        self.options = {'sequence': ['Init'], 'cold': False, 'id': -100, 'previous_phase': 20,
                        'ability_usage': 10, 'picked_count': 2, 'callback': True,
                        'death': 'absent', 'disguise': 'live', 'clone_null': False, **options}
        o = self.options
        self.phase, self.occurrence = 'Construct', 0
        self.events, self.counts, self.phase_counts, self.byte_prefixes = [], {}, {}, []
        self.lists, self.continuations, self.error = [], [], None
        self.ui = {p: True for p in [self.acted + 0x800, *self.controls[:-1]]}
        self.u.mem_write(self.actor, bytes([0xA5] * 0x1B8))
        bindings = {0x48: self.number, 0x78: self.rip, 0x88: 0 if o['disguise'] == 'absent' else self.disguise,
                    0x98: 0 if o['death'] == 'absent' else self.old_death, 0xA8: self.acted,
                    0xF0: self.status, 0x180: self.callback if o['callback'] else 0,
                    0x188: self.pickeds, 0x1A8: self.pickable}
        for offset, value in bindings.items(): self.q(self.actor + offset, value)
        for offset in [0xD8, 0xEC, 0xED, 0x11C]: self.u.mem_write(self.actor + offset, b'\1')
        self.d(self.actor + 0xDC, 7)
        self.d(self.actor + 0xE0, 10)
        self.d(self.actor + 0xE4, 20)
        self.d(self.actor + 0xF8, 10)
        self.d(self.actor + 0x118, 73)
        for slot, value in self.slot_values.items(): self.q(self.base + slot, value)
        for pointer in self.types.values(): self.d(pointer + 0xE0, 0 if o['cold'] else 1)
        for flag in self.all_flags: self.u.mem_write(self.base + flag, bytes([not o['cold']]))
        self.q(self.types['Gameplay_TypeInfo'] + 0xB8, self.gameplay_static)
        self.d(self.gameplay_static + 0x2C, o['previous_phase'])
        for index, pointer in enumerate(self.data):
            self.q(pointer + 0x28, self.name)
            self.d(pointer + 0x134, [20, 30][index])
            self.d(pointer + 0x138, o['ability_usage'])
            self.q(pointer + 0x140, self.source_roles[index])
        self.d(self.pickeds + 0x18, o['picked_count'])
        for index, pointer in enumerate(self.picked_controls): self.q(self.pickeds + 0x20 + index * 8, pointer)
        self.q(self.number, self.number_class)
        self.q(self.number_class + 0x558, self.stop + 0x200)
        self.q(self.number_class + 0x560, self.number_class + 0x800)
        self.q(self.callback + 0x18, self.stop + 0x100)
        self.q(self.callback + 0x28, self.callback_target + 0x80)
        self.q(self.callback + 0x40, self.callback_target)
        for component, active in [(self.status, self.active), (self.alternate_status, self.alternate_active)]:
            self.q(component + 0x10, active)
            self.q(component + 0x18, self.resistant)
            self.q(component + 0x20, self.callback_target)
            self.q(active + 0x10, active + 0x100)
            self.d(active + 0x18, 3)
            self.d(active + 0x1C, 23)
            self.u.mem_write(active + 0x120, struct.pack('<iii', 10, 30, 50))
        initial = bytes(self.u.mem_read(self.actor, 0x1B8))
        self.invoke(0x3697C0)
        stages = []
        if not self.returned:
            return {'input': o, 'returned': False, 'error': self.error, 'events': self.events, 'stages': [], 'final': self.snapshot()}
        constructed = bytes(self.u.mem_read(self.actor, 0x1B8))
        allowed = {n for offset, size, _ in self.fields.values() for n in range(offset, offset + size)}
        assert all(a == b for n, (a, b) in enumerate(zip(initial, constructed)) if n not in allowed)
        produced = self.snapshot()
        assert produced['actor']['acted_infos'] == self.fresh_lists[0] and produced['actor']['hover_infos'] == self.fresh_lists[1]
        assert produced['actor']['saved_act'] == self.empty_string and produced['actor']['act'] == produced['actor']['uses'] == 1
        stages.append({'phase': 'Construct', 'snapshot': produced})
        for self.occurrence, name in enumerate(o['sequence'], 1):
            assert name in ['Init', 'InitWithNoReset']
            self.phase, self.counts = f'{name}#{self.occurrence}', {}
            self.data_index = (self.occurrence - 1) % 2
            self.iterator = self.arena + 0x30000 + self.occurrence * 0x100
            self.wait = self.arena + 0x34000 + self.occurrence * 0x100
            self.clone = self.arena + 0x38000 + self.occurrence * 0x100
            self.current_id = o['id']
            before = bytes(self.u.mem_read(self.actor, 0x1B8))
            self.invoke(self.ENTRIES[name], data=self.data[self.data_index], id=self.current_id)
            after = bytes(self.u.mem_read(self.actor, 0x1B8))
            changed = {0x50: 8, 0x58: 8, 0x98: 8, 0xD8: 1, 0xDC: 4, 0xE0: 4, 0xE4: 4,
                       0xED: 1, 0x118: 4, 0x11C: 1, 0x168: 8}
            if name == 'Init': changed.update({0x60: 8, 0x68: 8, 0x70: 8, 0xF8: 4})
            for effect in o.get('effects', {}).values():
                changed.update({self.actor_fields[field][0]: self.actor_fields[field][1] for field in effect})
            allowed = {n for offset, size in changed.items() for n in range(offset, offset + size)}
            assert all(a == b for n, (a, b) in enumerate(zip(before, after)) if n not in allowed)
            final = self.snapshot()
            assert final['actor']['acted_infos'] == self.fresh_lists[0] and final['actor']['hover_infos'] == self.fresh_lists[1]
            assert final['actor']['saved_act'] == self.empty_string and final['actor']['act'] == 1
            assert final['lists'][1]['count'] == 0 and final['lists'][1]['version'] == 0
            if self.returned:
                assert final['lists'][0]['count'] == 0 and final['lists'][0]['version'] == self.occurrence
                assert final['actor']['state'] == 5 and final['actor']['revealed'] == 0
                assert final['continuations'][-1]['state'] == 1 and final['continuations'][-1]['actor'] == self.actor
                stages.append({'phase': f'{name}#{self.occurrence}', 'snapshot': final})
            else: break
        return {'input': o, 'returned': self.returned, 'error': self.error, 'events': self.events,
                'stages': stages, 'final': self.snapshot(), 'constructor_and_initializer_retention_verified': True}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, mutations, failures = [], [], []
    def verify_normal(result):
        o, final = result['input'], result['final']
        actor = final['actor']
        method = o['sequence'][0]
        assert result['returned'] and result['error'] is None
        assert actor['data'] == m.data[0] and actor['bluff'] == actor['revealed'] == 0
        assert actor['uses'] == 1 and actor['previous'] == 20 and actor['state'] == 5
        assert actor['killed_hidden'] == 1 and actor['killed_demon'] == actor['started'] == 0
        assert actor['id'] == (73 if o['id'] == -100 else o['id'] & 0xFFFFFFFF)
        assert actor['role'] == (0 if o['clone_null'] else m.arena + 0x38100)
        assert actor['dead_prefab'] == (m.old_death if o['death'] == 'destroyed' else 0)
        assert actor['alignment'] == (20 if method == 'Init' else 10)
        assert all(actor[field] == (0 if method == 'Init' else 0xA5A5A5A5A5A5A5A5)
                   for field in ['register_as', 'trailer', 'runtime'])
        assert final['statuses'][0]['count'] == (0 if method == 'Init' else 3)
        assert final['statuses'][0]['version'] == (24 if method == 'Init' else 23)
        controls = {r['identity']: r['active'] for r in final['controls']}
        assert not controls[m.acted + 0x800] and controls[m.pickable]
        assert all(controls[p] == (o['picked_count'] == 0) for p in m.picked_controls)
        assert controls[m.rip] == (o['death'] != 'live')
        assert controls[m.disguise] == (o['disguise'] != 'live')
        assert final['continuations'] == [{'identity': m.arena + 0x30100, 'actor': m.actor,
                                          'state': 1, 'current': m.arena + 0x34100}]
        callbacks = [e for e in result['events'] if e['kind'] == 'state_callback']
        assert len(callbacks) == int(o['callback'])
        if callbacks:
            assert callbacks[0]['snapshot']['actor']['state'] == 5
            assert callbacks[0]['snapshot']['statuses'][0]['count'] == 3
            assert callbacks[0]['snapshot']['lists'][0]['count'] == 0
            assert callbacks[0]['snapshot']['lists'][0]['version'] == 1
    for method, cold, id, previous, usage, count, callback in itertools.product(
            ['Init', 'InitWithNoReset'], [False, True], [-100, 19], [0, 20], [0, 10], [0, 2],
            [False, True]):
        result = m.run_join({'sequence': [method], 'cold': cold, 'id': id, 'previous_phase': previous,
                             'ability_usage': usage, 'picked_count': count, 'callback': callback,
                             'death': 'absent', 'disguise': 'live', 'clone_null': False})
        verify_normal(result)
        cases.append({'input': result['input'], 'stages': result['stages'], 'final': result['final'],
                      'api_order': [{'phase': e['phase'], 'kind': e['kind']} for e in result['events']]})
    for method, death, disguise in itertools.product(['Init', 'InitWithNoReset'], ['absent', 'destroyed', 'live'],
                                                    ['absent', 'destroyed', 'live']):
        result = m.run_join({'sequence': [method], 'cold': True, 'death': death, 'disguise': disguise})
        verify_normal(result)
        cases.append({'input': result['input'], 'stages': result['stages'], 'final': result['final'],
                      'api_order': [{'phase': e['phase'], 'kind': e['kind']} for e in result['events']]})
    for method in ['Init', 'InitWithNoReset']:
        result = m.run_join({'sequence': [method], 'cold': True, 'clone_null': True})
        verify_normal(result)
        cases.append({'input': result['input'], 'stages': result['stages'], 'final': result['final'],
                      'api_order': [{'phase': e['phase'], 'kind': e['kind']} for e in result['events']]})
    sequences = []
    for names in [['Init', 'InitWithNoReset'], ['InitWithNoReset', 'Init', 'InitWithNoReset'], ['Init', 'Init', 'Init']]:
        result = m.run_join({'sequence': names, 'cold': True})
        assert result['returned'] and len(result['final']['continuations']) == len(names)
        for ordinal, stage in enumerate(result['stages'][1:], 1):
            assert stage['snapshot']['continuations'] == result['final']['continuations'][:ordinal]
        sequences.append(result)
    for method in ['Init', 'InitWithNoReset']:
        for effects in [{f'{method}#1:state_callback:1': {'statuses': m.alternate_status}},
                        {f'{method}#1:state_callback:1': {'uses': 0}}]:
            result = m.run_join({'sequence': [method], 'effects': effects, 'previous_phase': 0})
            assert result['returned']
            if 'statuses' in next(iter(effects.values())):
                assert result['final']['statuses'][0]['count'] == 3
                assert result['final']['statuses'][1]['count'] == (0 if method == 'Init' else 3)
            else:
                assert result['final']['actor']['uses'] == 0
                assert not next(c['active'] for c in result['final']['controls'] if c['identity'] == m.pickable)
            mutations.append(result)
    baselines = []
    for method in ['Init', 'InitWithNoReset']:
        options = {'sequence': [method], 'cold': True, 'id': 19, 'death': 'live'}
        baseline = m.run_join(options)
        baseline_bytes = m.byte_prefixes.copy()
        baselines.append(baseline)
        counts = {}
        for index, e in enumerate(baseline['events']):
            key = (e['phase'], e['kind'])
            counts[key] = counts.get(key, 0) + 1
            failure = [*key, counts[key]]
            result = m.run_join({**options, 'failure': failure})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == e['snapshot']
            assert bytes(m.u.mem_read(m.actor, 0x1B8)) == baseline_bytes[index]
            failures.append({'baseline': method, 'failure': failure, 'prefix_length': index + 1,
                             'exact_snapshot_and_actor_bytes_verified': True})
    return {'build': BUILD, 'constructor': m.method, 'methods': m.methods,
            'instruction_assertions': m.instruction_assertions + len(m.join_checks) + 2,
            'case_count': len(cases), 'cases': cases, 'repeated_sequence_count': len(sequences), 'repeated_sequences': sequences,
            'mutation_case_count': len(mutations), 'mutation_cases': mutations, 'failure_baselines': baselines,
            'failure_case_count': len(failures), 'failure_cases': failures,
            'native_instructions_executed': len(m.visited),
            'decoded_instructions_by_owner': {name: sum(owner == name for owner in m.address_phase.values()) for name in m.ENTRIES},
            'executed_by_owner': {name: len({p for p in m.visited if m.address_phase.get(p) == name}) for name in m.ENTRIES},
            'constructor_instructions_executed': len(m.constructor_decoded & m.visited),
            'limits': ['Constructor-produced physical Lists, uses, act and savedAct pass unchanged into native Init/NoReset in one emulator.',
                       'Serialized UI/data/status components are authored pre-entry fixtures; Unity scene load and status-constructor provenance are not inferred.',
                       'RefreshCharacter and RefreshView execute natively only with Hidden state; callback probes preserve that state.',
                       'StartCoroutine explicitly invokes native MoveNext only to its first yield; actual engine registration, later resumption and scheduling remain external.',
                       'Metadata, runtime, diagnostics, GC, List construction, object operations, cloning, wait construction and callback effects remain explicit supplied services.',
                       'Complete actor-byte and phase-tagged semantic prefix checks cover controlled failures without reconstructing managed exception unwinding.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'mutation_case_count', 'failure_case_count', 'native_instructions_executed']}))
