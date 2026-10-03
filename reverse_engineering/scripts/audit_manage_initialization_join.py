"""Actual shared pool construction and Init occurrences before board publication.

Uses the original pool audit's retained memory and services, rather than replaying
its output into a fresh initializer fixture. No live process or hidden gameplay
oracle is read. RefreshView and Unity operations are explicit supplied services.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_manage_pool_composition import audit as pool_audit


class InitializationJoin:
    def decode(self, metadata, dump, pe, cs, decoded):
        root = Path(__file__).parents[1]
        prior = json.loads((root / f'reports/{BUILD}_character_init.json').read_text(encoding='utf-8'))
        assert prior['build'] == BUILD and prior['case_count'] == 475
        self.fields = prior['fields']
        for name, declarations in self.fields.items():
            body = re.search(r'^[^\n]*class ' + re.escape(name)
                             + r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', dump, re.M | re.S)
            assert body and all(d in body[1] for d in declarations), name
        addresses = {0x365a20, 0x367970, 0x3756b0}
        self.methods = [m for m in prior['methods'] if m['Address'] in addresses]
        assert {m['Address'] for m in self.methods} == addresses
        self.fingerprints = {}
        for method in self.methods:
            assert method in metadata['ScriptMethod']
            start = method['Address']
            end = min(m['Address'] for m in metadata['ScriptMethod'] if m['Address'] > start)
            instructions = list(cs.disasm(pe.get_data(start, end - start), start))
            while instructions[-1].mnemonic == 'int3': instructions.pop()
            assert instructions[0].address == start
            assert all(a.address + a.size == b.address for a, b in zip(instructions, instructions[1:]))
            end = instructions[-1].address + instructions[-1].size
            self.fingerprints[method['Name']] = hashlib.sha256(pe.get_data(start, end - start)).hexdigest()
            decoded.update({i.address: i for i in instructions})
        noop = list(cs.disasm(pe.get_data(0x33ed50, 3), 0x33ed50))
        assert len(noop) == 1 and (noop[0].mnemonic, noop[0].op_str) == ('ret', '0')
        decoded[0x33ed50] = noop[0]
        self.checks = {
            0x36cfda: ('call', '0x365a20'),
            0x36d01e: ('mov', 'rbx, qword ptr [r12 + 0x20]'),
            0x365aad: ('mov', 'qword ptr [rcx], r15'),
            0x365c33: ('mov', 'eax, dword ptr [rsi + 0x134]'),
            0x365cbf: ('call', 'qword ptr [rax + 0x18]'),
            0x365ce2: ('mov', 'dword ptr [rax + 0x18], r15d'),
            0x365d39: ('mov', 'dword ptr [rbx + 0x10], r15d'),
            0x365d62: ('jmp', '0x1c7f160'),
            0x3756f4: ('mov', 'dword ptr [rdi + 0x10], 0xffffffff'),
            0x375769: ('mov', 'dword ptr [rdi + 0x10], 1'),
        }
        for address, expected in self.checks.items():
            assert address in decoded and (decoded[address].mnemonic, decoded[address].op_str) == expected
        self.return_site = decoded[0x36cfda].address + decoded[0x36cfda].size
        self.native_seen = set()

    def bind(self, **runtime):
        # Explicit allowlisted fields; this runtime never enters report JSON.
        for name, value in runtime.items(): setattr(self, name, value)
        required = ['Character.<DelayReveal>d__84_TypeInfo', 'UnityEngine.WaitForSeconds_TypeInfo',
                    'Method$ClassConv.CreateCopyNonGeneric<Role>()', 'Gameplay_TypeInfo']
        assert all(name in self.bindings for name in required)
        self.sources = {ident: self.arena + 0x180000 + ident * 0x100 for ident in self.data}
        self.callback_code = self.stop + 0x300
        self.text_code = self.stop + 0x400
        self.yield_return = self.stop + 0x500
        self.services = {0x2b6ff0: 'barrier', 0x1c79fd0: 'game_object', 0x1c7d810: 'inactive',
                         0x112b9d0: 'clear_infos', 0x1c82480: 'unity_live',
                         0xf71c60: 'concat', 0x1c4b450: 'log', 0x1c4b380: 'log',
                         0x282580: 'box', 0xf74df0: 'format', 0x367b60: 'refresh_view',
                         0x2b7d40: 'allocate', 0x1c7f160: 'start_coroutine',
                         0x603240: 'clone_role', 0x1c961f0: 'wait_constructor',
                         0x1c822c0: 'unity_equal'}

    def actor_snapshot(self, actor):
        return {'actor': actor, 'data': self.labels[self.rq(actor + 0x50)],
                'data_pointer': self.rq(actor + 0x50), 'id': self.rd(actor + 0x118),
                'state': self.rd(actor + 0xe4), 'previous': self.rd(actor + 0xe0),
                'alignment': self.rd(actor + 0xf8), 'role': self.rq(actor + 0x168),
                'bluff': self.rq(actor + 0x58), 'runtime': self.rq(actor + 0x70),
                'register_as': self.rq(actor + 0x60),
                'trailer': self.rq(actor + 0x68), 'dead_prefab': self.rq(actor + 0x98),
                'revealed': int(self.uc.mem_read(actor + 0xd8, 1)[0]), 'uses': self.rd(actor + 0xdc),
                'killed_hidden': int(self.uc.mem_read(actor + 0xec, 1)[0]),
                'killed_demon': int(self.uc.mem_read(actor + 0xed, 1)[0]),
                'started': int(self.uc.mem_read(actor + 0x11c, 1)[0]),
                'bluff_role': self.rq(actor + 0x170), 'saved_act': self.rq(actor + 0x198),
                'state_callback': self.rq(actor + 0x180),
                'info_count': self.rd(actor + 0x418), 'info_version': self.rd(actor + 0x41c),
                'info_values': [self.rq(self.rq(actor + 0x410) + 0x20 + i * 8)
                                for i in range(self.rd(actor + 0x418))],
                'status_count': self.rd(actor + 0x398), 'status_version': self.rd(actor + 0x39c),
                'status_values': [self.rd(self.rq(actor + 0x390) + 0x20 + i * 4)
                                  for i in range(self.rd(actor + 0x398))],
                'resistance_values': [self.rd(self.rq(actor + 0x3d0) + 0x20 + i * 4)
                                      for i in range(self.rd(actor + 0x3d8))],
                'resistance': self.rq(actor + 0x318), 'status_target': self.rq(actor + 0x320)}

    def snapshot(self):
        return {'actors': {name: self.actor_snapshot(p) for name, p in self.actors.items()},
                'continuations': [{'identity': p, 'state': self.rd(p + 0x10),
                    'owner': next(name for name, a in self.actors.items() if a == self.rq(p + 0x20)),
                    'current': self.rq(p + 0x18)} for p, _ in self.retained]}

    def prepare(self):
        self.calls, self.events, self.retained = [], [], []
        self.active = None
        self.nonvolatile = [self.x.UC_X86_REG_RBX, self.x.UC_X86_REG_RBP,
                            self.x.UC_X86_REG_RSI, self.x.UC_X86_REG_RDI,
                            self.x.UC_X86_REG_R12, self.x.UC_X86_REG_R13,
                            self.x.UC_X86_REG_R14, self.x.UC_X86_REG_R15]
        self.uc.mem_write(self.arena + 0x180000, bytes(0x40000))
        for i, actor in enumerate(self.actors.values()):
            self.uc.mem_write(actor, bytes(0x1000))
            for offset, pointer in {0x48: actor + 0x600, 0x50: self.data[7], 0xa8: actor + 0x500,
                                    0xf0: actor + 0x300, 0x148: actor + 0x400,
                                    0x188: actor + 0x900, 0x180: actor + 0xa00}.items():
                self.q(actor + offset, pointer)
            self.q(actor + 0x310, actor + 0x380)
            self.q(actor + 0x390, actor + 0xb00); self.q(actor + 0xb18, 3)
            for j, status in enumerate((10, 30, 50)): self.d(actor + 0xb20 + j * 4, status)
            self.q(actor + 0x318, actor + 0x3c0)
            self.q(actor + 0x3d0, actor + 0xb80); self.q(actor + 0xb98, 1)
            self.d(actor + 0x3d8, 1); self.d(actor + 0xba0, 10)
            self.q(actor + 0x320, self.actors['other_character'])
            self.q(actor + 0x410, actor + 0x480)
            self.q(actor + 0x498, 2)
            self.d(actor + 0x418, 2); self.d(actor + 0x41c, 17)
            self.d(actor + 0x398, 3); self.d(actor + 0x39c, 23)
            self.q(actor + 0x600, actor + 0x700)
            self.q(actor + 0xc58, self.text_code); self.q(actor + 0xc60, actor + 0xd00)
            self.q(actor + 0xa18, self.callback_code)
            self.q(actor + 0xa28, actor + 0xa00); self.q(actor + 0xa40, actor)
            self.d(actor + 0xe4, 20); self.d(actor + 0xe0, 10)
            self.d(actor + 0xf8, 10); self.d(actor + 0x118, 73)
        for ident, pointer in self.data.items():
            self.q(pointer + 0x140, self.sources[ident])
            self.q(pointer + 0x28, self.arena + 0x190000)
        for pointer in self.bindings.values(): self.d(pointer + 0xe0, 1)
        self.d(self.gs + 0x2c, 20)
        self.initial_actors = self.snapshot()

    def emit(self, kind, **details):
        self.events.append({'kind': kind, 'init_index': len(self.calls) - 1,
                            **details, 'snapshot': self.snapshot()})

    def hook(self, address, rva, c, dx, r8):
        x = self.x
        if rva == 0x36d01e:
            assert self.active is None
            self.state['boundary'] = 'before_publication'
            self.uc.emu_stop(); return True
        if rva == self.return_site and self.active is not None:
            assert self.reg(x.UC_X86_REG_RSP) == self.entry_sp + 8
            assert [self.reg(r) for r in self.nonvolatile] == self.saved_registers
            self.active['completed'] = True
            self.active['after'] = self.actor_snapshot(self.actor)
            assert all(bytes(self.uc.mem_read(p, 0x28)) == raw for p, raw in self.retained)
            self.active = None
            return False
        if rva == 0x365a20:
            assert self.active is None and c in self.actors.values() and self.reg(x.UC_X86_REG_R9) == 0
            assert dx in self.labels and self.rq(self.reg(x.UC_X86_REG_RSP)) == self.base + self.return_site
            self.actor = c
            self.entry_sp = self.reg(x.UC_X86_REG_RSP)
            self.saved_registers = [self.reg(r) for r in self.nonvolatile]
            index = len(self.calls)
            self.iterator = self.arena + 0x1a0000 + index * 0x200
            self.wait = self.iterator + 0x100
            self.clone = self.arena + 0x1b0000 + index * 0x100
            self.active = {'actor': next(name for name, p in self.actors.items() if p == c),
                           'data': self.labels[dx], 'display_id': r8 & 0xffffffff,
                           'before': self.actor_snapshot(c), 'completed': False,
                           'iterator': self.iterator, 'clone': None}
            self.calls.append(self.active)
            self.emit('init_entry')
            self.native_seen.add(rva)
            return True  # Continue the actual native entry.
        if self.active is None: return False
        if address == self.callback_code:
            assert c == self.actor and dx == self.actor + 0xa00
            self.emit('state_callback')
            if self.opt.get('stop_callback') == len(self.calls):
                self.halt('state_callback'); return True
            if self.opt.get('replace_callback') == len(self.calls):
                self.q(self.owner + 0x20, self.alternate)
            self.ret(); return True
        if address == self.text_code:
            assert c == self.actor + 0x600 and dx == self.arena + 0x190000 and r8 == self.actor + 0xd00
            self.emit('set_text'); self.ret(); return True
        if address == self.yield_return:
            assert self.reg(x.UC_X86_REG_RAX) & 255 == 1
            assert self.rd(self.iterator + 0x10) == 1 and self.rq(self.iterator + 0x18) == self.wait
            self.uc.reg_write(x.UC_X86_REG_RSP, self.scheduler_sp)
            self.retained.append((self.iterator, bytes(self.uc.mem_read(self.iterator, 0x28))))
            self.emit('first_yield'); self.ret(); return True
        if rva == 0x367970:
            self.emit('refresh_hidden')
            self.native_seen.add(rva)
            return True  # Hidden RefreshCharacter is actual native code in both stages.
        if rva == 0x33ed50:
            self.native_seen.add(rva)
            return False  # Execute the verified folded no-op.
        if rva not in self.services:
            self.native_seen.add(rva)
            return False
        kind = self.services[rva]
        self.emit(kind, caller=hex(self.rq(self.reg(x.UC_X86_REG_RSP)) - self.base))
        if kind == 'barrier':
            assert self.rq(c) == dx; self.ret()
        elif kind == 'game_object':
            assert c in (self.actor, self.actor + 0x500) and dx == 0; self.ret(c + 0x80)
        elif kind == 'inactive':
            assert dx == 0 and r8 == 0; self.ret()
        elif kind == 'clear_infos':
            assert c == self.actor + 0x480 and dx == 0
            self.uc.mem_write(c + 0x20, bytes(r8 * 8)); self.ret()
        elif kind == 'unity_live':
            assert c == 0 and dx == 0; self.ret(0)
        elif kind == 'unity_equal':
            assert c == 0 and dx == 0; self.ret(1)
        elif kind in ('concat', 'box', 'format'):
            self.ret(self.arena + 0x190000)
        elif kind == 'allocate':
            typ = self.bindings
            assert c in (typ['Character.<DelayReveal>d__84_TypeInfo'], typ['UnityEngine.WaitForSeconds_TypeInfo'])
            target = self.iterator if c == typ['Character.<DelayReveal>d__84_TypeInfo'] else self.wait
            self.uc.mem_write(target, bytes(0x40)); self.q(target, c); self.ret(target)
        elif kind == 'start_coroutine':
            assert c == self.actor and dx == self.iterator and r8 == 0
            assert self.rd(dx + 0x10) == 0 and self.rq(dx + 0x20) == self.actor and self.rq(dx + 0x18) == 0
            if self.opt.get('first_yield'):
                self.scheduler_sp = self.reg(x.UC_X86_REG_RSP)
                nested_sp = self.scheduler_sp - 0x30
                assert nested_sp % 16 == 8
                self.q(nested_sp, self.yield_return)
                self.uc.reg_write(x.UC_X86_REG_RSP, nested_sp)
                self.uc.reg_write(x.UC_X86_REG_RCX, dx)
                self.uc.reg_write(x.UC_X86_REG_RDX, 0)
                self.uc.reg_write(x.UC_X86_REG_RIP, self.base + 0x3756b0)
            else:
                self.retained.append((self.iterator, bytes(self.uc.mem_read(self.iterator, 0x28))))
                self.ret()
        elif kind == 'clone_role':
            assert c == self.sources[self.active['data']]
            assert dx == self.bindings['Method$ClassConv.CreateCopyNonGeneric<Role>()']
            self.active['clone'] = self.clone; self.ret(self.clone)
        elif kind == 'wait_constructor':
            assert c == self.wait and r8 == 0
            assert self.reg(x.UC_X86_REG_XMM1) & 0xffffffff == 0x3e99999a
            self.ret()
        else:
            assert kind in ('refresh_view', 'log'); self.ret()
        return True

    def report(self, run, starts, rosters, fallback, **evidence):
        cases, baselines = [], {}
        options = {'board': ['character', 'character'], 'manage_roster': [7, 1],
                   'alternate_board': ['other_character']}
        for first_yield in (False, True):
            for variant in ('alias', 'replace', 'stop_second', 'replace_stop_second'):
                selected = {**options, 'first_yield': first_yield}
                if variant in ('replace', 'replace_stop_second'): selected['replace_callback'] = 1
                if variant in ('stop_second', 'replace_stop_second'): selected['stop_callback'] = 2
                result = run(starts, rosters, fallback, selected)
                assert len(self.calls) == 2
                stopped = 'stop_callback' in selected
                assert result['error'] == ('state_callback' if stopped else None)
                assert result['boundary'] == (None if stopped else 'before_publication')
                assert [c['display_id'] for c in self.calls] == [2, 0 if 'replace_callback' in selected else 1]
                assert [c['data'] for c in self.calls] == [7, 1]
                assert [c['completed'] for c in self.calls] == [True, not stopped]
                assert all(bytes(self.uc.mem_read(p, 0x28)) == raw for p, raw in self.retained)
                assert len(self.retained) == (1 if stopped else 2)
                assert len({p for p, _ in self.retained}) == len(self.retained)
                final = self.snapshot()
                actor = final['actors']['character']
                assert actor['data'] == 1 and actor['state'] == 5
                assert actor['info_version'] == 19 and actor['info_count'] == 0
                assert actor['status_version'] == (24 if stopped else 25)
                assert actor['status_count'] == 0
                if first_yield:
                    assert actor['role'] == self.calls[0 if stopped else 1]['clone']
                else: assert actor['role'] == 0
                assert final['actors']['other_character'] == self.initial_actors['actors']['other_character']
                assert all(c['state'] == int(first_yield) for c in final['continuations'])
                assert all(c['owner'] == 'character' for c in final['continuations'])
                result.update(family=variant, stage='first_yield' if first_yield else 'registration_only',
                              initializer_calls=self.calls, initializer_events=self.events,
                              initial_actors=self.initial_actors, final_actors=final)
                if stopped:
                    baseline = baselines[(first_yield, 'replace' if 'replace_callback' in selected else 'alias')]
                    callback_index = next(i for i, e in enumerate(baseline['initializer_events'])
                                          if e['kind'] == 'state_callback' and e['init_index'] == 1)
                    assert self.events == baseline['initializer_events'][:callback_index + 1]
                    assert final == self.events[-1]['snapshot']
                    assert self.calls[0] == baseline['initializer_calls'][0]
                else:
                    baselines[(first_yield, variant)] = result
                cases.append(result)
        # Snapshot pooling keeps chronological data while avoiding repeated corpora.
        snapshots, indices = [], {}
        for case in cases:
            for event in case['events']:
                snapshot = event.pop('snapshot'); key = json.dumps(snapshot, sort_keys=True)
                if key not in indices: indices[key] = len(snapshots); snapshots.append(snapshot)
                event['snapshot_index'] = indices[key]
        return {'build_id': BUILD, 'schema': 'manage_initialization_join_v1',
                'case_count': len(cases), 'initializer_calls': sum(len(c['initializer_calls']) for c in cases),
                'completed_initializers': sum(call['completed'] for c in cases for call in c['initializer_calls']),
                'metadata_verified': evidence['metadata'] + self.methods,
                'native_assertions': len(evidence['checks']) + len(self.checks),
                'native_instructions_executed': len(evidence['visited'] | self.native_seen),
                'initializer_body_sha256': self.fingerprints,
                'field_declarations': {**evidence['fields'], **self.fields},
                'snapshot_table': snapshots, 'cases': cases,
                'limits': ['Development fixtures, not held-out or original live observations.',
                    'Actual Manage/pool/source/filter/predicate/Init/Hidden Refresh and optional first MoveNext bodies share retained memory.',
                    'Stops before 36D01E publication; no Act Init/Start, resumed Reveal, queue admission or solver integration.',
                    'Stable collection services, allocation, class metadata, RNG, UI, RefreshView, cloning and delegate effects are supplied.',
                    'First MoveNext invocation is supplied synchronous scheduling; registration-only stage retains state-zero iterators.',
                    'Controlled board-field replacement preserves the original enumerator list; no managed List version/reentrancy claim.',
                    'Stopped callback retains actual completed effects; no exception unwinding or rollback.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args()
    report = pool_audit(args.game_root, args.dumper_root, initialization=InitializationJoin())
    with args.output.open('w', encoding='utf-8', newline='\n') as output:
        output.write(json.dumps(report, indent=2) + '\n')
    print(json.dumps({k: report[k] for k in ('case_count', 'initializer_calls', 'completed_initializers', 'native_instructions_executed')}))
