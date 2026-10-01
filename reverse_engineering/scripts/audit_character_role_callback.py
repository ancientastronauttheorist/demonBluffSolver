"""Join actual RoleAct callback installation to delayed-result first publication."""
import argparse
import hashlib
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine


TARGETS = {0x368790: 'Character$$RoleAct',
           0x377120: 'Character.<>c__DisplayClass125_0$$<RoleAct>b__0',
           0x368A50: 'Character$$ShowActedDelayed',
           0x375FE0: 'Character.<ShowActedDelayed>d__133$$MoveNext'}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root)
        ext = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == ext['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json'))
        self.dump = pin('dump.cs', 'dump_cs')
        self.instructions, self.ranges = {}, {}
        self.targets = []
        for address, name in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1
            self.targets += rows
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4:
                    root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == address:
                    a, b = entry.struct.BeginAddress, entry.struct.EndAddress
                    chunks.append([hex(a), hex(b)])
                    ins = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
                    assert sum(i.size for i in ins) == b - a
                    self.instructions.update({i.address: i for i in ins})
            assert chunks
            self.ranges[hex(address)] = chunks
        self.flags, references = set(), set()
        for i in self.instructions.values():
            for op in i.operands:
                if op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + op.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.x86.X86_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
                self.flags.add(i.address + i.size + i.operands[0].mem.disp)
        self.bindings, self.metadata_slots = {}, {}
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                token = self.arena + 0x4000 + len(self.bindings) * 0x200
                self.bindings[row['Name']] = token
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
        for name in ['System.Action<ActedInfo>_TypeInfo', 'Character.<>c__DisplayClass125_0_TypeInfo',
                     'Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()',
                     'Character.<ShowActedDelayed>d__133_TypeInfo', 'UnityEngine.WaitForSeconds_TypeInfo']:
            assert name in self.bindings
        self.actor, self.role, self.role_class, self.info = [self.arena + n for n in [0x11000, 0x12000, 0x13000, 0x14000]]
        self.role_act, self.role_bluff = self.stop + 0x100, self.stop + 0x110
        self.after_callback, self.after_wait = self.stop + 0x200, self.stop + 0x210
        self.q(self.role, self.role_class)
        for offset, value in [(0x208, self.role_act), (0x210, self.role_class + 0x500),
                              (0x258, self.role_bluff), (0x260, self.role_class + 0x580)]:
            self.q(self.role_class + offset, value)

    def object_id(self, pointer):
        if not pointer:
            return None
        if pointer in self.objects:
            return self.objects[pointer]
        assert pointer in [self.actor, self.role, self.info]
        return {self.actor: 'actor', self.role: 'role', self.info: 'info'}[pointer]

    def iterator(self, pointer):
        return {'id': self.object_id(pointer), 'state': self.rd(pointer + 0x10),
                'current': self.object_id(self.rq(pointer + 0x18)),
                'delay_bits': self.rd(pointer + 0x20), 'owner': self.object_id(self.rq(pointer + 0x28)),
                'info': self.object_id(self.rq(pointer + 0x30)), 'trigger': self.rd(pointer + 0x38)}

    def snapshot(self):
        return {'role_delegate': self.object_id(self.rq(self.role + 0x28)),
                'objects': list(self.objects.values()),
                'closures': {name: {'owner': self.object_id(self.rq(p + 0x10)), 'trigger': self.rd(p + 0x18)}
                             for p, name in self.objects.items() if name.startswith('closure')},
                'iterators': [self.iterator(p) for p, name in self.objects.items() if name.startswith('iterator')],
                'waits': {name: self.rd(p + 0x10) for p, name in self.objects.items() if name.startswith('wait')},
                'registered': self.registered.copy(), 'completed_first_steps': self.first_steps.copy(),
                'role_calls': self.role_calls.copy()}

    def call_native(self, address, cx, dx, return_address):
        x = self.x
        old_sp = self.reg(x.UC_X86_REG_RSP)
        self.native_calls.append((return_address, old_sp))
        sp = old_sp - 0x30  # Shadow space, alignment, and pushed return address.
        assert sp & 15 == 8
        self.q(sp, return_address)
        self.q(sp + 0x28, 0)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        self.u.reg_write(x.UC_X86_REG_RCX, cx)
        self.u.reg_write(x.UC_X86_REG_RDX, dx)
        self.u.reg_write(x.UC_X86_REG_R8, 0)
        self.u.reg_write(x.UC_X86_REG_RIP, self.base + address)

    def complete_native_call(self, return_address):
        expected, old_sp = self.native_calls.pop()
        assert expected == return_address and self.reg(self.x.UC_X86_REG_RSP) == old_sp - 0x28
        self.u.reg_write(self.x.UC_X86_REG_RSP, old_sp)

    def invoke_delegate(self, frame):
        delegate = frame['delegate']
        self.call_native(0x377120, self.delegates[delegate], 0 if self.options.get('null_info') else self.info, self.after_callback)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9]]
        if address in [self.role_act, self.role_bluff]:
            route = 'act' if address == self.role_act else 'bluff_act'
            assert cx == self.role and r8 == self.actor_arg
            assert r9 == self.role_class + (0x500 if route == 'act' else 0x580)
            delegate = self.rq(self.role + 0x28)
            assert delegate in self.delegates
            call = {'route': route, 'trigger': dx & 0xFFFFFFFF, 'delegate': self.object_id(delegate)}
            if self.event('role_virtual_service', [call]):
                self.role_calls.append(call)
                repeats = self.options.get('callback_repeats', 1)
                if repeats:
                    frame = {'kind': 'role', 'remaining': repeats, 'delegate': delegate}
                    self.frames.append(frame)
                    self.invoke_delegate(frame)
                else:
                    self.ret()
        elif address == self.after_callback:
            self.complete_native_call(address)
            frame = self.frames[-1]
            assert frame['kind'] == 'role'
            frame['remaining'] -= 1
            if frame['remaining']:
                self.invoke_delegate(frame)
            else:
                self.frames.pop()
                self.ret()
        elif address == self.after_wait:
            self.complete_native_call(address)
            frame = self.frames.pop()
            assert frame['kind'] == 'start' and self.reg(x.UC_X86_REG_RAX) & 0xFF == 1
            self.first_steps.append(self.iterator(frame['iterator']))
            self.ret(self.arena + 0x20000 + len(self.registered) * 0x100)
        elif rva == 0x2B7B40:
            assert cx in self.metadata_slots
            if self.event('metadata_service', [cx - self.base]):
                self.ret(self.metadata_slots[cx])
        elif rva == 0x2B7D40:
            kinds = {self.bindings['Character.<>c__DisplayClass125_0_TypeInfo']: 'closure',
                     self.bindings['System.Action<ActedInfo>_TypeInfo']: 'delegate',
                     self.bindings['Character.<ShowActedDelayed>d__133_TypeInfo']: 'iterator',
                     self.bindings['UnityEngine.WaitForSeconds_TypeInfo']: 'wait'}
            assert cx in kinds
            kind = kinds[cx]
            if self.event('allocate_service', [kind]):
                pointer = self.alloc(0x100)
                self.q(pointer, cx)
                self.objects[pointer] = kind + str(len(self.objects))
                self.ret(pointer)
        elif rva == 0x2B6FF0:
            assert self.rq(cx) == dx
            if self.event('barrier_service', [cx - self.arena, self.object_id(dx)]):
                self.ret()
        elif rva == 0x4D5B60:
            assert self.objects[cx].startswith('delegate') and self.objects[dx].startswith('closure')
            assert r8 == self.bindings['Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()'] and r9 == 0
            if self.event('delegate_constructor_service', [self.object_id(cx), self.object_id(dx)]):
                self.delegates[cx] = dx
                self.ret()
        elif rva == 0x1C7F160:
            assert cx == self.actor_arg and r8 == 0 and self.objects[dx].startswith('iterator')
            item = self.iterator(dx)
            assert item['state'] == item['delay_bits'] == 0 and item['current'] is None
            if self.event('start_coroutine_service', [item]):
                self.registered.append(item)
                self.frames.append({'kind': 'start', 'iterator': dx})
                self.call_native(0x375FE0, dx, 0, self.after_wait)
        elif rva == 0x1C961F0:
            assert self.objects[cx].startswith('wait') and r8 == 0
            bits = self.reg(x.UC_X86_REG_XMM1) & 0xFFFFFFFF
            assert bits == 0
            if self.event('wait_constructor_service', [self.object_id(cx), bits]):
                self.d(cx + 0x10, bits)
                self.ret()
        elif rva == 0x2B7D90:
            self.error = 'native_null_guard'
            uc.emu_stop()

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        self.allocations, self.next_alloc, self.objects, self.delegates = {}, self.arena + 0x30000, {}, {}
        self.registered, self.first_steps, self.role_calls, self.frames = [], [], [], []
        self.native_calls = []
        self.actor_arg = 0 if options.get('null_actor') else self.actor
        self.u.mem_write(self.actor, bytes(0x200))
        self.u.mem_write(self.info, bytes(0x80))
        self.q(self.role + 0x28, 0)
        for flag in self.flags:
            self.u.mem_write(self.base + flag, bytes([int(not options.get('cold'))]))

    def invoke(self, address, cx, dx=0, r8=0, r9=0):
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        self.q(sp + 0x28, 0)
        keep = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(keep):
            self.u.reg_write(register, 0xFAB00000 + i)
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, cx), (x.UC_X86_REG_RDX, dx),
                                (x.UC_X86_REG_R8, r8), (x.UC_X86_REG_R9, r9)]:
            self.u.reg_write(register, value & 0xFFFFFFFFFFFFFFFF)
        saved_xmm6 = 0xFEDCBA98765432100123456789ABCDEF
        self.u.reg_write(x.UC_X86_REG_XMM6, saved_xmm6)
        self.u.emu_start(self.base + address, self.stop, timeout=10_000_000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(register) == 0xFAB00000 + i for i, register in enumerate(keep))
            assert self.reg(x.UC_X86_REG_XMM6) == saved_xmm6 and not self.frames and not self.native_calls
        assert bytes(self.u.mem_read(self.actor, 0x200)) == bytes(0x200)
        assert bytes(self.u.mem_read(self.info, 0x80)) == bytes(0x80)
        return returned

    def run(self, options=None):
        self.prepare(options or {})
        returned = self.invoke(0x368790, self.actor_arg, 0 if self.options.get('null_role') else self.role,
                               self.options.get('trigger', 5), self.options.get('route', 0))
        return {'options': self.options, 'returned': returned, 'error': self.error,
                'events': self.events.copy(), 'final': self.snapshot()}


def verify_native(m):
    checks = {0x368813: ('mov', 'qword ptr [rcx], rsi'), 0x36881B: ('mov', 'dword ptr [rbx + 0x18], ebp'),
              0x36884E: ('mov', 'qword ptr [rcx], rbp'), 0x368851: ('call', '0x2b6ff0'),
              0x36886E: ('call', 'qword ptr [rax + 0x258]'),
              0x368896: ('call', 'qword ptr [rax + 0x208]'),
              0x377142: ('xorps', 'xmm1, xmm1'), 0x377145: ('call', '0x368a50'),
              0x377158: ('jmp', '0x1c7f160'), 0x368AB4: ('mov', 'dword ptr [rbx + 0x10], 0'),
              0x368ACA: ('movss', 'dword ptr [rbx + 0x20], xmm6'),
              0x368AE1: ('mov', 'dword ptr [rbx + 0x38], ebp'),
              0x37603B: ('mov', 'dword ptr [rdi + 0x10], 0xffffffff'),
              0x37605A: ('call', '0x1c961f0'), 0x376066: ('mov', 'qword ptr [rcx], rbx'),
              0x376069: ('call', '0x2b6ff0'), 0x376075: ('mov', 'dword ptr [rdi + 0x10], 1')}
    for address, expected in checks.items():
        assert address in m.instructions and (m.instructions[address].mnemonic, m.instructions[address].op_str) == expected
    declarations = {
        'private sealed class Character.<>c__DisplayClass125_0 // TypeDefIndex: 5481':
            ['public Character <>4__this; // 0x10', 'public ETriggerPhase trigger; // 0x18'],
        'private sealed class Character.<ShowActedDelayed>d__133 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5485':
            ['private int <>1__state; // 0x10', 'private object <>2__current; // 0x18',
             'public float delay; // 0x20', 'public Character <>4__this; // 0x28',
             'public ActedInfo info; // 0x30', 'public ETriggerPhase trigger; // 0x38'],
        'public abstract class Role //': ['public Action<ActedInfo> onActed; // 0x28']}
    for declaration, fields in declarations.items():
        assert m.dump.count(declaration) == 1
        block = m.dump.split(declaration, 1)[1].split('// Methods', 1)[0]
        assert all(field in block for field in fields)
    noop = list(m.cs.disasm(m.pe.get_data(0x33ED50, 3), 0x33ED50))
    assert [(i.mnemonic, i.op_str) for i in noop] == [('ret', '0')]
    for name, address in [('UnityEngine.MonoBehaviour$$StartCoroutine', 0x1C7F160),
                          ('UnityEngine.WaitForSeconds$$.ctor', 0x1C961F0)]:
        assert len([r for r in m.metadata['ScriptMethod'] if r['Name'] == name and r['Address'] == address]) == 1
    return {'targets': m.targets, 'native_ranges': m.ranges, 'instruction_assertions': len(checks) + 1,
            'metadata_bindings': sorted(m.bindings), 'scope': 'Actual RoleAct, closure callback, delayed-result factory and explicit first MoveNext execute. Metadata, GC/delegate allocation/construction/barriers, role virtual callbacks, WaitForSeconds construction and StartCoroutine registration/resume adapter remain supplied. No second resume, scheduler timing, real delegate implementation, role clue generation or native exception unwinding is claimed.'}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    verified = verify_native(m)
    cases, baselines, failures = [], [], []
    for trigger, route, repeats, null_info, cold in itertools.product([0, 3, 5, 30, 0xFFFFFFFF], [0, 1, 0xFFFFFFFF], [0, 1, 2], [False, True], [False, True]):
        result = m.run({'trigger': trigger, 'route': route, 'callback_repeats': repeats, 'null_info': null_info, 'cold': cold})
        assert result['returned'] and len(result['final']['registered']) == repeats
        assert len(result['final']['completed_first_steps']) == repeats
        assert result['final']['role_calls'][0]['route'] == ('act' if route == 0 else 'bluff_act')
        assert all(item['trigger'] == trigger and item['state'] == 1 and item['delay_bits'] == 0
                   for item in result['final']['completed_first_steps'])
        assert all(item['owner'] == 'actor' and item['info'] == (None if null_info else 'info')
                   for item in result['final']['completed_first_steps'])
        assert list(result['final']['closures'].values()) == [{'owner': 'actor', 'trigger': trigger}]
        assert len({item['id'] for item in result['final']['completed_first_steps']}) == repeats
        assert len(result['final']['waits']) == repeats
        cases.append({k: v for k, v in result.items() if k != 'events'} | {'event_kinds': [e['kind'] for e in result['events']]})
    for null_actor, null_role in itertools.product([False, True], repeat=2):
        result = m.run({'null_actor': null_actor, 'null_role': null_role})
        assert result['returned'] == (not null_actor and not null_role)
        cases.append(result)
    result = m.run({'null_actor': True, 'callback_repeats': 0})
    assert result['returned'] and result['final']['role_calls'] and not result['final']['registered']
    cases.append(result)
    m.prepare({'callback_repeats': 0})
    assert m.invoke(0x368790, m.actor, m.role, 3, 0)
    old = m.rq(m.role + 0x28)
    assert m.invoke(0x368790, m.actor, m.role, 5, 1)
    new = m.rq(m.role + 0x28)
    assert old != new and m.invoke(0x377120, m.delegates[old], m.info) and m.invoke(0x377120, m.delegates[new], m.info)
    assert [item['trigger'] for item in m.first_steps] == [3, 5]
    assert m.rq(m.role + 0x28) == new
    assert all(item['owner'] == 'actor' and item['info'] == 'info' for item in m.first_steps)
    sequence = {'old_delegate': m.object_id(old), 'new_delegate': m.object_id(new), 'final': m.snapshot(), 'events': m.events.copy()}
    for options in [{'cold': True}, {'cold': True, 'callback_repeats': 2}]:
        baseline = m.run(options)
        baseline_id = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            result = m.run({**options, 'failure': [kind, counts[kind]]})
            assert not result['returned'] and result['events'] == baseline['events'][:index + 1]
            assert result['final'] == event['snapshot']
            failures.append({'baseline': baseline_id, 'prefix_length': index + 1, 'failure': [kind, counts[kind]], 'exact_snapshot_verified': True})
    return {'build_id': BUILD, **verified, 'cases': cases, 'case_count': len(cases), 'retained_delegate_sequence': sequence,
            'failure_baselines': baselines, 'failure_cases': failures, 'failure_case_count': len(failures),
            'executed_address_count': len(m.executed)}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['case_count'], report['failure_case_count'], report['executed_address_count'])
