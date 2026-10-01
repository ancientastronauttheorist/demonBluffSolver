"""Execute Character coroutine factories and immediate speech entry callers."""
import argparse
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_role_publication import Machine as PublicationMachine, verify_publication


TARGETS = {0x364A20: 'DelayReveal', 0x364A90: 'DelayedDemonKill',
           0x368B00: 'ShowActed', 0x369350: 'ShowInfoDelayed',
           0x3693E0: 'ShowTrailerAct', 0x365220: 'HideActed'}
TYPES = {'Character.<DelayReveal>d__84_TypeInfo': 'reveal_factory',
         'Character.<DelayedDemonKill>d__103_TypeInfo': 'kill_factory',
         'Character.<ShowInfoDelayed>d__134_TypeInfo': 'speech'}
SIGNATURES = {
    'DelayReveal': 'System_Collections_IEnumerator_o* Character__DelayReveal (Character_o* __this, const MethodInfo* method);',
    'DelayedDemonKill': 'System_Collections_IEnumerator_o* Character__DelayedDemonKill (Character_o* __this, Character_o* evilRef, const MethodInfo* method);',
    'ShowActed': 'void Character__ShowActed (Character_o* __this, ActedInfo_o* info, int32_t trigger, float delay, const MethodInfo* method);',
    'ShowInfoDelayed': 'System_Collections_IEnumerator_o* Character__ShowInfoDelayed (Character_o* __this, System_String_o* info, const MethodInfo* method);',
    'ShowTrailerAct': 'void Character__ShowTrailerAct (Character_o* __this, System_String_o* info, const MethodInfo* method);',
    'HideActed': 'void Character__HideActed (Character_o* __this, const MethodInfo* method);'}


class Machine(PublicationMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
        self.entry_targets = []
        self.entry_addresses = set()
        for address, name in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod']
                    if r['Address'] == address and r['Name'] == 'Character$$' + name]
            assert len(rows) == 1
            assert rows[0]['Signature'] == SIGNATURES[name]
            self.entry_targets.extend(rows)
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4:
                    root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == address:
                    chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks
            self.ranges[hex(address)] = [[hex(a), hex(b)] for a, b in chunks]
            for a, b in chunks:
                ins = list(self.cs.disasm(self.pe.get_data(a, b - a), a))
                assert sum(i.size for i in ins) == b - a
                self.instructions.update({i.address: i for i in ins})
                self.entry_addresses.update(i.address for i in ins)
        references = set()
        for address in self.entry_addresses:
            i = self.instructions[address]
            for op in i.operands:
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + op.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1:
                assert i.operands[0].mem.base == capstone.x86.X86_REG_RIP
                self.flags.add(i.address + i.size + i.operands[0].mem.disp)
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                if row['Name'] not in self.bindings:
                    self.bindings[row['Name']] = self.arena + 0x4000 + len(self.bindings) * 0x200
                token = self.bindings[row['Name']]
                self.metadata_slots[self.base + row['Address']] = token
                self.q(self.base + row['Address'], token)
        assert all(name in self.bindings for name in TYPES)
        for name, index, fields in [
                ('Character.<DelayReveal>d__84', 5482, ['public Character <>4__this; // 0x20']),
                ('Character.<DelayedDemonKill>d__103', 5483,
                 ['public Character <>4__this; // 0x20', 'public Character evilRef; // 0x28']),
                ('Character.<ShowInfoDelayed>d__134', 5486,
                 ['public Character <>4__this; // 0x20', 'public string info; // 0x28'])]:
            match = re.search(r'^private sealed class ' + re.escape(name) +
                              r' : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: ' +
                              str(index) + r'\s*\{(.*?)// Properties', self.dump, re.M | re.S)
            assert match and all(field in match[1] for field in fields)
            assert 'private int <>1__state; // 0x10' in match[1]
            assert 'private object <>2__current; // 0x18' in match[1]
        self.checks = {
            0x364A69: ('mov', 'qword ptr [rcx], rdi'),
            0x364A6C: ('mov', 'dword ptr [rbx + 0x10], 0'),
            0x364AEB: ('call', '0x2b6ff0'),
            0x364AF7: ('mov', 'qword ptr [rcx], rdi'),
            0x368B06: ('mov', 'r9d, r8d'),
            0x368B15: ('movaps', 'xmm1, xmm3'),
            0x368B1B: ('call', '0x368a50'),
            0x368B2E: ('jmp', '0x1c7f160'),
            0x3693A4: ('mov', 'dword ptr [rbx + 0x10], 0'),
            0x3693B7: ('mov', 'qword ptr [rcx], rdi'),
            0x369410: ('call', '0x1c7d810'),
            0x369415: ('mov', 'rcx, qword ptr [rbx + 0xa8]'),
            0x369431: ('jmp', '0x35dd10'),
            0x365248: ('jmp', '0x1c7d810')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == expected
                   for a, expected in self.checks.items())

    def factory_object(self, p):
        name = self.objects[p]
        if name.startswith('iterator'):
            return self.iterator(p)
        out = {'id': name, 'state': self.rd(p + 0x10),
               'current': self.object_id(self.rq(p + 0x18)),
               'owner': self.object_id(self.rq(p + 0x20))}
        if name.startswith(('kill_factory', 'speech')):
            out['argument'] = self.object_id(self.rq(p + 0x28))
        return out

    def snapshot(self):
        out = super().snapshot()
        out['factory_objects'] = [self.factory_object(p) for p in self.objects
                                  if self.objects[p].startswith(('reveal_factory', 'kill_factory', 'speech', 'iterator'))]
        out['entry_registered'] = self.entry_registered.copy()
        out['acted_reference'] = self.object_id(self.rq(self.actor + 0xA8))
        out['metadata_flags'] = {hex(p): self.u.mem_read(self.base + p, 1)[0] for p in sorted(self.flags)}
        return out

    def prepare(self, options):
        self.entry_registered = []
        super().prepare(options)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        cx, dx, r8 = [self.reg(r) for r in [x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8]]
        if rva == 0x2B7D40 and cx in [self.bindings[n] for n in TYPES]:
            self.executed.add(rva)
            kind = next(v for n, v in TYPES.items() if self.bindings[n] == cx)
            if self.event('allocate_service', [kind]):
                p = self.alloc(0x100)
                self.q(p, cx)
                self.objects[p] = kind + str(len(self.objects))
                self.ret(p)
        elif rva == 0x1C7F160:
            self.executed.add(rva)
            assert cx == self.actor_arg and r8 == 0
            item = self.factory_object(dx)
            assert item['state'] == 0 and item['current'] is None
            if self.event('entry_coroutine_registration_service', [item]):
                self.entry_registered.append(item)
                self.ret(0xABCDEF1234567890)
        elif rva == 0x1C7D810 and self.options.get('clear_acted_after_activation'):
            self.executed.add(rva)
            assert cx == self.fixtures['game'] and dx & 0xFF == 1 and r8 == 0
            if self.event('set_active_service', ['game', True]):
                self.active['game'] = True
                self.q(self.actor + 0xA8, 0)
                self.ret()
        else:
            super().hook(uc, address, size, data)

    def invoke(self, address, cx, dx=0, r8=0, r9=0):
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        self.q(sp + 0x28, 0)
        integers = [getattr(x, 'UC_X86_REG_' + name) for name in
                    ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, register in enumerate(integers):
            self.u.reg_write(register, 0xFAB00000 + i)
        for i, register in enumerate(vectors):
            self.u.reg_write(register, (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64))
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, cx),
                                (x.UC_X86_REG_RDX, dx), (x.UC_X86_REG_R8, r8), (x.UC_X86_REG_R9, r9)]:
            self.u.reg_write(register, value & 0xFFFFFFFFFFFFFFFF)
        self.u.emu_start(self.base + address, self.stop, timeout=10_000_000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(register) == 0xFAB00000 + i for i, register in enumerate(integers))
            assert all(self.reg(register) == (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64)
                       for i, register in enumerate(vectors))
            assert not self.frames and not self.native_calls
        return returned

    def run_entry(self, name, options=None, retained=False):
        if not retained:
            self.prepare(options or {})
        else:
            self.options = options or {}
            self.error = None
        address = next(a for a, n in TARGETS.items() if n == name)
        if name == 'DelayedDemonKill':
            argument = 0 if self.options.get('null_argument') else self.role
        elif name == 'ShowActed':
            argument = 0 if self.options.get('null_argument') else self.info
        else:
            argument = 0 if self.options.get('null_argument') else self.string_pointers['original']
        if name in ['DelayReveal', 'HideActed']:
            argument = 0
        if name in ['ShowActed', 'ShowInfoDelayed', 'ShowTrailerAct'] and self.options.get('empty_argument'):
            assert name != 'ShowActed'
            argument = self.string_pointers['empty']
        before_actor = bytes(self.u.mem_read(self.actor, 0x200))
        self.u.reg_write(self.x.UC_X86_REG_XMM3, self.options.get('delay_bits', 0))
        returned = self.invoke(address, self.actor_arg, argument,
                               self.options.get('trigger', 30) | 0xFACE000000000000)
        after_actor = bytes(self.u.mem_read(self.actor, 0x200))
        if self.options.get('clear_acted_after_activation'):
            assert before_actor[:0xA8] == after_actor[:0xA8] and before_actor[0xB0:] == after_actor[0xB0:]
        else:
            assert before_actor == after_actor
        result = self.reg(self.x.UC_X86_REG_RAX) if returned and name in ['DelayReveal', 'DelayedDemonKill', 'ShowInfoDelayed'] else None
        if result is not None:
            assert result in self.objects
            item = self.factory_object(result)
            assert item['state'] == 0 and item['current'] is None and item['owner'] == self.object_id(self.actor_arg)
            if name != 'DelayReveal':
                assert item['argument'] == self.object_id(argument)
        if returned and name == 'ShowActed':
            item = self.entry_registered[-1]
            assert item['trigger'] == self.options.get('trigger', 30) & 0xFFFFFFFF
            assert item['delay_bits'] == self.options.get('delay_bits', 0)
            assert item['info'] == self.object_id(argument)
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'result': self.object_id(result) if result else None, 'events': self.events.copy(), 'final': self.snapshot()}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    dependency = verify_publication(m)
    cases, sequences, baselines, failures = [], [], [], []
    for name, cold, null_argument in itertools.product(TARGETS.values(), [False, True], [False, True]):
        result = m.run_entry(name, {'cold': cold, 'null_argument': null_argument})
        assert result['returned']
        cases.append(result)
    for name, cold in itertools.product(['DelayReveal', 'DelayedDemonKill', 'ShowInfoDelayed'], [False, True]):
        result = m.run_entry(name, {'cold': cold, 'null_actor': True})
        assert result['returned']
        cases.append(result)
    for trigger, bits, cold in itertools.product([0, 3, 5, 30, 0xFFFFFFFF, 0x80000000],
                                                [0, 0x80000000, 0x3E99999A, 0x3ECCCCCD, 0x7FC01234, 0x7F800000],
                                                [False, True]):
        result = m.run_entry('ShowActed', {'trigger': trigger, 'delay_bits': bits, 'cold': cold})
        assert result['returned']
        cases.append(result)
    for name in ['ShowInfoDelayed', 'ShowTrailerAct']:
        result = m.run_entry(name, {'empty_argument': True})
        assert result['returned']
        cases.append(result)
    for name, null in itertools.product(['ShowTrailerAct', 'HideActed'], ['null_acteds', 'null_game_call', 'null_version', 'null_layouts']):
        options = {null: 1 if null == 'null_game_call' else True}
        result = m.run_entry(name, options)
        assert result['returned'] == (name == 'HideActed' and null in ['null_version', 'null_layouts'])
        cases.append(result)
    for name in ['DelayReveal', 'DelayedDemonKill', 'ShowInfoDelayed', 'ShowActed']:
        m.prepare({'cold': True})
        calls = [m.run_entry(name, {'cold': True}, retained=True),
                 m.run_entry(name, {'null_argument': True}, retained=True)]
        assert all(c['returned'] for c in calls)
        assert len(calls[-1]['final']['factory_objects']) == 2
        sequences.append({'method': name, 'calls': calls})
    mutation = m.run_entry('ShowTrailerAct', {'clear_acted_after_activation': True})
    assert not mutation['returned'] and mutation['error'] == 'native_null_guard'
    assert mutation['final']['acted_reference'] is None and mutation['final']['active']['game']
    assert not mutation['final']['shown'] and mutation['final']['saved_speech'] == 'old'
    for name in TARGETS.values():
        baseline = m.run_entry(name, {'cold': True})
        assert baseline['returned']
        ordinal = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run_entry(name, {'cold': True, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index + 1]
            assert stopped['final'] == event['snapshot']
            failures.append({'baseline': ordinal, 'failure': [kind, counts[kind]],
                             'prefix_length': index + 1, 'exact_snapshot_verified': True})
    return {'build': BUILD, 'targets': m.entry_targets, 'native_ranges': {hex(a): m.ranges[hex(a)] for a in TARGETS},
            'instruction_assertions': len(m.checks), 'case_count': len(cases), 'cases': cases,
            'dependency_instruction_assertions': dependency['instruction_assertions'],
            'mutation_cases': [mutation],
            'retained_sequences': sequences, 'failure_baselines': baselines, 'failure_case_count': len(failures),
            'failure_cases': failures, 'entry_instructions_executed': len(m.executed & m.entry_addresses),
            'entry_instructions_decoded': len(m.entry_addresses), 'native_execution_addresses': len(m.executed),
            'scope': 'Six complete native Character factory/direct speech entry callers execute. Metadata/allocation/GC/UI services and coroutine registration are supplied; no generator MoveNext, actual scheduling, animation, runtime lifetime or managed unwinding is inferred. Actor bytes remain unchanged in the inert-service normal corpus.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'failure_case_count', 'entry_instructions_executed', 'native_execution_addresses']}))
