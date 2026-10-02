"""Pinned RevealOrder constructor caller and supplied-base presentation joins.

The two-instruction folded caller is executed. Unity's constructor gateway,
component/TMP/formatting services remain supplied, and no engine allocation,
serialized field initialization, renderer or scheduler is inferred.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_reveal_order_presentation import Machine as PresentationMachine
from audit_reveal_order_presentation import serialize_report


ENTRY, END, FOLLOWING, BASE_GATEWAY = 0x33E820, 0x33E827, 0x33E830, 0x1C79770


class Machine(PresentationMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        rows = [r for r in self.metadata['ScriptMethod']
                if r['Address'] == ENTRY and r['Name'] == 'RevealOrder$$.ctor']
        assert len(rows) == 1
        assert rows[0]['Signature'] == 'void RevealOrder___ctor (RevealOrder_o* __this, const MethodInfo* method);'
        assert rows[0]['TypeSignature'] == 'vii'
        self.constructor_target = rows[0]
        self.wrapper_aliases = [r for r in self.metadata['ScriptMethod'] if r['Address'] == ENTRY]
        assert len(self.wrapper_aliases) == 218
        self.base_aliases = [r for r in self.metadata['ScriptMethod'] if r['Address'] == BASE_GATEWAY]
        nominal = [r for r in self.base_aliases if r['Name'] == 'UnityEngine.MonoBehaviour$$.ctor']
        assert len(nominal) == 1 and nominal[0]['TypeSignature'] == 'vii'
        assert nominal[0]['Signature'] == 'void UnityEngine_MonoBehaviour___ctor (UnityEngine_MonoBehaviour_o* __this, const MethodInfo* method);'
        assert len(self.base_aliases) == 5
        assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > ENTRY) == FOLLOWING
        # Direct file-backed bytes; a PE's virtual data must not invent a body.
        section = self.pe.get_section_by_rva(ENTRY)
        assert section is not None
        assert ENTRY + (FOLLOWING - ENTRY) <= section.VirtualAddress + section.SizeOfRawData
        raw = self.pe.get_data(ENTRY, FOLLOWING - ENTRY)
        assert len(raw) == FOLLOWING - ENTRY
        assert raw[END - ENTRY:] == b'\xcc' * (FOLLOWING - END)
        decoded = list(self.cs.disasm(raw[:END - ENTRY], ENTRY))
        assert sum(i.size for i in decoded) == END - ENTRY
        assert [(i.address, i.mnemonic, i.op_str) for i in decoded] == [
            (ENTRY, 'xor', 'edx, edx'), (ENTRY + 2, 'jmp', '0x1c79770')]
        self.constructor_instructions = {i.address: i for i in decoded}
        self.unwind_entries = [{'begin': hex(e.struct.BeginAddress), 'end': hex(e.struct.EndAddress)}
            for e in self.pe.DIRECTORY_ENTRY_EXCEPTION
            if e.struct.BeginAddress <= ENTRY < e.struct.EndAddress]
        # Field declarations are exact pinned inputs, independent of shared RVA aliases.
        dump = (Path(dumper_root) / 'dump.cs').read_text(encoding='utf-8-sig')
        declaration = re.search(r'^public class RevealOrder : MonoBehaviour // TypeDefIndex: 5735\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert declaration
        self.field_declaration = declaration[1].split('// Methods')[0].strip()
        assert self.field_declaration == '// Fields\n\tpublic TextMeshProUGUI text; // 0x20'
        self.p['reveal_class'] = self.arena + 0x70000
        self.ids[self.p['reveal_class']] = 'reveal_class'
        self.sizes['reveal_class'] = 0x100

    def snapshot(self):
        out = super().snapshot()
        out['owner_class_ref'] = self.oid(self.rq(self.p['reveal']))
        out['base_acceptances'] = self.base_acceptances.copy()
        out['stack_window_hex'] = bytes(self.u.mem_read(self.entry_sp, 0x40)).hex()
        return out

    def prepare(self, options):
        self.base_acceptances = []
        super().prepare(options)
        storage = options.get('owner_storage', 'serialized')
        assert storage in ['fresh_zero', 'serialized', 'reused']
        if storage == 'fresh_zero': self.u.mem_write(self.p['reveal'], bytes(self.sizes['reveal']))
        self.q(self.p['reveal'], self.p['reveal_class'])
        self.q(self.p['reveal'] + 8, 0)
        if storage == 'reused': self.q(self.p['reveal'] + 0x20, self.p['other_text'])
        # This is a diagnostic window, not an asserted managed object extent.
        self.q(self.entry_sp, self.stop)
        self.u.mem_write(self.entry_sp + 8, bytes([0xB6]) * 0x38)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        if rva in self.constructor_instructions:
            self.executed.add(rva)
            return
        if rva == BASE_GATEWAY:
            self.executed.add(rva)
            owner = 0 if self.options.get('null_owner') else self.p['reveal']
            assert self.reg(x.UC_X86_REG_RCX) == owner
            assert self.reg(x.UC_X86_REG_RDX) == 0  # EDX writes zero-extend.
            assert self.reg(x.UC_X86_REG_R8) == 0xDEADBEEF11111111
            assert self.reg(x.UC_X86_REG_R9) == 0xDEADBEEF22222222
            assert self.reg(x.UC_X86_REG_RSP) == self.entry_sp
            assert self.rq(self.entry_sp) == self.stop  # Actual tail-call return address.
            if self.event('monobehaviour_constructor_service', [self.oid(owner), 0,
                    self.reg(x.UC_X86_REG_R8), self.reg(x.UC_X86_REG_R9)]):
                # Explicit service effects, never attributed to the native caller.
                mutation = self.options.get('base_effect', 'retain')
                assert mutation in ['retain', 'clear_text', 'replace_text']
                if mutation != 'retain' and owner:
                    self.q(owner + 0x20, self.p['other_text'] if mutation == 'replace_text' else 0)
                self.base_acceptances.append({'owner': self.oid(owner), 'effect': mutation})
                self.ret(0xFADE1234DEADBEEF)  # Void caller has no object-return contract.
            return
        super().hook(uc, address, size, data)

    def invoke_constructor(self):
        x, sp = self.x, self.entry_sp
        self.q(sp, self.stop)
        integer_regs = [getattr(x, 'UC_X86_REG_' + n) for n in
                        ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, r in enumerate(integer_regs): self.u.reg_write(r, 0xFAB00000 + i)
        for i, r in enumerate(vectors): self.u.reg_write(r, (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64))
        method_info = self.options.get('method_info', 'nonnull')
        info = {'null': 0, 'nonnull': self.p['method'], 'poison': 0xFACE123456789ABC}[method_info]
        for r, v in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, 0 if self.options.get('null_owner') else self.p['reveal']),
                     (x.UC_X86_REG_RDX, info), (x.UC_X86_REG_R8, 0xDEADBEEF11111111),
                     (x.UC_X86_REG_R9, 0xDEADBEEF22222222)]: self.u.reg_write(r, v)
        self.u.emu_start(self.base + ENTRY, self.stop, timeout=10_000_000, count=1000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error == 'monobehaviour_constructor_service'
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(r) == 0xFAB00000 + i for i, r in enumerate(integer_regs))
            assert all(self.reg(r) == (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64) for i, r in enumerate(vectors))
            assert all(self.reg(getattr(x, 'UC_X86_REG_' + n)) == 0xFACE123456789090
                       for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11'])
            assert all(self.reg(getattr(x, f'UC_X86_REG_XMM{i}')) == (1 << 127) | i for i in range(6))
            assert self.reg(x.UC_X86_REG_RAX) == 0xFADE1234DEADBEEF
        return returned

    def run_constructor(self, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        initial, old = self.snapshot(), len(self.events)
        returned = self.invoke_constructor()
        final, events = self.snapshot(), self.events[old:].copy()
        assert len(events) == 1 and events[0]['kind'] == 'monobehaviour_constructor_service'
        assert events[0]['snapshot'] == initial  # No native object or stack writes.
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            allowed = range(0x20, 0x28) if (n == 'reveal' and returned and
                not self.options.get('null_owner') and self.options.get('base_effect', 'retain') != 'retain') else []
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        assert final['stack_window_hex'] == initial['stack_window_hex']
        if self.options.get('base_effect', 'retain') == 'retain': assert final['memory'] == initial['memory']
        return {'method': '.ctor', 'options': self.options.copy(), 'returned': returned,
                'error': self.error, 'initial': initial, 'events': events, 'final': final,
                'native_object_and_stack_writes': 0,
                'win64_nonvolatile_stack_and_supplied_volatile_poison_verified': returned}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, stops, baselines, sequences = [], [], [], []
    for storage in ['fresh_zero', 'serialized', 'reused']:
        for null in [False, True]:
            for info in ['null', 'nonnull', 'poison']:
                for effect in ['retain', 'clear_text', 'replace_text']:
                    cases.append(m.run_constructor({'owner_storage': storage, 'null_owner': null,
                        'method_info': info, 'base_effect': effect}))
            options = {'owner_storage': storage, 'null_owner': null}
            baseline = m.run_constructor(options); baselines.append(baseline)
            stop = m.run_constructor({**options, 'failure': ['monobehaviour_constructor_service', 1]})
            assert not stop['returned']
            assert stop['events'] == baseline['events']
            assert stop['final'] == baseline['events'][0]['snapshot']
            stops.append(stop)
    # These joins receive an explicit already-serialized component graph. The
    # constructor neither discovers nor initializes its text reference.
    for storage in ['serialized', 'reused', 'fresh_zero']:
        for effect in ['retain', 'replace_text', 'clear_text']:
            m.prepare({'owner_storage': storage})
            rows = [m.run_constructor({'base_effect': effect}, retained=True)]
            null_text = m.rq(m.p['reveal'] + 0x20) == 0
            rows += [m.run('Init', {'order_bits': 0x80000003, 'null_text': null_text}, retained=True),
                     m.run('Hide', retained=True)]
            assert rows[0]['returned'] and rows[2]['returned']
            expected_init = effect == 'replace_text' or (effect == 'retain' and storage != 'fresh_zero')
            assert rows[1]['returned'] == expected_init
            sequences.append(rows)
    for storage in ['serialized', 'reused']:
        m.prepare({'owner_storage': storage})
        rows = [m.run_constructor(retained=True), m.run('Init', {'order_bits': 0xFFFFFFFF}, retained=True),
                m.run_constructor(retained=True), m.run('Hide', retained=True),
                m.run('Init', {'order_bits': 0}, retained=True)]
        assert all(r['returned'] for r in rows)
        assert rows[2]['initial']['memory'] == rows[2]['final']['memory']
        sequences.append(rows)
    assert set(m.constructor_instructions) <= m.executed
    return {'schema': 'reveal_order_constructor_native_v1', 'build': BUILD,
            'scope': 'exact RevealOrder folded constructor caller; base services supplied; explicit serialized presentation joins',
            'target': m.constructor_target, 'shared_wrapper_alias_count': len(m.wrapper_aliases),
            'base_gateway_aliases': m.base_aliases, 'nominal_base': 'UnityEngine.MonoBehaviour',
            'body_bounds': {'start': hex(ENTRY), 'end_exclusive': hex(END), 'next_managed': hex(FOLLOWING),
                            'alignment_cc_bytes': FOLLOWING - END},
            'unwind_entries_containing_entry': m.unwind_entries,
            'field_pin': {'type': 'RevealOrder', 'type_def_index': 5735, 'declaration': m.field_declaration},
            'constructor_instructions': [{'rva': hex(a), 'mnemonic': i.mnemonic, 'operands': i.op_str}
                                         for a, i in m.constructor_instructions.items()],
            'diagnostic_windows_not_object_extent_or_unused_typed_field_claims': m.sizes,
            'joined_targets': m.targets, 'joined_supplied_targets': m.service_targets,
            'presentation_instruction_assertions': len(m.checks),
            'cases': cases, 'baselines': baselines, 'failure_stops': stops, 'retained_sequences': sequences,
            'summary': {'cases': len(cases), 'baselines': len(baselines), 'stops': len(stops),
                        'retained_sequences': len(sequences), 'constructor_instructions': 2}}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--game-root', type=Path, required=True)
    p.add_argument('--dumper-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(serialize_report(result), encoding='utf-8')
    print(json.dumps(result['summary'], sort_keys=True))
