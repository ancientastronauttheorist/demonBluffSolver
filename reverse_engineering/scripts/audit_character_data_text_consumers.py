"""Six actual CharacterData text callers; RNG/localization/Unity helpers supplied."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

from audit_character_assets import BUILD
from audit_character_data_consumers import Machine as DataVerifier
from audit_character_oracle_reveal_join import expand_memory, pool_memory
from audit_report_snapshots import expand_snapshots, pool_snapshots

TARGETS = {
    'GetFlavorText': (0x3B4CA0, 0x3B4CEB, 0x3B4CF0, 3, 'iii'),
    'GetTranslatedName': (0x3B4D60, 0x3B4D97, 0x3B4DA0, 4, 'iiii'),
    'GetIWasTranslated': (0x3B4D10, 0x3B4D47, 0x3B4D50, 5, 'iiii'),
    'GetIfLies': (0x3B4D50, 0x3B4D5E, 0x3B4D60, 16, 'iii'),
    'GetHints': (0x3B4D00, 0x3B4D0B, 0x3B4D10, 17, 'iii'),
    'UpdateCharacterName': (0x3B5070, 0x3B5094, 0x3B50A0, 19, 'vii'),
}
TEXT_FIELDS = {'name': 0x28, 'flavor': 0x68, 'array': 0x70, 'hints': 0x78,
               'if_lies': 0x80, 'translation': 0x148}
SERVICES = {0x1C86600: 'UnityEngine.Random$$Range', 0x1C82250: 'UnityEngine.Object$$get_name',
            0x3A83C0: 'StringHelper$$ConvertTextToTextWithTooltips',
            0x3F5920: 'CharacterLoc$$GetTranslatedName', 0x3F5690: 'CharacterLoc$$GetIWasTranslated'}
POISON = 0xFACE123456789090


class Machine(DataVerifier):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        dump_raw = (Path(dumper_root) / 'dump.cs').read_bytes()
        extraction = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(dump_raw).hexdigest().upper() == extraction['outputs']['dump_cs']['sha256'].upper()
        dump = dump_raw.decode('utf-8-sig')
        body = re.search(r'^public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert body
        for line in ['public string characterName; // 0x28', 'public string flavorText; // 0x68',
                     'public string[] additionalFlavorTexts; // 0x70', 'public string hints; // 0x78',
                     'public string ifLies; // 0x80', 'public CharacterLoc translation; // 0x148']:
            assert line in body[1]
        self.instructions, self.targets, self.bounds, self.flags = {}, [], {}, {}
        for name, (start, end, following, ordinal, abi) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == 'CharacterData$$'+name and r['Address'] == start]
            ret = 'void' if name == 'UpdateCharacterName' else 'System_String_o*'
            extra = 'System_String_o* localeCode, ' if abi == 'iiii' else ''
            assert len(rows) == 1 and rows[0]['TypeSignature'] == abi
            assert rows[0]['Signature'] == f'{ret} CharacterData__{name} (CharacterData_o* __this, {extra}const MethodInfo* method);'
            assert len([r for r in self.metadata['ScriptMethod'] if r['Address'] == start]) == 1
            self.targets.append(dict(rows[0], method_id=f'tdi5845.m{ordinal:04}'))
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start) == following
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks == ([] if name in ['GetIfLies', 'GetHints'] else [(start, end)])
            section = self.pe.get_section_by_rva(start)
            assert section and following <= section.VirtualAddress + section.SizeOfRawData
            raw = self.pe.get_data(start, following-start)
            assert len(raw) == following-start and raw[end-start:] == b'\xcc'*(following-end)
            ins = list(self.cs.disasm(raw[:end-start], start))
            assert sum(i.size for i in ins) == end-start
            self.instructions.update({i.address: i for i in ins})
            self.bounds[name] = dict(start=hex(start), end_exclusive=hex(end), next_managed=hex(following),
                                     unwind=[[hex(a), hex(b)] for a, b in chunks], byte_length=end-start,
                                     decoded_instructions=len(ins), body_sha256=hashlib.sha256(raw[:end-start]).hexdigest())
        signatures = {
            0x1C86600: 'int32_t UnityEngine_Random__Range (int32_t minInclusive, int32_t maxExclusive, const MethodInfo* method);',
            0x1C82250: 'System_String_o* UnityEngine_Object__get_name (UnityEngine_Object_o* __this, const MethodInfo* method);',
            0x3A83C0: 'System_String_o* StringHelper__ConvertTextToTextWithTooltips (System_String_o* inputText, const MethodInfo* method);',
            0x3F5920: 'System_String_o* CharacterLoc__GetTranslatedName (CharacterLoc_o* __this, System_String_o* localeCode, const MethodInfo* method);',
            0x3F5690: 'System_String_o* CharacterLoc__GetIWasTranslated (CharacterLoc_o* __this, System_String_o* localeCode, const MethodInfo* method);',
        }
        self.supplied = []
        for address, name in SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['Signature'] == signatures[address]
            self.supplied += rows
        self.checks = {
            0x3B4CA6: ('mov', 'rbx, qword ptr [rcx + 0x70]'),
            0x3B4CAF: ('cmp', 'qword ptr [rbx + 0x18], 0'),
            0x3B4CB6: ('mov', 'rax, qword ptr [rcx + 0x68]'),
            0x3B4CC0: ('mov', 'edx, dword ptr [rbx + 0x18]'),
            0x3B4CC8: ('call', '0x1c86600'), 0x3B4CCD: ('cdqe', ''),
            0x3B4CCF: ('cmp', 'eax, dword ptr [rbx + 0x18]'),
            0x3B4CD2: ('jae', '0x3b4ce5'),
            0x3B4CD4: ('mov', 'rax, qword ptr [rbx + rax*8 + 0x20]'),
            0x3B4CDF: ('call', '0x2b7d90'), 0x3B4CE5: ('call', '0x2b7d80'),
            0x3B4D69: ('mov', 'rcx, qword ptr [rcx + 0x148]'),
            0x3B4D78: ('call', '0x3f5920'), 0x3B4D8C: ('jmp', '0x1c82250'),
            0x3B4D19: ('mov', 'rcx, qword ptr [rcx + 0x148]'),
            0x3B4D28: ('call', '0x3f5690'), 0x3B4D3C: ('jmp', '0x1c82250'),
            0x3B4D50: ('mov', 'rcx, qword ptr [rcx + 0x80]'),
            0x3B4D00: ('mov', 'rcx, qword ptr [rcx + 0x78]'),
            0x3B5080: ('lea', 'rcx, [rbx + 0x28]'),
            0x3B5087: ('mov', 'qword ptr [rcx], rax'),
            0x3B508F: ('jmp', '0x2b6ff0'),
        }
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        names = ['data', 'other_data', 'data_class', 'array', 'other_array', 'array_class', 'flavor', 'hints',
                 'if_lies', 'element0', 'element1', 'element2', 'translation', 'other_translation', 'locale',
                 'other_locale', 'translated', 'converted', 'object_name', 'other_name', 'replacement']
        self.p = {n: self.arena+0x180000+i*0x1000 for i, n in enumerate(names)}
        self.ids = {p: n for n, p in self.p.items()}
        self.sizes = {n: 0x200 if n in ['data', 'other_data'] else 0x100 if 'class' in n or 'array' in n else 0x80 for n in names}
        self.tracking_writes = False
        self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE, self.observe_text_write)

    def observe_text_write(self, uc, access, address, size, value, unused):
        if not self.tracking_writes: return
        if address == self.p['data']+0x28:
            assert self.reg(self.x.UC_X86_REG_RIP)-self.base == 0x3B5087 and size == 8
            self.allowed.setdefault('data', set()).update(range(0x28, 0x30))

    def snapshot(self):
        return dict(fields={n: self.oid(self.rq(self.p['data']+off)) for n, off in TEXT_FIELDS.items()},
                    arrays={n: dict(length_bits=self.rq(self.p[n]+0x18),
                        elements=[self.oid(self.rq(self.p[n]+0x20+i*8)) for i in range(3)]) for n in ['array', 'other_array']},
                    service_history=deepcopy(self.history), native_entries=deepcopy(self.entries),
                    memory={n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()})

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options.copy(), [], {}, None
        self.allowed, self.history, self.entries = {}, [], []
        for n, p in self.p.items(): self.u.mem_write(p, bytes([0xA5])*self.sizes[n])
        for n in ['data', 'other_data']:
            self.q(self.p[n], self.p['data_class'])
            for field, off in TEXT_FIELDS.items():
                target = 'object_name' if field == 'name' else field
                if field == 'array': target = options.get('initial_array', 'array')
                if options.get('alias_texts') and field in ['name', 'flavor', 'hints', 'if_lies']: target = 'element1'
                self.q(self.p[n]+off, 0 if options.get('null_field') == field else self.p[target])
        for n in ['array', 'other_array']:
            self.q(self.p[n], self.p['array_class'])
            self.q(self.p[n]+0x18, options.get('length_bits', 3) if n == 'array' else 2)
            for i in range(3): self.q(self.p[n]+0x20+i*8, 0 if options.get('null_element') == i else self.p['element1' if options.get('alias_texts') else 'element'+str(2-i if n == 'other_array' else i)])
        self.phase_counts = {}

    def mutate(self, phase):
        self.phase_counts[phase] = self.phase_counts.get(phase, 0)+1
        if self.options.get('mutation_phase') != phase or self.options.get('mutation_occurrence', 1) != self.phase_counts[phase]: return
        action = self.options['mutation']
        def write(n, off, value):
            self.q(self.p[n]+off, value); self.allowed.setdefault(n, set()).update(range(off, off+8))
        if action in ['replace_array', 'clear_array']: write('data', 0x70, self.p['other_array'] if action == 'replace_array' else 0)
        elif action in ['replace_translation', 'clear_translation']: write('data', 0x148, self.p['other_translation'] if action == 'replace_translation' else 0)
        elif action == 'replace_element': write('array', 0x28, self.p['replacement'])
        elif action == 'clear_element': write('array', 0x28, 0)
        elif action == 'shrink_array': write('array', 0x18, 1)
        elif action == 'zero_array': write('array', 0x18, 0)
        elif action == 'replace_name': write('data', 0x28, self.p['other_name'])
        else: raise AssertionError(action)

    def event(self, kind, args):
        result = super().event(kind, args)
        self.events[-1].update(raw_args=[self.reg(getattr(self.x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']],
                               caller_return_bits=self.rq(self.reg(self.x.UC_X86_REG_RSP)), native_site=hex(self.last_native))
        return result

    def hook(self, uc, address, size, unused):
        if address == self.stop: return
        rva, x = address-self.base, self.x; self.executed.add(rva)
        if rva in self.instructions:
            self.last_native = rva
            if rva == TARGETS[self.method][0]: self.entries.append(dict(method=self.method, raw_args=[self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']]))
            return
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        phase, value = None, 0
        if rva == 0x1C86600:
            captured_array = self.reg(x.UC_X86_REG_RBX)
            assert self.oid(captured_array) in ['array', 'other_array'] and cx == r8 == 0 and dx == self.rd(captured_array+0x18)
            value = self.options.get('random_return_bits', 0xCAFE123400000001)
            kind, args, phase = SERVICES[rva], [cx, dx, r8, value], 'random'
        elif rva in [0x3F5920, 0x3F5690]:
            assert self.oid(cx) in ['translation', 'other_translation'] and dx == self.locale_arg and r8 == 0
            value = 0 if self.options.get('null_translation_result') else self.p[self.options.get('translation_result', 'translated')]
            kind, args, phase = SERVICES[rva], [self.oid(cx), self.oid(dx), r8, self.oid(value)], 'translation'
        elif rva == 0x1C82250:
            assert cx == self.owner_arg and dx == 0
            value = 0 if self.options.get('null_name_result') else self.p[self.options.get('name_result', 'object_name')]
            kind, args, phase = SERVICES[rva], [self.oid(cx), dx, self.oid(value)], 'name'
        elif rva == 0x3A83C0:
            assert cx in [0, self.p['hints'], self.p['if_lies'], self.p['element1']] and dx == 0
            value = 0 if self.options.get('null_converter_result') else self.p[self.options.get('converter_result', 'converted')]
            kind, args, phase = SERVICES[rva], [self.oid(cx), dx, self.oid(value)], 'converter'
        elif rva == 0x2B6FF0:
            assert cx == self.owner_arg+0x28 and dx == self.last_name and self.rq(cx) == dx
            kind, args, phase = 'reference_barrier', ['data', 0x28, self.oid(dx)], 'barrier'
        elif rva in [0x2B7D90, 0x2B7D80]:
            kind = 'native_null_guard' if rva == 0x2B7D90 else 'native_bounds_guard'
            self.event(kind, []); self.error = kind; uc.emu_stop(); return
        else: raise AssertionError(f'unclaimed {rva:x}')
        if self.event(kind, args):
            self.history.append(dict(kind=kind, args=deepcopy(args)))
            self.mutate(phase)
            if rva == 0x1C82250: self.last_name = value
            self.ret(value)

    def run(self, method, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error, self.allowed, self.counts, self.phase_counts = options or {}, None, {}, {}, {}
        self.method = method; self.fault = None
        self.owner_arg = 0 if self.options.get('null_owner') else self.p['data']
        self.locale_arg = 0 if self.options.get('null_locale') else self.p['locale']
        x, sp = self.x, self.stack+0x18008; self.q(sp, self.stop)
        initial, old = self.snapshot(), len(self.events)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_'+n), 0xFAB00000+i)
        for i in range(6, 16): self.u.reg_write(getattr(x, f'UC_X86_REG_XMM{i}'), (1<<125)|i)
        incoming = [self.owner_arg, self.locale_arg if method in ['GetTranslatedName', 'GetIWasTranslated'] else 0xDEAD123400000002,
                    self.options.get('entry_r8_bits', 0xDEAD123400000008), self.options.get('entry_r9_bits', 0xDEAD123400000009)]
        for n, value in zip(['RCX', 'RDX', 'R8', 'R9'], incoming): self.u.reg_write(getattr(x, 'UC_X86_REG_'+n), value)
        for n, value in [('RSP', sp), ('R10', 0xABCD0010), ('R11', 0xABCD0011), ('MXCSR', 0x1F80)]: self.u.reg_write(getattr(x, 'UC_X86_REG_'+n), value)
        self.tracking_writes = True
        try: self.u.emu_start(self.base+TARGETS[method][0], self.stop, timeout=10_000_000, count=10000)
        except self.unicorn.UcError as exc:
            assert self.options.get('null_owner')
            pc = self.reg(x.UC_X86_REG_RIP)-self.base
            expected = {'GetFlavorText': 0x3B4CA6, 'GetTranslatedName': 0x3B4D69, 'GetIWasTranslated': 0x3B4D19,
                        'GetIfLies': 0x3B4D50, 'GetHints': 0x3B4D00, 'UpdateCharacterName': 0x3B5087}[method]
            assert pc == expected and exc.errno == (self.unicorn.UC_ERR_WRITE_UNMAPPED if method == 'UpdateCharacterName' else self.unicorn.UC_ERR_READ_UNMAPPED)
            self.error, self.fault = 'native_owner_access_fault', hex(pc)
        finally: self.tracking_writes = False
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        result = self.reg(x.UC_X86_REG_RAX) if returned else None
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp+8
            assert all(self.reg(getattr(x, 'UC_X86_REG_'+n)) == 0xFAB00000+i for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']))
            assert all(self.reg(getattr(x, f'UC_X86_REG_XMM{i}')) == (1<<125)|i for i in range(6, 16))
            if method != 'UpdateCharacterName': self.oid(result)
        final = self.snapshot()
        for n, raw in initial['memory'].items():
            assert all(i in self.allowed.get(n, set()) or b == bytes.fromhex(final['memory'][n])[i] for i, b in enumerate(bytes.fromhex(raw))), n
        row = dict(method=method, options=self.options.copy(), entry_raw_args=incoming, returned=returned, result_bits=result,
                   error=self.error, fault_rva=self.fault, initial=initial, events=deepcopy(self.events[old:]), final=final,
                   completed_memory_write_offsets={n: sorted(v) for n, v in self.allowed.items()}, normal_abi_verified=returned)
        verify(row, self)
        return row


def verify(row, m):
    """Independent full-byte, raw-ABI ordered model; no reads of native output."""
    model = deepcopy(row['initial']); mem = {n: bytearray.fromhex(v) for n, v in model['memory'].items()}
    options, regs = row['options'], row['entry_raw_args'].copy(); events, phases = [], {}
    error, fault, returned, result = None, None, False, None
    def word(n, off): return int.from_bytes(mem[n][off:off+8], 'little')
    def snapshot():
        out = deepcopy(model); out['memory'] = {n: bytes(v).hex() for n, v in mem.items()}
        out['fields'] = {n: m.oid(word('data', off)) for n, off in TEXT_FIELDS.items()}
        out['arrays'] = {n: dict(length_bits=word(n, 0x18), elements=[m.oid(word(n, 0x20+i*8)) for i in range(3)]) for n in ['array', 'other_array']}
        return out
    def write(n, off, value): mem[n][off:off+8] = value.to_bytes(8, 'little')
    def mutate(phase):
        phases[phase] = phases.get(phase, 0)+1
        if options.get('mutation_phase') != phase or options.get('mutation_occurrence', 1) != phases[phase]: return
        action = options['mutation']
        if action in ['replace_array', 'clear_array']: write('data', 0x70, m.p['other_array'] if action == 'replace_array' else 0)
        elif action in ['replace_translation', 'clear_translation']: write('data', 0x148, m.p['other_translation'] if action == 'replace_translation' else 0)
        elif action == 'replace_element': write('array', 0x28, m.p['replacement'])
        elif action == 'clear_element': write('array', 0x28, 0)
        elif action == 'shrink_array': write('array', 0x18, 1)
        elif action == 'zero_array': write('array', 0x18, 0)
        elif action == 'replace_name': write('data', 0x28, m.p['other_name'])
        else: raise AssertionError(action)
    class Stopped(Exception): pass
    def emit(kind, args, raw, site, phase=None, tail=False, terminal=False):
        nonlocal regs, error
        event = dict(kind=kind, args=args, snapshot=snapshot(), raw_args=raw,
                     caller_return_bits=m.stop if tail else m.base+site+m.instructions[site].size, native_site=hex(site))
        events.append(event)
        if terminal or options.get('failure') == [kind, sum(e['kind'] == kind for e in events)]: error = kind; raise Stopped
        model['service_history'].append(dict(kind=kind, args=deepcopy(args)))
        if phase: mutate(phase)
        regs = [POISON]*4
    name = row['method']; model['native_entries'].append(dict(method=name, raw_args=regs.copy()))
    try:
        if name == 'UpdateCharacterName':
            out = 0 if options.get('null_name_result') else m.p[options.get('name_result', 'object_name')]
            emit(SERVICES[0x1C82250], [m.oid(regs[0]), 0, m.oid(out)], [regs[0], 0, regs[2], regs[3]], 0x3B507B, 'name')
            if row['entry_raw_args'][0] == 0: error, fault = 'native_owner_access_fault', '0x3b5087'
            else:
                write('data', 0x28, out)
                emit('reference_barrier', ['data', 0x28, m.oid(out)], [m.p['data']+0x28, out, regs[2], regs[3]], 0x3B508F, 'barrier', tail=True)
                returned, result = True, 0
        elif row['entry_raw_args'][0] == 0:
            error = 'native_owner_access_fault'
            fault = hex({'GetFlavorText': 0x3B4CA6, 'GetTranslatedName': 0x3B4D69, 'GetIWasTranslated': 0x3B4D19, 'GetIfLies': 0x3B4D50, 'GetHints': 0x3B4D00}[name])
        elif name == 'GetFlavorText':
            array = m.oid(word('data', 0x70))
            if array is None: emit('native_null_guard', [], regs, 0x3B4CDF, terminal=True)
            if word(array, 0x18) == 0: result = word('data', 0x68)
            else:
                bits = options.get('random_return_bits', 0xCAFE123400000001); maximum = word(array, 0x18)&0xFFFFFFFF
                emit(SERVICES[0x1C86600], [0, maximum, 0, bits], [0, maximum, 0, regs[3]], 0x3B4CC8, 'random')
                index = bits&0xFFFFFFFF
                if index >= word(array, 0x18)&0xFFFFFFFF: emit('native_bounds_guard', [], regs, 0x3B4CE5, terminal=True)
                assert index < 3
                result = word(array, 0x20+index*8)
            returned = True
        elif name in ['GetTranslatedName', 'GetIWasTranslated']:
            translation = m.oid(word('data', 0x148)); out = 0
            if translation:
                address, site = (0x3F5920, 0x3B4D78) if name == 'GetTranslatedName' else (0x3F5690, 0x3B4D28)
                out = 0 if options.get('null_translation_result') else m.p[options.get('translation_result', 'translated')]
                emit(SERVICES[address], [translation, m.oid(row['entry_raw_args'][1]), 0, m.oid(out)],
                     [m.p[translation], row['entry_raw_args'][1], 0, regs[3]], site, 'translation')
            if out == 0:
                out = 0 if options.get('null_name_result') else m.p[options.get('name_result', 'object_name')]
                emit(SERVICES[0x1C82250], ['data', 0, m.oid(out)], [m.p['data'], 0, regs[2], regs[3]],
                     0x3B4D8C if name == 'GetTranslatedName' else 0x3B4D3C, 'name', tail=True)
            returned, result = True, out
        else:
            text = m.oid(word('data', 0x78 if name == 'GetHints' else 0x80))
            out = 0 if options.get('null_converter_result') else m.p[options.get('converter_result', 'converted')]
            emit(SERVICES[0x3A83C0], [text, 0, m.oid(out)], [m.p[text] if text else 0, 0, regs[2], regs[3]],
                 0x3B4D06 if name == 'GetHints' else 0x3B4D59, 'converter', tail=True)
            returned, result = True, out
    except Stopped: pass
    assert row['events'] == events, (name, options, 'events')
    assert (row['returned'], row['error'], row['fault_rva'], row['result_bits']) == (returned, error, fault, result), (name, options, 'outcome')
    assert row['final'] == snapshot(), (name, options, 'final')


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for length, bits, null in itertools.product([0, 1, 2, 3, 1<<32], [0xCAFE123400000000, 0xFFFF123400000001, 0x8000000000000002, 0xCAFE0000FFFFFFFF], [None, 1]):
        cases.append(m.run('GetFlavorText', dict(length_bits=length, random_return_bits=bits, null_element=null)))
    for name in TARGETS:
        for options in [{}, {'null_owner': True}, {'entry_r8_bits': 0xFEDCBA9876543210, 'entry_r9_bits': 0x123456789ABCDEF0}]: cases.append(m.run(name, options))
    for name in ['GetTranslatedName', 'GetIWasTranslated']:
        for null_loc, null_result, null_locale, null_name in itertools.product([False, True], repeat=4):
            options = dict(null_translation_result=null_result, null_locale=null_locale, null_name_result=null_name)
            if null_loc: options['null_field'] = 'translation'
            cases.append(m.run(name, options))
        for phase, action in itertools.product(['translation', 'name'], ['replace_translation', 'clear_translation', 'replace_name']):
            cases.append(m.run(name, dict(null_translation_result=True, mutation_phase=phase, mutation=action)))
    for name, field in [('GetHints', 'hints'), ('GetIfLies', 'if_lies')]:
        for null_input, null_output in itertools.product([False, True], repeat=2):
            options = dict(null_converter_result=null_output)
            if null_input: options['null_field'] = field
            cases.append(m.run(name, options))
    cases.append(m.run('GetFlavorText', dict(null_field='array')))
    cases.append(m.run('GetFlavorText', dict(length_bits=0, null_field='flavor')))
    for action in ['replace_array', 'clear_array', 'replace_element', 'clear_element', 'shrink_array', 'zero_array', 'replace_name']:
        cases.append(m.run('GetFlavorText', dict(mutation_phase='random', mutation=action)))
        cases.append(m.run('GetFlavorText', dict(length_bits=0, mutation_phase='random', mutation=action)))
    for phase, action in itertools.product(['name', 'barrier'], ['replace_name', 'clear_array', 'replace_translation']):
        cases.append(m.run('UpdateCharacterName', dict(mutation_phase=phase, mutation=action)))
    cases.append(m.run('UpdateCharacterName', dict(null_name_result=True)))
    for name in TARGETS:
        cases.append(m.run(name, dict(alias_texts=True)))
        for options in [dict(name_result='element1'), dict(translation_result='locale'), dict(converter_result='element1')]:
            cases.append(m.run(name, dict(options, alias_texts=True)))
    for bits in [0xFFFFFFFF00000000, 0xCAFE123400000001]:
        cases.append(m.run('GetFlavorText', dict(initial_array='other_array', random_return_bits=bits)))
    for first in ['GetFlavorText', 'GetTranslatedName', 'GetHints']:
        rows = [m.run(first), m.run('UpdateCharacterName', dict(mutation_phase='barrier', mutation='replace_name'), True), m.run('GetIWasTranslated', dict(null_translation_result=True), True)]
        assert all(r['returned'] for r in rows); sequences.append(rows)
    rows = [m.run('GetFlavorText', dict(mutation_phase='random', mutation='replace_array')),
            m.run('GetFlavorText', dict(random_return_bits=0xFFFFFFFF00000000), True),
            m.run('GetFlavorText', dict(random_return_bits=0xCAFE123400000001), True)]
    assert all(r['returned'] for r in rows); sequences.append(rows)
    profiles = [(name, {}) for name in TARGETS]
    profiles += [('GetFlavorText', dict(mutation_phase='random', mutation='replace_array')),
                 ('GetFlavorText', dict(mutation_phase='random', mutation='shrink_array')),
                 ('GetFlavorText', dict(initial_array='other_array', random_return_bits=0xFFFFFFFF00000000)),
                 ('GetHints', dict(alias_texts=True, converter_result='element1')),
                 ('GetTranslatedName', dict(null_translation_result=True, mutation_phase='translation', mutation='clear_translation')),
                 ('UpdateCharacterName', dict(mutation_phase='name', mutation='replace_name'))]
    for name, options in profiles:
        baseline = m.run(name, options); bid = len(baselines); baselines.append(baseline); counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0)+1
            stopped = m.run(name, {**options, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index+1] and stopped['final'] == event['snapshot']
            stops.append(dict(baseline=bid, prefix_length=index+1, result=stopped))
    for rows in sequences:
        assert all(current['initial'] == prior['final'] for prior, current in zip(rows, rows[1:]))
    missing = sorted(set(m.instructions)-m.executed)
    assert all(m.instructions[a].mnemonic == 'int3' for a in missing), [hex(a) for a in missing]
    return dict(schema='character_data_text_consumers_native_v1', build=BUILD, targets=m.targets, supplied_targets=m.supplied,
                bounds=m.bounds, field_offsets=TEXT_FIELDS, instruction_assertions=len(m.checks),
                decoded_instructions=len(m.instructions), covered_instructions=len(set(m.instructions)&m.executed),
                unexecuted_terminal_traps=[hex(a) for a in missing], cases=cases, retained_sequences=sequences,
                baselines=baselines, failure_stops=stops, summary=dict(cases=len(cases), sequences=len(sequences), baselines=len(baselines), stops=len(stops)),
                scope='six exact CharacterData text callers; Random.Range, CharacterLoc, StringHelper, Object.get_name and barriers supplied; GetDescription, renderer, localization bodies, runtime admission, scheduling and unwinding excluded')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('game_root'); parser.add_argument('dumper_root'); parser.add_argument('--output', required=True)
    args = parser.parse_args(); full = audit(args.game_root, args.dumper_root)
    report = pool_snapshots(pool_memory(full))
    assert expand_memory(expand_snapshots(report)) == full
    Path(args.output).write_text(json.dumps(report, sort_keys=True, separators=(',', ':'))+'\n', encoding='utf-8')
    print(json.dumps(full['summary']))
