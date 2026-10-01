"""Execute thirteen exact Character field accessors/mutators offline."""
import argparse
import hashlib
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine


# Name, exact entry, operation, consumed field offset, width, declaring field.
TARGETS = [
    ('get_onClick', 0x3698B0, 'get', 0x100, 8, 'private Action <onClick>k__BackingField'),
    ('set_onClick', 0x3698D0, 'set_reference', 0x100, 8, 'private Action <onClick>k__BackingField'),
    ('get_onReveal', 0x3698C0, 'get', 0x108, 8, 'private Action <onReveal>k__BackingField'),
    ('set_onReveal', 0x3698E0, 'set_reference', 0x108, 8, 'private Action <onReveal>k__BackingField'),
    ('CreateRuntimeData', 0x364A10, 'set_reference', 0x70, 8, 'private RuntimeCharacterData runtimeData'),
    ('GetRuntimeData', 0x365140, 'get', 0x70, 8, 'private RuntimeCharacterData runtimeData'),
    ('GetRealAlignment', 0x365020, 'get', 0xF8, 4, 'public EAlignment alignment'),
    ('GetState', 0x365150, 'get', 0xE4, 4, 'public ECharacterState state'),
    ('ChangeAlignment', 0x364960, 'set_scalar', 0xF8, 4, 'public EAlignment alignment'),
    ('UpdateTrailerInfo', 0x369460, 'set_reference', 0x68, 8, 'private CharacterTrailerInfo trailerInfo'),
    ('Uninteractable', 0x369440, 'set_true', 0x178, 1, 'private bool uninteractable'),
    ('Interactable', 0x365D80, 'set_false', 0x178, 1, 'private bool uninteractable'),
    ('UpdateRegisterAsRole', 0x369450, 'set_reference', 0x60, 8, 'public CharacterData registerAs'),
]


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root)
        ext = json.loads((Path(__file__).parents[1] /
                          f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pinned(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == ext['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        script = json.loads(pinned('script.json', 'script_json'))
        dump = pinned('dump.cs', 'dump_cs')
        block = re.search(r'^public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487\s*\{(.*?)// Properties',
                          dump, re.M | re.S)
        assert block
        self.targets, self.instruction_count = [], 0
        pointer_args = {'set_onClick': ('System_Action_o*', 'value'), 'set_onReveal': ('System_Action_o*', 'value'),
                        'CreateRuntimeData': ('RuntimeCharacterData_o*', 'rData'),
                        'UpdateTrailerInfo': ('CharacterTrailerInfo_o*', 'trailerInfo'),
                        'UpdateRegisterAsRole': ('CharacterData_o*', 'cd')}
        returns = {'get_onClick': 'System_Action_o*', 'get_onReveal': 'System_Action_o*',
                   'GetRuntimeData': 'RuntimeCharacterData_o*', 'GetRealAlignment': 'int32_t', 'GetState': 'int32_t'}
        for name, rva, operation, offset, width, field in TARGETS:
            assert f'{field}; // 0x{offset:X}' in block[1]
            rows = [r for r in script['ScriptMethod'] if r['Name'] == 'Character$$' + name and r['Address'] == rva]
            assert len(rows) == 1
            argument = ''
            if name in pointer_args:
                ty, arg = pointer_args[name]
                argument = f'{ty} {arg}, '
            elif name == 'ChangeAlignment':
                argument = 'int32_t alig, '
            signature = f'{returns.get(name, "void")} Character__{name} (Character_o* __this, {argument}const MethodInfo* method);'
            assert rows[0]['Signature'] == signature
            # These pointer-backed leaves have no containing unwind record.
            assert not any(e.struct.BeginAddress <= rva < e.struct.EndAddress for e in self.pe.DIRECTORY_ENTRY_EXCEPTION)
            decoded = []
            for i in self.cs.disasm(self.pe.get_data(rva, 16), rva):
                decoded.append(i)
                if i.mnemonic in ['ret', 'jmp']:
                    break
            if operation == 'get':
                expected = [('mov', f'{"rax, qword" if width == 8 else "eax, dword"} ptr [rcx + {hex(offset)}]'), ('ret', '')]
            elif operation == 'set_reference':
                expected = [('add', f'rcx, {hex(offset)}'), ('mov', 'qword ptr [rcx], rdx'), ('jmp', '0x2b6ff0')]
            elif operation == 'set_scalar':
                expected = [('mov', f'dword ptr [rcx + {hex(offset)}], edx'), ('ret', '')]
            else:
                expected = [('mov', f'byte ptr [rcx + {hex(offset)}], {int(operation == "set_true")}'), ('ret', '')]
            assert [(i.mnemonic, i.op_str) for i in decoded] == expected
            self.instruction_count += len(decoded)
            self.targets.append({**rows[0], 'operation': operation, 'field': field,
                                 'field_offset': hex(offset), 'width': width,
                                 'decoded_end': hex(decoded[-1].address + decoded[-1].size)})
        self.actor = self.arena + 0x10000
        self.field_offsets = sorted({(offset, width) for _, _, _, offset, width, _ in TARGETS})

    def snapshot(self):
        raw = bytes(self.u.mem_read(self.actor, 0x200))
        return {'fields': {hex(offset): int.from_bytes(raw[offset:offset + width], 'little')
                           for offset, width in self.field_offsets},
                'actor_sha256': hashlib.sha256(raw).hexdigest()}

    def hook(self, uc, address, size, data):
        self.executed.add(address - self.base)
        if address != self.base + 0x2B6FF0:
            return
        cx, dx = self.reg(self.x.UC_X86_REG_RCX), self.reg(self.x.UC_X86_REG_RDX)
        assert cx == self.actor + self.target[3] and self.rq(cx) == dx
        if self.event('write_barrier_service', [cx - self.actor, dx]):
            self.ret(0xABCDEF1234567890)  # Void callers must not reinterpret it.

    def run(self, name, initial, argument=0, fill=0xA5, stop_barrier=False, retained=False):
        self.target = next(t for t in TARGETS if t[0] == name)
        _, rva, operation, offset, width, _ = self.target
        if not retained:
            self.u.mem_write(self.actor, bytes([fill]) * 0x200)
            self.u.mem_write(self.actor + offset, (initial & ((1 << (width * 8)) - 1)).to_bytes(width, 'little'))
        before_raw = bytes(self.u.mem_read(self.actor, 0x200))
        before = self.snapshot()
        self.events, self.counts, self.error = [], {}, None
        self.options = {'failure': ['write_barrier_service', 1]} if stop_barrier else {}
        x = self.x
        sp = self.stack + 0x18008
        self.q(sp, self.stop)
        integers = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                    x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, register in enumerate(integers):
            self.u.reg_write(register, 0xFEDC0000 + i)
        for i, register in enumerate(vectors):
            self.u.reg_write(register, (0xA5A50000 + i) | ((0xBCBC0000 + i) << 64))
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, self.actor),
                                (x.UC_X86_REG_RDX, argument), (x.UC_X86_REG_R8, 0),
                                (x.UC_X86_REG_RAX, 0xFAFAFAFA12345678)]:
            self.u.reg_write(register, value)
        self.u.emu_start(self.base + rva, self.stop, timeout=1_000_000, count=32)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned == (not stop_barrier) and (returned or self.error == 'write_barrier_service')
        expected = bytearray(before_raw)
        if operation != 'get':
            value = argument if operation in ['set_reference', 'set_scalar'] else int(operation == 'set_true')
            expected[offset:offset + width] = (value & ((1 << (8 * width)) - 1)).to_bytes(width, 'little')
        assert bytes(self.u.mem_read(self.actor, 0x200)) == expected
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, register in enumerate(integers):
                assert self.reg(register) == 0xFEDC0000 + i
            for i, register in enumerate(vectors):
                assert self.reg(register) == (0xA5A50000 + i) | ((0xBCBC0000 + i) << 64)
        result = self.reg(x.UC_X86_REG_RAX) if operation == 'get' and returned else None
        if result is not None:
            assert result == int.from_bytes(before_raw[offset:offset + width], 'little')
        return {'method': name, 'initial': initial, 'argument': argument, 'fill': fill, 'retained': retained,
                'before': before, 'events': self.events.copy(), 'returned': returned,
                'error': self.error, 'result': result, 'final': self.snapshot(),
                'all_unrelated_actor_bytes_retained': True}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, failures, sequences = [], [], []
    pointers = [0, 1, 0x123456789ABCDEF0, 0x8000000000000001, 0xFFFFFFFFFFFFFFFF, m.actor]
    scalars = [0, 10, 20, 30, 0xFFFFFFFF, 0x80000000, 0xFEDCBA9812345678]
    for name, _, operation, _, width, _ in TARGETS:
        values = pointers if width == 8 else scalars if width == 4 else [0, 255]
        for fill in [0xA5, 0x5A]:
            for value in values:
                cases.append(m.run(name, value if operation == 'get' else 0x87654321, value, fill))
        if operation == 'set_reference':
            baseline = m.run(name, 0, pointers[2])
            stopped = m.run(name, 0, pointers[2], stop_barrier=True)
            assert stopped['events'] == baseline['events'][:1]
            assert stopped['final'] == baseline['events'][0]['snapshot']
            failures.append({'baseline': baseline, 'stopped': stopped})
    for setter, getter in [('set_onClick', 'get_onClick'), ('set_onReveal', 'get_onReveal'),
                            ('CreateRuntimeData', 'GetRuntimeData'), ('ChangeAlignment', 'GetRealAlignment')]:
        calls = []
        for i, value in enumerate(pointers if setter != 'ChangeAlignment' else scalars):
            calls.append(m.run(setter, 0, value, retained=i != 0))
            calls.append(m.run(getter, 0, retained=True))
            assert calls[-1]['result'] == (value if setter != 'ChangeAlignment' else value & 0xFFFFFFFF)
        sequences.append({'setter': setter, 'getter': getter, 'calls': calls})
    return {'build_id': BUILD, 'targets': m.targets, 'target_count': len(m.targets),
            'instruction_assertions': m.instruction_count, 'cases': cases, 'case_count': len(cases),
            'failure_cases': failures, 'failure_case_count': len(failures),
            'retained_sequences': sequences, 'retained_sequence_count': len(sequences),
            'executed_address_count': len(m.executed),
            'scope': 'Thirteen exact complete Character field leaves execute with full unrelated-byte and normal ABI retention. Pointer setters store before their sole supplied GC barrier; value getters read raw fields with decoded return width. Nonnull receivers and mapped authored storage are required; null/destroyed references are retained as raw values, without Unity lifetime tests. No callback invocation, allocation, appearance/status inference, runtime exception handling or live process access is claimed.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({k: result[k] for k in ['target_count', 'case_count', 'failure_case_count', 'retained_sequence_count']}))
