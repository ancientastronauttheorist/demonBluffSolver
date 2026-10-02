"""Execute four pinned DeckView visibility callers with supplied Unity/tween services.

All four bodies execute offline, including Update's native tail-call to Close.
Input, formatting, liveness, CanvasGroup and DOTween implementations are services.
"""
import argparse
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine


TARGETS = {
    0x39C7E0: ('Start', 0x39C86B, 'tdi5744.m0002'),
    0x39B7B0: ('OpenDeckView', 0x39B8A1, 'tdi5744.m0010'),
    0x39B1A0: ('CloseDeckView', 0x39B28C, 'tdi5744.m0011'),
    0x39D310: ('Update', 0x39D33C, 'tdi5744.m0009'),
}
FIELDS = [('tutorial', 0x20), ('villagers', 0x28), ('outsiders', 0x30),
          ('minions', 0x38), ('demons', 0x40), ('char_prefab', 0x48),
          ('reveal_clips', 0x50), ('layouts', 0x58), ('on_enable', 0x60),
          ('canvas', 0x68), ('animation', 0x70)]
SERVICE_NAMES = {
    0x1C79FD0: 'UnityEngine.Component$$get_gameObject',
    0x1C81060: 'UnityEngine.Object$$GetInstanceID',
    0xF74DF0: 'System.String$$Format',
    0x1C822C0: 'UnityEngine.Object$$op_Equality',
    0x5044D0: 'DG.Tweening.DOTween$$Kill',
    0x349E80: 'DG.Tweening.DOTweenModuleUI$$DOFade',
    0x6BC9D0: 'DG.Tweening.TweenSettingsExtensions$$SetId<object>',
    0x1EADA80: 'UnityEngine.CanvasGroup$$set_interactable',
    0x1EADA30: 'UnityEngine.CanvasGroup$$set_blocksRaycasts',
    0x1CD3C80: 'UnityEngine.Input$$GetKeyDownInt',
}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root)
        extraction = json.loads((Path(__file__).parents[1] /
            f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pinned(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pinned('script.json', 'script_json'))
        dump = pinned('dump.cs', 'dump_cs')
        declaration = re.search(r'^public class DeckView : MonoBehaviour // TypeDefIndex: 5744\s*\{(.*?)// Methods',
                                dump, re.M | re.S)
        assert declaration
        expected_fields = [
            'public TutorialNote tutorial; // 0x20', 'public Transform villagers; // 0x28',
            'public Transform outsiders; // 0x30', 'public Transform minions; // 0x38',
            'public Transform demons; // 0x40', 'public DeckCharacter charPrefab; // 0x48',
            'public AudioClip[] revealClips; // 0x50', 'public RectTransform[] layoutsToRebuild; // 0x58',
            'public Action onEnable; // 0x60', 'public CanvasGroup canvasGroup; // 0x68',
            'private string animId; // 0x70', 'public static List<CharacterData> ObscuredCharacters; // 0x0']
        assert all(field in declaration[1] for field in expected_fields)
        self.targets, self.instructions, self.ranges = [], {}, {}
        for address, (name, end, method) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == 'DeckView$$' + name]
            assert len(rows) == 1
            assert rows[0]['Signature'] == f'void DeckView__{name} (DeckView_o* __this, const MethodInfo* method);'
            self.targets.append(dict(rows[0], stable_method_id=method))
            next_entry = min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > address)
            section = self.pe.get_section_by_rva(address)
            assert section and next_entry - section.VirtualAddress <= section.SizeOfRawData
            raw = self.pe.get_data(address, next_entry - address)
            assert len(raw) == next_entry - address
            instructions = list(self.cs.disasm(raw, address))
            assert sum(i.size for i in instructions) == len(raw)
            while instructions[-1].mnemonic == 'int3':
                instructions.pop()
            assert instructions[-1].address + instructions[-1].size == end
            assert all(a.address + a.size == b.address for a, b in zip(instructions, instructions[1:]))
            self.ranges[hex(address)] = [hex(address), hex(end)]
            self.instructions.update({i.address: i for i in instructions})
        self.services = []
        for address, name in SERVICE_NAMES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1
            self.services += rows
        references, self.flags = set(), set()
        for i in self.instructions.values():
            for operand in i.operands:
                if operand.type == capstone.CS_OP_MEM and operand.mem.base == capstone.x86.X86_REG_RIP:
                    references.add(i.address + i.size + operand.mem.disp)
            if i.mnemonic == 'cmp' and i.operands[0].type == capstone.CS_OP_MEM and i.operands[0].size == 1 and i.operands[0].mem.base == capstone.x86.X86_REG_RIP:
                self.flags.add(i.address + i.size + i.operands[0].mem.disp)
        self.bindings, self.slots, self.labels = {}, {}, {0: None}
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] not in references:
                continue
            pointer = self.arena + 0x4000 + len(self.bindings) * 0x400
            self.bindings[row['Name']] = pointer
            self.slots[row['Address']] = pointer
            self.q(self.base + row['Address'], pointer)
            self.labels[pointer] = row['Name']
        self.set_id_method = 'Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<float, float, FloatOptions>>()'
        assert set(self.bindings) == {'int_TypeInfo', 'UnityEngine.Object_TypeInfo', 'DG.Tweening.DOTween_TypeInfo', self.set_id_method}
        literals = [r for r in self.metadata['ScriptString'] if r['Address'] in references]
        assert len(literals) == 1 and literals[0]['Value'] == 'deckView_{0}'
        self.literal_rva = literals[0]['Address']
        names = ['owner', 'owner_class', 'owner_game_object', 'alternate_game_object', 'boxed',
                 'canvas0', 'canvas1', 'canvas2', 'tween0', 'tween1', 'tween2',
                 'animation0', 'animation1', 'animation2', 'formatted', 'literal', 'string_class',
                 'tutorial', 'villagers', 'outsiders', 'minions', 'demons', 'char_prefab',
                 'reveal_clips', 'layouts', 'on_enable']
        self.p = {name: self.arena + 0x10000 + i * 0x1000 for i, name in enumerate(names)}
        self.labels.update({pointer: name for name, pointer in self.p.items()})
        self.q(self.base + self.literal_rva, self.p['literal'])
        self.checks = {
            0x39C836: ('mov', 'dword ptr [rsp + 0x40], eax'),
            0x39C858: ('mov', 'qword ptr [rcx], rax'),
            0x39C85B: ('call', '0x2b6ff0'),
            0x39B7F8: ('mov', 'rdi, qword ptr [rbx + 0x68]'),
            0x39B817: ('test', 'al, al'),
            0x39B822: ('mov', 'rdi, qword ptr [rbx + 0x70]'),
            0x39B837: ('mov', 'dl, 1'),
            0x39B850: ('xorps', 'xmm2, xmm2'),
            0x39B862: ('mov', 'rdx, qword ptr [rbx + 0x70]'),
            0x39B877: ('mov', 'dl, 1'),
            0x39B88A: ('mov', 'dl, 1'),
            0x39B1E8: ('mov', 'rdi, qword ptr [rbx + 0x68]'),
            0x39B212: ('mov', 'rdi, qword ptr [rbx + 0x70]'),
            0x39B227: ('mov', 'dl, 1'),
            0x39B23B: ('xorps', 'xmm1, xmm1'),
            0x39B238: ('xorps', 'xmm2, xmm2'),
            0x39B24D: ('mov', 'rdx, qword ptr [rbx + 0x70]'),
            0x39B262: ('xor', 'edx, edx'),
            0x39B275: ('xor', 'edx, edx'),
            0x39D31B: ('lea', 'ecx, [rdx + 0x1b]'),
            0x39D323: ('test', 'al, al'),
            0x39D331: ('jmp', '0x39b1a0'),
        }
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == value for a, value in self.checks.items())
        instruction = self.instructions[0x39B841]
        operand = instruction.operands[1]
        assert operand.type == capstone.CS_OP_MEM and operand.mem.base == capstone.x86.X86_REG_RIP
        literal = instruction.address + instruction.size + operand.mem.disp
        section = self.pe.get_section_by_rva(literal)
        assert section and literal - section.VirtualAddress + 4 <= section.SizeOfRawData
        value = self.pe.get_data(literal, 4)
        assert len(value) == 4 and struct.unpack('<I', value)[0] == 0x3F800000
        self.float_literal = {'instruction': '0x39b841', 'rva': hex(literal), 'bits': 0x3F800000}
        assert len(self.flags) == 3
        self.entered = []

    def oid(self, pointer):
        assert pointer in self.labels, hex(pointer)
        return self.labels[pointer]

    def put_string(self, name, text):
        pointer = self.p[name]
        raw = text.encode('utf-16-le')
        self.u.mem_write(pointer, bytes(0x100))
        self.q(pointer, self.p['string_class'])
        self.d(pointer + 0x10, len(raw) // 2)
        self.u.mem_write(pointer + 0x14, raw + b'\0\0')
        self.texts[name] = text

    def snapshot(self):
        owner = self.p['owner']
        return {
            'owner_fields': {name: self.oid(self.rq(owner + offset)) for name, offset in FIELDS},
            'owner_bytes_hex': bytes(self.u.mem_read(owner, 0x90)).hex(),
            'runtime_classes': {name: {'initialized_word': self.rd(pointer + 0xE0),
                                      'bytes_hex': bytes(self.u.mem_read(pointer, 0x100)).hex()}
                                for name, pointer in self.bindings.items() if name.endswith('_TypeInfo')},
            'metadata_flags': {hex(address): self.u.mem_read(self.base + address, 1)[0] for address in sorted(self.flags)},
            'metadata_slots': {hex(address): self.oid(self.rq(self.base + address)) for address in sorted(self.slots)},
            'literal_slot': self.oid(self.rq(self.base + self.literal_rva)),
            'canvas_services': {name: value.copy() for name, value in self.canvases.items()},
            'canvas_bytes_hex': {name: bytes(self.u.mem_read(self.p[name], 0x40)).hex() for name in self.canvases},
            'string_services': self.texts.copy(),
            'string_bytes_hex': {name: bytes(self.u.mem_read(self.p[name], 0x100)).hex() for name in self.texts},
            'boxed_int_bytes_hex': bytes(self.u.mem_read(self.p['boxed'], 0x20)).hex(),
            'supplied_game_object_bytes_hex': {name: bytes(self.u.mem_read(self.p[name], 0x40)).hex()
                                               for name in ['owner_game_object', 'alternate_game_object']},
            'supplied_tween_bytes_hex': {name: bytes(self.u.mem_read(self.p[name], 0x40)).hex()
                                        for name in ['tween0', 'tween1', 'tween2']},
            'kill_requests': [r.copy() for r in self.kill_requests],
            'fade_requests': [r.copy() for r in self.fade_requests],
            'set_id_requests': [r.copy() for r in self.set_id_requests],
            'native_entries': self.entered.copy(),
            'applied_mutations': [{**entry, 'fields': entry['fields'].copy()} for entry in self.applied_mutations],
        }

    def prepare(self, options):
        self.options = options.copy()
        self.events, self.counts, self.error, self.entered = [], {}, None, []
        self.kill_requests, self.fade_requests, self.set_id_requests = [], [], []
        self.equality_capture, self.kill_capture = None, None
        self.applied_mutations = []
        self.texts = {}
        for name, pointer in self.p.items():
            self.u.mem_write(pointer, bytes([0xA5]) * 0x100)
        self.q(self.p['owner'], self.p['owner_class'])
        for name, offset in FIELDS:
            self.q(self.p['owner'] + offset, self.p[name] if name not in ['canvas', 'animation'] else
                   (0 if options.get('null_' + name) else self.p[name + '0']))
        if options.get('alias_layout_parents'):
            for offset in [0x30, 0x38, 0x40]:
                self.q(self.p['owner'] + offset, self.p['villagers'])
        self.canvases = {name: {'interactable': bool(i & 1), 'blocks_raycasts': not bool(i & 1)}
                         for i, name in enumerate(['canvas0', 'canvas1', 'canvas2'])}
        for name in self.canvases:
            self.q(self.p[name] + 0x10, self.p['owner_game_object'])
        self.q(self.p['boxed'], self.bindings['int_TypeInfo'])
        self.d(self.p['boxed'] + 0x10, 0xCAFEBABE)
        for name, text in [('animation0', 'old ID'), ('animation1', 'callback ID'),
                           ('animation2', 'second callback ID'), ('formatted', 'old formatted value'),
                           ('literal', 'deckView_{0}')]:
            self.put_string(name, text)
        if options.get('equal_animation_texts'):
            self.put_string('animation1', 'old ID')
        for name, pointer in self.bindings.items():
            self.u.mem_write(pointer, bytes([0xA5]) * 0x100)
            if name.endswith('_TypeInfo'):
                self.d(pointer + 0xE0, 0 if options.get('class_cold') else options.get('class_word', 1))
        for address in self.flags:
            self.u.mem_write(self.base + address, bytes([0 if options.get('cold') else options.get('warm_byte', 1)]))

    def mutate(self, kind, ordinal):
        mutation = self.options.get('mutations', {}).get(kind + ':' + str(ordinal))
        if not mutation:
            return
        self.applied_mutations.append({'service': kind, 'ordinal': ordinal, 'fields': mutation.copy()})
        for name in ['canvas', 'animation']:
            if name in mutation:
                value = mutation[name]
                self.q(self.p['owner'] + dict(FIELDS)[name], 0 if value is None else self.p[value])

    def event(self, kind, args):
        self.counts[kind] = self.counts.get(kind, 0) + 1
        ordinal = self.counts[kind]
        self.events.append({'kind': kind, 'args': args, 'snapshot': self.snapshot()})
        if self.options.get('failure') == [kind, ordinal]:
            self.error = kind
            self.u.emu_stop()
            return False
        self.mutate(kind, ordinal)
        return True

    def ret(self, value=0xBADF00DDEAD0000):
        x = self.x
        for ordinal, name in enumerate(['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']):
            self.u.reg_write(getattr(x, 'UC_X86_REG_' + name), 0xABCDEF9876543200 + ordinal)
        for ordinal in range(6):
            self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(ordinal)), 0xABCDEF9876543210 + ordinal)
        super().ret(value)

    def hook(self, uc, address, size, data):
        if address == self.stop:
            return
        rva, x = address - self.base, self.x
        self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + name)) for name in ['RCX', 'RDX', 'R8', 'R9']]
        if rva in TARGETS:
            assert cx == self.p['owner']
            self.entered.append(TARGETS[rva][0])
        if rva in self.instructions:
            return
        if rva == 0x2B7D90:
            self.event('native_null_guard', [])
            self.error = 'native_null_guard'
            uc.emu_stop()
        elif rva == 0x2B7B40:
            assert cx - self.base in self.slots or cx - self.base == self.literal_rva
            if self.event('metadata', [hex(cx - self.base)]):
                self.ret(self.rq(cx))
        elif rva == 0x281D90:
            assert cx in [self.bindings[n] for n in ['UnityEngine.Object_TypeInfo', 'DG.Tweening.DOTween_TypeInfo']]
            if cx == self.bindings['UnityEngine.Object_TypeInfo']:
                self.equality_capture = self.rq(self.p['owner'] + 0x68)
            else:
                self.kill_capture = self.rq(self.p['owner'] + 0x70)
            if self.event('class_initialization', [self.oid(cx)]):
                self.d(cx + 0xE0, 1)
                self.ret()
        elif rva == 0x1C79FD0:
            assert cx == self.p['owner'] and dx == 0
            if self.event('game_object', [self.oid(cx)]):
                self.ret(0 if self.options.get('null_game_object') else self.p['owner_game_object'])
        elif rva == 0x1C81060:
            assert cx == self.p['owner_game_object'] and dx == 0
            if self.event('instance_id', [self.oid(cx)]):
                self.ret(0xFACE000000000000 | (self.options.get('instance_id', -7) & 0xFFFFFFFF))
        elif rva == 0x282580:
            assert cx == self.bindings['int_TypeInfo']
            bits = self.rd(dx)
            if self.event('box_int32', [bits]):
                self.d(self.p['boxed'] + 0x10, bits)
                self.ret(self.p['boxed'])
        elif rva == 0xF74DF0:
            assert cx == self.p['literal'] and dx == self.p['boxed'] and r8 == 0
            bits = self.rd(dx + 0x10)
            if self.event('string_format', [self.oid(cx), self.oid(dx), bits]):
                signed = bits if bits < 0x80000000 else bits - 0x100000000
                self.put_string('formatted', 'deckView_' + str(signed))
                self.ret(0 if self.options.get('null_formatted') else self.p['formatted'])
        elif rva == 0x2B6FF0:
            assert cx == self.p['owner'] + 0x70 and self.rq(cx) == dx
            if self.event('reference_barrier', [self.oid(dx)]):
                self.ret()
        elif rva == 0x1C822C0:
            assert self.oid(cx) in [None, 'canvas0', 'canvas1', 'canvas2'] and dx == 0 and r8 == 0
            assert cx == (self.rq(self.p['owner'] + 0x68) if self.equality_capture is None else self.equality_capture)
            self.equality_capture = None
            result = self.options.get('equality_return_bits', 0xFACE000000000000 | int(cx == 0))
            if self.event('unity_equality', [self.oid(cx), None, result]):
                self.ret(result)
        elif rva == 0x5044D0:
            assert self.oid(cx) in [None, 'animation0', 'animation1', 'animation2', 'formatted'] and dx & 0xFF == 1 and r8 == 0
            assert dx == 0xABCDEF9876543201
            assert cx == (self.rq(self.p['owner'] + 0x70) if self.kill_capture is None else self.kill_capture)
            self.kill_capture = None
            request = {'animation': self.oid(cx), 'complete_byte': dx & 0xFF,
                       'complete_register_bits': dx}
            if self.event('kill', request):
                self.kill_requests.append(request)
                self.ret(self.options.get('kill_return_bits', 0xFACE0000FFFFFFFF))
        elif rva == 0x349E80:
            assert self.oid(cx) in [None, 'canvas0', 'canvas1', 'canvas2'] and r9 == 0
            assert cx == self.rq(self.p['owner'] + 0x68)
            alpha = self.reg(x.UC_X86_REG_XMM1) & 0xFFFFFFFF
            duration = self.reg(x.UC_X86_REG_XMM2) & 0xFFFFFFFF
            assert duration == 0 and alpha == (0x3F800000 if self.entered[-1] == 'OpenDeckView' else 0)
            request = {'canvas': self.oid(cx), 'alpha_bits': alpha, 'duration_bits': duration}
            if self.event('fade', request):
                index = min(len(self.fade_requests), 2)
                self.fade_requests.append(request)
                self.ret(0 if self.options.get('null_tween') else self.p['tween' + str(index)])
        elif rva == 0x6BC9D0:
            assert self.oid(cx) in [None, 'tween0', 'tween1', 'tween2'] and r8 == self.bindings[self.set_id_method]
            assert dx == self.rq(self.p['owner'] + 0x70)
            request = {'tween': self.oid(cx), 'animation': self.oid(dx), 'method': self.oid(r8)}
            if self.event('set_id', request):
                self.set_id_requests.append(request)
                self.ret(cx)
        elif rva in [0x1EADA80, 0x1EADA30]:
            assert self.oid(cx) in self.canvases and r8 == 0
            assert cx == self.rq(self.p['owner'] + 0x68)
            name = 'interactable' if rva == 0x1EADA80 else 'blocks_raycasts'
            value = dx & 0xFF
            assert value == int(self.entered[-1] == 'OpenDeckView')
            assert dx == (0xABCDEF9876543201 if self.entered[-1] == 'OpenDeckView' else 0)
            if self.event(name, {'canvas': self.oid(cx), 'enabled_byte': value,
                                 'enabled_register_bits': dx}):
                self.canvases[self.oid(cx)][name] = bool(value)
                self.ret()
        elif rva == 0x1CD3C80:
            assert cx == 27 and dx == 0
            result = self.options.get('key_return_bits', 0xFACE000000000000)
            if self.event('key_down', [27, result]):
                self.ret(result)
        else:
            raise AssertionError(f'unexpected execution {rva:x}')

    def run(self, name, options=None, retained=False):
        if not retained:
            self.prepare(options or {})
        elif options is not None:
            self.options.update(options)
        start = next(address for address, (method, _, _) in TARGETS.items() if method == name)
        initial, old_count = self.snapshot(), len(self.events)
        self.error = None
        x, sp = self.x, self.stack + 0x18008
        self.q(sp, self.stop)
        integers = ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']
        for ordinal, register in enumerate(integers):
            self.u.reg_write(getattr(x, 'UC_X86_REG_' + register), 0xFAB0000000000000 + ordinal)
        for index in range(6, 16):
            self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(index)), (0x123456789ABCDEF0 << 64) | index)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        self.u.reg_write(x.UC_X86_REG_RCX, self.p['owner'])
        self.u.reg_write(x.UC_X86_REG_RDX, 0xFACE000000000000)
        self.u.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        self.u.emu_start(self.base + start, self.stop, timeout=10_000_000, count=100000)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for ordinal, register in enumerate(integers):
                assert self.reg(getattr(x, 'UC_X86_REG_' + register)) == 0xFAB0000000000000 + ordinal
            for index in range(6, 16):
                assert self.reg(getattr(x, 'UC_X86_REG_XMM' + str(index))) == (0x123456789ABCDEF0 << 64) | index
        final = self.snapshot()
        old_bytes, new_bytes = bytes.fromhex(initial['owner_bytes_hex']), bytes.fromhex(final['owner_bytes_hex'])
        allowed = set(range(0x70, 0x78)) if name == 'Start' else set()
        for mutation in final['applied_mutations'][len(initial['applied_mutations']):]:
            for field in mutation['fields']:
                offset = dict(FIELDS)[field]
                allowed |= set(range(offset, offset + 8))
        assert all(a == b for offset, (a, b) in enumerate(zip(old_bytes, new_bytes)) if offset not in allowed)
        completed = self.events[old_count:] if returned else self.events[old_count:-1]
        verify_storage_retention(initial, final, completed)
        result = {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                  'initial': initial, 'events': self.events[old_count:].copy(), 'final': final,
                  'other_owner_bytes_retained': True, 'win64_nonvolatile_verified': returned}
        verify_normal_expectations(result)
        return result


def verify_storage_retention(initial, final, completed):
    """Retain unconsumed diagnostic bytes around each explicitly supplied effect."""
    for key in ['metadata_slots', 'literal_slot', 'canvas_bytes_hex',
                'supplied_game_object_bytes_hex', 'supplied_tween_bytes_hex']:
        assert initial[key] == final[key], key
    for name in initial['runtime_classes']:
        before = bytes.fromhex(initial['runtime_classes'][name]['bytes_hex'])
        after = bytes.fromhex(final['runtime_classes'][name]['bytes_hex'])
        assert before[:0xE0] == after[:0xE0] and before[0xE4:] == after[0xE4:]
        if initial['runtime_classes'][name]['initialized_word'] != final['runtime_classes'][name]['initialized_word']:
            assert name in ['UnityEngine.Object_TypeInfo', 'DG.Tweening.DOTween_TypeInfo']
            assert any(event['kind'] == 'class_initialization' and event['args'] == [name] for event in completed)
            assert initial['runtime_classes'][name]['initialized_word'] == 0
            assert final['runtime_classes'][name]['initialized_word'] == 1
    for name, raw in initial['string_bytes_hex'].items():
        if name != 'formatted' or not any(event['kind'] == 'string_format' for event in completed):
            assert final['string_bytes_hex'][name] == raw
            assert final['string_services'][name] == initial['string_services'][name]
    before = bytes.fromhex(initial['boxed_int_bytes_hex'])
    after = bytes.fromhex(final['boxed_int_bytes_hex'])
    assert before[:0x10] == after[:0x10] and before[0x14:] == after[0x14:]
    boxes = [event for event in completed if event['kind'] == 'box_int32']
    assert struct.unpack('<I', after[0x10:0x14])[0] == (boxes[-1]['args'][0] if boxes else struct.unpack('<I', before[0x10:0x14])[0])
    for address, byte in initial['metadata_flags'].items():
        assert final['metadata_flags'][address] == byte or (byte == 0 and final['metadata_flags'][address] == 1)
    expected_canvas = {name: fields.copy() for name, fields in initial['canvas_services'].items()}
    for event in completed:
        if event['kind'] in ['interactable', 'blocks_raycasts']:
            expected_canvas[event['args']['canvas']][event['kind']] = bool(event['args']['enabled_byte'])
    assert final['canvas_services'] == expected_canvas
    for key in ['kill_requests', 'fade_requests', 'set_id_requests']:
        kind = {'kill_requests': 'kill', 'fade_requests': 'fade', 'set_id_requests': 'set_id'}[key]
        assert final[key] == initial[key] + [event['args'] for event in completed if event['kind'] == kind]


def verify_normal_expectations(result):
    """Independent complete-path ordering and observable UI expectations.

    Mutations and injected stops retain their exact native event snapshots instead.
    This check also catches a silently replaced native tail-call or missing effect.
    """
    options, method = result['options'], result['method']
    if options.get('mutations') or options.get('failure'):
        return
    events = result['events']
    kinds = [event['kind'] for event in events if event['kind'] not in ['metadata', 'class_initialization']]
    if method == 'Start':
        if options.get('null_game_object'):
            assert kinds == ['game_object', 'native_null_guard'] and not result['returned']
            assert result['final']['owner_fields']['animation'] == result['initial']['owner_fields']['animation']
        else:
            assert kinds == ['game_object', 'instance_id', 'box_int32', 'string_format', 'reference_barrier']
            assert result['returned']
            expected_id = options.get('instance_id', -7)
            assert result['final']['string_services']['formatted'] == 'deckView_' + str(expected_id)
            assert result['final']['owner_fields']['animation'] == (None if options.get('null_formatted') else 'formatted')
        return
    prefix = ['key_down'] if method == 'Update' else []
    if method == 'Update' and not (options.get('key_return_bits', 0xFACE000000000000) & 0xFF):
        assert kinds == prefix and result['returned']
        assert result['final']['canvas_services'] == result['initial']['canvas_services']
        assert result['final']['native_entries'][-1] == 'Update'
        return
    if method == 'Update':
        assert result['final']['native_entries'][-2:] == ['Update', 'CloseDeckView']
    equality = options.get('equality_return_bits', int(result['initial']['owner_fields']['canvas'] is None)) & 0xFF
    if equality:
        assert kinds == prefix + ['unity_equality'] and result['returned']
        assert result['final']['canvas_services'] == result['initial']['canvas_services']
    elif result['initial']['owner_fields']['canvas'] is None:
        assert kinds == prefix + ['unity_equality', 'kill', 'fade', 'set_id', 'native_null_guard']
        assert not result['returned']
    else:
        assert kinds == prefix + ['unity_equality', 'kill', 'fade', 'set_id', 'interactable', 'blocks_raycasts']
        assert result['returned']
        canvas = result['initial']['owner_fields']['canvas']
        value = method == 'OpenDeckView'
        assert result['final']['canvas_services'][canvas] == {'interactable': value, 'blocks_raycasts': value}


def audit(game_root, dumper_root):
    machine = Machine(game_root, dumper_root)
    cases, sequences, baselines, failures = [], [], [], []
    for name, cold, class_cold in itertools.product([target[0] for target in TARGETS.values()], [False, True], [False, True]):
        cases.append(machine.run(name, {'cold': cold, 'class_cold': class_cold}))
    for instance_id, cold in itertools.product([-2147483648, -7, 0, 7, 2147483647], [False, True]):
        cases.append(machine.run('Start', {'instance_id': instance_id, 'cold': cold}))
    for name, bits, canvas_null, id_null, tween_null in itertools.product(
            ['OpenDeckView', 'CloseDeckView'], [0xFFFFFFFFFFFFFF00, 0xFACE000000000001, 0xFACE000000000080, 0xFACE0000000000FF],
            [False, True], [False, True], [False, True]):
        cases.append(machine.run(name, {'equality_return_bits': bits, 'null_canvas': canvas_null,
                                       'null_animation': id_null, 'null_tween': tween_null}))
    for bits in [0xFFFFFFFFFFFFFF00, 0xFACE000000000001, 0xFACE000000000080, 0xFACE0000000000FF]:
        cases.append(machine.run('Update', {'key_return_bits': bits, 'cold': True, 'class_cold': True}))
    for options in [{'null_game_object': True}, {'null_formatted': True},
                    {'warm_byte': 0x80, 'class_word': 0xDEADBEEF}]:
        cases.append(machine.run('Start', options))
    mutation_cases = [
        ('class_initialization:1', {'canvas': 'canvas1'}),
        ('unity_equality:1', {'canvas': 'canvas1', 'animation': 'animation1'}),
        ('class_initialization:2', {'animation': 'animation2'}),
        ('kill:1', {'canvas': 'canvas2', 'animation': 'animation2'}),
        ('fade:1', {'animation': 'animation1'}),
        ('fade:1', {'canvas': None}),
        ('set_id:1', {'canvas': 'canvas1'}),
        ('interactable:1', {'canvas': 'canvas2'}),
        ('interactable:1', {'canvas': None}),
        ('unity_equality:1', {'canvas': None}),
        ('set_id:1', {'animation': None}),
    ]
    for name, (phase, mutation) in itertools.product(['OpenDeckView', 'CloseDeckView'], mutation_cases):
        cases.append(machine.run(name, {'cold': True, 'class_cold': True, 'mutations': {phase: mutation}}))
    cases.append(machine.run('Update', {'key_return_bits': 0xFACE000000000080,
        'mutations': {'key_down:1': {'canvas': 'canvas1', 'animation': 'animation1'}}}))
    for phase in ['string_format:1', 'reference_barrier:1']:
        cases.append(machine.run('Start', {'mutations': {phase: {'animation': 'animation2'}}}))
    for null_canvas, class_word in itertools.product([False, True], [2, 0xDEADBEEF]):
        for name in ['OpenDeckView', 'CloseDeckView']:
            cases.append(machine.run(name, {'null_canvas': null_canvas, 'warm_byte': 0xFF, 'class_word': class_word}))
    for name in [target[0] for target in TARGETS.values()]:
        cases.append(machine.run(name, {'alias_layout_parents': True}))
    for name in ['OpenDeckView', 'CloseDeckView']:
        cases.append(machine.run(name, {'equal_animation_texts': True,
            'mutations': {'fade:1': {'animation': 'animation1'}}}))
        for kill_result in [0, 1, 0xFFFFFFFFFFFFFFFF]:
            cases.append(machine.run(name, {'kill_return_bits': kill_result}))
    for alias in [False, True]:
        machine.prepare({'cold': True, 'class_cold': True})
        if alias:
            machine.q(machine.p['owner'] + 0x68, machine.p['canvas1'])
        calls = [machine.run('Start', retained=True), machine.run('OpenDeckView', retained=True),
                 machine.run('Update', {'key_return_bits': 0xFACE000000000000}, retained=True),
                 machine.run('Update', {'key_return_bits': 0xFACE000000000080}, retained=True),
                 machine.run('CloseDeckView', retained=True), machine.run('OpenDeckView', retained=True)]
        assert all(row['returned'] for row in calls)
        sequences.append({'alternate_canvas': alias, 'calls': calls})
    # Cold complete paths and four capture/reload profiles retain every stop.
    baseline_specs = [(target[0], {}) for target in TARGETS.values()] + [
        ('Start', {'mutations': {'string_format:1': {'animation': 'animation2'}}}),
        ('OpenDeckView', {'mutations': {'class_initialization:2': {'animation': 'animation2'}}}),
        ('CloseDeckView', {'mutations': {'fade:1': {'animation': 'animation1'}}}),
        ('OpenDeckView', {'mutations': {'interactable:1': {'canvas': 'canvas2'}}}),
    ]
    for name, supplied in baseline_specs:
        options = dict({'cold': True, 'class_cold': True, 'key_return_bits': 0xFACE000000000080}, **supplied)
        baseline = machine.run(name, options)
        assert baseline['returned']
        ordinal = len(baselines)
        baselines.append(baseline)
        counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']
            counts[kind] = counts.get(kind, 0) + 1
            stopped = machine.run(name, dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned']
            assert stopped['events'] == baseline['events'][:index + 1]
            assert stopped['final'] == event['snapshot']
            failures.append({'baseline': ordinal, 'failure': [kind, counts[kind]], 'prefix_length': index + 1,
                             'stopped': stopped, 'exact_snapshot_verified': True})
    missing = set(machine.instructions) - machine.executed
    assert not missing, [hex(address) for address in sorted(missing)]
    return {'build': BUILD, 'targets': machine.targets, 'ranges': machine.ranges,
            'instruction_assertions': len(machine.checks), 'float_literal': machine.float_literal,
            'string_literal': {'rva': hex(machine.literal_rva), 'value': 'deckView_{0}'},
            'metadata_bindings': sorted(machine.bindings), 'supplied_services': machine.services,
            'case_count': len(cases), 'cases': cases, 'retained_sequences': sequences,
            'failure_baselines': baselines, 'failure_case_count': len(failures), 'failure_cases': failures,
            'body_instructions_decoded': len(machine.instructions),
            'body_instructions_executed': len(machine.executed & machine.instructions.keys()),
            'native_execution_addresses': len(machine.executed),
            'scope': 'Four complete native DeckView callers; Update executes its native CloseDeckView tail-call. Unity liveness/input/component identity/CanvasGroup, DOTween tween creation/identity, String.Format/boxing, runtime metadata/class initialization and GC barriers are explicit services. No renderer, scheduler, engine input collection, concrete tween effects, obscured-list lifecycle, other DeckView bodies or actual managed exception unwinding are inferred. Owner bytes include a diagnostic 0x90 window, not an object-size/type proof.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('dumper_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(json.dumps({key: report[key] for key in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
