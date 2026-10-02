"""Complete native MouseExit with supplied callback, Unity and DOTween services."""
import argparse
from copy import deepcopy
import itertools
import json
from pathlib import Path
import re

from audit_character_assets import BUILD
from audit_card_interaction_audio import Machine as AudioMachine, POISON, XPOISON, INCOMING, XINCOMING
from audit_character_oracle_reveal_join import expand_memory, pool_memory
from audit_report_snapshots import expand_snapshots, pool_snapshots

START, END, NEXT = 0x35F810, 0x35FA2A, 0x35FA30
FLAGS = [0x288C13B, 0x288C102]
DURATION = 0x3E4CCCCD
SHADOW_Y = 0xC1780000
NONVOL_XMM6 = (0xABCDEF9876543210 << 64) | 6
SERVICES = {0x5044D0: 'DG.Tweening.DOTween$$Kill', 0x1C7D810: 'UnityEngine.GameObject$$SetActive',
            0x50FB60: 'DG.Tweening.ShortcutExtensions$$DOLocalMoveY', 0x513BF0: 'DG.Tweening.ShortcutExtensions$$DOScale',
            0x6BC9D0: 'DG.Tweening.TweenSettingsExtensions$$SetId<object>'}


class Machine(AudioMachine):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        import capstone
        dump = (Path(dumper_root) / 'dump.cs').read_text(encoding='utf-8-sig')
        body = re.search(r'^public class CardInteraction : MonoBehaviour // TypeDefIndex: 5468\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert body
        for field in ['public GameObject[] highlights; // 0x28', 'public Transform card; // 0x30', 'public Transform shadow; // 0x38', 'private string animationId; // 0x40', 'public Action onHoverExit; // 0x68', 'private bool blockHover; // 0x70']:
            assert field in body[1]
        vector = re.search(r'^public struct Vector3 :[^\n]* // TypeDefIndex: 6699\s*\{(.*?)\n\}', dump, re.M | re.S)
        assert vector and 'private static readonly Vector3 oneVector; // 0xC' in vector[1]
        for field in ['public float x; // 0x0', 'public float y; // 0x4', 'public float z; // 0x8']: assert field in vector[1]
        self.instructions, self.body_addresses, self.ranges = {}, set(), {}
        self.decode('MouseExit', START, END, NEXT)
        rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == START]
        assert len(rows) == 1 and rows[0]['Name'] == 'CardInteraction$$MouseExit'
        assert rows[0]['Signature'] == 'void CardInteraction__MouseExit (CardInteraction_o* __this, const MethodInfo* method);' and rows[0]['TypeSignature'] == 'vii'
        self.targets = [dict(rows[0], stable_method_id='tdi5468.m0006')]
        assert len(self.ranges['MouseExit']['unwind_chunks']) == 4
        entry = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == START); assert entry.unwindinfo.Flags == 0
        self.checks = {0x35F841: ('cmp', 'byte ptr [rsi + 0x70], 0'), 0x35F845: ('jne', '0x35fa18'),
            0x35F855: ('mov', 'rax, qword ptr [rsi + 0x68]'), 0x35F85E: ('mov', 'rdx, qword ptr [rax + 0x28]'),
            0x35F862: ('mov', 'rcx, qword ptr [rax + 0x40]'), 0x35F866: ('call', 'qword ptr [rax + 0x18]'),
            0x35F870: ('mov', 'rbx, qword ptr [rsi + 0x40]'), 0x35F874: ('cmp', 'dword ptr [rcx + 0xe0], 0'),
            0x35F882: ('xor', 'r8d, r8d'), 0x35F885: ('mov', 'dl, 1'), 0x35F88F: ('mov', 'rdi, qword ptr [rsi + 0x28]'),
            0x35F8A0: ('cmp', 'eax, dword ptr [rdi + 0x18]'), 0x35F8A3: ('jge', '0x35f8cf'),
            0x35F8A5: ('cmp', 'ebx, dword ptr [rdi + 0x18]'), 0x35F8A8: ('jae', '0x35fa24'),
            0x35F8AE: ('movsxd', 'rax, ebx'), 0x35F8B1: ('mov', 'rcx, qword ptr [rdi + rax*8 + 0x20]'),
            0x35F8BF: ('xor', 'r8d, r8d'), 0x35F8C2: ('xor', 'edx, edx'), 0x35F8C9: ('inc', 'ebx'), 0x35F8CB: ('mov', 'eax, ebx'),
            0x35F8CF: ('mov', 'rcx, qword ptr [rsi + 0x30]'), 0x35F8E6: ('movaps', 'xmm2, xmm6'),
            0x35F8E9: ('mov', 'qword ptr [rsp + 0x20], 0'), 0x35F901: ('mov', 'rdx, qword ptr [rsi + 0x40]'),
            0x35F915: ('mov', 'rcx, qword ptr [rsi + 0x38]'), 0x35F919: ('movaps', 'xmm2, xmm6'),
            0x35F944: ('mov', 'rbx, qword ptr [rsi + 0x30]'), 0x35F96D: ('mov', 'rdx, qword ptr [rax + 0xb8]'),
            0x35F974: ('movsd', 'xmm0, qword ptr [rdx + 0xc]'), 0x35F979: ('mov', 'eax, dword ptr [rdx + 0x14]'),
            0x35F97C: ('lea', 'rdx, [rsp + 0x30]'), 0x35F981: ('movsd', 'qword ptr [rsp + 0x30], xmm0'),
            0x35F987: ('mov', 'dword ptr [rsp + 0x38], eax'), 0x35F9AA: ('mov', 'rbx, qword ptr [rsi + 0x38]'),
            0x35F9D3: ('mov', 'rdx, qword ptr [rax + 0xb8]'), 0x35F9DA: ('movsd', 'xmm0, qword ptr [rdx + 0xc]'),
            0x35FA09: ('movaps', 'xmm6, xmmword ptr [rsp + 0x40]'),
            0x35FA1E: ('call', '0x2b7d90'), 0x35FA24: ('call', '0x2b7d80')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == value for a, value in self.checks.items())
        refs, flag_refs = set(), set()
        for i in self.instructions.values():
            for o in i.operands:
                if o.type == capstone.CS_OP_MEM and o.mem.base == capstone.x86.X86_REG_RIP:
                    a = i.address + i.size + o.mem.disp; refs.add(a)
                    if i.mnemonic == 'cmp' and o.size == 1: flag_refs.add(a)
        assert flag_refs == set(FLAGS)
        names = ['owner', 'owner_class', 'tween_class', 'other_tween_class', 'vector_class', 'other_vector_class', 'vector_statics', 'other_vector_statics',
                 'set_id_method', 'other_method', 'array0', 'array1', 'array_class', 'object0', 'object1', 'object2', 'object_class',
                 'card0', 'card1', 'shadow0', 'shadow1', 'transform_class', 'animation0', 'animation1', 'string_class',
                 'action0', 'action1', 'action_class', 'code0', 'code1', 'method0', 'method1', 'tween0', 'tween1', 'tween2', 'tween3', 'tweener_class']
        self.p = {n: self.arena + 0x270000 + i * 0x1000 for i, n in enumerate(names)}
        self.callbacks = [self.stop + 0x100, self.stop + 0x200]
        self.labels = {0: None, **{p: n for n, p in self.p.items()}, **{v: 'callback' + str(i) for i, v in enumerate(self.callbacks)}}
        self.sizes = {n: 256 if n.endswith('_class') else 128 for n in names}
        keys = {'DG.Tweening.DOTween_TypeInfo': 'tween_class', 'UnityEngine.Vector3_TypeInfo': 'vector_class',
                'Method$DG.Tweening.TweenSettingsExtensions.SetId<TweenerCore<Vector3, Vector3, VectorOptions>>()': 'set_id_method'}
        self.slots, self.slot_names = {}, {}
        for r in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if r['Address'] in refs:
                assert r['Name'] in keys
                if r['Name'].startswith('Method$'): assert r['MethodAddress'] == 0
                self.slots[r['Address']] = self.p[keys[r['Name']]]; self.slot_names[keys[r['Name']]] = r['Address']
        assert len(self.slots) == 3
        self.metadata_order = []
        ordered = [i for _, i in sorted(self.instructions.items())]
        for previous, i in zip(ordered, ordered[1:]):
            if i.mnemonic == 'call' and i.op_str == '0x2b7b40':
                assert previous.mnemonic == 'lea'; slot = previous.address + previous.size + previous.operands[1].mem.disp
                assert slot in self.slots; self.metadata_order.append((slot, i.address + i.size))
        assert len(self.metadata_order) == 4
        self.float_pins = []
        for pc, slot, bits in [(0x35F8DE, 0x1F34B10, DURATION), (0x35F90A, 0x1F34CA8, SHADOW_Y)]:
            i = self.instructions[pc]; assert i.mnemonic == 'movss' and i.address + i.size + i.operands[1].mem.disp == slot
            section = self.pe.get_section_by_rva(slot); assert section and slot + 4 - section.VirtualAddress <= section.SizeOfRawData
            raw = self.pe.get_data(slot, 4); assert len(raw) == 4 and int.from_bytes(raw, 'little') == bits
            self.float_pins.append({'instruction_rva': hex(pc), 'slot_rva': hex(slot), 'bits': bits})
        self.service_targets = []
        signatures = ['int32_t DG_Tweening_DOTween__Kill (Il2CppObject* targetOrId, bool complete, const MethodInfo* method);',
            'void UnityEngine_GameObject__SetActive (UnityEngine_GameObject_o* __this, bool value, const MethodInfo* method);',
            'DG_Tweening_Core_TweenerCore_Vector3__Vector3__VectorOptions__o* DG_Tweening_ShortcutExtensions__DOLocalMoveY (UnityEngine_Transform_o* target, float endValue, float duration, bool snapping, const MethodInfo* method);',
            'DG_Tweening_Core_TweenerCore_Vector3__Vector3__VectorOptions__o* DG_Tweening_ShortcutExtensions__DOScale (UnityEngine_Transform_o* target, UnityEngine_Vector3_o endValue, float duration, const MethodInfo* method);',
            'Il2CppObject* DG_Tweening_TweenSettingsExtensions__SetId_object_ (Il2CppObject* t, System_String_o* stringId, const MethodInfo_6BC9D0* method);']
        for (address, name), signature in zip(SERVICES.items(), signatures):
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['Signature'] == signature; self.service_targets += rows
        self.frame = self.entry_sp - 0x58

    def snapshot(self):
        return {'metadata_flags': {hex(a): self.u.mem_read(self.base + a, 1)[0] for a in FLAGS},
                'metadata_slots': {hex(a): self.oid(self.rq(self.base + a)) for a in self.slots},
                'vector_argument_hex': bytes(self.u.mem_read(self.frame + 0x30, 16)).hex(),
                'fifth_argument_bits': self.rq(self.frame + 0x20), 'saved_xmm6_bits': int.from_bytes(self.u.mem_read(self.frame + 0x40, 16), 'little'),
                'native_entries': deepcopy(self.entries), 'service_history': deepcopy(self.history),
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()}}

    def abi(self):
        result = super().abi(); result['raw_xmm'].append(self.reg(self.x.UC_X86_REG_XMM6)); return result

    def prepare(self, options):
        self.options = deepcopy(options); self.entries, self.history, self.events, self.counts = [], [], [], {}
        for n, p in self.p.items(): self.u.mem_write(p, b'\xA5' * self.sizes[n])
        for n in ['owner', 'object0', 'object1', 'object2', 'card0', 'card1', 'shadow0', 'shadow1', 'tween0', 'tween1', 'tween2', 'tween3', 'code0', 'code1', 'method0', 'method1', 'set_id_method', 'other_method']:
            self.q(self.p[n], self.p['owner_class']); self.q(self.p[n] + 8, 0)
        for n, fields in [('owner', [(0x28, 'array0'), (0x30, 'card0'), (0x38, 'shadow0'), (0x40, 'animation0'), (0x68, 'action0')]),
                          ('vector_class', [(0xB8, 'vector_statics')]), ('other_vector_class', [(0xB8, 'other_vector_statics')])]:
            for off, value in fields: self.q(self.p[n] + off, self.p[value])
        self.u.mem_write(self.p['owner'] + 0x70, bytes([options.get('block_byte', 0)]))
        if options.get('null_action'): self.q(self.p['owner'] + 0x68, 0)
        if options.get('null_array'): self.q(self.p['owner'] + 0x28, 0)
        if options.get('null_animation'): self.q(self.p['owner'] + 0x40, 0)
        if options.get('null_card'): self.q(self.p['owner'] + 0x30, 0)
        if options.get('null_shadow'): self.q(self.p['owner'] + 0x38, 0)
        for i, n in enumerate(['action0', 'action1']):
            self.q(self.p[n], self.p['action_class']); self.q(self.p[n] + 8, 0)
            for off, value in [(0x18, self.callbacks[i]), (0x20, self.p['code' + str(1 - i)]), (0x28, self.p['method' + str(i)]), (0x40, self.p['code' + str(i)])]: self.q(self.p[n] + off, value)
        for n, length in [('array0', options.get('length_qword', 2)), ('array1', 3)]:
            self.q(self.p[n], self.p['array_class']); self.q(self.p[n] + 8, 0); self.q(self.p[n] + 0x10, 0); self.q(self.p[n] + 0x18, length)
            for i in range(3): self.q(self.p[n] + 0x20 + i * 8, 0 if options.get('null_element') == i else self.p['object' + str(i)])
        for n, bits in [('vector_statics', options.get('vector_bits', [0x3F800000] * 3)), ('other_vector_statics', [0x7FC01234, 0x80000000, 0x7F800000])]:
            self.u.mem_write(self.p[n] + 0xC, b''.join(v.to_bytes(4, 'little') for v in bits))
        for n in ['tween_class', 'other_tween_class']: self.d(self.p[n] + 0xE0, options.get('initialized_word', 1))
        for i, a in enumerate(FLAGS): self.u.mem_write(self.base + a, bytes([0 if options.get('cold') else options.get('warm_flag', 1)]))
        for a, value in self.slots.items(): self.q(self.base + a, value)
        for n in ['animation0', 'animation1']:
            text = 'authored ' + n; self.u.mem_write(self.p[n], bytes(self.sizes[n])); self.q(self.p[n], self.p['string_class']); self.d(self.p[n] + 0x10, len(text)); self.u.mem_write(self.p[n] + 0x14, text.encode('utf-16-le') + b'\0\0')
        self.u.mem_write(self.frame + 0x30, b'\xA7' * 16); self.u.mem_write(self.frame + 0x40, b'\xB8' * 16); self.q(self.frame + 0x20, 0xAABBCCDD11223344)

    def mutate(self, kind, relative):
        for name, off, width, value in self.options.get('mutations', {}).get(kind + ':' + str(relative), []):
            if name == 'flag': self.u.mem_write(self.base + FLAGS[off], bytes([value]))
            elif name == 'slot': self.q(self.base + self.slot_names[off], self.labels_to_pointer(value))
            else:
                bits = self.labels_to_pointer(value) if isinstance(value, str) or value is None else value
                self.u.mem_write(self.p[name] + off, bits.to_bytes(width, 'little')); self.allow(name, off, width)

    def hook(self, uc, address, size, data):
        if address == self.stop: return
        rva = address - self.base; self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(self.x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == START: self.entries.append({'method': 'MouseExit', **self.abi()})
        if rva in self.instructions: return
        if rva == 0x2B7B40:
            assert cx - self.base in self.slots
            if self.event('metadata', [hex(cx - self.base), self.oid(self.rq(cx))]): self.history.append(['metadata', hex(cx - self.base), self.oid(self.rq(cx))]); self.ret(self.rq(cx))
        elif address in self.callbacks:
            action = self.reg(self.x.UC_X86_REG_RAX)
            assert cx == self.rq(action + 0x40) and dx == self.rq(action + 0x28)
            args = [self.oid(action), self.oid(address), self.oid(cx), self.oid(dx)]
            if self.event('hover_exit', args): self.history.append(['hover_exit', *args]); self.ret()
        elif rva == 0x281D90:
            assert self.oid(cx) in ['tween_class', 'other_tween_class'] and self.rd(cx + 0xE0) == 0
            name = self.oid(cx)
            if self.event('class_init', [name]): self.d(cx + 0xE0, 1); self.allow(name, 0xE0, 4); self.history.append(['class_init', name]); self.ret()
        elif rva == 0x5044D0:
            assert dx & 0xFF == 1 and r8 == 0
            result = self.options.get('kill_result_bits', 0x12345678FFFFFFFF)
            args = [self.oid(cx), dx, r8, result]
            if self.event('kill', args): self.history.append(['kill', *args]); self.ret(result)
        elif rva == 0x1C7D810:
            assert self.oid(cx) in ['object0', 'object1', 'object2'] and dx == 0 and r8 == 0
            args = [self.oid(cx), dx, r8]
            if self.event('set_active', args): self.history.append(['set_active', *args]); self.ret()
        elif rva in [0x50FB60, 0x513BF0]:
            kind = 'move_y' if rva == 0x50FB60 else 'scale'
            relative = self.counts.get(kind, 0) - self.entry_counts.get(kind, 0)
            result = self.options.get('tween_results', ['tween0', 'tween1', 'tween2', 'tween3'])[(0 if kind == 'move_y' else 2) + relative]
            xmm = self.abi()['raw_xmm']; assert xmm[2] == DURATION and r9 == 0 and self.rq(self.frame + 0x20) == 0
            args = [self.oid(cx), self.oid(self.labels_to_pointer(result))]
            if kind == 'move_y': assert xmm[1] == (0 if relative == 0 else SHADOW_Y); args += [xmm[1], xmm[2], r9, 0]
            else:
                assert dx == self.frame + 0x30; args += [bytes(self.u.mem_read(dx, 12)).hex(), xmm[2], r9]
            if self.event(kind, args): self.history.append([kind, *args]); self.ret(self.labels_to_pointer(result))
        elif rva == 0x6BC9D0:
            assert r8 == self.rq(self.base + self.slot_names['set_id_method'])
            result = self.options.get('set_id_result', 'tween3'); args = [self.oid(cx), self.oid(dx), self.oid(r8), result]
            if self.event('set_id', args): self.history.append(['set_id', *args]); self.ret(self.labels_to_pointer(result))
        elif rva == 0x2B7D90: self.event('native_null_guard', []); self.error = 'native_null_guard'; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def run(self, options=None, retained=False):
        if not retained: self.prepare(options or {})
        elif options is not None: self.options.update(deepcopy(options))
        self.method, self.error, self.fault, self.allowed = 'MouseExit', None, None, {}
        self.entry_counts = self.counts.copy(); initial, old = self.snapshot(), len(self.events)
        x, sp = self.x, self.entry_sp; self.q(sp, self.stop)
        raw = [0 if self.options.get('null_owner') else self.p['owner'], *INCOMING[1:]]
        for n, v in zip(['RCX', 'RDX', 'R8', 'R9'], raw): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), v)
        for i, v in enumerate(XINCOMING): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), v)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFAB0000000000000 + i)
        for i in range(6, 16): self.u.reg_write(getattr(x, 'UC_X86_REG_XMM' + str(i)), (0xABCDEF9876543210 << 64) | i)
        self.u.reg_write(x.UC_X86_REG_RSP, sp)
        try: self.u.emu_start(self.base + START, self.stop, timeout=10_000_000, count=20000)
        except self.unicorn.UcError as exc:
            assert exc.errno == self.unicorn.UC_ERR_READ_UNMAPPED
            pc = self.reg(x.UC_X86_REG_RIP) - self.base; assert pc in [0x35F841, 0x35F874, 0x35F96D, 0x35F974, 0x35F9D3, 0x35F9DA]
            self.error, self.fault = 'native_storage_fault', hex(pc)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): assert self.reg(getattr(x, 'UC_X86_REG_' + n)) == 0xFAB0000000000000 + i
            for i in range(6, 16): assert self.reg(getattr(x, 'UC_X86_REG_XMM' + str(i))) == (0xABCDEF9876543210 << 64) | i
        final = self.snapshot()
        for n, before in initial['memory'].items():
            before, after = bytes.fromhex(before), bytes.fromhex(final['memory'][n]); assert all(i in self.allowed.get(n, set()) or b == after[i] for i, b in enumerate(before)), n
        assert final['vector_argument_hex'][24:] == initial['vector_argument_hex'][24:]
        for pin in self.float_pins: assert self.rd(self.base + int(pin['slot_rva'], 16)) == pin['bits']
        row = {'options': deepcopy(self.options), 'entry_raw_args': raw, 'entry_raw_xmm': XINCOMING + [NONVOL_XMM6],
               'returned': returned, 'error': self.error, 'fault_rva': self.fault, 'initial': initial, 'events': deepcopy(self.events[old:]), 'final': final,
               'service_counts_before': self.entry_counts.copy(), 'service_counts_after': self.counts.copy(), 'nonvolatile_abi_verified': returned, 'reached_only_storage_retention_verified': True}
        self.verify(row); row['independent_ordered_model_verified'] = True; return row

    def verify(self, row):
        model, options, counts = deepcopy(row['initial']), row['options'], row['service_counts_before'].copy()
        raw, xmm = row['entry_raw_args'].copy(), row['entry_raw_xmm'].copy()
        model['native_entries'].append({'method': 'MouseExit', 'raw_args': raw.copy(), 'raw_xmm': xmm.copy(), 'caller_return': self.caller_return(self.stop)})
        index, error, fault, returned = 0, None, None, False
        class Stopped(Exception): pass
        def read(n, off, width=8): return int.from_bytes(bytes.fromhex(model['memory'][n])[off:off + width], 'little')
        def ptr(label): return self.labels_to_pointer(label)
        def oid(bits): return self.oid(bits)
        def write(n, off, width, bits):
            data = bytearray.fromhex(model['memory'][n]); data[off:off + width] = bits.to_bytes(width, 'little'); model['memory'][n] = data.hex()
        def mutate(kind, relative):
            for n, off, width, value in options.get('mutations', {}).get(kind + ':' + str(relative), []):
                if n == 'flag': model['metadata_flags'][hex(FLAGS[off])] = value
                elif n == 'slot': model['metadata_slots'][hex(self.slot_names[off])] = value
                else: write(n, off, width, ptr(value) if isinstance(value, str) or value is None else value)
        def emit(kind, args, caller, terminal=False):
            nonlocal index, raw, xmm, error
            ordinal = counts.get(kind, 0) + 1; counts[kind] = ordinal
            event = {'kind': kind, 'ordinal': ordinal, 'args': args, 'abi': {'raw_args': raw.copy(), 'raw_xmm': xmm.copy(), 'caller_return': self.caller_return(self.base + caller)}, 'snapshot': model}
            assert row['events'][index] == event, (kind, index, row['options'])
            index += 1; relative = ordinal - row['service_counts_before'].get(kind, 0)
            if terminal or options.get('failure') == [kind, relative]: error = kind; raise Stopped()
            mutate(kind, relative); raw, xmm = POISON.copy(), XPOISON.copy() + [xmm[6]]
            model['service_history'].append([kind, *args])
        def metadata(slot, caller):
            raw[0] = self.base + slot; emit('metadata', [hex(slot), model['metadata_slots'][hex(slot)]], caller)
            # The resolver may replace its own slot before producing the completed diagnostic.
            model['service_history'][-1] = ['metadata', hex(slot), model['metadata_slots'][hex(slot)]]
        def storage_fault(pc):
            nonlocal error, fault
            error, fault = 'native_storage_fault', hex(pc); raise Stopped()
        def null_guard(): emit('native_null_guard', [], 0x35FA23, True)
        def set_id(tween, caller):
            raw[0], raw[1], raw[2] = ptr(tween), read('owner', 0x40), ptr(model['metadata_slots'][hex(self.slot_names['set_id_method'])])
            emit('set_id', [tween, oid(raw[1]), oid(raw[2]), options.get('set_id_result', 'tween3')], caller)
        try:
            if model['metadata_flags'][hex(FLAGS[0])] == 0:
                for slot, caller in self.metadata_order[:2]: metadata(slot, caller)
                model['metadata_flags'][hex(FLAGS[0])] = 1
            if row['entry_raw_args'][0] == 0: storage_fault(0x35F841)
            if read('owner', 0x70, 1) != 0: returned = True
            else:
                action = oid(read('owner', 0x68))
                if action:
                    raw[0], raw[1] = read(action, 0x40), read(action, 0x28)
                    emit('hover_exit', [action, oid(read(action, 0x18)), oid(raw[0]), oid(raw[1])], 0x35F869)
                cls = model['metadata_slots'][hex(self.slot_names['tween_class'])]; raw[0] = ptr(cls)
                animation = oid(read('owner', 0x40))
                if cls is None: storage_fault(0x35F874)
                if read(cls, 0xE0, 4) == 0:
                    emit('class_init', [cls], 0x35F882); write(cls, 0xE0, 4, 1)
                raw[0], raw[1], raw[2] = ptr(animation), (raw[1] & ~0xFF) | 1, 0
                emit('kill', [animation, raw[1], 0, options.get('kill_result_bits', 0x12345678FFFFFFFF)], 0x35F88F)
                array = oid(read('owner', 0x28))
                if array is None: null_guard()
                i = 0
                while True:
                    length = read(array, 0x18, 4); signed = length if length < (1 << 31) else length - (1 << 32)
                    if i >= signed: break
                    assert 0 <= i < length <= 3
                    obj = oid(read(array, 0x20 + i * 8)); raw[0] = ptr(obj)
                    if obj is None: null_guard()
                    raw[1], raw[2] = 0, 0; emit('set_active', [obj, 0, 0], 0x35F8C9); i += 1
                model['saved_xmm6_bits'] = xmm[6]; xmm[6] = DURATION
                for j, caller in enumerate([0x35F8F7, 0x35F92A]):
                    raw[0], raw[3] = read('owner', 0x30 if j == 0 else 0x38), 0
                    xmm[1], xmm[2] = (0 if j == 0 else SHADOW_Y), xmm[6]; model['fifth_argument_bits'] = 0
                    tween = options.get('tween_results', ['tween0', 'tween1', 'tween2', 'tween3'])[j]
                    emit('move_y', [oid(raw[0]), tween, xmm[1], xmm[2], 0, 0], caller); set_id(tween, 0x35F90A if j == 0 else 0x35F93D)
                for j, caller in enumerate([0x35F990, 0x35F9F6]):
                    captured = oid(read('owner', 0x30 if j == 0 else 0x38))
                    if model['metadata_flags'][hex(FLAGS[1])] == 0:
                        metadata(*self.metadata_order[2 + j]); model['metadata_flags'][hex(FLAGS[1])] = 1
                    cls = model['metadata_slots'][hex(self.slot_names['vector_class'])]
                    raw[0], raw[3], xmm[2] = ptr(captured), 0, xmm[6]
                    if cls is None: storage_fault(0x35F96D if j == 0 else 0x35F9D3)
                    static = oid(read(cls, 0xB8))
                    if static is None: storage_fault(0x35F974 if j == 0 else 0x35F9DA)
                    data = bytes.fromhex(model['memory'][static])[0xC:0x18]
                    xmm[0] = int.from_bytes(data[:8], 'little'); raw[1] = self.frame + 0x30
                    model['vector_argument_hex'] = data.hex() + model['vector_argument_hex'][24:]
                    tween = options.get('tween_results', ['tween0', 'tween1', 'tween2', 'tween3'])[2 + j]
                    emit('scale', [captured, tween, data.hex(), xmm[2], 0], caller); set_id(tween, 0x35F9A3 if j == 0 else 0x35FA09)
                returned = True
        except Stopped: pass
        assert index == len(row['events']) and (returned, error, fault) == (row['returned'], row['error'], row['fault_rva'])
        assert model == row['final'] and counts == row['service_counts_after']


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for cold, action, initialized in itertools.product([False, True], [False, True], [0, 1]): cases.append(m.run({'cold': cold, 'null_action': action, 'initialized_word': initialized}))
    for value in [1, 0x80, 0xFF]: cases.append(m.run({'cold': True, 'block_byte': value}))
    for value in [0x80, 0xFF]: cases.append(m.run({'warm_flag': value}))
    for value in [0x80000000, 0xFFFFFFFF]: cases.append(m.run({'initialized_word': value}))
    cases.append(m.run({'vector_bits': [0x7FC01234, 0x80000000, 0x7F800000]}))
    for value in [0, 1, 2, 3, 0x80000000, 0xFFFFFFFF, 0x100000000, 0x100000002]: cases.append(m.run({'length_qword': value}))
    for i in range(3): cases.append(m.run({'length_qword': 3, 'null_element': i}))
    for option in ['null_owner', 'null_array', 'null_animation', 'null_card', 'null_shadow']: cases.append(m.run({'cold': True, option: True}))
    for value in [0, 1, 0x80000000, 0xABCDEF1200000001]: cases.append(m.run({'kill_result_bits': value}))
    for results in [[None] * 4, ['tween0'] * 4, ['card0', 'shadow0', 'tween0', 'tween1']]: cases.append(m.run({'tween_results': results, 'set_id_result': None}))
    mutations = [
        ('hover_exit:1', [['owner', 0x70, 1, 1], ['owner', 0x40, 8, 'animation1']]),
        ('class_init:1', [['owner', 0x40, 8, 'animation1']]),
        ('class_init:1', [['slot', 'tween_class', 8, 'other_tween_class']]),
        ('kill:1', [['owner', 0x28, 8, 'array1']]),
        ('set_active:1', [['owner', 0x28, 8, 'array1'], ['array0', 0x18, 8, 3], ['array0', 0x28, 8, 'object2']]),
        ('set_active:1', [['array0', 0x18, 8, 1]]),
        ('set_active:1', [['array0', 0x28, 8, None]]),
        ('move_y:1', [['owner', 0x38, 8, 'shadow1'], ['owner', 0x40, 8, 'animation1']]),
        ('set_id:1', [['owner', 0x30, 8, 'card1'], ['slot', 'set_id_method', 8, 'other_method']]),
        ('metadata:3', [['owner', 0x30, 8, 'card1'], ['slot', 'vector_class', 8, 'other_vector_class']]),
        ('scale:1', [['owner', 0x38, 8, 'shadow1'], ['vector_statics', 0xC, 4, 0x7FC01234]]),
        ('set_id:3', [['flag', 1, 1, 0], ['owner', 0x38, 8, 'shadow1']]),
        ('metadata:4', [['owner', 0x38, 8, 'shadow0'], ['slot', 'vector_class', 8, 'other_vector_class']]),
        ('hover_exit:1', [['slot', 'tween_class', 8, None]]),
        ('metadata:3', [['slot', 'vector_class', 8, None]]),
        ('metadata:3', [['vector_class', 0xB8, 8, None]]),
        ('set_id:3', [['slot', 'vector_class', 8, None]]),
        ('set_id:3', [['vector_class', 0xB8, 8, None]]),
        ('hover_exit:1', [['owner', 0x68, 8, 'action1'], ['action1', 0x40, 8, 'code0'], ['action1', 0x28, 8, 'method0']])]
    for phase, actions in mutations:
        combined = {phase: actions}
        if phase == 'metadata:4': combined['set_id:3'] = [['flag', 1, 1, 0]]
        cases.append(m.run({'cold': True, 'initialized_word': 0, 'mutations': combined}))
    for options in [{'cold': True, 'initialized_word': 0}, {'cold': True, 'mutations': {'hover_exit:1': mutations[-1][1]}},
                    {'mutations': {'set_active:1': mutations[4][1], 'set_id:3': mutations[11][1]}}]:
        m.prepare(options); rows = [m.run(retained=True) for _ in range(3)]
        assert all(r['returned'] for r in rows) and all(b['initial'] == a['final'] for a, b in zip(rows, rows[1:])); sequences.append(rows)
    for kind in ['metadata', 'hover_exit', 'class_init', 'set_active', 'move_y', 'set_id', 'scale']:
        m.prepare({'cold': True, 'initialized_word': 0, 'failure': [kind, 1]})
        rows = [m.run(retained=True), m.run({'failure': None}, True), m.run(retained=True)]
        assert not rows[0]['returned'] and all(r['returned'] for r in rows[1:]) and all(b['initial'] == a['final'] for a, b in zip(rows, rows[1:])); sequences.append(rows)
    m.prepare({'cold': True})
    rows = [m.run(retained=True), m.run({'tween_results': [None, 'tween0', 'card0', 'shadow0'], 'set_id_result': None}, True), m.run({'tween_results': ['tween3'] * 4, 'set_id_result': 'tween1'}, True)]
    assert all(r['returned'] for r in rows) and all(b['initial'] == a['final'] for a, b in zip(rows, rows[1:])); sequences.append(rows)
    for options in [{'cold': True, 'initialized_word': 0}, {'cold': True, 'block_byte': 1}, {'cold': True, 'null_array': True}, {'cold': True, 'null_element': 1},
                    {'cold': True, 'initialized_word': 0, 'mutations': {'set_active:1': mutations[4][1], 'set_id:3': mutations[11][1], 'metadata:4': mutations[12][1]}},
                    {'null_action': True, 'length_qword': 0}, {'cold': True, 'mutations': {'metadata:3': mutations[14][1]}}]:
        baseline = m.run(options); bid = len(baselines); baselines.append(baseline); counts = {}
        for i, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(dict(options, failure=[kind, counts[kind]]))
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:i + 1] and stopped['final'] == event['snapshot']
            stops.append({'baseline': bid, 'prefix_length': i + 1, 'stopped': stopped})
    missing = m.body_addresses - m.executed; assert missing == {0x35FA23, 0x35FA24, 0x35FA29}
    return {'build': BUILD, 'targets': m.targets, 'ranges': m.ranges, 'supplied_targets': m.service_targets,
            'float_literals': m.float_pins, 'metadata_slot_rvas': {k: hex(v) for k, v in m.slot_names.items()}, 'metadata_flag_rvas': [hex(v) for v in FLAGS],
            'case_count': len(cases), 'cases': cases, 'retained_sequences': sequences, 'failure_baselines': baselines, 'failure_case_count': len(stops), 'failure_stops': stops,
            'instruction_assertions': len(m.checks), 'body_instructions_decoded': len(m.body_addresses), 'body_instructions_executed': len(m.body_addresses & m.executed),
            'native_execution_addresses': len(m.executed), 'unexecuted_addresses': [hex(a) for a in sorted(missing)],
            'scope': 'Complete actual MouseExit caller. Metadata, hover callback, DOTween class initializer/Kill, GameObject.SetActive, DOLocalMoveY, DOScale, SetId and null gateway supplied. Native signed loop exit and subsequent unsigned bounds check share unchanged lowDWORD length with identical nonnegative index: bounds gateway unreachable without asynchronous writes. Bounded authored arrays have at most three elements; arbitrary enormous/concurrent runtime arrays excluded. No Image calls occur. Unity/DOTween/CLR admission, service algorithms and exception unwinding remain supplied. Native class/static/owner faults stop at actual read.'}


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__); p.add_argument('game_root'); p.add_argument('dumper_root'); p.add_argument('--output', required=True)
    args = p.parse_args(); full = audit(args.game_root, args.dumper_root)
    memory = pool_memory(full); assert expand_memory(memory) == full
    report = pool_snapshots(memory); assert expand_snapshots(report) == memory and expand_memory(expand_snapshots(report)) == full
    Path(args.output).write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps({k: report[k] for k in ['case_count', 'failure_case_count', 'body_instructions_decoded', 'body_instructions_executed', 'native_execution_addresses']}))
