"""Exact PickCharacter/Update callers with supplied picker/input/UI boundaries."""
import argparse
import hashlib
import itertools
import json
import re
from copy import deepcopy
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots

TARGETS = {'PickCharacter': (0x367790, 0x36788E, 0x367890, 'tdi5487.m0040'),
           'Update': (0x369700, 0x3697B7, 0x3697C0, 'tdi5487.m0053')}
POISON = 0xFACE123456789090


def snapshot(memory, roots, flags, state, ids):
    def word(n, off): return int.from_bytes(memory[n][off:off+8], 'little')
    def oid(p): return ids[p] if p else None
    picker, gameplay, ui = [oid(roots[n]) for n in ['CharacterPicker_TypeInfo', 'Gameplay_TypeInfo', 'UIEvents_TypeInfo']]
    picker_static, gameplay_static, ui_static = [oid(word(n, 0xB8)) for n in [picker, gameplay, ui]]
    return dict(memory={n: bytes(v).hex() for n, v in memory.items()}, metadata_roots={n: oid(v) for n, v in roots.items()},
                metadata_flags=flags.copy(),
                actor=dict(hover=memory['actor'][0x190], killed=memory['actor'][0xED],
                           state_bits=int.from_bytes(memory['actor'][0xE4:0xE8], 'little'), pickeds=oid(word('actor', 0x188))),
                class_words={n: int.from_bytes(memory[n][0xE0:0xE4], 'little') for n in ['picker_class', 'other_picker_class', 'gameplay_class', 'other_gameplay_class', 'ui_class']},
                picker_list=oid(word(picker_static, 0)), gameplay_state_bits=int.from_bytes(memory[gameplay_static][0x28:0x2C], 'little'),
                detail_delegate=oid(word(ui_static, 0x30)),
                array_length_bits={n: word(n, 0x18) for n in ['pickeds', 'other_pickeds']},
                array_slots={n: [oid(word(n, 0x20+i*8)) for i in range(3)] for n in ['pickeds', 'other_pickeds']},
                supplied_state=deepcopy(state))


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root)
        manifest = json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root)/name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == manifest['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json')); dump = pin('dump.cs', 'dump_cs')
        self.instructions, self.targets, self.bounds, self.flags, refs = {}, [], {}, {}, set()
        for name, (start, end, following, mid) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start]
            assert len(rows) == 1 and rows[0]['Name'] == 'Character$$'+name
            assert rows[0]['TypeSignature'] == 'vii' and rows[0]['Signature'] == f'void Character__{name} (Character_o* __this, const MethodInfo* method);'
            self.targets.append(dict(rows[0], method_id=mid))
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start) == following
            chunks = []
            for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = e
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append((e.struct.BeginAddress, e.struct.EndAddress))
            assert chunks == [(start, end)]
            section = self.pe.get_section_by_rva(start)
            assert section and following <= section.VirtualAddress+section.SizeOfRawData
            raw = self.pe.get_data(start, following-start)
            assert len(raw) == following-start and raw[end-start:] == b'\xcc'*(following-end)
            ins = list(self.cs.disasm(raw[:end-start], start)); assert sum(i.size for i in ins) == end-start
            self.instructions.update({i.address: i for i in ins})
            self.bounds[name] = dict(start=hex(start), end_exclusive=hex(end), next_managed=hex(following), padding_bytes=following-end)
            for i in ins:
                for op in i.operands:
                    if op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                        a = i.address+i.size+op.mem.disp; refs.add(a)
                        if i.mnemonic == 'cmp' and i.operands[0].size == 1: self.flags[name] = a
        names = ['actor', 'other_actor', 'picker_class', 'other_picker_class', 'gameplay_class', 'other_gameplay_class', 'ui_class',
                 'picker_static', 'other_picker_static', 'gameplay_static', 'other_gameplay_static', 'ui_static',
                 'picked_list', 'other_list', 'contains_method', 'pickeds', 'other_pickeds', 'game0', 'game1', 'game2',
                 'detail_action', 'replacement_action', 'action_target', 'action_method']
        self.p = {n: self.arena+0x10000+i*0x1000 for i, n in enumerate(names)}; self.ids = {v: k for k, v in self.p.items()}
        self.sizes = {n: 0x200 if n in ['actor', 'other_actor'] else 0x100 for n in names}
        self.slot_names = {r['Address']: r['Name'] for cat in ['ScriptMetadata', 'ScriptMetadataMethod'] for r in self.metadata[cat] if r['Address'] in refs}
        assert set(self.slot_names.values()) == {'CharacterPicker_TypeInfo', 'Gameplay_TypeInfo', 'UIEvents_TypeInfo', 'Method$System.Collections.Generic.List<Character>.Contains()'}
        self.root_records = {'CharacterPicker_TypeInfo': 'picker_class', 'Gameplay_TypeInfo': 'gameplay_class', 'UIEvents_TypeInfo': 'ui_class',
                             'Method$System.Collections.Generic.List<Character>.Contains()': 'contains_method'}
        self.slots = {n: a for a, n in self.slot_names.items()}
        self.details_gateway = self.stop+0x300
        self.supplied = []
        for a, name, signature, typesig in [
                (0x378DB0, 'CharacterPicker$$ClickedCharacter', 'void CharacterPicker__ClickedCharacter (Character_o* ch, const MethodInfo* method);', 'vii'),
                (0xB55950, 'System.Collections.Generic.List<object>$$Contains', 'bool System_Collections_Generic_List_object___Contains (System_Collections_Generic_List_object__o* __this, Il2CppObject* item, const MethodInfo_B55950* method);', 'iiii'),
                (0x1CD3DD0, 'UnityEngine.Input$$GetMouseButtonDown', 'bool UnityEngine_Input__GetMouseButtonDown (int32_t button, const MethodInfo* method);', 'iii'),
                (0x1C7D810, 'UnityEngine.GameObject$$SetActive', 'void UnityEngine_GameObject__SetActive (UnityEngine_GameObject_o* __this, bool value, const MethodInfo* method);', 'viii')]:
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == a and r['Name'] == name]
            assert len(rows) == 1 and rows[0]['Signature'] == signature and rows[0]['TypeSignature'] == typesig
            self.supplied += rows
        fields = {
            'public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487': ['public bool killedByDemon; // 0xED', 'public ECharacterState state; // 0xE4', 'public GameObject[] pickeds; // 0x188', 'public bool hover; // 0x190'],
            'public class CharacterPicker : MonoBehaviour // TypeDefIndex: 5599': ['public static List<Character> PickedCharacters; // 0x0'],
            'public class Gameplay : MonoBehaviour // TypeDefIndex: 5604': ['public static EGameplayState GameplayState; // 0x28'],
            'public static class UIEvents // TypeDefIndex: 5523': ['public static Action<Character> OnShowCharacterDetails; // 0x30'],
            'public abstract class Delegate : ICloneable, ISerializable // TypeDefIndex: 419': ['private IntPtr invoke_impl; // 0x18', 'private IntPtr method; // 0x28', 'private IntPtr method_code; // 0x40']}
        for declaration, required in fields.items():
            b = re.search('^'+re.escape(declaration)+r'\s*\{(.*?)(?=\n// Namespace:)', dump, re.M|re.S)
            assert b and all(line in b[1] for line in required)
        assert re.search(r'^public sealed class Action<T> : MulticastDelegate // TypeDefIndex: 154$', dump, re.M)
        for declaration, required in [('public enum ECharacterState // TypeDefIndex: 5489', 'public const ECharacterState Hidden = 5;'),
                                      ('public enum EGameplayState // TypeDefIndex: 5607', 'public const EGameplayState Night = 20;')]:
            b = re.search('^'+re.escape(declaration)+r'\s*\{(.*?)\n\}', dump, re.M|re.S); assert b and required in b[1]
        self.fields = fields
        self.checks = {0x3677E0: ('call', '0x378db0'), 0x3677F3: ('mov', 'rcx, qword ptr [rcx]'),
                       0x367809: ('call', '0xb55950'), 0x36780E: ('mov', 'rdi, qword ptr [rdi + 0x188]'),
                       0x367817: ('test', 'al, al'), 0x367822: ('cmp', 'eax, dword ptr [rdi + 0x18]'),
                       0x36782F: ('mov', 'rcx, qword ptr [rdi + rax*8 + 0x20]'), 0x36786A: ('mov', 'dl, 1'),
                       0x369731: ('cmp', 'byte ptr [rbx + 0x190], 0'), 0x36973C: ('lea', 'ecx, [rdx + 1]'),
                       0x369744: ('test', 'al, al'), 0x369748: ('cmp', 'byte ptr [rbx + 0xed], 0'),
                       0x369751: ('cmp', 'dword ptr [rbx + 0xe4], 5'), 0x369772: ('mov', 'rax, qword ptr [rip + 0x238e9c7]'),
                       0x369780: ('cmp', 'dword ptr [rax + 0x28], 0x14'), 0x369794: ('mov', 'rax, qword ptr [rcx + 0x30]'),
                       0x36979D: ('mov', 'r8, qword ptr [rax + 0x28]'), 0x3697A4: ('mov', 'rcx, qword ptr [rax + 0x40]'),
                       0x3697AD: ('jmp', 'qword ptr [rax + 0x18]')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        self.tracking = False; self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE, self.observe_write)

    def oid(self, p):
        if not p: return None
        assert p in self.ids, hex(p)
        return self.ids[p]

    def observe_write(self, uc, access, address, size, value, user_data):
        if self.tracking and address-self.base in self.flags.values():
            assert size == value == 1 and self.reg(self.x.UC_X86_REG_RIP)-self.base in self.instructions
            self.flag_writes[hex(address-self.base)] = 1

    def snapshot(self):
        mem = {n: bytes(self.u.mem_read(p, self.sizes[n])) for n, p in self.p.items()}
        roots = {n: self.rq(self.base+a) for n, a in self.slots.items()}
        return snapshot(mem, roots, {hex(a): self.u.mem_read(self.base+a, 1)[0] for a in self.flags.values()}, self.state, self.ids)

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        self.state = dict(games={n: False for n in ['game0', 'game1', 'game2']},
                          members={'picked_list': ['actor'] if options.get('picked', True) else [], 'other_list': []},
                          clicked=[], contains=[], input=[], details=[], entries=[])
        for n, p in self.p.items(): self.u.mem_write(p, bytes([0xA5])*self.sizes[n])
        for n, target in self.root_records.items(): self.q(self.base+self.slots[n], self.p[target])
        for cls, static in [('picker_class', 'picker_static'), ('other_picker_class', 'other_picker_static'),
                            ('gameplay_class', 'gameplay_static'), ('other_gameplay_class', 'other_gameplay_static'), ('ui_class', 'ui_static')]:
            self.q(self.p[cls]+0xB8, self.p[static]); self.d(self.p[cls]+0xE0, options.get('class_word', 0) if not cls.startswith('other_') else 0xFFFFFFFF)
        self.q(self.p['picker_static'], 0 if options.get('null_list') else self.p['picked_list'])
        self.q(self.p['other_picker_static'], self.p['other_list'])
        self.d(self.p['gameplay_static']+0x28, options.get('gameplay_bits', 10)); self.d(self.p['other_gameplay_static']+0x28, options.get('other_gameplay_bits', 20))
        self.q(self.p['ui_static']+0x30, 0 if options.get('null_delegate') else self.p['detail_action'])
        for n in ['detail_action', 'replacement_action']:
            self.q(self.p[n]+0x18, self.details_gateway); self.q(self.p[n]+0x28, self.p['action_method'])
            target = options.get('action_target', 'action_target'); assert target in [None, 'actor', 'other_actor', 'action_target']
            self.q(self.p[n]+0x40, self.p[target] if target else 0)
            self.q(self.p[n]+0x20, self.p['other_actor'])  # Deliberately different unused m_target.
        self.u.mem_write(self.p['actor']+0x190, bytes([options.get('hover', 1)])); self.u.mem_write(self.p['actor']+0xED, bytes([options.get('killed', 0)]))
        self.d(self.p['actor']+0xE4, options.get('state_bits', 10)); self.q(self.p['actor']+0x188, 0 if options.get('null_array') else self.p['pickeds'])
        for n, slots in [('pickeds', ['game0', 'game1', 'game2']), ('other_pickeds', ['game2', 'game0', 'game1'])]:
            self.q(self.p[n]+0x18, options.get('array_length_bits', 0xFACE000000000002))
            for i, item in enumerate(slots):
                self.q(self.p[n]+0x20+i*8, 0 if options.get('null_index') == i else self.p['game0' if options.get('alias_games') else item])
        for a in self.flags.values(): self.u.mem_write(self.base+a, bytes([options.get('warm_byte', 0)]))

    def mutate(self, phase):
        self.phase_counts[phase] = self.phase_counts.get(phase, 0)+1
        if self.options.get('mutation_phase') != phase or self.options.get('mutation_occurrence', 1) != self.phase_counts[phase]: return
        action = self.options['mutation']
        def write(n, off, value, size=8):
            self.u.mem_write(self.p[n]+off, value.to_bytes(size, 'little'))
            self.writes.setdefault(n, set()).update(range(off, off+size))
        if action in ['clear_array', 'replace_array']: write('actor', 0x188, 0 if action == 'clear_array' else self.p['other_pickeds'])
        elif action == 'clear_second': write('pickeds', 0x28, 0)
        elif action == 'replace_second': write('pickeds', 0x28, self.p['game2'])
        elif action == 'shrink_array': write('pickeds', 0x18, 1)
        elif action == 'clear_list': write('picker_static', 0, 0)
        elif action == 'replace_list': write('picker_static', 0, self.p['other_list'])
        elif action in ['picker_root', 'gameplay_root']:
            name = 'CharacterPicker_TypeInfo' if action == 'picker_root' else 'Gameplay_TypeInfo'
            target = 'other_picker_class' if action == 'picker_root' else 'other_gameplay_class'
            self.q(self.base+self.slots[name], self.p[target]); self.root_writes[name] = target
        elif action in ['clear_delegate', 'replace_delegate']: write('ui_static', 0x30, 0 if action == 'clear_delegate' else self.p['replacement_action'])
        elif action == 'clear_hover': write('actor', 0x190, 0, 1)
        elif action == 'set_killed': write('actor', 0xED, 0x80, 1)
        elif action == 'hide_actor': write('actor', 0xE4, 5, 4)
        elif action == 'night': write('gameplay_static', 0x28, 20, 4)
        else: raise AssertionError(action)

    def event(self, kind, args):
        result = super().event(kind, args)
        e = self.events[-1]; ret = self.rq(self.reg(self.x.UC_X86_REG_RSP))
        e.update(raw_args=[self.reg(getattr(self.x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']],
                 caller=hex(ret-self.base), caller_kind='fixture_return_sentinel' if ret == self.stop else 'native_return', native_phase=self.entry)
        return result

    def ret(self, value=0):
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']: self.u.reg_write(getattr(self.x, 'UC_X86_REG_'+n), POISON)
        for i in range(6): self.u.reg_write(getattr(self.x, f'UC_X86_REG_XMM{i}'), (1<<127)|i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        rva, x = address-self.base, self.x; self.executed.add(rva)
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_'+n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == TARGETS[self.entry][0]: self.state['entries'].append(dict(method=self.entry, raw_args=[cx, dx, r8, r9]))
        if rva in self.instructions: return
        if rva == 0x2B7B40:
            assert cx-self.base in self.slot_names
            name = self.slot_names[cx-self.base]
            if self.event('metadata_service', [name]): self.mutate('metadata'); self.ret(self.rq(cx))
        elif rva == 0x281D90:
            assert cx in [self.p[n] for n in ['picker_class', 'other_picker_class', 'gameplay_class', 'other_gameplay_class']] and self.rd(cx+0xE0) == 0
            n = self.oid(cx)
            if self.event('class_initialization_service', [n]):
                self.d(cx+0xE0, 1); self.writes.setdefault(n, set()).update(range(0xE0, 0xE4)); self.mutate('class_init'); self.ret()
        elif rva == 0x378DB0:
            assert cx == self.p['actor'] and dx == 0
            if self.event('clicked_character_service', ['actor', dx]):
                self.state['clicked'].append('actor')
                if self.options.get('clicked_toggles_membership'):
                    cls = self.rq(self.base+self.slots['CharacterPicker_TypeInfo']); static = self.rq(cls+0xB8); name = self.oid(self.rq(static))
                    assert name in self.state['members']; self.state['members'][name] = [] if 'actor' in self.state['members'][name] else ['actor']
                self.mutate('clicked'); self.ret()
        elif rva == 0xB55950:
            assert self.oid(cx) in self.state['members'] and dx == self.p['actor'] and r8 == self.p['contains_method']
            result = self.options.get('contains_bits', 0xABCD123456789000 | (0xFE if 'actor' in self.state['members'][self.oid(cx)] else 0))
            args = [self.oid(cx), 'actor', self.oid(r8), result]
            if self.event('contains_service', args): self.state['contains'].append(args); self.mutate('contains'); self.ret(result)
        elif rva == 0x1C7D810:
            assert self.oid(cx) in self.state['games'] and dx in [0, (POISON & ~255)|1] and r8 == 0
            args = [self.oid(cx), dx, r8]
            if self.event('set_active_service', args): self.state['games'][self.oid(cx)] = bool(dx&255); self.mutate('active'); self.ret()
        elif rva == 0x1CD3DD0:
            assert cx == 1 and dx == 0
            result = self.options.get('mouse_bits', 0xABCD123456789001)
            args = [cx, dx, result]
            if self.event('mouse_button_down_service', args): self.state['input'].append(args); self.mutate('input'); self.ret(result)
        elif address == self.details_gateway:
            cb = self.reg(x.UC_X86_REG_RAX); assert cb in [self.p['detail_action'], self.p['replacement_action']]
            assert cx == self.rq(cb+0x40) and dx == self.p['actor'] and r8 == self.rq(cb+0x28)
            args = [self.oid(cb), self.oid(cx), 'actor', self.oid(r8)]
            if self.event('details_action_service', args): self.state['details'].append(args); self.mutate('details'); self.ret()
        elif rva in [0x2B7D80, 0x2B7D90]:
            kind = 'native_bounds_guard' if rva == 0x2B7D80 else 'native_null_guard'
            self.event(kind, []); self.error = kind; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error, self.counts = options or {}, None, {}
        self.entry = name; self.writes, self.root_writes, self.flag_writes, self.phase_counts = {}, {}, {}, {}
        initial, old = self.snapshot(), len(self.events); x, sp = self.x, self.stack+0x18008; self.q(sp, self.stop)
        for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']): self.u.reg_write(getattr(x, 'UC_X86_REG_'+n), 0xFAB00000+i)
        for i in range(6, 16): self.u.reg_write(getattr(x, f'UC_X86_REG_XMM{i}'), (1<<125)|i)
        args = [self.p['actor'], 0, self.options.get('unused_r8', 0xCAFE123400000008), self.options.get('unused_r9', 0xCAFE123400000009)]
        for n, v in zip(['RCX', 'RDX', 'R8', 'R9'], args): self.u.reg_write(getattr(x, 'UC_X86_REG_'+n), v)
        self.u.reg_write(x.UC_X86_REG_RSP, sp); self.tracking = True
        try: self.u.emu_start(self.base+TARGETS[name][0], self.stop, timeout=10000000, count=10000)
        finally: self.tracking = False
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp+8
            assert all(self.reg(getattr(x, 'UC_X86_REG_'+n)) == 0xFAB00000+i for i, n in enumerate(['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']))
            assert all(self.reg(getattr(x, f'UC_X86_REG_XMM{i}')) == (1<<125)|i for i in range(6, 16))
        final = self.snapshot()
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            assert all(i in self.writes.get(n, set()) or b == after[i] for i, b in enumerate(before)), n
        assert final['metadata_roots'] == {**initial['metadata_roots'], **self.root_writes}
        assert final['metadata_flags'] == {**initial['metadata_flags'], **self.flag_writes}
        row = dict(method=name, options=self.options.copy(), entry_raw_args=args, returned=returned, error=self.error,
                   initial=initial, events=self.events[old:].copy(), final=final,
                   completed_memory_write_offsets={n: sorted(v) for n, v in self.writes.items()}, completed_root_writes=self.root_writes.copy(),
                   reached_native_flag_writes=self.flag_writes.copy(), normal_abi_verified=returned, unrelated_storage_retained=True)
        verify(row, self.p, self.ids, self.slots, self.flags, self.base, self.stop)
        return row


def verify(row, p, ids, slots, flags, base, stop):
    """Independent full service-entry snapshots, ordered effects and consumed ABI."""
    raw = {n: bytearray.fromhex(v) for n, v in row['initial']['memory'].items()}
    roots = {n: p[v] for n, v in row['initial']['metadata_roots'].items()}; native_flags = row['initial']['metadata_flags'].copy()
    state = deepcopy(row['initial']['supplied_state']); options = row['options']; name = row['method']; expected = []
    phase_counts, counts = {}, {}; residual = row['entry_raw_args'][1:].copy()
    state['entries'].append(dict(method=name, raw_args=row['entry_raw_args']))
    def word(n, off): return int.from_bytes(raw[n][off:off+8], 'little')
    def oid(v): return ids[v] if v else None
    def write(n, off, value, size=8): raw[n][off:off+size] = value.to_bytes(size, 'little')
    def mutate(phase):
        phase_counts[phase] = phase_counts.get(phase, 0)+1
        if options.get('mutation_phase') != phase or options.get('mutation_occurrence', 1) != phase_counts[phase]: return
        action = options['mutation']
        if action in ['clear_array', 'replace_array']: write('actor', 0x188, 0 if action == 'clear_array' else p['other_pickeds'])
        elif action in ['clear_second', 'replace_second']: write('pickeds', 0x28, 0 if action == 'clear_second' else p['game2'])
        elif action == 'shrink_array': write('pickeds', 0x18, 1)
        elif action in ['clear_list', 'replace_list']: write('picker_static', 0, 0 if action == 'clear_list' else p['other_list'])
        elif action == 'picker_root': roots['CharacterPicker_TypeInfo'] = p['other_picker_class']
        elif action == 'gameplay_root': roots['Gameplay_TypeInfo'] = p['other_gameplay_class']
        elif action in ['clear_delegate', 'replace_delegate']: write('ui_static', 0x30, 0 if action == 'clear_delegate' else p['replacement_action'])
        elif action == 'clear_hover': write('actor', 0x190, 0, 1)
        elif action == 'set_killed': write('actor', 0xED, 0x80, 1)
        elif action == 'hide_actor': write('actor', 0xE4, 5, 4)
        elif action == 'night': write('gameplay_static', 0x28, 20, 4)
        else: raise AssertionError(action)
    class ModelStop(Exception): pass
    def emit(kind, args, caller, registers=None):
        snap = snapshot(raw, roots, native_flags, state, ids)
        expected.append(dict(kind=kind, args=args, snapshot=snap, caller=hex(caller), native_phase=name,
                             caller_kind='fixture_return_sentinel' if caller == stop-base else 'native_return'))
        if registers is not None: expected[-1]['raw_args'] = registers
        counts[kind] = counts.get(kind, 0)+1
        if options.get('failure') == [kind, counts[kind]]: raise ModelStop
    def poison(): residual[:] = [POISON]*3
    def guard(cx): emit('native_null_guard', [], 0x36788D, [cx, POISON, POISON, POISON]); raise ModelStop
    def init_class(root_name, caller):
        cls = oid(roots[root_name])
        if int.from_bytes(raw[cls][0xE0:0xE4], 'little') == 0:
            emit('class_initialization_service', [cls], caller, [p[cls], *residual])
            write(cls, 0xE0, 1, 4); mutate('class_init'); poison()
    completed = False; error = None
    try:
        if native_flags[hex(flags[name])] == 0:
            requested = ['CharacterPicker_TypeInfo', 'Method$System.Collections.Generic.List<Character>.Contains()'] if name == 'PickCharacter' else ['Gameplay_TypeInfo', 'UIEvents_TypeInfo']
            callers = [0x3677AE, 0x3677BA] if name == 'PickCharacter' else [0x36971E, 0x36972A]
            for root, caller in zip(requested, callers):
                emit('metadata_service', [root], caller, [base+slots[root], *residual]); mutate('metadata'); poison()
            native_flags[hex(flags[name])] = 1
        if name == 'PickCharacter':
            init_class('CharacterPicker_TypeInfo', 0x3677D6)
            emit('clicked_character_service', ['actor', 0], 0x3677E5, [p['actor'], 0, residual[1], residual[2]])
            state['clicked'].append('actor')
            if options.get('clicked_toggles_membership'):
                cls = oid(roots['CharacterPicker_TypeInfo']); static = oid(word(cls, 0xB8)); lst = oid(word(static, 0))
                state['members'][lst] = [] if 'actor' in state['members'][lst] else ['actor']
            mutate('clicked'); poison()
            cls = oid(roots['CharacterPicker_TypeInfo']); static = oid(word(cls, 0xB8)); lst = oid(word(static, 0))
            if lst is None: guard(0)
            bits = options.get('contains_bits', 0xABCD123456789000 | (0xFE if 'actor' in state['members'][lst] else 0))
            args = [lst, 'actor', 'contains_method', bits]
            emit('contains_service', args, 0x36780E, [p[lst], p['actor'], p['contains_method'], POISON])
            state['contains'].append(args); mutate('contains'); poison()
            array = oid(word('actor', 0x188))
            if array is None: guard(POISON)
            index = 0
            while True:
                count = int.from_bytes(raw[array][0x18:0x1C], 'little', signed=True)
                if index >= count: break
                assert index < 3
                game = oid(word(array, 0x20+index*8))
                if game is None: guard(0)
                on = bits & 255 != 0; dx = (POISON & ~255)|1 if on else 0
                args = [game, dx, 0]
                emit('set_active_service', args, 0x367871 if on else 0x367843, [p[game], dx, 0, POISON])
                state['games'][game] = on; mutate('active'); poison(); index += 1
        elif raw['actor'][0x190]:
            bits = options.get('mouse_bits', 0xABCD123456789001); args = [1, 0, bits]
            emit('mouse_button_down_service', args, 0x369744, [1, 0, residual[1], residual[2]])
            state['input'].append(args); mutate('input'); poison()
            actor_state = int.from_bytes(raw['actor'][0xE4:0xE8], 'little')
            if bits & 255 and raw['actor'][0xED] == 0 and actor_state != 5:
                init_class('Gameplay_TypeInfo', 0x369772)
                cls = oid(roots['Gameplay_TypeInfo']); static = oid(word(cls, 0xB8))
                if int.from_bytes(raw[static][0x28:0x2C], 'little') != 20:
                    ui = oid(roots['UIEvents_TypeInfo']); static = oid(word(ui, 0xB8)); cb = oid(word(static, 0x30))
                    if cb:
                        target, method = oid(word(cb, 0x40)), oid(word(cb, 0x28)); args = [cb, target, 'actor', method]
                        emit('details_action_service', args, stop-base, [p[target] if target else 0, p['actor'], p[method], POISON])
                        state['details'].append(args); mutate('details'); poison()
        completed = True
    except ModelStop: error = expected[-1]['kind']
    assert row['returned'] == completed and row['error'] == error
    assert len(row['events']) == len(expected)
    for actual, predicted in zip(row['events'], expected):
        assert all(actual[k] == v for k, v in predicted.items()), (name, options, actual['kind'], predicted)
    assert row['final'] == snapshot(raw, roots, native_flags, state, ids)


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root); cases, sequences, baselines, stops = [], [], [], []
    for byte, count, warm, alias in itertools.product([0, 1, 0x80, 0xFF], [0, 1, 2, 3, 0x80000000, 0xFFFFFFFF], [0, 0xFE], [False, True]):
        cases.append(m.run('PickCharacter', dict(contains_bits=0xABCD123456789000|byte, array_length_bits=0xFACE000000000000|count, warm_byte=warm, alias_games=alias)))
    for hover, mouse, killed, char_state, game_state in itertools.product([0, 1, 0x80], [0, 1, 0x80, 0xFF], [0, 0x80], [5, 10, 0xFFFFFFFF], [10, 20, 0x80000014]):
        cases.append(m.run('Update', dict(hover=hover, mouse_bits=0xABCD123456789000|mouse, killed=killed, state_bits=char_state, gameplay_bits=game_state)))
    for name in TARGETS:
        for options in [dict(class_word=0xFFFFFFFF, warm_byte=0xFE), dict(unused_r8=0xFEDCBA9876543210, unused_r9=0x123456789ABCDEF0)]: cases.append(m.run(name, options))
    for options in [dict(null_list=True), dict(null_array=True), dict(null_index=0), dict(null_index=1), dict(picked=False), dict(clicked_toggles_membership=True)]: cases.append(m.run('PickCharacter', options))
    for options in [dict(null_delegate=True), dict(action_target=None), dict(action_target='actor'), dict(action_target='other_actor')]: cases.append(m.run('Update', options))
    pick_actions = ['clear_array', 'replace_array', 'clear_second', 'replace_second', 'shrink_array', 'clear_list', 'replace_list', 'picker_root']
    for phase, action in itertools.product(['metadata', 'class_init', 'clicked', 'contains', 'active'], pick_actions): cases.append(m.run('PickCharacter', dict(mutation_phase=phase, mutation=action)))
    for phase, action in itertools.product(['metadata', 'input', 'class_init', 'details'], ['clear_hover', 'set_killed', 'hide_actor', 'night', 'clear_delegate', 'replace_delegate', 'gameplay_root']):
        cases.append(m.run('Update', dict(mutation_phase=phase, mutation=action)))
    for action in pick_actions:
        cases.append(m.run('PickCharacter', dict(mutation_phase='active', mutation_occurrence=2, mutation=action, array_length_bits=0xFACE000000000003)))
        cases.append(m.run('PickCharacter', dict(mutation_phase='class_init', mutation=action, class_word=0xFFFFFFFF)))
    for phase, gate in itertools.product(['metadata', 'input', 'class_init', 'details'], [dict(hover=0, warm_byte=0xFE), dict(mouse_bits=0xABCD123456789000)]):
        cases.append(m.run('Update', dict(gate, mutation_phase=phase, mutation='gameplay_root')))
    for options in [dict(picked=False, null_array=True), dict(picked=False, null_index=1)]: cases.append(m.run('PickCharacter', options))
    for options in [{}, dict(clicked_toggles_membership=True), dict(alias_games=True), dict(mutation_phase='contains', mutation='replace_array')]:
        rows = [m.run('PickCharacter', options), m.run('Update', retained=True), m.run('PickCharacter', dict(clicked_toggles_membership=True), True), m.run('Update', dict(mutation_phase='details', mutation='clear_hover'), True)]
        assert all(r['returned'] for r in rows); sequences.append(rows)
    profiles = [('PickCharacter', {}), ('PickCharacter', dict(picked=False)), ('PickCharacter', dict(mutation_phase='contains', mutation='replace_array')),
                ('PickCharacter', dict(mutation_phase='active', mutation='clear_second')), ('PickCharacter', dict(mutation_phase='clicked', mutation='picker_root')),
                ('PickCharacter', dict(mutation_phase='active', mutation_occurrence=2, mutation='shrink_array', array_length_bits=0xFACE000000000003)),
                ('Update', {}), ('Update', dict(mutation_phase='input', mutation='replace_delegate')), ('Update', dict(mutation_phase='class_init', mutation='gameplay_root'))]
    for name, options in profiles:
        baseline = m.run(name, options); bid = len(baselines); baselines.append(baseline); counts = {}
        for index, e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind, 0)+1
            stopped = m.run(name, {**options, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index+1] and stopped['final'] == e['snapshot']
            stops.append(dict(baseline=bid, prefix_length=index+1, result=stopped))
    missing = sorted(set(m.instructions)-m.executed)
    assert missing == [0x367882, 0x367887, 0x36788D], [hex(a) for a in missing]
    return dict(schema='character_pick_details_native_v1', build=BUILD, targets=m.targets, body_bounds=m.bounds,
                scope='exact PickCharacter and Update callers; picker/List.Contains/input/delegate/runtime/SetActive supplied; ShowDescription and lifecycle dispatch excluded',
                supplied_targets=m.supplied, fields=m.fields, metadata_slots={n: hex(a) for n, a in m.slots.items()}, metadata_flags={n: hex(a) for n, a in m.flags.items()},
                diagnostic_windows_not_object_extents=m.sizes, instruction_assertions=len(m.checks), decoded_instructions=len(m.instructions),
                body_identities={name: dict(byte_length=end-start, instruction_count=sum(start <= a < end for a in m.instructions),
                                           sha256=hashlib.sha256(m.pe.get_data(start, end-start)).hexdigest())
                                 for name, (start, end, _, _) in TARGETS.items()},
                covered_instructions=len(set(m.instructions)&m.executed), unexecuted_guard_and_traps=[hex(a) for a in missing],
                cases=cases, retained_sequences=sequences, baselines=baselines, failure_stops=stops,
                summary=dict(cases=len(cases), sequences=len(sequences), baselines=len(baselines), stops=len(stops)))


if __name__ == '__main__':
    p = argparse.ArgumentParser(); p.add_argument('--game-root', required=True); p.add_argument('--dumper-root', required=True); p.add_argument('--output', required=True)
    args = p.parse_args(); report = pool_snapshots(pool_memory(audit(args.game_root, args.dumper_root)))
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True)+'\n', encoding='utf-8')
    print(json.dumps(report['summary'], sort_keys=True))
