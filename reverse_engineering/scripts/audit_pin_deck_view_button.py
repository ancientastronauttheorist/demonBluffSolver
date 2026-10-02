"""Execute four exact PinDeckViewButton callers with explicit bounded services."""
import argparse
import hashlib
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine


ENTRIES = {'OnEnable': (0x3A5660, 0x3A57C3, 0x3A57D0),
           'OnDisable': (0x3A5530, 0x3A565D, 0x3A5660),
           'UpdateView': (0x3A57D0, 0x3A5816, 0x3A5820),
           'OnClick': (0x3A54C0, 0x3A5524, 0x3A5530)}
SERVICES = {0x3C70A0: 'Settings$$get_PinnedDeck', 0x3C7310: 'Settings$$set_PinnedDeck',
            0x1EDB120: 'UnityEngine.UI.Toggle$$SetIsOnWithoutNotify',
            0x4D5170: 'System.Action$$.ctor', 0x116BCC0: 'System.Delegate$$Combine',
            0x116E070: 'System.Delegate$$Remove'}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root)
        self.capstone = capstone
        manifest = json.loads((Path(__file__).parents[1] /
            f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name, key):
            raw = (Path(dumper_root) / name).read_bytes()
            assert hashlib.sha256(raw).hexdigest().upper() == manifest['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json'))
        dump = pin('dump.cs', 'dump_cs')
        def block(pattern):
            found = re.search(pattern, dump, re.M | re.S)
            assert found
            return found[1]
        owner = block(r'^public class PinDeckViewButton : MonoBehaviour // TypeDefIndex: 5728\s*\{(.*?)\n\}')
        assert owner.split('// Methods')[0].strip() == '// Fields\n\tpublic Toggle toggle; // 0x20'
        settings = block(r'^public static class Settings // TypeDefIndex: 5795\s*\{(.*?)\n\}')
        assert 'public static int PinnedDeck { get; set; }' in settings
        ui = block(r'^public static class UIEvents // TypeDefIndex: 5523\s*\{(.*?)\n\}')
        self.ui_fields = [(n, int(a, 16)) for n, a in re.findall(r'public static Action(?:<[^\n]+>)? (\w+); // (0x[\dA-Fa-f]+)', ui)]
        assert len(self.ui_fields) == 21 and ('OnSettingsChange', 0x88) in self.ui_fields
        delegate = block(r'^public abstract class Delegate : ICloneable, ISerializable // TypeDefIndex: 419\s*\{(.*?)\n\}')
        for field in ['private IntPtr invoke_impl; // 0x18', 'private object m_target; // 0x20',
                      'private IntPtr method; // 0x28', 'private IntPtr method_code; // 0x40']:
            assert field in delegate
        multicast = block(r'^public abstract class MulticastDelegate : Delegate // TypeDefIndex: 440\s*\{(.*?)\n\}')
        assert 'private Delegate[] delegates; // 0x78' in multicast
        action = block(r'^public sealed class Action : MulticastDelegate // TypeDefIndex: 153\s*\{(.*?)\n\}')
        assert 'public virtual void Invoke()' in action
        self.targets, self.instructions, self.ranges, self.flags, self.bindings = [], {}, {}, {}, {}
        slots = {r['Address']: ('metadata', r['Name']) for r in self.metadata['ScriptMetadata']}
        slots.update({r['Address']: ('method', r['Name']) for r in self.metadata['ScriptMetadataMethod']})
        for name, (start, end, following) in ENTRIES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == start and r['Name'] == 'PinDeckViewButton$$' + name]
            assert len(rows) == 1 and rows[0]['TypeSignature'] == 'vii'
            assert rows[0]['Signature'] == f'void PinDeckViewButton__{name} (PinDeckViewButton_o* __this, const MethodInfo* method);'
            self.targets += rows
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start) == following
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks and min(a for a, _ in chunks) == start and max(b for _, b in chunks) == end
            self.ranges[name] = [[hex(a), hex(b)] for a, b in chunks]
            section = self.pe.get_section_by_rva(start)
            assert section is not None and following <= section.VirtualAddress + section.SizeOfRawData
            raw = self.pe.get_data(start, following - start)
            assert len(raw) == following - start and raw[end-start:] == b'\xcc' * (following-end)
            decoded = list(self.cs.disasm(raw[:end-start], start))
            assert sum(i.size for i in decoded) == end-start
            self.instructions.update({i.address: i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type == capstone.x86.X86_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                        slot = i.address + i.size + op.mem.disp
                        if slot in slots: self.bindings[slot] = slots[slot]
                        elif i.mnemonic == 'cmp' and i.operands[0].size == 1: self.flags[name] = slot
        assert set(self.flags) == {'OnEnable', 'OnDisable', 'OnClick'}
        assert {name for _, name in self.bindings.values()} == {
            'System.Action_TypeInfo', 'UIEvents_TypeInfo', 'Method$PinDeckViewButton.UpdateView()'}
        self.service_targets = []
        for address, name in SERVICES.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1
            self.service_targets += rows
        assert next(r for r in self.service_targets if r['Name'] == 'Settings$$get_PinnedDeck')['Signature'] == 'int32_t Settings__get_PinnedDeck (const MethodInfo* method);'
        assert next(r for r in self.service_targets if r['Name'] == 'Settings$$set_PinnedDeck')['Signature'] == 'void Settings__set_PinnedDeck (int32_t value, const MethodInfo* method);'
        assert next(r for r in self.service_targets if r['Name'] == 'UnityEngine.UI.Toggle$$SetIsOnWithoutNotify')['Signature'] == 'void UnityEngine_UI_Toggle__SetIsOnWithoutNotify (UnityEngine_UI_Toggle_o* __this, bool value, const MethodInfo* method);'
        method = next(r for r in self.metadata['ScriptMetadataMethod'] if r['Name'] == 'Method$PinDeckViewButton.UpdateView()')
        assert method['MethodAddress'] == ENTRIES['UpdateView'][0]
        self.method_slot = method['Address']
        self.action_slot = next(a for a, (_, n) in self.bindings.items() if n == 'System.Action_TypeInfo')
        self.ui_slot = next(a for a, (_, n) in self.bindings.items() if n == 'UIEvents_TypeInfo')
        self.checks = {
            0x3A56A8: ('test', 'eax, eax'), 0x3A56B5: ('mov', 'dl, 1'),
            0x3A56C2: ('xor', 'dl, dl'), 0x3A56D1: ('call', '0x1edb120'),
            0x3A56E4: ('mov', 'rdi, qword ptr [rcx + 0x88]'),
            0x3A570A: ('call', '0x4d5170'), 0x3A5718: ('call', '0x116bcc0'),
            0x3A5794: ('add', 'rcx, 0x88'), 0x3A57A0: ('jmp', '0x2b6ff0'),
            0x3A5584: ('mov', 'rdi, qword ptr [rcx + 0x88]'),
            0x3A55AA: ('call', '0x4d5170'), 0x3A55B8: ('call', '0x116e070'),
            0x3A5640: ('jmp', '0x2b6ff0'),
            0x3A57E4: ('test', 'eax, eax'), 0x3A57F0: ('mov', 'dl, 1'),
            0x3A5804: ('xor', 'edx, edx'),
            0x3A54E9: ('test', 'eax, eax'), 0x3A54EB: ('sete', 'cl'),
            0x3A54F0: ('call', '0x3c7310'),
            0x3A5503: ('mov', 'rax, qword ptr [rcx + 0x88]'),
            0x3A550F: ('mov', 'rdx, qword ptr [rax + 0x28]'),
            0x3A5513: ('mov', 'rcx, qword ptr [rax + 0x40]'),
            0x3A551B: ('jmp', 'qword ptr [rax + 0x18]')}
        assert all((self.instructions[a].mnemonic, self.instructions[a].op_str) == v for a, v in self.checks.items())
        for name, (start, end, _) in ENTRIES.items():
            targets = [i.op_str for i in self.instructions.values() if start <= i.address < end and i.mnemonic == 'call']
            assert targets.count('0x3c70a0') == int(name != 'OnDisable')
            assert targets.count('0x116bcc0') == int(name == 'OnEnable')
            assert targets.count('0x116e070') == int(name == 'OnDisable')
        self.p = {n: self.arena + 0x50000 + i * 0x1000 for i, n in enumerate(
            ['button', 'toggle', 'other_toggle', 'button_class', 'ui_class', 'ui_static', 'other_static',
             'action_class', 'foreign_class', 'update_method', 'prior_method', 'callback_target'])}
        self.ids = {p: n for n, p in self.p.items()}
        self.sizes = {n: 0x100 if n.endswith('class') else 0xA8 if n.endswith('static') else 0x80 for n in self.p}
        self.callback = self.stop + 0x100
        self.entry_sp = self.stack + 0x18008

    def oid(self, value):
        if not value: return None
        assert value in self.ids, hex(value)
        return self.ids[value]

    def snapshot(self):
        return {'toggle_ref': self.oid(self.rq(self.p['button'] + 0x20)),
                'ui_static_ref': self.oid(self.rq(self.p['ui_class'] + 0xB8)),
                'metadata_flags_u8': {n: self.u.mem_read(self.base + a, 1)[0] for n, a in self.flags.items()},
                'pinned_deck_bits': self.pinned, 'toggle_values': self.toggle_values.copy(),
                'setting_writes': self.setting_writes.copy(), 'callbacks': self.callbacks.copy(),
                'allocation_order': self.allocation_order.copy(),
                'delegates': {self.oid(p): list(v) for p, v in self.delegates.items()},
                'memory': {n: bytes(self.u.mem_read(p, self.sizes[n])).hex() for n, p in self.p.items()},
                'metadata_slots': {n: self.rq(self.base + a) for a, (_, n) in self.bindings.items()}}

    def new_delegate(self, invocations=None, foreign=False):
        token = self.arena + 0x80000 + len(self.delegates) * 0x100
        name = 'delegate_' + str(len(self.delegates))
        self.p[name], self.ids[token], self.sizes[name] = token, name, 0x80
        self.u.mem_write(token, bytes(0x80))
        self.q(token, self.p['foreign_class' if foreign else 'action_class'])
        self.q(token + 0x18, self.callback)
        self.q(token + 0x28, self.p['prior_method'])
        callback_argument = self.options.get('callback_argument', 'callback_target')
        self.q(token + 0x40, 0 if callback_argument is None else self.p[callback_argument])
        self.delegates[token] = list(invocations or [])
        return token

    def prepare(self, options):
        self.options, self.events, self.counts, self.error = options, [], {}, None
        for n in list(self.p):
            if n.startswith('delegate_'):
                self.ids.pop(self.p[n]); self.p.pop(n); self.sizes.pop(n)
        for n, p in self.p.items(): self.u.mem_write(p, bytes([0xA5]) * self.sizes[n])
        self.q(self.p['button'], self.p['button_class']); self.q(self.p['button'] + 8, 0)
        self.q(self.p['button'] + 0x20, 0 if options.get('null_toggle') else self.p['toggle'])
        self.q(self.p['ui_class'] + 0xB8, self.p['ui_static'])
        self.d(self.p['ui_class'] + 0xE0, options.get('class_word', 0))
        for stat in ['ui_static', 'other_static']:
            self.u.mem_write(self.p[stat], bytes(self.sizes[stat]))
        for a, (_, n) in self.bindings.items():
            self.q(self.base + a, self.p[{'UIEvents_TypeInfo': 'ui_class', 'System.Action_TypeInfo': 'action_class',
                'Method$PinDeckViewButton.UpdateView()': 'update_method'}[n]])
        for a in self.flags.values(): self.u.mem_write(self.base + a, bytes([options.get('warm_flag', 0)]))
        self.delegates, self.allocation_order = {}, []
        self.pinned = options.get('pinned_bits', 0)
        self.toggle_values = {'toggle': options.get('initial_toggle', False), 'other_toggle': True}
        self.setting_writes, self.callbacks = [], []
        invocations = []
        if options.get('prior'): invocations.append(['callback_target', 'prior_method'])
        if options.get('preexisting_own'): invocations.append([None if options.get('null_owner') else 'button', 'update_method'])
        if options.get('trailing_prior'): invocations.append(['toggle', 'prior_method'])
        if invocations: self.q(self.p['ui_static'] + 0x88, self.new_delegate(invocations))
        self.replacement = self.new_delegate([['callback_target', 'prior_method']])
        self.q(self.p['other_static'] + 0x88, self.replacement)
        self.allowed = {}

    def mutate(self, phase):
        if self.options.get('mutation_phase') != phase: return
        mutation = self.options['mutation']
        if mutation in ['replace_toggle', 'clear_toggle']:
            self.q(self.p['button'] + 0x20, self.p['other_toggle'] if mutation == 'replace_toggle' else 0)
            self.allowed.setdefault('button', set()).update(range(0x20, 0x28))
        elif mutation in ['replace_event', 'clear_event']:
            static = self.rq(self.p['ui_class'] + 0xB8)
            self.q(static + 0x88, self.replacement if mutation == 'replace_event' else 0)
            self.allowed.setdefault(self.oid(static), set()).update(range(0x88, 0x90))
        elif mutation == 'swap_statics':
            self.q(self.p['ui_class'] + 0xB8, self.p['other_static'])
            self.allowed.setdefault('ui_class', set()).update(range(0xB8, 0xC0))
        else: raise AssertionError(mutation)

    def ret(self, value=0):
        x = self.x
        for n in ['RCX', 'RDX', 'R8', 'R9', 'R10', 'R11']:
            self.u.reg_write(getattr(x, 'UC_X86_REG_' + n), 0xFACE123456789090)
        for i in range(6): self.u.reg_write(getattr(x, f'UC_X86_REG_XMM{i}'), (1 << 127) | i)
        super().ret(value)

    def hook(self, uc, address, size, data):
        rva, x = address - self.base, self.x
        self.executed.add(rva)
        if rva in self.instructions: return
        cx, dx, r8, r9 = [self.reg(getattr(x, 'UC_X86_REG_' + n)) for n in ['RCX', 'RDX', 'R8', 'R9']]
        if rva == 0x2B7B40:
            assert cx - self.base in self.bindings
            if self.event('metadata_service', [self.bindings[cx-self.base][1]]):
                self.mutate('metadata'); self.ret(self.rq(cx))
        elif rva == 0x3C70A0:
            assert cx == 0
            result = self.pinned
            if self.event('pinned_deck_get_service', [cx, result]):
                self.mutate('getter'); self.ret(0xFACE123400000000 | result)
        elif rva == 0x3C7310:
            assert cx in [0, 1] and dx == 0
            if self.event('pinned_deck_set_service', [cx, dx]):
                self.pinned = cx; self.setting_writes.append(cx); self.mutate('setter'); self.ret()
        elif rva == 0x1EDB120:
            assert cx in [self.p['toggle'], self.p['other_toggle']] and r8 == 0
            enabled = self.pinned != 0
            raw = (0xFACE123456789001 if enabled else 0) if self.method == 'UpdateView' else 0xFACE123456789000 | int(enabled)
            assert dx == raw, (self.method, hex(dx), hex(raw))
            if self.event('toggle_without_notify_service', [self.oid(cx), dx, dx & 255, r8]):
                self.toggle_values[self.oid(cx)] = bool(dx & 255); self.mutate('toggle'); self.ret()
        elif rva == 0x2B7D40:
            assert cx == self.p['action_class']
            if self.event('action_allocate_service', [self.oid(cx)]):
                token = self.new_delegate(); self.allocation_order.append(self.oid(token))
                self.mutate('allocate'); self.ret(token)
        elif rva == 0x4D5170:
            assert cx in self.delegates and dx == (0 if self.options.get('null_owner') else self.p['button'])
            assert r8 == self.p['update_method'] and r9 == 0
            if self.event('action_constructor_service', [self.oid(cx), self.oid(dx), self.oid(r8), r9]):
                self.delegates[cx] = [[self.oid(dx), self.oid(r8)]]
                self.q(cx + 0x20, dx); self.q(cx + 0x40, dx); self.q(cx + 0x28, r8)
                self.mutate('constructor'); self.ret()
        elif rva in [0x116BCC0, 0x116E070]:
            assert (cx == 0 or cx in self.delegates) and dx in self.delegates and r8 == 0
            kind = 'delegate_combine_service' if rva == 0x116BCC0 else 'delegate_remove_service'
            if self.event(kind, [self.oid(cx), self.oid(dx), r8]):
                before = self.delegates[cx].copy() if cx else []
                own = self.delegates[dx]
                if rva == 0x116BCC0:
                    result = dx if not cx else self.new_delegate(before + own)
                else:
                    last = next((i for i in range(len(before)-len(own), -1, -1) if before[i:i+len(own)] == own), None)
                    rest = before if last is None else before[:last] + before[last+len(own):]
                    result = cx if last is None else self.new_delegate(rest) if rest else 0
                mode = self.options.get('result', 'normal')
                if mode == 'null': result = 0
                elif mode == 'foreign': result = self.new_delegate([], foreign=True)
                elif mode == 'source': result = cx
                assert mode in ['normal', 'null', 'foreign', 'source']
                self.mutate('combination'); self.ret(result)
        elif rva == 0x2B6FF0:
            assert cx in [self.p['ui_static'] + 0x88, self.p['other_static'] + 0x88] and self.rq(cx) == dx
            # The exact native event store already occurred before this barrier,
            # including when the supplied barrier itself stops at entry.
            self.allowed.setdefault(self.oid(cx - 0x88), set()).update(range(0x88, 0x90))
            if self.event('static_write_barrier_service', [self.oid(cx - 0x88), 'OnSettingsChange', self.oid(dx)]):
                self.mutate('barrier'); self.ret()
        elif address == self.callback:
            assert cx in self.ids or cx == 0
            assert dx in [self.p['update_method'], self.p['prior_method']]
            assert r8 == r9 == 0xFACE123456789090
            if self.event('settings_change_callback_service', [self.oid(cx), self.oid(dx), r8, r9]):
                self.callbacks.append([self.oid(cx), self.oid(dx)])
                self.mutate('callback'); self.ret()
        elif rva in [0x2B7D90, 0x2B7040]:
            kind = 'native_null_guard' if rva == 0x2B7D90 else 'native_cast_failure'
            args = [] if rva == 0x2B7D90 else [self.oid(cx), self.oid(dx)]
            self.event(kind, args); self.error = kind; uc.emu_stop()
        else: raise AssertionError(f'unclaimed native address {rva:x}')

    def invoke(self, name):
        self.method = name
        x, sp = self.x, self.entry_sp
        self.q(sp, self.stop)
        regs = [getattr(x, 'UC_X86_REG_' + n) for n in ['RBX', 'RBP', 'RSI', 'RDI', 'R12', 'R13', 'R14', 'R15']]
        vectors = [getattr(x, f'UC_X86_REG_XMM{i}') for i in range(6, 16)]
        for i, r in enumerate(regs): self.u.reg_write(r, 0xFAB00000 + i)
        for i, r in enumerate(vectors): self.u.reg_write(r, (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64))
        for r, v in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, 0 if self.options.get('null_owner') else self.p['button']),
                     (x.UC_X86_REG_RDX, 0xDEAD123456789ABC), (x.UC_X86_REG_R8, 0xDEADBEEF11111111),
                     (x.UC_X86_REG_R9, 0xDEADBEEF22222222)]: self.u.reg_write(r, v)
        try: self.u.emu_start(self.base + ENTRIES[name][0], self.stop, timeout=10_000_000, count=5000)
        except self.unicorn.UcError as exc:
            assert self.options.get('null_owner') and name in ['OnEnable', 'UpdateView']
            assert exc.errno == self.unicorn.UC_ERR_READ_UNMAPPED
            assert self.reg(x.UC_X86_REG_RIP) == self.base + {'OnEnable': 0x3A56A4, 'UpdateView': 0x3A57E0}[name]
            self.error = 'native_null_owner_dereference'
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop
        assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.reg(r) == 0xFAB00000 + i for i, r in enumerate(regs))
            assert all(self.reg(r) == (0xABCD0000 + i) | ((0xFEDC0000 + i) << 64) for i, r in enumerate(vectors))
        return returned

    def run(self, name, options=None, retained=False):
        if not retained: self.prepare(options or {})
        else: self.options, self.error = options or {}, None
        self.allowed = {}
        initial, old = self.snapshot(), len(self.events)
        returned = self.invoke(name)
        final, events = self.snapshot(), self.events[old:].copy()
        # Caller stores are metadata flags and one reloaded static event slot.
        # Supplied services additionally allocate/initialize delegate records or
        # explicitly mutate fields. All other physical storage must remain exact.
        for n, raw in initial['memory'].items():
            before, after = bytes.fromhex(raw), bytes.fromhex(final['memory'][n])
            allowed = self.allowed.get(n, set())
            assert all(i in allowed or b == after[i] for i, b in enumerate(before)), n
        return {'method': name, 'options': self.options.copy(), 'returned': returned, 'error': self.error,
                'initial': initial, 'events': events, 'final': final,
                'win64_nonvolatile_and_stack_preserved_on_return': returned,
                'unrelated_storage_retained': True}


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    cases, sequences, baselines, stops = [], [], [], []
    for name in ENTRIES:
        for bits, warm in itertools.product([0, 1, 2, 0x7FFFFFFF, 0x80000000, 0xFFFFFFFF], [0, 0xFE]):
            r = m.run(name, {'pinned_bits': bits, 'warm_flag': warm, 'prior': True, 'preexisting_own': True})
            assert r['returned']
            if name in ['OnEnable', 'UpdateView']: assert r['final']['toggle_values']['toggle'] == (bits != 0)
            if name == 'OnClick': assert r['final']['setting_writes'] == [int(bits == 0)]
            cases.append(r)
        for options in [{'null_owner': True}, {'null_toggle': True}, {'prior': True}, {},
                        {'class_word': 0xFFFFFFFF}, {'callback_argument': 'button', 'prior': True},
                        {'callback_argument': 'toggle', 'prior': True}, {'callback_argument': 'other_toggle', 'prior': True},
                        {'callback_argument': None, 'prior': True}]:
            cases.append(m.run(name, options))
    for name in ['OnEnable', 'OnDisable']:
        for prior, own, trailing, mode in itertools.product([False, True], [False, True], [False, True], ['normal', 'null', 'source', 'foreign']):
            r = m.run(name, {'prior': prior, 'preexisting_own': own, 'trailing_prior': trailing, 'result': mode})
            assert r['returned'] == (mode != 'foreign')
            cases.append(r)
    for name, phases in [('OnEnable', ['getter', 'toggle', 'allocate', 'constructor', 'combination']),
                         ('OnDisable', ['allocate', 'constructor', 'combination']),
                         ('UpdateView', ['getter', 'toggle']), ('OnClick', ['getter', 'setter', 'callback'])]:
        for phase, action in itertools.product(phases, ['replace_toggle', 'clear_toggle', 'replace_event', 'clear_event', 'swap_statics']):
            r = m.run(name, {'prior': True, 'mutation_phase': phase, 'mutation': action})
            kinds = [e['kind'] for e in r['events']]
            if phase == 'getter' and name in ['OnEnable', 'UpdateView']:
                if action == 'clear_toggle': assert not r['returned'] and r['error'] == 'native_null_guard'
                elif action == 'replace_toggle':
                    assert next(e for e in r['events'] if e['kind'] == 'toggle_without_notify_service')['args'][0] == 'other_toggle'
            if name == 'OnClick' and phase == 'setter':
                if action == 'clear_event': assert 'settings_change_callback_service' not in kinds
                elif action in ['replace_event', 'swap_statics']:
                    assert next(e for e in r['events'] if e['kind'] == 'settings_change_callback_service')['args'][:2] == ['callback_target', 'prior_method']
            if name in ['OnEnable', 'OnDisable'] and phase in ['allocate', 'constructor', 'combination']:
                combination = next(e for e in r['events'] if e['kind'] in ['delegate_combine_service', 'delegate_remove_service'])
                assert combination['args'][0] == 'delegate_0'  # Native captured old event before the mutating service.
                if action == 'swap_statics':
                    assert next(e for e in r['events'] if e['kind'] == 'static_write_barrier_service')['args'][0] == 'other_static'
            cases.append(r)
    for prior, own in itertools.product([False, True], [False, True]):
        for names in [['OnEnable', 'OnClick', 'UpdateView', 'OnDisable'],
                      ['OnEnable', 'OnEnable', 'OnDisable', 'OnClick', 'OnDisable'],
                      ['OnDisable', 'OnClick', 'OnEnable', 'OnClick']]:
            m.prepare({'prior': prior, 'preexisting_own': own, 'pinned_bits': 0xFFFFFFFF})
            rows = [m.run(n, retained=True) for n in names]
            assert all(r['returned'] for r in rows)
            expected = ([['callback_target', 'prior_method']] if prior else []) + ([['button', 'update_method']] if own else [])
            for r in rows:
                if r['method'] == 'OnEnable': expected.append(['button', 'update_method'])
                elif r['method'] == 'OnDisable':
                    last = next((i for i in range(len(expected)-1, -1, -1) if expected[i] == ['button', 'update_method']), None)
                    if last is not None: expected.pop(last)
                raw = bytes.fromhex(r['final']['memory']['ui_static'])
                token = int.from_bytes(raw[0x88:0x90], 'little')
                assert (r['final']['delegates'][m.oid(token)] if token else []) == expected
            sequences.append(rows)
    for name in ENTRIES:
        options = {'prior': True, 'preexisting_own': True}
        baseline = m.run(name, options); assert baseline['returned']
        baseline_id = len(baselines); baselines.append(baseline); counts = {}
        for index, event in enumerate(baseline['events']):
            kind = event['kind']; counts[kind] = counts.get(kind, 0) + 1
            stopped = m.run(name, {**options, 'failure': [kind, counts[kind]]})
            assert not stopped['returned'] and stopped['events'] == baseline['events'][:index+1]
            assert stopped['final'] == event['snapshot']
            stops.append({'baseline': baseline_id, 'prefix_length': index+1, 'result': stopped})
    unexecuted = sorted(set(m.instructions) - m.executed)
    assert set(unexecuted) == {0x3A5650, 0x3A5651, 0x3A5654, 0x3A5657, 0x3A565C,
        0x3A57AA, 0x3A57AB, 0x3A57AE, 0x3A57B1, 0x3A57B6, 0x3A57C2, 0x3A5815}
    return {'schema': 'pin_deck_view_button_native_v1', 'build': BUILD,
            'scope': 'four complete caller bodies; Settings/UI/runtime/delegate/callback services supplied',
            'targets': m.targets, 'supplied_targets': m.service_targets, 'unwind_ranges': m.ranges,
            'field_pin': {'type': 'PinDeckViewButton', 'type_def_index': 5728, 'toggle_offset': '0x20'},
            'settings_pin': {'type': 'Settings', 'type_def_index': 5795, 'property': 'PinnedDeck', 'property_type': 'int32'},
            'event_pin': {'type': 'UIEvents', 'type_def_index': 5523, 'field': 'OnSettingsChange', 'offset': '0x88'},
            'delegate_pin': {'type': 'System.Delegate', 'type_def_index': 419, 'action_type_def_index': 153,
                'multicast_type_def_index': 440, 'invoke_impl': '0x18', 'm_target': '0x20',
                'method': '0x28', 'method_code': '0x40', 'delegates': '0x78'},
            'event_fields': [{'name': n, 'offset': hex(a)} for n, a in m.ui_fields],
            'metadata_bindings': [{'rva': hex(a), 'kind': k, 'name': n} for a, (k, n) in m.bindings.items()],
            'metadata_flag_rvas': {n: hex(a) for n, a in m.flags.items()},
            'instruction_assertions': len(m.checks), 'decoded_instructions': len(m.instructions),
            'covered_instructions': len(set(m.instructions) & m.executed),
            'unexecuted_instructions': [{'rva': hex(a), 'mnemonic': m.instructions[a].mnemonic,
                                        'operands': m.instructions[a].op_str} for a in unexecuted],
            'normal_and_edge_cases': cases, 'retained_sequences': sequences, 'baselines': baselines,
            'failure_stops': stops,
            'summary': {'cases': len(cases), 'sequences': len(sequences), 'stops': len(stops)}}


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('--game-root', type=Path, required=True)
    p.add_argument('--dumper-root', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args(); report = audit(args.game_root, args.dumper_root)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, sort_keys=True, separators=(',', ':'), ensure_ascii=True) + '\n', encoding='utf-8')
    print(json.dumps(report['summary'], sort_keys=True))
