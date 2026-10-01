"""Native close/reveal callbacks through note hiding, queue processing and saving."""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_storage_join import compact_trace, values
from audit_tutorial_persistence_join import Machine as TutorialMachine, StorageMachine

TARGETS = {
    'close': (0x38C8C0, 0x38C94D, 'TutorialsController$$CloseTutorialIfAble', 'tdi5650.m0013'),
    'reveal': (0x38CC30, 0x38CDB7, 'TutorialsController$$OnCharacterReveal', 'tdi5650.m0010'),
    'hide': (0x38BCB0, 0x38BD7F, 'TutorialNote$$HideNote', 'tdi5633.m0002'),
    'hidden': (0x38DB80, 0x38DC86, 'TutorialsController$$OnTutHidden', 'tdi5650.m0024'),
    'queue': (0x38DDB0, 0x38DF7D, 'TutorialsController$$ProcessTutorialQueue', 'tdi5650.m0025'),
    'appearance': (0x364C40, 0x364CC6, 'Character$$GetCharacterBluffIfAble', 'tdi5487.m0013'),
}
SIGNATURES = {
    'close': 'void TutorialsController__CloseTutorialIfAble (TutorialsController_o* __this, int32_t type, const MethodInfo* method);',
    'reveal': 'void TutorialsController__OnCharacterReveal (TutorialsController_o* __this, Character_o* ch, const MethodInfo* method);',
    'hide': 'void TutorialNote__HideNote (TutorialNote_o* __this, const MethodInfo* method);',
    'hidden': 'void TutorialsController__OnTutHidden (TutorialsController_o* __this, TutorialNote_o* tn, const MethodInfo* method);',
    'queue': 'void TutorialsController__ProcessTutorialQueue (TutorialsController_o* __this, const MethodInfo* method);',
    'appearance': 'CharacterData_o* Character__GetCharacterBluffIfAble (Character_o* __this, const MethodInfo* method);'}


class Machine(TutorialMachine):
    def __init__(self, game_root, dumper_root):
        self.handler_ready = False
        super().__init__(game_root, dumper_root)
        script = json.loads((Path(dumper_root)/'script.json').read_text(encoding='utf-8-sig'))
        dump = (Path(dumper_root)/'dump.cs').read_text(encoding='utf-8')
        self.handler_fields = {
            'public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487': ['CharacterData dataRef; // 0x50', 'CharacterData bluff; // 0x58', 'bool revealed; // 0xD8', 'ECharacterState state; // 0xE4'],
            'public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845': ['bool picking; // 0x13E'],
            'private sealed class TutorialsController.<HoverCharacterRoutine>d__26 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5642': ['int <>1__state; // 0x10', 'object <>2__current; // 0x18', 'TutorialsController <>4__this; // 0x20', 'Character ch; // 0x28'],
            'private sealed class TutorialsController.<PickableCharacterType>d__25 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5645': ['int <>1__state; // 0x10', 'object <>2__current; // 0x18', 'TutorialsController <>4__this; // 0x20', 'Character ch; // 0x28'],
        }
        for declaration, fields in self.handler_fields.items():
            block = re.search(r'^'+re.escape(declaration)+r'\s*\{(.*?)// (?:Methods|Properties)', dump, re.M|re.S)
            assert block and all(field in block[1] for field in fields)
        slots = {r['Address']: ('metadata', r['Name']) for r in script['ScriptMetadata']}
        slots.update({r['Address']: ('method', r['Name']) for r in script['ScriptMetadataMethod']})
        self.handler_bindings, self.handler_flags, self.handler_instructions, self.handler_targets = {}, {}, {}, []
        for label, (a, b, name, method_id) in TARGETS.items():
            rows = [r for r in script['ScriptMethod'] if r['Address'] == a and r['Name'] == name]
            assert len(rows) == 1
            assert rows[0]['Signature'] == SIGNATURES[label]
            self.handler_targets.append(dict(rows[0], method_id=method_id))
            decoded = list(self.cs.disasm(self.pe.get_data(a, b-a), a))
            assert sum(i.size for i in decoded) == b-a, (label, hex(decoded[-1].address+decoded[-1].size), hex(b))
            self.handler_instructions.update({i.address: i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type == self.capstone.x86.X86_OP_MEM and op.mem.base == self.capstone.x86.X86_REG_RIP:
                        slot = i.address+i.size+op.mem.disp
                        if slot in slots:
                            self.handler_bindings[slot] = slots[slot]
                        elif i.mnemonic == 'cmp' and i.operands[0].size == 1:
                            self.handler_flags[slot] = label
        entry = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == 0x357700)
        assert entry.struct.EndAddress == 0x357724
        ctor = list(self.cs.disasm(self.pe.get_data(0x357700, 0x24), 0x357700))
        assert [(i.mnemonic, i.op_str) for i in ctor if i.address == 0x357716] == [('mov', 'dword ptr [rbx + 0x10], edi')]
        self.handler_instructions.update({i.address: i for i in ctor})
        self.restore_time_bits = struct.unpack('<I', self.pe.get_data(0x1F34B18, 4))[0]
        assert self.restore_time_bits == 0x3F800000
        checks = {0x38C921: ('call', '0x38bcb0'), 0x38CD90: ('call', '0x38bcb0'),
                  0x38CCC0: ('call', '0x364c40'), 0x38CCCE: ('cmp', 'byte ptr [rax + 0x13e], 0'),
                  0x38BCDF: ('mov', 'dword ptr [rbx + 0x48], edx'), 0x38BD25: ('jmp', 'qword ptr [rax + 0x18]'),
                  0x38DC66: ('jmp', '0x38ddb0'), 0x38DF3C: ('call', '0x38e1a0'),
                  0x38DF53: ('call', '0xb59ce0'), 0x38DEA9: ('inc', 'edi'), 0x38DF05: ('mov', 'edx, edi')}
        for a, expected in checks.items():
            i = self.handler_instructions[a]; assert (i.mnemonic, i.op_str) == expected
        self.handler_assertions = len(checks)+2

    def setup_handler(self):
        self.setup_tutorials()
        self.entry_pending = False
        self.actor, self.real_data, self.bluff_data = self.alloc(0x200), self.alloc(0x180), self.alloc(0x180)
        self.u.mem_write(self.actor, bytes([0xA5])*0x1B8)
        self.q(self.actor+0x50, 0 if self.options.get('null_real') else self.real_data)
        self.q(self.actor+0x58, self.bluff_data if self.options.get('bluff_present') else 0)
        self.d(self.actor+0xE4, self.options.get('actor_state', 0))
        self.u.mem_write(self.actor+0xD8, bytes([int(self.options.get('revealed', False))]))
        self.u.mem_write(self.real_data+0x13E, bytes([self.options.get('real_picking', 0)]))
        self.u.mem_write(self.bluff_data+0x13E, bytes([self.options.get('bluff_picking', 0)]))
        self.retained_actor = bytes(self.u.mem_read(self.actor, 0x1B8))
        self.handler_types = {name: self.alloc() for _, name in self.handler_bindings.values() if name.startswith('TutorialsController.<')}
        self.handler_tokens = {}
        for slot, (kind, name) in self.handler_bindings.items():
            if slot in self.tutorial_tokens:
                token = self.tutorial_tokens[slot]
            elif name == 'UnityEngine.Object_TypeInfo':
                token = self.ui_type
            elif name == 'System.Action<TutorialNote>_TypeInfo':
                token = self.delegate_type
            elif name in self.handler_types:
                token = self.handler_types[name]
            else:
                assert kind == 'method', name
                token = self.alloc()
            self.handler_tokens[slot] = token; self.q(self.base+slot, token)
        for slot in self.handler_flags:
            self.u.mem_write(self.base+slot, bytes([int(self.options.get('warm', False))]))
        self.handler_routines, self.enumerators = {}, {}
        self.queue_types = []
        for row in self.options.get('queue', []):
            if row.get('null'):
                self.queue_types.append(0)
                continue
            q = self.alloc(); self.queues.append(q)
            self.d(q+0x10, row.get('type', 20)); self.q(q+0x18, self.pivot)
            self.u.mem_write(q+0x20, bytes([int(row.get('restriction', False))]))
            self.queue_types.append(q)
        self.q(self.controller+0x28, self.raw_list(self.queue_types, 8) if not self.options.get('null_queue') else 0)
        self.hide_method = next(self.tutorial_tokens[a] for a, (_, n) in self.tutorial_bindings.items() if n == 'Method$TutorialsController.OnTutHidden()')
        for note, row in zip(self.notes, self.options.get('notes', [{}]*len(self.notes))):
            self.u.mem_write(note+0x4C, bytes([int(row.get('clickable', True))]))
            if row.get('on_hide', False):
                token = self.alloc(); self.delegates[token] = {'target': self.controller, 'method': self.hide_method}
                self.write_delegate(token); self.q(note+0x38, token)
        self.initial_notes = {n: bytes(self.u.mem_read(n, 0x80)) for n in self.notes}
        self.initial_controller = bytes(self.u.mem_read(self.controller, 0x40))
        self.initial_data = {p: bytes(self.u.mem_read(p, 0x180)) for p in [self.real_data, self.bluff_data]}
        self.handler_ready = True

    def handler_method(self, name):
        return next(self.handler_tokens[a] for a, (_, n) in self.handler_bindings.items() if n == name)

    def write_delegate(self, token):
        self.q(token, self.delegate_type)
        self.q(token+0x18, self.base+TARGETS['hidden'][0])
        self.q(token+0x28, self.hide_method); self.q(token+0x40, self.controller)

    def snapshot(self):
        result = super().snapshot()
        if self.handler_ready:
            result['handler'] = {'actor': {'identity': self.actor, 'real': self.rq(self.actor+0x50),
                'bluff': self.rq(self.actor+0x58), 'state': self.rd(self.actor+0xE4), 'revealed': self.u.mem_read(self.actor+0xD8, 1)[0],
                'real_picking': self.u.mem_read(self.real_data+0x13E, 1)[0], 'bluff_picking': self.u.mem_read(self.bluff_data+0x13E, 1)[0]},
                'routines': {k: {'type': kind, 'state': self.rd(k+0x10), 'current': self.rq(k+0x18),
                    'controller': self.rq(k+0x20), 'character': self.rq(k+0x28)} for k, kind in self.handler_routines.items()},
                'metadata_initialized': {hex(a): bool(self.u.mem_read(self.base+a, 1)[0]) for a in self.handler_flags}}
        return result

    def invoke(self, rva, receiver, argument=0):
        if rva == 0x3EAAA0 and self.options.get('handler_entry'):
            self.setup_handler()
            label = self.options['handler_entry']
            receiver = self.notes[0] if label == 'hide' else self.actor if label == 'appearance' else self.controller
            argument = (0 if self.options.get('null_character') else self.actor) if label == 'reveal' else self.notes[0] if label == 'hidden' else self.options.get('type', 10) if label == 'close' else 0
            return StorageMachine.invoke(self, TARGETS[label][0], receiver, argument)
        return StorageMachine.invoke(self, rva, receiver, argument)

    def hook(self, uc, address, size, data):
        if not self.handler_ready:
            return super().hook(uc, address, size, data)
        rva, x = address-self.base, self.x
        cx, dx, r8 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8)]
        self.executed.add(rva)
        if rva == 0x2B7B40 and cx-self.base in self.handler_bindings:
            self.service('handler_metadata_service', [self.handler_bindings[cx-self.base][1]], lambda: self.ret(self.rq(cx)))
        elif rva == 0x2B7D40 and cx in self.handler_types.values():
            kind = next(n for n, t in self.handler_types.items() if t == cx)
            def allocation():
                token = self.alloc(); self.handler_routines[token] = kind; self.ret(token)
            self.service('handler_routine_allocate_service', [kind], allocation)
        elif rva == 0x1C7F160 and dx in self.handler_routines:
            assert cx == self.controller and r8 == 0 and self.rd(dx+0x10) == 0 and self.rq(dx+0x20) == cx
            def publication():
                self.coroutines.append({'identity': dx, 'kind': self.handler_routines[dx], 'controller': cx, 'character': self.rq(dx+0x28), 'state': self.rd(dx+0x10)})
                self.ret(self.alloc())
            self.service('handler_start_coroutine_service', [cx, dx], publication)
        elif rva == 0x1C822C0:
            assert cx in [0, self.bluff_data] and dx == r8 == 0
            null = not cx or not self.options.get('bluff_live', True)
            self.service('handler_unity_null_service', [cx, null], lambda: self.ret(0xFACE000000000000 | int(null)))
        elif rva == 0x1C8E540:
            bits = self.reg(x.UC_X86_REG_XMM0)&0xFFFFFFFF
            assert dx == 0 and bits in [0, self.restore_time_bits]
            def time_scale():
                self.time_scale_bits = bits; self.ret()
            self.service('joined_time_scale_service', [bits], time_scale)
        elif rva == 0x4D5B60:
            assert cx in self.delegates and dx == self.controller and r8 == self.hide_method and self.reg(x.UC_X86_REG_R9) == 0
            def constructor():
                self.delegates[cx] = {'target': dx, 'method': r8}; self.write_delegate(cx); self.ret()
            self.service('joined_hide_delegate_constructor_service', [cx, dx, r8], constructor)
        elif rva == 0x116E070:
            assert r8 == 0 and dx in self.delegates and (not cx or cx in self.delegates)
            assert not cx or 'combined' not in self.delegates[cx]
            result = 0 if not cx or self.delegates[cx] == self.delegates[dx] else cx
            self.service('joined_hide_delegate_remove_service', [cx, dx, result], lambda: self.ret(result))
        elif rva == 0xB16640:
            assert dx in self.raw_lists and self.raw_lists[dx] == 8 and self.stack <= cx <= self.stack+0x20000-24
            assert r8 == self.handler_method('Method$System.Collections.Generic.List<TutorialQueue>.GetEnumerator()')
            def enumerator():
                self.q(cx, dx); self.d(cx+8, 0); self.d(cx+12, self.rd(dx+0x1C)); self.q(cx+16, 0); self.ret(cx)
            self.service('queue_get_enumerator_service', [dx, r8], enumerator)
        elif rva == 0x9693D0:
            assert self.stack <= cx <= self.stack+0x20000-24
            assert dx == self.handler_method('Method$System.Collections.Generic.List.Enumerator<TutorialQueue>.MoveNext()')
            token, index = self.rq(cx), self.rd(cx+8)
            assert token in self.raw_lists and self.rd(cx+12) == self.rd(token+0x1C)
            count = self.rd(token+0x18)
            def move_next():
                current = self.rq(self.rq(token+0x10)+0x20+index*8) if index < count else 0
                self.d(cx+8, index+1); self.q(cx+16, current); self.ret(0xFACE000000000000 | int(index<count))
            self.service('queue_move_next_service', [token, index, count, dx], move_next)
        elif rva == 0xB22150:
            assert cx in self.raw_lists and dx < self.rd(cx+0x18)
            assert r8 == self.handler_method('Method$System.Collections.Generic.List<TutorialQueue>.get_Item()')
            result = self.rq(self.rq(cx+0x10)+0x20+8*dx)
            self.service('queue_item_service', [cx, dx, r8, result], lambda: self.ret(result))
        elif rva == 0xB59CE0:
            assert cx in self.raw_lists and dx < self.rd(cx+0x18)
            assert r8 == self.handler_method('Method$System.Collections.Generic.List<TutorialQueue>.RemoveAt()')
            def remove_at():
                backing, count = self.rq(cx+0x10), self.rd(cx+0x18)
                for index in range(dx, count-1):
                    self.q(backing+0x20+8*index, self.rq(backing+0x20+8*(index+1)))
                self.q(backing+0x20+8*(count-1), 0); self.d(cx+0x18, count-1)
                self.d(cx+0x1C, self.rd(cx+0x1C)+1); self.ret()
            self.service('queue_remove_at_service', [cx, dx, r8], remove_at)
        else:
            return super().hook(uc, address, size, data)

    def run_handler(self, label, state, options=None, storage=None):
        self.handler_ready = self.tutorial_ready = False
        result = self.run_data('Save', state, {'handler_entry': label, **(options or {})}, storage=storage)
        result['method'] = label
        assert bytes(self.u.mem_read(self.actor, 0x1B8)) == self.retained_actor
        assert bytes(self.u.mem_read(self.controller, 0x40)) == self.initial_controller
        assert all(bytes(self.u.mem_read(p, len(before))) == before for p, before in self.initial_data.items())
        assert all(bytes(self.u.mem_read(p, len(before))) == before for p, before in self.retained_arrays.items())
        allowed = set(range(0x28, 0x30))|set(range(0x38, 0x40))|set(range(0x48, 0x4C))
        assert all(all(a==b for i, (a,b) in enumerate(zip(before, self.u.mem_read(n,0x80))) if i not in allowed)
                   for n,before in self.initial_notes.items())
        result['actor_and_unconsumed_note_fields_retained_verified'] = True
        return result


def audit(game_root, dumper_root):
    m = Machine(game_root, dumper_root)
    old = {'key': 'Tutorials', 'completedTutorials': ['old-t'], 'unlockedCharactersId': ['c']}
    cases, baselines, failures = [], [], []
    for label, warm, active, clickable, stop_time, stage, count in itertools.product(['close','reveal'], [False,True], [False,True], [False,True], [False,True], [0,1,7], [0,1,2]):
        opts = {'warm':warm, 'notes':[{'id':'old-t','type':10,'active':active,'clickable':clickable,'stop_time':stop_time,'stage':stage,'stages':[True]*count}], 'real_picking':0}
        result=m.run_handler(label,old,opts,{})
        assert result['returned'],result['error']
        note=result['final']['tutorials']['notes'][0]
        advances=active and clickable
        assert note['stage']==stage+int(advances)
        assert values(result['final']['save'])==old and result['final']['storage']=={}
        assert len(result['final']['tutorials']['coroutines'])==int(label=='reveal')
        cases.append(result)
    for state, revealed, present, live, real_pick, bluff_pick in itertools.product([0,10,20,30],[False,True],[False,True],[False,True],[0,1],[0,1]):
        opts={'actor_state':state,'revealed':revealed,'bluff_present':present,'bluff_live':live,'real_picking':real_pick,'bluff_picking':bluff_pick,'notes':[]}
        result=m.run_handler('reveal',old,opts,{})
        assert result['returned']
        picking=real_pick if state in [20,30] or revealed or not present or not live else bluff_pick
        assert len(result['final']['tutorials']['coroutines'])==1+int(bool(picking))
        cases.append(result)
    joined_notes=[{'id':'old-t','type':10,'active':True,'stage':0,'stages':[True],'on_hide':True},
                  {'id':'new-t','type':20,'state':0,'active':False,'stage':0,'stages':[True]}]
    for label, restrictions in itertools.product(['close','reveal','queue'], [[],[False],[True],[True,False],[False,True],[True,False,True]]):
        opts={'notes':joined_notes,'queue':[{'type':20,'restriction':flag} for flag in restrictions]}
        result=m.run_handler(label,old,opts,{})
        assert result['returned'],result['error']
        selected=sum(restrictions)<len(restrictions)
        index_events=[e for e in result['events'] if e['kind'] in ['queue_item_service','queue_remove_at_service']]
        assert len(index_events)==3*int(selected)
        assert all(e['args'][1]==sum(restrictions) for e in index_events)
        if label=='queue':
            assert result['final']['tutorials']['queued']['count']==len(restrictions)
            assert result['final']['storage']=={}
        elif selected:
            assert values(result['final']['save'])['completedTutorials']==['old-t','new-t']
            assert result['final']['tutorials']['queued']['count']==len(restrictions)-1
            assert len(result['final']['storage_calls'])==1
        else:
            assert values(result['final']['save'])==old
        cases.append(result)
    for label,opts in [('reveal',{'null_character':True}),('reveal',{'null_real':True}),('close',{'null_notes':True}),
                       ('close',{'null_note':True}),('queue',{'null_queue':True}),
                       ('queue',{'queue':[{'null':True}]}),
                       ('hide',{'notes':[{'null_stages':True,'stage':0}]}),
                       ('hide',{'notes':[{'stages':[True,True],'stage':-1}]}),
                       ('hide',{'notes':[{'stages':[True,True],'stage':0x7FFFFFFF}]})]:
        result=m.run_handler(label,old,opts,{})
        assert not result['returned'] and result['error'] in ['null_reference','bounds_failure']
        cases.append(result)
    for storage_options in [{'registry_status':5},{'create_status':5}]:
        opts={'notes':joined_notes,'queue':[{'type':20,'restriction':False}],'storage_options':storage_options}
        result=m.run_handler('close',old,opts,{})
        assert not result['returned'] and result['error']=='preference_exception'
        assert values(result['final']['save'])['completedTutorials']==['old-t','new-t']
        assert result['final']['tutorials']['queued']['count']==1
        assert result['final']['tutorials']['notes'][0]['on_hide']==0
        assert result['final']['tutorials']['notes'][1]['state']==0
        assert not any(e['kind']=='queue_remove_at_service' for e in result['events'])
        cases.append(result)
    for label,opts in [('close',{'notes':joined_notes,'queue':[{'type':20,'restriction':False}]}),
                       ('reveal',{'notes':joined_notes,'queue':[{'type':20,'restriction':False}],'real_picking':1}),
                       ('hide',{'notes':[{'stages':[True,False],'stage':0,'stop_time':True}]}),
                       ('queue',{'notes':[joined_notes[1]],'queue':[{'type':20,'restriction':False}],'show_callback':True})]:
        baseline=m.run_handler(label,old,opts,{});assert baseline['returned']
        baseline_id=len(baselines);baselines.append(baseline);counts={}
        for index,event in enumerate(baseline['events']):
            kind=event['kind'];counts[kind]=counts.get(kind,0)+1
            result=m.run_handler(label,old,dict(opts,failure=[kind,counts[kind]]),{})
            assert not result['returned'] and result['events']==baseline['events'][:index+1]
            assert result['final']==event['snapshot']
            failures.append({'baseline':baseline_id,'prefix_length':index+1,'failure':[kind,counts[kind]],'exact_snapshot_verified':True})
    return {'build_id':BUILD,'targets':m.handler_targets,'fields':m.handler_fields,
            'native_ranges':{label:[hex(row[0]),hex(row[1])] for label,row in TARGETS.items()},
            'metadata_bindings':[{'rva':hex(a),'kind':kind,'name':name} for a,(kind,name) in sorted(m.handler_bindings.items())],
            'instruction_assertions':m.handler_assertions,'cases':cases,'case_count':len(cases),
            'failure_baselines':baselines,'failure_cases':failures,'failure_case_count':len(failures),'executed_address_count':len(m.executed),
            'scope':'Actual close/reveal, appearance getter, note-hide, hidden callback and queue processing bodies run into actual ShowTutorial/Note.Show/native AddTutorial/Save/JSON/storage. Runtime list enumeration/item/remove, delegate/runtime allocation/metadata/barrier, Unity and coroutine publication are explicit services. No generator MoveNext, engine dispatch/readiness or native exception unwinding is claimed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(compact_trace(r),indent=2)+'\n',encoding='utf-8')
    print(r['case_count'],r['failure_case_count'],r['executed_address_count'])
