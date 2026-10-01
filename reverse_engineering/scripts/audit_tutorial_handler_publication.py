"""Execute four native tutorial handlers through explicit coroutine publication."""
import argparse
import itertools
import json
import re
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_storage_join import compact_trace, values
from audit_tutorial_persistence_join import Machine as TutorialMachine, StorageMachine

TARGETS = {
    'info': (0x38C170, 0x38C202, 'CharacterInfoNote', 'tdi5650.m0012'),
    'killed': (0x38C330, 0x38C42F, 'CharacterKilledTutorial', 'tdi5650.m0006'),
    'level': (0x38CAD0, 0x38CC2A, 'LevelIdTutorial', 'tdi5650.m0003'),
    'start': (0x38E3D0, 0x38E560, 'StartTutorials', 'tdi5650.m0011'),
}
# Verified generated declarations, including the reversed poison captures.
GENERATORS = {
    'CharacterInfoTutorial>d__18': (5640, 0x20, 0x28),
    'CharacterKillRoutine>d__11': (5641, 0x20, 0x28),
    'PoisonKilledRoutine>d__12': (5646, 0x28, 0x20),
    'RevealCardTutorial>d__19': (5647, 0x20, None),
    'KillTutorial>d__20': (5644, 0x20, None),
    'HoverObjective>d__21': (5643, 0x20, None),
    'CancelKillTutorial>d__22': (5639, 0x20, None),
    'SecondLevelNoteCoroutine>d__8': (5648, 0x20, None),
    'ThirdLevelNoteCoroutine>d__9': (5649, 0x20, None),
}


class Machine(TutorialMachine):
    def __init__(self, game_root, dumper_root):
        self.publication_ready = False
        super().__init__(game_root, dumper_root)
        script = json.loads((Path(dumper_root)/'script.json').read_text(encoding='utf-8-sig'))
        dump = (Path(dumper_root)/'dump.cs').read_text(encoding='utf-8')
        self.capture_layouts, self.publication_fields = {}, {}
        for suffix, (tdi, controller, character) in GENERATORS.items():
            name = 'TutorialsController.<'+suffix
            declaration = 'private sealed class '+name+' : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: '+str(tdi)
            fields = ['int <>1__state; // 0x10', 'object <>2__current; // 0x18', 'TutorialsController <>4__this; // '+hex(controller).upper().replace('0X','0x')]
            if character is not None:
                fields.append('Character ch; // '+hex(character).upper().replace('0X','0x'))
            if suffix in ['KillTutorial>d__20', 'HoverObjective>d__21']:
                fields.append('TutorialsController.<>c__DisplayClass'+('20' if suffix.startswith('Kill') else '21')+'_0 <>8__1; // 0x28')
            block = re.search(r'^'+re.escape(declaration)+r'\s*\{(.*?)// Properties', dump, re.M|re.S)
            assert block and all(f in block[1] for f in fields), declaration
            self.publication_fields[declaration] = fields
            self.capture_layouts[name+'_TypeInfo'] = (controller, character)
        declaration = 'public class Gameplay : MonoBehaviour // TypeDefIndex: 5604'
        fields = ['static Gameplay Instance; // 0x10', 'int currentLevel; // 0x78']
        block = re.search(r'^'+re.escape(declaration)+r'\s*\{(.*?)// Methods', dump, re.M|re.S)
        assert block and all(f in block[1] for f in fields)
        self.publication_fields[declaration] = fields
        slots = {r['Address']: ('metadata', r['Name']) for r in script['ScriptMetadata']}
        slots.update({r['Address']: ('method', r['Name']) for r in script['ScriptMetadataMethod']})
        self.publication_bindings, self.publication_flags, self.publication_targets = {}, {}, []
        self.publication_instructions = {}
        for label, (a,b,name,method_id) in TARGETS.items():
            rows = [r for r in script['ScriptMethod'] if r['Address']==a and r['Name']=='TutorialsController$$'+name]
            assert len(rows)==1
            args = 'TutorialsController_o* __this, '+('Character_o* ch, ' if label in ['info','killed'] else '')+'const MethodInfo* method'
            assert rows[0]['Signature']=='void TutorialsController__'+name+' ('+args+');'
            assert rows[0]['TypeSignature']==('viii' if label in ['info','killed'] else 'vii')
            family=[e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress<b and e.struct.EndAddress>a]
            assert [(e.struct.BeginAddress,e.struct.EndAddress) for e in family]==[(a,b+1)]
            following=min(r['Address'] for r in script['ScriptMethod'] if r['Address']>a)
            assert following=={'info':0x38C210,'killed':0x38C430,'level':0x38CC30,'start':0x38E570}[label]
            assert self.pe.get_data(b,following-b)==bytes([0xCC])*(following-b)
            self.publication_targets.append(dict(rows[0], method_id=method_id))
            decoded = list(self.cs.disasm(self.pe.get_data(a,b-a),a))
            assert sum(i.size for i in decoded)==b-a
            self.publication_instructions.update({i.address:i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type==self.capstone.x86.X86_OP_MEM and op.mem.base==self.capstone.x86.X86_REG_RIP:
                        slot=i.address+i.size+op.mem.disp
                        if slot in slots:
                            self.publication_bindings[slot]=slots[slot]
                        elif i.mnemonic=='cmp' and i.operands[0].size==1:
                            self.publication_flags[slot]=label
        assert {n for _,n in self.publication_bindings.values()}==set(self.capture_layouts)|{'Gameplay_TypeInfo'}
        ctor = next(e for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress==0x357700)
        assert ctor.struct.EndAddress==0x357724
        decoded=list(self.cs.disasm(self.pe.get_data(0x357700,0x24),0x357700))
        self.publication_instructions.update({i.address:i for i in decoded})
        checks = {
            0x357716: ('mov','dword ptr [rbx + 0x10], edi'),
            0x38C3EF: ('lea','rcx, [rbx + 0x28]'), 0x38C3FE: ('lea','rcx, [rbx + 0x20]'),
            0x38CB2A: ('cmp','dword ptr [rdx + 0x78], 1'), 0x38CBBA: ('cmp','dword ptr [rcx + 0x78], 2'),
            0x38CB8E: ('mov','rcx, qword ptr [rip + 0x236b5ab]'),
            0x38C1F8: ('jmp','0x1c7f160'), 0x38C425: ('jmp','0x1c7f160'),
            0x38E556: ('jmp','0x1c7f160'),
        }
        for a, expected in checks.items():
            i=self.publication_instructions[a];assert (i.mnemonic,i.op_str)==expected
        for label, expected in [('info',1),('killed',2),('start',4),('level',2)]:
            a,b,*_=TARGETS[label]
            body=[i for x,i in self.publication_instructions.items() if a<=x<b]
            assert sum(i.mnemonic=='call' and i.op_str=='0x357700' for i in body)==expected
            assert sum(i.mnemonic in ['call','jmp'] and i.op_str=='0x1c7f160' for i in body)==expected
        self.publication_assertions=len(checks)+8+16

    def setup_publication(self):
        self.setup_tutorials();self.entry_pending=False
        self.incoming_controller=0 if self.options.get('null_controller') else self.controller
        self.character=self.alloc(0x200)
        self.u.mem_write(self.character,bytes([0xA5])*0x200)
        self.incoming_character=0 if self.options.get('null_character') else self.character
        self.publication_types={name:self.alloc(0x180) for _,name in self.publication_bindings.values()}
        for slot,(_,name) in self.publication_bindings.items():
            self.q(self.base+slot,self.publication_types[name])
        for slot in self.publication_flags:
            self.u.mem_write(self.base+slot,bytes([int(self.options.get('warm',False))]))
        self.gameplay_type=self.publication_types['Gameplay_TypeInfo']
        self.gameplay_static=self.alloc(0x100);self.gameplay=self.alloc(0x100)
        self.u.mem_write(self.gameplay_static,bytes([0xCE])*0x100)
        self.u.mem_write(self.gameplay,bytes([0xBA])*0x100)
        self.q(self.gameplay_type+0xB8,self.gameplay_static)
        self.d(self.gameplay_type+0xE0,int(self.options.get('class_initialized',True)))
        self.q(self.gameplay_static+0x10,0 if self.options.get('null_gameplay') else self.gameplay)
        self.d(self.gameplay+0x78,self.options.get('level',0))
        self.initial_gameplay=bytes(self.u.mem_read(self.gameplay,0x100))
        self.initial_static=bytes(self.u.mem_read(self.gameplay_static,0x100))
        self.initial_controller=bytes(self.u.mem_read(self.controller,0x40))
        self.publication_routines={};self.publication_calls=[];self.runtime_init_calls=[]
        self.publication_ready=True

    def routine_state(self, token, kind):
        controller,character=self.capture_layouts[kind]
        return {'identity':token,'kind':kind,'state':self.rd(token+0x10),
                'current':self.rq(token+0x18),'controller':self.rq(token+controller),
                'character':self.rq(token+character) if character is not None else None,
                'controller_offset':controller,'character_offset':character,
                'closure':self.rq(token+0x28) if kind in ['TutorialsController.<KillTutorial>d__20_TypeInfo','TutorialsController.<HoverObjective>d__21_TypeInfo'] else None}

    def snapshot(self):
        r=super().snapshot()
        if self.publication_ready:
            r['publication']={'incoming_controller':self.incoming_controller,'incoming_character':self.incoming_character,
                'routines':[self.routine_state(p,k) for p,k in self.publication_routines.items()],
                'calls':self.publication_calls.copy(),'runtime_init_calls':self.runtime_init_calls.copy(),
                'gameplay_instance':self.rq(self.gameplay_static+0x10),'current_level':self.rd(self.gameplay+0x78),
                'class_initialized':self.rd(self.gameplay_type+0xE0),
                'metadata_initialized':{hex(a):bool(self.u.mem_read(self.base+a,1)[0]) for a in self.publication_flags}}
        return r

    def invoke(self,rva,receiver,argument=0):
        if rva==0x3EAAA0 and self.options.get('publication_entry'):
            self.setup_publication();label=self.options['publication_entry']
            return StorageMachine.invoke(self,TARGETS[label][0],self.incoming_controller,
                                         self.incoming_character if label in ['info','killed'] else 0)
        return StorageMachine.invoke(self,rva,receiver,argument)

    def hook(self,uc,address,size,data):
        if not self.publication_ready:
            return super().hook(uc,address,size,data)
        rva,x=address-self.base,self.x
        cx,dx,r8=[self.reg(r) for r in [x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8]]
        self.executed.add(rva)
        if rva==0x2B7B40 and cx-self.base in self.publication_bindings:
            self.service('publication_metadata_service',[self.publication_bindings[cx-self.base][1]],lambda:self.ret(self.rq(cx)))
        elif rva==0x2B7D40 and cx in self.publication_types.values():
            kind=next(n for n,t in self.publication_types.items() if t==cx);assert kind in self.capture_layouts
            def allocate():
                token=self.alloc();self.publication_routines[token]=kind;self.q(token,cx);self.ret(token)
            self.service('publication_allocate_service',[kind],allocate)
        elif rva==0x281D90:
            assert cx==self.gameplay_type
            def initialize():
                self.runtime_init_calls.append(cx);self.d(cx+0xE0,1)
                if 'init_level' in self.options:self.d(self.gameplay+0x78,self.options['init_level'])
                self.ret()
            self.service('gameplay_class_initialize_service',[cx],initialize)
        elif rva==0x1C7F160 and dx in self.publication_routines:
            assert cx==self.incoming_controller and r8==0
            row=self.routine_state(dx,self.publication_routines[dx])
            assert row['state']==row['current']==0 and row['controller']==cx
            assert row['character'] in [None,self.incoming_character] and row['closure'] in [None,0]
            def publish():
                self.publication_calls.append(row)
                actions=self.options.get('publication_actions',[])
                if len(self.publication_calls)<=len(actions):
                    action=actions[len(self.publication_calls)-1]
                    if 'level' in action:self.d(self.gameplay+0x78,action['level'])
                    if 'instance_null' in action:self.q(self.gameplay_static+0x10,0 if action['instance_null'] else self.gameplay)
                    if 'class_initialized' in action:self.d(self.gameplay_type+0xE0,int(action['class_initialized']))
                self.ret(self.alloc())
            self.service('publication_start_coroutine_service',[cx,dx,row['kind']],publish)
        else:
            return super().hook(uc,address,size,data)

    def run_publication(self,label,state,options=None):
        self.publication_ready=self.tutorial_ready=False
        r=self.run_data('Save',state,{'publication_entry':label,'notes':[],**(options or {})},storage={})
        r['method']=label
        assert bytes(self.u.mem_read(self.character,0x200))==bytes([0xA5])*0x200
        assert bytes(self.u.mem_read(self.controller,0x40))==self.initial_controller
        assert all(a==b for i,(a,b) in enumerate(zip(self.initial_gameplay,self.u.mem_read(self.gameplay,0x100))) if i not in range(0x78,0x7C))
        assert all(a==b for i,(a,b) in enumerate(zip(self.initial_static,self.u.mem_read(self.gameplay_static,0x100))) if i not in range(0x10,0x18))
        assert values(r['final']['save'])==state and r['final']['storage']=={} and not r['final']['storage_calls']
        assert not r['final']['tutorials']['coroutines']
        r['unconsumed_input_storage_retained_verified']=True
        return r


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root)
    state={'key':'Tutorials','completedTutorials':['old-t'],'unlockedCharactersId':['c']}
    cases,baselines,failures=[],[],[]
    for label,warm,null_controller,null_character in itertools.product(['start','info','killed'],[False,True],[False,True],[False,True]):
        r=m.run_publication(label,state,dict(warm=warm,null_controller=null_controller,null_character=null_character))
        assert r['returned']
        expected={'start':['RevealCardTutorial>d__19','KillTutorial>d__20','HoverObjective>d__21','CancelKillTutorial>d__22'],
                  'info':['CharacterInfoTutorial>d__18'],'killed':['CharacterKillRoutine>d__11','PoisonKilledRoutine>d__12']}[label]
        assert [c['kind'] for c in r['final']['publication']['calls']]==['TutorialsController.<'+s+'_TypeInfo' for s in expected]
        cases.append(r)
    for warm,initialized,level in itertools.product([False,True],[False,True],[-1,0,1,2,3,0x7FFFFFFF]):
        r=m.run_publication('level',state,dict(warm=warm,class_initialized=initialized,level=level))
        assert r['returned']
        suffix='SecondLevelNoteCoroutine>d__8' if level==1 else 'ThirdLevelNoteCoroutine>d__9' if level==2 else None
        assert [c['kind'] for c in r['final']['publication']['calls']]==(['TutorialsController.<'+suffix+'_TypeInfo'] if suffix else [])
        assert len(r['final']['publication']['runtime_init_calls'])==int(not initialized)
        cases.append(r)
    for action in [{'level':2},{'level':3},{'instance_null':True},{'level':2,'class_initialized':False}]:
        r=m.run_publication('level',state,{'level':1,'publication_actions':[action]})
        assert r['returned']==(not action.get('instance_null',False))
        assert len(r['final']['publication']['calls'])==1+int(action.get('level')==2)
        if not r['returned']:assert r['error']=='null_reference'
        cases.append(r)
    for initialized in [False,True]:
        r=m.run_publication('level',state,{'null_gameplay':True,'class_initialized':initialized})
        assert not r['returned'] and r['error']=='null_reference' and not r['final']['publication']['calls']
        cases.append(r)
    r=m.run_publication('level',state,{'class_initialized':False,'level':0,'init_level':2})
    assert r['returned'] and len(r['final']['publication']['calls'])==1;cases.append(r)
    for label,opts in [('start',{}),('info',{}),('killed',{}),('level',{'level':1,'class_initialized':False,'publication_actions':[{'level':2,'class_initialized':False}]})]:
        baseline=m.run_publication(label,state,opts);assert baseline['returned'];bid=len(baselines);baselines.append(baseline);counts={}
        for index,event in enumerate(baseline['events']):
            kind=event['kind'];counts[kind]=counts.get(kind,0)+1
            r=m.run_publication(label,state,dict(opts,failure=[kind,counts[kind]]))
            assert not r['returned'] and r['events']==baseline['events'][:index+1]
            assert r['final']==event['snapshot']
            failures.append({'baseline':bid,'prefix_length':index+1,'failure':[kind,counts[kind]],'exact_snapshot_verified':True})
    return {'build_id':BUILD,'targets':m.publication_targets,'fields':m.publication_fields,
            'native_ranges':{label:[hex(row[0]),hex(row[1])] for label,row in TARGETS.items()},
            'metadata_bindings':[{'rva':hex(a),'kind':k,'name':n} for a,(k,n) in sorted(m.publication_bindings.items())],
            'instruction_assertions':m.publication_assertions,'cases':cases,'case_count':len(cases),
            'failure_baselines':baselines,'failure_cases':failures,'failure_case_count':len(failures),'executed_address_count':len(m.executed),
            'scope':'Actual four tutorial handlers and shared generator constructor run. Runtime allocation, metadata/class initialization, write barriers and StartCoroutine publication with explicitly authored optional state mutations are supplied services. No MoveNext, Unity event dispatch/readiness/scheduling, engine class initializer or persistence invocation is inferred.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(compact_trace(r),indent=2)+'\n',encoding='utf-8')
    print(r['case_count'],r['failure_case_count'],r['executed_address_count'])
