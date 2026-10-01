"""Native death tutorial publication/resumes through status gates and saves."""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_storage_join import compact_trace, values
from audit_tutorial_character_generators import Machine as GeneratorMachine, StorageMachine

TARGETS={
    'handler':(0x38C330,0x38C42F,'TutorialsController$$CharacterKilledTutorial','tdi5650.m0006',
        'void TutorialsController__CharacterKilledTutorial (TutorialsController_o* __this, Character_o* ch, const MethodInfo* method);','viii',0x38C430),
    'kill':(0x3A9330,0x3A9436,'TutorialsController.<CharacterKillRoutine>d__11$$MoveNext','tdi5641.m0002',
        'bool TutorialsController__CharacterKillRoutine_d__11__MoveNext (TutorialsController__CharacterKillRoutine_d__11_o* __this, const MethodInfo* method);','iii',0x3A9440),
    'poison':(0x3AAD40,0x3AAE6F,'TutorialsController.<PoisonKilledRoutine>d__12$$MoveNext','tdi5646.m0002',
        'bool TutorialsController__PoisonKilledRoutine_d__12__MoveNext (TutorialsController__PoisonKilledRoutine_d__12_o* __this, const MethodInfo* method);','iii',0x3AAE70),
    'contains':(0x363C40,0x363C91,'CharacterStatuses$$Contains','tdi5488.m0003',
        'bool CharacterStatuses__Contains (CharacterStatuses_o* __this, int32_t status, const MethodInfo* method);','iiii',0x363CA0),
}
CAPTURES={'kill':(0x20,0x28),'poison':(0x28,0x20)}


class Machine(GeneratorMachine):
    def __init__(self,game_root,dumper_root):
        self.death_ready=False
        super().__init__(game_root,dumper_root)
        s=json.loads((Path(dumper_root)/'script.json').read_text(encoding='utf-8-sig'))
        dump=(Path(dumper_root)/'dump.cs').read_text(encoding='utf-8')
        self.death_fields={
            'public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487':['Transform icon; // 0x20','CharacterStatuses statuses; // 0xF0'],
            'public class CharacterStatuses // TypeDefIndex: 5488':['List<ECharacterStatus> statuses; // 0x10','List<ECharacterStatus> resistances; // 0x18','Character targetCharacter; // 0x20'],
            'public class Gameplay : MonoBehaviour // TypeDefIndex: 5604':['static EGameplayState GameplayState; // 0x28'],
            'private sealed class TutorialsController.<CharacterKillRoutine>d__11 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5641':
                ['int <>1__state; // 0x10','object <>2__current; // 0x18','TutorialsController <>4__this; // 0x20','Character ch; // 0x28'],
            'private sealed class TutorialsController.<PoisonKilledRoutine>d__12 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5646':
                ['int <>1__state; // 0x10','object <>2__current; // 0x18','Character ch; // 0x20','TutorialsController <>4__this; // 0x28'],
        }
        for declaration,fields in self.death_fields.items():
            block=re.search(r'^'+re.escape(declaration)+r'\s*\{(.*?)// (?:Methods|Properties)',dump,re.M|re.S)
            assert block and all(f in block[1] for f in fields)
        for enum,tdi,constants in [('EGameplayState',5607,[('Summary',50)]),('ECharacterStatus',5491,[('Corrupted',10)]),('ETutorialType',5635,[('Poison',45),('KilledCharacter',100)])]:
            block=re.search(r'^public enum '+enum+r' // TypeDefIndex: '+str(tdi)+r'\s*\{(.*?)^\}',dump,re.M|re.S)
            assert block and all(enum+' '+n+' = '+str(v)+';' in block[1] for n,v in constants)
        slots={r['Address']:('metadata',r['Name']) for r in s['ScriptMetadata']}
        slots.update({r['Address']:('method',r['Name']) for r in s['ScriptMetadataMethod']})
        self.death_bindings,self.death_flags,self.death_targets,self.death_instructions={},{},[],{}
        for label,(a,b,name,method_id,signature,type_signature,following) in TARGETS.items():
            rows=[r for r in s['ScriptMethod'] if r['Address']==a and r['Name']==name];assert len(rows)==1
            assert rows[0]['Signature']==signature and rows[0]['TypeSignature']==type_signature
            family=[(e.struct.BeginAddress,e.struct.EndAddress) for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress<b and e.struct.EndAddress>a]
            assert family==[(a,b+1)]
            assert min(r['Address'] for r in s['ScriptMethod'] if r['Address']>a)==following
            assert self.pe.get_data(b,following-b)==bytes([0xCC])*(following-b)
            decoded=list(self.cs.disasm(self.pe.get_data(a,b-a),a));assert sum(i.size for i in decoded)==b-a
            self.death_targets.append(dict(rows[0],method_id=method_id));self.death_instructions.update({i.address:i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type==self.capstone.x86.X86_OP_MEM and op.mem.base==self.capstone.x86.X86_REG_RIP:
                        slot=i.address+i.size+op.mem.disp
                        if slot in slots:self.death_bindings[slot]=slots[slot]
                        elif i.mnemonic=='cmp' and i.operands[0].size==1:self.death_flags[slot]=label
        self.death_wait_literals={}
        for a in [0x3A9383,0x3AAD93]:
            i=self.death_instructions[a];assert i.mnemonic=='movss'
            op=i.operands[1];slot=i.address+i.size+op.mem.disp
            bits=struct.unpack('<I',self.pe.get_data(slot,4))[0];assert bits==0x3E4CCCCD
            self.death_wait_literals[hex(a)]={'rva':hex(slot),'bits':bits}
        checks={0x3A93EE:('cmp','dword ptr [rax + 0x28], 0x32'),0x3AAE02:('cmp','dword ptr [rax + 0x28], 0x32'),
            0x3AAE11:('mov','rcx, qword ptr [rax + 0xf0]'),0x3AAE20:('lea','edx, [r8 + 0xa]'),
            0x3AAE24:('call','0x363c40'),0x363C87:('jmp','0xb45070'),
            0x3A941F:('call','0x38e1a0'),0x3AAE58:('call','0x38e1a0'),
            0x38C3EF:('lea','rcx, [rbx + 0x28]'),0x38C3FE:('lea','rcx, [rbx + 0x20]')}
        for a,expected in checks.items():
            i=self.death_instructions[a];assert (i.mnemonic,i.op_str)==expected
        rows=[r for r in s['ScriptMethod'] if r['Address']==0x33ED50 and r['Name']=='System.Object$$.ctor'];assert len(rows)==1
        assert rows[0]['Signature']=='void System_Object___ctor (Il2CppObject* __this, const MethodInfo* method);' and rows[0]['TypeSignature']=='vii'
        leaf=list(self.cs.disasm(self.pe.get_data(0x33ED50,3),0x33ED50));assert [(i.address,i.size,i.mnemonic,i.op_str) for i in leaf]==[(0x33ED50,3,'ret','0')]
        for i in [self.generator_instructions[0x1C96203],self.handler_instructions[0x357711]]:assert (i.mnemonic,i.op_str)==('call','0x33ed50')
        self.death_assertions=4*6+len(checks)+2+5

    def setup_death(self):
        self.setup_generator()
        self.statuses=self.alloc();self.status_list=self.raw_list(self.options.get('statuses',[10]),4)
        self.q(self.statuses+0x10,0 if self.options.get('null_status_list') else self.status_list)
        self.q(self.statuses+0x18,self.raw_list([10,30],4));self.q(self.statuses+0x20,self.actor)
        self.q(self.actor+0xF0,0 if self.options.get('null_statuses') else self.statuses)
        self.death_actor_before=bytes(self.u.mem_read(self.actor,0x1B8));self.statuses_before=bytes(self.u.mem_read(self.statuses,0x80))
        self.status_list_before=self.raw_list_state(self.status_list)
        self.death_tokens={};self.death_types={}
        for slot,(kind,name) in self.death_bindings.items():
            if slot in self.generator_tokens:token=self.generator_tokens[slot]
            else:token=self.alloc(0x180 if kind=='metadata' else 0x80)
            self.death_tokens[slot]=token;self.q(self.base+slot,token)
            if kind=='metadata':self.death_types[name]=token
        for slot in self.death_flags:self.u.mem_write(self.base+slot,bytes([int(self.options.get('warm',False))]))
        self.gameplay_type=self.death_types['Gameplay_TypeInfo'];self.gameplay_static=self.alloc(0x100)
        self.u.mem_write(self.gameplay_static,bytes([0xCE])*0x100)
        self.q(self.gameplay_type+0xB8,self.gameplay_static);self.d(self.gameplay_type+0xE0,int(self.options.get('class_initialized',True)))
        self.d(self.gameplay_static+0x28,self.options.get('gameplay_state',10))
        self.static_before=bytes(self.u.mem_read(self.gameplay_static,0x100))
        self.death_routines,self.death_publications,self.death_steps,self.death_initializers={},{},[],[]
        self.death_ready=True

    def death_routine_state(self,p,label):
        controller,character=CAPTURES[label]
        return {'identity':p,'kind':label,'state':self.rd(p+0x10),'current':self.rq(p+0x18),
                'controller':self.rq(p+controller),'character':self.rq(p+character),'controller_offset':controller,'character_offset':character}

    def snapshot(self):
        r=super().snapshot()
        if self.death_ready:
            r['death']={'routines':[self.death_routine_state(p,k) for p,k in self.death_routines.items()],
                'publications':list(self.death_publications.values()),'manual_steps':self.death_steps.copy(),
                'class_initialized':self.rd(self.gameplay_type+0xE0),'gameplay_state':self.rd(self.gameplay_static+0x28),
                'class_initializer_calls':self.death_initializers.copy(),'statuses':self.raw_list_state(self.status_list),
                'metadata_initialized':{hex(a):bool(self.u.mem_read(self.base+a,1)[0]) for a in self.death_flags}}
        return r

    def invoke(self,rva,receiver,argument=0):
        if rva==0x3EAAA0 and self.options.get('death_entry'):
            self.setup_death()
            if not StorageMachine.invoke(self,TARGETS['handler'][0],self.incoming_controller,self.incoming_character):return False
            assert list(self.death_publications)==['kill','poison']
            for step in self.options.get('steps',['kill','poison','kill','poison']):
                if step in CAPTURES:
                    token=self.death_publications[step]['identity'];entry,rc,rd=TARGETS[step][0],token,0
                else:
                    assert step in ['close_kill','close_poison'];entry,rc,rd=0x38C8C0,self.controller,100 if step=='close_kill' else 45
                if not StorageMachine.invoke(self,entry,rc,rd):return False
                row={'kind':step}
                if step in CAPTURES:row.update(self.death_routine_state(token,step),returned_bool=self.reg(self.x.UC_X86_REG_RAX)&255)
                self.death_steps.append(row)
            return True
        return StorageMachine.invoke(self,rva,receiver,argument)

    def hook(self,uc,address,size,data):
        if not self.death_ready:return super().hook(uc,address,size,data)
        rva,x=address-self.base,self.x
        cx,dx,r8=[self.reg(r) for r in [x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8]]
        self.executed.add(rva)
        if rva==0x2B7B40 and cx-self.base in self.death_bindings:
            self.service('death_metadata_service',[self.death_bindings[cx-self.base][1]],lambda:self.ret(self.rq(cx)))
        elif rva==0x2B7D40 and cx in [self.death_types['TutorialsController.<CharacterKillRoutine>d__11_TypeInfo'],self.death_types['TutorialsController.<PoisonKilledRoutine>d__12_TypeInfo']]:
            kind='kill' if cx==self.death_types['TutorialsController.<CharacterKillRoutine>d__11_TypeInfo'] else 'poison'
            def allocate():
                token=self.alloc();self.death_routines[token]=kind;self.q(token,cx);self.d(token+0x10,0xDEADBEEF);self.ret(token)
            self.service('death_routine_allocate_service',[kind],allocate)
        elif rva==0x1C7F160 and dx in self.death_routines:
            kind=self.death_routines[dx];row=self.death_routine_state(dx,kind)
            assert cx==self.incoming_controller and r8==0 and row['state']==row['current']==0
            assert row['controller']==cx and row['character']==self.incoming_character
            def publish():self.death_publications[kind]=row;self.ret(self.alloc())
            self.service('death_start_coroutine_service',[cx,dx,kind],publish)
        elif rva==0x281D90:
            assert cx==self.gameplay_type
            def initialize():
                self.death_initializers.append(cx);self.d(cx+0xE0,1)
                if 'initializer_gameplay_state' in self.options:self.d(self.gameplay_static+0x28,self.options['initializer_gameplay_state'])
                self.ret()
            self.service('death_gameplay_class_initialize_service',[cx],initialize)
        elif rva==0xB45070 and cx==self.status_list:
            method=next(self.death_tokens[a] for a,(_,n) in self.death_bindings.items() if n=='Method$System.Collections.Generic.List<ECharacterStatus>.Contains()')
            assert dx==10 and r8==method
            yes=dx in self.raw_list_state(cx)['values']
            self.service('death_status_contains_service',[cx,dx,r8,yes],lambda:self.ret(0xFACE000000000000|int(yes)))
        else:return super().hook(uc,address,size,data)

    def run_death(self,state,options=None):
        self.death_ready=self.generator_ready=self.handler_ready=self.tutorial_ready=False;self.manual_show_pending=False
        r=self.run_data('Save',state,{'death_entry':True,'pivot_live':True,'notes':[],**(options or {})},storage={})
        r['method']='CharacterKilledTutorial/death MoveNext'
        assert bytes(self.u.mem_read(self.actor,0x1B8))==self.death_actor_before
        assert bytes(self.u.mem_read(self.statuses,0x80))==self.statuses_before and self.raw_list_state(self.status_list)==self.status_list_before
        assert all(a==b for i,(a,b) in enumerate(zip(self.static_before,self.u.mem_read(self.gameplay_static,0x100))) if i not in range(0x28,0x2C))
        assert all(bytes(self.u.mem_read(a,len(before)))==before for a,before in self.retained_arrays.items())
        r['character_status_and_unconsumed_static_storage_retained_verified']=True
        return r


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);state={'key':'Tutorials','completedTutorials':['old-t'],'unlockedCharactersId':['c']}
    cases,baselines,failures=[],[],[]
    for warm,initialized,game_state,statuses in itertools.product([False,True],[False,True],[10,50,60],[[],[10],[15,10,10],[55]]):
        r=m.run_death(state,dict(warm=warm,class_initialized=initialized,gameplay_state=game_state,statuses=statuses));assert r['returned']
        assert [s['returned_bool'] for s in r['final']['death']['manual_steps']]==[1,1,0,0]
        assert all(w['seconds_bits']==0x3E4CCCCD for w in r['final']['generator']['waits'])
        assert len(r['final']['death']['class_initializer_calls'])==int(not initialized)
        assert values(r['final']['save'])==state;cases.append(r)
    notes=[{'id':'kill-t','type':100,'state':0,'stage':0,'stages':[True],'active':False},
           {'id':'poison-t','type':45,'state':0,'stage':0,'stages':[True],'active':False}]
    for steps,expected in [(['kill','poison','kill','poison','close_kill'],['kill-t','poison-t']),
                           (['poison','kill','poison','kill','close_poison'],['poison-t','kill-t'])]:
        r=m.run_death(state,{'notes':notes,'steps':steps,'show_callback':True});assert r['returned'],r['error']
        assert values(r['final']['save'])['completedTutorials']==['old-t']+expected
        assert len(r['final']['storage_calls'])==2 and r['final']['tutorials']['queued']['count']==0;cases.append(r)
    for kind,option in itertools.product(['kill','poison'],['null_character','null_icon','null_controller','null_statuses','null_status_list']):
        r=m.run_death(state,{option:True,'steps':[kind,kind]})
        expects_error=kind=='poison' or option not in ['null_statuses','null_status_list']
        assert r['returned']==(not expects_error)
        if expects_error:assert r['error']=='null_reference'
        cases.append(r)
    for opts in [{'null_character':True,'gameplay_state':50},{'class_initialized':False,'initializer_gameplay_state':50,'null_character':True}]:
        r=m.run_death(state,opts);assert r['returned'] and not any(e['kind']=='character_icon_transform_service' for e in r['events']);cases.append(r)
    for kind in ['kill','poison']:
        r=m.run_death(state,{'steps':[kind,kind,kind],'notes':[]});assert r['returned']
        assert [s['returned_bool'] for s in r['final']['death']['manual_steps']]==[1,0,0];cases.append(r)
    for opts in [{'notes':[notes[0]],'steps':['kill','kill'],'class_initialized':False},
                 {'notes':[notes[1]],'steps':['poison','poison']},
                 {'notes':notes,'steps':['kill','poison','kill','poison','close_kill'],'show_callback':True}]:
        baseline=m.run_death(state,opts);assert baseline['returned'];bid=len(baselines);baselines.append(baseline);counts={}
        for index,event in enumerate(baseline['events']):
            kind=event['kind'];counts[kind]=counts.get(kind,0)+1
            r=m.run_death(state,dict(opts,failure=[kind,counts[kind]]))
            assert not r['returned'] and r['events']==baseline['events'][:index+1] and r['final']==event['snapshot']
            failures.append({'baseline':bid,'prefix_length':index+1,'failure':[kind,counts[kind]],'exact_snapshot_verified':True})
    return {'build_id':BUILD,'targets':m.death_targets,'fields':m.death_fields,'wait_literals':m.death_wait_literals,
        'native_ranges':{n:[hex(r[0]),hex(r[1])] for n,r in TARGETS.items()},
        'metadata_bindings':[{'rva':hex(a),'kind':k,'name':n} for a,(k,n) in sorted(m.death_bindings.items())],
        'instruction_assertions':m.death_assertions,'cases':cases,'case_count':len(cases),'failure_baselines':baselines,
        'failure_cases':failures,'failure_case_count':len(failures),'executed_address_count':len(m.executed),
        'scope':'Actual CharacterKilledTutorial publication, both generated MoveNext bodies, state/WaitForSeconds/Object constructors, native status Contains wrapper and native Show/Close/Hide/hidden/queue/save/JSON/storage callers execute under explicit manual resumes. Gameplay class initialization, List membership and Unity/runtime operations are named supplied services. No real scheduler admission, elapsed time, event readiness/dispatch or exception unwinding is claimed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(compact_trace(r),indent=2)+'\n',encoding='utf-8')
    print(r['case_count'],r['failure_case_count'],r['executed_address_count'])
