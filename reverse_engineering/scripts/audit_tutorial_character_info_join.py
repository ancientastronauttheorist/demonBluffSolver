"""Native CharacterInfo publication/resumes through Acted and tutorial storage."""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_storage_join import compact_trace, values
from audit_tutorial_character_generators import Machine as GeneratorMachine, StorageMachine, TARGETS as GENERATOR_TARGETS

TARGETS = {
    'handler': (0x38C170,0x38C202,'TutorialsController$$CharacterInfoNote','tdi5650.m0012',
        'void TutorialsController__CharacterInfoNote (TutorialsController_o* __this, Character_o* ch, const MethodInfo* method);','viii'),
    'move': (0x3A9210,0x3A92E1,'TutorialsController.<CharacterInfoTutorial>d__18$$MoveNext','tdi5640.m0002',
        'bool TutorialsController__CharacterInfoTutorial_d__18__MoveNext (TutorialsController__CharacterInfoTutorial_d__18_o* __this, const MethodInfo* method);','iii'),
}


class Machine(GeneratorMachine):
    def __init__(self,game_root,dumper_root):
        self.info_ready=False
        super().__init__(game_root,dumper_root)
        s=json.loads((Path(dumper_root)/'script.json').read_text(encoding='utf-8-sig'))
        dump=(Path(dumper_root)/'dump.cs').read_text(encoding='utf-8')
        self.info_fields={
            'public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487':['Acted acteds; // 0xA8'],
            'public class Acted : MonoBehaviour // TypeDefIndex: 5477':['ActedVersion acted; // 0x20','RectTransform[] layoutsToRebuild; // 0x28'],
            'private sealed class TutorialsController.<CharacterInfoTutorial>d__18 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5640':
                ['int <>1__state; // 0x10','object <>2__current; // 0x18','TutorialsController <>4__this; // 0x20','Character ch; // 0x28'],
        }
        for declaration,fields in self.info_fields.items():
            block=re.search(r'^'+re.escape(declaration)+r'\s*\{(.*?)// (?:Methods|Properties)',dump,re.M|re.S)
            assert block and all(f in block[1] for f in fields)
        slots={r['Address']:('metadata',r['Name']) for r in s['ScriptMetadata']}
        slots.update({r['Address']:('method',r['Name']) for r in s['ScriptMetadataMethod']})
        self.info_bindings,self.info_flags,self.info_targets,self.info_instructions={},{},[],{}
        for label,(a,b,name,method_id,signature,type_signature) in TARGETS.items():
            rows=[r for r in s['ScriptMethod'] if r['Address']==a and r['Name']==name];assert len(rows)==1
            assert rows[0]['Signature']==signature and rows[0]['TypeSignature']==type_signature
            family=[(e.struct.BeginAddress,e.struct.EndAddress) for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress<b and e.struct.EndAddress>a]
            assert family==[(a,b+1)]
            next_addr=min(r['Address'] for r in s['ScriptMethod'] if r['Address']>a)
            assert next_addr=={'handler':0x38C210,'move':0x3A92F0}[label]
            assert self.pe.get_data(b,next_addr-b)==bytes([0xCC])*(next_addr-b)
            decoded=list(self.cs.disasm(self.pe.get_data(a,b-a),a));assert sum(i.size for i in decoded)==b-a
            self.info_targets.append(dict(rows[0],method_id=method_id));self.info_instructions.update({i.address:i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type==self.capstone.x86.X86_OP_MEM and op.mem.base==self.capstone.x86.X86_REG_RIP:
                        slot=i.address+i.size+op.mem.disp
                        if slot in slots:self.info_bindings[slot]=slots[slot]
                        elif i.mnemonic=='cmp' and i.operands[0].size==1:self.info_flags[slot]=label
        assert {n for _,n in self.info_bindings.values()}=={'TutorialsController.<CharacterInfoTutorial>d__18_TypeInfo','UnityEngine.WaitForSeconds_TypeInfo'}
        checks={0x38C1F8:('jmp','0x1c7f160'),0x3A92A5:('mov','rcx, qword ptr [rcx + 0xa8]'),
            0x3A92B3:('call','0x1c7a010'),0x3A92CA:('call','0x38e1a0'),0x3A92C6:('lea','edx, [r9 + 0x1e]'),
            0x3A9299:('mov','dword ptr [rdi + 0x10], 0xffffffff')}
        for a,expected in checks.items():
            i=self.info_instructions[a];assert (i.mnemonic,i.op_str)==expected
        i=self.info_instructions[0x3A9257];assert i.mnemonic=='movss'
        op=i.operands[1];slot=i.address+i.size+op.mem.disp
        self.info_wait_literal={'rva':hex(slot),'bits':struct.unpack('<I',self.pe.get_data(slot,4))[0]}
        assert self.info_wait_literal['bits']==0x3D4CCCCD
        rows=[r for r in s['ScriptMethod'] if r['Address']==0x33ED50 and r['Name']=='System.Object$$.ctor']
        assert len(rows)==1
        assert rows[0]['Signature']=='void System_Object___ctor (Il2CppObject* __this, const MethodInfo* method);' and rows[0]['TypeSignature']=='vii'
        self.object_base_target=rows[0]
        leaf=list(self.cs.disasm(self.pe.get_data(0x33ED50,3),0x33ED50))
        assert [(i.address,i.size,i.mnemonic,i.op_str) for i in leaf]==[(0x33ED50,3,'ret','0')]
        for i in [self.generator_instructions[0x1C96203],self.handler_instructions[0x357711]]:
            assert (i.mnemonic,i.op_str)==('call','0x33ed50')
        self.info_assertions=len(checks)+1+2*6+5

    def setup_info(self):
        self.setup_generator()
        self.acted=self.alloc(0x80);self.u.mem_write(self.acted,bytes([0xBC])*0x80)
        self.q(self.actor+0xA8,0 if self.options.get('null_acted') else self.acted)
        self.info_actor_before=bytes(self.u.mem_read(self.actor,0x1B8))
        self.info_tokens={}
        for slot,(_,name) in self.info_bindings.items():
            token=self.generator_tokens[slot] if slot in self.generator_tokens else self.alloc(0x180)
            self.info_tokens[slot]=token;self.q(self.base+slot,token)
            if name=='TutorialsController.<CharacterInfoTutorial>d__18_TypeInfo':self.info_type=token
        for slot in self.info_flags:self.u.mem_write(self.base+slot,bytes([int(self.options.get('warm',False))]))
        self.info_routines,self.info_publications,self.info_steps={},[],[]
        self.info_selected=None;self.info_ready=True

    def info_routine_state(self,token):
        return {'identity':token,'state':self.rd(token+0x10),'current':self.rq(token+0x18),
                'controller':self.rq(token+0x20),'character':self.rq(token+0x28)}

    def snapshot(self):
        r=super().snapshot()
        if self.info_ready:
            r['character_info']={'selected':self.info_selected,'acted':self.rq(self.actor+0xA8),
                'routines':[self.info_routine_state(p) for p in self.info_routines],
                'publications':self.info_publications.copy(),'manual_steps':self.info_steps.copy(),
                'metadata_initialized':{hex(a):bool(self.u.mem_read(self.base+a,1)[0]) for a in self.info_flags}}
        return r

    def invoke(self,rva,receiver,argument=0):
        if rva==0x3EAAA0 and self.options.get('info_entry'):
            self.setup_info()
            if self.options.get('pick_queue'):
                if not StorageMachine.invoke(self,GENERATOR_TARGETS['pick_factory'][0],self.incoming_controller,self.incoming_character):return False
                self.selected_generator=self.reg(self.x.UC_X86_REG_RAX)
                for _ in range(2):
                    if not StorageMachine.invoke(self,GENERATOR_TARGETS['pick'][0],self.selected_generator):return False
                    self.manual_steps.append({'kind':'move','returned_bool':self.reg(self.x.UC_X86_REG_RAX)&255,
                        'state':self.rd(self.selected_generator+0x10),'current':self.rq(self.selected_generator+0x18)})
            if not StorageMachine.invoke(self,TARGETS['handler'][0],self.incoming_controller,self.incoming_character):return False
            assert len(self.info_publications)==1
            self.info_selected=self.info_publications[0]['identity']
            if 'initial_state' in self.options:self.d(self.info_selected+0x10,self.options['initial_state'])
            for kind in self.options.get('steps',['move','move']):
                entry,rc,rd=(TARGETS['move'][0],self.info_selected,0) if kind=='move' else (0x38C8C0,self.controller,30)
                assert kind in ['move','close']
                if not StorageMachine.invoke(self,entry,rc,rd):return False
                row={'kind':kind,**self.info_routine_state(self.info_selected)}
                if kind=='move':row['returned_bool']=self.reg(self.x.UC_X86_REG_RAX)&255
                self.info_steps.append(row)
            return True
        return StorageMachine.invoke(self,rva,receiver,argument)

    def hook(self,uc,address,size,data):
        if not self.info_ready:return super().hook(uc,address,size,data)
        rva,x=address-self.base,self.x
        cx,dx,r8=[self.reg(r) for r in [x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8]]
        self.executed.add(rva)
        if rva==0x2B7B40 and cx-self.base in self.info_bindings:
            self.service('character_info_metadata_service',[self.info_bindings[cx-self.base][1]],lambda:self.ret(self.rq(cx)))
        elif rva==0x2B7D40 and cx==self.info_type:
            def allocate():
                token=self.alloc();self.info_routines[token]=None;self.q(token,cx);self.d(token+0x10,0xDEADBEEF);self.ret(token)
            self.service('character_info_allocate_service',[cx],allocate)
        elif rva==0x1C7F160 and dx in self.info_routines:
            row=self.info_routine_state(dx)
            assert cx==self.incoming_controller and r8==0 and row['controller']==cx and row['character']==self.incoming_character
            assert row['state']==row['current']==0
            def publish():self.info_publications.append(row);self.ret(self.alloc())
            self.service('character_info_start_coroutine_service',[cx,dx,self.incoming_character],publish)
        elif rva==0x1C7A010 and cx==self.acted:
            assert dx==0
            result=0 if self.options.get('null_acted_transform') else self.pivot
            self.service('acted_transform_service',[cx,result],lambda:self.ret(result))
        else:return super().hook(uc,address,size,data)

    def run_info(self,state,options=None):
        self.info_ready=self.generator_ready=self.handler_ready=self.tutorial_ready=False
        self.manual_show_pending=False
        r=self.run_data('Save',state,{'info_entry':True,'pivot_live':True,'showed':[],'notes':[],**(options or {})},storage={})
        r['method']='CharacterInfoNote/MoveNext'
        assert bytes(self.u.mem_read(self.actor,0x1B8))==self.info_actor_before
        assert bytes(self.u.mem_read(self.acted,0x80))==bytes([0xBC])*0x80
        assert all(bytes(self.u.mem_read(a,len(before)))==before for a,before in self.retained_arrays.items())
        r['character_acted_and_stage_array_storage_retained_verified']=True
        return r


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root)
    state={'key':'Tutorials','completedTutorials':['old-t'],'unlockedCharactersId':['c']}
    cases,baselines,failures=[],[],[]
    note={'id':'info-t','type':30,'state':0,'stage':0,'stages':[True],'active':False}
    for warm,initial in itertools.product([False,True],[0,-1,7]):
        r=m.run_info(state,{'warm':warm,'initial_state':initial,'steps':['move']})
        assert r['returned']
        step=r['final']['character_info']['manual_steps'][0]
        assert step['returned_bool']==int(initial==0) and step['state']==(1 if initial==0 else initial&0xFFFFFFFF)
        if initial==0:assert r['final']['generator']['waits'][0]['seconds_bits']==0x3D4CCCCD
        assert values(r['final']['save'])==state;cases.append(r)
    for warm,null_transform in itertools.product([False,True],[False,True]):
        r=m.run_info(state,{'warm':warm,'null_acted_transform':null_transform,'notes':[note]})
        assert r['returned'] and [s['returned_bool'] for s in r['final']['character_info']['manual_steps']]==[1,0]
        assert values(r['final']['save'])['completedTutorials']==['old-t','info-t'] and len(r['final']['storage_calls'])==1
        cases.append(r)
    for option in ['null_character','null_acted','null_controller']:
        r=m.run_info(state,{option:True});assert not r['returned'] and r['error']=='null_reference'
        assert len(r['final']['character_info']['publications'])==1
        assert r['final']['character_info']['routines'][0]['state']==0xFFFFFFFF;cases.append(r)
    joined=[note,{'id':'pick-t','type':80,'state':0,'stage':0,'stages':[True],'active':False}]
    chain={'pick_queue':True,'notes':joined,'steps':['move','move','close'],'show_callback':True}
    for warm in [False,True]:
        r=m.run_info(state,dict(chain,warm=warm));assert r['returned'],r['error']
        assert values(r['final']['save'])['completedTutorials']==['old-t','info-t','pick-t']
        assert len(r['final']['storage_calls'])==2 and r['final']['tutorials']['queued']['count']==0
        queue=r['final']['tutorials']['queue_records'][0]
        assert queue['restriction']==0 and r['final']['generator']['closures'][0]['new_tutorial']==queue['identity']
        assert not r['final']['tutorials']['callback_observations']
        cases.append(r)
    for opts in [{'notes':[note]},chain]:
        baseline=m.run_info(state,opts);assert baseline['returned'];bid=len(baselines);baselines.append(baseline);counts={}
        for index,event in enumerate(baseline['events']):
            kind=event['kind'];counts[kind]=counts.get(kind,0)+1
            r=m.run_info(state,dict(opts,failure=[kind,counts[kind]]))
            assert not r['returned'] and r['events']==baseline['events'][:index+1]
            assert r['final']==event['snapshot']
            failures.append({'baseline':bid,'prefix_length':index+1,'failure':[kind,counts[kind]],'exact_snapshot_verified':True})
    return {'build_id':BUILD,'targets':m.info_targets,'fields':m.info_fields,'wait_literal':m.info_wait_literal,
        'object_base_target':m.object_base_target,'object_base_range':['0x33ed50','0x33ed53'],
        'native_ranges':{n:[hex(r[0]),hex(r[1])] for n,r in TARGETS.items()},
        'metadata_bindings':[{'rva':hex(a),'kind':k,'name':n} for a,(k,n) in sorted(m.info_bindings.items())],
        'instruction_assertions':m.info_assertions,'cases':cases,'case_count':len(cases),
        'failure_baselines':baselines,'failure_cases':failures,'failure_case_count':len(failures),'executed_address_count':len(m.executed),
        'scope':'Actual CharacterInfoNote publication, shared constructor, CharacterInfoTutorial MoveNext and WaitForSeconds constructor run. Resumes call actual Show30/save and optionally execute existing Pickable queue/callback/close/show80/save chain. Metadata/allocation/barrier/delegate/list/Unity services and publication are supplied. No engine scheduler admission, elapsed time, event dispatch/readiness or exception unwinding is inferred.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(compact_trace(r),indent=2)+'\n',encoding='utf-8')
    print(r['case_count'],r['failure_case_count'],r['executed_address_count'])
