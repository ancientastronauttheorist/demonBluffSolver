"""Native Character tutorial generator resumes through UI/queue/save callers."""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_saved_game_storage_join import compact_trace, values
from audit_tutorial_close_reveal_join import Machine as CloseMachine, StorageMachine

TARGETS = {
    'hover_factory': (0x38C960,0x38C9E8,'TutorialsController$$HoverCharacterRoutine','tdi5650.m0022',
        'System_Collections_IEnumerator_o* TutorialsController__HoverCharacterRoutine (TutorialsController_o* __this, Character_o* ch, const MethodInfo* method);','iiii'),
    'pick_factory': (0x38DC90,0x38DD18,'TutorialsController$$PickableCharacterType','tdi5650.m0021',
        'System_Collections_IEnumerator_o* TutorialsController__PickableCharacterType (TutorialsController_o* __this, Character_o* ch, const MethodInfo* method);','iiii'),
    'hover': (0x3AA0D0,0x3AA19E,'TutorialsController.<HoverCharacterRoutine>d__26$$MoveNext','tdi5642.m0002',
        'bool TutorialsController__HoverCharacterRoutine_d__26__MoveNext (TutorialsController__HoverCharacterRoutine_d__26_o* __this, const MethodInfo* method);','iii'),
    'pick': (0x3AAA50,0x3AACF2,'TutorialsController.<PickableCharacterType>d__25$$MoveNext','tdi5645.m0002',
        'bool TutorialsController__PickableCharacterType_d__25__MoveNext (TutorialsController__PickableCharacterType_d__25_o* __this, const MethodInfo* method);','iii'),
    'closure': (0x3AD4F0,0x3AD510,'TutorialsController.<>c__DisplayClass25_0$$<PickableCharacterType>b__0','tdi5638.m0001',
        'void TutorialsController___c__DisplayClass25_0___PickableCharacterType_b__0 (TutorialsController___c__DisplayClass25_0_o* __this, int32_t tutType, const MethodInfo* method);','viii'),
    'wait_ctor': (0x1C961F0,0x1C96218,'UnityEngine.WaitForSeconds$$.ctor','tdi6799.m0000',
        'void UnityEngine_WaitForSeconds___ctor (UnityEngine_WaitForSeconds_o* __this, float seconds, const MethodInfo* method);','vifi'),
}
FAMILIES = {
    'hover_factory':[(0x38C960,0x38C9E9)],'pick_factory':[(0x38DC90,0x38DD19)],
    'hover':[(0x3AA0D0,0x3AA19F)],
    'pick':[(0x3AAA50,0x3AAAD5),(0x3AAAD5,0x3AAC60),(0x3AAC60,0x3AAC9C),(0x3AAC9C,0x3AACED),(0x3AACED,0x3AACF3)],
    'closure':[(0x3AD4F0,0x3AD511)],'wait_ctor':[(0x1C961F0,0x1C96218)],
}


class Machine(CloseMachine):
    def __init__(self,game_root,dumper_root):
        self.generator_ready=False
        super().__init__(game_root,dumper_root)
        s=json.loads((Path(dumper_root)/'script.json').read_text(encoding='utf-8-sig'))
        dump=(Path(dumper_root)/'dump.cs').read_text(encoding='utf-8')
        self.generator_fields={
            'public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487':['Transform icon; // 0x20'],
            'private sealed class TutorialsController.<>c__DisplayClass25_0 // TypeDefIndex: 5638':['TutorialQueue newTut; // 0x10'],
            'public sealed class WaitForSeconds : YieldInstruction // TypeDefIndex: 6799':['float m_Seconds; // 0x10'],
        }
        for declaration,fields in self.generator_fields.items():
            block=re.search(r'^'+re.escape(declaration)+r'\s*\{(.*?)// (?:Methods|Properties)',dump,re.M|re.S)
            assert block and all(f in block[1] for f in fields)
        enum=re.search(r'^public enum ETutorialType // TypeDefIndex: 5635\s*\{(.*?)^\}',dump,re.M|re.S)
        assert enum and all('ETutorialType '+n+' = '+str(v)+';' in enum[1] for n,v in [('CharacterInfo',30),('HoverCharacter',40),('PickableCharacter',80)])
        slots={r['Address']:('metadata',r['Name']) for r in s['ScriptMetadata']}
        slots.update({r['Address']:('method',r['Name']) for r in s['ScriptMetadataMethod']})
        self.generator_bindings,self.generator_flags,self.generator_targets,self.generator_instructions={},{},[],{}
        for label,(a,b,name,method_id,signature,type_signature) in TARGETS.items():
            rows=[r for r in s['ScriptMethod'] if r['Address']==a and r['Name']==name];assert len(rows)==1
            assert rows[0]['Signature']==signature and rows[0]['TypeSignature']==type_signature
            self.generator_targets.append(dict(rows[0],method_id=method_id))
            family=[(e.struct.BeginAddress,e.struct.EndAddress) for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress<b and e.struct.EndAddress>a]
            assert family==FAMILIES[label]
            next_addr=min(r['Address'] for r in s['ScriptMethod'] if r['Address']>a)
            if label!='wait_ctor':assert self.pe.get_data(b,next_addr-b)==bytes([0xCC])*(next_addr-b)
            decoded=list(self.cs.disasm(self.pe.get_data(a,b-a),a));assert sum(i.size for i in decoded)==b-a
            self.generator_instructions.update({i.address:i for i in decoded})
            for i in decoded:
                for op in i.operands:
                    if op.type==self.capstone.x86.X86_OP_MEM and op.mem.base==self.capstone.x86.X86_REG_RIP:
                        slot=i.address+i.size+op.mem.disp
                        if slot in slots:self.generator_bindings[slot]=slots[slot]
                        elif i.mnemonic=='cmp' and i.operands[0].size==1:self.generator_flags[slot]=label
        self.wait_literals={}
        for a,expected in [(0x3AA117,0x3F000000),(0x3AAC6C,0x3F000000),(0x3AACAF,0x3D4CCCCD)]:
            i=self.generator_instructions[a];assert i.mnemonic=='movss'
            op=i.operands[1];slot=i.address+i.size+op.mem.disp
            bits=struct.unpack('<I',self.pe.get_data(slot,4))[0];assert bits==expected
            self.wait_literals[hex(a)]={'rva':hex(slot),'bits':bits}
        checks={0x3AA187:('call','0x38e1a0'),0x3AAB2B:('call','0x38e1a0'),
            0x3AAB5E:('call','0xb45070'),0x3AAC1F:('mov','qword ptr [rcx], rbx'),
            0x3AAC3F:('call','0x2eb0'),0x3AABD7:('mov','byte ptr [rdi + 0x20], 1'),
            0x3AD4F4:('cmp','edx, 0x1e'),0x3AD502:('mov','byte ptr [rax + 0x20], 0'),
            0x1C96208:('movss','dword ptr [rbx + 0x10], xmm6')}
        for a,expected in checks.items():
            i=self.generator_instructions[a];assert (i.mnemonic,i.op_str)==expected
        self.generator_assertions=6*4+3+len(checks)

    def setup_generator(self):
        self.setup_handler()
        self.icon=self.alloc();self.q(self.actor+0x20,0 if self.options.get('null_icon') else self.icon)
        self.actor_before=bytes(self.u.mem_read(self.actor,0x1B8))
        self.incoming_controller=0 if self.options.get('null_controller') else self.controller
        self.incoming_character=0 if self.options.get('null_character') else self.actor
        self.prior_show=self.rq(self.controller+0x38)
        if not self.options.get('null_showed'):
            self.q(self.controller+0x30,self.raw_list(self.options.get('showed',[30]),4))
        self.generator_tokens,self.generator_types={},{}
        for slot,(kind,name) in self.generator_bindings.items():
            if slot in self.tutorial_tokens:token=self.tutorial_tokens[slot]
            elif slot in self.handler_tokens:token=self.handler_tokens[slot]
            else:token=self.alloc(0x180 if kind=='metadata' else 0x80)
            self.generator_tokens[slot]=token;self.q(self.base+slot,token)
            if kind=='metadata':self.generator_types[name]=token
        for slot in self.generator_flags:self.u.mem_write(self.base+slot,bytes([int(self.options.get('warm',False))]))
        self.waits,self.closures,self.show_delegates,self.manual_steps={},{},{},[]
        self.selected_generator=None
        self.generator_ready=True

    def generator_method(self,name):
        return next(self.generator_tokens[a] for a,(_,n) in self.generator_bindings.items() if n==name)

    def snapshot(self):
        r=super().snapshot()
        if self.generator_ready:
            r['generator']={'selected':self.selected_generator,'prior_on_tutorial_show':self.prior_show,
                'on_tutorial_show':self.rq(self.controller+0x38),'manual_steps':self.manual_steps.copy(),
                'waits':[{'identity':p,'seconds_bits':self.rd(p+0x10)} for p in self.waits],
                'closures':[{'identity':p,'new_tutorial':self.rq(p+0x10)} for p in self.closures],
                'show_delegates':self.show_delegates.copy(),
                'metadata_initialized':{hex(a):bool(self.u.mem_read(self.base+a,1)[0]) for a in self.generator_flags}}
        return r

    def invoke(self,rva,receiver,argument=0):
        if rva==0x3EAAA0 and self.options.get('generator_entry'):
            self.setup_generator();label=self.options['generator_entry']
            if not StorageMachine.invoke(self,TARGETS[label+'_factory'][0],self.incoming_controller,self.incoming_character):return False
            self.selected_generator=self.reg(self.x.UC_X86_REG_RAX)
            assert self.selected_generator in self.handler_routines
            if 'initial_state' in self.options:self.d(self.selected_generator+0x10,self.options['initial_state'])
            for step in self.options.get('steps',[{'kind':'move'}]* (2 if label=='hover' else 3)):
                kind=step['kind']
                if kind=='move':entry,rc,rd=TARGETS[label][0],self.selected_generator,0
                elif kind=='show':entry,rc,rd=0x38E1A0,self.controller,step['type']
                elif kind=='close':entry,rc,rd=0x38C8C0,self.controller,step['type']
                elif kind=='closure':entry,rc,rd=TARGETS['closure'][0],next(iter(self.closures)),step['type']
                else:assert False,kind
                # Show's third argument is a real supplied Transform token.
                self.manual_show_pending=kind=='show'
                if not StorageMachine.invoke(self,entry,rc,rd):return False
                row={'kind':kind,'state':self.rd(self.selected_generator+0x10),'current':self.rq(self.selected_generator+0x18)}
                if kind=='move':row['returned_bool']=self.reg(self.x.UC_X86_REG_RAX)&255
                self.manual_steps.append(row)
            return True
        return StorageMachine.invoke(self,rva,receiver,argument)

    def hook(self,uc,address,size,data):
        if not self.generator_ready:return super().hook(uc,address,size,data)
        rva,x=address-self.base,self.x
        if rva==0x38E1A0 and self.manual_show_pending:
            self.manual_show_pending=False;uc.reg_write(x.UC_X86_REG_R8,self.pivot)
        cx,dx,r8,r9=[self.reg(r) for r in [x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9]]
        self.executed.add(rva)
        if rva==0x2B7B40 and cx-self.base in self.generator_bindings:
            self.service('generator_metadata_service',[self.generator_bindings[cx-self.base][1]],lambda:self.ret(self.rq(cx)))
        elif rva==0x2B7D40 and cx in [self.generator_types[n] for n in ['UnityEngine.WaitForSeconds_TypeInfo','TutorialsController.<>c__DisplayClass25_0_TypeInfo','System.Action<ETutorialType>_TypeInfo']]:
            name=next(n for n,t in self.generator_types.items() if t==cx)
            def allocation():
                token=self.alloc();self.q(token,cx)
                {'UnityEngine.WaitForSeconds_TypeInfo':self.waits,'TutorialsController.<>c__DisplayClass25_0_TypeInfo':self.closures,'System.Action<ETutorialType>_TypeInfo':self.show_delegates}[name][token]=None
                self.ret(token)
            self.service('generator_allocate_service',[name],allocation)
        elif rva==0x1C7A010 and cx==self.icon:
            assert dx==0
            result=0 if self.options.get('null_transform') else self.pivot
            self.service('character_icon_transform_service',[cx,result],lambda:self.ret(result))
        elif rva==0xB45070:
            assert cx in self.raw_lists and self.raw_lists[cx]==4 and dx==30
            assert r8==self.generator_method('Method$System.Collections.Generic.List<ETutorialType>.Contains()')
            yes=dx in self.raw_list_state(cx)['values']
            self.service('showed_type_contains_service',[cx,dx,yes],lambda:self.ret(0xFACE000000000000|int(yes)))
        elif rva==0x4D5E50:
            method=self.generator_method('Method$TutorialsController.<>c__DisplayClass25_0.<PickableCharacterType>b__0()')
            assert cx in self.show_delegates and dx in self.closures and r8==method and r9==0
            def constructor():
                self.show_delegates[cx]={'target':dx,'method':r8}
                self.q(cx+0x18,self.base+TARGETS['closure'][0]);self.q(cx+0x28,r8);self.q(cx+0x40,dx);self.ret()
            self.service('pickable_show_delegate_constructor_service',[cx,dx,r8],constructor)
        else:return super().hook(uc,address,size,data)

    def run_generator(self,label,state,options=None):
        self.generator_ready=self.handler_ready=self.tutorial_ready=False
        self.manual_show_pending=False
        r=self.run_data('Save',state,{'generator_entry':label,'pivot_live':True,'notes':[],**(options or {})},storage={})
        r['method']=label
        assert bytes(self.u.mem_read(self.actor,0x1B8))==self.actor_before
        assert all(bytes(self.u.mem_read(a,len(before)))==before for a,before in self.retained_arrays.items())
        r['character_and_stage_arrays_retained_verified']=True
        return r


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root)
    state={'key':'Tutorials','completedTutorials':['old-t'],'unlockedCharactersId':['c']}
    cases,baselines,failures=[],[],[]
    for label,warm,shown,initial in itertools.product(['hover','pick'],[False,True],[False,True],[0,-1,7]):
        r=m.run_generator(label,state,{'warm':warm,'showed':[30] if shown else [],'initial_state':initial,'steps':[{'kind':'move'}]})
        assert r['returned']
        step=r['final']['generator']['manual_steps'][0]
        assert step['returned_bool']==int(initial==0)
        assert step['state']==(1 if initial==0 else initial&0xFFFFFFFF)
        if initial==0:assert r['final']['generator']['waits'][0]['seconds_bits']==(0x3F000000 if label=='hover' else 0x3D4CCCCD)
        assert values(r['final']['save'])==state;cases.append(r)
    notes=lambda t:[{'id':'hover-t' if t==40 else 'pick-t','type':t,'state':0,'stage':0,'stages':[True],'active':False}]
    for label,warm,shown in itertools.product(['hover','pick'],[False,True],[False,True]):
        r=m.run_generator(label,state,{'warm':warm,'showed':[30] if shown else [],'notes':notes(40 if label=='hover' else 80),'show_callback':True})
        assert r['returned']
        steps=r['final']['generator']['manual_steps']
        assert [s['returned_bool'] for s in steps]==([1,0] if label=='hover' else [1,1,0] if shown else [1,0,0])
        if label=='hover' or shown:
            assert values(r['final']['save'])['completedTutorials']==['old-t','hover-t' if label=='hover' else 'pick-t']
            assert len(r['final']['storage_calls'])==1
        else:
            records=r['final']['tutorials']['queue_records'];assert len(records)==1 and records[0]['restriction']==1
            assert r['final']['generator']['on_tutorial_show']!=r['final']['generator']['prior_on_tutorial_show']
            assert not r['final']['storage_calls']
        cases.append(r)
    joined_notes=[{'id':'info-t','type':30,'state':0,'stage':0,'stages':[True],'active':False},
                  {'id':'pick-t','type':80,'state':0,'stage':0,'stages':[True],'active':False}]
    chain=[{'kind':'move'},{'kind':'move'},{'kind':'show','type':30},{'kind':'close','type':30}]
    for warm in [False,True]:
        r=m.run_generator('pick',state,{'warm':warm,'showed':[],'notes':joined_notes,'steps':chain,'show_callback':True})
        assert r['returned'],r['error']
        assert values(r['final']['save'])['completedTutorials']==['old-t','info-t','pick-t']
        assert r['final']['tutorials']['queued']['count']==0 and r['final']['tutorials']['queue_records'][0]['restriction']==0
        assert len(r['final']['storage_calls'])==2
        cases.append(r)
    for typ in [0,20,30,40,80]:
        r=m.run_generator('pick',state,{'showed':[],'steps':[{'kind':'move'},{'kind':'move'},{'kind':'closure','type':typ}]})
        assert r['returned'] and r['final']['tutorials']['queue_records'][0]['restriction']==int(typ!=30);cases.append(r)
    for label,opts in [('hover',{'null_character':True}),('hover',{'null_icon':True}),('hover',{'null_controller':True}),
                       ('pick',{'null_character':True}),('pick',{'null_icon':True}),('pick',{'null_controller':True}),
                       ('pick',{'null_showed':True}),('pick',{'null_queue':True,'showed':[]})]:
        r=m.run_generator(label,state,opts);assert not r['returned'] and r['error']=='null_reference';cases.append(r)
    for label,opts in [('hover',{'notes':notes(40)}),('pick',{'notes':notes(80)}),
                       ('pick',{'showed':[],'notes':joined_notes,'steps':chain,'show_callback':True})]:
        baseline=m.run_generator(label,state,opts);assert baseline['returned'];bid=len(baselines);baselines.append(baseline);counts={}
        for index,event in enumerate(baseline['events']):
            kind=event['kind'];counts[kind]=counts.get(kind,0)+1
            r=m.run_generator(label,state,dict(opts,failure=[kind,counts[kind]]))
            assert not r['returned'] and r['events']==baseline['events'][:index+1]
            assert r['final']==event['snapshot']
            failures.append({'baseline':bid,'prefix_length':index+1,'failure':[kind,counts[kind]],'exact_snapshot_verified':True})
    return {'build_id':BUILD,'targets':m.generator_targets,'fields':m.generator_fields,'wait_literals':m.wait_literals,
            'native_ranges':{n:[hex(r[0]),hex(r[1])] for n,r in TARGETS.items()},'unwind_families':{n:[[hex(a),hex(b)] for a,b in rows] for n,rows in FAMILIES.items()},
            'instruction_assertions':m.generator_assertions,'cases':cases,'case_count':len(cases),'failure_baselines':baselines,
            'failure_cases':failures,'failure_case_count':len(failures),'executed_address_count':len(m.executed),
            'scope':'Native factories, shared constructor, Hover/Pickable MoveNext, WaitForSeconds constructor, predicate closure, actual Show/Close/Hide/hidden/queue/save/JSON/storage callers run under explicitly ordered manual resumes. Runtime allocation/metadata/barriers/delegate construction/list and Unity UI/Transform services are supplied. No real scheduler admission, elapsed time, engine event dispatch/readiness or exception unwinding is claimed.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(compact_trace(r),indent=2)+'\n',encoding='utf-8')
    print(r['case_count'],r['failure_case_count'],r['executed_address_count'])
