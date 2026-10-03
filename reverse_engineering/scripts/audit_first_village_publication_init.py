"""Retain the original five native initializers through publication and Act Init.

Scene/CLR/class/delegate gateways are named providers. Native bytes stay private.
The same original Manage frame stops before the ordered-Start array is read.
"""
import argparse
import copy
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_first_village_initialization import InitializationWitness, ORDER, sha
from audit_first_village_role_setup import METHODS, ROLES
from audit_first_village_profile_generation import ROOT, load_inputs
from audit_report_snapshots import pool_snapshots, expand_snapshots


class PublicationInitWitness(InitializationWitness):
    def __init__(self,inputs,profiles,characters,init_report,role_report):
        super().__init__(inputs,profiles,characters,init_report)
        import capstone
        self.post=False
        self.post_ordinals=[]; self.actions=[]; self.action=None; self.concrete_calls=[]
        self.callbacks={}; self.publication_list=0; self.pre_publication=None
        self.post_bodies=[]; self.role_entries={}; self.method_entries={}
        wanted=list(METHODS)+['Gameplay.UpdateCharacters']
        for name in wanted:
            rows=[r for r in self.meta['ScriptMethod'] if r['Name']==name.replace('.','$$',1)]
            assert len(rows)==1,name
            row=rows[0]; start=row['Address']
            end=min(r['Address'] for r in self.meta['ScriptMethod'] if r['Address']>start)
            raw=self.pe.get_data(start,end-start)
            ins=list(self.cs.disasm(raw,start))
            while ins[-1].mnemonic=='int3': ins.pop()
            assert ins[0].address==start and all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]))
            self.instructions.update({i.address:i for i in ins})
            self.method_entries[name]=start
            self.role_entries.setdefault(start,[]).append(name)
            body={'name':name,'rva':hex(start),'signature':row['Signature'],'end_rva':hex(ins[-1].address+ins[-1].size),
                  'instruction_count':len(ins),'body_sha256':sha(raw[:ins[-1].address+ins[-1].size-start])}
            self.post_bodies.append(body)
            if name!='Gameplay.UpdateCharacters': assert body in role_report['concrete_bodies'],name
        slots={r['Address']:r for k in ('ScriptMetadata','ScriptMetadataMethod','ScriptString') for r in self.meta[k]}
        for i in self.instructions.values():
            for op in i.operands:
                if op.type!=capstone.CS_OP_MEM or op.mem.base!=capstone.x86.X86_REG_RIP: continue
                a=i.address+i.size+op.mem.disp
                if a in slots:
                    row=slots[a]; name=row.get('Name','literal:'+row.get('Value',''))
                    if name not in self.names:
                        p=self.allocate(name); self.names[name]=p; self.metadata_names[p]=name; self.d(p+0xE0,1)
                    self.q(self.base+a,self.names[name])
                elif i.mnemonic=='cmp' and op.size==1: self.uc.mem_write(self.base+a,b'\1')
        self.post_pins={int(p['rva'],16):(p['mnemonic'],p['operands']) for p in role_report['instruction_assertions']}
        self.post_pins.update({0x36D03D:('call','0x3811b0'),0x36D0BA:('call','0x3645c0'),
                              0x36D0BF:('jmp','0x36d090'),0x36D0F9:('mov','r13, qword ptr [r12 + 0x28]'),
                              0x38123B:('mov','qword ptr [rax + 0x18], rbx')})
        for a,expected in self.post_pins.items(): assert (self.instructions[a].mnemonic,self.instructions[a].op_str)==expected
        # A named pre-entry hydration delta on the ORIGINAL source classes.
        # No initialized actor or retained clone is replaced or modified here.
        self.class_hydration=[]
        for ident in ORDER:
            binding=self.asset_fields[str(ident)]; klass=binding['source_class']; role=ROLES[ident]
            for offset,name in ((0x208,role+'.Act'),(0x258,role+'.BluffAct' if role!='Minion' else 'Role.BluffAct')):
                context=self.allocate('virtual_context:'+name)
                self.q(klass+offset,self.base+self.method_entries[name]); self.q(klass+offset+8,context)
                self.class_hydration.append({'asset':ident,'class_identity':klass,'slot_offset':offset,'method':name,
                                            'method_rva':hex(self.method_entries[name]),'method_context':context})
            if role=='Confessor':
                context=self.allocate('virtual_context:Confessor.OnInit')
                self.q(klass+0x1C8,self.base+self.method_entries['Confessor.OnInit']); self.q(klass+0x1D0,context)
                self.class_hydration.append({'asset':ident,'class_identity':klass,'slot_offset':0x1C8,'method':'Confessor.OnInit',
                                            'method_rva':hex(self.method_entries['Confessor.OnInit']),'method_context':context})
        source_roles={row['source_role'] for row in self.asset_fields.values()}
        self.asset_storage=[(p,bytes(self.uc.mem_read(p,0x48 if p in source_roles else len(raw)))) for p,raw in self.asset_storage]
        self.enum_lists={self.scene[a][k] for a in self.actors for k in ('active','resistance')}

    def snapshot(self):
        out=super().snapshot()
        if not hasattr(self,'post'): return out
        p=self.rq(self.gameplay_static+0x18)
        out['publication']={'identity':p,'actors':self.values(p) if p else [],
                            'board_identity':self.rq(self.owner+0x20),'board_actors':self.values(self.rq(self.owner+0x20))}
        out['runtime_roles']=[{'actor_identity':a,'identity':self.rq(a+0x168),'class_identity':self.rq(self.rq(a+0x168)) if self.rq(a+0x168) else 0,
                               'on_acted':self.rq(self.rq(a+0x168)+0x28) if self.rq(a+0x168) else 0,
                               'saved_fields':[self.rq(self.rq(a+0x168)+o) for o in (0x30,0x38,0x40)] if self.rq(a+0x168) else []}
                              for a in self.actors]
        out['source_role_saved_fields']=[{'asset':int(ident),'identity':row['source_role'],
                                         'saved_fields':[self.rq(row['source_role']+o) for o in (0x30,0x38,0x40)]}
                                        for ident,row in self.asset_fields.items()]
        return out

    def preserve(self):
        if not getattr(self,'post',False): return super().preserve()
        assert all(bytes(self.uc.mem_read(p,len(raw)))==raw for p,raw in self.asset_storage)
        conf_active=self.scene[self.actors[1]]['active']
        clones={row['clone'] for row in self.init_calls if row.get('clone')}
        for storage in self.retained:
            for p,raw in storage:
                allowed=set()
                if p in clones: allowed.update(range(0x28,0x30))
                if p==conf_active: allowed.update(range(0x18,0x20)); allowed.update(range(0x420,0x424))
                now=bytes(self.uc.mem_read(p,len(raw)))
                assert all(a==b for i,(a,b) in enumerate(zip(raw,now)) if i not in allowed),(self.labels.get(p),self.phase)
        for clone in clones:
            assert self.rq(clone+0x28) in (0,self.callbacks.get(clone,0))
            assert [self.rq(clone+o) for o in (0x30,0x38,0x40)]==[0,0,0]
        count=self.rd(conf_active+0x18)
        assert count in (0,1) and self.rd(conf_active+0x1C)==1+count
        if count: assert self.rd(self.rq(conf_active+0x10)+0x20)==25

    def post_service(self,name,**details):
        return self.own_service(name,**details)

    def service(self,name,**details):
        # Inherited enumeration/liveness gateways are part of this same caller.
        # Collect them too, before their effects, without double-counting.
        if getattr(self,'post',False):
            self.preserve()
            self.post_ordinals.append(len(self.events)+1)
            details.setdefault('phase',self.phase)
            details.setdefault('raw_arguments',{n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ('RCX','RDX','R8','R9')})
        return super().service(name,**details)

    def hook(self,uc,address,size,user):
        x=self.x; r=address-self.base
        c,t,m,n=[self.reg(z) for z in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        if r==0x36D01E:
            assert self.active is None and len(self.init_calls)==5 and all(row['completed'] for row in self.init_calls)
            self.preserve(); self.pre_publication=self.snapshot()
            self.first_post_service=len(self.events)+1
            self.pre_sp=self.reg(x.UC_X86_REG_RSP)
            assert self.pre_sp==self.init_calls[0]['entry_sp']+8 and self.rq(self.stack+0x10008)==self.stop
            self.post=True; self.phase='publication'
            return  # Let the SAME retained Manage instruction execute.
        if not self.post: return super().hook(uc,address,size,user)
        if r==0x36D0F9:
            assert len(self.actions)==5 and all(row['completed'] for row in self.actions) and self.action is None
            assert self.reg(x.UC_X86_REG_RSP)==self.pre_sp and self.rq(self.stack+0x10008)==self.stop
            self.preserve(); self.boundary={'kind':'before_ordered_start','rva':hex(r),'completed_act_init':5,
                'live_manage_sp':self.pre_sp,'root_return_sentinel':self.stop}; uc.emu_stop(); return
        if r==0x3645C0:
            assert self.action is None and c==self.actors[len(self.actions)] and t==3 and m==0
            regs=self.nonvolatiles+[getattr(x,f'UC_X86_REG_XMM{i}') for i in range(6,16)]
            self.action={'actor':self.labels[c],'actor_identity':c,'trigger':t,'entry_sp':self.reg(x.UC_X86_REG_RSP),
                         'entry_nonvolatiles':[self.reg(z) for z in regs],'before':self.actor_snapshot(c),
                         'native_call_start':len(self.concrete_calls),'completed':False}
            self.actions.append(self.action); self.phase='act_init:'+str(len(self.actions))
        if r==0x36D0BF:
            assert self.action is not None
            regs=self.nonvolatiles+[getattr(x,f'UC_X86_REG_XMM{i}') for i in range(6,16)]
            assert self.reg(x.UC_X86_REG_RSP)==self.action['entry_sp']+8
            assert [self.reg(z) for z in regs]==self.action['entry_nonvolatiles']
            self.preserve(); a=self.action['actor_identity']; clone=self.rq(a+0x168)
            assert self.rq(clone+0x28)==self.callbacks[clone]
            self.action.update(completed=True,after=self.actor_snapshot(a),snapshot=self.snapshot(),
                               concrete_calls=copy.deepcopy(self.concrete_calls[self.action['native_call_start']:]))
            self.action=None; self.phase='init_loop'
        if r in self.role_entries:
            role=next((ROLES[ORDER[i]] for i,a in enumerate(self.actors) if self.rq(a+0x168)==c),None)
            names=self.role_entries[r]; name=next((v for v in names if role and v.startswith(role+'.')),names[0])
            if r==0x33ED50 and role is None: name=None
            if name:
                self.concrete_calls.append({'method':name,'receiver':c,'rdx':t,'r8':m,'r9':n,
                                             'caller_return_rva':hex(self.rq(self.reg(x.UC_X86_REG_RSP))-self.base)})
                if name=='CharacterStatuses.AddStatus':
                    assert c==self.scene[self.actors[1]]['status'] and t==25 and m==self.actors[1] and n==0
                    assert self.rq(self.reg(x.UC_X86_REG_RSP)+0x28)==0
        if r in self.instructions:
            self.visited.add(r); return
        if address==self.stop+0x600: raise AssertionError('Act Init invoked unexpected onActed callback')
        if r in (0x2B7D40,0x2B6FF0,0xB610A0,0xB45070,0x41A0,0x282580,0xF74DF0,0x1C4B450,0x4D5B60):
            kind={0x2B7D40:'post_allocate',0x2B6FF0:'post_barrier',0xB610A0:'publication_list_ctor',
                  0xB45070:'enum_contains',0x41A0:'enum_add',0x282580:'trigger_box',0xF74DF0:'log_format',
                  0x1C4B450:'log',0x4D5B60:'role_delegate_ctor'}[r]
            details={}; result=0
            if kind=='post_allocate':
                name=self.metadata_names[c]
                assert name in ('System.Collections.Generic.List<Character>_TypeInfo','Character.<>c__DisplayClass125_0_TypeInfo','System.Action<ActedInfo>_TypeInfo')
                details.update(object_type=name,result=self.cursor)
            elif kind=='post_barrier': assert self.rq(c)==t
            elif kind=='publication_list_ctor':
                assert c==self.publication_list and t==self.board and m==self.names['Method$System.Collections.Generic.List<Character>..ctor()']
                details['actors']=self.values(t)
            elif kind in ('enum_contains','enum_add'):
                assert c in self.enum_lists
                values=[self.rd(self.rq(c+0x10)+0x20+i*4) for i in range(self.rd(c+0x18))]
                details.update(list_identity=c,status=t&0xFFFFFFFF,method=self.metadata_names.get(m))
                if kind=='enum_add': assert c==self.scene[self.actors[1]]['active'] and values==[] and t==25
                else: result=int((t&0xFFFFFFFF) in values)
            elif kind=='role_delegate_ctor':
                assert self.action and self.rq(t+0x10)==self.action['actor_identity'] and self.rd(t+0x18)==3
                assert m==self.names['Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()'] and n==0
                details.update(identity=c,closure=t,actor=self.rq(t+0x10),trigger=self.rd(t+0x18))
            elif kind=='trigger_box': details.update(type=self.metadata_names.get(c),value=self.rd(t),result=self.cursor)
            elif kind=='log_format': details.update(literal=self.metadata_names.get(c),boxed=t,result=self.cursor)
            if not self.post_service(kind,**details): return
            if kind=='post_allocate':
                result=self.allocate(name+':post:'+str(len(self.post_ordinals))); self.q(result,c)
                if name=='System.Collections.Generic.List<Character>_TypeInfo': self.publication_list=result
            elif kind=='publication_list_ctor': self.write_list(c,self.values(t),0)
            elif kind=='enum_add':
                self.d(self.rq(c+0x10)+0x20,25); self.d(c+0x18,1); self.d(c+0x1C,self.rd(c+0x1C)+1)
            elif kind=='role_delegate_ctor':
                self.q(c+0x18,self.stop+0x600); self.q(c+0x20,t); self.q(c+0x28,m); self.q(c+0x40,t)
                self.callbacks[self.rq(self.action['actor_identity']+0x168)]=c
            elif kind in ('trigger_box','log_format'):
                result=self.allocate(kind+':'+str(len(self.post_ordinals)))
                if kind=='trigger_box': self.d(result+0x10,self.rd(t))
            self.ret(result); return
        return super().hook(uc,address,size,user)


def audit(game_root,dumper_root):
    dependencies=['audit_first_village_publication_init.py','audit_first_village_initialization.py','audit_first_village_role_setup.py',
                  'audit_first_village_bluff_generation.py','audit_first_village_profile_generation.py','audit_manage_pool_composition.py',
                  'audit_character_assets.py','audit_ascension_assets.py','audit_report_snapshots.py','audit_round_candidate_composition.py','audit_spy.py']
    hashes={n:sha((Path(__file__).parent/n).read_bytes()) for n in dependencies}
    names=('ascension_assets_audit','character_assets_audit','first_village_bluff_generation','character_init','first_village_initialization','first_village_role_setup')
    paths={n:ROOT/f'reports/{BUILD}_{n}.json' for n in names}
    prior_hashes={n:sha(p.read_bytes()) for n,p in paths.items()}
    reports={n:json.loads(p.read_text(encoding='utf-8')) for n,p in paths.items()}
    from audit_ascension_assets import audit as profile_audit
    from audit_character_assets import audit as character_audit
    assert reports['ascension_assets_audit']==json.loads(json.dumps(profile_audit(Path(game_root),Path(dumper_root))))
    assert reports['character_assets_audit']==json.loads(json.dumps(character_audit(Path(game_root),Path(dumper_root))))
    m=PublicationInitWitness(load_inputs(Path(game_root),Path(dumper_root)),reports['ascension_assets_audit'],reports['character_assets_audit'],reports['character_init'],reports['first_village_role_setup'])
    constructors=[]
    for a in m.actors:
        row=m.run('Character.ctor',a); assert row['failure'] is None; constructors.append(row)
        f=m.actor_snapshot(a); assert f['uses']==f['act']==1 and f['saved_act']==m.literals['']
    m.phase='setup'; setup=[]
    for name,this,arg,choices in [('GameData.SetupCurrentAscension',m.game,0,[]),('AscensionsData.ClearCurrentPickedScript',m.temporary,0,[]),
        ('AscensionsData.SetupCharactersCount',m.temporary,0,[0]),('AscensionsData.SetupStartingCharacters',m.temporary,0,[]),('Gameplay.GetCurrentScript',m.gameplay,0,[])]:
        row=m.run(name,this,arg,choices); assert not row['failure']; setup.append(row)
    m.q(m.gameplay_static+0x30,m.reg(m.x.UC_X86_REG_RAX))
    prior=reports['first_village_bluff_generation']; genrow=prior['generation_index_factor']['cases'][0]; poolrow=prior['pool_index_factors']['cases'][0]
    generation=m.run('Gameplay.GetRandomCharacters',m.gameplay,5,genrow['choices']); assert generation['final']['returned_order']==ORDER
    returned=m.reg(m.x.UC_X86_REG_RAX); saved=m.save(); initial=m.snapshot(); m.phase='manage'; m.new_services=[]
    joined=m.run('Characters.ManageCharacters',m.owner,returned,poolrow['choices'])
    assert not joined['failure'] and joined['boundary']['kind']=='before_ordered_start'
    assert m.post_ordinals==list(range(m.first_post_service,len(joined['services'])+1))
    assert joined['final']['publication']['actors']==m.actors and joined['final']['publication']['identity']!=m.board
    assert joined['final']['publication']['board_identity']==m.board and joined['final']['pools']==poolrow['pool_identities_and_contents']
    graph_keys=('source_starting','source_inline','temporary_starting','temporary_cache','selected_counts','temporary_counts',
                'rosters','saved_rosters','roster_identities','board_order','current_script_identity','current_script_fields','asset_bindings')
    expected_graph={k:initial[k] for k in graph_keys}
    assert {k:m.pre_publication[k] for k in graph_keys}==expected_graph
    assert {k:joined['final'][k] for k in graph_keys}==expected_graph
    assert m.pre_publication['pools']==poolrow['pool_identities_and_contents']
    for i,action in enumerate(m.actions):
        f=action['after']; assert f['id']==5-i and f['data']==m.assets[ORDER[i]] and f['uses']==1 and f['state']==5 and f['started']==0
        assert f['acted_infos_storage']['count']==0 and f['acted_infos_storage']['version']==1
        assert f['hover_infos_storage']['count']==f['hover_infos_storage']['version']==0
        assert f['saved_act']==m.literals[''] and f['act']==1
        status=f['status_storage']; assert status['active_values']==([25] if i==1 else []) and status['version']==(2 if i==1 else 1)
        assert status['resistance_values']==[] and status['resistance_version']==0 and status['target']==0
        assert sum(c['method']=='Confessor.OnInit' for c in action['concrete_calls'])==int(i==1)
        assert {k:action['snapshot'][k] for k in graph_keys}==expected_graph
        assert action['snapshot']['pools']==poolrow['pool_identities_and_contents']
    assert all(row['state']==1 and row['wait_bits']==0x3E99999A for row in joined['final']['continuations'])
    before_publication=copy.deepcopy(m.pre_publication); initializers=copy.deepcopy(m.init_calls); first_yields=copy.deepcopy(m.first_yields)
    actions=copy.deepcopy(m.actions); calls=copy.deepcopy(m.concrete_calls); stop_ordinals=list(m.post_ordinals); stops=[]
    for ordinal in stop_ordinals:
        m.restore(saved); m.phase='manage'; m.post=False; m.active=None; m.action=None
        m.init_calls=[]; m.first_yields=[]; m.retained=[]; m.frames=[]; m.iterators=[]; m.new_services=[]
        m.post_ordinals=[]; m.actions=[]; m.concrete_calls=[]; m.callbacks={}; m.publication_list=0; m.pre_publication=None
        result=m.run('Characters.ManageCharacters',m.owner,returned,poolrow['choices'],stop_service=ordinal,record_events=False)
        assert result['failure']=='service:'+joined['services'][ordinal-1]['service']
        assert [{k:v for k,v in e.items() if k!='snapshot'} for e in result['services']]==[{k:v for k,v in e.items() if k!='snapshot'} for e in joined['services'][:ordinal]]
        assert result['final']==joined['services'][ordinal-1]['snapshot']
        assert {k:result['final'][k] for k in graph_keys}==expected_graph
        assert result['final']['pools']==poolrow['pool_identities_and_contents']
        stops.append({'service_ordinal':ordinal,'service':joined['services'][ordinal-1]['service'],'completed_act_init':sum(a['completed'] for a in m.actions),'final':result['final']})
    assert all(sha(p.read_bytes())==prior_hashes[n] for n,p in paths.items())
    assert all(sha((Path(__file__).parent/n).read_bytes())==hashes[n] for n in dependencies)
    return {'schema_version':'first_village_publication_init_v1','build_id':BUILD,
            'domain':{'generation_row':0,'pool_row':0,'asset_order':ORDER,'initial_actor_state':20,'class_hydration':'Named pre-entry virtual-slot provider on original source classes; clones/actors retained.',
                      'services':'Inherited scene/CLR/clone/first-step providers; publication shallow copy, int32 enum list membership/add, allocation/delegate constructor/logging/barriers supplied.'},
            'scope':'Same retained Manage through actual publication and five ActInit calls; stop before ordered Start. No queue drain, acquisition completion, Day results or pixel/player-history admission.',
            'source_hashes':hashes,'prior_report_hashes':prior_hashes,'class_hydration':m.class_hydration,'retained_graph_invariants':expected_graph,
            'selected_instruction_assertions':[{'rva':hex(a),'mnemonic':v[0],'operands':v[1]} for a,v in m.post_pins.items()],
            'bodies':m.body_evidence+m.new_bodies+m.post_bodies,'constructors':constructors,'setup':setup,'generation':generation,
            'initial':initial,'pre_publication':{'snapshot':before_publication},'initializers':initializers,'first_yields':first_yields,
            'joined':joined,'act_init_returns':actions,'concrete_calls':calls,'stopped_prefixes':stops,
            'counters':{'constructors':len(constructors),'initializers':len(initializers),'first_yields':len(first_yields),'act_init_calls':len(actions),
                        'manage_services':len(joined['services']),'post_services':len(stop_ordinals),'stopped_prefixes':len(stops),
                        'native_instruction_addresses':len(m.visited),'selected_post_pins':len(m.post_pins),'python_source_hashes':len(dependencies)}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--game-root',required=True); p.add_argument('--dumper-root',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); report=audit(a.game_root,a.dumper_root); assert report==json.loads(json.dumps(report))
    packed=pool_snapshots(report); assert expand_snapshots(json.loads(json.dumps(packed)))==report
    Path(a.output).write_text(json.dumps(packed,indent=2,sort_keys=True,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':Path(a.output).name,**report['counters']},sort_keys=True),flush=True)


if __name__=='__main__': main()
