"""Original generation row 0 retained through five native Init first yields.

Only authored operands and normalized storage projections are exported. Runtime,
scene hydration, role cloning and synchronous coroutine entry remain providers.
"""
import argparse
import copy
import hashlib
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_first_village_bluff_generation import BluffJoin
from audit_first_village_profile_generation import ROOT, load_inputs
from audit_report_snapshots import pool_snapshots, expand_snapshots


ORDER = [21596, 21614, 21626, 21621, 21618]
FIELDS = {'data': (0x50, 8), 'bluff': (0x58, 8), 'register_as': (0x60, 8),
          'trailer': (0x68, 8), 'runtime': (0x70, 8), 'dead_prefab': (0x98, 8), 'revealed': (0xD8, 1),
          'uses': (0xDC, 4), 'previous': (0xE0, 4), 'state': (0xE4, 4),
          'killed_hidden': (0xEC, 1), 'killed_demon': (0xED, 1), 'alignment': (0xF8, 4),
          'id': (0x118, 4), 'started': (0x11C, 1), 'acted_infos': (0x148, 8),
          'hover_infos': (0x150, 8), 'role': (0x168, 8), 'bluff_role': (0x170, 8),
          'saved_act': (0x198, 8), 'act': (0x1A1, 1), 'statuses': (0xF0, 8), 'state_callback': (0x180, 8)}


def sha(raw): return hashlib.sha256(raw).hexdigest()


class InitializationWitness(BluffJoin):
    def __init__(self, inputs, profiles, characters, init_report):
        super().__init__(inputs, profiles, characters)
        import capstone
        self.phase, self.active = 'hydration', None
        self.init_calls, self.first_yields, self.retained, self.constructed = [], [], [], []
        self.frames = []
        self.new_services = []
        self.new_bodies, self.new_decoded = [], set()
        for name, declarations in init_report['fields'].items():
            block = re.search(r'^[^\n]*class '+re.escape(name)+r'(?: :[^\n]*)? // TypeDefIndex: \d+\s*\{(.*?)^\}', self.dump, re.M | re.S)
            assert block and all(d in block[1] for d in declarations), name
        methods = {0x3697C0: 'Character$$.ctor', 0x365A20: 'Character$$Init',
                   0x367970: 'Character$$RefreshCharacter', 0x367B60: 'Character$$RefreshView',
                   0x3756B0: 'Character.<DelayReveal>d__84$$MoveNext'}
        for start, name in methods.items():
            rows = [m for m in self.meta['ScriptMethod'] if m['Address'] == start and m['Name'] == name]
            assert len(rows) == 1
            end = min(m['Address'] for m in self.meta['ScriptMethod'] if m['Address'] > start)
            body = self.pe.get_data(start, end-start)
            ins = list(self.cs.disasm(body, start))
            while ins[-1].mnemonic == 'int3': ins.pop()
            assert ins[0].address == start and all(a.address+a.size == b.address for a,b in zip(ins,ins[1:]))
            self.instructions.update({i.address:i for i in ins})
            self.new_decoded.update(i.address for i in ins)
            self.new_bodies.append({'name':name, 'rva':hex(start), 'end_rva':hex(ins[-1].address+ins[-1].size),
                                    'instruction_count':len(ins), 'signature':rows[0]['Signature'],
                                    'body_sha256':sha(body[:ins[-1].address+ins[-1].size-start])})
        self.entries['Character.ctor'] = 0x3697C0
        ins = list(self.cs.disasm(self.pe.get_data(0x33ED50,3),0x33ED50))
        assert [(i.mnemonic,i.op_str) for i in ins] == [('ret','0')]
        self.instructions[0x33ED50] = ins[0]
        self.new_decoded.add(0x33ED50)
        self.pins = {0x36CFDA: ('call','0x365a20'), 0x36D01E: ('mov','rbx, qword ptr [r12 + 0x20]'),
                     0x369801: ('mov','dword ptr [rdi + 0xdc], 1'),
                     0x369833: ('mov','qword ptr [rcx], rbx'), 0x369863: ('mov','qword ptr [rcx], rbx'),
                     0x369879: ('mov','qword ptr [rcx], rax'), 0x36988A: ('mov','byte ptr [rdi + 0x1a1], 1'),
                     0x365CDF: ('inc','dword ptr [rax + 0x1c]'),
                     0x365CE2: ('mov','dword ptr [rax + 0x18], r15d'),
                     0x3756F4: ('mov','dword ptr [rdi + 0x10], 0xffffffff'),
                     0x37572E: ('mov','qword ptr [rcx], rax'),
                     0x375769: ('mov','dword ptr [rdi + 0x10], 1'), 0x33ED50: ('ret','0')}
        for a,expected in self.pins.items():
            assert a in self.instructions and (self.instructions[a].mnemonic,self.instructions[a].op_str) == expected
        self.literals = {}
        slots = {r['Address']:r for key in ('ScriptMetadata','ScriptMetadataMethod','ScriptString') for r in self.meta[key]}
        for a in self.new_decoded:
            i = self.instructions[a]
            for op in i.operands:
                if op.type != capstone.CS_OP_MEM or op.mem.base != capstone.x86.X86_REG_RIP: continue
                address = i.address+i.size+op.mem.disp
                if address in slots:
                    row = slots[address]
                    name = row.get('Name', 'literal:'+row.get('Value',''))
                    if name not in self.names:
                        p = self.allocate(name); self.names[name] = p; self.metadata_names[p] = name
                        self.d(p+0xE0,1)
                    self.q(self.base+address,self.names[name])
                    if 'Value' in row: self.literals[row['Value']] = self.names[name]
                elif i.mnemonic == 'cmp' and op.size == 1: self.uc.mem_write(self.base+address,b'\1')
        assert {'','# {0}','INIT: '} <= self.literals.keys()
        i = self.instructions[0x375742]
        literal = i.address+i.size+i.operands[1].mem.disp
        assert struct.unpack('<I', self.pe.get_data(literal,4))[0] == 0x3E99999A
        self.asset_fields, self.scene = {}, {}
        records = {r['path_id']:r for r in characters['records']}
        for ident in ORDER:
            row, p = records[ident], self.assets[ident]
            declaration = 'public class '+row['role_type']+' : Role // TypeDefIndex: '+str(row['role_type_def_index'])
            assert declaration in self.dump, declaration
            role_class = self.allocate('source_class:'+row['role_type'])
            source = self.allocate('source_role:'+str(ident)); self.q(source,role_class)
            name = self.allocate('asset_name:'+str(ident))
            self.q(p+0x140,source); self.q(p+0x28,name)
            self.d(p+0x138,row['abilityUsage']); self.uc.mem_write(p+0x13E,bytes([row['picking']]))
            self.asset_fields[str(ident)] = {k:row[k] for k in ('name','characterName','type','startingAlignment','abilityUsage','picking','role_rid','role_type')}
            self.asset_fields[str(ident)].update(identity=p, source_role=source, source_class=role_class, name_identity=name)
        for index,a in enumerate(self.actors):
            status = self.allocate(f'status:{index}')
            active = self.list([],f'active:{index}'); resist = self.list([],f'resistant:{index}')
            self.q(status+0x10,active); self.q(status+0x18,resist)
            self.q(a+0xF0,status)
            controls = {name:self.allocate(f'{name}:{index}') for name in ('acted','number','number_class','rip','pickable','text','boxed')}
            self.q(a+0xA8,controls['acted']); self.q(a+0x48,controls['number'])
            self.q(controls['number'],controls['number_class'])
            self.q(controls['number_class']+0x558,self.stop+0x400)
            self.q(controls['number_class']+0x560,controls['number_class']+0x800)
            self.q(a+0x78,controls['rip']); self.q(a+0x1A8,controls['pickable'])
            self.q(a+0x188,self.array([],f'pickeds:{index}'))
            self.d(a+0xE4,20); self.d(a+0xE0,10)
            self.scene[a] = {**controls,'status':status,'active':active,'resistance':resist}
        self.d(self.gameplay_static+0x2C,0)
        self.asset_storage=[]
        for row in self.asset_fields.values():
            for p,n in ((row['identity'],0x148),(row['source_role'],0x40),(row['source_class'],0x1000)):
                self.asset_storage.append((p,bytes(self.uc.mem_read(p,n))))

    def reg(self,r): return self.uc.reg_read(r)

    def actor_snapshot(self,a):
        out = {k:int.from_bytes(self.uc.mem_read(a+offset,width),'little') for k,(offset,width) in FIELDS.items()}
        out['identity'] = a; out['label'] = self.labels[a]
        for field in ('acted_infos','hover_infos'):
            p = out[field]
            out[field+'_storage'] = None if not p else {'identity':p,'backing':self.rq(p+0x10),'count':self.rd(p+0x18),'version':self.rd(p+0x1C)}
        s = self.scene[a]
        out['status_storage'] = {k:s[k] for k in ('status','active','resistance')}
        out['status_storage'].update(count=self.rd(s['active']+0x18),version=self.rd(s['active']+0x1C),
                                     resistance_count=self.rd(s['resistance']+0x18),target=self.rq(s['status']+0x20))
        out['status_storage'].update(active_backing=self.rq(s['active']+0x10),resistance_backing=self.rq(s['resistance']+0x10),
                                     resistance_version=self.rd(s['resistance']+0x1C))
        out['status_storage'].update(active_values=[self.rd(self.rq(s['active']+0x10)+0x20+i*4) for i in range(self.rd(s['active']+0x18))],
                                     resistance_values=[self.rd(self.rq(s['resistance']+0x10)+0x20+i*4) for i in range(self.rd(s['resistance']+0x18))])
        out['actor_storage_sha256']=sha(bytes(self.uc.mem_read(a,0x1B8)))
        return out

    def snapshot(self):
        out = super().snapshot()
        if not hasattr(self,'scene'): return out
        out['actors'] = [self.actor_snapshot(a) for a in self.actors]
        out['continuations'] = [{'identity':p, 'owner':self.rq(p+0x20),'state':self.rd(p+0x10),
                                 'current':self.rq(p+0x18),'wait_bits':self.rd(self.rq(p+0x18)+0x10) if self.rq(p+0x18) else None}
                                for p in getattr(self,'iterators',[])]
        out['asset_bindings'] = self.asset_fields
        return out

    def preserve(self):
        assert all(bytes(self.uc.mem_read(p,len(raw)))==raw for p,raw in self.asset_storage), 'selected asset/source/class changed'
        for storage in self.retained:
            assert all(bytes(self.uc.mem_read(p,len(raw))) == raw for p,raw in storage), 'earlier actor/continuation changed'

    def own_service(self,name,**details):
        self.preserve()
        ordinal = len(self.events)+1
        self.new_services.append(ordinal)
        return self.service(name,phase=self.phase,raw_arguments={n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ('RCX','RDX','R8','R9')},**details)

    def run(self,*args,**kwargs):
        for i in range(16): self.uc.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),0 if i<6 else 0xABC100+i)
        # Retained snapshots cost more than the sparse base harness's five-second
        # budget. Continue the LIVE context after timeout/count exhaustion;
        # never reseed its frame or treat partial execution as completion.
        original = self.uc.emu_start
        def bounded(begin,end,timeout=0,count=0):
            for _ in range(24):
                original(begin,end,timeout=5_000_000,count=count)
                if self.failure or self.boundary or self.reg(self.x.UC_X86_REG_RIP)==end: return
                begin=self.reg(self.x.UC_X86_REG_RIP)
            raise AssertionError('Retained invocation exhausted its bounded execution budget')
        self.uc.emu_start=bounded
        try: return super().run(*args,**kwargs)
        finally: self.uc.emu_start=original

    def hook(self,uc,address,size,user):
        x, r = self.x,address-self.base
        c,t,m = [self.reg(z) for z in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8)]
        if r == 0x36D01E:
            assert self.active is None and len(self.init_calls)==5 and all(row['completed'] for row in self.init_calls)
            sp=self.reg(x.UC_X86_REG_RSP)
            assert sp==self.init_calls[0]['entry_sp']+8
            assert self.rq(self.stack+0x10008)==self.stop
            self.preserve(); self.boundary={'kind':'before_publication','rva':hex(r),'completed_initializers':5,
                'live_manage_sp':sp,'root_entry_sp':self.stack+0x10008,'root_return_sentinel':self.stop}; uc.emu_stop(); return
        if r == 0x36CFDF and self.active is not None:
            row = self.active
            assert self.reg(x.UC_X86_REG_RSP)==row['entry_sp']+8
            assert [self.reg(z) for z in self.nonvolatiles] == row['entry_nonvolatiles']
            assert [self.reg(getattr(x,f'UC_X86_REG_XMM{i}')) for i in range(6,16)]==row['entry_xmm_nonvolatiles']
            self.preserve()
            row['completed']=True; row['after']=self.actor_snapshot(row['actor_identity']); row['snapshot']=self.snapshot()
            a=row['actor_identity']; s=self.scene[a]; p=row['iterator']; wait=self.rq(p+0x18); clone=self.rq(a+0x168)
            pointers=[(a,0x1B8),(self.rq(a+0x148),0xC20),(self.rq(a+0x150),0xC20),
                      (s['status'],0x28),(s['active'],0xC20),(s['resistance'],0xC20),(p,0x28),(wait,0x20),(clone,0x40)]
            self.retained.append([(p,bytes(uc.mem_read(p,n))) for p,n in pointers])
            self.active=None; self.phase='manage'
        if r == 0x365A20:
            self.preserve(); assert self.active is None and c==self.actors[len(self.init_calls)]
            assert t==self.assets[ORDER[len(self.init_calls)]] and m==5-len(self.init_calls) and self.reg(x.UC_X86_REG_R9)==0
            sp=self.reg(x.UC_X86_REG_RSP); assert self.rq(sp)==self.base+0x36CFDF
            self.nonvolatiles=[getattr(x,'UC_X86_REG_'+n) for n in ('RBX','RBP','RSI','RDI','R12','R13','R14','R15')]
            self.active={'actor':self.labels[c],'actor_identity':c,'asset_id':self.asset_ids[t],'display_id':m,
                         'before':self.actor_snapshot(c),'entry_sp':sp,'entry_return_rva':'0x36cfdf',
                         'entry_nonvolatiles':[self.reg(z) for z in self.nonvolatiles],
                         'entry_xmm_nonvolatiles':[self.reg(getattr(x,f'UC_X86_REG_XMM{i}')) for i in range(6,16)],'completed':False}
            self.init_calls.append(self.active); self.phase='init:'+str(len(self.init_calls))
        if r==0x3697C0: self.phase='ctor'; self.ctor_actor=c
        if address==self.stop+0x400:
            assert self.active and c==self.scene[self.active['actor_identity']]['number']
            if self.own_service('set_text'): self.ret()
            return
        if address==self.stop+0x300:
            assert self.frames and self.active
            ctx,sp,regs=self.frames.pop()
            assert self.reg(x.UC_X86_REG_RAX)&255==1 and self.reg(x.UC_X86_REG_RSP)==sp+8
            assert all(self.reg(z)==v for z,v in regs.items())
            p=self.active['iterator']; wait=self.rq(p+0x18)
            assert self.rd(p+0x10)==1 and self.rd(wait+0x10)==0x3E99999A
            self.first_yields.append({'actor':self.active['actor'],'iterator':p,'snapshot':self.snapshot()})
            if not self.own_service('synchronous_first_yield_return',iterator=p): return
            uc.context_restore(ctx); self.ret(self.allocate('coroutine_handle:'+self.active['actor'])); return
        if r in self.new_decoded:
            if r==0x365D2F: assert self.reg(x.UC_X86_REG_RAX)==self.active['iterator']
            self.visited.add(r); return
        if self.phase not in ('ctor',) and self.active is None: return super().hook(uc,address,size,user)
        services={0x2B7D40:'allocate',0xB02160:'list_constructor',0x1C79770:'base_constructor',
                  0x2B6FF0:'barrier',0x2B7B40:'metadata',0x281D90:'class_init',
                  0x1C79FD0:'get_game_object',0x1C7D810:'set_active',0x1C82480:'unity_live',
                  0x1C822C0:'unity_null',0xF71C60:'concat',0x1C4B380:'context_log',0x1C4B450:'log',
                  0x282580:'box',0xF74DF0:'format',0x1C7F160:'start_coroutine',
                  0x603240:'clone_role',0x1C961F0:'wait_constructor'}
        assert r in services,(self.phase,hex(r))
        kind=services[r]; result=0
        a=self.ctor_actor if self.phase=='ctor' else self.active['actor_identity']; s=self.scene[a]
        details={}
        if kind=='allocate': details.update(object_type=self.metadata_names[c],result=self.cursor)
        elif kind=='barrier': assert self.rq(c)==t
        elif kind=='list_constructor': assert self.metadata_names[t]=='Method$System.Collections.Generic.List<ActedInfo>..ctor()'
        elif kind=='base_constructor': assert c==a and t==0
        elif kind=='get_game_object': assert c in (a,s['acted']) and t==0; result=c+0x800
        elif kind=='set_active': assert m==0 and t&255 in (0,1)
        elif kind in ('unity_live','unity_null'): assert c==t==0; result=int(kind=='unity_null')
        elif kind in ('concat','format','box'): result=s['text'] if kind!='box' else s['boxed']
        elif kind=='clone_role':
            assert c==self.rq(self.rq(a+0x50)+0x140) and t==self.names['Method$ClassConv.CreateCopyNonGeneric<Role>()']
            details.update(source=c,result=self.cursor)
        elif kind=='wait_constructor': assert m==0 and self.reg(x.UC_X86_REG_XMM1)&0xFFFFFFFF==0x3E99999A; details['seconds_f32_bits']=0x3E99999A
        elif kind=='start_coroutine':
            assert c==a and t==self.active['iterator'] and m==0 and self.rd(t+0x10)==0 and self.rq(t+0x20)==a
        if not self.own_service(kind,**details): return
        if kind=='allocate':
            name=self.metadata_names[c]; result=self.allocate(name+':'+str(len(self.new_services))); self.q(result,c)
            if name=='Character.<DelayReveal>d__84_TypeInfo':
                self.active['iterator']=result
                if not hasattr(self,'iterators'): self.iterators=[]
                self.iterators.append(result)
            else: assert name in ('System.Collections.Generic.List<ActedInfo>_TypeInfo','UnityEngine.WaitForSeconds_TypeInfo'),name
        elif kind=='list_constructor': self.write_list(c,[],0); result=0xC0DE000000000002
        elif kind=='class_init': self.d(c+0xE0,1)
        elif kind=='clone_role':
            result=self.allocate('clone:'+self.labels[a]); uc.mem_write(result,bytes(uc.mem_read(c,0x40))); self.active['clone']=result
        elif kind=='wait_constructor': self.d(c+0x10,0x3E99999A)
        elif kind=='start_coroutine':
            ctx=uc.context_save(); sp=self.reg(x.UC_X86_REG_RSP)-0x2010
            assert sp%16==8
            regs={z:self.reg(z) for z in self.nonvolatiles+[getattr(x,f'UC_X86_REG_XMM{i}') for i in range(6,16)]}
            self.frames.append((ctx,sp,regs)); self.q(sp,self.stop+0x300)
            for name in ('RAX','R10','R11','R8','R9'): uc.reg_write(getattr(x,'UC_X86_REG_'+name),0)
            for i in range(6): uc.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),0)
            uc.reg_write(x.UC_X86_REG_RSP,sp); uc.reg_write(x.UC_X86_REG_RCX,t); uc.reg_write(x.UC_X86_REG_RDX,0)
            uc.reg_write(x.UC_X86_REG_RIP,self.base+0x3756B0); return
        self.ret(result)


def audit(game_root,dumper_root):
    dependencies=['audit_first_village_initialization.py','audit_first_village_bluff_generation.py','audit_first_village_profile_generation.py',
                  'audit_manage_pool_composition.py','audit_character_assets.py','audit_ascension_assets.py','audit_report_snapshots.py',
                  'audit_round_candidate_composition.py','audit_spy.py']
    source_hashes={n:sha((Path(__file__).parent/n).read_bytes()) for n in dependencies}
    report_paths={name:ROOT/f'reports/{BUILD}_{name}.json' for name in
                  ('ascension_assets_audit','character_assets_audit','first_village_bluff_generation','character_init','character_constructor_init')}
    prior_hashes={name:sha(p.read_bytes()) for name,p in report_paths.items()}
    reports={name:json.loads(p.read_text(encoding='utf-8')) for name,p in report_paths.items()}
    from audit_ascension_assets import audit as profile_audit
    from audit_character_assets import audit as character_audit
    assert reports['ascension_assets_audit']==json.loads(json.dumps(profile_audit(Path(game_root),Path(dumper_root))))
    assert reports['character_assets_audit']==json.loads(json.dumps(character_audit(Path(game_root),Path(dumper_root))))
    inputs=load_inputs(Path(game_root),Path(dumper_root))
    m=InitializationWitness(inputs,reports['ascension_assets_audit'],reports['character_assets_audit'],reports['character_init'])
    constructors=[]
    for a in m.actors:
        result=m.run('Character.ctor',a)
        assert result['failure'] is None
        f=m.actor_snapshot(a)
        assert f['uses']==f['act']==1 and f['saved_act']==m.literals['']
        assert f['acted_infos']!=f['hover_infos'] and f['acted_infos_storage']['version']==f['hover_infos_storage']['version']==0
        constructors.append(result)
    setup=[]
    m.phase='setup'
    for name,this,arg,choices in [('GameData.SetupCurrentAscension',m.game,0,[]),
        ('AscensionsData.ClearCurrentPickedScript',m.temporary,0,[]),
        ('AscensionsData.SetupCharactersCount',m.temporary,0,[0]),
        ('AscensionsData.SetupStartingCharacters',m.temporary,0,[]),('Gameplay.GetCurrentScript',m.gameplay,0,[])]:
        result=m.run(name,this,arg,choices); assert result['failure'] is None; setup.append(result)
    m.q(m.gameplay_static+0x30,m.reg(m.x.UC_X86_REG_RAX))
    prior=reports['first_village_bluff_generation']; genrow=prior['generation_index_factor']['cases'][0]; poolrow=prior['pool_index_factors']['cases'][0]
    generation=m.run('Gameplay.GetRandomCharacters',m.gameplay,5,genrow['choices'])
    assert generation['final']['returned_order']==genrow['returned_order']==ORDER
    assert generation['sort_keys']==genrow['sort_keys']
    returned=m.reg(m.x.UC_X86_REG_RAX); prepared=m.save(); initial=m.snapshot()
    m.phase='manage'; m.new_services=[]
    joined=m.run('Characters.ManageCharacters',m.owner,returned,poolrow['choices'])
    assert joined['failure'] is None and joined['boundary']['completed_initializers']==5
    assert joined['final']['pools']==poolrow['pool_identities_and_contents']
    assert joined['final']['rosters']==[genrow['current_villagers'],[],[21596],[]]
    graph_keys=('source_starting','source_inline','temporary_starting','temporary_cache','selected_counts',
                'temporary_counts','rosters','saved_rosters','roster_identities','board_order','current_script_identity','current_script_fields')
    expected_graph={k:initial[k] for k in graph_keys}
    assert {k:joined['final'][k] for k in graph_keys}==expected_graph
    for i,row in enumerate(m.init_calls):
        f=row['after']; assert f['id']==5-i and f['data']==m.assets[ORDER[i]]
        assert f['alignment']==m.asset_fields[str(ORDER[i])]['startingAlignment']
        assert f['previous']==20 and f['state']==5 and f['uses']==1 and f['revealed']==f['started']==0
        assert f['bluff']==f['register_as']==f['trailer']==f['runtime']==0
        assert f['acted_infos_storage']['count']==0 and f['acted_infos_storage']['version']==1
        assert f['hover_infos_storage']['count']==f['hover_infos_storage']['version']==0
        assert f['status_storage']['count']==0 and f['status_storage']['version']==1
        assert f['status_storage']['active_values']==f['status_storage']['resistance_values']==[]
        assert f['saved_act']==m.literals[''] and f['act']==1 and f['role']==row['clone']
        assert {k:row['snapshot'][k] for k in graph_keys}==expected_graph
        assert row['snapshot']['pools']==poolrow['pool_identities_and_contents']
    assert len({row['clone'] for row in m.init_calls})==len(m.first_yields)==5
    final_calls=copy.deepcopy(m.init_calls); yields=copy.deepcopy(m.first_yields)
    stop_ordinals=list(m.new_services); stops=[]
    for ordinal in stop_ordinals:
        m.restore(prepared); m.phase='manage'; m.active=None; m.init_calls=[]; m.first_yields=[]; m.retained=[]; m.frames=[]; m.iterators=[]; m.new_services=[]
        result=m.run('Characters.ManageCharacters',m.owner,returned,poolrow['choices'],stop_service=ordinal,record_events=False)
        assert result['failure']=='service:'+joined['services'][ordinal-1]['service']
        assert [{k:v for k,v in e.items() if k!='snapshot'} for e in result['services']]==[{k:v for k,v in e.items() if k!='snapshot'} for e in joined['services'][:ordinal]]
        assert result['final']==joined['services'][ordinal-1]['snapshot']
        stops.append({'service_ordinal':ordinal,'service':joined['services'][ordinal-1]['service'],'completed_initializers':sum(r['completed'] for r in m.init_calls),'final':result['final']})
        if len(stops)%25==0: print('Verified new-service prefixes: '+str(len(stops)),flush=True)
    assert all(sha(p.read_bytes())==prior_hashes[name] for name,p in report_paths.items())
    assert all(sha((Path(__file__).parent/n).read_bytes())==source_hashes[n] for n in dependencies)
    return {'schema_version':'first_village_initialization_v1','build_id':BUILD,
            'scope':'Original N5 generation row0/poolrow0; five retained constructors, Manage Init and authored synchronous DelayReveal first steps; stop before publication. No queue drain, role action, captured UI or complete player history.',
            'domain':{'generation_row':0,'pool_row':0,'asset_order':ORDER,'initial_actor_state':20,'previous_gameplay_phase':0,'empty_active_statuses':True,
                      'services':'Inherited generation/CLR/RNG providers; scene hydration, base constructor, role clone, text/UI and synchronous StartCoroutine first-step providers.'},
            'prior_report_hashes':prior_hashes,'source_hashes':source_hashes,'retained_graph_invariants':expected_graph,
            'selected_instruction_assertions':[{'rva':hex(a),'mnemonic':v[0],'operands':v[1]} for a,v in m.pins.items()],
            'bodies':m.body_evidence+m.new_bodies,'constructors':constructors,'setup':setup,'generation':generation,
            'initial':initial,'joined':joined,'initializers':final_calls,'first_yields':yields,'stopped_prefixes':stops,
            'counters':{'constructor_count':len(constructors),'initializer_count':len(final_calls),'first_yield_count':len(yields),
                        'manage_service_count':len(joined['services']),'new_manage_service_count':len(stop_ordinals),'stopped_prefix_count':len(stops),
                        'native_instruction_count':len(m.visited),'selected_pin_count':len(m.pins),'python_source_hash_count':len(dependencies)}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--game-root',required=True); p.add_argument('--dumper-root',required=True); p.add_argument('--output',required=True)
    args=p.parse_args(); report=audit(args.game_root,args.dumper_root)
    assert report==json.loads(json.dumps(report))
    packed=pool_snapshots(report); assert expand_snapshots(json.loads(json.dumps(packed)))==report
    Path(args.output).write_text(json.dumps(packed,indent=2,sort_keys=True,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':Path(args.output).name,**report['counters']},sort_keys=True),flush=True)


if __name__=='__main__': main()
