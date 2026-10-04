"""Retained original N5 Manage return with conditional null onSetup and Shuffle wait.

Native bytes remain private. Scene/runtime creation providers stay explicit;
paused/reentered services are not independent failure fixtures.
"""
import argparse
import copy
import inspect
import json
import re
import struct
from pathlib import Path

from audit_first_village_start_queue import (
    BUILD, ORDER, START_ORDER, GRAPH_KEYS, ROOT, Cursor, sha, cpu,
    first_difference, scene_order, load_inputs, StartQueueWitness,
    MultiOwnerEngine, ScheduledEngine, NativeTree, validate_tree,
    AbortedPrefix, ENGINE_SHA256, verify_fingerprint,
)
from audit_report_snapshots import pool_snapshots, expand_snapshots

ACQUISITION_BITS = 0x3E99999A
SHUFFLE_BITS = 0x3F000000


class ShuffleWitness(StartQueueWitness):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        import capstone
        self.q(self.owner+0x58, 0)  # Explicit conditional runtime binding.
        self.tail = False
        self.before_on_setup = None
        self.shuffle_iterator = 0
        self.shuffle_first_yield = None
        self.prior_five_storage = []
        self.return_cpu = None
        self.entry_cpu = None
        declaration = re.search(r'^private sealed class Characters\.<ShuffleDeck>d__16 :[^\n]*\n\{(.*?)\n\}', self.dump, re.M|re.S)
        assert declaration
        assert 'private int <>1__state; // 0x10' in declaration[1]
        assert 'private object <>2__current; // 0x18' in declaration[1]
        assert '<>4__this' not in declaration[1]
        start = 0x376B00
        rows = [r for r in self.meta['ScriptMethod'] if r['Address']==start and r['Name']=='Characters.<ShuffleDeck>d__16$$MoveNext']
        assert len(rows)==1
        assert rows[0]['Signature']=='bool Characters__ShuffleDeck_d__16__MoveNext (Characters__ShuffleDeck_d__16_o* __this, const MethodInfo* method);'
        end = min(r['Address'] for r in self.meta['ScriptMethod'] if r['Address']>start)
        raw = self.read_file_backed(start, end-start)
        ins = list(self.cs.disasm(raw, start))
        assert sum(i.size for i in ins)==len(raw)
        while ins[-1].mnemonic=='int3': ins.pop()
        assert ins[0].address==start and all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]))
        self.shuffle_decoded = {i.address for i in ins}
        self.instructions.update({i.address:i for i in ins})
        self.shuffle_body = {'name':rows[0]['Name'], 'signature':rows[0]['Signature'], 'rva':hex(start),
                             'end_rva':hex(ins[-1].address+ins[-1].size),
                             'body_sha256':sha(raw[:ins[-1].address+ins[-1].size-start])}
        slots = {r['Address']:r for key in ('ScriptMetadata','ScriptMetadataMethod','ScriptString') for r in self.meta[key]}
        for i in ins:
            for op in i.operands:
                if op.type!=capstone.CS_OP_MEM or op.mem.base!=capstone.x86.X86_REG_RIP: continue
                address=i.address+i.size+op.mem.disp
                if address in slots:
                    row=slots[address]; name=row.get('Name','literal:'+row.get('Value',''))
                    if name not in self.names:
                        p=self.allocate(name); self.names[name]=p; self.metadata_names[p]=name; self.d(p+0xE0,1)
                    self.q(self.base+address,self.names[name])
                elif i.mnemonic=='cmp' and op.size==1:
                    self.uc.mem_write(self.base+address,b'\1')
        load=self.instructions[0x376B68]
        literal=load.address+load.size+load.operands[1].mem.disp
        assert struct.unpack('<I',self.read_file_backed(literal,4))[0]==SHUFFLE_BITS
        self.shuffle_literal={'load_rva':hex(load.address),'literal_rva':hex(literal),'duration_bits':SHUFFLE_BITS}
        self.tail_pins={0x36D2DB:('mov','rax, qword ptr [r12 + 0x58]'),
                        0x36D2ED:('call','qword ptr [rax + 0x18]'),
                        0x36D313:('call','0x2b7d40'),0x36D320:('call','0x33ed50'),
                        0x36D325:('mov','dword ptr [rbx + 0x10], 0'),
                        0x36D332:('mov','rcx, r12'),0x36D335:('call','0x1c7f160'),
                        0x36D356:('ret',''),0x376B55:('mov','dword ptr [rdi + 0x10], 0xffffffff'),
                        0x376B79:('call','0x1c961f0'),0x376B88:('call','0x2b6ff0'),
                        0x376B94:('mov','dword ptr [rdi + 0x10], 1')}
        for a,wanted in self.tail_pins.items():
            assert (self.instructions[a].mnemonic,self.instructions[a].op_str)==wanted

    def read_file_backed(self,rva,size):
        section=self.pe.get_section_by_rva(rva)
        assert section is not None and section.VirtualAddress<=rva
        assert rva+size<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(rva,size); assert len(raw)==size
        return raw

    def snapshot(self):
        out=super().snapshot()
        if hasattr(self,'tail'):
            p=self.shuffle_iterator; wait=self.rq(p+0x18) if p else 0
            out['conditional_on_setup']={'owner':self.owner,'identity':self.rq(self.owner+0x58),
                                         'contract':'supplied runtime null; original binding unestablished'}
            out['shuffle_continuation']=None if not p else {'identity':p,'class':self.rq(p),
                'state':self.rd(p+0x10),'current':wait,'wait_bits':self.rd(wait+0x10) if wait else None}
            out['prior_five_storage']=[{'identity':p,'size':len(raw),'sha256':sha(bytes(self.uc.mem_read(p,len(raw))))}
                                       for p,raw in self.prior_five_storage]
        return out

    def capture_prior_five(self):
        regions={p:len(raw) for group in self.retained for p,raw in group}
        regions.update({a:0x1B8 for a in self.actors})
        for a in self.actors:
            clone=self.rq(a+0x168); regions[clone]=0x48
            callback=self.rq(clone+0x28)
            if callback: regions[callback]=0x48
        for p in self.iterators:
            regions[p]=0x28; regions[self.rq(p+0x18)]=0x20
        regions[self.publication_list]=0xC20
        regions[self.owner]=0x60
        self.prior_five_storage=[(p,bytes(self.uc.mem_read(p,n))) for p,n in regions.items()]

    def preserve(self):
        super().preserve()
        for p,raw in getattr(self,'prior_five_storage',[]):
            assert bytes(self.uc.mem_read(p,len(raw)))==raw,('post-ordered retention',self.labels.get(p),self.phase)
        if hasattr(self,'tail'): assert self.rq(self.owner+0x58)==0

    def step(self,iterator):
        if iterator!=self.shuffle_iterator: return super().step(iterator)
        assert self.tail and self.active is None and self.rd(iterator+0x10)==0
        context=self.uc.context_save(); caller=cpu(self); sp=caller['RSP']-0x2010
        assert sp%16==8
        end=self.stop+0x900; self.q(sp,end); self.uc.mem_write(self.bridge_out,b'\x7f')
        for n in ('RAX','R10','R11','R8','R9'): self.uc.reg_write(getattr(self.x,'UC_X86_REG_'+n),0)
        for i in range(6): self.uc.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),0)
        self.uc.reg_write(self.x.UC_X86_REG_RSP,sp)
        self.uc.reg_write(self.x.UC_X86_REG_RCX,iterator); self.uc.reg_write(self.x.UC_X86_REG_RDX,self.bridge_out)
        prior=self.active_machine; self.active_machine=self
        try:
            self.drive_managed(self.base+0x1C8A780,end)
            if self.failure: raise AbortedPrefix()
            assert self.reg(self.x.UC_X86_REG_RIP)==end and self.reg(self.x.UC_X86_REG_RSP)==sp+8
            after=cpu(self)
            assert all(after[n]==caller[n] for n in ('RBX','RBP','RSI','RDI','R12','R13','R14','R15')+tuple('XMM'+str(i) for i in range(6,16)))
            assert self.uc.mem_read(self.bridge_out,1)==b'\1'
            assert self.rd(iterator+0x10)==1 and self.rd(self.rq(iterator+0x18)+0x10)==SHUFFLE_BITS
            self.shuffle_first_yield={'iterator':iterator,'owner':self.owner,'entry_cpu':caller,'return_cpu':after,'snapshot':self.snapshot()}
            return 1
        finally:
            self.active_machine=prior
            if not self.failure: self.uc.context_restore(context); assert cpu(self)==caller

    def hook(self,uc,address,size,user):
        r=address-self.base; x=self.x
        if r==self.entries['Characters.ManageCharacters']:
            self.entry_cpu=cpu(self)
        if r==0x36D2DB:
            assert self.before_start is not None and len(self.scan)==75
            assert self.reg(x.UC_X86_REG_RSP)==self.pre_sp and self.rq(self.stack+0x10008)==self.stop
            self.preserve(); assert self.snapshot()==self.before_start
            self.before_on_setup=self.snapshot(); self.capture_prior_five()
            self.tail=True; self.phase='shuffle_admission'; self.visited.add(r); return
        if self.tail:
            c,t,m=[self.reg(z) for z in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8)]
            if r==0x36D2ED: raise AssertionError('conditional null onSetup invoked a subscriber')
            if r==0x36D356:
                self.preserve(); assert self.reg(x.UC_X86_REG_RSP)==self.stack+0x10008
                assert self.rq(self.reg(x.UC_X86_REG_RSP))==self.stop
                after=cpu(self)
                assert all(after[n]==self.entry_cpu[n] for n in ('RBX','RBP','RSI','RDI','R12','R13','R14','R15')+tuple('XMM'+str(i) for i in range(6,16)))
                self.return_cpu=after; self.visited.add(r); return
            if r==0x1C7F160:
                assert c==self.owner and t==self.shuffle_iterator and m==0 and self.rd(t+0x10)==0
                if self.service('shuffle_start_coroutine',owner=c,iterator=t):
                    self.pending_start=(c,t); uc.emu_stop()
                return
            if r==0x4060:
                assert c==0 and t==self.bridge_type and m==self.shuffle_iterator
                if self.service('shuffle_ienumerator_slot_zero',slot=0,iterator=m,method_rva='0x376b00'):
                    for n,v in [('RCX',m),('RDX',0),('R8',0),('R9',0)]: uc.reg_write(getattr(x,'UC_X86_REG_'+n),v)
                    uc.reg_write(x.UC_X86_REG_RIP,self.base+0x376B00)
                return
            if r==0x2B7D40:
                name=self.metadata_names[c]
                assert name in ('Characters.<ShuffleDeck>d__16_TypeInfo','UnityEngine.WaitForSeconds_TypeInfo')
                if self.service('shuffle_allocate',object_type=name,result=self.cursor):
                    p=self.allocate(name+':shuffle'); self.q(p,c)
                    if name.startswith('Characters.'): assert not self.shuffle_iterator; self.shuffle_iterator=p
                    self.ret(p)
                return
            if r==0x1C961F0:
                assert c and self.rq(c)==self.names['UnityEngine.WaitForSeconds_TypeInfo'] and m==0
                assert self.reg(x.UC_X86_REG_XMM1)&0xFFFFFFFF==SHUFFLE_BITS
                if self.service('shuffle_wait_constructor',identity=c,seconds_f32_bits=SHUFFLE_BITS):
                    self.d(c+0x10,SHUFFLE_BITS); self.ret()
                return
            if r==0x2B6FF0:
                assert c==self.shuffle_iterator+0x18 and self.rq(c)==t
                if self.service('shuffle_current_barrier',destination=c,value=t): self.ret()
                return
            if r in self.shuffle_decoded or r==0x33ED50:
                self.visited.add(r); return
        return super().hook(uc,address,size,user)


class ShuffleEngine(MultiOwnerEngine):
    def __init__(self,data,managed):
        super().__init__(data,managed)
        p=self.arena+0x180000+5*0x1000
        row={'managed_actor':managed.owner,'native_owner':p,'key':106}
        self.owner_bindings[managed.owner]=row
        self.write_q(p+0x40,self.owner_vtable); self.write_d(p+8,row['key'])
        self.write_q(p+0x70,p+0x70); self.write_q(p+0x78,p+0x70)

    def queue_state(self):
        out=ScheduledEngine.queue_state(self)
        if not hasattr(self,'owner_bindings'): return out
        reverse={p:i for i,p in self.nodes.items()}; order=[]
        def walk(p):
            if p==self.head: return
            assert p in reverse
            walk(self.qword(p)); order.append(reverse[p]); walk(self.qword(p+0x10))
        walk(self.qword(self.head+8)); assert order==[r['id'] for r in out['entries']]
        for node,cached in self.records.items():
            live=bytes(self.uc.mem_read(node+0x20,0x40)); assert live==cached
            payload=struct.unpack_from('<Q',live,0x18)[0]; row=self.payload_registry[payload]
            original=self.managed.rq(row['pointer']+0x18); bits=self.managed.rd(original+0x10)
            assert bits==(SHUFFLE_BITS if row['kind']=='shuffle' else ACQUISITION_BITS)
            duration=struct.unpack('<f',struct.pack('<I',bits))[0]
            assert struct.unpack_from('<dq',live,0)==(1.0+duration,8)
            assert struct.unpack_from('<QQ',live,0x20)==(self.base+0x778B30,self.base+0x778BD0)
            assert struct.unpack_from('<III',live,0x30)==(row['key'],0xA,0)
            assert self.gc_targets[self.qword(payload+0x10)]==self.qword(payload+0x20)==row['pointer']
            assert self.qword(payload+0x58)==self.owner_bindings[row['actor']]['native_owner']
        for mirror,original in self.wait_mirrors.items():
            assert self.qword(mirror)==self.wait_class
            assert bytes(self.uc.mem_read(mirror+8,0x18))==bytes(self.managed.uc.mem_read(original+8,0x18))
        if self.pending_insert is None: validate_tree(self.uc.mem_read,self.container,self.head,self.records,[0]*len(order))
        out['actual_tree_order']=order
        return out

    def _on_code(self,uc,address,size,data):
        if not hasattr(self,'owner_bindings'): return super()._on_code(uc,address,size,data)
        r=address-self.base; m=self.managed; x=self.x86
        if r==0x779070:
            self.executed.add(r)
            if not m.service('engine_current_yield'): return
            payload=uc.reg_read(x.UC_X86_REG_RCX); row=self.payload_registry[payload]
            original=m.rq(row['pointer']+0x18); bits=m.rd(original+0x10)
            assert bits==(SHUFFLE_BITS if row['kind']=='shuffle' else ACQUISITION_BITS)
            mirror=self.arena+0x1A0000+len(self.wait_mirrors)*0x100
            self.wait_mirrors[mirror]=original; uc.mem_write(mirror,bytes(m.uc.mem_read(original,0x20))); self.write_q(mirror,self.wait_class)
            self.join_trace.append({'kind':'current_yield_gateway','iterator':row['label'],'wait':original,'duration_bits':bits})
            uc.reg_write(x.UC_X86_REG_RCX,payload); uc.reg_write(x.UC_X86_REG_RDX,mirror); uc.reg_write(x.UC_X86_REG_RIP,self.base+0x779370); return
        if r==0x440F00:
            self.executed.add(r)
            record=bytes(uc.mem_read(uc.reg_read(x.UC_X86_REG_R8),0x40)); payload=struct.unpack_from('<Q',record,0x18)[0]
            row=self.payload_registry[payload]; bits=m.rd(m.rq(row['pointer']+0x18)+0x10)
            assert struct.unpack_from('<I',record,0x30)[0]==row['key']
            assert struct.unpack_from('<QQ',record,0x20)==(self.base+0x778B30,self.base+0x778BD0)
            self.pending_insert={'id':self.next_identity,'record':record,'node':self.next_node,'kind':row['kind'],'iterator':row['label'],
                                 'producer':{'time':self.producer_time,'frame':self.producer_frame,'duration_bits':bits}}
            self.next_identity+=1; NativeTree._on_code(self,uc,address,size,data); return
        return super()._on_code(uc,address,size,data)

    def start_owner(self,actor,iterator):
        m=self.managed; row=self.owner_bindings[actor]; self.native_owner=row['native_owner']
        kind='shuffle' if actor==m.owner else 'acquisition'
        assert (iterator==m.shuffle_iterator)==(kind=='shuffle')
        prior=m.active_machine; m.active_machine=m; m.direct_gateway=True
        try:
            if not m.service('native_record_creation',actor=actor,iterator=iterator,native_owner=self.native_owner,owner_key=row['key'],kind=kind): raise AbortedPrefix()
        finally: m.direct_gateway=False; m.active_machine=prior
        payload=self.arena+0x1B0000+len(self.registry)*0x100
        self.uc.mem_write(payload,bytes(0x88)); label=m.labels[iterator]
        rec={'pointer':iterator,'payload':payload,'kind':kind,'label':label,'actor':actor,'key':row['key']}
        self.registry[iterator]=rec; self.payload_registry[payload]=rec; self.payload_labels[payload]=label
        handle=17+len(self.registry); self.gc_targets[handle]=iterator
        for off,val in ((0x10,handle),(0x20,iterator),(0x58,self.native_owner)): self.write_q(payload+off,val)
        self.write_d(payload+0x18,2); self.write_d(payload+0x60,1)
        head=self.native_owner+0x70; last=self.qword(head+8)
        self.write_q(payload,head); self.write_q(payload+8,last); self.write_q(last,payload); self.write_q(head+8,payload)
        self.run_native('dispatch',payload,0)
        assert struct.unpack('<i',self.uc.mem_read(payload+0x60,4))[0]==2
        self.run_native('release_native',payload)
        assert struct.unpack('<i',self.uc.mem_read(payload+0x60,4))[0]==1
        wrapper=m.allocate('native_coroutine_handle:'+m.labels[actor]); m.q(wrapper+0x10,payload)
        self.completed_storage.extend([(payload,bytes(self.uc.mem_read(payload,0x88))),
                                       (self.native_owner,bytes(self.uc.mem_read(self.native_owner,0x100)))])
        return wrapper


def audit(game_root,dumper_root):
    game_root=Path(game_root); dumper_root=Path(dumper_root); inputs=load_inputs(game_root,dumper_root)
    names=('ascension_assets_audit','character_assets_audit','first_village_bluff_generation','character_init',
           'first_village_role_setup','first_village_publication_init','hunter_scheduled_publication','first_village_start_queue')
    paths={n:ROOT/f'reports/{BUILD}_{n}.json' for n in names}
    prior_hashes={n:sha(p.read_bytes()) for n,p in paths.items()}
    reports={n:json.loads(p.read_text(encoding='utf-8')) for n,p in paths.items()}
    reports={n:expand_snapshots(r) if 'snapshot_encoding' in r else r for n,r in reports.items()}
    from audit_ascension_assets import audit as profile_audit
    from audit_character_assets import audit as asset_audit
    assert reports['ascension_assets_audit']==json.loads(json.dumps(profile_audit(game_root,dumper_root)))
    assert reports['character_assets_audit']==json.loads(json.dumps(asset_audit(game_root,dumper_root)))
    original=scene_order(game_root,inputs,reports['character_assets_audit'])
    assert original==reports['first_village_start_queue']['original_scene_order']
    engine_data=(game_root/'UnityPlayer.dll').read_bytes(); verify_fingerprint(engine_data,ENGINE_SHA256)
    source_paths={Path(inspect.getfile(c)) for c in ShuffleWitness.__mro__[:-1]+ShuffleEngine.__mro__[:-1]}
    source_paths.update([Path(__file__),Path(inspect.getfile(Cursor)),Path(inspect.getfile(pool_snapshots)),
                         Path(inspect.getfile(load_inputs)),Path(inspect.getfile(profile_audit)),Path(inspect.getfile(verify_fingerprint))])
    source_hashes={p.relative_to(ROOT.parent).as_posix():sha(p.read_bytes()) for p in sorted(source_paths)}
    prior=reports['first_village_bluff_generation']; genrow=prior['generation_index_factor']['cases'][0]; poolrow=prior['pool_index_factors']['cases'][0]

    def run(paused=False,abort=None):
        m=ShuffleWitness(inputs,reports['ascension_assets_audit'],reports['character_assets_audit'],reports['character_init'],reports['first_village_role_setup'],original_order=original)
        constructors=[m.run('Character.ctor',a) for a in m.actors]; assert all(not r['failure'] for r in constructors)
        m.phase='setup'; setup=[]
        for name,this,arg,choices in [('GameData.SetupCurrentAscension',m.game,0,[]),('AscensionsData.ClearCurrentPickedScript',m.temporary,0,[]),('AscensionsData.SetupCharactersCount',m.temporary,0,[0]),('AscensionsData.SetupStartingCharacters',m.temporary,0,[]),('Gameplay.GetCurrentScript',m.gameplay,0,[])]:
            row=m.run(name,this,arg,choices); assert not row['failure']; setup.append(row)
        m.q(m.gameplay_static+0x30,m.reg(m.x.UC_X86_REG_RAX))
        generation=m.run('Gameplay.GetRandomCharacters',m.gameplay,5,genrow['choices']); assert generation['final']['returned_order']==ORDER
        returned=m.reg(m.x.UC_X86_REG_RAX); e=ShuffleEngine(engine_data,m); e.set_producer(1.0,7)
        m.phase='manage'; m.pause_mode=paused; initial=m.snapshot()
        joined=m.run('Characters.ManageCharacters',m.owner,returned,poolrow['choices'],stop_service=abort)
        if abort is not None: return m,e,{'joined':joined}
        assert joined['failure'] is None and joined['boundary'] is None
        assert m.return_cpu and m.reg(m.x.UC_X86_REG_RIP)==m.stop and m.reg(m.x.UC_X86_REG_RSP)==m.stack+0x10010
        assert len(m.init_calls)==len(m.first_yields)==len(m.actions)==5
        assert len(e.registry)==len(e.owner_bindings)==len(e.nodes)==6 and m.shuffle_first_yield
        assert m.scan==[[a,b] for a in START_ORDER for b in ORDER]
        assert {k:joined['final'][k] for k in GRAPH_KEYS}=={k:initial[k] for k in GRAPH_KEYS}
        assert joined['final']['pools']==poolrow['pool_identities_and_contents']
        prior_keys=('actors','continuations','runtime_roles','source_role_saved_fields','publication')
        assert {k:joined['final'][k] for k in prior_keys}=={k:m.before_on_setup[k] for k in prior_keys}
        assert all(r['reference_count']==1 and r['owner_linked'] and r['gc_handle'] and r['cached_enumerator']==r['iterator'] for r in e.retained_native_records())
        assert all(len(row['payloads'])==1 for row in e.owner_snapshot())
        assert e.queue_state()['actual_tree_order']==list(range(6))
        assert all(r['state']==1 and r['wait_bits']==ACQUISITION_BITS for r in joined['final']['continuations'])
        assert joined['final']['shuffle_continuation']['state']==1 and joined['final']['shuffle_continuation']['wait_bits']==SHUFFLE_BITS
        if paused: assert m.reentries==len(joined['services'])==len(m.prefixes) and not m.reentry and not m.paused
        admissions=[]
        for event in e.join_trace:
            if event['kind']!='native_wait_inserted': continue
            i=event['id']; node=e.nodes[i]; record=bytes(e.uc.mem_read(node+0x20,0x40)); assert record==e.records[node]
            payload=struct.unpack_from('<Q',record,0x18)[0]; row=e.payload_registry[payload]
            bits=SHUFFLE_BITS if row['kind']=='shuffle' else ACQUISITION_BITS
            assert event['producer']=={'time':1.0,'frame':7,'duration_bits':bits}
            assert struct.unpack_from('<d',record,0)[0]==1.0+struct.unpack('<f',struct.pack('<I',bits))[0]
            assert struct.unpack_from('<q',record,8)[0]==8 and struct.unpack_from('<I',record,0x38)[0]==0
            admissions.append({'id':i,'kind':row['kind'],'iterator':row['pointer'],'iterator_label':row['label'],'actor':row['actor'],
                'payload':payload,'native_owner':e.qword(payload+0x58),'owner_key':row['key'],
                'producer':event['producer'],'deadline':struct.unpack_from('<d',record,0)[0],
                'frame_threshold':8,'generation':0,'queue':event['queue']})
        return m,e,{'constructors':constructors,'setup':setup,'generation':generation,'initial':initial,
            'pre_publication':m.pre_publication,'before_start':m.before_start,'before_on_setup':m.before_on_setup,
            'initializers':m.init_calls,'first_yields':m.first_yields,'act_init_returns':m.actions,
            'shuffle_first_yield':m.shuffle_first_yield,'joined':joined,'ordered_comparisons':m.scan,'engine_admissions':admissions,
            'completion':{'returned':True,'entry_cpu':m.entry_cpu,'before_return_cpu':m.return_cpu,
                          'final_cpu':{'managed':cpu(m),'engine':cpu(e)},'root_return_sentinel':m.stop}}

    normal_m,normal_e,normal=run(); paused_m,paused_e,paused=run(True)
    assert normal==paused,first_difference(normal,paused)
    services=normal['joined']['services']; representatives={}
    for i,event in enumerate(services,1):
        if event['phase']=='shuffle_admission': representatives.setdefault(event['service'],i)
    aborts=[]
    for service,ordinal in representatives.items():
        m,e,result=run(abort=ordinal); joined=result['joined']
        assert joined['failure']=='service:'+service and joined['services']==services[:ordinal]
        assert joined['final']==services[ordinal-1]['snapshot']
        aborts.append({'service_ordinal':ordinal,'service':service,'final':joined['final']})
    assert all(sha(p.read_bytes())==prior_hashes[n] for n,p in paths.items())
    assert all(sha(p.read_bytes())==source_hashes[p.relative_to(ROOT.parent).as_posix()] for p in source_paths)
    return {'schema_version':'first_village_shuffle_admission_v1','build_id':BUILD,
        'decision':'Same original N5 retained Manage returns normally after actual Shuffle first wait and sixth queue admission.',
        'domain':{'generation_row':0,'pool_row':0,'asset_order':ORDER,'producer_time':1.0,'producer_signed_frame':7,'generation':0,
                  'on_setup':'Explicit supplied runtime null; original runtime subscriber binding remains unestablished.',
                  'exit':'normal Manage return at 0x36d356','scene_actor_initial_state':20},
        'exclusions':['original runtime onSetup absence','subscriber effects','Shuffle state1/Gameplay.ShuffleDeck/events',
                      'acquisition resumes/queue drains','rendered/player observation','API failure effects/managed exceptions',
                      'equivalence of every aborted continuation'],
        'source_hashes':source_hashes,'prior_report_hashes':prior_hashes,'original_scene_order':original,
        'ordered_source_bindings':normal_m.ordered_source_bindings,'class_metadata_aliases':normal_m.class_metadata_aliases,
        'shuffle_literal':normal_m.shuffle_literal,
        'selected_instruction_assertions':[{'rva':hex(a),'mnemonic':v[0],'operands':v[1]} for a,v in normal_m.tail_pins.items()],
        'bodies':normal_m.body_evidence+normal_m.new_bodies+normal_m.post_bodies+normal_m.bridge_bodies+[normal_m.shuffle_body],
        **normal,'paused_reentered_prefixes':paused_m.prefixes,'restart_aborted_prefixes':aborts,
        'counters':{'constructors':len(normal['constructors']),'initializers':len(normal_m.init_calls),'acquisition_first_yields':len(normal_m.first_yields),
                    'shuffle_first_yields':int(normal_m.shuffle_first_yield is not None),'act_init_calls':len(normal_m.actions),
                    'start_calls':0,'on_setup_calls':0,'ordered_entries':len(START_ORDER),'ordered_comparisons':len(normal_m.scan),
                    'queue_records':len(normal_e.nodes),'physical_owners':len(normal_e.owner_bindings),
                    'manage_services':len(services),'paused_reentered_prefixes':len(paused_m.prefixes),'reentry_tokens_consumed':paused_m.reentries,
                    'tail_services':sum(e['phase']=='shuffle_admission' for e in services),'restart_aborted_prefixes':len(aborts),
                    'managed_native_addresses':len(normal_m.visited),'engine_native_addresses':len(normal_e.executed),
                    'python_source_hashes':len(source_hashes)}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--game-root',required=True); p.add_argument('--dumper-root',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); output=Path(a.output); assert output.parent.is_dir()
    report=audit(a.game_root,a.dumper_root); assert report==json.loads(json.dumps(report))
    packed=pool_snapshots(report); assert expand_snapshots(json.loads(json.dumps(packed)))==report
    output.write_text(json.dumps(packed,indent=2,sort_keys=True,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':Path(a.output).name,**report['counters']},sort_keys=True),flush=True)


if __name__=='__main__': main()
