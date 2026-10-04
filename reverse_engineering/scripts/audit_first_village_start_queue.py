"""Original N5 ordered Start scan and retained multiowner wait admission.

Native bytes remain private. Pre-service pause/reentry is distinguished from
restart-based aborts; runtime/CLR/scene/creation gateways remain supplied.
"""
import argparse
import copy
import inspect
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD, Cursor
from audit_first_village_initialization import ORDER, sha
from audit_first_village_publication_init import PublicationInitWitness
from audit_first_village_bluff_generation import BluffJoin
from audit_first_village_profile_generation import ROOT, load_inputs
from audit_hunter_scheduled_publication import ScheduledEngine
from audit_report_snapshots import pool_snapshots, expand_snapshots
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint
from audit_unityplayer_wait_tree import NativeTree, validate_tree

START_ORDER = [21594,21593,21597,21605,21602,21595,21599,21590,21606,21600,21609,21634,21598,21607,21591]
GRAPH_KEYS = ('source_starting','source_inline','temporary_starting','temporary_cache','selected_counts',
              'temporary_counts','rosters','saved_rosters','roster_identities','board_order',
              'current_script_identity','current_script_fields','asset_bindings')


class AbortedPrefix(Exception): pass


def first_difference(a,b,path='root'):
    if type(a)!=type(b): return path+':type'
    if isinstance(a,dict):
        if a.keys()!=b.keys(): return path+':keys'
        for k in a:
            out=first_difference(a[k],b[k],path+'/'+str(k))
            if out:return out
    elif isinstance(a,list):
        if len(a)!=len(b):return path+':length'
        for i,(x,y) in enumerate(zip(a,b)):
            out=first_difference(x,y,path+'/'+str(i))
            if out:return out
    elif a!=b:
        return path+':'+str(a)[:80]+' != '+str(b)[:80]
    return None


def cpu(machine):
    u=machine.uc; x=machine.x86 if hasattr(machine,'x86') else machine.x
    names=['RIP','RSP','RAX','RCX','RDX','R8','R9','R10','R11','RBX','RBP','RSI','RDI','R12','R13','R14','R15','EFLAGS','MXCSR']
    names += ['XMM'+str(i) for i in range(16)]
    return {n:u.reg_read(getattr(x,'UC_X86_REG_'+n)) for n in names}


def scene_order(game_root,inputs,characters):
    import UnityPy
    raw=(Path(game_root)/'Demon Bluff_Data/level0').read_bytes()
    assert sha(raw).upper()==inputs[-1]['inputs']['level0']['sha256']
    env=UnityPy.load(str(Path(game_root)/'Demon Bluff_Data/level0'))
    file=next(iter(env.files.values()))
    assert [x.path for x in file.externals]==['globalgamemanagers.assets','sharedassets0.assets','Library/unity default resources']
    obj=next(o for o in env.objects if o.path_id==137026)
    data=obj.get_raw_data(); digest=sha(data)
    assert len(data)==332 and digest.upper()=='544328634CD77D551B5864CDC1B643029F3B30BFFC5BB4350DFCF83C66226BB0'
    assert obj.read_typetree(check_read=False)['m_Script']=={'m_FileID':1,'m_PathID':1528}
    manager=UnityPy.load(str(Path(game_root)/'Demon Bluff_Data/globalgamemanagers.assets'))
    script=next(o for o in manager.objects if o.path_id==1528).read_typetree()
    assert script['m_ClassName']=='Characters' and script['m_AssemblyName']=='Assembly-CSharp' and script['m_Namespace']==''
    block=re.search(r'^public class Characters : MonoBehaviour //[^\n]*\n\{(.*?)\n\}',inputs[2],re.M|re.S)[1]
    assert all(text in block for text in ('public List<Character> characters; // 0x20','public CharacterData[] startGameActOrder; // 0x28','public Action onSetup; // 0x58'))
    c=Cursor(data); name=c.string()
    value={'name':name,'characters':c.array(c.pointer),'start_game_act_order':c.array(c.pointer),
           'character_pool':c.array(c.pointer),'current_pool':c.pointer(),
           'unique_pool':c.array(c.pointer),'duplicates_pool':c.array(c.pointer),'bluff_must_include':c.array(c.pointer)}
    assert c.offset==len(data) and value['start_game_act_order']==[(2,i) for i in START_ORDER]
    assert not set(ORDER)&set(START_ORDER)
    records={r['path_id']:r for r in characters['records']}
    return {'file_sha256':sha(raw),'path_id':137026,'script_reference':[1,1528],
            'script_binding':{'class_name':script['m_ClassName'],'namespace':script['m_Namespace'],'assembly':script['m_AssemblyName']},
            'object_sha256':digest,'object_size':len(data),'consumed_bytes':c.offset,
            'serialized_fields':json.loads(json.dumps(value)),
            'ordered_role_bindings':[{k:records[i][k] for k in ('path_id','name','role_type','role_rid','object_sha256')} for i in START_ORDER]}


class StartQueueWitness(PublicationInitWitness):
    def __init__(self,*args,original_order,**kwargs):
        super().__init__(*args,**kwargs)
        self.original_order=original_order
        records={r['path_id']:r for r in args[2]['records']}
        classes={r['role_type']:r['source_class'] for r in self.asset_fields.values()}
        classes['Role']=self.allocate('source_class:Role')
        self.ordered_source_bindings=[]; self.hierarchy_arrays=[]
        for ident in START_ORDER:
            row=records[ident]; name=row['role_type']
            if name not in classes: classes[name]=self.allocate('source_class:'+name)
            source=self.allocate('source_role:'+str(ident)); self.q(source,classes[name])
            p=self.assets[ident]; self.q(p+0x140,source)
            self.d(p+0x138,row['abilityUsage']); self.uc.mem_write(p+0x13E,bytes([row['picking']]))
            self.asset_fields[str(ident)]={k:row[k] for k in ('name','characterName','type','startingAlignment','abilityUsage','picking','role_rid','role_type')}
            self.asset_fields[str(ident)].update(identity=p,source_role=source,source_class=classes[name],name_identity=self.rq(p+0x28))
        for ident in ORDER+START_ORDER:
            binding=self.asset_fields[str(ident)]; name=binding['role_type']; chain=[]
            while True:
                chain.append(name)
                if name=='Role': break
                matches=re.findall(r'^public class '+re.escape(name)+r' : (\w+)(?:[^\n]*) // TypeDefIndex: \d+$',self.dump,re.M)
                assert len(matches)==1,(name,matches)
                name=matches[0]
                if name not in classes: classes[name]=self.allocate('source_class:'+name)
                assert len(chain)<=16
            chain.reverse()
            binding['ancestors']=[{'managed_role':n,'identity':classes[n]} for n in chain]
            # Explicit runtime class-hierarchy provider, installed before calls.
            klass=binding['source_class']; hierarchy=self.allocate('class_hierarchy:'+str(ident))
            for i,n in enumerate(chain): self.q(hierarchy+i*8,classes[n])
            self.q(klass+0xC8,hierarchy); self.uc.mem_write(klass+0x130,bytes([len(chain)]))
            binding['hierarchy_identity']=hierarchy
            self.hierarchy_arrays.append(hierarchy)
            if ident in START_ORDER:
                self.ordered_source_bindings.append({'asset_id':ident,**copy.deepcopy(binding)})
        root_hierarchy=self.allocate('class_hierarchy:Role'); self.q(root_hierarchy,classes['Role'])
        self.q(classes['Role']+0xC8,root_hierarchy); self.uc.mem_write(classes['Role']+0x130,b'\1')
        self.hierarchy_arrays.append(root_hierarchy)
        self.class_metadata_aliases=[]
        for row in self.meta['ScriptMetadata']:
            name=row['Name']; managed=name[:-9] if name.endswith('_TypeInfo') else None
            if managed not in classes: continue
            klass=classes[managed]; self.names[name]=klass; self.metadata_names[klass]=name
            self.q(self.base+row['Address'],klass); self.d(klass+0xE0,1)
            self.class_metadata_aliases.append({'name':name,'slot_rva':hex(row['Address']),'class_identity':klass})
        tracked={p:len(raw) for p,raw in self.asset_storage}
        for binding in self.asset_fields.values():
            tracked.update({binding['identity']:0x148,binding['source_role']:0x48,binding['source_class']:0x1000})
        tracked.update({p:0x1000 for p in self.hierarchy_arrays}); tracked[classes['Role']]=0x1000
        self.asset_storage=[(p,bytes(self.uc.mem_read(p,n))) for p,n in tracked.items()]
        self.start_array=self.array([self.assets[i] for i in START_ORDER],'original_start_order')
        self.q(self.owner+0x28,self.start_array)
        self.order_storage=bytes(self.uc.mem_read(self.start_array,0x1000))
        self.scan=[]; self.before_start=None; self.pending_start=None; self.engine_join=None
        self.pause_mode=False; self.paused=None; self.reentry=None; self.prefixes=[]; self.reentries=0
        self.active_machine=self; self.direct_gateway=False
        self.bridge_out=self.allocate('bridge_result')
        self.bridge_type=self.allocate('IEnumerator_TypeInfo')
        self.q(self.base+0x26FE930,self.bridge_type)
        self.bridge_bodies=[]; self.bridge_decoded=set()
        for name,start,length in [('UnityEngine.SetupCoroutine$$InvokeMoveNext',0x1C8A780,None),
                                  ('identity_wrapper',0x3E8F30,4),('pointer_equality_wrapper',0x4A0210,7)]:
            end=min(r['Address'] for r in self.meta['ScriptMethod'] if r['Address']>start) if length is None else start+length
            raw=self.pe.get_data(start,end-start); ins=list(self.cs.disasm(raw,start))
            while ins[-1].mnemonic=='int3': ins.pop()
            assert all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]))
            self.instructions.update({i.address:i for i in ins})
            self.bridge_decoded.update(i.address for i in ins)
            self.bridge_bodies.append({'name':name,'rva':hex(start),'end_rva':hex(ins[-1].address+ins[-1].size),
                                       'body_sha256':sha(raw[:ins[-1].address+ins[-1].size-start])})
        self.queue_pins={0x36D0F9:('mov','r13, qword ptr [r12 + 0x28]'),
                         0x36D12C:('mov','r14, qword ptr [r13 + r14*8 + 0x20]'),
                         0x36D1C9:('call','0x1c822c0'),0x36D1DC:('call','0x3645c0'),
                         0x36D2DB:('mov','rax, qword ptr [r12 + 0x58]'),
                         0x36D204:('movzx','ecx, byte ptr [r8 + 0x130]'),
                         0x36D214:('mov','rax, qword ptr [rax + 0xc8]'),
                         0x1C8A7DB:('call','0x4060'),0x1C8A7E0:('mov','byte ptr [rbx], al')}
        for a,wanted in self.queue_pins.items(): assert (self.instructions[a].mnemonic,self.instructions[a].op_str)==wanted
        self.lowlevel=self.uc.emu_start

    def snapshot(self):
        out=super().snapshot()
        if hasattr(self,'start_array'):
            out['ordered_start']={'identity':self.rq(self.owner+0x28),'asset_ids':self.norm([self.rq(self.start_array+0x20+i*8) for i in range(self.rq(self.start_array+0x18))])}
            out['ordered_source_bindings']=self.ordered_source_bindings
            out['class_hierarchy_storage']=[{'identity':p,'sha256':sha(bytes(self.uc.mem_read(p,0x1000)))} for p in self.hierarchy_arrays]
        if getattr(self,'engine_join',None) is not None:
            out['engine_queue']=self.engine_join.queue_state()
            out['native_records']=self.engine_join.retained_native_records()
            out['physical_owners']=self.engine_join.owner_snapshot()
            out['engine_storage']=self.engine_join.storage_snapshot()
        return out

    def preserve(self):
        super().preserve()
        if hasattr(self,'order_storage'):
            assert self.rq(self.owner+0x28)==self.start_array
            assert bytes(self.uc.mem_read(self.start_array,0x1000))==self.order_storage

    def own_service(self,name,**details):
        # The inherited allocator uses this diagnostic ledger in object labels.
        # A retry must not advance it before consuming the one-shot token.
        self.preserve()
        if self.reentry is None: self.new_services.append(len(self.events)+1)
        return self.service(name,phase=self.phase,**details)

    def service(self,name,**details):
        machine=getattr(self,'active_machine',self)
        registers=cpu(machine); sp=registers['RSP']
        read=machine.rq if machine is self else machine.qword
        raw={n:registers[n] for n in ('RCX','RDX','R8','R9')}
        details.pop('raw_arguments',None)
        entry={'service':name,'machine':'managed' if machine is self else 'engine','phase':self.phase,
               'caller_return_rva':hex(read(sp)-machine.base),'raw_arguments':raw,**details}
        entry['entry_rva']=hex(registers['RIP']-machine.base)
        snapshot=self.snapshot()
        if self.reentry is not None:
            token=self.reentry
            assert token['entry']==entry and token['cpu']==registers and token['snapshot']==snapshot
            self.reentry=None; self.reentries+=1
            return True
        self.preserve()
        if getattr(self,'engine_join',None) is not None: self.engine_join.preserve_previous()
        event={**entry,'snapshot':snapshot if self.record_events else None}
        self.events.append(event)
        ordinal=len(self.events)
        if self.stop_service==ordinal:
            self.failure='service:'+name
            machine.uc.emu_stop()
            if machine is not self: raise AbortedPrefix()
            return False
        if self.pause_mode:
            assert self.paused is None
            self.paused={'entry':entry,'cpu':registers,'snapshot':snapshot,'ordinal':ordinal,
                         'pause_kind':'python_direct_provider' if self.direct_gateway else 'unicorn_pre_instruction',
                         'stack_bytes':bytes(machine.uc.mem_read(sp,0x30)).hex()}
            if self.direct_gateway:
                self.acknowledge_pause(machine)
                return self.service(name,**details)
            machine.uc.emu_stop(); return False
        return True

    def acknowledge_pause(self,machine):
        token=self.paused; assert token is not None
        now=cpu(machine)
        # Unicorn must have stopped at the precise pre-instruction hook.
        assert now==token['cpu'],(now['RIP'],token['cpu']['RIP'])
        assert bytes(machine.uc.mem_read(now['RSP'],0x30)).hex()==token['stack_bytes']
        assert self.snapshot()==token['snapshot']
        self.prefixes.append({'service_ordinal':token['ordinal'],**token['entry'],
                              'pause_kind':token['pause_kind'],
                              'entry_cpu':token['cpu'],'stack_bytes':token['stack_bytes'],
                              'final':token['snapshot']})
        self.reentry=token; self.paused=None
        machine.uc.reg_write(getattr(machine.x86 if hasattr(machine,'x86') else machine.x,'UC_X86_REG_RIP'),now['RIP'])

    def drive_managed(self,begin,end,outer=False):
        for _ in range(4096):
            self.lowlevel(begin,end,timeout=5_000_000,count=500000)
            if self.failure: return
            if self.paused is not None:
                self.acknowledge_pause(self); begin=self.reg(self.x.UC_X86_REG_RIP); continue
            if self.pending_start is not None:
                assert outer
                actor,iterator=self.pending_start; self.pending_start=None
                context=self.uc.context_save(); before=cpu(self)
                handle=self.engine_join.start_owner(actor,iterator)
                self.uc.context_restore(context); assert cpu(self)==before
                self.ret(handle); begin=self.reg(self.x.UC_X86_REG_RIP); continue
            if self.boundary or self.reg(self.x.UC_X86_REG_RIP)==end: return
            begin=self.reg(self.x.UC_X86_REG_RIP)
        raise AssertionError('retained managed driver budget exhausted')

    def step(self,iterator):
        assert self.active is not None and self.active['iterator']==iterator
        assert self.rd(iterator+0x10)==0 and self.rq(iterator+0x20)==self.active['actor_identity']
        context=self.uc.context_save(); caller=cpu(self); sp=caller['RSP']-0x2010
        assert sp%16==8
        end=self.stop+0x900
        self.q(sp,end); self.uc.mem_write(self.bridge_out,b'\x7f')
        for name in ('RAX','R10','R11','R8','R9'): self.uc.reg_write(getattr(self.x,'UC_X86_REG_'+name),0)
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
            assert self.rd(iterator+0x10)==1 and self.rd(self.rq(iterator+0x18)+0x10)==0x3E99999A
            self.first_yields.append({'actor':self.active['actor'],'iterator':iterator,'snapshot':self.snapshot()})
            return 1
        finally:
            self.active_machine=prior
            if not self.failure: self.uc.context_restore(context); assert cpu(self)==caller

    def run(self,*args,**kwargs):
        if args[0]!='Characters.ManageCharacters': return super().run(*args,**kwargs)
        for i in range(16): self.uc.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),0 if i<6 else 0xABC100+i)
        original=self.uc.emu_start
        self.uc.emu_start=lambda begin,end,**unused:self.drive_managed(begin,end,True)
        try:
            try: return BluffJoin.run(self,*args,**kwargs)
            except AbortedPrefix:
                assert self.failure
                return {'method':args[0],'failure':self.failure,'boundary':None,'services':list(self.events),
                        'final':self.snapshot(),'draws':list(self.draws)}
        finally: self.uc.emu_start=original

    def hook(self,uc,address,size,user):
        r=address-self.base
        if r==0x1C86600:
            minimum,maximum=[self.reg(z) for z in (self.x.UC_X86_REG_RCX,self.x.UC_X86_REG_RDX)]
            choice=self.choices[0] if self.choices else 0
            assert minimum<=choice<maximum
            draw={'minimum':minimum,'maximum_exclusive':maximum,'width':maximum-minimum,'index':choice,
                  'caller_return_rva':hex(self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base)}
            if self.service('rng',**draw):
                if self.choices: self.choices.pop(0)
                self.draws.append(draw); self.ret(choice)
            return
        if r==0x1C7F160 and self.active:
            actor,iterator,context=[self.reg(z) for z in (self.x.UC_X86_REG_RCX,self.x.UC_X86_REG_RDX,self.x.UC_X86_REG_R8)]
            assert actor==self.active['actor_identity'] and iterator==self.active['iterator'] and context==0
            assert self.rd(iterator+0x10)==0
            if self.service('original_start_coroutine',actor=actor,iterator=iterator):
                self.pending_start=(actor,iterator); uc.emu_stop()
            return
        if r==0x4060:
            slot,typ,iterator=[self.reg(z) for z in (self.x.UC_X86_REG_RCX,self.x.UC_X86_REG_RDX,self.x.UC_X86_REG_R8)]
            assert slot==0 and typ==self.bridge_type and iterator==self.active['iterator']
            if self.service('ienumerator_slot_zero',slot=slot,iterator=iterator,method_rva='0x3756b0'):
                for n,v in [('RCX',iterator),('RDX',0),('R8',0),('R9',0)]: uc.reg_write(getattr(self.x,'UC_X86_REG_'+n),v)
                uc.reg_write(self.x.UC_X86_REG_RIP,self.base+0x3756B0)
            return
        if r in self.bridge_decoded:
            self.visited.add(r); return
        if r==0x36D0F9:
            assert len(self.actions)==5 and all(a['completed'] for a in self.actions)
            assert self.reg(self.x.UC_X86_REG_RSP)==self.pre_sp
            self.preserve(); self.before_start=self.snapshot(); self.phase='ordered_start'
            self.visited.add(r); return
        if r==0x3645C0 and self.post:
            assert self.reg(self.x.UC_X86_REG_RDX)==3,'unexpected ordered Start request'
        if r==0x1C822C0 and self.phase=='ordered_start':
            c,t=[self.reg(z) for z in (self.x.UC_X86_REG_RCX,self.x.UC_X86_REG_RDX)]
            assert c in self.assets.values() and t in self.assets.values()
            assert self.asset_ids[c] in START_ORDER and self.asset_ids[t] in ORDER
            if self.service('ordered_asset_equality',left=c,right=t):
                self.scan.append([self.asset_ids[c],self.asset_ids[t]])
                self.ret(int(c==t))
            return
        if r==0x36D2DB:
            assert self.before_start is not None and len(self.scan)==len(START_ORDER)*len(ORDER)
            assert self.reg(self.x.UC_X86_REG_RSP)==self.pre_sp and self.rq(self.stack+0x10008)==self.stop
            self.preserve(); assert self.snapshot()==self.before_start
            self.boundary={'kind':'before_on_setup','rva':hex(r),'ordered_entries':len(START_ORDER),
                           'start_calls':0,'live_manage_sp':self.pre_sp,'root_return_sentinel':self.stop}
            uc.emu_stop(); return
        return super().hook(uc,address,size,user)


class MultiOwnerEngine(ScheduledEngine):
    def __init__(self,data,managed):
        super().__init__(data,managed)
        self.owner_bindings={a:{'managed_actor':a,'native_owner':self.arena+0x180000+i*0x1000,'key':101+i}
                             for i,a in enumerate(managed.actors)}
        for row in self.owner_bindings.values():
            p=row['native_owner']; self.write_q(p+0x40,self.owner_vtable); self.write_d(p+8,row['key'])
            head=p+0x70; self.write_q(head,head); self.write_q(head+8,head)
        self.completed_storage=[]

    def owner_snapshot(self):
        out=[]
        for row in self.owner_bindings.values():
            assert self.qword(row['native_owner']+0x40)==self.owner_vtable
            assert struct.unpack('<I',self.uc.mem_read(row['native_owner']+8,4))[0]==row['key']
            head=row['native_owner']+0x70; links=[]; p=self.qword(head); previous=head
            while p!=head:
                assert p in self.payload_registry and p not in links
                assert self.qword(p+0x58)==row['native_owner']
                assert self.qword(p+8)==previous
                links.append(p); previous=p; p=self.qword(p)
            assert self.qword(head+8)==previous
            out.append({**row,'list_head':head,'payloads':links})
        return out

    def preserve_previous(self):
        for address,raw in self.completed_storage:
            assert bytes(self.uc.mem_read(address,len(raw)))==raw,hex(address)

    def storage_snapshot(self):
        blocks=[(self.owner,0x50),(self.head,0x60),(self.cache,0x1000),(self.owner_vtable,0x20),
                (self.method,0x100),(self.allocator,0x100),(self.profiler,0x1000),(self.engine,0x1000)]
        blocks += [(row['native_owner'],0x100) for row in self.owner_bindings.values()]
        blocks += [(p,0x88) for p in self.payload_registry]
        blocks += [(p,0x60) for p in range(self.arena+0x1000,self.next_node,0x80)]
        blocks += [(p,0x20) for p in self.wait_mirrors]
        order=[]; reverse={p:i for i,p in self.nodes.items()}
        def walk(p):
            if p==self.head:return
            assert p in reverse
            walk(self.qword(p)); order.append(reverse[p]); walk(self.qword(p+0x10))
        walk(self.qword(self.head+8))
        return {'blocks':[{'identity':p,'size':n,'sha256':sha(bytes(self.uc.mem_read(p,n)))} for p,n in blocks],
                'payload_bytes':{str(p):bytes(self.uc.mem_read(p,0x88)).hex() for p in self.payload_registry},
                'owner_bytes':{str(r['native_owner']):bytes(self.uc.mem_read(r['native_owner'],0x100)).hex() for r in self.owner_bindings.values()},
                'tree_node_bytes':{str(p):bytes(self.uc.mem_read(p,0x60)).hex() for p in range(self.arena+0x1000,self.next_node,0x80)},
                'next_node':self.next_node,'pending_insert':copy.deepcopy(self.pending_insert) if self.pending_insert is None else {k:v.hex() if isinstance(v,bytes) else v for k,v in self.pending_insert.items()},
                'payload_registry':{str(p):dict(row) for p,row in self.payload_registry.items()},
                'gc_targets':{str(k):v for k,v in self.gc_targets.items()},'wait_mirrors':{str(k):v for k,v in self.wait_mirrors.items()},
                'actual_tree_order':order,'queue_owner':self.owner,'container':self.container,'head':self.head}

    def queue_state(self):
        out=super().queue_state()
        if hasattr(self,'owner_bindings'):
            reverse={p:i for i,p in self.nodes.items()}; order=[]
            def walk(p):
                if p==self.head:return
                assert p in reverse
                walk(self.qword(p)); order.append(reverse[p]); walk(self.qword(p+0x10))
            walk(self.qword(self.head+8))
            assert order==[r['id'] for r in out['entries']]
            for node,cached in self.records.items():
                live=bytes(self.uc.mem_read(node+0x20,0x40)); assert live==cached
                payload=struct.unpack_from('<Q',live,0x18)[0]; row=self.payload_registry[payload]
                assert struct.unpack_from('<dq',live,0)==(1.0+struct.unpack('<f',struct.pack('<I',0x3E99999A))[0],8)
                assert struct.unpack_from('<QQ',live,0x20)==(self.base+0x778B30,self.base+0x778BD0)
                assert struct.unpack_from('<III',live,0x30)==(row['key'],0xA,0)
                assert self.gc_targets[self.qword(payload+0x10)]==self.qword(payload+0x20)==row['pointer']
                assert self.qword(payload+0x58)==self.owner_bindings[row['actor']]['native_owner']
            for mirror,original in self.wait_mirrors.items():
                assert self.qword(mirror)==self.wait_class
                assert bytes(self.uc.mem_read(mirror+8,0x18))==bytes(self.managed.uc.mem_read(original+8,0x18))
            if self.pending_insert is None:
                validate_tree(self.uc.mem_read,self.container,self.head,self.records,[0]*len(order))
            out['actual_tree_order']=order
        return out

    def retained_native_records(self):
        if not hasattr(self,'owner_bindings'): return []
        linked={p for row in self.owner_snapshot() for p in row['payloads']}
        return [{'payload':p,'iterator':row['pointer'],'managed_actor':row['actor'],'native_owner':self.qword(p+0x58),
                 'owner_key':row['key'],'reference_count':struct.unpack('<i',self.uc.mem_read(p+0x60,4))[0],
                 'gc_handle':self.qword(p+0x10),'cached_enumerator':self.qword(p+0x20),'owner_linked':p in linked}
                for p,row in self.payload_registry.items()]

    def _on_code(self,uc,address,size,data):
        if not hasattr(self,'owner_bindings'): return super()._on_code(uc,address,size,data)
        r=address-self.base; m=self.managed; x=self.x86
        self.executed.add(r if address>=self.base else 'gateway:'+hex(address-self.stop))
        gateways={self.param_count:'parameter_count',self.invoke:'runtime_invoke',self.owner_context:'owner_context',
                  self.gc_write:'gc_write',self.gc_target:'gc_target',self.free_handle:'gc_free',
                  self.object_class:'yield_object_class',self.subclass:'wait_subclass',
                  self.base+0x779070:'current_yield',self.base+0x677920:'tree_allocator',
                  self.base+0x17D6A84:'numeric_conversion',self.base+0x6E78E0:'profiler',self.base+0x355F00:'profile_sink'}
        if address in gateways:
            if not m.service('engine_'+gateways[address]): return
        if address==self.owner_context:
            c=uc.reg_read(x.UC_X86_REG_RCX)
            assert c in [row['native_owner']+0x40 for row in self.owner_bindings.values()]
            out=uc.reg_read(x.UC_X86_REG_RDX); self.write_q(out,0); self._return(out); return
        if r==0x779070:
            payload=uc.reg_read(x.UC_X86_REG_RCX); row=self.payload_registry[payload]
            original=m.rq(row['pointer']+0x18); assert m.rd(original+0x10)==0x3E99999A
            mirror=self.arena+0x1A0000+len(self.wait_mirrors)*0x100
            self.wait_mirrors[mirror]=original; uc.mem_write(mirror,bytes(m.uc.mem_read(original,0x20))); self.write_q(mirror,self.wait_class)
            self.join_trace.append({'kind':'current_yield_gateway','iterator':row['label'],'wait':original,'duration_bits':m.rd(original+0x10)})
            uc.reg_write(x.UC_X86_REG_RCX,payload); uc.reg_write(x.UC_X86_REG_RDX,mirror); uc.reg_write(x.UC_X86_REG_RIP,self.base+0x779370); return
        if r==0x440F00:
            record=bytes(uc.mem_read(uc.reg_read(x.UC_X86_REG_R8),0x40)); payload=struct.unpack_from('<Q',record,0x18)[0]
            row=self.payload_registry[payload]
            assert struct.unpack_from('<I',record,0x30)[0]==row['key']
            assert struct.unpack_from('<QQ',record,0x20)==(self.base+0x778B30,self.base+0x778BD0)
            self.pending_insert={'id':self.next_identity,'record':record,'node':self.next_node,'kind':row['kind'],'iterator':row['label'],
                'producer':{'time':self.producer_time,'frame':self.producer_frame,'duration_bits':0x3E99999A}}
            self.next_identity+=1
            NativeTree._on_code(self,uc,address,size,data); return
        if r in (0x778B30,0x151EF0): raise AssertionError('acquisition/lookup callback outside predeadline scope')
        return super()._on_code(uc,address,size,data)

    def run_native(self,name,*args):
        m=self.managed; prior=m.active_machine; m.active_machine=self
        x=self.x86; sp=self.stack+0xF008
        self.uc.mem_write(sp,struct.pack('<Q',self.stop)+bytes(0x28))
        for n in ('RAX','RCX','RDX','R8','R9','R10','R11'): self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),0)
        self.uc.reg_write(x.UC_X86_REG_RSP,sp)
        for i,n in enumerate(('RBX','RBP','RSI','RDI','R12','R13','R14','R15')): self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),0xFEA00000+i)
        for i in range(16): self.uc.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),0 if i<6 else 0xFED100+i)
        self.uc.reg_write(x.UC_X86_REG_EFLAGS,2); self.uc.reg_write(x.UC_X86_REG_MXCSR,0x1F80)
        for n,v in zip(('RCX','RDX','R8','R9'),args): self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        before=cpu(self); pc=self.base+self.routines[name][0]
        try:
            for _ in range(1024):
                self.pending_invoke=None
                self.uc.emu_start(pc,self.stop,timeout=5_000_000,count=500000)
                if m.failure: raise AbortedPrefix()
                if m.paused is not None:
                    m.acknowledge_pause(self); pc=self.uc.reg_read(x.UC_X86_REG_RIP); continue
                if self.pending_invoke is not None:
                    iterator,out,error=self.pending_invoke; self.pending_invoke=None
                    context=self.uc.context_save(); invocation_cpu=cpu(self)
                    result=m.step(iterator); self.uc.context_restore(context)
                    assert cpu(self)==invocation_cpu
                    self.uc.mem_write(out,bytes([result])); self.write_q(error,0)
                    self.join_trace.append({'kind':'managed_move_next_return','iterator':self.registry[iterator]['label'],'result':result})
                    self._return(); pc=self.uc.reg_read(x.UC_X86_REG_RIP); continue
                if self.uc.reg_read(x.UC_X86_REG_RIP)==self.stop: break
                pc=self.uc.reg_read(x.UC_X86_REG_RIP)
            else: raise AssertionError('engine retained driver budget exhausted')
            after=cpu(self)
            assert after['RSP']==sp+8
            assert all(after[n]==before[n] for n in ('RBX','RBP','RSI','RDI','R12','R13','R14','R15')+tuple('XMM'+str(i) for i in range(6,16)))
            return after['RAX']
        finally: m.active_machine=prior

    def start_owner(self,actor,iterator):
        m=self.managed; row=self.owner_bindings[actor]; self.native_owner=row['native_owner']
        prior=m.active_machine; m.active_machine=m; m.direct_gateway=True
        try:
            if not m.service('native_record_creation',actor=actor,iterator=iterator,native_owner=self.native_owner,owner_key=row['key']): raise AbortedPrefix()
        finally: m.direct_gateway=False; m.active_machine=prior
        payload=self.arena+0x1B0000+len(self.registry)*0x100
        self.uc.mem_write(payload,bytes(0x88)); label=m.labels[iterator]
        rec={'pointer':iterator,'payload':payload,'kind':'acquisition','label':label,'actor':actor,'key':row['key']}
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
    game_root=Path(game_root); dumper_root=Path(dumper_root)
    inputs=load_inputs(game_root,dumper_root)
    names=('ascension_assets_audit','character_assets_audit','first_village_bluff_generation','character_init','first_village_role_setup','first_village_publication_init','hunter_scheduled_publication')
    paths={n:ROOT/f'reports/{BUILD}_{n}.json' for n in names}
    prior_hashes={n:sha(p.read_bytes()) for n,p in paths.items()}
    reports={n:json.loads(p.read_text(encoding='utf-8')) for n,p in paths.items()}
    from audit_ascension_assets import audit as profile_audit
    from audit_character_assets import audit as asset_audit
    assert reports['ascension_assets_audit']==json.loads(json.dumps(profile_audit(game_root,dumper_root)))
    assert reports['character_assets_audit']==json.loads(json.dumps(asset_audit(game_root,dumper_root)))
    original=scene_order(game_root,inputs,reports['character_assets_audit'])
    engine_data=(game_root/'UnityPlayer.dll').read_bytes(); verify_fingerprint(engine_data,ENGINE_SHA256)
    source_paths={Path(inspect.getfile(c)) for c in StartQueueWitness.__mro__[:-1]+MultiOwnerEngine.__mro__[:-1]}
    source_paths.update([Path(__file__),Path(inspect.getfile(Cursor)),Path(inspect.getfile(pool_snapshots)),Path(inspect.getfile(load_inputs)),Path(inspect.getfile(profile_audit)),Path(inspect.getfile(verify_fingerprint))])
    source_hashes={p.relative_to(ROOT.parent).as_posix():sha(p.read_bytes()) for p in sorted(source_paths)}
    prior=reports['first_village_bluff_generation']; genrow=prior['generation_index_factor']['cases'][0]; poolrow=prior['pool_index_factors']['cases'][0]

    def run(paused=False,abort=None):
        m=StartQueueWitness(inputs,reports['ascension_assets_audit'],reports['character_assets_audit'],reports['character_init'],reports['first_village_role_setup'],original_order=original)
        constructors=[m.run('Character.ctor',a) for a in m.actors]; assert all(not r['failure'] for r in constructors)
        m.phase='setup'; setup=[]
        for name,this,arg,choices in [('GameData.SetupCurrentAscension',m.game,0,[]),('AscensionsData.ClearCurrentPickedScript',m.temporary,0,[]),('AscensionsData.SetupCharactersCount',m.temporary,0,[0]),('AscensionsData.SetupStartingCharacters',m.temporary,0,[]),('Gameplay.GetCurrentScript',m.gameplay,0,[])]:
            row=m.run(name,this,arg,choices); assert not row['failure']; setup.append(row)
        m.q(m.gameplay_static+0x30,m.reg(m.x.UC_X86_REG_RAX))
        generation=m.run('Gameplay.GetRandomCharacters',m.gameplay,5,genrow['choices']); assert generation['final']['returned_order']==ORDER
        returned=m.reg(m.x.UC_X86_REG_RAX)
        e=MultiOwnerEngine(engine_data,m); e.set_producer(1.0,7)
        m.phase='manage'; m.pause_mode=paused; initial=m.snapshot()
        joined=m.run('Characters.ManageCharacters',m.owner,returned,poolrow['choices'],stop_service=abort)
        if abort is not None: return m,e,{'joined':joined}
        assert joined['failure'] is None and joined['boundary']['kind']=='before_on_setup'
        assert len(m.init_calls)==len(m.first_yields)==len(m.actions)==len(e.registry)==5
        assert m.scan==[[a,b] for a in START_ORDER for b in ORDER]
        assert all(r['reference_count']==1 and r['owner_linked'] and r['gc_handle'] and r['cached_enumerator']==r['iterator'] for r in e.retained_native_records())
        assert all(len(row['payloads'])==1 for row in e.owner_snapshot())
        assert {k:joined['final'][k] for k in GRAPH_KEYS}=={k:initial[k] for k in GRAPH_KEYS}
        assert joined['final']['pools']==poolrow['pool_identities_and_contents']
        assert all(r['state']==1 and r['wait_bits']==0x3E99999A for r in joined['final']['continuations'])
        if paused: assert m.reentries==len(joined['services'])==len(m.prefixes) and not m.reentry and not m.paused
        admissions=[]
        for event in e.join_trace:
            if event['kind']!='native_wait_inserted': continue
            i=event['id']; record=e.records[e.nodes[i]]; payload=struct.unpack_from('<Q',record,0x18)[0]; row=e.payload_registry[payload]
            assert event['producer']=={'time':1.0,'frame':7,'duration_bits':0x3E99999A}
            assert struct.unpack_from('<d',record,0)[0]==1.0+struct.unpack('<f',struct.pack('<I',0x3E99999A))[0]
            assert struct.unpack_from('<q',record,8)[0]==8 and struct.unpack_from('<I',record,0x38)[0]==0
            admissions.append({'id':i,'iterator':row['pointer'],'iterator_label':row['label'],'actor':row['actor'],
                               'payload':payload,'native_owner':e.qword(payload+0x58),'owner_key':row['key'],
                               'producer':event['producer'],'frame_threshold':8,'generation':0,'queue':event['queue']})
        return m,e,{'constructors':constructors,'setup':setup,'generation':generation,'initial':initial,
                    'pre_publication':m.pre_publication,'before_start':m.before_start,'initializers':m.init_calls,
                    'first_yields':m.first_yields,'act_init_returns':m.actions,'joined':joined,'ordered_comparisons':m.scan,
                    'engine_admissions':admissions,'final_cpu':{'managed':cpu(m),'engine':cpu(e)}}

    normal_m,normal_e,normal=run()
    paused_m,paused_e,paused=run(True)
    assert normal==paused,first_difference(normal,paused)
    services=normal['joined']['services']
    # One independent aborted restart per materially new gateway family/phase.
    representatives={}
    for i,event in enumerate(services,1):
        if event['service'].startswith('engine_') or event['service'] in ('original_start_coroutine','native_record_creation','ienumerator_slot_zero','ordered_asset_equality'):
            representatives.setdefault((event['service'],event['phase'].split(':')[0]),i)
    aborts=[]
    for key,ordinal in representatives.items():
        m,e,result=run(abort=ordinal); joined=result['joined']
        assert joined['failure']=='service:'+services[ordinal-1]['service']
        assert joined['services']==services[:ordinal] and joined['final']==services[ordinal-1]['snapshot']
        aborts.append({'service_ordinal':ordinal,'service':key[0],'phase_family':key[1],'final':joined['final']})
    assert all(sha(p.read_bytes())==prior_hashes[n] for n,p in paths.items())
    assert all(sha(p.read_bytes())==source_hashes[p.relative_to(ROOT.parent).as_posix()] for p in source_paths)
    return {'schema_version':'first_village_start_queue_v1','build_id':BUILD,
            'decision':'Retained original N5 unchanged through genuine ordered Start and all five original first waits admitted before acquisition.',
            'domain':{'generation_row':0,'pool_row':0,'asset_order':ORDER,'producer_time':1.0,'producer_signed_frame':7,'generation':0,
                      'exit':'0x36d2db before onSetup read','scene_actor_initial_state':20,'creation_runtime_and_scene_gateways':'Explicit supplied providers; native dispatcher, managed first MoveNext, wait producer/tree and reference release execute.'},
            'exclusions':['onSetup identity/absence','Shuffle','acquisition resumes','rendered/player observation','managed exception/API failure effects','all abort continuations equivalence'],
            'source_hashes':source_hashes,'prior_report_hashes':prior_hashes,'original_scene_order':original,
            'ordered_source_bindings':normal_m.ordered_source_bindings,
            'class_metadata_aliases':normal_m.class_metadata_aliases,
            'selected_instruction_assertions':[{'rva':hex(a),'mnemonic':v[0],'operands':v[1]} for a,v in normal_m.queue_pins.items()],
            'bodies':normal_m.body_evidence+normal_m.new_bodies+normal_m.post_bodies+normal_m.bridge_bodies,
            **normal,'paused_reentered_prefixes':paused_m.prefixes,'restart_aborted_prefixes':aborts,
            'counters':{'constructors':len(normal['constructors']),'initializers':len(normal_m.init_calls),'first_yields':len(normal_m.first_yields),
                        'act_init_calls':len(normal_m.actions),'start_calls':0,'ordered_entries':len(START_ORDER),'ordered_comparisons':len(normal_m.scan),
                        'queue_records':len(normal_e.nodes),'physical_owners':len(normal_e.owner_bindings),
                        'manage_services':len(services),'paused_reentered_prefixes':len(paused_m.prefixes),'reentry_tokens_consumed':paused_m.reentries,
                        'restart_aborted_prefixes':len(aborts),'managed_native_addresses':len(normal_m.visited),
                        'engine_native_addresses':len(normal_e.executed),'python_source_hashes':len(source_hashes)}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--game-root',required=True); p.add_argument('--dumper-root',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); report=audit(a.game_root,a.dumper_root); assert report==json.loads(json.dumps(report))
    packed=pool_snapshots(report); assert expand_snapshots(json.loads(json.dumps(packed)))==report
    Path(a.output).write_text(json.dumps(packed,indent=2,sort_keys=True,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':Path(a.output).name,**report['counters']},sort_keys=True),flush=True)


if __name__=='__main__': main()
