"""Actual installed animation subscriber and nested retained queue admissions.

Unity lifecycle, closed-instance delegate construction/Combine, scene components,
vector/tween providers and nullable audio binding are explicit finite contracts.
"""
import argparse
import copy
import inspect
import json
import re
import struct
from pathlib import Path

from audit_first_village_shuffle_admission import ShuffleWitness, ShuffleEngine, ACQUISITION_BITS, SHUFFLE_BITS
from audit_first_village_start_queue import (
    BUILD, ORDER, START_ORDER, GRAPH_KEYS, ROOT, Cursor, sha, cpu, first_difference,
    scene_order, load_inputs, StartQueueWitness, MultiOwnerEngine, ScheduledEngine,
    NativeTree, validate_tree, AbortedPrefix, ENGINE_SHA256, verify_fingerprint,
)
from audit_first_village_on_setup_binding import audit as binding_audit, PINS
from audit_first_village_bluff_generation import BluffJoin
from audit_report_snapshots import pool_snapshots, expand_snapshots

BITS={'acquisition':ACQUISITION_BITS,'shuffle':SHUFFLE_BITS,'animation':0x3D4CCCCD,'audio':0x3ECCCCCD}
ENTRY={'animation':0x375130,'audio':0x375DB0}
NONVOL=('RBX','RBP','RSI','RDI','R12','R13','R14','R15')+tuple('XMM'+str(i) for i in range(6,16))


class SubscriberWitness(ShuffleWitness):
    def __init__(self,*args,binding,**kwargs):
        super().__init__(*args,**kwargs)
        import capstone
        self.binding=binding
        self.animation=self.allocate('scene_animation:137027'); self.q(self.animation+0x20,self.owner)
        self.delegates={}; self.installed_delegate=0; self.installed=False; self.installed_event_fields=()
        self.invocation='hydration'; self.visual_effects=[]; self.subscriber_yields=[]
        self.iterator_kinds={}; self.managed_step_stack=[]; self.nested_handoffs=[]
        self.callback_calls=[]; self.subscriber_bodies=[]; self.subscriber_decoded=set()
        methods={0x363500:'CharacterShuffleAnimation$$OnEnable',0x363060:'CharacterShuffleAnimation$$Animates',
                 0x375130:'CharacterShuffleAnimation.<Animate>d__6$$MoveNext',0x375660:'CharacterShuffleAnimation.<Animate>d__6$$<>m__Finally1',
                 0x375DB0:'CharacterShuffleAnimation.<PlayAudioDelay>d__9$$MoveNext',0x4D5170:'System.Action$$.ctor'}
        slots={r['Address']:r for key in ('ScriptMetadata','ScriptMetadataMethod','ScriptString') for r in self.meta[key]}
        for start,name in methods.items():
            rows=[r for r in self.meta['ScriptMethod'] if r['Address']==start and r['Name']==name]; assert len(rows)==1,name
            end=min(r['Address'] for r in self.meta['ScriptMethod'] if r['Address']>start)
            raw=self.read_file_backed(start,end-start); ins=list(self.cs.disasm(raw,start))
            assert ins and ins[0].address==start and all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]))
            self.instructions.update({i.address:i for i in ins})
            if start!=0x4D5170: self.subscriber_decoded.update(i.address for i in ins)
            self.subscriber_bodies.append({'name':name,'rva':hex(start),'interval_end_rva':hex(end),
                'decoded_prefix_end_rva':hex(ins[-1].address+ins[-1].size),'body_sha256':sha(raw),
                'execution':'supplied constructor layout evidence' if start==0x4D5170 else 'actual reached instructions; other paths excluded'})
            for i in ins:
                for op in i.operands:
                    if op.type!=capstone.CS_OP_MEM or op.mem.base!=capstone.x86.X86_REG_RIP: continue
                    address=i.address+i.size+op.mem.disp
                    if address in slots:
                        row=slots[address]; n=row.get('Name','literal:'+row.get('Value',''))
                        if n not in self.names:
                            p=self.allocate(n); self.names[n]=p; self.metadata_names[p]=n; self.d(p+0xE0,1)
                        self.q(self.base+address,self.names[n])
                    elif i.mnemonic=='cmp' and op.size==1: self.uc.mem_write(self.base+address,b'\1')
        self.entries['Animation.OnEnable']=0x363500
        self.method_bindings={}
        for name,entry in [('Animates',0x363060),('ShuffleCards',0x363960)]:
            key='Method$CharacterShuffleAnimation.'+name+'()'; p=self.names[key]
            rows=[r for r in self.meta['ScriptMetadataMethod'] if r['Name']==key]
            assert rows and all(r['MethodAddress']==entry for r in rows)
            self.q(p+8,self.base+entry); self.uc.mem_write(p+0x4C,struct.pack('<H',6)); self.uc.mem_write(p+0x52,b'\0')
            self.method_bindings[p]={'name':name,'identity':p,'code':self.base+entry,'flags':6,'parameters_count':0}
        self.event_static=self.allocate('GameplayEvents_static'); self.q(self.names['GameplayEvents_TypeInfo']+0xB8,self.event_static)
        self.audio_static=self.allocate('AudioEvents_static'); self.q(self.names['AudioEvents_TypeInfo']+0xB8,self.audio_static)
        self.vector_static=self.allocate('Vector3_static'); self.q(self.names['UnityEngine.Vector3_TypeInfo']+0xB8,self.vector_static)
        for declaration,field in [('public static class AudioEvents',r'public static Action<ESFX> OnPlaySfxOneShot; // 0x0'),
                                  ('public struct Vector3',r'private static readonly Vector3 zeroVector; // 0x0')]:
            block=re.search(r'^'+re.escape(declaration)+r'[^\n]*\n\{(.*?)\n\}',self.dump,re.M|re.S)
            assert block and field in block.group(1),declaration
        self.uc.mem_write(self.vector_static,bytes(120))
        self.draws_by_actor={}
        for a in self.actors:
            component=self.allocate('draw_component:'+self.labels[a]); pivot=self.allocate('pivot:'+self.labels[a]); self.q(component+0x20,pivot)
            self.draws_by_actor[a]={'actor':a,'component':component,'pivot':pivot,'local_position_bits':[0x3F800000,0x40000000,0x40400000]}
        self.subscriber_pins={a:(m,o) for a,m,o in PINS if a in self.instructions}
        self.subscriber_pins.update({0x4D518A:('mov','rax, qword ptr [r8 + 8]'),0x4D5191:('mov','qword ptr [rcx + 0x10], rax'),
            0x4D5198:('mov','qword ptr [rcx + 0x28], rbx'),0x4D519F:('mov','qword ptr [rcx + 0x20], rdx'),
            0x4D51D8:('mov','qword ptr [rdi + 0x40], rax'),0x4D51E0:('mov','qword ptr [rdi + 0x18], rax'),
            0x4D51FA:('mov','qword ptr [rdi + 0x38], rax'),
            0x37536E:('mov','rax, qword ptr [rsi + 0x20]'),0x37537B:('mov','rdx, qword ptr [rax + 0x20]'),
            0x3753A6:('movups','xmmword ptr [rax + 0x28], xmm0'),0x3753B0:('movsd','qword ptr [rax + 0x38], xmm1'),
            0x3753BD:('add','rcx, 0x28'),0x3753C1:('xor','edx, edx'),0x3753C3:('call','0x2b6ff0')})
        for a,wanted in self.subscriber_pins.items(): assert (self.instructions[a].mnemonic,self.instructions[a].op_str)==wanted
        stub=self.instructions[0x4D51E4]; assert stub.mnemonic=='lea'
        self.action_stub=self.base+stub.address+stub.size+stub.operands[1].mem.disp; assert self.action_stub==self.base+0x60E0

    def snapshot(self):
        out=StartQueueWitness.snapshot(self)
        if not hasattr(self,'animation'): return out
        p=self.shuffle_iterator; wait=self.rq(p+0x18) if p else 0
        out['shuffle_continuation']=None if not p else {'identity':p,'class':self.rq(p),'state':self.rd(p+0x10),
                                                      'current':wait,'wait_bits':self.rd(wait+0x10) if wait else None}
        out['subscriber_binding']={'component':self.animation,'scene_path_id':137027,'characters':self.rq(self.animation+0x20),
            'on_setup':self.rq(self.owner+0x58),'delegates':copy.deepcopy(list(self.delegates.values())),
            'later_gameplay_events':{hex(o):self.rq(self.event_static+o) for o in (0x18,0x30,0x38)} if hasattr(self,'event_static') else {},
            'audio_on_play_sfx':self.rq(self.audio_static) if hasattr(self,'audio_static') else 0,
            'contracts':'supplied lifecycle invocation; initially null delegate lists; AudioEvents null conditional'}
        out['auxiliary_continuations']=[{'identity':p,'kind':kind,'state':self.rd(p+0x10),'current':self.rq(p+0x18),
            'wait_bits':self.rd(self.rq(p+0x18)+0x10) if self.rq(p+0x18) else None,
            'captured_component':self.rq(p+0x20) if kind=='animation' else None} for p,kind in self.iterator_kinds.items()]
        out['visual_provider_state']=copy.deepcopy(list(self.draws_by_actor.values())) if hasattr(self,'draws_by_actor') else []
        out['prior_five_storage']=[{'identity':p,'size':len(raw),'sha256':sha(bytes(self.uc.mem_read(p,len(raw))))} for p,raw in self.prior_five_storage]
        return out

    def preserve(self):
        StartQueueWitness.preserve(self)
        for p,raw in getattr(self,'prior_five_storage',[]): assert bytes(self.uc.mem_read(p,len(raw)))==raw
        for p,row in getattr(self,'delegates',{}).items():
            assert [self.rq(p+o) for o in (0x10,0x18,0x20,0x28,0x38,0x40)]==row['fields']
        if getattr(self,'installed',False):
            assert self.rq(self.owner+0x58)==self.installed_delegate
            assert tuple(self.rq(self.event_static+o) for o in (0x18,0x30,0x38))==self.installed_event_fields
        if hasattr(self,'audio_static'): assert self.rq(self.audio_static)==0
        if hasattr(self,'vector_static'): assert bytes(self.uc.mem_read(self.vector_static,120))==bytes(120)

    def service(self,name,**details):
        details['invocation']=self.invocation if hasattr(self,'invocation') else 'hydration'
        return super().service(name,**details)

    def run(self,*args,**kwargs):
        if args[0]!='Animation.OnEnable': return super().run(*args,**kwargs)
        for i in range(16): self.uc.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),0 if i<6 else 0xABC100+i)
        original=self.uc.emu_start
        self.uc.emu_start=lambda begin,end,**unused:self.drive_managed(begin,end,True)
        try:
            try:
                result=BluffJoin.run(self,*args,**kwargs)
                if not self.failure:
                    assert all(self.reg(getattr(self.x,f'UC_X86_REG_XMM{i}'))==0xABC100+i for i in range(6,16))
                return result
            except AbortedPrefix:
                assert self.failure
                return {'method':args[0],'failure':self.failure,'boundary':None,'services':list(self.events),'final':self.snapshot()}
        finally: self.uc.emu_start=original

    def drive_managed(self,begin,end,outer=False):
        for _ in range(4096):
            self.lowlevel(begin,end,timeout=5_000_000,count=500000)
            if self.failure: return
            if self.paused is not None:
                self.acknowledge_pause(self); begin=self.reg(self.x.UC_X86_REG_RIP); continue
            if self.pending_start is not None:
                actor,iterator=self.pending_start; self.pending_start=None
                context=self.uc.context_save(); before=cpu(self); sp=before['RSP']
                stack=bytes(self.uc.mem_read(sp-0x1000,0x2000))
                parent=self.managed_step_stack[-1] if self.managed_step_stack else None
                retained=bytes(self.uc.mem_read(parent,0x40)) if parent else None
                handle=self.engine_join.start_owner(actor,iterator)
                assert bytes(self.uc.mem_read(sp-0x1000,0x2000))==stack
                if parent: assert bytes(self.uc.mem_read(parent,0x40))==retained
                self.uc.context_restore(context); assert cpu(self)==before
                if parent: self.nested_handoffs.append({'parent_iterator':parent,'child_iterator':iterator,
                    'caller_cpu':before,'restored_cpu':cpu(self),'stack_sha256':sha(stack),'parent_iterator_sha256':sha(retained)})
                self.ret(handle); begin=self.reg(self.x.UC_X86_REG_RIP); continue
            if self.boundary or self.reg(self.x.UC_X86_REG_RIP)==end: return
            begin=self.reg(self.x.UC_X86_REG_RIP)
        raise AssertionError('subscriber managed driver budget exhausted')

    def step(self,iterator):
        if iterator not in self.iterator_kinds: return super().step(iterator)
        kind=self.iterator_kinds[iterator]; assert self.rd(iterator+0x10)==0
        context=self.uc.context_save(); caller=cpu(self); sp=caller['RSP']-0x2010; assert sp%16==8
        end=self.stop+0x900; self.q(sp,end); self.uc.mem_write(self.bridge_out,b'\x7f')
        for n in ('RAX','R10','R11','R8','R9'): self.uc.reg_write(getattr(self.x,'UC_X86_REG_'+n),0)
        for i in range(6): self.uc.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),0)
        self.uc.reg_write(self.x.UC_X86_REG_RSP,sp); self.uc.reg_write(self.x.UC_X86_REG_RCX,iterator); self.uc.reg_write(self.x.UC_X86_REG_RDX,self.bridge_out)
        prior=self.active_machine; self.active_machine=self; self.managed_step_stack.append(iterator)
        try:
            self.drive_managed(self.base+0x1C8A780,end)
            if self.failure: raise AbortedPrefix()
            after=cpu(self); assert after['RIP']==end and after['RSP']==sp+8 and all(after[n]==caller[n] for n in NONVOL)
            assert self.uc.mem_read(self.bridge_out,1)==b'\1' and self.rd(iterator+0x10)==1
            assert self.rd(self.rq(iterator+0x18)+0x10)==BITS[kind]
            self.subscriber_yields.append({'iterator':iterator,'kind':kind,'suspended_caller_cpu':caller,'bridge_return_cpu':after,'snapshot':self.snapshot()})
            return 1
        finally:
            self.active_machine=prior
            if not self.failure:
                assert self.managed_step_stack.pop()==iterator
                self.uc.context_restore(context); assert cpu(self)==caller

    def hook(self,uc,address,size,user):
        r=address-self.base; x=self.x; c,t,m,n=[self.reg(z) for z in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        if r==0x36D2DB:
            assert self.installed and self.rq(self.owner+0x58)==self.installed_delegate and len(self.scan)==75
            self.preserve(); assert self.snapshot()==self.before_start
            self.before_on_setup=self.snapshot(); self.capture_prior_five(); self.tail=True; self.phase='subscriber'
            self.visited.add(r); return
        if r==0x36D2ED:
            d=self.installed_delegate; assert c==self.animation and t==self.rq(d+0x28) and self.rq(d+0x18)==self.base+0x363060
            self.visited.add(r); return
        if r==0x363060:
            assert self.tail and c==self.animation and t==self.names['Method$CharacterShuffleAnimation.Animates()']
            self.callback_calls.append({'delegate':self.installed_delegate,'target':c,'method':t,'code':address,
                'caller_return_rva':hex(self.rq(self.reg(x.UC_X86_REG_RSP))-self.base)})
        if r==0x36D2F0 and self.tail: self.phase='shuffle_admission'
        if r==0x375E17: raise AssertionError('conditional null AudioEvents invoked callback')
        if r==0x4060 and self.managed_step_stack:
            p=self.managed_step_stack[-1]; kind=self.iterator_kinds[p]
            assert c==0 and t==self.bridge_type and m==p
            if self.service('subscriber_ienumerator_slot_zero',iterator=p,kind=kind,method_rva=hex(ENTRY[kind])):
                for name,val in [('RCX',p),('RDX',0),('R8',0),('R9',0)]: uc.reg_write(getattr(x,'UC_X86_REG_'+name),val)
                uc.reg_write(x.UC_X86_REG_RIP,self.base+ENTRY[kind])
            return
        active=self.phase in ('installer','subscriber')
        if active and r==0x1C7F160:
            assert c==self.animation and t in self.iterator_kinds and m==0 and self.rd(t+0x10)==0
            if self.service('subscriber_start_coroutine',owner=c,iterator=t,kind=self.iterator_kinds[t]):
                self.pending_start=(c,t); uc.emu_stop()
            return
        if active and r==0x2B7D40:
            name=self.metadata_names[c]
            assert name in ('System.Action_TypeInfo','CharacterShuffleAnimation.<Animate>d__6_TypeInfo',
                            'CharacterShuffleAnimation.<PlayAudioDelay>d__9_TypeInfo','UnityEngine.WaitForSeconds_TypeInfo')
            if self.service('subscriber_allocate',object_type=name,result=self.cursor):
                p=self.allocate(name+':subscriber:'+str(self.cursor-self.arena)); self.q(p,c)
                if name.endswith('<Animate>d__6_TypeInfo'): self.iterator_kinds[p]='animation'
                if name.endswith('<PlayAudioDelay>d__9_TypeInfo'): self.iterator_kinds[p]='audio'
                self.ret(p)
            return
        if active and r==0x4D5170:
            assert self.phase=='installer' and t==self.animation and m in self.method_bindings and n==0
            code=self.rq(m+8)
            if self.service('closed_action_ctor',identity=c,target=t,method=m,code=code):
                for off,value in ((0x10,code),(0x18,code),(0x20,t),(0x28,m),(0x38,self.action_stub),(0x40,t)): self.q(c+off,value)
                self.delegates[c]={'identity':c,'target':t,'method':m,'method_name':self.method_bindings[m]['name'],'code':code,
                    'fields':[self.rq(c+o) for o in (0x10,0x18,0x20,0x28,0x38,0x40)]}
                self.ret()
            return
        if active and r==0x116BCC0:
            assert self.phase=='installer' and c==0 and t in self.delegates and m==0
            if self.service('combine_existing_null',existing=c,added=t,result=t): self.ret(t)
            return
        if active and r==0x2B6FF0:
            caller=self.rq(self.reg(x.UC_X86_REG_RSP))-self.base
            if caller==0x3753C8:
                parent=self.managed_step_stack[-1]
                assert c==parent+0x28 and t==0 and self.rq(c)==self.rq(self.rq(self.animation+0x20)+0x20)==self.board
                assert self.board!=self.publication_list
                assert self.values(self.rq(c))==self.actors
            else: assert self.rq(c)==t
            allowed={self.owner+0x58,*[self.event_static+o for o in (0x18,0x30,0x38)]}
            assert c in allowed or any(p<=c<p+0x40 for p in self.iterator_kinds)
            if self.service('subscriber_barrier',destination=c,value=t,stored_value=self.rq(c),caller_rva=hex(caller)):
                self.ret()
            return
        if active and r==0x606FC0:
            assert c in self.draws_by_actor and self.metadata_names[t]=='Method$UnityEngine.Component.GetComponent<SingleCharacterDrawAnimation>()'
            result=self.draws_by_actor[c]['component']
            if self.service('draw_get_component',actor=c,component=result): self.ret(result)
            return
        if active and r==0x1C92130:
            row=next(row for row in self.draws_by_actor.values() if row['pivot']==c)
            value=list(struct.unpack('<III',bytes(uc.mem_read(t,12)))); assert value==[0,0,0] and m==0
            if self.service('transform_set_local_position',pivot=c,vector_bits=value):
                row['local_position_bits']=value; self.visual_effects.append({'kind':'position','actor':row['actor'],'pivot':c,'bits':value}); self.ret()
            return
        if active and r==0x50FB60:
            assert c==self.draws_by_actor[self.actors[0]]['pivot'] and self.reg(x.UC_X86_REG_XMM1)&0xFFFFFFFF==0x43C30000
            assert self.reg(x.UC_X86_REG_XMM2)&0xFFFFFFFF==ACQUISITION_BITS and n&255==0 and self.rq(self.reg(x.UC_X86_REG_RSP)+0x28)==0
            if self.service('tween_local_move_y',pivot=c,end_bits=0x43C30000,duration_bits=ACQUISITION_BITS,snapping=False):
                opaque=self.allocate('opaque_tween'); self.visual_effects.append({'kind':'tween_request','pivot':c,'end_bits':0x43C30000,
                    'duration_bits':ACQUISITION_BITS,'snapping':False,'unconsumed_return':opaque}); self.ret(opaque)
            return
        if active and r==0x1C961F0:
            kind=self.iterator_kinds[self.managed_step_stack[-1]]; bits=BITS[kind]
            assert self.reg(x.UC_X86_REG_XMM1)&0xFFFFFFFF==bits and m==0
            if self.service('subscriber_wait_constructor',identity=c,kind=kind,duration_bits=bits): self.d(c+0x10,bits); self.ret()
            return
        if r in self.subscriber_decoded:
            self.visited.add(r); return
        return super().hook(uc,address,size,user)


class SubscriberEngine(ShuffleEngine):
    def __init__(self,data,managed):
        super().__init__(data,managed)
        p=self.arena+0x180000+6*0x1000; self.owner_bindings[managed.animation]={'managed_actor':managed.animation,'native_owner':p,'key':107}
        self.write_q(p+0x40,self.owner_vtable); self.write_d(p+8,107); self.write_q(p+0x70,p+0x70); self.write_q(p+0x78,p+0x70)
        self.native_depth=0; self.borrowed=[]; self.engine_handoffs=[]; self.registrations=[]; self.native_frames=[]

    def storage_snapshot(self):
        out=super().storage_snapshot()
        if hasattr(self,'native_depth'):
            out['nested_runtime']={'depth':self.native_depth,'engine_frames':copy.deepcopy(self.native_frames),
                'managed_step_stack':list(self.managed.managed_step_stack),
                'borrowed_payloads':[{'identity':p,'body_bytes':raw.hex()} for p,raw in self.borrowed],
                'registrations':copy.deepcopy(self.registrations)}
        return out

    def preserve_previous(self):
        animation_owner=self.owner_bindings[self.managed.animation]['native_owner']
        for p,raw in self.completed_storage:
            if p in self.payload_registry and self.payload_registry[p]['actor']==self.managed.animation:
                assert bytes(self.uc.mem_read(p+0x10,0x78))==raw[0x10:]
            elif p==animation_owner:
                assert bytes(self.uc.mem_read(p,0x70))==raw[:0x70]
                assert bytes(self.uc.mem_read(p+0x80,0x80))==raw[0x80:]
            else: assert bytes(self.uc.mem_read(p,len(raw)))==raw
        for p,raw in self.borrowed: assert bytes(self.uc.mem_read(p+0x10,0x78))==raw
        for row in self.owner_snapshot():
            expected=[r['payload'] for r in self.registrations if r['native_owner']==row['native_owner']]
            assert row['payloads']==expected

    def queue_state(self):
        out=ScheduledEngine.queue_state(self)
        if not hasattr(self,'owner_bindings'): return out
        reverse={p:i for i,p in self.nodes.items()}; order=[]
        def walk(p):
            if p==self.head:return
            assert p in reverse
            walk(self.qword(p)); order.append(reverse[p]); walk(self.qword(p+0x10))
        walk(self.qword(self.head+8)); assert order==[r['id'] for r in out['entries']]
        for node,cached in self.records.items():
            live=bytes(self.uc.mem_read(node+0x20,0x40)); assert live==cached
            payload=struct.unpack_from('<Q',live,0x18)[0]; row=self.payload_registry[payload]; bits=BITS[row['kind']]
            assert self.managed.rd(self.managed.rq(row['pointer']+0x18)+0x10)==bits
            assert struct.unpack_from('<dq',live,0)==(1.0+struct.unpack('<f',struct.pack('<I',bits))[0],8)
            assert struct.unpack_from('<QQ',live,0x20)==(self.base+0x778B30,self.base+0x778BD0)
            assert struct.unpack_from('<III',live,0x30)==(row['key'],0xA,0)
            assert self.gc_targets[self.qword(payload+0x10)]==self.qword(payload+0x20)==row['pointer']
            assert self.qword(payload+0x58)==self.owner_bindings[row['actor']]['native_owner']
        for mirror,original in self.wait_mirrors.items():
            assert self.qword(mirror)==self.wait_class and bytes(self.uc.mem_read(mirror+8,0x18))==bytes(self.managed.uc.mem_read(original+8,0x18))
        if self.pending_insert is None: validate_tree(self.uc.mem_read,self.container,self.head,self.records,[0]*len(order))
        out['actual_tree_order']=order; return out

    def _on_code(self,uc,address,size,data):
        if not hasattr(self,'owner_bindings'): return super()._on_code(uc,address,size,data)
        r=address-self.base; m=self.managed; x=self.x86
        if r==0x779070:
            self.executed.add(r)
            if not m.service('engine_current_yield'): return
            payload=uc.reg_read(x.UC_X86_REG_RCX); row=self.payload_registry[payload]; original=m.rq(row['pointer']+0x18); bits=m.rd(original+0x10)
            assert bits==BITS[row['kind']]
            mirror=self.arena+0x1A0000+len(self.wait_mirrors)*0x100; self.wait_mirrors[mirror]=original
            uc.mem_write(mirror,bytes(m.uc.mem_read(original,0x20))); self.write_q(mirror,self.wait_class)
            self.join_trace.append({'kind':'current_yield_gateway','iterator':row['label'],'wait':original,'duration_bits':bits})
            uc.reg_write(x.UC_X86_REG_RCX,payload); uc.reg_write(x.UC_X86_REG_RDX,mirror); uc.reg_write(x.UC_X86_REG_RIP,self.base+0x779370); return
        if r==0x440F00:
            self.executed.add(r); record=bytes(uc.mem_read(uc.reg_read(x.UC_X86_REG_R8),0x40)); payload=struct.unpack_from('<Q',record,0x18)[0]
            row=self.payload_registry[payload]; assert struct.unpack_from('<I',record,0x30)[0]==row['key']
            self.pending_insert={'id':self.next_identity,'record':record,'node':self.next_node,'kind':row['kind'],'iterator':row['label'],
                'producer':{'time':self.producer_time,'frame':self.producer_frame,'duration_bits':BITS[row['kind']]}}
            self.next_identity+=1; NativeTree._on_code(self,uc,address,size,data); return
        return MultiOwnerEngine._on_code(self,uc,address,size,data)

    def run_native(self,name,*args):
        m=self.managed; prior=m.active_machine; m.active_machine=self; x=self.x86
        parent_context=self.uc.context_save(); parent_cpu=cpu(self); nested=self.native_depth>0
        parent_stack=bytes(self.uc.mem_read(parent_cpu['RSP']-0x1000,0x2000)) if nested else None
        sp=self.stack+0xF008-self.native_depth*0x4000; assert sp>=self.stack+0x1000
        self.native_depth+=1
        self.uc.mem_write(sp,struct.pack('<Q',self.stop)+bytes(0x28))
        for n in ('RAX','RCX','RDX','R8','R9','R10','R11'): self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),0)
        self.uc.reg_write(x.UC_X86_REG_RSP,sp)
        for i,n in enumerate(NONVOL[:8]): self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),0xFEA00000+i)
        for i in range(16): self.uc.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),0 if i<6 else 0xFED100+i)
        self.uc.reg_write(x.UC_X86_REG_EFLAGS,2); self.uc.reg_write(x.UC_X86_REG_MXCSR,0x1F80)
        for n,v in zip(('RCX','RDX','R8','R9'),args): self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        before=cpu(self); pc=self.base+self.routines[name][0]
        self.native_frames.append({'name':name,'root_sp':sp,'entry_cpu':before,
            'saved_parent_cpu':parent_cpu if nested else None,'parent_stack_sha256':sha(parent_stack) if nested else None})
        try:
            for _ in range(1024):
                self.pending_invoke=None; self.uc.emu_start(pc,self.stop,timeout=5_000_000,count=500000)
                if m.failure: raise AbortedPrefix()
                if m.paused is not None: m.acknowledge_pause(self); pc=self.uc.reg_read(x.UC_X86_REG_RIP); continue
                if self.pending_invoke is not None:
                    iterator,out,error=self.pending_invoke; self.pending_invoke=None
                    context=self.uc.context_save(); invocation_cpu=cpu(self)
                    payload=args[0]; borrowed=bytes(self.uc.mem_read(payload+0x10,0x78)); self.borrowed.append((payload,borrowed))
                    try: result=m.step(iterator)
                    finally:
                        if not m.failure: assert self.borrowed.pop()==(payload,borrowed)
                    assert bytes(self.uc.mem_read(payload+0x10,0x78))==borrowed
                    self.uc.context_restore(context); assert cpu(self)==invocation_cpu
                    self.uc.mem_write(out,bytes([result])); self.write_q(error,0)
                    self.join_trace.append({'kind':'managed_move_next_return','iterator':self.registry[iterator]['label'],'result':result})
                    self._return(); pc=self.uc.reg_read(x.UC_X86_REG_RIP); continue
                if self.uc.reg_read(x.UC_X86_REG_RIP)==self.stop: break
                pc=self.uc.reg_read(x.UC_X86_REG_RIP)
            else: raise AssertionError('nested engine driver budget exhausted')
            after=cpu(self); assert after['RSP']==sp+8 and all(after[n]==before[n] for n in NONVOL)
            return after['RAX']
        finally:
            m.active_machine=prior
            if not m.failure: self.native_depth-=1; self.native_frames.pop()
            if nested and not m.failure:
                assert bytes(self.uc.mem_read(parent_cpu['RSP']-0x1000,0x2000))==parent_stack
                self.uc.context_restore(parent_context); assert cpu(self)==parent_cpu
                self.engine_handoffs.append({'native_entry':name,'parent_cpu':parent_cpu,'restored_cpu':cpu(self),
                    'parent_stack_sha256':sha(parent_stack),'child_root_sp':sp,'depth':self.native_depth+1})

    def start_owner(self,actor,iterator):
        m=self.managed; row=self.owner_bindings[actor]; self.native_owner=row['native_owner']
        kind=m.iterator_kinds.get(iterator,'shuffle' if actor==m.owner else 'acquisition')
        prior=m.active_machine; m.active_machine=m; m.direct_gateway=True
        try:
            if not m.service('native_record_creation',actor=actor,iterator=iterator,native_owner=self.native_owner,owner_key=row['key'],kind=kind): raise AbortedPrefix()
        finally: m.direct_gateway=False; m.active_machine=prior
        payload=self.arena+0x1B0000+len(self.registry)*0x100; self.uc.mem_write(payload,bytes(0x88)); label=m.labels[iterator]
        rec={'pointer':iterator,'payload':payload,'kind':kind,'label':label,'actor':actor,'key':row['key']}
        self.registry[iterator]=rec; self.payload_registry[payload]=rec; self.payload_labels[payload]=label
        handle=17+len(self.registry); self.gc_targets[handle]=iterator
        for off,val in ((0x10,handle),(0x20,iterator),(0x58,self.native_owner)): self.write_q(payload+off,val)
        self.write_d(payload+0x18,2); self.write_d(payload+0x60,1)
        head=self.native_owner+0x70; last=self.qword(head+8)
        self.write_q(payload,head); self.write_q(payload+8,last); self.write_q(last,payload); self.write_q(head+8,payload)
        self.registrations.append({'registration_ordinal':len(self.registry),'iterator':iterator,'iterator_label':label,'kind':kind,
            'managed_owner':actor,'native_owner':self.native_owner,'owner_key':row['key'],'payload':payload})
        self.run_native('dispatch',payload,0); assert struct.unpack('<i',self.uc.mem_read(payload+0x60,4))[0]==2
        self.run_native('release_native',payload); assert struct.unpack('<i',self.uc.mem_read(payload+0x60,4))[0]==1
        wrapper=m.allocate('native_coroutine_handle:'+m.labels[actor]); m.q(wrapper+0x10,payload)
        self.completed_storage.extend([(payload,bytes(self.uc.mem_read(payload,0x88))),
            (row['native_owner'],bytes(self.uc.mem_read(row['native_owner'],0x100)))])
        return wrapper


def audit(game_root,dumper_root):
    game_root=Path(game_root); dumper_root=Path(dumper_root); inputs=load_inputs(game_root,dumper_root)
    names=('ascension_assets_audit','character_assets_audit','first_village_bluff_generation','character_init',
           'first_village_role_setup','first_village_publication_init','hunter_scheduled_publication',
           'first_village_start_queue','first_village_shuffle_admission','first_village_on_setup_binding')
    paths={n:ROOT/f'reports/{BUILD}_{n}.json' for n in names}; prior_hashes={n:sha(p.read_bytes()) for n,p in paths.items()}
    reports={n:json.loads(p.read_text(encoding='utf-8')) for n,p in paths.items()}
    reports={n:expand_snapshots(r) if 'snapshot_encoding' in r else r for n,r in reports.items()}
    from audit_ascension_assets import audit as profile_audit
    from audit_character_assets import audit as asset_audit
    assert reports['ascension_assets_audit']==json.loads(json.dumps(profile_audit(game_root,dumper_root)))
    assert reports['character_assets_audit']==json.loads(json.dumps(asset_audit(game_root,dumper_root)))
    assert reports['first_village_on_setup_binding']==json.loads(json.dumps(binding_audit(game_root,dumper_root)))
    original=scene_order(game_root,inputs,reports['character_assets_audit'])
    assert original==reports['first_village_start_queue']['original_scene_order']
    engine_data=(game_root/'UnityPlayer.dll').read_bytes(); verify_fingerprint(engine_data,ENGINE_SHA256)
    source_paths={Path(inspect.getfile(c)) for c in SubscriberWitness.__mro__[:-1]+SubscriberEngine.__mro__[:-1]}
    source_paths.update([Path(__file__),Path(inspect.getfile(Cursor)),Path(inspect.getfile(pool_snapshots)),Path(inspect.getfile(load_inputs)),
                        Path(inspect.getfile(profile_audit)),Path(inspect.getfile(binding_audit)),Path(inspect.getfile(verify_fingerprint))])
    source_hashes={p.relative_to(ROOT.parent).as_posix():sha(p.read_bytes()) for p in sorted(source_paths)}
    prior=reports['first_village_bluff_generation']; genrow=prior['generation_index_factor']['cases'][0]; poolrow=prior['pool_index_factors']['cases'][0]

    def run(paused=False,abort=None,abort_invocation='manage'):
        m=SubscriberWitness(inputs,reports['ascension_assets_audit'],reports['character_assets_audit'],reports['character_init'],
            reports['first_village_role_setup'],original_order=original,binding=reports['first_village_on_setup_binding'])
        constructors=[m.run('Character.ctor',a) for a in m.actors]; assert all(not r['failure'] for r in constructors)
        m.phase='setup'; setup=[]
        for name,this,arg,choices in [('GameData.SetupCurrentAscension',m.game,0,[]),('AscensionsData.ClearCurrentPickedScript',m.temporary,0,[]),
            ('AscensionsData.SetupCharactersCount',m.temporary,0,[0]),('AscensionsData.SetupStartingCharacters',m.temporary,0,[]),('Gameplay.GetCurrentScript',m.gameplay,0,[])]:
            row=m.run(name,this,arg,choices); assert not row['failure']; setup.append(row)
        m.q(m.gameplay_static+0x30,m.reg(m.x.UC_X86_REG_RAX))
        generation=m.run('Gameplay.GetRandomCharacters',m.gameplay,5,genrow['choices']); assert generation['final']['returned_order']==ORDER
        returned=m.reg(m.x.UC_X86_REG_RAX); e=SubscriberEngine(engine_data,m); e.set_producer(1.0,7)
        m.invocation='installation'; m.phase='installer'; m.pause_mode=paused
        installation=m.run('Animation.OnEnable',m.animation,stop_service=abort if abort_invocation=='installation' else None)
        if abort is not None and abort_invocation=='installation': return m,e,{'installation':installation}
        assert not installation['failure'] and installation['boundary'] is None
        m.installed_delegate=m.rq(m.owner+0x58); assert m.installed_delegate in m.delegates
        assert m.delegates[m.installed_delegate]['method_name']=='Animates'
        assert len(m.delegates)==4 and all(m.delegates[m.rq(m.event_static+o)]['method_name']=='ShuffleCards' for o in (0x18,0x30,0x38))
        m.installed_event_fields=tuple(m.rq(m.event_static+o) for o in (0x18,0x30,0x38))
        m.installed=True; m.invocation='manage'; m.phase='manage'; initial=m.snapshot()
        joined=m.run('Characters.ManageCharacters',m.owner,returned,poolrow['choices'],stop_service=abort if abort_invocation=='manage' else None)
        if abort is not None: return m,e,{'joined':joined}
        assert joined['failure'] is None and joined['boundary'] is None and m.return_cpu
        assert m.reg(m.x.UC_X86_REG_RIP)==m.stop and m.reg(m.x.UC_X86_REG_RSP)==m.stack+0x10010
        assert len(m.init_calls)==len(m.first_yields)==len(m.actions)==5 and len(m.callback_calls)==1
        assert len(m.subscriber_yields)==2 and [row['kind'] for row in m.subscriber_yields]==['audio','animation']
        assert len(m.nested_handoffs)==1 and len(e.engine_handoffs)==2
        assert m.scan==[[a,b] for a in START_ORDER for b in ORDER]
        assert {k:joined['final'][k] for k in GRAPH_KEYS}=={k:initial[k] for k in GRAPH_KEYS}
        assert joined['final']['pools']==poolrow['pool_identities_and_contents']
        protected=('actors','continuations','runtime_roles','source_role_saved_fields','publication','ordered_start','ordered_source_bindings')
        assert {k:joined['final'][k] for k in protected}=={k:m.before_on_setup[k] for k in protected}
        assert all(r['reference_count']==1 and r['owner_linked'] and r['gc_handle'] and r['cached_enumerator']==r['iterator'] for r in e.retained_native_records())
        owner_rows=e.owner_snapshot(); assert sorted(len(row['payloads']) for row in owner_rows)==[1]*6+[2]
        assert all(r['state']==1 and r['wait_bits']==ACQUISITION_BITS for r in joined['final']['continuations'])
        assert all(r['state']==1 and r['wait_bits']==BITS[r['kind']] for r in joined['final']['auxiliary_continuations'])
        assert joined['final']['shuffle_continuation']['state']==1 and joined['final']['shuffle_continuation']['wait_bits']==SHUFFLE_BITS
        assert [r['kind'] for r in e.registrations]==['acquisition']*5+['animation','audio','shuffle']
        assert len(m.visual_effects)==6 and all(row['local_position_bits']==[0,0,0] for row in m.draws_by_actor.values())
        total=len(installation['services'])+len(joined['services'])
        if paused: assert m.reentries==total==len(m.prefixes) and not m.reentry and not m.paused
        admissions=[]
        for event in e.join_trace:
            if event['kind']!='native_wait_inserted': continue
            i=event['id']; node=e.nodes[i]; record=bytes(e.uc.mem_read(node+0x20,0x40)); assert record==e.records[node]
            payload=struct.unpack_from('<Q',record,0x18)[0]; row=e.payload_registry[payload]; bits=BITS[row['kind']]
            assert event['producer']=={'time':1.0,'frame':7,'duration_bits':bits}
            admissions.append({'id':i,'kind':row['kind'],'iterator':row['pointer'],'iterator_label':row['label'],'actor':row['actor'],
                'payload':payload,'native_owner':e.qword(payload+0x58),'owner_key':row['key'],'producer':event['producer'],
                'deadline':struct.unpack_from('<d',record,0)[0],'frame_threshold':struct.unpack_from('<q',record,8)[0],
                'generation':struct.unpack_from('<I',record,0x38)[0],'queue':event['queue']})
        assert [r['kind'] for r in admissions]==['acquisition']*5+['audio','animation','shuffle']
        actual_order=e.queue_state()['actual_tree_order']
        assert [admissions[i]['kind'] for i in actual_order]==['animation']+['acquisition']*5+['audio','shuffle']
        return m,e,{'constructors':constructors,'setup':setup,'generation':generation,'installation':installation,'initial':initial,
            'pre_publication':m.pre_publication,'before_start':m.before_start,'before_on_setup':m.before_on_setup,
            'initializers':m.init_calls,'first_yields':m.first_yields,'act_init_returns':m.actions,'joined':joined,
            'ordered_comparisons':m.scan,
            'shuffle_first_yield':m.shuffle_first_yield,'subscriber_first_yields':m.subscriber_yields,
            'installed_delegate':copy.deepcopy(m.delegates[m.installed_delegate]),'callback_calls':m.callback_calls,
            'coroutine_registrations':e.registrations,'engine_admissions':admissions,'visual_effects':m.visual_effects,
            'nested_managed_handoffs':m.nested_handoffs,'nested_engine_handoffs':e.engine_handoffs,
            'completion':{'returned':True,'entry_cpu':m.entry_cpu,'before_return_cpu':m.return_cpu,
                'final_cpu':{'managed':cpu(m),'engine':cpu(e)},'root_return_sentinel':m.stop}}

    normal_m,normal_e,normal=run(); paused_m,paused_e,paused=run(True)
    assert normal==paused,first_difference(normal,paused)
    representatives={}
    for invocation,key in [('installation','installation'),('manage','joined')]:
        for ordinal,event in enumerate(normal[key]['services'],1):
            if event['phase'] in ('installer','subscriber'):
                representatives.setdefault((invocation,event['service']),ordinal)
    aborts=[]
    for (invocation,service),ordinal in representatives.items():
        m,e,result=run(abort=ordinal,abort_invocation=invocation); key='installation' if invocation=='installation' else 'joined'
        actual=result[key]; baseline=normal[key]['services']
        assert actual['failure']=='service:'+service and actual['services']==baseline[:ordinal]
        assert actual['final']==baseline[ordinal-1]['snapshot'],first_difference(actual['final'],baseline[ordinal-1]['snapshot'])
        aborts.append({'invocation':invocation,'service_ordinal':ordinal,'service':service,'final':actual['final']})
    assert all(sha(p.read_bytes())==prior_hashes[n] for n,p in paths.items())
    assert all(sha(p.read_bytes())==source_hashes[p.relative_to(ROOT.parent).as_posix()] for p in source_paths)
    return {'schema_version':'first_village_subscriber_admission_v1','build_id':BUILD,
        'decision':'Actual installed animation subscriber inserts retained nested waits before normal Manage return; compare native queue deadline order.',
        'domain':{'generation_row':0,'pool_row':0,'asset_order':ORDER,'scene_component_path_id':137027,'scene_manager_path_id':137026,
            'lifecycle':'OnEnable invocation supplied; actual body executes on hydrated original binding.',
            'delegate_lists':'existing onSetup and three later GameplayEvents lists supplied null before OnEnable',
            'audio_events':'OnPlaySfxOneShot supplied null; original runtime binding unresolved here; no original absence claim',
            'vector_static':'warm Vector3 class; 120 zero bytes supplied; inline first-state zero read executes',
            'tween':'exact request recorded; opaque return unconsumed; no completion simulated',
            'producer_time':1.0,'producer_signed_frame':7,'generation':0,'exit':'normal Manage return at 0x36d356'},
        'exclusions':['actual Unity lifecycle order/lifetime','multicast/repeated-enable/disable effects','SfxController/FMOD callback execution',
            'animation/audio/Shuffle state1 and acquisition resumes','queue drain','rendered/player observation','API failures/managed exceptions',
            'every aborted continuation equivalence'],
        'source_hashes':source_hashes,'prior_report_hashes':prior_hashes,'original_scene_order':original,
        'original_animation_binding':reports['first_village_on_setup_binding']['component'],
        'ordered_source_bindings':normal_m.ordered_source_bindings,'class_metadata_aliases':normal_m.class_metadata_aliases,
        'delegate_method_bindings':list(normal_m.method_bindings.values()),
        'selected_instruction_assertions':[{'rva':hex(a),'mnemonic':v[0],'operands':v[1]} for a,v in normal_m.subscriber_pins.items()],
        'bodies':normal_m.body_evidence+normal_m.new_bodies+normal_m.post_bodies+normal_m.bridge_bodies+[normal_m.shuffle_body]+normal_m.subscriber_bodies,
        **normal,'paused_reentered_prefixes':paused_m.prefixes,'restart_aborted_prefixes':aborts,
        'counters':{'constructors':len(normal['constructors']),'initializers':len(normal_m.init_calls),'act_init_calls':len(normal_m.actions),
            'start_calls':sum(r['trigger']==5 for r in normal_m.actions),'ordered_entries':len(normal_m.ordered_source_bindings),
            'ordered_comparisons':len(normal_m.scan),
            'acquisition_first_yields':len(normal_m.first_yields),'subscriber_first_yields':len(normal_m.subscriber_yields),
            'shuffle_first_yields':int(normal_m.shuffle_first_yield is not None),'on_setup_calls':len(normal_m.callback_calls),
            'installed_actions':len(normal_m.delegates),'coroutine_registrations':len(normal_e.registrations),
            'queue_records':len(normal_e.nodes),'physical_owners':len(normal_e.owner_bindings),
            'installation_services':len(normal['installation']['services']),'manage_services':len(normal['joined']['services']),
            'paused_reentered_prefixes':len(paused_m.prefixes),'reentry_tokens_consumed':paused_m.reentries,
            'restart_aborted_prefixes':len(aborts),'nested_managed_handoffs':len(normal_m.nested_handoffs),'nested_engine_handoffs':len(normal_e.engine_handoffs),
            'managed_native_addresses':len(normal_m.visited),'engine_native_addresses':len(normal_e.executed),'python_source_hashes':len(source_hashes)}}


def main():
    p=argparse.ArgumentParser(); p.add_argument('--game-root',required=True); p.add_argument('--dumper-root',required=True); p.add_argument('--output',required=True)
    a=p.parse_args(); output=Path(a.output); assert output.parent.is_dir()
    report=audit(a.game_root,a.dumper_root); assert report==json.loads(json.dumps(report))
    packed=pool_snapshots(report); assert expand_snapshots(json.loads(json.dumps(packed)))==report
    output.write_text(json.dumps(packed,indent=2,sort_keys=True,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':output.name,**report['counters']},sort_keys=True),flush=True)


if __name__=='__main__': main()

