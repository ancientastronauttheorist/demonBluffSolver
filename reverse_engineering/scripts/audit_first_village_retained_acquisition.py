"""Retained original N5 queue callbacks through animation and acquisition.

Native consumer/iterator/role/presentation bodies execute under explicit scene,
CLR, clone and engine providers. No Day request, renderer or player observation.
"""
import argparse
import copy
import inspect
import json
import math
import re
import struct
from pathlib import Path

from audit_first_village_subscriber_admission import SubscriberWitness, SubscriberEngine, BITS, ENTRY, NONVOL
from audit_first_village_start_queue import (
    BUILD, ORDER, START_ORDER, GRAPH_KEYS, ROOT, sha, cpu, first_difference,
    scene_order, load_inputs, ScheduledEngine, NativeTree, validate_tree,
    AbortedPrefix, ENGINE_SHA256, verify_fingerprint,
)
from audit_character_assets import Cursor as AssetCursor
from audit_first_village_on_setup_binding import audit as binding_audit
from audit_first_village_bluff_generation import BluffJoin
from audit_report_snapshots import pool_snapshots, expand_snapshots


ACQUISITION_METHODS={
    0x368410:'Character$$Reveal',0x365160:'Character$$GiveBluff',0x3682A0:'Character$$RevealReal',
    0x3694D0:'Character$$UpdateViewReal',0x3695A0:'Character$$UpdateView',0x3688B0:'Character$$SetupArt',
    0x3B4AB0:'CharacterData$$GetArt',0x3B4A20:'CharacterData$$GetArtType',0x364C40:'Character$$GetCharacterBluffIfAble',
    0x367B60:'Character$$RefreshView',
}


class RetainedAcquisition(SubscriberWitness):
    def __init__(self,*args,visual_assets,**kwargs):
        super().__init__(*args,**kwargs)
        import capstone
        self.consuming=False; self.resume_iterator=None; self.resume_actor=None
        self.resume_calls=[]; self.acquisition_calls=[]; self.acquisition_action=None
        self.acquired=[]; self.clone_calls=[]; self.presentation=[]; self.acquisition_bodies=[]
        self.copied_role_storage={}; self.selector_calls=[]
        self.acquisition_decoded=set(); self.pre_entry_hydration=[]; self.visual_assets=visual_assets
        slots={r['Address']:r for key in ('ScriptMetadata','ScriptMetadataMethod','ScriptString') for r in self.meta[key]}
        for start,name in ACQUISITION_METHODS.items():
            rows=[r for r in self.meta['ScriptMethod'] if r['Address']==start and r['Name']==name]; assert len(rows)==1,name
            end=min(r['Address'] for r in self.meta['ScriptMethod'] if r['Address']>start)
            raw=self.read_file_backed(start,end-start); ins=list(self.cs.disasm(raw,start))
            while ins and ins[-1].mnemonic=='int3': ins.pop()
            assert ins and ins[0].address==start and all(a.address+a.size==b.address for a,b in zip(ins,ins[1:]))
            self.instructions.update({i.address:i for i in ins}); self.acquisition_decoded.update(i.address for i in ins)
            self.acquisition_bodies.append({'name':name,'rva':hex(start),'signature':rows[0]['Signature'],
                'end_rva':hex(ins[-1].address+ins[-1].size),'body_sha256':sha(raw[:ins[-1].address+ins[-1].size-start])})
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
        null=list(self.cs.disasm(self.read_file_backed(0x3712B0,3),0x3712B0))
        assert [(i.mnemonic,i.op_str) for i in null]==[('xor','eax, eax'),('ret','')]
        self.instructions.update({i.address:i for i in null}); self.acquisition_decoded.update(i.address for i in null)
        self.acquisition_bodies.append({'name':'Role.GetRegisterAsRole/GetBluffIfAble','rva':'0x3712b0',
            'end_rva':'0x3712b3','body_sha256':sha(self.read_file_backed(0x3712B0,3))})
        for ident in ORDER:
            row=self.asset_fields[str(ident)]; klass=row['source_class']
            for offset,entry in ((0x288,0x3712B0),(0x278,0x3E49F0 if ident==21596 else 0x3712B0)):
                context=self.allocate('selector_context:'+str(ident)+':'+hex(offset))
                self.q(klass+offset,self.base+entry); self.q(klass+offset+8,context)
                self.pre_entry_hydration.append({'kind':'selector_vtable','asset':ident,'class':klass,
                    'offset':offset,'code':self.base+entry,'context':context})
        self.ui={}; self.ui_effects={}; self.opaque_sprites={}
        character=re.search(r'^public class Character :[^\n]*\n\{(.*?)\n\}',self.dump,re.M|re.S);assert character
        for field in ('public Image bg; // 0x120','public Image[] borders; // 0x128','public Image artBg; // 0x130'):
            assert field in character.group(1),field
        self.text_setter=self.stop+0xB00; self.color_setter=self.stop+0xB10
        self.uc.mem_write(self.text_setter,b'\xc3'*0x20)
        for a in self.actors:
            row={name:self.allocate('reveal_'+name+':'+self.labels[a]) for name in ('text','class','art','clip','background','art_background')}
            self.q(row['text'],row['class']); self.q(row['class']+0x558,self.text_setter)
            self.q(row['class']+0x560,row['class']+0x800); self.q(row['class']+0x2A8,self.color_setter)
            self.q(row['class']+0x2B0,row['class']+0x900)
            for o,k in ((0x40,'text'),(0x28,'art'),(0x30,'clip'),(0x120,'background')): self.q(a+o,row[k])
            for k in ('background','art_background'):self.q(row[k],row['class'])
            self.q(a+0x130,row['art_background']); self.q(a+0x128,self.array([],'card_borders:'+self.labels[a]))
            self.ui[a]=row; self.ui_effects[a]={'text':None,'color_bits':None,'active':{},'sprites':{}}
            self.pre_entry_hydration.append({'kind':'scene_reveal_components','actor':a,**row})
        for ident in ORDER:
            visual=visual_assets[str(ident)]; data=self.assets[ident]
            name=self.asset_fields[str(ident)]['name_identity']; text=self.asset_fields[str(ident)]['characterName']
            self.d(name+0x10,len(text)); self.uc.mem_write(name+0x14,text.encode('utf-16le'))
            for offset,key in ((0x90,'art'),(0x98,'art_cute'),(0xA0,'art_nice'),(0xA8,'art_animated'),
                               (0xB0,'randomArt'),(0xB8,'backgroundArt'),(0xC0,'currentSkin')):
                reference=visual['references'][key]
                if reference[1]==0: value=0
                else:
                    assert key!='currentSkin','selected native skin callees require separate evidence'
                    ref=tuple(reference)
                    if ref not in self.opaque_sprites: self.opaque_sprites[ref]=self.allocate('sprite_ref:'+str(ref))
                    value=self.opaque_sprites[ref]
                self.q(data+offset,value)
            for offset,key in ((0xD8,'color'),(0xE8,'artBgColor'),(0xF8,'cardBgColor'),(0x108,'cardBorderColor')):
                self.uc.mem_write(data+offset,struct.pack('<IIII',*visual['color_bits'][key]))
            self.pre_entry_hydration.append({'kind':'original_presentation_asset','asset':ident,'data':data,'name_identity':name,
                'source_object_sha256':visual['object_sha256'],'references':visual['references'],'color_bits':visual['color_bits']})
        # Named pre-entry hydration, before any constructor/Init or clone. Freeze
        # the resulting source graph; no later acquisition rewrites these assets.
        self.asset_storage=[(p,bytes(self.uc.mem_read(p,len(raw)))) for p,raw in self.asset_storage]
        self.resume_pins={0x37577B:('cmp','eax, 1'),0x375780:('mov','dword ptr [rdi + 0x10], 0xffffffff'),
            0x375791:('call','0x368410'),0x3753D2:('cmp','ecx, 1'),
            0x3754D1:('mov','dword ptr [rbx + 0x10], 0xffffffff'),
            0x368462:('mov','rax, qword ptr [r8 + 0x288]'),0x368469:('mov','r8, qword ptr [r8 + 0x290]'),
            0x368470:('call','rax'),0x368526:('mov','rax, qword ptr [r8 + 0x278]'),
            0x36852D:('mov','r8, qword ptr [r8 + 0x280]'),0x368534:('call','rax')}
        # Operands are checked after exact entry-based decode, never inferred
        # from the source-state model. Further callsite pins are added below.
        for address,wanted in self.resume_pins.items():
            assert address in self.instructions and (self.instructions[address].mnemonic,self.instructions[address].op_str)==wanted

    def snapshot(self):
        out=super().snapshot()
        if hasattr(self,'acquisition_calls'):
            out['retained_acquisition']={'resume_iterator':self.resume_iterator,'resume_actor':self.resume_actor,
                'acquired_actors':list(self.acquired),'clone_calls':copy.deepcopy(self.clone_calls),
                'action_calls':copy.deepcopy(self.acquisition_calls),'presentation':copy.deepcopy(self.presentation),
                'selector_calls':copy.deepcopy(self.selector_calls),
                'ui_effects':[{"actor":a,**copy.deepcopy(row)} for a,row in self.ui_effects.items()]}
        return out

    def preserve(self):
        if not getattr(self,'consuming',False): return super().preserve()
        assert all(bytes(self.uc.mem_read(p,len(raw)))==raw for p,raw in self.asset_storage)
        assert self.rq(self.owner+0x28)==self.start_array and bytes(self.uc.mem_read(self.start_array,0x1000))==self.order_storage
        assert all(bytes(self.uc.mem_read(p,len(raw)))==raw for p,raw in self.protected_setup_storage)
        assert self.rq(self.owner+0x58)==self.installed_delegate
        assert tuple(self.rq(self.event_static+o) for o in (0x18,0x30,0x38))==self.installed_event_fields
        for p,row in self.delegates.items(): assert [self.rq(p+o) for o in (0x10,0x18,0x20,0x28,0x38,0x40)]==row['fields']
        assert self.rq(self.audio_static)==0 and bytes(self.uc.mem_read(self.vector_static,120))==bytes(120)
        for a,raw in self.protected_actors.items():
            actual=bytes(self.uc.mem_read(a,len(raw)))
            if a!=self.resume_actor: assert actual==raw,self.labels[a]
            else:
                allowed=set(range(0x58,0x68))|set(range(0x170,0x178))
                assert all(x==y for i,(x,y) in enumerate(zip(actual,raw)) if i not in allowed),self.labels[a]
        for p,raw in self.protected_iterators.items():
            actual=bytes(self.uc.mem_read(p,len(raw)))
            if p!=self.resume_iterator: assert actual==raw,self.labels[p]
            else:
                allowed=set(range(0x10,0x14))
                if self.iterator_kinds.get(p)=='animation':allowed|=set(range(0x18,0x20))|set(range(0x28,0x40))
                assert all(x==y for i,(x,y) in enumerate(zip(actual,raw)) if i not in allowed),self.labels[p]
                if self.iterator_kinds.get(p)=='animation':
                    assert self.rq(p+0x28) in (self.board,0)
                    if self.rq(p+0x28):assert self.rd(p+0x34)==self.rd(self.board+0x1C)
        for p,raw in self.copied_role_storage.items():
            actual=bytes(self.uc.mem_read(p,0x48));assert actual[:0x28]==raw[:0x28] and actual[0x30:]==raw[0x30:]
        for a in self.actors:
            f=self.actor_snapshot(a)
            assert f['data']==self.assets[ORDER[self.actors.index(a)]] and f['state']==5 and f['previous']==20
            assert f['uses']==1 and f['revealed']==f['started']==f['killed_hidden']==f['killed_demon']==0
            assert f['acted_infos_storage']['count']==0 and f['acted_infos_storage']['version']==1
            assert f['hover_infos_storage']['count']==f['hover_infos_storage']['version']==0
            assert f['status_storage']['resistance_values']==[] and f['status_storage']['resistance_version']==0
        for actor,regions in self.protected_regions.items():
            for p,raw,kind in regions:
                actual=bytes(self.uc.mem_read(p,len(raw)))
                if actor!=self.resume_actor:assert actual==raw,(self.labels[actor],kind)
                elif kind=='runtime_role': assert actual[:0x28]==raw[:0x28] and actual[0x30:]==raw[0x30:]
                elif kind=='active_statuses' and actor==self.actors[0]:
                    allowed=set(range(0x18,0x20))|set(range(0x420,0x424))
                    assert all(x==y for i,(x,y) in enumerate(zip(actual,raw)) if i not in allowed)
                else:assert actual==raw,(self.labels[actor],kind)
        current=self.snapshot();assert {k:current[k] for k in GRAPH_KEYS}==self.protected_graph

    def step(self,iterator):
        if not self.consuming: return super().step(iterator)
        assert iterator in self.iterators or self.iterator_kinds.get(iterator)=='animation'
        assert self.rd(iterator+0x10)==1
        kind='acquisition' if iterator in self.iterators else 'animation'
        self.resume_iterator=iterator; self.resume_actor=self.rq(iterator+0x20) if kind=='acquisition' else None
        context=self.uc.context_save(); caller=cpu(self); sp=((caller['RSP']-0x2010)&~0xF)-8; assert sp%16==8
        parent_stack=bytes(self.uc.mem_read(caller['RSP']-0x800,0x1000))
        end=self.stop+0x900; self.q(sp,end); self.uc.mem_write(self.bridge_out,b'\x7f')
        for n in ('RAX','R10','R11','R8','R9'): self.uc.reg_write(getattr(self.x,'UC_X86_REG_'+n),0)
        for i in range(6): self.uc.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),0)
        self.uc.reg_write(self.x.UC_X86_REG_RSP,sp); self.uc.reg_write(self.x.UC_X86_REG_RCX,iterator)
        self.uc.reg_write(self.x.UC_X86_REG_RDX,self.bridge_out); prior=self.active_machine; self.active_machine=self
        before=self.snapshot()
        try:
            self.drive_managed(self.base+0x1C8A780,end)
            if self.failure: raise AbortedPrefix()
            after=cpu(self); assert after['RIP']==end and after['RSP']==sp+8 and all(after[n]==caller[n] for n in NONVOL)
            result=self.uc.mem_read(self.bridge_out,1)[0]; assert result in (0,1)
            assert self.rd(iterator+0x10)==(1 if result else 0xFFFFFFFF)
            self.preserve();assert bytes(self.uc.mem_read(caller['RSP']-0x800,0x1000))==parent_stack
            self.resume_calls.append({'iterator':iterator,'kind':kind,'actor':self.resume_actor,'result':result,
                'before':before,'after':self.snapshot(),'bridge_return_cpu':after,'suspended_caller_cpu':caller})
            if kind=='acquisition':
                assert result==0; self.acquired.append(self.resume_actor)
                self.protected_actors[self.resume_actor]=bytes(self.uc.mem_read(self.resume_actor,0x1B8))
                self.protected_regions[self.resume_actor]=[(p,bytes(self.uc.mem_read(p,len(raw))),k) for p,raw,k in self.protected_regions[self.resume_actor]]
                if self.resume_actor==self.actors[0]:assert self.rq(self.resume_actor+0x170)==self.clone_calls[0]['result']
            self.protected_iterators[iterator]=bytes(self.uc.mem_read(iterator,0x40 if kind=='animation' else 0x28))
            return result
        finally:
            self.active_machine=prior
            if not self.failure:
                self.uc.context_restore(context); assert cpu(self)==caller
                self.resume_iterator=None; self.resume_actor=None

    def hook(self,uc,address,size,user):
        if not getattr(self,'consuming',False): return super().hook(uc,address,size,user)
        x=self.x; r=address-self.base
        c,t,m,n=[self.reg(z) for z in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        caller=self.rq(self.reg(x.UC_X86_REG_RSP))-self.base
        if r==0x4060:
            p=self.resume_iterator; kind='acquisition' if p in self.iterators else 'animation'
            entry=0x3756B0 if kind=='acquisition' else ENTRY[kind]
            assert c==0 and t==self.bridge_type and m==p
            if self.service('retained_ienumerator_slot_zero',iterator=p,kind=kind,method_rva=hex(entry)):
                for k,v in [('RCX',p),('RDX',0),('R8',0),('R9',0)]: uc.reg_write(getattr(x,'UC_X86_REG_'+k),v)
                uc.reg_write(x.UC_X86_REG_RIP,self.base+entry)
            return
        # The state1 adapters below qualify every mutable service by actual
        # receiver and current native callback, rather than fresh actor fixtures.
        return self.resume_hook(uc,address,size,user,c,t,m,n,caller)

    def resume_hook(self,uc,address,size,user,c,t,m,n,caller):
        x=self.x; r=address-self.base; a=self.resume_actor
        if r in (0x3712B0,0x3E49F0):
            assert a is not None and c==self.rq(self.rq(a+0x50)+0x140) and t==a
            self.selector_calls.append({'rva':hex(r),'receiver':c,'source_asset':self.asset_ids[self.rq(a+0x50)],
                                        'actor':a,'context':m,'caller_return_rva':hex(caller)})
        if r==0x3645C0:
            assert a is not None and c==a and t in (3,7) and m==0
            self.acquisition_action={'actor':a,'trigger':t,'caller_return_rva':hex(caller)}
            self.acquisition_calls.append(copy.deepcopy(self.acquisition_action))
        roles={self.rq(actor+offset) for actor in self.actors for offset in (0x168,0x170)}-{0}
        if r in self.role_entries and (r!=0x33ED50 or c in roles):
            self.acquisition_calls.append({'role_method_rva':hex(r),'metadata_aliases':self.role_entries[r],
                                          'receiver':c,'rdx':t,'r8':m,'r9':n,'caller_return_rva':hex(caller)})
        if r in self.instructions:
            self.visited.add(r); return
        if address==self.stop+0x600: raise AssertionError('non-Day acquisition invoked onActed callback')
        if r==0x603240:
            assert a is not None and c==self.rq(self.assets[21614]+0x140) and a==self.actors[0]
            assert t==self.names['Method$ClassConv.CreateCopyNonGeneric<Role>()'] and m==0 and caller==0x3651F3
            if self.service('acquisition_clone',source=c,source_asset=21614,result=self.cursor):
                p=self.allocate('acquired_bluff_clone'); uc.mem_write(p,bytes(uc.mem_read(c,0x48)))
                self.copied_role_storage[p]=bytes(uc.mem_read(p,0x48))
                self.clone_calls.append({'source':c,'result':p,'actor':a,'copied_bytes_sha256':sha(bytes(uc.mem_read(c,0x48)))})
                self.ret(p)
            return
        if r==0xB45070 or r==0x41A0:
            assert c in self.enum_lists
            values=[self.rd(self.rq(c+0x10)+0x20+i*4) for i in range(self.rd(c+0x18))]
            if r==0x41A0: assert a==self.actors[0] and c==self.scene[a]['active'] and t==25 and values==[]
            if self.service('acquisition_enum_contains' if r==0xB45070 else 'acquisition_enum_add',list_identity=c,
                            status=t&0xFFFFFFFF,values=values,method=self.metadata_names.get(m)):
                if r==0xB45070: self.ret(int((t&0xFFFFFFFF) in values))
                else:
                    self.d(self.rq(c+0x10)+0x20+4*len(values),t); self.d(c+0x18,len(values)+1)
                    self.d(c+0x1C,self.rd(c+0x1C)+1); self.ret()
            return
        if r==0x2B7D40:
            name=self.metadata_names[c]
            assert name in ('UnityEngine.WaitForSeconds_TypeInfo','Character.<>c__DisplayClass125_0_TypeInfo','System.Action<ActedInfo>_TypeInfo')
            if self.service('retained_allocate',object_type=name,result=self.cursor):
                p=self.allocate(name+':retained:'+str(self.cursor)); self.q(p,c); self.ret(p)
            return
        if r==0x4D5B60:
            assert a is not None and self.rq(t+0x10)==a and self.rd(t+0x18) in (3,7)
            assert m==self.names['Method$Character.<>c__DisplayClass125_0.<RoleAct>b__0()'] and n==0
            if self.service('acquisition_role_delegate_ctor',identity=c,closure=t,actor=a,trigger=self.rd(t+0x18)):
                for off,val in ((0x18,self.stop+0x600),(0x20,t),(0x28,m),(0x40,t)): self.q(c+off,val)
                self.ret()
            return
        if r==0x2B6FF0:
            assert self.rq(c)==t
            if self.service('retained_barrier',destination=c,value=t,stored_value=self.rq(c)): self.ret()
            return
        if r==0x1C961F0:
            assert self.iterator_kinds.get(self.resume_iterator)=='animation' and self.reg(x.UC_X86_REG_XMM1)&0xFFFFFFFF==BITS['animation'] and m==0
            if self.service('animation_reyield_wait_ctor',identity=c,duration_bits=BITS['animation']): self.d(c+0x10,BITS['animation']); self.ret()
            return
        if r==0x50FB60:
            current=self.rq(self.resume_iterator+0x38); assert current in self.actors and c==self.draws_by_actor[current]['pivot']
            assert self.reg(x.UC_X86_REG_XMM1)&0xFFFFFFFF==0x43C30000 and self.reg(x.UC_X86_REG_XMM2)&0xFFFFFFFF==BITS['acquisition']
            assert n&255==0 and self.rq(self.reg(x.UC_X86_REG_RSP)+0x28)==0
            if self.service('animation_retained_tween',actor=current,pivot=c,end_bits=0x43C30000,duration_bits=BITS['acquisition']):
                p=self.allocate('retained_tween'); self.visual_effects.append({'kind':'tween_request','actor':current,'pivot':c,'unconsumed_return':p}); self.ret(p)
            return
        if r==0x606FC0:
            assert c in self.draws_by_actor and self.metadata_names[t]=='Method$UnityEngine.Component.GetComponent<SingleCharacterDrawAnimation>()'
            assert c==self.rq(self.resume_iterator+0x38)
            if self.service('animation_retained_component',actor=c,component=self.draws_by_actor[c]['component']): self.ret(self.draws_by_actor[c]['component'])
            return
        if r in (0x1C79FD0,0x1C7D810,0x1D49700,0x1D41930) or address in (self.text_setter,self.color_setter):
            assert a is not None; row=self.ui[a]; effects=self.ui_effects[a]
            if r==0x1C79FD0:
                assert c in (row['art'],row['clip']) and t==0
                name='reveal_image_game_object'; details={'image':c,'game_object':c+0x800}
            elif r==0x1C7D810:
                assert c in (row['art']+0x800,row['clip']+0x800) and t&255 in (0,1) and m==0
                name='reveal_image_active'; details={'game_object':c,'active':t&255}
            elif r==0x1D49700:
                assert c in (row['art'],row['art_background']) and t in self.opaque_sprites.values() and m==0,(hex(caller),self.labels.get(c),self.labels.get(t),hex(m))
                if caller==0x3683F5:assert c==row['art_background'] and t==self.rq(self.rq(a+0x50)+0xB8)
                name='reveal_image_sprite'; details={'image':c,'sprite':t}
            elif address==self.text_setter:
                assert c==row['text'] and m==row['class']+0x800
                name='reveal_name_text'; details={'text_component':c,'string_identity':t}
            else:
                assert c in (row['text'],row['background'],row['art_background']) and m in (0,row['class']+0x900)
                name='reveal_component_color'; details={'component':c,'color_bits':list(struct.unpack('<IIII',bytes(uc.mem_read(t,16))))}
            if self.service(name,**details):
                self.presentation.append({'actor':a,'service':name,**details})
                if r==0x1C79FD0: self.ret(c+0x800)
                else:
                    if r==0x1C7D810: effects['active'][str(c)]=t&255
                    elif r==0x1D49700: effects['sprites'][str(c)]=t
                    elif address==self.text_setter: effects['text']=t
                    else: effects['color_bits']=details['color_bits']
                    self.ret()
            return
        if r in (0x282580,0xF74DF0,0x1C4B450):
            name={0x282580:'retained_trigger_box',0xF74DF0:'retained_log_format',0x1C4B450:'retained_log'}[r]
            if self.service(name):
                p=0
                if r!=0x1C4B450:
                    p=self.allocate(name+':'+str(self.cursor))
                    if r==0x282580:self.d(p+0x10,self.rd(t))
                self.ret(p)
            return
        if r==0xF7B1B0:
            assert t==0
            length=self.rd(c+0x10);assert length<128
            text=bytes(uc.mem_read(c+0x14,length*2)).decode('utf-16le').upper()
            if self.service('reveal_uppercase',source_string=c,result=self.cursor):
                p=self.allocate('uppercase_name:'+text);self.d(p+0x10,len(text));uc.mem_write(p+0x14,text.encode('utf-16le'));self.ret(p)
            return
        if r==0x1C86600:
            choice=self.choices[0]; assert c<=choice<t
            draw={'minimum':c,'maximum_exclusive':t,'index':choice,'width':t-c,'caller_return_rva':hex(caller)}
            if self.service('acquisition_rng',**draw): self.choices.pop(0); self.draws.append(draw); self.ret(choice)
            return
        if r in (0x1C82480,0x1C822C0):
            known={0,*self.assets.values(),*self.opaque_sprites.values()}
            known.update(p for row in self.ui.values() for k,p in row.items() if k!='class')
            assert c in known and t==0,(hex(r),hex(c),self.labels.get(c),hex(caller))
            if self.service('acquisition_unity_live' if r==0x1C82480 else 'acquisition_unity_null',left=c,right=t): self.ret(int(c!=0) if r==0x1C82480 else int(c==0))
            return
        return BluffJoin.hook(self,uc,address,size,user)


class RetainedEngine(SubscriberEngine):
    def __init__(self,*args):
        super().__init__(*args)
        self.consumer_mode=False; self.active_payload=None; self.released=set(); self.insertions={}
        start,end=0x778BD0,0x778CC0
        section=next(s for s in self.pe.sections if s.VirtualAddress<=start<s.VirtualAddress+s.SizeOfRawData)
        assert end<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(start,end-start);assert len(raw)==end-start
        decoded={i.address:(i.mnemonic,i.op_str) for i in self.cs.disasm(raw,start)}
        self.release_pins={0x778BD6:('dec','dword ptr [rcx + 0x60]'),0x778BF4:('mov','byte ptr [rcx + 0x64], 1'),
            0x778C35:('mov','qword ptr [rbx], rdi'),0x778C38:('mov','qword ptr [rbx + 8], rdi'),
            0x778C74:('mov','dword ptr [rbx + 0x18], edi'),0x778C77:('mov','qword ptr [rbx + 0x10], rdi')}
        assert all(decoded.get(a)==v for a,v in self.release_pins.items())
        self.release_body={'name':'UnityPlayer retained reference release','rva':hex(start),'interval_end_rva':hex(end),'body_sha256':sha(raw)}

    def preserve_previous(self):
        if not self.consumer_mode: return super().preserve_previous()
        for p,raw in self.pending_storage.items():
            actual=bytes(self.uc.mem_read(p,len(raw)))
            if p in self.released:assert actual==self.released_storage[p]
            elif p==self.active_payload:
                allowed=set(range(0,0x1C))|set(range(0x60,0x65))
                assert all(x==y for i,(x,y) in enumerate(zip(actual,raw)) if i not in allowed),[(hex(i),x,y) for i,(x,y) in enumerate(zip(actual,raw)) if x!=y]
                assert self.qword(p+0x10) in (0,int.from_bytes(raw[0x10:0x18],'little'))
                assert 0<=struct.unpack_from('<i',actual,0x60)[0]<=3
                assert struct.unpack_from('<I',actual,0x18)[0] in (0,2) and actual[0x64] in (0,1)
            else:
                # Native unlink updates the surviving same-owner neighbour's
                # links. All payload body/GC/ref fields remain protected.
                if self.payload_registry[p]['actor']==self.managed.animation:
                    assert bytes(self.uc.mem_read(p+0x10,0x78))==raw[0x10:]
                else: assert bytes(self.uc.mem_read(p,len(raw)))==raw
        for p,raw in self.owner_storage.items():
            actual=bytes(self.uc.mem_read(p,len(raw)));assert actual[:0x70]==raw[:0x70] and actual[0x80:]==raw[0x80:]
        for p,raw in self.borrowed: assert bytes(self.uc.mem_read(p+0x10,0x78))==raw
        for row in self.owner_snapshot():
            expected=[r['payload'] for r in self.registrations if r['native_owner']==row['native_owner'] and r['payload'] not in self.released]
            if self.active_payload in expected:
                # Cleanup can unlink before its provider free call; constrain
                # this sole active deletion, never another payload/order.
                assert row['payloads'] in (expected,[p for p in expected if p!=self.active_payload])
            else: assert row['payloads']==expected

    def queue_state(self):
        if not self.consumer_mode: return super().queue_state()
        out=ScheduledEngine.queue_state(self); reverse={p:i for i,p in self.nodes.items()}; order=[]
        def walk(p):
            if p==self.head:return
            assert p in reverse
            walk(self.qword(p)); order.append(reverse[p]); walk(self.qword(p+0x10))
        walk(self.qword(self.head+8)); assert order==[r['id'] for r in out['entries']]
        for node,cached in self.records.items():
            live=bytes(self.uc.mem_read(node+0x20,0x40)); assert live==cached
            payload=struct.unpack_from('<Q',live,0x18)[0]; row=self.payload_registry[payload]
            event=self.insertions[reverse[node]]; producer=event['producer']; bits=BITS[row['kind']]
            assert struct.unpack_from('<dq',live,0)==(producer['time']+struct.unpack('<f',struct.pack('<I',bits))[0],producer['frame']+1)
            assert struct.unpack_from('<QQ',live,0x20)==(self.base+0x778B30,self.base+0x778BD0)
            assert struct.unpack_from('<III',live,0x30)==(row['key'],0xA,event['record_generation'])
            assert self.gc_targets[self.qword(payload+0x10)]==self.qword(payload+0x20)==row['pointer']
            assert self.qword(payload+0x58)==self.owner_bindings[row['actor']]['native_owner']
        for mirror,original in self.wait_mirrors.items():
            assert self.qword(mirror)==self.wait_class and bytes(self.uc.mem_read(mirror+8,0x18))==bytes(self.managed.uc.mem_read(original+8,0x18))
        if self.pending_insert is None: validate_tree(self.uc.mem_read,self.container,self.head,self.records,[0]*len(order))
        out['actual_tree_order']=order; return out

    def _on_code(self,uc,address,size,data):
        if not self.consumer_mode: return super()._on_code(uc,address,size,data)
        r=address-self.base; m=self.managed; x=self.x86
        self.executed.add(r if address>=self.base else 'gateway:'+hex(address-self.stop))
        gateways={self.param_count:'parameter_count',self.invoke:'runtime_invoke',self.owner_context:'owner_context',
            self.gc_write:'gc_write',self.gc_target:'gc_target',self.free_handle:'gc_free',self.object_class:'yield_object_class',
            self.subclass:'wait_subclass',self.base+0x779070:'current_yield',self.base+0x677920:'tree_allocator',
            self.base+0x17D6A84:'numeric_conversion',self.base+0x6E78E0:'profiler',self.base+0x355F00:'profile_sink',
            self.base+0x151EF0:'retained_owner_lookup',self.base+0x17A7808:'payload_free'}
        if address in gateways and not m.service('engine_'+gateways[address]):return
        if address==self.owner_context:
            c=uc.reg_read(x.UC_X86_REG_RCX); assert c in [r['native_owner']+0x40 for r in self.owner_bindings.values()]
            out=uc.reg_read(x.UC_X86_REG_RDX); self.write_q(out,0); self._return(out);return
        if r==0x151EF0:
            key=struct.unpack('<I',uc.mem_read(uc.reg_read(x.UC_X86_REG_RDX),4))[0]
            row=next(r for r in self.owner_bindings.values() if r['key']==key)
            uc.mem_write(self.owner_entry,struct.pack('<QQQ',key,0,row['native_owner']))
            self.join_trace.append({'kind':'owner_lookup_gateway','id':self.visit_identity,'key':key,'native_owner':row['native_owner'],'outcome':'valid'})
            self._return(self.owner_entry);return
        if r==0x779070:
            payload=uc.reg_read(x.UC_X86_REG_RCX); row=self.payload_registry[payload]; original=m.rq(row['pointer']+0x18)
            bits=m.rd(original+0x10); assert bits==BITS[row['kind']]
            mirror=self.arena+0x1A0000+len(self.wait_mirrors)*0x100; self.wait_mirrors[mirror]=original
            uc.mem_write(mirror,bytes(m.uc.mem_read(original,0x20))); self.write_q(mirror,self.wait_class)
            self.join_trace.append({'kind':'current_yield_gateway','iterator':row['label'],'wait':original,'duration_bits':bits})
            uc.reg_write(x.UC_X86_REG_RCX,payload); uc.reg_write(x.UC_X86_REG_RDX,mirror); uc.reg_write(x.UC_X86_REG_RIP,self.base+0x779370);return
        if r==0x440F00:
            record=bytes(uc.mem_read(uc.reg_read(x.UC_X86_REG_R8),0x40)); payload=struct.unpack_from('<Q',record,0x18)[0]
            row=self.payload_registry[payload]; assert struct.unpack_from('<I',record,0x30)[0]==row['key']
            item={'id':self.next_identity,'record':record,'node':self.next_node,'kind':row['kind'],'iterator':row['label'],
                'record_generation':struct.unpack_from('<I',record,0x38)[0],
                'producer':{'time':self.producer_time,'frame':self.producer_frame,'duration_bits':BITS[row['kind']]}}
            self.pending_insert=item; self.insertions[item['id']]=copy.deepcopy(item); self.next_identity+=1
            NativeTree._on_code(self,uc,address,size,data);return
        if r==0x778B30:
            payload=uc.reg_read(x.UC_X86_REG_RDX); assert payload in self.payload_registry
            self.active_payload=payload
        if r==0x17A7808:
            p=uc.reg_read(x.UC_X86_REG_RCX); assert p==self.active_payload
            assert self.qword(p+0x10)==0 and struct.unpack('<i',uc.mem_read(p+0x60,4))[0]==0
            self.preserve_previous();self.released.add(p);self.released_storage[p]=bytes(uc.mem_read(p,0x88))
        if r==0x1049F20: raise AssertionError('retained physical owner mismatch')
        return ScheduledEngine._on_code(self,uc,address,size,data)

    def run_native(self,name,*args):
        if not self.consumer_mode:return super().run_native(name,*args)
        m=self.managed; prior=m.active_machine; m.active_machine=self; x=self.x86
        assert self.native_depth==0
        sp=self.stack+0xF008; self.native_depth=1
        self.uc.mem_write(sp,struct.pack('<Q',self.stop)+bytes(0x28))
        for n in ('RAX','RCX','RDX','R8','R9','R10','R11'): self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),0)
        self.uc.reg_write(x.UC_X86_REG_RSP,sp)
        for i,n in enumerate(NONVOL[:8]):self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),0xFEA00000+i)
        for i in range(16):self.uc.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),0 if i<6 else 0xFED100+i)
        self.uc.reg_write(x.UC_X86_REG_EFLAGS,2);self.uc.reg_write(x.UC_X86_REG_MXCSR,0x1F80)
        for n,v in zip(('RCX','RDX','R8','R9'),args):self.uc.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        before=cpu(self); pc=self.base+self.routines[name][0]
        self.native_frames.append({'name':name,'root_sp':sp,'entry_cpu':before,'saved_parent_cpu':None,'parent_stack_sha256':None})
        try:
            for _ in range(1024):
                self.pending_invoke=None;self.uc.emu_start(pc,self.stop,timeout=5_000_000,count=500000)
                if m.failure:raise AbortedPrefix()
                if m.paused is not None:m.acknowledge_pause(self);pc=self.uc.reg_read(x.UC_X86_REG_RIP);continue
                if self.pending_invoke is not None:
                    iterator,out,error=self.pending_invoke;self.pending_invoke=None
                    context=self.uc.context_save();invocation_cpu=cpu(self)
                    stack=bytes(self.uc.mem_read(invocation_cpu['RSP']-0x800,0x1000))
                    payload=self.registry[iterator]['payload'];assert payload==self.active_payload
                    borrowed=bytes(self.uc.mem_read(payload+0x10,0x78));self.borrowed.append((payload,borrowed))
                    result=m.step(iterator)
                    assert self.borrowed.pop()==(payload,borrowed) and bytes(self.uc.mem_read(payload+0x10,0x78))==borrowed
                    assert bytes(self.uc.mem_read(invocation_cpu['RSP']-0x800,0x1000))==stack
                    self.uc.context_restore(context);assert cpu(self)==invocation_cpu
                    self.uc.mem_write(out,bytes([result]));self.write_q(error,0)
                    self.join_trace.append({'kind':'managed_move_next_return','iterator':self.registry[iterator]['label'],'result':result})
                    self._return();pc=self.uc.reg_read(x.UC_X86_REG_RIP);continue
                if self.uc.reg_read(x.UC_X86_REG_RIP)==self.stop:break
                pc=self.uc.reg_read(x.UC_X86_REG_RIP)
            else:raise AssertionError('retained consumer budget exhausted')
            after=cpu(self);assert after['RIP']==self.stop and after['RSP']==sp+8 and all(after[n]==before[n] for n in NONVOL)
            self.active_payload=None;self.native_depth=0;self.native_frames.pop()
            return after['RAX']
        finally:m.active_machine=prior

    def drain(self,time,frame,phase=2):
        m=self.managed;before=self.queue_state();start=len(self.join_trace);m.events=[];m.failure=None
        self.set_clock(time,frame);self.set_producer(time,frame)
        try:self.run_native('drain',self.owner,phase)
        except AbortedPrefix: assert m.failure
        out={'input':{'time':time,'frame':frame,'phase':phase,'generation_before':before['generation']},
            'before':before,'after':self.queue_state(),'events':copy.deepcopy(self.join_trace[start:]),
            'services':list(m.events),'failure':m.failure,'managed':m.snapshot(),'engine_cpu':cpu(self),'managed_cpu':cpu(m)}
        return out


def read_visual_assets(game_root,records):
    import UnityPy
    environment=UnityPy.load(str(Path(game_root)/'Demon Bluff_Data/sharedassets0.assets'))
    wanted={r['path_id']:r for r in records if r['path_id'] in ORDER}; result={}
    for obj in environment.objects:
        if obj.path_id not in wanted: continue
        raw=obj.get_raw_data(); assert sha(raw).upper()==wanted[obj.path_id]['object_sha256'].upper()
        cursor=AssetCursor(raw)
        strings=[cursor.string() for _ in range(5)]; assert strings[3]==wanted[obj.path_id]['characterName']
        cursor.read('i'); cursor.read('ifi'); cursor.array(cursor.pointer)
        for _ in range(4): cursor.string()
        cursor.array(cursor.string)
        for _ in range(3): cursor.string()
        refs={k:list(cursor.pointer()) for k in ('art','art_cute','art_nice','art_animated','randomArt','backgroundArt','currentSkin')}
        cursor.array(cursor.pointer); cursor.array(cursor.pointer)
        colors={k:list(cursor.read('IIII')) for k in ('color','artBgColor','cardBgColor','cardBorderColor')}
        result[str(obj.path_id)]={'object_sha256':sha(raw),'prefix_bytes':cursor.offset,'references':refs,'color_bits':colors}
        assert cursor.offset==int(wanted[obj.path_id]['field_offsets']['additionalStatuses'],16)
    assert set(result)==set(map(str,ORDER)); return result


def audit(game_root,dumper_root,probe=False):
    game_root=Path(game_root);dumper_root=Path(dumper_root);inputs=load_inputs(game_root,dumper_root)
    names=('ascension_assets_audit','character_assets_audit','first_village_bluff_generation','character_init',
           'first_village_role_setup','first_village_publication_init','hunter_scheduled_publication',
           'first_village_start_queue','first_village_shuffle_admission','first_village_on_setup_binding','first_village_subscriber_admission')
    paths={n:ROOT/f'reports/{BUILD}_{n}.json' for n in names}
    assert all(p.is_file() for p in paths.values())
    prior_hashes={n:sha(p.read_bytes()) for n,p in paths.items()}
    reports={n:json.loads(p.read_text(encoding='utf-8')) for n,p in paths.items()}
    reports={n:expand_snapshots(r) if 'snapshot_encoding' in r else r for n,r in reports.items()}
    original=scene_order(game_root,inputs,reports['character_assets_audit'])
    assert original==reports['first_village_start_queue']['original_scene_order']
    visual=read_visual_assets(game_root,reports['character_assets_audit']['records'])
    engine_data=(game_root/'UnityPlayer.dll').read_bytes();verify_fingerprint(engine_data,ENGINE_SHA256)
    source_paths={Path(inspect.getfile(c)) for c in RetainedAcquisition.__mro__[:-1]+RetainedEngine.__mro__[:-1]}
    source_paths.update([Path(__file__),Path(inspect.getfile(AssetCursor)),Path(inspect.getfile(pool_snapshots)),
                        Path(inspect.getfile(load_inputs)),Path(inspect.getfile(binding_audit)),Path(inspect.getfile(verify_fingerprint))])
    source_hashes={p.relative_to(ROOT.parent).as_posix():sha(p.read_bytes()) for p in sorted(source_paths)}
    prior=reports['first_village_bluff_generation'];genrow=prior['generation_index_factor']['cases'][0];poolrow=prior['pool_index_factors']['cases'][0]

    def run(paused=False,abort=None):
        m=RetainedAcquisition(inputs,reports['ascension_assets_audit'],reports['character_assets_audit'],reports['character_init'],
            reports['first_village_role_setup'],original_order=original,binding=reports['first_village_on_setup_binding'],visual_assets=visual)
        constructors=[m.run('Character.ctor',a) for a in m.actors];assert all(not r['failure'] for r in constructors)
        m.phase='setup';setup=[]
        for name,this,arg,choices in [('GameData.SetupCurrentAscension',m.game,0,[]),('AscensionsData.ClearCurrentPickedScript',m.temporary,0,[]),
            ('AscensionsData.SetupCharactersCount',m.temporary,0,[0]),('AscensionsData.SetupStartingCharacters',m.temporary,0,[]),('Gameplay.GetCurrentScript',m.gameplay,0,[])]:
            row=m.run(name,this,arg,choices);assert not row['failure'];setup.append(row)
        m.q(m.gameplay_static+0x30,m.reg(m.x.UC_X86_REG_RAX))
        generation=m.run('Gameplay.GetRandomCharacters',m.gameplay,5,genrow['choices']);assert generation['final']['returned_order']==ORDER
        returned=m.reg(m.x.UC_X86_REG_RAX);e=RetainedEngine(engine_data,m);e.set_producer(1.0,7)
        m.invocation='installation';m.phase='installer';m.pause_mode=paused
        installation=m.run('Animation.OnEnable',m.animation);assert not installation['failure'] and installation['boundary'] is None
        m.installed_delegate=m.rq(m.owner+0x58);assert m.installed_delegate in m.delegates
        m.installed_event_fields=tuple(m.rq(m.event_static+o) for o in (0x18,0x30,0x38));m.installed=True
        m.invocation='manage';m.phase='manage';initial=m.snapshot()
        joined=m.run('Characters.ManageCharacters',m.owner,returned,poolrow['choices'])
        assert not joined['failure'] and joined['boundary'] is None and len(e.nodes)==8
        assert m.scan==[[a,b] for a in START_ORDER for b in ORDER]
        before_consumption=m.snapshot();m.protected_graph={k:before_consumption[k] for k in GRAPH_KEYS}
        m.protected_actors={a:bytes(m.uc.mem_read(a,0x1B8)) for a in m.actors}
        m.protected_setup_storage=[(p,bytes(m.uc.mem_read(p,n))) for p,n in [(m.owner,0x60),(m.board,0xC20),(m.publication_list,0xC20)]+[(p,0xC20) for p in m.pools]]
        m.protected_iterators={p:bytes(m.uc.mem_read(p,0x28)) for p in m.iterators}
        m.protected_iterators.update({p:bytes(m.uc.mem_read(p,0x40 if kind=='animation' else 0x20)) for p,kind in m.iterator_kinds.items()})
        m.protected_iterators[m.shuffle_iterator]=bytes(m.uc.mem_read(m.shuffle_iterator,0x20))
        m.protected_regions={}
        for a in m.actors:
            s=m.scene[a];m.protected_regions[a]=[]
            for p,n,k in ((m.rq(a+0x148),0xC20,'history'),(m.rq(a+0x150),0xC20,'hover'),(s['status'],0x28,'status'),
                (s['active'],0xC20,'active_statuses'),(s['resistance'],0xC20,'resistance'),(m.rq(a+0x168),0x48,'runtime_role')):
                m.protected_regions[a].append((p,bytes(m.uc.mem_read(p,n)),k))
        e.pending_storage={p:bytes(e.uc.mem_read(p,0x88)) for p in e.payload_registry}
        e.owner_storage={r['native_owner']:bytes(e.uc.mem_read(r['native_owner'],0x100)) for r in e.owner_bindings.values()};e.released_storage={}
        for event in e.join_trace:
            if event['kind']=='native_wait_inserted':e.insertions[event['id']]={'producer':event['producer'],'record_generation':0}
        m.consuming=True;e.consumer_mode=True;m.phase='consumer';drains=[]
        first=e.queue_state()['entries'][0]['deadline']; assert e.queue_state()['entries'][0]['kind']=='animation'
        schedule=[('future',math.nextafter(first,-math.inf),8,2),('phase',first,8,1),('frame',first,7,2)]
        def consume(label,time,frame,phase=2):
            m.invocation=label;m.stop_service=abort[1] if abort and abort[0]==label else None
            row=e.drain(time,frame,phase);drains.append(row)
            if row['failure']:return False
            assert row['after']['generation']==(row['input']['generation_before']+1)&0xFFFFFFFF
            m.preserve();e.preserve_previous();return True
        for label,time,frame,phase in schedule:
            if not consume(label,time,frame,phase):return m,e,{'drains':drains}
            assert not any(r['kind']=='native_wait_callback' for r in drains[-1]['events']) and len(e.nodes)==8
        for i in range(5):
            current=e.queue_state();deadline=current['entries'][0]['deadline'];assert current['entries'][0]['kind']=='animation'
            if not consume('animation:'+str(i+1),deadline,8+i):return m,e,{'drains':drains}
            callbacks=[r for r in drains[-1]['events'] if r['kind']=='native_wait_callback'];assert len(callbacks)==1
            inserted=[r for r in drains[-1]['events'] if r['kind']=='native_wait_inserted']
            assert len(inserted)==int(i<4)
            if inserted:
                fresh=inserted[0]['id'];record=next(r for r in drains[-1]['after']['entries'] if r['id']==fresh)
                assert record['generation']==drains[-1]['after']['generation'] and record['frame_threshold']==9+i
                assert not any(r['kind']=='native_wait_visit' and r['id']==fresh for r in drains[-1]['events'])
            assert len(m.resume_calls)==i+1 and m.resume_calls[-1]['kind']=='animation' and m.resume_calls[-1]['result']==int(i<4)
            assert not m.acquired and [m.actor_snapshot(a)['status_storage']['active_values'] for a in m.actors]==[[],[25],[],[],[]]
        assert len(e.released)==1 and len(e.nodes)==7
        m.choices=[1,0];m.draws=[];acquisition_deadline=e.queue_state()['entries'][0]['deadline']
        assert e.queue_state()['entries'][0]['kind']=='acquisition'
        if not consume('acquisition',acquisition_deadline,13):return m,e,{'drains':drains}
        assert m.acquired==m.actors and len(e.released)==6 and len(e.nodes)==2
        assert [r['kind'] for r in e.queue_state()['entries']]==['audio','shuffle']
        assert [r['result'] for r in m.resume_calls]==[1,1,1,1,0]+[0]*5
        assert [(d['minimum'],d['maximum_exclusive'],d['index']) for d in m.draws]==[(1,11,1),(0,4,0)] and not m.choices
        assert len(m.clone_calls)==1 and m.clone_calls[0]['source']==m.rq(m.assets[21614]+0x140)
        for i,a in enumerate(m.actors):
            f=m.actor_snapshot(a);status=f['status_storage']
            assert status['active_values']==([25] if i<2 else []) and status['version']==(2 if i<2 else 1)
            assert f['bluff']==(m.assets[21614] if i==0 else 0)
        assert not m.resume_iterator and not m.resume_actor and m.reg(m.x.UC_X86_REG_RSP)==m.stack+0x10010
        records=e.retained_native_records()
        for r in records:
            if r['payload'] in e.released:assert r['reference_count']==r['gc_handle']==0 and not r['owner_linked']
            else:assert r['reference_count']==1 and r['gc_handle'] and r['owner_linked']
        return m,e,{'constructors':constructors,'setup':setup,'generation':generation,'installation':installation,'initial':initial,
            'joined':joined,'before_consumption':before_consumption,'before_start':m.before_start,'before_on_setup':m.before_on_setup,
            'initializers':m.init_calls,'first_yields':m.first_yields,'act_init_returns':m.actions,'ordered_comparisons':m.scan,
            'drains':drains,'resume_calls':copy.deepcopy(m.resume_calls),'selector_draws':copy.deepcopy(m.draws),
            'pre_entry_hydration':m.pre_entry_hydration,'clone_calls':m.clone_calls,'acquisition_calls':m.acquisition_calls,'selector_calls':m.selector_calls,
            'presentation':m.presentation,'visual_effects':m.visual_effects,'coroutine_registrations':e.registrations,
            'installed_delegate':m.delegates[m.installed_delegate],'completion':{'managed_cpu':cpu(m),'engine_cpu':cpu(e),
                'managed':m.snapshot(),'queue':e.queue_state(),'released_payloads':sorted(e.released)}}

    m,e,normal=run()
    if probe:return {'probe':True,'counters':{'resumes':len(m.resume_calls),'queue_records':len(e.nodes)},'normal':normal}
    print('Verified unpaused retained composition',flush=True)
    pm,pe,paused=run(True);assert normal==paused,first_difference(normal,paused)
    print('Verified complete paused/reentered ledger parity',flush=True)
    selected={'animation_retained_tween','acquisition_clone','acquisition_enum_add','reveal_name_text',
              'reveal_component_color','reveal_uppercase','engine_payload_free'}
    representatives={}
    for row in normal['drains']:
        for i,event in enumerate(row['services'],1):
            if event['service'] in selected:representatives.setdefault(event['service'],(event['invocation'],i))
    assert set(representatives)==selected
    aborts=[]
    for name,(invocation,ordinal) in representatives.items():
        am,ae,out=run(abort=(invocation,ordinal));actual=out['drains'][-1]
        baseline=next(row for row in normal['drains'] if row['services'] and row['services'][0]['invocation']==invocation)
        assert actual['failure']=='service:'+name and actual['services']==baseline['services'][:ordinal]
        assert actual['managed']==baseline['services'][ordinal-1]['snapshot'],first_difference(actual['managed'],baseline['services'][ordinal-1]['snapshot'])
        aborts.append({'invocation':invocation,'service_ordinal':ordinal,'service':name,'final':actual['managed']})
        print('Verified representative restart prefix '+str(len(aborts)),flush=True)
    assert all(sha(p.read_bytes())==prior_hashes[n] for n,p in paths.items())
    assert all(sha(p.read_bytes())==source_hashes[p.relative_to(ROOT.parent).as_posix()] for p in source_paths)
    services=sum(len(r['services']) for r in normal['drains'])
    return {'schema_version':'first_village_retained_acquisition_v1','build_id':BUILD,
        'decision':'Actual retained queue consumer must finish earlier animation before five original Hidden acquisitions; Minion copies original Confessor source.',
        'domain':{'generation_row':0,'pool_row':0,'asset_order':ORDER,'producer_time':1.0,'producer_signed_frame':7,
            'animation_frames':[8,9,10,11,12],'acquisition_frame':13,'phase':2,'selector_choices':[1,0],
            'services':'Inherited lifecycle/null-audio/tween/current mirror; exact original asset sprite resolution, CLR membership/clone/renderer API providers supplied.'},
        'exclusions':['Day/click/player observation','rendered readiness','Audio/Shuffle resume','probability law for recorded draws','Unity lifecycle runtime proof',
                      'API/managed exceptions','all aborted continuation equivalence','forced generation wrap/equality branch'],
        'source_hashes':source_hashes,'prior_report_hashes':prior_hashes,'original_scene_order':original,'presentation_assets':visual,
        'bodies':m.body_evidence+m.new_bodies+m.post_bodies+m.bridge_bodies+[m.shuffle_body]+m.subscriber_bodies+m.acquisition_bodies+[e.release_body],
        'selected_instruction_assertions':[{'rva':hex(a),'mnemonic':v[0],'operands':v[1]} for a,v in m.resume_pins.items()],
        'selected_engine_assertions':[{'rva':hex(a),'mnemonic':v[0],'operands':v[1]} for a,v in e.release_pins.items()],
        'ordered_source_bindings':m.ordered_source_bindings,'class_metadata_aliases':m.class_metadata_aliases,
        'restart_abort_selection':{'attempted':sorted(selected),'admitted':sorted(representatives),
            'excluded':'Other service families have exact pre-effect pause/reentry coverage, not new restart-abort coverage.'},
        **normal,'paused_reentered_prefixes':pm.prefixes,'restart_aborted_prefixes':aborts,
        'counters':{'animation_resumes':sum(r['kind']=='animation' for r in m.resume_calls),'acquisition_resumes':len(m.acquired),
            'animation_reyields':sum(r['result']==1 for r in m.resume_calls),'drains':len(normal['drains']),'gate_drains':3,
            'queue_records_at_setup':8,'queue_records_at_exit':len(e.nodes),'native_wait_insertions':e.next_identity,
            'released_payloads':len(e.released),'coroutine_registrations':len(e.registrations),'physical_owners':len(e.owner_bindings),
            'selector_draws':len(m.draws),'acquisition_clones':len(m.clone_calls),'acquisition_services':services,
            'paused_reentered_prefixes':len(pm.prefixes),'reentry_tokens_consumed':pm.reentries,'restart_aborted_prefixes':len(aborts),
            'managed_native_addresses':len(m.visited),'engine_native_addresses':len(e.executed),'python_source_hashes':len(source_hashes)}}


def main():
    p=argparse.ArgumentParser();p.add_argument('--game-root',required=True);p.add_argument('--dumper-root',required=True)
    p.add_argument('--output',required=True);p.add_argument('--probe',action='store_true');a=p.parse_args()
    output=Path(a.output);assert output.parent.is_dir()
    report=audit(a.game_root,a.dumper_root,a.probe);assert report==json.loads(json.dumps(report))
    packed=pool_snapshots(report);assert expand_snapshots(json.loads(json.dumps(packed)))==report
    output.write_text(json.dumps(packed,indent=2,sort_keys=True,ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps({'output':output.name,**report['counters']},sort_keys=True),flush=True)


if __name__=='__main__':main()
