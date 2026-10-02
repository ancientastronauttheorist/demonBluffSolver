"""Actual ChangeSkin -> actual CheckIfSkinUnlocked in one physical diagnostic graph."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_data_skin_lookup import Machine as LookupMachine, METHODS, MI_NAMES, VOL, NONVOL, POISON, snap
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots

ENTRY,END,NEXT=0x3B42C0,0x3B43A8,0x3B43B0
OBJECT_SLOT='UnityEngine.Object_TypeInfo';CONTAINS_SLOT='Method$System.Collections.Generic.List<SkinData>.Contains()'
EXCLUDED={0x3B43A7:'after nonreturning ChangeSkin guard int3',0x3B44CC:'Check cleanup-only exact MI load',0x3B44D3:'Check cleanup-only saved enumerator load',
 0x3B44D8:'Check cleanup-only Dispose call',0x3B44DD:'Check cleanup-only exception reload',0x3B44E2:'Check cleanup-only exception test',0x3B44E5:'Check cleanup-only rethrow branch',
 0x3B44FE:'after nonreturning null-list guard nop',0x3B44FF:'second null captured-skin guard unreachable with preserved RDI',0x3B4504:'after unreachable second captured-skin guard nop',
 0x3B450A:'after nonreturning null-current guard nop',0x3B450B:'Check cleanup-only rethrow call',0x3B4510:'after nonreturning rethrow int3'}


class Machine(LookupMachine):
    def __init__(self,game_root,dumper_root):
        import capstone
        super().__init__(game_root,dumper_root)
        self.instructions={a:i for a,i in self.instructions.items() if METHODS['CheckIfSkinUnlocked']['entry']<=a<METHODS['CheckIfSkinUnlocked']['end']}
        self.bounds={'CheckIfSkinUnlocked':self.bounds['CheckIfSkinUnlocked']};self.targets={'CheckIfSkinUnlocked':self.targets['CheckIfSkinUnlocked']};self.flags.pop('LoadSkin')
        extraction=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        raw=(Path(dumper_root)/'script.json').read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==extraction['outputs']['script_json']['sha256'].upper();metadata=json.loads(raw.decode('utf-8-sig'))
        rows=[r for r in metadata['ScriptMethod'] if r['Address']==ENTRY];assert len(rows)==1
        assert (rows[0]['Name'],rows[0]['Signature'],rows[0]['TypeSignature'])==('CharacterData$$ChangeSkin','void CharacterData__ChangeSkin (CharacterData_o* __this, SkinData_o* skin, const MethodInfo* method);','viii');self.targets['ChangeSkin']=rows[0]
        assert min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>ENTRY)==NEXT
        chunks=[]
        for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root=e
            while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
            if root.struct.BeginAddress==ENTRY:chunks.append((e.struct.BeginAddress,e.struct.EndAddress));assert e.unwindinfo.Flags==0
        assert chunks==[(ENTRY,END)]
        section=self.pe.get_section_by_rva(ENTRY);assert section and NEXT<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(ENTRY,NEXT-ENTRY);assert len(raw)==NEXT-ENTRY and raw[END-ENTRY:]==b'\xcc'*8
        ins=list(self.cs.disasm(raw[:END-ENTRY],ENTRY));assert sum(i.size for i in ins)==END-ENTRY and len(ins)==61
        assert (ins[-1].address,ins[-1].size,ins[-1].mnemonic,ins[-1].op_str)==(0x3B43A7,1,'int3','');self.instructions.update({i.address:i for i in ins})
        refs=set()
        for i in ins:
            for op in i.operands:
                if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+op.mem.disp)
            if i.mnemonic=='cmp' and i.operands[0].type==capstone.x86.X86_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:self.flags['ChangeSkin']=i.address+i.size+i.operands[0].mem.disp
        roots={r['Address']:r['Name'] for cat in ['ScriptMetadata','ScriptMetadataMethod'] for r in metadata[cat] if r['Address'] in refs};assert set(roots.values())=={OBJECT_SLOT,CONTAINS_SLOT}
        self.slot_names.update(roots);self.slots.update({n:a for a,n in roots.items()})
        for site,name in [(0x3B42E0,CONTAINS_SLOT),(0x3B42EC,OBJECT_SLOT)]:
            i=self.instructions[site-7];assert i.mnemonic=='lea' and self.slot_names[i.address+i.size+i.operands[1].mem.disp]==name
        for gw,sites in [(0x2B7B40,[0x3B42E0,0x3B42EC]),(0x281D90,[0x3B4308,0x3B434D]),(0x1C82480,[0x3B4315,0x3B435A]),(0xB55950,[0x3B4334]),(NEXT,[0x3B4372]),(0x2B6FF0,[0x3B4388]),(0x3874B0,[0x3B4392]),(0x2B7D90,[0x3B43A2])]:assert [i.address for i in ins if i.mnemonic=='call' and i.op_str==hex(gw)]==sites
        self.bounds['ChangeSkin']=dict(start=hex(ENTRY),end_exclusive=hex(END),next_managed=hex(NEXT),padding_bytes=8,unwind_ranges=[[hex(a),hex(b)] for a,b in chunks],eh_flags=0,byte_length=END-ENTRY,instruction_count=len(ins),sha256=hashlib.sha256(raw[:END-ENTRY]).hexdigest())
        for a,n,s,t in [(0x1C82480,'UnityEngine.Object$$op_Inequality','bool UnityEngine_Object__op_Inequality (UnityEngine_Object_o* x, UnityEngine_Object_o* y, const MethodInfo* method);','iiii'),
          (0xB55950,'System.Collections.Generic.List<object>$$Contains','bool System_Collections_Generic_List_object___Contains (System_Collections_Generic_List_object__o* __this, Il2CppObject* item, const MethodInfo_B55950* method);','iiii'),
          (0x3874B0,'SavesGame$$UpdateCharacterPreference','void SavesGame__UpdateCharacterPreference (CharacterData_o* cd, const MethodInfo* method);','vii')]:
            rows=[r for r in metadata['ScriptMethod'] if r['Address']==a and r['Name']==n];assert len(rows)==1 and rows[0]['Signature']==s and rows[0]['TypeSignature']==t;self.supplied.extend(rows)
        self.checks={a:v for a,v in self.checks.items() if NEXT<=a<METHODS['CheckIfSkinUnlocked']['end']}
        self.checks.update({0x3B42D1:('mov','rbx, rdx'),0x3B42D4:('mov','rdi, rcx'),0x3B42FF:('cmp','dword ptr [rcx + 0xe0], 0'),0x3B430D:('xor','r8d, r8d'),0x3B4310:('xor','edx, edx'),0x3B4312:('mov','rcx, rbx'),0x3B431E:('mov','rcx, qword ptr [rdi + 0xc8]'),0x3B4331:('mov','rdx, rbx'),0x3B4344:('cmp','dword ptr [rcx + 0xe0], 0'),0x3B4352:('xor','r8d, r8d'),0x3B4355:('xor','edx, edx'),0x3B4357:('mov','rcx, rbx'),0x3B4368:('mov','rdx, qword ptr [rbx + 0x18]'),0x3B436C:('xor','r8d, r8d'),0x3B436F:('mov','rcx, rdi'),0x3B4377:('test','al, al'),0x3B4385:('mov','qword ptr [rcx], rbx'),0x3B438D:('xor','edx, edx'),0x3B438F:('mov','rcx, rdi')})
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())
        for i,n in enumerate(['object_class','other_object_class','contains_mi','other_contains_mi']):self.p[n]=self.arena+0x2C0000+i*0x1000;self.sizes[n]=0x100 if n.endswith('class') else 0x80
        self.check_entry_sp=self.entry_sp-0x30;self.frame=self.check_entry_sp-0x68
        self.p.update(enum_output=self.frame+0x28,enum_state=self.frame+0x40,scratch=self.frame+0x20);self.ids={v:n for n,v in self.p.items()}

    def snapshot(self):
        s=super().snapshot();s['runtime_class_flags']={n:self.rd(self.p[n]+0xE0) for n in ['object_class','other_object_class']};return s
    def prepare(self,options):
        super().prepare(options)
        self.q(self.base+self.slots[OBJECT_SLOT],self.p['object_class']);self.q(self.base+self.slots[CONTAINS_SLOT],self.p['contains_mi'])
        self.d(self.p['object_class']+0xE0,options.get('object_class_flag',0));self.d(self.p['other_object_class']+0xE0,options.get('other_object_class_flag',0))
        for n,a in self.flags.items():self.u.mem_write(self.base+a,bytes([options.get('outer_warm' if n=='ChangeSkin' else 'check_warm',options.get('warm_byte',0))]))
        self.state.update(unity_comparisons=[],contains=[],class_initializations=[],preference_updates=[])
    def observe_write(self,uc,access,address,size,value,data):
        if not self.tracking:return
        pc=self.reg(self.x.UC_X86_REG_RIP)-self.base
        if address in [self.base+a for a in self.flags.values()]:
            assert pc==(0x3B42F1 if self.method=='ChangeSkin' else METHODS['CheckIfSkinUnlocked']['flagstore']) and size==value==1;self.flags_written.add(self.method)
        elif self.owner and self.owner<=address<self.owner+0x200:
            assert self.method=='ChangeSkin' and pc==0x3B4385 and address-self.owner==0xC0 and size==8;self.allowed.setdefault(self.oid(self.owner),set()).update(range(0xC0,0xC8))
        else:super().observe_write(uc,access,address,size,value,data)
    def entry(self):
        self.state['entries'].append(dict(method=self.method,owner=self.oid(self.owner),argument=self.oid(self.reg(self.x.UC_X86_REG_RDX)),argument_kind='SkinData' if self.method=='ChangeSkin' else 'String',volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm(),caller_return_bits=self.rq(self.reg(self.x.UC_X86_REG_RSP))))
    def hook(self,uc,address,size,data):
        rva=address-self.base;self.executed.add(rva)
        if rva in self.instructions:
            if rva==ENTRY:self.method='ChangeSkin';self.entry()
            elif rva==NEXT:
                self.method='CheckIfSkinUnlocked';self.query=self.reg(self.x.UC_X86_REG_RDX);assert self.reg(self.x.UC_X86_REG_RSP)==self.check_entry_sp;self.entry()
            elif rva==0x3B4377:self.method='ChangeSkin'
            return
        if self.method=='CheckIfSkinUnlocked':return super().hook(uc,address,size,data)
        cx,dx,r8=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8']];caller=self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base
        if rva==0x2B7B40:
            name=self.slot_names[cx-self.base];value=self.rq(cx);args=[name,self.oid(value)]
            if self.event('metadata',args):self.state['metadata'].append(args);self.mutate('metadata');self.ret(value)
        elif rva==0x281D90:
            assert self.oid(cx) in ['object_class','other_object_class'];args=[self.oid(cx)]
            if self.event('class_initialize',args):self.state['class_initializations'].append(args);self.write(self.oid(cx),0xE0,4,1);self.mutate('class_initialize');self.ret(self.options.get('class_init_return_bits',0xC111123456789076))
        elif rva==0x1C82480:
            assert cx==self.skin and dx==r8==0;args=[self.oid(cx),None,0]
            if self.event('unity_inequality',args):
                i=len(self.state['unity_comparisons'])-self.inequality_start;value=self.sequence_value('inequality_raw',i,0xBEEF123456789000|int(self.skin!=0))
                self.state['unity_comparisons'].append(dict(args=args,return_bits=value));self.mutate('unity_inequality');self.ret(value)
        elif rva==0xB55950:
            assert self.oid(cx) in ['list0','list1'] and dx==self.skin and self.oid(r8) in ['contains_mi','other_contains_mi'];args=[self.oid(cx),self.oid(dx),self.oid(r8)]
            if self.event('contains',args):
                count=self.rd(cx+0x18);assert count<=4;array=self.rq(cx+0x10);has=any(self.rq(array+0x20+i*8)==dx for i in range(count));value=self.options.get('contains_raw',0xCAFE123456789000|int(has))
                self.state['contains'].append(dict(args=args,return_bits=value));self.mutate('contains');self.ret(value)
        elif rva==0x2B6FF0:
            assert cx==self.owner+0xC0 and dx==self.skin and self.rq(cx)==dx;args=[self.oid(self.owner),0xC0,self.oid(dx)]
            if self.event('reference_barrier',args):self.state['barriers'].append(args);self.mutate('reference_barrier');self.ret(self.options.get('barrier_return_bits',0xBADD123456789056))
        elif rva==0x3874B0:
            assert cx==self.owner and dx==0;args=[self.oid(cx),0]
            if self.event('update_character_preference',args):self.state['preference_updates'].append(args);self.mutate('update_character_preference');self.ret(self.options.get('save_return_bits',0x5A7E123456789098))
        elif rva==0x2B7D90:
            assert caller==0x3B43A7
            if self.event('native_guard',['0x3b43a2']):self.error='native_guard';self.u.emu_stop()
        else:raise AssertionError(f'unclaimed ChangeSkin native {rva:x}')
    def run(self,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options=deepcopy(options or {});self.counts={};self.error=None
        self.method='ChangeSkin';self.mode='composition';self.owner=0 if self.options.get('null_owner') else self.p[self.options.get('owner','owner')];self.skin=0 if self.options.get('null_skin') else self.p[self.options.get('skin','skin0')];self.query=self.skin
        self.allowed,self.slot_writes,self.flags_written,self.completed={},{},set(),{};self.flag_written=False;self.comparison_start=len(self.state['comparisons']);self.inequality_start=len(self.state['unity_comparisons'])
        initial=self.snapshot();old=len(self.events);x=self.x;self.q(self.entry_sp,self.stop)
        incoming={n:self.options.get('entry_'+n.lower()+'_bits',0xDEAD123400000000+i) for i,n in enumerate(VOL)};incoming.update(RCX=self.owner,RDX=self.skin)
        initial_xmm=[f'{((1<<126)|i):032x}' for i in range(6)]
        for n,v in incoming.items():self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        for i,v in enumerate(initial_xmm):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),int(v,16))
        for i,n in enumerate(NONVOL):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),(1<<125)|i)
        self.u.reg_write(x.UC_X86_REG_RSP,self.entry_sp);self.tracking=True;fault=None
        try:self.u.emu_start(self.base+ENTRY,self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base;assert self.owner==0
            assert (pc,exc.errno) in [(0x3B431E,self.unicorn.UC_ERR_READ_UNMAPPED),(0x3B4385,self.unicorn.UC_ERR_WRITE_UNMAPPED),(0x3B4405,self.unicorn.UC_ERR_READ_UNMAPPED)]
            self.error='native_owner_fault';fault=hex(pc)
        finally:self.tracking=False
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==self.entry_sp+8
            assert all(self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB0000000000000+i for i,n in enumerate(NONVOL))
            assert all(self.reg(getattr(x,'UC_X86_REG_XMM'+str(i)))==(1<<125)|i for i in range(6,16))
        final=self.snapshot()
        for n,before in initial['memory'].items():
            a,b=bytes.fromhex(before),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or value==b[i] for i,value in enumerate(a)),n
        assert final['metadata_slots']=={**initial['metadata_slots'],**self.slot_writes}
        assert final['metadata_flags']=={**initial['metadata_flags'],**{n:1 for n in self.flags_written}}
        row=dict(options=self.options,initial=initial,events=deepcopy(self.events[old:]),final=final,returned=returned,error=self.error,fault_rva=fault,final_phase=self.method,
          entry_volatile_registers=incoming,entry_volatile_xmm_hex=initial_xmm,final_volatile_registers=self.volatile(),final_volatile_xmm_hex=self.xmm(),return_bits=self.reg(x.UC_X86_REG_RAX) if returned else None,
          normal_abi_verified=returned,completed_memory_write_offsets={n:sorted(v) for n,v in self.allowed.items()},completed_slot_writes=self.slot_writes.copy(),reached_native_flag_writes=sorted(self.flags_written),unrelated_storage_retained=True)
        verify(row,self.p,self.ids,self.slots,self.base,self.stop);row['independent_ordered_full_state_verified']=True;return row


def verify(row,p,ids,slot_addresses,base,stop):
    raw={n:bytearray.fromhex(v) for n,v in row['initial']['memory'].items()};slots={n:p[v] for n,v in row['initial']['metadata_slots'].items()};flags=row['initial']['metadata_flags'].copy();state=deepcopy(row['initial']['supplied_state']);options=row['options'];regs=row['entry_volatile_registers'].copy();xmm=row['entry_volatile_xmm_hex'].copy()
    owner=ids[regs['RCX']] if regs['RCX'] else None;skin=regs['RDX'];phase='ChangeSkin';events=[];counts={};completed={};returned=False;error=None;fault=None;result=None;inequality_start=len(state['unity_comparisons']);comparison_start=len(state['comparisons'])
    def oid(v):return ids[v] if v else None
    def q(n,off):return int.from_bytes(raw[n][off:off+8],'little')
    def rd(n,off):return int.from_bytes(raw[n][off:off+4],'little')
    def write(n,off,size,value):raw[n][off:off+size]=(p[value] if isinstance(value,str) else value).to_bytes(size,'little')
    def snapshot():
        s=snap(raw,slots,flags,state,ids);s['runtime_class_flags']={n:rd(n,0xE0) for n in ['object_class','other_object_class']};return s
    def entry(caller):state['entries'].append(dict(method=phase,owner=owner,argument=oid(regs['RDX']),argument_kind='SkinData' if phase=='ChangeSkin' else 'String',volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),caller_return_bits=caller))
    entry(stop)
    def mutate(kind):
        count=completed.get(kind,0)+1;completed[kind]=count
        for target,off,size,value in options.get('mutations',{}).get(kind+':'+str(count),[]):
            if target.startswith('slot:'):slots[target[5:]]=p[value]
            else:write(target,off,size,value)
    def series(key,index,default):return options.get(key,[])[index] if index<len(options.get(key,[])) else default
    def equal(a,b):
        if not a or not b:return a==b
        na,nb=oid(a),oid(b);return raw[na][0x14:0x14+rd(na,0x10)*2]==raw[nb][0x14:0x14+rd(nb,0x10)*2]
    class Stopped(Exception):pass
    def emit(kind,args,site,category=None,effect=None,value=None):
        events.append(dict(kind=kind,args=args,snapshot=snapshot(),raw_args=[regs[n] for n in ['RCX','RDX','R8','R9']],volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),caller=hex(site+5),native_phase=phase,entry_mode='composition'))
        counts[kind]=counts.get(kind,0)+1
        if options.get('failure')==[kind,counts[kind]] or kind in ['native_guard','rethrow']:raise Stopped
        if category:state[category].append(args)
        if effect:effect()
        mutate(kind);regs.update({n:POISON for n in VOL[1:]});regs['RAX']=value;xmm[:]=[f'{((1<<127)|i):032x}' for i in range(6)]
    def metadata(names,sites,name):
        if flags[name]==0:
            for n,site in zip(names,sites):
                regs['RCX']=base+slot_addresses[n];value=slots[n];emit('metadata',[n,oid(value)],site,'metadata',value=value)
            flags[name]=1
    def class_init(site):
        regs['RCX']=slots[OBJECT_SLOT];captured=oid(regs['RCX'])
        if rd(captured,0xE0)==0:emit('class_initialize',[captured],site,'class_initializations',effect=lambda:write(captured,0xE0,4,1),value=options.get('class_init_return_bits',0xC111123456789076))
    def inequality(site):
        regs['R8']=regs['RDX']=0;regs['RCX']=skin;args=[oid(skin),None,0];i=len(state['unity_comparisons'])-inequality_start;value=series('inequality_raw',i,0xBEEF123456789000|int(skin!=0))
        emit('unity_inequality',args,site,effect=lambda:state['unity_comparisons'].append(dict(args=args,return_bits=value)),value=value)
        return bool(regs['RAX']&0xFF)
    def fault_at(site):
        nonlocal error,fault
        error='native_owner_fault';fault=hex(site);raise Stopped
    def check():
        nonlocal phase
        phase='CheckIfSkinUnlocked';entry(base+0x3B4377);d=METHODS[phase];query=regs['RDX']
        metadata(MI_NAMES,d['metadata'],phase)
        if owner is None:fault_at(0x3B4405)
        regs['RDX']=q(owner,0xC8)
        if regs['RDX']==0:emit('native_guard',['0x3b44f9'],0x3B44F9)
        regs['R8']=slots[MI_NAMES[3]];regs['RCX']=p['enum_output'];listname=oid(regs['RDX']);args=[oid(regs['RCX']),listname,oid(regs['R8'])]
        entries=[oid(q(oid(q(listname,0x10)),0x20+i*8)) for i in range(rd(listname,0x18))];produced=p[listname].to_bytes(8,'little')+bytes(4)+rd(listname,0x1C).to_bytes(4,'little')+bytes(8)
        def make_enum():state['enumerators'].append(dict(args=args,produced_24_bytes=produced.hex(),supplied_entries=entries));state['iterator']=dict(list=listname,entries=entries,cursor=0);raw['scratch'][8:32]=produced
        emit('get_enumerator',args,d['get'],effect=make_enum,value=options.get('get_return_bits',p['enum_output']))
        xmm[0]=f'{int.from_bytes(raw["scratch"][8:24],"little"):032x}';xmm[1]=f'{int.from_bytes(raw["scratch"][24:32],"little"):032x}'
        raw['scratch'][32:56]=raw['scratch'][8:32];write('scratch',8,8,0);write('scratch',0x10,8,p['enum_state'])
        matched=False
        while True:
            regs['RDX']=slots[MI_NAMES[1]];regs['RCX']=p['enum_state'];args=[oid(regs['RCX']),oid(regs['RDX'])];it=state['iterator'];i=it['cursor'];has=i<len(it['entries']);current=it['entries'][i] if has else None;value=series('move_next_raw',i,0xBEEF123456789000|int(has))
            def move():it['cursor']+=1;write('scratch',0x28,4,i+1);write('scratch',0x30,8,current if current else 0);state['moves'].append(dict(args=args,current=current,return_bits=value))
            emit('move_next',args,d['move'],effect=move,value=value)
            if regs['RAX']&0xFF==0:break
            captured=q('scratch',0x30)
            if captured==0:emit('native_guard',['0x3b4505'],0x3B4505)
            regs['R8']=0;regs['RDX']=q(oid(captured),0x18);regs['RCX']=query;args=[oid(regs['RCX']),oid(regs['RDX']),0];i=len(state['comparisons'])-comparison_start;value=series('equality_raw',i,0xCAFE123456789000|int(equal(regs['RCX'],regs['RDX'])))
            emit('string_equality',args,d['equal'],effect=lambda:state['comparisons'].append(dict(args=args,return_bits=value)),value=value)
            if regs['RAX']&0xFF==0:continue
            regs['RDX']=0;regs['RCX']=captured;args=[oid(captured),0];value=options.get('unlocked_raw',{}).get(oid(captured),0xBEEF123456789001)
            emit('skin_unlocked',args,d['unlock'],effect=lambda:state['unlocks'].append(dict(args=args,return_bits=value)),value=value);captured_byte=regs['RAX']&0xFF
            regs['RDX']=slots[MI_NAMES[0]];regs['RCX']=p['enum_state'];emit('dispose',[oid(regs['RCX']),oid(regs['RDX'])],d['dispose'][0],'disposals',value=options.get('dispose_return_bits',0xD15E1234567890AB));regs['RAX']=captured_byte;matched=True;break
        if not matched:
            regs['RDX']=slots[MI_NAMES[0]];regs['RCX']=p['enum_state'];emit('dispose',[oid(regs['RCX']),oid(regs['RDX'])],d['dispose'][1],'disposals',value=options.get('dispose_return_bits',0xD15E1234567890AB));regs['RAX']&=~0xFF
        phase='ChangeSkin';return bool(regs['RAX']&0xFF)
    try:
        metadata([CONTAINS_SLOT,OBJECT_SLOT],[0x3B42E0,0x3B42EC],'ChangeSkin');class_init(0x3B4308)
        admitted=True
        if inequality(0x3B4315):
            if owner is None:fault_at(0x3B431E)
            regs['RCX']=q(owner,0xC8)
            if regs['RCX']==0:emit('native_guard',['0x3b43a2'],0x3B43A2)
            regs['R8']=slots[CONTAINS_SLOT];regs['RDX']=skin;listname=oid(regs['RCX']);args=[listname,oid(skin),oid(regs['R8'])];array=oid(q(listname,0x10));has=any(q(array,0x20+i*8)==skin for i in range(rd(listname,0x18)));value=options.get('contains_raw',0xCAFE123456789000|int(has))
            emit('contains',args,0x3B4334,effect=lambda:state['contains'].append(dict(args=args,return_bits=value)),value=value);admitted=bool(regs['RAX']&0xFF)
        if admitted:
            class_init(0x3B434D)
            if inequality(0x3B435A):
                if skin==0:emit('native_guard',['0x3b43a2'],0x3B43A2)
                regs['RDX']=q(oid(skin),0x18);regs['R8']=0;regs['RCX']=p[owner] if owner else 0;admitted=check()
            if admitted:
                regs['RCX']=(p[owner] if owner else 0)+0xC0;regs['RDX']=skin
                if owner is None:fault_at(0x3B4385)
                write(owner,0xC0,8,skin);emit('reference_barrier',[owner,0xC0,oid(skin)],0x3B4388,'barriers',value=options.get('barrier_return_bits',0xBADD123456789056))
                regs['RDX']=0;regs['RCX']=p[owner];emit('update_character_preference',[owner,0],0x3B4392,'preference_updates',value=options.get('save_return_bits',0x5A7E123456789098))
        returned=True;result=regs['RAX']
    except Stopped:
        if error is None:error=events[-1]['kind']
    assert (row['returned'],row['error'],row['fault_rva'],row['return_bits'],row['final_phase'])==(returned,error,fault,result,phase),options
    assert row['events']==events,('events',options)
    assert row['final']==snapshot(),('final',options)
    assert row['final_volatile_registers']==regs and row['final_volatile_xmm_hex']==xmm,('volatile ABI',options)


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];sequences=[];baselines=[];stops=[]
    for warm,class_flag,owner,skin,entries in itertools.product([0,0x80],[0,1],['owner','other_owner'],['skin0','skin1','skin2'],[['skin0','skin1','skin2'],['skin2','skin0'],[],['skin0','skin0']]):cases.append(m.run(dict(warm_byte=warm,object_class_flag=class_flag,owner=owner,skin=skin,entries=entries)))
    profiles=[dict(null_skin=True),dict(null_skin=True,inequality_raw=[1,1],contains_raw=1),dict(null_skin=True,inequality_raw=[0,0]),dict(null_list=True),dict(null_list=True,inequality_raw=[0,0]),dict(null_owner=True),dict(null_owner=True,inequality_raw=[0,0]),dict(null_owner=True,inequality_raw=[0,1]),dict(contains_raw=0xFFFFFFFFFFFFFF00),dict(contains_raw=0x8000000000000080),dict(inequality_raw=[0xFFFFFFFFFFFFFF00,0xFFFFFFFFFFFFFF00]),dict(inequality_raw=[0x8000000000000080,0xFFFFFFFFFFFFFF00]),dict(inequality_raw=[0xFFFFFFFFFFFFFF00,0x8000000000000080]),dict(unlocked_raw={'skin0':0xFFFFFFFFFFFFFF00}),dict(unlocked_raw={'skin0':0x123456789ABCDE80}),dict(unlocked_raw={'skin0':0x123456789ABCDEFF}),dict(entries=[None],contains_raw=1),dict(entries=[],contains_raw=1,dispose_return_bits=0xFFFFFFFFFFFFFFFF),dict(entries=['skin2','skin0'],unlocked_raw={'skin2':0x123456789ABCDE80}),dict(unlocked_raw={'skin0':0xBEEF000000000001},save_return_bits=0),dict(save_return_bits=0xFFFFFFFFFFFFFFFF),dict(get_return_bits=0),dict(entry_rax_bits=0xFFFFFFFFFFFFFFFF,entry_r8_bits=0x1122334455667788,entry_r9_bits=0x8877665544332211)]
    profiles.extend([dict(object_class_flag=0x10000000),dict(object_class_flag=0xFFFFFFFF),dict(skin='skin2',entries=['skin0','skin2'],unlocked_raw={'skin0':0xBEEF123456789000,'skin2':0xBEEF123456789001}),dict(contains_raw=0xFFFFFFFFFFFFFF00,mutations={'class_initialize:2':[['other_owner',0xC0,8,'skin1']]}),dict(entries=['skin2','skin0'],mutations={'string_equality:1':[['skin2',0x18,8,'different_id'],['scratch',0x30,8,'skin1']]})])
    for options in profiles:cases.append(m.run(options))
    plans=[('metadata:1',[['owner',0xC8,8,'list1']]),('metadata:2',[['slot:'+OBJECT_SLOT,0,8,'other_object_class']]),('class_initialize:1',[['slot:'+OBJECT_SLOT,0,8,'other_object_class'],['owner',0xC8,8,'list1']]),
      ('unity_inequality:1',[['slot:'+CONTAINS_SLOT,0,8,'other_contains_mi'],['owner',0xC8,8,'list1']]),('unity_inequality:2',[['skin0',0x18,8,'different_id']]),('unity_inequality:2',[['skin0',0x18,8,0],['skin1',0x18,8,0]]),
      ('contains:1',[['owner',0xC8,8,'list1']]),('contains:1',[['owner',0xC8,8,0]]),('contains:1',[['skin0',0x18,8,'different_id']]),('contains:1',[['slot:'+MI_NAMES[3],0,8,'other_mi3']]),
      ('get_enumerator:1',[['owner',0xC8,8,'list1'],['slot:'+MI_NAMES[1],0,8,'other_mi1']]),('move_next:1',[['scratch',0x30,8,'skin2']]),('string_equality:1',[['scratch',0x30,8,'skin1'],['skin0',0x18,8,'different_id']]),
      ('skin_unlocked:1',[['slot:'+MI_NAMES[0],0,8,'other_mi0'],['owner',0xC0,8,'skin1']]),('dispose:1',[['owner',0xC0,8,'skin1']]),('reference_barrier:1',[['owner',0xC0,8,'skin1']]),('update_character_preference:1',[['owner',0xC0,8,'skin2'],['other_owner',0xC8,8,'list1']])]
    for (phase,writes),warm,class_flag,owner in itertools.product(plans,[0,0xFE],[0,1],['owner','other_owner']):cases.append(m.run(dict(warm_byte=warm,object_class_flag=class_flag,owner=owner,mutations={phase:writes})))
    for options in [{},dict(owner='other_owner'),dict(entries=['skin2','skin0']),dict(mutations={'reference_barrier:1':[['owner',0xC0,8,'skin1']]})]:
        rows=[m.run(options),m.run(dict(options,failure=['string_equality',1]),True),m.run(options,True),m.run(dict(options,mutations={'update_character_preference:1':[['owner',0xC0,8,'skin2']]}),True)]
        assert [r['returned'] for r in rows]==[True,False,True,True] and all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
    profiles=[{},dict(warm_byte=0xFE,object_class_flag=1),dict(contains_raw=0),dict(inequality_raw=[0,0]),dict(null_skin=True,inequality_raw=[1,1],contains_raw=1),dict(null_list=True),dict(null_owner=True),dict(entries=[None],contains_raw=1),dict(entries=[],contains_raw=1),dict(entries=['skin2','skin0']),dict(mutations={'class_initialize:1':[['slot:'+OBJECT_SLOT,0,8,'other_object_class']]}),dict(mutations={'contains:1':[['owner',0xC8,8,0]]}),dict(mutations={'unity_inequality:2':[['skin0',0x18,8,0],['skin1',0x18,8,0]]}),dict(mutations={'reference_barrier:1':[['owner',0xC0,8,'skin1']]})]
    for options in profiles:
        baseline=m.run(options);bid=len(baselines);baselines.append(baseline);counts={}
        for i,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;stopped=m.run(dict(options,failure=[kind,counts[kind]]));assert not stopped['returned'] and stopped['events']==baseline['events'][:i+1] and stopped['final']==e['snapshot'];stops.append(dict(baseline=bid,prefix_length=i+1,result=stopped))
    missing=set(m.instructions)-m.executed;assert missing==set(EXCLUDED),(sorted(missing),sorted(set(EXCLUDED)-missing))
    return dict(schema='character_data_change_skin_join_native_v1',build=BUILD,methods={n:dict(method_id='tdi5845.m0014' if n=='ChangeSkin' else 'tdi5845.m0013',symbol_key='CharacterData::public void ChangeSkin(SkinData skin)' if n=='ChangeSkin' else 'CharacterData::public bool CheckIfSkinUnlocked(string skinId)',target=t) for n,t in m.targets.items()},body_bounds=m.bounds,
      scope='Actual ChangeSkin -> actual ordinary CheckIfSkinUnlocked in one physical graph. Unity inequality/class init, Contains, metadata/iterator/string/SkinData unlock/barrier/UpdateCharacterPreference/guards supplied. Complete Check cleanup extent pinned but no direct cleanup/unwind execution in this join.',
      supplied_targets=m.supplied,fields=m.fields,metadata_slots={n:hex(a) for n,a in m.slots.items()},metadata_flags={n:hex(a) for n,a in m.flags.items()},diagnostic_windows_not_object_extents=m.sizes,
      stack_layout=dict(outer_entry='fixture entry SP',outer_frame='entry SP-28',check_entry='entry SP-30',check_frame='entry SP-98',check_hidden_output='check frame+28',check_active_state='check frame+40',enumerator_bytes=24),
      instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),excluded_instruction_reasons={hex(a):r for a,r in EXCLUDED.items()},
      cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,summary=dict(cases=len(cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--game-root',required=True);parser.add_argument('--dumper-root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    report=pool_snapshots(pool_memory(audit(args.game_root,args.dumper_root)));Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps(report['summary'],sort_keys=True))
