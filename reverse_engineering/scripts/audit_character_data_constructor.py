"""Exact CharacterData constructor with supplied allocation/list/base services."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots

ENTRY, END, FOLLOWING = 0x3B50A0, 0x3B5274, 0x3B5280
POISON = 0xFACE123456789090
INTEGER_VOLATILE = ['RAX','RCX','RDX','R8','R9','R10','R11']
KINDS = ['CharacterData','SkinData','AchievementData','ECharacterStatus','ECharacterTag','CharacterData']
FIELDS = [('bundledCharacters',0x48),('skins',0xC8),('achievements',0xD0),('additionalStatuses',0x118),('tags',0x120),('canAppearIf',0x128)]
ALLOCATION_SITES = [0x3B513C,0x3B5169,0x3B5199,0x3B51C9,0x3B51F9,0x3B5229]
CTOR_SITES = [0x3B514E,0x3B517B,0x3B51AB,0x3B51DB,0x3B520B,0x3B523B]
STORE_SITES = [0x3B515A,0x3B518A,0x3B51BA,0x3B51EA,0x3B521A,0x3B524A]
BARRIER_SITES = [0x3B515D,0x3B518D,0x3B51BD,0x3B51ED,0x3B521D,0x3B524D]
METADATA_ORDER = [('ECharacterTag','method'),('SkinData','method'),('ECharacterStatus','method'),('CharacterData','method'),
                  ('AchievementData','method'),('ECharacterStatus','class'),('SkinData','class'),('ECharacterTag','class'),
                  ('CharacterData','class'),('AchievementData','class')]
METADATA_SITES = [0x3B50BD,0x3B50C9,0x3B50D5,0x3B50E1,0x3B50ED,0x3B50F9,0x3B5105,0x3B5111,0x3B511D,0x3B5129]


def slot_name(kind,category):
    return 'System.Collections.Generic.List<'+kind+'>_TypeInfo' if category=='class' else 'Method$System.Collections.Generic.List<'+kind+'>..ctor()'


def snapshot(memory,slots,flag,state,ids):
    def q(n,o):return int.from_bytes(memory[n][o:o+8],'little')
    def oid(v):return ids[v] if v else None
    return dict(memory={n:bytes(v).hex() for n,v in memory.items()},metadata_slots={n:oid(v) for n,v in slots.items()},metadata_flag=flag,
                owners={n:dict(fields={field:oid(q(n,off)) for field,off in FIELDS},bluffable=memory[n][0x13C],
                               usually_disguised=memory[n][0x13D],picking=memory[n][0x13E],current_skin=oid(q(n,0xC0))) for n in ['owner','other_owner']},
                supplied_state=deepcopy(state))


class Machine(NativeMachine):
    def __init__(self,game_root,dumper_root):
        import capstone
        super().__init__(game_root)
        extraction=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name,key):
            raw=(Path(dumper_root)/name).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==extraction['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata=json.loads(pin('script.json','script_json'));dump=pin('dump.cs','dump_cs')
        rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==ENTRY];assert len(rows)==1
        assert rows[0]['Name']=='CharacterData$$.ctor' and rows[0]['Signature']=='void CharacterData___ctor (CharacterData_o* __this, const MethodInfo* method);' and rows[0]['TypeSignature']=='vii'
        self.target=rows[0];assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address']>ENTRY)==FOLLOWING
        chunks=[]
        for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root=e
            while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
            if root.struct.BeginAddress==ENTRY:chunks.append((e.struct.BeginAddress,e.struct.EndAddress))
        assert chunks==[(ENTRY,END)]
        section=self.pe.get_section_by_rva(ENTRY);assert section and FOLLOWING<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(ENTRY,FOLLOWING-ENTRY);assert len(raw)==FOLLOWING-ENTRY and raw[END-ENTRY:]==b'\xcc'*12
        ins=list(self.cs.disasm(raw[:END-ENTRY],ENTRY));assert sum(i.size for i in ins)==END-ENTRY
        assert (ins[-1].address,ins[-1].size,ins[-1].mnemonic,ins[-1].op_str)==(0x3B526F,5,'jmp','0x1c8a5c0')
        self.instructions={i.address:i for i in ins};refs=set()
        for i in ins:
            for op in i.operands:
                if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+op.mem.disp)
            if i.mnemonic=='cmp' and i.operands[0].type==capstone.x86.X86_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:
                self.flag=i.address+i.size+i.operands[0].mem.disp
        self.slot_names={r['Address']:r['Name'] for category in ['ScriptMetadata','ScriptMetadataMethod'] for r in self.metadata[category] if r['Address'] in refs}
        self.slots={n:a for a,n in self.slot_names.items()};assert set(self.slots)=={slot_name(k,c) for k in set(KINDS) for c in ['class','method']}
        for (kind,category),site in zip(METADATA_ORDER,METADATA_SITES):
            lea=self.instructions[site-7];assert lea.mnemonic=='lea' and lea.operands[1].mem.base==capstone.x86.X86_REG_RIP
            assert self.slot_names[lea.address+lea.size+lea.operands[1].mem.disp]==slot_name(kind,category)
        for gateway,sites in [(0x2B7B40,METADATA_SITES),(0x2B7D40,ALLOCATION_SITES),(0xB02160,CTOR_SITES),(0x2B6FF0,BARRIER_SITES)]:
            assert [i.address for i in ins if i.mnemonic=='call' and i.op_str==hex(gateway)]==sites
        declaration=re.search(r'^public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845\s*\{(.*?)\n\}',dump,re.M|re.S);assert declaration
        self.field_declarations=['public List<CharacterData> bundledCharacters; // 0x48','public List<SkinData> skins; // 0xC8',
             'public List<AchievementData> achievements; // 0xD0','public List<ECharacterStatus> additionalStatuses; // 0x118',
             'public List<ECharacterTag> tags; // 0x120','public List<CharacterData> canAppearIf; // 0x128',
             'public bool bluffable; // 0x13C','public bool usuallyDisguised; // 0x13D','public bool picking; // 0x13E','public SkinData currentSkin; // 0xC0']
        assert all(f in declaration[1] for f in self.field_declarations)
        self.supplied=[]
        for address,name,signature in [(0xB02160,'System.Collections.Generic.List<object>$$.ctor','void System_Collections_Generic_List_object____ctor (System_Collections_Generic_List_object__o* __this, const MethodInfo_B02160* method);'),
                                      (0x1C8A5C0,'UnityEngine.ScriptableObject$$.ctor','void UnityEngine_ScriptableObject___ctor (UnityEngine_ScriptableObject_o* __this, const MethodInfo* method);')]:
            rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==address and r['Name']==name];assert len(rows)==1 and rows[0]['Signature']==signature and rows[0]['TypeSignature']=='vii'
            self.supplied+=rows
        self.checks={0x3B50B1:('mov','rdi, rcx'),0x3B5135:('mov','rcx, qword ptr [rip + 0x234b544]'),
                     0x3B514B:('mov','rbx, rax'),0x3B515A:('mov','qword ptr [rcx], rbx'),
                     0x3B518A:('mov','qword ptr [rcx], rbx'),0x3B51BA:('mov','qword ptr [rcx], rbx'),
                     0x3B51EA:('mov','qword ptr [rcx], rbx'),0x3B521A:('mov','qword ptr [rcx], rbx'),
                     0x3B524A:('mov','qword ptr [rcx], rbx'),0x3B5252:('xor','edx, edx'),
                     0x3B5254:('mov','byte ptr [rdi + 0x13c], 1'),0x3B525B:('mov','rcx, rdi'),
                     0x3B525E:('mov','byte ptr [rdi + 0x13e], 1'),0x3B5265:('mov','rbx, qword ptr [rsp + 0x30]')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())
        names=['owner','other_owner','data_class','old_skin','new_skin']+[f'list{i}' for i in range(6)]+[f'prior{i}' for i in range(6)]
        for kind in sorted(set(KINDS)):
            names += [kind+':'+category for category in ['class','method','other_class','other_method']]
        self.p={n:self.arena+0x280000+i*0x1000 for i,n in enumerate(names)};self.ids={v:n for n,v in self.p.items()}
        self.sizes={n:0x200 if n in ['owner','other_owner'] else 0x100 if ':' in n or n=='data_class' else 0x80 for n in names}
        self.list_kinds={f'{prefix}{i}':kind for prefix in ['list','prior'] for i,kind in enumerate(KINDS)}
        self.tracking=False;self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE,self.observe_write)

    def oid(self,v):return self.ids[v] if v else None
    def snapshot(self):
        return snapshot({n:bytes(self.u.mem_read(p,self.sizes[n])) for n,p in self.p.items()},
                        {n:self.rq(self.base+a) for n,a in self.slots.items()},self.u.mem_read(self.base+self.flag,1)[0],self.state,self.ids)

    def observe_write(self,uc,access,address,size,value,user_data):
        if not self.tracking:return
        pc=self.reg(self.x.UC_X86_REG_RIP)-self.base
        if address==self.base+self.flag:
            assert size==value==1 and pc==0x3B512E;self.flag_written=True
        elif self.owner and self.owner<=address<self.owner+0x200:
            off=address-self.owner
            assert (pc in STORE_SITES and off==FIELDS[STORE_SITES.index(pc)][1] and size==8) or (pc,off,size,value) in [(0x3B5254,0x13C,1,1),(0x3B525E,0x13E,1,1)]
            self.allowed.setdefault(self.oid(self.owner),set()).update(range(off,off+size))

    def effect(self,n,offset,size,value):
        if isinstance(value,str):value=self.p[value]
        self.u.mem_write(self.p[n]+offset,value.to_bytes(size,'little'));self.allowed.setdefault(n,set()).update(range(offset,offset+size))

    def mutate(self,kind):
        ordinal=self.completed_counts.get(kind,0)+1;self.completed_counts[kind]=ordinal
        for target,offset,size,value in self.options.get('mutations',{}).get(kind+':'+str(ordinal),[]):
            if target.startswith('slot:'):
                name=target[5:];self.q(self.base+self.slots[name],self.p[value]);self.slot_writes[name]=value
            else:self.effect(target,offset,size,value)

    def prepare(self,options):
        self.options=deepcopy(options);self.events=[];self.counts={};self.error=None
        self.state={n:[] for n in ['entries','metadata','allocations','list_constructors','barriers','base_constructors']}
        for n,p in self.p.items():self.u.mem_write(p,bytes([options.get('seed_byte',0xA5)])*self.sizes[n])
        for kind in set(KINDS):
            for category in ['class','method']:self.q(self.base+self.slots[slot_name(kind,category)],self.p[kind+':'+category])
        for n in ['owner','other_owner']:
            self.q(self.p[n],self.p['data_class']);self.q(self.p[n]+0xC0,self.p['old_skin'])
            for i,(_,off) in enumerate(FIELDS):self.q(self.p[n]+off,self.p[f'prior{i}'] if options.get('prefilled_owner',True) else 0)
        for name,kind in self.list_kinds.items():self.q(self.p[name],self.p[kind+':class'])
        self.u.mem_write(self.base+self.flag,bytes([options.get('warm_byte',0)]))

    def volatile(self):return {n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in INTEGER_VOLATILE}
    def xmm(self):return [f'{self.reg(getattr(self.x,"UC_X86_REG_XMM"+str(i))):032x}' for i in range(6)]
    def event(self,kind,args):
        okay=super().event(kind,args);ret=self.rq(self.reg(self.x.UC_X86_REG_RSP))
        self.events[-1].update(raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],
                              volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm(),caller=hex(ret-self.base),
                              caller_kind='fixture_return_sentinel' if ret==self.stop else 'native_return',native_phase='CharacterData.ctor')
        return okay

    def ret(self,value=0):
        for n in INTEGER_VOLATILE[1:]:self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),POISON)
        for i in range(6):self.u.reg_write(getattr(self.x,'UC_X86_REG_XMM'+str(i)),(1<<127)|i)
        super().ret(value)

    def allocation_record(self,index):return self.options.get('allocation_records',[f'list{i}' for i in range(6)])[index]

    def hook(self,uc,address,size,user_data):
        rva=address-self.base;self.executed.add(rva);x=self.x
        if rva==ENTRY:self.state['entries'].append(dict(owner=self.oid(self.owner),volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm()))
        if rva in self.instructions:return
        cx,dx=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX']]
        caller=self.rq(self.reg(x.UC_X86_REG_RSP))-self.base
        if rva==0x2B7B40:
            assert cx-self.base in self.slot_names;name=self.slot_names[cx-self.base];value=self.rq(cx);args=[name,self.oid(value)]
            if self.event('metadata',args):self.state['metadata'].append(args);self.mutate('metadata');self.ret(value)
        elif rva==0x2B7D40:
            index=ALLOCATION_SITES.index(caller-5);kind=KINDS[index];name=self.oid(cx)
            assert name in [kind+':class',kind+':other_class'];record=self.allocation_record(index);assert self.list_kinds[record]==kind
            args=[index,name,record]
            if self.event('allocation',args):self.state['allocations'].append(args);self.mutate('allocation');self.ret(self.p[record])
        elif rva==0xB02160:
            index=CTOR_SITES.index(caller-5);kind=KINDS[index];record=self.allocation_record(index)
            assert cx==self.p[record] and self.oid(dx) in [kind+':method',kind+':other_method']
            args=[index,record,self.oid(dx)]
            if self.event('list_constructor',args):self.state['list_constructors'].append(args);self.mutate('list_constructor');self.ret(self.options.get('list_ctor_return_bits',0xF00D123456789012))
        elif rva==0x2B6FF0:
            index=BARRIER_SITES.index(caller-5);field,off=FIELDS[index];record=self.allocation_record(index)
            assert cx==self.owner+off and dx==self.p[record] and self.rq(cx)==dx
            args=[index,self.oid(self.owner),field,off,record]
            if self.event('reference_barrier',args):self.state['barriers'].append(args);self.mutate('reference_barrier');self.ret(self.options.get('barrier_return_bits',0xBADD123456789034))
        elif rva==0x1C8A5C0:
            assert cx==self.owner and dx==0 and self.reg(x.UC_X86_REG_RSP)==self.entry_sp
            args=[self.oid(cx),0]
            if self.event('base_constructor',args):self.state['base_constructors'].append(args);self.mutate('base_constructor');self.ret(self.options.get('base_return_bits',0xABCD123456789056))
        else:raise AssertionError(f'unclaimed native {rva:x}')

    def run(self,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options=deepcopy(options or {});self.error=None;self.counts={}
        self.allowed,self.slot_writes,self.flag_written,self.completed_counts={},{},False,{}
        self.owner=0 if self.options.get('null_owner') else self.p[self.options.get('owner','owner')]
        initial=self.snapshot();old=len(self.events);x=self.x;sp=self.stack+0x18008;self.entry_sp=sp;self.q(sp,self.stop)
        for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),(1<<125)|i)
        incoming={n:self.options.get('entry_'+n.lower()+'_bits',0xDEAD123400000000+i) for i,n in enumerate(INTEGER_VOLATILE)}
        incoming['RCX']=self.owner;initial_xmm=[f'{((1<<126)|i):032x}' for i in range(6)]
        for n,v in incoming.items():self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        for i,v in enumerate(initial_xmm):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),int(v,16))
        self.u.reg_write(x.UC_X86_REG_RSP,sp);self.tracking=True;fault=None
        try:self.u.emu_start(self.base+ENTRY,self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base
            assert self.owner==0 and exc.errno==self.unicorn.UC_ERR_WRITE_UNMAPPED and pc==STORE_SITES[0]
            self.error='native_owner_write_fault';fault=hex(pc)
        finally:self.tracking=False
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==sp+8
            assert all(self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB0000000000000+i for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']))
            assert all(self.reg(getattr(x,'UC_X86_REG_XMM'+str(i)))==(1<<125)|i for i in range(6,16))
        final=self.snapshot()
        for n,before in initial['memory'].items():
            a,b=bytes.fromhex(before),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or value==b[i] for i,value in enumerate(a)),n
        assert final['metadata_slots']=={**initial['metadata_slots'],**self.slot_writes}
        assert final['metadata_flag']==(1 if self.flag_written else initial['metadata_flag'])
        row=dict(options=self.options,entry_volatile_registers=incoming,entry_volatile_xmm_hex=initial_xmm,returned=returned,error=self.error,fault_rva=fault,
                 initial=initial,events=deepcopy(self.events[old:]),final=final,normal_abi_verified=returned,
                 completed_memory_write_offsets={n:sorted(v) for n,v in self.allowed.items()},completed_slot_writes=self.slot_writes.copy(),reached_native_flag_write=self.flag_written,
                 return_bits=self.reg(x.UC_X86_REG_RAX) if returned else None,final_volatile_registers=self.volatile(),final_volatile_xmm_hex=self.xmm(),unrelated_storage_retained=True)
        verify(row,self.p,self.ids,self.slots,self.base,self.stop);row['independent_ordered_full_state_verified']=True
        return row


def verify(row,p,ids,slot_addresses,base,stop):
    raw={n:bytearray.fromhex(v) for n,v in row['initial']['memory'].items()};slots={n:p[v] for n,v in row['initial']['metadata_slots'].items()}
    flag=row['initial']['metadata_flag'];state=deepcopy(row['initial']['supplied_state']);options=row['options'];regs=row['entry_volatile_registers'].copy();xmm=row['entry_volatile_xmm_hex'].copy()
    owner=ids[regs['RCX']] if regs['RCX'] else None;state['entries'].append(dict(owner=owner,volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy()))
    events=[];counts={};completed={};returned=False;error=None;fault=None;result=None
    def oid(v):return ids[v] if v else None
    def write(n,off,size,value):raw[n][off:off+size]=(p[value] if isinstance(value,str) else value).to_bytes(size,'little')
    def mutate(kind):
        count=completed.get(kind,0)+1;completed[kind]=count
        for target,offset,size,value in options.get('mutations',{}).get(kind+':'+str(count),[]):
            if target.startswith('slot:'):slots[target[5:]]=p[value]
            else:write(target,offset,size,value)
    class Stopped(Exception):pass
    def emit(kind,args,caller,category,value):
        e=dict(kind=kind,args=args,snapshot=snapshot(raw,slots,flag,state,ids),raw_args=[regs[n] for n in ['RCX','RDX','R8','R9']],
               volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),caller=hex(caller),
               caller_kind='fixture_return_sentinel' if caller==stop-base else 'native_return',native_phase='CharacterData.ctor')
        events.append(e);counts[kind]=counts.get(kind,0)+1
        if options.get('failure')==[kind,counts[kind]]:raise Stopped
        state[category].append(args);mutate(kind)
        regs.update({n:POISON for n in INTEGER_VOLATILE[1:]});regs['RAX']=value;xmm[:]=[f'{((1<<127)|i):032x}' for i in range(6)]
    try:
        if flag==0:
            for (kind,category),site in zip(METADATA_ORDER,METADATA_SITES):
                name=slot_name(kind,category);regs['RCX']=base+slot_addresses[name];value=slots[name]
                emit('metadata',[name,oid(value)],site+5,'metadata',value)
            flag=1
        records=options.get('allocation_records',[f'list{i}' for i in range(6)])
        for index,(kind,(field,off)) in enumerate(zip(KINDS,FIELDS)):
            regs['RCX']=slots[slot_name(kind,'class')];record=records[index];captured=p[record]
            emit('allocation',[index,oid(regs['RCX']),record],ALLOCATION_SITES[index]+5,'allocations',captured)
            regs['RCX']=captured;regs['RDX']=slots[slot_name(kind,'method')]
            emit('list_constructor',[index,record,oid(regs['RDX'])],CTOR_SITES[index]+5,'list_constructors',options.get('list_ctor_return_bits',0xF00D123456789012))
            regs['RCX']=(p[owner] if owner else 0)+off;regs['RDX']=captured
            if owner is None:error='native_owner_write_fault';fault=hex(STORE_SITES[index]);raise Stopped
            write(owner,off,8,captured)
            emit('reference_barrier',[index,owner,field,off,record],BARRIER_SITES[index]+5,'barriers',options.get('barrier_return_bits',0xBADD123456789034))
        regs['RDX']=0;write(owner,0x13C,1,1);regs['RCX']=p[owner];write(owner,0x13E,1,1)
        value=options.get('base_return_bits',0xABCD123456789056);emit('base_constructor',[owner,0],stop-base,'base_constructors',value)
        returned=True;result=value
    except Stopped:
        if error is None:error=events[-1]['kind']
    assert (row['returned'],row['error'],row['fault_rva'],row['return_bits'])==(returned,error,fault,result),options
    assert row['final_volatile_registers']==regs and row['final_volatile_xmm_hex']==xmm,('final volatile ABI',options)
    assert row['events']==events,('events',options)
    assert row['final']==snapshot(raw,slots,flag,state,ids),('final',options)


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];sequences=[];baselines=[];stops=[]
    for warm,seed,prefilled,owner,alias in itertools.product([0,1,0x80,0xFF],[0,0xA5],[False,True],['owner','other_owner'],[False,True]):
        records=[f'list{i}' for i in range(6)]
        if alias:records[5]='list0'
        cases.append(m.run(dict(warm_byte=warm,seed_byte=seed,prefilled_owner=prefilled,owner=owner,allocation_records=records)))
    for warm in [0,0xFE]:cases.append(m.run(dict(null_owner=True,warm_byte=warm)))
    for results in [(0,0,0),(0xFFFFFFFFFFFFFFFF,0x8000000000000000,0x123456789ABCDEF0)]:
        cases.append(m.run(dict(list_ctor_return_bits=results[0],barrier_return_bits=results[1],base_return_bits=results[2],entry_rax_bits=0x1111222233334444,entry_r10_bits=0xAAAABBBBCCCCDDDD,entry_r11_bits=0xEEEEDDDDCCCCBBBB)))
    plans=[('metadata:1',[['owner',0xC0,8,'new_skin']]),('metadata:10',[['slot:'+slot_name('CharacterData','class'),0,8,'CharacterData:other_class']]),
           ('allocation:1',[['slot:'+slot_name('CharacterData','method'),0,8,'CharacterData:other_method']]),
           ('list_constructor:1',[['owner',0x48,8,'prior0'],['list0',0x10,8,'prior0']]),
           ('reference_barrier:1',[['owner',0x48,8,'prior0'],['owner',0x13C,1,0x80]]),
           ('reference_barrier:5',[['owner',0x13E,1,0x80]]),('reference_barrier:6',[['owner',0x13D,1,0x80],['owner',0xC0,8,'new_skin']]),
           ('base_constructor:1',[['owner',0x13C,1,0],['owner',0x13E,1,0],['other_owner',0xC0,8,'new_skin']]),
           ('allocation:6',[['list0',0x18,4,0xFFFFFFFF]]),('list_constructor:6',[['list0',0x10,8,'prior5']])]
    for phase,writes in plans:
        for warm,owner,alias in itertools.product([0,0xFE],['owner','other_owner'],[False,True]):
            records=[f'list{i}' for i in range(6)]
            if alias:records[5]='list0'
            cases.append(m.run(dict(warm_byte=warm,owner=owner,allocation_records=records,mutations={phase:writes})))
    for options in [{},dict(owner='other_owner'),dict(allocation_records=['list0','list1','list2','list3','list4','list0']),
                    dict(mutations={'reference_barrier:6':[['owner',0xC0,8,'new_skin']]})]:
        rows=[m.run(options),m.run(dict(options,failure=['reference_barrier',3]),True),m.run(options,True),m.run(dict(options,mutations={'base_constructor:1':[['owner',0x13D,1,0x80]]}),True)]
        assert rows[0]['returned'] and not rows[1]['returned'] and rows[2]['returned'] and rows[3]['returned']
        assert all(prior['final']==following['initial'] for prior,following in zip(rows,rows[1:]));sequences.append(rows)
    profiles=[{},dict(warm_byte=0xFE),dict(owner='other_owner'),dict(allocation_records=['list0','list1','list2','list3','list4','list0']),
              dict(mutations={'allocation:1':[['slot:'+slot_name('CharacterData','method'),0,8,'CharacterData:other_method']]}),
              dict(mutations={'reference_barrier:6':[['owner',0xC0,8,'new_skin']]}),dict(mutations={'base_constructor:1':[['owner',0x13C,1,0]]}),dict(null_owner=True)]
    for options in profiles:
        baseline=m.run(options);bid=len(baselines);baselines.append(baseline);counts={}
        for index,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;stopped=m.run(dict(options,failure=[kind,counts[kind]]))
            assert not stopped['returned'] and stopped['events']==baseline['events'][:index+1] and stopped['final']==e['snapshot']
            stops.append(dict(baseline=bid,prefix_length=index+1,result=stopped))
    assert set(m.instructions)<=m.executed
    return dict(schema='character_data_constructor_native_v1',build=BUILD,method_id='tdi5845.m0020',symbol_key='CharacterData::public void .ctor()',target=m.target,
                body_bounds=dict(start=hex(ENTRY),end_exclusive=hex(END),next_managed=hex(FOLLOWING),padding_bytes=12),
                body_identity=dict(byte_length=END-ENTRY,instruction_count=len(m.instructions),sha256=hashlib.sha256(m.pe.get_data(ENTRY,END-ENTRY)).hexdigest()),
                scope='Exact CharacterData ctor caller only; allocation, six exact-MI generic List ctor calls, barriers, metadata, and ScriptableObject base ctor tail supplied; no generic/shared alias, base/engine initialization or unassigned defaults promoted',
                supplied_targets=m.supplied,fields=m.field_declarations,metadata_slots={n:hex(a) for n,a in m.slots.items()},metadata_flag=hex(m.flag),
                diagnostic_windows_not_object_extents=m.sizes,instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),
                cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,summary=dict(cases=len(cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--game-root',required=True);parser.add_argument('--dumper-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=pool_snapshots(pool_memory(audit(args.game_root,args.dumper_root)))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],sort_keys=True))
