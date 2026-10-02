"""Actual FindLocaleLoc and actual CharacterLoc getter composition, offline."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

from audit_character_assets import BUILD
from audit_character_loc_text import Machine as TextMachine, TARGETS, VOLATILE, POISON
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots

ENTRY,END,NEXT,CLEANUP=0x3F5540,0x3F5684,0x3F5690,0x3F5645
MI_NAMES=['Method$System.Collections.Generic.List.Enumerator<LocaleLoc>.Dispose()',
 'Method$System.Collections.Generic.List.Enumerator<LocaleLoc>.MoveNext()',
 'Method$System.Collections.Generic.List.Enumerator<LocaleLoc>.get_Current()',
 'Method$System.Collections.Generic.List<LocaleLoc>.GetEnumerator()']
META_SITES=[0x3F5565,0x3F5571,0x3F557D,0x3F5589]
SEARCH_CALLS=[*META_SITES,0x3F55AE,0x3F55EC,0x3F5609,0x3F561C,0x3F563E,0x3F5651,0x3F5672,0x3F5678,0x3F567E]
NONVOL=['RBX','RBP','RSI','RDI','R12','R13','R14','R15']
EXCLUDED={0x3F5677:'after supplied nonreturning null-list guard nop',0x3F567D:'after supplied nonreturning null-current guard nop',0x3F5683:'after supplied nonreturning rethrow int3'}


def snapshot(memory,slots,flag,state,ids):
    def oid(v):return ids[v] if v else None
    def q(n,off):return int.from_bytes(memory[n][off:off+8],'little')
    return dict(memory={n:bytes(v).hex() for n,v in memory.items()},metadata_slots={n:oid(v) for n,v in slots.items()},metadata_flag=flag,
      owners={n:dict(localized_locs=oid(q(n,0x20))) for n in ['owner','other_owner']},
      locales={n:dict(locale_code=oid(q(n,0x10)),translated_name=oid(q(n,0x18)),i_was_translated=oid(q(n,0x20)),entries=oid(q(n,0x28))) for n in ['loc0','loc1','loc2']},
      supplied_state=deepcopy(state))


class Machine(TextMachine):
    def __init__(self,game_root,dumper_root):
        import capstone
        super().__init__(game_root,dumper_root)
        manifest=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name,key):
            raw=(Path(dumper_root)/name).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==manifest['outputs'][key]['sha256'].upper();return raw.decode('utf-8-sig')
        dump=pin('dump.cs','dump_cs');header=pin('il2cpp.h','il2cpp_h')
        self.fields=['public List<LocaleLoc> localizedLocs; // 0x20','public string localeCode; // 0x10','public string translatedName; // 0x18','public string iWasTranslated; // 0x20','public List<CustomValuess> entries; // 0x28']
        for name,index,fields in [('CharacterLoc',5972,self.fields[:1]),('LocaleLoc',5969,self.fields[1:])]:
            block=re.search(r'^public class '+name+r' // TypeDefIndex: '+str(index)+r'\s*\{(.*?)\n\}',dump,re.M|re.S);assert block and all(f in block[1] for f in fields)
        block=re.search(r'^public struct List\.Enumerator<T> : IEnumerator<T>, IDisposable, IEnumerator // TypeDefIndex: 1509\s*\{(.*?)\n\}',dump,re.M|re.S);assert block
        assert all(f in block[1] for f in ['private List<T> _list; // 0x0','private int _index; // 0x0','private int _version; // 0x0','private T _current; // 0x0'])
        block=re.search(r'struct System_Collections_Generic_List_Enumerator_T__Fields \{(.*?)\n\};',header,re.S);assert block
        assert [s.strip() for s in block[1].splitlines() if s.strip()]==['struct System_Collections_Generic_List_T__o* _list;','int32_t _index;','int32_t _version;','Il2CppObject* _current;']
        rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==ENTRY];assert len(rows)==1
        assert (rows[0]['Name'],rows[0]['Signature'],rows[0]['TypeSignature'])==('CharacterLoc$$FindLocaleLoc','LocaleLoc_o* CharacterLoc__FindLocaleLoc (CharacterLoc_o* __this, System_String_o* localeCode, const MethodInfo* method);','iiii');self.targets.append(dict(rows[0],method_id='tdi5972.m0001',symbol_key='CharacterLoc::private LocaleLoc FindLocaleLoc(string localeCode)'))
        assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address']>ENTRY)==NEXT
        chunks=[]
        for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root=e
            while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
            if root.struct.BeginAddress==ENTRY:chunks.append((e.struct.BeginAddress,e.struct.EndAddress));assert e.unwindinfo.Flags==3 and e.unwindinfo.ExceptionHandler==0x30CD28
        assert chunks==[(ENTRY,END)]
        section=self.pe.get_section_by_rva(ENTRY);assert section and NEXT<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(ENTRY,NEXT-ENTRY);assert len(raw)==NEXT-ENTRY and raw[END-ENTRY:]==b'\xcc'*12
        ins=list(self.cs.disasm(raw[:END-ENTRY],ENTRY));assert sum(i.size for i in ins)==END-ENTRY
        assert (ins[-1].address,ins[-1].size,ins[-1].mnemonic,ins[-1].op_str)==(0x3F5683,1,'int3','');self.instructions.update({i.address:i for i in ins})
        refs=set()
        for i in ins:
            for op in i.operands:
                if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+op.mem.disp)
            if i.mnemonic=='cmp' and i.operands[0].type==capstone.x86.X86_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:self.flag=i.address+i.size+i.operands[0].mem.disp
        self.slot_names={r['Address']:r['Name'] for cat in ['ScriptMetadata','ScriptMetadataMethod'] for r in self.metadata[cat] if r['Address'] in refs};self.slots={n:a for a,n in self.slot_names.items()};assert set(self.slots)==set(MI_NAMES)
        for n,site in zip(MI_NAMES,META_SITES):
            i=self.instructions[site-7];assert i.mnemonic=='lea' and self.slot_names[i.address+i.size+i.operands[1].mem.disp]==n
        for gw,sites in [(0x2B7B40,META_SITES),(0xB16640,[0x3F55AE]),(0x9693D0,[0x3F55EC]),(0xF73E00,[0x3F5609]),(0x33ED50,[0x3F561C,0x3F563E,0x3F5651]),(0x2B7D90,[0x3F5672,0x3F5678]),(0x246610,[0x3F567E])]:assert [i.address for i in ins if i.mnemonic=='call' and i.op_str==hex(gw)]==sites
        self.bounds['FindLocaleLoc']=dict(start=hex(ENTRY),end_exclusive=hex(END),next_managed=hex(NEXT),padding_bytes=12,unwind=[[hex(a),hex(b)] for a,b in chunks],eh_flags=3,handler_rva='0x30cd28',direct_cleanup=hex(CLEANUP),byte_length=END-ENTRY,instructions=len(ins),sha256=hashlib.sha256(raw[:END-ENTRY]).hexdigest())
        for name,(a,b,_,_,_) in TARGETS.items():assert (self.instructions[b-1].size,self.instructions[b-1].mnemonic,self.instructions[b-1].op_str)==(1,'ret','')
        self.checks.update({0x3F554F:('mov','rsi, rdx'),0x3F5552:('mov','rbx, rcx'),0x3F5595:('mov','rdx, qword ptr [rbx + 0x20]'),0x3F55A9:('lea','rcx, [rsp + 0x28]'),0x3F55B3:('movups','xmm0, xmmword ptr [rsp + 0x28]'),0x3F55B8:('movups','xmmword ptr [rsp + 0x40], xmm0'),0x3F55BD:('movsd','xmm1, qword ptr [rsp + 0x38]'),0x3F55C3:('movsd','qword ptr [rsp + 0x50], xmm1'),0x3F55C9:('mov','qword ptr [rsp + 0x28], 0'),0x3F55D7:('mov','qword ptr [rsp + 0x30], rbx'),0x3F55F5:('mov','rdi, qword ptr [rsp + 0x50]'),0x3F5602:('mov','rdx, rsi'),0x3F5605:('mov','rcx, qword ptr [rdi + 0x10]'),0x3F5621:('mov','rax, rdi'),0x3F564C:('mov','rcx, qword ptr [rsp + 0x30]'),0x3F5656:('mov','rcx, qword ptr [rsp + 0x28]'),0x3F5660:('xor','eax, eax')})
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())
        self.supplied=[r for r in self.supplied if r['Address']!=ENTRY]
        for a,n,s,t in [(0xB16640,'System.Collections.Generic.List<object>$$GetEnumerator','System_Collections_Generic_List_Enumerator_T__o System_Collections_Generic_List_object___GetEnumerator (System_Collections_Generic_List_object__o* __this, const MethodInfo_B16640* method);','iii'),(0x9693D0,'System.Collections.Generic.List.Enumerator<object>$$MoveNext','bool System_Collections_Generic_List_Enumerator_object___MoveNext (System_Collections_Generic_List_Enumerator_T__o __this, const MethodInfo_9693D0* method);','iii'),(0x33ED50,'System.Collections.Generic.List.Enumerator<object>$$Dispose','void System_Collections_Generic_List_Enumerator_object___Dispose (System_Collections_Generic_List_Enumerator_T__o __this, const MethodInfo_33ED50* method);','vii'),(0xF73E00,'System.String$$op_Equality','bool System_String__op_Equality (System_String_o* a, System_String_o* b, const MethodInfo* method);','iiii')]:
            rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==a and r['Name']==n];assert len(rows)==1 and rows[0]['Signature']==s and rows[0]['TypeSignature']==t;self.supplied.extend(rows)
        names=['loc2','list0','list1','array0','array1','string_class','exception','clone_enum']+[f'mi{i}' for i in range(4)]+[f'other_mi{i}' for i in range(4)]
        for i,n in enumerate(names):self.p[n]=self.arena+0xA20000+i*0x1000;self.sizes[n]=256 if n.startswith(('list','array')) or n.endswith('class') else 128
        self.entry_sp=self.stack+0x18008;self.graph_start=self.entry_sp-0x78;self.p['scratch']=self.graph_start;self.sizes['scratch']=0x70
        self.p.update(standalone_output=self.entry_sp-0x40,standalone_enum=self.entry_sp-0x28,nested_output=self.entry_sp-0x70,nested_enum=self.entry_sp-0x58);self.ids={v:n for n,v in self.p.items()}
        self.tracking=False;self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE,self.observe_write)

    def snapshot(self):return snapshot({n:bytes(self.u.mem_read(self.p[n],s)) for n,s in self.sizes.items()},{n:self.rq(self.base+a) for n,a in self.slots.items()},self.u.mem_read(self.base+self.flag,1)[0],self.state,self.ids)
    def volatile(self):return {n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in VOLATILE}
    def write(self,n,off,size,value):
        if isinstance(value,str):value=self.p[value]
        if n in ['active_enum','hidden_output']:
            off+=(self.enum_state if n=='active_enum' else self.output)-self.graph_start;n='scratch'
        self.u.mem_write(self.p[n]+off,value.to_bytes(size,'little'));self.allowed.setdefault(n,set()).update(range(off,off+size))
    def mutate(self,kind):
        count=self.completed.get(kind,0)+1;self.completed[kind]=count
        for target,off,size,value in self.options.get('mutations',{}).get(kind+':'+str(count),[]):
            if target.startswith('slot:'):
                n=target[5:];self.q(self.base+self.slots[n],self.p[value]);self.slot_writes[n]=value
            else:self.write(target,off,size,value)
    def observe_write(self,uc,access,address,size,value,data):
        if not self.tracking:return
        value &= (1 << (size*8))-1
        pc=self.reg(self.x.UC_X86_REG_RIP)-self.base
        if address==self.base+self.flag:assert pc==0x3F558E and size==value==1;self.flag_written=True
        elif self.graph_start<=address<self.graph_start+0x70:
            off=address-self.graph_start;expected=[]
            if pc in SEARCH_CALLS and not self.nested:expected=[(8,8,self.base+pc+5)]
            if pc in [a+9 for a,*_ in TARGETS.values()]+[a+28 for a,*_ in TARGETS.values()]:expected=[(0x48,8,self.base+pc+5)]
            if self.nested and pc in [0x3F5540,0x3F5545,0x3F554A]:expected=[({0x3F5540:0x50,0x3F5545:0x58,0x3F554A:0x40}[pc],8,0xFAB0000000000000+{0x3F5540:0,0x3F5545:2,0x3F554A:3}[pc])]
            out=self.output-self.graph_start;active=self.enum_state-self.graph_start
            ranges={0x3F55B8:(active,16),0x3F55C3:(active+16,8),0x3F55C9:(out,8),0x3F55D7:(out+8,8)}
            if pc in ranges:
                first,total=ranges[pc];assert (off,size)==(first,total) or (total==16 and (off,size) in [(first,8),(first+8,8)]),(hex(pc),off,size)
            else:assert (off,size,value) in expected,(hex(pc),off,size,hex(value),expected)
            self.allowed.setdefault('scratch',set()).update(range(off,off+size))

    def prepare(self,options):
        self.options=deepcopy(options);self.events=[];self.counts={};self.error=None
        self.state=dict(entries=[],metadata=[],enumerators=[],moves=[],comparisons=[],emptiness=[],disposals=[],iterator=None)
        for n,s in self.sizes.items():self.u.mem_write(self.p[n],bytes([options.get('seed_byte',0xA5)])*s)
        for i,n in enumerate(MI_NAMES):self.q(self.base+self.slots[n],self.p[f'mi{i}'])
        self.u.mem_write(self.base+self.flag,bytes([options.get('warm_byte',0)]))
        for n in ['owner','other_owner']:self.q(self.p[n],self.p['owner_class']);self.q(self.p[n]+0x20,self.p[options.get('owner_list','list0')] if not options.get('null_list') else 0)
        for n,text in [('code','target'),('other_code','other'),('name','name'),('other_name','other name'),('i_was','I was'),('other_i_was','other I was'),('replacement','replacement')]:
            self.q(self.p[n],self.p['string_class']);self.d(self.p[n]+0x10,len(text));self.u.mem_write(self.p[n]+0x14,text.encode('utf-16-le')+b'\0\0')
        for i,n in enumerate(['loc0','loc1','loc2']):
            self.q(self.p[n],self.p['locale_class'])
            for off,target in [(0x10,'other_code' if i==1 else 'code'),(0x18,'other_name' if i==1 else 'name'),(0x20,'other_i_was' if i else 'i_was'),(0x28,'entries')]:
                if options.get('alias_texts') and off in [0x18,0x20]:target='code'
                self.q(self.p[n]+off,0 if options.get('null_field')==off else self.p[target])
        for i in range(2):
            entries=options.get('entries' if i==0 else 'other_entries',['loc0','loc1','loc2'] if i==0 else ['loc1','loc0'])
            self.q(self.p[f'list{i}']+0x10,self.p[f'array{i}']);self.d(self.p[f'list{i}']+0x18,len(entries));self.d(self.p[f'list{i}']+0x1C,0x12345678+i);self.q(self.p[f'array{i}']+0x18,len(entries))
            for j,n in enumerate(entries):self.q(self.p[f'array{i}']+0x20+j*8,self.p[n] if n else 0)
        self.u.mem_write(self.p['clone_enum'],self.p['list1'].to_bytes(8,'little')+(2).to_bytes(4,'little')+(0xDEADBEEF).to_bytes(4,'little')+self.p['loc1'].to_bytes(8,'little'))
        if options.get('cleanup'):
            frame=self.entry_sp-0x68;self.q(frame+0x28,self.p['exception'] if options.get('exception_live') else 0);self.q(frame+0x30,self.p[options.get('cleanup_enum','standalone_enum')]);self.u.mem_write(self.p['standalone_enum'],bytes(self.u.mem_read(self.p['clone_enum'],24)))
    def event(self,kind,args):
        self.events.append(dict(kind=kind,args=args,snapshot=self.snapshot(),raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm(),caller_return_bits=self.rq(self.reg(self.x.UC_X86_REG_RSP)),native_phase=self.phase,entry_api=self.api,synthetic_cleanup=self.cleanup))
        self.counts[kind]=self.counts.get(kind,0)+1
        if self.options.get('failure')==[kind,self.counts[kind]]:self.error=kind;self.u.emu_stop();return False
        return True
    def entry(self):
        caller=self.rq(self.reg(self.x.UC_X86_REG_RSP))
        self.state['entries'].append(dict(method=self.phase,owner=self.oid(self.owner),code=self.oid(self.code),volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm(),caller_return_bits=caller,
          caller_kind='synthetic_frame_slot' if self.cleanup else 'fixture_return' if caller==self.stop else 'native_return',synthetic_cleanup=self.cleanup))
    def value(self,key,index,default):return self.options.get(key,[])[index] if index<len(self.options.get(key,[])) else default
    def equal(self,a,b):
        if not a or not b:return a==b
        return bytes(self.u.mem_read(a+0x14,self.rd(a+0x10)*2))==bytes(self.u.mem_read(b+0x14,self.rd(b+0x10)*2))
    def hook(self,uc,address,size,data):
        rva=address-self.base;self.executed.add(rva)
        if rva in self.instructions:
            if rva==ENTRY or (self.cleanup and rva==CLEANUP):self.phase='FindLocaleLoc';assert self.reg(self.x.UC_X86_REG_RSP)==(self.frame if self.cleanup else self.search_entry_sp);self.entry()
            elif self.api in TARGETS and rva==TARGETS[self.api][0]:self.phase=self.api;self.entry()
            elif self.api in TARGETS and rva==TARGETS[self.api][0]+14:self.phase=self.api
            return
        cx,dx,r8=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8']]
        if rva==0x2B7B40:
            n=self.slot_names[cx-self.base];value=self.rq(cx);args=[n,self.oid(value)]
            if self.event('metadata',args):self.state['metadata'].append(args);self.mutate('metadata');self.ret(value)
        elif rva==0xB16640:
            assert cx==self.output and self.oid(dx) in ['list0','list1'] and self.oid(r8) in ['mi3','other_mi3'];args=[self.oid(cx),self.oid(dx),self.oid(r8)]
            if self.event('get_enumerator',args):
                count=self.rd(dx+0x18);assert count<=4;array=self.rq(dx+0x10);entries=[self.oid(self.rq(array+0x20+i*8)) for i in range(count)];raw=dx.to_bytes(8,'little')+bytes(4)+self.rd(dx+0x1C).to_bytes(4,'little')+bytes(8)
                self.state['enumerators'].append(dict(args=args,produced_24_bytes=raw.hex(),supplied_entries=entries));self.state['iterator']=dict(list=self.oid(dx),entries=entries,cursor=0)
                for off in range(0,24,8):self.write('hidden_output',off,8,int.from_bytes(raw[off:off+8],'little'))
                self.mutate('get_enumerator');self.ret(self.options.get('get_return_bits',self.output))
        elif rva==0x9693D0:
            assert cx==self.enum_state and self.oid(dx) in ['mi1','other_mi1'];args=[self.oid(cx),self.oid(dx)]
            if self.event('move_next',args):
                it=self.state['iterator'];i=it['cursor'];has=i<len(it['entries']);current=it['entries'][i] if has else None;it['cursor']+=1;value=self.value('move_next_raw',i,0xBEEF123456789000|int(has))
                self.write('active_enum',8,4,i+1);self.write('active_enum',16,8,current if current else 0);self.state['moves'].append(dict(args=args,current=current,return_bits=value));self.mutate('move_next');self.ret(value)
        elif rva==0xF73E00:
            assert r8==0;args=[self.oid(cx),self.oid(dx),0]
            if self.event('string_equality',args):
                i=len(self.state['comparisons'])-self.comparison_start;value=self.value('equality_raw',i,0xCAFE123456789000|int(self.equal(cx,dx)));self.state['comparisons'].append(dict(args=args,return_bits=value));self.mutate('string_equality');self.ret(value)
        elif rva==0x33ED50:
            assert self.oid(cx) in ['standalone_enum','nested_enum','clone_enum'] and self.oid(dx) in ['mi0','other_mi0'];args=[self.oid(cx),self.oid(dx)]
            if self.event('dispose',args):self.state['disposals'].append(args);self.mutate('dispose');self.ret(self.options.get('dispose_return_bits',0xD15E1234567890AB))
        elif rva==0xF76390:
            assert self.phase in TARGETS and dx==0;args=[self.oid(cx),0];value=self.options.get('empty_result_bits',0xBADDF00D00000000)
            if self.event('is_null_or_empty',args):self.state['emptiness'].append(dict(args=args,return_bits=value));self.mutate('is_null_or_empty');self.ret(value)
        elif rva in [0x2B7D90,0x246610]:
            kind='native_guard' if rva==0x2B7D90 else 'rethrow';site=self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base-5;assert site in [0x3F5672,0x3F5678] if kind=='native_guard' else site==0x3F567E
            args=[hex(site)] if kind=='native_guard' else [self.oid(cx)]
            if self.event(kind,args):self.error=kind;self.u.emu_stop()
        else:raise AssertionError(f'unclaimed native {rva:x}')

    def run(self,api,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options=deepcopy(options or {});self.counts={};self.error=None
        self.api=api;self.cleanup=self.options.get('cleanup',False);assert not self.cleanup or api=='FindLocaleLoc';self.nested=api in TARGETS;self.phase=api
        self.search_entry_sp=self.entry_sp-0x30 if self.nested else self.entry_sp;self.frame=self.search_entry_sp-0x68;self.output=self.frame+0x28;self.enum_state=self.frame+0x40
        self.owner=self.pointer(self.options.get('owner','owner'));self.code=self.pointer(self.options.get('code','code'));self.allowed,self.slot_writes,self.flag_written,self.completed={},{},False,{};self.comparison_start=len(self.state['comparisons'])
        initial=self.snapshot();old=len(self.events);x=self.x;self.q(self.entry_sp,self.stop)
        incoming={n:self.options.get('entry_'+n.lower()+'_bits',0xDEAD123400000000+i) for i,n in enumerate(VOLATILE)};incoming.update(RCX=self.owner,RDX=self.code);initial_xmm=[f'{((1<<126)|i):032x}' for i in range(6)]
        for n,v in incoming.items():self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        for i,v in enumerate(initial_xmm):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),int(v,16))
        for i,n in enumerate(NONVOL):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),(1<<125)|i)
        if self.cleanup:
            for off,n in [(0x60,'RDI'),(0x70,'RBX'),(0x78,'RSI')]:self.q(self.frame+off,0xFAB0000000000000+NONVOL.index(n))
        self.u.reg_write(x.UC_X86_REG_RSP,self.frame if self.cleanup else self.entry_sp);self.tracking=True;fault=None
        try:self.u.emu_start(self.base+(CLEANUP if self.cleanup else ENTRY if api=='FindLocaleLoc' else TARGETS[api][0]),self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base;assert self.owner==0 and (pc,exc.errno)==(0x3F5595,self.unicorn.UC_ERR_READ_UNMAPPED);self.error='native_owner_read_fault';fault=hex(pc)
        finally:self.tracking=False
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==self.entry_sp+8
            assert all(self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB0000000000000+i for i,n in enumerate(NONVOL))
            assert all(self.reg(getattr(x,'UC_X86_REG_XMM'+str(i)))==(1<<125)|i for i in range(6,16))
        final=self.snapshot()
        for n,before in initial['memory'].items():
            a,b=bytes.fromhex(before),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or value==b[i] for i,value in enumerate(a)),n
        assert final['metadata_slots']=={**initial['metadata_slots'],**self.slot_writes};assert final['metadata_flag']==(1 if self.flag_written else initial['metadata_flag'])
        row=dict(api=api,options=self.options,synthetic_cleanup=self.cleanup,initial=initial,events=deepcopy(self.events[old:]),final=final,returned=returned,error=self.error,fault_rva=fault,final_phase=self.phase,
          entry_volatile_registers=incoming,entry_volatile_xmm_hex=initial_xmm,final_volatile_registers=self.volatile(),final_volatile_xmm_hex=self.xmm(),result_bits=self.reg(x.UC_X86_REG_RAX) if returned else None,
          normal_or_synthetic_frame_abi_verified=returned,completed_memory_write_offsets={n:sorted(v) for n,v in self.allowed.items()},completed_slot_writes=self.slot_writes.copy(),reached_native_flag_write=self.flag_written,unrelated_storage_retained=True)
        verify(row,self.p,self.ids,self.slots,self.base,self.stop,self.entry_sp);row['independent_ordered_full_state_verified']=True;return row


def verify(row,p,ids,slot_addresses,base,stop,entry_sp):
    raw={n:bytearray.fromhex(v) for n,v in row['initial']['memory'].items()};slots={n:p[v] for n,v in row['initial']['metadata_slots'].items()};flag=row['initial']['metadata_flag'];state=deepcopy(row['initial']['supplied_state']);options=row['options'];api=row['api'];cleanup=row['synthetic_cleanup'];nested=api in TARGETS;phase=api
    regs=row['entry_volatile_registers'].copy();xmm=row['entry_volatile_xmm_hex'].copy();owner=ids[regs['RCX']] if regs['RCX'] else None;code=regs['RDX'];events=[];counts={};completed={};returned=False;error=None;fault=None;result=None;comparison_start=len(state['comparisons']);out=8 if nested else 0x38;active=0x20 if nested else 0x50;output=p['nested_output' if nested else 'standalone_output'];enum=p['nested_enum' if nested else 'standalone_enum']
    def oid(v):return ids[v] if v else None
    def q(n,off):return int.from_bytes(raw[n][off:off+8],'little')
    def rd(n,off):return int.from_bytes(raw[n][off:off+4],'little')
    def write(n,off,size,value):
        if n in ['active_enum','hidden_output']:off+=active if n=='active_enum' else out;n='scratch'
        raw[n][off:off+size]=(p[value] if isinstance(value,str) else value).to_bytes(size,'little')
    def entry(caller):state['entries'].append(dict(method=phase,owner=owner,code=oid(code),volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),caller_return_bits=caller,
      caller_kind='synthetic_frame_slot' if cleanup else 'fixture_return' if caller==stop else 'native_return',synthetic_cleanup=cleanup))
    entry(q('scratch',0x10) if cleanup else stop)
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
        if phase in TARGETS:write('scratch',0x48,8,base+site+5)
        elif not nested:write('scratch',8,8,base+site+5)
        events.append(dict(kind=kind,args=args,snapshot=snapshot(raw,slots,flag,state,ids),raw_args=[regs[n] for n in ['RCX','RDX','R8','R9']],volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),caller_return_bits=base+site+5,native_phase=phase,entry_api=api,synthetic_cleanup=cleanup))
        counts[kind]=counts.get(kind,0)+1
        if options.get('failure')==[kind,counts[kind]] or kind in ['native_guard','rethrow']:raise Stopped
        if category:state[category].append(args)
        if effect:effect()
        mutate(kind);regs.update({n:POISON for n in VOLATILE[1:]});regs['RAX']=value;xmm[:]=[f'{((1<<127)|i):032x}' for i in range(6)]
    def dispose(site,receiver):
        regs['RDX']=slots[MI_NAMES[0]];regs['RCX']=receiver;emit('dispose',[oid(receiver),oid(regs['RDX'])],site,'disposals',value=options.get('dispose_return_bits',0xD15E1234567890AB))
    try:
        if nested:
            regs['R8']=0;write('scratch',0x48,8,base+TARGETS[api][0]+14);phase='FindLocaleLoc';entry(base+TARGETS[api][0]+14)
            for off,index in [(0x50,0),(0x58,2),(0x40,3)]:write('scratch',off,8,0xFAB0000000000000+index)
        if cleanup:
            dispose(0x3F5651,q('scratch',out+8));regs['RCX']=q('scratch',out)
            if regs['RCX']:emit('rethrow',[oid(regs['RCX'])],0x3F567E)
            regs['RAX']=0
        else:
            if flag==0:
                for n,site in zip(MI_NAMES,META_SITES):
                    regs['RCX']=base+slot_addresses[n];value=slots[n];emit('metadata',[n,oid(value)],site,'metadata',value=value)
                flag=1
            if owner is None:error='native_owner_read_fault';fault='0x3f5595';raise Stopped
            regs['RDX']=q(owner,0x20)
            if regs['RDX']==0:emit('native_guard',['0x3f5672'],0x3F5672)
            regs['R8']=slots[MI_NAMES[3]];regs['RCX']=output;listname=oid(regs['RDX']);args=[oid(output),listname,oid(regs['R8'])];entries=[oid(q(oid(q(listname,0x10)),0x20+i*8)) for i in range(rd(listname,0x18))];produced=p[listname].to_bytes(8,'little')+bytes(4)+rd(listname,0x1C).to_bytes(4,'little')+bytes(8)
            def make_enum():state['enumerators'].append(dict(args=args,produced_24_bytes=produced.hex(),supplied_entries=entries));state['iterator']=dict(list=listname,entries=entries,cursor=0);raw['scratch'][out:out+24]=produced
            emit('get_enumerator',args,0x3F55AE,effect=make_enum,value=options.get('get_return_bits',output))
            xmm[0]=f'{int.from_bytes(raw["scratch"][out:out+16],"little"):032x}';xmm[1]=f'{int.from_bytes(raw["scratch"][out+16:out+24],"little"):032x}';raw['scratch'][active:active+24]=raw['scratch'][out:out+24];write('scratch',out,8,0);write('scratch',out+8,8,enum)
            matched=False
            while True:
                regs['RDX']=slots[MI_NAMES[1]];regs['RCX']=enum;args=[oid(enum),oid(regs['RDX'])];it=state['iterator'];i=it['cursor'];has=i<len(it['entries']);current=it['entries'][i] if has else None;value=series('move_next_raw',i,0xBEEF123456789000|int(has))
                def move():it['cursor']+=1;write('active_enum',8,4,i+1);write('active_enum',16,8,current if current else 0);state['moves'].append(dict(args=args,current=current,return_bits=value))
                emit('move_next',args,0x3F55EC,effect=move,value=value)
                if regs['RAX']&0xFF==0:break
                captured=q('scratch',active+16)
                if captured==0:emit('native_guard',['0x3f5678'],0x3F5678)
                regs['R8']=0;regs['RDX']=code;regs['RCX']=q(oid(captured),0x10);args=[oid(regs['RCX']),oid(regs['RDX']),0];i=len(state['comparisons'])-comparison_start;value=series('equality_raw',i,0xCAFE123456789000|int(equal(regs['RCX'],regs['RDX'])))
                emit('string_equality',args,0x3F5609,effect=lambda:state['comparisons'].append(dict(args=args,return_bits=value)),value=value)
                if regs['RAX']&0xFF==0:continue
                dispose(0x3F561C,enum);regs['RAX']=captured;matched=True;break
            if not matched:dispose(0x3F563E,enum);regs['RAX']=0
        if nested:
            phase=api;captured=regs['RAX']
            if captured:
                field=TARGETS[api][4];regs['RCX']=q(oid(captured),field);regs['RDX']=0;args=[oid(regs['RCX']),0];value=options.get('empty_result_bits',0xBADDF00D00000000)
                emit('is_null_or_empty',args,TARGETS[api][0]+28,effect=lambda:state['emptiness'].append(dict(args=args,return_bits=value)),value=value);regs['RAX']=0 if value&0xFF else q(oid(captured),field)
            else:regs['RAX']=0
        returned=True;result=regs['RAX']
    except Stopped:
        if error is None:error=events[-1]['kind']
    assert (row['returned'],row['error'],row['fault_rva'],row['result_bits'],row['final_phase'])==(returned,error,fault,result,phase),options
    assert row['events']==events,('events',api,options)
    assert row['final']==snapshot(raw,slots,flag,state,ids),('final',api,options)
    assert row['final_volatile_registers']==regs and row['final_volatile_xmm_hex']==xmm,('volatile ABI',api,options)


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];sequences=[];baselines=[];stops=[];apis=['FindLocaleLoc',*TARGETS]
    profiles=[[],['loc0'],['loc1'],['loc0','loc1','loc2'],['loc2','loc0'],['loc0','loc0'],[None],['loc1',None],['loc0',None]]
    for api,warm,seed,owner,entries in itertools.product(apis,[0,1,0x80],[0,0xA5],['owner','other_owner'],profiles):cases.append(m.run(api,dict(warm_byte=warm,seed_byte=seed,owner=owner,entries=entries)))
    for api in apis:
        for options in [dict(owner=None),dict(null_list=True),dict(code=None),dict(owner_list='list1'),dict(alias_texts=True),dict(equality_raw=[0xFFFFFFFFFFFFFF00]*3),dict(equality_raw=[0x8000000000000080]*3),dict(move_next_raw=[0x123456789ABCDE00]),dict(move_next_raw=[0x80000000000000FF,0]),dict(dispose_return_bits=0xFFFFFFFFFFFFFFFF),dict(get_return_bits=0),dict(entries=[],dispose_return_bits=0xFFFFFFFFFFFFFFFF),dict(empty_result_bits=0xFFFFFFFFFFFFFF80),dict(empty_result_bits=0xFFFFFFFFFFFFFF00),dict(null_field=0x18),dict(null_field=0x20),dict(null_field=0x10,code=None),dict(entry_rax_bits=0xFFFFFFFFFFFFFFFF,entry_r8_bits=0x123456789ABCDEF0,entry_r9_bits=0x8877665544332211)]:cases.append(m.run(api,options))
    plans=[('metadata:1',[['owner',0x20,8,'list1']]),('metadata:4',[['slot:'+MI_NAMES[3],0,8,'other_mi3']]),('get_enumerator:1',[['owner',0x20,8,'list1'],['slot:'+MI_NAMES[1],0,8,'other_mi1']]),('move_next:1',[['active_enum',16,8,'loc1']]),('move_next:1',[['active_enum',16,8,0]]),('string_equality:1',[['active_enum',16,8,'loc1'],['loc0',0x10,8,'other_code']]),('string_equality:1',[['loc0',0x18,8,'replacement'],['loc0',0x20,8,'replacement']]),('dispose:1',[['active_enum',16,8,'loc1'],['loc0',0x18,8,'replacement'],['loc0',0x20,8,0]]),('dispose:1',[['slot:'+MI_NAMES[0],0,8,'other_mi0']]),('is_null_or_empty:1',[['loc0',0x18,8,0],['loc0',0x20,8,'replacement']])]
    for api,(kind,writes),warm,owner in itertools.product(apis,plans,[0,0xFE],['owner','other_owner']):cases.append(m.run(api,dict(warm_byte=warm,owner=owner,mutations={kind:writes})))
    for live,receiver in itertools.product([False,True],['standalone_enum','clone_enum']):cases.append(m.run('FindLocaleLoc',dict(cleanup=True,exception_live=live,cleanup_enum=receiver)))
    for value in [0,'exception']:cases.append(m.run('FindLocaleLoc',dict(cleanup=True,exception_live=(value==0),mutations={'dispose:1':[['hidden_output',0,8,value]]})))
    for api in apis:
        for options in [{},dict(owner='other_owner'),dict(entries=['loc2','loc0']),dict(mutations={'dispose:1':[['loc0',0x18,8,'replacement']]})]:
            rows=[m.run(api,options),m.run(api,dict(options,failure=['string_equality',1]),True),m.run(api,options,True),m.run(api,dict(options,mutations={'dispose:1':[['other_owner',0x20,8,'list1']]}),True)]
            assert [r['returned'] for r in rows]==[True,False,True,True] and all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
    rows=[m.run('FindLocaleLoc'),m.run('GetTranslatedName',{},True),m.run('GetIWasTranslated',dict(failure=['is_null_or_empty',1]),True),m.run('FindLocaleLoc',{},True)]
    assert [r['returned'] for r in rows]==[True,True,False,True] and all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
    for api in apis:
        baseline_options=[{},dict(warm_byte=0xFE),dict(entries=[]),dict(entries=[None]),dict(null_list=True),dict(owner=None),dict(mutations={'get_enumerator:1':[['slot:'+MI_NAMES[1],0,8,'other_mi1']]}),dict(mutations={'dispose:1':[['active_enum',16,8,'loc1'],['loc0',0x18,8,'replacement'],['loc0',0x20,8,0]]})]
        if api=='FindLocaleLoc':baseline_options += [dict(cleanup=True),dict(cleanup=True,exception_live=True),dict(cleanup=True,cleanup_enum='clone_enum',mutations={'dispose:1':[['hidden_output',0,8,'exception']]})]
        for options in baseline_options:
            baseline=m.run(api,options);bid=len(baselines);baselines.append(baseline);counts={}
            for i,e in enumerate(baseline['events']):
                kind=e['kind'];counts[kind]=counts.get(kind,0)+1;stopped=m.run(api,dict(options,failure=[kind,counts[kind]]));assert not stopped['returned'] and stopped['events']==baseline['events'][:i+1] and stopped['final']==e['snapshot'];stops.append(dict(baseline=bid,prefix_length=i+1,result=stopped))
    missing=set(m.instructions)-m.executed;assert missing==set(EXCLUDED),(sorted(missing),sorted(set(EXCLUDED)-missing))
    return dict(schema='character_loc_search_join_native_v1',build=BUILD,targets=m.targets,bounds=m.bounds,supplied_targets=m.supplied,fields=m.fields,
      scope='Actual standalone FindLocaleLoc, direct synthetic cleanup, and actual two getter -> actual FindLocaleLoc composition in one physical graph. Whole metadata/iterator/string/guards/rethrow supplied; no managed exception dispatch or unwind.',
      enumerator_layout=dict(bytes=24,list_offset=0,index_offset=8,version_offset=12,current_offset=16),stack_layout=dict(shared_window_bytes=112,shared_start='outer entry SP-78',standalone_frame='SP-68',nested_search_frame='SP-98',nested_search_entry='SP-30',standalone_output_offset=56,standalone_active_offset=80,nested_output_offset=8,nested_active_offset=32),
      metadata_slots={n:hex(a) for n,a in m.slots.items()},metadata_flag=hex(m.flag),diagnostic_windows_not_object_extents=m.sizes,instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),excluded_instruction_reasons={hex(a):s for a,s in EXCLUDED.items()},
      cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,summary=dict(cases=len(cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--game-root',required=True);parser.add_argument('--dumper-root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    report=pool_snapshots(pool_memory(audit(args.game_root,args.dumper_root)));Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps(report['summary'],sort_keys=True))
