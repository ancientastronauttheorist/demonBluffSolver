"""Exact LoadPreferences caller; collection, save and LoadSkin services supplied."""
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

ENTRY,END,NEXT,CLEANUP=0x3B4DB0,0x3B4F02,0x3B4F10,0x3B4EC5
MI_NAMES=['Method$System.Collections.Generic.List.Enumerator<CharacterPreference>.Dispose()',
 'Method$System.Collections.Generic.List.Enumerator<CharacterPreference>.MoveNext()',
 'Method$System.Collections.Generic.List.Enumerator<CharacterPreference>.get_Current()',
 'Method$System.Collections.Generic.List<CharacterPreference>.GetEnumerator()']
META_SITES=[0x3B4DD2,0x3B4DDE,0x3B4DEA,0x3B4DF6]
VOL=['RAX','RCX','RDX','R8','R9','R10','R11'];NONVOL=['RBX','RBP','RSI','RDI','R12','R13','R14','R15'];POISON=0xFACE123456789090
EXCLUDED={0x3B4EF5:'post nonreturning current guard nop',0x3B4EFB:'post nonreturning rethrow int3',0x3B4F01:'post nonreturning save/List guard int3'}

def snapshot(memory,slots,flag,state,ids):
    return dict(memory={n:bytes(v).hex() for n,v in memory.items()},metadata_slots={n:ids[v] if v else None for n,v in slots.items()},metadata_flag=flag,supplied_state=deepcopy(state))

class Machine(NativeMachine):
    def __init__(self,game_root,dumper_root):
        import capstone
        super().__init__(game_root)
        ext=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(n,k):
            b=(Path(dumper_root)/n).read_bytes();assert hashlib.sha256(b).hexdigest().upper()==ext['outputs'][k]['sha256'].upper();return b.decode('utf-8-sig')
        meta=json.loads(pin('script.json','script_json'));dump=pin('dump.cs','dump_cs');header=pin('il2cpp.h','il2cpp_h')
        rows=[r for r in meta['ScriptMethod'] if r['Address']==ENTRY];assert len(rows)==1
        assert (rows[0]['Name'],rows[0]['Signature'],rows[0]['TypeSignature'])==('CharacterData$$LoadPreferences','void CharacterData__LoadPreferences (CharacterData_o* __this, const MethodInfo* method);','vii');self.target=rows[0]
        assert min(r['Address'] for r in meta['ScriptMethod'] if r['Address']>ENTRY)==NEXT
        chunks=[]
        for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root=e
            while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
            if root.struct.BeginAddress==ENTRY:chunks.append((e.struct.BeginAddress,e.struct.EndAddress));assert e.unwindinfo.Flags==3 and e.unwindinfo.ExceptionHandler==0x30CD28
        assert chunks==[(ENTRY,END)]
        section=self.pe.get_section_by_rva(ENTRY);assert section and NEXT<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(ENTRY,NEXT-ENTRY);assert len(raw)==NEXT-ENTRY and raw[END-ENTRY:]==b'\xcc'*14
        ins=list(self.cs.disasm(raw[:END-ENTRY],ENTRY));assert sum(i.size for i in ins)==END-ENTRY
        assert (ins[-1].address,ins[-1].size,ins[-1].mnemonic,ins[-1].op_str)==(END-1,1,'int3','');self.instructions={i.address:i for i in ins}
        refs=set()
        for i in ins:
            for o in i.operands:
                if o.type==capstone.x86.X86_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+o.mem.disp)
            if i.mnemonic=='cmp' and i.operands[0].type==capstone.x86.X86_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:self.flag=i.address+i.size+i.operands[0].mem.disp
        assert self.flag==0x288C4DD
        self.slot_names={r['Address']:r['Name'] for c in ['ScriptMetadata','ScriptMetadataMethod'] for r in meta[c] if r['Address'] in refs};self.slots={n:a for a,n in self.slot_names.items()};assert set(self.slots)==set(MI_NAMES)
        for n,site in zip(MI_NAMES,META_SITES):
            i=self.instructions[site-7];assert i.mnemonic=='lea' and self.slot_names[i.address+i.size+i.operands[1].mem.disp]==n
        for gw,sites in [(0x2B7B40,META_SITES),(0x2B6FF0,[0x3B4E10]),(0x387870,[0x3B4E17]),(0xB16640,[0x3B4E3E]),(0x9693D0,[0x3B4E7C]),(0xF73E00,[0x3B4E9A]),(NEXT,[0x3B4EAD]),(0x33ED50,[0x3B4EBE,0x3B4ED1]),(0x2B7D90,[0x3B4EF0,0x3B4EFC]),(0x246610,[0x3B4EF6])]:assert [i.address for i in ins if i.mnemonic=='call' and i.op_str==hex(gw)]==sites
        self.fields=[]
        data_decl=re.search(r'^public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845\s*\{(.*?)\n\}',dump,re.M|re.S);assert data_decl
        method_declarations=[s.strip() for s in data_decl[1].splitlines() if s.strip().endswith('{ }')]
        assert len(method_declarations)==21 and method_declarations[11]=='public void LoadPreferences() { }'
        assert re.search(r'// RVA: 0x3B4DB0[^\n]*\n\s*public void LoadPreferences\(\) \{ \}',data_decl[1])
        for declaration,index,fields in [('CharacterData : ScriptableObject, ICharacterLocData, ICardData',5845,['public string characterId; // 0x18','public SkinData currentSkin; // 0xC0']),('SavedCharacters',5552,['public List<CharacterPreference> prefs; // 0x10']),('CharacterPreference',5553,['public string chId; // 0x10','public string prefSkinId; // 0x18'])]:
            b=re.search(r'^public class '+re.escape(declaration)+r' // TypeDefIndex: '+str(index)+r'\s*\{(.*?)\n\}',dump,re.M|re.S);assert b and all(f in b[1] for f in fields);self.fields.extend(fields)
        b=re.search(r'^public struct List\.Enumerator<T> : IEnumerator<T>, IDisposable, IEnumerator // TypeDefIndex: 1509\s*\{(.*?)\n\}',dump,re.M|re.S);assert b and all(f in b[1] for f in ['private List<T> _list; // 0x0','private int _index; // 0x0','private int _version; // 0x0','private T _current; // 0x0'])
        b=re.search(r'struct System_Collections_Generic_List_Enumerator_T__Fields \{(.*?)\n\};',header,re.S);assert b and [s.strip() for s in b[1].splitlines() if s.strip()]==['struct System_Collections_Generic_List_T__o* _list;','int32_t _index;','int32_t _version;','Il2CppObject* _current;']
        self.supplied=[]
        for a,n,s,t in [(0x387870,'SavesGame$$get_CharacterPreferences','SavedCharacters_o* SavesGame__get_CharacterPreferences (const MethodInfo* method);','ii'),(NEXT,'CharacterData$$LoadSkin','void CharacterData__LoadSkin (CharacterData_o* __this, System_String_o* skinId, const MethodInfo* method);','viii'),(0xB16640,'System.Collections.Generic.List<object>$$GetEnumerator','System_Collections_Generic_List_Enumerator_T__o System_Collections_Generic_List_object___GetEnumerator (System_Collections_Generic_List_object__o* __this, const MethodInfo_B16640* method);','iii'),(0x9693D0,'System.Collections.Generic.List.Enumerator<object>$$MoveNext','bool System_Collections_Generic_List_Enumerator_object___MoveNext (System_Collections_Generic_List_Enumerator_T__o __this, const MethodInfo_9693D0* method);','iii'),(0x33ED50,'System.Collections.Generic.List.Enumerator<object>$$Dispose','void System_Collections_Generic_List_Enumerator_object___Dispose (System_Collections_Generic_List_Enumerator_T__o __this, const MethodInfo_33ED50* method);','vii'),(0xF73E00,'System.String$$op_Equality','bool System_String__op_Equality (System_String_o* a, System_String_o* b, const MethodInfo* method);','iiii')]:
            rows=[r for r in meta['ScriptMethod'] if r['Address']==a and r['Name']==n];assert len(rows)==1 and rows[0]['Signature']==s and rows[0]['TypeSignature']==t;self.supplied.extend(rows)
        self.checks={0x3B4DBF:('mov','rsi, rcx'),0x3B4E02:('lea','rcx, [rsi + 0xc0]'),0x3B4E0B:('mov','qword ptr [rcx], rbx'),0x3B4E15:('xor','ecx, ecx'),0x3B4E25:('mov','rdx, qword ptr [rax + 0x10]'),0x3B4E39:('lea','rcx, [rsp + 0x28]'),0x3B4E43:('movups','xmm0, xmmword ptr [rsp + 0x28]'),0x3B4E48:('movups','xmmword ptr [rsp + 0x40], xmm0'),0x3B4E4D:('movsd','xmm1, qword ptr [rsp + 0x38]'),0x3B4E53:('movsd','qword ptr [rsp + 0x50], xmm1'),0x3B4E59:('mov','qword ptr [rsp + 0x28], rbx'),0x3B4E63:('mov','qword ptr [rsp + 0x30], rbx'),0x3B4E85:('mov','rdi, qword ptr [rsp + 0x50]'),0x3B4E92:('mov','rdx, qword ptr [rsi + 0x18]'),0x3B4E96:('mov','rcx, qword ptr [rdi + 0x10]'),0x3B4EA3:('xor','r8d, r8d'),0x3B4EA6:('mov','rdx, qword ptr [rdi + 0x18]'),0x3B4EAA:('mov','rcx, rsi'),0x3B4ECC:('mov','rcx, qword ptr [rsp + 0x30]'),0x3B4ED6:('mov','rcx, qword ptr [rsp + 0x28]')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())
        self.bounds=dict(start=hex(ENTRY),end_exclusive=hex(END),next_managed=hex(NEXT),padding_bytes=14,unwind_ranges=[[hex(a),hex(b)] for a,b in chunks],eh_flags=3,handler_rva='0x30cd28',byte_length=END-ENTRY,instruction_count=len(ins),sha256=hashlib.sha256(raw[:END-ENTRY]).hexdigest())
        names=['owner','other_owner','data_class','saved0','saved1','list0','list1','array0','array1','pref0','pref1','pref2','target_id','other_id','skin_id0','skin_id1','skin0','skin1','old_skin','string_class','exception','clone_enum']+[f'mi{i}' for i in range(4)]+[f'other_mi{i}' for i in range(4)]
        self.p={n:self.arena+0xA50000+i*0x1000 for i,n in enumerate(names)};self.sizes={n:512 if n in ['owner','other_owner'] else 256 if n.startswith(('list','array')) else 128 for n in names}
        self.entry_sp=self.stack+0x18008;self.frame=self.entry_sp-0x68;self.p.update(enum_output=self.frame+0x28,enum_state=self.frame+0x40,scratch=self.frame+0x20);self.sizes['scratch']=64;self.ids={v:n for n,v in self.p.items()}
        self.tracking=False;self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE,self.observe_write)
    def oid(self,v):return self.ids[v] if v else None
    def pointer(self,n):return self.p[n] if n else 0
    def snapshot(self):return snapshot({n:bytes(self.u.mem_read(self.p[n],s)) for n,s in self.sizes.items()},{n:self.rq(self.base+a) for n,a in self.slots.items()},self.u.mem_read(self.base+self.flag,1)[0],self.state,self.ids)
    def volatile(self):return {n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in VOL}
    def xmm(self):return [f'{self.reg(getattr(self.x,"UC_X86_REG_XMM"+str(i))):032x}' for i in range(6)]
    def write(self,n,off,size,value):
        self.u.mem_write(self.p[n]+off,(self.p[value] if isinstance(value,str) else value).to_bytes(size,'little'));self.allowed.setdefault(n,set()).update(range(off,off+size))
    def mutate(self,kind):
        count=self.completed.get(kind,0)+1;self.completed[kind]=count
        for n,off,size,v in self.options.get('mutations',{}).get(kind+':'+str(count),[]):
            if n.startswith('slot:'):self.q(self.base+self.slots[n[5:]],self.p[v]);self.slot_writes[n[5:]]=v
            else:self.write(n,off,size,v)
    def observe_write(self,uc,access,address,size,value,data):
        if not self.tracking:return
        pc=self.reg(self.x.UC_X86_REG_RIP)-self.base
        if address==self.base+self.flag:assert (pc,size,value)==(0x3B4DFB,1,1);self.flag_written=True
        elif self.owner and self.owner<=address<self.owner+512:
            assert (pc,address-self.owner,size,value)==(0x3B4E0B,0xC0,8,0);self.allowed.setdefault(self.oid(self.owner),set()).update(range(0xC0,0xC8))
        elif self.p['scratch']<=address<self.p['scratch']+64:
            off,total={0x3B4E48:(0x20,16),0x3B4E53:(0x30,8),0x3B4E59:(8,8),0x3B4E63:(0x10,8)}[pc];assert (address-self.p['scratch'],size)==(off,total) or (total==16 and (address-self.p['scratch'],size) in [(off,8),(off+8,8)]);self.allowed.setdefault('scratch',set()).update(range(address-self.p['scratch'],address-self.p['scratch']+size))
    def event(self,kind,args):
        self.counts[kind]=self.counts.get(kind,0)+1;self.events.append(dict(kind=kind,args=args,ordinal=self.counts[kind],snapshot=self.snapshot(),raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm(),caller_return_bits=self.rq(self.reg(self.x.UC_X86_REG_RSP)),synthetic_cleanup=self.cleanup))
        if self.options.get('failure')==[kind,self.counts[kind]]:self.error=kind;self.u.emu_stop();return False
        return True
    def finish(self,kind,args,value):
        self.state['history'].append(dict(kind=kind,args=deepcopy(args)));self.mutate(kind)
        for n in VOL[1:]:self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),POISON)
        for i in range(6):self.u.reg_write(getattr(self.x,'UC_X86_REG_XMM'+str(i)),(1<<127)|i)
        super().ret(value)
    def series(self,key,i,default):return self.options.get(key,[])[i] if i<len(self.options.get(key,[])) else default
    def equal(self,a,b):return a==b if not a or not b else bytes(self.u.mem_read(a+0x14,self.rd(a+0x10)*2))==bytes(self.u.mem_read(b+0x14,self.rd(b+0x10)*2))
    def prepare(self,options):
        self.options=deepcopy(options);self.events=[];self.counts={};self.error=None;self.state=dict(entries=[],history=[],iterator=None)
        for n,s in self.sizes.items():self.u.mem_write(self.p[n],bytes([options.get('seed_byte',0xA5)])*s)
        for i,n in enumerate(MI_NAMES):self.q(self.base+self.slots[n],self.p[f'mi{i}'])
        self.u.mem_write(self.base+self.flag,bytes([options.get('warm_byte',0)]))
        for n in ['owner','other_owner']:self.q(self.p[n],self.p['data_class']);self.q(self.p[n]+0x18,self.pointer(options.get('owner_id','target_id')));self.q(self.p[n]+0xC0,self.p['old_skin'])
        for n,text in [('target_id','target'),('other_id','other'),('skin_id0','first'),('skin_id1','second')]:self.q(self.p[n],self.p['string_class']);self.d(self.p[n]+0x10,len(text));self.u.mem_write(self.p[n]+0x14,text.encode('utf-16-le')+b'\0\0')
        for i in range(3):self.q(self.p[f'pref{i}']+0x10,self.pointer(None if options.get('null_pref_id') else 'other_id' if i==1 else 'target_id'));self.q(self.p[f'pref{i}']+0x18,self.pointer(None if options.get('null_skin_id') else 'skin_id1' if i==2 else 'skin_id0'))
        for i in range(2):
            entries=options.get('entries' if i==0 else 'other_entries',['pref0','pref1','pref2'] if i==0 else ['pref1','pref0']);assert len(entries)<=4
            self.q(self.p[f'saved{i}']+0x10,self.pointer(None if options.get('null_list') else f'list{i}'));self.q(self.p[f'list{i}']+0x10,self.p[f'array{i}']);self.d(self.p[f'list{i}']+0x18,len(entries));self.d(self.p[f'list{i}']+0x1C,0x12345678+i);self.q(self.p[f'array{i}']+0x18,len(entries))
            for j,n in enumerate(entries):self.q(self.p[f'array{i}']+0x20+j*8,self.pointer(n))
        self.u.mem_write(self.p['clone_enum'],self.p['list1'].to_bytes(8,'little')+(2).to_bytes(4,'little')+(0xDEADBEEF).to_bytes(4,'little')+self.p['pref1'].to_bytes(8,'little'))
        if options.get('cleanup'):self.q(self.frame+0x28,self.pointer('exception' if options.get('exception_live') else None));self.q(self.frame+0x30,self.p[options.get('cleanup_enum','enum_state')]);self.u.mem_write(self.p['enum_state'],bytes(self.u.mem_read(self.p['clone_enum'],24)))
    def hook(self,uc,address,size,data):
        rva=address-self.base;self.executed.add(rva);x=self.x
        if rva==(CLEANUP if self.cleanup else ENTRY):self.state['entries'].append(dict(owner=self.oid(self.owner),synthetic_cleanup=self.cleanup,volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm()))
        if rva in self.instructions:return
        cx,dx,r8=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8']]
        if rva==0x2B7B40:
            n=self.slot_names[cx-self.base];v=self.rq(cx);args=[n,self.oid(v)]
            if self.event('metadata',args):self.finish('metadata',args,v)
        elif rva==0x2B6FF0:
            assert cx==self.owner+0xC0 and dx==self.rq(cx)==0;args=[self.oid(self.owner),0xC0,None]
            if self.event('reference_barrier',args):self.finish('reference_barrier',args,self.options.get('barrier_return_bits',0xBADD123456789056))
        elif rva==0x387870:
            assert cx==0;v=self.pointer(self.options.get('saved_result','saved0'));args=[0,self.oid(v)]
            if self.event('character_preferences',args):self.finish('character_preferences',args,v)
        elif rva==0xB16640:
            assert cx==self.p['enum_output'] and self.oid(dx) in ['list0','list1'] and self.oid(r8) in ['mi3','other_mi3'];n=self.oid(dx);count=self.rd(dx+0x18);assert count<=4;array=self.rq(dx+0x10);entries=[self.oid(self.rq(array+0x20+i*8)) for i in range(count)];raw=dx.to_bytes(8,'little')+bytes(4)+self.rd(dx+0x1C).to_bytes(4,'little')+bytes(8);args=[self.oid(cx),n,self.oid(r8),raw.hex(),entries]
            if self.event('get_enumerator',args):
                self.state['iterator']=dict(list=n,entries=entries,cursor=0)
                for off in range(0,24,8):self.write('scratch',8+off,8,int.from_bytes(raw[off:off+8],'little'))
                self.finish('get_enumerator',args,self.options.get('get_return_bits',self.p['enum_output']))
        elif rva==0x9693D0:
            assert cx==self.p['enum_state'] and self.oid(dx) in ['mi1','other_mi1'];it=self.state['iterator'];i=it['cursor'];has=i<len(it['entries']);current=it['entries'][i] if has else None;v=self.series('move_next_raw',i,0xBEEF123456789000|int(has));args=[self.oid(cx),self.oid(dx),current,v]
            if self.event('move_next',args):it['cursor']+=1;self.write('scratch',0x28,4,i+1);self.write('scratch',0x30,8,current if current else 0);self.finish('move_next',args,v)
        elif rva==0xF73E00:
            assert r8==0;i=self.completed.get('string_equality',0);v=self.series('equality_raw',i,0xCAFE123456789000|int(self.equal(cx,dx)));args=[self.oid(cx),self.oid(dx),0,v]
            if self.event('string_equality',args):self.finish('string_equality',args,v)
        elif rva==NEXT:
            assert cx==self.owner and r8==0;args=[self.oid(cx),self.oid(dx),0]
            if self.event('load_skin',args):self.finish('load_skin',args,self.options.get('load_skin_return_bits',0x10AD123456789098))
        elif rva==0x33ED50:
            assert self.oid(cx) in ['enum_state','clone_enum'] and self.oid(dx) in ['mi0','other_mi0'];args=[self.oid(cx),self.oid(dx)]
            if self.event('dispose',args):self.finish('dispose',args,self.options.get('dispose_return_bits',0xD15E1234567890AB))
        elif rva in [0x2B7D90,0x246610]:
            kind='native_guard' if rva==0x2B7D90 else 'rethrow';caller=self.rq(self.reg(x.UC_X86_REG_RSP));site=caller-self.base-5;assert site in [0x3B4EF0,0x3B4EFC] if kind=='native_guard' else site==0x3B4EF6;args=[hex(site)] if kind=='native_guard' else [self.oid(cx)]
            if self.event(kind,args):self.error=kind;self.u.emu_stop()
        else:raise AssertionError(f'unclaimed native {rva:x}')
    def run(self,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options=deepcopy(options or {});self.error=None;self.counts={}
        self.owner=self.pointer(self.options.get('owner','owner'));self.cleanup=self.options.get('cleanup',False);self.allowed,self.slot_writes,self.flag_written,self.completed={},{},False,{};x=self.x
        incoming={n:self.options.get('entry_'+n.lower()+'_bits',0xDEAD123400000000+i) for i,n in enumerate(VOL)};incoming['RCX']=self.owner;initial_xmm=[f'{((1<<126)|i):032x}' for i in range(6)]
        for n,v in incoming.items():self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        for i,v in enumerate(initial_xmm):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),int(v,16))
        for i,n in enumerate(NONVOL):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),(1<<125)|i)
        self.q(self.entry_sp,self.stop)
        if self.cleanup:
            for off,n in [(0x60,'RDI'),(0x70,'RBX'),(0x78,'RSI')]:self.q(self.frame+off,0xFAB0000000000000+NONVOL.index(n))
        self.u.reg_write(x.UC_X86_REG_RSP,self.frame if self.cleanup else self.entry_sp);initial=self.snapshot();old=len(self.events);self.tracking=True;fault=None
        try:self.u.emu_start(self.base+(CLEANUP if self.cleanup else ENTRY),self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:assert self.owner==0 and not self.cleanup and exc.errno==self.unicorn.UC_ERR_WRITE_UNMAPPED and self.reg(x.UC_X86_REG_RIP)-self.base==0x3B4E0B;self.error='native_owner_write_fault';fault='0x3b4e0b'
        finally:self.tracking=False
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==self.entry_sp+8;assert all(self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB0000000000000+i for i,n in enumerate(NONVOL));assert all(self.reg(getattr(x,'UC_X86_REG_XMM'+str(i)))==(1<<125)|i for i in range(6,16))
        final=self.snapshot()
        for n,h in initial['memory'].items():
            a,b=bytes.fromhex(h),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or v==b[i] for i,v in enumerate(a)),n
        assert final['metadata_slots']=={**initial['metadata_slots'],**self.slot_writes};assert final['metadata_flag']==(1 if self.flag_written else initial['metadata_flag'])
        row=dict(options=self.options,synthetic_cleanup=self.cleanup,initial=initial,events=deepcopy(self.events[old:]),final=final,returned=returned,error=self.error,fault_rva=fault,entry_volatile_registers=incoming,entry_volatile_xmm_hex=initial_xmm,final_volatile_registers=self.volatile(),final_volatile_xmm_hex=self.xmm(),return_bits=self.reg(x.UC_X86_REG_RAX) if returned else None,normal_or_synthetic_frame_abi_verified=returned,completed_memory_write_offsets={n:sorted(v) for n,v in self.allowed.items()},completed_slot_writes=self.slot_writes.copy(),reached_native_flag_write=self.flag_written,unrelated_storage_retained=True)
        verify(row,self.p,self.ids,self.slots,self.base);row['independent_ordered_full_state_verified']=True;return row

def verify(row,p,ids,slot_addresses,base):
    raw={n:bytearray.fromhex(v) for n,v in row['initial']['memory'].items()};slots={n:p[v] for n,v in row['initial']['metadata_slots'].items()};flag=row['initial']['metadata_flag'];state=deepcopy(row['initial']['supplied_state']);o=row['options'];regs=row['entry_volatile_registers'].copy();xmm=row['entry_volatile_xmm_hex'].copy();owner=ids[regs['RCX']] if regs['RCX'] else None;cleanup=row['synthetic_cleanup'];events=[];counts={};completed={};returned=False;error=None;fault=None;result=None
    state['entries'].append(dict(owner=owner,synthetic_cleanup=cleanup,volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy()))
    def oid(v):return ids[v] if v else None
    def q(n,off):return int.from_bytes(raw[n][off:off+8],'little')
    def rd(n,off):return int.from_bytes(raw[n][off:off+4],'little')
    def write(n,off,size,v):raw[n][off:off+size]=(p[v] if isinstance(v,str) else v).to_bytes(size,'little')
    def mutate(kind):
        count=completed.get(kind,0)+1;completed[kind]=count
        for n,off,size,v in o.get('mutations',{}).get(kind+':'+str(count),[]):
            if n.startswith('slot:'):slots[n[5:]]=p[v]
            else:write(n,off,size,v)
    def series(k,i,d):return o.get(k,[])[i] if i<len(o.get(k,[])) else d
    def equal(a,b):return a==b if not a or not b else raw[oid(a)][0x14:0x14+rd(oid(a),0x10)*2]==raw[oid(b)][0x14:0x14+rd(oid(b),0x10)*2]
    class Stopped(Exception):pass
    def emit(kind,args,site,value=None,effect=None):
        counts[kind]=counts.get(kind,0)+1;events.append(dict(kind=kind,args=args,ordinal=counts[kind],snapshot=snapshot(raw,slots,flag,state,ids),raw_args=[regs[n] for n in ['RCX','RDX','R8','R9']],volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),caller_return_bits=base+site+5,synthetic_cleanup=cleanup))
        if o.get('failure')==[kind,counts[kind]] or kind in ['native_guard','rethrow']:raise Stopped
        if effect:effect()
        state['history'].append(dict(kind=kind,args=deepcopy(args)));mutate(kind);regs.update({n:POISON for n in VOL[1:]});regs['RAX']=value;xmm[:]=[f'{((1<<127)|i):032x}' for i in range(6)]
    def dispose(site,receiver):regs['RCX']=receiver;regs['RDX']=slots[MI_NAMES[0]];emit('dispose',[oid(receiver),oid(regs['RDX'])],site,o.get('dispose_return_bits',0xD15E1234567890AB))
    try:
        if cleanup:
            dispose(0x3B4ED1,q('scratch',0x10));regs['RCX']=q('scratch',8)
            if regs['RCX']:emit('rethrow',[oid(regs['RCX'])],0x3B4EF6)
        else:
            if flag==0:
                for n,site in zip(MI_NAMES,META_SITES):regs['RCX']=base+slot_addresses[n];v=slots[n];emit('metadata',[n,oid(v)],site,v)
                flag=1
            regs['RCX']=(p[owner] if owner else 0)+0xC0
            if owner is None:error='native_owner_write_fault';fault='0x3b4e0b';raise Stopped
            write(owner,0xC0,8,0);regs['RDX']=0;emit('reference_barrier',[owner,0xC0,None],0x3B4E10,o.get('barrier_return_bits',0xBADD123456789056))
            regs['RCX']=0;v=p[o.get('saved_result','saved0')] if o.get('saved_result','saved0') else 0;emit('character_preferences',[0,oid(v)],0x3B4E17,v)
            if not regs['RAX']:emit('native_guard',['0x3b4efc'],0x3B4EFC)
            regs['RDX']=q(oid(regs['RAX']),0x10)
            if not regs['RDX']:emit('native_guard',['0x3b4efc'],0x3B4EFC)
            regs['RCX']=p['enum_output'];regs['R8']=slots[MI_NAMES[3]];n=oid(regs['RDX']);entries=[oid(q(oid(q(n,0x10)),0x20+i*8)) for i in range(rd(n,0x18))];produced=p[n].to_bytes(8,'little')+bytes(4)+rd(n,0x1C).to_bytes(4,'little')+bytes(8);args=[oid(regs['RCX']),n,oid(regs['R8']),produced.hex(),entries]
            def make_enum():state['iterator']=dict(list=n,entries=entries,cursor=0);raw['scratch'][8:32]=produced
            emit('get_enumerator',args,0x3B4E3E,o.get('get_return_bits',p['enum_output']),make_enum);xmm[0]=f'{int.from_bytes(raw["scratch"][8:24],"little"):032x}';xmm[1]=f'{int.from_bytes(raw["scratch"][24:32],"little"):032x}';raw['scratch'][32:56]=raw['scratch'][8:32];write('scratch',8,8,0);write('scratch',0x10,8,p['enum_state'])
            while True:
                regs['RCX']=p['enum_state'];regs['RDX']=slots[MI_NAMES[1]];it=state['iterator'];i=it['cursor'];has=i<len(it['entries']);current=it['entries'][i] if has else None;v=series('move_next_raw',i,0xBEEF123456789000|int(has));args=[oid(regs['RCX']),oid(regs['RDX']),current,v]
                def move():it['cursor']+=1;write('scratch',0x28,4,i+1);write('scratch',0x30,8,current if current else 0)
                emit('move_next',args,0x3B4E7C,v,move)
                if not regs['RAX']&0xFF:break
                captured=q('scratch',0x30)
                if not captured:emit('native_guard',['0x3b4ef0'],0x3B4EF0)
                regs['R8']=0;regs['RDX']=q(owner,0x18);regs['RCX']=q(oid(captured),0x10);i=completed.get('string_equality',0);v=series('equality_raw',i,0xCAFE123456789000|int(equal(regs['RCX'],regs['RDX'])));emit('string_equality',[oid(regs['RCX']),oid(regs['RDX']),0,v],0x3B4E9A,v)
                if not regs['RAX']&0xFF:continue
                regs['R8']=0;regs['RDX']=q(oid(captured),0x18);regs['RCX']=p[owner];emit('load_skin',[owner,oid(regs['RDX']),0],0x3B4EAD,o.get('load_skin_return_bits',0x10AD123456789098))
            dispose(0x3B4EBE,p['enum_state'])
        returned=True;result=regs['RAX']
    except Stopped:
        if error is None:error=events[-1]['kind']
    assert (row['returned'],row['error'],row['fault_rva'],row['return_bits'])==(returned,error,fault,result),o
    assert row['events']==events,('events',o);assert row['final']==snapshot(raw,slots,flag,state,ids),('final',o);assert row['final_volatile_registers']==regs and row['final_volatile_xmm_hex']==xmm,('ABI',o)

def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];seqs=[];baselines=[];stops=[]
    profiles=[[],['pref0'],['pref1'],['pref0','pref1','pref2'],['pref2','pref0'],['pref0','pref0'],[None],['pref1',None],['pref0',None]]
    for warm,seed,owner,entries in itertools.product([0,1,0x80],[0,0xA5],['owner','other_owner'],profiles):cases.append(m.run(dict(warm_byte=warm,seed_byte=seed,owner=owner,entries=entries)))
    for o in [dict(owner=None),dict(saved_result=None),dict(null_list=True),dict(owner_id=None),dict(null_pref_id=True),dict(owner_id=None,null_pref_id=True),dict(null_skin_id=True),dict(saved_result='saved1'),dict(equality_raw=[0xFFFFFFFFFFFFFF00]*3),dict(equality_raw=[0x8000000000000080]*3),dict(move_next_raw=[0x123456789ABCDE00]),dict(move_next_raw=[0x80000000000000FF,0]),dict(get_return_bits=0),dict(dispose_return_bits=0),dict(load_skin_return_bits=0xFFFFFFFFFFFFFFFF)]:cases.append(m.run(o))
    plans=[('metadata:1',[['saved0',0x10,8,'list1']]),('metadata:4',[['slot:'+MI_NAMES[3],0,8,'other_mi3']]),('reference_barrier:1',[['owner',0xC0,8,'skin0']]),('character_preferences:1',[['saved0',0x10,8,'list1']]),('get_enumerator:1',[['saved0',0x10,8,'list1'],['slot:'+MI_NAMES[1],0,8,'other_mi1']]),('get_enumerator:1',[['scratch',8,8,'list1']]),('move_next:1',[['scratch',0x30,8,'pref1']]),('move_next:1',[['scratch',0x30,8,0]]),('string_equality:1',[['scratch',0x30,8,'pref1'],['pref0',0x18,8,'skin_id1']]),('string_equality:1',[['owner',0x18,8,'other_id']]),('load_skin:1',[['owner',0x18,8,'other_id'],['owner',0xC0,8,'skin1']]),('load_skin:1',[['slot:'+MI_NAMES[0],0,8,'other_mi0']]),('dispose:1',[['owner',0xC0,8,'skin0']])]
    for (kind,writes),warm,owner in itertools.product(plans,[0,0xFE],['owner','other_owner']):cases.append(m.run(dict(warm_byte=warm,owner=owner,mutations={kind:writes})))
    for live,receiver in itertools.product([False,True],['enum_state','clone_enum']):cases.append(m.run(dict(cleanup=True,exception_live=live,cleanup_enum=receiver)))
    for live,v in [(False,'exception'),(True,0)]:cases.append(m.run(dict(cleanup=True,exception_live=live,mutations={'dispose:1':[['scratch',8,8,v]]})))
    for o in [{},dict(owner='other_owner'),dict(entries=['pref0','pref0']),dict(mutations={'load_skin:1':[['owner',0xC0,8,'skin1']]})]:
        rows=[m.run(o),m.run(dict(o,failure=['string_equality',1]),True),m.run(o,True),m.run(dict(o,mutations={'dispose:1':[['other_owner',0xC0,8,'skin0']]}),True)];assert [r['returned'] for r in rows]==[True,False,True,True];assert all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));seqs.append(rows)
    rows=[m.run(dict(failure=['metadata',4])),m.run({},True),m.run({},True),m.run(dict(failure=['character_preferences',1]),True)]
    assert [r['returned'] for r in rows]==[False,True,True,False];assert all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));seqs.append(rows)
    for o in [{},dict(warm_byte=0xFE),dict(entries=[]),dict(entries=[None]),dict(saved_result=None),dict(null_list=True),dict(owner=None),dict(mutations={'string_equality:1':[['pref0',0x18,8,'skin_id1']]}),dict(cleanup=True),dict(cleanup=True,exception_live=True),dict(cleanup=True,cleanup_enum='clone_enum')]:
        baseline=m.run(o);bid=len(baselines);baselines.append(baseline);counts={}
        for i,e in enumerate(baseline['events']):
            k=e['kind'];counts[k]=counts.get(k,0)+1;r=m.run(dict(o,failure=[k,counts[k]]));assert not r['returned'] and r['events']==baseline['events'][:i+1] and r['final']==e['snapshot'];stops.append(dict(baseline=bid,prefix_length=i+1,result=r))
    missing=set(m.instructions)-m.executed;assert missing==set(EXCLUDED),(sorted(missing),sorted(set(EXCLUDED)-missing))
    return dict(schema='character_data_preferences_native_v1',build=BUILD,target=dict(m.target,method_id='tdi5845.m0011'),body_bounds=m.bounds,fields=m.fields,supplied_targets=m.supplied,metadata_slots={n:hex(a) for n,a in m.slots.items()},metadata_flag=hex(m.flag),diagnostic_windows_not_object_extents=m.sizes,scope='Exact LoadPreferences caller; SavedCharacters getter/iterator/string/LoadSkin/metadata/barrier/guards/rethrow wholly supplied. Synthetic cleanup frame only, no managed unwind or persistence.',instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),excluded_instruction_reasons={hex(a):v for a,v in EXCLUDED.items()},cases=cases,retained_sequences=seqs,baselines=baselines,failure_stops=stops,summary=dict(cases=len(cases),sequences=len(seqs),baselines=len(baselines),stops=len(stops)))

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--game-root',required=True);parser.add_argument('--dumper-root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    raw=audit(args.game_root,args.dumper_root);report=pool_snapshots(pool_memory(raw));from audit_report_snapshots import expand_snapshots;from audit_character_oracle_reveal_join import expand_memory
    assert expand_memory(expand_snapshots(report))==raw
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps(report['summary'],sort_keys=True))
