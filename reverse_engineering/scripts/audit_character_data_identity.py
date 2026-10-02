"""Exact offline CharacterData.GenerateCharacterId caller; all services supplied."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_oracle_reveal_join import pool_memory, expand_memory
from audit_report_snapshots import pool_snapshots, expand_snapshots

ENTRY, END, FOLLOWING = 0x3B4520, 0x3B4982, 0x3B4990
CHUNKS = [(ENTRY,0x3B4581,0),(0x3B4581,0x3B48DF,4),(0x3B48DF,0x3B48E6,4),(0x3B48E6,END,4)]
VOL = ['RAX','RCX','RDX','R8','R9','R10','R11']
NONVOL = ['RBX','RBP','RSI','RDI','R12','R13','R14','R15']
POISON = 0xFACE123456789090
SLOTS = {'int_TypeInfo':0x2707130,'object[]_TypeInfo':0x2720110,'format_literal':0x26EEE60}
META = [0x3B453A,0x3B4546,0x3B4552]
RNG = [0x3B45E3,0x3B463F,0x3B469B,0x3B46F7,0x3B4753,0x3B47AF,0x3B480B,0x3B4867]
BOX = [0x3B45F8,0x3B4654,0x3B46B0,0x3B470C,0x3B4768,0x3B47C4,0x3B4820,0x3B487C]
SCRATCH = [0x60,0x70,0x78,0x20,0x24,0x28,0x2C,0x30]
CAST = [0x3B45B3,0x3B460F,0x3B466B,0x3B46C7,0x3B4723,0x3B477F,0x3B47DB,0x3B4837,0x3B4893]
GATES = [0x3B45C1,0x3B461D,0x3B4679,0x3B46D5,0x3B4731,0x3B478D,0x3B47E9,0x3B4845,0x3B48A1]
STORES = [0x3B45D2,0x3B462E,0x3B468A,0x3B46E6,0x3B4742,0x3B479E,0x3B47FA,0x3B4856,0x3B48B2]
BARRIERS = [0x3B45D5,0x3B4631,0x3B468D,0x3B46E9,0x3B4745,0x3B47A1,0x3B47FD,0x3B4859,0x3B48B5,0x3B48D5]
MAKE = [0x3B48EC+i*16 for i in range(9)]
RAISE = [x+10 for x in MAKE]
TRAPS = [0x3B48EB]+[x+15 for x in MAKE]+[0x3B4981]
LITERAL = '{0}_{1}{2}{3}{4}{5}{6}{7}{8}'
SERVICES = ['metadata','is_empty','array_allocate','object_name','random','box','type_check','reference_barrier','format','make_exception','raise_exception','null_exception','bounds_exception']


def snapshot(raw, slots, flag, state, ids):
    def q(n,o):return int.from_bytes(raw[n][o:o+8],'little')
    def oid(v):return ids[v] if v else None
    return dict(memory={n:bytes(v).hex() for n,v in raw.items()},metadata_slots={n:oid(v) for n,v in slots.items()},metadata_flag=flag,
                owners={n:oid(q(n,0x18)) for n in ['owner','other_owner']},
                arrays={n:dict(class_identity=oid(q(n,0)),length_bits=q(n,0x18),items=[oid(q(n,0x20+i*8)) for i in range(9)]) for n in ['array','other_array']},
                supplied_state=deepcopy(state))


class Machine(NativeMachine):
    def __init__(self,game_root,dumper_root):
        import capstone
        super().__init__(game_root)
        build=json.loads((Path(__file__).parents[1]/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
        metadata_input=build['inputs']['global_metadata'];raw_metadata=(Path(game_root)/metadata_input['path']).read_bytes()
        assert len(raw_metadata)==metadata_input['size'] and hashlib.sha256(raw_metadata).hexdigest().upper()==metadata_input['sha256'].upper()
        extraction=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name,key):
            raw=(Path(dumper_root)/name).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==extraction['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata=json.loads(pin('script.json','script_json'));dump=pin('dump.cs','dump_cs');header=pin('il2cpp.h','il2cpp_h')
        rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==ENTRY];assert len(rows)==1
        self.target=rows[0];assert self.target==dict(Address=ENTRY,Name='CharacterData$$GenerateCharacterId',Signature='void CharacterData__GenerateCharacterId (CharacterData_o* __this, const MethodInfo* method);',TypeSignature='vii')
        assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address']>ENTRY)==FOLLOWING
        methods=Path(__file__).parents[1]/f'coverage/{BUILD}/methods.v1.jsonl'
        binding=[json.loads(s) for s in methods.read_text(encoding='utf-8').splitlines() if '"id":"tdi5845.m0010"' in s]
        assert len(binding)==1 and binding[0]['symbol_key']=='CharacterData::public void GenerateCharacterId()' and binding[0]['native']==[dict(body='ga:rva:003B4520',binding='direct')]
        chunks=[]
        for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root=e
            while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
            if root.struct.BeginAddress==ENTRY:chunks.append((e.struct.BeginAddress,e.struct.EndAddress,e.unwindinfo.Flags))
        assert chunks==CHUNKS
        section=self.pe.get_section_by_rva(ENTRY);assert section and FOLLOWING<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(ENTRY,FOLLOWING-ENTRY);assert len(raw)==FOLLOWING-ENTRY and raw[END-ENTRY:]==b'\xcc'*14
        ins=list(self.cs.disasm(raw[:END-ENTRY],ENTRY));assert len(ins)==290 and sum(i.size for i in ins)==END-ENTRY
        self.instructions={i.address:i for i in ins}
        declaration=re.search(r'^public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845\s*\{(.*?)\n\}',dump,re.M|re.S);assert declaration
        assert 'public string characterId; // 0x18' in declaration[1] and '// RVA: 0x3B4520 Offset: 0x3B3120 VA: 0x1803B4520\n\tpublic void GenerateCharacterId()' in declaration[1]
        for text in ['struct System_Int32_Fields {\n\tint32_t m_value;','struct System_Object_array {\n\tIl2CppObject obj;\n\tIl2CppArrayBounds *bounds;\n\til2cpp_array_size_t max_length;\n\tIl2CppObject* m_Items[65535];','Il2CppClass* element_class;']:
            assert text in header
        self.flag=0x288C4DC
        checks={0x3B4527:('cmp','byte ptr [rip + 0x24d7fae], 0'),0x3B452E:('mov','rdi, rcx'),
                0x3B4557:('mov','byte ptr [rip + 0x24d7f7e], 1'),0x3B455E:('mov','rcx, qword ptr [rdi + 0x18]'),
                0x3B4562:('lea','rsi, [rdi + 0x18]'),0x3B4566:('xor','edx, edx'),0x3B456D:('test','al, al'),
                0x3B457C:('mov','edx, 9'),0x3B4581:('mov','qword ptr [rsp + 0x40], rbx'),
                0x3B458B:('xor','edx, edx'),0x3B458D:('mov','rcx, rdi'),0x3B4590:('mov','rbx, rax'),0x3B4598:('mov','rdi, rax'),
                0x3B48C1:('xor','r8d, r8d'),0x3B48C4:('mov','rdx, rbx'),0x3B48CC:('mov','rdx, rax'),
                0x3B48CF:('mov','qword ptr [rsi], rax'),0x3B48D2:('mov','rcx, rsi'),0x3B48DA:('mov','rbx, qword ptr [rsp + 0x40]'),
                0x3B48E5:('ret','')}
        for j in range(9):
            checks[GATES[j]]=('cmp',f'dword ptr [rbx + 0x18], {j}')
            checks[GATES[j]+4]=('jbe','0x3b497c')
            checks[STORES[j]]=('mov','qword ptr [rcx], rdi')
            checks[CAST[j]-4]=('mov','rdx, qword ptr [rdx + 0x40]')
            checks[CAST[j]+5]=('test','rax, rax')
        for j,site in enumerate(BOX):
            checks[site-4]=('mov',f'dword ptr [rsp + {hex(SCRATCH[j])}], eax')
            checks[site+5]=('mov','rdi, rax')
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in checks.items());self.checks=checks
        gateway_sites={0x2B7B40:META,0xF76390:[0x3B4568],0x2B7080:[0x3B4586],0x1C82250:[0x3B4593],0x1C86600:RNG,0x282580:BOX,
                       0x2B7010:CAST,0x2B6FF0:BARRIERS,0xF74B10:[0x3B48C7],0x2B78A0:MAKE,0x2B7D50:RAISE,0x2B7D90:[0x3B48E6],0x2B7D80:[0x3B497C]}
        for gateway,sites in gateway_sites.items():assert [i.address for i in ins if i.mnemonic=='call' and i.op_str==hex(gateway)]==sites
        self.gateway_sites=gateway_sites
        refs={i.address+i.size+op.mem.disp for i in ins for op in i.operands if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP}
        for n,a in SLOTS.items():
            category='ScriptString' if n=='format_literal' else 'ScriptMetadata';rows=[r for r in self.metadata[category] if r['Address']==a];assert len(rows)==1 and a in refs
            assert rows[0].get('Value')==LITERAL if n=='format_literal' else rows[0]['Name']==n
        assert refs==set(SLOTS.values())|{self.flag}
        for n,site in zip(SLOTS,META):
            i=self.instructions[site-7];assert i.mnemonic=='lea' and i.address+i.size+i.operands[1].mem.disp==SLOTS[n]
        self.supplied=[]
        for address,name,signature,types in [
            (0xF76390,'System.String$$IsNullOrEmpty','bool System_String__IsNullOrEmpty (System_String_o* value, const MethodInfo* method);','iii'),
            (0xF74B10,'System.String$$Format','System_String_o* System_String__Format (System_String_o* format, System_Object_array* args, const MethodInfo* method);','iiii'),
            (0x1C82250,'UnityEngine.Object$$get_name','System_String_o* UnityEngine_Object__get_name (UnityEngine_Object_o* __this, const MethodInfo* method);','iii'),
            (0x1C86600,'UnityEngine.Random$$Range','int32_t UnityEngine_Random__Range (int32_t minInclusive, int32_t maxExclusive, const MethodInfo* method);','iiii')]:
            rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==address and r['Name']==name];assert len(rows)==1 and rows[0]['Signature']==signature and rows[0]['TypeSignature']==types;self.supplied+=rows
        names=['owner','other_owner','data_class','array_class','other_array_class','element_class','other_element_class','int_class','other_int_class','object_class',
               'array','other_array','old_id','other_id','name','other_name','format_literal','other_literal','formatted','other_formatted','exception']+[f'box{i}' for i in range(8)]+['other_box']
        self.p={n:self.arena+0x380000+i*0x1000 for i,n in enumerate(names)};self.ids={v:n for n,v in self.p.items()}
        self.sizes={n:0x200 if n in ['owner','other_owner'] else 0x100 if n.endswith('class') else 0x80 for n in names}
        self.sp=self.stack+0x18008;self.frame=self.sp-0x58;self.stack_begin=self.sp-0x68;self.sizes['stack_window']=0x98
        self.encountered_traps=set()
        self.tracking=False;self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE,self.observe_write)

    def oid(self,v):return self.ids[v] if v else None
    def volatile(self):return {n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in VOL}
    def xmm(self):return [f'{self.reg(getattr(self.x,"UC_X86_REG_XMM"+str(i))):032x}' for i in range(6)]
    def snapshot(self):
        raw={n:bytes(self.u.mem_read(p,self.sizes[n])) for n,p in self.p.items()};raw['stack_window']=bytes(self.u.mem_read(self.stack_begin,self.sizes['stack_window']))
        return snapshot(raw,{n:self.rq(self.base+a) for n,a in SLOTS.items()},self.u.mem_read(self.base+self.flag,1)[0],self.state,self.ids)
    def effect(self,n,off,size,value):
        value=self.p[value] if isinstance(value,str) else value;assert 0<=off<=self.sizes[n]-size
        self.u.mem_write((self.stack_begin if n=='stack_window' else self.p[n])+off,value.to_bytes(size,'little'));self.allowed.setdefault(n,set()).update(range(off,off+size))
    def mutate(self,kind):
        ordinal=self.completed_counts.get(kind,0)+1;self.completed_counts[kind]=ordinal
        for n,off,size,value in self.options.get('mutations',{}).get(kind+':'+str(ordinal),[]):
            if n.startswith('slot:'):
                name=n[5:];self.q(self.base+SLOTS[name],self.p[value] if value else 0);self.slot_writes[name]=value
            else:self.effect(n,off,size,value)
    def observe_write(self,uc,access,address,size,value,user_data):
        if not self.tracking:return
        pc=self.reg(self.x.UC_X86_REG_RIP)-self.base
        if address==self.base+self.flag:
            assert (pc,size,value)==(0x3B4557,1,1);self.flag_written=True;return
        if self.stack_begin<=address<self.stack_begin+self.sizes['stack_window']:
            assert pc in self.instructions
            valid={0x3B4520:(self.sp-8,8),0x3B4522:(self.sp-16,8),0x3B4581:(self.frame+0x40,8)}
            valid.update({site-4:(self.frame+off,4) for site,off in zip(BOX,SCRATCH)})
            assert (self.instructions[pc].mnemonic=='call' and address==self.frame-8 and size==8) or valid.get(pc)==(address,size)
            self.allowed.setdefault('stack_window',set()).update(range(address-self.stack_begin,address-self.stack_begin+size));return
        for n,p in self.p.items():
            if p<=address<p+self.sizes[n]:
                if pc in STORES:
                    j=STORES.index(pc);assert n==self.oid(self.array) and address==self.array+0x20+j*8 and size==8
                else:assert pc==0x3B48CF and address==self.owner+0x18 and size==8
                self.allowed.setdefault(n,set()).update(range(address-p,address-p+size));return
        raise AssertionError(('untracked native write',hex(pc),hex(address),size))
    def prepare(self,o):
        self.state={n:[] for n in ['entries']+SERVICES};self.events=[]
        for n,p in self.p.items():self.u.mem_write(p,bytes([o.get('seed_byte',0xA5)])*self.sizes[n])
        self.u.mem_write(self.stack_begin,bytes([0xA6])*self.sizes['stack_window']);self.q(self.sp,self.stop)
        for n in ['owner','other_owner']:self.q(self.p[n],self.p['data_class']);self.q(self.p[n]+0x18,self.p[o.get('initial_id','old_id')] if o.get('initial_id','old_id') else 0)
        for n in ['array','other_array']:
            self.q(self.p[n],self.p['array_class']);self.q(self.p[n]+0x10,0);self.q(self.p[n]+0x18,o.get('length_bits',9))
            for j in range(9):self.q(self.p[n]+0x20+j*8,0)
        for n in ['array_class','other_array_class']:self.q(self.p[n]+0x40,self.p['element_class'])
        for n,p in self.p.items():
            if n not in ['owner','other_owner','array','other_array'] and not n.endswith('class'):self.q(p,self.p['object_class'])
        for n,value in [('int_TypeInfo','int_class'),('object[]_TypeInfo','array_class'),('format_literal','format_literal')]:self.q(self.base+SLOTS[n],self.p[value])
        self.u.mem_write(self.base+self.flag,bytes([o.get('warm_byte',0)]))
    def ret(self,value=0):
        for n in VOL[1:]:self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),POISON)
        for i in range(6):self.u.reg_write(getattr(self.x,'UC_X86_REG_XMM'+str(i)),(1<<127)|i)
        super().ret(value)
    def event(self,kind,args):
        okay=super().event(kind,args);caller=self.rq(self.reg(self.x.UC_X86_REG_RSP))
        self.events[-1].update(raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm(),
                              caller=hex(caller-self.base),caller_kind='native_return',native_phase='CharacterData.GenerateCharacterId',rsp_bits=self.reg(self.x.UC_X86_REG_RSP))
        return okay
    def complete(self,kind,args,value,terminal=False):
        if self.event(kind,args):
            self.state[kind].append(args);self.mutate(kind)
            if terminal and not self.options.get('exception_service_returns'):
                self.error=kind;self.u.emu_stop()
            else:self.ret(value)
    def hook(self,uc,address,size,user_data):
        rva=address-self.base;x=self.x
        if rva==ENTRY:self.state['entries'].append(dict(owner=self.oid(self.owner),volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm()))
        if rva in TRAPS:
            self.encountered_traps.add(rva)
            self.error='native_int3';self.trap=hex(rva);self.u.emu_stop();return
        self.executed.add(rva)
        if rva in self.instructions:return
        cx,dx,r8,r9=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']];caller=self.rq(self.reg(x.UC_X86_REG_RSP))-self.base;site=caller-5
        if rva==0x2B7B40:
            n=next(n for n,a in SLOTS.items() if self.base+a==cx);value=self.rq(cx);self.complete('metadata',[n,self.oid(value)],value)
        elif rva==0xF76390:
            assert dx==0;self.complete('is_empty',[self.oid(cx),0],self.options.get('empty_return_bits',0xAABBCCDDEE000001))
        elif rva==0x2B7080:
            assert dx==9;self.array=self.p[self.options.get('array_result','array')] if self.options.get('array_result','array') else 0
            self.complete('array_allocate',[self.oid(cx),9,self.oid(self.array)],self.array)
        elif rva==0x1C82250:
            assert cx==self.owner and dx==0;v=self.options.get('name_result','name');self.complete('object_name',[self.oid(cx),0,v],self.p[v] if v else 0)
        elif rva==0x1C86600:
            j=RNG.index(site);assert (cx,dx,r8)==(0,10,0);value=self.options.get('random_return_bits',[0xFFFFFFFF00000000+i for i in range(8)])[j]
            self.complete('random',[j,0,10,0,value],value)
        elif rva==0x282580:
            j=BOX.index(site);assert dx==self.frame+SCRATCH[j];bits=self.rd(dx);v=self.options.get('box_results',[f'box{i}' for i in range(8)])[j]
            self.complete('box',[j,self.oid(cx),dx,bits,v],self.p[v] if v else 0)
        elif rva==0x2B7010:
            j=CAST.index(site);assert dx==self.rq(self.rq(self.array)+0x40)
            v=self.options.get('type_results',{}).get(str(j),self.oid(cx));self.complete('type_check',[j,self.oid(cx),self.oid(dx),v],self.p[v] if v else 0)
        elif rva==0x2B6FF0:
            j=BARRIERS.index(site);n=self.oid(self.array) if j<9 else self.oid(self.owner);off=0x20+j*8 if j<9 else 0x18
            assert cx==self.p[n]+off and self.rq(cx)==dx;self.complete('reference_barrier',[j,n,off,self.oid(dx)],self.options.get('barrier_return_bits',0xBADD123456789034))
        elif rva==0xF74B10:
            assert dx==self.array and r8==0;v=self.options.get('format_result','formatted');self.complete('format',[self.oid(cx),self.oid(dx),0,v],self.p[v] if v else 0)
        elif rva==0x2B78A0:
            j=MAKE.index(site);v=self.options.get('exception_result','exception');self.complete('make_exception',[j,v],self.p[v] if v else 0)
        elif rva==0x2B7D50:
            j=RAISE.index(site);assert dx==0;self.complete('raise_exception',[j,self.oid(cx),0],self.options.get('exception_return_bits',0xE00D123456789000),True)
        elif rva in [0x2B7D90,0x2B7D80]:
            kind='null_exception' if rva==0x2B7D90 else 'bounds_exception';self.complete(kind,[],self.options.get('exception_return_bits',0xE00D123456789000),True)
        else:raise AssertionError(('unclaimed entry',hex(rva)))
    def run(self,options=None,retained=False):
        self.options=deepcopy(options or {})
        if not retained:self.prepare(self.options)
        self.counts={};self.completed_counts={};self.error=None;self.trap=None;self.allowed={};self.slot_writes={};self.flag_written=False;self.array=0
        self.owner=self.p[self.options.get('owner','owner')] if not self.options.get('null_owner') else 0
        initial=self.snapshot();old=len(self.events);x=self.x
        for j,n in enumerate(NONVOL):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+j)
        for j in range(6,16):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(j)),(1<<125)|j)
        incoming={n:self.options.get('entry_'+n.lower()+'_bits',0xDEAD123400000000+j) for j,n in enumerate(VOL)};incoming['RCX']=self.owner
        initial_xmm=[f'{((1<<126)|j):032x}' for j in range(6)]
        for n,v in incoming.items():self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        for j,v in enumerate(initial_xmm):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(j)),int(v,16))
        self.u.reg_write(x.UC_X86_REG_RSP,self.sp);self.tracking=True;fault=None
        try:self.u.emu_start(self.base+ENTRY,self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base
            assert exc.errno==self.unicorn.UC_ERR_READ_UNMAPPED and pc in [0x3B455E]+[site-4 for site in CAST]
            self.error='native_read_fault';fault=hex(pc)
        finally:self.tracking=False
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==self.sp+8
            assert all(self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB0000000000000+j for j,n in enumerate(NONVOL))
            assert all(self.reg(getattr(x,'UC_X86_REG_XMM'+str(j)))==((1<<125)|j) for j in range(6,16))
        final=self.snapshot()
        for n,b in initial['memory'].items():assert all(i in self.allowed.get(n,set()) or v==bytes.fromhex(final['memory'][n])[i] for i,v in enumerate(bytes.fromhex(b))),n
        assert final['metadata_slots']=={**initial['metadata_slots'],**self.slot_writes};assert final['metadata_flag']==(1 if self.flag_written else initial['metadata_flag'])
        row=dict(options=self.options,initial=initial,events=deepcopy(self.events[old:]),final=final,entry_volatile_registers=incoming,entry_volatile_xmm_hex=initial_xmm,
                 returned=returned,error=self.error,fault_rva=fault,trap_rva=self.trap,final_volatile_registers=self.volatile(),final_volatile_xmm_hex=self.xmm(),final_rsp_bits=self.reg(x.UC_X86_REG_RSP),
                 return_bits=self.reg(x.UC_X86_REG_RAX) if returned else None,normal_abi_verified=returned,unrelated_storage_retained=True,
                 reached_write_offsets={n:sorted(v) for n,v in self.allowed.items()},completed_slot_writes=self.slot_writes.copy(),reached_flag_write=self.flag_written)
        verify(row,self.p,self.ids,self.base,self.stop,self.sp,self.stack_begin);row['independent_ordered_full_state_verified']=True
        return row


def verify(row,p,ids,base,stop,sp,stack_begin):
    raw={n:bytearray.fromhex(v) for n,v in row['initial']['memory'].items()};slots={n:p[v] if v else 0 for n,v in row['initial']['metadata_slots'].items()}
    flag=row['initial']['metadata_flag'];state=deepcopy(row['initial']['supplied_state']);o=row['options'];regs=row['entry_volatile_registers'].copy();xmm=row['entry_volatile_xmm_hex'].copy()
    owner=ids[regs['RCX']] if regs['RCX'] else None;owner_bits=regs['RCX'];frame=sp-0x58;events=[];counts={};completed={};error=None;fault=None;trap=None;returned=False;rsp=frame
    state['entries'].append(dict(owner=owner,volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy()))
    def oid(v):return ids[v] if v else None
    def q(n,off):return int.from_bytes(raw[n][off:off+8],'little')
    def write(n,off,size,v):raw[n][off:off+size]=(p[v] if isinstance(v,str) else v).to_bytes(size,'little')
    def stackwrite(address,size,v):write('stack_window',address-stack_begin,size,v)
    def mutate(kind):
        ordinal=completed.get(kind,0)+1;completed[kind]=ordinal
        for n,off,size,value in o.get('mutations',{}).get(kind+':'+str(ordinal),[]):
            if n.startswith('slot:'):slots[n[5:]]=p[value] if value else 0
            else:write(n,off,size,value)
    class Stop(Exception):pass
    def emit(kind,args,site,value,terminal=False):
        nonlocal rsp,error
        rsp=frame-8;stackwrite(rsp,8,base+site+5)
        events.append(dict(kind=kind,args=args,snapshot=snapshot(raw,slots,flag,state,ids),raw_args=[regs[n] for n in ['RCX','RDX','R8','R9']],
                           volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),caller=hex(site+5),caller_kind='native_return',native_phase='CharacterData.GenerateCharacterId',rsp_bits=rsp))
        counts[kind]=counts.get(kind,0)+1
        if o.get('failure')==[kind,counts[kind]]:error=kind;raise Stop
        state[kind].append(args);mutate(kind)
        if terminal and not o.get('exception_service_returns'):error=kind;raise Stop
        regs.update({n:POISON for n in VOL[1:]});regs['RAX']=value;xmm[:]=[f'{((1<<127)|j):032x}' for j in range(6)];rsp=frame
    def nativefault(pc):
        nonlocal error,fault
        error='native_read_fault';fault=hex(pc);raise Stop
    def native_trap(pc):
        nonlocal error,trap
        error='native_int3';trap=hex(pc);raise Stop
    stackwrite(sp-8,8,0xFAB0000000000002);stackwrite(sp-16,8,0xFAB0000000000003)
    try:
        if flag==0:
            for name,site in zip(SLOTS,META):
                regs['RCX']=base+SLOTS[name];value=slots[name];emit('metadata',[name,oid(value)],site,value)
            flag=1
        if owner is None:nativefault(0x3B455E)
        regs['RCX']=q(owner,0x18);regs['RDX']=0;value=o.get('empty_return_bits',0xAABBCCDDEE000001)
        emit('is_empty',[oid(regs['RCX']),0],0x3B4568,value)
        if value&255:
            regs['RCX']=slots['object[]_TypeInfo'];regs['RDX']=9;stackwrite(frame+0x40,8,0xFAB0000000000000)
            array=o.get('array_result','array');array_bits=p[array] if array else 0
            emit('array_allocate',[oid(regs['RCX']),9,array],0x3B4586,array_bits)
            regs['RDX']=0;regs['RCX']=owner_bits;name=o.get('name_result','name');value=p[name] if name else 0
            emit('object_name',[owner,0,name],0x3B4593,value)
            if not array:
                emit('null_exception',[],0x3B48E6,o.get('exception_return_bits',0xE00D123456789000),True);native_trap(0x3B48EB)
            for j in range(9):
                if j:
                    regs['R8']=0;regs['RCX']=0;regs['RDX']=10;value=o.get('random_return_bits',[0xFFFFFFFF00000000+i for i in range(8)])[j-1]
                    emit('random',[j-1,0,10,0,value],RNG[j-1],value)
                    regs['RCX']=slots['int_TypeInfo'];regs['RDX']=frame+SCRATCH[j-1];stackwrite(regs['RDX'],4,value&0xFFFFFFFF)
                    captured=o.get('box_results',[f'box{i}' for i in range(8)])[j-1];value=p[captured] if captured else 0
                    emit('box',[j-1,oid(regs['RCX']),regs['RDX'],int.from_bytes(raw['stack_window'][regs['RDX']-stack_begin:regs['RDX']-stack_begin+4],'little'),captured],BOX[j-1],value)
                else:captured=name
                if captured:
                    array_class=q(array,0);regs['RDX']=array_class;regs['RCX']=p[captured]
                    if not array_class:nativefault(CAST[j]-4)
                    regs['RDX']=q(oid(array_class),0x40);out=o.get('type_results',{}).get(str(j),captured);value=p[out] if out else 0
                    emit('type_check',[j,captured,oid(regs['RDX']),out],CAST[j],value)
                    if not value:
                        out=o.get('exception_result','exception');value=p[out] if out else 0;emit('make_exception',[j,out],MAKE[j],value)
                        regs['RCX']=value;regs['RDX']=0;emit('raise_exception',[j,out,0],RAISE[j],o.get('exception_return_bits',0xE00D123456789000),True);native_trap(MAKE[j]+15)
                if q(array,0x18)&0xFFFFFFFF<=j:
                    emit('bounds_exception',[],0x3B497C,o.get('exception_return_bits',0xE00D123456789000),True);native_trap(0x3B4981)
                regs['RCX']=array_bits+0x20+j*8;regs['RDX']=p[captured] if captured else 0;write(array,0x20+j*8,8,regs['RDX'])
                emit('reference_barrier',[j,array,0x20+j*8,captured],BARRIERS[j],o.get('barrier_return_bits',0xBADD123456789034))
            regs['RCX']=slots['format_literal'];regs['R8']=0;regs['RDX']=array_bits;out=o.get('format_result','formatted');value=p[out] if out else 0
            emit('format',[oid(regs['RCX']),array,0,out],0x3B48C7,value)
            regs['RDX']=value;write(owner,0x18,8,value);regs['RCX']=owner_bits+0x18
            emit('reference_barrier',[9,owner,0x18,out],BARRIERS[9],o.get('barrier_return_bits',0xBADD123456789034))
        returned=True;rsp=sp+8
    except Stop:pass
    assert (row['returned'],row['error'],row['fault_rva'],row['trap_rva'])==(returned,error,fault,trap),o
    assert row['events']==events,('events',o)
    assert row['final']==snapshot(raw,slots,flag,state,ids),('final',o)
    assert row['final_volatile_registers']==regs and row['final_volatile_xmm_hex']==xmm and row['final_rsp_bits']==rsp,('final ABI',o)
    assert row['return_bits']==(regs['RAX'] if returned else None)


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];sequences=[];baselines=[];stops=[]
    for warm,seed,owner,empty in itertools.product([0,1,0x80,0xFF],[0,0xA5],['owner','other_owner'],[0xAABBCCDDEE000001,0xAABBCCDDEE000000]):
        cases.append(m.run(dict(warm_byte=warm,seed_byte=seed,owner=owner,empty_return_bits=empty)))
    for options in [dict(initial_id=None),dict(name_result=None),dict(box_results=[None]*8),dict(array_result=None),dict(null_owner=True),
                    dict(format_result=None),dict(format_result='old_id'),dict(array_result='other_array'),dict(name_result='formatted'),
                    dict(box_results=['other_box']*8,type_results={str(i):'other_name' for i in range(9)}),
                    dict(random_return_bits=[0,9,10,0xFFFFFFFF,0x80000000,0x123456789ABCDEF0,0xFFFFFFFFFFFFFFFF,0x8000000000000000]),
                    dict(length_bits=0xFFFFFFFF00000009),dict(length_bits=0xFFFFFFFF00000000),dict(length_bits=0xFFFFFFFFFFFFFFFF),
                    dict(empty_return_bits=0xFFFFFFFFFFFFFF00),dict(empty_return_bits=0x8000000000000080),
                    dict(name_result='old_id',box_results=['old_id']*8,format_result='old_id'),
                    dict(exception_result=None,type_results={'0':None})]:cases.append(m.run(options))
    for j in range(9):
        cases.append(m.run(dict(length_bits=j)));cases.append(m.run(dict(type_results={str(j):None})))
        cases.append(m.run(dict(type_results={str(j):None},exception_service_returns=True)))
    for options in [dict(array_result=None),dict(length_bits=0)]:cases.append(m.run(dict(options,exception_service_returns=True)))
    plans=[('metadata:1',[['owner',0x18,8,'other_id']]),('metadata:3',[['slot:object[]_TypeInfo',0,8,'other_array_class']]),
           ('is_empty:1',[['owner',0x18,8,'other_id']]),('array_allocate:1',[['owner',0x18,8,'other_id'],['slot:int_TypeInfo',0,8,'other_int_class']]),
           ('object_name:1',[['array',0,8,'other_array_class'],['other_array_class',0x40,8,'other_element_class']]),
           ('type_check:1',[['array',0x18,4,0]]),('reference_barrier:1',[['array',0x20,8,'other_name'],['array_class',0x40,8,'other_element_class']]),
           ('random:1',[['slot:int_TypeInfo',0,8,'other_int_class']]),('box:1',[['array_class',0x40,8,'other_element_class']]),
           ('type_check:2',[['array',0x18,4,1]]),('reference_barrier:5',[['array',0x18,4,5]]),
           ('reference_barrier:9',[['slot:format_literal',0,8,'other_literal'],['owner',0x18,8,'other_id']]),
           ('format:1',[['owner',0x18,8,'other_id'],['array',0x60,8,'other_box']]),
           ('reference_barrier:10',[['owner',0x18,8,'other_id'],['other_owner',0x18,8,'formatted']]),
           ('object_name:1',[['array',0,8,0]]),('make_exception:1',[['owner',0x18,8,'other_id']]),('raise_exception:1',[['owner',0x18,8,'other_id']])]
    for kind,writes in plans:
        for warm in [0,0xFE]:
            o=dict(warm_byte=warm,mutations={kind:writes})
            if kind.startswith(('make_exception','raise_exception')):o['type_results']={'0':None}
            cases.append(m.run(o))
    indexed_plans=[]
    for j in range(9):
        # The type service can lower the array's current low-DWORD length before
        # the current store, and every later cast reloads the class/element root.
        indexed_plans.append(dict(mutations={f'type_check:{j+1}':[['array',0x18,4,j]]}))
        phase='object_name:1' if j==0 else f'box:{j}'
        indexed_plans.append(dict(mutations={phase:[['array',0,8,0]]}))
        cases.append(m.run(indexed_plans[-2]));cases.append(m.run(indexed_plans[-1]))
    for j in range(1,10):
        cases.append(m.run(dict(mutations={f'reference_barrier:{j}':[['array',0,8,'other_array_class'],['other_array_class',0x40,8,'other_element_class'],['other_owner',0x18,8,'other_id']]})))
    cases.append(m.run(dict(mutations={'box:1':[['slot:int_TypeInfo',0,8,'other_int_class']], 'type_check:1':[['slot:object[]_TypeInfo',0,8,'other_array_class']]})))
    for first,second in [({},dict(empty_return_bits=0xABCDEF0000000000)),({},dict(owner='other_owner',array_result='other_array')),
                         (dict(failure=['reference_barrier',5]),{}),(dict(failure=['metadata',2]),{}),
                         (dict(format_result=None),{}),(dict(mutations={'reference_barrier:10':[['owner',0x18,8,'other_id']]}),{})]:
        rows=[m.run(first),m.run(second,True),m.run(dict(second,box_results=['other_box']*8,format_result='other_formatted'),True)]
        assert all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
    profiles=[{},dict(empty_return_bits=0xFFFFFFFFFFFFFF00),dict(name_result=None),dict(box_results=[None]*8),dict(array_result=None),dict(length_bits=4),
              dict(type_results={'0':None}),dict(type_results={'8':None}),dict(type_results={'4':None},exception_service_returns=True),
              dict(mutations={'reference_barrier:9':[['slot:format_literal',0,8,'other_literal']]}),dict(null_owner=True)]
    profiles += [dict(type_results={str(j):None}) for j in range(1,8)]
    profiles += [dict(length_bits=j) for j in range(9) if j!=4]
    profiles += indexed_plans
    for o in profiles:
        baseline=m.run(o);bid=len(baselines);baselines.append(baseline);counts={}
        for index,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;stopped=m.run(dict(o,failure=[kind,counts[kind]]))
            assert not stopped['returned'] and stopped['events']==baseline['events'][:index+1] and stopped['final']==e['snapshot']
            stops.append(dict(baseline=bid,prefix_length=index+1,result=stopped))
    assert set(m.instructions)-set(TRAPS)<=m.executed
    assert m.encountered_traps==set(TRAPS)
    return dict(schema='character_data_identity_native_v1',build=BUILD,method_id='tdi5845.m0010',symbol_key='CharacterData::public void GenerateCharacterId()',target=m.target,
                scope='Exact caller only; metadata, allocation, Unity name, integer RNG, boxing, casts, format, barriers and exception services supplied whole; no runtime/Unity policy, saved IDs or aliases promoted',
                body_bounds=dict(start=hex(ENTRY),end_exclusive=hex(END),next_managed=hex(FOLLOWING),padding_bytes=14,chunks=[[hex(a),hex(b),f] for a,b,f in CHUNKS]),
                body_identity=dict(byte_length=END-ENTRY,instruction_count=len(m.instructions),sha256=hashlib.sha256(m.pe.get_data(ENTRY,END-ENTRY)).hexdigest()),
                supplied_targets=m.supplied,metadata_slots={n:hex(a) for n,a in SLOTS.items()},metadata_flag=hex(m.flag),literal=LITERAL,
                supplied_gateway_call_sites={hex(a):[hex(s) for s in sites] for a,sites in m.gateway_sites.items()},
                field_declarations=['public string characterId; // 0x18'],diagnostic_windows_not_object_extents=m.sizes,
                stack_window=dict(start_relative_to_entry_rsp=-0x68,size=0x98,frame_relative_to_entry_rsp=-0x58,box_scratch_offsets_relative_to_frame=SCRATCH),
                instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),
                excluded_trap_rvas=[hex(a) for a in TRAPS],trap_boundary_probes=[hex(a) for a in sorted(m.encountered_traps)],
                cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,
                summary=dict(cases=len(cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--game-root',required=True);parser.add_argument('--dumper-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();raw=audit(args.game_root,args.dumper_root);report=pool_snapshots(pool_memory(raw));assert expand_memory(expand_snapshots(report))==raw
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf-8')
    print(json.dumps(report['summary'],sort_keys=True))
