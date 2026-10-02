"""Actual CharacterLoc text getters; locale search and string emptiness supplied."""
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

TARGETS = {'GetIWasTranslated':(0x3F5690,0x3F56C7,0x3F56D0,3,0x20),
           'GetTranslatedName':(0x3F5920,0x3F5957,0x3F5960,2,0x18)}
VOLATILE = ['RAX','RCX','RDX','R8','R9','R10','R11']
POISON = 0xFACE123456789090


class Machine(NativeMachine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root)
        manifest=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pinned(name,key):
            raw=(Path(dumper_root)/name).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==manifest['outputs'][key]['sha256'].upper()
            return raw.decode('utf-8-sig')
        self.metadata=json.loads(pinned('script.json','script_json'));dump=pinned('dump.cs','dump_cs')
        loc=re.search(r'^public class LocaleLoc // TypeDefIndex: 5969\s*\{(.*?)\n\}',dump,re.M|re.S);assert loc
        for line in ['public string localeCode; // 0x10','public string translatedName; // 0x18','public string iWasTranslated; // 0x20','public List<CustomValuess> entries; // 0x28']:assert line in loc[1]
        owner=re.search(r'^public class CharacterLoc // TypeDefIndex: 5972\s*\{(.*?)\n\}',dump,re.M|re.S);assert owner
        self.targets=[];self.instructions={};self.bounds={};self.checks={}
        for name,(start,end,following,ordinal,field) in TARGETS.items():
            rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==start];assert len(rows)==1
            assert rows[0]['Name']=='CharacterLoc$$'+name and rows[0]['TypeSignature']=='iiii'
            assert rows[0]['Signature']==f'System_String_o* CharacterLoc__{name} (CharacterLoc_o* __this, System_String_o* localeCode, const MethodInfo* method);'
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address']>start)==following
            chunks=[]
            for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root=e
                while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
                if root.struct.BeginAddress==start:chunks.append((e.struct.BeginAddress,e.struct.EndAddress));assert e.unwindinfo.Flags==0
            assert chunks==[(start,end)]
            section=self.pe.get_section_by_rva(start);assert section and following<=section.VirtualAddress+section.SizeOfRawData
            raw=self.pe.get_data(start,following-start);assert len(raw)==following-start and raw[end-start:]==b'\xcc'*(following-end)
            ins=list(self.cs.disasm(raw[:end-start],start));assert sum(i.size for i in ins)==end-start and len(ins)==20
            self.instructions.update({i.address:i for i in ins});self.targets.append(dict(rows[0],method_id=f'tdi5972.m{ordinal:04}'))
            self.bounds[name]=dict(start=hex(start),end_exclusive=hex(end),next_managed=hex(following),unwind=[[hex(a),hex(b)] for a,b in chunks],byte_length=end-start,instructions=len(ins),sha256=hashlib.sha256(raw[:end-start]).hexdigest())
            self.checks.update({start+6:('xor','r8d, r8d'),start+9:('call','0x3f5540'),start+14:('mov','rbx, rax'),start+22:('mov',f'rcx, qword ptr [rax + {hex(field)}]'),start+26:('xor','edx, edx'),start+28:('call','0xf76390'),start+33:('test','al, al'),start+35:('jne',hex(start+47)),start+37:('mov',f'rax, qword ptr [rbx + {hex(field)}]'),start+47:('xor','eax, eax')})
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())
        self.supplied=[]
        for address,name,signature in [(0x3F5540,'CharacterLoc$$FindLocaleLoc','LocaleLoc_o* CharacterLoc__FindLocaleLoc (CharacterLoc_o* __this, System_String_o* localeCode, const MethodInfo* method);'),(0xF76390,'System.String$$IsNullOrEmpty','bool System_String__IsNullOrEmpty (System_String_o* value, const MethodInfo* method);')]:
            rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==address and r['Name']==name];assert len(rows)==1 and rows[0]['Signature']==signature
            self.supplied+=rows
        names=['owner','other_owner','owner_class','loc0','loc1','locale_class','code','other_code','name','other_name','i_was','other_i_was','replacement','entries']
        self.p={n:self.arena+0xA00000+i*0x1000 for i,n in enumerate(names)};self.ids={p:n for n,p in self.p.items()}
        self.sizes={n:256 if n.endswith('class') else 128 for n in names}

    def oid(self,bits):return self.ids[bits] if bits else None
    def pointer(self,name):return self.p[name] if name else 0
    def snapshot(self):return dict(memory={n:bytes(self.u.mem_read(p,self.sizes[n])).hex() for n,p in self.p.items()},native_entries=deepcopy(self.entries),service_history=deepcopy(self.history))
    def registers(self):return {n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in VOLATILE}
    def xmm(self):return [f'{self.reg(getattr(self.x,"UC_X86_REG_XMM"+str(i))):032x}' for i in range(6)]
    def prepare(self,options):
        self.options=deepcopy(options);self.entries=[];self.history=[];self.events=[];self.counts={}
        for n,p in self.p.items():self.u.mem_write(p,bytes([options.get('seed_byte',0xA5)])*self.sizes[n])
        for n in ['owner','other_owner']:self.q(self.p[n],self.p['owner_class'])
        for i,n in enumerate(['loc0','loc1']):
            self.q(self.p[n],self.p['locale_class'])
            for field,target in [(0x10,'code' if i==0 else 'other_code'),(0x18,'name' if i==0 else 'other_name'),(0x20,'i_was' if i==0 else 'other_i_was'),(0x28,'entries')]:
                if options.get('alias_texts') and field in [0x18,0x20]:target='code'
                self.q(self.p[n]+field,0 if options.get('null_field')==field else self.p[target])
    def mutate(self,kind,relative):
        for name,offset,target in self.options.get('mutations',{}).get(kind+':'+str(relative),[]):
            self.q(self.p[name]+offset,self.pointer(target));self.allowed.setdefault(name,set()).update(range(offset,offset+8))
    def event(self,kind,args):
        ordinal=self.counts.get(kind,0)+1;self.counts[kind]=ordinal
        self.events.append(dict(kind=kind,ordinal=ordinal,args=args,raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],volatile_registers=self.registers(),volatile_xmm_hex=self.xmm(),native_site=hex(self.last_native),caller_return_bits=self.rq(self.reg(self.x.UC_X86_REG_RSP)),snapshot=self.snapshot()))
        relative=ordinal-self.entry_counts.get(kind,0)
        if self.options.get('failure')==[kind,relative]:self.error=kind;self.u.emu_stop();return False
        self.mutate(kind,relative);self.history.append(dict(kind=kind,args=deepcopy(args)));return True
    def ret(self,value=0):
        for n in VOLATILE[1:]:self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),POISON)
        for i in range(6):self.u.reg_write(getattr(self.x,'UC_X86_REG_XMM'+str(i)),(1<<127)|i)
        super().ret(value)
    def hook(self,uc,address,size,data):
        if address==self.stop:return
        rva=address-self.base;self.executed.add(rva)
        if rva in self.instructions:
            self.last_native=rva
            if rva==TARGETS[self.method][0]:self.entries.append(dict(method=self.method,volatile_registers=self.registers(),volatile_xmm_hex=self.xmm()))
            return
        cx,dx,r8,r9=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
        if rva==0x3F5540:
            assert cx==self.owner and dx==self.code and r8==0
            result=self.options.get('search_result','loc0');args=[self.oid(cx),self.oid(dx),0,result]
            if self.event('find_locale',args):self.ret(self.pointer(result))
        elif rva==0xF76390:
            loc=self.reg(self.x.UC_X86_REG_RBX);assert cx==self.rq(loc+TARGETS[self.method][4]) and dx==0
            result=self.options.get('empty_result_bits',0xBADDF00D00000000);args=[self.oid(cx),0,result]
            if self.event('is_null_or_empty',args):self.ret(result)
        else:raise AssertionError(f'unclaimed {rva:x}')
    def run(self,method,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options=deepcopy(options or {})
        self.method=method;self.error=None;self.allowed={};self.entry_counts=self.counts.copy()
        self.owner=self.pointer(self.options.get('owner','owner'));self.code=self.pointer(self.options.get('code','code'))
        initial=self.snapshot();old=len(self.events);x=self.x;sp=self.stack+0x18008;self.q(sp,self.stop)
        incoming={n:0xDEAD123400000000+i for i,n in enumerate(VOLATILE)};incoming['RCX']=self.owner;incoming['RDX']=self.code
        for n in ['R8','R9']:incoming[n]=self.options.get('entry_'+n.lower()+'_bits',incoming[n])
        for n,v in incoming.items():self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        initial_xmm=[f'{((1<<126)|i):032x}' for i in range(6)]
        for i,v in enumerate(initial_xmm):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),int(v,16))
        for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),(1<<125)|i)
        self.u.reg_write(x.UC_X86_REG_RSP,sp);self.u.emu_start(self.base+TARGETS[method][0],self.stop,timeout=10000000,count=10000)
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==sp+8
            assert all(self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB0000000000000+i for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']))
            assert all(self.reg(getattr(x,'UC_X86_REG_XMM'+str(i)))==(1<<125)|i for i in range(6,16))
        final=self.snapshot()
        for n,raw in initial['memory'].items():assert all(i in self.allowed.get(n,set()) or b==bytes.fromhex(final['memory'][n])[i] for i,b in enumerate(bytes.fromhex(raw))),n
        row=dict(method=method,options=self.options,entry_volatile_registers=incoming,entry_volatile_xmm_hex=initial_xmm,initial=initial,events=deepcopy(self.events[old:]),final=final,returned=returned,error=self.error,result_bits=self.reg(x.UC_X86_REG_RAX) if returned else None,final_volatile_registers=self.registers(),final_volatile_xmm_hex=self.xmm(),service_counts_before=self.entry_counts,service_counts_after=self.counts.copy(),normal_abi_verified=returned,reached_only_writes_verified=True)
        self.verify(row);row['independent_complete_model_verified']=True;return row
    def verify(self,row):
        s=deepcopy(row['initial']);options=row['options'];regs=row['entry_volatile_registers'].copy();xmm=row['entry_volatile_xmm_hex'].copy();counts=row['service_counts_before'].copy();events=[];returned=False;error=None;result=None
        s['native_entries'].append(dict(method=row['method'],volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy()))
        def read(name,off):return int.from_bytes(bytes.fromhex(s['memory'][name])[off:off+8],'little')
        class Stopped(Exception):pass
        def emit(kind,args,site,out):
            nonlocal regs,xmm,error
            ordinal=counts.get(kind,0)+1;counts[kind]=ordinal
            events.append(dict(kind=kind,ordinal=ordinal,args=args,raw_args=[regs[n] for n in ['RCX','RDX','R8','R9']],volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),native_site=hex(site),caller_return_bits=self.base+site+5,snapshot=deepcopy(s)))
            relative=ordinal-row['service_counts_before'].get(kind,0)
            if options.get('failure')==[kind,relative]:error=kind;raise Stopped
            for n,off,target in options.get('mutations',{}).get(kind+':'+str(relative),[]):
                raw=bytearray.fromhex(s['memory'][n]);raw[off:off+8]=self.pointer(target).to_bytes(8,'little');s['memory'][n]=raw.hex()
            s['service_history'].append(dict(kind=kind,args=deepcopy(args)))
            regs.update({n:POISON for n in VOLATILE[1:]});regs['RAX']=out;xmm[:]=[f'{((1<<127)|i):032x}' for i in range(6)]
        start,_,_,_,field=TARGETS[row['method']]
        try:
            loc=options.get('search_result','loc0');regs['R8']=0
            emit('find_locale',[self.oid(regs['RCX']),self.oid(regs['RDX']),0,loc],start+9,self.pointer(loc))
            if loc is None:regs['RAX']=0
            else:
                text=self.oid(read(loc,field));bits=options.get('empty_result_bits',0xBADDF00D00000000);regs['RCX']=self.pointer(text);regs['RDX']=0
                emit('is_null_or_empty',[text,0,bits],start+28,bits)
                regs['RAX']=0 if bits&0xFF else read(loc,field)
            result=regs['RAX'];returned=True
        except Stopped:pass
        assert row['events']==events and row['final']==s
        assert (row['returned'],row['error'],row['result_bits'])==(returned,error,result)
        assert row['service_counts_after']==counts and row['final_volatile_registers']==regs and row['final_volatile_xmm_hex']==xmm


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];sequences=[];baselines=[];stops=[]
    for method in TARGETS:
        for loc,bits,owner,code in itertools.product(['loc0','loc1',None],[0,1,0xBADDF00D00000000,0xFFFFFFFFFFFFFF80],['owner',None],['code',None]):cases.append(m.run(method,dict(search_result=loc,empty_result_bits=bits,owner=owner,code=code)))
        for field in [0x18,0x20]:cases.append(m.run(method,dict(null_field=field)))
        cases.append(m.run(method,dict(alias_texts=True,entry_r8_bits=0xFEDCBA9876543210,entry_r9_bits=0x123456789ABCDEF0)))
        for kind,loc,field,target in itertools.product(['find_locale','is_null_or_empty'],['loc0','loc1'],[0x18,0x20],['replacement',None]):cases.append(m.run(method,dict(search_result=loc,mutations={kind+':1':[[loc,field,target]]})))
    for options in [{},dict(search_result=None),dict(search_result='loc1'),dict(mutations={'is_null_or_empty:1':[['loc0',0x18,'replacement'],['loc0',0x20,None]]})]:
        rows=[m.run('GetTranslatedName',options),m.run('GetIWasTranslated',{},True),m.run('GetTranslatedName',dict(empty_result_bits=0xABCD000000000001),True)]
        assert all(r['returned'] for r in rows) and all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
    for kind in ['find_locale','is_null_or_empty']:
        rows=[m.run('GetTranslatedName',dict(failure=[kind,1])),m.run('GetIWasTranslated',{},True),m.run('GetTranslatedName',{},True)]
        assert not rows[0]['returned'] and all(r['returned'] for r in rows[1:]) and all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
    for method in TARGETS:
        for options in [{},dict(search_result=None),dict(mutations={'is_null_or_empty:1':[['loc0',0x18,None],['loc0',0x20,'replacement']]})]:
            baseline=m.run(method,options);bid=len(baselines);baselines.append(baseline);counts={}
            for i,e in enumerate(baseline['events']):
                kind=e['kind'];counts[kind]=counts.get(kind,0)+1;row=m.run(method,dict(options,failure=[kind,counts[kind]]))
                assert not row['returned'] and row['events']==baseline['events'][:i+1] and row['final']==e['snapshot'];stops.append(dict(baseline=bid,prefix_length=i+1,result=row))
    assert set(m.instructions)<=m.executed
    return dict(schema='character_loc_text_native_v1',build=BUILD,targets=m.targets,bounds=m.bounds,supplied_targets=m.supplied,instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,summary=dict(cases=len(cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)),scope='Exact two CharacterLoc getters; whole FindLocaleLoc and String.IsNullOrEmpty supplied; AL branch, captured LocaleLoc and post-string-call field reload; no real locale lookup/string semantics/runtime admission/unwind')


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('game_root');parser.add_argument('dumper_root');parser.add_argument('--output',required=True);args=parser.parse_args()
    full=audit(args.game_root,args.dumper_root);memory=pool_memory(full);assert expand_memory(memory)==full
    report=pool_snapshots(memory);assert expand_snapshots(report)==memory and expand_memory(expand_snapshots(report))==full
    Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8');print(json.dumps(full['summary']))
