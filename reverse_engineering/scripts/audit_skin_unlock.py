"""Offline SkinData unlock callers; save/List services remain supplied whole."""
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

TARGETS={'CheckIfUnlocked':(0x3EBAF0,0x3EBB8E,0x3EBB90),'UnlockSkin':(0x3EBE50,0x3EBEF0,0x3EBEF0)}
SERVICES={0x2B7B40:'metadata',0x387CB0:'get_unlocked',0xB55950:'contains',0x2EB0:'add',0x388000:'set_unlocked',0x2B7D90:'null_guard'}
GPR=['rax','rcx','rdx','r8','r9','r10','r11','rbx','rbp','rsi','rdi','r12','r13','r14','r15','rsp']
VOL=GPR[:7];NONVOL=GPR[7:15];MASK=(1<<64)-1;POISON=0xFACE123456789090


def wire_registers(registers):
    return {n:f'{v:032x}' if n.startswith('xmm') else v for n,v in registers.items()}


def unwire_registers(registers):
    return {n:int(v,16) if n.startswith('xmm') else v for n,v in registers.items()}


class Fault(Exception):
    def __init__(self,address,size,write=False):self.address,self.size,self.write=address,size,write


class Memory:
    def __init__(self,layout,raw):self.layout=layout;self.raw={n:bytearray.fromhex(raw[n]) for n in layout}
    def locate(self,address,size,write=False):
        for n,(p,length) in self.layout.items():
            if p<=address and address+size<=p+length:return n,address-p
        raise Fault(address,size,write)
    def read(self,address,size):
        n,o=self.locate(address,size);return bytes(self.raw[n][o:o+size])
    def integer(self,address,size):return int.from_bytes(self.read(address,size),'little')
    def write(self,address,raw):
        n,o=self.locate(address,len(raw),True);self.raw[n][o:o+len(raw)]=raw
    def store(self,address,value,size):self.write(address,(value&((1<<(8*size))-1)).to_bytes(size,'little'))


def snap(mem,m,history,writes,entries):
    return dict(memory={n:mem.read(p,size).hex() for n,(p,size) in m.layout.items()},history=deepcopy(history),native_writes=deepcopy(writes),native_entries=deepcopy(entries))


def contract(kind,args,mem,m,o,ordinal,site):
    """Only authored completed service effects; no save files or native List code."""
    c,d,r8,r9=args;effects=[];p=m.p
    if kind=='metadata':result=mem.integer(c,8)
    elif kind=='get_unlocked':result=p[o.get('saved_result','saved')] if o.get('saved_result','saved') else 0
    elif kind=='contains':result=o.get('contains_bits',0xCAFE123456789000)
    elif kind=='add':
        count=mem.integer(c+0x18,4);assert count<4
        array=mem.integer(c+0x10,8)
        effects=[(array+0x20+count*8,d.to_bytes(8,'little')),(c+0x18,(count+1).to_bytes(4,'little'))]
        result=o.get('add_return_bits',0xADD01234567890AB)
    elif kind=='set_unlocked':result=o.get('set_return_bits',0x5E701234567890AB)
    else:raise AssertionError(kind)
    for plan in o.get('mutations',[]):
        if plan['kind']==kind and plan['ordinal']==ordinal and plan['site']==hex(site):
            for n,offset,size,value in plan['writes']:
                value=p[value] if isinstance(value,str) else value
                effects.append((m.layout[n][0]+offset,value.to_bytes(size,'little')))
    return result,effects


class Model:
    """Independent byte-addressed instruction model, never reads observed effects."""
    def __init__(self,m,row):
        self.m,self.row=m,row;self.mem=Memory(m.layout,row['initial']['memory']);self.r=unwire_registers(row['entry_registers']);self.pc=m.base+TARGETS[row['method']][0]
        self.history=deepcopy(row['initial']['history']);self.writes=deepcopy(row['initial']['native_writes']);self.entries=deepcopy(row['initial']['native_entries'])
        self.entries.append(dict(method=row['method'],registers=wire_registers(self.r)))
        self.events=[];self.counts={};self.error=self.fault=None;self.returned=False;self.zf=self.sf=self.of=self.cf=False
    def alias(self,n):
        if n in self.r:return n,128 if n.startswith('xmm') else 64
        if n in {'eax','ecx','edx','ebx','esi','edi','esp'}:return {'eax':'rax','ecx':'rcx','edx':'rdx','ebx':'rbx','esi':'rsi','edi':'rdi','esp':'rsp'}[n],32
        if n in {'al','cl','dl','bl','dil','sil'}:return {'al':'rax','cl':'rcx','dl':'rdx','bl':'rbx','dil':'rdi','sil':'rsi'}[n],8
        if re.fullmatch(r'r\d+[db]',n):return n[:-1],32 if n[-1]=='d' else 8
        raise AssertionError(n)
    def get(self,n):
        k,b=self.alias(n);return self.r[k]&((1<<b)-1)
    def set(self,n,v):
        k,b=self.alias(n);mask=(1<<b)-1;self.r[k]=v&mask if b>=32 else (self.r[k]&~mask)|(v&mask)
    def address(self,op,i):
        reg=self.m.cs.reg_name
        b=i.address+self.m.base+i.size if reg(op.mem.base)=='rip' else self.get(reg(op.mem.base)) if op.mem.base else 0
        return (b+(self.get(reg(op.mem.index))*op.mem.scale if op.mem.index else 0)+op.mem.disp)&MASK
    def read(self,op,i):
        return self.get(self.m.cs.reg_name(op.reg)) if op.type==1 else op.imm&((1<<(op.size*8))-1) if op.type==2 else self.mem.integer(self.address(op,i),op.size)
    def write(self,op,i,v):
        if op.type==1:self.set(self.m.cs.reg_name(op.reg),v)
        else:self.store(self.address(op,i),v,op.size,i.address)
    def store(self,a,v,size,site):
        self.mem.store(a,v,size);self.writes.append([hex(site),a,size,v&((1<<(size*8))-1)])
    def flags(self,a,b,bits,sub=False):
        mask=(1<<bits)-1;sign=1<<(bits-1);v=(a-b if sub else a&b)&mask
        self.zf=v==0;self.sf=bool(v&sign);self.of=bool((a^b)&(a^v)&sign) if sub else False;self.cf=a<b if sub else False
    def run(self):
        m,r=self.m,self.r
        try:
            for _ in range(500):
                if self.pc==m.stop:self.returned=True;break
                i=m.instructions[self.pc-m.base];op=i.mnemonic;v=i.operands;following=self.pc+i.size
                if op in ['mov','movzx']:self.write(v[0],i,self.read(v[1],i))
                elif op=='lea':self.write(v[0],i,self.address(v[1],i))
                elif op in ['cmp','test']:self.flags(self.read(v[0],i),self.read(v[1],i),v[0].size*8,op=='cmp')
                elif op in ['xor','add','sub']:
                    a,b=self.read(v[0],i),self.read(v[1],i);value=a^b if op=='xor' else a+b if op=='add' else a-b
                    self.write(v[0],i,value)
                    if op=='xor':self.flags(value,value,v[0].size*8)
                elif op=='push':
                    value=self.read(v[0],i);r['rsp']-=8;self.store(r['rsp'],value,8,i.address)
                elif op=='pop':
                    value=self.mem.integer(r['rsp'],8);r['rsp']+=8;self.write(v[0],i,value)
                elif op=='ret':following=self.mem.integer(r['rsp'],8);r['rsp']+=8
                elif op in ['je','jne','jb','jle']:
                    take=self.zf if op=='je' else not self.zf if op=='jne' else self.cf if op=='jb' else self.zf or self.sf!=self.of
                    if take:following=m.base+v[0].imm
                elif op=='call':
                    r['rsp']-=8;self.store(r['rsp'],following,8,i.address);kind=SERVICES[v[0].imm];args=[r[n] for n in ['rcx','rdx','r8','r9']]
                    ordinal=self.counts.get(kind,0)+1;self.counts[kind]=ordinal
                    self.events.append(dict(kind=kind,ordinal=ordinal,site=hex(i.address),caller=following,entry_sp=r['rsp'],raw_args=args,registers=wire_registers(r),snapshot=snap(self.mem,m,self.history,self.writes,self.entries)))
                    if kind=='null_guard' or self.row['options'].get('failure')==[kind,ordinal]:self.error=kind;break
                    result,effects=contract(kind,args,self.mem,m,self.row['options'],ordinal,i.address)
                    for a,raw in effects:self.mem.write(a,raw)
                    self.history.append(dict(kind=kind,ordinal=ordinal,site=hex(i.address),args=args,return_bits=result,writes=[[a,b.hex()] for a,b in effects]))
                    r.update({n:POISON for n in VOL[1:]});r['rax']=result
                    for j in range(6):r[f'xmm{j}']=(1<<127)|j
                    r['rsp']+=8
                else:raise AssertionError((op,i.op_str))
                self.pc=following
            else:raise AssertionError('model instruction limit')
        except Fault as e:
            self.error='native_write_fault' if e.write else 'native_read_fault';self.fault=dict(address=e.address,size=e.size,rva=hex(i.address),write=e.write)
        return dict(events=self.events,final=snap(self.mem,m,self.history,self.writes,self.entries),final_registers=wire_registers(r),returned=self.returned,error=self.error,fault=self.fault)


class Machine(NativeMachine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root)
        repo=Path(__file__).parents[1];build=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
        pinmeta=build['inputs']['global_metadata'];raw=(Path(game_root)/pinmeta['path']).read_bytes();assert len(raw)==pinmeta['size'] and hashlib.sha256(raw).hexdigest().upper()==pinmeta['sha256'].upper()
        extraction=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(n,k):
            raw=(Path(dumper_root)/n).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==extraction['outputs'][k]['sha256'].upper();return raw.decode('utf-8-sig')
        metadata=json.loads(pin('script.json','script_json'));dump=pin('dump.cs','dump_cs');header=pin('il2cpp.h','il2cpp_h')
        self.declarations={}
        for declaration in ['public class SkinData : ScriptableObject // TypeDefIndex: 5945','public class SavedSkins // TypeDefIndex: 5551','public class AutoUnlocked : UnlockWith // TypeDefIndex: 5949']:
            match=re.search('^'+re.escape(declaration)+r'\s*\{(.*?)\n\}',dump,re.M|re.S);assert match;self.declarations[declaration]=match[0]
        assert 'public string skinId; // 0x18' in self.declarations[next(iter(self.declarations))]
        assert 'public UnlockWith unlockWith; // 0x68' in self.declarations[next(iter(self.declarations))]
        assert 'public List<string> ids; // 0x10' in self.declarations['public class SavedSkins // TypeDefIndex: 5551']
        assert all(s in header for s in ['Il2CppClass** typeHierarchy;','uint8_t typeHierarchyDepth;','struct __declspec(align(8)) System_String_Fields {\n\tint32_t _stringLength;\n\tuint16_t _firstChar;\n};'])
        self.header_layouts={}
        for name in ['Il2CppClass_1','Il2CppClass_2']:
            block=re.search(r'^struct '+name+r'\s*\{(.*?)\n\};',header,re.M|re.S);assert block
            offset,alignment,fields=0,1,{}
            for line in block[1].strip().splitlines():
                line=line.strip().rstrip(';');field=line.rsplit(' ',1)[1]
                if '*' in line:size,align=8,8
                elif line.startswith('Il2CppType '):size,align=16,8
                elif line.startswith(('uint32_t ','int32_t ','unsigned int ')):size,align=4,4
                elif line.startswith('uint16_t '):size,align=2,2
                elif line.startswith('uint8_t '):size,align=1,1
                elif line.startswith('size_t '):size,align=8,8
                else:raise AssertionError(line)
                offset=(offset+align-1)//align*align;fields[field]=offset;offset+=size;alignment=max(alignment,align)
            self.header_layouts[name]=dict(size=(offset+alignment-1)//alignment*alignment,fields=fields)
        class2=self.header_layouts['Il2CppClass_1']['size']+16
        assert re.search(r'^struct Il2CppClass\s*\{\s*Il2CppClass_1 _1;\s*void\* static_fields;\s*Il2CppRGCTXData\* rgctx_data;\s*Il2CppClass_2 _2;',header,re.M)
        assert class2+self.header_layouts['Il2CppClass_2']['fields']['typeHierarchy']==0xC8
        assert class2+self.header_layouts['Il2CppClass_2']['fields']['typeHierarchyDepth']==0x12C
        assert class2+self.header_layouts['Il2CppClass_2']['fields']['naturalAligment']==0x130
        # The decoded native gate consumes +130, whereas Dumper's generated
        # header places typeHierarchyDepth at +12C. Preserve the discrepancy;
        # this audit binds the byte operand without inventing a runtime member.
        self.class_gate=dict(native_byte_offset=0x130,generated_type_hierarchy_depth_offset=0x12C,runtime_member_name=None,generated_header_member_at_native_offset='naturalAligment')
        self.instructions={};self.targets=[];self.bounds={};self.refs=set()
        import capstone
        for name,(start,end,next_entry) in TARGETS.items():
            rows=[r for r in metadata['ScriptMethod'] if r['Address']==start];assert len(rows)==1 and rows[0]['Name']=='SkinData$$'+name
            assert rows[0]['TypeSignature']==('iii' if name=='CheckIfUnlocked' else 'vii')
            assert rows[0]['Signature']==f"{'bool' if name=='CheckIfUnlocked' else 'void'} SkinData__{name} (SkinData_o* __this, const MethodInfo* method);"
            declaration=self.declarations['public class SkinData : ScriptableObject // TypeDefIndex: 5945']
            method_declarations=re.findall(r'^\s*public[^\n]*\([^\n]*\) \{ \}',declaration.split('// Methods')[1],re.M)
            assert len(method_declarations)==4 and re.search(r'\b'+name+r'\(',method_declarations[1 if name=='CheckIfUnlocked' else 2])
            self.targets+=rows
            assert min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>start)==next_entry
            chunks=[(e.struct.BeginAddress,e.struct.EndAddress,e.unwindinfo.Flags) for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress==start]
            assert chunks==[(start,end,0)]
            section=self.pe.get_section_by_rva(start);assert section and next_entry<=section.VirtualAddress+section.SizeOfRawData
            raw=self.pe.get_data(start,next_entry-start);assert len(raw)==next_entry-start and raw[end-start:]==b'\xcc'*(next_entry-end)
            ins=list(self.cs.disasm(raw[:end-start],start));assert sum(i.size for i in ins)==end-start
            assert (ins[-1].address,ins[-1].size,ins[-1].mnemonic,ins[-1].op_str)==(end-1,1,'int3','')
            self.instructions.update({i.address:i for i in ins});self.bounds[name]=dict(start=hex(start),end_exclusive=hex(end),next_managed=hex(next_entry),padding_bytes=next_entry-end,instruction_count=len(ins),byte_length=end-start,sha256=hashlib.sha256(raw[:end-start]).hexdigest(),unwind_chunks=chunks)
            self.refs.update(i.address+i.size+op.mem.disp for i in ins for op in i.operands if op.type==3 and op.mem.base==capstone.x86.X86_REG_RIP)
        rows=[r for category in ['ScriptMetadata','ScriptMetadataMethod'] for r in metadata[category] if r['Address'] in self.refs]
        self.slots={r['Name']:r['Address'] for r in rows};assert set(self.slots)=={'AutoUnlocked_TypeInfo','Method$System.Collections.Generic.List<string>.Contains()','Method$System.Collections.Generic.List<string>.Add()'}
        self.flags={'CheckIfUnlocked':0x288C682,'UnlockSkin':0x288C683};assert self.refs==set(self.slots.values())|set(self.flags.values())
        self.gateways={hex(a):[hex(i.address) for i in self.instructions.values() if i.mnemonic=='call' and i.operands[0].imm==a] for a in SERVICES}
        assert self.gateways=={'0x2b7b40':['0x3ebb09','0x3ebb15','0x3ebe6d','0x3ebe79'],'0x387cb0':['0x3ebb53','0x3ebe87'],'0xb55950':['0x3ebb71','0x3ebea8'],'0x2eb0':['0x3ebed0'],'0x388000':['0x3ebeda'],'0x2b7d90':['0x3ebb88','0x3ebeea']}
        self.p={n:self.arena+0x480000+j*0x1000 for j,n in enumerate(['owner','other_owner','skin_class','unlock','unlock_class','auto_class','other_auto_class','hierarchy','saved','other_saved','list','other_list','array','other_array','id','other_id','string_class','contains_mi','other_contains_mi','add_mi','other_add_mi'])}
        self.entry_sp=self.stack+0x18008
        self.layout={n:(p,0x180 if n.endswith('class') else 0x100 if n in ['owner','other_owner','list','other_list','array','other_array'] else 0x80) for n,p in self.p.items()}
        self.layout.update({f'slot_{a:x}':(self.base+a,8) for a in self.slots.values()});self.layout.update({f'flag_{a:x}':(self.base+a,1) for a in self.flags.values()})
        self.layout['native_stack']=(self.entry_sp-0x40,0x70)
        self.checks={0x3EBB21:('mov','rax, qword ptr [rbx + 0x68]'),0x3EBB34:('movzx','ecx, byte ptr [rdx + 0x130]'),0x3EBB3B:('cmp','byte ptr [rax + 0x130], cl'),0x3EBB41:('jb','0x3ebb51'),0x3EBB43:('mov','rax, qword ptr [rax + 0xc8]'),0x3EBB4A:('cmp','qword ptr [rax + rcx*8 - 8], rdx'),0x3EBB6D:('mov','rdx, qword ptr [rbx + 0x18]'),0x3EBB76:('test','al, al'),0x3EBB80:('mov','al, 1'),0x3EBEA4:('mov','rdx, qword ptr [rdi + 0x18]'),0x3EBEAD:('test','al, al'),0x3EBEB1:('mov','rdx, qword ptr [rdi + 0x18]'),0x3EBEBA:('cmp','dword ptr [rdx + 0x10], 2'),0x3EBEBE:('jle','0x3ebedf'),0x3EBEC0:('mov','rcx, qword ptr [rbx + 0x10]'),0x3EBED7:('mov','rcx, rbx')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())
        self.supplied=[r for r in metadata['ScriptMethod'] if r['Address'] in [0x387CB0,0xB55950,0x388000] and r['Name'] in ['SavesGame$$get_UnlockedSkins','System.Collections.Generic.List<object>$$Contains','SavesGame$$set_UnlockedSkins']];assert len(self.supplied)==3
        expected={0x387CB0:('SavedSkins_o* SavesGame__get_UnlockedSkins (const MethodInfo* method);','ii'),0xB55950:('bool System_Collections_Generic_List_object___Contains (System_Collections_Generic_List_object__o* __this, Il2CppObject* item, const MethodInfo_B55950* method);','iiii'),0x388000:('void SavesGame__set_UnlockedSkins (SavedSkins_o* value, const MethodInfo* method);','vii')}
        assert all((r['Signature'],r['TypeSignature'])==expected[r['Address']] for r in self.supplied)
        # This generated generic entry has no ScriptMethod row. Bind the whole
        # supplied gateway by the exact caller's List<string>.Add MI argument,
        # with selected entry evidence; do not invent a managed declaration.
        assert not [r for r in metadata['ScriptMethod'] if r['Address']==0x2EB0]
        section=self.pe.get_section_by_rva(0x2EB0);assert section and 0x2EB0+11<=section.VirtualAddress+section.SizeOfRawData
        add_raw=self.pe.get_data(0x2EB0,11);assert len(add_raw)==11
        add_entry=list(self.cs.disasm(add_raw,0x2EB0))
        assert [(i.mnemonic,i.op_str,i.size) for i in add_entry]==[('sub','rsp, 0x28',4),('inc','dword ptr [rcx + 0x1c]',3),('mov','r10, qword ptr [rcx + 0x10]',4)]
        add_rows=[r for r in rows if r['Name']=='Method$System.Collections.Generic.List<string>.Add()'];assert len(add_rows)==1
        self.add_binding=dict(gateway='0x2eb0',native_caller='0x3ebed0',methodinfo_slot=hex(self.slots['Method$System.Collections.Generic.List<string>.Add()']),metadata=add_rows[0],managed_declaration=None,whole_supplied=True,selected_entry_assertions=[['0x2eb0','sub','rsp, 0x28'],['0x2eb4','inc','dword ptr [rcx + 0x1c]'],['0x2eb7','mov','r10, qword ptr [rcx + 0x10]']])
        self.tracking=False;self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE,self.observe);self.u.hook_add(self.unicorn.UC_HOOK_MEM_INVALID,self.invalid)
    def read(self,a,size):return bytes(self.u.mem_read(a,size))
    def integer(self,a,size):return int.from_bytes(self.read(a,size),'little')
    def write(self,a,raw):self.u.mem_write(a,raw)
    def registers(self):return {n:self.reg(getattr(self.x,'UC_X86_REG_'+n.upper())) for n in GPR+[f'xmm{i}' for i in range(16)]}
    def snapshot(self):return snap(self,self,self.history,self.writes,self.entries)
    def observe(self,u,access,address,size,value,data):
        if self.tracking:
            assert any(p<=address and address+size<=p+length for p,length in self.layout.values())
            self.writes.append([hex(self.reg(self.x.UC_X86_REG_RIP)-self.base),address,size,value&((1<<(size*8))-1)])
    def invalid(self,u,access,address,size,value,data):
        write=access==self.unicorn.UC_MEM_WRITE_UNMAPPED;self.error='native_write_fault' if write else 'native_read_fault';self.fault=dict(address=address,size=size,rva=hex(self.reg(self.x.UC_X86_REG_RIP)-self.base),write=write);return False
    def hook(self,u,address,size,data):
        if address==self.stop:self.returned=True;u.emu_stop();return
        rva=address-self.base
        if rva in self.instructions:self.executed.add(rva);self.site=rva;return
        kind=SERVICES[rva];r=self.registers();args=[r[n] for n in ['rcx','rdx','r8','r9']];ordinal=self.counts.get(kind,0)+1;self.counts[kind]=ordinal
        self.events.append(dict(kind=kind,ordinal=ordinal,site=hex(self.site),caller=self.integer(r['rsp'],8),entry_sp=r['rsp'],raw_args=args,registers=wire_registers(r),snapshot=self.snapshot()))
        if kind=='null_guard' or self.options.get('failure')==[kind,ordinal]:self.error=kind;u.emu_stop();return
        assert r['rsp']%16==8
        result,effects=contract(kind,args,self,self,self.options,ordinal,self.site)
        for a,raw in effects:self.write(a,raw)
        self.history.append(dict(kind=kind,ordinal=ordinal,site=hex(self.site),args=args,return_bits=result,writes=[[a,b.hex()] for a,b in effects]))
        for n in VOL[1:]:u.reg_write(getattr(self.x,'UC_X86_REG_'+n.upper()),POISON)
        for i in range(6):u.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),(1<<127)|i)
        super().ret(result)
    def prepare(self,o):
        self.history=[];self.writes=[];self.entries=[];p=self.p
        for n,(address,size) in self.layout.items():self.write(address,bytes([o.get('seed_byte',0xA5)])*size)
        for n in ['owner','other_owner']:self.q(p[n],p['skin_class']);self.q(p[n]+0x18,p['id'] if not o.get('null_id') else 0);self.q(p[n]+0x68,0 if o.get('unlock_null') else p['unlock'])
        self.q(p['unlock'],p['unlock_class']);self.q(p['unlock_class']+0xC8,0 if o.get('null_hierarchy') else p['hierarchy']);self.write(p['unlock_class']+0x130,bytes([o.get('unlock_byte',2)]))
        for n in ['auto_class','other_auto_class']:self.write(p[n]+0x130,bytes([o.get('auto_byte',2)]))
        for j in range(4):self.q(p['hierarchy']+j*8,p['auto_class'] if o.get('auto_match',False) else p['other_auto_class'])
        for saved,lst,array in [('saved','list','array'),('other_saved','other_list','other_array')]:
            self.q(p[saved]+0x10,0 if o.get('null_list') else p[lst]);self.q(p[lst]+0x10,p[array]);self.d(p[lst]+0x18,0);self.d(p[lst]+0x1C,0x80000000)
        for n in ['id','other_id']:self.q(p[n],p['string_class']);self.d(p[n]+0x10,o.get('id_length',3));self.write(p[n]+0x14,b'abc\0')
        for n,a in self.slots.items():self.q(self.base+a,p['auto_class'] if n=='AutoUnlocked_TypeInfo' else p['contains_mi'] if 'Contains' in n else p['add_mi'])
        for a in self.flags.values():self.write(self.base+a,bytes([o.get('warm_byte',0)]))
    def run(self,method,options=None,retained=False):
        o=deepcopy(options or {});self.tracking=False
        if not retained:self.prepare(o)
        self.options=o;self.events=[];self.counts={};self.error=self.fault=None;self.returned=False
        r={n:0xA110000000000000+j for j,n in enumerate(GPR)};r.update({f'xmm{i}':(0xB110000000000000+i)|((0xC110000000000000+i)<<64) for i in range(16)})
        r.update(rcx=0 if o.get('null_owner') else self.p[o.get('owner','owner')],rsp=self.entry_sp)
        self.q(self.entry_sp,self.stop)
        for n,v in r.items():self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n.upper()),v)
        initial=self.snapshot();self.entries.append(dict(method=method,registers=wire_registers(r)));self.tracking=True
        try:self.u.emu_start(self.base+TARGETS[method][0],self.stop+0x1000,count=500)
        except self.unicorn.UcError:assert self.error
        self.tracking=False
        row=dict(method=method,options=o,entry_registers=wire_registers(r),initial=initial,events=deepcopy(self.events),final=self.snapshot(),final_registers=wire_registers(self.registers()),returned=self.returned,error=self.error,fault=self.fault)
        expected=Model(self,row).run()
        for k,v in expected.items():assert row[k]==v,(method,o,k)
        if self.returned:
            assert row['final_registers']['rsp']==self.entry_sp+8
            final=unwire_registers(row['final_registers']);assert all(final[n]==r[n] for n in NONVOL+[f'xmm{i}' for i in range(6,16)])
        row['independent_full_state_verified']=True;return row


def audit(game_root,dumper_root):
    primitive=object.__new__(Model);primitive.r={'rax':MASK,'rdx':POISON}
    primitive.set('al',1);assert primitive.r['rax']==0xFFFFFFFFFFFFFF01
    primitive.set('edx',0);assert primitive.r['rdx']==0
    primitive.flags(0x80000000,2,32,True);assert primitive.sf!=primitive.of
    primitive.flags(1,2,8,True);assert primitive.cf
    registers={n:MASK for n in GPR};registers.update({f'xmm{i}':(1<<127)|i for i in range(16)})
    assert unwire_registers(wire_registers(registers))==registers
    m=Machine(game_root,dumper_root);cases=[];sequences=[];baselines=[];stops=[]
    for method in TARGETS:
        for warm,contains,seed in itertools.product([0,1,0x80,0xFF],[0xCAFE123456789000,0xCAFE123456789001,0xCAFE1234567890FE],[0,0xA5]):cases.append(m.run(method,dict(warm_byte=warm,contains_bits=contains,seed_byte=seed)))
        profiles=[{},dict(owner='other_owner'),dict(null_owner=True),dict(saved_result=None),dict(null_list=True),dict(null_id=True),dict(null_id=True,contains_bits=0xFACE123456789001),dict(contains_bits=0xCAFE1234567890FF),dict(mutations=[dict(kind='get_unlocked',ordinal=1,site=hex(0x3EBE87 if method=='CheckIfUnlocked' else 0x3EBB53),writes=[['owner',0x18,8,'other_id']])])]
        if method=='CheckIfUnlocked':
            profiles += [dict(unlock_null=True),dict(auto_match=True),dict(unlock_byte=1),dict(auto_byte=1,auto_match=True),dict(auto_byte=3,unlock_byte=2),dict(auto_byte=4,unlock_byte=4,auto_match=True),dict(null_hierarchy=True),dict(auto_match=True,saved_result=None)]
        else:profiles += [dict(id_length=v) for v in [0,1,2,3,0x7FFFFFFF,0x80000000,0xFFFFFFFF]]
        for o in profiles:cases.append(m.run(method,o))
        plans=[dict(kind='get_unlocked',ordinal=1,site=hex(0x3EBB53 if method=='CheckIfUnlocked' else 0x3EBE87),writes=[['owner',0x18,8,'other_id']]),
               dict(kind='contains',ordinal=1,site=hex(0x3EBB71 if method=='CheckIfUnlocked' else 0x3EBEA8),writes=[['owner',0x18,8,'other_id'],['saved',0x10,8,'other_list']])]
        contains_slot=f"slot_{m.slots['Method$System.Collections.Generic.List<string>.Contains()']:x}"
        add_slot=f"slot_{m.slots['Method$System.Collections.Generic.List<string>.Add()']:x}"
        plans += [dict(kind='metadata',ordinal=2,site=hex(0x3EBB15 if method=='CheckIfUnlocked' else 0x3EBE79),writes=[[contains_slot,0,8,'other_contains_mi']])]
        if method=='UnlockSkin':plans += [dict(kind='contains',ordinal=1,site='0x3ebea8',writes=[['owner',0x18,8,0]]),dict(kind='add',ordinal=1,site='0x3ebed0',writes=[['owner',0x18,8,'other_id'],['saved',0x10,8,'other_list']]),dict(kind='contains',ordinal=1,site='0x3ebea8',writes=[[add_slot,0,8,'other_add_mi']]),dict(kind='set_unlocked',ordinal=1,site='0x3ebeda',writes=[['owner',0x18,8,'other_id'],['saved',0x10,8,'other_list']])]
        for plan in plans:
            o=dict(mutations=[plan]);cases.append(m.run(method,o));profiles.append(o)
        for o in profiles:
            baseline=m.run(method,o);bid=len(baselines);baselines.append(baseline);counts={}
            for index,event in enumerate(baseline['events']):
                kind=event['kind'];counts[kind]=counts.get(kind,0)+1;row=m.run(method,dict(o,failure=[kind,counts[kind]]))
                assert row['events']==baseline['events'][:index+1] and row['final']==event['snapshot']
                stops.append(dict(baseline=bid,prefix_length=index+1,result=row))
    for first,second in [('CheckIfUnlocked','UnlockSkin'),('UnlockSkin','CheckIfUnlocked'),('UnlockSkin','UnlockSkin')]:
        rows=[m.run(first),m.run(second,retained=True),m.run(first,dict(contains_bits=0xF00D000000000001),True)]
        assert all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
    for method in TARGETS:
        rows=[m.run(method,dict(failure=['metadata',2])),m.run(method,retained=True),m.run(method,dict(failure=['get_unlocked',1]),True)]
        assert all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
    # Only terminal nonreturning guard padding is outside supported execution.
    omitted=set(m.instructions)-m.executed
    assert omitted=={0x3EBB8D,0x3EBEEF},[hex(a) for a in omitted]
    report=dict(schema='skin_unlock_native_v1',build=BUILD,targets=m.targets,body_bounds=m.bounds,field_declarations=m.declarations,derived_header_layouts=m.header_layouts,native_class_byte_gate=m.class_gate,metadata_slots={n:hex(a) for n,a in m.slots.items()},metadata_flags={n:hex(a) for n,a in m.flags.items()},supplied_targets=m.supplied,whole_add_gateway_binding=m.add_binding,supplied_gateway_sites=m.gateways,instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(m.instructions)-len(omitted),excluded_rvas=[hex(a) for a in sorted(omitted)],diagnostic_windows_not_object_extents={n:s for n,(p,s) in m.layout.items()},scope='Two actual SkinData callers only; whole save getter/setter,List Contains/Add,metadata and exception services supplied; native inline byte/index gate and signed ID-length gate included. Runtime name of native class byte+130 unresolved against generated header; no save persistence or runtime admission.',cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,summary=dict(cases=len(cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)))
    encoded=pool_snapshots(pool_memory(report));assert expand_memory(expand_snapshots(encoded))==report;return encoded


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('game_root',type=Path);parser.add_argument('dumper_root',type=Path);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    report=audit(a.game_root,a.dumper_root);a.output.parent.mkdir(parents=True,exist_ok=True);a.output.write_text(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps(report['summary'],sort_keys=True))
