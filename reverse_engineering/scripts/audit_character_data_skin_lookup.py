"""Exact CharacterData skin lookup callers; iterator/string/skin services supplied."""
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

METHODS={
 'LoadSkin':dict(id='tdi5845.m0012',symbol='CharacterData::public void LoadSkin(string skinId)',entry=0x3B4F10,end=0x3B5061,next=0x3B5070,signature='void CharacterData__LoadSkin (CharacterData_o* __this, System_String_o* skinId, const MethodInfo* method);',typesig='viii',cleanup=0x3B501B,
  metadata=[0x3B4F3B,0x3B4F47,0x3B4F53,0x3B4F5F],flagstore=0x3B4F64,get=0x3B4F9A,move=0x3B4FD0,equal=0x3B4FED,dispose=[0x3B5014,0x3B5027],barriers=[0x3B4F79,0x3B5003],guards=[0x3B504F,0x3B505B],rethrow=0x3B5055),
 'CheckIfSkinUnlocked':dict(id='tdi5845.m0013',symbol='CharacterData::public bool CheckIfSkinUnlocked(string skinId)',entry=0x3B43B0,end=0x3B4511,next=0x3B4520,signature='bool CharacterData__CheckIfSkinUnlocked (CharacterData_o* __this, System_String_o* skinId, const MethodInfo* method);',typesig='iiii',cleanup=0x3B44CC,
  metadata=[0x3B43D5,0x3B43E1,0x3B43ED,0x3B43F9],flagstore=0x3B43FE,get=0x3B4421,move=0x3B445C,equal=0x3B447D,dispose=[0x3B44A2,0x3B44C5,0x3B44D8],barriers=[],guards=[0x3B44F9,0x3B44FF,0x3B4505],rethrow=0x3B450B,unlock=0x3B4490)}
MI_NAMES=['Method$System.Collections.Generic.List.Enumerator<SkinData>.Dispose()',
          'Method$System.Collections.Generic.List.Enumerator<SkinData>.MoveNext()',
          'Method$System.Collections.Generic.List.Enumerator<SkinData>.get_Current()',
          'Method$System.Collections.Generic.List<SkinData>.GetEnumerator()']
VOL=['RAX','RCX','RDX','R8','R9','R10','R11'];NONVOL=['RBX','RBP','RSI','RDI','R12','R13','R14','R15'];POISON=0xFACE123456789090
EXCLUDED={0x3B44FE:'post nonreturning null-list guard nop',0x3B44FF:'second null captured-skin guard unreachable with preserved RDI',
 0x3B4504:'post unreachable second captured-skin guard nop',0x3B450A:'post nonreturning null-current guard nop',0x3B4510:'post nonreturning rethrow int3',
 0x3B5054:'post nonreturning null-current guard nop',0x3B505A:'post nonreturning rethrow int3',0x3B5060:'post nonreturning null-list guard int3'}


def snap(memory,slots,flags,state,ids):
    def q(n,o):return int.from_bytes(memory[n][o:o+8],'little')
    def oid(v):return ids[v] if v else None
    return dict(memory={n:bytes(v).hex() for n,v in memory.items()},metadata_slots={n:oid(v) for n,v in slots.items()},metadata_flags=flags.copy(),
      owners={n:dict(current_skin=oid(q(n,0xC0)),skins=oid(q(n,0xC8))) for n in ['owner','other_owner']},
      skins={n:dict(skin_id=oid(q(n,0x18))) for n in ['skin0','skin1','skin2']},supplied_state=deepcopy(state))


class Machine(NativeMachine):
    def __init__(self,game_root,dumper_root):
        import capstone
        super().__init__(game_root)
        extraction=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name,key):
            b=(Path(dumper_root)/name).read_bytes();assert hashlib.sha256(b).hexdigest().upper()==extraction['outputs'][key]['sha256'].upper();return b.decode('utf-8-sig')
        metadata=json.loads(pin('script.json','script_json'));dump=pin('dump.cs','dump_cs');header=pin('il2cpp.h','il2cpp_h')
        self.targets={};self.instructions={};self.flags={};refs=set();self.bounds={}
        for name,d in METHODS.items():
            rows=[r for r in metadata['ScriptMethod'] if r['Address']==d['entry']];assert len(rows)==1
            assert (rows[0]['Name'],rows[0]['Signature'],rows[0]['TypeSignature'])==('CharacterData$$'+name,d['signature'],d['typesig']);self.targets[name]=rows[0]
            assert min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>d['entry'])==d['next']
            chunks=[]
            for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root=e
                while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
                if root.struct.BeginAddress==d['entry']:
                    chunks.append((e.struct.BeginAddress,e.struct.EndAddress));assert e.unwindinfo.Flags==3 and e.unwindinfo.ExceptionHandler==0x30CD28
            assert chunks==[(d['entry'],d['end'])]
            section=self.pe.get_section_by_rva(d['entry']);assert section and d['next']<=section.VirtualAddress+section.SizeOfRawData
            raw=self.pe.get_data(d['entry'],d['next']-d['entry']);assert len(raw)==d['next']-d['entry'] and raw[d['end']-d['entry']:]==b'\xcc'*(d['next']-d['end'])
            ins=list(self.cs.disasm(raw[:d['end']-d['entry']],d['entry']));assert sum(i.size for i in ins)==d['end']-d['entry']
            assert (ins[-1].address,ins[-1].size,ins[-1].mnemonic,ins[-1].op_str)==(d['end']-1,1,'int3','')
            self.instructions.update({i.address:i for i in ins})
            for i in ins:
                for op in i.operands:
                    if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:refs.add(i.address+i.size+op.mem.disp)
                if i.mnemonic=='cmp' and i.operands[0].type==capstone.x86.X86_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:self.flags[name]=i.address+i.size+i.operands[0].mem.disp
            for gw,sites in [(0x2B7B40,d['metadata']),(0xB16640,[d['get']]),(0x9693D0,[d['move']]),(0xF73E00,[d['equal']]),(0x33ED50,d['dispose']),(0x2B6FF0,d['barriers']),(0x2B7D90,d['guards']),(0x246610,[d['rethrow']])]:
                assert [i.address for i in ins if i.mnemonic=='call' and i.op_str==hex(gw)]==sites
            self.bounds[name]=dict(start=hex(d['entry']),end_exclusive=hex(d['end']),next_managed=hex(d['next']),padding_bytes=d['next']-d['end'],unwind_ranges=[[hex(a),hex(b)] for a,b in chunks],eh_flags=3,handler_rva='0x30cd28',direct_cleanup=hex(d['cleanup']),byte_length=d['end']-d['entry'],instruction_count=len(ins),sha256=hashlib.sha256(raw[:d['end']-d['entry']]).hexdigest())
        self.slot_names={r['Address']:r['Name'] for cat in ['ScriptMetadata','ScriptMetadataMethod'] for r in metadata[cat] if r['Address'] in refs};self.slots={n:a for a,n in self.slot_names.items()};assert set(self.slots)==set(MI_NAMES)
        for d in METHODS.values():
            for name,site in zip(MI_NAMES,d['metadata']):
                i=self.instructions[site-7];assert i.mnemonic=='lea' and i.operands[1].mem.base==capstone.x86.X86_REG_RIP
                assert self.slot_names[i.address+i.size+i.operands[1].mem.disp]==name
        self.fields=['public SkinData currentSkin; // 0xC0','public List<SkinData> skins; // 0xC8','public string skinId; // 0x18']
        for declaration,index,fields in [('CharacterData : ScriptableObject, ICharacterLocData, ICardData',5845,self.fields[:2]),('SkinData : ScriptableObject',5945,self.fields[2:])]:
            block=re.search(r'^public class '+re.escape(declaration)+r' // TypeDefIndex: '+str(index)+r'\s*\{(.*?)\n\}',dump,re.M|re.S);assert block and all(f in block[1] for f in fields)
        block=re.search(r'^public struct List\.Enumerator<T> : IEnumerator<T>, IDisposable, IEnumerator // TypeDefIndex: 1509\s*\{(.*?)\n\}',dump,re.M|re.S);assert block
        assert all(f in block[1] for f in ['private List<T> _list; // 0x0','private int _index; // 0x0','private int _version; // 0x0','private T _current; // 0x0'])
        block=re.search(r'struct System_Collections_Generic_List_Enumerator_T__Fields \{(.*?)\n\};',header,re.S);assert block
        assert [line.strip() for line in block[1].splitlines() if line.strip()]==['struct System_Collections_Generic_List_T__o* _list;','int32_t _index;','int32_t _version;','Il2CppObject* _current;']
        self.supplied=[]
        for a,n,s,t in [(0xB16640,'System.Collections.Generic.List<object>$$GetEnumerator','System_Collections_Generic_List_Enumerator_T__o System_Collections_Generic_List_object___GetEnumerator (System_Collections_Generic_List_object__o* __this, const MethodInfo_B16640* method);','iii'),
          (0x9693D0,'System.Collections.Generic.List.Enumerator<object>$$MoveNext','bool System_Collections_Generic_List_Enumerator_object___MoveNext (System_Collections_Generic_List_Enumerator_T__o __this, const MethodInfo_9693D0* method);','iii'),
          (0x33ED50,'System.Collections.Generic.List.Enumerator<object>$$Dispose','void System_Collections_Generic_List_Enumerator_object___Dispose (System_Collections_Generic_List_Enumerator_T__o __this, const MethodInfo_33ED50* method);','vii'),
          (0xF73E00,'System.String$$op_Equality','bool System_String__op_Equality (System_String_o* a, System_String_o* b, const MethodInfo* method);','iiii'),
          (0x3EBAF0,'SkinData$$CheckIfUnlocked','bool SkinData__CheckIfUnlocked (SkinData_o* __this, const MethodInfo* method);','iii')]:
            rows=[r for r in metadata['ScriptMethod'] if r['Address']==a and r['Name']==n];assert len(rows)==1 and rows[0]['Signature']==s and rows[0]['TypeSignature']==t;self.supplied.extend(rows)
        self.checks={0x3B441C:('lea','rcx, [rsp + 0x28]'),0x3B4426:('movups','xmm0, xmmword ptr [rsp + 0x28]'),0x3B442B:('movups','xmmword ptr [rsp + 0x40], xmm0'),0x3B4430:('movsd','xmm1, qword ptr [rsp + 0x38]'),0x3B4436:('movsd','qword ptr [rsp + 0x50], xmm1'),
         0x3B443C:('mov','qword ptr [rsp + 0x28], 0'),0x3B4465:('mov','rdi, qword ptr [rsp + 0x50]'),0x3B4476:('mov','rdx, qword ptr [rdi + 0x18]'),0x3B448B:('xor','edx, edx'),0x3B4495:('movzx','edi, al'),0x3B44A7:('movzx','eax, dil'),0x3B44E7:('xor','al, al'),
         0x3B4F74:('mov','qword ptr [rcx], rbx'),0x3B4F95:('lea','rcx, [rsp + 0x28]'),0x3B4F9F:('movups','xmm0, xmmword ptr [rsp + 0x28]'),0x3B4FA4:('movups','xmmword ptr [rsp + 0x40], xmm0'),0x3B4FA9:('movsd','xmm1, qword ptr [rsp + 0x38]'),0x3B4FAF:('movsd','qword ptr [rsp + 0x50], xmm1'),0x3B4FB5:('mov','qword ptr [rsp + 0x28], rbx'),0x3B4FD9:('mov','rdi, qword ptr [rsp + 0x50]'),0x3B4FE9:('mov','rcx, qword ptr [rdi + 0x18]'),0x3B4FFD:('mov','qword ptr [rcx], rdi')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())
        names=['owner','other_owner','data_class','list0','list1','array0','array1','skin0','skin1','skin2','skin_class','query','equal_id','different_id','string_class','exception','clone_enum']+[f'mi{i}' for i in range(4)]+[f'other_mi{i}' for i in range(4)]
        self.p={n:self.arena+0x290000+i*0x1000 for i,n in enumerate(names)};self.sizes={n:0x200 if n in ['owner','other_owner'] else 0x100 if n.startswith(('list','array')) else 0x80 for n in names}
        self.entry_sp=self.stack+0x18008;self.frame=self.entry_sp-0x68
        self.p.update(enum_output=self.frame+0x28,enum_state=self.frame+0x40,scratch=self.frame+0x20);self.sizes['scratch']=0x40
        self.ids={v:n for n,v in self.p.items()};self.tracking=False;self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE,self.observe_write)

    def oid(self,v):return self.ids[v] if v else None
    def snapshot(self):return snap({n:bytes(self.u.mem_read(self.p[n],size)) for n,size in self.sizes.items()},{n:self.rq(self.base+a) for n,a in self.slots.items()},{n:self.u.mem_read(self.base+a,1)[0] for n,a in self.flags.items()},self.state,self.ids)
    def volatile(self):return {n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in VOL}
    def xmm(self):return [f'{self.reg(getattr(self.x,"UC_X86_REG_XMM"+str(i))):032x}' for i in range(6)]

    def observe_write(self,uc,access,address,size,value,data):
        if not self.tracking:return
        pc=self.reg(self.x.UC_X86_REG_RIP)-self.base
        if address in [self.base+a for a in self.flags.values()]:
            assert pc==METHODS[self.method]['flagstore'] and size==value==1;self.flag_written=True
        elif self.owner and self.owner<=address<self.owner+0x200:
            assert self.method=='LoadSkin' and pc in [0x3B4F74,0x3B4FFD] and address-self.owner==0xC0 and size==8
            self.allowed.setdefault(self.oid(self.owner),set()).update(range(0xC0,0xC8))
        elif self.p['scratch']<=address<self.p['scratch']+self.sizes['scratch']:
            ranges={0x3B442B:(0x20,16),0x3B4436:(0x30,8),0x3B443C:(8,8),0x3B444A:(0x10,8),0x3B4FA4:(0x20,16),0x3B4FAF:(0x30,8),0x3B4FB5:(8,8),0x3B4FBF:(0x10,8)}
            off,expected_size=ranges[pc];actual=(address-self.p['scratch'],size)
            assert actual==(off,expected_size) or (expected_size==16 and actual in [(off,8),(off+8,8)]),(hex(pc),actual)
            self.allowed.setdefault('scratch',set()).update(range(address-self.p['scratch'],address-self.p['scratch']+size))

    def write(self,n,offset,size,value):
        if isinstance(value,str):value=self.p[value]
        self.u.mem_write(self.p[n]+offset,value.to_bytes(size,'little'));self.allowed.setdefault(n,set()).update(range(offset,offset+size))
    def mutate(self,kind):
        count=self.completed.get(kind,0)+1;self.completed[kind]=count
        for target,off,size,value in self.options.get('mutations',{}).get(kind+':'+str(count),[]):
            if target.startswith('slot:'):
                name=target[5:];self.q(self.base+self.slots[name],self.p[value]);self.slot_writes[name]=value
            else:self.write(target,off,size,value)
    def event(self,kind,args):
        okay=super().event(kind,args);caller=self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base
        self.events[-1].update(raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm(),caller=hex(caller),native_phase=self.method,entry_mode=self.mode)
        return okay
    def ret(self,value=0):
        for n in VOL[1:]:self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),POISON)
        for i in range(6):self.u.reg_write(getattr(self.x,'UC_X86_REG_XMM'+str(i)),(1<<127)|i)
        super().ret(value)
    def string_equal(self,a,b):
        if not a or not b:return a==b
        return bytes(self.u.mem_read(a+0x14,self.rd(a+0x10)*2))==bytes(self.u.mem_read(b+0x14,self.rd(b+0x10)*2))
    def sequence_value(self,key,index,default):return self.options.get(key,[])[index] if index<len(self.options.get(key,[])) else default

    def prepare(self,options):
        self.options=deepcopy(options);self.events=[];self.counts={};self.error=None
        self.state=dict(entries=[],metadata=[],barriers=[],enumerators=[],moves=[],comparisons=[],unlocks=[],disposals=[],guards=[],rethrows=[],iterator=None)
        for n,size in self.sizes.items():self.u.mem_write(self.p[n],bytes([options.get('seed_byte',0xA5)])*size)
        for i,name in enumerate(MI_NAMES):self.q(self.base+self.slots[name],self.p[f'mi{i}'])
        for name,a in self.flags.items():self.u.mem_write(self.base+a,bytes([options.get('warm_byte',0)]))
        for n in ['owner','other_owner']:self.q(self.p[n],self.p['data_class']);self.q(self.p[n]+0xC0,self.p['skin2']);self.q(self.p[n]+0xC8,self.p[options.get('owner_list','list0')] if not options.get('null_list') else 0)
        for n,text in [('query','target'),('equal_id','target'),('different_id','other')]:
            self.q(self.p[n],self.p['string_class']);self.d(self.p[n]+0x10,len(text));self.u.mem_write(self.p[n]+0x14,text.encode('utf-16-le')+b'\0\0')
        for i in range(3):self.q(self.p[f'skin{i}'],self.p['skin_class']);self.q(self.p[f'skin{i}']+0x18,self.p['equal_id' if i in [0,2] else 'different_id'])
        for i in range(2):
            entries=options.get('entries' if i==0 else 'other_entries',['skin0','skin1','skin2'] if i==0 else ['skin1','skin0'])
            self.q(self.p[f'list{i}']+0x10,self.p[f'array{i}']);self.d(self.p[f'list{i}']+0x18,len(entries));self.d(self.p[f'list{i}']+0x1C,0x12345678+i)
            self.q(self.p[f'array{i}']+0x18,len(entries))
            for j,n in enumerate(entries):self.q(self.p[f'array{i}']+0x20+j*8,self.p[n] if n else 0)
        self.u.mem_write(self.p['clone_enum'],self.p['list1'].to_bytes(8,'little')+(2).to_bytes(4,'little')+(0xDEADBEEF).to_bytes(4,'little')+self.p['skin1'].to_bytes(8,'little'))
        if options.get('mode')=='cleanup':
            self.q(self.frame+0x28,self.p['exception'] if options.get('exception_live') else 0)
            self.q(self.frame+0x30,self.p[options.get('cleanup_enum','enum_state')]);self.u.mem_write(self.p['enum_state'],bytes(self.u.mem_read(self.p['clone_enum'],24)))

    def hook(self,uc,address,size,data):
        rva=address-self.base;self.executed.add(rva);d=METHODS[self.method]
        if rva==d['entry'] or (self.mode=='cleanup' and rva==d['cleanup']):self.state['entries'].append(dict(method=self.method,mode=self.mode,owner=self.oid(self.owner),query=self.oid(self.query),volatile_registers=self.volatile(),volatile_xmm_hex=self.xmm()))
        if rva in self.instructions:return
        cx,dx,r8=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8']];caller=self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base;site=caller-5
        if rva==0x2B7B40:
            name=self.slot_names[cx-self.base];value=self.rq(cx);args=[name,self.oid(value)]
            if self.event('metadata',args):self.state['metadata'].append(args);self.mutate('metadata');self.ret(value)
        elif rva==0x2B6FF0:
            assert cx==self.owner+0xC0 and self.rq(cx)==dx;args=[self.oid(self.owner),0xC0,self.oid(dx)]
            if self.event('reference_barrier',args):self.state['barriers'].append(args);self.mutate('reference_barrier');self.ret(self.options.get('barrier_return_bits',0xBADD123456789056))
        elif rva==0xB16640:
            assert cx==self.p['enum_output'] and self.oid(dx) in ['list0','list1'] and self.oid(r8) in ['mi3','other_mi3']
            args=[self.oid(cx),self.oid(dx),self.oid(r8)]
            if self.event('get_enumerator',args):
                count=self.rd(dx+0x18);assert count<=4;array=self.rq(dx+0x10);entries=[self.oid(self.rq(array+0x20+i*8)) for i in range(count)]
                raw=dx.to_bytes(8,'little')+(0).to_bytes(4,'little')+self.rd(dx+0x1C).to_bytes(4,'little')+(0).to_bytes(8,'little')
                self.state['enumerators'].append(dict(args=args,produced_24_bytes=raw.hex(),supplied_entries=entries));self.state['iterator']=dict(list=self.oid(dx),entries=entries,cursor=0)
                for off in range(0,24,8):self.write('scratch',8+off,8,int.from_bytes(raw[off:off+8],'little'))
                self.mutate('get_enumerator');self.ret(self.options.get('get_return_bits',self.p['enum_output']))
        elif rva==0x9693D0:
            assert cx==self.p['enum_state'] and self.oid(dx) in ['mi1','other_mi1'];args=[self.oid(cx),self.oid(dx)]
            if self.event('move_next',args):
                it=self.state['iterator'];i=it['cursor'];has=i<len(it['entries']);current=it['entries'][i] if has else None;it['cursor']+=1
                value=self.sequence_value('move_next_raw',i,0xBEEF123456789000|int(has));self.write('scratch',0x28,4,i+1);self.write('scratch',0x30,8,current if current else 0)
                self.state['moves'].append(dict(args=args,current=current,return_bits=value));self.mutate('move_next');self.ret(value)
        elif rva==0xF73E00:
            assert r8==0;args=[self.oid(cx),self.oid(dx),0]
            if self.event('string_equality',args):
                i=len(self.state['comparisons'])-self.comparison_start;value=self.sequence_value('equality_raw',i,0xCAFE123456789000|int(self.string_equal(cx,dx)))
                self.state['comparisons'].append(dict(args=args,return_bits=value));self.mutate('string_equality');self.ret(value)
        elif rva==0x3EBAF0:
            assert self.oid(cx) in ['skin0','skin1','skin2'] and dx==0;args=[self.oid(cx),0]
            if self.event('skin_unlocked',args):
                value=self.options.get('unlocked_raw',{}).get(self.oid(cx),0xBEEF123456789001);self.state['unlocks'].append(dict(args=args,return_bits=value));self.mutate('skin_unlocked');self.ret(value)
        elif rva==0x33ED50:
            assert self.oid(cx) in ['enum_state','clone_enum'] and self.oid(dx) in ['mi0','other_mi0'];args=[self.oid(cx),self.oid(dx)]
            if self.event('dispose',args):self.state['disposals'].append(args);self.mutate('dispose');self.ret(self.options.get('dispose_return_bits',0xD15E1234567890AB))
        elif rva in [0x2B7D90,0x246610]:
            kind='native_guard' if rva==0x2B7D90 else 'rethrow';assert site in d['guards'] if kind=='native_guard' else site==d['rethrow']
            args=[hex(site)] if kind=='native_guard' else [self.oid(cx)]
            if self.event(kind,args):self.error=kind;self.u.emu_stop()
        else:raise AssertionError(f'unclaimed native {rva:x}')

    def run(self,method,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options=deepcopy(options or {});self.counts={};self.error=None
        self.method=method;self.mode=self.options.get('mode','ordinary');self.owner=0 if self.options.get('null_owner') else self.p[self.options.get('owner','owner')];self.query=0 if self.options.get('null_query') else self.p[self.options.get('query','query')]
        self.allowed,self.slot_writes,self.flag_written,self.completed={},{},False,{};self.comparison_start=len(self.state['comparisons'])
        x=self.x
        for i,n in enumerate(NONVOL):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),(1<<125)|i)
        incoming={n:0xDEAD123400000000+i for i,n in enumerate(VOL)};incoming.update(RCX=self.owner,RDX=self.query)
        initial_xmm=[f'{((1<<126)|i):032x}' for i in range(6)]
        for n,v in incoming.items():self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        for i,v in enumerate(initial_xmm):self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),int(v,16))
        self.q(self.entry_sp,self.stop)
        if self.mode=='cleanup':
            for off,n in [(0x60,'RDI' if method=='CheckIfSkinUnlocked' else 'R14'),(0x70,'RBX'),(0x78,'RSI')]+([(0x80,'RDI')] if method=='LoadSkin' else []):self.q(self.frame+off,0xFAB0000000000000+NONVOL.index(n))
        self.u.reg_write(x.UC_X86_REG_RSP,self.frame if self.mode=='cleanup' else self.entry_sp)
        initial=self.snapshot();old=len(self.events);self.tracking=True;fault=None;d=METHODS[method]
        try:self.u.emu_start(self.base+(d['cleanup'] if self.mode=='cleanup' else d['entry']),self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base;assert self.owner==0 and self.mode=='ordinary'
            assert (pc,exc.errno)==((0x3B4F74,self.unicorn.UC_ERR_WRITE_UNMAPPED) if method=='LoadSkin' else (0x3B4405,self.unicorn.UC_ERR_READ_UNMAPPED))
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
        assert final['metadata_flags']=={**initial['metadata_flags'],**({method:1} if self.flag_written else {})}
        row=dict(method=method,mode=self.mode,options=self.options,initial=initial,events=deepcopy(self.events[old:]),final=final,returned=returned,error=self.error,fault_rva=fault,
                 entry_volatile_registers=incoming,entry_volatile_xmm_hex=initial_xmm,final_volatile_registers=self.volatile(),final_volatile_xmm_hex=self.xmm(),return_bits=self.reg(x.UC_X86_REG_RAX) if returned else None,
                 normal_or_synthetic_frame_abi_verified=returned,completed_memory_write_offsets={n:sorted(v) for n,v in self.allowed.items()},completed_slot_writes=self.slot_writes.copy(),reached_native_flag_write=self.flag_written,unrelated_storage_retained=True)
        verify(row,self.p,self.ids,self.slots,self.base);row['independent_ordered_full_state_verified']=True;return row


def verify(row,p,ids,slot_addresses,base):
    raw={n:bytearray.fromhex(v) for n,v in row['initial']['memory'].items()};slots={n:p[v] for n,v in row['initial']['metadata_slots'].items()};flags=row['initial']['metadata_flags'].copy();state=deepcopy(row['initial']['supplied_state'])
    method=row['method'];mode=row['mode'];d=METHODS[method];options=row['options'];regs=row['entry_volatile_registers'].copy();xmm=row['entry_volatile_xmm_hex'].copy();owner=ids[regs['RCX']] if regs['RCX'] else None;query=regs['RDX'];events=[];counts={};completed={};returned=False;error=None;fault=None;result=None
    comparison_start=len(state['comparisons']);state['entries'].append(dict(method=method,mode=mode,owner=owner,query=ids[query] if query else None,volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy()))
    def oid(v):return ids[v] if v else None
    def q(n,off):return int.from_bytes(raw[n][off:off+8],'little')
    def rd(n,off):return int.from_bytes(raw[n][off:off+4],'little')
    def write(n,off,size,value):raw[n][off:off+size]=(p[value] if isinstance(value,str) else value).to_bytes(size,'little')
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
        events.append(dict(kind=kind,args=args,snapshot=snap(raw,slots,flags,state,ids),raw_args=[regs[n] for n in ['RCX','RDX','R8','R9']],volatile_registers=regs.copy(),volatile_xmm_hex=xmm.copy(),caller=hex(site+5),native_phase=method,entry_mode=mode))
        counts[kind]=counts.get(kind,0)+1
        if options.get('failure')==[kind,counts[kind]]:raise Stopped
        if kind in ['native_guard','rethrow']:raise Stopped
        if category:state[category].append(args)
        if effect:effect()
        mutate(kind);regs.update({n:POISON for n in VOL[1:]});regs['RAX']=value;xmm[:]=[f'{((1<<127)|i):032x}' for i in range(6)]
    def dispose(site,receiver):
        regs['RDX']=slots[MI_NAMES[0]];regs['RCX']=receiver;emit('dispose',[oid(receiver),oid(regs['RDX'])],site,'disposals',value=options.get('dispose_return_bits',0xD15E1234567890AB))
    def guard(site):emit('native_guard',[hex(site)],site)
    def barrier(site,captured):
        regs['RCX']=p[owner]+0xC0;regs['RDX']=captured;emit('reference_barrier',[owner,0xC0,oid(captured)],site,'barriers',value=options.get('barrier_return_bits',0xBADD123456789056))
    try:
        if mode=='cleanup':
            dispose(d['dispose'][-1],q('scratch',0x10));regs['RCX']=q('scratch',8)
            if regs['RCX']:emit('rethrow',[oid(regs['RCX'])],d['rethrow'])
            if method=='CheckIfSkinUnlocked':regs['RAX']&=~0xFF
        else:
            if flags[method]==0:
                for name,site in zip(MI_NAMES,d['metadata']):
                    regs['RCX']=base+slot_addresses[name];value=slots[name];emit('metadata',[name,oid(value)],site,'metadata',value=value)
                flags[method]=1
            if method=='LoadSkin':
                regs['RCX']=(p[owner] if owner else 0)+0xC0
                if owner is None:error='native_owner_fault';fault='0x3b4f74';raise Stopped
                write(owner,0xC0,8,0);barrier(d['barriers'][0],0)
            elif owner is None:error='native_owner_fault';fault='0x3b4405';raise Stopped
            regs['RDX']=q(owner,0xC8)
            if regs['RDX']==0:guard(d['guards'][-1] if method=='LoadSkin' else d['guards'][0])
            regs['R8']=slots[MI_NAMES[3]];regs['RCX']=p['enum_output'];listname=oid(regs['RDX']);args=[oid(regs['RCX']),listname,oid(regs['R8'])]
            entries=[oid(q(oid(q(listname,0x10)),0x20+i*8)) for i in range(rd(listname,0x18))]
            produced=p[listname].to_bytes(8,'little')+bytes(4)+rd(listname,0x1C).to_bytes(4,'little')+bytes(8)
            def make_enum():
                state['enumerators'].append(dict(args=args,produced_24_bytes=produced.hex(),supplied_entries=entries));state['iterator']=dict(list=listname,entries=entries,cursor=0);raw['scratch'][8:32]=produced
            emit('get_enumerator',args,d['get'],effect=make_enum,value=options.get('get_return_bits',p['enum_output']))
            xmm[0]=f'{int.from_bytes(raw["scratch"][8:24],"little"):032x}';xmm[1]=f'{int.from_bytes(raw["scratch"][24:32],"little"):032x}'
            raw['scratch'][32:56]=raw['scratch'][8:32];write('scratch',8,8,0);write('scratch',0x10,8,p['enum_state'])
            matched=False
            while True:
                regs['RDX']=slots[MI_NAMES[1]];regs['RCX']=p['enum_state'];args=[oid(regs['RCX']),oid(regs['RDX'])];it=state['iterator'];i=it['cursor'];has=i<len(it['entries']);current=it['entries'][i] if has else None;value=series('move_next_raw',i,0xBEEF123456789000|int(has))
                def move():
                    it['cursor']+=1;write('scratch',0x28,4,i+1);write('scratch',0x30,8,current if current else 0);state['moves'].append(dict(args=args,current=current,return_bits=value))
                emit('move_next',args,d['move'],effect=move,value=value)
                if regs['RAX']&0xFF==0:break
                captured=q('scratch',0x30)
                if captured==0:guard(d['guards'][0] if method=='LoadSkin' else d['guards'][-1])
                regs['R8']=0
                if method=='LoadSkin':regs['RDX']=query;regs['RCX']=q(oid(captured),0x18)
                else:regs['RDX']=q(oid(captured),0x18);regs['RCX']=query
                args=[oid(regs['RCX']),oid(regs['RDX']),0];i=len(state['comparisons'])-comparison_start;value=series('equality_raw',i,0xCAFE123456789000|int(equal(regs['RCX'],regs['RDX'])))
                def comparison():state['comparisons'].append(dict(args=args,return_bits=value))
                emit('string_equality',args,d['equal'],effect=comparison,value=value)
                if regs['RAX']&0xFF==0:continue
                if method=='LoadSkin':write(owner,0xC0,8,captured);barrier(d['barriers'][1],captured)
                else:
                    regs['RDX']=0;regs['RCX']=captured;args=[oid(captured),0];value=options.get('unlocked_raw',{}).get(oid(captured),0xBEEF123456789001)
                    def unlocked():state['unlocks'].append(dict(args=args,return_bits=value))
                    emit('skin_unlocked',args,d['unlock'],effect=unlocked,value=value);captured_byte=regs['RAX']&0xFF
                    dispose(d['dispose'][0],p['enum_state']);regs['RAX']=captured_byte;matched=True;break
            if not matched:
                dispose(d['dispose'][0 if method=='LoadSkin' else 1],p['enum_state'])
                if method=='CheckIfSkinUnlocked':regs['RAX']&=~0xFF
        returned=True;result=regs['RAX']
    except Stopped:
        if error is None:error=events[-1]['kind']
    assert (row['returned'],row['error'],row['fault_rva'],row['return_bits'])==(returned,error,fault,result),row['options']
    assert row['events']==events,('events',method,options)
    assert row['final']==snap(raw,slots,flags,state,ids),('final',method,options)
    assert row['final_volatile_registers']==regs and row['final_volatile_xmm_hex']==xmm,('volatile ABI',method,options)


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];sequences=[];baselines=[];stops=[]
    profiles=[[],['skin0'],['skin1'],['skin0','skin1','skin2'],['skin1','skin2','skin0'],['skin0','skin0'],[None],['skin1',None],['skin0',None]]
    for method,warm,seed,owner,entries in itertools.product(METHODS,[0,1,0x80],[0,0xA5],['owner','other_owner'],profiles):
        cases.append(m.run(method,dict(warm_byte=warm,seed_byte=seed,owner=owner,entries=entries)))
    for method in METHODS:
        for options in [dict(null_owner=True),dict(null_list=True),dict(null_query=True),dict(entries=[]),dict(owner_list='list1'),dict(equality_raw=[0xFFFFFFFFFFFFFF00]*3),dict(equality_raw=[0x8000000000000080]*3),dict(unlocked_raw={'skin0':0xFFFFFFFFFFFFFF00,'skin2':0x123456789ABCDE80}),dict(unlocked_raw={'skin0':0x123456789ABCDEFF}),dict(unlocked_raw={'skin0':0x123456789ABCDE80}),dict(dispose_return_bits=0xFFFFFFFFFFFFFFFF),dict(entries=[],dispose_return_bits=0xFFFFFFFFFFFFFFFF),dict(entries=[],dispose_return_bits=0),dict(move_next_raw=[0x123456789ABCDE00]),dict(move_next_raw=[0x80000000000000FF,0]),dict(get_return_bits=0),dict(dispose_return_bits=0),dict(entries=['skin0','skin0','skin2'],mutations={'string_equality:1':[['skin0',0x18,8,'different_id'],['scratch',0x30,8,'skin1']]})]:cases.append(m.run(method,options))
    plans=[('metadata:1',[['owner',0xC8,8,'list1']]),('metadata:4',[['slot:'+MI_NAMES[3],0,8,'other_mi3']]),
      ('get_enumerator:1',[['owner',0xC8,8,'list1'],['slot:'+MI_NAMES[1],0,8,'other_mi1']]),
      ('move_next:1',[['scratch',0x30,8,'skin1']]),('move_next:1',[['scratch',0x30,8,0]]),
      ('string_equality:1',[['scratch',0x30,8,'skin1'],['skin0',0x18,8,'different_id']]),
      ('string_equality:1',[['owner',0xC0,8,'skin1'],['owner',0xC8,8,'list1']]),
      ('skin_unlocked:1',[['slot:'+MI_NAMES[0],0,8,'other_mi0'],['owner',0xC0,8,'skin1']]),
      ('dispose:1',[['owner',0xC0,8,'skin1'],['scratch',0x30,8,'skin2']]),
      ('reference_barrier:1',[['owner',0xC8,8,'list1'],['owner',0xC0,8,'skin1']]),
      ('reference_barrier:2',[['owner',0xC0,8,'skin1'],['array0',0x20,8,'skin2']])]
    for method,(phase,writes),warm,owner in itertools.product(METHODS,plans,[0,0xFE],['owner','other_owner']):cases.append(m.run(method,dict(warm_byte=warm,owner=owner,mutations={phase:writes})))
    for method,live,alias in itertools.product(METHODS,[False,True],['enum_state','clone_enum']):
        cases.append(m.run(method,dict(mode='cleanup',exception_live=live,cleanup_enum=alias)))
    for method in METHODS:
        for value in [0,'exception']:cases.append(m.run(method,dict(mode='cleanup',exception_live=(value==0),mutations={'dispose:1':[['scratch',8,8,value]]})))
        for options in [{},dict(owner='other_owner'),dict(entries=['skin0','skin0']),dict(mutations={'dispose:1':[['owner',0xC0,8,'skin1']]})]:
            rows=[m.run(method,options),m.run(method,dict(options,failure=['string_equality',1]),True),m.run(method,options,True),m.run(method,dict(options,mutations={'dispose:1':[['other_owner',0xC0,8,'skin2']]}),True)]
            assert [row['returned'] for row in rows]==[True,False,True,True]
            assert all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
        rows=[m.run(method,dict(null_list=True,failure=['metadata',4])),m.run(method,dict(mutations={'metadata:1':[['owner',0xC8,8,'list0']]}),True),m.run(method,dict(query='different_id'),True),m.run(method,{},True)]
        assert [row['returned'] for row in rows]==[False,True,True,True]
        assert all(a['final']==b['initial'] for a,b in zip(rows,rows[1:]));sequences.append(rows)
        baseline_options=[{},dict(warm_byte=0xFE),dict(entries=[]),dict(entries=[None]),dict(null_list=True),dict(null_owner=True),dict(mutations={'get_enumerator:1':[['slot:'+MI_NAMES[1],0,8,'other_mi1']]}),dict(mutations={'string_equality:1':[['scratch',0x30,8,'skin1']]}),dict(mode='cleanup'),dict(mode='cleanup',exception_live=True),dict(mode='cleanup',cleanup_enum='clone_enum',mutations={'dispose:1':[['scratch',8,8,'exception']]})]
        for options in baseline_options:
            baseline=m.run(method,options);bid=len(baselines);baselines.append(baseline);counts={}
            for index,e in enumerate(baseline['events']):
                kind=e['kind'];counts[kind]=counts.get(kind,0)+1;stopped=m.run(method,dict(options,failure=[kind,counts[kind]]))
                assert not stopped['returned'] and stopped['events']==baseline['events'][:index+1] and stopped['final']==e['snapshot'];stops.append(dict(baseline=bid,prefix_length=index+1,result=stopped))
    missing=set(m.instructions)-m.executed;assert missing==set(EXCLUDED),(sorted(missing),sorted(set(EXCLUDED)-missing))
    return dict(schema='character_data_skin_lookup_native_v1',build=BUILD,methods={n:dict(method_id=d['id'],symbol_key=d['symbol'],target=m.targets[n]) for n,d in METHODS.items()},body_bounds=m.bounds,
      scope='Exact two CharacterData caller declarations; whole iterator/string/skin/metadata/barrier/guard/rethrow services supplied. Separate direct cleanup frame diagnostics do not execute managed exception dispatch or unwind.',
      supplied_targets=m.supplied,fields=m.fields,enumerator_layout=dict(size=24,list_offset=0,index_offset=8,version_offset=12,current_offset=16,hidden_output='stack frame+28',active_state='stack frame+40'),
      metadata_slots={n:hex(a) for n,a in m.slots.items()},metadata_flags={n:hex(a) for n,a in m.flags.items()},diagnostic_windows_not_object_extents=m.sizes,
      instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),excluded_instruction_reasons={hex(a):reason for a,reason in EXCLUDED.items()},
      cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,summary=dict(cases=len(cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)))


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--game-root',required=True);parser.add_argument('--dumper-root',required=True);parser.add_argument('--output',required=True)
    args=parser.parse_args();report=pool_snapshots(pool_memory(audit(args.game_root,args.dumper_root)))
    Path(args.output).parent.mkdir(parents=True,exist_ok=True);Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'),ensure_ascii=True)+'\n',encoding='utf-8');print(json.dumps(report['summary'],sort_keys=True))
