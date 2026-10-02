"""Exact RevealBluff body with frozen authored storage/ABI infrastructure."""
import argparse
import hashlib
import itertools
import json
import re
import struct
from copy import deepcopy
from pathlib import Path
import capstone
from audit_character_assets import BUILD
from audit_character_reward_presentation import Machine as FrozenMachine, FIELDS, POISON
from audit_report_snapshots import pool_snapshots

START,END,NEXT=0x368130,0x368292,0x3682A0
BLUFF_FIELDS=dict(FIELDS,bluff=0x58)
SERVICES={0xF7B1B0:'System.String$$ToUpper',0x3B4AB0:'CharacterData$$GetArt',0x3B4A20:'CharacterData$$GetArtType',0x3688B0:'Character$$SetupArt',0x1C82480:'UnityEngine.Object$$op_Inequality',0x1D49700:'UnityEngine.UI.Image$$set_sprite',0x3695A0:'Character$$UpdateView',0x367B60:'Character$$RefreshView'}

class Machine(FrozenMachine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root,dumper_root)
        ext=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pinned(name,key):
            raw=(Path(dumper_root)/name).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==ext['outputs'][key]['sha256'].upper();return raw.decode('utf-8-sig')
        metadata=json.loads(pinned('script.json','script_json'));dump=pinned('dump.cs','dump_cs')
        rows=[r for r in metadata['ScriptMethod'] if r['Name']=='Character$$RevealBluff' and r['Address']==START]
        assert len(rows)==1 and rows[0]['TypeSignature']=='vii' and rows[0]['Signature']=='void Character__RevealBluff (Character_o* __this, const MethodInfo* method);'
        self.targets=[dict(rows[0],method_id='tdi5487.m0046')]
        assert min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>START)==NEXT
        chunks=[]
        for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root=entry
            while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
            if root.struct.BeginAddress==START:chunks.append([entry.struct.BeginAddress,entry.struct.EndAddress])
        assert chunks==[[START,END+1]]
        section=self.pe.get_section_by_rva(START);assert section and NEXT<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(START,NEXT-START);assert len(raw)==NEXT-START and raw[END-START:]==bytes([0xCC])*(NEXT-END)
        ins=list(self.cs.disasm(raw[:END-START],START));assert sum(i.size for i in ins)==END-START
        self.instructions={i.address:i for i in ins}
        self.bounds={'start':hex(START),'end_exclusive':hex(END),'terminal_trap':hex(END),'next_managed':hex(NEXT),'unwind_chunks':[[hex(a),hex(b)] for a,b in chunks]}
        block=re.search(r'^public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487\s*\{(.*?)// Properties',dump,re.M|re.S)
        assert block and 'public CharacterData bluff; // 0x58' in block[1]
        refs={i.address+i.size+o.mem.disp for i in ins for o in i.operands if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP}
        rows=[r for r in metadata['ScriptMetadata']+metadata['ScriptMetadataMethod'] if r['Address'] in refs]
        assert len(rows)==1 and rows[0]['Name']=='UnityEngine.Object_TypeInfo' and self.base+rows[0]['Address']==self.slot
        assert not [r for r in metadata['ScriptString'] if r['Address'] in refs]
        flags=[i.address+i.size+i.operands[0].mem.disp for i in ins if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP];assert len(flags)==1;self.flag=flags[0]
        self.supplied=[]
        for address,name in list(SERVICES.items())+[(0x1BE7620,'TMPro.TMP_Text$$set_text'),(0x1BE64E0,'TMPro.TMP_Text$$set_color')]:
            rows=[r for r in metadata['ScriptMethod'] if r['Address']==address and r['Name']==name];assert len(rows)==1;self.supplied+=rows
        self.checks={0x368159:('mov','rax, qword ptr [rbx + 0x58]'),0x36815D:('mov','rdi, qword ptr [rbx + 0x40]'),0x36817E:('test','rdi, rdi'),
            0x368187:('mov','r9, qword ptr [rdi]'),0x36818A:('mov','rdx, rax'),0x368190:('mov','r8, qword ptr [r9 + 0x560]'),0x368197:('call','qword ptr [r9 + 0x558]'),
            0x36819E:('mov','rdx, qword ptr [rbx + 0x58]'),0x3681AB:('mov','rcx, qword ptr [rbx + 0x40]'),0x3681B8:('movups','xmm0, xmmword ptr [rdx + 0xd8]'),
            0x3681C7:('movaps','xmmword ptr [rsp + 0x20], xmm0'),0x3681CC:('mov','r8, qword ptr [rax + 0x2b0]'),0x3681D3:('call','qword ptr [rax + 0x2a8]'),
            0x3681ED:('mov','rcx, qword ptr [rbx + 0x58]'),0x368207:('mov','r8d, eax'),0x368215:('mov','rdi, qword ptr [rbx + 0x58]'),
            0x368225:('mov','rdi, qword ptr [rdi + 0xb8]'),0x368247:('test','al, al'),0x36824B:('mov','rdx, qword ptr [rbx + 0x58]'),
            0x368260:('mov','rdx, qword ptr [rdx + 0xb8]'),0x368274:('call','0x3695a0'),0x368288:('jmp','0x367b60'),0x36828D:('call','0x2b7d90')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())

    def snapshot(self):
        result=super().snapshot();result['fields']['bluff']=self.oid(self.rq(self.p['actor']+0x58));return result

    def prepare(self,options):
        super().prepare(options)
        source='data0' if options.get('alias_bluff_data') else options.get('bluff_source','data1')
        self.q(self.p['actor']+0x58,0 if options.get('null_field')=='bluff' else self.p[source])
        if options.get('null_bluff_name'):self.q(self.p[source]+0x28,0)
        if options.get('null_bluff_background'):self.q(self.p[source]+0xB8,0)
        self.types[source]=options.get('type_bits',10);self.live['background'+source[-1]]=options.get('background_live',True)
        self.d(self.p['object_class']+0xE0,int(options.get('class_warm',options.get('warm',False))))
        self.u.mem_write(self.base+self.flag,bytes([int(options.get('metadata_warm',options.get('warm',False)))]))

    def mutate(self,phase):
        if self.options.get('mutation_phase')!=phase:return
        action=self.options['mutation']
        if action in ['clear_bluff','replace_bluff']:
            self.q(self.p['actor']+0x58,0 if action=='clear_bluff' else self.p['data2']);self.allowed.setdefault('actor',set()).update(range(0x58,0x60))
        elif action=='replace_background_sprite':
            n=self.oid(self.rq(self.p['actor']+0x58));self.q(self.p[n]+0xB8,self.p['background0']);self.allowed.setdefault(n,set()).update(range(0xB8,0xC0))
        else:super().mutate(phase)

    def hook(self,uc,address,size,data):
        if address==self.stop:return
        rva,x=address-self.base,self.x;self.executed.add(rva)
        if rva in self.instructions:
            self.last_native=rva
            if rva==START:self.entries.append({'entry':'RevealBluff','raw_args':[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]})
            return
        cx,dx,r8,r9=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']];value=0;phase=None
        if rva==0x2B7B40:
            assert cx==self.slot and self.last_native==0x36814D;kind='metadata_service';args=[cx-self.base,'object_class'];value=self.rq(cx);phase='metadata'
        elif rva==0x281D90:
            assert cx==self.p['object_class'] and self.last_native==0x368235;kind='class_initialization_service';args=['object_class'];phase='class_init'
        elif rva==0x2B7D90:
            assert self.last_native==0x36828D;self.event('native_null_guard',[]);self.error='native_null_guard';uc.emu_stop();return
        elif address==self.text_gateway:
            assert self.oid(cx) in self.components and dx in [0,self.p['upper0'],self.p['upper1'],self.p['upper2']] and r8==self.rq(self.rq(cx)+0x560) and r9==self.rq(cx) and self.last_native==0x368197
            kind='tmp_text_service';args=[self.oid(cx),self.oid(dx),self.oid(r8),self.oid(r9)];phase='text'
        elif address==self.color_gateway:
            assert self.oid(cx) in self.components and self.stack<=dx<self.stack+0x20000 and r8==self.rq(self.rq(cx)+0x2B0) and self.last_native==0x3681D3
            kind='tmp_color_service';args=[self.oid(cx),list(struct.unpack('<IIII',uc.mem_read(dx,16))),self.oid(r8)];phase='color'
        elif rva==0xF7B1B0:
            assert self.oid(cx) in ['name0','name1','name2'] and dx==0 and self.last_native==0x368179;kind=SERVICES[rva];value=0 if self.options.get('uppercase_null') else self.p['upper'+self.oid(cx)[-1]];args=[self.oid(cx),dx,self.oid(value)];phase='uppercase'
        elif rva==0x3B4AB0:
            assert self.oid(cx) in self.types and dx==0 and self.last_native==0x3681E8;kind=SERVICES[rva];value=0 if self.options.get('null_art_sprite') else self.p['sprite'+self.oid(cx)[-1]];args=[self.oid(cx),dx,self.oid(value)];phase='art'
        elif rva==0x3B4A20:
            assert self.oid(cx) in self.types and dx==0 and self.last_native==0x3681FF;kind=SERVICES[rva];value=0xFACE123400000000|self.types[self.oid(cx)];args=[self.oid(cx),dx,value&0xFFFFFFFF,value];phase='type'
        elif rva==0x3688B0:
            assert cx==self.p['actor'] and dx in [0,self.p['sprite0'],self.p['sprite1'],self.p['sprite2']] and r8<=0xFFFFFFFF and r9==0 and self.last_native==0x368210;kind=SERVICES[rva];args=['actor',self.oid(dx),r8,r9];phase='setup_art'
        elif rva==0x1C82480:
            assert dx==0 and r8==0 and self.last_native==0x368242;kind=SERVICES[rva];value=0x1234567800000000|(self.options.get('live_true_byte',0xFE) if cx and self.live[self.oid(cx)] else 0);args=[self.oid(cx),None,r8,value];phase='inequality'
        elif rva==0x1D49700:
            assert self.oid(cx) in self.components and dx in [0,self.p['background0'],self.p['background1'],self.p['background2']] and r8==0 and self.last_native==0x36826A;kind=SERVICES[rva];args=[self.oid(cx),self.oid(dx),r8];phase='background_set'
        elif rva in [0x3695A0,0x367B60]:
            assert cx==self.p['actor'] and dx==0 and self.last_native==(0x368274 if rva==0x3695A0 else 0x368288);kind=SERVICES[rva];args=['actor',dx];phase='view' if rva==0x3695A0 else 'refresh'
        else:raise AssertionError(f'unclaimed instruction {rva:x}')
        if self.event(kind,args):
            if kind=='class_initialization_service':self.d(cx+0xE0,1);self.allowed.setdefault('object_class',set()).update(range(0xE0,0xE4))
            elif kind=='tmp_text_service':self.components[self.oid(cx)]['text']=self.oid(dx)
            elif kind=='tmp_color_service':self.components[self.oid(cx)]['color_bits']=args[1].copy()
            elif rva==0x1D49700:self.components[self.oid(cx)]['sprite']=self.oid(dx)
            self.requests.append({'kind':kind,'args':deepcopy(args)});self.mutate(phase);self.ret(value)

    def run(self,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options,self.counts,self.error,self.allowed,self.fault=options or {},{},None,{},None
        initial,old=self.snapshot(),len(self.events);x,sp=self.x,self.stack+0x18008;self.q(sp,self.stop)
        for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB00000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),(1<<125)|i)
        entry=[0 if self.options.get('null_owner') else self.p['actor'],0,0,0xABCDEF1234567890]
        for n,v in zip(['RCX','RDX','R8','R9'],entry):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        self.u.reg_write(x.UC_X86_REG_RSP,sp);self.u.reg_write(x.UC_X86_REG_MXCSR,0x1F80)
        try:self.u.emu_start(self.base+START,self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base;assert self.options.get('null_owner') and exc.errno==self.unicorn.UC_ERR_READ_UNMAPPED and pc==0x368159;self.error='native_owner_read_fault';self.fault=hex(pc)
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==sp+8
            for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):assert self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB00000+i
            for i in range(6,16):assert self.reg(getattr(x,f'UC_X86_REG_XMM{i}'))==(1<<125)|i
        final=self.snapshot();events=self.events[old:].copy()
        for n,raw in initial['memory'].items():
            before,after=bytes.fromhex(raw),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or b==after[i] for i,b in enumerate(before)),n
        assert final['metadata_slots']==initial['metadata_slots']
        row={'entry':'RevealBluff','entry_raw_args':entry,'options':self.options.copy(),'initial':initial,'events':events,'final':final,'returned':returned,'error':self.error,'fault_rva':self.fault,'normal_abi_verified':returned}
        verify_semantics(row,self);row['independent_ordered_state_verified']=True;return row

def verify_semantics(row,m):
    s=deepcopy(row['initial']);mem={n:bytearray.fromhex(raw) for n,raw in s['memory'].items()};events=[];o=row['options'];error=None;fault=None;regs=row['entry_raw_args'].copy();p=m.p;sp=m.stack+0x18008
    def q(n,off):return struct.unpack_from('<Q',mem[n],off)[0]
    def field(n):return m.oid(q('actor',BLUFF_FIELDS[n]))
    def snapshot():
        r=deepcopy(s);r['memory']={n:raw.hex() for n,raw in mem.items()};r['fields']={n:field(n) for n in BLUFF_FIELDS};r['left_act_bits']=mem['actor'][0xB0]
        r['data']={n:{'name':m.oid(q(n,0x28)),'background':m.oid(q(n,0xB8)),'color_bits':list(struct.unpack_from('<IIII',mem[n],0xD8))} for n in ['data0','data1','data2']};return r
    def mutate(phase):
        if o.get('mutation_phase')!=phase:return
        a=o['mutation']
        if a.startswith(('clear_','replace_')) and a.partition('_')[2] in BLUFF_FIELDS:
            n=a.partition('_')[2];v=0 if a.startswith('clear_') else p['data2' if n=='bluff' else 'data1' if n=='data' else 'tmp1' if n=='name' else 'bg1' if n=='background' else 'acted2'];struct.pack_into('<Q',mem['actor'],BLUFF_FIELDS[n],v)
        elif a=='replace_tmp_class':struct.pack_into('<Q',mem[field('name')],0,p['tmp_class1'])
        elif a=='replace_background_sprite':struct.pack_into('<Q',mem[field('bluff')],0xB8,p['background0'])
        else:raise AssertionError(a)
    def service(kind,args,site,raw,phase,effect=None,tail=False):
        nonlocal regs,error
        events.append({'kind':kind,'args':deepcopy(args),'snapshot':snapshot(),'raw_args':raw.copy(),'native_site':hex(site),'return_target':m.stop if tail else m.base+site+m.instructions[site].size})
        if o.get('failure')==[kind,sum(e['kind']==kind for e in events)]:error=kind;return False
        if effect:effect()
        s['requests'].append({'kind':kind,'args':deepcopy(args)});mutate(phase);regs=[POISON]*4;return True
    def guard(raw):
        nonlocal error
        events.append({'kind':'native_null_guard','args':[],'snapshot':snapshot(),'raw_args':raw.copy(),'native_site':'0x36828d','return_target':m.base+END});error='native_null_guard'
    def body():
        nonlocal error,fault
        if not s['metadata_flag']:
            if not service('metadata_service',[m.slot-m.base,'object_class'],0x36814D,[m.slot,regs[1],regs[2],regs[3]],'metadata'):return
            s['metadata_flag']=1
        if not row['entry_raw_args'][0]:error='native_owner_read_fault';fault='0x368159';return
        data,name_receiver=field('bluff'),field('name')
        if data is None:guard(regs);return
        if q(data,0x28)==0:guard([0,regs[1],regs[2],regs[3]]);return
        name=m.oid(q(data,0x28));upper=None if o.get('uppercase_null') else 'upper'+name[-1]
        if not service(SERVICES[0xF7B1B0],[name,0,upper],0x368179,[p[name],0,regs[2],regs[3]],'uppercase'):return
        if name_receiver is None:guard(regs);return
        cls=m.oid(q(name_receiver,0));mi=m.oid(q(cls,0x560))
        if not service('tmp_text_service',[name_receiver,upper,mi,cls],0x368197,[p[name_receiver],0 if upper is None else p[upper],p[mi],p[cls]],'text',lambda:s['components'][name_receiver].update(text=upper)):return
        data,name_receiver=field('bluff'),field('name')
        if data is None:guard([regs[0],0,regs[2],regs[3]]);return
        if name_receiver is None:guard([0,p[data],regs[2],regs[3]]);return
        color=list(struct.unpack_from('<IIII',mem[data],0xD8));cls=m.oid(q(name_receiver,0));mi=m.oid(q(cls,0x2B0))
        if not service('tmp_color_service',[name_receiver,color,mi],0x3681D3,[p[name_receiver],sp-0x18,p[mi],regs[3]],'color',lambda:s['components'][name_receiver].update(color_bits=color.copy())):return
        data=field('bluff')
        if data is None:guard([0,regs[1],regs[2],regs[3]]);return
        sprite=None if o.get('null_art_sprite') else 'sprite'+data[-1]
        if not service(SERVICES[0x3B4AB0],[data,0,sprite],0x3681E8,[p[data],0,regs[2],regs[3]],'art'):return
        data=field('bluff')
        if data is None:guard([0,regs[1],regs[2],regs[3]]);return
        typ=s['types'][data];raw_return=0xFACE123400000000|typ
        if not service(SERVICES[0x3B4A20],[data,0,typ,raw_return],0x3681FF,[p[data],0,regs[2],regs[3]],'type'):return
        if not service(SERVICES[0x3688B0],['actor',sprite,typ,0],0x368210,[p['actor'],0 if sprite is None else p[sprite],typ,0],'setup_art'):return
        data=field('bluff')
        if data is None:guard(regs);return
        background=m.oid(q(data,0xB8))
        if not struct.unpack_from('<I',mem['object_class'],0xE0)[0]:
            if not service('class_initialization_service',['object_class'],0x368235,[p['object_class'],regs[1],regs[2],regs[3]],'class_init',lambda:struct.pack_into('<I',mem['object_class'],0xE0,1)):return
        result=0x1234567800000000|(o.get('live_true_byte',0xFE) if background and s['background_liveness'][background] else 0)
        if not service(SERVICES[0x1C82480],[background,None,0,result],0x368242,[0 if background is None else p[background],0,0,regs[3]],'inequality'):return
        if result&255:
            data,bg=field('bluff'),field('background')
            if data is None:guard([regs[0],0,regs[2],regs[3]]);return
            if bg is None:guard([0,p[data],regs[2],regs[3]]);return
            sprite_bg=m.oid(q(data,0xB8))
            if not service(SERVICES[0x1D49700],[bg,sprite_bg,0],0x36826A,[p[bg],0 if sprite_bg is None else p[sprite_bg],0,regs[3]],'background_set',lambda:s['components'][bg].update(sprite=sprite_bg)):return
        if not service(SERVICES[0x3695A0],['actor',0],0x368274,[p['actor'],0,regs[2],regs[3]],'view'):return
        service(SERVICES[0x367B60],['actor',0],0x368288,[p['actor'],0,regs[2],regs[3]],'refresh',tail=True)
    s['native_entries'].append({'entry':'RevealBluff','raw_args':regs.copy()});body()
    assert row['events']==events,('events',o);assert row['error']==error and row['returned']==(error is None) and row['fault_rva']==fault
    assert row['final']==snapshot(),('final',o)

def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];seq=[];bases=[];stops=[]
    for meta,cls,live,upper,typ,alias in itertools.product([False,True],[False,True],[False,True],[False,True],[0,10,0xFFFFFFFF],[False,True]):cases.append(m.run({'metadata_warm':meta,'class_warm':cls,'background_live':live,'uppercase_null':upper,'type_bits':typ,'alias_bluff_data':alias}))
    for options in [{'null_owner':True},{'null_field':'bluff'},{'null_field':'name'},{'null_field':'background'},{'null_field':'data'},{'null_bluff_name':True},{'null_bluff_background':True},{'null_art_sprite':True},{'alias_bg_tmp':True},{'bluff_source':'data2'},{'live_true_byte':0},{'live_true_byte':0x80}]:cases.append(m.run(options))
    mutations=[('metadata','replace_name'),('metadata','replace_bluff'),('uppercase','replace_name'),('uppercase','clear_name'),('uppercase','replace_bluff'),('uppercase','clear_bluff'),('uppercase','clear_data'),('uppercase','replace_tmp_class'),('text','replace_name'),('text','clear_name'),('text','replace_bluff'),('text','clear_bluff'),('text','replace_tmp_class'),('color','replace_bluff'),('color','clear_bluff'),('art','replace_bluff'),('art','clear_bluff'),('type','replace_bluff'),('setup_art','replace_bluff'),('setup_art','clear_bluff'),('class_init','replace_bluff'),('class_init','clear_bluff'),('class_init','replace_background_sprite'),('inequality','replace_bluff'),('inequality','clear_bluff'),('inequality','replace_background_sprite'),('inequality','replace_background'),('inequality','clear_background'),('view','clear_bluff'),('refresh','replace_bluff')]
    for phase,action in mutations:cases.append(m.run({'mutation_phase':phase,'mutation':action}))
    for phase,action in [('class_init','clear_bluff'),('class_init','replace_background_sprite'),('inequality','clear_bluff'),('inequality','replace_background_sprite')]:cases.append(m.run({'background_live':False,'mutation_phase':phase,'mutation':action}))
    for phase in ['metadata','class_init']:cases.append(m.run({'warm':True,'mutation_phase':phase,'mutation':'clear_bluff'}))
    for options in [{},{'alias_bluff_data':True},{'alias_bg_tmp':True}]:seq.append([m.run(options),m.run({'mutation_phase':'art','mutation':'replace_bluff'},True),m.run({'uppercase_null':True},True)])
    profiles=[{}, {'uppercase_null':True},{'metadata_warm':True,'class_warm':False},{'mutation_phase':'uppercase','mutation':'replace_tmp_class'},{'mutation_phase':'art','mutation':'replace_bluff'},{'mutation_phase':'inequality','mutation':'replace_background_sprite'}]
    for options in profiles:
        baseline=m.run(options);assert baseline['returned'];bid=len(bases);bases.append(baseline);counts={}
        for i,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;row=m.run(dict(options,failure=[kind,counts[kind]]));assert not row['returned'] and row['events']==baseline['events'][:i+1] and row['final']==e['snapshot'];stops.append({'baseline':bid,'prefix_length':i+1,'result':row})
    assert not set(m.instructions)-m.executed,[hex(a) for a in sorted(set(m.instructions)-m.executed)]
    return {'build':BUILD,'schema':'character_bluff_presentation_native_v1','targets':m.targets,'supplied_declarations':m.supplied,'bounds':m.bounds,'operand_assertions':{hex(a):list(v) for a,v in m.checks.items()},'metadata_flag_rva':hex(m.flag),'metadata_slot_rva':hex(m.slot-m.base),'virtual_slots':{'text':{'slot':66,'function_offset':0x558,'method_offset':0x560,'r9':'physical TMP class'},'color':{'slot':23,'function_offset':0x2A8,'method_offset':0x2B0}},'decoded_instructions':len(m.instructions),'executed_instructions':len(set(m.instructions)&m.executed),'observed_addresses':len(m.executed),'cases':cases,'sequences':seq,'baselines':bases,'stops':stops,'summary':{'cases':len(cases),'normal_returns':sum(r['returned'] for r in cases),'native_stops':sum(not r['returned'] for r in cases),'sequences':len(seq),'baselines':len(bases),'stops':len(stops)},'scope':'Exact RevealBluff only; frozen reward presentation allocation/ABI/type pins reused, other bodies excluded. Nullable ToUpper result forwarded unchanged; R9 physical TMP class; actor.bluff source. SetupArt/Data getters, TMP/Unity, metadata/class runtime, ordered UpdateView and tail RefreshView supplied. Authored aliases/callbacks, no renderer/scheduler/runtime object admission/unwinding.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('game_root');p.add_argument('dumper_root');p.add_argument('--output',required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);Path(a.output).write_text(json.dumps(pool_snapshots(r),sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8');print(json.dumps(r['summary']))
