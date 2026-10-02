"""Actual SetupObject and RevealReal; reward initialization and UI supplied."""
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
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots

TARGETS={0x3689D0:('SetupObject','tdi5487.m0019',0x368A44,0x368A50),0x3682A0:('RevealReal','tdi5487.m0044',0x36840E,0x368410)}
FIELDS={'name':0x40,'data':0x50,'acted':0xA8,'left':0xB8,'up':0xC0,'down':0xC8,'right':0xD0,'background':0x130}
SIDES={10:'up',20:'left',30:'down',40:'right'}
COLORS=[[0,0,0,0],[0x3F800000]*4,[0x80000000,0x7FC01234,0x7F800000,0xFF800000]]
POISON=0xFACE123456789090
SERVICES={0xF7B1B0:'System.String$$ToUpper',0x3B4AB0:'CharacterData$$GetArt',0x3B4A20:'CharacterData$$GetArtType',0x3688B0:'Character$$SetupArt',0x1C82480:'UnityEngine.Object$$op_Inequality',0x1D49700:'UnityEngine.UI.Image$$set_sprite',0x3694D0:'Character$$UpdateViewReal'}

class Machine(NativeMachine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root)
        ext=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(name,key):
            raw=(Path(dumper_root)/name).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==ext['outputs'][key]['sha256'].upper();return raw.decode('utf-8-sig')
        metadata=json.loads(pin('script.json','script_json'));dump=pin('dump.cs','dump_cs')
        self.targets=[];self.instructions={};self.bounds={}
        for start,(name,mid,end,next_entry) in TARGETS.items():
            rows=[r for r in metadata['ScriptMethod'] if r['Name']=='Character$$'+name and r['Address']==start]
            args='int32_t actedSide, ' if name=='SetupObject' else ''
            assert len(rows)==1 and rows[0]['Signature']==f'void Character__{name} (Character_o* __this, {args}const MethodInfo* method);'
            assert rows[0]['TypeSignature']==('viii' if args else 'vii');self.targets.append(dict(rows[0],method_id=mid))
            assert min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>start)==next_entry
            chunks=[]
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root=entry
                while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
                if root.struct.BeginAddress==start:chunks.append([entry.struct.BeginAddress,entry.struct.EndAddress])
            assert chunks==([] if name=='SetupObject' else [[start,end+1]])
            section=self.pe.get_section_by_rva(start);assert section and next_entry<=section.VirtualAddress+section.SizeOfRawData
            raw=self.pe.get_data(start,next_entry-start);assert len(raw)==next_entry-start and raw[end-start:]==bytes([0xCC])*(next_entry-end)
            ins=list(self.cs.disasm(raw[:end-start],start));assert sum(i.size for i in ins)==end-start
            self.instructions.update({i.address:i for i in ins});self.bounds[hex(start)]={'end_exclusive':hex(end),'next_managed':hex(next_entry),'unwind_chunks':[[hex(a),hex(b)] for a,b in chunks]}
        def declaration(name,tdi):
            m=re.search(r'^public (?:abstract )?(?:class|enum) '+name+r'(?: :[^\n]*)? // TypeDefIndex: '+str(tdi)+r'\s*\{(.*?)\n\}',dump,re.M|re.S);assert m;return m[1]
        assert all(s in declaration('Character',5487) for s in ['public TextMeshProUGUI chName; // 0x40','public CharacterData dataRef; // 0x50','public Acted acteds; // 0xA8','public bool leftAct; // 0xB0','public Acted leftActed; // 0xB8','public Acted upActed; // 0xC0','public Acted downActed; // 0xC8','public Acted rightActed; // 0xD0','public Image artBg; // 0x130'])
        assert all(s in declaration('CharacterData',5845) for s in ['public string characterName; // 0x28','public Sprite backgroundArt; // 0xB8','public Color color; // 0xD8'])
        assert all(f'public const EActedSide {n} = {v};' in declaration('EActedSide',5470) for n,v in [('None',0),('Up',10),('Left',20),('Down',30),('Right',40)])
        tmp=declaration('TMP_Text',9110)
        assert '// RVA: 0x1BE7620 Offset: 0x1BE6220 VA: 0x181BE7620 Slot: 66\n\tpublic virtual void set_text(string value)' in tmp
        assert '// RVA: 0x1BE64E0 Offset: 0x1BE50E0 VA: 0x181BE64E0 Slot: 23\n\tpublic override void set_color(Color value)' in tmp
        assert 'TMP_Text' in dump and declaration('TextMeshProUGUI',8974)
        self.supplied=[]
        for address,name in list(SERVICES.items())+[(0x1BE7620,'TMPro.TMP_Text$$set_text'),(0x1BE64E0,'TMPro.TMP_Text$$set_color')]:
            rows=[r for r in metadata['ScriptMethod'] if r['Address']==address and r['Name']==name];assert len(rows)==1;self.supplied+=rows
        refs={i.address+i.size+o.mem.disp for i in self.instructions.values() for o in i.operands if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP}
        rows=[r for r in metadata['ScriptMetadata']+metadata['ScriptMetadataMethod'] if r['Address'] in refs];assert len(rows)==1 and rows[0]['Name']=='UnityEngine.Object_TypeInfo'
        literals=[r for r in metadata['ScriptString'] if r['Address'] in refs];assert len(literals)==1 and literals[0]['Value']==''
        self.slot=self.base+rows[0]['Address'];self.literal_slot=self.base+literals[0]['Address']
        flags=[i.address+i.size+i.operands[0].mem.disp for i in self.instructions.values() if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP];assert len(flags)==1;self.flag=flags[0]
        names=['actor','tmp0','tmp1','tmp2','bg0','bg1','acted0','acted1','acted2','acted3','data0','data1','data2','name0','name1','name2','upper0','upper1','upper2','empty','sprite0','sprite1','sprite2','background0','background1','background2','object_class','tmp_class0','tmp_class1','text_mi0','text_mi1','color_mi0','color_mi1']
        self.p={n:self.arena+0x10000+i*0x1000 for i,n in enumerate(names)};self.ids={p:n for n,p in self.p.items()}
        self.sizes={n:0x600 if n.startswith('tmp_class') else 0x200 if n in ['actor','object_class'] else 0x180 if n.startswith('data') else 0x80 for n in names}
        self.text_gateway,self.color_gateway=self.stop+0x100,self.stop+0x110
        self.checks={0x3689D0:('cmp','edx, 0x28'),0x3689DC:('mov','byte ptr [rcx + 0xb0], 1'),0x3689ED:('jmp','0x2b6ff0'),0x368A43:('ret',''),
            0x3682D9:('mov','rdi, qword ptr [rbx + 0x40]'),0x368304:('cmovne','rdx, rax'),0x368317:('mov','r8, qword ptr [rax + 0x560]'),0x36831E:('call','qword ptr [rax + 0x558]'),
            0x368324:('mov','rdx, qword ptr [rbx + 0x50]'),0x368331:('mov','rcx, qword ptr [rbx + 0x40]'),0x36833E:('movups','xmm0, xmmword ptr [rdx + 0xd8]'),0x36834D:('movaps','xmmword ptr [rsp + 0x20], xmm0'),
            0x368352:('mov','r8, qword ptr [rax + 0x2b0]'),0x368359:('call','qword ptr [rax + 0x2a8]'),0x368373:('mov','rcx, qword ptr [rbx + 0x50]'),0x36838D:('mov','r8d, eax'),
            0x36839B:('mov','rdi, qword ptr [rbx + 0x50]'),0x3683AB:('mov','rdi, qword ptr [rdi + 0xb8]'),0x3683CD:('test','al, al'),0x3683D1:('mov','rdx, qword ptr [rbx + 0x50]'),
            0x3683E6:('mov','rdx, qword ptr [rdx + 0xb8]'),0x368404:('jmp','0x3694d0'),0x368409:('call','0x2b7d90')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.checks.items())

    def oid(self,p):
        if not p:return None
        assert p in self.ids,hex(p);return self.ids[p]

    def snapshot(self):
        return {'fields':{n:self.oid(self.rq(self.p['actor']+off)) for n,off in FIELDS.items()},'left_act_bits':self.u.mem_read(self.p['actor']+0xB0,1)[0],
                'data':{n:{'name':self.oid(self.rq(self.p[n]+0x28)),'background':self.oid(self.rq(self.p[n]+0xB8)),'color_bits':[self.rd(self.p[n]+0xD8+i*4) for i in range(4)]} for n in ['data0','data1','data2']},
                'components':deepcopy(self.components),'types':self.types.copy(),'background_liveness':self.live.copy(),'requests':deepcopy(self.requests),'native_entries':deepcopy(self.entries),
                'metadata_flag':self.u.mem_read(self.base+self.flag,1)[0],'metadata_slots':{'object_class':self.oid(self.rq(self.slot)),'empty':self.oid(self.rq(self.literal_slot))},
                'memory':{n:bytes(self.u.mem_read(p,self.sizes[n])).hex() for n,p in self.p.items()}}

    def prepare(self,options):
        self.options,self.events,self.counts,self.error=options.copy(),[],{},None;self.allowed={};self.requests=[];self.entries=[];self.last_native=None;self.fault=None
        for n,p in self.p.items():self.u.mem_write(p,bytes([0xA5])*self.sizes[n])
        for i in range(2):
            c=self.p['tmp_class'+str(i)];self.q(c+0x558,self.text_gateway);self.q(c+0x560,self.p['text_mi'+str(i)]);self.q(c+0x2A8,self.color_gateway);self.q(c+0x2B0,self.p['color_mi'+str(i)])
        for n in ['tmp0','tmp1','tmp2']:self.q(self.p[n],self.p['tmp_class0'])
        fields={'name':'tmp0','data':'data0','acted':'acted0','left':'acted0','up':'acted1','down':'acted2','right':'acted3','background':'tmp0' if options.get('alias_bg_tmp') else 'bg0'}
        if options.get('alias_acteds'):fields.update({n:'acted0' for n in ['left','up','down','right']})
        if options.get('null_field'):fields[options['null_field']]=None
        for n,off in FIELDS.items():self.q(self.p['actor']+off,0 if fields[n] is None else self.p[fields[n]])
        self.u.mem_write(self.p['actor']+0xB0,bytes([options.get('left_act_bits',0x80)]))
        for i in range(3):
            d=self.p['data'+str(i)];self.q(d+0x28,0 if options.get('null_data_name') and i==0 else self.p['name'+str(i)]);self.q(d+0xB8,0 if options.get('null_background_sprite') and i==0 else self.p['background'+str(i)]);self.u.mem_write(d+0xD8,struct.pack('<IIII',*COLORS[i]))
        self.components={n:{'text':None,'color_bits':[0xA5A5A5A5]*4,'sprite':None} for n in ['tmp0','tmp1','tmp2','bg0','bg1']}
        self.types={'data0':options.get('type_bits',0),'data1':10,'data2':0xFFFFFFFF};self.live={'background0':options.get('background_live',True),'background1':True,'background2':False}
        self.q(self.slot,self.p['object_class']);self.q(self.literal_slot,self.p['empty']);self.d(self.p['object_class']+0xE0,int(options.get('warm',False)));self.u.mem_write(self.base+self.flag,bytes([int(options.get('warm',False))]))

    def mutate(self,phase):
        if self.options.get('mutation_phase')!=phase:return
        action=self.options['mutation']
        if action.startswith(('clear_','replace_')) and action.partition('_')[2] in FIELDS:
            n=action.partition('_')[2];off=FIELDS[n];value=0 if action.startswith('clear_') else self.p['data1' if n=='data' else 'tmp1' if n=='name' else 'bg1' if n=='background' else 'acted2']
            self.q(self.p['actor']+off,value);self.allowed.setdefault('actor',set()).update(range(off,off+8))
        elif action=='replace_tmp_class':
            n=self.oid(self.rq(self.p['actor']+FIELDS['name']));assert n in ['tmp0','tmp1','tmp2'];self.q(self.p[n],self.p['tmp_class1']);self.allowed.setdefault(n,set()).update(range(8))
        elif action=='replace_background_sprite':
            n=self.oid(self.rq(self.p['actor']+FIELDS['data']));self.q(self.p[n]+0xB8,self.p['background1']);self.allowed.setdefault(n,set()).update(range(0xB8,0xC0))
        else:raise AssertionError(action)

    def ret(self,value=0):
        for n in ['RCX','RDX','R8','R9','R10','R11']:self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),POISON)
        for i in range(6):self.u.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),(1<<127)|i)
        super().ret(value)

    def event(self,kind,args):
        result=super().event(kind,args);self.events[-1].update(raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],native_site=hex(self.last_native),return_target=self.rq(self.reg(self.x.UC_X86_REG_RSP)))
        return result

    def hook(self,uc,address,size,data):
        if address==self.stop:return
        rva,x=address-self.base,self.x;self.executed.add(rva)
        if rva in self.instructions:
            self.last_native=rva
            if rva in TARGETS:self.entries.append({'entry':TARGETS[rva][0],'raw_args':[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]})
            return
        cx,dx,r8,r9=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']];kind=None;args=None;value=0;phase=None
        if rva==0x2B7B40:
            assert cx in [self.slot,self.literal_slot] and self.last_native in [0x3682BD,0x3682C9];kind='metadata_service';args=[cx-self.base,self.oid(self.rq(cx))];value=self.rq(cx);phase='metadata'
        elif rva==0x281D90:
            assert cx==self.p['object_class'] and self.last_native==0x3683BB;kind='class_initialization_service';args=['object_class'];phase='class_init'
        elif rva==0x2B6FF0:
            assert cx==self.p['actor']+0xA8 and dx==self.rq(cx) and self.last_native in [0x3689ED,0x368A08,0x368A23,0x368A3E];kind='reference_barrier';args=['actor',0xA8,self.oid(dx)];phase='barrier'
        elif rva==0x2B7D90:
            assert self.last_native==0x368409;self.event('native_null_guard',[]);self.error='native_null_guard';uc.emu_stop();return
        elif address==self.text_gateway:
            assert self.oid(cx) in self.components and self.oid(dx) in ['upper0','upper1','upper2','empty'] and r8==self.rq(self.rq(cx)+0x560) and self.last_native==0x36831E;kind='tmp_text_service';args=[self.oid(cx),self.oid(dx),self.oid(r8)];phase='text'
        elif address==self.color_gateway:
            assert self.oid(cx) in self.components and self.stack<=dx<self.stack+0x20000 and r8==self.rq(self.rq(cx)+0x2B0) and self.last_native==0x368359;kind='tmp_color_service';args=[self.oid(cx),list(struct.unpack('<IIII',uc.mem_read(dx,16))),self.oid(r8)];phase='color'
        elif rva==0xF7B1B0:
            assert self.oid(cx) in ['name0','name1','name2'] and dx==0 and self.last_native==0x3682F5;kind=SERVICES[rva];value=0 if self.options.get('uppercase_null') else self.p['upper'+self.oid(cx)[-1]];args=[self.oid(cx),dx,self.oid(value)];phase='uppercase'
        elif rva==0x3B4AB0:
            assert self.oid(cx) in self.types and dx==0 and self.last_native==0x36836E;kind=SERVICES[rva];value=0 if self.options.get('null_art_sprite') else self.p['sprite'+self.oid(cx)[-1]];args=[self.oid(cx),dx,self.oid(value)];phase='art'
        elif rva==0x3B4A20:
            assert self.oid(cx) in self.types and dx==0 and self.last_native==0x368385;kind=SERVICES[rva];value=0xFACE123400000000|self.types[self.oid(cx)];args=[self.oid(cx),dx,value&0xFFFFFFFF,value];phase='type'
        elif rva==0x3688B0:
            assert cx==self.p['actor'] and dx in [0,self.p['sprite0'],self.p['sprite1'],self.p['sprite2']] and r8<=0xFFFFFFFF and r9==0 and self.last_native==0x368396;kind=SERVICES[rva];args=['actor',self.oid(dx),r8,r9];phase='setup_art'
        elif rva==0x1C82480:
            assert dx==0 and r8==0 and self.last_native==0x3683C8;kind=SERVICES[rva];value=0x1234567800000000|(self.options.get('live_true_byte',0xFE) if cx and self.live[self.oid(cx)] else 0);args=[self.oid(cx),None,r8,value];phase='inequality'
        elif rva==0x1D49700:
            assert self.oid(cx) in self.components and dx in [0,self.p['background0'],self.p['background1'],self.p['background2']] and r8==0 and self.last_native==0x3683F0;kind=SERVICES[rva];args=[self.oid(cx),self.oid(dx),r8];phase='background_set'
        elif rva==0x3694D0:
            assert cx==self.p['actor'] and dx==0 and self.last_native==0x368404;kind=SERVICES[rva];args=['actor',dx];phase='view'
        else:raise AssertionError(f'unclaimed {rva:x}')
        if self.event(kind,args):
            if kind=='class_initialization_service':self.d(cx+0xE0,1);self.allowed.setdefault('object_class',set()).update(range(0xE0,0xE4))
            elif kind=='tmp_text_service':self.components[self.oid(cx)]['text']=self.oid(dx)
            elif kind=='tmp_color_service':self.components[self.oid(cx)]['color_bits']=args[1].copy()
            elif rva==0x1D49700:self.components[self.oid(cx)]['sprite']=self.oid(dx)
            self.requests.append({'kind':kind,'args':deepcopy(args)});self.mutate(phase);self.ret(value)

    def run(self,name,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options,self.counts,self.error,self.allowed,self.fault=options or {},{},None,{},None
        initial,old=self.snapshot(),len(self.events);x,sp=self.x,self.stack+0x18008;self.q(sp,self.stop)
        for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB00000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),(1<<125)|i)
        entry=[0 if self.options.get('null_owner') else self.p['actor'],(0xFACE000000000000|self.options.get('side_bits',0)) if name=='SetupObject' else 0,0,0xABCDEF1234567890]
        for n,v in zip(['RCX','RDX','R8','R9'],entry):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        self.u.reg_write(x.UC_X86_REG_RSP,sp);self.u.reg_write(x.UC_X86_REG_MXCSR,0x1F80)
        start=next(a for a,(n,_,_,_) in TARGETS.items() if n==name)
        try:self.u.emu_start(self.base+start,self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base
            assert self.options.get('null_owner') and exc.errno==self.unicorn.UC_ERR_READ_UNMAPPED and pc in [0x3689D5,0x3689F7,0x368A12,0x368A2D,0x3682D5]
            self.error='native_owner_read_fault';self.fault=hex(pc)
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==sp+8
            for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):assert self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB00000+i
            for i in range(6,16):assert self.reg(getattr(x,f'UC_X86_REG_XMM{i}'))==(1<<125)|i
        final=self.snapshot();events=self.events[old:].copy()
        if name=='SetupObject' and (entry[1]&0xFFFFFFFF) in SIDES and not self.options.get('null_owner'):
            self.allowed.setdefault('actor',set()).update(range(0xA8,0xB0))
            if entry[1]&0xFFFFFFFF==40:self.allowed['actor'].add(0xB0)
        for n,raw in initial['memory'].items():
            before,after=bytes.fromhex(raw),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or b==after[i] for i,b in enumerate(before)),n
        assert final['metadata_slots']==initial['metadata_slots']
        row={'entry':name,'entry_raw_args':entry,'options':self.options.copy(),'initial':initial,'events':events,'final':final,'returned':returned,'error':self.error,'fault_rva':self.fault,'normal_abi_verified':returned}
        verify_semantics(row,self);row['independent_ordered_state_verified']=True;return row

def verify_semantics(row,m):
    s=deepcopy(row['initial']);mem={n:bytearray.fromhex(raw) for n,raw in s['memory'].items()};events=[];o=row['options'];error=None;fault=None;regs=row['entry_raw_args'].copy()
    p=m.p;sp=m.stack+0x18008
    def q(n,off):return struct.unpack_from('<Q',mem[n],off)[0]
    def oid(v):return m.oid(v)
    def snapshot():
        r=deepcopy(s);r['memory']={n:raw.hex() for n,raw in mem.items()};r['fields']={n:oid(q('actor',off)) for n,off in FIELDS.items()};r['left_act_bits']=mem['actor'][0xB0]
        r['data']={n:{'name':oid(q(n,0x28)),'background':oid(q(n,0xB8)),'color_bits':list(struct.unpack_from('<IIII',mem[n],0xD8))} for n in ['data0','data1','data2']};return r
    def field(n):return oid(q('actor',FIELDS[n]))
    def mutate(phase):
        if o.get('mutation_phase')!=phase:return
        a=o['mutation']
        if a.startswith(('clear_','replace_')) and a.partition('_')[2] in FIELDS:
            n=a.partition('_')[2];value=0 if a.startswith('clear_') else p['data1' if n=='data' else 'tmp1' if n=='name' else 'bg1' if n=='background' else 'acted2'];struct.pack_into('<Q',mem['actor'],FIELDS[n],value)
        elif a=='replace_tmp_class':struct.pack_into('<Q',mem[field('name')],0,p['tmp_class1'])
        elif a=='replace_background_sprite':struct.pack_into('<Q',mem[field('data')],0xB8,p['background1'])
        else:raise AssertionError(a)
    def service(kind,args,site,raw,phase,effect=None,tail=False):
        nonlocal regs,error
        return_target=m.stop if tail else m.base+site+m.instructions[site].size
        events.append({'kind':kind,'args':deepcopy(args),'snapshot':snapshot(),'raw_args':raw.copy(),'native_site':hex(site),'return_target':return_target})
        ordinal=sum(e['kind']==kind for e in events)
        if o.get('failure')==[kind,ordinal]:error=kind;return False
        if effect:effect()
        s['requests'].append({'kind':kind,'args':deepcopy(args)});mutate(phase);regs=[POISON]*4;return True
    def guard(raw):
        nonlocal error
        # These are exact call-site register residues, not invented parameters
        # for the runtime null-throw gateway.
        events.append({'kind':'native_null_guard','args':[],'snapshot':snapshot(),'raw_args':raw.copy(),'native_site':'0x368409','return_target':m.base+0x36840E})
        error='native_null_guard';return False
    s['native_entries'].append({'entry':row['entry'],'raw_args':regs.copy()})
    if row['entry']=='SetupObject':
        side=regs[1]&0xFFFFFFFF
        if side in SIDES:
            if not regs[0]:error='native_owner_read_fault';fault=hex({40:0x3689D5,30:0x3689F7,10:0x368A12,20:0x368A2D}[side])
            else:
                value=q('actor',FIELDS[SIDES[side]])
                if side==40:mem['actor'][0xB0]=1
                struct.pack_into('<Q',mem['actor'],0xA8,value)
                service('reference_barrier',['actor',0xA8,oid(value)],{40:0x3689ED,30:0x368A08,10:0x368A23,20:0x368A3E}[side],[p['actor']+0xA8,value,regs[2],regs[3]],'barrier',tail=True)
    else:
        if not s['metadata_flag']:
            for slot,site,label in [(m.slot,0x3682BD,'object_class'),(m.literal_slot,0x3682C9,'empty')]:
                if not service('metadata_service',[slot-m.base,label],site,[slot,regs[1],regs[2],regs[3]],'metadata'):break
            if error is None:s['metadata_flag']=1
        if error is None and not row['entry_raw_args'][0]:error='native_owner_read_fault';fault='0x3682d5'
        if error is None:
            data,name_receiver=field('data'),field('name')
            if data is None:guard(regs)
            elif q(data,0x28)==0:guard([0,regs[1],regs[2],regs[3]])
            else:
                name=oid(q(data,0x28));upper=None if o.get('uppercase_null') else 'upper'+name[-1]
                if service(SERVICES[0xF7B1B0],[name,0,upper],0x3682F5,[p[name],0,regs[2],regs[3]],'uppercase'):
                    text='empty' if upper is None else upper
                    if name_receiver is None:guard([regs[0],p[text],regs[2],regs[3]])
                    else:
                        cls=oid(q(name_receiver,0));mi=oid(q(cls,0x560))
                        if service('tmp_text_service',[name_receiver,text,mi],0x36831E,[p[name_receiver],p[text],p[mi],regs[3]],'text',lambda:s['components'][name_receiver].update(text=text)):
                            data,name_receiver=field('data'),field('name')
                            if data is None:guard([regs[0],0,regs[2],regs[3]])
                            elif name_receiver is None:guard([0,p[data],regs[2],regs[3]])
                            else:
                                color=list(struct.unpack_from('<IIII',mem[data],0xD8));cls=oid(q(name_receiver,0));mi=oid(q(cls,0x2B0))
                                if service('tmp_color_service',[name_receiver,color,mi],0x368359,[p[name_receiver],sp-0x18,p[mi],regs[3]],'color',lambda:s['components'][name_receiver].update(color_bits=color.copy())):
                                    data=field('data')
                                    if data is None:guard([0,regs[1],regs[2],regs[3]])
                                    else:
                                        sprite=None if o.get('null_art_sprite') else 'sprite'+data[-1]
                                        if service(SERVICES[0x3B4AB0],[data,0,sprite],0x36836E,[p[data],0,regs[2],regs[3]],'art'):
                                            data=field('data')
                                            if data is None:guard([0,regs[1],regs[2],regs[3]])
                                            else:
                                                typ=s['types'][data];raw_return=0xFACE123400000000|typ
                                                if service(SERVICES[0x3B4A20],[data,0,typ,raw_return],0x368385,[p[data],0,regs[2],regs[3]],'type'):
                                                    if service(SERVICES[0x3688B0],['actor',sprite,typ,0],0x368396,[p['actor'],0 if sprite is None else p[sprite],typ,0],'setup_art'):
                                                        data=field('data')
                                                        if data is None:guard(regs)
                                                        else:
                                                            background=oid(q(data,0xB8));initialized=struct.unpack_from('<I',mem['object_class'],0xE0)[0]
                                                            if not initialized:
                                                                service('class_initialization_service',['object_class'],0x3683BB,[p['object_class'],regs[1],regs[2],regs[3]],'class_init',lambda:struct.pack_into('<I',mem['object_class'],0xE0,1))
                                                            if error is None:
                                                                result=0x1234567800000000|(o.get('live_true_byte',0xFE) if background and s['background_liveness'][background] else 0)
                                                                if service(SERVICES[0x1C82480],[background,None,0,result],0x3683C8,[0 if background is None else p[background],0,0,regs[3]],'inequality'):
                                                                    if result&255:
                                                                        data,bg=field('data'),field('background')
                                                                        if data is None:guard([regs[0],0,regs[2],regs[3]])
                                                                        elif bg is None:guard([0,p[data],regs[2],regs[3]])
                                                                        else:
                                                                            sprite_bg=oid(q(data,0xB8))
                                                                            service(SERVICES[0x1D49700],[bg,sprite_bg,0],0x3683F0,[p[bg],0 if sprite_bg is None else p[sprite_bg],0,regs[3]],'background_set',lambda:s['components'][bg].update(sprite=sprite_bg))
                                                                    if error is None:service(SERVICES[0x3694D0],['actor',0],0x368404,[p['actor'],0,regs[2],regs[3]],'view',tail=True)
    assert row['events']==events,(row['entry'],o,'events')
    assert row['error']==error and row['returned']==(error is None) and row['fault_rva']==fault
    assert row['final']==snapshot(),(row['entry'],o,'final')

def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];seq=[];bases=[];stops=[]
    for side,left,alias,null in itertools.product([0,10,20,30,40,9,0xFFFFFFFF,0x8000000A],[0,0x80,0xFE],[False,True],[None,'left','up','down','right']):cases.append(m.run('SetupObject',{'side_bits':side,'left_act_bits':left,'alias_acteds':alias,**({} if null is None else {'null_field':null})}))
    for side in [0,10,20,30,40]:cases.append(m.run('SetupObject',{'side_bits':side,'null_owner':True}))
    for warm,bg_live,upper_null,typ,alias in itertools.product([False,True],[False,True],[False,True],[0,10,0xFFFFFFFF],[False,True]):cases.append(m.run('RevealReal',{'warm':warm,'background_live':bg_live,'uppercase_null':upper_null,'type_bits':typ,'alias_bg_tmp':alias}))
    for options in [{'null_owner':True},{'null_field':'data'},{'null_field':'name'},{'null_field':'background'},
                    {'null_field':'background','background_live':False},{'null_field':'background','null_background_sprite':True},
                    {'null_data_name':True},{'null_background_sprite':True},{'null_art_sprite':True},{'live_true_byte':0},{'live_true_byte':0x80}]:cases.append(m.run('RevealReal',options))
    mutations=[('metadata','replace_name'),('metadata','replace_data'),('uppercase','replace_name'),('uppercase','clear_name'),('uppercase','replace_data'),('uppercase','clear_data'),('uppercase','replace_tmp_class'),('text','replace_name'),('text','clear_name'),('text','replace_data'),('text','clear_data'),('text','replace_tmp_class'),('color','replace_data'),('color','clear_data'),('art','replace_data'),('art','clear_data'),('type','replace_data'),('setup_art','clear_data'),('setup_art','replace_data'),('class_init','replace_data'),('class_init','clear_data'),('class_init','replace_background_sprite'),('inequality','replace_data'),('inequality','clear_data'),('inequality','replace_background_sprite'),('inequality','replace_background'),('inequality','clear_background')]
    for phase,action in mutations:cases.append(m.run('RevealReal',{'mutation_phase':phase,'mutation':action}))
    for phase,action in [('class_init','clear_data'),('class_init','replace_background_sprite'),('inequality','clear_data'),('inequality','replace_background_sprite')]:cases.append(m.run('RevealReal',{'background_live':False,'mutation_phase':phase,'mutation':action}))
    for phase in ['metadata','class_init']:cases.append(m.run('RevealReal',{'warm':True,'mutation_phase':phase,'mutation':'clear_data'}))
    for side in [10,20,30,40]:cases.append(m.run('SetupObject',{'side_bits':side,'mutation_phase':'barrier','mutation':'replace_acted'}))
    for options in [{},{'alias_bg_tmp':True},{'alias_acteds':True}]:
        seq.append([m.run('SetupObject',dict(options,side_bits=40)),m.run('RevealReal',{},True),m.run('SetupObject',{'side_bits':10},True),m.run('RevealReal',{},True)])
    for name,options in [('SetupObject',{'side_bits':40}),('RevealReal',{}),('RevealReal',{'uppercase_null':True}),('RevealReal',{'mutation_phase':'uppercase','mutation':'replace_name'}),('RevealReal',{'mutation_phase':'art','mutation':'replace_data'}),('RevealReal',{'mutation_phase':'inequality','mutation':'replace_background_sprite'})]:
        baseline=m.run(name,options);assert baseline['returned'];bid=len(bases);bases.append(baseline);counts={}
        for i,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;row=m.run(name,dict(options,failure=[kind,counts[kind]]));assert not row['returned'] and row['events']==baseline['events'][:i+1] and row['final']==e['snapshot'];stops.append({'baseline':bid,'prefix_length':i+1,'result':row})
    assert not set(m.instructions)-m.executed,[hex(a) for a in sorted(set(m.instructions)-m.executed)]
    return {'build':BUILD,'schema':'character_reward_presentation_native_v1','targets':m.targets,'supplied_declarations':m.supplied,'bounds':m.bounds,'operand_assertions':{hex(a):list(v) for a,v in m.checks.items()},'metadata_flag_rva':hex(m.flag),'metadata_slot_rvas':[hex(m.slot-m.base),hex(m.literal_slot-m.base)],'virtual_slots':{'text':{'slot':66,'function_offset':0x558,'method_offset':0x560},'color':{'slot':23,'function_offset':0x2A8,'method_offset':0x2B0}},'decoded_instructions':len(m.instructions),'executed_instructions':len(set(m.instructions)&m.executed),'observed_addresses':len(m.executed),'cases':cases,'sequences':seq,'baselines':bases,'stops':stops,'summary':{'cases':len(cases),'normal_returns':sum(r['returned'] for r in cases),'native_stops':sum(not r['returned'] for r in cases),'sequences':len(seq),'baselines':len(bases),'stops':len(stops)},'scope':'Actual SetupObject/RevealReal only. InitReward excluded. SetupArt/GetArt/GetArtType/UpdateViewReal, String.ToUpper, TMP/Unity, GC/runtime services supplied. Authored diagnostic memory/alias callbacks; no renderer, runtime object admission or native unwinding.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('game_root');p.add_argument('dumper_root');p.add_argument('--output',required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);Path(a.output).write_text(json.dumps(pool_snapshots(pool_memory(r)),sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8');print(json.dumps(r['summary']))
