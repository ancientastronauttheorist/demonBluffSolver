"""Exact nominal SkinData folded caller; ScriptableObject constructor supplied.

The caller clears EDX and tail-transfers without reading or writing SkinData.
Base effects/returns are authored inputs. Runtime construction, other folded
declarations, allocation defaults and Color initialization are not promoted.
"""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re
import struct

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_reward_refresh_join import (
    pool_memory, pool_histories, pool_memory_maps, pool_state_maps,
    pool_snapshots, expand_report, expand_memory, verify_history_codec,
    verify_memory_map_codec, verify_state_map_codec, verify_full_report_codec)

ENTRY, END, NEXT, BASE_CTOR, SITE = 0x373A40, 0x373A47, 0x373A50, 0x1C8A5C0, 0x373A42
BODY_SHA = '20e9f56f8427c04c0925a2326da4e8aa3579d033d7c316b96cb5b2782fae51df'
VOL = ['RAX','RCX','RDX','R8','R9','R10','R11']
NONVOL = ['RBX','RBP','RSI','RDI','R12','R13','R14','R15']
GPRS = VOL+NONVOL+['RSP']
MASK = (1 << 64)-1


def sha(raw): return hashlib.sha256(raw).hexdigest()


def abi(registers):
    return {'volatile_registers':{n:registers[n] for n in VOL},
            'volatile_gpr_hex':{n:f'{registers[n]:016x}' for n in VOL},
            'volatile_xmm_hex':[registers[f'XMM{i}'] for i in range(6)],
            'all_registers':deepcopy(registers)}


def supplied_effects(options,m):
    effects=[]
    for change in options.get('base_writes',[]):
        record,offset,raw=change
        data=bytes.fromhex(raw)
        assert record in m.layout and 0<=offset and offset+len(data)<=m.layout[record][1]
        effects.append([record,offset,data.hex()])
    return effects


def model(row,m):
    """Independent complete byte/register chronology from authored initial state."""
    state=deepcopy(row['initial']);registers=deepcopy(row['entry_registers'])
    # Native entry captures the incoming method bits before the byte-width
    # distinction: EDX zeroing clears every high DWORD bit of RDX.
    entry={'owner':registers['RCX'],'native_entry':m.base+ENTRY,**abi(registers)}
    state['native_entries'].append(entry)
    registers['RDX']=0
    event={'kind':'scriptable_object_constructor','ordinal':1,'native_site':hex(SITE),
           'gateway':m.base+BASE_CTOR,'caller_return_bits':m.stop,'entry_sp':m.entry_sp,
           'raw_args':[registers[n] for n in ['RCX','RDX','R8','R9']],
           **abi(registers),'snapshot':deepcopy(state)}
    events=[event]
    returned=not row['options'].get('stop_base',False)
    if returned:
        effects=supplied_effects(row['options'],m)
        for record,offset,raw in effects:
            memory=bytearray.fromhex(state['memory'][record]);data=bytes.fromhex(raw)
            memory[offset:offset+len(data)]=data;state['memory'][record]=memory.hex()
        state['requests'].append({'kind':'scriptable_object_constructor','ordinal':1,'site':hex(SITE),
                                  'raw_args':event['raw_args'],'return_bits':row['options'].get('return_bits',0xD00D1234567890AB),
                                  'writes':effects,'normal_callee_abi':{'entry_sp':m.entry_sp,'return_sp':m.entry_sp+8,
                                      'caller':m.stop,'preserved':{n:registers[n] for n in NONVOL+[f'XMM{i}' for i in range(6,16)]}}})
        for n in VOL[1:]:registers[n]=row['options'].get('volatile_poison',0xFACE123456789090)
        for i in range(6):registers[f'XMM{i}']=f'{((1 << 127)|i):032x}'
        registers['RAX']=row['options'].get('return_bits',0xD00D1234567890AB)
        registers['RSP']+=8
    assert row['events']==events,('events',row['options'])
    assert row['final']==state,('state',row['options'])
    assert row['final_registers']==registers,('registers',row['options'])
    assert row['returned']==returned and row['error']==(None if returned else 'supplied_base_stop')
    # All apparent field changes belong to the completed supplied base contract.
    if not row['options'].get('base_writes') or not returned:
        assert row['initial']['memory']==row['final']['memory']


class Machine(NativeMachine):
    def __init__(self,game_root,dumper_root):
        manifest=json.loads((Path(__file__).parents[1]/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
        self.input_pins={}
        for key in ['game_assembly','global_metadata']:
            expected=manifest['inputs'][key];raw=(game_root/expected['path']).read_bytes()
            assert len(raw)==expected['size'] and sha(raw).upper()==expected['sha256'].upper()
            self.input_pins[key]={'size':len(raw),'sha256':sha(raw)}
        super().__init__(game_root)
        extraction=json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        texts={}
        for filename,key in [('dump.cs','dump_cs'),('script.json','script_json'),('il2cpp.h','il2cpp_h')]:
            raw=(dumper_root/filename).read_bytes();expected=extraction['outputs'][key]
            assert len(raw)==expected['size'] and sha(raw).upper()==expected['sha256'].upper()
            self.input_pins[key]={'size':len(raw),'sha256':sha(raw)};texts[key]=raw.decode('utf-8-sig')
        metadata=json.loads(texts['script_json']);dump=texts['dump_cs']
        declarations=[
            ('SkinData','public class SkinData : ScriptableObject // TypeDefIndex: 5945'),
            ('ERarity','public enum ERarity // TypeDefIndex: 5847'),
            ('EArtType','public enum EArtType // TypeDefIndex: 5946'),
            ('Color','public struct Color : IEquatable<Color>, IFormattable // TypeDefIndex: 6690'),
            ('ScriptableObject','public class ScriptableObject : Object // TypeDefIndex: 6779')]
        self.declarations={}
        for name,declaration in declarations:
            block=re.search('^'+re.escape(declaration)+r'\s*\{(.*?)\n\}',dump,re.M|re.S);assert block,declaration
            self.declarations[name]={'declaration':declaration,'fields':block[1].split('// Methods')[0].strip()}
            if name=='SkinData':skin=block[1]
            if name=='ScriptableObject':base_body=block[1]
        self.fields={n:int(off,16) for n,off in re.findall(r'\b(\w+); // (0x[\dA-Fa-f]+)',skin.split('// Methods')[0])}
        assert self.fields=={'skinId':0x18,'artistName':0x20,'artistLink':0x28,'skinRarity':0x30,'art':0x38,
                            'animated_art':0x40,'lockedArt':0x48,'type':0x50,'glowColor':0x54,'unlockWith':0x68,
                            'flavor':0x70,'notes':0x78,'skinFor':0x80}
        methods=[line.strip() for line in skin.splitlines() if line.strip().endswith('{ }')]
        assert methods==['public void GenerateSkinId() { }','public bool CheckIfUnlocked() { }','public void UnlockSkin() { }','public void .ctor() { }']
        assert '// RVA: 0x373A40 Offset: 0x372640 VA: 0x180373A40\n\tpublic void .ctor() { }' in skin
        rows=[r for r in metadata['ScriptMethod'] if r['Address']==ENTRY]
        assert len(rows)==29
        selected=[r for r in rows if r['Name']=='SkinData$$.ctor']
        assert len(selected)==1 and selected[0]['Signature']=='void SkinData___ctor (SkinData_o* __this, const MethodInfo* method);' and selected[0]['TypeSignature']=='vii'
        self.target=dict(selected[0],symbol_key='tdi5945.m0003',shared_rva_declaration_count=29,folded_aliases_promoted=False)
        selected=[r for r in metadata['ScriptMethod'] if r['Address']==BASE_CTOR and r['Name']=='UnityEngine.ScriptableObject$$.ctor']
        assert len(selected)==1 and selected[0]['Signature']=='void UnityEngine_ScriptableObject___ctor (UnityEngine_ScriptableObject_o* __this, const MethodInfo* method);' and selected[0]['TypeSignature']=='vii'
        self.supplied=selected[0]
        assert '// RVA: 0x1C8A5C0 Offset: 0x1C891C0 VA: 0x181C8A5C0\n\tpublic void .ctor() { }' in base_body
        assert min(r['Address'] for r in metadata['ScriptMethod'] if r['Address']>ENTRY)==NEXT
        following=[r for r in metadata['ScriptMethod'] if r['Address']==NEXT]
        assert len(following)==1 and following[0]['Name']=='SfxController$$Awake' and following[0]['Signature']=='void SfxController__Awake (SfxController_o* __this, const MethodInfo* method);' and following[0]['TypeSignature']=='vii'
        self.next_declaration=following[0]
        for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            assert not entry.struct.BeginAddress<=ENTRY<entry.struct.EndAddress
        section=self.pe.get_section_by_rva(ENTRY)
        assert section and NEXT<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(ENTRY,NEXT-ENTRY)
        assert len(raw)==NEXT-ENTRY and raw[END-ENTRY:]==b'\xcc'*9 and sha(raw[:END-ENTRY])==BODY_SHA
        ins=list(self.cs.disasm(raw[:END-ENTRY],ENTRY))
        assert len(ins)==2 and sum(i.size for i in ins)==7
        assert (ins[0].address,ins[0].mnemonic,ins[0].op_str)==(ENTRY,'xor','edx, edx')
        assert (ins[1].address,ins[1].mnemonic,ins[1].op_str)==(SITE,'jmp',hex(BASE_CTOR))
        self.instructions={i.address:i for i in ins}
        self.bounds={'entry':hex(ENTRY),'end_exclusive':hex(END),'next_managed':hex(NEXT),'byte_length':7,
                     'instruction_count':2,'sha256':BODY_SHA,'unwind_ranges':[],'alignment_padding':9,'terminal_traps':[]}
        names=['skin0','skin1','skin_class','skin_id','artist','link','flavor','notes','art','animated','locked','unlock','character_data']
        self.p={n:self.arena+0xE80000+i*0x1000 for i,n in enumerate(names)}
        self.entry_sp=self.stack+0x18008
        self.layout={n:(p,256 if n in ['skin0','skin1'] else 512 if n=='character_data' else 128) for n,p in self.p.items()}
        self.layout['native_stack']=(self.entry_sp-0x80,0x100)
        self.tracking=False
        self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE,self.observe_write)

    def observe_write(self,uc,access,address,size,value,user_data):
        if self.tracking:raise AssertionError(('unexpected caller store',hex(address),size,value))

    def registers(self):
        result={n:self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in GPRS}
        result.update({f'XMM{i}':f'{self.reg(getattr(self.x,f"UC_X86_REG_XMM{i}")):032x}' for i in range(16)})
        return result

    def snapshot(self):
        return {'memory':{n:bytes(self.u.mem_read(p,size)).hex() for n,(p,size) in self.layout.items()},
                'native_entries':deepcopy(self.native_entries),'requests':deepcopy(self.history)}

    def prepare(self,options):
        self.native_entries,self.history=[],[]
        for n,(p,size) in self.layout.items():self.u.mem_write(p,bytes([options.get('seed',0xA5)])*size)
        for skin in ['skin0','skin1']:
            self.q(self.p[skin],self.p['skin_class'])
            for name,offset in self.fields.items():
                if name in ['skinRarity','type','glowColor']:continue
                identity={'skinId':'skin_id','artistName':'artist','artistLink':'link','animated_art':'animated','lockedArt':'locked','unlockWith':'unlock','skinFor':'character_data'}.get(name,name)
                self.q(self.p[skin]+offset,0 if options.get('null_fields') else self.p[identity])
            self.d(self.p[skin]+0x30,options.get('rarity_bits',40));self.d(self.p[skin]+0x50,options.get('type_bits',0x8000000A))
            self.u.mem_write(self.p[skin]+0x54,struct.pack('<4I',*options.get('color_bits',[0x80000000,0x7FC01234,0x3F800000,0xFF800000])))

    def hook(self,uc,address,size,user_data):
        if address==self.stop:self.returned=True;uc.emu_stop();return
        rva=address-self.base;self.executed.add(rva)
        if rva==ENTRY:
            registers=self.registers()
            self.native_entries.append({'owner':registers['RCX'],'native_entry':address,**abi(registers)})
            return
        if rva==SITE:return
        assert rva==BASE_CTOR,hex(rva)
        registers=self.registers();assert registers['RDX']==0 and registers['RSP']==self.entry_sp
        caller=self.rq(self.entry_sp);assert caller==self.stop
        event={'kind':'scriptable_object_constructor','ordinal':1,'native_site':hex(SITE),'gateway':address,
               'caller_return_bits':caller,'entry_sp':self.entry_sp,'raw_args':[registers[n] for n in ['RCX','RDX','R8','R9']],
               **abi(registers),'snapshot':self.snapshot()}
        self.events.append(event)
        if self.options.get('stop_base'):self.error='supplied_base_stop';uc.emu_stop();return
        effects=supplied_effects(self.options,self)
        for record,offset,raw in effects:self.u.mem_write(self.layout[record][0]+offset,bytes.fromhex(raw))
        result=self.options.get('return_bits',0xD00D1234567890AB)
        self.history.append({'kind':'scriptable_object_constructor','ordinal':1,'site':hex(SITE),'raw_args':event['raw_args'],
                             'return_bits':result,'writes':effects,'normal_callee_abi':{'entry_sp':self.entry_sp,'return_sp':self.entry_sp+8,
                                'caller':caller,'preserved':{n:registers[n] for n in NONVOL+[f'XMM{i}' for i in range(6,16)]}}})
        for n in VOL[1:]:self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),self.options.get('volatile_poison',0xFACE123456789090))
        for i in range(6):self.u.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),(1 << 127)|i)
        self.u.reg_write(self.x.UC_X86_REG_RAX,result);self.u.reg_write(self.x.UC_X86_REG_RSP,self.entry_sp+8);self.u.reg_write(self.x.UC_X86_REG_RIP,caller)
        now=self.registers();assert all(now[n]==registers[n] for n in NONVOL+[f'XMM{i}' for i in range(6,16)])

    def run(self,options,retained=False):
        self.tracking=False
        if not retained:self.prepare(options)
        self.options=deepcopy(options);self.events=[];self.error=None;self.returned=False
        if not retained:self.u.mem_write(self.entry_sp-0x80,bytes([0xCC])*0x100)
        self.q(self.entry_sp,self.stop)
        registers={n:options.get('register_seed',0xDEAD123400000000)+i for i,n in enumerate(GPRS)}
        registers.update(RCX=0 if options.get('null_owner') else self.p[options.get('owner','skin0')],RDX=options.get('method_bits',0xFFFFFFFF12345678),RSP=self.entry_sp)
        registers.update({f'XMM{i}':f'{((1 << 126)|i):032x}' for i in range(16)})
        for n,v in registers.items():self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n),int(v,16) if n.startswith('XMM') else v)
        initial=self.snapshot();self.tracking=True
        try:self.u.emu_start(self.base+ENTRY,self.stop+0x1000,count=32)
        finally:self.tracking=False
        row={'options':deepcopy(options),'entry_registers':registers,'initial':initial,'events':deepcopy(self.events),
             'final':self.snapshot(),'final_registers':self.registers(),'returned':self.returned,'error':self.error,
             'normal_win64_verified':self.returned,'actual_caller_native_writes':[]}
        model(row,self)
        if self.returned:
            assert row['final_registers']['RSP']==self.entry_sp+8
            assert all(row['final_registers'][n]==registers[n] for n in NONVOL+[f'XMM{i}' for i in range(6,16)])
        return row


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[]
    for owner,seed,method,null_fields in itertools.product(['skin0','skin1'],[0,0xA5,0xFF],[0,0xFFFFFFFF12345678],[False,True]):
        cases.append(m.run({'owner':owner,'seed':seed,'method_bits':method,'null_fields':null_fields}))
    for value in [0,MASK,0x8000000000000080]:cases.append(m.run({'return_bits':value,'rarity_bits':0xFFFFFFFF,'type_bits':0xFFFFFFFF}))
    for color in [[0,0,0,0],[0x3F800000]*4,[0x7FA01234,0xFF800000,0x80000000,1]]:cases.append(m.run({'color_bits':color}))
    for writes in [[['skin0',0x54,'0102030405060708090a0b0c0d0e0f10']], [['skin0',0x48,'0000000000000000']]]:
        cases.append(m.run({'base_writes':writes}))
    cases.append(m.run({'null_owner':True,'stop_base':True}))
    sequences=[]
    for first,second in [({},{}),({'owner':'skin1'},{'owner':'skin0'}),({'stop_base':True},{}),
                         ({'base_writes':[['skin0',0x54,'0102030405060708090a0b0c0d0e0f10']]},{})]:
        rows=[m.run(first),m.run(second,True)]
        assert rows[0]['final']==rows[1]['initial']
        sequences.append(rows)
    baselines=[m.run({}),m.run({'owner':'skin1','method_bits':MASK}),m.run({'base_writes':[['skin0',0x48,'0000000000000000']]})]
    stops=[]
    for i,baseline in enumerate(baselines):
        row=m.run(dict(baseline['options'],stop_base=True))
        assert row['events']==baseline['events'] and row['final']==baseline['events'][0]['snapshot']
        assert row['final_registers']==baseline['events'][0]['all_registers']
        stops.append({'baseline':i,'prefix_length':1,'result':row})
    assert set(m.instructions)<=m.executed
    raw={'schema':'skin_data_constructor_native_v1','build':BUILD,'input_pins':m.input_pins,'target':m.target,
         'body_bounds':m.bounds,'selected_operand_assertions':{hex(ENTRY):['xor','edx, edx']},
         'next_managed_declaration_not_executed':m.next_declaration,
         'tail_gateway':{'native_site':hex(SITE),'target_rva':hex(BASE_CTOR),'caller_return_retained':True},
         'supplied_base_declaration':m.supplied,'class_declarations_and_fields':m.declarations,'skin_field_offsets':m.fields,
         'physical_layout':{n:{'pointer':p,'size':size} for n,(p,size) in m.layout.items()},
         'cases':cases,'sequences':sequences,'baselines':baselines,'stops':stops,
         'summary':{'cases':len(cases),'normal_returns':sum(r['returned'] for r in cases),'supplied_stops':sum(not r['returned'] for r in cases),
                    'sequences':len(sequences),'baselines':len(baselines),'stops':len(stops),'decoded_instructions':2,'executed_instructions':2,
                    'physical_records':len(m.layout),'independently_compared_rows':len(cases)+8+len(baselines)+len(stops)},
         'scope':'Exact two-instruction SkinData nominal folded wrapper only. ScriptableObject constructor wholly supplied; no owner reads/stores/metadata/Color initialization in caller. Other folded declarations, runtime construction/admission, allocation defaults, base implementation and proprietary complete native export excluded. Null owner only reaches a supplied stop, no managed runtime admission asserted.'}
    verify_history_codec();verify_memory_map_codec();verify_state_map_codec();verify_full_report_codec()
    encoded=pool_snapshots(pool_state_maps(pool_memory_maps(pool_histories(pool_memory(raw)))))
    assert expand_report(encoded)==raw
    return encoded


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('game_root',type=Path);parser.add_argument('dumper_root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.game_root,args.dumper_root)
    raw=(json.dumps(report,separators=(',',':'),ensure_ascii=True)+'\n').encode('utf-8')
    assert len(raw)<100*1024*1024;args.output.write_bytes(raw)
    print(json.dumps({'summary':report['summary'],'bytes':len(raw),'sha256':sha(raw)}))
