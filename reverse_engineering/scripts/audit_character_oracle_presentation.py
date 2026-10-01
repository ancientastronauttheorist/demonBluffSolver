"""Execute OracleEyeActive/HideOracleInfo callers with named supplied services."""
import argparse
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_character_presentation_helpers import Machine as PresentationMachine

TARGETS = {0x3675A0: ('OracleEyeActive', 'tdi5487.m0030', 0x36778D),
           0x3654F0: ('HideOracleInfo', 'tdi5487.m0031', 0x36563D)}
SUPPLIED = {0x35DD10: 'Acted$$Act', 0x3A71C0: 'RevealOrder$$Init',
            0x3A7190: 'RevealOrder$$Hide', 0x364C40: 'Character$$GetCharacterBluffIfAble',
            0x363D50: 'CharacterView$$AnimateIn', 0x363DF0: 'CharacterView$$AnimateOut',
            0x363F10: 'CharacterView$$Init', 0x1C82480: 'UnityEngine.Object$$op_Inequality',
            0x1C79FD0: 'UnityEngine.Component$$get_gameObject',
            0x1C7DC50: 'UnityEngine.GameObject$$get_activeSelf',
            0x1C7D810: 'UnityEngine.GameObject$$SetActive'}


class Machine(PresentationMachine):
    def __init__(self, game_root, dumper_root):
        import capstone
        super().__init__(game_root, dumper_root)
        self.oracle_targets, self.oracle_instructions, self.oracle_ranges = [], {}, {}
        references = set()
        for start, (name, method_id, end) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == 'Character$$' + name and r['Address'] == start]
            assert len(rows) == 1 and rows[0]['Signature'] == f'void Character__{name} (Character_o* __this, const MethodInfo* method);'
            assert rows[0]['TypeSignature'] == 'vii'
            self.oracle_targets.append(dict(rows[0], method_id=method_id))
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append((entry.struct.BeginAddress, entry.struct.EndAddress))
            assert chunks and min(a for a, _ in chunks) == start and max(b for _, b in chunks) >= end
            following = min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start)
            assert following == (0x367790 if name == 'OracleEyeActive' else 0x365640)
            assert self.pe.get_data(end, following-end) == bytes([0xCC])*(following-end)
            ins = list(self.cs.disasm(self.pe.get_data(start, end-start), start))
            assert sum(i.size for i in ins) == end-start
            self.oracle_ranges[hex(start)] = [[hex(a), hex(b)] for a,b in chunks]
            self.oracle_instructions.update({i.address:i for i in ins})
            for i in ins:
                for op in i.operands:
                    if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                        references.add(i.address+i.size+op.mem.disp)
                        if i.mnemonic == 'cmp' and i.operands[0].size == 1: self.flags.add(i.address+i.size+op.mem.disp)
        self.instructions.update(self.oracle_instructions)
        for row in self.metadata['ScriptMetadata'] + self.metadata['ScriptMetadataMethod']:
            if row['Address'] in references:
                if row['Name'] not in self.bindings: self.bindings[row['Name']] = self.arena+0x6000+len(self.bindings)*0x200
                token = self.bindings[row['Name']]
                self.metadata_slots[self.base+row['Address']] = token
                self.q(self.base+row['Address'], token); self.ids[token] = row['Name']
        self.list_item_method = 'Method$System.Collections.Generic.List<ActedInfo>.get_Item()'
        assert self.list_item_method in self.bindings
        self.supplied_metadata = []
        for address, name in SUPPLIED.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address']==address and r['Name']==name]
            assert len(rows)==1; self.supplied_metadata += rows
        assert any(r['Address']==0xB22150 and r['Name']=='System.Collections.Generic.List<object>$$get_Item' for r in self.metadata['ScriptMethod'])
        self.field_pins = [
            ('Character',5487,['public Acted acteds; // 0xA8','private int pickableUses; // 0xDC','public ECharacterState state; // 0xE4','public bool killedByDemon; // 0xED','public GameObject pickHighlight; // 0x138','public CharacterView charBluff; // 0x140','public List<ActedInfo> actedInfos; // 0x148','public RevealOrder revealOrder; // 0x158','private int order; // 0x160']),
            ('Acted',5477,['public GameObject highlight; // 0x30','public Image arrowImage; // 0x38','private Color savedArrowColor; // 0x40']),
            ('ActedInfo',5498,['public string desc; // 0x10']),
            ('CharacterData',5845,['public bool picking; // 0x13E'])]
        for name,index,fields in self.field_pins:
            m=re.search(r'^public class '+name+r'(?: :[^\n]*)? // TypeDefIndex: '+str(index)+r'\s*\{(.*?)// (?:Properties|Methods)',self.dump,re.M|re.S)
            assert m and all(f in m[1] for f in fields),name
        m=re.search(r'^public enum ECharacterState // TypeDefIndex: 5489\s*\{(.*?)\n\}',self.dump,re.M|re.S)
        assert m and all(f'public const ECharacterState {name} = {value};' in m[1] for name,value in [('Hidden',5),('Alive',10),('Dead',20),('Revealed',30)])
        self.checks_oracle = {
            0x3675DD:('cmp','dword ptr [rbx + 0xe4], 5'),0x367619:('cmp','dword ptr [rcx + 0x18], 1'),
            0x36762C:('mov','rdi, qword ptr [rbx + 0xa8]'),0x367679:('mov','dl, 1'),
            0x367690:('lea','rcx, [rsp + 0x20]'),0x36769C:('mov','r8, qword ptr [r8 + 0x2a0]'),
            0x3676B3:('movups','xmmword ptr [rdi + 0x40], xmm0'),0x3676D5:('call','qword ptr [rax + 0x2a8]'),
            0x3676EE:('cmp','byte ptr [rax + 0x13e], 0'),0x3676F7:('cmp','dword ptr [rbx + 0xdc], 0'),
            0x367716:('cmp','dword ptr [rbx + 0xe4], 0x14'),0x36774E:('test','al, al'),
            0x365561:('dec','edx'),0x365563:('mov','rdi, qword ptr [rbx + 0xa8]'),
            0x3655C8:('movups','xmm0, xmmword ptr [rdi + 0x40]'),0x365616:('test','al, al')}
        assert all((self.oracle_instructions[a].mnemonic,self.oracle_instructions[a].op_str)==v for a,v in self.checks_oracle.items())
        i=self.oracle_instructions[0x3676A9];op=i.operands[-1]
        assert op.type==capstone.CS_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP and op.size==16
        literal=i.address+i.size+op.mem.disp;section=self.pe.get_section_by_rva(literal)
        assert section and literal-section.VirtualAddress+16<=section.SizeOfRawData
        raw=self.pe.get_data(literal,16);assert len(raw)==16
        self.color_literal={'rva':hex(literal),'bits':list(struct.unpack('<IIII',raw))}
        assert self.color_literal=={'rva':'0x1f34cb0','bits':[0,0x3F800000,0x3F800000,0x3F800000]}
        for i,name in enumerate(['acted','other_acted','history','info0','info1','info2','speech0','speech1','speech2','description_game','pick_game','view_game','arrow','other_arrow','image_class','color_get_method','color_set_method','reveal','history_array']):
            self.p[name]=self.arena+0x50000+i*0x1000;self.ids[self.p[name]]=name
        self.color_get,self.color_set=self.stop+0x240,self.stop+0x250
        self.oracle_ready=False

    def snapshot(self):
        out=super().snapshot()
        if not getattr(self,'oracle_ready',False):return out
        a=self.p['actor']
        out['oracle']={'state_bits':self.rd(a+0xE4),'uses_bits':self.rd(a+0xDC),'order_bits':self.rd(a+0x160),
            **{name:self.oid(self.rq(a+off)) for name,off in [('acted',0xA8),('history',0x148),('reveal',0x158),('pick',0x138)]},
            'count_bits':self.rd(self.p['history']+0x18),
            'data_picking_bits':{n:self.byte(self.p[n]+0x13E) for n in ['data','bluff']},
            'acted_fields':{n:{'game':self.oid(self.rq(self.p[n]+0x30)),'arrow':self.oid(self.rq(self.p[n]+0x38)),'saved_color_bits':[self.rd(self.p[n]+0x40+i*4) for i in range(4)]} for n in ['acted','other_acted']},
            'images':{n:v.copy() for n,v in self.images.items()},'games':self.games.copy(),'reveal_requests':self.reveal_requests.copy(),'speech_requests':self.speech_requests.copy(),
            'memory':{n:bytes(self.u.mem_read(self.p[n],size)).hex() for n,size in [('actor',0x200),('acted',0x80),('other_acted',0x80),('history',0x40),('history_array',0x40),('info0',0x30),('info1',0x30),('info2',0x30),('data',0x160),('bluff',0x160),('arrow',0x40),('other_arrow',0x40),('view',0x100)]}}
        return out

    def prepare(self, options):
        self.oracle_ready=False
        super().prepare(options)
        a=self.p['actor'];self.images={n:[0x80000000,0x7FC01234,0x3F000001,0x3F800000] for n in ['arrow','other_arrow']}
        self.games={'description_game':False,'pick_game':False,'view_game':options.get('view_active',True)}
        self.reveal_requests,self.speech_requests=[],[]
        for off,name in [(0xA8,'acted'),(0x138,'pick_game'),(0x148,'history'),(0x158,'reveal')]:self.q(a+off,0 if options.get('null_'+name) else self.p[name])
        self.d(a+0xE4,options.get('state',20));self.d(a+0xDC,options.get('uses_bits',1));self.d(a+0x160,options.get('order_bits',0x80000003))
        self.d(self.p['history']+0x18,options.get('count_bits',2))
        self.q(self.p['history']+0x10,self.p['history_array']);self.d(self.p['history']+0x1C,0xFFFFFFFF)
        self.q(self.p['history_array']+0x18,3)
        for index in range(3):self.q(self.p['history_array']+0x20+8*index,self.p['info'+str(index)])
        for index in range(3):self.q(self.p['info'+str(index)]+0x10,0 if options.get('null_text') else self.p['speech'+str(index)])
        for n in ['data','bluff']:self.u.mem_write(self.p[n]+0x13E,bytes([options.get('picking_bits',0x80)]))
        self.q(self.p['image_class']+0x298,self.color_get);self.q(self.p['image_class']+0x2A0,self.p['color_get_method'])
        self.q(self.p['image_class']+0x2A8,self.color_set);self.q(self.p['image_class']+0x2B0,self.p['color_set_method'])
        for n in ['arrow','other_arrow']:self.q(self.p[n],self.p['image_class'])
        for n in ['acted','other_acted']:
            self.q(self.p[n]+0x30,0 if options.get('null_description_game') else self.p['pick_game' if options.get('alias_games') else 'description_game'])
            self.q(self.p[n]+0x38,0 if options.get('null_arrow') else self.p['arrow' if n=='acted' or options.get('alias_arrows') else 'other_arrow'])
            for i,b in enumerate([0x3E000001,0x80000000,0x7FC0ABCD,0x3F800000]):self.d(self.p[n]+0x40+i*4,b)
        self.oracle_ready=True;self.memory_mutations={}

    def mutation(self, phase):
        if self.options.get('mutation_phase')!=phase:return
        name=self.options['mutation'];a=self.p['actor']
        if name in ['clear_acted','other_acted','clear_view','bluff_to_data','clear_pick']:
            off={'clear_acted':0xA8,'other_acted':0xA8,'clear_view':0x140,'bluff_to_data':0x58,'clear_pick':0x138}[name]
            value=self.p['other_acted'] if name=='other_acted' else self.p['data'] if name=='bluff_to_data' else 0
            self.q(a+off,value);self.authored_offsets.update(range(off,off+8))
        elif name=='clear_arrow':
            n=self.oid(self.rq(a+0xA8));self.q(self.p[n]+0x38,0);self.memory_mutations.setdefault(n,set()).update(range(0x38,0x40))
        elif name=='count_one':self.d(self.p['history']+0x18,1);self.memory_mutations.setdefault('history',set()).update(range(0x18,0x1C))
        else:raise AssertionError(name)

    def hook(self,uc,address,size,data):
        if not getattr(self,'oracle_ready',False):return super().hook(uc,address,size,data)
        rva,x=address-self.base,self.x;self.executed.add(rva)
        if rva in self.instructions:return
        cx,dx,r8=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8']]
        if rva==0xB22150:
            assert cx==self.p['history'] and r8==self.bindings[self.list_item_method]
            index=dx&0xFFFFFFFF;assert index<3 and dx==index
            result=0 if self.options.get('null_info') else self.p['info'+str(index)]
            if self.event('list_item_service',[self.oid(cx),index,self.oid(r8),self.oid(result)]):self.mutation('item');self.ret(result)
        elif rva==0x35DD10:
            assert cx in [self.p['acted'],self.p['other_acted']] and dx in [0,*[self.p['speech'+str(i)] for i in range(3)]] and r8==0
            args=[self.oid(cx),self.oid(dx)]
            if self.event('supplied_acted_act',args):self.speech_requests.append(args);self.mutation('act');self.ret()
        elif rva in [0x3A71C0,0x3A7190]:
            assert cx==self.p['reveal'] and (dx==self.rd(self.p['actor']+0x160) and r8==0 if rva==0x3A71C0 else dx==0)
            args=[self.oid(cx),dx&0xFFFFFFFF] if rva==0x3A71C0 else [self.oid(cx)]
            if self.event('supplied_reveal_init' if rva==0x3A71C0 else 'supplied_reveal_hide',args):self.reveal_requests.append(args);self.mutation('reveal');self.ret()
        elif rva==0x364C40:
            assert cx==self.p['actor'] and dx==0
            result=0 if self.options.get('null_selected_data') else self.p[self.options.get('selected_data','data')]
            if self.event('supplied_get_bluff_if_able',[self.oid(cx),self.oid(result)]):self.mutation('selected');self.ret(result)
        elif rva==0x1C82480:
            assert cx in [0,self.p['data'],self.p['bluff']] and dx==r8==0
            result=self.options.get('inequality_return_bits',0xFACE000000000000|int(cx!=0 and not self.options.get('destroyed_bluff')))
            if self.event('unity_inequality_service',[self.oid(cx),result]):self.mutation('inequality');self.ret(result)
        elif rva==0x1C7D810:
            assert cx in [self.p[n] for n in self.games] and r8==0 and dx&255 in [0,1]
            args=[self.oid(cx),dx,dx&255]
            if self.event('set_active_service',args):self.games[self.oid(cx)]=bool(dx&255);self.mutation('active');self.ret()
        elif address==self.color_get:
            assert self.stack<=cx<self.stack+0x20000 and dx in [self.p['arrow'],self.p['other_arrow']] and r8==self.p['color_get_method']
            captured=self.reg(x.UC_X86_REG_RDI);assert captured in [self.p['acted'],self.p['other_acted']]
            bits=self.images[self.oid(dx)].copy()
            if self.event('color_get_service',[cx-self.stack,self.oid(dx),self.oid(r8),bits,self.oid(captured)]):uc.mem_write(cx,struct.pack('<IIII',*bits));self.mutation('color_get');self.ret(cx)
        elif address==self.color_set:
            assert cx in [self.p['arrow'],self.p['other_arrow']] and self.stack<=dx<self.stack+0x20000 and r8==self.p['color_set_method']
            bits=list(struct.unpack('<IIII',uc.mem_read(dx,16)))
            if self.event('color_set_service',[self.oid(cx),bits,self.oid(r8)]):self.images[self.oid(cx)]=bits;self.mutation('color_set');self.ret()
        elif rva==0x1C79FD0:
            assert cx==self.p['view'] and dx==0
            result=0 if self.options.get('null_view_game') else self.p['view_game']
            if self.event('view_game_object_service',[self.oid(cx),self.oid(result)]):self.mutation('view_game');self.ret(result)
        elif rva==0x1C7DC50:
            assert cx==self.p['view_game'] and dx==0
            result=self.options.get('active_return_bits',0xFACE000000000000|int(self.games['view_game']))
            if self.event('active_self_service',[self.oid(cx),result]):self.mutation('active_self');self.ret(result)
        elif rva in [0x363D50,0x363DF0,0x363F10]:
            assert cx==self.p['view'] and (dx in [0,self.p['bluff'],self.p['data']] and r8==0 if rva==0x363F10 else dx==0)
            name={0x363D50:'AnimateIn',0x363DF0:'AnimateOut',0x363F10:'Init'}[rva]
            args=[self.oid(cx),self.oid(dx)] if name=='Init' else [self.oid(cx)]
            if self.event('supplied_view_'+name,args):
                if name=='Init':self.view_state['data']=self.oid(dx)
                else:self.view_state['active']=name=='AnimateIn'
                self.mutation('view_'+name);self.ret()
        else:return super().hook(uc,address,size,data)

    def run_oracle(self,name,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options,self.error=options or {},None
        self.authored_offsets=set();self.memory_mutations={};initial=self.snapshot();old=len(self.events)
        returned=self.invoke(next(a for a,(n,_,_) in TARGETS.items() if n==name))
        final=self.snapshot()
        new_events=self.events[old:]
        gets=[e for e in new_events if e['kind']=='color_get_service']
        if gets and self.error!='color_get_service':
            captured=gets[0]['args'][4]
            assert final['oracle']['acted_fields'][captured]['saved_color_bits']==gets[0]['args'][3]
        if returned and not self.options.get('mutation_phase'):
            count=initial['oracle']['count_bits'];count=count if count<0x80000000 else count-0x100000000
            speech=[e for e in new_events if e['kind']=='supplied_acted_act']
            expected_index=0 if name=='OracleEyeActive' else count-1
            assert [e['args'][1] for e in speech]==([None if self.options.get('null_text') else 'speech'+str(expected_index)] if count>1 else [])
            kinds=[e['kind'] for e in new_events]
            if name=='OracleEyeActive':
                assert kinds.count('supplied_reveal_init')==int(initial['oracle']['state_bits']!=5)
                assert kinds.count('color_get_service')==int(count>1)
                selected=self.options.get('selected_data','data')
                expected_pick=initial['oracle']['data_picking_bits'][selected]!=0 and 0<initial['oracle']['uses_bits']<0x80000000
                if expected_pick:assert final['oracle']['games']['pick_game']
                inequality=[e for e in new_events if e['kind']=='unity_inequality_service']
                assert len(inequality)==int(initial['oracle']['state_bits']==20 and initial['actor']['killed_by_demon_bits']==0)
                show=bool(inequality and inequality[0]['args'][1]&255)
                assert kinds.count('supplied_view_AnimateIn')==int(show) and kinds.count('supplied_view_Init')==int(show)
                if count>1:assert [e['args'][1] for e in new_events if e['kind']=='color_set_service']==[self.color_literal['bits']]
            else:
                assert kinds.count('supplied_reveal_hide')==1 and not final['oracle']['games']['pick_game']
                active=[e for e in new_events if e['kind']=='active_self_service'];assert len(active)==1
                assert kinds.count('supplied_view_AnimateOut')==int(active[0]['args'][1]&255!=0)
                if count>1:assert [e['args'][1] for e in new_events if e['kind']=='color_set_service']==[initial['oracle']['acted_fields']['acted']['saved_color_bits']]
        for n,raw in initial['oracle']['memory'].items():
            before,after=bytes.fromhex(raw),bytes.fromhex(final['oracle']['memory'][n])
            allowed=self.authored_offsets if n=='actor' else self.memory_mutations.get(n,set())| (set(range(0x40,0x50)) if name=='OracleEyeActive' and n in ['acted','other_acted'] else set())
            assert all(i in allowed or b==after[i] for i,b in enumerate(before)),n
        return {'method':name,'options':self.options.copy(),'returned':returned,'error':self.error,'initial':initial,'events':new_events.copy(),'final':final,'unconsumed_memory_retained':True}


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases,sequences,baselines,stops=[],[],[],[]
    for state,count,uses,picking in itertools.product([5,10,20,30],[0,1,2],[0,1,0xFFFFFFFF],[0,0x80]):
        r=m.run_oracle('OracleEyeActive',{'state':state,'count_bits':count,'uses_bits':uses,'picking_bits':picking});assert r['returned'];cases.append(r)
    for options in [{'cold':True,'class_cold':True},{'warm_byte':0x80,'class_word':0xDEADBEEF},{'killed':0x80},{'bluff':'absent'},{'destroyed_bluff':True},{'same_data_bluff':True},{'selected_data':'bluff'},{'null_text':True},{'alias_games':True},{'count_bits':0x80000000},{'inequality_return_bits':0xFACE000000000000},{'inequality_return_bits':0xFACE000000000080}]:
        r=m.run_oracle('OracleEyeActive',options);assert r['returned'];cases.append(r)
    for count,active in itertools.product([0,1,2,3],[False,True]):
        r=m.run_oracle('HideOracleInfo',{'count_bits':count,'view_active':active});assert r['returned'];cases.append(r)
    for options in [{'cold':True},{'active_return_bits':0xFACE000000000000},{'active_return_bits':0xFACE000000000080},{'alias_games':True},{'null_text':True},{'count_bits':0x80000000}]:
        r=m.run_oracle('HideOracleInfo',options);assert r['returned'];cases.append(r)
    for name in ['OracleEyeActive','HideOracleInfo']:
        for field in ['reveal','history','acted','description_game','arrow','view']:
            options={'null_'+field:True}
            if name=='OracleEyeActive' and field=='reveal':options['state']=10
            r=m.run_oracle(name,options);assert not r['returned'] and r['error']=='native_null_guard';cases.append(r)
        for options in [{'null_info':True},{'null_selected_data':True},{'null_pick_game':True}] if name=='OracleEyeActive' else [{'null_info':True},{'null_pick_game':True},{'null_view_game':True}]:
            r=m.run_oracle(name,options);assert not r['returned'] and r['error']=='native_null_guard';cases.append(r)
    for name,phase,mutation in [('OracleEyeActive','item','clear_acted'),('OracleEyeActive','item','other_acted'),('OracleEyeActive','act','clear_acted'),('OracleEyeActive','act','other_acted'),('OracleEyeActive','active','other_acted'),('OracleEyeActive','color_get','clear_arrow'),('OracleEyeActive','view_AnimateIn','clear_view'),('OracleEyeActive','inequality','bluff_to_data'),('OracleEyeActive','reveal','count_one'),('HideOracleInfo','item','other_acted'),('HideOracleInfo','act','clear_acted'),('HideOracleInfo','color_set','clear_pick'),('HideOracleInfo','active_self','clear_view')]:
        options={'mutation_phase':phase,'mutation':mutation};r=m.run_oracle(name,options)
        assert r['returned']==(mutation not in ['clear_acted','clear_arrow','clear_view','clear_pick']), (name,phase,mutation,r['returned'])
        cases.append(r)
    for alias in [False,True]:
        m.prepare({'cold':True,'class_cold':True,'alias_games':alias})
        sequences.append([m.run_oracle(n,retained=True) for n in ['OracleEyeActive','HideOracleInfo','OracleEyeActive']])
    for name in ['OracleEyeActive','HideOracleInfo']:
        baseline=m.run_oracle(name,{'cold':True,'class_cold':True});assert baseline['returned'];baselines.append(baseline);counts={}
        for index,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1
            stopped=m.run_oracle(name,{'cold':True,'class_cold':True,'failure':[kind,counts[kind]]})
            assert not stopped['returned'] and stopped['events']==baseline['events'][:index+1] and stopped['final']==e['snapshot']
            stops.append({'method':name,'failure':[kind,counts[kind]],'exact_prefix_verified':True})
    missing=set(m.oracle_instructions)-m.executed;assert not missing
    return {'build':BUILD,'targets':m.oracle_targets,'ranges':m.oracle_ranges,'field_pins':m.field_pins,'supplied_metadata':m.supplied_metadata,'instruction_assertions':len(m.checks_oracle),'color_literal':m.color_literal,'case_count':len(cases),'cases':cases,'retained_sequences':sequences,'failure_baselines':baselines,'failure_case_count':len(stops),'failures':stops,'caller_instructions_executed':len(m.oracle_instructions),'native_execution_addresses':len(m.executed),'scope':'Actual OracleEyeActive/HideOracleInfo callers only. Direct signed List count DWORD; GetItem physical outcomes supplied. Acted.Act, RevealOrder, GetCharacterBluffIfAble, CharacterView bodies, Unity/color virtual methods, metadata/class work are supplied. No renderer/scheduler/exception unwinding; raw retained memory is authored diagnostic storage, not complete valid typed runtime objects.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True)
    args=p.parse_args();r=audit(args.game_root,args.dumper_root);args.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:r[k] for k in ['case_count','failure_case_count','caller_instructions_executed','native_execution_addresses']}))
