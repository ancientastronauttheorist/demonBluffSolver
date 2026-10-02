"""Exact CharacterData.GetDescription caller; converter/runtime supplied."""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
from pathlib import Path
import re

from audit_character_assets import BUILD
from audit_character_data_consumers import Machine as DataVerifier
from audit_character_oracle_reveal_join import expand_memory, pool_memory
from audit_report_snapshots import expand_snapshots, pool_snapshots

ENTRY, END, FOLLOWING = 0x3B4BF0, 0x3B4C98, 0x3B4CA0
FLAG, SLOT = 0x288C4E1, 0x271F268
POISON = 0xFACE123456789090


class Machine(DataVerifier):
    def __init__(self, game_root, dumper_root):
        super().__init__(game_root, dumper_root)
        raw = (Path(dumper_root)/'dump.cs').read_bytes()
        extraction = json.loads((Path(__file__).parents[1]/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        assert hashlib.sha256(raw).hexdigest().upper() == extraction['outputs']['dump_cs']['sha256'].upper()
        dump = raw.decode('utf-8-sig')
        for pattern, fields in [
            (r'^public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845', ['public string description; // 0x50', 'public string descriptionPL; // 0x58', 'public string descriptionCHN; // 0x60']),
            (r'^public class ProjectContext : MonoBehaviour // TypeDefIndex: 5546', ['public GameData gameData; // 0x20', 'public static ProjectContext Instance; // 0x0']),
            (r'^public class GameData : ScriptableObject // TypeDefIndex: 5928', ['public ELanguage language; // 0x20']),
            (r'^public enum ELanguage // TypeDefIndex: 5985', ['public const ELanguage English = 0;', 'public const ELanguage Polish = 10;']),
        ]:
            block = re.search(pattern+r'\s*\{(.*?)\n\}', dump, re.M | re.S); assert block
            assert all(f in block[1] for f in fields)
        rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == ENTRY]
        assert len(rows) == 1 and rows[0]['Name'] == 'CharacterData$$GetDescription'
        assert rows[0]['Signature'] == 'System_String_o* CharacterData__GetDescription (CharacterData_o* __this, const MethodInfo* method);' and rows[0]['TypeSignature'] == 'iii'
        self.targets = [dict(rows[0], method_id='tdi5845.m0015')]
        assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > ENTRY) == FOLLOWING
        chunks = []
        for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root = e
            while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
            if root.struct.BeginAddress == ENTRY: chunks.append((e.struct.BeginAddress, e.struct.EndAddress))
        assert chunks == [(ENTRY, END)]
        section = self.pe.get_section_by_rva(ENTRY); assert section and FOLLOWING <= section.VirtualAddress+section.SizeOfRawData
        body = self.pe.get_data(ENTRY, FOLLOWING-ENTRY)
        assert len(body) == FOLLOWING-ENTRY and body[END-ENTRY:] == b'\xcc'*(FOLLOWING-END)
        ins = list(self.cs.disasm(body[:END-ENTRY], ENTRY)); assert sum(i.size for i in ins) == END-ENTRY
        self.instructions = {i.address:i for i in ins}
        self.bounds = {'start':hex(ENTRY),'end_exclusive':hex(END),'next_managed':hex(FOLLOWING),'unwind_chunks':[[hex(a),hex(b)] for a,b in chunks], 'body_sha256':hashlib.sha256(body[:END-ENTRY]).hexdigest(), 'byte_length':END-ENTRY}
        self.checks = {
            0x3B4BF6:('cmp','byte ptr [rip + 0x24d78e4], 0'),
            0x3B4C02:('lea','rcx, [rip + 0x236a65f]'),
            0x3B4C0E:('mov','byte ptr [rip + 0x24d78cc], 1'),
            0x3B4C15:('mov','rcx, qword ptr [rbx + 0x50]'),
            0x3B4C20:('mov','r8, qword ptr [rip + 0x236a641]'),
            0x3B4C2A:('mov','rcx, qword ptr [r8 + 0xb8]'),
            0x3B4C31:('mov','rax, qword ptr [rcx]'),
            0x3B4C39:('mov','rax, qword ptr [rax + 0x20]'),
            0x3B4C42:('cmp','dword ptr [rax + 0x20], 0'),
            0x3B4C48:('mov','rcx, qword ptr [rbx + 0x50]'),
            0x3B4C53:('mov','r8, qword ptr [rip + 0x236a60e]'),
            0x3B4C5D:('mov','rax, qword ptr [r8 + 0xb8]'),
            0x3B4C64:('mov','rax, qword ptr [rax]'),
            0x3B4C6C:('mov','rax, qword ptr [rax + 0x20]'),
            0x3B4C75:('cmp','dword ptr [rax + 0x20], 0xa'),
            0x3B4C7B:('mov','rcx, qword ptr [rbx + 0x50]'),
            0x3B4C89:('mov','rax, rdx'), 0x3B4C92:('call','0x2b7d90'), 0x3B4C97:('int3',''),
        }
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str) == v for a,v in self.checks.items())
        assert [i.address for i in ins if i.mnemonic == 'call' and i.op_str == '0x3a83c0'] == [0x3B4C1B,0x3B4C4E,0x3B4C81]
        slots = [r for r in self.metadata['ScriptMetadata'] if r['Address'] == SLOT]
        assert len(slots) == 1 and slots[0]['Name'] == 'ProjectContext_TypeInfo'
        self.supplied = [r for r in self.metadata['ScriptMethod'] if r['Address'] == 0x3A83C0]
        assert len(self.supplied) == 1 and self.supplied[0]['Name'] == 'StringHelper$$ConvertTextToTextWithTooltips'
        assert self.supplied[0]['Signature'] == 'System_String_o* StringHelper__ConvertTextToTextWithTooltips (System_String_o* inputText, const MethodInfo* method);'
        names = ['data','other_data','data_class','project_class','other_class','statics','other_statics','context','other_context','game','other_game', 'description','polish_description','chinese_description','replacement','result0','result1','result2']
        self.p = {n:self.arena+0x200000+i*0x1000 for i,n in enumerate(names)}; self.ids = {p:n for n,p in self.p.items()}
        self.sizes = {n:512 if n in ['data','other_data'] else 256 if n.endswith('class') else 128 for n in names}

    def snapshot(self):
        return {'metadata_flag_byte':self.u.mem_read(self.base+FLAG,1)[0], 'metadata_slot':self.oid(self.rq(self.base+SLOT)),
                'native_entries':deepcopy(self.entries), 'service_history':deepcopy(self.history),
                'memory':{n:bytes(self.u.mem_read(p,self.sizes[n])).hex() for n,p in self.p.items()}}

    def prepare(self, options):
        self.options = deepcopy(options); self.events,self.counts,self.entries,self.history,self.allowed = [],{},[],[],{}
        self.error = None
        for n,p in self.p.items(): self.u.mem_write(p,b'\xa5'*self.sizes[n])
        for n in ['data','other_data']:
            self.q(self.p[n],self.p['data_class'])
            for off,key in [(0x50,'description'),(0x58,'polish_description'),(0x60,'chinese_description')]: self.q(self.p[n]+off,0 if options.get('null_description') and off == 0x50 else self.p[key])
        for cls,sta,ctx,game in [('project_class','statics','context','game'),('other_class','other_statics','other_context','other_game')]:
            self.q(self.p[cls]+0xB8,0 if options.get('null_statics') else self.p[sta]); self.d(self.p[cls]+0xE0,options.get('class_word',0))
            self.q(self.p[sta],0 if options.get('null_context') else self.p[ctx]); self.q(self.p[ctx]+0x20,0 if options.get('null_game') else self.p[game])
            self.d(self.p[game]+0x20,options.get('language_bits',0) if game == 'game' else options.get('other_language_bits',10))
        self.q(self.base+SLOT,0 if options.get('null_class') else self.p['project_class'])
        self.u.mem_write(self.base+FLAG,bytes([options.get('warm_flag',0)]))

    def mutate(self, phase):
        actions = self.options.get('mutations',{}).get(phase,[])
        if isinstance(actions,str): actions = [actions]
        def write(n,off,width,value):
            self.u.mem_write(self.p[n]+off,value.to_bytes(width,'little')); self.allowed.setdefault(n,set()).update(range(off,off+width))
        for action in actions:
            if action == 'replace_class': self.q(self.base+SLOT,self.p['other_class'])
            elif action == 'clear_class': self.q(self.base+SLOT,0)
            elif action == 'replace_statics': write('project_class',0xB8,8,self.p['other_statics'])
            elif action == 'clear_statics': write('project_class',0xB8,8,0)
            elif action == 'replace_context': write('statics',0,8,self.p['other_context'])
            elif action == 'clear_context': write('statics',0,8,0)
            elif action == 'replace_game': write('context',0x20,8,self.p['other_game'])
            elif action == 'clear_game': write('context',0x20,8,0)
            elif action.startswith('language:'): write('game',0x20,4,int(action.partition(':')[2],0))
            elif action in ['replace_description','clear_description']: write('data',0x50,8,self.p['replacement'] if action.startswith('replace') else 0)
            else: raise AssertionError(action)

    def event(self,kind,args):
        ordinal = self.counts.get(kind,0)+1; self.counts[kind] = ordinal
        self.events.append(dict(kind=kind,ordinal=ordinal,args=deepcopy(args),snapshot=self.snapshot(),raw_args=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']],native_site=hex(self.last_native),caller_return_bits=self.rq(self.reg(self.x.UC_X86_REG_RSP))))
        relative = ordinal-self.entry_counts.get(kind,0)
        if self.options.get('failure') == [kind,relative]: self.error = kind; self.u.emu_stop(); return False
        return True

    def hook(self,uc,address,size,unused):
        if address == self.stop: return
        rva,x = address-self.base,self.x; self.executed.add(rva)
        if rva in self.instructions:
            assert rva != 0x3B4C97; self.last_native = rva
            if rva == ENTRY: self.entries.append([self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']])
            return
        cx,dx,r8,r9 = [self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
        if rva == 0x2B7B40:
            assert cx == self.base+SLOT and self.last_native == 0x3B4C09
            if self.event('metadata',[hex(SLOT),self.oid(self.rq(cx))]): self.history.append(['metadata',self.oid(self.rq(cx))]); self.mutate('metadata'); self.ret()
        elif rva == 0x3A83C0:
            ordinal = self.counts.get('converter',0)-self.entry_counts.get('converter',0)+1
            assert ordinal in [1,2,3] and dx == 0 and cx == self.rq(self.owner+0x50)
            outputs = self.options.get('converter_results',['result0','result1','result2']); result = outputs[ordinal-1]
            assert result is None or result in self.p
            if self.event('converter',[self.oid(cx),0,result]): self.history.append(['converter',self.oid(cx),result]); self.mutate('converter:'+str(ordinal)); self.ret(0 if result is None else self.p[result])
        elif rva == 0x2B7D90: self.event('native_null_guard',[]); self.error = 'native_null_guard'; uc.emu_stop()
        else: raise AssertionError(f'unclaimed {rva:x}')

    def run(self,options=None,retained=False):
        if not retained: self.prepare(options or {})
        else: self.options = deepcopy(options or {})
        self.error,self.fault,self.allowed = None,None,{}; self.entry_counts = self.counts.copy()
        self.owner = 0 if self.options.get('null_owner') else self.p['data']; x,sp = self.x,self.stack+0x18008
        initial,old = self.snapshot(),len(self.events); self.q(sp,self.stop)
        for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']): self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16): self.u.reg_write(getattr(x,'UC_X86_REG_XMM'+str(i)),(1<<125)|i)
        incoming = [self.owner,0xDEAD123400000002,self.options.get('entry_r8_bits',0xDEAD123400000008),self.options.get('entry_r9_bits',0xDEAD123400000009)]
        for n,v in zip(['RCX','RDX','R8','R9'],incoming): self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        for n,v in [('RSP',sp),('R10',0xABCD0010),('R11',0xABCD0011),('MXCSR',0x1F80)]: self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        try: self.u.emu_start(self.base+ENTRY,self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc = self.reg(x.UC_X86_REG_RIP)-self.base
            assert exc.errno == self.unicorn.UC_ERR_READ_UNMAPPED and pc in [0x3B4C15,0x3B4C2A,0x3B4C31,0x3B4C5D,0x3B4C64]
            self.error,self.fault = 'native_read_fault',hex(pc)
        returned = self.reg(x.UC_X86_REG_RIP) == self.stop; assert returned or self.error
        result = self.reg(x.UC_X86_REG_RAX) if returned else None
        if returned:
            self.oid(result); assert self.reg(x.UC_X86_REG_RSP) == sp+8
            assert all(self.reg(getattr(x,'UC_X86_REG_'+n)) == 0xFAB0000000000000+i for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']))
            assert all(self.reg(getattr(x,'UC_X86_REG_XMM'+str(i))) == (1<<125)|i for i in range(6,16))
        final = self.snapshot()
        for n,raw in initial['memory'].items():
            assert all(i in self.allowed.get(n,set()) or b == bytes.fromhex(final['memory'][n])[i] for i,b in enumerate(bytes.fromhex(raw))),n
        row = dict(options=deepcopy(self.options),entry_raw_args=incoming,initial=initial,events=deepcopy(self.events[old:]),final=final,returned=returned,result_bits=result,error=self.error,fault_rva=self.fault,service_counts_before=self.entry_counts.copy(),service_counts_after=self.counts.copy(),normal_abi_verified=returned)
        verify(row,self); row['independent_full_state_verified'] = True; return row


def verify(row,m):
    state = deepcopy(row['initial']); mem = {n:bytearray.fromhex(raw) for n,raw in state['memory'].items()}
    options,regs,counts = row['options'],row['entry_raw_args'].copy(),row['service_counts_before'].copy()
    events,error,fault,result,returned = [],None,None,None,False
    def word(n,off,width=8): return int.from_bytes(mem[n][off:off+width],'little')
    def ptr(n): return 0 if n is None else m.p[n]
    def snapshot():
        s = deepcopy(state); s['memory'] = {n:raw.hex() for n,raw in mem.items()}; return s
    def mutate(phase):
        actions = options.get('mutations',{}).get(phase,[])
        if isinstance(actions,str): actions = [actions]
        def write(n,off,width,value): mem[n][off:off+width] = value.to_bytes(width,'little')
        for action in actions:
            if action in ['replace_class','clear_class']: state['metadata_slot'] = 'other_class' if action.startswith('replace') else None
            elif action in ['replace_statics','clear_statics']: write('project_class',0xB8,8,ptr('other_statics' if action.startswith('replace') else None))
            elif action in ['replace_context','clear_context']: write('statics',0,8,ptr('other_context' if action.startswith('replace') else None))
            elif action in ['replace_game','clear_game']: write('context',0x20,8,ptr('other_game' if action.startswith('replace') else None))
            elif action.startswith('language:'): write('game',0x20,4,int(action.partition(':')[2],0))
            elif action in ['replace_description','clear_description']: write('data',0x50,8,ptr('replacement' if action.startswith('replace') else None))
            else: raise AssertionError(action)
    class Stopped(Exception): pass
    def emit(kind,args,raw,site,phase=None,terminal=False):
        nonlocal regs,error
        counts[kind] = counts.get(kind,0)+1
        events.append(dict(kind=kind,ordinal=counts[kind],args=deepcopy(args),snapshot=snapshot(),raw_args=raw.copy(),native_site=hex(site),caller_return_bits=m.base+site+m.instructions[site].size))
        relative = counts[kind]-row['service_counts_before'].get(kind,0)
        if terminal or options.get('failure') == [kind,relative]: error = kind; raise Stopped
        if kind == 'metadata': state['service_history'].append(['metadata',state['metadata_slot']])
        else: state['service_history'].append(['converter',args[0],args[2]])
        if phase: mutate(phase)
        regs = [POISON]*4
    def convert(site,ordinal):
        text = word('data',0x50); output = options.get('converter_results',['result0','result1','result2'])[ordinal-1]
        emit('converter',[m.oid(text),0,output],[text,0,regs[2],regs[3]],site,'converter:'+str(ordinal))
        return ptr(output)
    def read_context(second=False):
        nonlocal fault,error
        cls = state['metadata_slot'] if not second else m.oid(regs[2])
        if cls is None: error,fault = 'native_read_fault',hex(0x3B4C5D if second else 0x3B4C2A); raise Stopped
        static = m.oid(word(cls,0xB8))
        if not second: regs[0] = ptr(static)
        if static is None: error,fault = 'native_read_fault',hex(0x3B4C64 if second else 0x3B4C31); raise Stopped
        context = m.oid(word(static,0))
        if context is None: emit('native_null_guard',[],regs,0x3B4C92,terminal=True)
        game = m.oid(word(context,0x20))
        if game is None: emit('native_null_guard',[],regs,0x3B4C92,terminal=True)
        return word(game,0x20,4)
    state['native_entries'].append(regs.copy())
    try:
        if state['metadata_flag_byte'] == 0:
            emit('metadata',[hex(SLOT),state['metadata_slot']],[m.base+SLOT,*regs[1:]],0x3B4C09,'metadata')
            state['metadata_flag_byte'] = 1
        if row['entry_raw_args'][0] == 0: error,fault = 'native_read_fault','0x3b4c15'; raise Stopped
        first = convert(0x3B4C1B,1); regs[2] = ptr(state['metadata_slot']); regs[1] = first
        language = read_context()
        if language == 0:
            second = convert(0x3B4C4E,2); regs[2] = ptr(state['metadata_slot']); regs[1] = second
        language = read_context(True)
        if language == 10:
            ordinal = counts.get('converter',0)-row['service_counts_before'].get('converter',0)+1
            third = convert(0x3B4C81,ordinal); regs[1] = third
        result,returned = regs[1],True
    except Stopped: pass
    assert row['events'] == events,('events',options)
    assert (row['returned'],row['error'],row['fault_rva'],row['result_bits']) == (returned,error,fault,result),('outcome',options)
    assert row['final'] == snapshot() and row['service_counts_after'] == counts,('state',options)


def audit(game_root,dumper_root):
    m = Machine(game_root,dumper_root); cases,sequences,baselines,stops = [],[],[],[]
    for warm,lang,null,outputs in itertools.product([0,1,0x80,0xFF],[0,10,1,0x80000000,0x8000000A,0xFFFFFFFF],[False,True],[['result0','result1','result2'],[None,None,None],['description','description','description']]):
        cases.append(m.run(dict(warm_flag=warm,language_bits=lang,null_description=null,converter_results=outputs)))
    for option in ['null_owner','null_class','null_statics','null_context','null_game']:
        for warm in [0,1]: cases.append(m.run({option:True,'warm_flag':warm}))
    actions = ['replace_class','clear_class','replace_statics','clear_statics','replace_context','clear_context','replace_game','clear_game','language:0','language:10','language:1','language:0xffffffff','replace_description','clear_description']
    for phase,action,lang in itertools.product(['metadata','converter:1','converter:2'],actions,[0,10,1]):
        cases.append(m.run(dict(language_bits=lang,mutations={phase:action})))
    for compound in [['language:10','replace_description'],['replace_class','replace_description'],['clear_context','clear_description']]:
        cases.append(m.run(dict(mutations={'converter:2':compound})))
    for action in actions:
        cases.append(m.run(dict(mutations={'converter:2':'language:10','converter:3':action})))
    cases.append(m.run(dict(entry_r8_bits=0x123456789ABCDEF0,entry_r9_bits=0xFEDCBA9876543210)))
    for options in [{},{'language_bits':10},{'language_bits':1},{'mutations':{'converter:1':'replace_description'}},{'mutations':{'converter:2':'language:10'}},{'mutations':{'converter:1':'replace_class'}}]:
        rows = [m.run(options),m.run({},True),m.run({'converter_results':['result2','result0','result1']},True)]
        assert all(r['returned'] for r in rows) and all(b['initial'] == a['final'] for a,b in zip(rows,rows[1:])); sequences.append(rows)
    for options,recovery in [({'null_owner':True},{}),({'null_context':True},{'mutations':{'converter:1':'replace_context'}}),({'null_game':True},{'mutations':{'converter:1':'replace_game'}})]:
        rows = [m.run(options),m.run(recovery,True),m.run({},True)]
        assert not rows[0]['returned'] and all(r['returned'] for r in rows[1:]) and all(b['initial'] == a['final'] for a,b in zip(rows,rows[1:])); sequences.append(rows)
    profiles = [{},{'language_bits':10},{'language_bits':1},{'warm_flag':1},{'null_context':True},{'mutations':{'converter:1':'clear_game'}},{'mutations':{'converter:2':'clear_context'}},{'mutations':{'converter:2':['language:10','replace_description']}},{'mutations':{'metadata':'replace_class'}}]
    for options in profiles:
        baseline = m.run(options); bid = len(baselines); baselines.append(baseline); counts = {}
        for i,e in enumerate(baseline['events']):
            kind = e['kind']; counts[kind] = counts.get(kind,0)+1; row = m.run(dict(options,failure=[kind,counts[kind]]))
            assert row['events'] == baseline['events'][:i+1] and row['final'] == e['snapshot'] and not row['returned']
            stops.append(dict(baseline=bid,prefix_length=i+1,result=row))
    missing = set(m.instructions)-m.executed; assert missing == {0x3B4C97}
    return dict(build=BUILD,schema='character_data_description_native_v1',targets=m.targets,supplied_targets=m.supplied,bounds=m.bounds,metadata_flag_rva=hex(FLAG),metadata_slot_rva=hex(SLOT),field_offsets={'description':0x50,'unused_polish_description':0x58,'unused_chinese_description':0x60,'project_static_fields':0xB8,'project_instance':0,'game_data':0x20,'language':0x20},instruction_assertions=len(m.checks),decoded_instructions=len(m.instructions),covered_instructions=len(set(m.instructions)&m.executed),unexecuted_terminal_traps=[hex(a) for a in missing],cases=cases,retained_sequences=sequences,baselines=baselines,failure_stops=stops,summary=dict(cases=len(cases),normal_returns=sum(r['returned'] for r in cases),native_stops=sum(not r['returned'] for r in cases),sequences=len(sequences),baselines=len(baselines),stops=len(stops)),scope='Exact CharacterData.GetDescription with supplied StringHelper converter and metadata/null runtime gateways. Current native body reloads the same description field, not legacy PL/CHN fields. Independent complete authored storage/ABI/caller/final model, exact stops/faults and retained sequences; no runtime admission/localization body/rendering/scheduling/unwinding.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(); parser.add_argument('game_root'); parser.add_argument('dumper_root'); parser.add_argument('--output',required=True)
    args = parser.parse_args(); full = audit(args.game_root,args.dumper_root); report = pool_snapshots(pool_memory(full))
    assert expand_memory(expand_snapshots(report)) == full
    Path(args.output).write_text(json.dumps(report,sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8'); print(json.dumps(full['summary']))
