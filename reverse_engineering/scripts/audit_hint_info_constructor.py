"""Execute the exact HintInfo constructor with arbitrary authored argument profiles.

Actual folded Object return executes; GC barriers are supplied inert boundaries.
Stack argument callbacks are explicit diagnostics, not engine admission claims.
"""
import argparse
import itertools
import json
import struct
from copy import deepcopy
from pathlib import Path
from audit_character_assets import BUILD
from audit_deck_character_surface import Machine as FrozenMachine, HINT_CTOR, HINT_FIELDS
from audit_report_snapshots import pool_snapshots

STORE_PINS={0x3BC22E:(0x18,8),0x3BC23F:(0x10,8),0x3BC24E:(0x30,8),0x3BC25D:(0x20,8),0x3BC26E:(0x28,8),0x3BC288:(0x38,16)}
COLORS=[[0,0,0,0],[0x3F800000]*4,[0x80000000,0x7FC01234,0x7F800000,0xFF800000],[0xFFFFFFFF,0x00000001,0x00800000,0x3E4CCCCD]]
SLOTS={'flavor':0x28,'title':0x30,'color':0x38,'method_info':0x40}

class Machine(FrozenMachine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root,dumper_root)
        # The frozen verifier pins exact constructor signature/bounds/type fields.
        self.targets=self.helper_targets
        self.body={a for a in self.instructions if HINT_CTOR[0]<=a<HINT_CTOR[1]}
        self.p={n:self.arena+0xB0000+i*0x1000 for i,n in enumerate(['hint','other_hint','text','title','hints','flavor','image','replacement','color','other_color'])}
        self.sizes={n:0x80 for n in self.p}
        self.labels={p:n for n,p in self.p.items()}
        self.entry_sp=self.stack+0x18008
        self.extra_checks={
            0x3BC214:('mov','rbx, rdx'),0x3BC217:('mov','rsi, r9'),0x3BC21C:('mov','rdi, r8'),
            0x3BC236:('mov','rdx, qword ptr [rsp + 0x58]'),0x3BC265:('mov','rdx, qword ptr [rsp + 0x50]'),
            0x3BC276:('mov','rax, qword ptr [rsp + 0x60]')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.extra_checks.items())

    def snapshot(self):
        return {'memory':{n:bytes(self.u.mem_read(p,self.sizes[n])).hex() for n,p in self.p.items()},
                'hint_field_bits':{n:self.rq(self.p['hint']+off) for n,off in HINT_FIELDS},
                'hint_color_bits':list(struct.unpack('<IIII',self.u.mem_read(self.p['hint']+0x38,16))),
                'stack_argument_bits':{n:self.rq(self.entry_sp+off) for n,off in SLOTS.items()},
                'register_argument_bits':{n:self.input_refs[n] for n in ['text','image','hints']},
                'completed_barriers':[r.copy() for r in self.barriers], 'native_entries':self.entries.copy()}

    def prepare(self,options):
        self.options=options.copy();self.events=[];self.counts={};self.error=None
        self.barriers=[];self.entries=[];self.allowed={};self.stack_allowed=set()
        for n,p in self.p.items():self.u.mem_write(p,bytes([0xA5])*self.sizes[n])
        refs={n:0 if options.get('null_mask',0)&(1<<i) else self.p['text'] if options.get('alias_strings') and n!='image' else self.p[n]
              for i,n in enumerate(['text','image','hints','flavor','title'])}
        self.input_refs=refs
        for n,off in SLOTS.items():
            value=refs[n] if n in refs else 0 if n=='color' and options.get('null_color') else self.p['color'] if n=='color' else options.get('method_bits',0xDEAD123456789ABC)
            self.q(self.entry_sp+off,value)
        self.u.mem_write(self.p['color'],struct.pack('<IIII',*options.get('color_bits',COLORS[0])))
        self.u.mem_write(self.p['other_color'],struct.pack('<IIII',*COLORS[2]))
        self.q(self.entry_sp,self.stop)

    def mutate(self,kind,ordinal):
        action=self.options.get('mutations',{}).get(kind+':'+str(ordinal))
        if action is None:return
        if action in ['replace_title_slot','replace_flavor_slot','replace_color_slot','clear_color_slot']:
            name=action.split('_')[1];off=SLOTS[name]
            value=0 if action=='clear_color_slot' else self.p['other_color'] if name=='color' else self.p['replacement']
            self.q(self.entry_sp+off,value);self.stack_allowed.update(range(off,off+8))
        elif action=='replace_color_bytes':
            self.u.mem_write(self.p['color'],struct.pack('<IIII',*COLORS[3]));self.allowed.setdefault('color',set()).update(range(16))
        elif action=='overwrite_early_text':
            self.q(self.p['hint']+0x18,self.p['replacement']);self.allowed.setdefault('hint',set()).update(range(0x18,0x20))
        else:raise AssertionError(action)

    def event(self,kind,args):
        # Frozen event applies the callback only after capturing its snapshot;
        # callbacks change authored memory, never these service-entry registers.
        result=super().event(kind,args)
        self.events[-1]['raw_args']=[self.reg(getattr(self.x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
        self.events[-1]['caller_return_rva']=hex(self.rq(self.reg(self.x.UC_X86_REG_RSP))-self.base)
        return result

    def hook(self,uc,address,size,data):
        if address==self.stop:return
        rva,x=address-self.base,self.x;self.executed.add(rva)
        cx,dx=self.reg(x.UC_X86_REG_RCX),self.reg(x.UC_X86_REG_RDX)
        if rva==HINT_CTOR[0]:self.entries.append('HintInfo.ctor')
        if rva in STORE_PINS:
            off,width=STORE_PINS[rva]
            if self.reg(x.UC_X86_REG_RBP):self.allowed.setdefault('hint',set()).update(range(off,off+width))
        if rva in self.body:return
        if rva==0x33ED50:
            assert cx==(0 if self.options.get('null_owner') else self.p['hint']) and dx==0
            self.entries.append('folded_Object_ret0');return
        if rva==0x2B6FF0:
            off=cx-self.p['hint'];assert off in dict(HINT_FIELDS).values() and self.rq(cx)==dx
            caller=self.rq(self.reg(x.UC_X86_REG_RSP))-self.base
            assert caller=={0x18:0x3BC236,0x10:0x3BC247,0x30:0x3BC256,0x20:0x3BC265,0x28:0x3BC276}[off]
            field=next(n for n,o in HINT_FIELDS if o==off)
            args={'field':field,'receiver':self.oid(cx-off),'value_bits':dx,'field_offset':off}
            if self.event('reference_barrier',args):self.barriers.append(args);self.ret()
            return
        raise AssertionError(f'unclaimed instruction {rva:x}')

    def run(self,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:
            self.options=options or {};self.error=None;self.allowed={};self.stack_allowed=set();self.counts={}
            # Explicit next invocation rewrites caller argument slots; prior object
            # and completed-service storage persist in the same physical graph.
            for n in ['flavor','title']:self.q(self.entry_sp+SLOTS[n],self.input_refs[n])
            self.q(self.entry_sp+SLOTS['color'],self.p['color'])
        initial=self.snapshot();old=len(self.events)
        x=self.x;sp=self.entry_sp;self.q(sp,self.stop)
        for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB0000000000000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),(0xABCDEF9876543210<<64)|i)
        for n,v in [('RSP',sp),('RCX',0 if self.options.get('null_owner') else self.p['hint']),('RDX',self.input_refs['text']),('R8',self.input_refs['image']),('R9',self.input_refs['hints'])]:self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        try:self.u.emu_start(self.base+HINT_CTOR[0],self.stop,timeout=10_000_000,count=2000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base
            assert pc in [0x3BC22E,0x3BC285]
            assert exc.errno==(self.unicorn.UC_ERR_WRITE_UNMAPPED if pc==0x3BC22E else self.unicorn.UC_ERR_READ_UNMAPPED)
            assert pc!=0x3BC22E or self.options.get('null_owner')
            assert pc!=0x3BC285 or self.rq(sp+SLOTS['color'])==0
            self.error='native_receiver_write_fault' if pc==0x3BC22E else 'native_color_read_fault'
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==sp+8
            for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):assert self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB0000000000000+i
            for i in range(6,16):assert self.reg(getattr(x,f'UC_X86_REG_XMM{i}'))==(0xABCDEF9876543210<<64)|i
        final=self.snapshot();events=self.events[old:].copy()
        for n,raw in initial['memory'].items():
            before,after=bytes.fromhex(raw),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or a==after[i] for i,a in enumerate(before)),n
        for n,off in SLOTS.items():assert final['stack_argument_bits'][n]==initial['stack_argument_bits'][n] or off in self.stack_allowed
        completed=events if returned or self.error=='native_color_read_fault' else events[:-1] if events else []
        assert final['completed_barriers']==initial['completed_barriers']+[e['args'] for e in completed]
        row={'options':self.options.copy(),'returned':returned,'error':self.error,'initial':initial,'events':events,'final':final,'nonvolatile_verified':returned,'unrelated_bytes_retained':True,
             'fault_rva':hex(self.reg(x.UC_X86_REG_RIP)-self.base) if self.error in ['native_receiver_write_fault','native_color_read_fault'] else None}
        verify_semantics(row,self.p)
        row['independent_ordered_state_verified']=True
        return row

def verify_semantics(row,p):
    """Simulate authored stores/callbacks from the initial graph, including stops.

    The model never reads final native fields or current emulator color storage.
    Every event snapshot, full raw argument tuple, final byte and slot is checked.
    """
    state=deepcopy(row['initial']);memory={n:bytearray.fromhex(raw) for n,raw in state['memory'].items()}
    slots=state['stack_argument_bits'];options=row['options'];expected_events=[]
    def snapshot():
        result=deepcopy(state)
        result['memory']={n:raw.hex() for n,raw in memory.items()}
        result['hint_field_bits']={n:struct.unpack_from('<Q',memory['hint'],off)[0] for n,off in HINT_FIELDS}
        result['hint_color_bits']=list(struct.unpack_from('<IIII',memory['hint'],0x38))
        return result
    def mutate(ordinal):
        action=options.get('mutations',{}).get('reference_barrier:'+str(ordinal))
        if action in ['replace_title_slot','replace_flavor_slot','replace_color_slot','clear_color_slot']:
            name=action.split('_')[1]
            slots[name]=0 if action=='clear_color_slot' else p['other_color'] if name=='color' else p['replacement']
        elif action=='replace_color_bytes':memory['color'][:16]=struct.pack('<IIII',*COLORS[3])
        elif action=='overwrite_early_text':struct.pack_into('<Q',memory['hint'],0x18,p['replacement'])
        else:assert action is None
    state['native_entries']+=['HintInfo.ctor','folded_Object_ret0']
    error=None
    if options.get('null_owner'):error='native_receiver_write_fault'
    else:
        for ordinal,(field,off,caller) in enumerate([('text',0x18,0x3BC236),('title',0x10,0x3BC247),('image',0x30,0x3BC256),('hints',0x20,0x3BC265),('flavor',0x28,0x3BC276)],1):
            value=slots[field] if field in ['title','flavor'] else state['register_argument_bits'][field]
            struct.pack_into('<Q',memory['hint'],off,value)
            args={'field':field,'receiver':'hint','value_bits':value,'field_offset':off}
            raw=[p['hint']+off,value,state['register_argument_bits']['image'] if ordinal==1 else 0xFACE123456789002,state['register_argument_bits']['hints'] if ordinal==1 else 0xFACE123456789003]
            expected_events.append({'kind':'reference_barrier','args':args,'snapshot':snapshot(),'raw_args':raw,'caller_return_rva':hex(caller)})
            if options.get('failure')==['reference_barrier',ordinal]:error='reference_barrier';break
            mutate(ordinal);state['completed_barriers'].append(args)
        if error is None:
            if slots['color']==0:error='native_color_read_fault'
            else:
                source=next(n for n,pointer in p.items() if pointer==slots['color'])
                memory['hint'][0x38:0x48]=memory[source][:16]
    assert row['events']==expected_events
    assert row['error']==error and row['returned']==(error is None)
    assert row['final']==snapshot()
    assert row['fault_rva']==('0x3bc22e' if error=='native_receiver_write_fault' else '0x3bc285' if error=='native_color_read_fault' else None)

def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];seq=[];bases=[];stops=[]
    for mask,color,alias in itertools.product(range(32),COLORS,[False,True]):
        cases.append(m.run({'null_mask':mask,'color_bits':color,'alias_strings':alias}))
    for ordinal,action in itertools.product(range(1,6),['replace_title_slot','replace_flavor_slot','replace_color_slot','replace_color_bytes','overwrite_early_text']):
        cases.append(m.run({'color_bits':COLORS[1],'mutations':{'reference_barrier:'+str(ordinal):action}}))
    cases.append(m.run({'null_owner':True}));cases.append(m.run({'null_color':True}))
    for ordinal in range(1,6):cases.append(m.run({'mutations':{'reference_barrier:'+str(ordinal):'clear_color_slot'}}))
    for options in [{'color_bits':COLORS[2]},{'null_mask':31,'alias_strings':True}]:
        m.prepare(options);seq.append([m.run(retained=True),m.run({'mutations':{'reference_barrier:1':'overwrite_early_text'}},retained=True),m.run(retained=True)])
    for options in [{},{'null_mask':31},{'mutations':{'reference_barrier:1':'replace_title_slot'}},{'mutations':{'reference_barrier:1':'replace_flavor_slot'}},{'mutations':{'reference_barrier:5':'replace_color_slot'}}]:
        baseline=m.run(options);assert baseline['returned'];bid=len(bases);bases.append(baseline)
        for i,e in enumerate(baseline['events']):
            stopped=m.run({**options,'failure':['reference_barrier',i+1]})
            assert not stopped['returned'] and stopped['events']==baseline['events'][:i+1] and stopped['final']==e['snapshot']
            stops.append({'baseline':bid,'prefix_length':i+1,'result':stopped})
    assert not m.body-m.executed and 0x33ED50 in m.executed
    return {'build':BUILD,'schema':'hint_info_constructor_native_v1','targets':m.targets,'constructor_bounds':[hex(v) for v in HINT_CTOR],
            'instruction_assertions':len(m.extra_checks)+sum(a in m.body for a in m.checks),'decoded_instructions':len(m.body),'executed_instructions':len(m.body&m.executed),'folded_base_executed':True,
            'cases':cases,'sequences':seq,'baselines':bases,'stops':stops,'summary':{'cases':len(cases),'sequences':len(seq),'baselines':len(bases),'stops':len(stops)},
            'scope':'Exact HintInfo constructor only; actual folded Object ret0; supplied inert GC barriers. Authored stack-slot mutations and raw float/nullable pointer diagnostics preserve capture/load chronology, not valid engine allocation or callback admission. No UI renderer/GC implementation/runtime object admission or native unwinding; folded aliases excluded.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(pool_snapshots(r),sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8');print(json.dumps(r['summary']))
