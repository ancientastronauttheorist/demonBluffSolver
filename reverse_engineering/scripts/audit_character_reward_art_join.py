"""Actual reward initialization, data art getters and SetupArt in one native graph."""
import argparse
import itertools
import json
import struct
from copy import deepcopy
from pathlib import Path
from audit_character_assets import BUILD
from audit_character_reward_presentation import Machine as PresentationMachine, FIELDS, SIDES, SERVICES, POISON
from audit_character_init_reward import Machine as InitVerifier, ENTRY, END, FOLLOWING
from audit_character_art_preferences import Machine as ArtVerifier, TARGETS as ART_TARGETS
from audit_character_data_consumers import Machine as DataVerifier, TARGETS as DATA_TARGETS
from audit_character_oracle_reveal_join import pool_memory
from audit_report_snapshots import pool_snapshots

POINTERS={'bluff':0x58,'register_as':0x60,'state_action':0x180}
WORDS={'uses':0xDC,'prev_state':0xE0,'state':0xE4,'alignment':0xF8}

class RewardMachine(PresentationMachine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root,dumper_root)
        verifier=InitVerifier(game_root,dumper_root)
        self.init_target=verifier.target.copy();self.init_ranges=deepcopy(verifier.ranges);self.init_services=deepcopy(verifier.services)
        self.init_instructions=verifier.instructions.copy();self.init_checks=verifier.checks.copy();self.init_stores=verifier.stores.copy()
        assert len(self.init_instructions)==50 and (self.init_instructions[0x365711].mnemonic,self.init_instructions[0x365711].op_str)==('int3','')
        self.instructions.update(self.init_instructions);self.targets.append(self.init_target)
        self.bounds[hex(ENTRY)]={'end_exclusive':hex(END),'next_managed':hex(FOLLOWING),'unwind_chunks':self.init_ranges['InitReward']['unwind_chunks'],'terminal_trap':'0x365711'}
        self.supplied=[r for r in self.supplied if r['Name']!='Character$$RevealReal']+[r for r in self.init_services if r['Name']!='Character$$RevealReal']
        self.extra_names=['game0','game1','game2','game3','action0','action1','action_class','code0','code1','method0','method1','unused_target']
        for i,n in enumerate(self.extra_names):self.p[n]=self.arena+0x400000+i*0x1000;self.ids[self.p[n]]=n;self.sizes[n]=0x80
        self.callback_gateway=self.stop+0x300

    def snapshot(self):
        r=super().snapshot()
        if not getattr(self,'join_ready',False):return r
        r['reward_join']={'pointers':{n:self.oid(self.rq(self.p['actor']+off)) for n,off in POINTERS.items()},
            'words':{n:self.rd(self.p['actor']+off) for n,off in WORDS.items()},
            'starting_alignment_bits':{n:self.rd(self.p[n]+0x134) for n in ['data0','data1','data2']},
            'acted_games':self.acted_games.copy(),'active':self.active.copy(),'callbacks':deepcopy(self.callbacks),'captured_init_entries':deepcopy(self.captures)}
        return r

    def prepare(self,options):
        self.join_ready=False;super().prepare(options)
        for n,off in POINTERS.items():self.q(self.p['actor']+off,0 if options.get('null_'+n) else self.p['data2' if n in ['bluff','register_as'] else 'action0'])
        for n,off in WORDS.items():self.d(self.p['actor']+off,options.get(n,{'uses':0xF0000007,'prev_state':30,'state':10,'alignment':20}[n]))
        for i,n in enumerate(['data0','data1','data2']):self.d(self.p[n]+0x134,options.get('input_alignment', [10,20,0xFFFFFFFF][i]))
        for i,n in enumerate(['action0','action1']):
            self.q(self.p[n],self.p['action_class']);self.q(self.p[n]+0x18,self.callback_gateway);self.q(self.p[n]+0x20,self.p['unused_target']);self.q(self.p[n]+0x28,self.p['method'+str(i)]);self.q(self.p[n]+0x40,0 if options.get('null_method_code') else self.p['code'+str(i)])
        self.acted_games={n:'game0' if options.get('alias_games') else 'game'+n[-1] for n in ['acted0','acted1','acted2','acted3']}
        self.active={n:True for n in ['game0','game1','game2','game3']};self.callbacks=[];self.captures=[];self.current_input=0;self.current_acted=0;self.last_game=0;self.join_ready=True
        self.d(self.p['object_class']+0xE0,int(options.get('class_warm',options.get('warm',False))))
        self.u.mem_write(self.base+self.flag,bytes([int(options.get('metadata_warm',options.get('warm',False)))]))

    def mutate(self,phase):
        action=self.options.get('init_mutations',{}).get(phase)
        if action is None:return super().mutate(phase)
        fields={'replace_acteds':(0xA8,'acted2'),'clear_acteds':(0xA8,None),'replace_data':(0x50,'data2'),'clear_data':(0x50,None),
            'replace_name':(0x40,'tmp1'),'clear_name':(0x40,None),'replace_state_action':(0x180,'action1'),'clear_state_action':(0x180,None),
            'replace_bluff':(0x58,'data2'),'replace_register_as':(0x60,'data2')}
        if action in fields:
            off,n=fields[action];self.q(self.p['actor']+off,0 if n is None else self.p[n]);self.allowed.setdefault('actor',set()).update(range(off,off+8))
        elif action=='change_input_alignment':
            n=self.oid(self.current_input);assert n in ['data0','data1','data2'];self.d(self.current_input+0x134,0xF1234567);self.allowed.setdefault(n,set()).update(range(0x134,0x138))
        elif action.startswith('change_') and action[7:] in WORDS:
            off=WORDS[action[7:]];self.d(self.p['actor']+off,0xE1234567);self.allowed.setdefault('actor',set()).update(range(off,off+4))
        elif action=='replace_game_map':self.acted_games[self.oid(self.current_acted)]='game2'
        else:raise AssertionError(action)

    def hook(self,uc,address,size,data):
        rva,x=address-self.base,self.x
        if rva==0x365711:raise AssertionError('terminal trap executed')
        if rva in self.init_instructions:
            self.executed.add(rva);self.last_native=rva
            if rva==ENTRY:
                self.current_input=self.reg(x.UC_X86_REG_RDX);self.entries.append({'entry':'InitReward','raw_args':[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]})
            if rva==0x365657:
                self.current_acted=self.reg(x.UC_X86_REG_RCX);self.captures.append({'input':self.oid(self.current_input),'acted':self.oid(self.current_acted)})
            if rva in self.init_stores:
                off,width=self.init_stores[rva];assert self.reg(x.UC_X86_REG_RBX)==self.p['actor'];self.allowed.setdefault('actor',set()).update(range(off,off+width))
            return
        if rva in [0x3689EA,0x368A05,0x368A20,0x368A3B]:self.allowed.setdefault('actor',set()).update(range(0xA8,0xB0))
        if rva==0x3689DC:self.allowed.setdefault('actor',set()).add(0xB0)
        cx,dx,r8,r9=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
        if rva==0x1C79FD0:
            self.executed.add(rva);assert cx==self.current_acted and self.oid(cx) in self.acted_games and dx==0 and self.last_native==0x365662
            mode=self.options.get('gameobject_result','normal');assert mode in ['normal','null','other']
            result=0 if mode=='null' else self.p['game2'] if mode=='other' else self.p[self.acted_games[self.oid(cx)]]
            if self.event('get_gameobject',[self.oid(cx),dx,self.oid(result)]):
                self.requests.append({'kind':'get_gameobject','args':[self.oid(cx),dx,self.oid(result)]});self.last_game=result;self.mutate('get_gameobject');self.ret(result)
            return
        if rva==0x1C7D810:
            self.executed.add(rva);assert cx==self.last_game and self.oid(cx) in self.active and dx==0 and r8==0 and self.last_native==0x365678
            args=[self.oid(cx),dx,r8]
            if self.event('set_active',args):self.active[self.oid(cx)]=False;self.requests.append({'kind':'set_active','args':args.copy()});self.mutate('set_active');self.ret()
            return
        if rva==0x2B6FF0 and self.last_native in [0x36568A,0x365699,0x3656AB]:
            self.executed.add(rva);off,ordinal={0x36568A:(0x58,1),0x365699:(0x50,2),0x3656AB:(0x60,3)}[self.last_native]
            assert cx==self.p['actor']+off and dx==(self.current_input if off==0x50 else 0) and self.rq(cx)==dx
            args=['actor',off,self.oid(dx)]
            if self.event('reference_barrier',args):self.requests.append({'kind':'reference_barrier','args':args.copy()});self.mutate('init_barrier:'+str(ordinal));self.ret()
            return
        if address==self.callback_gateway:
            self.executed.add(rva);action=self.reg(x.UC_X86_REG_RAX);assert action in [self.p['action0'],self.p['action1']] and cx==self.rq(action+0x40) and dx==self.rq(action+0x28) and r8==POISON and r9==POISON and self.last_native==0x3656F5
            args=[self.oid(action),self.oid(cx),self.oid(dx),r8,r9]
            if self.event('state_callback',args):self.callbacks.append(args.copy());self.requests.append({'kind':'state_callback','args':args.copy()});self.mutate('callback');self.ret()
            return
        if rva==0x2B7D90 and self.last_native==0x36570C:
            self.executed.add(rva);self.event('native_null_guard',[]);self.error='native_null_guard';uc.emu_stop();return
        return super().hook(uc,address,size,data)

    def run(self,name,options=None,retained=False):
        if not retained:self.prepare(options or {})
        else:self.options,self.counts,self.error,self.allowed,self.fault=options or {},{},None,{},None
        initial,old=self.snapshot(),len(self.events);x,sp=self.x,self.stack+0x18008;self.q(sp,self.stop)
        for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),0xFAB00000+i)
        for i in range(6,16):self.u.reg_write(getattr(x,f'UC_X86_REG_XMM{i}'),(1<<125)|i)
        arg=0 if self.options.get('null_input_data') else self.p[self.options.get('input_data','data1')]
        entry=[0 if self.options.get('null_owner') else self.p['actor'],arg if name=='InitReward' else 0xFACE000000000000|self.options.get('side_bits',0) if name=='SetupObject' else 0,0xDEAD123400000008 if name=='InitReward' else 0,0xDEAD123400000009]
        for n,v in zip(['RCX','RDX','R8','R9'],entry):self.u.reg_write(getattr(x,'UC_X86_REG_'+n),v)
        self.u.reg_write(x.UC_X86_REG_RSP,sp);self.u.reg_write(x.UC_X86_REG_MXCSR,0x1F80)
        start=ENTRY if name=='InitReward' else 0x3689D0 if name=='SetupObject' else 0x3682A0
        try:self.u.emu_start(self.base+start,self.stop,timeout=10000000,count=10000)
        except self.unicorn.UcError as exc:
            pc=self.reg(x.UC_X86_REG_RIP)-self.base
            assert self.options.get('null_owner') and exc.errno==self.unicorn.UC_ERR_READ_UNMAPPED and pc in [0x365650,0x3689D5,0x3689F7,0x368A12,0x368A2D,0x3682D5];self.error='native_owner_read_fault';self.fault=hex(pc)
        returned=self.reg(x.UC_X86_REG_RIP)==self.stop;assert returned or self.error
        if returned:
            assert self.reg(x.UC_X86_REG_RSP)==sp+8
            for i,n in enumerate(['RBX','RBP','RSI','RDI','R12','R13','R14','R15']):assert self.reg(getattr(x,'UC_X86_REG_'+n))==0xFAB00000+i
            for i in range(6,16):assert self.reg(getattr(x,f'UC_X86_REG_XMM{i}'))==(1<<125)|i
        final=self.snapshot();events=self.events[old:].copy()
        for n,raw in initial['memory'].items():
            before,after=bytes.fromhex(raw),bytes.fromhex(final['memory'][n]);assert all(i in self.allowed.get(n,set()) or b==after[i] for i,b in enumerate(before)),n
        assert final['metadata_slots']==initial['metadata_slots']
        row={'entry':name,'entry_raw_args':entry,'options':self.options.copy(),'initial':initial,'events':events,'final':final,'returned':returned,'error':self.error,'fault_rva':self.fault,'normal_abi_verified':returned}
        verify_semantics(row,self);row['independent_ordered_state_verified']=True;return row

class Machine(RewardMachine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root,dumper_root)
        dv,av=DataVerifier(game_root,dumper_root),ArtVerifier(game_root,dumper_root)
        assert dv.bindings[0x2718BF0]==('metadata','UnityEngine.Object_TypeInfo') and av.slot-av.base==self.slot-self.base==0x2718BF0
        self.added_entries={DATA_TARGETS[n][0]:n for n in ['GetArt','GetArtType']};self.added_entries[0x3688B0]='SetupArt'
        self.extra_flags={n:dv.flags[n] for n in ['GetArt','GetArtType']};self.extra_flags['SetupArt']=av.entry_flags['SetupArt']
        self.added_instructions={};self.added_checks={}
        for name in ['GetArt','GetArtType']:
            start,end,*_=DATA_TARGETS[name]
            self.added_instructions.update({a:i for a,i in dv.instructions.items() if start<=a<end})
            self.targets.append(next(r for r in dv.targets if r['Name']=='CharacterData$$'+name));self.bounds[hex(start)]=deepcopy(dv.bounds[name]);self.bounds[hex(start)]['unwind_chunks']=dv.ranges[name]
        start,end=0x3688B0,ART_TARGETS[0x3688B0][2]
        self.added_instructions.update({a:i for a,i in av.instructions.items() if start<=a<end})
        trap=list(self.cs.disasm(self.pe.get_data(end,1),end));assert len(trap)==1 and trap[0].mnemonic=='int3'
        self.added_instructions[end]=trap[0]
        chunks=[(e.struct.BeginAddress,e.struct.EndAddress) for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress==start];assert chunks==[(start,end+1)]
        self.bounds[hex(start)]={**av.bounds[hex(start)],'end_exclusive':hex(end+1),'unwind_chunks':[[hex(a),hex(b)] for a,b in chunks]}
        self.targets.append(next(r for r in av.targets if r['Name']=='Character$$SetupArt'))
        self.instructions.update(self.added_instructions)
        self.added_checks.update({a:v for a,v in dv.checks.items() if a in self.added_instructions});self.added_checks.update({a:v for a,v in av.checks.items() if a in self.added_instructions})
        self.supplied=[r for r in self.supplied if r['Name'] not in ['CharacterData$$GetArt','CharacterData$$GetArtType','Character$$SetupArt']]
        self.supplied += [r for r in av.supplied if r['Name']=='UnityEngine.Object$$op_Equality']
        for i,n in enumerate(['skin0','skin1','skin2','art0','clipping0','art1','clipping1','artgame0','artgame1','artgame2']):
            self.p[n]=self.arena+0x600000+i*0x1000;self.ids[self.p[n]]=n;self.sizes[n]=0x100 if n.startswith('skin') else 0x80
        self.traps={0x365711,0x36840E,0x3B4B38,0x3B4AA2,0x3689C0};self.art_ready=False

    def prepare(self,options):
        self.art_ready=False;super().prepare(options)
        for i,n in enumerate(['data0','data1','data2']):
            skin=options.get('skin'+str(i),'skin'+str(i));self.q(self.p[n]+0xC0,0 if skin is None else self.p[skin]);self.q(self.p[n]+0x98,0 if options.get('null_art_sprite') else self.p['sprite'+str(i)])
        for i,n in enumerate(['skin0','skin1','skin2']):
            self.q(self.p[n]+0x38,0 if options.get('null_skin_sprite') else self.p['sprite'+str((i+1)%3)]);self.d(self.p[n]+0x50,options.get('skin_type_bits',[0,10,0xFFFFFFFF][i]))
        for n,off in [('art',0x28),('clipping',0x30)]:
            target='art0' if n=='clipping' and options.get('alias_images') else n+'0';self.q(self.p['actor']+off,0 if options.get('null_art_field')==n else self.p[target])
        self.art_games={'art0':'artgame0','clipping0':'artgame0' if options.get('alias_art_games') else 'artgame1','art1':'artgame2','clipping1':'artgame2'}
        self.active.update({n:False for n in ['artgame0','artgame1','artgame2']});self.components.update({n:{'text':None,'color_bits':[0xA5A5A5A5]*4,'sprite':None} for n in self.art_games})
        self.art_live={n:True for n in ['skin0','skin1','skin2','sprite0','sprite1','sprite2']};self.art_live.update(options.get('art_liveness',{}))
        if options.get('alias_skins'):
            for n in ['data0','data1','data2']:self.q(self.p[n]+0xC0,self.p['skin0'])
        if options.get('alias_sprites'):
            for n in ['data0','data1','data2']:self.q(self.p[n]+0x98,self.p['sprite0'])
            for n in ['skin0','skin1','skin2']:self.q(self.p[n]+0x38,self.p['sprite0'])
        for n,f in self.extra_flags.items():self.u.mem_write(self.base+f,bytes([options.get('extra_metadata_flags',{}).get(n,int(options.get('warm',False)))]))
        self.frames=[];self.results=[];self.callee_abi=[];self.saved_frames=[];self.art_mutation_log=[];self.active_frame_start=0;self.art_ready=True

    def snapshot(self):
        r=super().snapshot()
        if not getattr(self,'art_ready',False):return r
        r['art_join']={'fields':{n:self.oid(self.rq(self.p['actor']+off)) for n,off in [('art',0x28),('clipping',0x30)]},
            'data':{n:{'skin':self.oid(self.rq(self.p[n]+0xC0)),'default':self.oid(self.rq(self.p[n]+0x98))} for n in ['data0','data1','data2']},
            'skins':{n:{'sprite':self.oid(self.rq(self.p[n]+0x38)),'type_bits':self.rd(self.p[n]+0x50)} for n in ['skin0','skin1','skin2']},
            'metadata_flags':{n:self.u.mem_read(self.base+f,1)[0] for n,f in self.extra_flags.items()},'component_games':self.art_games.copy(),'liveness':self.art_live.copy(),
            'frames':deepcopy(self.frames),'results':deepcopy(self.results),'normal_callee_abi':self.callee_abi.copy(),'mutations':deepcopy(self.art_mutation_log)}
        return r

    def mutate(self,phase):
        plans=self.options.get('art_mutations',{});a=plans.get(phase)
        if a is None:return super().mutate(phase)
        self.art_mutation_log.append({'phase':phase,'action':a})
        def write(n,off,v,width=8):
            (self.q if width==8 else self.d)(self.p[n]+off,v);self.allowed.setdefault(n,set()).update(range(off,off+width))
        if a in ['replace_skin','clear_skin']:write(self.options.get('mutation_data','data1'),0xC0,self.p['skin2'] if a=='replace_skin' else 0)
        elif a in ['replace_skin_sprite','clear_skin_sprite']:write(self.options.get('mutation_skin','skin1'),0x38,self.p['sprite0'] if a=='replace_skin_sprite' else 0)
        elif a=='replace_skin_type':write(self.options.get('mutation_skin','skin1'),0x50,self.options.get('replacement_type',0x8000000A),4)
        elif a=='replace_default':write(self.options.get('mutation_data','data1'),0x98,self.p['sprite0'])
        elif a=='class_cold':write('object_class',0xE0,0,4)
        elif a in ['replace_art','clear_art','replace_clipping','clear_clipping']:
            n=a.split('_')[1];write('actor',0x28 if n=='art' else 0x30,self.p[n+'1'] if a.startswith('replace') else 0)
        elif a=='replace_data':write('actor',0x50,self.p['data2'])
        elif a=='clear_data':write('actor',0x50,0)
        elif a=='sprite_dead':self.art_live['sprite2']=False
        elif a=='swap_games':self.art_games['art0'],self.art_games['clipping0']=self.art_games['clipping0'],self.art_games['art0']
        else:raise AssertionError(a)

    def hook(self,uc,address,size,data):
        rva,x=address-self.base,self.x
        if rva in self.traps:raise AssertionError(('terminal trap',hex(rva)))
        if rva in self.added_entries:
            name=self.added_entries[rva];raw=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
            assert self.last_native=={'GetArt':0x36836E,'GetArtType':0x368385,'SetupArt':0x368396}[name]
            frame={'method':name,'raw_args':raw,'entry_sp':self.reg(x.UC_X86_REG_RSP),'return_target':self.rq(self.reg(x.UC_X86_REG_RSP))}
            self.frames.append(frame);self.saved_frames.append({n:self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RBX','RBP','RSI','RDI','R12','R13','R14','R15',*[f'XMM{i}' for i in range(6,16)]]});self.entries.append({'entry':name,'raw_args':raw.copy()})
        if rva in self.added_instructions:
            self.executed.add(rva);self.last_native=rva
            if rva in [0x3B4AE7,0x3B4A57]:self.frames[-1]['captured_skin']=self.oid(self.reg(x.UC_X86_REG_RDI))
            if self.instructions[rva].mnemonic=='ret':
                frame=self.frames[-1];assert self.reg(x.UC_X86_REG_RSP)==frame['entry_sp'];assert all(self.reg(getattr(x,'UC_X86_REG_'+n))==v for n,v in self.saved_frames[-1].items())
                name=frame['method'];result=self.reg(x.UC_X86_REG_RAX);self.callee_abi.append(name)
                self.results.append({'method':name,'result_bits':result if name!='SetupArt' else None,'result_identity':self.oid(result) if name=='GetArt' else None})
                self.frames.pop();self.saved_frames.pop()
            return
        cx,dx,r8,r9=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
        if len(self.frames)>self.active_frame_start:
            name=self.frames[-1]['method'];phase=name+':';kind=None;args=None;value=0;effect=None
            if rva==0x2B7B40:
                assert cx==self.slot;kind='metadata_service';args=[cx-self.base,'object_class'];value=self.rq(cx);phase+='metadata'
            elif rva==0x281D90:
                assert cx==self.p['object_class'] and self.rd(cx+0xE0)==0;kind='class_initialization_service';args=['object_class'];phase+='class_init'
                effect=lambda:(self.d(cx+0xE0,1),self.allowed.setdefault('object_class',set()).update(range(0xE0,0xE4)))
            elif rva==0x1C822C0:
                assert dx==r8==0;live=cx and self.art_live[self.oid(cx)];bits=self.options.get('art_equal_bits' if name=='SetupArt' else name+'_equal_bits')
                value=bits if bits is not None else (0x1234567800000000 if name=='SetupArt' else 0xFACE123456789000)|(0 if live else self.options.get('equal_true_byte',0xFE))
                kind='art_equality_service' if name=='SetupArt' else 'skin_equality_service';args=[self.oid(cx),None,0,value];phase+='equality'
            elif rva==0x1C79FD0:
                assert name=='SetupArt' and self.oid(cx) in self.art_games and dx==0;field='art' if self.last_native==0x36891E else 'clipping' if self.last_native==0x36898A else 'other'
                target=None if self.options.get('null_art_game')==self.oid(cx) else self.art_games[self.oid(cx)];value=0 if target is None else self.p[target];kind='get_gameobject';args=[self.oid(cx),0,target];phase+='getter:'+field
            elif rva==0x1C7D810:
                assert name=='SetupArt' and self.oid(cx) in self.active and r8==0;assert dx==(POISON&~255|1 if self.last_native in [0x368934,0x36899C] else 0)
                kind='set_active';args=[self.oid(cx),dx,0];phase+='active:'+('true' if dx&255 else 'false');effect=lambda:self.active.update({self.oid(cx):bool(dx&255)})
            elif rva==0x1D49700:
                assert name=='SetupArt' and self.oid(cx) in self.components and r8==0;kind=SERVICES[rva];args=[self.oid(cx),self.oid(dx),0];phase+='sprite_set';effect=lambda:self.components[self.oid(cx)].update(sprite=self.oid(dx))
            elif rva==0x2B7D90:
                assert self.last_native in [0x3B4B33,0x3B4A9D,0x3689BB];self.event('native_null_guard',[]);self.error='native_null_guard';uc.emu_stop();return
            if kind:
                self.executed.add(rva)
                if self.event(kind,args):
                    if effect:effect()
                    self.requests.append({'kind':kind,'args':deepcopy(args)});self.mutate(phase);self.ret(value)
                return
        return super().hook(uc,address,size,data)

    def run(self,name,options=None,retained=False):
        self.active_frame_start=len(self.frames) if retained else 0
        prior=len(self.art_mutation_log) if retained else 0
        row=super().run(name,options,retained)
        if not row['options'].get('failure'):
            reached={r['phase'] for r in row['final']['art_join']['mutations'][prior:]}
            assert set(row['options'].get('art_mutations',{}))<=reached,('unreached authored mutation',row['options'],reached)
        return row

def verify_semantics(row,m):
    s=deepcopy(row['initial']);mem={n:bytearray.fromhex(raw) for n,raw in s['memory'].items()};events=[];o=row['options'];error=None;fault=None;regs=row['entry_raw_args'].copy();p=m.p;sp=m.stack+0x18008;input_name=None;captured_acted=None
    def q(n,off):return struct.unpack_from('<Q',mem[n],off)[0]
    def field(n):return m.oid(q('actor',FIELDS[n]))
    def put(off,width,value):struct.pack_into('<Q' if width==8 else '<I',mem['actor'],off,value)
    def snapshot():
        r=deepcopy(s);r['memory']={n:raw.hex() for n,raw in mem.items()};r['fields']={n:field(n) for n in FIELDS};r['left_act_bits']=mem['actor'][0xB0]
        r['data']={n:{'name':m.oid(q(n,0x28)),'background':m.oid(q(n,0xB8)),'color_bits':list(struct.unpack_from('<IIII',mem[n],0xD8))} for n in ['data0','data1','data2']}
        r['reward_join']['pointers']={n:m.oid(q('actor',off)) for n,off in POINTERS.items()};r['reward_join']['words']={n:struct.unpack_from('<I',mem['actor'],off)[0] for n,off in WORDS.items()};r['reward_join']['starting_alignment_bits']={n:struct.unpack_from('<I',mem[n],0x134)[0] for n in ['data0','data1','data2']}
        r['art_join']['fields']={n:m.oid(q('actor',off)) for n,off in [('art',0x28),('clipping',0x30)]};r['art_join']['data']={n:{'skin':m.oid(q(n,0xC0)),'default':m.oid(q(n,0x98))} for n in ['data0','data1','data2']};r['art_join']['skins']={n:{'sprite':m.oid(q(n,0x38)),'type_bits':struct.unpack_from('<I',mem[n],0x50)[0]} for n in ['skin0','skin1','skin2']};return r
    def mutate(phase):
        a=o.get('art_mutations',{}).get(phase)
        if a is not None:
            s['art_join']['mutations'].append({'phase':phase,'action':a})
            def write(n,off,v,width=8):struct.pack_into('<Q' if width==8 else '<I',mem[n],off,v)
            if a in ['replace_skin','clear_skin']:write(o.get('mutation_data','data1'),0xC0,p['skin2'] if a=='replace_skin' else 0)
            elif a in ['replace_skin_sprite','clear_skin_sprite']:write(o.get('mutation_skin','skin1'),0x38,p['sprite0'] if a=='replace_skin_sprite' else 0)
            elif a=='replace_skin_type':write(o.get('mutation_skin','skin1'),0x50,o.get('replacement_type',0x8000000A),4)
            elif a=='replace_default':write(o.get('mutation_data','data1'),0x98,p['sprite0'])
            elif a=='class_cold':write('object_class',0xE0,0,4)
            elif a in ['replace_art','clear_art','replace_clipping','clear_clipping']:
                n=a.split('_')[1];put(0x28 if n=='art' else 0x30,8,p[n+'1'] if a.startswith('replace') else 0)
            elif a=='replace_data':put(0x50,8,p['data2'])
            elif a=='clear_data':put(0x50,8,0)
            elif a=='sprite_dead':s['art_join']['liveness']['sprite2']=False
            elif a=='swap_games':s['art_join']['component_games']['art0'],s['art_join']['component_games']['clipping0']=s['art_join']['component_games']['clipping0'],s['art_join']['component_games']['art0']
            else:raise AssertionError(a)
            return
        a=o.get('init_mutations',{}).get(phase)
        if a is not None:
            fields={'replace_acteds':(0xA8,'acted2'),'clear_acteds':(0xA8,None),'replace_data':(0x50,'data2'),'clear_data':(0x50,None),'replace_name':(0x40,'tmp1'),'clear_name':(0x40,None),
                'replace_state_action':(0x180,'action1'),'clear_state_action':(0x180,None),'replace_bluff':(0x58,'data2'),'replace_register_as':(0x60,'data2')}
            if a in fields:off,n=fields[a];put(off,8,0 if n is None else p[n])
            elif a=='change_input_alignment':assert input_name;struct.pack_into('<I',mem[input_name],0x134,0xF1234567)
            elif a.startswith('change_') and a[7:] in WORDS:put(WORDS[a[7:]],4,0xE1234567)
            elif a=='replace_game_map':s['reward_join']['acted_games'][captured_acted]='game2'
            else:raise AssertionError(a)
        elif o.get('mutation_phase')==phase:
            a=o['mutation']
            if a.startswith(('clear_','replace_')) and a.partition('_')[2] in FIELDS:
                n=a.partition('_')[2];v=0 if a.startswith('clear_') else p['data1' if n=='data' else 'tmp1' if n=='name' else 'bg1' if n=='background' else 'acted2'];put(FIELDS[n],8,v)
            elif a=='replace_tmp_class':struct.pack_into('<Q',mem[field('name')],0,p['tmp_class1'])
            elif a=='replace_background_sprite':struct.pack_into('<Q',mem[field('data')],0xB8,p['background1'])
            else:raise AssertionError(a)
    def service(kind,args,site,raw,phase,effect=None,tail=False):
        nonlocal regs,error
        events.append({'kind':kind,'args':deepcopy(args),'snapshot':snapshot(),'raw_args':raw.copy(),'native_site':hex(site),'return_target':m.stop if tail else m.base+site+m.instructions[site].size})
        if o.get('failure')==[kind,sum(e['kind']==kind for e in events)]:error=kind;return False
        if effect:effect()
        s['requests'].append({'kind':kind,'args':deepcopy(args)});mutate(phase);regs=[POISON]*4;return True
    def guard(raw,site=0x368409):
        nonlocal error
        events.append({'kind':'native_null_guard','args':[],'snapshot':snapshot(),'raw_args':raw.copy(),'native_site':hex(site),'return_target':m.base+site+5});error='native_null_guard'
    def entry(name,raw):s['native_entries'].append({'entry':name,'raw_args':raw.copy()})
    def begin_callee(name,raw,caller):
        entry(name,raw);s['art_join']['frames'].append({'method':name,'raw_args':raw.copy(),'entry_sp':sp-0x40,'return_target':m.base+caller+m.instructions[caller].size})
    def end_callee(name,result):
        s['art_join']['normal_callee_abi'].append(name);s['art_join']['results'].append({'method':name,'result_bits':result if name!='SetupArt' else None,'result_identity':m.oid(result) if name=='GetArt' else None});s['art_join']['frames'].pop()
    def data_getter(name,data,caller):
        nonlocal regs
        raw=[p[data],0,regs[2],regs[3]];regs=raw.copy();begin_callee(name,raw,caller)
        sites={'GetArt':(0x3B4ACD,0x3B4AF0,0x3B4AFD,0x3B4B33),'GetArtType':(0x3B4A3D,0x3B4A60,0x3B4A6D,0x3B4A9D)};meta,init,eq,null=sites[name]
        if not s['art_join']['metadata_flags'][name]:
            if not service('metadata_service',[m.slot-m.base,'object_class'],meta,[m.slot,regs[1],regs[2],regs[3]],name+':metadata'):return None
            s['art_join']['metadata_flags'][name]=1
        skin=m.oid(q(data,0xC0));s['art_join']['frames'][-1]['captured_skin']=skin
        if not struct.unpack_from('<I',mem['object_class'],0xE0)[0]:
            if not service('class_initialization_service',['object_class'],init,[p['object_class'],regs[1],regs[2],regs[3]],name+':class_init',lambda:struct.pack_into('<I',mem['object_class'],0xE0,1)):return None
        result=o.get(name+'_equal_bits');result=(0xFACE123456789000|(0 if skin and s['art_join']['liveness'][skin] else o.get('equal_true_byte',0xFE))) if result is None else result
        if not service('skin_equality_service',[skin,None,0,result],eq,[0 if skin is None else p[skin],0,0,regs[3]],name+':equality'):return None
        if result&255:out=q(data,0x98) if name=='GetArt' else 0
        else:
            skin=m.oid(q(data,0xC0))
            if skin is None:guard(regs,null);return None
            out=q(skin,0x38) if name=='GetArt' else struct.unpack_from('<I',mem[skin],0x50)[0]
        end_callee(name,out);return out
    def setup_art(sprite,typ):
        nonlocal regs
        raw=[p['actor'],sprite,typ,0];regs=raw.copy();begin_callee('SetupArt',raw,0x368396)
        if not s['art_join']['metadata_flags']['SetupArt']:
            if not service('metadata_service',[m.slot-m.base,'object_class'],0x3688D8,[m.slot,regs[1],regs[2],regs[3]],'SetupArt:metadata'):return
            s['art_join']['metadata_flags']['SetupArt']=1
        if not struct.unpack_from('<I',mem['object_class'],0xE0)[0]:
            if not service('class_initialization_service',['object_class'],0x3688F4,[p['object_class'],regs[1],regs[2],regs[3]],'SetupArt:class_init',lambda:struct.pack_into('<I',mem['object_class'],0xE0,1)):return
        sn=m.oid(sprite);result=o.get('art_equal_bits');result=(0x1234567800000000|(0 if sn and s['art_join']['liveness'][sn] else o.get('equal_true_byte',0xFE))) if result is None else result
        if not service('art_equality_service',[sn,None,0,result],0x368901,[sprite,0,0,regs[3]],'SetupArt:equality'):return
        if not result&255:
            selected='clipping' if typ==10 else 'art';other='art' if selected=='clipping' else 'clipping'
            off={'art':0x28,'clipping':0x30};firstget,active,setter=(0x36898A,0x36899C,0x3689B0) if selected=='clipping' else (0x36891E,0x368934,0x368948)
            component=m.oid(q('actor',off[selected]))
            if component is None:guard([0,regs[1],regs[2],regs[3]],0x3689BB);return
            target=None if o.get('null_art_game')==component else s['art_join']['component_games'][component]
            if not service('get_gameobject',[component,0,target],firstget,[p[component],0,regs[2],regs[3]],'SetupArt:getter:'+selected):return
            if target is None:guard(regs,0x3689BB);return
            dx=POISON&~255|1
            if not service('set_active',[target,dx,0],active,[p[target],dx,0,regs[3]],'SetupArt:active:true',lambda:s['reward_join']['active'].update({target:True})):return
            component=m.oid(q('actor',off[selected]))
            if component is None:guard([0,regs[1],regs[2],regs[3]],0x3689BB);return
            if not service(SERVICES[0x1D49700],[component,sn,0],setter,[p[component],sprite,0,regs[3]],'SetupArt:sprite_set',lambda:s['components'][component].update(sprite=sn)):return
            component=m.oid(q('actor',off[other]))
            if component is None:guard([0,regs[1],regs[2],regs[3]],0x3689BB);return
            target=None if o.get('null_art_game')==component else s['art_join']['component_games'][component]
            if not service('get_gameobject',[component,0,target],0x368958,[p[component],0,regs[2],regs[3]],'SetupArt:getter:other'):return
            if target is None:guard(regs,0x3689BB);return
            if not service('set_active',[target,0,0],0x36896A,[p[target],0,0,regs[3]],'SetupArt:active:false',lambda:s['reward_join']['active'].update({target:False})):return
        end_callee('SetupArt',None)
    def reveal():
        nonlocal error,fault
        if not s['metadata_flag']:
            for slot,site,label in [(m.slot,0x3682BD,'object_class'),(m.literal_slot,0x3682C9,'empty')]:
                if not service('metadata_service',[slot-m.base,label],site,[slot,regs[1],regs[2],regs[3]],'metadata'):return
            s['metadata_flag']=1
        if not row['entry_raw_args'][0]:error='native_owner_read_fault';fault='0x3682d5';return
        data,name_receiver=field('data'),field('name')
        if data is None:guard(regs);return
        if q(data,0x28)==0:guard([0,regs[1],regs[2],regs[3]]);return
        name=m.oid(q(data,0x28));upper=None if o.get('uppercase_null') else 'upper'+name[-1]
        if not service(SERVICES[0xF7B1B0],[name,0,upper],0x3682F5,[p[name],0,regs[2],regs[3]],'uppercase'):return
        text='empty' if upper is None else upper
        if name_receiver is None:guard([regs[0],p[text],regs[2],regs[3]]);return
        cls=m.oid(q(name_receiver,0));mi=m.oid(q(cls,0x560))
        if not service('tmp_text_service',[name_receiver,text,mi],0x36831E,[p[name_receiver],p[text],p[mi],regs[3]],'text',lambda:s['components'][name_receiver].update(text=text)):return
        data,name_receiver=field('data'),field('name')
        if data is None:guard([regs[0],0,regs[2],regs[3]]);return
        if name_receiver is None:guard([0,p[data],regs[2],regs[3]]);return
        color=list(struct.unpack_from('<IIII',mem[data],0xD8));cls=m.oid(q(name_receiver,0));mi=m.oid(q(cls,0x2B0))
        if not service('tmp_color_service',[name_receiver,color,mi],0x368359,[p[name_receiver],sp-0x18,p[mi],regs[3]],'color',lambda:s['components'][name_receiver].update(color_bits=color.copy())):return
        data=field('data')
        if data is None:guard([0,regs[1],regs[2],regs[3]]);return
        sprite=data_getter('GetArt',data,0x36836E)
        if error:return
        data=field('data')
        if data is None:guard([0,regs[1],regs[2],regs[3]]);return
        typ=data_getter('GetArtType',data,0x368385)
        if error:return
        setup_art(sprite,typ)
        if error:return
        data=field('data')
        if data is None:guard(regs);return
        background=m.oid(q(data,0xB8))
        if not struct.unpack_from('<I',mem['object_class'],0xE0)[0]:
            if not service('class_initialization_service',['object_class'],0x3683BB,[p['object_class'],regs[1],regs[2],regs[3]],'class_init',lambda:struct.pack_into('<I',mem['object_class'],0xE0,1)):return
        result=0x1234567800000000|(o.get('live_true_byte',0xFE) if background and s['background_liveness'][background] else 0)
        if not service(SERVICES[0x1C82480],[background,None,0,result],0x3683C8,[0 if background is None else p[background],0,0,regs[3]],'inequality'):return
        if result&255:
            data,bg=field('data'),field('background')
            if data is None:guard([regs[0],0,regs[2],regs[3]]);return
            if bg is None:guard([0,p[data],regs[2],regs[3]]);return
            sprite_bg=m.oid(q(data,0xB8))
            if not service(SERVICES[0x1D49700],[bg,sprite_bg,0],0x3683F0,[p[bg],0 if sprite_bg is None else p[sprite_bg],0,regs[3]],'background_set',lambda:s['components'][bg].update(sprite=sprite_bg)):return
        service(SERVICES[0x3694D0],['actor',0],0x368404,[p['actor'],0,regs[2],regs[3]],'view',tail=True)
    def initialize():
        nonlocal error,fault,regs,input_name,captured_acted
        input_name=m.oid(regs[1])
        if not regs[0]:error='native_owner_read_fault';fault='0x365650';return
        captured_acted=field('acted');s['reward_join']['captured_init_entries'].append({'input':input_name,'acted':captured_acted})
        if captured_acted is None:guard([0,regs[1],regs[2],regs[3]],0x36570C);return
        mode=o.get('gameobject_result','normal');game=None if mode=='null' else 'game2' if mode=='other' else s['reward_join']['acted_games'][captured_acted]
        if not service('get_gameobject',[captured_acted,0,game],0x365662,[p[captured_acted],0,regs[2],regs[3]],'get_gameobject'):return
        if game is None:guard(regs,0x36570C);return
        if not service('set_active',[game,0,0],0x365678,[p[game],0,0,regs[3]],'set_active',lambda:s['reward_join']['active'].update({game:False})):return
        for ordinal,(off,site,value) in enumerate([(0x58,0x36568A,0),(0x50,0x365699,0 if input_name is None else p[input_name]),(0x60,0x3656AB,0)],1):
            put(off,8,value)
            if not service('reference_barrier',['actor',off,m.oid(value)],site,[p['actor']+off,value,regs[2],regs[3]],'init_barrier:'+str(ordinal)):return
        put(0xDC,4,1)
        if input_name is None:guard(regs,0x36570C);return
        alignment=struct.unpack_from('<I',mem[input_name],0x134)[0];put(0xF8,4,alignment)
        state=struct.unpack_from('<I',mem['actor'],0xE4)[0];put(0xE0,4,state)
        action=m.oid(q('actor',0x180));put(0xE4,4,5)
        if action:
            code,method=m.oid(q(action,0x40)),m.oid(q(action,0x28));args=[action,code,method,regs[2],regs[3]]
            if not service('state_callback',args,0x3656F5,[0 if code is None else p[code],p[method],regs[2],regs[3]],'callback',lambda:s['reward_join']['callbacks'].append(args.copy())):return
        regs=[p['actor'],0,regs[2],regs[3]];entry('RevealReal',regs);reveal()
    entry(row['entry'],regs)
    if row['entry']=='InitReward':initialize()
    elif row['entry']=='RevealReal':reveal()
    else:
        side=regs[1]&0xFFFFFFFF
        if side in SIDES:
            if not regs[0]:error='native_owner_read_fault';fault=hex({40:0x3689D5,30:0x3689F7,10:0x368A12,20:0x368A2D}[side])
            else:
                value=q('actor',FIELDS[SIDES[side]])
                if side==40:mem['actor'][0xB0]=1
                put(0xA8,8,value);service('reference_barrier',['actor',0xA8,m.oid(value)],{40:0x3689ED,30:0x368A08,10:0x368A23,20:0x368A3E}[side],[p['actor']+0xA8,value,regs[2],regs[3]],'barrier',tail=True)
    assert row['events']==events,('events',row['entry'],o)
    assert row['error']==error and row['returned']==(error is None) and row['fault_rva']==fault
    assert row['final']==snapshot(),('final',row['entry'],o)

def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root);cases=[];seq=[];bases=[];stops=[]
    for meta,cls,data,action in itertools.product([False,True],[False,True],['data0','data1','data2'],[False,True]):cases.append(m.run('InitReward',{'metadata_warm':meta,'class_warm':cls,'input_data':data,'null_state_action':action}))
    for alignment,state in itertools.product([0,10,20,0xFFFFFFFF],[0,5,10,20,30,0x80000000]):cases.append(m.run('InitReward',{'input_alignment':alignment,'state':state}))
    for options in [{'null_owner':True},{'null_field':'acted'},{'gameobject_result':'null'},{'gameobject_result':'other'},{'null_input_data':True},{'null_method_code':True},{'null_field':'name'},{'null_field':'background'},
                    {'input_data':'data0','null_background_sprite':True,'null_field':'background'}, {'input_data':'data2','null_field':'background'},
                    {'alias_games':True},{'alias_bg_tmp':True},{'uppercase_null':True}]:cases.append(m.run('InitReward',options))
    for phase in ['get_gameobject','set_active','init_barrier:1','init_barrier:2','init_barrier:3','callback']:
        for action in ['replace_acteds','clear_acteds','replace_data','clear_data','replace_name','clear_name','replace_state_action','clear_state_action','change_input_alignment','change_state']:
            cases.append(m.run('InitReward',{'init_mutations':{phase:action}}))
    for action in ['change_prev_state','change_alignment','change_uses','replace_bluff','replace_register_as','replace_game_map']:cases.append(m.run('InitReward',{'init_mutations':{'callback' if action!='replace_game_map' else 'get_gameobject':action}}))
    for phase,action in [('uppercase','replace_name'),('uppercase','replace_tmp_class'),('text','replace_data'),('text','clear_data'),('inequality','replace_background_sprite')]:cases.append(m.run('InitReward',{'mutation_phase':phase,'mutation':action}))
    for options in [{'init_mutations':{'init_barrier:3':'replace_name','callback':'replace_data'}},
                    {'init_mutations':{'init_barrier:3':'change_input_alignment','callback':'change_input_alignment'}},
                    {'alias_games':True,'init_mutations':{'get_gameobject':'replace_game_map'}},
                    {'null_state_action':True,'init_mutations':{'init_barrier:3':'replace_state_action'}}]:cases.append(m.run('InitReward',options))
    for side in [0,10,20,30,40,0xFFFFFFFF]:cases.append(m.run('SetupObject',{'side_bits':side}))
    for side,alias in itertools.product([10,20,30,40],[False,True]):
        seq.append([m.run('SetupObject',{'side_bits':side,'alias_games':alias}),m.run('InitReward',{'input_data':'data1','init_mutations':{'callback':'replace_data'}},True),m.run('SetupObject',{'side_bits':40},True),m.run('InitReward',{'input_data':'data0','init_mutations':{'callback':'replace_name'}},True)])
    seq.append([m.run('InitReward',{'null_input_data':True}),m.run('InitReward',{'input_data':'data1'},True),m.run('RevealReal',{},True)])
    seq.append([m.run('SetupObject',{'side_bits':10,'alias_acteds':True}),m.run('InitReward',{'init_mutations':{'get_gameobject':'replace_acteds'}},True),m.run('SetupObject',{'side_bits':40},True),m.run('InitReward',{'input_data':'data0'},True)])
    for warm,class_warm,flags,skin in itertools.product([False,True],[False,True],itertools.product([0,0xFE],repeat=3),['skin1',None]):
        cases.append(m.run('InitReward',{'metadata_warm':warm,'class_warm':class_warm,'extra_metadata_flags':dict(zip(['GetArt','GetArtType','SetupArt'],flags)),'skin1':skin}))
    for bits,skin,live,forced in itertools.product([0,10,0x8000000A,0xFFFFFFFF],['skin1',None],[False,True],[None,0xFEDCBA9876543200,0xFEDCBA98765432FE]):
        cases.append(m.run('InitReward',{'skin_type_bits':bits,'skin1':skin,'art_liveness':{'skin1':live},'GetArt_equal_bits':forced,'GetArtType_equal_bits':forced}))
    for options in [{'alias_images':True},{'alias_art_games':True},{'alias_skins':True},{'alias_sprites':True},{'alias_images':True,'alias_art_games':True,'alias_skins':True,'alias_sprites':True},
                    {'null_art_field':'art'},{'null_art_field':'clipping'},{'null_art_game':'art0'},{'null_art_game':'clipping0'},{'null_skin_sprite':True},
                    {'null_art_field':'art','art_equal_bits':0x12345678000000FE},{'null_art_field':'clipping','null_skin_sprite':True}]:cases.append(m.run('InitReward',options))
    phases=['GetArt:metadata','GetArt:class_init','GetArt:equality','GetArtType:metadata','GetArtType:class_init','GetArtType:equality','SetupArt:metadata','SetupArt:class_init','SetupArt:equality']
    for phase,action in itertools.product(phases,['replace_skin','clear_skin','replace_skin_sprite','clear_skin_sprite','replace_skin_type','replace_default','replace_data','clear_data','class_cold']):
        plans={phase:action}
        if phase=='GetArtType:class_init':plans['GetArt:equality']='class_cold'
        if phase=='SetupArt:class_init':plans['GetArtType:equality']='class_cold'
        cases.append(m.run('InitReward',{'art_mutations':plans}))
    for phase,action in itertools.product(['SetupArt:getter:clipping','SetupArt:active:true','SetupArt:sprite_set','SetupArt:getter:other','SetupArt:active:false'],['replace_art','clear_art','replace_clipping','clear_clipping','swap_games','replace_data','clear_data','class_cold']):
        cases.append(m.run('InitReward',{'art_mutations':{phase:action}}))
    for options in [{'art_mutations':{'GetArt:equality':'replace_data'}},{'art_mutations':{'GetArt:equality':'class_cold','GetArtType:equality':'class_cold','SetupArt:equality':'class_cold'}},
                    {'art_mutations':{'GetArt:equality':'replace_skin','GetArtType:equality':'replace_skin_type'}},
                    {'mutation_skin':'skin2','art_mutations':{'GetArt:equality':'replace_skin','GetArtType:equality':'replace_skin_type'}},
                    {'skin_type_bits':0,'art_mutations':{'SetupArt:active:true':'replace_art'}},
                    {'skin_type_bits':0,'art_mutations':{'SetupArt:getter:art':'swap_games'}},
                    {'art_equal_bits':0x12345678000000FE,'art_mutations':{'SetupArt:equality':'clear_data'}}]:cases.append(m.run('InitReward',options))
    for options in [{'alias_images':True},{'alias_art_games':True},{'art_mutations':{'GetArt:equality':'replace_data'}},{'art_mutations':{'GetArt:equality':'class_cold','GetArtType:equality':'class_cold','SetupArt:equality':'class_cold'}}]:
        setup_options={k:v for k,v in options.items() if k!='art_mutations'}
        seq.append([m.run('SetupObject',{'side_bits':40,**setup_options}),m.run('InitReward',options,True),m.run('SetupObject',{'side_bits':10},True),m.run('InitReward',{'input_data':'data0'},True),m.run('RevealReal',{},True)])
    profiles=[('SetupObject',{'side_bits':40}),('InitReward',{}),('InitReward',{'metadata_warm':True,'class_warm':False}),('InitReward',{'init_mutations':{'get_gameobject':'clear_acteds'}}),('InitReward',{'init_mutations':{'init_barrier:3':'replace_state_action'}}),('InitReward',{'init_mutations':{'callback':'replace_data'}}),('InitReward',{'init_mutations':{'callback':'replace_name'}}),('InitReward',{'art_mutations':{'GetArt:equality':'replace_data'}}),('InitReward',{'init_mutations':{'init_barrier:3':'replace_name','callback':'replace_data'}}),
              ('InitReward',{'skin1':None}),('InitReward',{'skin_type_bits':0}),('InitReward',{'alias_images':True}),('InitReward',{'art_mutations':{'GetArt:equality':'class_cold','GetArtType:equality':'class_cold','SetupArt:equality':'class_cold'}}),('InitReward',{'art_mutations':{'SetupArt:active:true':'replace_clipping'}})]
    for name,options in profiles:
        baseline=m.run(name,options);assert baseline['returned'];bid=len(bases);bases.append(baseline);counts={}
        for i,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;row=m.run(name,dict(options,failure=[kind,counts[kind]]));assert not row['returned'] and row['events']==baseline['events'][:i+1] and row['final']==e['snapshot'];stops.append({'baseline':bid,'prefix_length':i+1,'result':row})
    excluded=m.traps;missing=set(m.instructions)-m.executed-excluded;assert not missing,[hex(a) for a in sorted(missing)];assert not m.executed&excluded
    return {'build':BUILD,'schema':'character_reward_art_join_native_v1','targets':m.targets,'bounds':m.bounds,'supplied_declarations':m.supplied,
        'field_offsets':{'character_pointers':dict(FIELDS,**POINTERS,art=0x28,clipping=0x30),'character_words':WORDS,'left_act_byte':0xB0,'data':{'name':0x28,'default_art':0x98,'skin':0xC0,'background':0xB8,'color_bytes16':0xD8,'starting_alignment_word':0x134},'skin':{'art':0x38,'type_word':0x50},'delegate':{'invoke_impl':0x18,'unused_managed_target':0x20,'method':0x28,'method_code':0x40}},
        'metadata_flag_rva':hex(m.flag),'metadata_slot_rvas':[hex(m.slot-m.base),hex(m.literal_slot-m.base)],
        'virtual_slots':{'text':{'slot':66,'function_offset':0x558,'method_offset':0x560},'color':{'slot':23,'function_offset':0x2A8,'method_offset':0x2B0}},
        'authored_records':{n:{'address_bits':p,'retained_bytes':m.sizes[n]} for n,p in m.p.items()},
        'init_operand_assertions':{hex(a):list(v) for a,v in m.init_checks.items()},'presentation_operand_assertions':{hex(a):list(v) for a,v in m.checks.items()},'added_operand_assertions':{hex(a):list(v) for a,v in m.added_checks.items()},'extra_metadata_flag_rvas':{n:hex(f) for n,f in m.extra_flags.items()},'terminal_traps_excluded':[hex(a) for a in sorted(m.traps)],'decoded_instructions':len(m.instructions),'executed_instructions':len(set(m.instructions)&m.executed),'observed_addresses':len(m.executed),'cases':cases,'sequences':seq,'baselines':bases,'stops':stops,'summary':{'cases':len(cases),'normal_returns':sum(r['returned'] for r in cases),'native_stops':sum(not r['returned'] for r in cases),'sequences':len(seq),'baselines':len(bases),'stops':len(stops)},'scope':'Actual SetupObject/InitReward/RevealReal/GetArt/GetArtType/SetupArt one graph. GetAnimatedArt excluded. GC/Unity/Action, TMP, uppercase and UpdateViewReal/runtime services supplied; no renderer/scheduler/runtime admission/unwinding. Full snapshots and independently modeled service-entry ABI/caller/state/native frames.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('game_root');p.add_argument('dumper_root');p.add_argument('--output',required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);Path(a.output).write_text(json.dumps(pool_snapshots(pool_memory(r)),sort_keys=True,separators=(',',':'))+'\n',encoding='utf-8');print(json.dumps(r['summary']))
