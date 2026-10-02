"""Eight actual reward/art/color/RefreshView callers in one physical graph."""
import argparse
import itertools
import json
import struct
import hashlib
import re
from copy import deepcopy
from pathlib import Path
from audit_character_assets import BUILD
from audit_character_reward_presentation import Machine as PresentationMachine, FIELDS, SIDES, SERVICES, POISON
from audit_character_init_reward import Machine as InitVerifier, ENTRY, END, FOLLOWING
from audit_character_art_preferences import Machine as ArtVerifier, TARGETS as ART_TARGETS
from audit_character_data_consumers import Machine as DataVerifier, TARGETS as DATA_TARGETS
from audit_character_oracle_reveal_join import pool_memory, expand_memory as expand_raw_memory
from audit_report_snapshots import pool_snapshots, expand_snapshots
from audit_character_view_presentation import Machine as ViewVerifier

POINTERS={'bluff':0x58,'register_as':0x60,'state_action':0x180}
WORDS={'uses':0xDC,'prev_state':0xE0,'state':0xE4,'alignment':0xF8}

HISTORY_KEYS={'requests','native_entries','results','normal_callee_abi','callbacks','captured_init_entries','mutations'}

def history_digest(value):return hashlib.sha256(json.dumps(value,sort_keys=True,separators=(',',':'),ensure_ascii=True).encode('utf-8')).hexdigest()

def pool_histories(report):
    """Intern complete ordered diagnostic histories without dropping any entries."""
    assert 'history_blobs' not in report and 'history_encoding' not in report
    blobs={}
    def encode(value):
        if isinstance(value,list):return [encode(v) for v in value]
        if not isinstance(value,dict):return value
        assert set(value)!={'history_sha256'}
        out={}
        for key,item in value.items():
            if key in HISTORY_KEYS and isinstance(item,list):
                digest=history_digest(item);assert digest not in blobs or blobs[digest]==item
                blobs[digest]=deepcopy(item);out[key]={'history_sha256':digest}
            else:out[key]=encode(item)
        return out
    result=encode(report);result['history_encoding']='sha256-complete-ordered-history-v1';result['history_blobs']=blobs
    assert expand_histories(result)==report
    return result

def expand_histories(report):
    assert report['history_encoding']=='sha256-complete-ordered-history-v1'
    blobs=report['history_blobs']
    for digest,value in blobs.items():assert isinstance(value,list) and history_digest(value)==digest
    def decode(value):
        if isinstance(value,list):return [decode(v) for v in value]
        if not isinstance(value,dict):return value
        if set(value)=={'history_sha256'}:return deepcopy(blobs[value['history_sha256']])
        return {k:decode(v) for k,v in value.items()}
    return decode({k:v for k,v in report.items() if k not in ['history_encoding','history_blobs']})

def verify_history_codec():
    history=[{'address_bits':(1<<64)-1,'nullable':None,'raw':'00a5ff','flag':False,'arguments':[2,1,0]}]
    source={'initial':{'requests':deepcopy(history),'callbacks':[]},'final':{'requests':deepcopy(history)},'options':{'mutations':{'service:1':[['actor',8,8,0]]}}}
    encoded=pool_histories(source);assert encoded['initial']['requests']==encoded['final']['requests'] and encoded['options']==source['options']
    decoded=expand_histories(encoded);decoded['initial']['requests'][0]['arguments'][0]=99;assert decoded['final']['requests']==history and source['initial']['requests']==history
    reversed_history=[{'ordinal':1},{'ordinal':2}];assert history_digest(reversed_history)!=history_digest(list(reversed(reversed_history)))
    tampered=deepcopy(encoded);key=encoded['initial']['requests']['history_sha256'];tampered['history_blobs'][key][0]['nullable']='changed'
    try:expand_histories(tampered)
    except AssertionError:pass
    else:raise AssertionError('corrupt ordered history accepted')

def pool_memory_maps(report):
    """Intern entire named memory-reference maps after the raw-byte codec."""
    assert 'memory_map_blobs' not in report and 'memory_map_encoding' not in report
    blobs={}
    def encode(value):
        if isinstance(value,list):return [encode(v) for v in value]
        if not isinstance(value,dict):return value
        assert set(value)!={'memory_map_sha256'}
        out={}
        for key,item in value.items():
            if key=='memory':
                assert isinstance(item,dict) and all(set(v)=={'memory_sha256'} for v in item.values())
                digest=history_digest(item);assert digest not in blobs or blobs[digest]==item
                blobs[digest]=deepcopy(item);out[key]={'memory_map_sha256':digest}
            else:out[key]=encode(item)
        return out
    result=encode(report);result['memory_map_encoding']='sha256-complete-named-memory-map-v1';result['memory_map_blobs']=blobs
    assert expand_memory_maps(result)==report
    return result

def expand_memory_maps(report):
    assert report['memory_map_encoding']=='sha256-complete-named-memory-map-v1'
    blobs=report['memory_map_blobs']
    for digest,value in blobs.items():assert isinstance(value,dict) and history_digest(value)==digest
    def decode(value):
        if isinstance(value,list):return [decode(v) for v in value]
        if not isinstance(value,dict):return value
        if set(value)=={'memory_map_sha256'}:return deepcopy(blobs[value['memory_map_sha256']])
        return {k:decode(v) for k,v in value.items()}
    return decode({k:v for k,v in report.items() if k not in ['memory_map_encoding','memory_map_blobs']})

def verify_memory_map_codec():
    mapping={'actor':{'memory_sha256':'a'*64},'image':{'memory_sha256':'b'*64}}
    source={'initial':{'memory':deepcopy(mapping)},'final':{'memory':deepcopy(mapping)},'other':{'memory':{}}}
    encoded=pool_memory_maps(source);assert encoded['initial']['memory']==encoded['final']['memory']
    assert history_digest(mapping)!=history_digest({'actor':mapping['image'],'image':mapping['actor']})
    decoded=expand_memory_maps(encoded);decoded['initial']['memory']['actor']['memory_sha256']='c'*64;assert decoded['final']['memory']==mapping and source['initial']['memory']==mapping
    tampered=deepcopy(encoded);key=encoded['initial']['memory']['memory_map_sha256'];tampered['memory_map_blobs'][key]['actor']['memory_sha256']='d'*64
    try:expand_memory_maps(tampered)
    except AssertionError:pass
    else:raise AssertionError('corrupt complete memory map accepted')

STATE_MAP_KEYS={'reward_join','art_join','color_join','refresh_join','components','types','background_liveness','liveness','transforms','active','acted_games','component_games','data_colors','arrays','fields','pointers','words','data','skins','starting_alignment_bits','metadata_flags','metadata_slots'}

def pool_state_maps(report):
    """Intern complete named logical maps without deleting fields or ordered values."""
    assert 'state_map_blobs' not in report and 'state_map_encoding' not in report
    blobs={}
    def encode(value):
        if isinstance(value,list):return [encode(v) for v in value]
        if not isinstance(value,dict):return value
        assert set(value)!={'state_map_sha256'}
        out={}
        for key,item in value.items():
            if key in STATE_MAP_KEYS and isinstance(item,dict):
                digest=history_digest(item);assert digest not in blobs or blobs[digest]==item
                blobs[digest]=deepcopy(item);out[key]={'state_map_sha256':digest}
            else:out[key]=encode(item)
        return out
    result=encode(report);result['state_map_encoding']='sha256-complete-authored-state-map-v1';result['state_map_blobs']=blobs
    assert expand_state_maps(result)==report
    return result

def expand_state_maps(report):
    assert report['state_map_encoding']=='sha256-complete-authored-state-map-v1'
    blobs=report['state_map_blobs']
    for digest,value in blobs.items():assert isinstance(value,dict) and history_digest(value)==digest
    def decode(value):
        if isinstance(value,list):return [decode(v) for v in value]
        if not isinstance(value,dict):return value
        if set(value)=={'state_map_sha256'}:return deepcopy(blobs[value['state_map_sha256']])
        return {k:decode(v) for k,v in value.items()}
    return decode({k:v for k,v in report.items() if k not in ['state_map_encoding','state_map_blobs']})

def verify_state_map_codec():
    value={'actor':{'position_bits':[0x80000000,0x7FC01234,1],'identity_bits':(1<<64)-1,'nullable':None},'other':{'position_bits':[3,2,1]}}
    joined={'transforms':deepcopy(value),'liveness':{'actor':True}}
    source={'first':{'transforms':deepcopy(value),'fields':{},'refresh_join':deepcopy(joined)},'second':{'transforms':deepcopy(value),'refresh_join':deepcopy(joined)},'metadata':{'caption':'retained'}}
    encoded=pool_state_maps(source);assert encoded['first']['transforms']==encoded['second']['transforms']
    decoded=expand_state_maps(encoded);assert decoded==source;decoded['first']['transforms']['actor']['position_bits'][0]=9;assert decoded['second']['transforms']==value and source['first']['transforms']==value
    assert encoded['first']['refresh_join']==encoded['second']['refresh_join'];decoded['first']['refresh_join']['liveness']['actor']=False;assert decoded['second']['refresh_join']==joined and source['first']['refresh_join']==joined
    changed=deepcopy(value);changed['actor']['position_bits'].reverse();assert history_digest(changed)!=history_digest(value)
    renamed={'other':value['actor'],'actor':value['other']};assert history_digest(renamed)!=history_digest(value)
    corrupt=deepcopy(encoded);digest=encoded['first']['transforms']['state_map_sha256'];corrupt['state_map_blobs'][digest]['actor']['nullable']='corruption'
    try:expand_state_maps(corrupt)
    except AssertionError:pass
    else:raise AssertionError('corrupt complete state map accepted')

def expand_memory(report):
    """Root integration adapter: input has already expanded full snapshots."""
    return expand_raw_memory(expand_histories(expand_memory_maps(expand_state_maps(report))))

def expand_report(report):
    """Complete decoder: snapshots, state maps, memory maps, histories, raw bytes."""
    return expand_memory(expand_snapshots(report))

def verify_full_report_codec():
    history=[{'ordinal':1,'address_bits':(1<<64)-1,'nullable':None},{'ordinal':2,'raw_args':[0,2,1]}]
    snapshot={'memory':{'actor':'00a5ff','image':'010203'},'requests':history,'components':{'image':{'nullable':None,'ordered':[2,1,0]}}}
    source={'initial':deepcopy(snapshot),'final':deepcopy(snapshot),'summary':{'complete':True}}
    encoded=pool_snapshots(pool_state_maps(pool_memory_maps(pool_histories(pool_memory(source)))))
    decoded=expand_report(encoded);assert decoded==source and expand_memory(expand_snapshots(encoded))==source
    decoded['initial']['requests'][0]['ordinal']=99;decoded['initial']['memory']['actor']='ffff';decoded['initial']['components']['image']['ordered'][0]=9;assert decoded['final']==snapshot and source['initial']==snapshot
    for name in ['snapshot_blobs','state_map_blobs','memory_map_blobs','history_blobs','memory_blobs']:
        corrupt=deepcopy(encoded);key=next(iter(corrupt[name]))
        if name=='snapshot_blobs':corrupt[name][key]['unexpected']=True
        elif name=='state_map_blobs':corrupt[name][key]['unexpected']=True
        elif name=='memory_map_blobs':corrupt[name][key]['actor']['memory_sha256']='f'*64
        elif name=='history_blobs':corrupt[name][key].reverse()
        else:corrupt[name][key]='aa'+corrupt[name][key][2:]
        try:expand_report(corrupt)
        except AssertionError:pass
        else:raise AssertionError(('corrupt full report accepted',name))

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
            assert exc.errno==self.unicorn.UC_ERR_READ_UNMAPPED
            if pc in [0x367CA7,0x367D15]:self.error='native_vector_read_fault'
            else:assert self.options.get('null_owner') and pc in [0x365650,0x3689D5,0x3689F7,0x368A12,0x368A2D,0x3682D5];self.error='native_owner_read_fault'
            self.fault=hex(pc)
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

class RewardArtMachine(RewardMachine):
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

class Machine(RewardArtMachine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root,dumper_root)
        vv=ViewVerifier(game_root,dumper_root)
        self.metadata=vv.metadata
        rows=[r for r in self.metadata['ScriptMethod'] if r['Name']=='UnityEngine.UI.Graphic$$set_color'];assert len(rows)==1 and rows[0]['Address']==0x1D41930 and rows[0]['TypeSignature']=='viii' and rows[0]['Signature']=='void UnityEngine_UI_Graphic__set_color (UnityEngine_UI_Graphic_o* __this, UnityEngine_Color_o value, const MethodInfo* method);'
        graphic=re.search(r'^public abstract class Graphic : UIBehaviour, ICanvasElement // TypeDefIndex: 9897\s*\{(.*?)\n\}',vv.dump,re.M|re.S);assert graphic and '// RVA: 0x1D41930 Offset: 0x1D40530 VA: 0x181D41930 Slot: 23\n\tpublic virtual void set_color(Color value) { }' in graphic[1]
        self.supplied+=rows
        for declaration,fields in [('public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487',['public Image bg; // 0x120','public Image[] borders; // 0x128']),('public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845',['public Color cardBgColor; // 0xF8','public Color cardBorderColor; // 0x108'])]:
            block=re.search('^'+re.escape(declaration)+r'\s*\{(.*?)\n\}',vv.dump,re.M|re.S);assert block and all(f in block[1] for f in fields)
            if 'TypeDefIndex: 5487' in declaration:
                declarations=re.findall(r'// RVA: (0x[0-9A-F]+).*?\n\s*([^\n]+)',block[1]);assert declarations[65]==('0x3694D0','public void UpdateViewReal() { }')
        rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==0x3694D0];assert len(rows)==1 and rows[0]['Name']=='Character$$UpdateViewReal' and rows[0]['Signature']=='void Character__UpdateViewReal (Character_o* __this, const MethodInfo* method);' and rows[0]['TypeSignature']=='vii'
        self.targets.append(dict(rows[0],method_id='tdi5487.m0065'));assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address']>0x3694D0)==0x3695A0
        chunks=[(e.struct.BeginAddress,e.struct.EndAddress) for e in self.pe.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress==0x3694D0];assert chunks==[(0x3694D0,0x36959C)]
        section=self.pe.get_section_by_rva(0x3694D0);assert section and 0x3695A0<=section.VirtualAddress+section.SizeOfRawData
        raw=self.pe.get_data(0x3694D0,0xD0);assert len(raw)==0xD0 and raw[0xCC:]==bytes([0xCC])*4
        self.color_body_identity={'byte_length':0xCC,'sha256':hashlib.sha256(raw[:0xCC]).hexdigest()}
        ins=list(self.cs.disasm(raw[:0xCC],0x3694D0));assert sum(i.size for i in ins)==0xCC;self.color_instructions={i.address:i for i in ins};self.instructions.update(self.color_instructions)
        self.bounds['0x3694d0']={'end_exclusive':'0x36959c','next_managed':'0x3695a0','unwind_chunks':[['0x3694d0','0x36959c']],'padding_bytes':4}
        self.color_checks={0x3694DF:('mov','rdx, qword ptr [rcx + 0x50]'),0x3694EF:('mov','rcx, qword ptr [rcx + 0x120]'),0x3694FF:('movups','xmm0, xmmword ptr [rdx + 0xf8]'),0x369513:('mov','r8, qword ptr [rax + 0x2b0]'),0x36951A:('call','qword ptr [rax + 0x2a8]'),0x369520:('mov','rdi, qword ptr [rsi + 0x128]'),0x369530:('cmp','eax, dword ptr [rdi + 0x18]'),0x369535:('cmp','ebx, dword ptr [rdi + 0x18]'),0x36953A:('mov','rdx, qword ptr [rsi + 0x50]'),0x369546:('mov','rcx, qword ptr [rdi + rax*8 + 0x20]'),0x369550:('movups','xmm0, xmmword ptr [rdx + 0x108]'),0x36956B:('call','qword ptr [rax + 0x2a8]'),0x36958B:('jmp','0x367b60'),0x369590:('call','0x2b7d90'),0x369595:('int3',''),0x369596:('call','0x2b7d80'),0x36959B:('int3','')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.color_checks.items())
        assert not any(i.op_str in ['0x363f10','0x3643f0'] for i in ins if i.mnemonic in ['call','jmp'])
        self.traps.update([0x369595,0x36959B]);self.unreached_gateway=0x369596
        self.supplied=[r for r in self.supplied if r['Name']!='Character$$UpdateViewReal'];rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==0x367B60 and r['Name']=='Character$$RefreshView'];assert len(rows)==1 and rows[0]['Signature']=='void Character__RefreshView (Character_o* __this, const MethodInfo* method);';self.supplied+=rows
        for i,n in enumerate(['colorbg0','colorbg1','border0','border1','border2','borders0','borders1','image_class0','image_class1','image_mi0','image_mi1']):
            self.p[n]=self.arena+0x800000+i*0x1000;self.ids[self.p[n]]=n;self.sizes[n]=0x400 if n.startswith('image_class') else 0x100 if n.startswith('borders') else 0x80
        self.color_ready=False

    def prepare(self,options):
        self.color_ready=False;super().prepare(options)
        for i,n in enumerate(['image_class0','image_class1']):self.q(self.p[n]+0x2A8,self.color_gateway);self.q(self.p[n]+0x2B0,self.p['image_mi'+str(i)])
        for n in ['colorbg0','colorbg1','border0','border1','border2']:self.q(self.p[n],self.p['image_class0'])
        self.components.update({n:{'text':None,'color_bits':[0xA5A5A5A5]*4,'sprite':None} for n in ['colorbg0','colorbg1','border0','border1','border2']})
        self.q(self.p['actor']+0x120,0 if options.get('null_color_bg') else self.p[options.get('color_bg','colorbg0')]);self.q(self.p['actor']+0x128,0 if options.get('null_borders') else self.p['borders0'])
        for i,n in enumerate(['borders0','borders1']):
            self.q(self.p[n]+0x18,options.get('border_length_bits',0xFACE000000000000|3) if i==0 else 1)
            for j in range(3):
                ref=None if options.get('null_border_index')==j and i==0 else options.get('border_alias','border'+str(j)) if i==0 else 'colorbg1';self.q(self.p[n]+0x20+8*j,0 if ref is None else self.p[ref])
        for n in [options.get('color_bg','colorbg0'),options.get('border_alias','border0')]:
            if not n.startswith('tmp'):self.q(self.p[n],self.p['image_class0'])
        self.colors={n:{'background':[0x3F800000,0x80000000,0x7FC01234,i],'border':[0x7F800000,0xFF800000,0x3F000000,i+3]} for i,n in enumerate(['data0','data1','data2'])}
        for n,values in self.colors.items():
            for k,off in [('background',0xF8),('border',0x108)]:self.u.mem_write(self.p[n]+off,struct.pack('<IIII',*values[k]))
        self.color_mutations=[];self.color_count=0;self.color_ready=True

    def snapshot(self):
        r=super().snapshot()
        if not getattr(self,'color_ready',False):return r
        r['color_join']={'fields':{n:self.oid(self.rq(self.p['actor']+off)) for n,off in [('background',0x120),('borders',0x128)]},
            'data_colors':{n:{k:list(struct.unpack('<IIII',self.u.mem_read(self.p[n]+off,16))) for k,off in [('background',0xF8),('border',0x108)]} for n in ['data0','data1','data2']},
            'arrays':{n:{'length_bits':self.rq(self.p[n]+0x18),'slots':[self.oid(self.rq(self.p[n]+0x20+8*i)) for i in range(3)]} for n in ['borders0','borders1']},'mutations':deepcopy(self.color_mutations)}
        return r

    def mutate(self,phase):
        a=self.options.get('color_mutations',{}).get(phase)
        if a is None:return super().mutate(phase)
        self.color_mutations.append({'phase':phase,'action':a})
        def write(n,off,v,width=8):
            if width==16:self.u.mem_write(self.p[n]+off,struct.pack('<IIII',*v))
            else:(self.q if width==8 else self.d)(self.p[n]+off,v)
            self.allowed.setdefault(n,set()).update(range(off,off+width))
        if a=='replace_data':write('actor',0x50,self.p['data2'])
        elif a=='clear_data':write('actor',0x50,0)
        elif a in ['replace_borders','clear_borders']:write('actor',0x128,self.p['borders1'] if a=='replace_borders' else 0)
        elif a in ['replace_background','clear_background']:write('actor',0x120,self.p['colorbg1'] if a=='replace_background' else 0)
        elif a in ['shrink_array','grow_array','negative_array']:write('borders0',0x18,{'shrink_array':0,'grow_array':3,'negative_array':0xFFFFFFFF}[a],4)
        elif a in ['clear_next_border','replace_next_border']:write('borders0',0x28,self.p['colorbg1'] if a=='replace_next_border' else 0)
        elif a=='replace_image_class':write(self.options.get('class_component','border1'),0,self.p['image_class1'])
        elif a=='replace_border_color':write(self.options.get('color_data','data1'),0x108,[0x80000000,0x7FC0ABCD,0x00000001,0xFFFFFFFF],16)
        else:raise AssertionError(a)

    def hook(self,uc,address,size,data):
        rva,x=address-self.base,self.x
        if rva==0x3694D0:
            assert self.last_native==0x368404;raw=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
            self.frames.append({'method':'UpdateViewReal','raw_args':raw,'entry_sp':self.reg(x.UC_X86_REG_RSP),'return_target':self.rq(self.reg(x.UC_X86_REG_RSP))});self.saved_frames.append({n:self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RBX','RBP','RSI','RDI','R12','R13','R14','R15',*[f'XMM{i}' for i in range(6,16)]]});self.entries.append({'entry':'UpdateViewReal','raw_args':raw.copy()})
        if rva in self.color_instructions:
            assert rva not in self.traps and rva!=self.unreached_gateway;self.executed.add(rva);self.last_native=rva
            if rva==0x3694EF:self.frames[-1]['background_data']=self.oid(self.reg(x.UC_X86_REG_RDX))
            if rva==0x369527:self.frames[-1]['captured_array']=self.oid(self.reg(x.UC_X86_REG_RDI))
            if rva==0x369543:self.frames[-1]['border_data']=self.oid(self.reg(x.UC_X86_REG_RDX))
            if rva==0x36958B:
                assert self.reg(x.UC_X86_REG_RSP)==self.frames[-1]['entry_sp'] and all(self.reg(getattr(x,'UC_X86_REG_'+n))==v for n,v in self.saved_frames[-1].items())
                self.frames.pop();self.saved_frames.pop();self.callee_abi.append('UpdateViewReal');self.results.append({'method':'UpdateViewReal','result_bits':None,'result_identity':None})
            return
        cx,dx,r8,r9=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']]
        if address==self.color_gateway and self.last_native in [0x36951A,0x36956B]:
            self.executed.add(rva);assert self.oid(cx) in self.components and r8==self.rq(self.rq(cx)+0x2B0)
            bits=list(struct.unpack('<IIII',uc.mem_read(dx,16)));args=[self.oid(cx),bits,self.oid(r8)];phase='UpdateViewReal:background' if self.last_native==0x36951A else 'UpdateViewReal:border:'+str(self.color_count+1)
            if self.event('image_color_service',args):
                self.components[self.oid(cx)]['color_bits']=bits.copy();self.requests.append({'kind':'image_color_service','args':deepcopy(args)});self.mutate(phase);self.color_count+=self.last_native==0x36956B;self.ret()
            return
        if rva==0x2B7D90 and self.last_native==0x369590:
            self.executed.add(rva);self.event('native_null_guard',[]);self.error='native_null_guard';uc.emu_stop();return
        if rva==0x367B60:
            self.executed.add(rva);assert cx==self.p['actor'] and dx==0 and self.last_native==0x36958B;args=['actor',0]
            if self.event('Character$$RefreshView',args):self.requests.append({'kind':'Character$$RefreshView','args':args.copy()});self.mutate('refresh_view');self.ret()
            return
        return super().hook(uc,address,size,data)

    def run(self,name,options=None,retained=False):
        self.color_count=0;prior=len(self.color_mutations) if retained else 0
        row=super().run(name,options,retained)
        if not row['options'].get('failure'):
            reached={r['phase'] for r in row['final']['color_join']['mutations'][prior:]};assert set(row['options'].get('color_mutations',{}))<=reached,('unreached color mutation',row['options'],reached)
        return row

class RefreshMachine(Machine):
    def __init__(self,game_root,dumper_root):
        super().__init__(game_root,dumper_root)
        self.refresh_ready=False
        dump=(Path(dumper_root)/'dump.cs').read_text(encoding='utf-8-sig')
        block=re.search(r'^public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487\s*\{(.*?)\n\}',dump,re.M|re.S);assert block
        assert '// RVA: 0x367B60 Offset: 0x366760 VA: 0x180367B60\n\tprivate void RefreshView() { }' in block[1]
        declarations=re.findall(r'// RVA: (0x[0-9A-F]+).*?\n\s*([^\n]+)',block[1]);assert declarations[63]==('0x367B60','private void RefreshView() { }')
        states=re.search(r'^public enum ECharacterState // TypeDefIndex: 5489\s*\{(.*?)\n\}',dump,re.M|re.S);assert states
        for n,v in [('None',0),('Hidden',5),('Alive',10),('Dead',20),('Revealed',30)]:assert f'public const ECharacterState {n} = {v};' in states[1]
        self.refresh_fields={'icon':0x20,'rip':0x78,'prefab':0x80,'disguise':0x88,'created':0x98,'pickable':0x1A8}
        for line in ['public Transform icon; // 0x20','public GameObject ripView; // 0x78','public GameObject deadPrefab; // 0x80','public GameObject disguiseIcon; // 0x88','public GameObject createdDeadPrefab; // 0x98','public GameObject pickable; // 0x1A8','public bool revealed; // 0xD8','public bool killedByDemon; // 0xED']:assert line in block[1]
        vector=re.search(r'^public struct Vector3 : [^\n]* // TypeDefIndex: 6699\s*\{(.*?)\n\}',dump,re.M|re.S);assert vector
        namespace=dump.rfind('// Namespace:',0,vector.start());assert dump[namespace:dump.find('\n',namespace)].strip()=='// Namespace: UnityEngine'
        for line in ['public float x; // 0x0','public float y; // 0x4','public float z; // 0x8','private static readonly Vector3 zeroVector; // 0x0']:assert line in vector[1]
        chunks=[]
        for e in self.pe.DIRECTORY_ENTRY_EXCEPTION:
            root=e
            while root.unwindinfo.Flags&4:root=root.unwindinfo._chained_entry
            if root.struct.BeginAddress==0x367B60:chunks.append((e.struct.BeginAddress,e.struct.EndAddress))
        assert chunks==[(0x367B60,0x367C08),(0x367C08,0x367C67),(0x367C67,0x367DFC)]
        assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address']>0x367B60)==0x367E00
        section=self.pe.get_section_by_rva(0x367B60);assert section and section.VirtualAddress+section.SizeOfRawData>=0x367E00
        raw=self.pe.get_data(0x367B60,0x2A0);assert len(raw)==0x2A0 and raw[0x29C:]==b'\xcc'*4
        ins=list(self.cs.disasm(raw[:0x29C],0x367B60));assert sum(i.size for i in ins)==0x29C and len(ins)==156
        self.refresh_instructions={i.address:i for i in ins};assert (ins[-1].address,ins[-1].mnemonic)==(0x367DFB,'int3')
        self.refresh_call_sites={hex(i.address):i.op_str for i in ins if i.mnemonic=='call'}
        assert self.refresh_call_sites=={hex(a):hex(b) for a,b in [(0x367B79,0x2B7B40),(0x367B85,0x2B7B40),(0x367BB9,0x1C7D810),(0x367BE2,0x281D90),(0x367BEF,0x1C822C0),(0x367C0D,0x1C7A010),(0x367C25,0x281D90),(0x367C37,0x668010),(0x367C4D,0x2B6FF0),(0x367C69,0x1C7DDF0),(0x367C80,0x1C7A010),(0x367C99,0x1C91B80),(0x367CC3,0x1C923D0),(0x367CDA,0x1C7DDF0),(0x367CF2,0x2B7B40),(0x367D31,0x1C91EC0),(0x367D48,0x1C7D810),(0x367D64,0x281D90),(0x367D71,0x1C82480),(0x367DA4,0x1C7D810),(0x367DCD,0x281D90),(0x367DDA,0x1C822C0),(0x367DF6,0x2B7D90)]}
        self.instructions.update(self.refresh_instructions);self.traps.add(0x367DFB)
        rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==0x367B60];assert len(rows)==1 and rows[0]['Name']=='Character$$RefreshView' and rows[0]['TypeSignature']=='vii' and rows[0]['Signature']=='void Character__RefreshView (Character_o* __this, const MethodInfo* method);'
        self.targets.append(dict(rows[0],method_id='tdi5487.m0063'));self.supplied=[r for r in self.supplied if r['Name']!='Character$$RefreshView']
        self.bounds['0x367b60']={'end_exclusive':'0x367dfc','next_managed':'0x367e00','unwind_chunks':[[hex(a),hex(b)] for a,b in chunks],'terminal_trap':'0x367dfb','byte_length':0x29C,'decoded_instructions':156,'body_sha256':hashlib.sha256(raw[:0x29C]).hexdigest()}
        self.refresh_checks={0x367B91:('cmp','dword ptr [rbx + 0xdc], 0'),0x367BA2:('jg','0x367bbe'),0x367BBE:('cmp','dword ptr [rbx + 0xe4], 0x14'),0x367BFC:('mov','rsi, qword ptr [rbx + 0x80]'),0x367C25:('call','0x281d90'),0x367C37:('call','0x668010'),0x367C3F:('mov','qword ptr [rbx + 0x98], rax'),0x367C4D:('call','0x2b6ff0'),0x367C52:('mov','rcx, qword ptr [rbx + 0x98]'),0x367C99:('call','0x1c91b80'),0x367CA7:('movsd','xmm0, qword ptr [rax]'),0x367CB0:('mov','eax, dword ptr [rax + 8]'),0x367CC3:('call','0x1c923d0'),0x367CC8:('mov','rcx, qword ptr [rbx + 0x98]'),0x367CDA:('call','0x1c7ddf0'),0x367D05:('mov','rax, qword ptr [rcx + 0xb8]'),0x367D15:('movsd','xmm0, qword ptr [rax]'),0x367D1E:('mov','eax, dword ptr [rax + 8]'),0x367D31:('call','0x1c91ec0'),0x367D46:('mov','dl, 1'),0x367D48:('call','0x1c7d810'),0x367D7A:('cmp','byte ptr [rbx + 0xed], 0'),0x367DCD:('call','0x281d90'),0x367DDA:('call','0x1c822c0'),0x367DDF:('mov','rcx, qword ptr [rbx + 0x88]'),0x367DE6:('test','al, al'),0x367DF2:('mov','dl, 1'),0x367DF6:('call','0x2b7d90')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in self.refresh_checks.items())
        captures={0x367BD2:('mov','rsi, qword ptr [rbx + 0x98]'),0x367D54:('mov','rdi, qword ptr [rbx + 0x88]'),0x367DC0:('mov','rdi, qword ptr [rbx + 0x58]')}
        assert all((self.instructions[a].mnemonic,self.instructions[a].op_str)==v for a,v in captures.items());self.refresh_checks.update(captures)
        bindings={0x2718BF0:'UnityEngine.Object_TypeInfo',0x26EA2F8:'UnityEngine.Vector3_TypeInfo',0x26D7A30:'Method$UnityEngine.Object.Instantiate<GameObject>()'}
        rows=[r for section in ['ScriptMetadata','ScriptMetadataMethod'] for r in self.metadata[section] if r['Address'] in bindings];assert {r['Address']:r['Name'] for r in rows}==bindings
        self.instantiate_mi_metadata=next(r for r in rows if r['Address']==0x26D7A30);assert self.instantiate_mi_metadata['MethodAddress']==0
        import capstone
        refs={i.address+i.size+op.mem.disp for i in ins for op in i.operands if op.type==capstone.CS_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP};assert refs==set(bindings)|{0x288C189,0x288C0E7}
        self.refresh_slots={n:self.base+a for a,n in [(0x2718BF0,'object_class'),(0x26EA2F8,'vector_class'),(0x26D7A30,'instantiate_mi')]};self.refresh_flags={'RefreshView':0x288C189,'Vector3':0x288C0E7}
        for i,n in enumerate(['vector_class','vector_static','instantiate_mi','icon','parent_transform','icon_transform','created_transform0','created_transform1','alternate_transform','prefab','created_old','created_new','created_other','rip','disguise','disguise_other','pickable','returned_vector']):
            self.p[n]=self.arena+0xA00000+i*0x1000;self.ids[self.p[n]]=n;self.sizes[n]=0x100 if n=='vector_class' else 0x80
        services={0x1C7A010:'UnityEngine.Component$$get_transform',0x1C7DDF0:'UnityEngine.GameObject$$get_transform',0x1C91B80:'UnityEngine.Transform$$get_position',0x1C923D0:'UnityEngine.Transform$$set_position',0x1C91EC0:'UnityEngine.Transform$$set_eulerAngles',0x668010:'UnityEngine.Object$$Instantiate<object>'}
        self.refresh_services={}
        for a,n in services.items():
            rs=[r for r in self.metadata['ScriptMethod'] if r['Address']==a and r['Name']==n];assert len(rs)==1;self.refresh_services[a]=rs[0];self.supplied.append(rs[0])
        signatures={0x1C7A010:('iii','UnityEngine_Transform_o* UnityEngine_Component__get_transform (UnityEngine_Component_o* __this, const MethodInfo* method);'),0x1C7DDF0:('iii','UnityEngine_Transform_o* UnityEngine_GameObject__get_transform (UnityEngine_GameObject_o* __this, const MethodInfo* method);'),0x1C91B80:('iii','UnityEngine_Vector3_o UnityEngine_Transform__get_position (UnityEngine_Transform_o* __this, const MethodInfo* method);'),0x1C923D0:('viii','void UnityEngine_Transform__set_position (UnityEngine_Transform_o* __this, UnityEngine_Vector3_o value, const MethodInfo* method);'),0x1C91EC0:('viii','void UnityEngine_Transform__set_eulerAngles (UnityEngine_Transform_o* __this, UnityEngine_Vector3_o value, const MethodInfo* method);'),0x668010:('iiii','Il2CppObject* UnityEngine_Object__Instantiate_object_ (Il2CppObject* original, UnityEngine_Transform_o* parent, const MethodInfo_668010* method);')}
        for a,(abi,signature) in signatures.items():assert self.refresh_services[a]['TypeSignature']==abi and self.refresh_services[a]['Signature']==signature

    def prepare(self,options):
        self.refresh_ready=False;super().prepare(options)
        for n,off in self.refresh_fields.items():
            default='created_old' if n=='created' else n
            ref=options.get('refresh_fields',{}).get(n,default)
            if n=='created' and 'created' not in options.get('refresh_fields',{}):ref=None
            self.q(self.p['actor']+off,0 if ref is None else self.p[ref])
        for n,a in self.refresh_slots.items():self.q(a,self.p[n])
        for n,a in self.refresh_flags.items():self.u.mem_write(self.base+a,bytes([options.get('refresh_flags',{}).get(n,int(options.get('warm',False)))]))
        self.q(self.p['vector_class']+0xB8,0 if options.get('null_vector_static') else self.p['vector_static'])
        self.u.mem_write(self.p['vector_static'],struct.pack('<III',*options.get('zero_bits',[0,0,0])))
        self.u.mem_write(self.p['returned_vector'],struct.pack('<III',*options.get('alternate_position_bits',[0x80000000,0x7FC0ABCD,1])))
        self.u.mem_write(self.p['actor']+0xD8,bytes([options.get('revealed_bits',0xFE)]));self.u.mem_write(self.p['actor']+0xED,bytes([options.get('killed_bits',0)]))
        self.refresh_live={n:True for n in self.p};self.refresh_live['created_old']=False;self.refresh_live.update(options.get('refresh_liveness',{}))
        self.active.update({n:options.get('refresh_active',True) for n in ['prefab','created_old','created_new','created_other','rip','disguise','disguise_other','pickable']})
        self.transform_state={n:{'position_bits':[0xA5A5A5A5]*3,'euler_bits':[0xA5A5A5A5]*3} for n in ['icon','parent_transform','icon_transform','created_transform0','created_transform1','alternate_transform']}
        self.refresh_log=[];self.refresh_vectors=[];self.refresh_ready=True

    def snapshot(self):
        r=super().snapshot()
        if not getattr(self,'refresh_ready',False):return r
        r['refresh_join']={'fields':{n:self.oid(self.rq(self.p['actor']+off)) for n,off in self.refresh_fields.items()},'revealed_bits':self.u.mem_read(self.p['actor']+0xD8,1)[0],'killed_bits':self.u.mem_read(self.p['actor']+0xED,1)[0],'metadata_flags':{n:self.u.mem_read(self.base+a,1)[0] for n,a in self.refresh_flags.items()},'metadata_slots':{n:self.oid(self.rq(a)) for n,a in self.refresh_slots.items()},'liveness':self.refresh_live.copy(),'transforms':deepcopy(self.transform_state),'mutations':deepcopy(self.refresh_log),'vector_outputs':deepcopy(self.refresh_vectors),'static_pointer':self.oid(self.rq(self.p['vector_class']+0xB8))}
        return r

    def mutate(self,phase):
        a=self.options.get('refresh_mutations',{}).get(phase)
        if a is None:return super().mutate(phase)
        self.refresh_log.append({'phase':phase,'action':deepcopy(a)})
        for n,off,width,value in a:
            if n=='liveness':self.refresh_live[off]=value;continue
            v=self.p[value] if isinstance(value,str) else value
            self.u.mem_write(self.p[n]+off,int(v).to_bytes(width,'little'));self.allowed.setdefault(n,set()).update(range(off,off+width))

    def hook(self,uc,address,size,data):
        rva,x=address-self.base,self.x
        if address==self.stop:return
        if rva in self.refresh_instructions:
            assert rva!=0x367DFB;self.executed.add(rva)
            if rva==0x367B60:
                assert self.last_native==0x36958B;raw=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']];assert raw[0]==self.p['actor'] and raw[1]==0
                self.frames.append({'method':'RefreshView','raw_args':raw.copy(),'entry_sp':self.reg(x.UC_X86_REG_RSP),'return_target':self.rq(self.reg(x.UC_X86_REG_RSP))});self.saved_frames.append({n:self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RBX','RBP','RSI','RDI','R12','R13','R14','R15',*[f'XMM{i}' for i in range(6,16)]]});self.entries.append({'entry':'RefreshView','raw_args':raw.copy()})
            if rva==0x367BFC:self.frames[-1]['creation_checked']=self.oid(self.reg(x.UC_X86_REG_RSI))
            if rva==0x367C08:self.frames[-1]['captured_prefab']=self.oid(self.reg(x.UC_X86_REG_RSI))
            if rva==0x367C3F:self.allowed.setdefault('actor',set()).update(range(0x98,0xA0))
            if rva==0x367C75:self.frames[-1]['captured_transform']=self.oid(self.reg(x.UC_X86_REG_RSI))
            if rva==0x367C9E:assert bytes(uc.mem_read(self.stack+0x17FF0,12)).hex()==self.refresh_vectors[-1]['written_bytes12']
            if rva==0x367DB8:
                assert self.reg(x.UC_X86_REG_RSP)==self.frames[-1]['entry_sp'] and all(self.reg(getattr(x,'UC_X86_REG_'+n))==v for n,v in self.saved_frames[-1].items());self.frames.pop();self.saved_frames.pop();self.callee_abi.append('RefreshView');self.results.append({'method':'RefreshView','result_bits':None,'result_identity':None})
            self.last_native=rva;return
        if rva in self.instructions or self.last_native not in self.refresh_instructions:return super().hook(uc,address,size,data)
        self.executed.add(rva);cx,dx,r8,r9=[self.reg(getattr(x,'UC_X86_REG_'+n)) for n in ['RCX','RDX','R8','R9']];site=self.last_native;value=0;effect=None
        if rva==0x2B7D90:self.event('native_null_guard',[]);self.error='native_null_guard';uc.emu_stop();return
        if rva==0x2B7B40:
            n=next(n for n,a in self.refresh_slots.items() if a==cx);kind='metadata_service';args=[cx-self.base,n];phase='RefreshView:metadata:'+n;value=self.rq(cx)
        elif rva==0x281D90:
            assert cx==self.p['object_class'] and self.rd(cx+0xE0)==0;kind='class_initialization_service';args=['object_class'];phase='RefreshView:class_init:'+hex(site)
            effect=lambda:(self.d(cx+0xE0,1),self.allowed.setdefault('object_class',set()).update(range(0xE0,0xE4)))
        elif rva in [0x1C822C0,0x1C82480]:
            assert dx==r8==0;n=self.oid(cx);live=bool(n and self.refresh_live[n]);truth=not live if rva==0x1C822C0 else live;value=self.options.get('refresh_boolean_bits',0xFACE123456789000)|(self.options.get('refresh_true_byte',0xFE) if truth else 0);kind='refresh_unity_equality' if rva==0x1C822C0 else 'refresh_unity_inequality';args=[n,None,0,value];phase='RefreshView:liveness:'+hex(site)
        elif rva==0x1C7A010:
            assert dx==0;n=self.oid(cx);target=self.options.get('refresh_parent','parent_transform') if site==0x367C0D else self.options.get('refresh_icon_transform','icon_transform');value=0 if target is None else self.p[target];kind='refresh_component_transform';args=[n,0,target];phase='RefreshView:component:'+hex(site)
        elif rva==0x668010:
            assert r8==self.p['instantiate_mi'];target=self.options.get('clone_result','created_new');value=0 if target is None else self.p[target];kind='refresh_instantiate';args=[self.oid(cx),self.oid(dx),'instantiate_mi',target];phase='RefreshView:instantiate'
        elif rva==0x2B6FF0:
            assert cx==self.p['actor']+0x98 and self.rq(cx)==dx;kind='reference_barrier';args=['actor',0x98,self.oid(dx)];phase='RefreshView:barrier'
        elif rva==0x1C7DDF0:
            assert dx==0;n=self.oid(cx);target=self.options.get('refresh_transform'+('0' if site==0x367C69 else '1'),'created_transform0' if site==0x367C69 else 'created_transform1');value=0 if target is None else self.p[target];kind='refresh_gameobject_transform';args=[n,0,target];phase='RefreshView:created:'+hex(site)
        elif rva==0x1C91B80:
            assert r8==0 and cx==self.stack+0x17FF0;bits=self.options.get('position_bits',[0x3F800000,0xC0000000,0x40400000]);target=self.options.get('position_return','buffer');value=cx if target=='buffer' else 0 if target is None else self.p[target];kind='refresh_get_position';args=[cx,self.oid(dx),0,bits,value];phase='RefreshView:position_get'
            def effect():uc.mem_write(cx,struct.pack('<III',*bits));self.refresh_vectors.append({'buffer_address_bits':cx,'written_bytes12':struct.pack('<III',*bits).hex(),'returned_address_bits':value})
        elif rva in [0x1C923D0,0x1C91EC0]:
            assert r8==0 and dx==self.stack+0x17FE0;bits=list(struct.unpack('<III',uc.mem_read(dx,12)));n=self.oid(cx);kind='refresh_set_position' if rva==0x1C923D0 else 'refresh_set_euler';args=[n,bits,0];phase='RefreshView:position_set' if rva==0x1C923D0 else 'RefreshView:euler_set';off=0x20 if rva==0x1C923D0 else 0x30
            def effect():uc.mem_write(cx+off,struct.pack('<III',*bits));self.allowed.setdefault(n,set()).update(range(off,off+12));self.transform_state[n]['position_bits' if off==0x20 else 'euler_bits']=bits.copy()
        elif rva==0x1C7D810:
            assert r8==0 and dx&255 in [0,1];n=self.oid(cx);kind='set_active';args=[n,dx,0];phase='RefreshView:active:'+hex(site);effect=lambda:self.active.update({n:bool(dx&255)})
        else:raise AssertionError(('unclaimed refresh service',hex(rva),hex(site)))
        if self.event(kind,args):
            if effect:effect()
            self.requests.append({'kind':kind,'args':deepcopy(args)});self.mutate(phase);self.ret(value)

    def run(self,name,options=None,retained=False):
        prior=len(self.refresh_log) if retained else 0;row=super().run(name,options,retained)
        assert row['final']['refresh_join']['metadata_slots']==row['initial']['refresh_join']['metadata_slots']
        if not row['options'].get('failure'):
            reached={r['phase'] for r in row['final']['refresh_join']['mutations'][prior:]};assert set(row['options'].get('refresh_mutations',{}))<=reached,('unreached refresh mutation',row['options'],reached)
        return row

Machine=RefreshMachine

def verify_semantics(row,m):
    s=deepcopy(row['initial']);mem={n:bytearray.fromhex(raw) for n,raw in s['memory'].items()};events=[];o=row['options'];error=None;fault=None;regs=row['entry_raw_args'].copy();p=m.p;sp=m.stack+0x18008;input_name=None;captured_acted=None
    def q(n,off):return struct.unpack_from('<Q',mem[n],off)[0]
    def field(n):return m.oid(q('actor',FIELDS[n]))
    def put(off,width,value):struct.pack_into('<Q' if width==8 else '<I',mem['actor'],off,value)
    def snapshot():
        r=deepcopy(s);r['memory']={n:raw.hex() for n,raw in mem.items()};r['fields']={n:field(n) for n in FIELDS};r['left_act_bits']=mem['actor'][0xB0]
        r['data']={n:{'name':m.oid(q(n,0x28)),'background':m.oid(q(n,0xB8)),'color_bits':list(struct.unpack_from('<IIII',mem[n],0xD8))} for n in ['data0','data1','data2']}
        r['reward_join']['pointers']={n:m.oid(q('actor',off)) for n,off in POINTERS.items()};r['reward_join']['words']={n:struct.unpack_from('<I',mem['actor'],off)[0] for n,off in WORDS.items()};r['reward_join']['starting_alignment_bits']={n:struct.unpack_from('<I',mem[n],0x134)[0] for n in ['data0','data1','data2']}
        r['art_join']['fields']={n:m.oid(q('actor',off)) for n,off in [('art',0x28),('clipping',0x30)]};r['art_join']['data']={n:{'skin':m.oid(q(n,0xC0)),'default':m.oid(q(n,0x98))} for n in ['data0','data1','data2']};r['art_join']['skins']={n:{'sprite':m.oid(q(n,0x38)),'type_bits':struct.unpack_from('<I',mem[n],0x50)[0]} for n in ['skin0','skin1','skin2']}
        r['color_join']['fields']={n:m.oid(q('actor',off)) for n,off in [('background',0x120),('borders',0x128)]};r['color_join']['data_colors']={n:{k:list(struct.unpack_from('<IIII',mem[n],off)) for k,off in [('background',0xF8),('border',0x108)]} for n in ['data0','data1','data2']};r['color_join']['arrays']={n:{'length_bits':q(n,0x18),'slots':[m.oid(q(n,0x20+8*i)) for i in range(3)]} for n in ['borders0','borders1']}
        r['refresh_join']['fields']={n:m.oid(q('actor',off)) for n,off in m.refresh_fields.items()};r['refresh_join']['revealed_bits']=mem['actor'][0xD8];r['refresh_join']['killed_bits']=mem['actor'][0xED];r['refresh_join']['static_pointer']=m.oid(q('vector_class',0xB8));return r
    def mutate(phase):
        a=o.get('refresh_mutations',{}).get(phase)
        if a is not None:
            s['refresh_join']['mutations'].append({'phase':phase,'action':deepcopy(a)})
            for n,off,width,value in a:
                if n=='liveness':s['refresh_join']['liveness'][off]=value;continue
                v=p[value] if isinstance(value,str) else value;mem[n][off:off+width]=int(v).to_bytes(width,'little')
            return
        a=o.get('color_mutations',{}).get(phase)
        if a is not None:
            s['color_join']['mutations'].append({'phase':phase,'action':a})
            def write(n,off,v,width=8):
                if width==16:struct.pack_into('<IIII',mem[n],off,*v)
                else:struct.pack_into('<Q' if width==8 else '<I',mem[n],off,v)
            if a=='replace_data':put(0x50,8,p['data2'])
            elif a=='clear_data':put(0x50,8,0)
            elif a in ['replace_borders','clear_borders']:put(0x128,8,p['borders1'] if a=='replace_borders' else 0)
            elif a in ['replace_background','clear_background']:put(0x120,8,p['colorbg1'] if a=='replace_background' else 0)
            elif a in ['shrink_array','grow_array','negative_array']:write('borders0',0x18,{'shrink_array':0,'grow_array':3,'negative_array':0xFFFFFFFF}[a],4)
            elif a in ['clear_next_border','replace_next_border']:write('borders0',0x28,p['colorbg1'] if a=='replace_next_border' else 0)
            elif a=='replace_image_class':write(o.get('class_component','border1'),0,p['image_class1'])
            elif a=='replace_border_color':write(o.get('color_data','data1'),0x108,[0x80000000,0x7FC0ABCD,0x00000001,0xFFFFFFFF],16)
            else:raise AssertionError(a)
            return
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
    def update_colors():
        nonlocal regs
        raw=[p['actor'],0,regs[2],regs[3]];regs=raw.copy();entry('UpdateViewReal',raw);s['art_join']['frames'].append({'method':'UpdateViewReal','raw_args':raw.copy(),'entry_sp':sp,'return_target':m.stop})
        data=field('data')
        if data is None:guard([p['actor'],0,regs[2],regs[3]],0x369590);return
        s['art_join']['frames'][-1]['background_data']=data;bg=m.oid(q('actor',0x120))
        if bg is None:guard([0,p[data],regs[2],regs[3]],0x369590);return
        bits=list(struct.unpack_from('<IIII',mem[data],0xF8));cls=m.oid(q(bg,0));mi=m.oid(q(cls,0x2B0))
        if not service('image_color_service',[bg,bits,mi],0x36951A,[p[bg],sp-0x18,p[mi],regs[3]],'UpdateViewReal:background',lambda:s['components'][bg].update(color_bits=bits.copy())):return
        arr=m.oid(q('actor',0x128));s['art_join']['frames'][-1]['captured_array']=arr
        if arr is None:guard(regs,0x369590);return
        index=0
        while index<((q(arr,0x18)&0x7FFFFFFF)-(q(arr,0x18)&0x80000000)):
            assert index<3
            data=field('data')
            if data is None:guard([regs[0],0,regs[2],regs[3]],0x369590);return
            s['art_join']['frames'][-1]['border_data']=data;image=m.oid(q(arr,0x20+8*index))
            if image is None:guard([0,p[data],regs[2],regs[3]],0x369590);return
            bits=list(struct.unpack_from('<IIII',mem[data],0x108));cls=m.oid(q(image,0));mi=m.oid(q(cls,0x2B0))
            if not service('image_color_service',[image,bits,mi],0x36956B,[p[image],sp-0x18,p[mi],regs[3]],'UpdateViewReal:border:'+str(index+1),lambda:s['components'][image].update(color_bits=bits.copy())):return
            index+=1
        s['art_join']['frames'].pop();s['art_join']['normal_callee_abi'].append('UpdateViewReal');s['art_join']['results'].append({'method':'UpdateViewReal','result_bits':None,'result_identity':None})
        refresh()
    def refresh():
        nonlocal regs,error,fault
        raw=[p['actor'],0,regs[2],regs[3]];regs=raw.copy();entry('RefreshView',raw);s['art_join']['frames'].append({'method':'RefreshView','raw_args':raw.copy(),'entry_sp':sp,'return_target':m.stop})
        def reference(off):return m.oid(q('actor',off))
        def class_init(site):
            if not struct.unpack_from('<I',mem['object_class'],0xE0)[0]:
                return service('class_initialization_service',['object_class'],site,[p['object_class'],regs[1],regs[2],regs[3]],'RefreshView:class_init:'+hex(site),lambda:struct.pack_into('<I',mem['object_class'],0xE0,1))
            return True
        def unity(n,eq,site):
            live=bool(n and s['refresh_join']['liveness'][n]);truth=not live if eq else live;bits=o.get('refresh_boolean_bits',0xFACE123456789000)|(o.get('refresh_true_byte',0xFE) if truth else 0)
            ok=service('refresh_unity_equality' if eq else 'refresh_unity_inequality',[n,None,0,bits],site,[0 if n is None else p[n],0,0,regs[3]],'RefreshView:liveness:'+hex(site));return bits if ok else None
        def active(n,value,site):return service('set_active',[n,value,0],site,[p[n],value,0,regs[3]],'RefreshView:active:'+hex(site),lambda:s['reward_join']['active'].update({n:bool(value&255)}))
        if not s['refresh_join']['metadata_flags']['RefreshView']:
            for n,site in [('instantiate_mi',0x367B79),('object_class',0x367B85)]:
                if not service('metadata_service',[m.refresh_slots[n]-m.base,n],site,[m.refresh_slots[n],regs[1],regs[2],regs[3]],'RefreshView:metadata:'+n):return
            s['refresh_join']['metadata_flags']['RefreshView']=1
        uses=struct.unpack_from('<i',mem['actor'],0xDC)[0]
        if uses<=0:
            n=reference(0x1A8)
            if n is None:guard([0,regs[1],regs[2],regs[3]],0x367DF6);return
            if not active(n,0,0x367BB9):return
        if struct.unpack_from('<I',mem['actor'],0xE4)[0]==20:
            created=reference(0x98)
            if not class_init(0x367BE2):return
            bits=unity(created,True,0x367BEF)
            if bits is None:return
            if bits&255:
                s['art_join']['frames'][-1]['creation_checked']=created
                prefab=reference(0x80);s['art_join']['frames'][-1]['captured_prefab']=prefab;parent=o.get('refresh_parent','parent_transform')
                if not service('refresh_component_transform',['actor',0,parent],0x367C0D,[p['actor'],0,regs[2],regs[3]],'RefreshView:component:0x367c0d'):return
                if not class_init(0x367C25):return
                clone=o.get('clone_result','created_new')
                if not service('refresh_instantiate',[prefab,parent,'instantiate_mi',clone],0x367C37,[0 if prefab is None else p[prefab],0 if parent is None else p[parent],p['instantiate_mi'],regs[3]],'RefreshView:instantiate'):return
                put(0x98,8,0 if clone is None else p[clone])
                if not service('reference_barrier',['actor',0x98,clone],0x367C4D,[p['actor']+0x98,0 if clone is None else p[clone],regs[2],regs[3]],'RefreshView:barrier'):return
                created=reference(0x98)
                if created is None:guard([0,regs[1],regs[2],regs[3]],0x367DF6);return
                first=o.get('refresh_transform0','created_transform0')
                if not service('refresh_gameobject_transform',[created,0,first],0x367C69,[p[created],0,regs[2],regs[3]],'RefreshView:created:0x367c69'):return
                s['art_join']['frames'][-1]['captured_transform']=first;icon=reference(0x20)
                if icon is None:guard([0,regs[1],regs[2],regs[3]],0x367DF6);return
                it=o.get('refresh_icon_transform','icon_transform')
                if not service('refresh_component_transform',[icon,0,it],0x367C80,[p[icon],0,regs[2],regs[3]],'RefreshView:component:0x367c80'):return
                if it is None:guard(regs,0x367DF6);return
                bits=o.get('position_bits',[0x3F800000,0xC0000000,0x40400000]);target=o.get('position_return','buffer');value=sp-0x18 if target=='buffer' else 0 if target is None else p[target]
                def getter_effect():s['refresh_join']['vector_outputs'].append({'buffer_address_bits':sp-0x18,'written_bytes12':struct.pack('<III',*bits).hex(),'returned_address_bits':value})
                if not service('refresh_get_position',[sp-0x18,it,0,bits,value],0x367C99,[sp-0x18,p[it],0,regs[3]],'RefreshView:position_get',getter_effect):return
                if first is None:guard(regs,0x367DF6);return
                if value==0:error='native_vector_read_fault';fault='0x367ca7';return
                pos=bits.copy() if target=='buffer' else list(struct.unpack_from('<III',mem[target],0))
                def transform_effect(n,off,values,key):mem[n][off:off+12]=struct.pack('<III',*values);s['refresh_join']['transforms'][n][key]=values.copy()
                if not service('refresh_set_position',[first,pos,0],0x367CC3,[p[first],sp-0x28,0,regs[3]],'RefreshView:position_set',lambda:transform_effect(first,0x20,pos,'position_bits')):return
                created=reference(0x98)
                if created is None:guard([0,regs[1],regs[2],regs[3]],0x367DF6);return
                second=o.get('refresh_transform1','created_transform1')
                if not service('refresh_gameobject_transform',[created,0,second],0x367CDA,[p[created],0,regs[2],regs[3]],'RefreshView:created:0x367cda'):return
                if not s['refresh_join']['metadata_flags']['Vector3']:
                    if not service('metadata_service',[m.refresh_slots['vector_class']-m.base,'vector_class'],0x367CF2,[m.refresh_slots['vector_class'],regs[1],regs[2],regs[3]],'RefreshView:metadata:vector_class'):return
                    s['refresh_join']['metadata_flags']['Vector3']=1
                if second is None:guard([p['vector_class'],regs[1],regs[2],regs[3]],0x367DF6);return
                static=m.oid(q('vector_class',0xB8))
                if static is None:error='native_vector_read_fault';fault='0x367d15';return
                zero=list(struct.unpack_from('<III',mem[static],0))
                if not service('refresh_set_euler',[second,zero,0],0x367D31,[p[second],sp-0x28,0,regs[3]],'RefreshView:euler_set',lambda:transform_effect(second,0x30,zero,'euler_bits')):return
                rip=reference(0x78)
                if rip is None:guard([0,regs[1],regs[2],regs[3]],0x367DF6);return
                if not active(rip,(regs[1]&~255)|1,0x367D48):return
        disguise=reference(0x88)
        if not class_init(0x367D64):return
        bits=unity(disguise,False,0x367D71)
        if bits is None:return
        if bits&255 and not mem['actor'][0xED]:
            state=struct.unpack_from('<I',mem['actor'],0xE4)[0]
            if state in [20,30]:
                bluff=reference(0x58)
                if not class_init(0x367DCD):return
                bits=unity(bluff,True,0x367DDA)
                if bits is None:return
                disguise=reference(0x88)
                if disguise is None:guard([0,regs[1],regs[2],regs[3]],0x367DF6);return
                if not active(disguise,0 if bits&255 else (regs[1]&~255)|1,0x367DA4):return
            else:
                disguise=reference(0x88)
                if disguise is None:guard([0,regs[1],regs[2],regs[3]],0x367DF6);return
                if not active(disguise,0,0x367DA4):return
        s['art_join']['frames'].pop();s['art_join']['normal_callee_abi'].append('RefreshView');s['art_join']['results'].append({'method':'RefreshView','result_bits':None,'result_identity':None})
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
        update_colors()
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
    verify_history_codec();verify_memory_map_codec();verify_state_map_codec();verify_full_report_codec()
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
    for count,alias in itertools.product([0,1,3,0xFFFFFFFF],[None,'border0','tmp0','art0']):
        options={'border_length_bits':0xABCD123400000000|count}
        if alias:options['border_alias']=alias
        cases.append(m.run('InitReward',options))
    for options in [{'null_color_bg':True},{'null_borders':True},{'null_border_index':0},{'null_border_index':1},{'null_border_index':2},
                    {'color_bg':'tmp0','border_alias':'tmp0'},{'color_bg':'art0','border_alias':'art0'},
                    {'mutation_phase':'background_set','mutation':'clear_data'},
                    {'border_length_bits':0xABCD000000000001,'color_mutations':{'UpdateViewReal:border:1':'grow_array'}},
                    {'border_alias':'border0','class_component':'border0','color_mutations':{'UpdateViewReal:border:1':'replace_image_class'}},
                    {'color_mutations':{'UpdateViewReal:background':'replace_borders','UpdateViewReal:border:1':'replace_data'}},
                    {'color_mutations':{'UpdateViewReal:border:1':'replace_borders','UpdateViewReal:border:2':'replace_data'}},
                    {'color_mutations':{'UpdateViewReal:background':'replace_data','UpdateViewReal:border:1':'replace_border_color'},'color_data':'data2'}]:cases.append(m.run('InitReward',options))
    for phase,action in itertools.product(['UpdateViewReal:background','UpdateViewReal:border:1','UpdateViewReal:border:2','UpdateViewReal:border:3'],
                                         ['replace_data','clear_data','replace_borders','clear_borders','replace_background','clear_background','shrink_array','grow_array','negative_array','clear_next_border','replace_next_border','replace_image_class','replace_border_color']):
        cases.append(m.run('InitReward',{'color_mutations':{phase:action}}))
    for shape,first,second in [({'border_alias':'tmp0'},'replace_data','replace_image_class'),
                               ({'border_alias':'border0'},'replace_borders','replace_data'),
                               ({'color_bg':'art0'},'replace_background','replace_border_color')]:
        seq.append([m.run('SetupObject',dict(shape,side_bits=40)),m.run('InitReward',{'color_mutations':{'UpdateViewReal:border:1':first}},True),
                    m.run('SetupObject',{'side_bits':10},True),m.run('InitReward',{'input_data':'data0','color_mutations':{'UpdateViewReal:background':second}},True),m.run('RevealReal',{},True)])
    profiles=[('SetupObject',{'side_bits':40}),('InitReward',{}),('InitReward',{'metadata_warm':True,'class_warm':False}),('InitReward',{'init_mutations':{'get_gameobject':'clear_acteds'}}),('InitReward',{'init_mutations':{'init_barrier:3':'replace_state_action'}}),('InitReward',{'init_mutations':{'callback':'replace_data'}}),('InitReward',{'init_mutations':{'callback':'replace_name'}}),('InitReward',{'art_mutations':{'GetArt:equality':'replace_data'}}),('InitReward',{'init_mutations':{'init_barrier:3':'replace_name','callback':'replace_data'}}),
              ('InitReward',{'skin1':None}),('InitReward',{'skin_type_bits':0}),('InitReward',{'alias_images':True}),('InitReward',{'art_mutations':{'GetArt:equality':'class_cold','GetArtType:equality':'class_cold','SetupArt:equality':'class_cold'}}),('InitReward',{'art_mutations':{'SetupArt:active:true':'replace_clipping'}})]
    profiles += [('InitReward',{'border_length_bits':0xABCD123400000000}),('InitReward',{'border_alias':'tmp0'}),
                 ('InitReward',{'color_mutations':{'UpdateViewReal:background':'replace_borders'}}),
                 ('InitReward',{'color_mutations':{'UpdateViewReal:border:1':'replace_data','UpdateViewReal:border:2':'replace_borders'}}),
                 ('InitReward',{'border_alias':'border0','class_component':'border0','color_mutations':{'UpdateViewReal:border:1':'replace_image_class'}})]
    for name,options in profiles:
        baseline=m.run(name,options);assert baseline['returned'];bid=len(bases);bases.append(baseline);counts={}
        for i,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;row=m.run(name,dict(options,failure=[kind,counts[kind]]));assert not row['returned'] and row['events']==baseline['events'][:i+1] and row['final']==e['snapshot'];stops.append({'baseline':bid,'prefix_length':i+1,'result':row})
    for state,uses,created,disguise,bluff,killed in itertools.product([5,20,30,0xFFFFFFFF],[0,1,0xFFFFFFFF],['created_old',None],[True,False],[True,False],[0,0xFE]):
        cases.append(m.run('RevealReal',{'state':state,'uses':uses,'refresh_fields':{'created':created},'refresh_liveness':{'created_old':True,'disguise':disguise,'data2':bluff},'killed_bits':killed}))
    dead={'state':20,'uses':0}
    extra=[{}, {'refresh_parent':None},{'refresh_fields':{'prefab':None}},{'clone_result':None},{'refresh_fields':{'icon':None}},{'refresh_icon_transform':None},{'refresh_transform0':None},{'refresh_transform1':None},{'refresh_fields':{'rip':None}},{'refresh_fields':{'pickable':None}},{'position_return':None},{'position_return':'returned_vector'},{'null_vector_static':True},{'refresh_fields':{'disguise':None}},{'refresh_fields':{'created':'created_old'},'refresh_liveness':{'created_old':True}}, {'refresh_fields':{'pickable':'game0','rip':'game0','disguise':'game0'}},{'refresh_fields':{'icon':'actor'},'refresh_transform1':'created_transform0'},{'clone_result':'prefab'},{'refresh_flags':{'RefreshView':0xFE,'Vector3':0xFE}}, {'refresh_fields':{'disguise':'created_new'}}, {'refresh_true_byte':1,'refresh_boolean_bits':0x1234567800000000}]
    for opts in extra:cases.append(m.run('RevealReal',dict(dead,**opts)))
    for bits in [[0,0x80000000,1],[0x7F800000,0xFF800000,0x7FC01234],[0xFFFFFFFF,0x00800000,0x007FFFFF]]:
        cases.append(m.run('RevealReal',dict(dead,position_bits=bits,zero_bits=list(reversed(bits)))))
    for state,revealed in itertools.product([5,20,30],[0,1,0x80,0xFF]):cases.append(m.run('RevealReal',dict(dead,state=state,revealed_bits=revealed)))
    for state,uses in itertools.product([5,20],[0x7FFFFFFF,0x80000000]):cases.append(m.run('RevealReal',dict(dead,state=state,uses=uses)))
    for opts in [{'state':5,'uses':1,'refresh_fields':{'pickable':None}},
                 {'state':20,'uses':1,'refresh_fields':{'created':'created_old','icon':None,'rip':None,'prefab':None,'pickable':None,'disguise':None},'refresh_liveness':{'created_old':True}},
                 {'state':5,'uses':1,'refresh_fields':{'icon':None,'rip':None,'prefab':None}},
                 dict(dead,refresh_mutations={'RefreshView:liveness:0x367d71':[['actor',0xED,1,0xFE],['actor',0x88,8,0]]}),
                 dict(dead,refresh_liveness={'disguise':False},refresh_mutations={'RefreshView:liveness:0x367d71':[['actor',0x88,8,0]]}),
                 dict(dead,refresh_fields={'created':'created_old'},refresh_liveness={'created_old':True},refresh_mutations={'RefreshView:liveness:0x367bef':[['actor',0x98,8,0]]}),
                 dict(dead,refresh_mutations={'RefreshView:liveness:0x367bef':[['actor',0xE4,4,5]]})]:cases.append(m.run('RevealReal',opts))
    plans=[('RefreshView:metadata:instantiate_mi',[['actor',0x80,8,'created_other']]),('RefreshView:component:0x367c0d',[['actor',0x80,8,'created_other']]),('RefreshView:component:0x367c0d',[['object_class',0xE0,4,0]]),('RefreshView:barrier',[['actor',0x98,8,'created_other']]),('RefreshView:barrier',[['actor',0x98,8,0]]),('RefreshView:created:0x367c69',[['actor',0x20,8,0]]),('RefreshView:position_get',[['actor',0x98,8,0]]),('RefreshView:position_get',[['actor',0x98,8,'created_other']]),('RefreshView:position_set',[['actor',0x98,8,'created_other']]),('RefreshView:metadata:vector_class',[['vector_static',0,4,0x80000000],['vector_static',4,4,0x7FC0ABCD],['vector_static',8,4,1]]),('RefreshView:euler_set',[['actor',0x78,8,0]]),('RefreshView:liveness:0x367d71',[['object_class',0xE0,4,0]]),('RefreshView:liveness:0x367d71',[['actor',0xED,1,0xFE]]),('RefreshView:liveness:0x367d71',[['actor',0xE4,4,5]]),('RefreshView:liveness:0x367d71',[['actor',0x88,8,'disguise_other']]),('RefreshView:liveness:0x367dda',[['actor',0x88,8,'disguise_other']]),('RefreshView:liveness:0x367dda',[['actor',0x88,8,0]]),('RefreshView:liveness:0x367dda',[['actor',0x58,8,0]]),('RefreshView:liveness:0x367bef',[['object_class',0xE0,4,0]]),('RefreshView:instantiate',[['actor',0x98,8,'created_other']]),('RefreshView:active:0x367bb9',[['actor',0xDC,4,9]]),('RefreshView:active:0x367d48',[['actor',0xE4,4,30]]),('RefreshView:created:0x367cda',[['vector_class',0xB8,8,0]])]
    for phase,changes in plans:cases.append(m.run('RevealReal',dict(dead,refresh_mutations={phase:changes})))
    for state in [5,20]:cases.append(m.run('RevealReal',dict(dead,state=state,refresh_mutations={'UpdateViewReal:border:3':[['object_class',0xE0,4,0]]})))
    class_capture_profiles=[dict(dead,refresh_mutations={'UpdateViewReal:border:3':[['object_class',0xE0,4,0]],'RefreshView:class_init:0x367be2':[['actor',0x98,8,'created_old']]}),
        dict(dead,refresh_mutations={'RefreshView:component:0x367c0d':[['object_class',0xE0,4,0]],'RefreshView:class_init:0x367c25':[['actor',0x80,8,0]]}),
        dict(dead,state=5,refresh_liveness={'disguise':False},refresh_mutations={'UpdateViewReal:border:3':[['object_class',0xE0,4,0]],'RefreshView:class_init:0x367d64':[['actor',0x88,8,'disguise_other']]}),
        dict(dead,refresh_mutations={'RefreshView:liveness:0x367d71':[['object_class',0xE0,4,0]],'RefreshView:class_init:0x367dcd':[['actor',0x58,8,0]]})]
    for opts in class_capture_profiles:cases.append(m.run('RevealReal',opts))
    cases.append(m.run('RevealReal',dict(dead,position_return='returned_vector',refresh_mutations={'RefreshView:position_get':[['returned_vector',0,4,0x7FC00011]]})))
    for opts in [dict(dead,refresh_mutations={'RefreshView:component:0x367c0d':[['object_class',0xE0,4,0]],'RefreshView:liveness:0x367d71':[['object_class',0xE0,4,0]]}),dict(dead,refresh_fields={'pickable':'game0','rip':'game0','disguise':'game0'}),dict(dead,position_return='returned_vector'),dict(dead,refresh_transform1='created_transform0')]:
        setup={k:v for k,v in opts.items() if k!='refresh_mutations'}
        init=dict(opts,refresh_mutations=dict(opts.get('refresh_mutations',{}),**{'UpdateViewReal:border:3':[['actor',0xE4,4,20],['actor',0x58,8,'data2']]}))
        seq.append([m.run('SetupObject',dict(setup,side_bits=40)),m.run('InitReward',init,True),m.run('RevealReal',{},True),m.run('RevealReal',{'refresh_mutations':{'RefreshView:liveness:0x367d71':[['actor',0xE4,4,30]]}},True)])
    profiles=[('RevealReal',dict(dead,**opts)) for opts in [{},{'position_return':'returned_vector'},{'refresh_fields':{'pickable':'game0','rip':'game0','disguise':'game0'}},{'refresh_mutations':{'RefreshView:component:0x367c0d':[['object_class',0xE0,4,0]],'RefreshView:liveness:0x367d71':[['object_class',0xE0,4,0]]}},{'refresh_mutations':{'RefreshView:barrier':[['actor',0x98,8,'created_other']]}},{'refresh_flags':{'RefreshView':0xFE,'Vector3':0xFE}},{'refresh_fields':{'created':'created_old'},'refresh_liveness':{'created_old':True}},{'position_bits':[0x80000000,0x7FC0ABCD,1],'zero_bits':[0x7F800000,0xFF800000,0xFFFFFFFF]}]]
    profiles += [('RevealReal',dict(dead,state=state,refresh_mutations={'UpdateViewReal:border:3':[['object_class',0xE0,4,0]]})) for state in [5,20]]
    profiles += [('RevealReal',opts) for opts in class_capture_profiles]
    for name,opts in profiles:
        baseline=m.run(name,opts);assert baseline['returned'];bid=len(bases);bases.append(baseline);counts={}
        for i,e in enumerate(baseline['events']):
            kind=e['kind'];counts[kind]=counts.get(kind,0)+1;row=m.run(name,dict(opts,failure=[kind,counts[kind]]));assert not row['returned'] and row['events']==baseline['events'][:i+1] and row['final']==e['snapshot'];stops.append({'baseline':bid,'prefix_length':i+1,'result':row})
    excluded=m.traps|{m.unreached_gateway};missing=set(m.instructions)-m.executed-excluded;assert not missing,[hex(a) for a in sorted(missing)];assert not m.executed&excluded
    for sequence in seq:
        for first,second in zip(sequence,sequence[1:]):assert first['final']==second['initial']
    return {'build':BUILD,'schema':'character_reward_refresh_join_native_v1','targets':m.targets,'bounds':m.bounds,'supplied_declarations':m.supplied,
        'field_offsets':{'character_pointers':dict(FIELDS,**POINTERS,art=0x28,clipping=0x30,color_background=0x120,borders=0x128),'character_words':WORDS,'left_act_byte':0xB0,'data':{'name':0x28,'default_art':0x98,'skin':0xC0,'background':0xB8,'color_bytes16':0xD8,'card_bg_color_bytes16':0xF8,'card_border_color_bytes16':0x108,'starting_alignment_word':0x134},'skin':{'art':0x38,'type_word':0x50},'delegate':{'invoke_impl':0x18,'unused_managed_target':0x20,'method':0x28,'method_code':0x40}},
        'metadata_flag_rva':hex(m.flag),'metadata_slot_rvas':[hex(m.slot-m.base),hex(m.literal_slot-m.base)],'color_body_identity':m.color_body_identity,
        'virtual_slots':{'text':{'slot':66,'function_offset':0x558,'method_offset':0x560},'color':{'slot':23,'function_offset':0x2A8,'method_offset':0x2B0}},
        'authored_records':{n:{'address_bits':p,'retained_bytes':m.sizes[n]} for n,p in m.p.items()},
        'init_operand_assertions':{hex(a):list(v) for a,v in m.init_checks.items()},'presentation_operand_assertions':{hex(a):list(v) for a,v in m.checks.items()},'added_operand_assertions':{hex(a):list(v) for a,v in m.added_checks.items()},'extra_metadata_flag_rvas':{n:hex(f) for n,f in m.extra_flags.items()},'color_operand_assertions':{hex(a):list(v) for a,v in m.color_checks.items()},'array_layout':{'signed_length_dword':0x18,'retained_length_bits':64,'elements':0x20,'retained_slots':3},'unexecuted_bounds_gateway':{'rva':hex(m.unreached_gateway),'opcode':'call 0x2b7d80','reason':'No mutation between adjacent native loop/index checks; concurrent size change and exception unwind excluded'},'terminal_traps_excluded':[hex(a) for a in sorted(m.traps)],'decoded_instructions':len(m.instructions),'executed_instructions':len(set(m.instructions)&m.executed),'observed_addresses':len(m.executed),'cases':cases,'sequences':seq,'baselines':bases,'stops':stops,'summary':{'cases':len(cases),'normal_returns':sum(r['returned'] for r in cases),'native_stops':sum(not r['returned'] for r in cases),'sequences':len(seq),'baselines':len(bases),'stops':len(stops)},'refresh_operand_assertions':{hex(a):list(v) for a,v in m.refresh_checks.items()},'refresh_metadata_slots':{n:hex(a-m.base) for n,a in m.refresh_slots.items()},'refresh_metadata_flags':{n:hex(a) for n,a in m.refresh_flags.items()},'refresh_fields':m.refresh_fields,'refresh_supplied_declarations':m.refresh_services,'refresh_call_sites':m.refresh_call_sites,'instantiate_boundary':{'methodinfo_metadata':m.instantiate_mi_metadata,'selected_supplied_declaration':m.refresh_services[0x668010],'helper_body_actual':False,'folded_aliases_promoted':False},'vector_abi':{'copied_bytes':12,'copy_widths':[8,4],'hidden_getter_buffer_relative_to_entry_sp':-24,'setter_buffer_relative_to_entry_sp':-40,'getter_buffer_padding_inferred':False,'transform_authored_diagnostic_offsets':{'position_bytes12':32,'euler_bytes12':48}},'scope':'Actual eight-body reward art/UpdateViewReal/RefreshView graph; no CharacterView edge. GetAnimatedArt excluded. GC/Unity/Action, Image/TMP, uppercase/runtime services supplied; no renderer/scheduler/runtime admission/unwinding/concurrent bounds race. Full snapshots and independently modeled service-entry ABI/caller/state/native frames.'}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('game_root');p.add_argument('dumper_root');p.add_argument('--output',required=True);a=p.parse_args()
    r=audit(a.game_root,a.dumper_root);print(json.dumps({'native_verified':r['summary']}),flush=True)
    encoded=pool_snapshots(pool_state_maps(pool_memory_maps(pool_histories(pool_memory(r)))))
    assert expand_report(encoded)==r
    output=json.dumps(encoded,sort_keys=True,separators=(',',':'))+'\n'
    assert len(output.encode('utf-8'))<100*1024*1024,'Complete report requires additional lossless interning or complete profile splitting'
    Path(a.output).write_text(output,encoding='utf-8');print(json.dumps(r['summary']),flush=True)
