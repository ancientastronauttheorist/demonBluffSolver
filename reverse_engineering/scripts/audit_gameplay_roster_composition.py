"""Pinned Gameplay roster composition with native loops and explicit collection services."""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path
from audit_character_assets import BUILD


def audit(game_root,dumper_root):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    assert unicorn.__version__=='2.1.4'
    root=Path(__file__).parents[1]
    lock=json.loads((root/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extract=json.loads((root/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(p,h):
        raw=p.read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==h.upper();return raw
    raw=pinned(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',extract['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump=pinned(Path(dumper_root)/'dump.cs',extract['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    assert 'public List<CharacterData> canAppearIf; // 0x128' in dump and 'public EAlignment startingAlignment; // 0x134' in dump
    entries={'GetAllCurrentCharacters':0x37C320,'CleanupCharactersList':0x37B750,'FilterIfCanAppearCharacters':0x37BE30,'GetNotInPlayCharacters':0x37CC00,'GetNotInDeckCharacters':0x37C8E0,'GetScriptCharactersOfAlignment':0x37D910,'UpdateCurrentCharacters':0x381260}
    exact=[]
    for n,a in entries.items():
        r=[r for r in script['ScriptMethod'] if r['Name']=='Gameplay$$'+n and r['Address']==a];assert len(r)==1;exact.append(r[0])
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    spans=[(0x37B750,0x37BDDF),(0x37BE30,0x37C19C),(0x37C320,0x37C3E3),(0x37C8E0,0x37CBF7),(0x37CC00,0x37CE04),(0x37D910,0x37DB0F),(0x381260,0x3814E2),
           (0x2EB0,0x2F0E),(0xB16640,0xB16675),(0x9674A0,0x9674E0),(0x9693D0,0x969470),(0x33ED50,0x33ED53)]
    ins={}
    for a,b in spans:
        rows=list(cs.disasm(pe.get_data(a,b-a),a));assert rows[-1].address+rows[-1].size==b;ins.update({i.address:i for i in rows})
    checks=[(0x37B80C,'call','0x37c320'),(0x37B8AE,'call','0x37be30'),(0x37BA10,'call','0x37be30'),(0x37BB79,'call','0x37be30'),(0x37BCDB,'call','0x37be30'),
            (0x9693E8,'cmp','dword ptr [rbx + 0xc], eax'),(0x969444,'call','0x113b490'),(0x2EB4,'inc','dword ptr [rcx + 0x1c]')]
    for a,m,o in checks:assert (ins[a].mnemonic,ins[a].op_str)==(m,o)
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,stop=0x300000000,0x400000000,0x500000000
    uc.mem_map(arena,0x1000000);uc.mem_map(stack,0x20000);uc.mem_map(stop,0x1000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
    def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
    def reg(r):return uc.reg_read(r)
    def ret(v=0):
        sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
    slots={r['Address']:r['Name'] for k in ['ScriptMetadata','ScriptMetadataMethod'] for r in script[k]};used=set()
    for i in ins.values():
        for op in i.operands:
            if op.type==capstone.CS_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:
                t=i.address+i.size+op.mem.disp
                if t in slots:used.add(t)
                elif i.mnemonic=='cmp' and op.size==1:uc.mem_write(base+t,b'\1')
    names={}
    for n,a in enumerate(sorted(used)):
        p=arena+0x1000+n*0x1000;q(base+a,p);names[slots[a]]=p
        d(p+0xE0,1);q(p+0x20,p+0x200);q(p+0x2C0,p+0x400);q(p+0x538,p+0x600);uc.mem_write(p+0x335,b'\1')
    assert {'Gameplay_TypeInfo','ProjectContext_TypeInfo','System.Collections.Generic.List<CharacterData>_TypeInfo'}<=set(names)
    actor,gs,ps,project,game,profile,counts=[arena+n for n in range(0x80000,0x87000,0x1000)]
    q(names['Gameplay_TypeInfo']+0xB8,gs);q(names['ProjectContext_TypeInfo']+0xB8,ps);q(ps,project);q(project+0x20,game);q(game+0x78,profile);q(gs+0x30,counts)
    assets={i:arena+0x90000+i*0x400 for i in range(1,10)};ids={0:None,**{v:k for k,v in assets.items()}}
    kinds={1:10,2:20,3:30,4:100,5:20,6:20,7:123,8:10,9:100};aligns={1:10,2:10,3:20,4:20,5:10,6:10,7:77,8:20,9:10}
    list_names={};array_names={};state={};opts={};visited=set();cases=[]
    def values(p):
        assert p in list_names,p;return [rq(rq(p+0x10)+0x20+n*8) for n in range(rd(p+0x18))]
    def write_list(p,vals,version=None):
        assert len(vals)<=256;q(p+0x10,p+0x100);q(p+0x118,256);d(p+0x18,len(vals))
        if version is not None:d(p+0x1C,version)
        for n,v in enumerate(vals):q(p+0x120+n*8,v)
    def new_list(p,name,vals):list_names[p]=name;array_names[p+0x100]=name;write_list(p,vals,3);return p
    def collection(p):
        if p in list_names:return values(p)
        assert p in array_names,p;return [rq(p+0x20+n*8) for n in range(rq(p+0x18))]
    def label(p):return list_names.get(p,'null' if not p else 'unknown')
    def normalized(vals):return [ids[v] for v in vals]
    def roster_values():return [None if rq(actor+0x28+n*8)==0 else normalized(values(rq(actor+0x28+n*8))) for n in range(4)]
    def halt(error):state['error']=error;uc.emu_stop()
    def hook(_,address,size,__):
        a=address-base;rcx,rdx,r8,r9=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        if address==stop:uc.emu_stop();return
        if a==0x2B7D40:
            assert rcx==names['System.Collections.Generic.List<CharacterData>_TypeInfo'];state['alloc']+=1;p=arena+0x200000+state['alloc']*0x2000;new_list(p,'allocated'+str(state['alloc']),[]);ret(p)
        elif a==0xB02160:assert rcx in list_names;write_list(rcx,[],0);ret()
        elif a==0xB610A0:
            state['copies'].append(label(rdx))
            if not rdx:halt('collection');return
            write_list(rcx,collection(rdx),0);ret()
        elif a==0xB53F50:
            state['appends'].append((label(rcx),label(rdx)))
            if not rdx or opts.get('append_fail')==len(state['appends']):halt('collection');return
            write_list(rcx,values(rcx)+collection(rdx),rd(rcx+0x1C)+1);ret()
        elif a==0xB59E70:
            state['removes'].append((label(rcx),ids[rdx]));vals=values(rcx)
            if opts.get('remove_fail')==len(state['removes']):halt('remove');return
            found=rdx in vals
            if found:vals.remove(rdx);write_list(rcx,vals,rd(rcx+0x1C)+1)
            ret(int(found))
        elif a==0xB55950:state['contains'].append(ids[rdx]);ret(int(rdx in values(rcx)))
        elif a==0xB22150:
            vals=values(rcx)
            if rdx>=len(vals):halt('index');return
            ret(vals[rdx])
        elif a==0x112B9D0:
            state['clears'].append(array_names[rcx]);assert rdx==0
            if opts.get('clear_fail')==len(state['clears']):halt('clear');return
            for n in range(r8):q(rcx+0x20+n*8,0)
            ret()
        elif a==0x3B1E10:
            assert rcx==profile;state['typed'].append(rdx)
            if opts.get('typed_fail')==len(state['typed']):halt('typed');return
            ret(state['pools'][rdx])
        elif a==0x1C86600:
            assert rcx==0 and rdx>0;n=len(state['draws']);index=state['choices'][n] if n<len(state['choices']) else 0;assert index<rdx
            state['draws'].append([rdx,index]);ret(index)
        elif a==0x2B6FF0:assert rq(rcx)==rdx;ret()
        elif a==0x281D90:
            state['cctor']+=1
            if opts.get('class_fail'):halt('cctor');return
            d(rcx+0xE0,1);ret()
        elif a==0x113B490:halt('enumerator_version')
        elif a in (0x2B7D90,0x2B7D80):halt('null' if a==0x2B7D90 else 'index')
        elif a in ins:
            visited.add(a)
            if a==0x2EB0:
                state['adds'].append((label(rcx),ids[rdx]))
                if opts.get('add_fail')==len(state['adds']):halt('add')
        else:raise AssertionError(f'unhandled {a:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    def run(name,rosters=None,in_data=None,out_data=None,current=None,pools=None,deps=None,kind=20,alignment=10,options=None,choices=(),counts_in=None):
        opts.clear();opts.update(options or {});state.clear();state.update(error=None,alloc=0,copies=[],appends=[],removes=[],contains=[],clears=[],typed=[],draws=[],choices=list(choices),adds=[],cctor=0,pools={})
        list_names.clear();array_names.clear()
        rosters=rosters if rosters is not None else [[1,1],[2],[3],[4]]
        for n,vals in enumerate(rosters):
            p=arena+0x100000+n*0x2000;new_list(p,'roster'+str(n),[assets.get(v,0) for v in vals or []]);q(actor+0x28+n*8,0 if vals is None else p)
        for i,p in assets.items():
            d(p+0x130,kinds[i]);d(p+0x134,aligns[i]);dep=(deps or {}).get(i,[])
            dp=arena+0x120000+i*0x2000;new_list(dp,'deps'+str(i),[assets.get(v,0) for v in dep or []]);q(p+0x128,0 if dep is None else dp)
        inp=arena+0x150000;out=arena+0x152000
        new_list(inp,'input',[assets.get(v,0) for v in in_data or []]);new_list(out,'output',[assets.get(v,0) for v in out_data or []])
        if opts.get('input_roster') is not None:inp=rq(actor+0x28+opts['input_roster']*8)
        if opts.get('output_is_input'):out=inp
        if opts.get('null_input'):inp=0
        if opts.get('null_output'):out=0
        physical=[]
        for n,v in enumerate(current or []):
            p=arena+0x180000+n*0x200;q(p+0x50,assets.get(v,0));physical.append(0 if v=='null_character' else p)
        live=arena+0x170000;new_list(live,'currentCharacters',physical);q(gs+0x18,0 if opts.get('null_current') else live)
        pools=pools if pools is not None else {10:[1,8],20:[2,5,6],30:[3],100:[4,9]}
        for n,k in enumerate([10,20,30,100]):
            vals=pools[k];p=arena+0x190000+n*0x2000;array_names[p]='pool'+str(k);q(p+0x18,len(vals or []))
            for j,v in enumerate(vals or []):q(p+0x20+j*8,assets.get(v,0))
            state['pools'][k]=0 if vals is None else p
        numbers=counts_in or [2,1,1,1,0,0,0,0]
        for n,v in enumerate(numbers):d(counts+0x14+n*4,v)
        q(gs+0x30,0 if opts.get('null_counts') else counts);d(names['Gameplay_TypeInfo']+0xE0,0 if opts.get('cold') else 1)
        q(ps,0 if opts.get('null_project') else project)
        sp=stack+0x10000;q(sp,stop);q(sp+0x28,0);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,0 if opts.get('null_this') else actor)
        uc.reg_write(x.UC_X86_REG_RDX,(kind&0xffffffff) if name=='FilterIfCanAppearCharacters' else (alignment&0xffffffff) if name=='GetScriptCharactersOfAlignment' else inp if name=='UpdateCurrentCharacters' else 0)
        uc.reg_write(x.UC_X86_REG_R8,inp);uc.reg_write(x.UC_X86_REG_R9,out)
        uc.emu_start(base+entries[name],stop,count=30000)
        if not state['error']:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8
        result=reg(x.UC_X86_REG_RAX);returned=normalized(values(result)) if not state['error'] and name not in ('UpdateCurrentCharacters','CleanupCharactersList') and result else None
        record={'method':name,'input':{'rosters':rosters,'in_data':in_data,'out_data':out_data,'current':current,'pools':pools,'deps':deps,'kind':kind,'alignment':alignment,'options':dict(opts),'choices':list(choices),'counts':numbers},
                'error':state['error'],'returned':returned,'rosters':roster_values(),'typed_order':list(state['typed']),'draws':list(state['draws']),'adds':list(state['adds']),'removes':list(state['removes']),'clears':list(state['clears']),
                'versions':[None if not rq(actor+0x28+n*8) else rd(rq(actor+0x28+n*8)+0x1C) for n in range(4)],'partial_lists':{label(p):normalized(values(p)) for p in list_names if label(p).startswith('allocated') or label(p)=='output'}}
        cases.append(record);return record
    for rosters in [[],[[1,1],[2],[3],[4]],[[None,1],[],[3,3],[]],[[],[],[],[]]]:
        if not rosters:continue
        r=run('GetAllCurrentCharacters',rosters);assert r['returned']==sum(rosters,[]) and r['rosters']==rosters
    for n in range(1,5):
        r=run('GetAllCurrentCharacters',options={'append_fail':n});assert r['error']=='collection' and r['partial_lists']['allocated1']==sum([[1,1],[2],[3],[4]][:n-1],[])
    for current in [[],[1],[1,1],[1,1,1],[2,4],[None],['null_character'],[1,'null_character']]:
        for method in ('GetNotInPlayCharacters','GetNotInDeckCharacters'):
            r=run(method,current=current);basevals=[1,1,2,3,4] if method=='GetNotInPlayCharacters' else [4,9,2,5,6,3,1,8]
            exp=basevals.copy()
            for v in current:
                if v=='null_character':break
                if v in exp:exp.remove(v)
            assert r['error']==('null' if 'null_character' in current else None)
            assert (r['returned'] if not r['error'] else r['partial_lists']['allocated1'])==exp
            assert r['typed_order']==([100,20,30,10] if method=='GetNotInDeckCharacters' else [])
    r=run('GetNotInDeckCharacters',rosters=[None]*4,current=[1],options={'null_this':True});assert r['returned']==[4,9,2,5,6,3,8] and r['error'] is None
    for method in ('GetNotInPlayCharacters','GetNotInDeckCharacters'):
        r=run(method,options={'null_current':True});assert r['error']=='null' and r['partial_lists']['allocated1']
    for n in range(1,5):
        r=run('GetNotInDeckCharacters',options={'typed_fail':n});assert r['error']=='typed' and r['typed_order']==[100,20,30,10][:n]
    for alignment in [-1,0,10,20,77,2147483647]:
        r=run('GetScriptCharactersOfAlignment',alignment=alignment);assert r['returned']==[v for v in [1,1,2,3,4] if aligns[v]==alignment]
    r=run('GetScriptCharactersOfAlignment',rosters=[[1,None],[2],[3],[4]]);assert r['error']=='null' and r['partial_lists']['allocated2']==[1]
    for vals in [[],[4,3,2,1,1,7,8],[1,None,2],[None]]:
        r=run('UpdateCurrentCharacters',in_data=vals);expected=[[],[],[],[]]
        for v in vals:
            if v is None:break
            if kinds[v] in [10,20,30,100]:expected[[10,20,30,100].index(kinds[v])].append(v)
        assert r['rosters']==expected and r['error']==('null' if None in vals else None) and r['clears']==['roster3','roster2','roster1','roster0']
    r=run('UpdateCurrentCharacters',options={'null_input':True});assert r['error']=='null' and r['rosters']==[[],[],[],[]]
    for n in range(1,5):
        r=run('UpdateCurrentCharacters',options={'clear_fail':n});assert r['error']=='clear';assert r['rosters']==[[1,1],[2],[3],[4]][:4-n]+[[]]*n
        rosters=[[1],[2],[3],[4]];rosters[4-n]=None;r=run('UpdateCurrentCharacters',rosters);assert r['error']=='null' and r['rosters'][4-n+1:]==[[]]*(n-1)
    for index in range(4):
        r=run('UpdateCurrentCharacters',options={'input_roster':index});assert r['rosters']==[[],[],[],[]] and not r['adds']
    # Filter considers only existing inData roles of requested type, and tests
    # whether ANY dependency appears in that same input, without early exit.
    for kind,dependency in itertools.product([10,20,30,100,123],[[],[1],[8],[None],[8,1]]):
        role={10:1,20:2,30:3,100:4,123:7}[kind];rosters=[[1,1],[2,2],[3,3],[4,4]]
        inp=[role,role,1];outp=[role,role,8];r=run('FilterIfCanAppearCharacters',rosters,inp,outp,deps={role:dependency},kind=kind)
        reject=bool(dependency) and not any(d in inp for d in dependency)
        expected=outp.copy();rr=[v.copy() for v in rosters]
        if reject:
            for v in inp:
                if kinds[v]==kind:
                    if kind in [10,20,30,100] and v in rr[[10,20,30,100].index(kind)]:rr[[10,20,30,100].index(kind)].remove(v)
                    if v in expected:expected.remove(v)
        assert r['returned']==expected and r['rosters']==rr and r['error'] is None
    r=run('FilterIfCanAppearCharacters',in_data=[1],out_data=[6],deps={6:[8]});assert r['returned']==[6] and not r['removes']
    for opts_,err in [({'null_input':True},'null'),({'null_output':True},None)]:
        r=run('FilterIfCanAppearCharacters',in_data=[],options=opts_);assert r['error']==err
    r=run('FilterIfCanAppearCharacters',in_data=[2],out_data=[2],deps={2:None});assert r['error']=='null' and not r['removes']
    r=run('FilterIfCanAppearCharacters',in_data=[2],out_data=[2],deps={2:[8]},options={'null_output':True});assert r['error']=='null' and r['rosters'][1]==[]
    r=run('FilterIfCanAppearCharacters',in_data=[2,2],deps={2:[8]},options={'output_is_input':True});assert r['error']=='enumerator_version' and r['rosters'][1]==[]
    # Native Cleanup calls native aggregation and filtering on an unchanged
    # snapshot. Defaults have no missing count, but typed getters still occur.
    r=run('CleanupCharactersList');assert r['typed_order']==[20,30,100,10] and not r['draws'] and r['rosters']==[[1,1],[2],[3],[4]]
    for counts_ in [[3,2,2,2,0,0,0,0],[0,0,0,0,3,2,2,2],[-1,-1,-1,-1,-2,-2,-2,-2],[2147483647,0,0,0,0,0,0,0]]:
        r=run('CleanupCharactersList',rosters=[[],[],[],[]],counts_in=counts_);assert r['typed_order']==[20,30,100,10]
        pools_={10:[1,8],20:[2,5,6],30:[3],100:[4,9]};expected=[]
        for n,k in enumerate([10,20,30,100]):
            ci={10:0,20:2,30:3,100:1}[k];need=max(counts_[ci],counts_[ci+4]);expected.append(pools_[k][:max(0,min(need,len(pools_[k])))])
        assert r['rosters']==expected
    r=run('CleanupCharactersList',rosters=[[1],[2],[],[]],deps={2:[8]},counts_in=[1,0,1,0,0,0,0,0]);assert r['rosters'][1]==[5] and r['draws']==[[2,0]]
    for n in range(1,5):
        r=run('CleanupCharactersList',rosters=[[],[],[],[]],counts_in=[1,1,1,1,0,0,0,0],options={'typed_fail':n});assert r['error']=='typed' and len(r['adds'])==n-1
    r=run('CleanupCharactersList',rosters=[[1],[2],[3],[]],deps={2:[8],3:[2]},counts_in=[1,0,1,1,0,0,0,0])
    assert r['rosters']==[[1],[5],[3],[]] and r['removes']==[('roster1',2),('allocated4',2),('allocated4',5)]
    r=run('CleanupCharactersList',rosters=[[],[],[],[]],pools={10:[],20:[2,5,2],30:[],100:[]},counts_in=[0,0,3,0,0,0,0,0],choices=[2,1,0])
    assert r['rosters']==[[],[2,2,5],[],[]] and r['draws']==[[3,2],[2,1],[1,0]]
    r=run('CleanupCharactersList',rosters=[[],[],[],[]],pools={10:[],20:[None],30:[],100:[]},counts_in=[0,0,1,0,0,0,0,0])
    assert r['rosters']==[[],[None],[],[]] and r['error'] is None
    for n in range(4):
        pools_={10:[1],20:[2],30:[3],100:[4]};pools_[[20,30,100,10][n]]=None
        r=run('CleanupCharactersList',rosters=[[],[],[],[]],pools=pools_,counts_in=[1,1,1,1,0,0,0,0])
        assert r['error']=='collection' and len(r['adds'])==n
    r=run('UpdateCurrentCharacters',rosters=[[],[],[],[]],in_data=[]);assert r['versions']==[4,4,4,4] and not r['clears']
    for n in range(1,5):
        r=run('UpdateCurrentCharacters',in_data=[4,3,2,1],options={'add_fail':n});assert r['error']=='add' and sum(map(len,r['rosters']))==n-1
    for n in range(1,3):
        r=run('FilterIfCanAppearCharacters',in_data=[2],out_data=[2],deps={2:[8]},options={'remove_fail':n})
        assert r['error']=='remove' and r['rosters'][1]==([2] if n==1 else []) and r['partial_lists']['output']==[2]
    r=run('CleanupCharactersList',rosters=[[1],[2],[3],[4]],counts_in=[-2147483648]*8)
    assert r['rosters']==[[1,1,8],[2,2,5,6],[3,3],[4,4,9]] and len(r['draws'])==8
    for method in ['GetNotInPlayCharacters','GetNotInDeckCharacters','CleanupCharactersList']:
        r=run(method,options={'cold':True,'class_fail':True});assert r['error']=='cctor'
    return {'build':BUILD,'status':'passed','exact_methods':exact,'native_cases':len(cases),'instruction_assertions':len(checks),'visited_instructions':len(visited),'cases':cases,
            'boundaries':['all seven managed bodies execute; Cleanup calls native aggregation/filter','native enumerator creation, MoveNext/version rejection, disposal stub, and capacity-sufficient Add execute','ToArray starting getter is an explicit previously-audited service with ordered calls','allocation, collection constructors/AddRange/Contains/Remove/index access and Array.Clear are services','comparison service is first matching reference identity in these fixtures; runtime comparer/Unity object liveness not reconstructed','Random.Range supplied indices and widths recorded; Unity RNG state not recovered','generic exception unwinding/finally on thrown service errors is not emulated']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--game-root',required=True);p.add_argument('--dumper-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if k not in ('exact_methods','cases')}))
