"""Native GetNotInDeck -> actual lazy typed getters -> actual board enumeration."""
import argparse
import hashlib
import json
import struct
from fractions import Fraction
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
    extraction=json.loads((root/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(p,h):
        b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
    raw=pinned(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    exact=[]
    for name,a in [('Gameplay$$GetNotInDeckCharacters',0x37C8E0),('AscensionsData$$GetStartingtCharactersOfType',0x3B1E10)]:
        rows=[r for r in script['ScriptMethod'] if r['Name']==name and r['Address']==a];assert len(rows)==1;exact.append(rows[0])
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase;cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    ins={}
    for a,b in [(0x37C8E0,0x37CBF7),(0x3B1E10,0x3B2001),(0xB16640,0xB16675),(0x9674A0,0x9674E0),(0x9693D0,0x969470),(0x33ED50,0x33ED53)]:
        rows=list(cs.disasm(pe.get_data(a,b-a),a));assert rows[-1].address+rows[-1].size==b;ins.update({i.address:i for i in rows})
    checks=[(0x37C9DB,'call','0x3b1e10'),(0x37CA36,'call','0x3b1e10'),(0x37CA88,'call','0x3b1e10'),(0x37CADA,'call','0x3b1e10'),(0x3B1E67,'call','0x1c86600'),(0x3B1EB2,'call','0x1c86600'),(0x3B1F7B,'jmp','0xb01f50')]
    for a,m,o in checks:assert (ins[a].mnemonic,ins[a].op_str)==(m,o)
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,stop=0x300000000,0x400000000,0x500000000
    uc.mem_map(arena,0x400000);uc.mem_map(stack,0x20000);uc.mem_map(stop,0x1000)
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
        p=arena+0x1000+n*0x1000;q(base+a,p);names[slots[a]]=p;d(p+0xE0,1);q(p+0x20,p+0x200);q(p+0x2C0,p+0x400);q(p+0x538,p+0x600);uc.mem_write(p+0x335,b'\1')
    assert {'Gameplay_TypeInfo','ProjectContext_TypeInfo','Method$System.Collections.Generic.List<CharacterData>.ToArray()'}<=set(names)
    gs,ps,project,game,profile,result,live=[arena+n for n in range(0x80000,0x87000,0x1000)]
    q(names['Gameplay_TypeInfo']+0xB8,gs);q(names['ProjectContext_TypeInfo']+0xB8,ps);q(ps,project);q(project+0x20,game);q(game+0x78,profile);q(gs+0x18,live)
    scripts={1:arena+0x90000,2:arena+0x91000};script_ids={0:None,**{p:n for n,p in scripts.items()}}
    order=[100,20,30,10];factions=[10,20,30,100]
    starting=[[1001,1001,None],[2001],[],[4001,4001,None]]
    script_lists={1:[[1101],[2101,2101],[3101],[4101]],2:[[1201],[2201],[3201,None],[4201,4201]]}
    arrays={};list_arrays={};visited=set();state={};options={}
    def array(p,vals):
        arrays[p]=vals;q(p+0x18,len(vals))
        for n,v in enumerate(vals):q(p+0x20+n*8,v or 0)
        return p
    def result_values():return [rq(rq(result+0x10)+0x20+n*8) or None for n in range(rd(result+0x18))]
    def write_result(vals):
        d(result+0x18,len(vals))
        for n,v in enumerate(vals):q(rq(result+0x10)+0x20+n*8,v or 0)
    def halt(error):state['error']=error;uc.emu_stop()
    def hook(_,address,size,__):
        a=address-base;rcx,rdx,r8=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8)]
        if address==stop:uc.emu_stop();return
        if a==0x2B7D40:state['allocations']+=1;assert rcx==names['System.Collections.Generic.List<CharacterData>_TypeInfo'];ret(result)
        elif a==0xB02160:assert rcx==result;q(result+0x10,result+0x100);q(result+0x118,256);d(result+0x18,0);d(result+0x1C,0);ret()
        elif a==0xB53F50:
            state['appends']+=1
            if not rdx or options.get('append_fail')==state['appends']:halt('collection');return
            assert rcx==result;write_result(result_values()+arrays[rdx]);d(result+0x1C,rd(result+0x1C)+1);ret()
        elif a==0xB01F50:
            state['to_arrays']+=1;assert rdx==names['Method$System.Collections.Generic.List<CharacterData>.ToArray()']
            if options.get('to_array_fail')==state['to_arrays']:halt('to_array');return
            ret(list_arrays[rcx])
        elif a==0x1C86600:
            n=len(state['draws']);index=state['choices'][n];assert rcx==0 and 0<=index<rdx;caller=rq(reg(x.UC_X86_REG_RSP))-base
            source={0x3B1E6C:'inline',0x3B1EB7:'custom'}[caller];state['draws'].append({'source':source,'width':rdx,'index':index});ret(index)
        elif a==0x2B6FF0:
            assert rq(rcx)==rdx
            if rcx==profile+0x60:state['cache_writes'].append(script_ids[rdx])
            ret()
        elif a==0xB59E70:
            state['removes'].append(rdx or None)
            if options.get('remove_fail')==len(state['removes']):halt('remove');return
            vals=result_values();item=rdx or None;found=item in vals
            if found:vals.remove(item);write_result(vals);d(result+0x1C,rd(result+0x1C)+1)
            ret(int(found))
        elif a==0x281D90:
            state['cctor']+=1
            if options.get('class_fail'):halt('cctor');return
            d(rcx+0xE0,1);ret()
        elif a in (0x2B7D90,0x2B7D80):halt('null' if a==0x2B7D90 else 'index')
        elif a in ins:
            visited.add(a)
            if a==0x3B1E10:
                assert rcx==profile;state['typed_order'].append(rdx)
                if options.get('typed_fail')==len(state['typed_order']):halt('typed')
        else:raise AssertionError(f'unhandled {a:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    def run(inline,custom,cached,choices,current,opts=None):
        options.clear();options.update(opts or {});state.clear();state.update(error=None,allocations=0,appends=0,to_arrays=0,choices=choices,draws=[],cache_writes=[],removes=[],cctor=0,typed_order=[])
        q(profile+0x60,scripts.get(cached,0));q(profile+0x20,0 if inline is None else array(arena+0xA0000,[scripts.get(k,0) for k in inline]))
        if custom is None:q(profile+0x18,0)
        else:
            records=[]
            for n,k in enumerate(custom):
                p=arena+0xB0000+n*0x100;q(p+0x18,scripts.get(k,0));records.append(0 if k=='missing' else p)
            q(profile+0x18,array(arena+0xA1000,records))
        normalized_lists={k:[None if options.get('null_list')==[k,n] else vals for n,vals in enumerate(v)] for k,v in script_lists.items()}
        normalized_starting=[None if options.get('null_starting')==n else vals for n,vals in enumerate(starting)]
        for n,vals in enumerate(normalized_starting):q(profile+0x40+n*8,0 if vals is None else array(arena+0xC0000+n*0x1000,vals))
        for k,lists in normalized_lists.items():
            for n,vals in enumerate(lists):
                p=arena+0xD0000+k*0x10000+n*0x1000;q(scripts[k]+0x10+n*8,0 if vals is None else p)
                if vals is not None:list_arrays[p]=array(p+0x800,vals)
        physical=[]
        for n,v in enumerate(current or []):
            p=arena+0x100000+n*0x200;q(p+0x50,v if isinstance(v,int) else 0);physical.append(0 if v=='null_character' else p)
        q(live+0x10,array(live+0x100,physical));d(live+0x18,len(physical));d(live+0x1C,3);q(gs+0x18,0 if current is None else live)
        d(names['Gameplay_TypeInfo']+0xE0,0 if options.get('cold') else 1);q(ps,0 if options.get('null_project') else project)
        sp=stack+0x10000;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,0);uc.reg_write(x.UC_X86_REG_RDX,0)
        uc.emu_start(base+0x37C8E0,stop,count=10000)
        if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8 and reg(x.UC_X86_REG_RAX)==result
        return {'input':{'inline':inline,'custom':None if custom is None else [None if k=='missing' else {'script_info':k} for k in custom],'initial_cache':cached,'script_lists':normalized_lists,'starting':normalized_starting,'choices':choices,'current':current,'options':dict(options)},
                'expected':{'error':state['error'],'cache':script_ids[rq(profile+0x60)],'draws':list(state['draws']),'cache_writes':list(state['cache_writes']),'typed_order':list(state['typed_order']),'output_prefix':result_values(),'version':rd(result+0x1C),'removes':list(state['removes']),'append_count':state['appends'],'to_array_count':state['to_arrays']}}
    def expected_paths(inline,custom,cached,current,opts):
        def finish(s):
            s=dict(s);s['removes']=list(s['removes']);s['output_prefix']=list(s['output_prefix'])
            if s['error'] is not None:return s
            if opts.get('cold') and opts.get('class_fail'):s['error']='cctor';return s
            if current is None:s['error']='null';return s
            for v in current:
                if v=='null_character':s['error']='null';break
                s['removes'].append(v)
                if opts.get('remove_fail')==len(s['removes']):s['error']='remove';break
                if v in s['output_prefix']:s['output_prefix'].remove(v);s['version']+=1
            return s
        def visit(phase,s):
            if phase==4:yield finish(s);return
            if opts.get('null_project'):s['error']='null';yield s;return
            kind=order[phase];faction=factions.index(kind);s['typed_order']=s['typed_order']+[kind]
            if opts.get('typed_fail')==phase+1:s['error']='typed';yield s;return
            candidates=[s]
            if s['cache'] is None:
                if inline is None:s['error']='null';yield s;return
                if inline:
                    candidates=[]
                    for index,k in enumerate(inline):
                        b=dict(s);b.update(cache=k,choices=s['choices']+[index],draws=s['draws']+[{'source':'inline','width':len(inline),'index':index}],cache_writes=s['cache_writes']+[k],probability=s['probability']/len(inline));candidates.append(b)
                after=[]
                for s_ in candidates:
                    if custom is None:s_['error']='null';after.append(s_)
                    elif custom:
                        for index,k in enumerate(custom):
                            b=dict(s_);b.update(choices=s_['choices']+[index],draws=s_['draws']+[{'source':'custom','width':len(custom),'index':index}],probability=s_['probability']/len(custom))
                            if k=='missing':b['error']='null'
                            else:b.update(cache=k,cache_writes=s_['cache_writes']+[k])
                            after.append(b)
                    else:
                        if not inline:s_['cache_writes']=s_['cache_writes']+[None]
                        after.append(s_)
                candidates=after
            for branch in candidates:
                if branch['error'] is not None:yield branch;continue
                k=branch['cache'];values=script_lists[k][faction] if k else starting[faction]
                if k:
                    if opts.get('null_list')==[k,faction]:branch['error']='null';yield branch;continue
                    branch['to_array_count']+=1
                    if opts.get('to_array_fail')==branch['to_array_count']:branch['error']='to_array';yield branch;continue
                elif opts.get('null_starting')==faction:values=None
                branch['append_count']+=1
                if values is None or opts.get('append_fail')==branch['append_count']:branch['error']='collection';yield branch;continue
                branch['output_prefix']=branch['output_prefix']+values;branch['version']+=1
                yield from visit(phase+1,branch)
        initial={'error':None,'cache':cached,'choices':[],'probability':Fraction(1),'draws':[],'cache_writes':[],'typed_order':[],'output_prefix':[],'version':0,'removes':[],'append_count':0,'to_array_count':0}
        return list(visit(0,initial))
    cases=[]
    configurations=[([],[]),([2],[]),([None,2],[]),([2,2],[None,1]),([2,2],[None,None]),([2,2],[None,'missing',1]),([],None),(None,[])]
    def check(inline,custom,cached,current,opts):
        expected=expected_paths(inline,custom,cached,current,opts);assert sum((e['probability'] for e in expected),Fraction())==1
        for e in expected:
            case=run(inline,custom,cached,e['choices'],current,opts)
            assert case['expected']=={k:v for k,v in e.items() if k not in ('choices','probability')},(case,e)
            case['probability']=str(e['probability']);cases.append(case)
    for inline,custom in configurations:
        for cached in [None,2]:
            for current in [[],[None,None,4001,4001,4001],[4101,2101,2101],[4001,'null_character']]:check(inline,custom,cached,current,{})
    for n in range(1,5):
        check([2,2],[None,1],None,[],{'append_fail':n});check([2,2],[None,1],None,[],{'typed_fail':n});check([2,2],[None,1],None,[],{'to_array_fail':n})
        check([],[],1,[],{'null_list':[1,n-1]});check([],[],None,[],{'null_starting':n-1})
    for opts in [{'cold':True},{'cold':True,'class_fail':True},{'remove_fail':1},{'remove_fail':3},{'null_project':True}]:check([2,2],[None,1],None,[4001,2101,None],opts)
    check([2,2],[None,1],None,None,{})
    return {'build':BUILD,'status':'passed','exact_methods':exact,'native_cases':len(cases),'instruction_assertions':len(checks),'visited_instructions':len(visited),'cases':cases,
            'boundaries':['actual GetNotInDeck directly calls actual AscensionsData typed getter in100,20,30,10 order','native global graph reads and board enumerator creation/MoveNext execute','fixed existing temporary profile graph; no mode provider or snapshot cloning inferred','List.ToArray snapshot and AddRange/Remove reference equality are explicit stable services','valid uniform RNG indices supplied; no Unity PRNG recovered','GC barriers observe native cache writes; exception unwinding excluded']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--game-root',required=True);p.add_argument('--dumper-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);Path(a.output).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if k not in ('exact_methods','cases')}))
