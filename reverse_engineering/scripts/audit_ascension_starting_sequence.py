"""Execute ascension starting-list concatenators and ordered lazy-selection calls."""
import argparse
import hashlib
import itertools
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
    def pinned(path,digest):
        b=path.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==digest.upper();return b
    raw=pinned(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    entries={'AscensionsData$$GetAllStartingCharacters':0x3B18A0,'AscensionsData$$GetAllStartingtCharacters':0x3B1BE0,'AscensionsData$$GetStartingtCharactersOfType':0x3B1E10,'GameData$$GetAllStartingCharactersFromAscension':0x3DC180}
    exact=[]
    for name,a in entries.items():
        rows=[r for r in script['ScriptMethod'] if r['Name']==name and r['Address']==a];assert len(rows)==1;exact.append(rows[0])
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    spans=[(0x3B18A0,0x3B1BDF),(0x3B1BE0,0x3B1CA3),(0x3B1E10,0x3B2001),(0x3DC180,0x3DC288),(0xB53F50,0xB53F6D)]
    ins={}
    for a,b in spans:
        rows=list(cs.disasm(pe.get_data(a,b-a),a));assert rows[-1].address+rows[-1].size==b
        ins.update({i.address:i for i in rows})
    checks=[(0x3B190E,'call','0x3b1e10'),(0x3B1938,'call','0x3b1e10'),(0x3B1A71,'call','0xb01f50'),(0x3B1BA4,'call','0xb01f50'),(0x3DC1F2,'call','0x3b1e10'),(0x3DC21C,'call','0x3b1e10'),(0x3DC23D,'call','0x3b1e10'),(0x3DC25E,'call','0x3b1e10'),(0xB53F57,'mov','edx, dword ptr [rcx + 0x18]'),(0xB53F68,'jmp','0xb57180')]
    for a,m,o in checks:assert (ins[a].mnemonic,ins[a].op_str)==(m,o)
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,stop=0x300000000,0x400000000,0x500000000
    uc.mem_map(arena,0x200000);uc.mem_map(stack,0x20000);uc.mem_map(stop,0x1000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def d(a,v):uc.mem_write(a,struct.pack('<I',v))
    def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
    def reg(r):return uc.reg_read(r)
    def ret(v=0):
        sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
    slots={r['Address']:r['Name'] for k in ['ScriptMetadata','ScriptMetadataMethod'] for r in script[k]};used=set()
    for i in ins.values():
        for op in i.operands:
            if op.type==capstone.CS_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:
                target=i.address+i.size+op.mem.disp
                if target in slots:used.add(target)
                elif i.mnemonic=='cmp' and op.size==1:uc.mem_write(base+target,b'\1')
    names={}
    for n,a in enumerate(sorted(used)):
        p=arena+0x1000+n*0x1000;q(base+a,p);names[slots[a]]=p
    required={'System.Collections.Generic.List<CharacterData>_TypeInfo','Method$System.Collections.Generic.List<CharacterData>..ctor()','Method$System.Collections.Generic.List<CharacterData>.AddRange()','Method$System.Collections.Generic.List<CharacterData>.ToArray()'}
    assert required<=set(names),names
    add=names['Method$System.Collections.Generic.List<CharacterData>.AddRange()'];q(add+0x20,arena+0x18000);q(arena+0x180C0,arena+0x19000);q(arena+0x19090,arena+0x1A000)
    profile,output,inline_array,custom_array=[arena+n for n in (0x20000,0x21000,0x22000,0x23000)]
    scripts={'A':arena+0x30000,'B':arena+0x31000};labels={0:None,**{v:k for k,v in scripts.items()}}
    array_values={};list_values={};array_labels={};list_labels={}
    def array(p,values,label):
        q(p+0x18,len(values));array_values[p]=list(values);array_labels[p]=label
        for n,v in enumerate(values):q(p+0x20+n*8,v)
        return p
    factions=[10,20,30,100]
    # Equal references and null elements are intentional preserved occurrences.
    plain=[[1001,1001,0],[2001],[],[4001]]
    payload={'A':[[1101],[2101,2101],[3101],[4101]],'B':[[1201],[2201],[3201,0],[4201,4201]]}
    for n,vals in enumerate(plain):q(profile+0x40+n*8,array(arena+0x40000+n*0x1000,vals,('plain',n)))
    for j,(name,p) in enumerate(scripts.items()):
        for n,vals in enumerate(payload[name]):
            lp=arena+0x50000+j*0x10000+n*0x1000;list_values[lp]=vals;list_labels[lp]=(name,n);q(p+0x10+n*8,lp)
            array(lp+0x800,vals,(name,n))
    state={};opts={};visited=set();corpus=[]
    script_ids={None:None,'A':1,'B':2}
    def nullable_ids(values):return None if values is None else [v or None for v in values]
    def halt(error):state['error']=error;uc.emu_stop()
    def hook(_,address,size,__):
        a=address-base;rcx,rdx,r8,r9=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        if address==stop:uc.emu_stop();return
        if a==0x2B7D40:
            assert rcx==names['System.Collections.Generic.List<CharacterData>_TypeInfo'];state['alloc']+=1;ret(output)
        elif a==0xB02160:
            assert rcx==output and rdx==names['Method$System.Collections.Generic.List<CharacterData>..ctor()'];d(output+0x18,0);ret()
        elif a==0x1C86600:
            n=len(state['draws']);assert rcx==0 and n<len(state['choices']);index=state['choices'][n];assert 0<=index<rdx
            state['draws'].append([rdx,index]);caller=rq(reg(x.UC_X86_REG_RSP))-base
            source={0x3B1E6C:'inline',0x3B1EB7:'custom',0x3B1997:'inline',0x3B19E2:'custom',0x3B1AD6:'inline',0x3B1B21:'custom'}[caller];state['draw_details'].append({'source':source,'width':rdx,'index':index});state['events'].append(['draw',rdx,index]);ret(index)
        elif a==0x2B6FF0:
            assert rcx==profile+0x60 and rq(rcx)==rdx;state['cache_writes'].append(labels[rdx]);state['events'].append(['cache',labels[rdx]]);ret()
        elif a==0xB01F50:
            assert rcx in list_values and rdx==names['Method$System.Collections.Generic.List<CharacterData>.ToArray()'];state['arrays'].append(list_labels[rcx]);state['events'].append(['to_array',*list_labels[rcx]])
            if opts.get('to_array_fail')==len(state['arrays']):halt('to_array');return
            ret(rcx+0x800)
        elif a==0xB57180:
            assert rcx==output and rdx==len(state['output']) and r9==arena+0x1A000;state['appends'].append(array_labels.get(r8));state['events'].append(['append',array_labels.get(r8)])
            if opts.get('append_fail')==len(state['appends']) or not r8:halt('collection');return
            assert r8 in array_values;state['output'].extend(array_values[r8]);d(output+0x18,len(state['output']));ret()
        elif a in (0x2B7D90,0x2B7D80):halt('null' if a==0x2B7D90 else 'index')
        elif a in ins:
            visited.add(a)
            if a==0x3B1E10:state['factions'].append(rdx)
            elif a in (0x3B194F,0x3B1A8E):state['factions'].append(30 if a==0x3B194F else 100)
            elif a in (0x3B1C4E,0x3B1C61,0x3B1C74,0x3B1C87):state['factions'].append({0x3B1C4E:10,0x3B1C61:20,0x3B1C74:30,0x3B1C87:100}[a])
        else:raise AssertionError(f'unhandled {a:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    def run(method,inline,custom,choices,cached=None,options=None,kind=10):
        opts.clear();opts.update(options or {});state.clear();state.update(error=None,choices=list(choices),draws=[],cache_writes=[],arrays=[],appends=[],output=[],events=[],alloc=0,draw_details=[],factions=[])
        q(profile+0x60,scripts.get(cached,0))
        q(profile+0x20,0 if inline is None else array(inline_array,[scripts.get(k,0) for k in inline],('inline',0)))
        if custom is None:q(profile+0x18,0)
        else:
            records=[]
            for n,k in enumerate(custom):
                p=arena+0x80000+n*0x100;q(p+0x18,scripts.get(k,0));records.append(0 if k=='missing' else p)
            q(profile+0x18,array(custom_array,records,('custom',0)))
        for n in range(4):q(profile+0x40+n*8,0 if opts.get('null_plain')==n+1 else arena+0x40000+n*0x1000)
        for j,(name,p) in enumerate(scripts.items()):
            for n in range(4):q(p+0x10+n*8,0 if opts.get('null_list')==(name,n) else arena+0x50000+j*0x10000+n*0x1000)
        sp=stack+0x10000;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp)
        uc.reg_write(x.UC_X86_REG_RCX,0 if method.startswith('GameData') else profile)
        uc.reg_write(x.UC_X86_REG_RDX,(0 if opts.get('null_profile') else profile) if method.startswith('GameData') else kind)
        uc.reg_write(x.UC_X86_REG_R8,0)
        uc.emu_start(base+entries[method],stop,count=4000)
        if not state['error']:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8
        normalized_lists={str(script_ids[name]):[None if opts.get('null_list')==(name,n) else nullable_ids(vals) for n,vals in enumerate(lists)] for name,lists in payload.items()}
        corpus.append({'input':{'method':{'AscensionsData$$GetAllStartingCharacters':'AllLazy','GameData$$GetAllStartingCharactersFromAscension':'AllLazyGameData','AscensionsData$$GetAllStartingtCharacters':'AllStored','AscensionsData$$GetStartingtCharactersOfType':'Typed'}[method],
            'inline':None if inline is None else [script_ids[k] for k in inline],
            'custom':None if custom is None else [None if k=='missing' else {'script_info':script_ids[k]} for k in custom],
            'initial_cache':script_ids[cached],'script_lists':normalized_lists,
            'starting':[None if opts.get('null_plain')==n+1 else nullable_ids(vals) for n,vals in enumerate(plain)],'choices':list(choices),'type':kind,'service_options':dict(opts)},
            'expected':{'error':state['error'],'cache':script_ids[labels[rq(profile+0x60)]],
            'cache_writes':[script_ids[k] for k in state['cache_writes']],
            'draws':list(state['draw_details']),'faction_read_order':list(state['factions']),
            'output_prefix':nullable_ids(state['output']),
            'typed_result':nullable_ids(array_values.get(reg(x.UC_X86_REG_RAX))) if method.endswith('$$GetStartingtCharactersOfType') and not state['error'] else None,
            'returned_successfully':state['error'] is None,'append_count':len(state['appends'])}})
        return {'error':state['error'],'draws':list(state['draws']),'cache_writes':list(state['cache_writes']),'output':list(state['output']),'cache':labels[rq(profile+0x60)],'appends':[list(a) if a else None for a in state['appends']],'events':list(state['events']),'returned':reg(x.UC_X86_REG_RAX),'arrays':list(state['arrays'])}
    # Independent model enumerates only consumed draws, assigning exact rational
    # path weights; it never pads choices after a cache becomes nonnull.
    def expected_paths(inline,custom,cached=None):
        def visit(faction,cache,choices,draws,writes,partial,appends,weight):
            if faction==4:
                yield dict(choices=choices,draws=draws,cache_writes=writes,cache=cache,output=partial,appends=appends,error=None,weight=weight);return
            candidates=[(cache,choices,draws,writes,weight,None)]
            if cache is None:
                if inline is None:candidates=[(cache,choices,draws,writes,weight,'null')]
                else:
                    candidates=[(k,choices+[n],draws+[[len(inline),n]],writes+[k],weight/Fraction(len(inline)),None) for n,k in enumerate(inline)] if inline else candidates
                    after=[]
                    for c,ch_,dr,wr,w,e in candidates:
                        if custom is None:after.append((c,ch_,dr,wr,w,'null'))
                        elif custom:
                            for n,k in enumerate(custom):
                                after.append((c if k=='missing' else k,ch_+[n],dr+[[len(custom),n]],wr if k=='missing' else wr+[k],w/Fraction(len(custom)),'null' if k=='missing' else None))
                        else:after.append((c,ch_,dr,wr+([None] if not inline else []),w,None))
                    candidates=after
            for c,ch_,dr,wr,w,error in candidates:
                if error:
                    yield dict(choices=ch_,draws=dr,cache_writes=wr,cache=c,output=partial,appends=appends,error=error,weight=w)
                else:
                    source=c or 'plain';vals=payload[c][faction] if c else plain[faction]
                    yield from visit(faction+1,c,ch_,dr,wr,partial+vals,appends+[[source,faction]],w)
        return list(visit(0,cached,[],[],[],[],[],Fraction(1)))
    cases=[];weighted=[]
    wrappers=['AscensionsData$$GetAllStartingCharacters','GameData$$GetAllStartingCharactersFromAscension']
    configurations=[([],[]),(['B'],[]),([None,'B'],[]),(['B','B'],[None,'A']),(['B','B'],[None,None]),(['B'],['missing','A']),(['B','B'],[None,'missing','A']),([None,'B'],[None,'A']),([],['A','A']),([],['missing']),([],None),(None,[])]
    for inline,custom in configurations:
        for cached in (None,'B'):
            expected=expected_paths(inline,custom,cached);assert sum((r['weight'] for r in expected),Fraction())==1
            for e in expected:
                for method in wrappers:
                    r=run(method,inline,custom,e['choices'],cached)
                    for key in ('error','draws','cache_writes','cache','output','appends'):assert r[key]==e[key],(method,inline,custom,e,key,r,e)
                    if not r['error']:assert r['returned']==output and state['factions']==factions
                    else:assert state['factions']==factions[:len(r['appends'])+1]
                    case={'method':method,'inline':inline,'custom_payloads':custom,'initial_cache':cached,'choices':e['choices'],'probability':str(e['weight']),**{k:r[k] for k in ('error','draws','cache_writes','cache','output','appends')}}
                    cases.append(case)
                    if inline==['B','B'] and custom==[None,'A'] and cached is None and method==wrappers[0]:weighted.append(case)
    assert len(weighted)==46 and sum((Fraction(r['probability']) for r in weighted),Fraction())==1
    failures=[]
    for method in wrappers:
        # Selection begins with a null payload, then caches A for faction 20.
        choices=[0,0,1,1]
        for n in range(1,5):
            r=run(method,['B','B'],[None,'A'],choices,options={'append_fail':n});assert r['error']=='collection' and len(r['appends'])==n
            assert r['output']==sum([plain[0]]+[payload['A'][i] for i in range(1,n-1)],[]) if n>1 else r['output']==[]
            failures.append({'method':method,'failure':'append','at':n,**{k:r[k] for k in ('error','draws','cache','output')}})
        for n in range(3):
            r=run(method,['B','B'],[None,'A'],choices,options={'to_array_fail':n+1});assert r['error']=='to_array' and len(r['appends'])==n+1
            failures.append({'method':method,'failure':'to_array','at':n+1,**{k:r[k] for k in ('error','draws','cache','output')}})
        for n in range(4):
            r=run(method,[],[],[],cached='A',options={'null_list':('A',n)});assert r['error']=='null' and r['output']==sum(payload['A'][:n],[]) and len(r['appends'])==n
            r=run(method,[],[],[],options={'null_plain':n+1});assert r['error']=='collection' and r['output']==sum(plain[:n],[]) and len(r['appends'])==n+1
            failures.append({'method':method,'failure':'null selected list / null plain collection','at':n+1})
    stored='AscensionsData$$GetAllStartingtCharacters'
    for inline,custom in configurations:
        for cached in (None,'A'):
            r=run(stored,inline,custom,[],cached);assert not r['draws'] and not r['cache_writes'] and r['cache']==cached and r['output']==sum(plain,[]) and r['appends']==[['plain',n] for n in range(4)]
            cases.append({'method':stored,'inline':inline,'custom_payloads':custom,'initial_cache':cached,'output':r['output'],'draws':[]})
    for n in range(1,5):
        r=run(stored,[],[],[],options={'append_fail':n});assert r['error']=='collection' and r['output']==sum(plain[:n-1],[]) and not r['draws'];failures.append({'method':stored,'failure':'append','at':n})
    r=run(wrappers[1],[],[],[],options={'null_profile':True});assert r['error']=='null' and state['alloc']==1 and not r['draws'];failures.append({'method':wrappers[1],'failure':'null profile after result-list construction'})
    typed='AscensionsData$$GetStartingtCharactersOfType'
    for kind in [0,10,20,30,100,101]:
        r=run(typed,['B'],[None],[0,0],kind=kind);assert r['draws']==[[1,0],[1,0]] and r['cache'] is None
        assert r['returned']==(arena+0x40000+factions.index(kind)*0x1000 if kind in factions else 0)
        cases.append({'method':typed,'kind':kind,'draws':r['draws'],'cache':None})
    return {'build':BUILD,'status':'passed','exact_methods':exact,'native_cases':len(corpus),'instruction_assertions':len(checks),'visited_instructions':len(visited),'weighted_native_cases':weighted,'cases':corpus,'failure_cases':failures,
            'boundaries':['native four selected bodies and actual List.AddRange dispatch execute','allocation and list constructor authored services','List.InsertRange arrays append authored service: injected failures stop before current append','List.ToArray authored snapshot service','Unity Random.Range authored uniform index service; no Unity PRNG recovered','GC barriers recorded after native cache writes','partial output is retained in unreturned local list on failure; caller receives no successful result']}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--game-root',required=True);p.add_argument('--dumper-root',required=True);p.add_argument('--output',required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root)
    Path(a.output).write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in r.items() if k not in ('exact_methods','weighted_native_cases','cases','failure_cases')}))
