"""Actual ManageCharacters prefix through both actual pool builders and sources."""
import argparse
import hashlib
import itertools
import json
import struct
from fractions import Fraction
from pathlib import Path
from audit_character_assets import BUILD
from audit_round_candidate_composition import ASSETS

ENTRIES = {0x36ce30:'ManageCharacters', 0x36d720:'PickRoundDuplicates',
           0x36d3a0: 'PickRoundBluffs', 0x377170: 'ContainsScriptCharacter',
           0x37c3f0: 'GetAscensionAllStartingCharacters', 0x37c1a0: 'GetAllAscensionCharacters',
           0x37dc00: 'GetScriptCharacters', 0x3b1e10: 'GetStartingtCharactersOfType',
           0x36a550: 'FilterBluffableCharacters', 0x36b9c0: 'FilterRealCharacterType',
           0x369eb0: 'FilterAlignmentCharacters'}


def audit(game_root, dumper_root):
    import capstone, pefile, unicorn
    from unicorn import x86_const as x
    assert unicorn.__version__ == '2.1.4'
    root = Path(__file__).parents[1]
    lock = json.loads((root/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    ext = json.loads((root/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pin(p, expected):
        b = p.read_bytes(); assert hashlib.sha256(b).hexdigest().upper() == expected.upper(); return b
    raw = pin(game_root/'GameAssembly.dll', lock['inputs']['game_assembly']['sha256'])
    meta = json.loads(pin(dumper_root/'script.json', ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump = pin(dumper_root/'dump.cs', ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    declarations = {
        'CharacterData': ['public ECharacterType type; // 0x130', 'public EAlignment startingAlignment; // 0x134', 'public bool bluffable; // 0x13C'],
        'AscensionsData': ['public CustomScriptData[] possibleScriptsData; // 0x18','public ScriptInfo[] possibleScripts; // 0x20',
            'public CharacterData[] startingTownsfolks; // 0x40','public CharacterData[] startingOutsiders; // 0x48',
            'public CharacterData[] startingMinions; // 0x50','public CharacterData[] startingDemons; // 0x58',
            'public ScriptInfo currentPickedScript; // 0x60','public CharacterData[] townsfolks; // 0x68',
            'public CharacterData[] outsiders; // 0x70','public CharacterData[] minions; // 0x78','public CharacterData[] demons; // 0x80'],
        'Gameplay': ['public List<CharacterData> currentTownsfolks; // 0x28','public List<CharacterData> currentOutsiders; // 0x30',
            'public List<CharacterData> currentMinions; // 0x38','public List<CharacterData> currentDemons; // 0x40'],
        'ProjectContext': ['public GameData gameData; // 0x20'],
        'Characters': ['public List<Character> characters; // 0x20','public List<CharacterData> UniquePool; // 0x40','public List<CharacterData> DuplicatesPool; // 0x48'],
        'ScriptInfo': ['public List<CharacterData> startingTownsfolks; // 0x10','public List<CharacterData> startingOutsiders; // 0x18',
            'public List<CharacterData> startingMinions; // 0x20','public List<CharacterData> startingDemons; // 0x28'],
        'GameData': ['public AscensionsData currentTemporaryAscension; // 0x78']}
    for name, fields in declarations.items():
        matches=[s for s in dump.split('\n') if s.startswith('public class '+name+' ')]
        assert len(matches)==1
        block=dump.split(matches[0],1)[1].split('\n}',1)[0]
        for field in fields: assert field in block,(name,field)
    pe = pefile.PE(data=raw, fast_load=True); base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    decoded = {}; exact = []; ends = {}
    for start, name in ENTRIES.items():
        method = 'Characters.<>c__DisplayClass22_0$$<PickRoundBluffs>b__0' if start == 0x377170 else ('Gameplay$$' if start in (0x37c3f0, 0x37c1a0, 0x37dc00) else 'AscensionsData$$' if start == 0x3b1e10 else 'Characters$$')+name
        rows = [r for r in meta['ScriptMethod'] if r['Address'] == start and r['Name'] == method]
        assert len(rows) == 1
        if start==0x377170: signature='bool Characters___c__DisplayClass22_0___PickRoundBluffs_b__0 (Characters___c__DisplayClass22_0_o* __this, CharacterData_o* cd, const MethodInfo* method);'
        elif start==0x36ce30: signature='void Characters__ManageCharacters (Characters_o* __this, System_Collections_Generic_List_CharacterData__o* charactersList, const MethodInfo* method);'
        elif start in (0x36d3a0,0x36d720): signature=f'void Characters__{name} (Characters_o* __this, const MethodInfo* method);'
        elif start==0x3b1e10: signature='CharacterData_array* AscensionsData__GetStartingtCharactersOfType (AscensionsData_o* __this, int32_t type, const MethodInfo* method);'
        elif method.startswith('Gameplay$$'): signature=f'System_Collections_Generic_List_CharacterData__o* Gameplay__{name} (Gameplay_o* __this, const MethodInfo* method);'
        else:
            extra=', int32_t type' if start==0x36b9c0 else ', int32_t alignment' if start==0x369eb0 else ''
            signature=f'System_Collections_Generic_List_CharacterData__o* Characters__{name} (Characters_o* __this, System_Collections_Generic_List_CharacterData__o* inpuCharacters{extra}, const MethodInfo* method);'
        assert rows[0]['Signature']==signature; exact.append(rows[0])
        end = min(r['Address'] for r in meta['ScriptMethod'] if r['Address'] > start)
        ins = list(cs.disasm(pe.get_data(start, end-start), start))
        while ins[-1].mnemonic == 'int3': ins.pop()
        assert ins[0].address == start and all(a.address+a.size == b.address for a, b in zip(ins, ins[1:]))
        ends[name] = hex(ins[-1].address+ins[-1].size); decoded.update({i.address:i for i in ins})
    gateways=[]
    for address,name,signature in [
        (0x36e4e0,'Characters$$UpdateCharacterPositions','void Characters__UpdateCharacterPositions (Characters_o* __this, const MethodInfo* method);'),
        (0x365a20,'Character$$Init','void Character__Init (Character_o* __this, CharacterData_o* character, int32_t id, const MethodInfo* method);')]:
        rows=[r for r in meta['ScriptMethod'] if r['Address']==address and r['Name']==name]
        assert len(rows)==1 and rows[0]['Signature']==signature; gateways.append(rows[0])
    assert ends == {'ManageCharacters':'0x36d396','PickRoundDuplicates':'0x36da3c',
                   'PickRoundBluffs':'0x36d71a', 'ContainsScriptCharacter':'0x3771c3',
                   'GetAscensionAllStartingCharacters':'0x37c59b', 'GetAllAscensionCharacters':'0x37c31f',
                   'GetScriptCharacters':'0x37dcc3', 'GetStartingtCharactersOfType':'0x3b2001',
                   'FilterBluffableCharacters':'0x36a6c4', 'FilterRealCharacterType':'0x36bb3d', 'FilterAlignmentCharacters':'0x36a02d'},ends
    checks = {0x37c48f:('call','0x3b1e10'), 0x37c4ea:('call','0x3b1e10'),
              0x37c530:('call','0x3b1e10'), 0x37c576:('call','0x3b1e10'),
              0x37c24b:('mov','rdx, qword ptr [rdx + 0x68]'), 0x37c292:('mov','rdx, qword ptr [rdx + 0x70]'),
              0x37c2cd:('mov','rdx, qword ptr [rdx + 0x78]'), 0x37c308:('mov','rdx, qword ptr [rdx + 0x68]'),
              0x36a64f:('cmp','byte ptr [rdx + 0x13c], 0'), 0x36bac1:('cmp','dword ptr [rdx + 0x130], esi'),
              0x369fb1:('cmp','dword ptr [rdx + 0x134], esi'), 0x3771b9:('jmp','0xb55950'),
              0x36cf00:('call','0x36e4e0'),0x36cf0a:('call','0x36d3a0'),0x36cf14:('call','0x36d720'),
              0x36cf7a:('mov','rax, qword ptr [r12 + 0x20]'),0x36cfda:('call','0x365a20'),
              0x36d7ca:('call','0x37dc00'),0x36d846:('call','0x369eb0'),0x36d8e0:('call','0x2b6ff0')}
    for a, expected in checks.items(): assert (decoded[a].mnemonic, decoded[a].op_str) == expected
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095); uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop = 0x200000000, 0x300000000, 0x400000000
    uc.mem_map(arena, 0x200000); uc.mem_map(stack, 0x10000); uc.mem_map(stop, 0x1000)
    def q(a,v): uc.mem_write(a,struct.pack('<Q',v))
    def d(a,v): uc.mem_write(a,struct.pack('<I',v&0xffffffff))
    def rq(a): return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def rd(a): return struct.unpack('<I',uc.mem_read(a,4))[0]
    def reg(r): return uc.reg_read(r)
    def ret(v=0):
        sp=reg(x.UC_X86_REG_RSP); uc.reg_write(x.UC_X86_REG_RAX,v); uc.reg_write(x.UC_X86_REG_RSP,sp+8); uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
    refs=set()
    for i in decoded.values():
        for op in i.operands:
            if op.type==capstone.CS_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP: refs.add(i.address+i.size+op.mem.disp)
    bindings={}
    for row in meta['ScriptMetadata']+meta['ScriptMetadataMethod']:
        if row['Address'] in refs:
            p=arena+0x1000+len(bindings)*0x200; bindings[row['Name']]=p; q(base+row['Address'],p)
    for name in ['Gameplay_TypeInfo','ProjectContext_TypeInfo','System.Collections.Generic.List<CharacterData>_TypeInfo',
                 'Characters.<>c__DisplayClass22_0_TypeInfo','System.Predicate<CharacterData>_TypeInfo',
                 'Method$Characters.<>c__DisplayClass22_0.<PickRoundBluffs>b__0()',
                 'Method$System.Collections.Generic.List<CharacterData>.Contains()',
                 'Method$System.Collections.Generic.List<CharacterData>.ToArray()',
                 'Method$System.Collections.Generic.List<Character>.GetEnumerator()',
                 'Method$System.Collections.Generic.List.Enumerator<Character>.MoveNext()',
                 'Method$System.Collections.Generic.List.Enumerator<Character>.Dispose()']: assert name in bindings, name
    math_ins=decoded[0x36cf8b]
    math_slot=base+math_ins.address+math_ins.size+math_ins.operands[1].mem.disp
    math_name=next(name for name,p in bindings.items() if p==rq(math_slot))
    assert math_name=='System.Math_TypeInfo'
    for i in decoded.values():
        if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:
            uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
    owner,game,gs,ps,project,settings,closure,predicate=[arena+n for n in range(0x10000,0x18000,0x1000)]
    profiles=[arena+0x18000,arena+0x19000]; scripts=[arena+0x1a000,arena+0x1b000]
    pool,duplicate,board,alternate,manage_roster=arena+0x20000,arena+0x24000,arena+0x28000,arena+0x2b000,arena+0x2e000
    actors={'character':arena+0x3a000,'other_character':arena+0x3c000}
    board_names={board:'board',alternate:'alternate'}
    data={i:arena+0x40000+i*0x200 for i in ASSETS}; labels={0:None}|{p:i for i,p in data.items()}
    q(bindings['Gameplay_TypeInfo']+0xb8,gs); q(bindings['ProjectContext_TypeInfo']+0xb8,ps)
    lists={}; arrays={}; state={}; opt={}; visited=set()
    def contents(p): return [rq(rq(p+0x10)+0x20+i*8) for i in range(rd(p+0x18))]
    def fill(p,values,version=0):
        q(p+0x10,p+0x1000); d(p+0x18,len(values)); d(p+0x1c,version); q(p+0x1018,128)
        for i,v in enumerate(values): q(p+0x1020+i*8,v)
    def append(p,v):
        n=rd(p+0x18); q(rq(p+0x10)+0x20+n*8,v); d(p+0x18,n+1); d(p+0x1c,rd(p+0x1c)+1)
    def snap():
        return {'lists':{name:{'items':[labels[v] for v in contents(p)],'version':rd(p+0x1c)} for p,name in lists.items()},
                'profile':profiles.index(rq(settings+0x78)) if rq(settings+0x78) in profiles else None,
                'project':bool(rq(ps)), 'settings':bool(rq(project+0x20)), 'game':bool(rq(gs+0x10)),
                'captured_script':lists.get(rq(closure+0x10)),
                'cache':[scripts.index(rq(p+0x60)) if rq(p+0x60) in scripts else None for p in profiles],
                'roster_sources':[lists.get(rq(game+0x28+i*8)) for i in range(4)],
                'classes':{name:bool(rd(bindings[name]+0xe0)) for name in ['Gameplay_TypeInfo',math_name]},
                'board':{'current':board_names.get(rq(owner+0x20)),
                         'items':{name:[next((k for k,v in actors.items() if v==p),None) for p in contents(ptr)] for ptr,name in board_names.items()}}}
    def emit(kind,**kw):
        state['counts'][kind]=state['counts'].get(kind,0)+1
        state['events'].append({'kind':kind,**kw,'snapshot':snap()})
        if opt.get('fail')==[kind,state['counts'][kind]]: state['error']=kind; uc.emu_stop(); return False
        return True
    def halt(error): state['error']=error; uc.emu_stop()
    def callback():
        key=[state['stage'],state['counts'].get('append_range',0)]
        for change in opt.get('callbacks',[]):
            if change['at']==key:
                if change['kind']=='profile': q(settings+0x78,profiles[change['value']] if change['value'] is not None else 0)
                elif change['kind']=='project': q(ps,0)
                elif change['kind']=='settings': q(project+0x20,0)
                elif change['kind']=='game': q(gs+0x10,0)
                elif change['kind']=='roster':
                    target=change['value']; values=change['items']; ptr=rq(game+0x28+target*8)
                    if values is None: q(game+0x28+target*8,0)
                    else: fill(ptr,[data[v] if v is not None else 0 for v in values],rd(ptr+0x1c)+1)
                elif change['kind']=='board': q(owner+0x20,alternate if change['value'] else 0)
                else: raise AssertionError(change)
                state['callbacks'].append(change)
    # RemoveAll is an explicit stable service. Each predicate invocation executes
    # the actual native closure body; commit is deferred until every result arrives.
    def predicate_step():
        pending=state['pending']
        if pending['index']==len(pending['before']):
            old=pending['before']; after=pending['after']; target=pending['target']
            fill(target,after,rd(target+0x1c)+int(len(old)!=len(after)))
            uc.reg_write(x.UC_X86_REG_RSP,pending['sp']); ret(len(old)-len(after)); state['pending']=None; return
        psp=pending['sp']-0x100; q(psp,stop+0x100)
        uc.reg_write(x.UC_X86_REG_RSP,psp); uc.reg_write(x.UC_X86_REG_RCX,closure)
        uc.reg_write(x.UC_X86_REG_RDX,pending['before'][pending['index']]); uc.reg_write(x.UC_X86_REG_R8,0)
        uc.reg_write(x.UC_X86_REG_RIP,base+0x377170)
    def hook(_,a,size,__):
        r=a-base; c,t,m=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
        if a==stop: uc.emu_stop(); return
        if r==0x36d01e: state['boundary']='empty_before_publication'; uc.emu_stop(); return
        if r==0x365a20:
            assert reg(x.UC_X86_REG_R9)==0
            state['boundary']='before_first_init'
            state['init_arguments']={'character':next(k for k,v in actors.items() if v==c),'data':labels[t],'display_id_bits':m&0xffffffff}
            uc.emu_stop(); return
        if a==stop+0x100:
            pending=state['pending']; accepted=reg(x.UC_X86_REG_RAX)&255
            state['predicate_results'].append(accepted)
            if not accepted: pending['after'].append(pending['before'][pending['index']])
            pending['index']+=1; predicate_step(); return
        if r in ENTRIES:
            if r==0x36d720: state['builder']='duplicates'
            elif r==0x36d3a0: state['builder']='unique'
            if r==0x3b1e10:
                assert c in profiles; state['typed'].append({'profile':profiles.index(c),'type':t})
            elif r not in (0x377170,0x36ce30):
                if state['builder']=='duplicates':
                    state['stage']={0x36d720:'duplicates',0x37dc00:'duplicate_script',0x36a550:'duplicate_bluffable',0x36b9c0:'duplicate_villagers' if m==10 else 'duplicate_outcasts',0x369eb0:'duplicate_discarded_good'}[r]
                else:
                    state['stage']=('fallback_' if state.get('fallback') else '')+('villagers' if m==10 else 'outcasts') if r==0x36b9c0 else {0x36d3a0:'unique',0x37c3f0:'starting',0x37c1a0:'fallback',0x37dc00:'script',0x36a550:'fallback_bluffable' if state.get('fallback') else 'bluffable',0x369eb0:'fallback_good'}[r]
                if r==0x37c1a0: state['fallback']=True
            state['entries'].append(ENTRIES[r])
        if r==0x36e4e0:
            assert c==owner and t==0
            if emit('positions'):
                if opt.get('replace_after_positions'): q(owner+0x20,alternate)
                ret()
        elif r==0x281d90:
            assert c in (bindings['Gameplay_TypeInfo'],bindings[math_name])
            if emit('class_init',class_name=next(n for n,p in bindings.items() if p==c)): d(c+0xe0,1); ret()
        elif r==0x2b7d40:
            kind='closure' if c==bindings['Characters.<>c__DisplayClass22_0_TypeInfo'] else 'predicate' if c==bindings['System.Predicate<CharacterData>_TypeInfo'] else state['stage']
            assert c in (bindings['Characters.<>c__DisplayClass22_0_TypeInfo'],bindings['System.Predicate<CharacterData>_TypeInfo'],bindings['System.Collections.Generic.List<CharacterData>_TypeInfo'])
            if emit('allocate',object=kind):
                if kind in ('closure','predicate'): ret(closure if kind=='closure' else predicate)
                else:
                    state['alloc']+=1; p=arena+0x100000+state['alloc']*0x3000; uc.mem_write(p,bytes(0x2000)); lists[p]=kind; q(p,c); fill(p,[]); ret(p)
        elif r==0xb02160:
            assert c in lists
            if emit('list_ctor',list=lists[c]): ret()
        elif r==0x33ed50:
            if emit('object_ctor' if c==closure else 'dispose'): ret()
        elif r==0xb53f50:
            assert c in lists
            if emit('append_range',destination=lists[c],source=lists.get(t,arrays.get(t))):
                if not t: halt('null_collection')
                else:
                    values=contents(t) if t in lists else [rq(t+0x20+i*8) for i in range(rq(t+0x18))]
                    assert t in lists or t in arrays
                    fill(c,contents(c)+values,rd(c+0x1c)+1); callback(); ret()
        elif r==0xb01f50:
            assert c in lists and t==bindings['Method$System.Collections.Generic.List<CharacterData>.ToArray()']
            if emit('to_array',source=lists[c]):
                p=arena+0x1e0000+state['counts']['to_array']*0x1000; arrays[p]=lists[c]+'.snapshot'; values=contents(c); q(p+0x18,len(values))
                for i,v in enumerate(values): q(p+0x20+i*8,v)
                ret(p)
        elif r==0x2b6ff0:
            assert rq(c)==t
            if c==closure+0x10:
                if emit('capture',source=lists.get(t)): ret()
            elif duplicate+0x1020<=c<duplicate+0x1420:
                if emit('duplicate_store',value=labels[t]): ret()
            else:
                assert c-0x60 in profiles
                if emit('cache_store',profile=profiles.index(c-0x60),script=scripts.index(t) if t in scripts else None): ret()
        elif r==0x112b9d0:
            assert c in (pool+0x1000,duplicate+0x1000) and t==0
            if emit('clear',list='unique' if c==pool+0x1000 else 'duplicates',count=m): uc.mem_write(c+0x20,bytes(m*8)); ret()
        elif r==0xc8b620:
            assert c==predicate and t==closure and m==bindings['Method$Characters.<>c__DisplayClass22_0.<PickRoundBluffs>b__0()']
            if emit('predicate_ctor'): ret()
        elif r==0xb59980:
            assert lists[c]=='starting' and t==predicate
            if emit('remove_all'):
                state['pending']={'target':c,'before':contents(c),'after':[],'index':0,'sp':reg(x.UC_X86_REG_RSP)}; predicate_step()
        elif r==0xb55950:
            assert c==rq(closure+0x10) and m==bindings['Method$System.Collections.Generic.List<CharacterData>.Contains()']
            if emit('contains',source=lists[c],value=labels[t]): ret(int(t in contents(c)))
        elif r==0xb16640:
            assert t in lists or t in board_names
            if t in board_names: assert m==bindings['Method$System.Collections.Generic.List<Character>.GetEnumerator()']
            if emit('enumerator',source=lists.get(t,board_names.get(t))): uc.mem_write(c,bytes(24)); q(c,t); d(c+8,0); ret(c)
        elif r==0x9693d0:
            source=rq(c); index=rd(c+8)
            if source in board_names: assert t==bindings['Method$System.Collections.Generic.List.Enumerator<Character>.MoveNext()']
            if emit('move_next',source=lists.get(source,board_names.get(source))):
                values=contents(source); d(c+8,index+1); q(c+0x10,values[index] if index<len(values) else 0); ret(int(index<len(values)))
                if source in board_names and opt.get('replace_on_move'): q(owner+0x20,alternate)
                if source in board_names and opt.get('null_after_move'): q(owner+0x20,0)
        elif r==0x2eb0:
            assert c in lists
            if emit('pool_add' if c in (pool,duplicate) else 'filter_add',list=lists[c],value=labels[t]): append(c,t); ret()
        elif r==0x1c86600:
            assert c==0 and m==0
            caller=rq(reg(x.UC_X86_REG_RSP))-base
            source='inline' if caller==0x3b1e6c else 'custom' if caller==0x3b1eb7 else state['builder']
            index=opt.get('choices',[])[len(state['draws'])] if len(state['draws'])<len(opt.get('choices',[])) else 0
            state['draws'].append({'source':source,'width':t,'index':index})
            if emit('rng',source=source,width=t,index=index): ret(index)
        elif r==0xb22150:
            if c==manage_roster:
                if not emit('get_item',index=t): return
            if t>=rd(c+0x18): halt('bounds')
            else: ret(contents(c)[t])
        elif r==0xb59e70:
            assert lists[c] in ('villagers','outcasts','duplicate_villagers','duplicate_outcasts')
            if emit('remove',list=lists[c],value=labels[t]):
                values=contents(c); values.remove(t); fill(c,values,rd(c+0x1c)+1); ret(1)
        elif r in (0x2b7d90,0x2b7d80): halt('null' if r==0x2b7d90 else 'bounds')
        else:
            assert r in decoded and decoded[r].size==size,hex(r); visited.add(r)
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    def run(starting,rosters,fallback,options=None):
        opt.clear(); opt.update(options or {}); state.clear()
        state.update(counts={},events=[],entries=[],typed=[],draws=[],callbacks=[],predicate_results=[],pending=None,alloc=0,error=None,fallback=False,builder=None,boundary=None,init_arguments=None)
        lists.clear(); arrays.clear(); lists[pool]='unique'; fill(pool,[data[7]],3); lists[duplicate]='duplicates'; fill(duplicate,[data[6]],9); q(closure+0x10,0)
        q(owner+0x40,0 if opt.get('null_pool') else pool); q(gs+0x10,0 if opt.get('null_game') else game)
        q(owner+0x48,0 if opt.get('null_duplicate_pool') else duplicate)
        fill(board,[actors[v] if v is not None else 0 for v in opt.get('board',['character','other_character'])])
        fill(alternate,[actors[v] if v is not None else 0 for v in opt.get('alternate_board',['other_character']*3)])
        q(owner+0x20,0 if opt.get('null_board') else board)
        manage_values=opt.get('manage_roster',[7,1,2]); lists[manage_roster]='manage_roster'; fill(manage_roster,[data[v] if v is not None else 0 for v in manage_values or []])
        q(ps,project); q(project+0x20,settings); q(settings+0x78,profiles[0]); d(bindings['Gameplay_TypeInfo']+0xe0,0 if opt.get('cold') else 1)
        d(bindings[math_name]+0xe0,0 if opt.get('cold') else 1)
        for ident,(typ,alignment,bluffable) in ASSETS.items():
            d(data[ident]+0x130,typ); d(data[ident]+0x134,alignment); uc.mem_write(data[ident]+0x13c,bytes([bluffable]))
        serial=0
        def make_list(name,values):
            nonlocal serial
            if values is None: return 0
            p=arena+0x50000+serial*0x3000; serial+=1; lists[p]=name; fill(p,[0 if v is None else data[v] for v in values]); return p
        roster_ptrs=[make_list(f'roster{i}',values) for i,values in enumerate(rosters)]
        for i,target in enumerate(opt.get('roster_aliases',[0,1,2,3])): q(game+0x28+i*8,roster_ptrs[target])
        for pi,p in enumerate(profiles):
            starts=opt.get('alternate_starting',starting) if pi else starting
            falls=opt.get('alternate_fallback',fallback) if pi else fallback
            for i in range(4):
                collection=make_list(f'profile{pi}.starting{i}',starts[i]); arr=collection+0x1000 if collection else 0
                if arr: arrays[arr]=f'profile{pi}.starting{i}'; q(arr+0x18,len(starts[i]))
                q(p+0x40+i*8,arr)
                collection=make_list(f'profile{pi}.all{i}',falls[i]); arr=collection+0x1000 if collection else 0
                if arr: arrays[arr]=f'profile{pi}.all{i}'; q(arr+0x18,len(falls[i]))
                q(p+0x68+i*8,arr)
                q(scripts[pi]+0x10+i*8,make_list(f'script{pi}.{i}',starts[i]))
            inline=arena+0xf0000+pi*0x1000; custom=inline+0x200
            ids=opt.get('inline',[]); q(inline+0x18,len(ids))
            for i,v in enumerate(ids): q(inline+0x20+i*8,scripts[v] if v is not None else 0)
            q(custom+0x18,0); q(p+0x20,inline); q(p+0x18,custom); q(p+0x60,scripts[pi] if opt.get('cached') else 0)
        initial=snap(); sp=stack+0x8008; q(sp,stop); uc.reg_write(x.UC_X86_REG_RSP,sp)
        uc.reg_write(x.UC_X86_REG_RCX,owner); uc.reg_write(x.UC_X86_REG_RDX,manage_roster if manage_values is not None else 0)
        keep=[x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
        for rr in keep: uc.reg_write(rr,0xabc000+rr)
        uc.emu_start(base+0x36ce30,stop,count=50000)
        assert state['error'] or state['boundary']
        return {'input':{'starting':starting,'rosters':rosters,'fallback':fallback,'options':dict(opt)},'initial':initial,'final':snap(),
                **{k:state[k][:] for k in ['entries','typed','events','draws','callbacks','predicate_results']},'error':state['error'],
                'boundary':state['boundary'],'init_arguments':state['init_arguments']}
    cases=[]; starts=[[0,0,1,4],[2,2,5],[3],[6]]; empty=[[],[],[],[]]
    rosters=[[1,7],[5],[3],[6]]; fallback=[[1,7],[5],[3],[6]]
    def support(start,roster,options,family):
        pending=[([],Fraction(1))]; result_cases=[]
        while pending:
            choices,probability=pending.pop()
            result=run(start,roster,fallback,{**options,'choices':choices})
            if len(result['draws'])>len(choices) and result['draws'][len(choices)]['width']:
                width=result['draws'][len(choices)]['width']
                pending.extend((choices+[index],probability/width) for index in range(width))
            else:
                assert result['error'] is None,(family,result['error'])
                assert result['boundary']=='before_first_init'
                result['probability']=str(probability); result['support_family']=family; result_cases.append(result)
        assert sum(Fraction(c['probability']) for c in result_cases)==1
        cases.extend(result_cases); return result_cases
    mixed=support(starts,rosters,{},'mixed')
    assert len(mixed)==8
    for case in mixed:
        assert case['final']['lists']['unique']['items']==[0,0,2]
        assert sorted(case['final']['lists']['duplicates']['items'])==[1,5,7]
        assert [d['source'] for d in case['draws']]==['unique']*3+['duplicates']*3
        assert case['entries'].count('GetScriptCharacters')==2
        assert case['final']['lists']['script']['items']==case['final']['lists']['duplicate_script']['items']==[1,7,5,3,6]
        assert case['init_arguments']=={'character':'character','data':7,'display_id_bits':2}
    support(empty,[[1],[5],[],[]],{},'fallback')
    support([[0],[2],[],[]],[[1,7],[5],[],[]],{'roster_aliases':[0,0,2,3]},'aliased_rosters')
    for inline in [[0,1],[None,1]]:
        support(starts,rosters,{'inline':inline,'alternate_starting':[[7],[5],[],[]]},'inline_'+str(inline))
    # Exact service failure prefixes span both builders and subsequent board access.
    for start,options in [(starts,{'cold':True}),(empty,{'cold':True}),(starts,{'cached':True}),(starts,{'inline':[0,1]})]:
        baseline=run(start,rosters,fallback,options); assert baseline['error'] is None; cases.append(baseline); counts={}
        for index,event in enumerate(baseline['events']):
            kind=event['kind']; counts[kind]=counts.get(kind,0)+1
            result=run(start,rosters,fallback,{**options,'fail':[kind,counts[kind]]})
            assert result['error']==kind and result['events']==baseline['events'][:index+1] and result['final']==event['snapshot']; cases.append(result)
    for options,error in [({'null_pool':True},'null'),({'null_duplicate_pool':True},'null'),({'null_game':True},'null'),
        ({'null_board':True},'null'),({'manage_roster':None},'null'),({'manage_roster':[]},'bounds'),
        ({'board':[None]},'null'),({'board':[None],'manage_roster':[]},'bounds'),({'null_after_move':True},'null')]:
        result=run(starts,rosters,fallback,options); assert result['error']==error; cases.append(result)
        if options.get('null_pool') or options.get('null_game'): assert 'PickRoundDuplicates' not in result['entries']
    for options,expected in [({'replace_after_positions':True},{'character':'other_character','data':7,'display_id_bits':3}),
        ({'replace_on_move':True},{'character':'character','data':7,'display_id_bits':3}),
        ({'manage_roster':[None]},{'character':'character','data':None,'display_id_bits':2})]:
        result=run(starts,rosters,fallback,options); assert result['error'] is None and result['init_arguments']==expected; cases.append(result)
    result=run(starts,rosters,fallback,{'board':[],'manage_roster':None}); assert result['error'] is None and result['boundary']=='empty_before_publication'; cases.append(result)
    # Mutating a current roster after the unique script snapshot must affect the
    # later duplicate snapshot without rewriting captured membership history.
    result=run(starts,rosters,fallback,{'callbacks':[{'at':['script',8],'kind':'roster','value':0,'items':[7,7]}]})
    assert result['error'] is None
    assert result['final']['lists']['script']['items']==[1,7,5,3,6]
    assert result['final']['lists']['duplicate_script']['items']==[7,7,5,3,6]
    assert result['final']['lists']['duplicates']['items']==[7,7,5]; cases.append(result)
    # First script getters are complete before these callbacks return. A vanished
    # singleton can still finish unique sampling but the duplicate entry fails.
    for stage,ordinal in [('starting',4),('script',8),('duplicate_script',12)]:
        result=run(starts,rosters,fallback,{'callbacks':[{'at':[stage,ordinal],'kind':'game','value':None}]})
        expected='null' if stage!='duplicate_script' else None
        assert result['error']==expected; cases.append(result)
        assert result['final']['lists']['duplicates']['items']==([6] if expected else [1,7,5])
    # A null mutation after unique capture fails only the later script copy and
    # preserves its old pool; a physical-null occurrence fails after pool clear.
    for values,error,expected in [(None,'null_collection',[6]),([None],'null',[])]:
        result=run(starts,rosters,fallback,{'callbacks':[{'at':['script',8],'kind':'roster','value':0,'items':values}]})
        assert result['error']==error and result['final']['lists']['duplicates']['items']==expected; cases.append(result)
    result=run(starts,empty,fallback); assert result['error']=='bounds' and result['draws'][-1]=={'source':'duplicates','width':0,'index':0}; cases.append(result)
    for n in range(1,5):
        result=run(starts,rosters,fallback,{'callbacks':[{'at':['starting',n],'kind':'profile','value':1}],'alternate_starting':[[7],[5],[],[]]})
        assert result['typed']==[{'profile':int(i>=n),'type':t} for i,t in enumerate([100,20,30,10])]
        assert result['error'] is None; cases.append(result)
    for i in range(4):
        bad=[v[:] for v in starts]; bad[i]=None; result=run(bad,rosters,fallback)
        assert result['error']=='null_collection' and 'PickRoundDuplicates' not in result['entries']; cases.append(result)
    snapshots=[]; indices={}
    for case in cases:
        for event in case['events']:
            snapshot=event.pop('snapshot'); key=json.dumps(snapshot,sort_keys=True)
            if key not in indices: indices[key]=len(snapshots); snapshots.append(snapshot)
            event['snapshot_index']=indices[key]
    return {'build_id':BUILD,'cases_passed':len(cases),'metadata_verified':exact,'verified_gateways':gateways,'entry_ends':ends,'native_assertions':len(checks),
            'native_instructions_executed':len(visited),'asset_fields':ASSETS,'field_declarations':declarations,'snapshot_table':snapshots,'cases':cases,
            'scope':'Actual ManageCharacters prefix invokes both actual pool builders, exact source getters, lazy typed getter, native filters and actual predicate callbacks. Stops before first Character.Init or empty-board publication. Positions, allocation/list/Contains/RNG/GC/enumeration/class initialization remain explicit services. RemoveAll defers its commit. Distinct pool and returned-list identities; optional aliased roster providers and controlled post-AddRange mutations. Warm metadata; no managed unwind, live scene/layout implementation or Unity PRNG claim.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('game_root',type=Path); p.add_argument('dumper_root',type=Path); p.add_argument('--output',required=True,type=Path)
    args=p.parse_args(); report=audit(args.game_root,args.dumper_root); args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8'); print(report['cases_passed'],report['native_instructions_executed'])
