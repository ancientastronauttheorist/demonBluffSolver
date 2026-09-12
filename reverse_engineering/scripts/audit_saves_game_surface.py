"""Audit remaining SavesGame JSON properties and character preference replacement."""
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
    manifest=json.loads((root/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction=json.loads((root/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(path,digest):
        b=path.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==digest.upper();return b
    raw=pinned(Path(game_root)/'GameAssembly.dll',manifest['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    properties={'AdvancedMode':(0x3877D0,0x387DC0,'SavedAdvanced','AdvancedMode'),
                'RoguelikeMode':(0x387980,0x387E60,'SavedRoguelike','RoguelikeMode'),
                'CharacterPreferences':(0x387870,0x387E10,'SAVED_CHARACTERS','SavedCharacters'),
                'UnlockedAchievements':(0x387BA0,0x387FB0,'SAVED_ACHIEVEMENTS','SavedAchievements'),
                'UnlockedSkins':(0x387CB0,0x388000,'SAVED_SKINS','SavedSkins')}
    selected={f'SavesGame$${prefix}_{name}':a for name,(get,set_,key,typ) in properties.items() for prefix,a in [('get',get),('set',set_)]}
    selected.update({'SavesGame$$UpdateCharacterPreference':0x3874B0,'SavesGame$$GetCharacterPreference':0x33ED50,
                     'SavesGame.<>c__DisplayClass15_0$$<UpdateCharacterPreference>b__0':0x393420,
                     'SavedCharacters$$.ctor':0x3873B0,'SavedSkins$$.ctor':0x387430,'SavedAchievements$$.ctor':0x387330})
    exact=[]
    for name,address in selected.items():
        rows=[r for r in script['ScriptMethod'] if r['Name']==name and r['Address']==address]
        assert len(rows)==1;exact.append(rows[0])
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    spans=[(0x387330,0x3873A6),(0x3873B0,0x387426),(0x387430,0x3874A6),(0x3874B0,0x3877C9),
           (0x3877D0,0x387870),(0x387870,0x387973),(0x387980,0x387A20),(0x387BA0,0x387CA3),
           (0x387CB0,0x387DB3),(0x387DC0,0x387E06),(0x387E10,0x387E56),(0x387E60,0x387EA6),
           (0x387FB0,0x387FF6),(0x388000,0x388046),(0x393420,0x39344F),(0xB59980,0xB59AF5),
           (0x357920,0x357927),(0x33ED50,0x33ED53)]
    ins={}
    for a,b in spans:
        rows=list(cs.disasm(pe.get_data(a,b-a),a));assert rows[-1].address+rows[-1].size==b
        ins.update({i.address:i for i in rows})
    checks=[(0x387689,'call','0xb59980'),(0x3876B4,'mov','rdx, qword ptr [rdx + 0x18]'),
            (0x3876DA,'mov','rdx, qword ptr [rdx + 0xc0]'),(0x3876E6,'mov','rdx, qword ptr [rdx + 0x18]'),
            (0x387716,'inc','dword ptr [rcx + 0x1c]'),(0x38774D,'mov','dword ptr [rcx + 0x18], eax'),
            (0x38778D,'call','0x1cd6420'),(0x3877B8,'jmp','0x1c86170'),
            (0x393444,'jmp','0xf73e00'),(0xB59AB3,'mov','qword ptr [rcx], rdx'),
            (0xB59AD4,'call','0x112b9d0'),(0xB59AE1,'mov','dword ptr [rdi + 0x18], ebp'),
            (0x357922,'jmp','0x33ed50'),(0x33ED50,'ret','0')]
    for a,m,o in checks:assert (ins[a].mnemonic,ins[a].op_str)==(m,o)
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64)
    uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,stop=0x300000000,0x400000000,0x500000000
    uc.mem_map(arena,0x200000);uc.mem_map(stack,0x20000);uc.mem_map(stop,0x1000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def d(a,v):uc.mem_write(a,struct.pack('<I',v))
    def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
    def reg(r):return uc.reg_read(r)
    def ret(v=0):
        rsp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,rsp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(rsp))
    slots={r['Address']:('metadata',r['Name']) for r in script['ScriptMetadata']}
    slots.update({r['Address']:('method',r['Name']) for r in script['ScriptMetadataMethod']})
    slots.update({r['Address']:('string',r['Value']) for r in script['ScriptString']})
    used=set()
    for i in ins.values():
        for op in i.operands:
            if op.type==capstone.x86.X86_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:
                target=i.address+i.size+op.mem.disp
                if target in slots:used.add(target)
                elif i.mnemonic=='cmp' and op.size==1:uc.mem_write(base+target,b'\x01')
    names={};string_values={};ptr_names={}
    for j,a in enumerate(sorted(used)):
        kind,name=slots[a];pointer=arena+0x1000+j*0x200;q(base+a,pointer);ptr_names[pointer]=name
        if kind=='string':string_values[pointer]=name
        else:names[name]=pointer
    required=['SavedCharacters_TypeInfo','SavedSkins_TypeInfo','SavedAchievements_TypeInfo','CharacterPreference_TypeInfo',
              'RoguelikeMode_TypeInfo','AdvancedMode_TypeInfo','System.Collections.Generic.List<CharacterPreference>_TypeInfo',
              'System.Collections.Generic.List<string>_TypeInfo','SavesGame.<>c__DisplayClass15_0_TypeInfo',
              'System.Predicate<CharacterPreference>_TypeInfo']
    assert set(required)<=set(names)
    add=names['Method$System.Collections.Generic.List<CharacterPreference>.Add()']
    q(add+0x20,arena+0x30000);q(arena+0x300C0,arena+0x30200);q(arena+0x30270,arena+0x30400)
    saved,listptr,array,cd,skin,firstjson,secondjson,parsed,serialized=[arena+n for n in range(0x40000,0x49000,0x1000)]
    string_values.update({firstjson:'first',secondjson:'second',serialized:'serialized'})
    custom={}
    def text_pointer(value):
        if value is None:return 0
        if value not in custom:
            pointer=arena+0x50000+len(custom)*0x100;custom[value]=pointer;string_values[pointer]=value
        return custom[value]
    state={}
    def allocate(kind):
        state['allocations'].append(kind);p=arena+0x80000+len(state['allocations'])*0x400
        uc.mem_write(p,bytes(0x300));return p
    def list_values(p):return [rq(rq(p+0x10)+0x20+i*8) for i in range(rd(p+0x18))]
    def hook(_,address,size,__):
        a=address-base;rcx,rdx,r8,r9=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        if a==0x2B7D40:
            assert rcx in ptr_names;ret(allocate(ptr_names[rcx]))
        elif a==0xB02160:
            state['lists'].append(rcx);p=allocate('backing array');q(rcx+0x10,p);q(p+0x18,0);ret()
        elif a==0x2B6FF0:assert rq(rcx)==rdx;ret()
        elif a==0x1C85F20:
            assert string_values[rcx]==state['key'] and rdx==0
            state['reads']+=1;ret(state['read_values'][min(state['reads']-1,len(state['read_values'])-1)])
        elif a==0xF76390:
            assert rdx==0;ret(rcx==0 or string_values[rcx]=='')
        elif a==0x645DA0:
            assert ptr_names[rdx]==f"Method$UnityEngine.JsonUtility.FromJson<{state['type']}>()"
            state['json_input']=rcx
            if state['failure']=='fromjson':state['error']='fromjson';uc.emu_stop()
            else:ret(state['parsed'])
        elif a==0x1CD6420:
            assert rdx==0;state['serialized_object']=rcx
            if state['failure']=='tojson':state['error']='tojson';uc.emu_stop()
            else:ret(serialized)
        elif a==0x1C86170:
            assert string_values[rcx]==state['key'] and rdx==serialized and r8==0
            if state['failure']=='setstring':state['error']='setstring';uc.emu_stop()
            else:state['writes']+=1;ret()
        elif a==0xC8B620:
            assert ptr_names[r8]=='Method$SavesGame.<>c__DisplayClass15_0.<UpdateCharacterPreference>b__0()' and r9==0
            q(rcx+0x18,base+0x393420);q(rcx+0x28,r8);q(rcx+0x40,rdx);ret()
        elif a==0xF73E00:
            assert r8==0;state['comparisons'].append((string_values.get(rcx),string_values.get(rdx)))
            ret(string_values.get(rcx)==string_values.get(rdx))
        elif a==0x112B9D0:
            assert r9==0
            for i in range(rdx,rdx+r8):q(rcx+0x20+i*8,0)
            ret()
        elif a==0xB54090:
            assert r8==arena+0x30400
            p=allocate('grown array');values=list_values(rcx);q(p+0x18,len(values)+4)
            for i,v in enumerate(values+[rdx]):q(p+0x20+i*8,v)
            q(rcx+0x10,p);d(rcx+0x18,len(values)+1);state['growth']+=1;ret()
        elif a in (0x2B7D90,0x2B7D80):state['error']='null' if a==0x2B7D90 else 'index';uc.emu_stop()
        elif a not in ins:raise AssertionError(f'left SavesGame boundary {a:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    regs=[x.UC_X86_REG_RBX,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_RBP,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
    def run(entry,arg=0,arg2=0,**options):
        state.clear();state.update(allocations=[],lists=[],reads=0,writes=0,comparisons=[],growth=0,failure=None)
        state.update(options);rsp=stack+0x18008;q(rsp,stop)
        for i,r in enumerate(regs):uc.reg_write(r,0xBCDE0000+i)
        for r,v in [(x.UC_X86_REG_RSP,rsp),(x.UC_X86_REG_RCX,arg),(x.UC_X86_REG_RDX,arg2),(x.UC_X86_REG_R8,0)]:uc.reg_write(r,v)
        uc.emu_start(base+entry,stop,timeout=1_000_000,count=20000)
        if 'error' not in state:
            assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==rsp+8
            for i,r in enumerate(regs):assert reg(r)==0xBCDE0000+i
        return reg(x.UC_X86_REG_RAX)
    property_cases=[]
    for prop,(get,set_,key,typ) in properties.items():
        for mode,values,result,failure in [('missing',[0],parsed,None),('empty',[text_pointer('')],parsed,None),
                                          ('present',[firstjson,secondjson],parsed,None),('changed read',[firstjson,0],0,None),
                                          ('null parse',[firstjson],0,None),('parse failure',[firstjson],0,'fromjson')]:
            value=run(get,key=key,type=typ,read_values=values,parsed=result,failure=failure)
            if mode in ('missing','empty'):
                assert state['reads']==1 and typ+'_TypeInfo' in state['allocations']
                if typ.startswith('Saved'):assert rq(value+0x10) in state['lists'] and list_values(rq(value+0x10))==[]
                else:assert bytes(uc.mem_read(value+16,0x100))==bytes(0x100)
            else:
                assert state['reads']==2 and state['json_input']==values[-1] and not state['allocations']
                if failure:assert state['error']==failure
                else:assert value==result
            assert state['writes']==0
            property_cases.append({'property':prop,'get_case':mode,'reads':state['reads'],'error':state.get('error')})
        for value,failure in itertools.product((0,parsed),(None,'tojson','setstring')):
            run(set_,value,key=key,type=typ,failure=failure)
            assert state['serialized_object']==value and state['writes']==int(failure is None)
            assert state.get('error')==failure
            property_cases.append({'property':prop,'setter_null':value==0,'error':failure})
    update_cases=[]
    for labels,capacity,skin_mode,failure in itertools.product(
            ([],['match'],['other','match','match','other'],['match','other',None]),(0,8),('selected','no skin','null skin id'),(None,'tojson','setstring')):
        q(saved+0x10,listptr);q(listptr+0x10,array);d(listptr+0x18,len(labels));d(listptr+0x1C,0);q(array+0x18,max(len(labels),capacity))
        pointers=[]
        for i,label in enumerate(labels):
            p=0 if label is None else arena+0x60000+i*0x100
            if p:q(p+0x10,text_pointer('id' if label=='match' else 'other'));q(p+0x18,text_pointer('old'))
            q(array+0x20+i*8,p);pointers.append(p)
        q(cd+0x18,text_pointer('id'));q(cd+0xC0,0 if skin_mode=='no skin' else skin)
        q(skin+0x18,0 if skin_mode=='null skin id' else text_pointer('skin'))
        run(0x3874B0,cd,key='SAVED_CHARACTERS',type='SavedCharacters',read_values=[firstjson],parsed=saved,failure=failure)
        actual=list_values(listptr)
        if None in labels:
            assert state['error']=='null' and state['writes']==0 and rd(listptr+0x18)==len(labels)
            assert actual==[pointers[1],pointers[1],0] and rd(listptr+0x1C)==0
        else:
            assert state.get('error')==failure
            survivors=[p for label,p in zip(labels,pointers) if label!='match']
            assert actual[:-1]==survivors and len(actual)==len(survivors)+1
            assert rq(actual[-1]+0x10)==text_pointer('id')
            assert string_values[rq(actual[-1]+0x18)]==('skin' if skin_mode=='selected' else '')
            assert state['serialized_object']==saved and state['writes']==int(failure is None)
            assert rd(listptr+0x1C)==1+int('match' in labels)
        update_cases.append({'input_labels':labels,'capacity':max(len(labels),capacity),'skin':skin_mode,'requested_failure':failure,
                             'error':state.get('error'),'result_count':len(actual),'version':rd(listptr+0x1C),'writes':state['writes'],'growth':state['growth']})
    # Null source character fails after reads/removal setup; no saved write occurs.
    q(saved+0x10,listptr);q(listptr+0x10,array);d(listptr+0x18,0);q(array+0x18,8)
    run(0x3874B0,0,key='SAVED_CHARACTERS',type='SavedCharacters',read_values=[firstjson],parsed=saved)
    assert state['error']=='null' and state['writes']==0
    update_cases.append({'null_character':True,'error':'null','writes':0})
    q(cd+0x18,text_pointer('id'));q(cd+0xC0,skin);q(skin+0x18,text_pointer('skin'))
    for broken in ('null parsed','null prefs','null backing array'):
        q(saved+0x10,0 if broken=='null prefs' else listptr)
        q(listptr+0x10,0 if broken=='null backing array' else array);d(listptr+0x18,0);d(listptr+0x1C,0)
        run(0x3874B0,cd,key='SAVED_CHARACTERS',type='SavedCharacters',read_values=[firstjson],parsed=0 if broken=='null parsed' else saved)
        assert state['error']=='null' and state['writes']==0
        if broken=='null backing array':assert rd(listptr+0x1C)==1
        update_cases.append({'broken_input':broken,'error':'null','writes':0})
    run(0x3874B0,cd,key='SAVED_CHARACTERS',type='SavedCharacters',read_values=[0],parsed=0)
    assert state['reads']==1 and state['writes']==1 and state['growth']==1
    fresh=state['serialized_object'];values=list_values(rq(fresh+0x10));assert len(values)==1 and rq(values[0]+0x10)==text_pointer('id')
    update_cases.append({'missing_store':True,'writes':1,'fresh_preference_count':1})
    predicate_cases=[];closure=arena+0x70000;pref=arena+0x70100
    for stored_id,character_id in itertools.product((None,'id','other'),repeat=2):
        q(closure+0x10,cd);q(pref+0x10,text_pointer(stored_id));q(cd+0x18,text_pointer(character_id))
        value=run(0x393420,closure,pref);assert bool(value)==(stored_id==character_id)
        predicate_cases.append({'stored_id':stored_id,'character_id':character_id,'matches':bool(value)})
    constructor_cases=[]
    for typ,address in [('SavedAchievements',0x387330),('SavedCharacters',0x3873B0),('SavedSkins',0x387430)]:
        run(address,saved);assert rq(saved+0x10) in state['lists'] and list_values(rq(saved+0x10))==[]
        constructor_cases.append({'type':typ,'empty_fresh_list':True})
    for arg in (0,cd):run(0x33ED50,arg);assert not state['allocations'] and not state['writes']
    return {'schema_version':1,'build_id':BUILD,'game_assembly_sha256':manifest['inputs']['game_assembly']['sha256'],
            'exact_methods':exact,'native_checks':len(checks),'property_cases':property_cases,'update_cases':update_cases,'constructor_cases':constructor_cases,'predicate_cases':predicate_cases,
            'native_case_count':len(property_cases)+len(update_cases)+len(constructor_cases)+len(predicate_cases)+2,
            'keys':{name:data[2] for name,data in properties.items()},
            'scope':'Native SavesGame properties/update, nested predicate, List.RemoveAll, inline append and default constructors. JSON, PlayerPrefs, string equality, allocation, delegate setup, array clearing and list capacity growth are explicit gateways. No persistence flush, live saves, JSON internals or exception unwinding is executed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('game_root',type=Path);parser.add_argument('--dumper-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.game_root,args.dumper_root);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {report['native_case_count']} SavesGame surface cases")
