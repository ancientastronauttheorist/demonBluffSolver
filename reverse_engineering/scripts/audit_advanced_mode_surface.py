"""Native AdvancedMode declaration audit; external persistence/UI/delegate services are explicit."""
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
    extraction=json.loads((root/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(path,digest):
        raw=Path(path).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
    raw=pinned(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump=pinned(Path(dumper_root)/'dump.cs',extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    decl=dump.split('public class AdvancedMode : GameMode // TypeDefIndex: 5935',1)[1].split('// Namespace:',1)[0]
    fields=['currentScore','bestScore','roundScore','currentSavedVillage','currentSavedVillageHighscoreTracker','allSavedVillagesEver','bestOverallSavedVillage']
    for n,name in enumerate(fields):assert f'public int {name}; // 0x{16+4*n:X}' in decl
    selected={r['Name'].split('$$')[1]:r for r in script['ScriptMethod'] if r['Name'].startswith('AdvancedMode$$') and r['Name'].split('$$')[1] not in ('GetCurrentAscension','GetPreviousAscension')}
    expected_entries={'GetGameMode':0x3712B0,'Init':0x3D1960,'LoadGame':0x3D1D70,'OnLoadGame':0x3D1E40,'DeInit':0x3D1420,'GetMaxLevel':0x3712B0,'GetCurrentLevel':0x3712B0,'GetScore':0x3712B0,'OnStageCompleted':0x3D1E60,'CheckAchievements':0x3D13D0,'OnCharacterKilled':0x3D1D80,'GetSummaryScores':0x3D18F0,'OnFailed':0x3D1E20,'GetResetLevel':0x379600,'CanResetLevel':0x3BCC90,'GetStartingLevel':0x379600,'UpdateScore':0x33ED50,'IsLocked':0x3D1CF0,'Save':0x3D1EE0,'GetScores':0x3D17E0,'.ctor':0x357920}
    assert {n:r['Address'] for n,r in selected.items()}==expected_entries
    rogue_decl=dump.split('public class RoguelikeStandard : GameMode // TypeDefIndex: 5936',1)[1].split('// Namespace:',1)[0]
    assert 'public int bestAscension; // 0x10' in rogue_decl
    assert 'public static Action OnUIUpdate; // 0x0' in dump
    assert 'public const ECharacterType Minion = 30;' in dump and 'public const ECharacterType Demon = 100;' in dump
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    spans=[(0x3D13D0,0x3D1414),(0x3D1420,0x3D1734),(0x3D17E0,0x3D18E1),(0x3D18F0,0x3D1951),(0x3D1960,0x3D1CE4),
           (0x3D1CF0,0x3D1D62),(0x3D1D70,0x3D1D77),(0x3D1D80,0x3D1E13),(0x3D1E20,0x3D1E33),(0x3D1E40,0x3D1E56),
           (0x3D1E60,0x3D1ED9),(0x3D1EE0,0x3D1EE7),(0x3712B0,0x3712B3),(0x379600,0x379604),(0x3BCC90,0x3BCC93),(0x33ED50,0x33ED53),(0x357920,0x357927)]
    ins={}
    for a,b in spans:
        decoded=list(cs.disasm(pe.get_data(a,b-a),a));assert decoded and decoded[-1].address+decoded[-1].size==b
        ins.update({i.address:i for i in decoded})
    assert all(r['Address'] in ins for r in selected.values())
    checks=[(0x3712B0,'xor','eax, eax'),(0x379600,'mov','eax, dword ptr [rcx + 0x1c]'),(0x3BCC90,'xor','al, al'),
            (0x33ED50,'ret','0'),(0x357922,'jmp','0x33ed50'),(0x3D1D72,'jmp','0x3877d0'),(0x3D1EE2,'jmp','0x387dc0')]
    for a,m,o in checks:assert (ins[a].mnemonic,ins[a].op_str)==(m,o)
    for a,b,expected in [(0x3D1420,0x3D1734,['0x116bcc0','0x116e070','0x116e070']),(0x3D1960,0x3D1CE4,['0x116bcc0']*3)]:
        assert [i.op_str for i in ins.values() if a<=i.address<b and i.mnemonic=='call' and i.op_str in ('0x116bcc0','0x116e070')]==expected
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,stop=0x300000000,0x400000000,0x500000000
    uc.mem_map(arena,0x200000);uc.mem_map(stack,0x20000);uc.mem_map(stop,0x1000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
    def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def rd(a):return struct.unpack('<i',uc.mem_read(a,4))[0]
    def signed(v):return (v+2**31)%2**32-2**31
    def reg(r):return uc.reg_read(r)
    def ret(v=0):
        sp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,sp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(sp))
    slots={r['Address']:('metadata',r['Name']) for r in script['ScriptMetadata']}
    slots.update({r['Address']:('method',r['Name']) for r in script['ScriptMetadataMethod']})
    slots.update({r['Address']:('string',r['Value']) for r in script['ScriptString']})
    used=set()
    for i in ins.values():
        for op in i.operands:
            if op.type==capstone.CS_OP_MEM and op.mem.base==capstone.x86.X86_REG_RIP:
                target=i.address+i.size+op.mem.disp
                if target in slots:used.add(target)
                elif i.mnemonic=='cmp' and op.size==1:uc.mem_write(base+target,b'\1')
    names={};strings={}
    for n,a in enumerate(sorted(used)):
        kind,name=slots[a];p=arena+0x1000+n*0x400;q(base+a,p)
        if kind=='string':strings[p]=name
        else:names[name]=p;d(p+0xE0,1)
    required={'GameData_TypeInfo','GameplayEvents_TypeInfo','UIEvents_TypeInfo','System.Action_TypeInfo','System.Action<Character>_TypeInfo','int_TypeInfo','Method$AdvancedMode.OnFailed()','Method$AdvancedMode.OnCharacterKilled()'}
    assert required<=set(names)
    events,game,ui,mode,klass,ch,data,loaded,notify=[arena+n for n in range(0x30000,0x39000,0x1000)]
    for name,p in [('GameplayEvents_TypeInfo',events),('GameData_TypeInfo',game),('UIEvents_TypeInfo',ui)]:q(names[name]+0xB8,p)
    q(mode,klass);virtual=arena+0x40000;q(klass+0x240,virtual)
    q(ch+0x50,data);q(notify+0x18,stop+0x100);q(notify+0x40,arena+0x45000);q(notify+0x28,arena+0x46000)
    state={};options={};visited=set()
    def halt(error):state['error']=error;uc.emu_stop()
    def values():return [rd(mode+16+4*n) for n in range(7)]
    def hook(_,address,size,__):
        a=address-base;rcx,rdx,r8,r9=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        if address==stop:uc.emu_stop();return
        if address==stop+0x100:
            assert (rcx,rdx)==(arena+0x45000,arena+0x46000);state['notify']+=1
            if options.get('notify_fail'):halt('notify');return
            ret();return
        if a==0x2B7D40:
            state['alloc']+=1;p=arena+0x50000+state['alloc']*0x400;q(p,rcx);ret(p)
        elif a in (0x4D5170,0x4D5B60):state['ctors'].append((rdx,r8));ret()
        elif a in (0x116BCC0,0x116E070):
            state['ops'].append('combine' if a==0x116BCC0 else 'remove');n=len(state['ops'])
            if options.get('fail_op')==n:halt('delegate');return
            p=arena+0x60000+n*0x400;q(p,0 if options.get('bad_op')==n else names['System.Action<Character>_TypeInfo' if n==1 else 'System.Action_TypeInfo']);ret(0 if options.get('null_results') else p)
        elif a==0x2B7010:ret(rcx if rq(rcx)==rdx else 0)
        elif a==0x2B6FF0:assert rq(rcx)==rdx;state['writes'].append(rcx-events);ret()
        elif a in (0x2B7040,0x2B7D90):halt('cast' if a==0x2B7040 else 'null')
        elif a==0x281D90:
            state['cctor']+=1
            if options.get('class_fail'):halt('cctor');return
            d(rcx+0xE0,1);ret()
        elif a==0x3877D0:
            assert rcx==0;state['load']+=1
            if options.get('load_fail'):halt('load');return
            ret(options.get('loaded',loaded))
        elif a==0x387DC0:
            assert rcx==mode and rdx==0;state['saved'].append(values())
            if options.get('save_fail'):halt('save');return
            ret()
        elif a==0x385AA0:
            assert strings[rcx]=='Lilis_ACHIV_5030';state['achievement'].append(values())
            if options.get('achievement_fail'):halt('achievement');return
            ret()
        elif a==0x398D10:
            state['count']+=1
            if options.get('count_fail'):halt('count');return
            ret(options.get('unrevealed',3)&0xffffffff)
        elif a==0x387A20:
            state['rogue']+=1
            if options.get('rogue_fail'):halt('rogue');return
            ret(0 if options.get('rogue_null') else loaded)
        elif a==0x282580:
            assert rcx==names['int_TypeInfo'];p=arena+0x70000+len(state['boxes'])*0x100;state['boxes'].append(rd(rdx));d(p+16,rd(rdx));ret(p)
        elif a==0xF74DF0:
            state['formats'].append((strings[rcx],rd(rdx+16)))
            if options.get('format_fail')==len(state['formats']):halt('format');return
            p=arena+0x80000+len(state['formats'])*0x100;strings[p]=strings[rcx].replace('{0}',str(rd(rdx+16)));ret(p)
        elif a==0xF713F0:
            assert rq(reg(x.UC_X86_REG_RSP)+0x28)==0;state['concat']=[strings[p] for p in (rcx,rdx,r8,r9)];strings[arena+0x90000]=''.join(state['concat']);ret(arena+0x90000)
        elif a in ins:visited.add(a)
        else:raise AssertionError(f'unhandled {a:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    defaults=[23,40,7,9,2,4,5]
    results=[]
    def run(name,vals=None,opts=None):
        options.clear();options.update(opts or {});state.clear();state.update(error=None,alloc=0,ctors=[],ops=[],writes=[],cctor=0,load=0,saved=[],achievement=[],count=0,notify=0,rogue=0,boxes=[],formats=[],concat=[])
        for n,v in enumerate(vals or defaults):d(mode+16+4*n,v)
        uc.mem_write(mode+0x2C,bytes([options.get('reset',0),1]))
        for off in (0x48,0x20,0xB0):q(events+off,arena+0xA0000+off)
        d(game+0x18,99);uc.mem_write(game+0x1C,bytes([options.get('debug',0)]));d(names['GameData_TypeInfo']+0xE0,0 if options.get('cold') else 1)
        q(ui,notify if options.get('notify') else 0);q(ch+0x50,0 if options.get('null_data') else data);d(data+0x130,options.get('kind',100));d(loaded+0x10,options.get('rogue_level',3))
        sp=stack+0x10000;q(sp,stop);uc.reg_write(x.UC_X86_REG_RSP,sp);uc.reg_write(x.UC_X86_REG_RCX,0 if options.get('null_this') else mode);uc.reg_write(x.UC_X86_REG_RDX,(0 if options.get('null_ch') else ch) if name=='OnCharacterKilled' else 0);uc.reg_write(x.UC_X86_REG_RAX,0xABCDEFAA)
        uc.emu_start(base+selected[name]['Address'],stop,count=5000)
        if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==sp+8
        results.append({'method':name,'input':vals or defaults,'options':dict(options),'error':state['error']})
        return state
    for name in ('Init','DeInit'):
        for score,nulls in itertools.product([-1,0,7,2147483647],[False,True]):
            vals=[score,*defaults[1:]];r=run(name,vals,{'null_results':nulls})
            assert r['error'] is None and r['writes']==[0x48,0x20,0xB0] and values()==vals
            assert r['ops']==(['combine']*3 if name=='Init' else ['combine','remove','remove'])
            assert rd(game+0x18)==(score if name=='Init' else 99)
            assert r['ctors']==[(mode,names['Method$AdvancedMode.OnCharacterKilled()']),(mode,virtual),(mode,names['Method$AdvancedMode.OnFailed()'])]
        for failure,n in itertools.product(['bad_op','fail_op'],[1,2,3]):
            r=run(name,opts={failure:n});assert r['error']==('cast' if failure=='bad_op' else 'delegate') and r['writes']==[0x48,0x20,0xB0][:n-1] and values()==defaults and rd(game+0x18)==99
        r=run(name,opts={'null_this':True});assert r['error']=='null' and r['writes']==[0x48]
    for fail in [False,True]:
        r=run('Init',opts={'cold':True,'class_fail':fail});assert r['cctor']==1 and r['writes']==[0x48,0x20,0xB0] and rd(game+0x18)==(99 if fail else 23)
    for village,all_,reset in itertools.product([-2,0,5,2147483647],[-1,0,5,2147483647],[0,1]):
        vals=defaults.copy();vals[3]=village;vals[5]=all_;r=run('OnLoadGame',vals,{'reset':reset});expected=vals.copy()
        if not reset and village>all_:expected[5]=village
        assert values()==expected and bytes(uc.mem_read(mode+0x2C,2))==bytes([int(bool(reset) or village>all_),1]) and not r['saved']
    for village in [-2147483648,-1,8,9,10,2147483647]:
        for score,best in [(2,40),(45,40),(-1,-2)]:
            vals=[score,best,7,village,2147483647,-8,-5]
            for fail in [None,'achievement_fail','save_fail']:
                r=run('OnStageCompleted',vals,{fail:True} if fail else {})
                exp=vals.copy();exp[2]=0;exp[3]=signed(village+1);exp[4]=signed(vals[4]+1);exp[5]=signed(vals[5]+1)
                achievement=exp[3]>=10
                assert len(r['achievement'])==int(achievement)
                if not (achievement and fail=='achievement_fail'):
                    exp[5]=max(exp[3],exp[5]);exp[1]=max(exp[0],exp[1]);exp[6]=max(exp[4],exp[6]);assert r['saved']==[exp]
                else:assert not r['saved']
                assert values()==exp
    for village,fail in itertools.product([-2147483648,9,10,2147483647],[False,True]):
        vals=defaults.copy();vals[3]=village;r=run('CheckAchievements',vals,{'achievement_fail':fail});assert len(r['achievement'])==int(village>=10) and not r['saved'] and values()==vals
    for fail in [False,True]:
        r=run('OnFailed',opts={'save_fail':fail});assert values()==[0,40,0,0,0,4,5] and r['saved']==[[0,40,0,0,0,4,5]] and bytes(uc.mem_read(mode+0x2C,2))==b'\0\1'
    for kind,count,notify_ in itertools.product([0,29,30,31,99,100,101],[-6,-5,0,3,2147483647],[False,True]):
        vals=[2147483640,40,-8,9,2,4,5];r=run('OnCharacterKilled',vals,{'kind':kind,'unrevealed':count,'notify':notify_});exp=vals.copy();valid=kind in (30,100)
        if valid:delta=signed(10*signed(count+5));exp[0]=signed(exp[0]+delta);exp[2]=signed(exp[2]+delta)
        assert values()==exp and r['count']==int(valid) and r['notify']==int(valid and notify_) and not r['saved']
    for opt,error in [({'null_ch':True},'null'),({'null_data':True},'null'),({'count_fail':True},'count'),({'notify':True,'notify_fail':True},'notify')]:
        r=run('OnCharacterKilled',opts=opt);assert r['error']==error
        assert values()==([103,40,87,9,2,4,5] if error=='notify' else defaults)
    for debug,level,cold in itertools.product([0,1],[-2147483648,-1,0,2,3,4,2147483647],[False,True]):
        r=run('IsLocked',opts={'debug':debug,'rogue_level':level,'cold':cold});assert (reg(x.UC_X86_REG_RAX)&255)==int(not debug and level<3) and r['rogue']==int(not debug) and r['cctor']==int(cold)
    for opts,error in [({'rogue_null':True},'null'),({'rogue_fail':True},'rogue'),({'cold':True,'class_fail':True},'cctor')]:assert run('IsLocked',opts=opts)['error']==error
    for name in ('GetGameMode','GetMaxLevel','GetCurrentLevel','GetScore','GetResetLevel','GetStartingLevel','CanResetLevel','UpdateScore','.ctor'):
        for village in [-2147483648,-1,0,9,2147483647]:
            vals=defaults.copy();vals[3]=village;r=run(name,vals);assert values()==vals and not r['saved']
            if name in ('GetGameMode','GetMaxLevel','GetCurrentLevel','GetScore'):assert reg(x.UC_X86_REG_RAX)==0
            elif name in ('GetResetLevel','GetStartingLevel'):assert reg(x.UC_X86_REG_RAX)==(village&0xffffffff)
            elif name=='CanResetLevel':assert reg(x.UC_X86_REG_RAX)==0xABCDEF00
    for opts in ({},{'loaded':0},{'load_fail':True}):
        r=run('LoadGame',opts=opts);assert r['load']==1 and values()==defaults
        if not r['error']:assert reg(x.UC_X86_REG_RAX)==opts.get('loaded',loaded)
    for opts in ({},{'save_fail':True}):assert run('Save',opts=opts)['saved']==[defaults]
    for vals in [defaults,[-2147483648,2147483647,-9,-8,-7,-6,-5]]:
        for name in ('GetScores','GetSummaryScores'):
            r=run(name,vals);expected=[vals[1],vals[6],vals[5]] if name=='GetScores' else [vals[0]]
            expected_formats=['<size=24>Highest Score: <size=24><color=green>{0}</color></color></size>\n','<color=grey><size=24>Saves in a row: <color=white><size=24>{0}</color></color></size>\n','<color=grey><size=24>Saved villages: <color=white><size=24><color=white>{0}</size></color>\n'] if name=='GetScores' else ['Score: <color=green>{0}<size=22></color>\n\nScore resets on round loss.']
            assert [p[0] for p in r['formats']]==expected_formats
            assert r['boxes']==expected and [p[1] for p in r['formats']]==expected and values()==vals
            if name=='GetScores':assert len(r['concat'])==4 and r['concat'][0]=='\n', repr(r['concat'])
            for n in range(1,len(expected)+1):
                r=run(name,vals,{'format_fail':n});assert r['error']=='format' and len(r['boxes'])==n and not r['concat']
    return {'build':BUILD,'status':'passed','exact_methods':list(selected.values()),'native_cases':len(results),'instruction_assertions':len(checks)+2,'visited_instructions':len(visited),
            'findings':{'deinit_delegate_operations':['combine character killed','remove stage completed','remove failed'],'init_current_village_source':'currentScore','achievement_key':'Lilis_ACHIV_5030','scoring_types':[30,100],'score_formula':'i32(10*i32(unrevealed+5))','get_score':'constant zero','unlock_condition':'GameData.DebugMode or RoguelikeStandard.bestAscension >= 3'},
            'services':['delegate allocation/construction/combine/remove and cast runtime','GameData class initialization','SavesGame Advanced getter/setter and RoguelikeStandard getter','ProjectContext achievement unlock','unrevealed character count','UIEvents.OnUIUpdate delegate','boxing and String.Format/Concat; exact input strings/values checked'],
            'cases':results}

if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--game-root',required=True);parser.add_argument('--dumper-root',required=True);parser.add_argument('--output',required=True);args=parser.parse_args()
    report=audit(args.game_root,args.dumper_root);Path(args.output).write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8');print(json.dumps({k:v for k,v in report.items() if k not in ('cases','exact_methods')}))
