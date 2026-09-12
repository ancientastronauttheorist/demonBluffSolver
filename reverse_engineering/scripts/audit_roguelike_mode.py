"""Remaining seventeen RoguelikeMode declarations, with explicit external services."""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path
from audit_character_assets import BUILD


def i32(v):return ((v+0x80000000)&0xffffffff)-0x80000000


def audit(game_root,dumper_root):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    assert unicorn.__version__=='2.1.4'
    root=Path(__file__).parents[1]
    lock=json.loads((root/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    ext=json.loads((root/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(p,h):
        b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
    raw=pinned(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',ext['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump=pinned(Path(dumper_root)/'dump.cs',ext['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    for field in ('public int currentDay; // 0x7C','public static Gameplay Instance; // 0x10','public const EGameplayState Init = 1;','public const EGameplayState Map = 70;','public static bool DebugMode; // 0x1C'):
        assert field in dump
    decl=[r for r in script['ScriptMethod'] if r['Name'].startswith('RoguelikeMode$$') and r['Name'] not in ('RoguelikeMode$$GetCurrentAscension','RoguelikeMode$$GetPreviousAscension')]
    assert len(decl)==17
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={};ranges=[]
    for a in sorted({r['Address'] for r in decl}|{0x33ED50}):
        b=min(r['Address'] for r in script['ScriptMethod'] if r['Address']>a);ins=list(cs.disasm(pe.get_data(a,b-a),a));assert ins[-1].address+ins[-1].size==b
        while ins[-1].mnemonic=='int3':ins.pop()
        ranges.append((a,ins[-1].address+ins[-1].size));decoded.update({i.address:i for i in ins})
    checks=[(0x3E9449,'mov','dword ptr [rcx + 0x18], esi'),(0x3E948E,'call','0x116bcc0'),(0x3E9553,'call','0x116bcc0'),(0x3E8FD9,'call','0x116e070'),(0x3E90A0,'call','0x116e070'),(0x3E96EF,'mov','dword ptr [rcx + 0x7c], 0'),(0x3E973A,'je','0x3e9773'),(0x3E973C,'jle','0x3e9745'),(0x3E973E,'mov','dword ptr [rbx + 0x18], 0'),(0x3E9783,'jge','0x3e9793'),(0x3E978E,'jmp','0x387e60'),(0x3E965E,'call','0x387a20'),(0x3E9668,'cmp','dword ptr [rax + 0x10], 2'),(0x3E9347,'mov','rax, qword ptr [rdx + 0x258]'),(0x3E9355,'call','rax')]
    for a,m,o in checks:assert (decoded[a].mnemonic,decoded[a].op_str)==(m,o)
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,stop=0x200000000,0x300000000,0x400000000
    uc.mem_map(arena,0x100000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
    def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
    def reg(r):return uc.reg_read(r)
    def ret(v=0):
        rsp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v&0xffffffffffffffff);uc.reg_write(x.UC_X86_REG_RSP,rsp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(rsp))
    types={name:arena+0x1000+n*0x400 for n,name in enumerate(['System.Action_TypeInfo','GameData_TypeInfo','GameplayEvents_TypeInfo','Gameplay_TypeInfo','int_TypeInfo'])};found=set()
    for r in script['ScriptMetadata']:
        if r['Name'] in types:q(base+r['Address'],types[r['Name']]);found.add(r['Name'])
    assert found==set(types)
    tokens={}
    for r in script['ScriptMetadataMethod']:
        if r['Name'] in ('Method$RoguelikeMode.OnFailed()','Method$RoguelikeMode.OnRestartLevel()'):
            ptr=arena+0x4000+len(tokens)*0x100;tokens[r['Name']]=ptr;q(base+r['Address'],ptr)
    assert len(tokens)==2
    template=next(r for r in script['ScriptString'] if r['Address']==0x26E0D58)['Value'];q(base+0x26E0D58,arena+0x4500)
    gs,es,ps,mode,klass,gameplay,loaded= [arena+n for n in (0x5000,0x6000,0x7000,0x8000,0x9000,0xA000,0xB000)]
    for name,stat in [('GameData_TypeInfo',gs),('GameplayEvents_TypeInfo',es),('Gameplay_TypeInfo',ps)]:q(types[name]+0xB8,stat)
    q(mode,klass)
    for ins in decoded.values():
        if ins.mnemonic=='cmp' and ins.operands[0].type==capstone.CS_OP_MEM and ins.operands[0].size==1 and ins.operands[0].mem.base==capstone.x86.X86_REG_RIP:uc.mem_write(base+ins.address+ins.size+ins.operands[0].mem.disp,b'\1')
    seen=set();state={};opt={};boxes={};cursor=arena+0x20000
    def alloc():
        nonlocal cursor
        p=cursor;cursor+=0x100;uc.mem_write(p,bytes(0x100));return p
    def halt(e):state['error']=e;uc.emu_stop()
    def hook(_,a,size,__):
        rva=a-base;rcx,rdx,r8=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8)]
        if a==stop:uc.emu_stop();return
        if a==stop+0x100:state['virtual_reads']+=1;assert rcx==mode;ret(opt['virtual']);return
        if rva==0x281D90:
            state['class_init']+=1
            if opt.get('class_fail'):halt('class');return
            d(rcx+0xE0,1);ret()
        elif rva==0x2B7D40:p=alloc();q(p,rcx);ret(p)
        elif rva==0x4D5170:state['ctors'].append((rdx,r8));ret()
        elif rva in (0x116BCC0,0x116E070):
            state['ops']+=1;n=state['ops'];state['op_names'].append('combine' if rva==0x116BCC0 else 'remove')
            if opt.get('fail_op')==n:halt('delegate');return
            p=alloc();q(p,0 if opt.get('bad_op')==n else types['System.Action_TypeInfo']);ret(0 if opt.get('null_result') else p)
        elif rva==0x2B6FF0:state['writes'].append(rcx-es);ret()
        elif rva in (0x2B7040,0x2B7D90):halt('cast' if rva==0x2B7040 else 'null')
        elif rva==0x387980:state['loads'].append('roguelike');ret(0 if opt.get('null_load') else loaded)
        elif rva==0x387A20:state['loads'].append('roguelike_standard');ret(0 if opt.get('null_load') else loaded)
        elif rva==0x387E60:
            state['saves'].append([i32(rd(rcx+0x10)),i32(rd(rcx+0x14))]);assert rcx==mode
            if opt.get('save_fail'):halt('save')
            else:ret()
        elif rva==0x37B620:
            state['states'].append(rcx&0xffffffff)
            if opt.get('state_fail')==len(state['states']):halt('state')
            else:ret()
        elif rva==0x282580:assert rcx==types['int_TypeInfo'];p=alloc();boxes[p]=i32(rd(rdx));ret(p)
        elif rva==0xF74DF0:
            assert rcx==arena+0x4500;state['format'].append(boxes[rdx]);ret(arena+0xC000)
        elif rva in decoded:seen.add(rva)
        else:raise AssertionError(f'unhandled {rva:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    def run(entry,score=17,level=4,deaths=3,arg=0,arg2=0,options=None):
        nonlocal cursor
        cursor=arena+0x20000;boxes.clear();state.clear();state.update(error=None,ctors=[],ops=0,op_names=[],writes=[],loads=[],saves=[],states=[],format=[],class_init=0,virtual_reads=0);opt.clear();opt.update(options or {})
        for name,ptr in types.items():d(ptr+0xE0,0 if opt.get('cold') else 1)
        d(mode+0x10,score);d(mode+0x14,level);d(mode+0x18,deaths);d(gs+0x18,99);uc.mem_write(gs+0x1C,bytes([opt.get('debug',0)]));d(loaded+0x10,opt.get('best',0));q(ps+0x10,0 if opt.get('null_gameplay') else gameplay);d(gameplay+0x7C,123)
        for off in (0xB0,0x38):q(es+off,arena+0xD000+off)
        q(klass+0x258,stop+0x100 if 'virtual' in opt else base+0x358300)
        rsp=stack+0x8000;q(rsp,stop);uc.reg_write(x.UC_X86_REG_RSP,rsp);uc.reg_write(x.UC_X86_REG_RCX,mode);uc.reg_write(x.UC_X86_REG_RDX,arg&0xffffffff);uc.reg_write(x.UC_X86_REG_R8,arg2&0xffffffff)
        preserved={r:0x12340000+r for r in (x.UC_X86_REG_RBX,x.UC_X86_REG_RBP,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI)}
        for r,v in preserved.items():uc.reg_write(r,v)
        uc.emu_start(base+entry,stop,count=10000)
        if state['error'] is None:assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==rsp+8 and all(reg(r)==v for r,v in preserved.items())
        return reg(x.UC_X86_REG_RAX)
    cases=[]
    for name,entry in [('Init',0x3E9390),('DeInit',0x3E8F40)]:
        for options in ({},{'null_result':True},{'fail_op':1},{'fail_op':2},{'bad_op':1},{'bad_op':2}):
            run(entry,options=options);failure=options.get('fail_op',options.get('bad_op'));expected_n=2 if failure is None else failure-1
            assert state['writes']==[0xB0,0x38][:expected_n] and rd(gs+0x18)==(0 if name=='Init' else 99)
            assert [m for _,m in state['ctors']]==[tokens['Method$RoguelikeMode.OnFailed()'],tokens['Method$RoguelikeMode.OnRestartLevel()']][:len(state['ctors'])]
            assert state['op_names']==(['combine'] if name=='Init' else ['remove'])*state['ops']
            assert [rd(mode+off) for off in (0x10,0x14,0x18)]==[17,4,3]
            cases.append({'method':name,'options':options,'writes':state['writes'],'error':state['error']})
    for fail in (False,True):
        run(0x3E9390,options={'cold':True,'class_fail':fail});assert rd(gs+0x18)==(99 if fail else 0) and state['class_init']==1
        cases.append({'method':'Init','cold_class':True,'failure':fail})
    for score,new,level in itertools.product((-2147483648,-1,0,7,2147483647),(-2147483648,-1,0,7,2147483647),(-1,0,9)):
        run(0x3E9780,score=score,arg=new,arg2=level)
        assert bool(state['saves'])==(new>score) and i32(rd(mode+0x10))==(new if new>score else score) and i32(rd(mode+0x14))==(level if new>score else 4)
        cases.append({'method':'UpdateScore','old':score,'new':new,'level':level,'saved':bool(state['saves'])})
    run(0x3E9780,score=1,arg=2,arg2=-3,options={'save_fail':True});assert state['error']=='save' and i32(rd(mode+0x10))==2 and i32(rd(mode+0x14))==-3;cases.append({'method':'UpdateScore','failure':'save','writes_preserved':True})
    for deaths in (-2147483648,-1,0,1,2147483647):
        for fail_at in (0,1,2):
            run(0x3E9710,deaths=deaths,options={'state_fail':fail_at});assert i32(rd(mode+0x18))==(0 if deaths>0 else deaths)
            expected=[] if deaths==0 else [1,70][:(fail_at or 2)];assert state['states']==expected
            cases.append({'method':'OnRestartLevel','deaths':deaths,'failure_at':fail_at,'states':state['states']})
    for deaths in (-1,1):
        run(0x3E9710,deaths=deaths,options={'cold':True,'class_fail':True})
        assert state['error']=='class' and not state['states'] and i32(rd(mode+0x18))==(0 if deaths>0 else deaths)
        cases.append({'method':'OnRestartLevel','deaths':deaths,'failure':'class'})
    for missing in (False,True):
        run(0x3E96A0,options={'null_gameplay':missing});assert state['error']==('null' if missing else None) and rd(gameplay+0x7C)==(123 if missing else 0);cases.append({'method':'OnFailed','missing_gameplay':missing})
    run(0x3E96A0,options={'cold':True,'class_fail':True});assert state['error']=='class' and rd(gameplay+0x7C)==123
    cases.append({'method':'OnFailed','failure':'class'})
    for debug,best,null in itertools.product((0,1),(-2147483648,-1,0,1,2,7),(False,True)):
        v=run(0x3E9610,options={'debug':debug,'best':best,'null_load':null})
        if debug:assert not state['loads'] and not state['error'] and v&0xff==0
        elif null:assert state['error']=='null'
        else:assert not state['error'] and bool(v&0xff)==(best<2)
        cases.append({'method':'IsLocked','debug':debug,'best_standard_ascension':best,'null_save':null,'error':state['error']})
    for value in (-2147483648,-1,0,7,2147483647):
        for virtual in (False,True):
            run(0x3E9310,score=31,options={'virtual':value} if virtual else {});assert state['format']==[value if virtual else 31];cases.append({'method':'GetScores','override':virtual,'formatted':state['format'][0]})
    for null in (False,True):
        result=run(0x3E9690,options={'null_load':null});assert result==(0 if null else loaded) and state['loads']==['roguelike'];cases.append({'method':'LoadGame','null_result':null})
    for row in decl:
        if row['Address'] not in (0x3E8E10,0x3E9250,0x358300,0x33ED50,0x3712B0,0x3BCC90,0x357920):continue
        v=run(row['Address'],score=-7,level=-9,deaths=11)
        expected={0x3E8E10:10,0x3E9250:-9,0x358300:-7,0x3712B0:0,0x3BCC90:0}
        if row['Address'] in expected:assert (v&0xff if row['Address']==0x3BCC90 else i32(v))==expected[row['Address']]
        assert [i32(rd(mode+off)) for off in (0x10,0x14,0x18)]==[-7,-9,11]
        cases.append({'method':row['Name'],'value':expected.get(row['Address']),'unchanged_fields':True})
    return {'schema_version':1,'build_id':BUILD,'exact_declarations':decl,'cases_passed':len(cases),'native_relationships_verified':len(checks),'native_instructions_executed':len(seen),'ranges':[[hex(a),hex(b)] for a,b in ranges],'cases':cases,'format_template':template,'scope':'Seventeen native declared methods; two previously audited ascension selectors excluded. Save getters/setter, delegate operations, allocation/boxing/formatting, class initialization and gameplay state events are explicit gateways. Exact load/save property implementations are separately audited. No inherited-surface or full runtime event lifecycle completeness claim.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} remaining RoguelikeMode cases")
