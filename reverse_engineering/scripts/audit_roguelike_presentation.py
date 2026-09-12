"""Supplemental native RoguelikeStandard level/presentation helpers."""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path
from audit_character_assets import BUILD


def signed(v):return ((v+0x80000000)&0xffffffff)-0x80000000


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
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
    raw=pinned(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    wanted={'RoguelikeStandard$$CheckIfLastLevel':0x3E9890,'RoguelikeStandard$$MaxLevel':0x3EA1C0,'RoguelikeStandard$$ShouldShowcaseNewCharacters':0x3EA460,'RoguelikeStandard$$GetKillScore':0x3E9CA0,'RoguelikeStandard$$GetSummaryScores':0x3E9E20,'RoguelikeStandard$$GetScores':0x3E9D60}
    declarations=[]
    for name,rva in wanted.items():
        rows=[r for r in script['ScriptMethod'] if r['Name']==name and r['Address']==rva];assert len(rows)==1;declarations+=rows
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True;decoded={};ranges=[]
    for a in wanted.values():
        b=min(r['Address'] for r in script['ScriptMethod'] if r['Address']>a)
        ins=list(cs.disasm(pe.get_data(a,b-a),a));assert ins[-1].address+ins[-1].size==b
        while ins[-1].mnemonic=='int3':ins.pop()
        ranges.append((a,ins[-1].address+ins[-1].size));decoded.update({i.address:i for i in ins})
    checks=[(0x3E98A1,'call','0x3ea1c0'),(0x3E98A6,'add','eax, edi'),(0x3E98A8,'lea','ecx, [rbx - 1]'),(0x3E98B2,'setge','al'),(0x3EA210,'cmp','ecx, dword ptr [rdx + 0x18]'),(0x3EA213,'jl','0x3ea21d'),(0x3EA21D,'jae','0x3ea246'),(0x3EA238,'dec','eax'),(0x3EA63B,'call','0x1c4b450'),(0x3EA64A,'setle','cl'),(0x3EA656,'cmovle','eax, edx'),(0x3E9E59,'add','eax, dword ptr [rbx + 0x28]')]
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
        rsp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,rsp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(rsp))
    types={name:arena+0x1000+i*0x400 for i,name in enumerate(['ProjectContext_TypeInfo','UnityEngine.Debug_TypeInfo','int_TypeInfo','object[]_TypeInfo'])}
    found=set()
    for r in script['ScriptMetadata']:
        if r['Name'] in types:q(base+r['Address'],types[r['Name']]);found.add(r['Name'])
    assert found==set(types)
    strings={};formats={}
    for addr in (0x26D96C0,0x270E950,0x27169B0,0x2716A38):
        row=next(r for r in script['ScriptString'] if r['Address']==addr)
        p=arena+0x4000+len(strings)*0x100;strings[p]=row['Value'];formats[hex(addr)]=row['Value'];q(base+addr,p)
    stat,project,data,outer,mode=arena+0x5000,arena+0x6000,arena+0x7000,arena+0x8000,arena+0x9000
    q(types['ProjectContext_TypeInfo']+0xB8,stat);d(types['UnityEngine.Debug_TypeInfo']+0xE0,1)
    q(types['object[]_TypeInfo']+0x40,arena+0xA000)
    for i in decoded.values():
        if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:
            uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
    state={};opt={};seen=set();boxvalues={};cursor=arena+0x20000
    def alloc(size=0x100):
        nonlocal cursor
        p=cursor;cursor+=(size+0xff)&~0xff
        if cursor>=arena+0xF0000:raise AssertionError('fixture arena exhausted')
        uc.mem_write(p,bytes(size));return p
    def fail(name):state['error']=name;uc.emu_stop()
    def hook(_,a,size,__):
        rva=a-base;rcx,rdx,r8=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8)]
        if a==stop:uc.emu_stop();return
        if rva==0x2B7080:
            assert rcx==types['object[]_TypeInfo'] and rdx==4
            p=alloc();q(p,rcx);d(p+0x18,rdx);ret(p)
        elif rva==0x282580:
            assert rcx==types['int_TypeInfo'];p=alloc();boxvalues[p]=signed(rd(rdx));ret(p)
        elif rva==0x2B7010:ret(rcx)
        elif rva==0x2B6FF0:ret()
        elif rva in (0xF74DF0,0xF74B10):
            vals=[boxvalues[rdx]] if rva==0xF74DF0 else [boxvalues[rq(rdx+0x20+8*i)] for i in range(rd(rdx+0x18))]
            state['formats'].append({'template':strings[rcx],'values':vals})
            if opt.get('format_fail')==len(state['formats']):fail('format');return
            p=alloc();strings[p]=strings[rcx].format(*vals);ret(p)
        elif rva==0xF71C60:
            state['concat']=[strings[rcx],strings[rdx]];p=alloc();strings[p]=strings[rcx]+strings[rdx];ret(p)
        elif rva==0x1C4B450:
            state['logs'].append(strings[rcx])
            if opt.get('log_fail'):fail('log')
            else:ret()
        elif rva==0x398D10:state['count_reads']+=1;ret(opt.get('count',0)&0xffffffff)
        elif rva in (0x2B7D90,0x2B7D80):fail('null' if rva==0x2B7D90 else 'range')
        elif rva in decoded:seen.add(rva)
        else:raise AssertionError(f'unhandled {rva:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    def run(entry,fields=None,arg=0,options=None,lengths=(2,0,7)):
        nonlocal cursor
        cursor=arena+0x20000;boxvalues.clear();state.clear();state.update(error=None,formats=[],logs=[],concat=[],count_reads=0);opt.clear();opt.update(options or {})
        uc.mem_write(mode,bytes([0xA5])*0x40)
        for off,val in (fields or {}).items():d(mode+off,val)
        q(stat,project);q(project+0x20,data);q(data+0x50,outer);d(outer+0x18,len(lengths))
        for i,n in enumerate(lengths):
            group=arena+0xB000+i*0x100;inner=arena+0xC000+i*0x100
            q(outer+0x20+i*8,group);q(group+0x10,inner);d(inner+0x18,n)
        null_at=opt.get('null_at')
        if null_at=='project':q(stat,0)
        if null_at=='data':q(project+0x20,0)
        if null_at=='outer':q(data+0x50,0)
        if null_at=='group':q(outer+0x20,0)
        if null_at=='inner':q(arena+0xB010,0)
        before=bytes(uc.mem_read(mode,0x40));rsp=stack+0x8000;q(rsp,stop)
        uc.reg_write(x.UC_X86_REG_RSP,rsp);uc.reg_write(x.UC_X86_REG_RCX,mode);uc.reg_write(x.UC_X86_REG_RDX,arg&0xffffffff);uc.reg_write(x.UC_X86_REG_R8,0)
        preserved={r:0x12340000+r for r in (x.UC_X86_REG_RBX,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI)}
        for r,v in preserved.items():uc.reg_write(r,v)
        uc.emu_start(base+entry,stop,count=10000)
        assert bytes(uc.mem_read(mode,0x40))==before
        if state['error'] is None:
            assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==rsp+8 and all(reg(r)==v for r,v in preserved.items())
        return reg(x.UC_X86_REG_RAX)
    cases=[]
    for lengths in ((),(0,),(1,),(2,0,7)):
        for index in (-2147483648,-1,0,1,2,3,99,2147483647):
            result=run(0x3EA1C0,{0x14:index},lengths=lengths)
            if index<0 or not lengths:assert state['error']=='range'
            else:assert state['error'] is None and signed(result)==lengths[min(index,len(lengths)-1)]-1
            cases.append({'method':'MaxLevel','index':index,'inner_lengths':list(lengths),'error':state['error'],'result':None if state['error'] else signed(result)})
    for null_at in ('project','data','outer','group','inner'):
        run(0x3EA1C0,{0x14:0},options={'null_at':null_at});assert state['error']=='null';cases.append({'method':'MaxLevel','null_at':null_at})
    for inner_length,village,mod in itertools.product((0,1,7),(-2147483648,-1,0,1,6,7,8,2147483647),(-2147483648,-1,0,1,2147483647)):
        result=run(0x3E9890,{0x14:0,0x20:village},arg=mod,lengths=(inner_length,))
        assert not state['error'] and result&0xff==int(signed(village-1)>=signed(inner_length-1+mod))
        cases.append({'method':'CheckIfLastLevel','inner_length':inner_length,'village':village,'mod':mod,'result':bool(result&0xff)})
    for current,best,village,bestvillage in itertools.product((-1,0,2),repeat=4):
        for showed in (0,1):
            result=run(0x3EA460,{0x14:current,0x10:best,0x20:village,0x1C:bestvillage,0x34:showed})
            assert not state['error'] and bool(result&0xff)==(best<=current and bestvillage<=village)
            assert state['formats'][0]['values']==[current,best,village,bestvillage] and len(state['logs'])==1
            cases.append({'method':'ShouldShowcaseNewCharacters','values':[current,best,village,bestvillage],'showed':showed,'result':bool(result&0xff)})
    for failure_options in ({'format_fail':1},{'log_fail':True}):
        run(0x3EA460,{0x14:2,0x10:0,0x20:2,0x1C:0},options=failure_options);assert state['error'] in ('format','log');cases.append({'method':'ShouldShowcaseNewCharacters','failure':state['error']})
    for count in (-2147483648,-5,-1,0,7,2147483647):
        value=run(0x3E9CA0,options={'count':count});assert signed(value)==signed((count+5)*10) and state['count_reads']==1
        cases.append({'method':'GetKillScore','count':count,'value':signed(value)})
    for first,second in itertools.product((-2147483648,-1,0,7,2147483647),repeat=2):
        out=run(0x3E9E20,{0x2C:first,0x28:second});assert state['formats'][0]['values']==[signed(first+second)] and out in strings
        cases.append({'method':'GetSummaryScores','ascension_score':first,'round_score':second,'format_value':signed(first+second)})
        out=run(0x3E9D60,{0x10:first,0x24:second});assert [r['values'] for r in state['formats']]==[[signed(first+1)],[second]] and strings[out]==''.join(state['concat'])
        cases.append({'method':'GetScores','best_ascension':first,'best_score':second,'format_values':[signed(first+1),second]})
    for entry,n in ((0x3E9E20,1),(0x3E9D60,1),(0x3E9D60,2)):
        run(entry,{0x10:2,0x24:77,0x28:3,0x2C:10},options={'format_fail':n})
        assert state['error']=='format' and len(state['formats'])==n and not state['concat']
        cases.append({'method':'GetSummaryScores' if entry==0x3E9E20 else 'GetScores','failure':'format','call':n})
    return {'schema_version':1,'build_id':BUILD,'exact_declarations':declarations,'native_relationships_verified':len(checks),'native_instructions_executed':len(seen),'ranges':[[hex(a),hex(b)] for a,b in ranges],'cases_passed':len(cases),'cases':cases,'format_templates':formats,'scope':'Six native helpers; boxing/format/concat/log/type/array allocation and unrevealed count are explicit service gateways. MaxLevel and CheckIfLastLevel execute together natively. Metadata warmed, no live globals/save/UI changes; no virtual slot invoked by these helper bodies.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} RoguelikeStandard helper cases")
