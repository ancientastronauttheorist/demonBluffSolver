"""Pinned RoguelikeStandard lifecycle and SavedRoguelikeStandard load audit; services are gateways."""
import argparse
import hashlib
import json
import struct
from pathlib import Path
from audit_character_assets import BUILD


def audit(game_root,dumper_root,target_manifest):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    assert unicorn.__version__=='2.1.4'
    repo=Path(__file__).parents[1]
    lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(path,digest):
        raw=path.read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
    raw=pinned(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump=pinned(Path(dumper_root)/'dump.cs',extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    game_decl=dump.split('public abstract class GameMode //',1)[1].split('// Namespace:',1)[0]
    for declaration in ('public abstract void Init();','public abstract GameMode LoadGame();'):
        assert declaration in game_decl
    targets=json.loads(Path(target_manifest).read_text(encoding='utf-8'))['functions']
    for f in targets:assert any(r['Name']==f['metadata_name'] and r['Signature']==f['signature'] and r['Address']==int(f['rva'],16) for r in script['ScriptMethod'])
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    ranges=[(0x3E9E90,0x3EA1A3),(0x3E98C0,0x3E9BD3),(0x3EA1B0,0x3EA1B7),(0x357920,0x357927),(0x33ED50,0x33ED53),(0x387A20,0x387AC0)]
    decoded={}
    for a,b in ranges:
        ins=list(cs.disasm(pe.get_data(a,b-a),a));assert ins[-1].address+ins[-1].size==b
        decoded.update({i.address:i for i in ins})
    checks=[(0x3EA1B0,'xor','ecx, ecx'),(0x3EA1B2,'jmp','0x387a20'),(0x357922,'jmp','0x33ed50'),(0x3E9F33,'call','0x116bcc0'),(0x3EA005,'call','0x116bcc0'),(0x3EA0C5,'call','0x116bcc0'),(0x3E9963,'call','0x116bcc0'),(0x3E9A35,'call','0x116e070'),(0x3E9AF5,'call','0x116e070')]
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
    types={name:arena+0x1000+n*0x400 for n,name in enumerate(['System.Action_TypeInfo','System.Action<Character>_TypeInfo','GameplayEvents_TypeInfo','GameData_TypeInfo','RoguelikeStandard_TypeInfo'])}
    found=set()
    for r in script['ScriptMetadata']:
        if r['Name'] in types:q(base+r['Address'],types[r['Name']]);found.add(r['Name'])
    assert found==set(types)
    method_tokens={}
    for r in script['ScriptMetadataMethod']:
        if r['Name'] in ('Method$RoguelikeStandard.OnFailed()','Method$RoguelikeStandard.OnCharacterKilled()'):
            token=arena+0x5000+len(method_tokens)*0x100;method_tokens[r['Name']]=token;q(base+r['Address'],token)
    assert len(method_tokens)==2
    json_method=next(r for r in script['ScriptMetadataMethod'] if r['Name']=='Method$UnityEngine.JsonUtility.FromJson<RoguelikeStandard>()')
    assert json_method['Address']==0x387AAD+7+0x238F5AC
    q(base+json_method['Address'],arena+0x5500)
    saved_key=next(r for r in script['ScriptString'] if r['Value']=='SavedRoguelikeStandard');q(base+saved_key['Address'],arena+0x6000)
    event_static,game_static,mode,klass=arena+0x7000,arena+0x8000,arena+0x9000,arena+0xA000
    for name,ptr in types.items():d(ptr+0xE0,1)
    q(types['GameplayEvents_TypeInfo']+0xB8,event_static);q(types['GameData_TypeInfo']+0xB8,game_static)
    q(mode,klass);q(klass+0x240,arena+0x5200)
    for i in decoded.values():
        if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:
            uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
    state={};opt={};visited=set()
    def halt(error):state['error']=error;uc.emu_stop()
    def hook(_,a,size,__):
        rva=a-base;rcx,rdx,r8=reg(x.UC_X86_REG_RCX),reg(x.UC_X86_REG_RDX),reg(x.UC_X86_REG_R8)
        if a==stop:uc.emu_stop();return
        if rva==0x2B7D40:
            state['alloc']+=1;p=arena+0x10000+state['alloc']*0x100;q(p,rcx);state['allocated'].append(p);ret(p)
        elif rva in (0x4D5170,0x4D5B60):
            state['ctors'].append({'target':rdx,'method':r8});ret()
        elif rva in (0x116BCC0,0x116E070):
            state['ops']+=1;n=state['ops'];state['inputs'].append((rcx,rdx))
            if opt.get('fail_op')==n:halt('delegate_gateway');return
            p=arena+0x20000+n*0x100
            typ=types['System.Action<Character>_TypeInfo'] if n==1 else types['System.Action_TypeInfo']
            state['operation_kinds'].append('combine' if rva==0x116BCC0 else 'remove')
            q(p,0 if opt.get('bad_op')==n else typ)
            ret(0 if opt.get('null_results') else p)
        elif rva==0x281D90:
            state['class_init']+=1
            if opt.get('class_fail'):halt('class_init');return
            d(rcx+0xE0,1);ret()
        elif rva==0x2B7010:ret(rcx if rq(rcx)==rdx else 0)
        elif rva==0x2B6FF0:state['writes'].append(rcx-event_static);ret()
        elif rva in (0x2B7040,0x2B7D90):halt('cast' if rva==0x2B7040 else 'null')
        elif rva==0x1C85F20:
            assert rcx==arena+0x6000;state['reads']+=1
            if opt.get('fail_read')==state['reads']:halt('preferences');return
            ret(opt.get('json_values',[opt.get('json',0)]*2)[state['reads']-1])
        elif rva==0xF76390:ret(int(rcx==0))
        elif rva==0x645DA0:
            assert rdx==arena+0x5500
            state['json_inputs'].append(rcx)
            if opt.get('json_fail'):halt('json');return
            ret(opt.get('loaded',arena+0x30000))
        elif rva in decoded:visited.add(rva)
        else:raise AssertionError(f'unhandled {rva:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    def run(name,entry,options=None,null=False):
        opt.clear();opt.update(options or {});state.clear();state.update(method=name,operation_kinds=[],alloc=0,allocated=[],ctors=[],ops=0,inputs=[],writes=[],reads=0,json_inputs=[],class_init=0,error=None)
        d(types['GameData_TypeInfo']+0xE0,0 if opt.get('cold_class') else 1)
        uc.mem_write(mode+0x10,bytes([0x55])*0x30);d(mode+0x14,opt.get('level',7));d(game_static+0x18,99)
        for off in (0x20,0xB0,0x48):q(event_static+off,arena+0x40000+off)
        rsp=stack+0x8000;q(rsp,stop);uc.reg_write(x.UC_X86_REG_RSP,rsp);uc.reg_write(x.UC_X86_REG_RCX,0 if null else mode);uc.reg_write(x.UC_X86_REG_RDX,0)
        preserved={r:0x12340000+r for r in (x.UC_X86_REG_RBX,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_R14)}
        for r,v in preserved.items():uc.reg_write(r,v)
        before_fields=bytes(uc.mem_read(mode+0x10,0x28));before_game=bytes(uc.mem_read(game_static,0x100))
        uc.emu_start(base+entry,stop,count=5000)
        assert bytes(uc.mem_read(mode+0x10,0x28))==before_fields
        assert bytes(uc.mem_read(game_static,0x100))==before_game
        if state['error'] is None:
            assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==rsp+8
            assert all(reg(r)==v for r,v in preserved.items())
        return dict(state)
    results=[]
    for name,entry in [('init',0x3E9E90),('deinit',0x3E98C0)]:
        order=[0x48,0x20,0xB0]
        for level in (-1,0,7,2147483647):
            for null_results in (False,True):
                r=run(name,entry,{'level':level,'null_results':null_results});assert r['error'] is None and r['writes']==order
                assert rd(game_static+0x18)==99 and rd(mode+0x28)==0x55555555
                expected=[method_tokens['Method$RoguelikeStandard.OnCharacterKilled()'],arena+0x5200,method_tokens['Method$RoguelikeStandard.OnFailed()']]
                assert [v['method'] for v in r['ctors']]==expected
                assert all(v['target']==mode for v in r['ctors'])
                assert r['operation_kinds']==(['combine']*3 if name=='init' else ['combine','remove','remove'])
                results.append({'method':name,'level':level,'null_delegate_results':null_results,'operations':r['operation_kinds'],'writes':r['writes']})
        for failure in ('bad_op','fail_op'):
            for n in (1,2,3):
                r=run(name,entry,{failure:n});assert r['error']==('cast' if failure=='bad_op' else 'delegate_gateway') and r['writes']==order[:n-1]
                assert rd(game_static+0x18)==99 and rd(mode+0x28)==0x55555555
                results.append({'method':name,'failure':failure,'operation':n,'writes':r['writes']})
    for options in ({},{'json':arena+0x6100},{'json':arena+0x6100,'loaded':0},{'fail_read':1},{'json':arena+0x6100,'fail_read':2},{'json':arena+0x6100,'json_fail':True},{'json_values':[arena+0x6100,arena+0x6200]}):
        r=run('load',0x3EA1B0,options)
        assert not r['writes'] and rd(game_static+0x18)==99
        if not options:
            assert r['reads']==1 and reg(x.UC_X86_REG_RAX)==r['allocated'][0]
        elif not r['error']:
            assert r['reads']==2 and reg(x.UC_X86_REG_RAX)==options.get('loaded',arena+0x30000)
            assert r['json_inputs']==[options.get('json_values',[0,options.get('json')])[1]]
        results.append({'method':'load','reads':r['reads'],'json_calls':len(r['json_inputs']),'error':r['error']})
    r=run('ctor',0x357920);assert r['error'] is None and rd(mode+0x38)==0x55555555
    results.append({'method':'ctor','no_explicit_field_initializers':True})
    return {'schema_version':1,'build_id':BUILD,'targets_verified':len(targets),'exact_json_method':json_method,'cases_passed':len(results),'native_relationships_verified':len(checks),'distinct_native_instructions':len(visited),'cases':results,'scope':'RoguelikeStandard Init/DeInit/LoadGame/shared constructor and SavedRoguelikeStandard getter execute natively. DeInit combines CharacterKilled while removing RoundWon/Died. Delegate and allocation/runtime/JSON/preferences internals are explicit gateways. No whole-mode progression or live event impact claimed.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--targets',type=Path,default=Path(__file__).parents[1]/'targets/roguelike_standard_lifecycle.json');p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.game_root,a.dumper_root,a.targets);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed {r['cases_passed']} lifecycle cases")
