"""Native multicast list mutation and RoguelikeStandard kill-score boundary."""
import argparse
import hashlib
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
    repo=Path(__file__).parents[1]
    lock=json.loads((repo/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction=json.loads((repo/f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    def pinned(p,h):
        b=p.read_bytes();assert hashlib.sha256(b).hexdigest().upper()==h.upper();return b
    raw=pinned(Path(game_root)/'GameAssembly.dll',lock['inputs']['game_assembly']['sha256'])
    script=json.loads(pinned(Path(dumper_root)/'script.json',extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump=pinned(Path(dumper_root)/'dump.cs',extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    for field in ('public CharacterData dataRef; // 0x50','public ECharacterType type; // 0x130','public const ECharacterType Minion = 30;','public const ECharacterType Demon = 100;','public static Action OnUIUpdate; // 0x0'):
        assert field in dump
    names={'RoguelikeStandard$$DeInit':0x3E98C0,'System.Action<object>$$.ctor':0x4D5B60,'System.Action<object>$$Invoke':0x4A86F0,'System.Delegate$$Combine':0x116BCC0,'System.Delegate$$Remove':0x116E070,'System.MulticastDelegate$$CombineImpl':0x1174850,'System.MulticastDelegate$$RemoveImpl':0x1175150,'System.MulticastDelegate$$Equals':0x1174C80,'System.Delegate$$Equals':0x116D410,'RoguelikeStandard$$OnCharacterKilled':0x3EA250}
    exact=[]
    for name,rva in names.items():
        rows=[r for r in script['ScriptMethod'] if r['Name']==name and r['Address']==rva];assert len(rows)==1;exact+=rows
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    entries=[0x116BCC0,0x116E070,0x1174850,0x1175150,0x1174C80,0x116D410,0x3EA250,0x4A0210,0x4A0220,0x1026E90,0x4D5B60,0x4A86F0,0x3E98C0]
    decoded={};ranges=[]
    for a in entries:
        b=min(r['Address'] for r in script['ScriptMethod'] if r['Address']>a)
        ins=list(cs.disasm(pe.get_data(a,b-a),a))
        assert ins and ins[-1].address+ins[-1].size==b
        while ins[-1].mnemonic=='int3':ins.pop()
        ranges.append((a,ins[-1].address+ins[-1].size))
        decoded.update({i.address:i for i in ins})
    pe.parse_data_directories(directories=[3])
    for a,b in [(0x25E600,0x25E657),(0x5C70,0x5CD0)]:
        assert any(e.struct.BeginAddress==a and e.struct.EndAddress==b for e in pe.DIRECTORY_ENTRY_EXCEPTION)
        ins=list(cs.disasm(pe.get_data(a,b-a),a));assert ins[-1].address+ins[-1].size==b
        decoded.update({i.address:i for i in ins});ranges.append((a,b))
    checks=[(0x3E9963,'call','0x116bcc0'),(0x3E9A35,'call','0x116e070'),(0x3E9AF5,'call','0x116e070'),(0x3EA281,'mov','rax, qword ptr [rbx + 0x50]'),(0x3EA28A,'cmp','dword ptr [rax + 0x130], 0x64'),(0x3EA293,'cmp','dword ptr [rax + 0x130], 0x1e'),(0x3EA2AB,'add','dword ptr [rdi + 0x28], ecx'),(0x4D5C38,'lea','rax, [rip - 0x4cffcf]'),(0x25E640,'mov','qword ptr [rbx + 0x18], rdx'),(0x5CAF,'call','qword ptr [rax + 0x18]')]
    for address,mnemonic,operands in checks:
        assert (decoded[address].mnemonic,decoded[address].op_str)==(mnemonic,operands)
    assert 0x4D5C38+decoded[0x4D5C38].size+decoded[0x4D5C38].operands[1].mem.disp==0x5C70
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64);uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,stop=0x200000000,0x300000000,0x400000000
    uc.mem_map(arena,0x1000000);uc.mem_map(stack,0x10000);uc.mem_map(stop,0x1000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def d(a,v):uc.mem_write(a,struct.pack('<I',v&0xffffffff))
    def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def rd(a):return struct.unpack('<I',uc.mem_read(a,4))[0]
    def reg(r):return uc.reg_read(r)
    def ret(v=0):
        rsp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,rsp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(rsp))
    delegate,multi,action,array,typ,uitype=[arena+n*0x1000 for n in range(1,7)]
    types={'System.Delegate_TypeInfo':delegate,'System.MulticastDelegate_TypeInfo':multi,'System.Delegate[]_TypeInfo':array,'System.Type_TypeInfo':typ,'UIEvents_TypeInfo':uitype}
    found=set()
    for r in script['ScriptMetadata']:
        if r['Name'] in types:q(base+r['Address'],types[r['Name']]);found.add(r['Name'])
    assert found==set(types),(found,set(types))
    for depth,t in enumerate((delegate,multi,action),1):
        uc.mem_write(t+0x130,bytes([depth]));q(t+0xC8,arena+0x8000)
        q(t+0x138,base+0x1174C80);q(t+0x1B8,stop+0x100);q(t+0x1E8,base+0x1174850);q(t+0x1F8,base+0x1175150)
    for n,t in enumerate((delegate,multi,action)):q(arena+0x8000+n*8,t)
    q(array+0x138,base+0x4A0210);q(array+0x40,delegate);d(typ+0xE0,1);q(uitype+0xB8,arena+0x9000)
    for i in decoded.values():
        if i.mnemonic=='cmp' and i.operands[0].type==capstone.CS_OP_MEM and i.operands[0].size==1 and i.operands[0].mem.base==capstone.x86.X86_REG_RIP:
            uc.mem_write(base+i.address+i.size+i.operands[0].mem.disp,b'\1')
    cursor=arena+0x10000
    def alloc(size=0x100):
        nonlocal cursor
        p=cursor;cursor+=(size+0xff)&~0xff;uc.mem_write(p,bytes(size));return p
    def single(target,method):
        p=alloc();q(p,action);run(0x4D5B60,p,target,method_arg=method);assert not state['error'];return p
    def members(p):
        if p==0:return []
        a=rq(p+0x78)
        return [rq(a+0x20+n*8) for n in range(rd(a+0x18))] if a else [p]
    seen=set();state={};services={}
    def hook(_,a,size,__):
        rva=a-base;rcx,rdx,r8,r9=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        if a==stop:uc.emu_stop();return
        if a==stop+0x100:services['method_info']=services.get('method_info',0)+1;ret(rq(rcx+0x28));return
        if a==stop+0x200:
            state['ui']+=1
            if state.get('fail_ui'):state['error']='ui';uc.emu_stop()
            else:ret()
            return
        if rva==0x29A490:ret(rq(rcx))
        elif rva==0x2C3B10:
            p=alloc();q(p,rcx);ret(p)
        elif rva==0x294E80:q(rcx,rdx);ret()
        elif rva==0x2B7020:ret(0)
        elif rva==0x2B7080:
            assert rcx==array and 0<=rdx<1000;p=alloc(0x20+rdx*8);q(p,array);q(p+0x18,rdx);ret(p)
        elif rva==0x2B7010:ret(rcx if rcx and rq(rcx)==action and rdx in (delegate,multi,action) else 0)
        elif rva==0x2B6FF0:ret()
        elif rva in (0x112C1E0,0x112BF60):
            if rva==0x112C1E0:src,si,dst,di,n=rcx,rdx,r8,r9,rq(reg(x.UC_X86_REG_RSP)+0x28)&0xffffffff
            else:src,si,dst,di,n=rcx,0,rdx,0,r8
            assert si+n<=rd(src+0x18) and di+n<=rd(dst+0x18)
            uc.mem_write(dst+0x20+di*8,bytes(uc.mem_read(src+0x20+si*8,n*8)));services['array_copy']=services.get('array_copy',0)+1;ret()
        elif rva==0x398D10:state['count_reads']+=1;ret(state['count']&0xffffffff)
        elif rva in (0x2B7040,0x2B7D90,0x2B7D80,0x2B7D50):state['error']=hex(rva);uc.emu_stop()
        elif rva in decoded:seen.add(rva)
        else:raise AssertionError(f'unhandled {rva:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    def run(entry,rcx,rdx=0,count=0,fail_ui=False,method_arg=0):
        state.clear();state.update(ui=0,count_reads=0,count=count,error=None,fail_ui=fail_ui)
        rsp=stack+0x8000;q(rsp,stop);uc.reg_write(x.UC_X86_REG_RSP,rsp);uc.reg_write(x.UC_X86_REG_RCX,rcx);uc.reg_write(x.UC_X86_REG_RDX,rdx);uc.reg_write(x.UC_X86_REG_R8,method_arg)
        preserved={r:0x12340000+r for r in (x.UC_X86_REG_RBX,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_RBP,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15)}
        for r,v in preserved.items():uc.reg_write(r,v)
        uc.emu_start(base+entry,stop,count=50000)
        if state['error'] is None:
            assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==rsp+8
            assert all(reg(r)==v for r,v in preserved.items())
        return reg(x.UC_X86_REG_RAX)
    target,other=arena+0xA000,arena+0xA100
    method1,method2,method_type=arena+0xE000,arena+0xE100,arena+0xF000
    q(method1,method_type);q(method2,method_type);q(method_type+0x138,base+0x4A0210)
    for method in (method1,method2):q(method+8,base+0x3EA250);uc.mem_write(method+0x52,b'\1')
    a=single(target,method1);equal=single(target,method1);different_target=single(other,method1);different_method=single(target,method2)
    for value,expected in [(a,1),(equal,1),(different_target,0),(different_method,0),(0,0)]:
        assert run(0x1174C80,a,value)&0xff==expected and not state['error']
    combo=run(0x116BCC0,a,equal);assert members(combo)==[a,equal] and not state['error']
    assert members(a)==[a] and members(equal)==[equal]
    combo3=run(0x116BCC0,combo,a);assert members(combo3)==[a,equal,a]
    combo4=run(0x116BCC0,combo,combo);assert members(combo4)==[a,equal,a,equal]
    head=run(0x116BCC0,a,combo);assert members(head)==[a,a,equal]
    assert run(0x116E070,a,equal)==0 and not state['error']
    assert run(0x116E070,a,different_target)==a and not state['error']
    equal_list_result=run(0x116E070,combo,run(0x116BCC0,equal,a))
    assert members(equal_list_result)==[] and not state['error']
    removed=run(0x116E070,combo4,combo);assert members(removed)==[a,equal] and not state['error']
    acc=0;accumulation=[]
    for phase in ['init','deinit','init','deinit','init']:
        handler=single(target,method1);acc=run(0x116BCC0,acc,handler);assert not state['error']
        accumulation.append({'phase':phase,'kill_handler_count':len(members(acc))})
    assert [r['kill_handler_count'] for r in accumulation]==[1,2,3,4,5]
    character,data,ui=arena+0xB000,arena+0xC000,arena+0xD000
    q(character+0x50,data);q(ui+0x18,stop+0x200)
    score_cases=[]
    for kind in (0,10,20,30,100):
        for count in (-1,0,1,7):
            for initial in (0,0x7fffffff):
                for enabled in (False,True):
                    d(data+0x130,kind);d(target+0x28,initial);q(arena+0x9000,ui if enabled else 0)
                    run(0x3EA250,target,character,count);assert not state['error']
                    eligible=kind in (30,100);expected=(initial+(count+5)*10*eligible)&0xffffffff
                    assert rd(target+0x28)==expected and state['count_reads']==int(eligible) and state['ui']==int(eligible and enabled)
                    score_cases.append({'type':kind,'count':count,'initial':initial,'ui':enabled,'final':expected})
    for null_character in (False,True):
        q(character+0x50,0 if not null_character else data);d(target+0x28,321)
        run(0x3EA250,target,0 if null_character else character)
        assert state['error']=='0x2b7d90' and rd(target+0x28)==321 and state['count_reads']==0
    q(character+0x50,data)
    d(data+0x130,100);d(target+0x28,0);q(arena+0x9000,ui)
    run(0x3EA250,target,character,7,True);assert state['error']=='ui' and rd(target+0x28)==120
    d(target+0x28,0)
    run(0x4A86F0,acc,character,7,True);assert state['error']=='ui' and rd(target+0x28)==120 and state['count_reads']==1
    d(target+0x28,0);q(arena+0x9000,0)
    run(0x4A86F0,acc,character,7)
    assert not state['error'] and state['count_reads']==5
    assert rd(target+0x28)==600
    old_new=run(0x116BCC0,run(0x116BCC0,single(target,method1),single(target,method1)),single(other,method1))
    d(target+0x28,0);d(other+0x28,0);run(0x4A86F0,old_new,character,7)
    assert rd(target+0x28)==240 and rd(other+0x28)==120
    return {'schema_version':1,'build_id':BUILD,'exact_declarations':exact,'native_instructions_executed':len(seen),'native_relationships_verified':len(checks)+1,'decoded_ranges':[[hex(a),hex(b)] for a,b in ranges],'distinct_mode_callback_scores':{'old':240,'new':120},'equal_list_removal_empty_nonnull':bool(equal_list_result),'accumulation':accumulation,'score_cases_passed':len(score_cases),'score_cases':score_cases,'replayed_accumulated_callbacks':5,'replay_score':600,'ui_failure_score_preserved':120,'service_counts':services,'scope':'Native Combine/Remove wrappers, multicast implementations, delegate equality, delegate constructor, specialized clone helper, Action Invoke and multicast trampoline execute with fixture classes. Method-info resolution, runtime allocation/type/array-copy/GC stores and instance-method classification remain gateways. Native score callback uses count-provider and UI gateways. Full lifecycle runtime instances and live initialization counts remain open.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('game_root',type=Path);p.add_argument('dumper_root',type=Path);p.add_argument('--output',type=Path,required=True);a=p.parse_args();r=audit(a.game_root,a.dumper_root);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8');print(f"Passed native delegate cases and {r['score_cases_passed']} score cases")
