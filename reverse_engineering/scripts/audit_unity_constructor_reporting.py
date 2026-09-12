"""Audit captured-constructor-error reporting and return policy, with constructor/log services."""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path
from audit_character_assets import BUILD
from audit_unityplayer_wait import ENGINE_SHA256


def audit(game_root):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    assert unicorn.__version__=='2.1.4'
    root=Path(__file__).parents[1]
    manifest=json.loads((root/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    def pinned(name,digest):
        raw=(Path(game_root)/name).read_bytes();assert hashlib.sha256(raw).hexdigest().upper()==digest.upper();return raw
    ep=pefile.PE(data=pinned('UnityPlayer.dll',ENGINE_SHA256),fast_load=True)
    gp=pefile.PE(data=pinned('GameAssembly.dll',manifest['inputs']['game_assembly']['sha256']),fast_load=True)
    gp.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXPORT']])
    assert [(e.name,e.address) for e in gp.DIRECTORY_ENTRY_EXPORT.symbols if e.name==b'il2cpp_runtime_object_init_exception']==[(b'il2cpp_runtime_object_init_exception',0x281DD0)]
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    def decode(pe,spans):
        result={}
        for start,end in spans:
            rows=list(cs.disasm(pe.get_data(start,end-start),start));assert rows[-1].address+rows[-1].size==end
            result.update({i.address:i for i in rows})
        return result
    ei=decode(ep,[(0x75A3A0,0x75A460),(0x820F70,0x82104F),(0x821050,0x8210A5),(0x75D3A0,0x75D44C),(0x7C17F0,0x7C1918)])
    gi=decode(gp,[(0x281DD0,0x281DD5),(0x2E5DF0,0x2E5E50),(0x2C4870,0x2C4875),(0x2E5310,0x2E5379)])
    # Decode the cache initializer from its verified entry before inspecting method stores.
    cache=decode(ep,[(0x81A880,0x81C539)])
    def rip(rows,address,operand=1):
        i=rows[address];return i.address+i.size+i.operands[operand].mem.disp
    def string(pe,address):
        section=pe.get_section_by_rva(address);assert section and address-section.VirtualAddress<section.SizeOfRawData
        b=pe.get_data(address,min(2048,section.SizeOfRawData-(address-section.VirtualAddress)));assert b'\0' in b
        return b.split(b'\0',1)[0].decode()
    assert string(ep,rip(cache,0x81C4C5))=='CallOverridenDebugHandler'
    assert string(ep,rip(cache,0x81C4FF))=='IsLoggingEnabled'
    assert string(ep,rip(cache,0x81C4AA))=='Debug'
    assert string(gp,rip(gi,0x2E5E08))=='.ctor'
    for lea,store,name,slot in [(0x76DFA5,0x76DFB1,'il2cpp_method_get_param_count',rip(ei,0x7C186D,0)),
                                (0x76E35B,0x76E367,'il2cpp_runtime_invoke',rip(ei,0x7C1873))]:
        rows=decode(ep,[(lea,store+7)])
        assert string(ep,rip(rows,lea))==name and rip(rows,store,0)==slot
    checks=[(ei,0x75A3D3,'mov','qword ptr [rbx], rcx'),(ei,0x75A428,'call','0x820f70'),
            (ei,0x75A42F,'jne','0x75a452'),(ei,0x75A431,'call','0x821050'),(ei,0x75A438,'je','0x75a452'),
            (ei,0x75A44D,'call','0x75b4c0'),(ei,0x75A457,'mov','rax, rbx'),(ei,0x75A45F,'ret',''),
            (ei,0x820FAE,'mov','rdx, qword ptr [rdx + 0x2f8]'),(ei,0x821082,'mov','rdx, qword ptr [rdx + 0x300]'),
            (ei,0x7C1889,'call','rax'),(ei,0x7C1900,'movzx','eax, byte ptr [rdi + 0x10]'),
            (ei,0x7C190D,'xor','al, al'),(ei,0x7C18E9,'call','0x75be00'),
            (gi,0x281DD0,'jmp','0x2e5df0'),(gi,0x2E5E12,'call','0x2f15e0'),
            (gi,0x2E5E1E,'cmp','dword ptr [r8 + 0x28], 0'),(gi,0x2C4870,'lea','rax, [rcx + 0x10]'),
            (gi,0x2E5E30,'mov','r9, rsi'),(gi,0x2E5E33,'xor','r8d, r8d'),(gi,0x2E5E4B,'jmp','0x2e5310'),
            (gi,0x2E5332,'mov','qword ptr [r9], 0'),(gi,0x2E535F,'call','0x2e5860'),
            (cache,0x81C4F8,'mov','qword ptr [rcx + 0x2f8], rax'),(cache,0x81C532,'mov','qword ptr [rcx + 0x300], rax')]
    for rows,a,m,o in checks:assert (rows[a].mnemonic,rows[a].op_str)==(m,o)
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64)
    eb,gb=ep.OPTIONAL_HEADER.ImageBase,0x600000000
    for pe,base in [(ep,eb),(gp,gb)]:
        uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,service,stop=0x300000000,0x400000000,0x500000000,0x500001000
    uc.mem_map(arena,0x100000);uc.mem_map(stack,0x20000);uc.mem_map(service,0x2000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def reg(r):return uc.reg_read(r)
    def ret(v=0):
        rsp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,rsp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(rsp))
    klass,obj,output,ctor,handler,enabled,boxed,exception,cache_object=[arena+n for n in range(0x1000,0xA000,0x1000)]
    q(obj,klass);q(ctor+0x20,klass)
    q(eb+rip(ei,0x820F83),cache_object);q(cache_object+0x2F8,handler);q(cache_object+0x300,enabled)
    bindings={rip(ei,0x75A3A6):service,rip(ei,0x75A3C8,0):service+0x10,
              rip(ei,0x75A3E0,0):gb+0x281DD0,rip(ei,0x7C186D,0):service+0x20,rip(ei,0x7C1873):service+0x30}
    for address,value in bindings.items():q(eb+address,value)
    state={}
    def hook(_,address,size,__):
        rcx,rdx,r8,r9=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        a=address-eb;g=address-gb
        if address==service:
            assert rcx==klass;state['events'].append('allocate');ret(obj if state['allocates'] else 0)
        elif address==service+0x10:
            assert rcx==0;q(rdx,r8);ret()
        elif g==0x2F15E0:
            assert rcx==klass and r8==0 and string(gp,rdx-gb)=='.ctor'
            assert rq(output)==obj;state['events'].append('lookup zero-argument ctor');ret(ctor)
        elif g==0x2E5310:
            assert rcx==ctor and rdx==obj+(16 if state['value_type'] else 0) and r8==0
            assert rq(output)==obj;state['events'].append('invoke ctor')
            # Constructor invocation/capture is an explicit service, not C++ unwinding.
            q(r9,exception if state['error'] else 0);q(obj+0x80,0xC701);ret()
        elif address==service+0x20:
            assert rcx in (handler,enabled);ret(2 if rcx==handler else 0)
        elif address==service+0x30:
            assert rcx in (handler,enabled) and rdx==0
            if rcx==handler:
                assert rq(r8)==exception and rq(r8+8)==0
                state['events'].append('override handler');result=state['handled'];fails=state['callback_failure']=='handler'
            else:
                state['events'].append('logging enabled');result=state['logging'];fails=state['callback_failure']=='enabled'
            uc.mem_write(boxed+16,bytes([int(result)]));q(r9,exception if fails else 0)
            ret(boxed)
        elif a==0x75BE00:
            assert rq(rcx)==exception and r8==0 and r9&255==1
            state['events'].append('report debug callback error');ret()
        elif a==0x75B4C0:
            assert rq(rcx)==exception and rdx==0 and r8==0 and r9&255==1
            assert rq(output)==obj;state['events'].append('fallback report')
            if state['reporter_failure']:state['stop']='reporter';uc.emu_stop()
            else:ret()
        elif a in ei or g in gi:pass
        else:raise AssertionError(f'left reporting boundary {address:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    cases=[]
    regs=[x.UC_X86_REG_RBX,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_RBP,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
    for allocates,value_type,error,handled,logging,callback_failure,reporter_failure in itertools.product(
            (False,True),(False,True),(False,True),(False,True),(False,True),(None,'handler','enabled'),(False,True)):
        state.clear();state.update(allocates=allocates,value_type=value_type,error=error,handled=handled,logging=logging,
                                   callback_failure=callback_failure,reporter_failure=reporter_failure,events=[])
        uc.mem_write(klass+0x28,struct.pack('<I',0x80000000 if value_type else 0));q(output,0xBAD0);q(obj+0x80,0)
        rsp=stack+0x18008;q(rsp,stop)
        for i,r in enumerate(regs):uc.reg_write(r,0xBCDE0000+i)
        uc.reg_write(x.UC_X86_REG_XMM6,0x112233445566778899AABBCCDDEEFF00)
        for r,v in [(x.UC_X86_REG_RSP,rsp),(x.UC_X86_REG_RCX,output),(x.UC_X86_REG_RDX,klass)]:uc.reg_write(r,v)
        uc.emu_start(eb+0x75A3A0,stop,timeout=1_000_000,count=10000)
        events=state['events'];active=allocates and error
        effective_handled=handled and callback_failure!='handler'
        queried=active and not effective_handled
        fallback=queried and logging and callback_failure!='enabled'
        assert ('invoke ctor' in events)==allocates
        assert ('override handler' in events)==active
        assert ('logging enabled' in events)==queried
        assert ('fallback report' in events)==fallback
        expected_stop=fallback and reporter_failure
        assert bool(state.get('stop'))==expected_stop
        assert rq(output)==(obj if allocates else 0)
        assert (rq(obj+0x80)==0xC701)==allocates
        if not expected_stop:
            assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==rsp+8 and reg(x.UC_X86_REG_RAX)==output
            for i,r in enumerate(regs):assert reg(r)==0xBCDE0000+i
            assert reg(x.UC_X86_REG_XMM6)==0x112233445566778899AABBCCDDEEFF00
        cases.append({'allocation_nonnull':allocates,'value_type':value_type,'ctor_captured_error':error,'handler_result':handled,
                      'logging_enabled':logging,'callback_failure':callback_failure,'reporter_failure':reporter_failure,
                      'events':events,'returns_published_allocation':not expected_stop,'stop_boundary':state.get('stop')})
    return {'schema_version':1,'build_id':BUILD,'engine_sha256':ENGINE_SHA256,'native_checks':len(checks),'native_case_count':len(cases),'cases':cases,
            'policy':'Captured constructor error is offered to Debug.CallOverridenDebugHandler(exception,null), then IsLoggingEnabled if unhandled, then fallback reporting if enabled. Returning services lead to the published allocation without local rethrow/reset.',
            'scope':'Native allocation helper, runtime constructor-selection/unboxing, Debug wrappers and bool-invoke bridge. Constructor invocation/capture, reflection metadata, final reporter and GC are services; no managed constructor body, C++ exception unwinding, or final logger implementation is claimed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('game_root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.game_root);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {report['native_case_count']} constructor reporting cases")
