"""Execute engine FromJson dispatch; parsing, field application and runtime remain services."""
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
    assert unicorn.__version__ == '2.1.4'
    raw=(Path(game_root)/'UnityPlayer.dll').read_bytes()
    assert hashlib.sha256(raw).hexdigest().upper()==ENGINE_SHA256.upper()
    pe=pefile.PE(data=raw,fast_load=True);base=pe.OPTIONAL_HEADER.ImageBase
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64);cs.detail=True
    ins={}
    for start,end in [(0x192170,0x1925B8),(0xAACEE0,0xAAD64B),(0x75A3A0,0x75A460)]:
        rows=list(cs.disasm(pe.get_data(start,end-start),start));assert rows[-1].address+rows[-1].size==end
        ins.update({i.address:i for i in rows})
    checks=[(0x19223C,'call','rax'),(0x19228C,'mov','r9, rdi'),(0x19228F,'mov','r8, rbx'),
            (0x19229F,'call','0xaacee0'),(0xAACF9A,'call','rax'),(0xAACFC2,'jne','0xaad051'),
            (0xAACFD9,'call','0xaacc80'),(0xAAD004,'call','0x75a3a0'),(0xAAD01E,'call','0xa8e030'),
            (0xAAD05B,'call','rax'),(0xAAD08F,'call','0xa8e030'),(0xAAD0F0,'call','0x765db0'),
            (0xAAD118,'mov','qword ptr [r13], r15'),(0xAAD5EB,'mov','qword ptr [r13], rax'),
            (0x75A3D3,'mov','qword ptr [rbx], rcx'),(0x75A3E0,'call','qword ptr [rip + 0x157beca]'),
            (0x1925B2,'call','0x765a20'),(0xAAD64A,'ret',''),(0x19259B,'ret','')]
    for address,mnemonic,operands in checks:assert (ins[address].mnemonic,ins[address].op_str)==(mnemonic,operands)
    def rip(address,operand=1):
        i=ins[address];return i.address+i.size+i.operands[operand].mem.disp
    def cstr(address):
        sec=pe.get_section_by_rva(address);assert sec and address-sec.VirtualAddress<sec.SizeOfRawData
        b=pe.get_data(address,min(2048,sec.SizeOfRawData-(address-sec.VirtualAddress)));assert b'\0' in b
        return b.split(b'\0',1)[0].decode()
    exports=[(0x76CFC3,0x76CFCF,'il2cpp_class_from_system_type',rip(0x192235)),
             (0x76D4A9,0x76D4B5,'il2cpp_class_get_rank',rip(0xAACF6D)),
             (0x76E16D,0x76E179,'il2cpp_object_get_class',rip(0xAAD051)),
             (0x76E1DF,0x76E1EB,'il2cpp_object_new',rip(0x75A3A6)),
             (0x76E3F3,0x76E3FF,'il2cpp_runtime_object_init_exception',rip(0x75A3E0,0))]
    for lea,store,name,slot in exports:
        a,b=list(cs.disasm(pe.get_data(lea,7),lea)),list(cs.disasm(pe.get_data(store,7),store))
        assert len(a)==len(b)==1 and a[0].mnemonic=='lea' and b[0].mnemonic=='mov'
        assert cstr(lea+7+a[0].operands[1].mem.disp)==name
        assert store+7+b[0].operands[0].mem.disp==slot
    rank_error=cstr(rip(0xAACFA6));assert rank_error=='Return type must represent an object type. Received an array.'
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64)
    uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095);uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,service,stop=0x300000000,0x400000000,0x500000000,0x500001000
    uc.mem_map(arena,0x100000);uc.mem_map(stack,0x20000);uc.mem_map(service,0x2000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def rq(a):return struct.unpack('<Q',uc.mem_read(a,8))[0]
    def reg(r):return uc.reg_read(r)
    def ret(v=0):
        rsp=reg(x.UC_X86_REG_RSP);uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,rsp+8);uc.reg_write(x.UC_X86_REG_RIP,rq(rsp))
    state={}
    def write_string(address,content):
        uc.mem_write(address,bytes(40))
        if len(content)<=24:
            uc.mem_write(address,content+b'\0');uc.mem_write(address+24,bytes([24-len(content)]));uc.mem_write(address+32,b'\x01')
        else:
            state['native_allocations']+=1;buffer=arena+0x20000+state['native_allocations']*0x100
            uc.mem_write(buffer,content+b'\0');q(address,buffer);q(address+16,len(content));uc.mem_write(address+32,b'\0')
        uc.mem_write(address+36,struct.pack('<I',1))
    def string_bytes(address):
        mode=uc.mem_read(address+32,1)[0]
        if mode==1:return bytes(uc.mem_read(address,24-struct.unpack('<b',uc.mem_read(address+24,1))[0]))
        return bytes(uc.mem_read(rq(address),rq(address+16)))
    slots={rip(0x1921BF,0):service,rip(0x192235):service+0x10,rip(0xAACF6D):service+0x20,
           rip(0xAAD051):service+0x30,rip(0x75A3A6):service+0x40,rip(0x75A3E0,0):service+0x50}
    for slot,target in slots.items():q(base+slot,target)
    manager,allocator,vtable=arena+0x8000,arena+0xA000,arena+0xB000
    q(base+rip(0xAAD02B),manager);uc.mem_write(manager+0x1836,b'\0');q(allocator,vtable);q(vtable+0x18,service+0x60)
    input_json,overwrite,requested_type,requested_class,actual_class,allocated,tree= [arena+n for n in range(0x1000,0x8000,0x1000)]
    def hook(_,address,size,__):
        a=address-base;rcx,rdx,r8,r9=[reg(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8,x.UC_X86_REG_R9)]
        if address==service:
            assert rcx==0;q(rdx,r8);state['stores'].append(r8);ret()
        elif address==service+0x10:
            assert rcx==requested_type;state['events'].append('type');ret(requested_class)
        elif address==service+0x20:
            assert rcx==(requested_class if state['type_present'] else 0)
            state['events'].append('rank')
            if not rcx:state['stop']='null class passed to runtime';uc.emu_stop()
            else:ret(state['rank'])
        elif address==service+0x30:
            assert rcx==overwrite;state['events'].append('actual class');ret(actual_class)
        elif address==service+0x40:
            assert rcx==requested_class;state['events'].append('allocate');ret(allocated)
        elif address==service+0x50:
            assert rcx==allocated;state['events'].append('initialize')
            if state['mode']=='constructor error':q(rdx,0xBAD1)
            ret()
        elif a==0x75A3F6:
            assert rq(reg(x.UC_X86_REG_RBX))==allocated
            state['allocation_published_before_constructor_error']=True
            state['stop']='constructor error handling';uc.emu_stop()
        elif a==0x4A86B0:
            assert rdx==input_json;state['events'].append('convert');write_string(rcx,state['json']);ret(rcx)
        elif a==0x14F740:
            write_string(rcx,string_bytes(rdx));ret(rcx)
        elif a==0x159230:
            write_string(rcx,bytes(uc.mem_read(rdx,r8)));ret(rcx)
        elif a==0xAACC80:
            text=bytes(uc.mem_read(rcx,256)).split(b'\0',1)[0]
            assert text==state['json']
            state['events'].append('parse');state['error_string']=r8
            if state['mode']=='parse error':write_string(r8,b'bad json');ret()
            elif state['mode']=='no tree':ret()
            else:ret(tree)
        elif a==0xA8E030:
            assert rcx==tree and rdx==(overwrite if state['overwrite'] else allocated)
            assert r8==(actual_class if state['overwrite'] else requested_class) and rq(r9)==0
            state['events'].append('apply');q(rdx+0x80,0xA991)
            if state['mode']=='apply failure':state['stop']='field application';uc.emu_stop()
            else:ret()
        elif a==0xAAC9B0:assert rcx==tree;state['events'].append('destroy parse tree');ret()
        elif a==0x354EC0:
            assert rcx==manager and r8==1;state['events'].append('free parse tree' if rdx==tree else 'free native string');ret()
        elif a==0x355150:assert rcx==manager;ret(allocator)
        elif address==service+0x60:assert rcx==allocator;state['events'].append('free native string');ret()
        elif a==0x765DB0:
            assert cstr(rdx-base)=='%s';state['events'].append('make error')
            q(rcx,0xBAD2);q(rcx+8,0);ret(rcx)
        elif a==0x47B50:state['events'].append('cleanup wrapper');ret()
        elif a==0x765A20:
            assert rq(rcx)==0xBAD2;state['stop']='raise';uc.emu_stop()
        elif a not in ins:raise AssertionError(f'left dispatch boundary {a:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    cases=[]
    regs=[x.UC_X86_REG_RBX,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_RBP,x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
    for ow,payload,type_present,rank,mode in itertools.product((False,True),(None,b'',b'{"a":1}'),(False,True),(0,1,2),('normal','no tree','parse error','constructor error','apply failure')):
        json_present=payload is not None
        state.clear();state.update(overwrite=ow,type_present=type_present,rank=rank,mode=mode,json=payload or b'',native_allocations=0,events=[],stores=[])
        q(overwrite+0x80,0);q(allocated+0x80,0)
        rsp=stack+0x18008;q(rsp,stop)
        for i,r in enumerate(regs):uc.reg_write(r,0xBCDE0000+i)
        for r,v in [(x.UC_X86_REG_RSP,rsp),(x.UC_X86_REG_RCX,input_json if json_present else 0),
                    (x.UC_X86_REG_RDX,overwrite if ow else 0),(x.UC_X86_REG_R8,requested_type if type_present else 0)]:uc.reg_write(r,v)
        uc.emu_start(base+0x192170,stop,timeout=1_000_000,count=100000)
        expected_stop='null class passed to runtime' if not type_present else ('raise' if rank or mode=='parse error' else
                       ('constructor error handling' if not ow and mode=='constructor error' else ('field application' if mode=='apply failure' else None)))
        assert state.get('stop')==expected_stop,(state,expected_stop)
        assert state['stores'][:6]==[input_json if json_present else 0]*2+[overwrite if ow else 0]*2+[requested_type if type_present else 0]*2
        events=state['events'];assert ('convert' in events)==json_present
        reaches_parse=type_present and rank==0
        assert ('parse' in events)==reaches_parse
        allocates=reaches_parse and not ow and mode not in ('parse error','no tree')
        assert ('allocate' in events)==allocates
        if allocates:assert events.index('parse')<events.index('allocate')<events.index('initialize')
        if reaches_parse and ow:assert events.index('actual class')<events.index('parse')
        if not expected_stop:
            assert reg(x.UC_X86_REG_RIP)==stop and reg(x.UC_X86_REG_RSP)==rsp+8
            for i,r in enumerate(regs):assert reg(r)==0xBCDE0000+i
            assert reg(x.UC_X86_REG_RAX)==(overwrite if ow else (0 if mode=='no tree' else allocated))
        mutated=rq((overwrite if ow else allocated)+0x80)==0xA991
        assert mutated==('apply' in events)
        if expected_stop=='field application':assert 'destroy parse tree' not in events and mutated
        cases.append({'overwrite':ow,'json_present':json_present,'json_length':len(payload or b''),'type_present':type_present,'rank':rank,'mode':mode,
                      'events':events,'stop_boundary':expected_stop,'object_mutated':mutated,
                      'allocation_published_before_constructor_error':state.get('allocation_published_before_constructor_error',False)})
    return {'schema_version':1,'build_id':BUILD,'engine_sha256':ENGINE_SHA256,'native_checks':len(checks),
            'export_identity_checks':len(exports),'native_case_count':len(cases),'cases':cases,
            'registration_evidence':'unity_json_gateway audit; no additional registration cases counted',
            'scope':'Native FromJson wrapper, create/overwrite dispatcher and allocation/init helper. Conversion/string ownership, class/runtime exports, parsing, field application, GC, exception machinery and allocator services are gateways. Constructor error path stops before reporting policy; null class stops at runtime rank service. No JSON field semantics or exception unwinding is claimed.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('game_root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.game_root);args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {report['native_case_count']} FromJson gateway cases")
