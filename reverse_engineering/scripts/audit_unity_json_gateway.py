"""Audit JSON module registration, IL2CPP fallback lookup and engine ToJson edge.

Serializer internals and heap-string destruction are explicit boundaries.
"""
import argparse
import hashlib
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
    root = Path(__file__).parents[1]
    manifest = json.loads((root/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    def pinned(name, digest):
        raw = (Path(game_root)/name).read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == digest.upper()
        return raw
    game = pefile.PE(data=pinned('GameAssembly.dll', manifest['inputs']['game_assembly']['sha256']), fast_load=True)
    engine = pefile.PE(data=pinned('UnityPlayer.dll', ENGINE_SHA256), fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    def decode(pe, spans):
        result = {}
        for start, end in spans:
            rows = list(cs.disasm(pe.get_data(start, end-start), start))
            assert rows[0].address == start and rows[-1].address+rows[-1].size == end
            result.update({i.address: i for i in rows})
        return result
    gi = decode(game, [(0x265BC0, 0x265F9C), (0x1CD63D0, 0x1CD6412), (0x2B7DF0, 0x2B7E1C)])
    ei = decode(engine, [(0xFA1850, 0xFA18FA), (0x191D80, 0x192164)])
    checks = [(gi, 0x1CD63F3, 'call', '0x2b7df0'), (gi, 0x2B7DF9, 'call', '0x265bc0'),
              (gi, 0x265DAE, 'mov', 'edx, 0x28'), (gi, 0x265DB6, 'call', '0x30d670'),
              (gi, 0x265DF4, 'mov', 'r8, rax'), (gi, 0x265DFB, 'call', '0x243950'),
              (gi, 0x265D4F, 'mov', 'rax, qword ptr [r15 + 0x40]'),
              (gi, 0x265F29, 'mov', 'r14, qword ptr [r15 + 0x40]'),
              (ei, 0xFA1880, 'mov', 'rsi, qword ptr [r14 + r12 + 0x18dedd8]'),
              (ei, 0xFA1891, 'mov', 'rbp, qword ptr [r14 + r12 + 0x18dede8]'),
              (ei, 0xFA18D5, 'cmp', 'r15d, 2'),
              (ei, 0x191DDD, 'je', '0x192135'), (ei, 0x191DF8, 'setne', 'r8b'),
              (ei, 0x191E08, 'call', '0xaaca50'),
              (ei, 0x191E1C, 'mov', 'edx, 0x18'), (ei, 0x191E21, 'sub', 'edx, eax'),
              (ei, 0x191E25, 'mov', 'edx, dword ptr [rbp + 7]'),
              (ei, 0x191E28, 'mov', 'rcx, qword ptr [rbp - 9]'),
              (ei, 0x192134, 'ret', ''), (ei, 0x19215E, 'call', '0x765a20')]
    for rows, address, mnemonic, operands in checks:
        assert address in rows and (rows[address].mnemonic, rows[address].op_str) == (mnemonic, operands)
    def rip(rows, address, operand=1):
        i = rows[address]; return i.address+i.size+i.operands[operand].mem.disp
    def cstring(pe, address):
        section = pe.get_section_by_rva(address)
        assert section and address-section.VirtualAddress < section.SizeOfRawData
        raw = pe.get_data(address, min(2048, section.SizeOfRawData-(address-section.VirtualAddress)))
        assert b'\0' in raw
        return raw.split(b'\0', 1)[0].decode('utf-8')
    request = cstring(game, rip(gi, 0x1CD63EC))
    assert request == 'UnityEngine.JsonUtility::ToJsonInternal(System.Object,System.Boolean)'
    name = request.split('(', 1)[0]
    bindings = []
    for i in range(2):
        target = struct.unpack('<Q', engine.get_data(0x18DEDD8+i*8, 8))[0]-engine.OPTIONAL_HEADER.ImageBase
        label = struct.unpack('<Q', engine.get_data(0x18DEDE8+i*8, 8))[0]-engine.OPTIONAL_HEADER.ImageBase
        bindings.append((cstring(engine, label), target))
    assert bindings == [(name, 0x191D80), ('UnityEngine.JsonUtility::FromJsonInternal', 0x192170)]
    # The module uses the already identified il2cpp_add_internal_call sink.
    assert rip(ei, 0xFA18C8, 0) == 0x1CD5F98
    # Independently resolve marshaling exports from their actual initialization sites.
    exports = decode(engine, [(0x76DAE5, 0x76DAF8), (0x76E4B1, 0x76E4C4)])
    assert cstring(engine, rip(exports, 0x76DAE5)) == 'il2cpp_gc_wbarrier_set_field'
    assert cstring(engine, rip(exports, 0x76E4B1)) == 'il2cpp_string_new_len'
    assert rip(exports, 0x76DAF1, 0) == rip(ei, 0x191DAC, 0)
    assert rip(exports, 0x76E4BD, 0) == rip(ei, 0x191E2C)
    def machine(pe, instructions):
        uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        base = pe.OPTIONAL_HEADER.ImageBase
        uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095)
        uc.mem_write(base, pe.get_memory_mapped_image())
        arena, stack, service, stop = 0x300000000, 0x400000000, 0x500000000, 0x500001000
        uc.mem_map(arena, 0x100000); uc.mem_map(stack, 0x20000); uc.mem_map(service, 0x2000)
        def q(a,v): uc.mem_write(a, struct.pack('<Q',v))
        def rq(a): return struct.unpack('<Q',uc.mem_read(a,8))[0]
        def ret(v=0):
            rsp=uc.reg_read(x.UC_X86_REG_RSP); uc.reg_write(x.UC_X86_REG_RAX,v)
            uc.reg_write(x.UC_X86_REG_RSP,rsp+8); uc.reg_write(x.UC_X86_REG_RIP,rq(rsp))
        registers = [x.UC_X86_REG_RBX,x.UC_X86_REG_RSI,x.UC_X86_REG_RDI,x.UC_X86_REG_RBP,
                     x.UC_X86_REG_R12,x.UC_X86_REG_R13,x.UC_X86_REG_R14,x.UC_X86_REG_R15]
        def run(entry, rcx=0, rdx=0):
            rsp=stack+0x18008; q(rsp,stop)
            for i,r in enumerate(registers): uc.reg_write(r,0xBCDE0000+i)
            for r,v in [(x.UC_X86_REG_RSP,rsp),(x.UC_X86_REG_RCX,rcx),(x.UC_X86_REG_RDX,rdx)]: uc.reg_write(r,v)
            uc.emu_start(base+entry,stop,timeout=1_000_000,count=50000)
            finished=uc.reg_read(x.UC_X86_REG_RIP)==stop
            if finished:
                assert uc.reg_read(x.UC_X86_REG_RSP)==rsp+8
                for i,r in enumerate(registers): assert uc.reg_read(r)==0xBCDE0000+i
            return uc.reg_read(x.UC_X86_REG_RAX),finished
        return uc,base,arena,service,q,rq,ret,run
    uc,base,arena,service,q,rq,ret,run = machine(game,gi)
    state={}
    def ghook(_,address,size,__):
        a=address-base; rcx,rdx,r8=[uc.reg_read(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8)]
        if a==0x243950:
            content=bytes(uc.mem_read(rdx,r8)); state['strings'].append(content.decode())
            storage=arena+0x10000+len(state['strings'])*0x1000
            uc.mem_write(storage,content+b'\0');q(rcx,storage);q(rcx+16,r8);q(rcx+24,max(r8,16));ret(rcx)
        elif a==0x30D900:
            left,right=bytes(uc.mem_read(rcx,r8)),bytes(uc.mem_read(rdx,r8))
            ret((left>right)-(left<right)&0xFFFFFFFF)
        elif a==0x30D670:
            content=bytes(uc.mem_read(rcx,r8)); index=content.find(bytes([rdx&255]));ret(0 if index<0 else rcx+index)
        elif a==0x30AF4C: ret()
        elif a not in gi: raise AssertionError(f'lookup left boundary {a:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,ghook)
    lookup=[]
    for requested, entries, expected, label in [
        (request,[(name,base+0x191D80)],base+0x191D80,'parameter fallback'),
        (request,[(name,base+0x191D80),(request,0x12345678)],0x12345678,'exact precedence'),
        (name,[(name,base+0x191D80)],base+0x191D80,'bare exact'),
        (request,[],0,'unregistered'),
        ('UnityEngine.JsonUtility::Missing(System.Object)',[(name,base+0x191D80)],0,'missing prefix'),
        ('UnityEngine.JsonUtility::Missing',[(name,base+0x191D80)],0,'missing bare'),
    ]:
        state.clear();state['strings']=[]
        sentinel=arena+0x100;uc.mem_write(sentinel,bytes(0x80));uc.mem_write(sentinel+0x19,b'\x01')
        nodes=[]
        for i,(key,value) in enumerate(sorted(entries)):
            node=arena+0x200+i*0x100; storage=arena+0x2000+i*0x100
            uc.mem_write(node,bytes(0x80));uc.mem_write(storage,key.encode()+b'\0')
            q(node+0x20,storage);q(node+0x30,len(key));q(node+0x38,len(key));q(node+0x40,value);nodes.append(node)
        for i,node in enumerate(nodes): q(node,sentinel);q(node+0x10,nodes[i+1] if i+1<len(nodes) else sentinel)
        q(sentinel+8,nodes[0] if nodes else sentinel);q(base+rip(gi,0x265C16),sentinel)
        uc.mem_write(arena+0x4000,requested.encode()+b'\0')
        value,done=run(0x265BC0,arena+0x4000);assert done and value==expected
        if label=='parameter fallback': assert state['strings']==[request,request,name]
        if label=='exact precedence': assert state['strings']==[request]
        lookup.append({'case':label,'constructed_keys':state['strings'],'resolved':bool(value)})
    uc,base,arena,service,q,rq,ret,run = machine(engine,ei)
    state={}
    q(base+rip(ei,0xFA18C8,0),service)
    q(base+rip(ei,0xFA188A,0),0)
    q(base+rip(ei,0x191DAC,0),service+0x10)
    q(base+rip(ei,0x191E2C),service+0x20)
    def ehook(_,address,size,__):
        a=address-base;rcx,rdx,r8=[uc.reg_read(r) for r in (x.UC_X86_REG_RCX,x.UC_X86_REG_RDX,x.UC_X86_REG_R8)]
        if address==service:
            state['registration'].append((cstring(engine,rcx-base),rdx-base));ret()
        elif address==service+0x10:
            assert rcx==0;q(rdx,r8);state['barriers'].append(r8);ret()
        elif a==0xAACA50:
            assert rcx==state['object'] and r8&255==bool(state['pretty'])
            state['serializer_calls']+=1
            if state['failure']=='serializer': state['stop']='serializer';uc.emu_stop();return
            payload=state['payload']
            if state['heap']:
                uc.mem_write(arena+0x6000,payload);q(rdx,arena+0x6000)
                uc.mem_write(rdx+16,struct.pack('<I',len(payload)));uc.mem_write(rdx+32,b'\0')
            else:
                uc.mem_write(rdx,payload);uc.mem_write(rdx+24,bytes([24-len(payload)]));uc.mem_write(rdx+32,b'\x01')
            ret()
        elif address==service+0x20:
            assert bytes(uc.mem_read(rcx,rdx))==state['payload']
            state['string_calls']+=1;ret(arena+0x7000)
        elif a==0x191E52:
            state['stop']='heap cleanup';uc.emu_stop()
        elif a==0x765D10:
            assert cstring(engine,rdx-base)=='obj'
            state['null_parameter']='obj';q(rcx,0xE001);q(rcx+8,0);ret(rcx)
        elif a==0x45AA0:
            uc.mem_write(rcx,bytes(uc.mem_read(rdx,16)));ret(rcx)
        elif a==0x765A20:
            assert rq(rcx)==0xE001;state['stop']='null argument';uc.emu_stop()
        elif a not in ei: raise AssertionError(f'engine left boundary {a:x}')
    uc.hook_add(unicorn.UC_HOOK_CODE,ehook)
    state['registration']=[]
    _,done=run(0xFA1850);assert done and state['registration']==bindings
    engine_cases=[]
    for pretty in (0,1,2,255):
        for mode,payload in [('empty',b''),('short',b'{"a":1}'),('capacity',b'x'*24),('heap',b'x'*40),('failure',b''),('null',b'')]:
            state.clear();state.update(object=0 if mode=='null' else arena+0x5000,pretty=pretty,payload=payload,heap=mode=='heap',
                failure='serializer' if mode=='failure' else None,barriers=[],serializer_calls=0,string_calls=0)
            value,done=run(0x191D80,state['object'],pretty)
            expected_stop={'heap':'heap cleanup','failure':'serializer','null':'null argument'}.get(mode)
            assert state.get('stop')==expected_stop and done==(expected_stop is None)
            assert state['serializer_calls']==int(mode!='null')
            assert state['string_calls']==int(mode not in ('null','failure'))
            assert state['barriers'][:2]==[state['object'],state['object']]
            if done: assert value==arena+0x7000 and state['barriers'][-1]==value
            engine_cases.append({'mode':mode,'pretty_byte':pretty,'normalized_pretty':bool(pretty),'serializer_calls':state['serializer_calls'],
                                 'string_calls':state['string_calls'],'stop_boundary':expected_stop})
    return {'schema_version':1,'build_id':BUILD,'engine_sha256':ENGINE_SHA256,
            'game_assembly_sha256':manifest['inputs']['game_assembly']['sha256'],'native_checks':len(checks),
            'request':request,'module_registration':'0xfa1850','bindings':{n:hex(a) for n,a in bindings},
            'lookup_cases':lookup,'engine_cases':engine_cases,'native_case_count':len(lookup)+len(engine_cases)+1,
            'scope':'Native lookup tree/fallback, two-entry JSON module registration, and engine ToJson boundary. String primitives, GC barrier, managed string construction, exception creation/raise and serializer are services. Heap-string cleanup stops before allocator policy; serializer internals remain open.'}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('game_root',type=Path);parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();report=audit(args.game_root)
    args.output.write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {report['native_case_count']} JSON gateway cases")
