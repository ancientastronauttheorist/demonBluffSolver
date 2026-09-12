"""Audit the pinned mask-16 wait dispatch lifecycle with explicit callee gateways."""
import argparse
import itertools
import json
import struct
from pathlib import Path
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


def audit(path):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    if unicorn.__version__ != '2.1.4': raise ValueError('Unicorn 2.1.4 required')
    raw = path.read_bytes(); digest = verify_fingerprint(raw, ENGINE_SHA256)
    pe = pefile.PE(data=raw, fast_load=True)
    pe.parse_data_directories(directories=[3])
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    decoded = {}
    for begin, end in [(0x59F5E0, 0x59F743), (0x5C62D0, 0x5C64BF), (0xBD81B0, 0xBD8535)]:
        ins = list(cs.disasm(pe.get_data(begin, end-begin), begin))
        assert ins and ins[-1].address + ins[-1].size == end
        decoded.update({i.address:i for i in ins})
    checks = [
        (0x59F629,'test','rdi, rdi'), (0x59F62C,'je','0x59f65d'),
        (0x59F676,'test','rcx, rcx'), (0x59F679,'je','0x59f683'),
        (0x59F68A,'mov','edx, 0x10'), (0x59F68F,'mov','rax, qword ptr [rcx]'),
        (0x59F692,'call','qword ptr [rax + 0xb8]'),
        (0x59F6B9,'cmp','r14d, 2'), (0x59F6C5,'je','0x59f70d'),
        (0x59F720,'movsd','xmm0, qword ptr [rax + 0x60]'),
        (0x59F72C,'movsd','qword ptr [rax + 0xf0], xmm0'),
        (0x59F734,'mov','word ptr [rax + 0xc0], 0x101'),
        (0x5C6349,'mov','ecx, dword ptr [rdi + 0x3e0]'),
        (0x5C634F,'test','ecx, ecx'), (0x5C6351,'je','0x5c63dd'),
        (0x5C6357,'sub','ecx, 1'), (0x5C635A,'je','0x5c636a'),
        (0x5C635C,'sub','ecx, 1'), (0x5C635F,'je','0x5c638e'),
        (0x5C6361,'cmp','ecx, 3'), (0x5C6364,'jne','0x5c6408'),
        (0x5C63B4,'call','0x59f5e0'), (0x5C6403,'call','0x59f5e0'),
        (0xBD8217,'cmp','byte ptr [rax + 0x78], 0'),
        (0xBD821B,'je','0xbd8518'),
        (0xBD84C6,'cmp','dword ptr [r12 + 8], 0'),
        (0xBD84EA,'jne','0xbd84f6'), (0xBD84EC,'xor','edx, edx'),
        (0xBD84F1,'call','0x59f5e0')]
    for address,mnemonic,operands in checks:
        assert address in decoded, hex(address)
        i=decoded[address]; assert (i.mnemonic,i.op_str)==(mnemonic,operands), hex(address)
    for address,target in [(0x59F683,0x1C6E720),(0x59F719,0x1C6E718),(0x59F725,0x1A74E90)]:
        i=decoded[address]
        refs=[i.address+i.size+o.mem.disp for o in i.operands if o.type==capstone.CS_OP_MEM and o.mem.base==capstone.x86.X86_REG_RIP]
        assert refs==[target]
    assert pe.get_data(0x1A74E90,16)==struct.pack('<QQ',1<<63,1<<63)
    chunks=[(e.struct.BeginAddress,e.struct.EndAddress) for e in pe.DIRECTORY_ENTRY_EXCEPTION if 0x59F5E0<=e.struct.BeginAddress<0x59F743]
    assert chunks==[(0x59f5e0,0x59f5e9),(0x59f5e9,0x59f62e),(0x59f62e,0x59f65d),(0x59f65d,0x59f67b),(0x59f67b,0x59f6c7),(0x59f6c7,0x59f743)]
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64)
    uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095); uc.mem_write(base,pe.get_memory_mapped_image())
    arena,stack,stop=0x40000000,0x50000000,0x60000000
    for a in (arena,stack,stop):uc.mem_map(a,0x10000)
    def q(a,v):uc.mem_write(a,struct.pack('<Q',v))
    def ret(v=0):
        rsp=uc.reg_read(x.UC_X86_REG_RSP); target=struct.unpack('<Q',uc.mem_read(rsp,8))[0]
        uc.reg_write(x.UC_X86_REG_RAX,v);uc.reg_write(x.UC_X86_REG_RSP,rsp+8);uc.reg_write(x.UC_X86_REG_RIP,target)
    visited=set(); events=[]
    gateway_calls={0x3A0BC0,0x466800,0x67F230,0x3E5AD0,0x103CA30,0x14EA00,0x595C30}
    def hook(_uc,a,size,_data):
        if a==stop:uc.emu_stop();return
        if a==base+0x6F25C0:events.append('service_get');ret(arena+0x1000);return
        if a-base in gateway_calls:
            events.append(hex(a-base));ret();return
        if stop+0x100<=a<=stop+0x400:
            events.append({'gateway':hex(a-stop),'arg':uc.reg_read(x.UC_X86_REG_RDX)&0xffffffff})
            ret();return
        assert 0x59F5E0<=a-base<0x59F743, hex(a)
        visited.add(a-base)
    uc.hook_add(unicorn.UC_HOOK_CODE,hook)
    results=[]
    for tree,service,mode,time in itertools.product((False,True),(False,True),(0,1,2,5),(0.0,123.25)):
        uc.mem_write(arena,bytes(0x10000));events.clear()
        for glob,ptr in [(0x1BE0800,arena+0x2000 if tree else 0),(0x1C6E728,arena+0x3000 if service else 0),(0x1C6E720,arena+0x4000),(0x1C6E718,arena+0x5000),(0x1CD60D0,stop+0x300)]:q(base+glob,ptr)
        for obj,vt,slot,fn in [(arena+0x1000,arena+0x1100,8,stop+0x100),(arena+0x3000,arena+0x3100,0x10,stop+0x200),(arena+0x4000,arena+0x4100,0xB8,stop+0x400)]:q(obj,vt);q(vt+slot,fn)
        q(arena+0x2000,arena+0x2100);q(arena+0x2008,9)
        uc.mem_write(arena+0x5060,struct.pack('<d',time))
        q(stack+0x8000,stop);uc.reg_write(x.UC_X86_REG_RSP,stack+0x8000)
        uc.reg_write(x.UC_X86_REG_RCX,arena);uc.reg_write(x.UC_X86_REG_RDX,mode)
        uc.emu_start(base+0x59F5E0,stop,count=1000)
        assert uc.reg_read(x.UC_X86_REG_RIP)==stop
        assert [e for e in events if isinstance(e,dict) and e['gateway']=='0x400']==[{'gateway':'0x400','arg':16}]
        assert ('0x595c30' in events)==(mode!=2)
        assert sum(isinstance(e,dict) and e['gateway']=='0x200' for e in events)==int(service)
        assert ('0x67f230' in events)==tree
        assert uc.mem_read(arena+0x50F0,8)==struct.pack('<d',-time)
        assert uc.mem_read(arena+0x50C0,2)==b'\1\1'
        if tree:
            assert struct.unpack('<QQQ',uc.mem_read(arena+0x2100,24))==(arena+0x2100,)*3
            assert uc.mem_read(arena+0x2008,8)==bytes(8)
        results.append({'tree_present':tree,'optional_service_present':service,'mode':mode,'clock_time':time,'dispatches':1,'mask':16})
    return {'schema_version':1,'engine_sha256':digest,'native_relationships_verified':len(checks)+4,'function_rva':'0x59F5E0','function_end_exclusive':'0x59F743','unwind_chunks':[[hex(a),hex(b)] for a,b in chunks],'distinct_native_instructions_executed':len(visited),'cases':results,'direct_callers':[{'function':'0x5C62D0','sites':['0x5C63B4','0x5C6403'],'gate':'mode at input+0x3E0 equals 0 or 2'},{'function':'0xBD81B0','sites':['0xBD84F1'],'gate':'within existing object+0x78 guard, mode at second argument+8 equals 0'}],'scope':'Native enclosing routine executes with all callees mocked. Mask16 is unconditional inside this routine; mode2 skips a later notification only. Native stores reset tree links when supplied and set clock+F0 to negated clock+60 and clock+C0/C1 to1. Caller modes have no assigned public identity. Not whole caller emulation, no live scheduling timestamps, no phase8 resolution.'}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('unityplayer',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();r=audit(a.unityplayer);a.output.write_text(json.dumps(r,indent=2)+'\n',encoding='utf-8')
    print(f"Verified {r['native_relationships_verified']} relationships and {len(r['cases'])} native lifecycle cases")
