"""Audit all direct Assembly-CSharp void aliases of the shared return stub.

Uses the independently regenerated method denominator, a pinned native body and
isolated execution. Classifications are a separate authored overlay; this tool
never changes them or claims complete caller/runtime behavior.
"""
import argparse
import hashlib
import json
import random
import re
import struct
from pathlib import Path

from build_method_coverage import create_outputs, compare_file

BUILD = "f530404b0f3f_807de4a83df4"
BODY = "ga:rva:0033ED50"
RVA = 0x33ED50


def select_definitions(methods):
    definitions = []
    for method in methods:
        if method["implementation"] != "concrete" or method["native"] != [{"body":BODY,"binding":"direct"}]:
            continue
        signature = method["symbol_key"].split("::",1)[1]
        if not re.search(r"\bvoid\s+[^()]+$",signature.split("(",1)[0]):
            raise ValueError("shared return body has an unsupported non-void definition")
        definitions.append({"id":method["id"],"declaring_type":method["declaring_type"],
                            "symbol_key":method["symbol_key"]})
    return definitions


def emulate(data):
    import pefile
    import unicorn
    from unicorn import x86_const as x

    if unicorn.__version__ != "2.1.4":
        raise ValueError("This audit requires Unicorn 2.1.4")
    pe=pefile.PE(data=data,fast_load=True)
    base=pe.OPTIONAL_HEADER.ImageBase
    uc=unicorn.Uc(unicorn.UC_ARCH_X86,unicorn.UC_MODE_64)
    uc.mem_map(base,(pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095)
    uc.mem_write(base,pe.get_memory_mapped_image())
    stack,heap,stop=0x200000000,0x300000000,0x400000000
    uc.mem_map(stack,0x10000); uc.mem_map(heap,0x1000); uc.mem_map(stop,0x1000)
    seen=[]
    def on_code(_,address,size,__):
        if address != base+RVA or size != 3:
            raise ValueError("execution left the exact shared-return body")
        seen.append(address-base)
    uc.hook_add(unicorn.UC_HOOK_CODE,on_code)
    gprs=[getattr(x,"UC_X86_REG_"+name) for name in
          ("RAX","RCX","RDX","RBX","RBP","RSI","RDI","R8","R9","R10","R11","R12","R13","R14","R15")]
    xmms=[getattr(x,f"UC_X86_REG_XMM{i}") for i in range(16)]
    rng=random.Random(0x33ED50)
    digest=hashlib.sha256()
    for case in range(64):
        memory=bytes(rng.getrandbits(8) for _ in range(0x1000))
        uc.mem_write(heap,memory)
        stack_bytes=bytearray(rng.getrandbits(8) for _ in range(0x1000))
        offset=0x808
        struct.pack_into("<Q",stack_bytes,offset,stop)
        uc.mem_write(stack,bytes(stack_bytes))
        rsp=stack+offset
        values=[rng.getrandbits(64) for _ in gprs]
        # Both mapped and null receivers demonstrate that this body itself
        # never touches an object; null checks in callers are outside scope.
        values[1]=heap if case%2 else 0
        vectors=[rng.getrandbits(128) for _ in xmms]
        for register,value in zip(gprs+xmms,values+vectors): uc.reg_write(register,value)
        flags=[2,0x202,0x246,0x247,0x602,0xA46][case%6]
        uc.reg_write(x.UC_X86_REG_RSP,rsp)
        uc.reg_write(x.UC_X86_REG_EFLAGS,flags)
        uc.emu_start(base+RVA,stop,timeout=1_000_000,count=2)
        if uc.reg_read(x.UC_X86_REG_RIP)!=stop or uc.reg_read(x.UC_X86_REG_RSP)!=rsp+8:
            raise ValueError("shared-return control/stack effect changed")
        if [uc.reg_read(r) for r in gprs+xmms] != values+vectors or uc.reg_read(x.UC_X86_REG_EFLAGS)!=flags:
            raise ValueError("shared-return body changed registers or flags")
        if bytes(uc.mem_read(heap,0x1000))!=memory or bytes(uc.mem_read(stack,0x1000))!=bytes(stack_bytes):
            raise ValueError("shared-return body wrote heap or stack")
        digest.update(memory); digest.update(stack_bytes)
    if seen != [RVA]*64:
        raise ValueError("unexpected executed instruction count")
    return {"cases":64,"executed_instructions":64,"input_digest_sha256":digest.hexdigest(),
            "gpr_values_preserved":15,"xmm_values_preserved":16,"flags_preserved":True,
            "heap_and_stack_unchanged":True,"stack_pointer_delta":8}


def audit(game_root,dumper_root):
    import capstone
    import pefile

    root=Path(game_root); dumper=Path(dumper_root); repo=Path(__file__).parents[1]
    manifest_path=repo/f"manifests/builds/{BUILD}.json"
    build=json.loads(manifest_path.read_text(encoding="utf-8"))
    metadata=(root/"Demon Bluff_Data/il2cpp_data/Metadata/global-metadata.dat").read_bytes()
    if hashlib.sha256(metadata).hexdigest().upper()!=build["inputs"]["global_metadata"]["sha256"]:
        raise ValueError("metadata fingerprint mismatch")
    generated,rows,methods,counts=create_outputs(build_manifest_path=manifest_path,
        type_index_path=repo/f"symbols/{BUILD}/assembly_csharp_types.json",
        dump_cs=dumper/"dump.cs",script_json=dumper/"script.json",game_assembly=root/"GameAssembly.dll")
    compare_file(repo/f"coverage/{BUILD}/manifest.v1.json",generated)
    compare_file(repo/f"coverage/{BUILD}/methods.v1.jsonl",rows)
    definitions=select_definitions(methods)
    if len(definitions)!=175:
        raise ValueError("unexpected shared-return definition count")
    data=(root/"GameAssembly.dll").read_bytes()
    pe=pefile.PE(data=data,fast_load=True)
    section=pe.get_section_by_rva(RVA)
    if section is None or not section.Characteristics & 0x20000000:
        raise ValueError("shared return body outside executable section")
    cs=capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
    instruction=next(cs.disasm(pe.get_data(RVA,3),RVA))
    if instruction.mnemonic!="ret" or instruction.op_str!="0" or instruction.size!=3:
        raise ValueError("shared return semantics changed")
    return {"schema_version":1,"build_id":BUILD,"game_assembly_sha256":build["inputs"]["game_assembly"]["sha256"],
        "global_metadata_sha256":build["inputs"]["global_metadata"]["sha256"],
        "denominator_methods":counts["method_definitions"],"body":BODY,
        "definition_count":len(definitions),"definitions":definitions,
        "native_emulation":emulate(data),
        "scope":"Direct native void bodies have only return control/stack effects, without heap/stack writes or data/vector-register/flag changes; caller/runtime actions remain separate",
        "classification_policy":"Preserve existing classifications and explicitly classify reviewed method identities; no automatic blanket terminal-state write",
        "unresolved":["call-site dispatch and null checking", "runtime class-initialization bookkeeping",
            "surrounding iterator and cancellation behavior", "all non-stub methods on these declaring types"]}


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_root",type=Path)
    parser.add_argument("--dumper-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args(); report=audit(args.game_root,args.dumper_root)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"Verified {report['definition_count']} direct void definitions and {report['native_emulation']['cases']} native cases")
