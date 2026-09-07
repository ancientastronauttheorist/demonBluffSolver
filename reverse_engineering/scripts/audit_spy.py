"""Pin Spy's six managed methods, inert dispatch, clue and cache contracts.

Audits local native/metadata inputs and selected negative serialized-name
observations. No live game state, raw native bytes or decompiler bodies emitted.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

BUILD = "f530404b0f3f_807de4a83df4"
ASSET_HASHES = {
    "sharedassets0.assets": "E0D239C03FA47EC138F9C2797779E83B65AEF41F30C5AD84D4CA76767A93E967",
    "level0": "B509AC15904F0926419CCCC1D2E86508C69B0E3AF2235A08FF8E9FC4D170C7F1",
    "resources.assets": "FED4E49586B019E3712C5EE3A8547885BF05CCB82997B790BC8A6B8ADF1226D6",
    "globalgamemanagers.assets": "38E56EF97CDA4C5DFCFCB397F8077335AEBBF199460FA5B536D91698C21ED2D9",
}


def audit(game_root, dumper_root):
    import pefile
    import capstone

    repo = Path(__file__).parents[1]
    manifest = json.loads((repo / f"manifests/builds/{BUILD}.json").read_text())
    extraction = json.loads((repo / f"manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json").read_text())
    root, dumper = Path(game_root), Path(dumper_root)
    def pinned(path, expected):
        data = path.read_bytes()
        if hashlib.sha256(data).hexdigest().upper() != expected.upper():
            raise ValueError(f"fingerprint mismatch: {path.name}")
        return data
    native = pinned(root / "GameAssembly.dll", manifest["inputs"]["game_assembly"]["sha256"])
    pinned(root / "Demon Bluff_Data/il2cpp_data/Metadata/global-metadata.dat", manifest["inputs"]["global_metadata"]["sha256"])
    dump = pinned(dumper / "dump.cs", extraction["outputs"]["dump_cs"]["sha256"]).decode("utf-8-sig")
    script = json.loads(pinned(dumper / "script.json", extraction["outputs"]["script_json"]["sha256"]).decode("utf-8-sig"))
    start = dump.index("public class Spy : Minion // TypeDefIndex: 5911")
    block = dump[start:dump.index("\n}",start)+2]
    expected_rvas = [0x3ED7D0,0x3ED640,0x33ED50,0x3ED4B0,0x3ED6A0,0x3CFFF0]
    if [int(rva,16) for rva in re.findall(r"// RVA: (0x[0-9A-Fa-f]+)",block)] != expected_rvas:
        raise ValueError("Spy declaration boundary changed")
    if "public CharacterData chData; // 0x48" not in block or "BluffAct(" in block:
        raise ValueError("Spy cache or inherited dispatch metadata changed")
    pe = pefile.PE(data=native,fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86,capstone.CS_MODE_64)
    cs.detail = True
    checks = [
        (0x33ED50,"ret","0"),
        (0x3C4CA0,"mov","rax, qword ptr [rcx]"),
        (0x3C4CA3,"mov","r9, qword ptr [rax + 0x210]"),
        (0x3C4CAA,"jmp","qword ptr [rax + 0x208]"),
        (0x3CFFF0,"xor","edx, edx"), (0x3CFFF2,"jmp","0x357920"),
        (0x357920,"xor","edx, edx"), (0x357922,"jmp","0x33ed50"),
        (0x3ED681,"xor","r9d, r9d"), (0x3ED684,"xor","r8d, r8d"),
        (0x3ED68D,"call","0x35d5d0"),
        (0x3E4B01,"xor","r9d, r9d"), (0x3E4B04,"xor","r8d, r8d"),
        (0x3E4B0D,"call","0x35d5d0"),
    ]
    for rva,mnemonic,operands in checks:
        ins = next(cs.disasm(pe.get_data(rva,15),rva))
        if (ins.mnemonic,ins.op_str) != (mnemonic,operands):
            raise ValueError(f"Spy native relationship mismatch at {rva:#x}")
    strings = {item["Address"]:item["Value"] for item in script["ScriptString"]}
    expected_description = "Can register as a Good Townsfolk. Demon will kill best targets."
    for rva,value in [(0x3ED7F0,expected_description),(0x3ED67A,""),(0x3E4AFA,"")]:
        ins = next(cs.disasm(pe.get_data(rva,15),rva))
        target = rva + ins.size + ins.operands[1].mem.disp
        if ins.mnemonic != "mov" or strings.get(target) != value:
            raise ValueError("Spy description/clue metadata binding mismatch")
    observations=[]
    for name,digest in ASSET_HASHES.items():
        raw=pinned(root / "Demon Bluff_Data" / name,digest)
        count=raw.count(b"Spy")
        if count:
            raise ValueError("previous negative Spy serialized-name observation changed")
        observations.append({"file":name,"sha256":digest,"size":len(raw),"ascii_spy_occurrences":count})
    return {"schema_version":1,"build_id":BUILD,"type_def_index":5911,
        "managed_method_count":6,"cache_field":{"name":"chData","offset":"0x48"},
        "native_relationships_verified":len(checks)+3,"declaration_boundary_verified":True,
        "method_rvas":[hex(rva) for rva in expected_rvas],
        "description":expected_description,"truth_and_inherited_bluff_clue":{"text":"","references":"null"},
        "act":"empty shared return body for all triggers",
        "inherited_bluff_act":"Role.BluffAct forwards to the concrete Act slot",
        "constructor":"two empty constructor forwarders reach the shared return body; no cache write",
        "asset_name_observations":observations,
        "asset_limit":"No current shipped Spy asset binding established; negative name searches alone do not prove global unreachability",
        "selector_evidence":"Existing bluff-acquisition audit and Spy-cache replay cover exact Villager occurrence selection and cache reuse"}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_root",type=Path)
    parser.add_argument("--dumper-root",type=Path,required=True)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.game_root,args.dumper_root)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"Verified {report['managed_method_count']} Spy declarations and {report['native_relationships_verified']} native relationships")
