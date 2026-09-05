"""Verify selected creation/MoveNext/wait edges across the pinned native images.

This audits the normal, valid-owner coroutine path. It does not reconstruct all
exception, cancellation, nested-iterator or yield-object branches.
"""
import argparse
import hashlib
import json
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, relative_target


def audit(game_root):
    import capstone
    import pefile

    root = Path(game_root)
    manifest = json.loads((Path(__file__).parents[1] / "manifests" / "builds" /
                           "f530404b0f3f_807de4a83df4.json").read_text(encoding="utf-8"))
    sources = {
        "UnityPlayer.dll": (root / "UnityPlayer.dll", ENGINE_SHA256),
        "GameAssembly.dll": (root / "GameAssembly.dll", manifest["inputs"]["game_assembly"]["sha256"]),
        "global-metadata.dat": (root / "Demon Bluff_Data/il2cpp_data/Metadata/global-metadata.dat",
                                manifest["inputs"]["global_metadata"]["sha256"]),
    }
    pinned = {}
    for label, (path, expected) in sources.items():
        data = path.read_bytes()
        digest = hashlib.sha256(data).hexdigest().upper()
        if digest != expected.upper():
            raise ValueError(f"{label} fingerprint mismatch")
        pinned[label] = {"data": data, "sha256": digest, "size": len(data)}
    images = {label: pefile.PE(data=pinned[label]["data"], fast_load=True)
              for label in ("UnityPlayer.dll", "GameAssembly.dll")}
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True

    def instruction(label, rva):
        pe = images[label]
        section = pe.get_section_by_rva(rva)
        if section is None or not section.Characteristics & 0x20000000:
            raise ValueError("Audit instruction outside executable section")
        return next(cs.disasm(pe.get_data(rva, 15), rva))

    engine_checks = [
        (0x100E0A, "call", "0x77bc80"),
        (0x77BE17, "mov", "edx, 0x88"),
        (0x77BE9C, "mov", "qword ptr [rdx + 0x58], r13"),
        (0x77BEAC, "mov", "r15d, 1"),
        (0x77BEB2, "mov", "dword ptr [rdx + 0x60], r15d"),
        (0x77BEFA, "call", "0x778d90"),
        (0x778DAB, "inc", "dword ptr [rcx + 0x60]"),
        (0x778DEF, "mov", "rdx, qword ptr [rdx + 0xcf0]"),
        (0x778DFB, "call", "0x75d3a0"),
        (0x778E57, "lea", "rcx, [rbp + 0xc0]"),
        (0x778F3E, "call", "rax"),
        (0x778F98, "cmp", "byte ptr [rbp + 0xc0], r15b"),
        (0x778F9F, "je", "0x778fb2"),
        (0x778FA1, "test", "rax, rax"),
        (0x778FA4, "jne", "0x778fb2"),
        (0x778FA6, "cmp", "qword ptr [rsp + 0x28], r15"),
        (0x778FAD, "mov", "dil, 1"),
        (0x778FBB, "call", "0x778bd0"),
        (0x778FC3, "je", "0x779034"),
        (0x778FF1, "test", "dil, dil"),
        (0x778FF4, "jne", "0x77904d"),
        (0x77904D, "cmp", "qword ptr [rsi + 0x58], r15"),
        (0x779056, "call", "0x779070"),
        (0x779086, "mov", "rdx, qword ptr [rcx + 0x50]"),
        (0x77921E, "call", "rax"),
        (0x779354, "call", "0x779370"),
        (0x7793DB, "mov", "rdx, qword ptr [rdx + 0xdc8]"),
        (0x7793D8, "mov", "r8b, 1"),
        (0x7793E5, "call", "rax"),
        (0x7793E9, "je", "0x7794fe"),
        (0x7793EF, "inc", "dword ptr [r14 + 0x60]"),
        (0x7793FB, "movss", "xmm6, dword ptr [rdi + 0x10]"),
        (0x7794EC, "call", "0x440f00"),
        (0x778BBE, "jmp", "0x778d90"),
        (0x8204F6, "call", "0x75fd10"),
        (0x820516, "mov", "qword ptr [rcx + 0xcf0], rax"),
        (0x820A58, "call", "0x75fc00"),
        (0x820A78, "mov", "qword ptr [rcx + 0xdc8], rax"),
    ]
    game_checks = [
        (0x1C8A791, "mov", "rbx, rdx"),
        (0x1C8A794, "mov", "rdi, rcx"),
        (0x1C8A7D6, "xor", "ecx, ecx"),
        (0x1C8A7D8, "mov", "r8, rdi"),
        (0x1C8A7DB, "call", "0x4060"),
        (0x1C8A7E0, "mov", "byte ptr [rbx], al"),
    ]
    for label, checks in (("UnityPlayer.dll", engine_checks), ("GameAssembly.dll", game_checks)):
        for rva, mnemonic, operands in checks:
            ins = instruction(label, rva)
            if (ins.mnemonic, ins.op_str) != (mnemonic, operands):
                raise ValueError(f"{label} semantic check failed at {rva:#x}")

    def reference(rva, operand=1):
        ins = instruction("UnityPlayer.dll", rva)
        return relative_target(ins.address, ins.size, ins.operands[operand].mem.disp)

    def string_reference(rva, expected):
        data = images["UnityPlayer.dll"].get_data(reference(rva), len(expected) + 1)
        if data != expected.encode("ascii") + b"\0":
            raise ValueError(f"Unexpected named reference at {rva:#x}")

    names = [(0x8204C8, "SetupCoroutine"), (0x8204D2, "UnityEngine.CoreModule.dll"),
             (0x8204E3, "InvokeMoveNext"), (0x820A36, "WaitForSeconds"),
             (0x76E35B, "il2cpp_runtime_invoke"),
             (0x76CF2B, "il2cpp_class_is_subclass_of")]
    for site, name in names:
        string_reference(site, name)
    refs = [(0x76E367, 0, 0x1CD62A0), (0x778F28, 1, 0x1CD62A0),
            (0x779208, 1, 0x1CD62A0), (0x76CF37, 0, 0x1CD6310),
            (0x7793D1, 1, 0x1CD6310), (0x8204FB, 1, 0x1CD6AF8),
            (0x778DB1, 1, 0x1CD6AF8), (0x820A5D, 1, 0x1CD6AF8),
            (0x77939C, 1, 0x1CD6AF8)]
    for site, operand, expected in refs:
        if reference(site, operand) != expected:
            raise ValueError(f"Unexpected shared pointer reference at {site:#x}")
    interface = instruction("GameAssembly.dll", 0x1C8A7CF)
    interface_slot = relative_target(interface.address, interface.size, interface.operands[1].mem.disp)
    if interface_slot != 0x26FE930:
        raise ValueError("Unexpected IEnumerator metadata usage slot")
    return {
        "schema_version": 1, "build_id": manifest["build_id"],
        "inputs": {label: {k: v for k, v in item.items() if k != "data"}
                   for label, item in pinned.items()},
        "semantic_checks_passed": len(engine_checks) + len(game_checks) + len(names) + len(refs) + 1,
        "scope": "normal valid-owner creation, managed step and WaitForSeconds dispatch edges",
        "engine_creation": "0x77BC80", "engine_move_next_dispatch": "0x778D90",
        "managed_move_next_bridge": "GameAssembly:0x1C8A780",
        "ienumerator_type_usage": "GameAssembly:0x26FE930",
        "yield_current_dispatch": "0x779070", "yield_object_type_dispatch": "0x779370",
        "wait_dispatch_callback": "0x778B30", "wait_release_callback": "0x778BD0",
        "cached_managed_step": "UnityEngine.SetupCoroutine.InvokeMoveNext",
        "cached_wait_type": "UnityEngine.WaitForSeconds",
        "runtime_invoke_storage": "0x1CD62A0",
        "findings": ["creation immediately invokes the managed-step dispatcher",
                     "managed IEnumerator slot-zero result is written to an engine-provided byte address",
                     "a successful continuing step with a retained owner reaches yielded-current dispatch",
                     "the WaitForSeconds type branch retains the payload and inserts its wait record",
                     "the wait callback re-enters the same managed-step dispatcher"],
        "unresolved": ["complete cancellation and reference-lifetime branches", "other yielded object kinds",
                       "native phase identities and mutation-safe equal-deadline ordering"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"verified {report['semantic_checks_passed']} cross-image coroutine checks")
