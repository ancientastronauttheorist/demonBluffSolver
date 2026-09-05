"""Fingerprint and verify selected native Unity wait-producer instruction sites.

Local native-input audit only; pefile and capstone are imported lazily. Outputs
addresses, classifications and check results, never native bytes/disassembly.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

ENGINE_SHA256 = "B5D48235E7CC02FF9496FB33A07D5921ADFC4B40DED1BC64C96A7A7C10B4DFB2"


def verify_fingerprint(data, expected):
    actual = hashlib.sha256(data).hexdigest().upper()
    if actual != expected.upper():
        raise ValueError("UnityPlayer fingerprint mismatch")
    return actual


def relative_target(instruction_rva, size, displacement):
    target = instruction_rva + size + displacement
    if instruction_rva < 0 or size <= 0 or not 0 <= target <= 0xFFFFFFFF:
        raise ValueError("Invalid relative reference")
    return target


def audit(path):
    import capstone
    import pefile

    data = Path(path).read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    pe = pefile.PE(data=data)
    decoder = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    decoder.detail = True

    def instruction(rva):
        section = pe.get_section_by_rva(rva)
        if section is None or not section.Characteristics & 0x20000000:
            raise ValueError("Instruction outside executable section")
        offset = pe.get_offset_from_rva(rva)
        return next(decoder.disasm(data[offset:offset + 15], rva))

    # Semantic checks at independently audited instruction boundaries. No byte
    # signatures or reconstructed proprietary method bodies are redistributed.
    checks = [
        (0x7793FB, "movss", "xmm6, dword ptr [rdi + 0x10]"),
        (0x77948B, "cvtps2pd", "xmm0, xmm6"),
        (0x7794A0, "movsd", "xmm1, qword ptr [rax + 0x60]"),
        (0x7794A5, "mov", "rax, qword ptr [rax + 0xc8]"),
        (0x7794B4, "inc", "rax"),
        (0x7794D4, "addsd", "xmm0, xmm1"),
        (0x7794DF, "movsd", "qword ptr [rbp - 0x39], xmm0"),
        (0x7794EC, "call", "0x440f00"),
        (0x440F1E, "mov", "edx, 0x60"),
        (0x440F95, "comisd", "xmm0, xmm1"),
        (0x440F9E, "jbe", "0x440fad"),
        (0x440FA0, "mov", "rcx, qword ptr [rcx]"),
        (0x440FAD, "mov", "rcx, qword ptr [rcx + 0x10]"),
        (0x440FDC, "call", "0x366cb0"),
    ]
    for rva, mnemonic, operands in checks:
        ins = instruction(rva)
        if (ins.mnemonic, ins.op_str) != (mnemonic, operands):
            raise ValueError(f"Semantic check failed at {rva:#x}")
    cap_instruction = instruction(0x7794AC)
    if cap_instruction.mnemonic != "minsd":
        raise ValueError("Missing time cap")
    cap_rva = relative_target(cap_instruction.address, cap_instruction.size,
                              cap_instruction.operands[1].mem.disp)
    cap = struct.unpack_from("<d", data, pe.get_offset_from_rva(cap_rva))[0]
    if cap != float.fromhex("0x1.fffffffffffffp+1023"):
        raise ValueError("Unexpected time cap")
    return {
        "schema_version": 1,
        "build_id": "f530404b0f3f_807de4a83df4",
        "engine": {"file": "UnityPlayer.dll", "sha256": digest,
                   "size": len(data), "pe_timestamp": pe.FILE_HEADER.TimeDateStamp,
                   "image_base": hex(pe.OPTIONAL_HEADER.ImageBase)},
        "semantic_checks_passed": len(checks) + 1,
        "wait_producer_chunk": "0x7793F3..0x7794FE",
        "deadline_tree_insert": "0x440F00",
        "tree_link_helper": "0x366CB0",
        "time_cap_rva": hex(cap_rva), "time_cap": cap.hex(),
        "resume_callback_candidate": "0x778B30",
        "release_callback_candidate": "0x778BD0",
        "unresolved": ["engine time/counter field ownership", "queue drain and eligibility predicate",
                       "equal-deadline dispatch order", "StartCoroutineManaged2 function binding"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unity_player", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.unity_player)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"verified {report['semantic_checks_passed']} native wait-boundary checks")
