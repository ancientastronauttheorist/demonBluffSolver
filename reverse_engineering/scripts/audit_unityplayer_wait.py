"""Fingerprint and verify selected native Unity wait producer/consumer sites.

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
        # Queue owner construction and consumer: node fields below are the
        # 0x40-byte wait record at node+0x20, not a second record layout.
        (0x43B73A, "mov", "qword ptr [rcx], rax"),
        (0x43B762, "mov", "qword ptr [rbx + 0x30], rax"),
        (0x43B766, "mov", "dword ptr [rbx + 0x48], edi"),
        (0x43BDE4, "mov", "rbx, qword ptr [rdi + 0x30]"),
        (0x43BDE8, "mov", "rdx, qword ptr [rax + 0xc8]"),
        (0x43BDEF, "mov", "rbx, qword ptr [rbx]"),
        (0x43BDF2, "movsd", "xmm6, qword ptr [rax + 0x90]"),
        (0x43BDFA, "inc", "dword ptr [rdi + 0x48]"),
        (0x43BE23, "comisd", "xmm6, xmmword ptr [rbx + 0x20]"),
        (0x43BE28, "jb", "0x43bfd2"),
        (0x43BE7C, "mov", "qword ptr [rdi + 0x40], rcx"),
        (0x43BE80, "mov", "r12d, dword ptr [rbx + 0x54]"),
        (0x43BE84, "test", "ebp, r12d"),
        (0x43BE87, "je", "0x43bfbd"),
        (0x43BE90, "cmp", "dword ptr [rbx + 0x58], eax"),
        (0x43BE93, "je", "0x43bfbd"),
        (0x43BE99, "cmp", "qword ptr [rbx + 0x28], rdx"),
        (0x43BE9D, "jg", "0x43bfbd"),
        (0x43BF17, "cmp", "byte ptr [rbx + 0x34], 0"),
        (0x43BF23, "jne", "0x43bf4e"),
        (0x43BF2F, "call", "0x43bc60"),
        (0x43BF34, "mov", "rdx, rbp"),
        (0x43BF37, "mov", "rcx, r14"),
        (0x43BF3A, "call", "r15"),
        (0x43BF42, "cmp", "eax, 1"),
        (0x43BF45, "jne", "0x43bfb5"),
        (0x43BF4A, "call", "rsi"),
        (0x43BFB0, "call", "0x43bb00"),
        (0x43BFBD, "mov", "rbx, qword ptr [rdi + 0x40]"),
        (0x43BC49, "call", "rbx"),
        (0x43BD6F, "call", "0x3eb6d0"),
        (0x778B3D, "cmp", "qword ptr [rdx + 0x58], rcx"),
        (0x778BBE, "jmp", "0x778d90"),
        (0x778BD6, "dec", "dword ptr [rcx + 0x60]"),
        (0x778BE0, "jle", "0x778bed"),
        (0x778C89, "call", "0x17a7808"),
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
    vtable_instruction = instruction(0x43B70C)
    if vtable_instruction.mnemonic != "lea":
        raise ValueError("Missing queue vtable assignment")
    vtable_rva = relative_target(vtable_instruction.address, vtable_instruction.size,
                                vtable_instruction.operands[1].mem.disp)
    consumer_va = struct.unpack_from("<Q", data,
                                    pe.get_offset_from_rva(vtable_rva + 0xB8))[0]
    if consumer_va - pe.OPTIONAL_HEADER.ImageBase != 0x43BD90:
        raise ValueError("Unexpected queue consumer virtual slot")
    return {
        "schema_version": 2,
        "build_id": "f530404b0f3f_807de4a83df4",
        "engine": {"file": "UnityPlayer.dll", "sha256": digest,
                   "size": len(data), "pe_timestamp": pe.FILE_HEADER.TimeDateStamp,
                   "image_base": hex(pe.OPTIONAL_HEADER.ImageBase)},
        "semantic_checks_passed": len(checks) + 2,
        "wait_producer_chunk": "0x7793F3..0x7794FE",
        "deadline_tree_insert": "0x440F00",
        "tree_link_helper": "0x366CB0",
        "time_cap_rva": hex(cap_rva), "time_cap": cap.hex(),
        "queue_constructor": "0x43B700", "queue_vtable": hex(vtable_rva),
        "queue_consumer": "0x43BD90", "consumer_virtual_slot": "0xB8",
        "queue_erase_without_release": "0x43BC60",
        "queue_erase_with_release": "0x43BB00",
        "record_fields": {"0x00": "double deadline", "0x08": "signed 64-bit eligibility threshold",
                          "0x10": "float repeat interval", "0x14": "repeat flag",
                          "0x18": "callback payload", "0x20": "dispatch callback",
                          "0x28": "release callback", "0x30": "owner lookup key",
                          "0x34": "phase mask", "0x38": "insertion generation"},
        "consumer_gates": ["stop traversal if sampled engine field 0x90 < deadline or unordered",
                           "skip when phase masks do not intersect",
                           "skip insertion generation equal to current dispatch generation",
                           "skip signed threshold greater than sampled engine field 0xC8"],
        "dispatch_callback": "0x778B30", "continuation_dispatch_target": "0x778D90",
        "release_callback": "0x778BD0",
        "unresolved": ["engine time/counter field ownership", "owner lookup and continuation dispatch internals",
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
