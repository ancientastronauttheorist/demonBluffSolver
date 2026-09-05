"""Verify the shipped UnityPlayer internal-call registration table pairing.

Audits local proprietary input; emits selected names/addresses and authored
findings only. Synthetic table-decoding tests do not need native dependencies.
"""
import argparse
import json
import struct
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, relative_target, verify_fingerprint


def decode_pointer_pairs(functions, names, image_base, read_name, executable):
    """Decode equally sized, nonempty VA tables with explicit image validation."""
    if not functions or len(functions) != len(names) or len(functions) % 8:
        raise ValueError("Invalid internal-call table sizes")
    bindings = {}
    for index, (function, name) in enumerate(zip(struct.iter_unpack("<Q", functions),
                                                 struct.iter_unpack("<Q", names))):
        function_rva, name_rva = function[0] - image_base, name[0] - image_base
        if not 0 <= function_rva <= 0xFFFFFFFF or not executable(function_rva):
            raise ValueError("Internal-call target outside executable image")
        if not 0 <= name_rva <= 0xFFFFFFFF:
            raise ValueError("Internal-call name outside image")
        decoded = read_name(name_rva)
        if not isinstance(decoded, str) or not decoded or "::" not in decoded:
            raise ValueError("Invalid internal-call name")
        if decoded in bindings:
            raise ValueError("Duplicate internal-call name")
        bindings[decoded] = {"index": index, "rva": function_rva}
    return bindings


def audit(path):
    import capstone
    import pefile

    data = Path(path).read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    pe = pefile.PE(data=data)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True

    def executable(rva):
        section = pe.get_section_by_rva(rva)
        return section is not None and bool(section.Characteristics & 0x20000000)

    def instruction(rva):
        if not executable(rva):
            raise ValueError("Audit instruction outside executable image")
        return next(cs.disasm(pe.get_data(rva, 15), rva))

    checks = [
        (0xFA1138, "xor", "r14d, r14d"),
        (0xFA1140, "mov", "rsi, qword ptr [r14 + r12 + 0x1894fc0]"),
        (0xFA1151, "mov", "rbp, qword ptr [r14 + r12 + 0x189bb80]"),
        (0xFA1182, "mov", "rdx, rsi"),
        (0xFA1185, "mov", "rcx, rbp"),
        (0xFA1188, "call", "qword ptr [rip + 0xd34e0a]"),
        (0xFA118E, "inc", "r15d"),
        (0xFA1191, "add", "r14, 8"),
        (0xFA1195, "cmp", "r15d, 0xd77"),
        (0xFA119C, "jb", "0xfa1140"),
        (0x10E107, "movsd", "xmm0, qword ptr [rax + 0x90]"),
        (0x10E10F, "cvtpd2ps", "xmm0, xmm0"),
        (0x10E127, "movsd", "xmm0, qword ptr [rax + 0x90]"),
        (0x10E517, "mov", "eax, dword ptr [rax + 0xc8]"),
        (0x10E527, "mov", "eax, dword ptr [rax + 0xd0]"),
        (0x100E0A, "call", "0x77bc80"),
        (0x76CC60, "call", "0x6a3420"),
    ]
    for rva, mnemonic, operands in checks:
        ins = instruction(rva)
        if (ins.mnemonic, ins.op_str) != (mnemonic, operands):
            raise ValueError(f"Semantic check failed at {rva:#x}")
    image_reference = instruction(0xFA1131)
    if (image_reference.mnemonic != "lea"
            or relative_target(image_reference.address, image_reference.size,
                               image_reference.operands[1].mem.disp) != 0):
        raise ValueError("Registration base is not the image base")
    for site in (0x10E100, 0x10E120, 0x10E510, 0x10E520):
        ins = instruction(site)
        if (ins.mnemonic != "mov"
                or relative_target(ins.address, ins.size, ins.operands[1].mem.disp) != 0x1C6E718):
            raise ValueError("Time getters do not use the audited engine object")

    def read_name(rva):
        section = pe.get_section_by_rva(rva)
        if section is None:
            raise ValueError("Name outside image sections")
        available = section.SizeOfRawData - (rva - section.VirtualAddress)
        if available <= 0:
            raise ValueError("Name in zero-filled virtual data")
        raw = pe.get_data(rva, min(2048, available))
        terminator = raw.find(b"\0")
        if terminator < 0:
            raise ValueError("Missing bounded name terminator")
        return raw[:terminator].decode("utf-8")

    export_name = instruction(0x76CC59)
    export_store = instruction(0x76CC65)
    registration_call = instruction(0xFA1188)
    if (export_name.mnemonic != "lea" or read_name(relative_target(
            export_name.address, export_name.size, export_name.operands[1].mem.disp))
            != "il2cpp_add_internal_call"):
        raise ValueError("Unexpected internal-call registration export name")
    if (export_store.mnemonic != "mov" or relative_target(
            export_store.address, export_store.size, export_store.operands[0].mem.disp)
            != relative_target(registration_call.address, registration_call.size,
                               registration_call.operands[0].mem.disp)):
        raise ValueError("Registration sink differs from resolved export storage")

    count = 0xD77
    bindings = decode_pointer_pairs(pe.get_data(0x1894FC0, count * 8),
                                    pe.get_data(0x189BB80, count * 8),
                                    pe.OPTIONAL_HEADER.ImageBase, read_name, executable)
    selected = {
        "UnityEngine.MonoBehaviour::StartCoroutineManaged": 0x1007F0,
        "UnityEngine.MonoBehaviour::StartCoroutineManaged2": 0x100CE0,
        "UnityEngine.Time::get_time": 0x10E100,
        "UnityEngine.Time::get_timeAsDouble": 0x10E120,
        "UnityEngine.Time::get_fixedTime": 0x10E180,
        "UnityEngine.Time::get_fixedTimeAsDouble": 0x10E1A0,
        "UnityEngine.Time::get_frameCount": 0x10E510,
        "UnityEngine.Time::get_renderedFrameCount": 0x10E520,
    }
    for name, expected in selected.items():
        if bindings.get(name, {}).get("rva") != expected:
            raise ValueError(f"Unexpected registered binding for {name}")
    return {
        "schema_version": 1,
        "build_id": "f530404b0f3f_807de4a83df4",
        "engine_sha256": digest,
        "registration_loop": "0xFA1110",
        "function_table": "0x1894FC0", "name_table": "0x189BB80",
        "registered_pair_count": len(bindings),
        "semantic_checks_passed": len(checks) + 7,
        "registration_sink": "il2cpp_add_internal_call",
        "selected_bindings_verified": len(selected),
        "bindings": {name: {"index": bindings[name]["index"], "rva": hex(rva)}
                     for name, rva in selected.items()},
        "field_findings": {
            "0x90": "double backing Time.time and Time.timeAsDouble; sampled by wait consumer",
            "0xC8": "low 32 bits back Time.frameCount; consumer uses signed 64-bit value",
            "0xD0": "low 32 bits back Time.renderedFrameCount",
            "0x60": "wait producer clock field; public identity/relationship unresolved",
        },
        "start_coroutine_managed2_creation_target": "0x77BC80",
        "unresolved": ["producer clock versus Time.time relationship", "engine phase identities",
                       "coroutine creation/MoveNext bridge", "equal-deadline mutation ordering"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unity_player", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.unity_player)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"verified {report['registered_pair_count']} internal-call pairs, "
          f"{report['semantic_checks_passed']} native relationships and "
          f"{report['selected_bindings_verified']} selected bindings")
