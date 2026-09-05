"""Bind selected native wait-dispatch masks to constructed PlayerLoop nodes.

Uses pinned static references plus isolated execution of the native loop builder
with a synthetic, already initialized type cache. No live game input or host DLL
execution. Requires local UnityPlayer.dll, pefile, Capstone and Unicorn 2.1.4.
"""
import argparse
import json
import struct
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


PHASES = [
    {"name": "EarlyUpdate/ScriptRunDelayedStartupFrame", "cache": 0x6A8,
     "type_arg": 0x81DCD5, "namespace_arg": 0x81DCDC, "assembly_arg": 0x81DCE3,
     "lookup": 0x81DCF8, "cache_load": 0x81DCFD, "unwrap": 0x81DD19,
     "store": 0x81DD1C, "callback_install": 0x59C892, "cell": 0x1CAA2D8,
     "callback": 0x5B7CE0, "queue_load": 0x5B7CE0, "mask_site": 0x5B7CE7,
     "dispatch": 0x5B7CEF, "mask": 4, "guard": "unconditional in callback"},
    {"name": "FixedUpdate/ScriptRunDelayedFixedFrameRate", "cache": 0x778,
     "type_arg": 0x81E1AA, "namespace_arg": 0x81E1B1, "assembly_arg": 0x81E1B8,
     "lookup": 0x81E1D0, "cache_load": 0x81E1D5, "unwrap": 0x81E1F1,
     "store": 0x81E1F4, "callback_install": 0x59C8AE, "cell": 0x1CAA380,
     "callback": 0x5B7D20, "queue_load": 0x5B7D29, "mask_site": 0x5B7D30,
     "dispatch": 0x5B7D38, "mask": 1, "guard": "byte at engine RVA 0x1cd5908 is nonzero"},
    {"name": "Update/ScriptRunDelayedDynamicFrameRate", "cache": 0xA48,
     "type_arg": 0x81F33E, "namespace_arg": 0x81F345, "assembly_arg": 0x81F34C,
     "lookup": 0x81F364, "cache_load": 0x81F369, "unwrap": 0x81F385,
     "store": 0x81F388, "callback_install": 0x59C8CA, "cell": 0x1CAA3E0,
     "callback": 0x5B7D50, "queue_load": 0x5B7D50, "mask_site": 0x5B7D57,
     "dispatch": 0x5B7D5F, "mask": 2, "guard": "unconditional in callback"},
    {"name": "PostLateUpdate/ScriptRunDelayedDynamicFrameRate", "cache": 0x8A8,
     "type_arg": 0x81E916, "namespace_arg": 0x81E91D, "assembly_arg": 0x81E92E,
     "lookup": 0x81E93C, "cache_load": 0x81E941, "unwrap": 0x81E95D,
     "store": 0x81E960, "callback_install": 0x59C8D8, "cell": 0x1CAA468,
     "callback": 0x5B7D50, "queue_load": 0x5B7D50, "mask_site": 0x5B7D57,
     "dispatch": 0x5B7D5F, "mask": 2, "guard": "unconditional in callback"},
    {"name": "PostLateUpdate/PlayerSendFrameComplete", "cache": 0x860,
     "type_arg": 0x81E754, "namespace_arg": 0x81E75B, "assembly_arg": 0x81E762,
     "lookup": 0x81E77A, "cache_load": 0x81E77F, "unwrap": 0x81E79B,
     "store": 0x81E79E, "callback_install": 0x59C7EA, "cell": 0x1CAA528,
     "callback": 0x5B7AF0, "queue_load": 0x5B7AFD, "mask_site": 0x5B7B04,
     "dispatch": 0x5B7B0C, "mask": 0x20, "guard": "dword at engine RVA 0x1cd5904 is zero"},
]


def find_phase_nodes(buffer, image_base, tag_base, phases):
    """Validate selected type/callback joins in a constructed 0x68-stride loop."""
    if len(buffer) < 0x68:
        raise ValueError("truncated loop header")
    count = struct.unpack_from("<Q", buffer, 0x28)[0]
    if not 1 <= count <= 256 or len(buffer) < count * 0x68:
        raise ValueError("invalid loop count or truncated nodes")
    results = []
    for phase in phases:
        matches = []
        for index in range(count):
            offset = index * 0x68
            value = struct.unpack_from("<Q", buffer, offset + 0x58)[0]
            if value == image_base + phase["cell"]:
                type_tag = struct.unpack_from("<Q", buffer, offset + 0x30)[0]
                if type_tag != tag_base + phase["cache"]:
                    raise ValueError("callback joined to wrong type-cache entry")
                matches.append(index)
        if len(matches) != 1:
            raise ValueError("selected callback missing or duplicated")
        results.append(matches[0])
    return count, results


def emulate_loop(data):
    import pefile
    import unicorn
    from unicorn import x86_const as x

    verify_fingerprint(data, ENGINE_SHA256)
    if unicorn.__version__ != "2.1.4":
        raise ValueError("This audit requires Unicorn 2.1.4")
    pe = pefile.PE(data=data, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    arena, stack, stop, tag_base = 0x200000000, 0x300000000, 0x400000000, 0x500000000
    uc.mem_map(arena, 0x10000)
    uc.mem_map(stack, 0x10000)
    uc.mem_map(stop, 0x1000)
    cache, loop = arena, arena + 0x2000
    for offset in range(0, 0x1000, 8):
        uc.mem_write(cache + offset, struct.pack("<Q", tag_base + offset))
    uc.mem_write(base + 0x1CD6AF8, struct.pack("<Q", cache))
    uc.mem_write(base + 0x1BDE140, struct.pack("<Q", loop))
    seen = set()

    def on_code(_, address, size, __):
        rva = address - base
        if not any(a <= rva and rva + size <= b for a, b in
                   [(0x5A6810, 0x5A7A5C), (0x42F6B0, 0x42FD73)]):
            raise ValueError(f"execution left audited loop construction: {rva:#x}")
        seen.add(rva)

    uc.hook_add(unicorn.UC_HOOK_CODE, on_code)
    rsp = stack + 0xF008
    uc.mem_write(rsp, struct.pack("<Q", stop) + bytes(0x28))
    uc.reg_write(x.UC_X86_REG_RSP, rsp)
    uc.reg_write(x.UC_X86_REG_EFLAGS, 2)
    uc.emu_start(base + 0x5A6810, stop, timeout=2_000_000, count=100_000)
    if uc.reg_read(x.UC_X86_REG_RIP) != stop:
        raise ValueError("loop construction exceeded its execution bound")
    count, indices = find_phase_nodes(bytes(uc.mem_read(loop, 0x8000)), base, tag_base, PHASES)
    return count, indices, len(seen)


def audit(path):
    import pefile
    import capstone

    data = Path(path).read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    pe = pefile.PE(data=data, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    checked = set()

    def instruction(rva):
        section = pe.get_section_by_rva(rva)
        if section is None or not section.Characteristics & 0x20000000:
            raise ValueError("instruction outside executable image")
        return next(cs.disasm(pe.get_data(rva, 15), rva))

    def exact(rva, mnemonic, operands):
        ins = instruction(rva)
        if (ins.mnemonic, ins.op_str) != (mnemonic, operands):
            raise ValueError(f"native relationship mismatch at {rva:#x}")
        checked.add(rva)

    def reference(rva, mnemonic, first, target):
        ins = instruction(rva)
        if ins.mnemonic != mnemonic or not ins.op_str.startswith(first):
            raise ValueError(f"native reference shape mismatch at {rva:#x}")
        refs = [ins.address + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if refs != [target]:
            raise ValueError(f"native reference target mismatch at {rva:#x}")
        checked.add(rva)

    def string_argument(rva, register, expected):
        ins = instruction(rva)
        refs = [ins.address + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if len(refs) != 1:
            raise ValueError("missing type-lookup string reference")
        reference(rva, "lea", register + ",", refs[0])
        if pe.get_data(refs[0], len(expected) + 1) != expected.encode("ascii") + b"\0":
            raise ValueError("type-lookup identity mismatch")

    for phase in PHASES:
        string_argument(phase["type_arg"], "r9", phase["name"])
        string_argument(phase["namespace_arg"], "r8", "UnityEngine.PlayerLoop")
        string_argument(phase["assembly_arg"], "rdx", "UnityEngine.CoreModule.dll")
        exact(phase["lookup"], "call", "0x75fc00")
        reference(phase["cache_load"], "mov", "rcx,", 0x1CD6AF8)
        exact(phase["unwrap"], "mov", "rax, qword ptr [rax]")
        exact(phase["store"], "mov", f"qword ptr [rcx + {phase['cache']:#x}], rax")
        install = phase["callback_install"]
        reference(install, "lea", "rax,", phase["callback"])
        reference(install + instruction(install).size, "mov", "qword ptr [rip", phase["cell"])
        reference(phase["queue_load"], "mov", "rcx,", 0x1C6E720)
        mask = phase["mask"]
        exact(phase["mask_site"], "mov", f"edx, {mask if mask < 10 else hex(mask)}")
        virtual_load = phase["mask_site"] + instruction(phase["mask_site"]).size
        exact(virtual_load, "mov", "rax, qword ptr [rcx]")
        exact(phase["dispatch"], "call" if mask == 0x20 else "jmp", "qword ptr [rax + 0xb8]")
    reference(0x5B7D20, "cmp", "byte ptr [rip", 0x1CD5908)
    exact(0x5B7D27, "je", "0x5b7d3f")
    reference(0x5B7AF4, "cmp", "dword ptr [rip", 0x1CD5904)
    exact(0x5B7AFB, "jne", "0x5b7b25")
    exact(0x5A681D, "call", "0x42f6b0")
    reference(0x42F6B6, "mov", "rdx,", 0x1CD6AF8)
    count, indices, instructions = emulate_loop(data)
    return {
        "schema_version": 1, "unityplayer_sha256": digest,
        "native_relationships_verified": len(checked),
        "emulated_builder": {"rva": "0x5a6810", "type_array_builder": "0x42f6b0",
            "emulator": "Unicorn 2.1.4 x86-64", "constructed_nodes": count,
            "distinct_executed_instructions": instructions,
            "input": "synthetic already-initialized type cache with unique offset tags"},
        "bindings": [{"name": p["name"], "mask": p["mask"], "guard": p["guard"],
            "cache_offset": hex(p["cache"]), "callback_cell_rva": hex(p["cell"]),
            "callback_rva": hex(p["callback"]), "constructed_node_index": index}
            for p, index in zip(PHASES, indices)],
        "unresolved": ["provenance or reachability of phase bit 8",
            "runtime clock snapshots and modifications of the default PlayerLoop",
            "callback-mutated wait traversal and full coroutine lifetime"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unityplayer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.unityplayer)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['native_relationships_verified']} native relationships and "
          f"{len(report['bindings'])} constructed phase bindings")
