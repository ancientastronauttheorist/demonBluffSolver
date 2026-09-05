"""Audit bounded native coroutine reference release with synthetic free sinks.

Executes the pinned release routine, including nested release calls. GC-handle
free and allocation free are recorded rather than executed on the host. Waiter
objects at +0x78 are covered only while another reference retains them; their
last-reference destructor and complete cross-caller lifetime remain outside scope.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from audit_unityplayer_cancellation import NativeCancellation
from audit_unityplayer_icalls import audit as audit_icalls
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


class NativeRelease(NativeCancellation):
    routines = {**NativeCancellation.routines, "release_native": (0x778BD0, 0x778CB8),
                "cleanup_gc": (0xF88B0, 0xF88B5), "cleanup_gc_body": (0x778CC0, 0x778D82)}

    def __init__(self, data):
        super().__init__(data)
        self.free_handle = self.stop + 0x500

    def _on_code(self, uc, address, size, _):
        if address == self.free_handle:
            self.events.append(["free_handle", uc.reg_read(self.x86.UC_X86_REG_ECX)])
            self._return()
            return
        if address - self.base == 0x17A7808:
            pointer = uc.reg_read(self.x86.UC_X86_REG_RCX)
            if uc.reg_read(self.x86.UC_X86_REG_RDX) != 0x88:
                raise ValueError("unexpected coroutine allocation size")
            self.events.append(["free_payload", self.payload_labels[pointer]])
            self._return()
            return
        rva = address - self.base
        if not any(a <= rva and rva + size <= b for a, b in
                   [(0x778BD0, 0x778CB8), (0xF88B0, 0xF88B5), (0x778CC0, 0x778D82),
                    (0x33CC90, 0x33CCB1), (0x33CF80, 0x33CF86)]):
            raise ValueError(f"execution left audited coroutine release: {rva:#x}")
        self.instructions.add(rva)

    def prepare(self, count=1, enumerator_handle=0, secondary_handle=0):
        self.reset_stop()
        self.uc.mem_write(self.base + 0x1CD6038, struct.pack("<Q", self.free_handle))
        self.uc.mem_write(self.payload + 0x60, struct.pack("<i", count))
        self.uc.mem_write(self.payload + 0x10, struct.pack("<Q", enumerator_handle))
        self.uc.mem_write(self.payload + 0x18, struct.pack("<I", 2))
        self.uc.mem_write(self.payload + 0x28, struct.pack("<Q", secondary_handle))
        self.uc.mem_write(self.payload + 0x30, struct.pack("<I", 2))
        self.events = []

    def snapshot(self, pointer=None):
        pointer = self.payload if pointer is None else pointer
        return {"references": struct.unpack("<i", self.uc.mem_read(pointer + 0x60, 4))[0],
                "cleanup_flag": self.uc.mem_read(pointer + 0x64, 1)[0],
                "list_links_zero": self.qword(pointer) == 0 and self.qword(pointer + 8) == 0,
                "enumerator_handle": self.qword(pointer + 0x10),
                "enumerator_mode": struct.unpack("<I", self.uc.mem_read(pointer + 0x18, 4))[0],
                "secondary_handle": self.qword(pointer + 0x28),
                "secondary_mode": struct.unpack("<I", self.uc.mem_read(pointer + 0x30, 4))[0],
                "link_68_zero": self.qword(pointer + 0x68) == 0,
                "link_70_zero": self.qword(pointer + 0x70) == 0,
                "auxiliary_78_zero": self.qword(pointer + 0x78) == 0}


def audit(path):
    import pefile
    import capstone

    data = Path(path).read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    registration = audit_icalls(path)
    root = Path(path).parent
    manifest = json.loads((Path(__file__).parents[1] / "manifests/builds/f530404b0f3f_807de4a83df4.json").read_text())
    pinned = {}
    for name, relative, key in [("GameAssembly.dll", "GameAssembly.dll", "game_assembly"),
            ("global-metadata.dat", "Demon Bluff_Data/il2cpp_data/Metadata/global-metadata.dat", "global_metadata")]:
        raw = (root / relative).read_bytes()
        sha = hashlib.sha256(raw).hexdigest().upper()
        if sha != manifest["inputs"][key]["sha256"].upper():
            raise ValueError(f"{name} fingerprint mismatch")
        pinned[name] = {"data": raw, "sha256": sha}
    pe = pefile.PE(data=data, fast_load=True)
    managed = pefile.PE(data=pinned["GameAssembly.dll"]["data"], fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    checks = [
        (0x778BD6, "dec", "dword ptr [rcx + 0x60]"),
        (0x778BDC, "cmp", "dword ptr [rcx + 0x60], 0"),
        (0x778BE0, "jle", "0x778bed"), (0x778BE2, "mov", "eax, 1"),
        (0x778BF4, "mov", "byte ptr [rcx + 0x64], 1"),
        (0x778C01, "call", "0x778bd0"),
        (0x778C06, "mov", "qword ptr [rbx + 0x68], rdi"),
        (0x778C13, "mov", "qword ptr [rax + 0x68], rdi"),
        (0x778C17, "mov", "qword ptr [rbx + 0x70], rdi"),
        (0x778C35, "mov", "qword ptr [rbx], rdi"),
        (0x778C38, "mov", "qword ptr [rbx + 8], rdi"),
        (0x778C6B, "mov", "ecx, dword ptr [rbx + 0x10]"),
        (0x778C74, "mov", "dword ptr [rbx + 0x18], edi"),
        (0x778C77, "mov", "qword ptr [rbx + 0x10], rdi"),
        (0x778C7B, "cmp", "qword ptr [rbx + 0x28], rdi"),
        (0x778C7F, "jne", "0x778c9b"),
        (0x778C81, "mov", "edx, 0x88"), (0x778C89, "call", "0x17a7808"),
        (0x778C9B, "mov", "ecx, dword ptr [rbx + 0x28]"),
        (0x778CA4, "mov", "dword ptr [rbx + 0x30], edi"),
        (0x778CA9, "mov", "qword ptr [rbx + 0x28], rdi"),
        (0x778CB7, "ret", ""), (0x76DCB4, "call", "0x6a3420"),
        (0xF88B0, "jmp", "0x778cc0"),
        (0x778CC9, "cmp", "dword ptr [rcx + 0x60], 0"),
        (0x778CD0, "je", "0x778cf4"),
        (0x778CD9, "mov", "ecx, dword ptr [rcx + 0x28]"),
        (0x778CE4, "mov", "dword ptr [rbx + 0x30], eax"),
        (0x778CE7, "mov", "qword ptr [rbx + 0x28], rax"),
        (0x778CF4, "cmp", "qword ptr [rcx], 0"),
        (0x778CF8, "je", "0x778d6d"),
        (0x778D6D, "mov", "edx, 0x88"),
        (0x778D7D, "jmp", "0x17a7808"),
        (0x778C45, "mov", "dword ptr [rax + 0x28], edi"),
        (0x778C48, "mov", "qword ptr [rax + 0x10], rdi"),
        (0x778C4C, "mov", "qword ptr [rax + 0x18], rdi"),
        (0x778C50, "mov", "qword ptr [rax + 0x20], rdi"),
        (0x778C58, "add", "rcx, 8"), (0x778C5C, "call", "0x33cc90"),
        (0x778C61, "mov", "qword ptr [rbx + 0x78], rdi"),
        (0x33CC96, "mov", "ebp, 0xffffffff"), (0x33CC9B, "mov", "eax, ebp"),
        (0x33CC9D, "lock xadd", "dword ptr [rcx + 4], eax"),
        (0x33CCA2, "cmp", "eax, 1"), (0x33CCA5, "jne", "0x33cf80"),
        (0x33CF85, "ret", ""),
        (0x81BD1D, "mov", "r8, rsi"), (0x81BD38, "call", "0x75fc00"),
        (0x81BD4B, "mov", "rax, qword ptr [rax]"),
        (0x81BD58, "mov", "qword ptr [rcx + 0x1c8], rax"),
        (0x8211D0, "call", "0x81a880"), (0x779730, "call", "0x8211c0"),
        (0x77973B, "mov", "rdx, qword ptr [rax + 0x1c8]"),
        (0x779742, "call", "rsi"), (0x779755, "mov", "rdi, qword ptr [rdi + 0x10]"),
        (0x7797FF, "mov", "qword ptr [rdi + 0x10], rax"),
        (0x77980A, "mov", "qword ptr [rdi + 0x18], rax"),
        (0x77980E, "mov", "qword ptr [rdi + 0x20], r14"),
        (0x779824, "mov", "qword ptr [r14 + 0x78], rdi"),
        (0x779828, "lock inc", "dword ptr [rdi + 0xc]"),
    ]
    for rva, mnemonic, operands in checks:
        ins = next(cs.disasm(pe.get_data(rva, 15), rva))
        if (ins.mnemonic, ins.op_str) != (mnemonic, operands):
            raise ValueError(f"native release relationship mismatch at {rva:#x}")
    for rva, mnemonic in [(0x76DCB9, "mov"), (0x778C6E, "call"), (0x778C9E, "call"),
                           (0x778CDC, "call")]:
        ins = next(cs.disasm(pe.get_data(rva, 15), rva))
        refs = [rva + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if ins.mnemonic != mnemonic or refs != [0x1CD6038]:
            raise ValueError("release GC-free export storage mismatch")
    ins = next(cs.disasm(pe.get_data(0x76DCAD, 15), 0x76DCAD))
    target = ins.address + ins.size + ins.operands[1].mem.disp
    if ins.mnemonic != "lea" or pe.get_data(target, 80).split(b"\0")[0] != b"il2cpp_gchandle_free":
        raise ValueError("unexpected release GC export")
    for rva, expected in [(0x81B729, "UnityEngine"), (0x81BD16, "AsyncOperation"),
                           (0x81BD20, "UnityEngine.CoreModule.dll")]:
        ins = next(cs.disasm(pe.get_data(rva, 15), rva))
        target = ins.address + ins.size + ins.operands[1].mem.disp
        if ins.mnemonic != "lea" or pe.get_data(target, len(expected) + 1) != expected.encode() + b"\0":
            raise ValueError("auxiliary AsyncOperation type identity mismatch")
    for rva, mnemonic, expected in [(0x81BD3D, "mov", 0x1CD6AF8),
            (0x8211C4, "mov", 0x1CD6AF8), (0x8211E8, "mov", 0x1CD6AF8),
            (0x7797F8, "lea", 0x778B30), (0x779803, "lea", 0x778BD0)]:
        ins = next(cs.disasm(pe.get_data(rva, 15), rva))
        refs = [rva + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if ins.mnemonic != mnemonic or refs != [expected]:
            raise ValueError("auxiliary callback/type-cache binding mismatch")
    name = "UnityEngine.Coroutine::ReleaseCoroutine"
    name_rva = struct.unpack("<Q", pe.get_data(0x189BB80 + 2225 * 8, 8))[0] - pe.OPTIONAL_HEADER.ImageBase
    target_rva = struct.unpack("<Q", pe.get_data(0x1894FC0 + 2225 * 8, 8))[0] - pe.OPTIONAL_HEADER.ImageBase
    if target_rva != 0xF88B0 or pe.get_data(name_rva, len(name) + 1) != name.encode() + b"\0":
        raise ValueError("registered Coroutine finalizer binding mismatch")
    managed_checks = [(0x1C7ADA2, "mov", "rbx, qword ptr [rax + 0x10]"),
        (0x1C7ADB9, "call", "0x2b7df0"), (0x1C7ADC5, "mov", "rcx, rbx"),
        (0x1C7ADC8, "call", "rax"), (0x1C7AE1C, "call", "0x2b7df0"),
        (0x1C7AE30, "jmp", "rax"), (0x2B7DF9, "call", "0x265bc0"),
        (0x265D4D, "je", "0x265d58"),
        (0x265DAE, "mov", "edx, 0x28"), (0x265DB6, "call", "0x30d670"),
        (0x265DC4, "sub", "rax, rsi"), (0x265DF4, "mov", "r8, rax"),
        (0x265DFB, "call", "0x243950"), (0x265F29, "mov", "r14, qword ptr [r15 + 0x40]")]
    for rva, mnemonic, operands in managed_checks:
        ins = next(cs.disasm(managed.get_data(rva, 15), rva))
        if (ins.mnemonic, ins.op_str) != (mnemonic, operands):
            raise ValueError("managed Coroutine finalizer bridge mismatch")
    cache_targets = []
    for rva in (0x1C7ADA6, 0x1C7ADBE, 0x1C7AE06, 0x1C7AE21):
        ins = next(cs.disasm(managed.get_data(rva, 15), rva))
        refs = [rva + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if ins.mnemonic != "mov" or len(refs) != 1:
            raise ValueError("missing finalizer internal-call cache reference")
        cache_targets.extend(refs)
    if len(set(cache_targets)) != 1:
        raise ValueError("finalizer and release wrapper use different caches")
    request_name = name + "(System.IntPtr)"
    for rva in (0x1C7ADB2, 0x1C7AE15):
        ins = next(cs.disasm(managed.get_data(rva, 15), rva))
        target = ins.address + ins.size + ins.operands[1].mem.disp
        if ins.mnemonic != "lea" or managed.get_data(target, len(request_name) + 1) != request_name.encode() + b"\0":
            raise ValueError("managed finalizer resolves a different internal call")
    lookup_tables = []
    for rva in (0x265C16, 0x265E4C):
        ins = next(cs.disasm(managed.get_data(rva, 15), rva))
        refs = [rva + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if ins.mnemonic != "mov" or len(refs) != 1:
            raise ValueError("missing internal-call lookup table")
        lookup_tables.extend(refs)
    if lookup_tables[0] != lookup_tables[1]:
        raise ValueError("signature fallback uses a different lookup table")
    native = NativeRelease(data)
    results = []

    def run(name, value, events, entry="release_native"):
        result = native.call(entry, native.payload)
        if (value is not None and result != value) or native.events != events:
            raise ValueError(f"native release mismatch: {name}: {result}, {native.events}")
        snapshot = native.snapshot()
        results.append({"name": name, "return_value": result if value is not None else None, "events": list(native.events),
                        "post_call_emulated_storage": snapshot})
        return snapshot

    native.prepare(count=2, enumerator_handle=17, secondary_handle=29)
    snapshot = run("positive remaining reference returns one without cleanup", 1, [])
    if snapshot["references"] != 1 or snapshot["cleanup_flag"] != 0 or snapshot["list_links_zero"]:
        raise ValueError("positive-reference fast path mutated lifetime state")
    native.prepare()
    snapshot = run("last reference without handles unlinks and frees payload", 0, [["free_payload", "requested"]])
    if snapshot["references"] != 0 or snapshot["cleanup_flag"] != 1 or not snapshot["list_links_zero"]:
        raise ValueError("last-reference cleanup mismatch")
    native.prepare(enumerator_handle=17)
    snapshot = run("enumerator handle freed before native payload", 0,
                   [["free_handle", 17], ["free_payload", "requested"]])
    if snapshot["enumerator_handle"] != 0 or snapshot["enumerator_mode"] != 0:
        raise ValueError("enumerator handle fields retained after cleanup")
    native.prepare(enumerator_handle=17, secondary_handle=29)
    snapshot = run("secondary handle branch clears handles without freeing payload in this call", 0,
                   [["free_handle", 17], ["free_handle", 29]])
    if any(snapshot[key] != 0 for key in ("enumerator_handle", "enumerator_mode", "secondary_handle", "secondary_mode")):
        raise ValueError("secondary handle cleanup mismatch")
    native.prepare(enumerator_handle=0x100000011, secondary_handle=0x20000001D)
    run("GC free receives low 32 bits of each nonzero handle field", 0,
        [["free_handle", 17], ["free_handle", 29]])
    native.prepare()
    child = native.payload + 0x100
    native.payload_labels[child] = "linked_68"
    native.uc.mem_write(child, bytes(0x88))
    native.uc.mem_write(child + 0x60, struct.pack("<i", 1))
    native.uc.mem_write(child + 0x70, struct.pack("<Q", native.payload))
    native.uc.mem_write(native.payload + 0x68, struct.pack("<Q", child))
    snapshot = run("linked 68 payload releases recursively before caller payload", 0,
                   [["free_payload", "linked_68"], ["free_payload", "requested"]])
    if not snapshot["link_68_zero"] or not native.snapshot(child)["link_70_zero"]:
        raise ValueError("recursive release retained reciprocal links")
    native.prepare()
    linked = native.payload + 0x100
    native.uc.mem_write(linked, bytes(0x88))
    native.uc.mem_write(linked + 0x68, struct.pack("<Q", native.payload))
    native.uc.mem_write(native.payload + 0x70, struct.pack("<Q", linked))
    snapshot = run("70 link clears reciprocal 68 without releasing linked object", 0,
                   [["free_payload", "requested"]])
    if not snapshot["link_70_zero"] or native.qword(linked + 0x68) != 0:
        raise ValueError("reciprocal link cleanup mismatch")
    native.prepare(count=2, enumerator_handle=17, secondary_handle=29)
    snapshot = run("managed cleanup while active clears secondary handle without decrementing references", None,
                   [["free_handle", 29]], entry="cleanup_gc")
    if snapshot["references"] != 2 or snapshot["secondary_handle"] != 0 or snapshot["list_links_zero"]:
        raise ValueError("active managed-wrapper cleanup changed native reference/list ownership")
    native.events = []
    run("first later native release remains active", 1, [])
    native.events = []
    run("last native release after wrapper cleanup frees allocation", 0,
        [["free_handle", 17], ["free_payload", "requested"]])
    native.prepare(enumerator_handle=17, secondary_handle=29)
    run("native completion before wrapper cleanup defers allocation release", 0,
        [["free_handle", 17], ["free_handle", 29]])
    native.events = []
    run("later managed cleanup frees zero-reference detached allocation", None,
        [["free_payload", "requested"]], entry="cleanup_gc")
    for count in (2, 3):
        native.prepare(enumerator_handle=17)
        auxiliary = native.payload + 0x400
        native.uc.mem_write(auxiliary, bytes(0x30))
        native.uc.mem_write(auxiliary + 0xC, struct.pack("<I", count))
        native.uc.mem_write(auxiliary + 0x10, struct.pack("<QQQI", 0x1111, 0x2222, 0x3333, 7))
        native.uc.mem_write(native.payload + 0x78, struct.pack("<Q", auxiliary))
        snapshot = run(f"auxiliary waiter retained by {count - 1} other references is cleared and released", 0,
                       [["free_handle", 17], ["free_payload", "requested"]])
        remaining = struct.unpack("<I", native.uc.mem_read(auxiliary + 0xC, 4))[0]
        cleared = bytes(native.uc.mem_read(auxiliary + 0x10, 0x1C)) == bytes(0x1C)
        if not snapshot["auxiliary_78_zero"] or remaining != count - 1 or not cleared:
            raise ValueError("retained auxiliary object cleanup mismatch")
        results[-1]["retained_auxiliary"] = {"remaining_references": remaining, "fields_10_through_28_cleared": cleared}
    return {"schema_version": 1, "unityplayer_sha256": digest,
            "native_relationships_verified": len(checks) + 5 + len(managed_checks) + 16,
            "managed_input_hashes": {name: item["sha256"] for name, item in pinned.items()},
            "registration_relationships_rechecked": registration["semantic_checks_passed"],
            "registered_binding": {"name": name, "index": 2225, "rva": "0xf88b0", "cleanup_rva": "0x778cc0"},
            "managed_finalizer_rva": "0x1c7ad80", "managed_release_wrapper_rva": "0x1c7ae00",
            "managed_request": request_name,
            "signature_fallback": "0x265bc0 retries the same table with the prefix before '(' after a full-name miss",
            "release_rva": "0x778bd0", "gc_free_export": "il2cpp_gchandle_free",
            "emulator": "Unicorn 2.1.4 x86-64", "cases": len(results), "results": results,
            "synthetic_boundaries": ["GC-handle free", "0x88-byte allocation free"],
            "storage_note": "free sinks record calls without unmapping memory; snapshots are emulated storage only",
            "auxiliary_support": "0x78 object fields cleared and atomic reference released only with initial count >=2; final-reference branch excluded",
            "auxiliary_type_binding": "UnityEngine.AsyncOperation cache+0x1c8; native pointer from yielded object+0x10; callbacks/payload installed before retaining into coroutine+0x78",
            "unresolved": ["concrete AsyncOperation subtype and last-reference destructor/allocator branch",
                           "zero-reference but still-linked diagnostic path in managed cleanup",
                           "validity of arbitrary reference/link graphs and full lifetime integration"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unityplayer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.unityplayer)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['cases']} native release cases and {report['native_relationships_verified']} relationships")
