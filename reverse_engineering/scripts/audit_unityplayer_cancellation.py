"""Audit registered coroutine stop bindings and bounded cancellation behavior.

Executes valid managed-wrapper inputs and native queue matching/unlinking in an
isolated emulator. GC field writes and release bodies are synthetic boundaries;
full reference-count destruction and nested coroutine lifetime are not claimed.
"""
import argparse
import json
import struct
from pathlib import Path

from audit_unityplayer_icalls import audit as audit_icalls
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint
from audit_unityplayer_wait_consumer import NativeConsumer


class NativeCancellation(NativeConsumer):
    routines = {**NativeConsumer.routines, "stop_handle": (0x100F80, 0x101182),
                "unlink_coroutine": (0x77D120, 0x77D33B),
                "stop_all": (0xFF000, 0xFF13D), "remove_owner_waits": (0x43B8E0, 0x43BAF2),
                "stop_enumerator": (0x101190, 0x1012B5), "match_enumerator": (0x77D340, 0x77D54D)}

    def __init__(self, data):
        super().__init__(data)
        self.gc_write = self.stop + 0x300
        self.gc_target = self.stop + 0x400
        self.native_owner = self.arena + 0x107000
        self.managed_owner = self.arena + 0x108000
        self.managed_handle = self.arena + 0x109000
        self.enumerator = self.arena + 0x10A000
        self.payload = self.arena + 0x110000
        self.reset_stop()

    def reset_stop(self):
        self.reset_case()
        self.payload_labels = {self.payload: "requested"}
        self.gc_targets = {}
        self.uc.mem_write(self.native_owner, bytes(0x100))
        self.uc.mem_write(self.native_owner + 8, struct.pack("<I", 11))
        self.uc.mem_write(self.payload, bytes(0x88))
        head = self.native_owner + 0x70
        self.uc.mem_write(head, struct.pack("<QQ", self.payload, self.payload))
        self.uc.mem_write(self.payload, struct.pack("<QQ", head, head))
        self.uc.mem_write(self.payload + 0x58, struct.pack("<Q", self.native_owner))
        self.uc.mem_write(self.managed_owner + 0x10, struct.pack("<Q", self.native_owner))
        self.uc.mem_write(self.managed_handle + 0x10, struct.pack("<Q", self.payload))
        self.uc.mem_write(self.base + 0x1CD6688, struct.pack("<Q", self.gc_write))
        self.uc.mem_write(self.base + 0x1CD6360, struct.pack("<Q", self.gc_target))
        self.uc.mem_write(self.base + 0x1C6E720, struct.pack("<Q", self.owner))

    def _on_code(self, uc, address, size, _):
        if address == getattr(self, "gc_target", None):
            handle = uc.reg_read(self.x86.UC_X86_REG_ECX)
            self._return(self.gc_targets[handle])
            return
        if address - self.base == 0x3EB6D0:
            # StopAll's owner-filter helper inlines cursor-aware removal instead
            # of entering 0x43bb00; track its actual native erase here as well.
            node = uc.reg_read(self.x86.UC_X86_REG_RDX)
            if node in self.records:
                identity = struct.unpack_from("<Q", self.records[node], 0x10)[0]
                self.events.append(["erase", identity])
                del self.records[node]
                del self.nodes[identity]
                del self.deadlines[identity]
        if address == getattr(self, "gc_write", None):
            target = uc.reg_read(self.x86.UC_X86_REG_RDX)
            value = uc.reg_read(self.x86.UC_X86_REG_R8)
            uc.mem_write(target, struct.pack("<Q", value))
            self._return()
            return
        if address == getattr(self, "release", None):
            pointer = uc.reg_read(self.x86.UC_X86_REG_RCX)
            self.events.append(["release_payload", self.payload_labels[pointer]])
            self._return()
            return
        super()._on_code(uc, address, size, _)

    def add_stop_wait(self, identity, owner_id=11, callback_matches=True,
                      payload_kind="requested", wrapper_flag=False, linked=False):
        self.add_wait(identity)
        pointer = self.payload
        if payload_kind != "requested":
            pointer = self.payload + identity * 0x100
            self.uc.mem_write(pointer, bytes(0x88))
            self.uc.mem_write(pointer + 0x80, bytes([int(wrapper_flag)]))
            self.uc.mem_write(pointer + 0x68, struct.pack("<Q", self.payload if linked else 0))
            self.payload_labels[pointer] = payload_kind
        node = self.nodes[identity]
        record = bytearray(self.records[node])
        struct.pack_into("<Q", record, 0x18, pointer)
        struct.pack_into("<Q", record, 0x20, self.base + (0x778B30 if callback_matches else 0x778BD0))
        struct.pack_into("<I", record, 0x30, owner_id)
        self.records[node] = bytes(record)
        self.uc.mem_write(node + 0x20, bytes(record))
        self.validate()

    def stop_handle(self, null_handle=False):
        if null_handle:
            self.uc.mem_write(self.managed_handle + 0x10, bytes(8))
        self.events = []
        self.call("stop_handle", self.managed_owner, self.managed_handle)
        self.validate()
        return self.stop_result()

    def stop_all(self):
        self.events = []
        self.call("stop_all", self.managed_owner)
        self.validate()
        return self.stop_result()

    def set_enumerator(self, pointer=None, matches=True, cached=True, has_handle=True):
        pointer = self.payload if pointer is None else pointer
        handle = (pointer - self.payload) // 0x100 + 1
        target = self.enumerator if matches else self.enumerator + 0x100
        self.gc_targets[handle] = target
        self.uc.mem_write(pointer + 0x10, struct.pack("<Q", handle if has_handle else 0))
        self.uc.mem_write(pointer + 0x18, struct.pack("<I", 2 if cached else 1))
        self.uc.mem_write(pointer + 0x20, struct.pack("<Q", target if cached else 0))

    def stop_enumerator(self, null_enumerator=False):
        self.events = []
        self.call("stop_enumerator", self.managed_owner, 0 if null_enumerator else self.enumerator)
        self.validate()
        return self.stop_result()

    def stop_result(self):
        head = self.native_owner + 0x70
        detached = (self.qword(self.payload) == 0 and self.qword(self.payload + 8) == 0 and
                    self.qword(self.payload + 0x58) == 0)
        return {"events": list(self.events), "remaining": list(self.deadlines),
                "requested_payload_detached": detached,
                "owner_list_empty": self.qword(head) == head and self.qword(head + 8) == head,
                "cursor_is_end": self.qword(self.owner + 0x40) == self.head}


def audit(path):
    import pefile
    import capstone

    data = Path(path).read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    registration = audit_icalls(path)
    pe = pefile.PE(data=data, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    selected = [(2264, "UnityEngine.MonoBehaviour::StopAllCoroutines", 0xFF000),
                (2275, "UnityEngine.MonoBehaviour::StopCoroutineManaged", 0x100F80),
                (2276, "UnityEngine.MonoBehaviour::StopCoroutineFromEnumeratorManaged", 0x101190)]
    for index, name, target in selected:
        pointer = struct.unpack("<Q", pe.get_data(0x189BB80 + index * 8, 8))[0] - base
        function = struct.unpack("<Q", pe.get_data(0x1894FC0 + index * 8, 8))[0] - base
        if function != target or pe.get_data(pointer, len(name) + 1) != name.encode() + b"\0":
            raise ValueError("registered cancellation binding mismatch")
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    checks = [
        (0x101008, "mov", "rdi, qword ptr [rax + 0x10]"),
        (0x101068, "test", "rdi, rdi"), (0x10106B, "je", "0x101141"),
        (0x10107D, "mov", "r14d, dword ptr [rax + 8]"),
        (0x1010EC, "cmp", "dword ptr [r8 + 0x50], r14d"),
        (0x1010F4, "jne", "0x101124"),
        (0x1010F6, "cmp", "r15, qword ptr [rdx + 0x20]"),
        (0x1010FA, "jne", "0x101124"),
        (0x1010FC, "mov", "rax, qword ptr [rdx + 0x18]"),
        (0x101100, "cmp", "rax, rdi"), (0x101103, "je", "0x10111c"),
        (0x101105, "cmp", "byte ptr [rax + 0x80], 0"),
        (0x10110C, "je", "0x101124"),
        (0x10110E, "mov", "rcx, qword ptr [rax + 0x68]"),
        (0x101112, "test", "rcx, rcx"), (0x101115, "je", "0x101124"),
        (0x101117, "cmp", "rcx, rdi"), (0x10111A, "jne", "0x101124"),
        (0x10111F, "call", "0x43bb00"),
        (0x101124, "mov", "r8, rbx"), (0x10112B, "jne", "0x1010a0"),
        (0x10113C, "call", "0x77d120"),
        (0x77D12D, "xor", "ebp, ebp"),
        (0x77D13B, "mov", "qword ptr [rcx + 8], rax"),
        (0x77D146, "mov", "qword ptr [rdx], rax"),
        (0x77D149, "mov", "qword ptr [rdi], rbp"),
        (0x77D14C, "mov", "qword ptr [rdi + 8], rbp"),
        (0x77D163, "mov", "qword ptr [rdi + 0x58], rbp"),
        (0xFF0DC, "call", "0x43b8e0"), (0xFF0EB, "call", "0x77d120"),
        (0x101270, "call", "0x77d340"), (0x76DAEC, "call", "0x6a3420"),
        (0xFF0C1, "lea", "rdi, [rbx + 0x70]"),
        (0xFF0C5, "cmp", "qword ptr [rdi + 8], rdi"),
        (0xFF0C9, "je", "0xff0f6"),
        (0xFF0E1, "cmp", "qword ptr [rdi + 8], rdi"),
        (0xFF0E5, "je", "0xff0f6"),
        (0xFF0E7, "mov", "rcx, qword ptr [rbx + 0x78]"),
        (0xFF0F4, "jne", "0xff0e7"),
        (0x43B96D, "cmp", "dword ptr [rdx + 0x50], edi"),
        (0x43B970, "jne", "0x43bac8"),
        (0x43B976, "mov", "rax, qword ptr [rdx + 0x40]"),
        (0x43B97A, "cmp", "r12, rax"), (0x43B97D, "je", "0x43b988"),
        (0x43B97F, "test", "rax, rax"), (0x43B982, "jne", "0x43bac8"),
        (0x43B9DD, "call", "0x3eb6d0"), (0x43BAA2, "call", "0x3eb6d0"),
        (0x43BAC6, "call", "rbp"),
        (0x77D3EC, "cmp", "dword ptr [rdi + 0x50], r12d"),
        (0x77D3F0, "jne", "0x77d4ac"),
        (0x77D3F6, "cmp", "rcx, qword ptr [rdi + 0x40]"),
        (0x77D3FA, "jne", "0x77d4ac"),
        (0x77D404, "mov", "rcx, qword ptr [rbp + 0x10]"),
        (0x77D408, "test", "rcx, rcx"), (0x77D40B, "je", "0x77d4a5"),
        (0x77D411, "cmp", "dword ptr [rbp + 0x18], 2"),
        (0x77D417, "mov", "rax, qword ptr [rbp + 0x20]"),
        (0x77D424, "call", "rax"),
        (0x77D441, "cmp", "rax, rsi"), (0x77D444, "je", "0x77d496"),
        (0x77D446, "cmp", "byte ptr [rbp + 0x80], 0"),
        (0x77D44D, "je", "0x77d4a5"),
        (0x77D44F, "mov", "rax, qword ptr [rbp + 0x68]"),
        (0x77D458, "cmp", "dword ptr [rax + 0x18], 2"),
        (0x77D45E, "mov", "rcx, qword ptr [rax + 0x20]"),
        (0x77D491, "cmp", "rcx, rsi"), (0x77D4A0, "call", "0x43bb00"),
        (0x77D536, "call", "0x77d120"), (0x77D54C, "ret", ""),
    ]
    for rva, mnemonic, operands in checks:
        ins = next(cs.disasm(pe.get_data(rva, 15), rva))
        if (ins.mnemonic, ins.op_str) != (mnemonic, operands):
            raise ValueError(f"native cancellation relationship mismatch at {rva:#x}")
    references = [(0x101099, "lea", 0x778B30), (0x101071, "mov", 0x1C6E720),
                  (0x76DAF1, "mov", 0x1CD6688), (0x43B908, "lea", 0x778B30),
                  (0x76DC93, "mov", 0x1CD6360), (0x77D41D, "mov", 0x1CD6360)]
    for rva, mnemonic, target in references:
        ins = next(cs.disasm(pe.get_data(rva, 15), rva))
        refs = [rva + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if ins.mnemonic != mnemonic or refs != [target]:
            raise ValueError("native cancellation reference mismatch")
    ins = next(cs.disasm(pe.get_data(0x76DAE5, 15), 0x76DAE5))
    name_rva = ins.address + ins.size + ins.operands[1].mem.disp
    if ins.mnemonic != "lea" or pe.get_data(name_rva, 80).split(b"\0")[0] != b"il2cpp_gc_wbarrier_set_field":
        raise ValueError("synthetic write-barrier boundary is not the resolved export")
    ins = next(cs.disasm(pe.get_data(0x76DC87, 15), 0x76DC87))
    name_rva = ins.address + ins.size + ins.operands[1].mem.disp
    if ins.mnemonic != "lea" or pe.get_data(name_rva, 80).split(b"\0")[0] != b"il2cpp_gchandle_get_target":
        raise ValueError("synthetic GC-target boundary is not the resolved export")
    tree = NativeCancellation(data)
    results = []

    def check(name, remaining, releases, detached=True, mode="handle", **kwargs):
        result = (tree.stop_handle(**kwargs) if mode == "handle" else
                  tree.stop_enumerator(**kwargs) if mode == "enumerator" else tree.stop_all())
        actual = [value for kind, value in result["events"] if kind == "release_payload"]
        if result["remaining"] != remaining or actual != releases or result["requested_payload_detached"] != detached:
            raise ValueError(f"native cancellation result mismatch: {name}: {result}")
        results.append({"name": name, **result})

    tree.add_stop_wait(1)
    check("direct owner callback and payload match", [], ["requested"])
    tree.reset_stop()
    tree.add_stop_wait(1, owner_id=22)
    check("same payload with different owner key is retained", [1], [])
    tree.reset_stop()
    tree.add_stop_wait(1, callback_matches=False)
    check("same owner and payload with different callback is retained", [1], [])
    tree.reset_stop()
    tree.add_stop_wait(1, payload_kind="other")
    check("different payload without wrapper marker is retained", [1], [])
    tree.reset_stop()
    tree.add_stop_wait(1, payload_kind="wrapper", wrapper_flag=True, linked=True)
    check("marked wrapper linked to requested payload is removed", [], ["wrapper"])
    tree.reset_stop()
    tree.add_stop_wait(1, payload_kind="unmarked_link", linked=True)
    check("link without wrapper marker is insufficient", [1], [])
    tree.reset_stop()
    tree.add_stop_wait(1, payload_kind="unlinked_wrapper", wrapper_flag=True)
    check("wrapper marker without nonnull link is insufficient", [1], [])
    tree.reset_stop()
    tree.add_stop_wait(1)
    check("null native Coroutine handle is a no-op", [1], [], detached=False, null_handle=True)
    tree.reset_stop()
    check("empty wait queue still detaches the coroutine", [], [])
    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.add_stop_wait(2, owner_id=22)
    tree.add_stop_wait(3, payload_kind="wrapper", wrapper_flag=True, linked=True)
    tree.add_stop_wait(4)
    tree.uc.mem_write(tree.owner + 0x40, struct.pack("<Q", tree.nodes[3]))
    check("all matching records removed and saved successor advanced to end", [2],
          ["requested", "wrapper", "requested"])
    if not results[-1]["cursor_is_end"]:
        raise ValueError("handle cancellation failed to advance the saved cursor")

    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.add_stop_wait(2, payload_kind="other")
    tree.add_stop_wait(3, owner_id=22)
    tree.add_stop_wait(4, callback_matches=False)
    check("StopAll removes all matching-owner coroutine callbacks regardless of payload", [3, 4],
          ["requested", "other"], mode="all")
    tree.reset_stop()
    tree.add_stop_wait(1)
    node = tree.nodes[1]
    record = bytearray(tree.records[node])
    struct.pack_into("<Q", record, 0x20, 0)
    tree.records[node] = bytes(record)
    tree.uc.mem_write(node + 0x20, bytes(record))
    check("StopAll also removes matching-owner null callback records", [], ["requested"], mode="all")
    tree.reset_stop()
    tree.add_stop_wait(1)
    node = tree.nodes[1]
    record = bytearray(tree.records[node])
    struct.pack_into("<Q", record, 0x28, 0)
    tree.records[node] = bytes(record)
    tree.uc.mem_write(node + 0x20, bytes(record))
    check("StopAll honors null release slots", [], [], mode="all")
    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.add_stop_wait(2)
    tree.uc.mem_write(tree.owner + 0x40, struct.pack("<Q", tree.nodes[1]))
    check("StopAll advances a saved cursor through all removed records", [], ["requested", "requested"], mode="all")
    if not results[-1]["cursor_is_end"]:
        raise ValueError("StopAll failed to advance the saved cursor")
    tree.reset_stop()
    tree.add_stop_wait(1)
    head = tree.native_owner + 0x70
    tree.uc.mem_write(head, struct.pack("<QQ", head, head))
    tree.uc.mem_write(tree.payload, bytes(16))
    tree.uc.mem_write(tree.payload + 0x58, bytes(8))
    check("empty coroutine owner list skips StopAll queue removal", [1], [], mode="all")
    tree.reset_stop()
    check("StopAll with an empty queue still detaches the linked coroutine", [], [], mode="all")

    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.set_enumerator()
    check("IEnumerator stop matches cached mode-2 target", [], ["requested"], mode="enumerator")
    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.set_enumerator(cached=False)
    check("IEnumerator stop matches GC-handle target outside cached mode", [], ["requested"], mode="enumerator")
    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.set_enumerator(matches=False)
    check("different IEnumerator leaves wait and owner link intact", [1], [], detached=False, mode="enumerator")
    tree.reset_stop()
    tree.add_stop_wait(1, owner_id=22)
    tree.set_enumerator()
    check("IEnumerator owner-key filtering retains other-owner wait", [1], [], mode="enumerator")
    tree.reset_stop()
    tree.add_stop_wait(1, payload_kind="wrapper", wrapper_flag=True, linked=True)
    tree.set_enumerator()
    tree.set_enumerator(tree.payload + 0x100, matches=False)
    check("IEnumerator stop follows marked linked payload", [], ["wrapper"], mode="enumerator")
    tree.reset_stop()
    tree.add_stop_wait(1, payload_kind="unmarked", linked=True)
    tree.set_enumerator()
    tree.set_enumerator(tree.payload + 0x100, matches=False)
    check("IEnumerator stop does not follow an unmarked link", [1], [], mode="enumerator")
    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.set_enumerator()
    check("null IEnumerator is a no-op", [1], [], detached=False, mode="enumerator", null_enumerator=True)
    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.set_enumerator(has_handle=False)
    check("queued cache match still requires nonzero handle field", [1], [], mode="enumerator")
    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.add_stop_wait(2, payload_kind="second")
    second = tree.payload + 0x200
    head = tree.native_owner + 0x70
    tree.set_enumerator()
    tree.set_enumerator(second)
    tree.uc.mem_write(head, struct.pack("<QQ", second, tree.payload))
    tree.uc.mem_write(tree.payload, struct.pack("<QQ", head, second))
    tree.uc.mem_write(second, struct.pack("<QQ", tree.payload, head))
    tree.uc.mem_write(second + 0x58, struct.pack("<Q", tree.native_owner))
    check("IEnumerator removes all queued matches then unlinks first surviving owner-list match", [],
          ["requested", "second"], mode="enumerator")
    if results[-1]["owner_list_empty"] or tree.qword(second + 0x58) != tree.native_owner:
        raise ValueError("IEnumerator stop unexpectedly unlinked every owner-list match")

    tree.reset_stop()
    tree.add_stop_wait(1)
    tree.add_stop_wait(2, payload_kind="second")
    second = tree.payload + 0x200
    head = tree.native_owner + 0x70
    tree.uc.mem_write(head, struct.pack("<QQ", second, tree.payload))
    tree.uc.mem_write(tree.payload, struct.pack("<QQ", head, second))
    tree.uc.mem_write(second, struct.pack("<QQ", tree.payload, head))
    tree.uc.mem_write(second + 0x58, struct.pack("<Q", tree.native_owner))
    check("StopAll unlinks every surviving owner-list coroutine", [], ["requested", "second"], mode="all")
    if not results[-1]["owner_list_empty"] or tree.qword(second + 0x58) != 0:
        raise ValueError("StopAll failed to clear the complete owner list")
    return {"schema_version": 1, "unityplayer_sha256": digest,
            "registration_relationships_rechecked": registration["semantic_checks_passed"],
            "registered_pair_count": registration["registered_pair_count"],
            "cancellation_relationships_verified": len(checks) + len(references) + 2,
            "bindings": {name: {"index": index, "rva": hex(target)} for index, name, target in selected},
            "emulator": "Unicorn 2.1.4 x86-64",
            "synthetic_boundaries": ["il2cpp_gc_wbarrier_set_field", "il2cpp_gchandle_get_target",
                                     "node allocation", "release bodies"],
            "emulated_scope": "valid handle, IEnumerator and StopAll wrappers, matching wait removal, simple owner-list unlink",
            "cases": len(results), "results": results,
            "unresolved": ["nested coroutine stop branches and real release/refcount destruction",
                           "invalid managed wrappers and exception paths", "mutation from release bodies"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unityplayer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.unityplayer)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['cases']} cancellation cases and "
          f"{report['cancellation_relationships_verified']} native relationships")
