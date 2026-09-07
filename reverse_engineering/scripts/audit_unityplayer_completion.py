"""Audit native MoveNext completion, queue release and in-invocation lifetime.

The dispatcher, invocation-frame initializer, callback, queue and reference
release execute in isolated Unicorn. Managed invocation, GC APIs, diagnostic
sinks and owner-context lookup remain explicit synthetic boundaries. Controlled
in-invocation releases/StopCoroutine calls run native code on a separate stack.
"""
import argparse
import hashlib
import itertools
import json
import struct
from pathlib import Path

from audit_unityplayer_coroutine_release import NativeRelease, audit as audit_release
from audit_unityplayer_cancellation import NativeCancellation
from audit_unityplayer_coroutines import audit as audit_bridge
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


class NativeCompletion(NativeRelease):
    routines = {**NativeRelease.routines, "dispatch": (0x778D90, 0x779062),
                "callback_native": (0x778B30, 0x778BC3), "frame_init": (0x75D3A0, 0x75D44C)}

    def __init__(self, data):
        super().__init__(data)
        self.param_count = self.stop + 0x600
        self.invoke = self.stop + 0x700
        self.owner_context = self.stop + 0x800
        self.cache = self.arena + 0x120000
        self.owner_vtable = self.arena + 0x122000
        self.method = self.arena + 0x123000
        self.error_out = self.arena + 0x124000

    def prepare_step(self, count=1, continuing=False, exception=False, cached=True,
                     stop_owner=False, drop_references=0, secondary_handle=0, stop_mode="handle"):
        if not 1 <= count <= 4 or not 0 <= drop_references <= count:
            raise ValueError("unsupported controlled reference graph")
        self.prepare(count=count, enumerator_handle=17, secondary_handle=secondary_handle)
        self.gc_targets[17] = self.enumerator
        self.uc.mem_write(self.payload + 0x18, struct.pack("<I", 2 if cached else 1))
        self.uc.mem_write(self.payload + 0x20, struct.pack("<Q", self.enumerator if cached else 0))
        self.uc.mem_write(self.native_owner + 0x40, struct.pack("<Q", self.owner_vtable))
        self.uc.mem_write(self.owner_vtable + 0x10, struct.pack("<Q", self.owner_context))
        self.uc.mem_write(self.cache, bytes(0x1000))
        self.uc.mem_write(self.cache + 0xCF0, struct.pack("<Q", self.method))
        for slot, target in [(0x1CD6AF8, self.cache), (0x1CD6250, self.param_count),
                             (0x1CD62A0, self.invoke)]:
            self.uc.mem_write(self.base + slot, struct.pack("<Q", target))
        self.uc.mem_write(self.error_out, b"\x7f")
        self.stop_mode = stop_mode
        self.continuing, self.exception = continuing, exception
        self.stop_owner, self.drop_references = stop_owner, drop_references
        self.invocations = 0
        self.pending_release = 0
        self.pending_stop = False
        self.queue_owner_mode = "match"

    def _on_code(self, uc, address, size, user):
        x = self.x86
        if address == getattr(self, "param_count", None):
            if uc.reg_read(x.UC_X86_REG_RCX) != self.method:
                raise ValueError("wrong managed method descriptor")
            self._return(2)
            return
        if address == getattr(self, "owner_context", None):
            if uc.reg_read(x.UC_X86_REG_RCX) != self.native_owner + 0x40:
                raise ValueError("wrong owner-context receiver")
            result = uc.reg_read(x.UC_X86_REG_RDX)
            uc.mem_write(result, bytes(8))
            self._return(result)
            return
        if address == getattr(self, "invoke", None):
            if uc.reg_read(x.UC_X86_REG_RCX) != self.method:
                raise ValueError("wrong managed method descriptor")
            if uc.reg_read(x.UC_X86_REG_RDX) != 0:
                raise ValueError("static invocation received an instance")
            args = uc.reg_read(x.UC_X86_REG_R8)
            if self.qword(args) != self.enumerator:
                raise ValueError("wrong IEnumerator invocation argument")
            result_address = self.qword(self.qword(args + 8))
            uc.mem_write(result_address, bytes([int(self.continuing)]))
            error = uc.reg_read(x.UC_X86_REG_R9)
            if self.exception:
                uc.mem_write(error, struct.pack("<Q", self.arena + 0x125000))
            self.events.append(["invoke", self.continuing, self.exception])
            self.invocations += 1
            if self.drop_references or self.stop_owner:
                self.pending_release = self.drop_references
                self.pending_stop = self.stop_owner
                uc.emu_stop()
                return
            self._return()
            return
        rva = address - self.base
        if rva == 0x75BE00:
            self.events.append(["report_exception"])
            self._return()
            return
        if rva == 0x1049F20:
            self.events.append(["report_owner_mismatch"])
            self._return()
            return
        if rva == 0x779070:
            self.events.append(["current_yield_boundary"])
            self._return()
            return
        if rva == 0x151EF0:
            self.uc.mem_write(self.owner_entry, struct.pack("<QQQ", 11, 0,
                self.native_owner if self.queue_owner_mode == "match" else self.native_owner + 0x100))
            self._return(self.owner_entry)
            return
        if rva == 0x778BD0:
            pointer = uc.reg_read(x.UC_X86_REG_RCX)
            count = struct.unpack("<i", uc.mem_read(pointer + 0x60, 4))[0]
            self.events.append(["release_reference", count])
        if any(a <= rva and rva + size <= b for a,b in NativeCancellation.routines.values()) or rva in (0x677920, 0x6E78E0, 0x355F00):
            return NativeCancellation._on_code(self,uc,address,size,user)
        if address in (getattr(self, "gc_write", None), getattr(self, "gc_target", None)):
            return NativeCancellation._on_code(self, uc, address, size, user)
        if any(a <= rva and rva + size <= b for a,b in [(0x778B30,0x778BC3),
                (0x778D90,0x779062),(0x75D3A0,0x75D44C)]):
            self.instructions.add(rva)
            return
        return super()._on_code(uc,address,size,user)

    def call(self, name, *args):
        x = self.x86
        rsp = self.stack + self.call_stack_offset
        self.uc.mem_write(rsp, struct.pack("<Q", self.stop) + bytes(0x28))
        self.uc.reg_write(x.UC_X86_REG_RSP, rsp)
        self.uc.reg_write(x.UC_X86_REG_EFLAGS, 2)
        self.uc.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        for register, value in zip((x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                x.UC_X86_REG_R8, x.UC_X86_REG_R9), args):
            self.uc.reg_write(register,value)
        address = self.base + self.routines[name][0]
        for _ in range(3):
            self.uc.emu_start(address, self.stop, timeout=1_000_000,count=100_000)
            if self.uc.reg_read(x.UC_X86_REG_RIP) == self.stop:
                break
            if not self.pending_release and not self.pending_stop:
                raise ValueError("dispatch exceeded execution bound")
            saved = self.uc.context_save()
            release_count = self.pending_release
            stop_owner = self.pending_stop
            self.pending_release = 0
            self.pending_stop = False
            offset = self.call_stack_offset
            try:
                self.call_stack_offset = 0x7008
                if stop_owner:
                    self.events.append([f"stop_{self.stop_mode}_during_invoke"])
                    if self.stop_mode == "all":
                        self.call("stop_all", self.managed_owner)
                    else:
                        self.call("stop_handle",self.managed_owner,self.managed_handle)
                for _ in range(release_count):
                    self.call("release_native",self.payload)
            finally:
                self.call_stack_offset = offset
            self.uc.context_restore(saved)
            self._return()
            address = self.uc.reg_read(x.UC_X86_REG_RIP)
        else:
            raise ValueError("too many in-invocation native actions")
        self.operation_count += 1
        return self.uc.reg_read(x.UC_X86_REG_RAX)

    def add_completion_wait(self, identity, payload_kind="requested"):
        self.add_stop_wait(identity,payload_kind=payload_kind)
        node = self.nodes[identity]
        record = bytearray(self.records[node])
        struct.pack_into("<Q",record,0x28,self.base + 0x778BD0)
        self.records[node] = bytes(record)
        self.uc.mem_write(node+0x20,bytes(record))

    def drain_completion(self):
        self.add_completion_wait(0)
        node = self.nodes[0]
        record = bytearray(self.records[node])
        struct.pack_into("<Q", record, 0x28, self.base + 0x778BD0)
        self.records[node] = bytes(record)
        self.uc.mem_write(node + 0x20,bytes(record))
        self.uc.mem_write(self.engine + 0x90,struct.pack("<d",1.0))
        self.uc.mem_write(self.engine + 0xC8,struct.pack("<q",1))
        self.events = []
        self.call("drain",self.owner,2)
        self.validate()



def expected_events(count, continuing, exception, stop_owner, dropped, secondary, queued):
    """Independent authored oracle for the bounded, unlinked-child graph."""
    events = [["erase", 0]] if queued else []
    events.append(["invoke", continuing, exception])
    if stop_owner:
        events.append(["stop_handle_during_invoke"])
    events.extend(["release_reference", count + 1 - i] for i in range(dropped))
    if exception:
        events.append(["report_exception"])
    remaining = count - dropped
    events.append(["release_reference", remaining + 1])
    if remaining and not exception:
        if continuing:
            if not stop_owner:
                events.append(["current_yield_boundary"])
        else:
            events.append(["release_reference", remaining + 1])
    if queued and remaining:
        events.append(["release_reference", remaining])
    final = max(0, remaining - int(queued))
    if final == 0:
        events.append(["free_handle", 17])
        events.append(["free_handle", secondary] if secondary else ["free_payload", "requested"])
    return events, final, int(remaining != 0)


def audit(path):
    import capstone
    import pefile

    data = Path(path).read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    bridge = audit_bridge(Path(path).parent)
    release = audit_release(path)
    pe = pefile.PE(data=data, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    checks = [
        (0x778B3D, "cmp", "qword ptr [rdx + 0x58], rcx"),
        (0x778B41, "je", "0x778bb2"), (0x778BA0, "call", "0x1049f20"),
        (0x778BA5, "mov", "eax, 1"), (0x778BB2, "xor", "edx, edx"),
        (0x778BB4, "mov", "rcx, rax"), (0x778BBE, "jmp", "0x778d90"),
        (0x75D3E8, "mov", "dword ptr [rcx + 0xf0], esi"),
        (0x75D431, "mov", "qword ptr [rdi + 0xf8], rbx"),
        (0x75D43D, "mov", "word ptr [rdi + 0x128], 0x101"),
        (0x778E03, "cmp", "dword ptr [rsi + 0x18], 2"),
        (0x778E09, "mov", "r8, qword ptr [rsi + 0x20]"),
        (0x778E0F, "mov", "rcx, qword ptr [rsi + 0x10]"),
        (0x778E75, "mov", "qword ptr [rsp + rax*8 + 0x60], rcx"),
        (0x778E87, "mov", "qword ptr [rbp + rcx*8 - 0x40], rax"),
        (0x778EA6, "call", "qword ptr [rax + 0x10]"),
        (0x778F2F, "lea", "r9, [rsp + 0x20]"),
        (0x778F34, "lea", "r8, [rbp - 0x40]"),
        (0x778F8E, "call", "0x75be00"),
        (0x778FB5, "mov", "ebx, dword ptr [rsi + 0x60]"),
        (0x778FBB, "call", "0x778bd0"),
        (0x778FC0, "cmp", "ebx, 1"), (0x778FC3, "je", "0x779034"),
        (0x778FE4, "mov", "byte ptr [r14], al"),
        (0x778FF6, "inc", "dword ptr [rsi + 0x60]"),
        (0x779002, "mov", "qword ptr [rbx + 0x70], r15"),
        (0x779006, "mov", "qword ptr [rsi + 0x68], r15"),
        (0x779017, "call", "0x778d90"), (0x77901F, "call", "0x778bd0"),
        (0x779024, "mov", "ebx, dword ptr [rsi + 0x60]"),
        (0x77902A, "call", "0x778bd0"), (0x77902F, "cmp", "ebx, 1"),
        (0x779034, "xor", "eax, eax"),
        (0x77904D, "cmp", "qword ptr [rsi + 0x58], r15"),
        (0x779056, "call", "0x779070"), (0x77905B, "mov", "eax, 1"),
    ]
    for rva, mnemonic, operands in checks:
        ins = next(cs.disasm(pe.get_data(rva,15),rva))
        if (ins.mnemonic,ins.op_str) != (mnemonic,operands):
            raise ValueError(f"completion relationship mismatch at {rva:#x}")
    references = [(0x778F22, "call", 0x1CD6250), (0x778F28,"mov",0x1CD62A0),
                  (0x76DFB1,"mov",0x1CD6250), (0x76E367,"mov",0x1CD62A0)]
    for rva, mnemonic, target in references:
        ins = next(cs.disasm(pe.get_data(rva,15),rva))
        refs = [rva + ins.size + op.mem.disp for op in ins.operands
                if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if ins.mnemonic != mnemonic or refs != [target]:
            raise ValueError("completion export storage mismatch")
    for rva, name in [(0x76DFA5,"il2cpp_method_get_param_count"),(0x76E35B,"il2cpp_runtime_invoke")]:
        ins = next(cs.disasm(pe.get_data(rva,15),rva))
        target = rva + ins.size + ins.operands[1].mem.disp
        if ins.mnemonic != "lea" or pe.get_data(target,len(name)+1) != name.encode() + b"\0":
            raise ValueError("unexpected managed invocation export name")
    native = NativeCompletion(data)
    results = []
    for queued, count, continuing, exception, cached, stop_owner, secondary in itertools.product(
            (False,True),(1,2,3,4),(False,True),(False,True),(False,True),(False,True),(0,29)):
        # A retained-owner continuing yield is only a gateway check; do not let
        # a synthetic empty current-yield body stand in for actual queue creation.
        if queued and continuing and not exception and not stop_owner:
            continue
        for dropped in range(count+1):
            native.prepare_step(count=count,continuing=continuing,exception=exception,cached=cached,
                stop_owner=stop_owner,drop_references=dropped,secondary_handle=secondary)
            if queued:
                native.drain_completion()
                value = None
            else:
                value = native.call("dispatch",native.payload,native.error_out)
            expected, remaining, return_value = expected_events(
                count,continuing,exception,stop_owner,dropped,secondary,queued)
            if native.events != expected or (not queued and value != return_value):
                raise ValueError(f"native completion mismatch: {queued,count,continuing,exception,cached,stop_owner,secondary,dropped}: {value}, {native.events} != {expected}")
            snapshot = native.snapshot()
            if snapshot["references"] != remaining or native.invocations != 1:
                raise ValueError("incorrect final references or managed invocation count")
            if snapshot["list_links_zero"] != (stop_owner or remaining == 0):
                raise ValueError("wrong owner-list cleanup")
            if remaining == 0 and (snapshot["enumerator_handle"] or not snapshot["cleanup_flag"]):
                raise ValueError("completed zero-reference payload retained enumerator")
            if not queued:
                flag = native.uc.mem_read(native.error_out,1)[0]
                if flag != (int(exception) if return_value else 0x7F):
                    raise ValueError("error flag written before survival guard or incorrectly encoded")
            results.append({"queued":queued,"initial_references":count,"continuing":continuing,
                "exception":exception,"cached_enumerator":cached,"stop_during_invoke":stop_owner,
                "secondary_handle_present":bool(secondary),"native_releases_during_invoke":dropped,
                "return_value":value,"remaining_references":remaining,"events":expected})
    # The callback's mismatch diagnostic returns one without entering MoveNext.
    for queued,count in itertools.product((False,True),(1,2)):
        native.prepare_step(count=count)
        native.queue_owner_mode = "mismatch"
        if queued:
            native.drain_completion()
            value = None
        else:
            value = native.call("callback_native",native.native_owner+0x100,native.payload)
        expected = ([["erase",0]] if queued else []) + [["report_owner_mismatch"]]
        if queued:
            expected.append(["release_reference",count])
            if count == 1:
                expected.extend([["free_handle",17],["free_payload","requested"]])
        if native.events != expected or native.invocations != 0 or (not queued and value != 1):
            raise ValueError("owner-mismatch callback/release behavior changed")
        results.append({"queued":queued,"initial_references":count,"owner_mismatch":True,
                        "return_value":value,"events":expected})
    # Two independent sibling coroutines: real release unlinks a final sibling
    # before StopAll traverses the owner list, or leaves a retained sibling for
    # StopAll to detach. Both removed successors must disappear from the drain.
    for sibling_count, secondary in itertools.product((1,2),(0,31)):
        native.prepare_step(stop_owner=True,stop_mode="all")
        native.add_completion_wait(1,payload_kind="sibling")
        sibling=native.payload+0x100
        head=native.native_owner+0x70
        native.uc.mem_write(head,struct.pack("<QQ",sibling,native.payload))
        native.uc.mem_write(native.payload,struct.pack("<QQ",head,sibling))
        native.uc.mem_write(sibling,struct.pack("<QQ",native.payload,head))
        native.uc.mem_write(sibling+0x58,struct.pack("<Q",native.native_owner))
        native.uc.mem_write(sibling+0x60,struct.pack("<i",sibling_count))
        native.uc.mem_write(sibling+0x10,struct.pack("<Q",19))
        native.uc.mem_write(sibling+0x18,struct.pack("<I",2))
        native.uc.mem_write(sibling+0x28,struct.pack("<Q",secondary))
        native.uc.mem_write(sibling+0x30,struct.pack("<I",2))
        # Primary wait must precede sibling despite being inserted second.
        sibling_node=native.nodes[1]
        record=bytearray(native.records[sibling_node])
        struct.pack_into("<d",record,0,0.5)
        native.records[sibling_node]=bytes(record)
        native.deadlines[1]=0.5
        native.uc.mem_write(sibling_node+0x20,bytes(record))
        native.drain_completion()
        expected=[["erase",0],["invoke",False,False],["stop_all_during_invoke"],
                  ["erase",1],["release_reference",sibling_count]]
        if sibling_count==1:
            expected.append(["free_handle",19])
            expected.append(["free_handle",secondary] if secondary else ["free_payload","sibling"])
        expected.extend([["release_reference",2],["release_reference",2],
                         ["release_reference",1],["free_handle",17],["free_payload","requested"]])
        if native.events != expected or native.nodes or native.invocations != 1:
            raise ValueError(f"StopAll sibling release/cursor mismatch: {native.events}")
        if native.qword(head)!=head or native.qword(head+8)!=head:
            raise ValueError("StopAll failed to empty owner list under actual release")
        sibling_state=native.snapshot(sibling)
        if sibling_state["references"]!=sibling_count-1 or not sibling_state["list_links_zero"]:
            raise ValueError("wrong sibling lifetime after StopAll")
        if sibling_count==2 and native.qword(sibling+0x58)!=0:
            raise ValueError("retained sibling owner was not detached")
        results.append({"queued":True,"stop_all_sibling":True,"sibling_initial_references":sibling_count,
            "sibling_secondary_handle":secondary,"events":expected,
            "post_call_sibling_emulated_storage":sibling_state})
    # Keep reviewable examples rather than serializing every repeated trace.
    corpus_digest = hashlib.sha256(json.dumps(results,sort_keys=True,separators=(",",":")).encode()).hexdigest()
    examples = [case for case in results if case.get("owner_mismatch") or case.get("stop_all_sibling") or (
        case["initial_references"] == 1 and case["cached_enumerator"]
        and not case["secondary_handle_present"])]
    return {"schema_version":1,"unityplayer_sha256":digest,
        "native_relationships_verified":len(checks)+len(references)+2,
        "bridge_relationships_rechecked":bridge["semantic_checks_passed"],
        "release_relationships_rechecked":release["native_relationships_verified"],
        "release_cases_rechecked":release["cases"],
        "emulator":"Unicorn 2.1.4 x86-64","cases":len(results),
        "verified_case_digest_sha256":corpus_digest,"representative_results":examples,
        "matrix":{"initial_references":[1,2,3,4],"cached_enumerator":[False,True],
            "continuing":[False,True],"exception":[False,True],"stop_during_invoke":[False,True],
            "secondary_handle":[0,29],"releases_during_invoke":"zero through initial reference count",
            "entries":["direct dispatcher","native one-shot queue"],
            "exclusion":"queued continuing nonexception with retained owner: real yield production is unaudited"},
        "synthetic_boundaries":["managed MoveNext invocation and chosen return/exception",
            "GC APIs", "owner-context virtual lookup", "method parameter-count lookup",
            "diagnostic sinks", "allocation/free sinks", "continuing current-yield gateway only"],
        "native_composition":["queue erasure and matching-owner callback", "invocation-frame initialization",
            "managed-step dispatcher", "reference release", "StopCoroutineManaged during invocation", "StopAll with actual sibling release and saved-cursor cancellation"],
        "storage_note":"free sinks record calls without unmapping; post-call snapshots are emulated storage only",
        "unresolved":["arbitrary linked/nested coroutine graphs", "actual managed bodies and exception construction",
            "current-yield production within continuing cases", "reentrant queue drains and complete lifetime graph"]}


if __name__ == "__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unityplayer",type=Path)
    parser.add_argument("--output",type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.unityplayer)
    args.output.write_text(json.dumps(report,indent=2)+"\n",encoding="utf-8")
    print(f"Verified {report['cases']} native completion cases and {report['native_relationships_verified']} relationships")
