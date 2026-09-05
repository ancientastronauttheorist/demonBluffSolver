"""Exercise one-shot native wait draining with controlled callback mutations.

Native queue traversal, gates and cursor-aware erasure execute in Unicorn.
Owner lookup, profiling, allocation and callback bodies are explicit synthetic
boundaries. This does not execute the game or reconstruct those boundaries.
"""
import argparse
import json
import math
import struct
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint
from audit_unityplayer_wait_tree import NativeTree, ROUTINES


def wait_record(deadline, frame, identity, generation, mask, callback, release):
    if not math.isfinite(deadline) or not 0 <= identity < 2**32 - 2:
        raise ValueError("finite deadline and bounded synthetic identity required")
    if not -(2**63) <= frame < 2**63 or not 0 <= generation < 2**32 or not 0 <= mask < 2**32:
        raise ValueError("invalid native counter or mask width")
    record = bytearray(0x40)
    struct.pack_into("<dq", record, 0, deadline, frame)
    # A u32 identity in the unused repeat-duration bytes; the repeat flag at
    # +0x14 remains zero. The tree oracle can also read this identity as a u64.
    struct.pack_into("<Q", record, 0x10, identity)
    struct.pack_into("<QQQIII", record, 0x18, identity, callback, release,
                     identity + 1, mask, generation)
    return bytes(record)


class NativeConsumer(NativeTree):
    routines = {**ROUTINES, "drain": (0x43BD90, 0x43C016),
                "remove": (0x43BC60, 0x43BD90), "cancel": (0x43BB00, 0x43BC56)}

    def __init__(self, data):
        super().__init__(data)
        self.owner = self.arena + 0x800
        self.container = self.owner + 0x30
        self.allocator = self.arena + 0x100000
        self.profiler = self.arena + 0x101000
        self.engine = self.arena + 0x102000
        self.owner_table = self.arena + 0x103000
        self.owner_entry = self.arena + 0x104000
        self.callback = self.stop + 0x100
        self.release = self.stop + 0x200
        self.reset_case()

    def qword(self, address):
        return struct.unpack("<Q", self.uc.mem_read(address, 8))[0]

    def reset_case(self, generation=0):
        super().reset()
        self.events = []
        self.actions = {}
        self.returns = {}
        self.owners = {}
        self.pending_callback = None
        self.native_visits = []
        self.next_identity = 0
        self.uc.mem_write(self.owner + 0x40, struct.pack("<QI", self.head, generation))
        self.uc.mem_write(self.allocator, bytes(0x100))
        self.uc.mem_write(self.profiler, bytes(0x1000))
        self.uc.mem_write(self.engine, bytes(0x1000))
        self.uc.mem_write(self.owner_table, struct.pack("<QI", self.owner_entry, 0))
        for rva, pointer in [(0x1BD5640, self.allocator), (0x1C6E718, self.engine),
                             (0x1CD5328, self.owner_table)]:
            self.uc.mem_write(self.base + rva, struct.pack("<Q", pointer))

    def _return(self, value=0):
        x = self.x86
        rsp = self.uc.reg_read(x.UC_X86_REG_RSP)
        self.uc.reg_write(x.UC_X86_REG_RAX, value)
        self.uc.reg_write(x.UC_X86_REG_RIP, self.qword(rsp))
        self.uc.reg_write(x.UC_X86_REG_RSP, rsp + 8)

    def _on_code(self, uc, address, size, _):
        rva = address - self.base
        x = self.x86
        if rva == 0x6E78E0:
            self._return(self.profiler)
            return
        if rva == 0x355F00:
            self._return()
            return
        if rva == 0x151EF0:
            owner_id = struct.unpack("<I", uc.mem_read(uc.reg_read(x.UC_X86_REG_RDX), 4))[0]
            state = self.owners.get(owner_id - 1, "valid")
            if state == "missing":
                self._return(self.owner_entry + 24)
            else:
                pointer = 0 if state == "null" else self.arena + 0x105000 + owner_id * 8
                uc.mem_write(self.owner_entry, struct.pack("<QQQ", owner_id, 0, pointer))
                self._return(self.owner_entry)
            return
        if address == self.callback:
            identity = uc.reg_read(x.UC_X86_REG_RDX)
            expected_owner = self.arena + 0x105000 + (identity + 1) * 8
            if uc.reg_read(x.UC_X86_REG_RCX) != expected_owner:
                raise ValueError("callback owner/payload mismatch")
            self.pending_callback = identity
            self.events.append(["callback", identity])
            uc.emu_stop()
            return
        if address == self.release:
            self.events.append(["release", uc.reg_read(x.UC_X86_REG_RCX)])
            self._return()
            return
        if rva in (0x43BC60, 0x43BB00):
            node = uc.reg_read(x.UC_X86_REG_R8)
            identity = struct.unpack_from("<Q", self.records[node], 0x10)[0]
            if uc.reg_read(x.UC_X86_REG_RCX) != self.owner:
                raise ValueError("wrong queue owner during erasure")
            self.events.append(["erase", identity])
            del self.nodes[identity]
            del self.records[node]
            del self.deadlines[identity]
        if rva == 0x43BE23:
            node = uc.reg_read(x.UC_X86_REG_RBX)
            identity = struct.unpack_from("<Q", self.records[node], 0x10)[0]
            self.native_visits.append(identity)
        super()._on_code(uc, address, size, _)

    def add_wait(self, identity, deadline=0.0, frame=0, generation=None, mask=10, release=True):
        if identity in self.nodes:
            raise ValueError("duplicate active identity")
        if generation is None:
            generation = struct.unpack("<I", self.uc.mem_read(self.owner + 0x48, 4))[0]
        record = wait_record(deadline, frame, identity, generation, mask,
                             self.callback, self.release if release else 0)
        self.uc.mem_write(self.input_record, record)
        result = self.call("insert", self.container, self.output, self.input_record)
        if result != self.output:
            raise ValueError("unexpected insert return")
        node = self.qword(self.output)
        self.nodes[identity] = node
        self.records[node] = record
        self.deadlines[identity] = deadline
        self.next_identity = max(self.next_identity, identity + 1)
        self.validate()

    def projection_state(self):
        entries = []
        for identity in sorted(self.deadlines, key=self.deadlines.__getitem__):
            record = self.records[self.nodes[identity]]
            entries.append({"logical_id": identity, "timing": {
                "deadline": struct.unpack_from("<d", record, 0)[0],
                "frame_threshold": struct.unpack_from("<q", record, 8)[0],
                "phase_mask": struct.unpack_from("<I", record, 0x34)[0],
                "insertion_generation": struct.unpack_from("<I", record, 0x38)[0]},
                "release_present": struct.unpack_from("<Q", record, 0x28)[0] != 0})
        return {"rule_version": "unity_wait_queue_native_v1", "entries": entries,
                "generation": struct.unpack("<I", self.uc.mem_read(self.owner + 0x48, 4))[0],
                "next_id": self.next_identity}

    def projection_context(self, time=1.0, frame=1, phase=2):
        state = self.projection_state()
        responses = {}
        for entry in state["entries"]:
            identity = entry["logical_id"]
            if self.qword(self.base + 0x1CD5328) == 0 or self.owners.get(identity) in ("missing", "null"):
                responses[str(identity)] = {"owner": "unavailable"}
                continue
            mutations = []
            for action in self.actions.get(identity, []):
                if action[0] == "cancel":
                    mutations.append({"operation": "cancel", "logical_id": action[1]})
                elif action[0] == "insert":
                    args = action[1]
                    if args.get("mask", 10) != 10 or "generation" in args:
                        raise ValueError("fixture insertion is not a current-generation WaitForSeconds")
                    threshold = args.get("frame", 0)
                    producer_frame = (threshold - 1 + 2**63) % 2**64 - 2**63
                    mutations.append({"operation": "insert", "duration": 0.0,
                        "producer_time": args.get("deadline", 0.0),
                        "producer_frame_counter": producer_frame,
                        "release_present": args.get("release", True)})
                elif action[0] != "clock":
                    raise ValueError("unknown projected action")
                # Callback engine-clock writes intentionally have no queue effect:
                # the native audit checks the retained entry-time samples.
            responses[str(identity)] = {"owner": "resolved",
                "callback_result": self.returns.get(identity, 1), "mutations": mutations}
        return {"rule_version": "unity_wait_queue_native_v1", "initial": state,
                "dispatch": {"rule_version": "unity_wait_eligibility_native_v1",
                    "sampled_time": time, "sampled_frame_counter": frame, "phase_mask": phase,
                    "generation_before": state["generation"]}, "responses": responses}

    def cancel(self, identity):
        node = self.nodes[identity]
        self.call("cancel", self.owner, node + 0x20, node)
        self.validate()

    def set_clock(self, time, frame):
        self.uc.mem_write(self.engine + 0x90, struct.pack("<d", time))
        self.uc.mem_write(self.engine + 0xC8, struct.pack("<q", frame))

    def drain(self, time=1.0, frame=1, phase=2):
        if not math.isfinite(time) or not -(2**63) <= frame < 2**63 or not 0 <= phase < 2**32:
            raise ValueError("invalid synthetic drain snapshot")
        self.set_clock(time, frame)
        self.events = []
        self.native_visits = []
        x = self.x86
        rsp = self.stack + 0xF008
        self.uc.mem_write(rsp, struct.pack("<Q", self.stop) + bytes(0x28))
        self.uc.reg_write(x.UC_X86_REG_RSP, rsp)
        self.uc.reg_write(x.UC_X86_REG_RCX, self.owner)
        self.uc.reg_write(x.UC_X86_REG_RDX, phase)
        self.uc.reg_write(x.UC_X86_REG_EFLAGS, 2)
        self.uc.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        pc = self.base + self.routines["drain"][0]
        for _ in range(256):
            self.pending_callback = None
            self.uc.emu_start(pc, self.stop, timeout=1_000_000, count=100_000)
            if self.pending_callback is None:
                if self.uc.reg_read(x.UC_X86_REG_RIP) != self.stop:
                    raise ValueError("native drain exceeded its execution bound")
                break
            identity = self.pending_callback
            self.validate()  # Native one-shot removal precedes the callback.
            context = self.uc.context_save()
            self.call_stack_offset = 0x7008
            for action in self.actions.get(identity, []):
                if action[0] == "insert":
                    self.add_wait(**action[1])
                elif action[0] == "cancel":
                    self.cancel(action[1])
                elif action[0] == "clock":
                    self.set_clock(*action[1:])
                else:
                    raise ValueError("unknown synthetic callback action")
            self.call_stack_offset = 0xF008
            self.uc.context_restore(context)
            self._return(self.returns.get(identity, 1))
            pc = self.uc.reg_read(x.UC_X86_REG_RIP)
        else:
            raise ValueError("synthetic callback count exceeded")
        self.validate()
        if struct.unpack("<I", self.uc.mem_read(self.profiler + 0x180, 4))[0] != 0:
            raise ValueError("unbalanced profiler scope")
        return {"events": list(self.events), "visits": list(self.native_visits),
                "remaining": sorted(self.deadlines, key=self.deadlines.__getitem__),
                "generation": struct.unpack("<I", self.uc.mem_read(self.owner + 0x48, 4))[0]}


def audit(path, projection_output=None):
    data = Path(path).read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    tree = NativeConsumer(data)
    results = []
    projections = []

    def check(name, expected_callbacks, expected_remaining, expected_visits=None, expected_releases=None, **snapshot):
        context = tree.projection_context(**snapshot)
        result = tree.drain(**snapshot)
        callbacks = [identity for kind, identity in result["events"] if kind == "callback"]
        releases = [identity for kind, identity in result["events"] if kind == "release"]
        if callbacks != expected_callbacks or result["remaining"] != expected_remaining:
            raise ValueError(f"native callback/order result mismatch: {name}: {result}")
        if expected_visits is not None and result["visits"] != expected_visits:
            raise ValueError(f"native visit result mismatch: {name}: {result}")
        if expected_releases is not None and releases != expected_releases:
            raise ValueError(f"native release result mismatch: {name}: {result}")
        results.append({"name": name, **result})
        projections.append({"name": name, "context": context,
            "expected_state": tree.projection_state(), "expected_events": result["events"],
            "expected_visits": result["visits"]})

    tree.reset_case()
    for identity in (7, 3, 9):
        tree.add_wait(identity)
    check("equal deadlines follow insertion order", [7, 3, 9], [], [7, 3, 9], [7, 3, 9])

    tree.reset_case()
    tree.add_wait(1, deadline=2, mask=1)
    tree.add_wait(2, deadline=3)
    check("future deadline stops before phase filtering", [], [1, 2], [1], [])

    tree.reset_case()
    tree.add_wait(1, mask=1)
    tree.add_wait(2, generation=1)
    tree.add_wait(3, frame=2)
    tree.add_wait(4)
    check("phase generation and signed frame skips continue traversal", [4], [1, 2, 3], [1, 2, 3, 4], [4])
    check("next generation still respects frame gate", [2], [1, 3], [1, 2, 3], [2])
    check("later frame admits retained wait", [3], [1], [1, 3], [3], frame=2)

    tree.reset_case(0xFFFFFFFF)
    tree.add_wait(1, generation=0)
    tree.add_wait(2, generation=0xFFFFFFFF, frame=-(2**63))
    tree.add_wait(3, generation=0xFFFFFFFF, frame=2**63 - 1)
    check("generation rollover and signed full-width frame gate", [2], [1, 3], [1, 2, 3], [2], frame=-(2**63))

    tree.reset_case()
    tree.add_wait(1, frame=2**32 + 1)
    tree.add_wait(2, frame=1)
    check("frame gate does not truncate to public u32 frameCount", [2], [1], [1, 2], [2])

    tree.reset_case()
    for identity in range(1, 4):
        tree.add_wait(identity)
    tree.owners = {1: "missing", 2: "null"}
    check("missing and null owners erase and release without callback", [3], [], [1, 2, 3], [1, 2, 3])

    tree.reset_case()
    tree.add_wait(1)
    tree.add_wait(2, release=False)
    tree.uc.mem_write(tree.base + 0x1CD5328, bytes(8))
    check("absent owner table removes waits and honors null release slot", [], [], [1, 2], [1])

    tree.reset_case()
    for identity in range(1, 5):
        tree.add_wait(identity, release=identity != 4)
    tree.returns = {1: 0, 2: 2}
    check("release requires callback result exactly one and nonnull slot", [1, 2, 3, 4], [],
          [1, 2, 3, 4], [3])

    tree.reset_case()
    for identity in range(1, 5):
        tree.add_wait(identity)
    tree.actions = {1: [("cancel", 2), ("cancel", 3)]}
    check("canceling saved successor advances cursor twice", [1, 4], [], [1, 4], [2, 3, 1, 4])

    tree.reset_case()
    for identity in range(1, 4):
        tree.add_wait(identity)
    tree.actions = {1: [("cancel", 2), ("cancel", 3), ("insert", {"identity": 4, "frame": 2})]}
    check("canceling through end then inserting retains end cursor", [1], [4], [1], [2, 3, 1])

    tree.reset_case()
    for identity in range(1, 5):
        tree.add_wait(identity)
    tree.actions = {1: [("cancel", 3)]}
    check("canceling later node preserves saved successor", [1, 2, 4], [], [1, 2, 4], [3, 1, 2, 4])

    tree.reset_case()
    tree.add_wait(1, deadline=0)
    tree.add_wait(2, deadline=2)
    tree.actions = {1: [("insert", {"identity": 3, "deadline": 1, "frame": 2})]}
    check("new insertion before saved successor is not visited", [1, 2], [3], [1, 2], [1, 2], time=3)
    check("next drain visits insertion but same frame remains too early", [], [3], [3], [], time=3)
    check("next frame admits the earlier insertion", [3], [], [3], [3], time=3, frame=2)

    tree.reset_case()
    tree.add_wait(1, deadline=0)
    tree.add_wait(2, deadline=1)
    tree.actions = {1: [("insert", {"identity": 3, "deadline": 2, "frame": 2})]}
    check("new insertion after saved successor is visited and generation-skipped", [1, 2], [3], [1, 2, 3],
          [1, 2], time=3)
    check("later drain admits the visited insertion", [3], [], [3], [3], time=3, frame=2)

    tree.reset_case()
    tree.add_wait(1)
    tree.actions = {1: [("insert", {"identity": 2, "frame": 2})]}
    check("insertion after last record preserves saved end sentinel", [1], [2], [1], [1])
    check("later frame admits insertion made at end", [2], [], [2], [2], frame=2)

    tree.reset_case()
    tree.add_wait(1)
    tree.add_wait(2, deadline=2)
    tree.actions = {1: [("clock", 100.0, 100)]}
    check("clock changes inside callback do not replace sampled deadline clock", [1], [2], [1, 2], [1])
    check("new drain samples changed clock", [2], [], [2], [2], time=100, frame=100)

    tree.reset_case()
    tree.add_wait(1)
    tree.add_wait(2, frame=2)
    tree.actions = {1: [("clock", 100.0, 100)]}
    check("clock changes inside callback do not replace sampled frame counter", [1], [2], [1, 2], [1])

    if projection_output is not None:
        projection_output = Path(projection_output)
        projection_output.parent.mkdir(parents=True, exist_ok=True)
        projection_output.write_text(json.dumps({"schema_version": 1,
            "native_input_sha256": digest,
            "scope": "authored synthetic queue inputs and native-emulated results; no native bytes",
            "clock_write_projection": "callback clock writes omitted; entry-time samples are retained",
            "cases": projections}, indent=2) + "\n", encoding="utf-8")
    return {"schema_version": 1, "unityplayer_sha256": digest,
            "emulator": "Unicorn 2.1.4 x86-64, MXCSR 0x1f80",
            "native_scope": "one-shot consumer, timing gates, traversal, tree insertion and cursor-aware erasure",
            "synthetic_boundaries": ["node allocator", "owner-hash lookup and owner objects", "profiling",
                                     "callback actions and return values", "release callback bodies"],
            "cases": len(results), "results": results,
            "routines": {name: {"rva": hex(a), "end_exclusive": hex(b),
                                "distinct_executed_instructions": sum(a <= r < b for r in tree.instructions)}
                         for name, (a, b) in tree.routines.items()},
            "unresolved": ["repeating waits", "reentrant drain/release callback mutation",
                           "native owner-table and coroutine lifetime behavior",
                           "actual game clock/phase observations and continuation admission"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unityplayer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--projection-fixture", type=Path)
    args = parser.parse_args()
    report = audit(args.unityplayer, args.projection_fixture)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['cases']} native one-shot consumer cases")
