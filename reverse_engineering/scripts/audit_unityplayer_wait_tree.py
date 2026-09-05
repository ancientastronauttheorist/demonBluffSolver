"""Differentially exercise pinned native wait-tree code in an isolated emulator.

Requires local UnityPlayer.dll, pefile and Unicorn 2.1.4. Never loads the DLL
into the host process. Only five audited tree routines and a synthetic allocator
may execute; synthetic records carry no game state. No native bytes are output.
"""
import argparse
import json
import math
import random
import struct
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


# Exclusive ends, established from native boundaries including leaf functions
# absent from the PE unwind table. Not guessed from preceding unwind entries.
ROUTINES = {
    "insert": (0x440F00, 0x441003),
    "link_and_balance": (0x366CB0, 0x366EF2),
    "erase": (0x3EB6D0, 0x3EBA51),
    "rotate_right": (0x3EBA60, 0x3EBAB9),
    "rotate_left": (0x3EBAC0, 0x3EBB15),
}
ALLOCATOR_RVA = 0x677920


def validate_tree(read, container, head, records, expected_order):
    """Check native links, red/black invariants, count, and unchanged payloads.

    records maps node addresses to their exact original 0x40-byte payloads.
    expected_order is an independent stable sort of surviving insertion IDs.
    This validator also runs in CI against authored synthetic nodes.
    """
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def qword(address):
        return struct.unpack("<Q", read(address, 8))[0]

    require(qword(container) == head, "container head changed")
    require(qword(container + 8) == len(records), "tree count mismatch")
    require(read(head + 0x18, 2) == b"\x01\x01", "invalid sentinel")
    root = qword(head + 8)
    seen = set()
    ordered_nodes = []

    def visit(node, parent):
        if node == head:
            return 1
        require(node in records, "unknown tree node")
        require(node not in seen, "tree cycle or duplicate link")
        seen.add(node)
        require(qword(node + 8) == parent, "parent mismatch")
        color, sentinel = read(node + 0x18, 2)
        require(sentinel == 0 and color in (0, 1), "invalid node flags")
        require(read(node + 0x20, 0x40) == records[node], "record payload changed")
        left, right = qword(node), qword(node + 0x10)
        if color == 0:
            require(read(left + 0x18, 1) == b"\x01" and
                    read(right + 0x18, 1) == b"\x01", "red parent and child")
        left_height = visit(left, node)
        ordered_nodes.append(node)
        right_height = visit(right, node)
        require(left_height == right_height, "black height mismatch")
        return left_height + color

    if root != head:
        require(read(root + 0x18, 1) == b"\x01", "root is not black")
    black_height = visit(root, head)
    require(seen == set(records), "unreachable tree nodes")
    require(qword(head) == (ordered_nodes[0] if ordered_nodes else head), "minimum mismatch")
    require(qword(head + 0x10) == (ordered_nodes[-1] if ordered_nodes else head), "maximum mismatch")
    actual = [struct.unpack_from("<Q", records[node], 0x10)[0] for node in ordered_nodes]
    require(actual == expected_order, "stable deadline order mismatch")
    return black_height


def make_record(deadline, identity):
    if not math.isfinite(deadline) or not 0 <= identity < 2**64:
        raise ValueError("finite deadline and u64 identity required")
    # The rest is opaque, deterministic payload: insertion/erasure must retain
    # every byte, including record fields unused by the ordering comparator.
    record = bytearray(((identity * 17 + i * 29) & 255) for i in range(0x40))
    struct.pack_into("<d", record, 0, deadline)
    struct.pack_into("<Q", record, 0x10, identity)
    return bytes(record)


class NativeTree:
    def __init__(self, data):
        import pefile
        import unicorn
        from unicorn import x86_const as x86

        if unicorn.__version__ != "2.1.4":
            raise ValueError("This audit requires Unicorn 2.1.4")
        verify_fingerprint(data, ENGINE_SHA256)
        pe = pefile.PE(data=data, fast_load=True)
        self.base = pe.OPTIONAL_HEADER.ImageBase
        self.uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        self.x86 = x86
        image_size = (pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095
        self.uc.mem_map(self.base, image_size)
        self.uc.mem_write(self.base, pe.get_memory_mapped_image())
        self.arena = 0x200000000
        self.stack = 0x300000000
        self.stop = 0x400000000
        self.uc.mem_map(self.arena, 0x200000)
        self.uc.mem_map(self.stack, 0x10000)
        self.uc.mem_map(self.stop, 0x1000)
        self.container = self.arena
        self.head = self.arena + 0x100
        self.input_record = self.arena + 0x200
        self.output = self.arena + 0x300
        self.instructions = set()
        self.operation_count = 0
        self.insertions = 0
        self.erasures = 0
        self.uc.hook_add(unicorn.UC_HOOK_CODE, self._on_code)
        self.reset()

    def _on_code(self, uc, address, size, _):
        rva = address - self.base
        if rva == ALLOCATOR_RVA:
            if uc.reg_read(self.x86.UC_X86_REG_RDX) != 0x60:
                raise ValueError("unexpected allocation size")
            allocated = self.next_node
            self.next_node += 0x80
            if self.next_node >= self.arena + 0x200000:
                raise ValueError("synthetic allocator capacity exceeded")
            uc.mem_write(allocated, bytes(0x60))
            uc.reg_write(self.x86.UC_X86_REG_RAX, allocated)
            rsp = uc.reg_read(self.x86.UC_X86_REG_RSP)
            target = struct.unpack("<Q", uc.mem_read(rsp, 8))[0]
            uc.reg_write(self.x86.UC_X86_REG_RSP, rsp + 8)
            uc.reg_write(self.x86.UC_X86_REG_RIP, target)
            return
        if not any(start <= rva and rva + size <= end for start, end in ROUTINES.values()):
            raise ValueError(f"execution left audited tree routines: {rva:#x}")
        self.instructions.add(rva)

    def reset(self):
        self.next_node = self.arena + 0x1000
        self.records = {}
        self.nodes = {}
        self.deadlines = {}
        self.uc.mem_write(self.container, struct.pack("<QQ", self.head, 0))
        self.uc.mem_write(self.head, struct.pack("<QQQ", self.head, self.head, self.head) +
                          b"\x01\x01" + bytes(0x60 - 26))

    def call(self, name, *args):
        x = self.x86
        rsp = self.stack + 0xF008  # Windows x64 entry alignment and shadow space.
        self.uc.mem_write(rsp, struct.pack("<Q", self.stop) + bytes(0x28))
        self.uc.reg_write(x.UC_X86_REG_RSP, rsp)
        self.uc.reg_write(x.UC_X86_REG_EFLAGS, 2)
        self.uc.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        for register, value in zip((x.UC_X86_REG_RCX, x.UC_X86_REG_RDX,
                                    x.UC_X86_REG_R8, x.UC_X86_REG_R9), args):
            self.uc.reg_write(register, value)
        self.uc.emu_start(self.base + ROUTINES[name][0], self.stop,
                          timeout=1_000_000, count=100_000)
        if self.uc.reg_read(x.UC_X86_REG_RIP) != self.stop:
            raise ValueError("native operation exceeded its execution bound")
        self.operation_count += 1
        return self.uc.reg_read(x.UC_X86_REG_RAX)

    def insert(self, deadline, identity):
        if identity in self.nodes:
            raise ValueError("duplicate active identity")
        record = make_record(deadline, identity)
        self.uc.mem_write(self.input_record, record)
        result = self.call("insert", self.container, self.output, self.input_record)
        if result != self.output:
            raise ValueError("unexpected insert return")
        node = struct.unpack("<Q", self.uc.mem_read(self.output, 8))[0]
        if node in self.records or not self.arena + 0x1000 <= node < self.next_node:
            raise ValueError("unexpected inserted node")
        self.nodes[identity] = node
        self.records[node] = record
        self.deadlines[identity] = deadline
        self.insertions += 1
        self.validate()

    def erase(self, identity):
        node = self.nodes[identity]
        result = self.call("erase", self.container, node)
        if result != node:
            raise ValueError("erase did not return the requested node")
        del self.nodes[identity]
        del self.records[node]
        del self.deadlines[identity]
        self.erasures += 1
        self.validate()

    def validate(self):
        # Dict insertion order is the independent occurrence order; equal keys
        # retain it under Python's stable sort, including -0.0 versus +0.0.
        expected = sorted(self.deadlines, key=self.deadlines.__getitem__)
        return validate_tree(self.uc.mem_read, self.container, self.head, self.records, expected)


def audit(path):
    data = Path(path).read_bytes()
    digest = verify_fingerprint(data, ENGINE_SHA256)
    tree = NativeTree(data)
    rng = random.Random(0xDB202209)
    cases = 0
    boundary = [-1.7976931348623157e308, -1.0, -5e-324, -0.0, 0.0,
                5e-324, 1.0, 1.7976931348623157e308]
    sequences = [list(range(64)), list(reversed(range(64))), [0.3] * 64,
                 [-0.0, 0.0] * 32, boundary * 8]
    for sequence in sequences:
        for removal in ("ascending", "descending", "shuffled"):
            tree.reset()
            for identity, deadline in enumerate(sequence):
                tree.insert(float(deadline), identity)
            identities = list(range(len(sequence)))
            if removal == "descending":
                identities.reverse()
            elif removal == "shuffled":
                rng.shuffle(identities)
            for identity in identities:
                tree.erase(identity)
            cases += 1
    for _ in range(128):
        tree.reset()
        for identity in range(96):
            if tree.nodes and rng.random() < 0.4:
                tree.erase(rng.choice(list(tree.nodes)))
            tree.insert(rng.choice(boundary + [-2.0, 0.3, 0.3, 2.0]), identity)
        remaining = list(tree.nodes)
        rng.shuffle(remaining)
        for identity in remaining:
            tree.erase(identity)
        cases += 1
    counts = {name: sum(start <= rva < end for rva in tree.instructions)
              for name, (start, end) in ROUTINES.items()}
    if not all(counts.values()):
        raise ValueError("a required native routine was never exercised")
    return {
        "schema_version": 1,
        "unityplayer_sha256": digest,
        "emulator": "Unicorn 2.1.4, x86-64, MXCSR 0x1f80",
        "synthetic_allocator": "0x677920; zeroed 0x60-byte nodes; no host DLL execution",
        "scope": "finite deadline insert/link/erase/rotations; no consumer or callback execution",
        "seed": "0xdb202209",
        "cases": cases,
        "operations": tree.operation_count,
        "insertions": tree.insertions,
        "erasures": tree.erasures,
        "verified_after_every_operation": ["stable finite deadline occurrence order",
            "unchanged 0x40-byte surviving payloads", "red-black invariants",
            "parent and sentinel links", "minimum, maximum and count"],
        "routines": {name: {"rva": hex(bounds[0]), "end_exclusive": hex(bounds[1]),
                             "distinct_executed_instructions": counts[name]}
                     for name, bounds in ROUTINES.items()},
        "unresolved": ["nonfinite keys and alternate floating-point modes",
                       "mutation during consumer traversal and callback dispatch",
                       "complete engine phase and lifetime provenance"],
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("unityplayer", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.unityplayer)
    args.output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(f"Verified {report['operations']} native tree operations across {report['cases']} cases")
