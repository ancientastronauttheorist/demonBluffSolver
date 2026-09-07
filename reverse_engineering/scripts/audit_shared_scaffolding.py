"""Verify exact shared constructor/getter bodies and per-definition field metadata."""
import argparse
import hashlib
import json
import random
import re
import struct
from pathlib import Path

from audit_folded_return import BUILD, audit as audit_return

BODIES = {0x357920: (152, "empty_constructor"),
          0x3CFFF0: (37, "empty_constructor_chain"),
          0x357700: (104, "int32_constructor"),
          0x353580: (210, "reference_getter")}


def definitions(dump, methods):
    headers = list(re.finditer(r"^.* // TypeDefIndex: (\d+)$", dump, re.M))
    blocks = {int(h[1]): dump[h.end():headers[i+1].start() if i+1 < len(headers) else len(dump)]
              for i, h in enumerate(headers)}
    result = []
    for rva, (count, behavior) in BODIES.items():
        body = f"ga:rva:{rva:08X}"
        selected = [m for m in methods if m["implementation"] == "concrete"
                    and m["native"] == [{"body": body, "binding": "direct"}]]
        if len(selected) != count:
            raise ValueError(f"definition count changed for {body}")
        for method in selected:
            signature = method["symbol_key"].split("::", 1)[1]
            entry = {"id": method["id"], "symbol_key": method["symbol_key"],
                     "body": body, "behavior": behavior}
            if behavior.startswith("empty_constructor"):
                if not re.fullmatch(r"(?:public|protected) void \.ctor\(\)", signature):
                    raise ValueError("unexpected empty constructor declaration")
            else:
                offset = 0x10 if behavior == "int32_constructor" else 0x18
                block = blocks[int(method["id"][3:].split(".")[0])]
                fields = block.split("// Properties")[0].split("// Methods")[0]
                declarations = re.findall(r"^\s*(.+); // 0x"+f"{offset:X}"+r"$", fields, re.M)
                if len(declarations) != 1 or " static " in declarations[0]:
                    raise ValueError("ambiguous or static field at native offset")
                field = re.sub(r"^(?:public|private|protected) (?:readonly )?", "", declarations[0])
                field_type, field_name = field.split(" ", 1)
                if behavior == "int32_constructor":
                    if field_type not in ("int", "Shugenja.EEvilDirection"):
                        raise ValueError("unverified four-byte field type")
                    if not re.fullmatch(r"public void \.ctor\("+re.escape(field_type)+r" [^,)]+\)", signature):
                        raise ValueError("constructor parameter differs from field")
                    if field_type == "Shugenja.EEvilDirection":
                        enum = re.search(r"public enum Shugenja\.EEvilDirection[^\n]*\n\{(.*?)\n\}", dump, re.S)
                        if not enum or "public int value__;" not in enum[1]:
                            raise ValueError("enum no longer backed by int32")
                else:
                    if field_type not in ("object", "string", "Collider", "Transform", "TailAnimator2.TailSegment"):
                        raise ValueError("unverified reference field type")
                    if not re.match(r"(?:public|private) "+re.escape(field_type)+r" .*get_[^()]+\(\)$", signature):
                        raise ValueError("getter return differs from field")
                entry["field"] = {"offset": hex(offset), "type": field_type, "name": field_name}
            result.append(entry)
    return result


def native_audit(data):
    import capstone
    import pefile
    import unicorn
    from unicorn import x86_const as x
    if unicorn.__version__ != "2.1.4":
        raise ValueError("requires Unicorn 2.1.4")
    pe = pefile.PE(data=data, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    expected = {
        0x33ED50: [("ret", "0")],
        0x357920: [("xor", "edx, edx"), ("jmp", hex(base+0x33ED50))],
        0x3CFFF0: [("xor", "edx, edx"), ("jmp", hex(base+0x357920))],
        0x353580: [("mov", "rax, qword ptr [rcx + 0x18]"), ("ret", "")],
        0x357700: [("mov", "qword ptr [rsp + 8], rbx"), ("push", "rdi"),
                   ("sub", "rsp, 0x20"), ("mov", "edi, edx"), ("mov", "rbx, rcx"),
                   ("xor", "edx, edx"), ("call", hex(base+0x33ED50)),
                   ("mov", "dword ptr [rbx + 0x10], edi"),
                   ("mov", "rbx, qword ptr [rsp + 0x30]"), ("add", "rsp, 0x20"),
                   ("pop", "rdi"), ("ret", "")],
    }
    allowed = {}
    for rva, sequence in expected.items():
        section = pe.get_section_by_rva(rva)
        if section is None or not section.Characteristics & 0x20000000:
            raise ValueError("entry outside executable section")
        instructions = list(cs.disasm(pe.get_data(rva, 0x40), base+rva))[:len(sequence)]
        if [(i.mnemonic, i.op_str) for i in instructions] != sequence:
            raise ValueError(f"native instruction relationship changed at {rva:x}")
        allowed.update({i.address: i.size for i in instructions})
    uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
    uc.mem_map(base, (pe.OPTIONAL_HEADER.SizeOfImage+4095)&~4095)
    uc.mem_write(base, pe.get_memory_mapped_image())
    stack, heap, stop = 0x200000000, 0x300000000, 0x400000000
    uc.mem_map(stack, 0x10000); uc.mem_map(heap, 0x1000); uc.mem_map(stop, 0x1000)
    writes, reads, trace = [], [], []
    def code(_, address, size, __):
        if allowed.get(address) != size:
            raise ValueError("unexpected native instruction")
        trace.append(address-base)
    def memory(_, access, address, size, value, __):
        (writes if access == unicorn.UC_MEM_WRITE else reads).append((address, size))
    uc.hook_add(unicorn.UC_HOOK_CODE, code)
    uc.hook_add(unicorn.UC_HOOK_MEM_WRITE | unicorn.UC_HOOK_MEM_READ, memory)
    names = ("RAX", "RCX", "RDX", "RBX", "RBP", "RSI", "RDI", "R8", "R9", "R10", "R11", "R12", "R13", "R14", "R15")
    gprs = [getattr(x, "UC_X86_REG_"+n) for n in names]
    xmms = [getattr(x, f"UC_X86_REG_XMM{i}") for i in range(16)]
    rng = random.Random(0x357700)
    digest = hashlib.sha256()
    cases = []
    for rva, (_, behavior) in BODIES.items():
        for case in range(96):
            object_bytes = bytes(rng.getrandbits(8) for _ in range(256))
            uc.mem_write(heap, object_bytes)
            stack_bytes = bytearray(rng.getrandbits(8) for _ in range(0x1000))
            offset = 0x808; rsp = stack+offset
            struct.pack_into("<Q", stack_bytes, offset, stop)
            uc.mem_write(stack, bytes(stack_bytes))
            values = [rng.getrandbits(64) for _ in gprs]
            values[1] = 0 if behavior.startswith("empty") and case%2 else heap
            if case < 8:
                values[2] = [0, 1, 0x7fffffff, 0x80000000, 0xffffffff, 0x100000000, 0xffffffffffffffff, 0x123456789abcdef0][case]
            vectors = [rng.getrandbits(128) for _ in xmms]
            for register, value in zip(gprs+xmms, values+vectors): uc.reg_write(register, value)
            flags = [2, 0x202, 0x246, 0x247, 0x602, 0xA46][case%6]
            uc.reg_write(x.UC_X86_REG_EFLAGS, flags); uc.reg_write(x.UC_X86_REG_RSP, rsp)
            writes.clear(); reads.clear(); trace.clear()
            uc.emu_start(base+rva, stop, timeout=1_000_000, count=32)
            if uc.reg_read(x.UC_X86_REG_RIP) != stop or uc.reg_read(x.UC_X86_REG_RSP) != rsp+8:
                raise ValueError("return stack/control mismatch")
            expected_values = values.copy(); expected_object = bytearray(object_bytes)
            expected_stack = stack_bytes.copy(); expected_writes = []
            if behavior.startswith("empty"):
                expected_values[2] = 0
            elif behavior == "reference_getter":
                expected_values[0] = struct.unpack_from("<Q", object_bytes, 0x18)[0]
                if reads != [(heap+0x18, 8), (rsp, 8)]:
                    raise ValueError("getter did more than one field read and return")
            else:
                expected_values[2] = 0
                struct.pack_into("<I", expected_object, 0x10, values[2]&0xffffffff)
                for delta, value in ((8, values[3]), (-8, values[6]), (-0x30, base+0x357716)):
                    struct.pack_into("<Q", expected_stack, offset+delta, value)
                expected_writes = [(rsp+8, 8), (rsp-8, 8), (rsp-0x30, 8), (heap+0x10, 4)]
            if writes != expected_writes:
                raise ValueError("unexpected memory writes")
            if bytes(uc.mem_read(heap, 256)) != bytes(expected_object) or bytes(uc.mem_read(stack, 0x1000)) != bytes(expected_stack):
                raise ValueError("object or stack effect mismatch")
            if [uc.reg_read(r) for r in gprs+xmms] != expected_values+vectors:
                raise ValueError("data/vector register effect mismatch")
            if behavior == "reference_getter" and uc.reg_read(x.UC_X86_REG_EFLAGS) != flags:
                raise ValueError("getter changed flags")
            digest.update(object_bytes); digest.update(stack_bytes)
            digest.update(b"".join(v.to_bytes(8, "little") for v in values))
        cases.append({"entry_rva": hex(rva), "behavior": behavior, "cases": 96})
    return {"cases": 384, "groups": cases, "input_digest_sha256": digest.hexdigest(),
            "instruction_relationships": sum(map(len, expected.values())),
            "all_nonvolatile_and_vector_registers_preserved": True,
            "exact_object_and_stack_writes_verified": True,
            "limitations": ["mapped receivers required for field access bodies", "arithmetic flags are unspecified for constructor bodies",
                            "native body only: allocation, GC and caller initialization are outside scope"]}


def audit(game_root, dumper_root):
    inherited = audit_return(game_root, dumper_root)
    repo = Path(__file__).parents[1]
    methods = [json.loads(s) for s in (repo/f"coverage/{BUILD}/methods.v1.jsonl").read_text(encoding="utf-8").splitlines()]
    selected = definitions((Path(dumper_root)/"dump.cs").read_text(encoding="utf-8"), methods)
    return {"schema_version": 1, "build_id": BUILD,
            "game_assembly_sha256": inherited["game_assembly_sha256"],
            "global_metadata_sha256": inherited["global_metadata_sha256"],
            "denominator_methods": inherited["denominator_methods"],
            "definition_count": len(selected), "definitions": selected,
            "native_emulation": native_audit((Path(game_root)/"GameAssembly.dll").read_bytes()),
            "prerequisite_return_audit": {"definitions": inherited["definition_count"], "native_cases": inherited["native_emulation"]["cases"]},
            "scope": "Exact direct bodies and per-definition fields; no complete iterator, class, allocation or runtime behavior claim",
            "excluded": ["generic-shared constructor binding", "MonoBehaviour constructor forwarding into Unity engine code"]}


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("game_root", type=Path)
    parser.add_argument("--dumper-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2)+"\n", encoding="utf-8")
    print(f"Verified {report['definition_count']} direct definitions and {report['native_emulation']['cases']} native cases")
