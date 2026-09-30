"""Reproducible boundary/ownership follow-up; phase-eight dataflow stays open.

Outputs classifications and addresses, never native bytes or method bodies.
"""
import argparse
import bisect
import json
import struct
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


def audit(path, inventory):
    import capstone
    import pefile

    raw = path.read_bytes()
    digest = verify_fingerprint(raw, ENGINE_SHA256)
    previous = json.loads(inventory.read_text(encoding='utf-8'))
    assert previous['engine_sha256'] == digest
    pe = pefile.PE(data=raw, fast_load=True)
    pe.parse_data_directories(directories=[3])
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    entries = pe.DIRECTORY_ENTRY_EXCEPTION
    starts = [e.struct.BeginAddress for e in entries]
    groups, owners = {}, {}
    for e in entries:
        root = e
        visited = set()
        while root.unwindinfo.Flags & 4:
            assert root.struct.BeginAddress not in visited
            visited.add(root.struct.BeginAddress)
            root = root.unwindinfo._chained_entry
        owner = root.struct.BeginAddress
        bounds = (e.struct.BeginAddress, e.struct.EndAddress)
        groups.setdefault(owner, []).append(bounds)
        owners[bounds[0]] = owner

    def containing(address):
        index = bisect.bisect_right(starts, address) - 1
        assert index >= 0
        e = entries[index].struct
        assert e.BeginAddress <= address < e.EndAddress
        return e.BeginAddress, e.EndAddress

    def decode(a, b):
        result = list(cs.disasm(pe.get_data(a, b - a), a))
        assert result and sum(i.size for i in result) == b - a
        return result

    def verified(address):
        a, b = containing(address)
        instructions = {i.address: i for i in decode(a, b)}
        assert address in instructions
        return instructions[address]

    def rip_target(ins):
        memories = [o.mem for o in ins.operands if o.type == capstone.CS_OP_MEM
                    and o.mem.base == capstone.x86.X86_REG_RIP]
        assert len(memories) == 1
        return ins.address + ins.size + memories[0].disp

    families = {}
    leaf_loads = []
    for item in previous['global_mov_loads']:
        address = int(item['load_rva'], 16)
        try:
            a, _ = containing(address)
        except AssertionError:
            leaf_loads.append(hex(address))
            continue
        assert rip_target(verified(address)) == 0x1c6e720
        families.setdefault(owners[a], []).append(hex(address))
    family_rows = []
    for root, loads in sorted(families.items()):
        instructions = [i for a, b in groups[root] for i in decode(a, b)]
        slots = [i.address for i in instructions if any(
            o.type == capstone.CS_OP_MEM and o.mem.disp == 0xb8 for o in i.operands)]
        family_rows.append({'root': hex(root), 'loads': loads,
                            'chunks': [[hex(a), hex(b)] for a, b in groups[root]],
                            'instruction_count': len(instructions),
                            'all_memory_b8_operand_sites': [hex(a) for a in slots]})

    # Two aligned static code pointers and one verified LEA/publication establish
    # leaf entries independently of the earlier raw FF byte matches.
    rows = []
    for entry, end, slot, expected_global, branch in [
            (0x304a50, 0x304a6a, 0x1932bf0, 0x1cd5710, 0x304a62),
            (0x6d0bc0, 0x6d0bd1, 0x196a7c8, 0x1c6e728, 0x6d0bca),
            (0xefaa30, 0xefaa47, None, 0x1cd1f28, 0xefaa3f)]:
        if slot is not None:
            data = pe.get_data(slot, 8)
            assert len(data) == 8
            assert struct.unpack('<Q', data)[0] == pe.OPTIONAL_HEADER.ImageBase + entry
        else:
            ins = verified(0xefcb7b)
            assert ins.mnemonic == 'lea' and rip_target(ins) == entry
            assert verified(0xefcb82).op_str == 'rax, qword ptr [rbx + 0x10]'
            assert verified(0xefcb86).op_str == 'qword ptr [rax + 0x550], rcx'
        body = {i.address: i for i in decode(entry, end)}
        assert rip_target(body[entry]) == expected_global
        assert body[branch].mnemonic == 'jmp'
        assert body[branch].op_str == 'qword ptr [rax + 0xb8]'
        assert not any(i.mnemonic == 'mov' and i.op_str.startswith('edx,') for i in body.values())
        rows.append({'entry': hex(entry), 'end_exclusive': hex(end),
                     'static_pointer_slot': hex(slot) if slot else None,
                     'address_publication_site': '0xefcb86' if slot is None else None,
                     'receiver_global': hex(expected_global), 'branch': hex(branch),
                     'mask': 'caller-supplied; no phase classification',
                     'wait_global_alias_relation': 'unproved'})

    # This newly included +b8 operand is an argument load, not a virtual target.
    assert rip_target(verified(0x7793a6)) == 0x1c6e708
    assert rip_target(verified(0x7795fb)) == 0x1cd6310
    assert verified(0x779605).op_str == 'rdx, qword ptr [rsi + 0xb8]'
    assert verified(0x77960f).op_str == 'rax'
    return {'schema_version': 1, 'engine_sha256': digest,
            'status': 'boundary/ownership handoff; no phase-eight dispatcher established',
            'global_load_count': 23, 'chained_unwind_families': family_rows,
            'prior_verified_leaf_loads': leaf_loads, 'new_pointer_backed_leaves': rows,
            'new_b8_argument_site': {'site': '0x779605', 'call': '0x77960f',
                                    'callee_pointer_global': '0x1cd6310',
                                    'argument_base_global': '0x1c6e708',
                                    'limit': 'Exact sites verified; path-sensitive RSI provenance not automated.'},
            'remaining_raw_candidate': '0x1467e7f',
            'remaining_work': ['Prove receiver alias/publication relations for the three forwarding globals.',
                               'Find verified entry or code-pointer provenance for raw 0x1467e7f.',
                               'Implement path-sensitive inter-chunk register/stack alias and computed-mask dataflow.',
                               'Trace callers of forwarding leaves and known wait dispatchers.',
                               'Ownership and operand inventory are not reachability or phase-admission proof.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unityplayer', type=Path)
    parser.add_argument('inventory', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.unityplayer, args.inventory)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {len(result['chained_unwind_families'])} unwind families and three pointer-backed leaves; phase eight unresolved")
