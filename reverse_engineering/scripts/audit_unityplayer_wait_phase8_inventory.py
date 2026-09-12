"""Bounded phase-eight search in pinned native wait-manager references.

This is an inventory, not proof that phase eight has no indirect dispatcher.
"""
import argparse
import bisect
import json
import re
import struct
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


GLOBAL = 0x1c6e720
CONSUMER = 0x43bd90
LEAVES = [(0x5b7ce0, 0x5b7cf6), (0x5b7d20, 0x5b7d40), (0x5b7d50, 0x5b7d66)]
KNOWN = [(0x59f683, 0x59f68a, 0x59f68f, 0x59f692, 16),
         (0x5b7afd, 0x5b7b04, 0x5b7b09, 0x5b7b0c, 32),
         (0x5b7ce0, 0x5b7ce7, 0x5b7cec, 0x5b7cef, 4),
         (0x5b7d29, 0x5b7d30, 0x5b7d35, 0x5b7d38, 1),
         (0x5b7d50, 0x5b7d57, 0x5b7d5c, 0x5b7d5f, 2)]


def audit(path):
    import capstone
    import pefile
    raw = path.read_bytes()
    digest = verify_fingerprint(raw, ENGINE_SHA256)
    pe = pefile.PE(data=raw, fast_load=True)
    pe.parse_data_directories(directories=[3])
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    chunks = [(entry.struct.BeginAddress, entry.struct.EndAddress) for entry in pe.DIRECTORY_ENTRY_EXCEPTION]
    starts = [a for a, _ in chunks]
    cache = {}
    incomplete_chunks = []
    def block(address):
        index = bisect.bisect_right(starts, address) - 1
        bounds = chunks[index] if index >= 0 else None
        source = 'unwind_chunk'
        if bounds is None or not bounds[0] <= address < bounds[1]:
            bounds = next(((a, b) for a, b in LEAVES if a <= address < b), None)
            source = 'previously_bound_PlayerLoop_leaf'
        if bounds is None:
            return None
        if bounds not in cache:
            a, b = bounds
            items = list(cs.disasm(pe.get_data(a, b - a), a))
            assert items and items[0].address == a
            consumed = sum(i.size for i in items)
            assert items[-1].address + items[-1].size == a + consumed
            if consumed != b - a:
                assert source == 'unwind_chunk'
                incomplete_chunks.append({'range': [hex(a), hex(b)], 'decoded_end_exclusive': hex(a + consumed)})
            cache[bounds] = items
        return bounds, source, cache[bounds]
    def verified(address, suffix_end=None):
        info = block(address)
        if info is None:
            return None
        bounds, source, items = info
        found = next((i for i in items if i.address == address), None) if suffix_end is None else next(
            (i for i in items if address - 2 <= i.address <= address and i.address + i.size == suffix_end), None)
        if found is None:
            return None
        return bounds, source, items, found
    global_sites = []
    local_slot_operands = []
    direct_raw = []
    slot_candidates = []
    executable_bytes = 0
    # x64 near indirect memory branches, displacement +0xB8; include SIB forms.
    slot_pattern = re.compile(rb'\xff(?:[\x90-\x93\x95-\x97\xa0-\xa3\xa5-\xa7]|[\x94\xa4].)\xb8\x00\x00\x00')
    for section in pe.sections:
        if not section.Characteristics & 0x20000000:
            continue
        data = section.get_data()
        executable_bytes += len(data)
        origin = section.VirtualAddress
        for prefix in [bytes((rex, 0x8b, modrm)) for rex in (0x48, 0x4c) for modrm in range(5, 0x40, 8)]:
            cursor = 0
            while True:
                cursor = data.find(prefix, cursor)
                if cursor < 0:
                    break
                rva = origin + cursor
                if cursor + 7 <= len(data) and rva + 7 + struct.unpack_from('<i', data, cursor + 3)[0] == GLOBAL:
                    item = verified(rva)
                    assert item is not None, f'Unverified global load {rva:#x}'
                    bounds, source, items, ins = item
                    assert ins.mnemonic == 'mov' and ins.operands[1].mem.base == capstone.x86.X86_REG_RIP
                    assert ins.address + ins.size + ins.operands[1].mem.disp == GLOBAL
                    later = [i for i in items if i.address > rva]
                    for later_ins in later:
                        if any(op.type == capstone.CS_OP_MEM and op.mem.disp == 0xb8 for op in later_ins.operands):
                            local_slot_operands.append({'global_load_rva': hex(rva), 'rva': hex(later_ins.address),
                                                        'mnemonic': later_ins.mnemonic, 'operand': later_ins.op_str})
                    first_call = next((i for i in later if i.mnemonic in ('call', 'jmp')), None)
                    global_sites.append({'load_rva': hex(rva), 'destination_register': ins.reg_name(ins.operands[0].reg),
                                         'verified_range': [hex(v) for v in bounds], 'boundary_source': source,
                                         'first_following_call_or_jump_in_range': None if first_call is None else
                                         {'rva': hex(first_call.address), 'mnemonic': first_call.mnemonic, 'operand': first_call.op_str}})
                cursor += 1
        for opcode in (b'\xe8', b'\xe9'):
            cursor = 0
            while True:
                cursor = data.find(opcode, cursor)
                if cursor < 0:
                    break
                if cursor + 5 <= len(data) and origin + cursor + 5 + struct.unpack_from('<i', data, cursor + 1)[0] == CONSUMER:
                    direct_raw.append(origin + cursor)
                cursor += 1
        for match in slot_pattern.finditer(data):
            slot_candidates.append((origin + match.start(), origin + match.end()))
    assert len(global_sites) == 23 and not direct_raw
    assert {int(item['rva'], 16) for item in local_slot_operands} == {item[3] for item in KNOWN}
    assert len(local_slot_operands) == 5
    verified_slots = []
    unverified_slots = []
    rejected_slots = []
    edx8_sites = []
    for rva, end in slot_candidates:
        item = verified(rva, end)
        if item is None:
            (unverified_slots if block(rva) is None else rejected_slots).append(hex(rva))
            continue
        bounds, source, items, ins = item
        assert ins.mnemonic in ('call', 'jmp') and len(ins.operands) == 1
        assert ins.operands[0].type == capstone.CS_OP_MEM and ins.operands[0].mem.disp == 0xb8
        preceding = [i for i in items if i.address < ins.address]
        matches = [i.address for i in preceding if i.mnemonic == 'mov' and i.op_str == 'edx, 8']
        if matches:
            edx8_sites.append({'dispatch_candidate': hex(ins.address), 'preceding_edx8': [hex(v) for v in matches]})
        verified_slots.append({'rva': hex(ins.address), 'boundary_source': source})
    known = []
    for load, mask_site, vtable_site, dispatch, mask in KNOWN:
        info = verified(load)
        assert info is not None
        instructions = {i.address: i for i in info[2]}
        assert all(rva in instructions for rva in (load, mask_site, vtable_site, dispatch))
        assert instructions[load].op_str.startswith('rcx, qword ptr [rip + ')
        assert (instructions[mask_site].mnemonic, instructions[mask_site].op_str) == ('mov', f'edx, {mask:#x}' if mask >= 10 else f'edx, {mask}')
        assert (instructions[vtable_site].mnemonic, instructions[vtable_site].op_str) == ('mov', 'rax, qword ptr [rcx]')
        assert instructions[dispatch].mnemonic in ('call', 'jmp') and instructions[dispatch].op_str == 'qword ptr [rax + 0xb8]'
        known.append({'global_load_rva': hex(load), 'mask_site_rva': hex(mask_site), 'dispatch_rva': hex(dispatch), 'mask': mask})
    assert len(verified_slots) == 158 and not edx8_sites
    assert unverified_slots == ['0x304a63', '0x6d0bcb', '0xefaa40', '0x1467e7f']
    return {'schema_version': 1, 'engine_sha256': digest, 'executable_file_bytes_searched': executable_bytes,
            'wait_manager_global_rva': hex(GLOBAL), 'concrete_consumer_rva': hex(CONSUMER),
            'global_mov_load_count': len(global_sites), 'global_mov_loads': sorted(global_sites, key=lambda r: int(r['load_rva'], 16)),
            'memory_slot_b8_operands_after_global_load_within_verified_range': local_slot_operands,
            'local_register_loaded_slot_target_count': sum(item['mnemonic'] not in ('call', 'jmp') for item in local_slot_operands),
            'raw_direct_relative_consumer_branch_count': len(direct_raw),
            'raw_memory_slot_b8_candidate_count': len(slot_candidates), 'verified_memory_slot_b8_branch_count': len(verified_slots),
            'verified_unwind_slot_branch_count': sum(v['boundary_source'] == 'unwind_chunk' for v in verified_slots),
            'verified_leaf_slot_branch_count': sum(v['boundary_source'] != 'unwind_chunk' for v in verified_slots),
            'unverified_no_boundary_candidates': unverified_slots, 'rejected_nonboundary_candidates': rejected_slots,
            'incomplete_unrelated_unwind_decodes': incomplete_chunks,
            'same_range_preceding_mov_edx8_candidates': edx8_sites, 'known_verified_wait_dispatches': known,
            'verified_slot_branches': verified_slots,
            'conclusion': 'No phase-eight dispatcher established by this bounded inventory.',
            'exclusions': ['Only RIP-relative REX.W MOV loads of the known global are inventoried; alternate addressing and global aliases are not exhaustive.',
                           'Only E8/E9 relative direct branches to the concrete consumer and immediate memory-indirect +0xB8 branches are searched.',
                           'Register-loaded virtual targets, indirect aliases, computed masks, caller-supplied masks, inter-chunk control flow and whole-program dataflow remain unproven.',
                           'Four leaf/no-unwind raw slot candidates are deliberately unverified; a raw byte match is not an instruction-boundary claim.',
                           'The absence of same-range mov edx,8 is a syntactic filter, not a proof that any slot call cannot receive phase bit8.',
                           'Verified +0xB8 calls on unrelated objects are not identified as wait-manager dispatches. No live PlayerLoop or startup-order claim.']}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unityplayer', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.unityplayer)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['global_mov_load_count']} global loads and {result['verified_memory_slot_b8_branch_count']} bounded slot branches; phase8 remains unresolved")
