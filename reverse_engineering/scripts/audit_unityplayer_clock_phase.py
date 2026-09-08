"""Join the pinned clock callback to its qualified default PlayerLoop node."""
import argparse
import json
from pathlib import Path

from audit_unityplayer_phases import PHASES, emulate_loop
from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


def audit(path):
    import capstone
    import pefile
    raw = path.read_bytes(); digest = verify_fingerprint(raw, ENGINE_SHA256)
    pe = pefile.PE(data=raw, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64); cs.detail = True
    instructions = {}
    for begin, end in [(0x81A880, 0x820F66), (0x59BFE0, 0x59C900)]:
        instructions.update({i.address: i for i in cs.disasm(pe.get_data(begin, end - begin), begin)})
    checked = set()
    def reference(address, mnemonic, prefix, expected):
        ins = instructions[address]
        targets = [ins.address + ins.size + op.mem.disp for op in ins.operands
                   if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP]
        if ins.mnemonic != mnemonic or not ins.op_str.startswith(prefix) or targets != [expected]:
            raise ValueError('Clock phase reference mismatch')
        checked.add(address)
    name = 'TimeUpdate/WaitForLastPresentationAndUpdateTime'
    for address, register, expected in [(0x81F276, 'r9', name),
                                         (0x81F27D, 'r8', 'UnityEngine.PlayerLoop'),
                                         (0x81F284, 'rdx', 'UnityEngine.CoreModule.dll')]:
        ins = instructions[address]; target = ins.address + ins.size + ins.operands[1].mem.disp
        reference(address, 'lea', register + ',', target)
        if pe.get_data(target, len(expected) + 1) != expected.encode('ascii') + b'\0':
            raise ValueError('Clock phase qualified type identity mismatch')
    for address, mnemonic, operands in [(0x81F29C, 'call', '0x75fc00'),
                                         (0x81F2BD, 'mov', 'rax, qword ptr [rax]'),
                                         (0x81F2C0, 'mov', 'qword ptr [rcx + 0xa28], rax')]:
        ins = instructions[address]
        if (ins.mnemonic, ins.op_str) != (mnemonic, operands): raise ValueError('Clock phase type cache mismatch')
        checked.add(address)
    reference(0x81F2A1, 'mov', 'rcx,', 0x1CD6AF8)
    reference(0x59C662, 'lea', 'rax,', 0x5B72C0)
    reference(0x59C669, 'mov', 'qword ptr [rip', 0x1CAA1E8)
    clock = {'name': name, 'cache': 0xA28, 'cell': 0x1CAA1E8}
    count, indices, executed = emulate_loop(raw, [clock, *PHASES])
    if count != 131 or indices != [2, 34, 56, 70, 89, 113]:
        raise ValueError('Default clock/wait node construction changed')
    return {'schema_version': 1, 'engine_sha256': digest,
            'native_relationships_verified': len(checked), 'constructed_nodes': count,
            'distinct_builder_instructions': executed,
            'clock': {'name': name, 'callback_rva': '0x5B72C0', 'callback_cell_rva': '0x1CAA1E8',
                      'type_cache_offset': '0xA28', 'node_index': indices[0]},
            'wait_nodes': [{'name': p['name'], 'mask': p['mask'], 'node_index': i} for p, i in zip(PHASES, indices[1:])],
            'scope': 'Qualified managed type-cache lookup, callback installation and native default-loop construction with synthetic initialized type tags. The clock callback body is audited separately. Runtime loop modifications, callback guards and phase bit 8 remain separate boundaries.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('unityplayer', type=Path); parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args(); result = audit(args.unityplayer)
    args.output.write_text(json.dumps(result, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {result['native_relationships_verified']} clock-phase relationships and {result['constructed_nodes']} default-loop nodes")
