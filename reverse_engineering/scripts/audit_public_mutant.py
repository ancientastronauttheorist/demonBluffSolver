"""Verify the pinned public Mutant/Skinwalker boundary and inherited empty clues.

This is instruction/metadata evidence. Allocator and shared List constructor
semantics are separate runtime contracts; this audit does not execute the game.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path
from audit_character_assets import BUILD


def audit(game_root, dumper_root):
    import capstone
    import pefile
    repo = Path(__file__).parents[1]
    manifest = json.loads((repo / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((repo / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))

    def pinned(path, digest):
        raw = path.read_bytes()
        if hashlib.sha256(raw).hexdigest().upper() != digest.upper():
            raise ValueError(f'fingerprint changed: {path.name}')
        return raw

    native = pinned(Path(game_root) / 'GameAssembly.dll', manifest['inputs']['game_assembly']['sha256'])
    dump = pinned(Path(dumper_root) / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    script = json.loads(pinned(Path(dumper_root) / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    boundaries = {'Skinwalker': (5921, 'Demon', [0x3EBEF0, 0x33ED50, 0x3CFFF0]),
                  'Demon': (5919, 'Role', [0x3D7060, 0x3D6BD0, 0x3D6B70, 0x33ED50, 0x3D6C30, 0x3D6E70, 0x3D6AC0, 0x3CFFF0])}
    for name, (tdi, parent, rvas) in boundaries.items():
        block = re.search(r'public class ' + name + r' : ' + parent + r' // TypeDefIndex: ' + str(tdi) + r'.*?\n\}', dump, re.S)
        if not block or [int(v, 16) for v in re.findall(r'// RVA: (0x[0-9A-Fa-f]+)', block[0])] != rvas:
            raise ValueError(f'declaration boundary changed: {name}')
    pe = pefile.PE(data=native, fast_load=True)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    instructions = {}
    for start, size in ((0x3EBEF0, 0x55), (0x3D6B70, 0x5B), (0x3D6BD0, 0x5B),
                        (0x3D7060, 0x2C), (0x3C4CA0, 0x11), (0x3712B0, 3),
                        (0x33ED50, 3), (0x3CFFF0, 7), (0x357920, 7)):
        instructions.update({i.address: i for i in cs.disasm(pe.get_data(start, size), start)})
    checks = [(0x3EBF25, 'call', '0x2b7d40'), (0x3EBF31, 'mov', 'rcx, rax'),
              (0x3EBF34, 'mov', 'rbx, rax'), (0x3EBF37, 'call', '0xb02160'),
              (0x3EBF3C, 'mov', 'rax, rbx'), (0x3EBF44, 'ret', ''),
              (0x3D6BA5, 'call', '0x2b7d40'), (0x3D6BB1, 'xor', 'r9d, r9d'),
              (0x3D6BB4, 'xor', 'r8d, r8d'), (0x3D6BBD, 'call', '0x35d5d0'),
              (0x3D6C05, 'call', '0x2b7d40'), (0x3D6C11, 'xor', 'r9d, r9d'),
              (0x3D6C14, 'xor', 'r8d, r8d'), (0x3D6C1D, 'call', '0x35d5d0'),
              (0x3C4CA3, 'mov', 'r9, qword ptr [rax + 0x210]'),
              (0x3C4CAA, 'jmp', 'qword ptr [rax + 0x208]'),
              (0x3712B0, 'xor', 'eax, eax'), (0x3712B2, 'ret', ''),
              (0x33ED50, 'ret', '0'), (0x3CFFF0, 'xor', 'edx, edx'),
              (0x3CFFF2, 'jmp', '0x357920'), (0x357920, 'xor', 'edx, edx'),
              (0x357922, 'jmp', '0x33ed50')]
    if any(address not in instructions for address, _, _ in checks):
        raise ValueError("requested assertion falls outside decoded instructions")
    for address, mnemonic, operands in checks:
        i = instructions[address]
        if (i.mnemonic, i.op_str) != (mnemonic, operands):
            raise ValueError(f'instruction relationship changed: {address:x}')

    def referenced_slot(address):
        i = instructions[address]
        if i.mnemonic != 'mov' or i.operands[1].type != capstone.x86.X86_OP_MEM or i.operands[1].mem.base != capstone.x86.X86_REG_RIP:
            raise ValueError('expected RIP-relative metadata load')
        return i.address + i.size + i.operands[1].mem.disp

    strings = {r['Address']: r['Value'] for r in script['ScriptString']}
    for address in (0x3D6BAA, 0x3D6C0A, 0x3D7080):
        if strings.get(referenced_slot(address)) != '':
            raise ValueError('empty inherited string changed')
    metadata = {r['Address']: r['Name'] for r in script['ScriptMetadata']}
    methods = {r['Address']: r['Name'] for r in script['ScriptMetadataMethod']}
    for address in (0x3D6B9E, 0x3D6BFE):
        if metadata.get(referenced_slot(address)) != 'ActedInfo_TypeInfo':
            raise ValueError('clue allocation type changed')
    if metadata.get(referenced_slot(0x3EBF1E)) != 'System.Collections.Generic.List<SpecialRule>_TypeInfo':
        raise ValueError('rule list allocation type changed')
    if methods.get(referenced_slot(0x3EBF2A)) != 'Method$System.Collections.Generic.List<SpecialRule>..ctor()':
        raise ValueError('rule list constructor instantiation changed')
    return {'schema_version': 1, 'build_id': BUILD,
            'game_assembly_sha256': manifest['inputs']['game_assembly']['sha256'],
            'declarations': {name: {'type_def_index': tdi, 'base_class': parent, 'rvas': [hex(r) for r in rvas]}
                             for name, (tdi, parent, rvas) in boundaries.items()},
            'native_relationships_verified': len(checks) + 7,
            'rules': 'Fresh List<SpecialRule> allocation and default constructor call; no rule additions',
            'action': 'Skinwalker and Demon Act share ret 0; inherited BluffAct dispatches to Act',
            'clues': 'Both inherited Demon clue methods allocate ActedInfo with empty text and null character references',
            'registration': 'Inherited Role.GetRegisterAsRole returns null',
            'scope': 'Static instruction and pinned metadata verification; asset binding and configuration references are separate reports; no global reachability or complete Demon lifecycle claim'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root, args.dumper_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"Verified {report['native_relationships_verified']} public Mutant/base relationships")
