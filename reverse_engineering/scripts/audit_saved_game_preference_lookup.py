"""Join exact preference request literals to pinned registrations and native lookup."""
import argparse
import hashlib
import json
from pathlib import Path

from audit_character_assets import BUILD
from audit_unity_json_gateway import audit as audit_gateway
from audit_unityplayer_icalls import audit as audit_registration, decode_pointer_pairs
from audit_unityplayer_wait import ENGINE_SHA256


def audit(game_root):
    import capstone
    import pefile
    root = Path(__file__).parents[1]
    manifest = json.loads((root / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    raw = (Path(game_root) / 'GameAssembly.dll').read_bytes()
    assert hashlib.sha256(raw).hexdigest().upper() == manifest['inputs']['game_assembly']['sha256'].upper()
    game = pefile.PE(data=raw, fast_load=True)
    engine_path = Path(game_root) / 'UnityPlayer.dll'
    # Execute all established registration-loop and export-sink assertions first.
    audit_registration(engine_path)
    raw = engine_path.read_bytes()
    assert hashlib.sha256(raw).hexdigest().upper() == ENGINE_SHA256
    engine = pefile.PE(data=raw, fast_load=True)
    engine.parse_data_directories(directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
    def cstring(pe, rva):
        section = pe.get_section_by_rva(rva)
        assert section and 0 <= rva - section.VirtualAddress < section.SizeOfRawData
        count = min(2048, section.SizeOfRawData - rva + section.VirtualAddress)
        data = pe.get_data(rva, count)
        assert len(data) == count and b'\0' in data
        return data.split(b'\0', 1)[0].decode('utf-8')
    def executable(rva):
        section = engine.get_section_by_rva(rva)
        return section is not None and bool(section.Characteristics & 0x20000000)
    bindings = decode_pointer_pairs(engine.get_data(0x1894FC0, 0xD77 * 8), engine.get_data(0x189BB80, 0xD77 * 8),
                                    engine.OPTIONAL_HEADER.ImageBase, lambda rva: cstring(engine, rva), executable)
    assert len(bindings) == 3447
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    requests, fixtures = [], []
    for entry, end, site, request, index, target in [
        (0x1C85F20, 0x1C85F82, 0x1C85F5C, 'UnityEngine.PlayerPrefs::GetString(System.String,System.String)', 2172, 0xF3150),
        (0x1C86170, 0x1C861FF, 0x1C8618C, 'UnityEngine.PlayerPrefs::TrySetSetString(System.String,System.String)', 2169, 0xF22B0)]:
        decoded = {i.address: i for i in cs.disasm(game.get_data(entry, end - entry), entry)}
        assert site in decoded
        ins = decoded[site]
        assert ins.mnemonic == 'lea' and ins.operands[1].mem.base == capstone.x86.X86_REG_RIP
        literal = ins.address + ins.size + ins.operands[1].mem.disp
        assert cstring(game, literal) == request
        name = request.split('(', 1)[0]
        assert bindings[name] == {'index': index, 'rva': target}
        unwind = next(e for e in engine.DIRECTORY_ENTRY_EXCEPTION if e.struct.BeginAddress == target)
        assert not unwind.unwindinfo.Flags & 4
        pointer = game.OPTIONAL_HEADER.ImageBase + target  # Authored lookup value, never invoked here.
        fixtures += [(request, [(name, pointer)], pointer, 'pref:' + name + ':fallback'),
                     (request, [(name, pointer), (request, 0x12345678)], 0x12345678, 'pref:' + name + ':exact-precedence'),
                     (name, [(name, pointer)], pointer, 'pref:' + name + ':bare'),
                     (request, [], 0, 'pref:' + name + ':missing'),
                     (request, [('UnityEngine.PlayerPrefs::Other', pointer)], 0, 'pref:' + name + ':wrong-prefix')]
        requests.append({'request': request, 'request_literal_rva': hex(literal), 'registration': name,
                         'registration_index': index, 'engine_target_rva': hex(target),
                         'first_unwind_fragment': [hex(target), hex(unwind.struct.EndAddress)],
                         'complete_engine_method_claimed': False})
    report = audit_gateway(game_root, fixtures)
    cases = [r for r in report['lookup_cases'] if r['case'].startswith('pref:')]
    assert len(cases) == 10
    for case in cases:
        if case['case'].endswith(':fallback'):
            request = next(r['request'] for r in requests if r['registration'] in case['case'])
            assert case['constructed_keys'] == [request, request, request.split('(', 1)[0]]
        elif case['case'].endswith(':exact-precedence'):
            assert len(case['constructed_keys']) == 1
    return {'build_id': BUILD, 'engine_sha256': ENGINE_SHA256, 'verified_registration_pair_count': len(bindings),
            'requests': requests, 'native_lookup_cases': cases, 'native_lookup_case_count': len(cases),
            'reused_gateway_native_checks': report['native_checks'],
            'scope': 'Exact native preference request literals and all 3447 shipped registration pairs are verified. Ten native GameAssembly map lookup/fallback fixtures join the names with authored pointer values. Registration population in a live map, engine preference entry bodies, storage and exception behavior remain open; no preference access occurs.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(report['native_lookup_case_count'], report['verified_registration_pair_count'])
