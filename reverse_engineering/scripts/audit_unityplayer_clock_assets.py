"""Pin shipped TimeManager settings and bind their names to native clock fields.

Configuration provenance only: no claim that all runtime writers are known or
that deserialization/normalization/refresh always run in a particular order.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from audit_unityplayer_wait import ENGINE_SHA256, verify_fingerprint


PINS = {
    'globalgamemanagers': (202560, '6BD99988279019EB3190F40940B1FF2242ED28A728BE22B9C48A26F118198C32'),
    'globalgamemanagers.assets': (705652, '38E56EF97CDA4C5DFCFCB397F8077335AEBBF199460FA5B536D91698C21ED2D9'),
}
FIELDS = [
    ('Fixed Timestep', 0x48, 0x3ca3d70a, 0x1954de0),
    ('Maximum Allowed Timestep', 0x100, 0x3eaaaaab, 0x1954df0),
    ('m_TimeScale', 0xfc, 0x3f800000, 0x1954dd0),
    ('Maximum Particle Timestep', 0x104, 0x3cf5c28f, 0x1954e10),
]


def audit(data_root, unityplayer):
    import UnityPy
    import capstone
    import pefile
    if UnityPy.__version__ != '1.25.0':
        raise ValueError('UnityPy 1.25.0 required for versioned fallback type trees')
    sources = {}
    found = []
    for name, (size, digest) in PINS.items():
        raw = (data_root / name).read_bytes()
        if len(raw) != size or hashlib.sha256(raw).hexdigest().upper() != digest:
            raise ValueError(f'Serialized source fingerprint changed: {name}')
        environment = UnityPy.load(raw)
        sources[name] = {'size': size, 'sha256': digest, 'time_manager_count': 0}
        for obj in environment.objects:
            if obj.type.name != 'TimeManager':
                continue
            sources[name]['time_manager_count'] += 1
            found.append((name, obj))
    assert len(found) == 1 and found[0][0] == 'globalgamemanagers'
    name, obj = found[0]
    assert obj.class_id == 5 and obj.path_id == 8 and obj.byte_size == 16
    assert obj.assets_file.unity_version == '2022.3.10f1'
    assert obj.serialized_type.node is None  # Fields use UnityPy's versioned fallback schema.
    typed = obj.read_typetree(check_read=True)
    assert obj.reader.Position == obj.byte_start + obj.byte_size
    assert list(typed) == [field[0] for field in FIELDS]
    payload = obj.get_raw_data()
    assert len(payload) == 16
    values = struct.unpack('<4f', payload)
    bits = struct.unpack('<4I', payload)
    assert list(bits) == [field[2] for field in FIELDS]
    assert values == tuple(typed[field[0]] for field in FIELDS)
    # Independent sequential scalar parsing consumes all sixteen bytes; there
    # is no MonoBehaviour header, guessed padding, or truncated type-tree read.
    native = unityplayer.read_bytes()
    digest = verify_fingerprint(native, ENGINE_SHA256)
    pe = pefile.PE(data=native, fast_load=True)
    base = pe.OPTIONAL_HEADER.ImageBase
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    decoded = {}
    ranges = [(0x552210, 0x552358, 'jmp'), (0x552360, 0x552443, 'ret')]
    for begin, end, terminal in ranges:
        items = list(cs.disasm(pe.get_data(begin, end - begin), begin))
        assert items[0].address == begin and sum(i.size for i in items) == end - begin
        assert items[-1].address + items[-1].size == end and items[-1].mnemonic == terminal
        decoded.update({i.address: i for i in items})
    def instruction(rva, mnemonic, operands):
        assert rva in decoded
        actual = decoded[rva]
        assert (actual.mnemonic, actual.op_str) == (mnemonic, operands)
        return actual
    # The field-description transfer passes both member pointer and name to
    # one helper, marking each as a four-byte float. The named-reader transfer
    # separately pairs the same member/name with another helper.
    description_sites = [(0x552244, 0x55224c, 0x552256, 0x55225d),
                         (0x552281, 0x55228c, 0x552296, 0x55229d),
                         (0x5522c1, 0x5522cc, 0x5522d6, 0x5522dd),
                         (0x552301, 0x55230c, 0x552316, 0x55231d)]
    reader_sites = [(0x55239d, 0x5523a4, 0x5523ab), (0x5523b0, 0x5523ba, 0x5523c1),
                    (0x5523c6, 0x5523d0, 0x5523d7), (0x5523dc, 0x5523e6, 0x5523ed)]
    def string_lea(rva, register, string_rva, value):
        item = decoded[rva]
        assert item.mnemonic == 'lea' and item.op_str.startswith(register + ', [rip + ')
        assert item.address + item.size + item.operands[1].mem.disp == string_rva
        assert pe.get_string_at_rva(string_rva).decode('utf-8') == value
    fields = []
    for index, ((field, offset, expected, string_rva), desc, reader) in enumerate(zip(FIELDS, description_sites, reader_sites)):
        instruction(desc[0], 'lea', f'r9, [rbx + {offset:#x}]')
        string_lea(desc[1], 'r8', 0x19410f1, 'float')
        string_lea(desc[2], 'rdx', string_rva, field)
        instruction(desc[3], 'call', '0x7df9b0')
        instruction(reader[0], 'lea', f'rdx, [rdi + {offset:#x}]')
        string_lea(reader[1], 'r8', string_rva, field)
        instruction(reader[2], 'call', '0x332830')
        fields.append({'serialized_name': field, 'serialized_offset': index * 4,
                       'native_clock_offset': hex(offset), 'binary32_bits': f'{expected:08x}',
                       'value': values[index], 'native_name_rva': hex(string_rva),
                       'description_call_rva': hex(desc[3]), 'named_reader_call_rva': hex(reader[2])})
    for rva in (0x552271, 0x5522b1, 0x5522f1, 0x552331):
        instruction(rva, 'mov', 'dword ptr [rdx + rcx + 0xc], 4')
    vtable = {}
    for slot in range(0, 0x100, 8):
        pointer = struct.unpack('<Q', pe.get_data(0x1954e38 + slot, 8))[0] - base
        if pointer in (0x552210, 0x552360):
            vtable[hex(slot)] = hex(pointer)
    assert vtable == {'0x78': '0x552360', '0x80': '0x552210'}
    return {'schema_version': 1, 'engine_sha256': digest, 'sources': sources,
            'unity_version': obj.assets_file.unity_version, 'unitypy_version': UnityPy.__version__,
            'time_manager': {'source': name, 'class_id': obj.class_id, 'path_id': obj.path_id,
                             'object_size': len(payload), 'consumed_bytes': 16,
                             'object_sha256': hashlib.sha256(payload).hexdigest().upper(),
                             'schema_provenance': 'UnityPy versioned fallback; serialized object has no embedded type tree',
                             'fields': fields},
            'native_vtable_rva': '0x1954e38', 'native_transfer_slots': vtable,
            'native_transfer_ranges': [[hex(a), hex(b)] for a, b, _ in ranges],
            'native_mapping_assertion_count': 32,
            'scope': 'Complete pinned shipped TimeManager configuration plus two independent native field/name transfer mappings. Static native instruction assertions, not execution of deserialization helpers. Does not establish runtime load ordering, normalization/refresh ordering, all runtime writers, or live state.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('data_root', type=Path)
    parser.add_argument('--unityplayer', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    result = audit(args.data_root, args.unityplayer)
    args.output.write_text(json.dumps(result, indent=2, allow_nan=False) + '\n', encoding='utf-8')
    print('Verified complete shipped TimeManager and four native clock field bindings')
