"""Execute the pinned engine JSON parser with bounded allocator/string services.

No engine bytes are retained. This audit uses the installed private PE at run time.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_unityplayer_wait import ENGINE_SHA256


class Machine:
    def __init__(self, game_root):
        import capstone
        import pefile
        import unicorn
        from unicorn import x86_const as x
        self.x, self.unicorn = x, unicorn
        assert unicorn.__version__ == '2.1.4'
        raw = (Path(game_root) / 'UnityPlayer.dll').read_bytes()
        assert hashlib.sha256(raw).hexdigest().upper() == ENGINE_SHA256.upper()
        self.pe = pefile.PE(data=raw, fast_load=True)
        self.pe.parse_data_directories(
            directories=[pefile.DIRECTORY_ENTRY['IMAGE_DIRECTORY_ENTRY_EXCEPTION']])
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        self.cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
        self.u = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        self.u.mem_map(self.base, (self.pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
        self.u.mem_write(self.base, self.pe.get_memory_mapped_image())
        self.arena, self.stack, self.services = 0x300000000, 0x400000000, 0x500000000
        self.stop = self.services + 0x1000
        self.u.mem_map(self.arena, 0x1000000)
        self.u.mem_map(self.stack, 0x20000)
        self.u.mem_map(self.services, 0x2000)
        self.manager = self.arena + 0x10000
        self.q(self.base + 0x1CD52F8, self.manager)
        self.executed = set()
        self.parser_error = None
        self.error_override = None
        self.u.hook_add(unicorn.UC_HOOK_CODE, self.hook)

    def q(self, a, v):
        self.u.mem_write(a, struct.pack('<Q', v))

    def rq(self, a):
        return struct.unpack('<Q', self.u.mem_read(a, 8))[0]

    def d(self, a, v):
        self.u.mem_write(a, struct.pack('<I', v))

    def rd(self, a):
        return struct.unpack('<I', self.u.mem_read(a, 4))[0]

    def reg(self, r):
        return self.u.reg_read(r)

    def ret(self, value=0):
        x = self.x
        sp = self.reg(x.UC_X86_REG_RSP)
        self.u.reg_write(x.UC_X86_REG_RAX, value)
        self.u.reg_write(x.UC_X86_REG_RSP, sp + 8)
        self.u.reg_write(x.UC_X86_REG_RIP, self.rq(sp))

    def alloc(self, size):
        assert 0 <= size <= 0x100000
        address = self.next_alloc
        self.next_alloc += (max(size, 16) + 15) & ~15
        assert self.next_alloc < self.arena + 0xF00000
        self.u.mem_write(address, bytes(max(size, 16)))
        self.allocations[address] = size
        return address

    def cstring(self, address):
        data = bytearray()
        for i in range(65536):
            byte = self.u.mem_read(address + i, 1)[0]
            if not byte:
                return bytes(data)
            data.append(byte)
        raise AssertionError('unbounded C string')

    def put_string(self, address, content):
        self.u.mem_write(address, bytes(40))
        if len(content) <= 24:
            self.u.mem_write(address, content)
            self.u.mem_write(address + 24, bytes([24 - len(content)]))
            self.u.mem_write(address + 32, b'\x01')
        else:
            buffer = self.alloc(len(content) + 1)
            self.u.mem_write(buffer, content)
            self.q(address, buffer)
            self.q(address + 16, len(content))
        self.d(address + 36, 1)

    def get_string(self, address):
        if self.u.mem_read(address + 32, 1)[0] == 1:
            length = 24 - struct.unpack('<b', self.u.mem_read(address + 24, 1))[0]
            return bytes(self.u.mem_read(address, length))
        length = self.rq(address + 16)
        return bytes(self.u.mem_read(self.rq(address), length)) if length else b''

    def hook(self, _, address, size, __):
        x = self.x
        a = address - self.base
        self.executed.add(a)
        cx, dx, r8, r9 = [self.reg(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        if a == 0x354970:
            assert cx == self.manager
            self.events.append(['allocate', dx, r8, r9])
            self.ret(self.alloc(dx))
        elif a == 0x354BA0:
            assert cx == self.manager
            out = self.alloc(r8)
            if dx:
                assert dx in self.allocations
                self.u.mem_write(out, bytes(self.u.mem_read(dx, min(r8, self.allocations[dx]))))
            self.events.append(['reallocate', self.allocations.get(dx, 0), r8, r9])
            self.ret(out)
        elif a == 0x354EC0:
            assert cx == self.manager
            self.events.append(['free', self.allocations.get(dx), r8])
            self.ret()
        elif a == 0x355150:
            assert cx == self.manager
            self.ret(0)
        elif a == 0x351BE0:
            self.events.append(['free_thunk', self.allocations.get(cx), dx])
            self.ret()
        elif a == 0x159230:
            content = bytes(self.u.mem_read(dx, r8))
            self.events.append(['string_assign_utf8_hex', content.hex()])
            self.put_string(cx, content)
            self.ret(cx)
        elif a == 0x14F740:
            self.put_string(cx, self.get_string(dx))
            self.ret(cx)
        elif a == 0x670140:
            assert self.cstring(dx) == b'JSON parse error: %s'
            error = self.cstring(r8)
            self.events.append(['format_error', error.decode('utf-8')])
            self.put_string(cx, b'JSON parse error: ' + error)
            self.ret(cx)
        elif a == 0xAACCEA:
            tree = self.reg(x.UC_X86_REG_RBX)
            if self.error_override is not None:
                self.d(tree + 0x120, self.error_override)
            self.parser_error = self.rd(tree + 0x120)

    def node(self, address, depth=0):
        assert depth < 32
        tag = self.rd(address + 16)
        kind = tag & 7
        if kind == 0:
            return {'kind': 'null'}
        if kind in (1, 2):
            return {'kind': 'bool', 'value': kind == 2}
        if kind in (3, 4):
            pointer, count = self.rq(address), self.rd(address + 8)
            assert count <= 256
            if kind == 3:
                return {'kind': 'object', 'members': [[self.node(pointer + i * 48, depth + 1), self.node(pointer + i * 48 + 24, depth + 1)] for i in range(count)]}
            return {'kind': 'array', 'items': [self.node(pointer + i * 24, depth + 1) for i in range(count)]}
        if kind == 5:
            pointer, length = self.rq(address), self.rd(address + 8)
            assert length <= 65536
            return {'kind': 'string', 'flags': tag,
                    'input_offset': pointer - self.arena if self.arena <= pointer < self.arena + self.input_size else None,
                    'utf8_hex': bytes(self.u.mem_read(pointer, length)).hex()}
        assert kind == 6
        return {'kind': 'number', 'flags': tag, 'payload_u64': self.rq(address)}

    def parse(self, payload, error_override=None):
        x = self.x
        self.next_alloc = self.arena + 0x20000
        self.allocations, self.events = {}, []
        self.parser_error, self.error_override = None, error_override
        source, error = self.arena, self.arena + 0x8000
        assert len(payload) < 0x7000
        self.input_size = len(payload) + 1
        self.u.mem_write(source, payload + b'\0')
        self.put_string(error, b'')
        sp = self.stack + 0x18008
        self.q(sp, self.stop)
        preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(preserved):
            self.u.reg_write(register, 0xBCDE0000 + i)
        self.u.reg_write(x.UC_X86_REG_MXCSR, 0x1F80)
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, source), (x.UC_X86_REG_RDX, 0), (x.UC_X86_REG_R8, error)]:
            self.u.reg_write(register, value)
        try:
            self.u.emu_start(self.base + 0xAACC80, self.stop, timeout=5_000_000, count=1_000_000)
        except Exception as exc:
            raise AssertionError(f'payload={payload!r}, RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        assert self.reg(x.UC_X86_REG_RIP) == self.stop, (payload, hex(self.reg(x.UC_X86_REG_RIP)-self.base))
        assert self.reg(x.UC_X86_REG_RSP) == sp + 8
        for i, register in enumerate(preserved):
            assert self.reg(register) == 0xBCDE0000 + i
        result = self.reg(x.UC_X86_REG_RAX)
        self.last_tree = result
        return {'input_utf8_hex': payload.hex(), 'error_code_override': error_override,
                'parser_error_code': self.parser_error,
                'error': self.get_string(error).decode('utf-8'),
                'tree': self.node(result + 0xC8) if result else None,
                'input_after_utf8_hex': bytes(self.u.mem_read(source, len(payload))).hex(),
                'events': self.events.copy()}

    def render(self, pretty=False):
        x = self.x
        assert self.last_tree
        context, output = self.arena + 0xA000, self.arena + 0xB000
        self.u.mem_write(context, bytes(0x140))
        self.u.mem_write(context + 0xB0, bytes(self.u.mem_read(self.last_tree + 0xC8, 24)))
        self.put_string(output, b'')
        sp = self.stack + 0x18008
        self.q(sp, self.stop)
        preserved = [x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI, x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15]
        for i, register in enumerate(preserved):
            self.u.reg_write(register, 0xCDEF0000 + i)
        for register, value in [(x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, context), (x.UC_X86_REG_RDX, output), (x.UC_X86_REG_R8, int(pretty))]:
            self.u.reg_write(register, value)
        try:
            self.u.emu_start(self.base + 0x1096690, self.stop, timeout=5_000_000, count=1_000_000)
        except Exception as exc:
            raise AssertionError(f'render RVA={self.reg(x.UC_X86_REG_RIP)-self.base:x}') from exc
        assert self.reg(x.UC_X86_REG_RIP) == self.stop
        assert self.reg(x.UC_X86_REG_RSP) == sp + 8
        for i, register in enumerate(preserved):
            assert self.reg(register) == 0xCDEF0000 + i
        return self.get_string(output)


ERRORS = [
    'The document is empty.',
    'The document root must not follow by other values.',
    'Invalid value.',
    'Missing a name for object member.',
    'Missing a colon after a name of object member.',
    "Missing a comma or '}' after an object member.",
    "Missing a comma or ']' after an array element.",
    'Incorrect hex digit after \\u escape in string.',
    'The surrogate pair in string is invalid.',
    'Invalid escape character in string.',
    'Missing a closing quotation mark in string.',
    'Invalid encoding in string.',
    'Number too big to be stored in double.',
    'Miss fraction part in number.',
    'Miss exponent in number.',
    'Terminate parsing due to Handler error.',
    'Unspecific syntax error.',
]


def verify_native(m):
    # Verified entries/callees and complete final instructions. The parser's
    # 18-entry RVA jump table starts immediately after its last return.
    ranges = [(0xAACC80, 0xAACE98), (0xAAC870, 0xAAC9A7), (0x1096690, 0x10969F9)]
    instructions = {}
    for start, end in ranges:
        assert any(e.struct.BeginAddress == start and e.struct.EndAddress >= end
                   for e in m.pe.DIRECTORY_ENTRY_EXCEPTION)
        raw = m.pe.get_data(start, end - start)
        assert len(raw) == end - start
        decoded = list(m.cs.disasm(raw, start))
        assert sum(i.size for i in decoded) == end - start
        instructions.update({i.address: i for i in decoded})
    checks = {
        0xAACCB2: ('mov', 'edx, 0x158'),
        0xAACCC3: ('call', '0x354970'),
        0xAACCCD: ('lea', 'r9d, [rbx + 9]'),
        0xAACCD1: ('mov', 'byte ptr [rsp + 0x20], 1'),
        0xAACCD6: ('mov', 'r8d, 0x4000'),
        0xAACCE2: ('call', '0xaac870'),
        0xAACCEA: ('movsxd', 'rdi, dword ptr [rbx + 0x120]'),
        0xAACD26: ('mov', 'ecx, dword ptr [rdx + rdi*4 + 0xaace98]'),
        0xAACD30: ('jmp', 'rcx'),
        0xAACE33: ('ret', ''),
        0xAACE34: ('cmp', 'byte ptr [rbx + 0xd8], 3'),
        0xAACE82: ('ret', ''),
        0xAACE97: ('ret', ''),
        0xAAC91E: ('call', '0xaad650'),
        0xAAC94C: ('call', '0xaac740'),
        0xAAC96F: ('call', '0xaad7d0'),
        0xAAC980: ('call', '0xaae190'),
        0xAAC9A6: ('ret', ''),
        0x10969F8: ('ret', ''),
    }
    for address, expected in checks.items():
        assert address in instructions, hex(address)
        actual = instructions[address]
        assert (actual.mnemonic, actual.op_str) == expected, (hex(address), actual.op_str)
    table = struct.unpack('<18I', m.pe.get_data(0xAACE98, 72))
    assert all(address in instructions for address in table)
    return {'instruction_assertions': len(checks),
            'entry_code_ranges': [[hex(a), hex(b)] for a, b in ranges],
            'error_jump_table_targets': [hex(a) for a in table]}


def audit(game_root):
    m = Machine(game_root)
    verified = verify_native(m)
    payloads = [b'', b' ', b'{}', b'{"a":1}', b'{"a":1,"a":2}', b'{"x":[true,false,null,1,-2,3.5,"abc"]}', b'[]', b'null', b'true', b'1', b'"s"', b'{}{}', b'{', b'{a:1}', b'{"a" 1}', b'{"a":}', b'{"a":1 "b":2}', b'{"a":[1 2]}', b'{"a":"\\uQQQQ"}', b'{"a":"\\uD800x"}', b'{"a":"\\x"}', b'{"a":"x}', b'{"a":1e9999}', b'{"a":1.}', b'{"a":1e}', b'{"a":"\\u0041\\uD83D\\uDE00"}']
    numbers = [b'-0', b'0.0', b'-0.0', b'2147483647', b'2147483648',
               b'-2147483648', b'-2147483649', b'4294967295', b'4294967296',
               b'9223372036854775807', b'9223372036854775808',
               b'18446744073709551615', b'18446744073709551616',
               b'-9223372036854775808', b'-9223372036854775809',
               b'1.7976931348623157e308', b'1.7976931348623159e308',
               b'2.2250738585072014e-308', b'1e-307', b'5e-324', b'1e-9999',
               b'NaN', b'Infinity', b'-Infinity', b'01', b'+1', b'.1',
               b'0x10', b'1e+3', b'1E-3']
    payloads += [b'{"a":' + n + b'}' for n in numbers]
    strings = [b'\xff', b'\xc0\xaf', b'\xed\xa0\x80', b'\xf4\x90\x80\x80',
               b'\x01', b'\\uDC00', b'\\uD800\\u0041', b'\\u0000',
               b'\\u001F', b'\\t\\r\\n\\b\\f\\/\\\\\\"', b'\xc3\xa9',
               b'\xf0\x9f\x98\x80']
    payloads += [b'{"a":"' + s + b'"}' for s in strings]
    payloads += [b'{"a":"' + b'x' * n + b'"}' for n in [0, 1, 23, 24, 25, 64, 512]]
    payloads += [b'{"a":[' + b','.join(str(i).encode() for i in range(n)) + b']}'
                 for n in [0, 1, 8, 32, 128]]
    payloads += [b'{}\0garbage', b'\xef\xbb\xbf{}', b'{"a":1,}', b'{"a":[1,]}',
                 b'{"a":truee}', b'{"a":/*comment*/1}', b'{"a":"\\uD800"}',
                 b'{"a":null,"a":{"a":[null,{},[]]}}',
                 b'{"score":42,"profile":{"name":"test","unlocked":[1,2,3],"enabled":true}}']
    cases = []
    for payload in payloads:
        case = m.parse(payload)
        if case['tree']:
            compact = m.render()
            case['compact_utf8_hex'] = compact.hex()
            case['pretty_utf8_hex'] = m.render(True).hex()
            roundtrip = m.parse(compact)
            assert roundtrip['tree'] is not None and roundtrip['error'] == ''
            case['compact_roundtrip_tree'] = roundtrip['tree']
        else:
            # The native parser/tree is destroyed before its diagnostic string
            # is assigned or formatted. Free services are inert observations.
            kinds = [e[0] for e in case['events']]
            diagnostic = next(i for i, k in enumerate(kinds)
                              if k in ['format_error', 'string_assign_utf8_hex'])
            assert any(k == 'free' for k in kinds[:diagnostic])
            if case['parser_error_code']:
                assert case['error'] == 'JSON parse error: ' + ERRORS[case['parser_error_code'] - 1]
        cases.append(case)
    # Controlled result-code overrides prove every native diagnostic arm; these
    # are deliberately separate from naturally produced syntax-error fixtures.
    overrides = []
    for code in range(1, 19):
        case = m.parse(b'{}', code)
        expected = ERRORS[code - 1] if code <= 17 else 'Unknown error.'
        assert case['tree'] is None
        assert case['error'] == 'JSON parse error: ' + expected
        overrides.append(case)
    by_input = {bytes.fromhex(c['input_utf8_hex']): c for c in cases}
    assert bytes.fromhex(by_input[b'{"a":1,"a":2}']['compact_utf8_hex']) == b'{"a":1,"a":2}'
    assert by_input[b'{"a":"\xff"}']['tree']['members'][0][1]['utf8_hex'] == 'ff'
    assert by_input[b'{"a":"\\uDC00"}']['tree']['members'][0][1]['utf8_hex'] == 'edb080'
    assert by_input[b'{"a":"\\uD800\\u0041"}']['parser_error_code'] == 9
    assert bytes.fromhex(by_input[b'{"a":NaN}']['compact_utf8_hex']) == b'{"a":NaN}'
    assert bytes.fromhex(by_input[b'{}\0garbage']['compact_utf8_hex']) == b'{}'
    assert by_input[b'{"a":18446744073709551615}']['tree']['members'][0][1] == {
        'kind': 'number', 'flags': 8710, 'payload_u64': 18446744073709551615}
    assert by_input[b'{"a":18446744073709551616}']['tree']['members'][0][1] == {
        'kind': 'number', 'flags': 16902, 'payload_u64': 4895412794951729152}
    negative_zero = by_input[b'{"a":-0.0}']
    assert negative_zero['tree']['members'][0][1]['payload_u64'] == 1 << 63
    assert bytes.fromhex(negative_zero['compact_utf8_hex']) == b'{"a":0.0}'
    assert negative_zero['compact_roundtrip_tree']['members'][0][1]['payload_u64'] == 0
    return {'schema_version': 1, 'build_id': BUILD, 'engine_sha256': ENGINE_SHA256,
            'native_case_count': len(cases), 'controlled_error_case_count': len(overrides),
            'executed_address_count': len(m.executed), 'mxcsr': '0x1f80',
            'native_verified': verified, 'cases': cases, 'error_overrides': overrides,
            'scope': 'Actual engine JSON parse entry, in-place recursive parser/tree construction, cleanup and tree rendering. Explicit always-success bounded allocation/reallocation/free and native-string services; no managed string conversion, metadata field selection/application, object-reference resolution, allocator ownership/failure or exception-unwind claim. Render inputs are parsed trees, not arbitrary managed objects. All byte strings remain hex, including invalid UTF-8. Controlled error codes do not claim natural reachability.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = audit(args.game_root)
    args.output.write_text(json.dumps(report, indent=2) + '\n', encoding='utf-8')
    print(f"{report['native_case_count']} natural cases, {report['controlled_error_case_count']} controlled error cases, {report['executed_address_count']} addresses")
