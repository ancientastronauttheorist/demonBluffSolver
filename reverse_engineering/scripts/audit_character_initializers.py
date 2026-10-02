"""Two exact Character initializers, with an independent byte-addressed model.

Only Init and InitWithNoReset execute as native bodies. Every engine/runtime,
refresh, iterator constructor and scheduler dependency is explicitly supplied.
The independent model executes the pinned instruction semantics over a separate
memory graph; it never consumes a native event, write, register or final value.
"""
import argparse
from copy import deepcopy
import hashlib
import itertools
import json
import re
import struct
from pathlib import Path

from audit_character_assets import BUILD
from audit_il2cpp_string_creation import Machine as NativeMachine
from audit_character_reward_refresh_join import (
    pool_memory, pool_histories, pool_memory_maps, pool_state_maps,
    pool_snapshots, expand_snapshots, expand_report as expand_prior_report,
    expand_memory as expand_prior_memory, verify_history_codec,
    verify_memory_map_codec, verify_state_map_codec, verify_full_report_codec)

TARGETS = {'InitWithNoReset': (0x365720, 0x365A16, 0x365A20),
           'Init': (0x365A20, 0x365D73, 0x365D80)}
TRAPS = {0x365A0F, 0x365A15, 0x365D6C, 0x365D72}
BODY_PINS = {
    'InitWithNoReset': {'length':758,'instructions':175,'sha256':'87631b963c82a7427bf7d8cd6ceea715d6f735e6973994d678e648073b238b67'},
    'Init': {'length':851,'instructions':196,'sha256':'019cf2e6d86b5f57bed01070da47d0dae5231adb57b9c19ac9a0a43fa2700bb1'}}
# Authored selected evidence for the consumed reset/list/AL/ID/capture/reload,
# callback and publication claims. Complete decode remains in private memory;
# tracked reports never export complete proprietary native disassembly.
CHECKS = {
    0x36578A: ('mov', 'rcx, qword ptr [rdi + 0xa8]'),
    0x3657C6: ('mov', 'rcx, qword ptr [rdi + 0x148]'),
    0x3657D6: ('mov', 'r8d, dword ptr [rcx + 0x18]'),
    0x3657DD: ('inc', 'dword ptr [rcx + 0x1c]'),
    0x3657E0: ('mov', 'dword ptr [rcx + 0x18], r15d'),
    0x3657E4: ('test', 'r8d, r8d'),
    0x3657E7: ('jle', '0x3657f7'),
    0x3657FE: ('mov', 'byte ptr [rdi + 0x11c], r15b'),
    0x365827: ('test', 'al, al'),
    0x365849: ('mov', 'rbp, qword ptr [rdi + 0x98]'),
    0x36586A: ('mov', 'qword ptr [rdi + 0x98], r15'),
    0x365883: ('mov', 'qword ptr [rcx], r15'),
    0x36588E: ('mov', 'byte ptr [rdi + 0xd8], r15b'),
    0x365899: ('mov', 'qword ptr [rdi + 0x50], r14'),
    0x3658A2: ('mov', 'rdx, qword ptr [rdi + 0x50]'),
    0x3658AF: ('mov', 'rdx, qword ptr [rdx + 0x28]'),
    0x365917: ('mov', 'rbx, qword ptr [rdi + 0x48]'),
    0x36591B: ('mov', 'dword ptr [rsp + 0x40], esi'),
    0x365906: ('cmp', 'esi, -0x64'),
    0x36593F: ('mov', 'r9, qword ptr [rbx]'),
    0x365956: ('mov', 'dword ptr [rdi + 0x118], esi'),
    0x365962: ('mov', 'dword ptr [rdi + 0xe0], eax'),
    0x365968: ('mov', 'rax, qword ptr [rdi + 0x180]'),
    0x36596F: ('mov', 'dword ptr [rdi + 0xe4], 5'),
    0x365986: ('call', 'qword ptr [rax + 0x18]'),
    0x3659D9: ('mov', 'qword ptr [rcx], rdi'),
    0x3659DC: ('mov', 'dword ptr [rbx + 0x10], r15d'),
    0x365A05: ('jmp', '0x1c7f160'),
    0x365AAD: ('mov', 'qword ptr [rcx], r15'),
    0x365AF2: ('mov', 'r8d, dword ptr [rcx + 0x18]'),
    0x365AF6: ('inc', 'dword ptr [rcx + 0x1c]'),
    0x365AF9: ('mov', 'dword ptr [rcx + 0x18], r15d'),
    0x365B00: ('jle', '0x365b10'),
    0x365B16: ('mov', 'qword ptr [rcx], r15'),
    0x365B19: ('mov', 'byte ptr [rdi + 0x11c], r15b'),
    0x365B2C: ('mov', 'r14, qword ptr [rdi + 0x98]'),
    0x365B4E: ('test', 'al, al'),
    0x365B70: ('mov', 'r14, qword ptr [rdi + 0x98]'),
    0x365B91: ('mov', 'qword ptr [rdi + 0x98], r15'),
    0x365BAA: ('mov', 'qword ptr [rcx], r15'),
    0x365BB5: ('mov', 'qword ptr [rdi + 0x50], rsi'),
    0x365BC2: ('mov', 'rdx, qword ptr [rdi + 0x50]'),
    0x365BCF: ('mov', 'rdx, qword ptr [rdx + 0x28]'),
    0x365C0A: ('mov', 'qword ptr [rcx], r15'),
    0x365C0D: ('mov', 'byte ptr [rdi + 0xd8], r15b'),
    0x365C20: ('mov', 'dword ptr [rdi + 0xdc], 1'),
    0x365C2A: ('test', 'rsi, rsi'),
    0x365C33: ('mov', 'eax, dword ptr [rsi + 0x134]'),
    0x365C39: ('mov', 'dword ptr [rdi + 0xf8], eax'),
    0x365C3F: ('cmp', 'ebp, -0x64'),
    0x365C50: ('mov', 'rbx, qword ptr [rdi + 0x48]'),
    0x365C54: ('mov', 'dword ptr [rsp + 0x40], ebp'),
    0x365C78: ('mov', 'r9, qword ptr [rbx]'),
    0x365C8F: ('mov', 'dword ptr [rdi + 0x118], ebp'),
    0x365C9B: ('mov', 'dword ptr [rdi + 0xe0], eax'),
    0x365CA1: ('mov', 'rax, qword ptr [rdi + 0x180]'),
    0x365CA8: ('mov', 'dword ptr [rdi + 0xe4], 5'),
    0x365CC2: ('mov', 'rax, qword ptr [rdi + 0xf0]'),
    0x365CD2: ('mov', 'rax, qword ptr [rax + 0x10]'),
    0x365CDF: ('inc', 'dword ptr [rax + 0x1c]'),
    0x365CE2: ('mov', 'dword ptr [rax + 0x18], r15d'),
    0x365D36: ('mov', 'qword ptr [rcx], rdi'),
    0x365D39: ('mov', 'dword ptr [rbx + 0x10], r15d'),
    0x365D62: ('jmp', '0x1c7f160')}
SERVICES = {
    0x2B7B40: 'metadata', 0x281D90: 'class_init', 0x2B6FF0: 'barrier',
    0x2B7D40: 'allocate_iterator', 0x2B7D90: 'native_null_guard',
    0x1C79FD0: 'component_game_object', 0x1C7D810: 'game_object_set_active',
    0x112B9D0: 'array_clear', 0x1C82480: 'unity_inequality',
    0x1C80520: 'unity_destroy', 0xF71C60: 'string_concat',
    0x1C4B380: 'context_log', 0x1C4B450: 'plain_log',
    0x282580: 'box_int32', 0xF74DF0: 'string_format',
    0x367970: 'refresh_character', 0x367B60: 'refresh_view',
    0x33ED50: 'object_constructor', 0x1C7F160: 'start_coroutine'}
DECLARATIONS = {
    0x1C79FD0: 'UnityEngine.Component$$get_gameObject',
    0x1C7D810: 'UnityEngine.GameObject$$SetActive',
    0x112B9D0: 'System.Array$$Clear',
    0x1C82480: 'UnityEngine.Object$$op_Inequality',
    0x1C80520: 'UnityEngine.Object$$Destroy',
    0xF71C60: 'System.String$$Concat', 0x1C4B380: 'UnityEngine.Debug$$Log',
    0x1C4B450: 'UnityEngine.Debug$$Log', 0xF74DF0: 'System.String$$Format',
    0x367970: 'Character$$RefreshCharacter', 0x367B60: 'Character$$RefreshView',
    0x1C7F160: 'UnityEngine.MonoBehaviour$$StartCoroutine', 0x33ED50:'System.Object$$.ctor'}
GPRS = ['rax', 'rcx', 'rdx', 'r8', 'r9', 'r10', 'r11', 'rbx', 'rbp',
        'rsi', 'rdi', 'r12', 'r13', 'r14', 'r15', 'rsp']
VOLATILE = GPRS[:7]
NONVOLATILE = GPRS[7:15]
POISON = 0xFACE123456789090
MASK = (1 << 64)-1
REGISTER_MAP_KEYS={'all_entry_registers','entry_registers','final_registers','raw_registers','preserved','normal_callee_abi'}


def register_map_hash(value):
    return hashlib.sha256(json.dumps(value,separators=(',',':'),ensure_ascii=True).encode('utf-8')).hexdigest()


def pool_register_maps(report):
    blobs={}
    def visit(value):
        if isinstance(value,list):return [visit(v) for v in value]
        if not isinstance(value,dict):return value
        output={}
        for key,item in value.items():
            if key in REGISTER_MAP_KEYS and isinstance(item,dict):
                digest=register_map_hash(item)
                if digest in blobs:assert blobs[digest]==item
                else:blobs[digest]=deepcopy(item)
                output[key]={'register_map_ref':digest}
            else:output[key]=visit(item)
        return output
    encoded=visit(report)
    encoded['register_map_codec']='sha256_ordered_json_v1'
    encoded['register_map_blobs']=blobs
    return encoded


def expand_register_maps(report):
    assert report['register_map_codec']=='sha256_ordered_json_v1'
    blobs=report['register_map_blobs']
    for digest,value in blobs.items():assert register_map_hash(value)==digest
    def visit(value):
        if isinstance(value,list):return [visit(v) for v in value]
        if not isinstance(value,dict):return value
        if 'register_map_ref' in value:
            assert set(value)=={'register_map_ref'} and value['register_map_ref'] in blobs
            return deepcopy(blobs[value['register_map_ref']])
        return {key:visit(item) for key,item in value.items() if key not in ['register_map_codec','register_map_blobs']}
    return visit(report)


def expand_report(report):
    """Expand prior five codecs first, then complete register/ABI maps."""
    return expand_register_maps(expand_prior_report(report))


def expand_memory(report):
    """Integration adapter for an input already expanded by expand_snapshots."""
    return expand_register_maps(expand_prior_memory(report))


def verify_register_map_codec():
    original={'events':[{'all_entry_registers':{'rax':MASK,'xmm0':(1<<128)-1}},
                        {'all_entry_registers':{'rax':MASK,'xmm0':(1<<128)-1}}],
              'entry_registers':{'nullable':None,'ordered':[3,2,1]}}
    encoded=pool_register_maps(original)
    assert expand_register_maps(encoded)==original
    decoded=expand_register_maps(encoded)
    decoded['events'][0]['all_entry_registers']['rax']=0
    assert decoded['events'][1]['all_entry_registers']['rax']==MASK and original['events'][0]['all_entry_registers']['rax']==MASK
    corrupt=deepcopy(encoded);key=next(iter(corrupt['register_map_blobs']));corrupt['register_map_blobs'][key]['bad']=1
    try:expand_register_maps(corrupt)
    except AssertionError:pass
    else:raise AssertionError('register corruption accepted')
    corrupt=deepcopy(encoded);corrupt['events'][0]['all_entry_registers']['extra']=1
    try:expand_register_maps(corrupt)
    except AssertionError:pass
    else:raise AssertionError('register ref shape accepted')
    corrupt=deepcopy(encoded);corrupt['events'][0]['all_entry_registers']['register_map_ref']='0'*64
    try:expand_register_maps(corrupt)
    except AssertionError:pass
    else:raise AssertionError('missing register ref accepted')


def encode_report(report):
    return pool_snapshots(pool_state_maps(pool_memory_maps(pool_histories(pool_memory(pool_register_maps(report))))))


def verify_combined_codec():
    history=[{'ordinal':1,'normal_callee_abi':{'entry_sp':8,'caller':MASK,'preserved':{'xmm15':(1<<128)-1}}},
             {'ordinal':2,'nullable':None,'raw_args':[3,2,1]}]
    state={'memory':{'actor':'00a5ff','array':'010203'},'requests':history,
           'components':{'actor':{'nullable':None,'ordered':[2,1,0]}}}
    source={'initial':deepcopy(state),'final':deepcopy(state),'entry_registers':{'rax':MASK}}
    encoded=encode_report(source)
    assert expand_report(encoded)==source and expand_memory(expand_snapshots(encoded))==source
    decoded=expand_report(encoded);decoded['initial']['requests'][0]['normal_callee_abi']['preserved']['xmm15']=0
    decoded['initial']['memory']['actor']='ff';decoded['initial']['components']['actor']['ordered'].reverse()
    assert decoded['final']==state and source['initial']==state
    for name in ['snapshot_blobs','state_map_blobs','memory_map_blobs','history_blobs','memory_blobs','register_map_blobs']:
        corrupt=deepcopy(encoded);key=next(iter(corrupt[name]))
        if name in ['snapshot_blobs','state_map_blobs','register_map_blobs']:corrupt[name][key]['unexpected']=True
        elif name=='memory_map_blobs':corrupt[name][key]['actor']['memory_sha256']='f'*64
        elif name=='history_blobs':corrupt[name][key].reverse()
        else:corrupt[name][key]='aa'+corrupt[name][key][2:]
        try:expand_report(corrupt)
        except AssertionError:pass
        else:raise AssertionError(('combined codec corruption accepted',name))


def verify_model_primitives():
    model=object.__new__(Model)
    model.registers={'rax':MASK,'rdx':POISON,'r8':MASK}
    model.setreg('al',1);assert model.registers['rax']==0xFFFFFFFFFFFFFF01
    model.setreg('edx',0);assert model.registers['rdx']==0
    model.setreg('r8d',0x100000013);assert model.registers['r8']==19
    model.flags(0xFFFFFFFF,0xFFFFFFFF,32);assert not model.zf and model.sf and not model.of
    model.flags(0x80000000,0,32,True);assert not model.zf and model.sf!=model.of
    model.flags(0xFFFFFF9C,0xFFFFFF9C,32,True);assert model.zf
    memory=Memory({'record':(0x1000,16)},{'record':'a5'*16})
    memory.store(0x1004,0x100000013,4);assert memory.read(0x1000,16)==bytes.fromhex('a5'*4+'13000000'+'a5'*8)
    for address,kind in [(0,'READ_UNMAPPED'),(0x1010,'READ_UNMAPPED')]:
        try:memory.read(address,1)
        except Fault as error:assert error.kind==kind
        else:raise AssertionError('model missing fault')


class Fault(Exception):
    def __init__(self, kind, address, size):
        self.kind, self.address, self.size = kind, address, size


class Memory:
    """Independent exact physical records, including globals and native stack."""
    def __init__(self, layout, initial):
        self.layout = layout
        self.records = {n: bytearray.fromhex(initial[n]) for n in layout}

    def locate(self, address, size, kind):
        for n, (p, length) in self.layout.items():
            if p <= address and address+size <= p+length:
                return n, address-p
        raise Fault(kind, address, size)

    def read(self, address, size):
        n, off = self.locate(address, size, 'READ_UNMAPPED')
        return bytes(self.records[n][off:off+size])

    def write(self, address, raw):
        n, off = self.locate(address, len(raw), 'WRITE_UNMAPPED')
        self.records[n][off:off+len(raw)] = raw

    def integer(self, address, size):
        return int.from_bytes(self.read(address, size), 'little')

    def store(self, address, value, size):
        self.write(address, (value & ((1 << (size*8))-1)).to_bytes(size, 'little'))


def raw_registers(registers):
    return {'gpr': {n: registers[n] for n in VOLATILE},
            'xmm': {f'xmm{i}': registers[f'xmm{i}'] for i in range(6)}}


def snapshot(memory, layout, history, writes, callbacks):
    return {'memory': {n: memory.read(p, s).hex() for n, (p, s) in layout.items()},
            'requests': deepcopy(history), 'native_entries': deepcopy(writes),
            'callbacks': deepcopy(callbacks)}


def contract(kind, args, memory, m, options, ordinal, allocation_index, site):
    """Authored whole supplied contracts, independent of observed native effects.

    Returns explicit writes, a return value and supplied diagnostic records.
    There is no renderer, coroutine execution, role cloning or implicit scheduler.
    """
    c, d, r8, r9 = args
    p = m.p
    writes, diagnostics = [], []
    def store(a, value, size=8):
        writes.append((a, (value & ((1 << (size*8))-1)).to_bytes(size, 'little')))
    value = 0
    if kind == 'metadata':
        assert c in m.slot_addresses
        diagnostics.append({'slot': c, 'identity': memory.integer(c, 8)})
    elif kind == 'class_init':
        assert c in [p['object_class'], p['other_object_class'], p['debug_class']]
        store(c+0xE0, 1, 4)
    elif kind == 'barrier':
        assert memory.integer(c, 8) == d
        diagnostics.append({'stored_field': c, 'value': d})
    elif kind == 'component_game_object':
        assert c in [p['actor'], p['acted'], p['other_acted']]
        value = 0 if options.get('null_game_object') or options.get('null_actor_game_object') and c==p['actor'] else p['actor_go'] if c == p['actor'] or options.get('alias_component_games') else p['acted_go']
    elif kind == 'game_object_set_active':
        assert c in [p['actor_go'], p['acted_go'], p['rip'], p['dead'], p['other_dead']]
        store(c+0x40, d & 255, 1)
    elif kind == 'array_clear':
        assert d == 0 and r9 == 0 and r8 <= 3
        assert c in [p['info_items'], p['active_items'], p['other_items']]
        writes.append((c+0x20, bytes(r8*8)))
    elif kind == 'unity_inequality':
        assert d == 0 and r8 == 0
        value = 0x123456789ABCDE00 | (options.get('true_byte', 0xFE) if c and options.get('dead', 'live') == 'live' else 0)
    elif kind == 'unity_destroy':
        assert c in [0, p['dead'], p['other_dead'], p['rip'], p['acted_go']]
        if c: store(c+0x41, 1, 1)
    elif kind == 'string_concat':
        assert c in [p['literal_init'], p['other_name']] and r8 == 0
        diagnostics.append({'captured_name': d})
        value = p['concat']
    elif kind in ['context_log', 'plain_log']:
        diagnostics.append({'message': c, 'context': d})
    elif kind == 'box_int32':
        assert c == p['int_class'] and d == m.entry_sp+8
        bits = memory.integer(d, 4)
        store(p['box']+0x10, bits, 4)
        diagnostics.append({'boxed_bits': bits})
        value = p['box']
    elif kind == 'string_format':
        assert c == p['literal_format'] and d == p['box'] and r8 == 0
        diagnostics.append({'boxed_bits': memory.integer(d+0x10, 4)})
        value = 0 if options.get('null_format') else p['formatted']
    elif kind == 'tmp_text':
        assert c in [p['number'], p['other_number']]
        assert r9 == memory.integer(c, 8) and r8 == memory.integer(r9+0x560, 8)
        store(c+0x20, d)
    elif kind == 'state_callback':
        assert c == p['callback_target'] and d == p['callback_method']
        diagnostics.append({'state': memory.integer(p['actor']+0xE4, 4),
                            'current_statuses': memory.integer(p['actor']+0xF0, 8)})
    elif kind in ['refresh_character', 'refresh_view']:
        assert c == p['actor'] and d == 0
    elif kind == 'allocate_iterator':
        assert c == p['iterator_class']
        value = 0 if options.get('null_allocate') else p['iterators'][allocation_index]
        if value:
            writes.append((value, bytes(128)))
            store(value, c)
    elif kind == 'object_constructor':
        assert d == 0
        # Exact shared RET 0 gateway supplied; no folded alias is promoted.
    elif kind == 'start_coroutine':
        assert c == p['actor'] and r8 == 0
        assert memory.integer(d+0x10, 4) == 0 and memory.integer(d+0x20, 8) == c
        value = p['coroutine']
        diagnostics.append({'iterator': d, 'native_state': 0, 'actual_first_yield': False})
    else:
        raise AssertionError(kind)
    # Qualified callback plans are applied only after this completed occurrence.
    matched = [plan for plan in options.get('callbacks', [])
               if plan['kind'] == kind and plan['ordinal'] == ordinal
               and plan['site'] == hex(site)]
    for plan in matched:
        for change in plan['writes']:
            if change['record'] == 'stack': address = m.entry_sp+change['offset']
            else: address = m.layout[change['record']][0]+change['offset']
            store(address, change['value'], change.get('size', 8))
        diagnostics.append({'callback_plan': deepcopy(plan)})
    return writes, value, diagnostics


class Model:
    """Small independently implemented x64 interpreter for the two pinned bodies.

    Operand widths, zero extension, retained upper byte-register bits and signed
    branches are evaluated separately from Unicorn. Service contracts are the
    explicitly authored fixture inputs, not an inference from native results.
    """
    def __init__(self, m, row):
        self.m, self.options = m, row['options']
        self.memory = Memory(m.layout, row['initial']['memory'])
        self.registers = dict(row['entry_registers'])
        self.pc = m.base+TARGETS[row['method']][0]
        self.history = deepcopy(row['initial']['requests'])
        self.writes = deepcopy(row['initial']['native_entries'])
        self.callbacks = deepcopy(row['initial']['callbacks'])
        self.events, self.counts = [], {}
        self.error, self.fault, self.returned = None, None, False
        self.allocation_index = row['allocation_index']
        self.zf = self.sf = self.of = False

    def register_alias(self, n):
        if n in self.registers: return n, 64
        aliases = {'eax': 'rax', 'ecx': 'rcx', 'edx': 'rdx', 'ebx': 'rbx',
                   'ebp': 'rbp', 'esi': 'rsi', 'edi': 'rdi', 'esp': 'rsp',
                   'al': 'rax', 'cl': 'rcx', 'dl': 'rdx', 'bl': 'rbx',
                   'sil': 'rsi', 'dil': 'rdi', 'bpl': 'rbp'}
        if n in aliases: return aliases[n], 8 if n in ['al', 'cl', 'dl', 'bl', 'sil', 'dil', 'bpl'] else 32
        if re.fullmatch(r'r\d+[dbw]', n): return n[:-1], {'d': 32, 'b': 8, 'w': 16}[n[-1]]
        raise AssertionError(n)

    def getreg(self, name):
        if name == 'rip': return self.pc
        n, bits = self.register_alias(name)
        return self.registers[n] & ((1 << bits)-1)

    def setreg(self, name, value):
        n, bits = self.register_alias(name)
        mask = (1 << bits)-1
        self.registers[n] = value & mask if bits >= 32 else (self.registers[n] & ~mask) | (value & mask)

    def address(self, operand, ins):
        mem = operand.mem
        base = ins.address+self.m.base+ins.size if self.m.cs.reg_name(mem.base) == 'rip' else self.getreg(self.m.cs.reg_name(mem.base)) if mem.base else 0
        index = self.getreg(self.m.cs.reg_name(mem.index)) if mem.index else 0
        return (base+index*mem.scale+mem.disp) & MASK

    def read(self, operand, ins):
        if operand.type == 1: return self.getreg(self.m.cs.reg_name(operand.reg))
        if operand.type == 2: return operand.imm & ((1 << (operand.size*8))-1)
        assert operand.type == 3
        return self.memory.integer(self.address(operand, ins), operand.size)

    def write(self, operand, ins, value):
        if operand.type == 1: self.setreg(self.m.cs.reg_name(operand.reg), value)
        else:
            a, size = self.address(operand, ins), operand.size
            self.memory.store(a, value, size)
            self.writes.append([hex(ins.address), a, size, value & ((1 << (size*8))-1)])

    def flags(self, a, b, bits, subtract=False):
        mask, sign = (1 << bits)-1, 1 << (bits-1)
        result = (a-b if subtract else a & b) & mask
        self.zf, self.sf = result == 0, bool(result & sign)
        self.of = bool((a ^ b) & (a ^ result) & sign) if subtract else False

    def service(self, kind, site, target, tail):
        registers, mem = self.registers, self.memory
        args = [registers[n] for n in ['rcx', 'rdx', 'r8', 'r9']]
        caller = mem.integer(registers['rsp'], 8)
        self.counts[kind] = self.counts.get(kind, 0)+1
        ordinal = self.counts[kind]
        event = {'kind': kind, 'ordinal': ordinal, 'native_site': hex(site),
                 'service_target': target, 'caller': caller, 'entry_sp': registers['rsp'],
                 'raw_args': args, 'raw_registers': raw_registers(registers),
                 'all_entry_registers': dict(registers),
                 'snapshot': snapshot(mem, self.m.layout, self.history, self.writes, self.callbacks)}
        self.events.append(event)
        if kind == 'native_null_guard': self.error = kind; return False
        if self.options.get('failure') == [kind, ordinal]: self.error = 'supplied_stop'; return False
        assert registers['rsp'] % 16 == 8
        effects, result, diagnostics = contract(kind, args, mem, self.m, self.options, ordinal, self.allocation_index, site)
        for address, raw in effects: mem.write(address, raw)
        self.history.append({'kind': kind, 'ordinal': ordinal, 'site': hex(site), 'args': args,
                             'return': result, 'diagnostics': diagnostics,
                             'normal_callee_abi': {'entry_sp':registers['rsp'],'return_sp':registers['rsp']+8,'caller':caller,
                                                  'preserved':{n:registers[n] for n in NONVOLATILE+[f'xmm{i}' for i in range(6,16)]}},
                             'writes': [[a, b.hex()] for a, b in effects]})
        for d in diagnostics:
            if 'callback_plan' in d: self.callbacks.append(d['callback_plan'])
        for n in VOLATILE[1:]: registers[n] = POISON
        for i in range(6): registers[f'xmm{i}'] = (1 << 127) | i
        registers['rax'] = result
        self.pc = caller
        registers['rsp'] += 8
        return True

    def run(self):
        m, r = self.m, self.registers
        for _ in range(2000):
            if self.pc == m.stop: self.returned = True; break
            ins = m.instructions[self.pc-m.base]
            op, operands, following = ins.mnemonic, ins.operands, self.pc+ins.size
            try:
                if op == 'mov': self.write(operands[0], ins, self.read(operands[1], ins))
                elif op == 'lea': self.write(operands[0], ins, self.address(operands[1], ins))
                elif op in ['xor', 'add', 'sub', 'inc']:
                    a = self.read(operands[0], ins)
                    b = 1 if op == 'inc' else self.read(operands[1], ins)
                    value = a ^ b if op == 'xor' else a+b if op in ['add', 'inc'] else a-b
                    self.write(operands[0], ins, value)
                    if op == 'xor': self.flags(value, value, operands[0].size*8)
                elif op in ['cmp', 'test']:
                    self.flags(self.read(operands[0], ins), self.read(operands[1], ins), operands[0].size*8, op == 'cmp')
                elif op == 'push':
                    value = self.read(operands[0], ins); r['rsp'] -= 8
                    memop = r['rsp']; self.memory.store(memop, value, 8)
                    self.writes.append([hex(ins.address), memop, 8, value])
                elif op == 'pop':
                    value = self.memory.integer(r['rsp'], 8); r['rsp'] += 8
                    self.write(operands[0], ins, value)
                elif op in ['je', 'jne', 'jle']:
                    take = self.zf if op == 'je' else not self.zf if op == 'jne' else self.zf or self.sf != self.of
                    if take: following = m.base+operands[0].imm
                elif op in ['call', 'jmp']:
                    target = m.base+operands[0].imm if operands[0].type == 2 else self.read(operands[0], ins)
                    if op == 'call':
                        r['rsp'] -= 8; self.memory.store(r['rsp'], following, 8)
                        self.writes.append([hex(ins.address), r['rsp'], 8, following])
                    kind = m.service_kind(target)
                    assert kind is not None, (hex(ins.address), hex(target))
                    if not self.service(kind, ins.address, target, op == 'jmp'): break
                    continue
                else: raise AssertionError((op, ins.op_str))
                self.pc = following
            except Fault as fault:
                self.error = fault.kind
                self.fault = {'kind': fault.kind, 'address': fault.address, 'size': fault.size, 'rva': hex(ins.address)}
                break
        else: raise AssertionError('model budget')
        return {'events': self.events, 'final': snapshot(self.memory, m.layout, self.history, self.writes, self.callbacks),
                'final_registers': self.registers, 'returned': self.returned, 'error': self.error, 'fault': self.fault}


class Machine(NativeMachine):
    def __init__(self, game_root, dumper_root):
        manifest=json.loads((Path(__file__).parents[1]/f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
        self.input_pins={}
        for key in ['game_assembly','global_metadata']:
            expected=manifest['inputs'][key]
            raw=(game_root/expected['path']).read_bytes()
            assert len(raw)==expected['size'] and hashlib.sha256(raw).hexdigest().upper()==expected['sha256'].upper()
            self.input_pins[key]={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
        super().__init__(game_root)
        extraction = json.loads((Path(__file__).parents[1] / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
        def pin(filename, key):
            raw = (dumper_root/filename).read_bytes()
            assert len(raw)==extraction['outputs'][key]['size'] and hashlib.sha256(raw).hexdigest().upper() == extraction['outputs'][key]['sha256'].upper()
            self.input_pins[key]={'size':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}
            return raw.decode('utf-8-sig')
        self.metadata = json.loads(pin('script.json', 'script_json'))
        dump = pin('dump.cs', 'dump_cs')
        header = pin('il2cpp.h', 'il2cpp_h')
        self.classes = {}
        for name, declaration in [
            ('Character', 'public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487'),
            ('CharacterData', 'public class CharacterData : ScriptableObject, ICharacterLocData, ICardData // TypeDefIndex: 5845'),
            ('CharacterStatuses', 'public class CharacterStatuses // TypeDefIndex: 5488'),
            ('ActedInfo', 'public class ActedInfo // TypeDefIndex: 5498'),
            ('Character.<DelayReveal>d__84', 'private sealed class Character.<DelayReveal>d__84 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5482'),
            ('ECharacterState', 'public enum ECharacterState // TypeDefIndex: 5489'),
            ('EAlignment', 'public enum EAlignment // TypeDefIndex: 5492'),
            ('ECharacterStatus', 'public enum ECharacterStatus // TypeDefIndex: 5491'),
            ('Delegate', 'public abstract class Delegate : ICloneable, ISerializable // TypeDefIndex: 419'),
            ('MulticastDelegate', 'public abstract class MulticastDelegate : Delegate // TypeDefIndex: 440'),
            ('Action', 'public sealed class Action : MulticastDelegate // TypeDefIndex: 153')]:
            match = re.search('^'+re.escape(declaration)+r'\s*\{(.*?)\n\}', dump, re.M|re.S)
            assert match, declaration
            self.classes[name] = {'declaration': declaration, 'fields': match[1].split('// Methods')[0].strip()}
        self.header_bindings = {}
        for name in ['System_Collections_Generic_List_ActedInfo__Fields', 'System_Collections_Generic_List_ECharacterStatus__Fields',
                     'ActedInfo_array', 'ECharacterStatus_array', 'Il2CppObject', 'VirtualInvokeData', 'Il2CppType', 'Il2CppClass_1', 'Il2CppClass_2']:
            match = re.search(r'^struct (?:__declspec\(align\(8\)\) )?'+re.escape(name)+r'\s*\{(.*?)\n\};',header,re.M|re.S)
            assert match,name
            self.header_bindings[name]=match[0]
        for name in ['System_Collections_Generic_List_ActedInfo__Fields','System_Collections_Generic_List_ECharacterStatus__Fields']:
            assert re.search(r'\* _items;\s+int32_t _size;\s+int32_t _version;\s+Il2CppObject\* _syncRoot;',self.header_bindings[name])
        assert 'int32_t m_Items[65535];' in self.header_bindings['ECharacterStatus_array']
        assert 'ActedInfo_o* m_Items[65535];' in self.header_bindings['ActedInfo_array']
        def header_layout(name):
            offset,fields,alignment=0,{},1
            for declaration in self.header_bindings[name].split('{',1)[1].split('}',1)[0].strip().splitlines():
                declaration=declaration.strip().rstrip(';')
                field=declaration.rsplit(' ',1)[1]
                if '*' in declaration:size,align=8,8
                elif declaration.startswith('Il2CppType '):size,align=16,8
                elif declaration.startswith(('uint32_t ','int32_t ','unsigned int ')):size,align=4,4
                elif declaration.startswith('uint16_t '):size,align=2,2
                elif declaration.startswith('uint8_t '):size,align=1,1
                elif declaration.startswith('size_t '):size,align=8,8
                else:raise AssertionError(declaration)
                offset=(offset+align-1)//align*align;fields[field]=offset;offset+=size;alignment=max(alignment,align)
            return {'size':(offset+alignment-1)//alignment*alignment,'fields':fields}
        self.header_layouts={n:header_layout(n) for n in ['Il2CppClass_1','Il2CppClass_2','Il2CppType']}
        assert self.header_layouts['Il2CppClass_1']['size']==0xB8 and self.header_layouts['Il2CppType']['size']==16
        assert self.header_layouts['Il2CppClass_1']['size']+16+self.header_layouts['Il2CppClass_2']['fields']['cctor_finished']==0xE0
        vtable=self.header_layouts['Il2CppClass_1']['size']+16+self.header_layouts['Il2CppClass_2']['size']
        assert vtable==0x138 and vtable+66*16==0x558 and vtable+66*16+8==0x560
        for declaration in ['public class TextMeshProUGUI : TMP_Text, ILayoutElement // TypeDefIndex: 8974',
                            'public abstract class TMP_Text : MaskableGraphic // TypeDefIndex: 9110']:
            match=re.search('^'+re.escape(declaration)+r'\s*\{(.*?)\n\}',dump,re.M|re.S);assert match
            if 'abstract' in declaration:
                assert '// RVA: 0x1BE7620 Offset: 0x1BE6220 VA: 0x181BE7620 Slot: 66\n\tpublic virtual void set_text(string value) { }' in match[1]
        rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==0x1BE7620 and r['Name']=='TMPro.TMP_Text$$set_text']
        assert len(rows)==1 and rows[0]['TypeSignature']=='viii'
        self.tmp_declaration=rows[0]
        rows=[r for r in self.metadata['ScriptMethod'] if r['Address']==0x357700 and r['Name']=='Character.<DelayReveal>d__84$$.ctor']
        assert len(rows)==1 and rows[0]['TypeSignature']=='viii'
        self.iterator_constructor_decl=rows[0]
        iterator_block=re.search(r'^private sealed class Character\.<DelayReveal>d__84 : IEnumerator<object>, IEnumerator, IDisposable // TypeDefIndex: 5482\s*\{(.*?)\n\}',dump,re.M|re.S)[1]
        assert '// RVA: 0x357700 Offset: 0x356300 VA: 0x180357700\n\tpublic void .ctor(int <>1__state) { }' in iterator_block
        self.header_bindings['TMP_slot66']=re.search(r'^struct TMPro_TMP_Text_VTable\s*\{(.*?)\n\};',header,re.M|re.S)[0]
        slots=re.findall(r'VirtualInvokeData (\w+);',self.header_bindings['TMP_slot66'])
        assert slots[66]=='_66_set_text'
        character = re.search(r'^public class Character : MonoBehaviour, ICard // TypeDefIndex: 5487\s*\{(.*?)\n\}', dump, re.M|re.S)[1]
        self.fields = {name: int(offset, 16) for name, offset in re.findall(r'\b(\w+); // (0x[\dA-Fa-f]+)', character.split('// Methods')[0])}
        required = {'number':0x48,'dataRef':0x50,'bluff':0x58,'registerAs':0x60,'trailerInfo':0x68,'runtimeData':0x70,
                    'ripView':0x78,'createdDeadPrefab':0x98,'acteds':0xA8,'revealed':0xD8,'pickableUses':0xDC,
                    'prevState':0xE0,'state':0xE4,'killedHidden':0xEC,'killedByDemon':0xED,'statuses':0xF0,
                    'alignment':0xF8,'id':0x118,'characterStartActed':0x11C,'actedInfos':0x148,
                    'role':0x168,'bluffRole':0x170,'onStateChange':0x180,'pickeds':0x188,'savedAct':0x198,'pickable':0x1A8}
        assert all(self.fields[n] == off for n, off in required.items())
        assert 'public const ECharacterState Hidden = 5;' in self.classes['ECharacterState']['fields']
        assert 'public EAlignment startingAlignment; // 0x134' in self.classes['CharacterData']['fields']
        assert all(line in self.classes['Delegate']['fields'] for line in ['private IntPtr invoke_impl; // 0x18', 'private IntPtr method; // 0x28', 'private IntPtr method_code; // 0x40'])
        self.instructions, self.bounds, self.methods = {}, {}, []
        allslots = {r['Address']: r for section in ['ScriptMetadata', 'ScriptMetadataMethod', 'ScriptString'] for r in self.metadata[section]}
        self.bindings, self.flags = {}, set()
        import capstone
        for name, (start, end, following) in TARGETS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Name'] == 'Character$$'+name]
            assert len(rows) == 1 and rows[0]['Address'] == start and rows[0]['TypeSignature'] == 'viiii'
            assert rows[0]['Signature'] == f'void Character__{name} (Character_o* __this, CharacterData_o* character, int32_t id, const MethodInfo* method);'
            signatures = re.findall(r'^\s*(?:public|private|protected)[^\n]*\([^\n]*\) \{ \}', character.split('// Methods')[1], re.M)
            ordinal = next(i for i, signature in enumerate(signatures) if re.search(r'\b'+name+r'\(', signature))
            assert ordinal=={'Init':25,'InitWithNoReset':26}[name]
            self.methods.append(dict(rows[0], symbol_key=f'tdi5487.m{ordinal:04}'))
            assert min(r['Address'] for r in self.metadata['ScriptMethod'] if r['Address'] > start) == following
            chunks = []
            for entry in self.pe.DIRECTORY_ENTRY_EXCEPTION:
                root = entry
                while root.unwindinfo.Flags & 4: root = root.unwindinfo._chained_entry
                if root.struct.BeginAddress == start: chunks.append([entry.struct.BeginAddress, entry.struct.EndAddress])
            assert chunks == [[start, end]]
            section = self.pe.get_section_by_rva(start)
            assert section and following <= section.VirtualAddress+section.SizeOfRawData
            raw = self.pe.get_data(start, following-start)
            assert len(raw) == following-start and raw[end-start:] == b'\xcc'*(following-end)
            instructions = list(self.cs.disasm(raw[:end-start], start))
            assert sum(i.size for i in instructions) == end-start
            assert instructions[-1].mnemonic == 'int3'
            self.instructions.update({i.address: i for i in instructions})
            self.bounds[name] = {'entry':hex(start),'end_exclusive':hex(end),'next_managed':hex(following),
                                 'unwind_chunks':[[hex(a),hex(b)] for a,b in chunks],'alignment_padding':following-end,
                                 'byte_length':end-start,'instruction_count':len(instructions),
                                 'sha256':hashlib.sha256(raw[:end-start]).hexdigest()}
            assert BODY_PINS[name]=={'length':end-start,'instructions':len(instructions),'sha256':self.bounds[name]['sha256']}
            for i in instructions:
                for op in i.operands:
                    if op.type == capstone.CS_OP_MEM and op.mem.base == capstone.x86.X86_REG_RIP:
                        address = i.address+i.size+op.mem.disp
                        if address in allslots: self.bindings[address] = allslots[address]
                        elif op.size == 1: self.flags.add(address)
                        else: raise AssertionError((hex(i.address), hex(address)))
        assert self.flags == {0x288C16D,0x288C16E,0x288C173}
        assert all(a in self.instructions and (self.instructions[a].mnemonic,self.instructions[a].op_str)==pin for a,pin in CHECKS.items())
        assert {i.address for i in self.instructions.values() if i.mnemonic == 'int3'} == TRAPS
        assert {r['Value'] for r in self.bindings.values() if 'Value' in r} == {'INIT: ', '# {0}'}
        folded = list(self.cs.disasm(self.pe.get_data(0x33ED50,3),0x33ED50))
        assert [(i.mnemonic,i.op_str,i.size) for i in folded] == [('ret','0',3)]
        assert len([r for r in self.metadata['ScriptMethod'] if r['Address']==0x33ED50])==3582
        self.service_declarations = {}
        for address, name in DECLARATIONS.items():
            rows = [r for r in self.metadata['ScriptMethod'] if r['Address'] == address and r['Name'] == name]
            assert len(rows) == 1, (hex(address),name)
            self.service_declarations[hex(address)] = rows[0]
        names = ['actor','data','other_data','acted','other_acted','info','info_items',
                 'statuses','other_statuses','active','other_active','resistance','active_items','other_items','resistance_items',
                 'target','number','other_number','text_class','other_text_class','text_mi','other_text_mi',
                 'callback','callback_target','callback_method','actor_go','acted_go','rip','dead','other_dead',
                 'object_class','other_object_class','debug_class','int_class','iterator_class','literal_init','literal_format',
                 'name','other_name','concat','box','formatted','coroutine','list_clear_mi','status_clear_mi',
                 'trailer','runtime','register_as','old_data','bluff','role','bluff_role','saved_act','pickeds','pickable']
        names += [f'iterator{i}' for i in range(4)]+[f'info{i}' for i in range(3)]
        self.p = {n:self.arena+0x200000+i*0x1000 for i,n in enumerate(names)}
        self.p['iterators'] = [self.p[f'iterator{i}'] for i in range(4)]
        self.entry_sp = self.stack+0x18008
        self.layout = {n:(p,512 if n in ['actor','data','other_data'] else 1536 if n in ['text_class','other_text_class'] else 512 if n.endswith('_class') else 128) for n,p in self.p.items() if n != 'iterators'}
        self.slot_addresses = {self.base+a for a in self.bindings}
        self.layout.update({f'slot_{a:x}':(self.base+a,8) for a in self.bindings})
        self.layout.update({f'flag_{a:x}':(self.base+a,1) for a in self.flags})
        self.layout['native_stack'] = (self.entry_sp-0x200,0x300)
        self.text_gateway, self.callback_gateway = self.stop+0x100, self.stop+0x200
        self.executed = set()
        self.tracking = False
        self.u.hook_add(self.unicorn.UC_HOOK_MEM_WRITE, self.observe_write)
        self.u.hook_add(self.unicorn.UC_HOOK_MEM_INVALID, self.invalid_memory)

    def read(self, address, size): return bytes(self.u.mem_read(address,size))
    def write(self, address, raw): self.u.mem_write(address,raw)
    def integer(self, address, size): return int.from_bytes(self.read(address,size),'little')

    def registers(self):
        return {n:self.reg(getattr(self.x,'UC_X86_REG_'+n.upper())) for n in GPRS+[f'xmm{i}' for i in range(16)]}

    def service_kind(self, target):
        if target == self.text_gateway: return 'tmp_text'
        if target == self.callback_gateway: return 'state_callback'
        return SERVICES.get(target-self.base)

    def observe_write(self, uc, access, address, size, value, user_data):
        if not self.tracking:return
        if any(p <= address and address+size <= p+s for p,s in self.layout.values()):
            site = self.reg(self.x.UC_X86_REG_RIP)-self.base
            self.native_writes.append([hex(site),address,size,value & ((1 << (size*8))-1)])
        else:
            # Unmapped attempted writes are recorded by the fault hook. No
            # successfully mapped write may escape the retained physical graph.
            assert not any(a<=address and address+size-1<=b for a,b,_ in self.u.mem_regions()),(hex(address),size)

    def invalid_memory(self, uc, access, address, size, value, user_data):
        kind = 'WRITE_UNMAPPED' if access == self.unicorn.UC_MEM_WRITE_UNMAPPED else 'READ_UNMAPPED'
        self.error = kind
        self.fault = {'kind':kind,'address':address,'size':size,'rva':hex(self.reg(self.x.UC_X86_REG_RIP)-self.base)}
        return False

    def current_snapshot(self):
        return snapshot(self,self.layout,self.history,self.native_writes,self.callbacks_done)

    def hook(self, uc, address, size, user_data):
        if address == self.stop:
            self.returned = True; uc.emu_stop(); return
        rva = address-self.base
        self.executed.add(rva)
        if rva in self.instructions:
            assert self.instructions[rva].size == size and rva not in TRAPS
            self.last_site = rva
            return
        kind = self.service_kind(address)
        assert kind, hex(address)
        registers = self.registers()
        args = [registers[n] for n in ['rcx','rdx','r8','r9']]
        self.counts[kind] = self.counts.get(kind,0)+1
        ordinal = self.counts[kind]
        caller = self.rq(registers['rsp'])
        self.events.append({'kind':kind,'ordinal':ordinal,'native_site':hex(self.last_site),
                            'service_target':address,'caller':caller,'entry_sp':registers['rsp'],
                            'raw_args':args,'raw_registers':raw_registers(registers),'all_entry_registers':registers.copy(),
                            'snapshot':self.current_snapshot()})
        if kind == 'native_null_guard': self.error=kind; uc.emu_stop(); return
        if self.options.get('failure') == [kind,ordinal]: self.error='supplied_stop'; uc.emu_stop(); return
        assert registers['rsp']%16==8
        effects, value, diagnostics = contract(kind,args,self,self,self.options,ordinal,self.allocation_index,self.last_site)
        for a, raw in effects: self.write(a,raw)
        self.history.append({'kind':kind,'ordinal':ordinal,'site':hex(self.last_site),'args':args,
                             'return':value,'diagnostics':diagnostics,
                             'normal_callee_abi':{'entry_sp':registers['rsp'],'return_sp':registers['rsp']+8,'caller':caller,
                                                 'preserved':{n:registers[n] for n in NONVOLATILE+[f'xmm{i}' for i in range(6,16)]}},
                             'writes':[[a,b.hex()] for a,b in effects]})
        for d in diagnostics:
            if 'callback_plan' in d: self.callbacks_done.append(d['callback_plan'])
        for n in VOLATILE[1:]: self.u.reg_write(getattr(self.x,'UC_X86_REG_'+n.upper()),POISON)
        for i in range(6): self.u.reg_write(getattr(self.x,f'UC_X86_REG_XMM{i}'),(1 << 127)|i)
        self.u.reg_write(self.x.UC_X86_REG_RAX,value)
        self.u.reg_write(self.x.UC_X86_REG_RSP,registers['rsp']+8)
        self.u.reg_write(self.x.UC_X86_REG_RIP,caller)
        current=self.registers()
        assert all(current[n]==registers[n] for n in NONVOLATILE+[f'xmm{i}' for i in range(6,16)])

    def prepare(self, options):
        self.history,self.native_writes,self.callbacks_done = [],[],[]
        p = self.p
        for n,(address,size) in self.layout.items():
            self.write(address,bytes([(sum(n.encode('ascii')) % 200)+20])*size)
        for off,n in [(0x48,'number'),(0x50,'old_data'),(0x58,'bluff'),(0x60,'register_as'),(0x68,'trailer'),
                      (0x70,'runtime'),(0x78,'rip'),(0x98,'dead'),(0xA8,'acted'),(0xF0,'statuses'),
                      (0x148,'info'),(0x168,'role'),(0x170,'bluff_role'),(0x180,'callback'),
                      (0x188,'pickeds'),(0x198,'saved_act'),(0x1A8,'pickable')]: self.q(p['actor']+off,p[n])
        for off,v in [(0xDC,7),(0xE0,10),(0xE4,20),(0xF8,10),(0x118,73)]:self.d(p['actor']+off,v)
        for off in [0xD8,0xEC,0xED,0x11C]:self.write(p['actor']+off,b'\xfe')
        if options.get('dead') == 'absent': self.q(p['actor']+0x98,0)
        if options.get('callback') is False:self.q(p['actor']+0x180,0)
        self.q(p['info']+0x10,p['info_items']);self.d(p['info']+0x18,options.get('info_count',2));self.d(p['info']+0x1C,options.get('version',17))
        self.q(p['info_items']+0x18,3)
        for i,n in enumerate(['name','other_name','saved_act']):
            self.q(p['info_items']+0x20+i*8,p[f'info{i}']);self.q(p[f'info{i}']+0x10,p[n])
        for statuses,active,items,count in [('statuses','active','active_items',3),('other_statuses','other_active','other_items',2)]:
            self.q(p[statuses]+0x10,p[active]);self.q(p[statuses]+0x18,p['resistance']);self.q(p[statuses]+0x20,p['target'])
            self.q(p[active]+0x10,p[items]);self.d(p[active]+0x18,count);self.d(p[active]+0x1C,23)
            self.q(p[items]+0x18,3);self.write(p[items]+0x20,struct.pack('<iii',10,30,50))
        self.q(p['resistance']+0x10,p['resistance_items']);self.d(p['resistance']+0x18,2);self.d(p['resistance']+0x1C,31)
        self.write(p['resistance_items']+0x20,struct.pack('<iii',45,30,50))
        for name,other in [('data',False),('other_data',True)]:
            self.q(p[name]+0x28,p['other_name'] if other else p['name']);self.d(p[name]+0x134,0x80000014 if other else options.get('alignment',20))
            self.q(p[name]+0x140,p['role'])
        if options.get('null_name'):self.q(p['data']+0x28,0)
        for number,cls,mi in [('number','text_class','text_mi'),('other_number','other_text_class','other_text_mi')]:
            self.q(p[number],p[cls]);self.q(p[cls]+0x558,self.text_gateway);self.q(p[cls]+0x560,p[mi])
        self.q(p['callback']+0x18,self.callback_gateway);self.q(p['callback']+0x28,p['callback_method']);self.q(p['callback']+0x40,p['callback_target'])
        type_names = {'UnityEngine.Object_TypeInfo':'object_class','UnityEngine.Debug_TypeInfo':'debug_class','int_TypeInfo':'int_class',
                      'Character.<DelayReveal>d__84_TypeInfo':'iterator_class','Method$System.Collections.Generic.List<ActedInfo>.Clear()':'list_clear_mi',
                      'Method$System.Collections.Generic.List<ECharacterStatus>.Clear()':'status_clear_mi'}
        for a,row in self.bindings.items():
            n = ('literal_init' if row['Value']=='INIT: ' else 'literal_format') if 'Value' in row else type_names[row['Name']]
            self.q(self.base+a,p[n])
            if 'Value' in row:self.d(p[n]+0x10,len(row['Value']));self.write(p[n]+0x14,row['Value'].encode('utf-16-le')+b'\0\0')
        for n in ['object_class','other_object_class','debug_class']:self.d(p[n]+0xE0,0 if options.get('cold',True) else 1)
        for a in self.flags:self.write(self.base+a,bytes([0 if options.get('cold',True) else 1]))
        nullable = {'number':0x48,'acteds':0xA8,'infos':0x148,'rip':0x78,'statuses':0xF0}
        if options.get('null') in nullable:self.q(p['actor']+nullable[options['null']],0)
        if options.get('null') == 'active':self.q(p['statuses']+0x10,0)
        if options.get('alias') == 'info_active':self.q(p['statuses']+0x10,p['info'])
        if options.get('alias') == 'info_resistance':self.q(p['statuses']+0x18,p['info'])
        if options.get('alias') == 'dead_rip':self.q(p['actor']+0x98,p['rip'])
        if options.get('alias') == 'rip_acted_go':self.q(p['actor']+0x78,p['acted_go'])
        if options.get('alias_component_games'):self.q(p['actor']+0x78,p['actor_go'])
        if options.get('null') == 'items':self.q(p['info']+0x10,0)

    def run(self, name, options, retained=False, allocation_index=0):
        self.tracking=False
        if not retained:self.prepare(options)
        self.options,self.allocation_index=deepcopy(options),allocation_index
        self.events,self.counts,self.error,self.fault,self.returned=[],{},None,None,False
        x,p,sp=self.x,self.p,self.entry_sp
        # A fresh external call frame is supplied; all physical game records and
        # completed histories persist in retained calls.
        self.write(sp-0x200,bytes([0xCC])*0x300);self.q(sp,self.stop)
        registers={n:0xA110000000000000+i for i,n in enumerate(GPRS)}
        registers.update({f'xmm{i}':(0xB110000000000000+i) | ((0xC110000000000000+i)<<64) for i in range(16)})
        registers.update(rcx=0 if options.get('null')=='actor' else p['actor'],rdx=0 if options.get('null')=='data' else p['data'],
                         r8=options.get('id',0xCAFE000000000013)&MASK,r9=p['list_clear_mi'],rsp=sp)
        for n,v in registers.items():self.u.reg_write(getattr(x,'UC_X86_REG_'+n.upper()),v)
        initial=self.current_snapshot()
        self.tracking=True
        try:self.u.emu_start(self.base+TARGETS[name][0],self.stop+0x1000,count=2000)
        except self.unicorn.UcError:assert self.error in ['READ_UNMAPPED','WRITE_UNMAPPED'],self.error
        self.tracking=False
        row={'method':name,'options':deepcopy(options),'allocation_index':allocation_index,
             'entry_registers':registers,'initial':initial,'events':deepcopy(self.events),'final':self.current_snapshot(),
             'final_registers':self.registers(),'returned':self.returned,'error':self.error,'fault':self.fault}
        expected=Model(self,row).run()
        for key in expected:assert row[key]==expected[key],(name,options,key,first_difference(row[key],expected[key]))
        if self.returned:
            assert row['final_registers']['rsp']==sp+8
            for n in NONVOLATILE+[f'xmm{i}' for i in range(6,16)]:assert row['final_registers'][n]==registers[n],n
        verify_ordered_semantics(row,self)
        return row


def first_difference(a,b,path=''):
    if type(a)!=type(b):return path,type(a).__name__,type(b).__name__
    if isinstance(a,dict):
        if a.keys()!=b.keys():return path,list(a),list(b)
        for key in a:
            if a[key]!=b[key]:return first_difference(a[key],b[key],path+'/'+str(key))
    elif isinstance(a,list):
        if len(a)!=len(b):return path,len(a),len(b)
        for i,(x,y) in enumerate(zip(a,b)):
            if x!=y:return first_difference(x,y,path+'/'+str(i))
    return path,a,b


def verify_ordered_semantics(row,m):
    """Additional semantic checks do not derive expectations from final writes."""
    o,p=row['options'],m.p
    def value(which,record,off,size=8):
        return int.from_bytes(bytes.fromhex(row[which]['memory'][record])[off:off+size],'little')
    kinds=[e['kind'] for e in row['events']]
    if not row['returned']:return
    assert kinds[-4:]==['allocate_iterator','object_constructor','barrier','start_coroutine']
    assert kinds.index('refresh_character')<kinds.index('refresh_view')<kinds.index('allocate_iterator')
    assert value('final','actor',0x58)==0 and value('final','actor',0xD8,1)==0
    assert value('final','actor',0xDC,4)==1 and value('final','actor',0xED,1)==0
    assert value('final','actor',0x11C,1)==0
    # Signed original list count controls Array.Clear even for diagnostic bits.
    initial_count=value('initial','info',0x18,4)
    assert ('array_clear' in kinds)==(0<initial_count<0x80000000)
    if 'array_clear' in kinds:
        initial_items=bytes.fromhex(row['initial']['memory']['info_items'])
        expected_items=bytearray(initial_items);expected_items[0x20:0x20+8*initial_count]=bytes(8*initial_count)
        assert row['final']['memory']['info_items']==expected_items.hex()
    if row['method']=='InitWithNoReset':
        for off,size in [(0x60,8),(0x68,8),(0x70,8),(0xF8,4)]:assert value('final','actor',off,size)==value('initial','actor',off,size)
        if not o.get('callbacks'):
            for record in ['statuses','other_statuses','active','other_active','resistance','active_items','other_items','resistance_items']:
                if record=='active' and o.get('alias')=='info_active':continue
                assert row['final']['memory'][record]==row['initial']['memory'][record],record
    else:
        for off in [0x60,0x68,0x70]:assert value('final','actor',off)==0
        original_alignment=value('initial','data',0x134,4)
        updates=[c for c in o.get('callbacks',[]) if any(w['record']=='data' and w['offset']==0x134 for w in c['writes'])]
        if not updates:assert value('final','actor',0xF8,4)==original_alignment
    id_bits=o.get('id',0xCAFE000000000013)&0xFFFFFFFF
    assert ('tmp_text' in kinds)==(id_bits!=0xFFFFFF9C)
    assert value('final','actor',0x118,4)==(value('initial','actor',0x118,4) if id_bits==0xFFFFFF9C else id_bits)
    if o.get('dead')=='destroyed' and not o.get('callbacks'):assert value('final','actor',0x98)==value('initial','actor',0x98)
    if not o.get('callbacks'):
        for off,size in [(0xEC,1),(0x168,8),(0x170,8),(0x198,8),(0x188,8),(0x1A8,8)]:assert value('final','actor',off,size)==value('initial','actor',off,size)
    if 'state_callback' in kinds:
        event=next(e for e in row['events'] if e['kind']=='state_callback')
        assert int.from_bytes(bytes.fromhex(event['snapshot']['memory']['actor'])[0xE4:0xE8],'little')==5
    current_events={e['kind']:e for e in row['events']}
    current_callbacks=row['final']['callbacks'][len(row['initial']['callbacks']):]
    for plan in current_callbacks:
        for change in plan['writes']:
            record,offset,new=change['record'],change['offset'],change['value']
            if plan['kind']=='box_int32' and record=='actor' and offset==0x48:
                assert current_events['tmp_text']['raw_args'][0]==value('initial','actor',0x48)
                assert value('final','actor',0x48)==new
            if plan['kind']=='string_format' and record=='number' and offset==0 and new:
                event=current_events['tmp_text'];assert event['raw_args'][0]==p['number'] and event['raw_args'][3]==new
                assert event['raw_args'][2]==p['other_text_mi']
            if plan['kind']=='barrier' and record=='actor' and offset==0x50 and plan['ordinal']==(3 if row['method']=='InitWithNoReset' else 5):
                assert current_events['string_concat']['raw_args'][1]==p['other_name']
                if row['method']=='Init':assert value('final','actor',0xF8,4)==value('initial','data',0x134,4)
            if plan['kind']=='unity_inequality' and record=='actor' and offset==0x98:
                assert current_events['unity_inequality']['raw_args'][0]==value('initial','actor',0x98)
                assert current_events['unity_destroy']['raw_args'][0]==new
            if plan['kind']=='state_callback' and record=='actor' and offset==0xF0 and new==p['other_statuses']:
                assert value('final','active',0x18,4)==3
                assert value('final','other_active',0x18,4)==(0 if row['method']=='Init' else 2)
            if plan['kind']=='game_object_set_active' and record=='slot_2718bf0':
                later=[e for e in row['events'] if e['kind']=='class_init' and e['raw_args'][0]==new]
                assert len(later)==1 and value('final','other_object_class',0xE0,4)==1
            if record=='stack' and offset==8:
                request=next(r for r in row['final']['requests'][len(row['initial']['requests']):] if r['kind']=='box_int32')
                assert request['diagnostics'][0]['boxed_bits']==id_bits


def audit(game_root,dumper_root):
    m=Machine(game_root,dumper_root)
    cases=[]
    # Bounded full branch matrix; high R8 bits and raw true AL exercise widths.
    for name,identity,dead,count,callback,cold in itertools.product(TARGETS,[0xBEEF0000FFFFFF9C,0xCAFE000000000013],['absent','destroyed','live'],[0,2],[False,True],[False,True]):
        cases.append(m.run(name,{'id':identity,'dead':dead,'info_count':count,'callback':callback,'cold':cold}))
    for name in TARGETS:
        for bits in [0x80000000,0xFFFFFFFF,1,3]:cases.append(m.run(name,{'info_count':bits,'version':0xFFFFFFFF,'dead':'live','id':0xFFFFFFFF00000000}))
        for null in ['actor','data','acteds','infos','rip','number','statuses','active']:
            cases.append(m.run(name,{'null':null,'dead':'live'}))
        # Array.Clear is whole supplied: a null backing array is characterized
        # by a supplied stop, never mislabeled as an actual callee native fault.
        cases.append(m.run(name,{'null':'items','failure':['array_clear',1]}))
        cases.append(m.run(name,{'null':'items','info_count':0}))
        for options in [{'null_game_object':True},{'null_allocate':True},{'null_format':True},
                        {'null_name':True},
                        {'null_actor_game_object':True},{'alias_component_games':True,'dead':'live'},
                        {'null':'number','id':0x12340000FFFFFF9C},{'null':'rip','dead':'absent'},
                        {'null':'statuses','id':0xFFFFFF9C},{'null':'active','id':0xFFFFFF9C}]:cases.append(m.run(name,options))
        for alias in ['info_active','info_resistance','dead_rip','rip_acted_go']:cases.append(m.run(name,{'alias':alias,'dead':'live'}))
        for byte in [1,128,255]:cases.append(m.run(name,{'true_byte':byte,'dead':'live','alignment':0xFFFFFFFF}))
        for bits in [0x80000000,0xFFFFFFFF,0xDEAD0000FFFFFF9C]:cases.append(m.run(name,{'id':bits,'dead':'live'}))
    def callback(kind,ordinal,record,offset,value,size=8):
        return {'kind':kind,'ordinal':ordinal,'writes':[{'record':record,'offset':offset,'value':value,'size':size}]}
    def qualified(name,plan):
        start,end,_=TARGETS[name]
        if plan['kind']=='state_callback':sites=[0x365986 if name=='InitWithNoReset' else 0x365CBF]
        elif plan['kind']=='tmp_text':sites=[0x36594F if name=='InitWithNoReset' else 0x365C88]
        else:
            sites=[a for a,i in m.instructions.items() if start<=a<end and i.mnemonic in ['call','jmp']
                   and i.operands[0].type==2 and SERVICES.get(i.operands[0].imm)==plan['kind']]
        assert len(sites)>=plan['ordinal'],(name,plan)
        return dict(plan,site=hex(sites[plan['ordinal']-1]))
    p=m.p
    plans=[
        callback('barrier',3,'actor',0x50,p['other_data']),
        callback('string_concat',1,'data',0x28,p['other_name']),
        callback('box_int32',1,'actor',0x48,p['other_number']),
        callback('string_format',1,'number',0,p['other_text_class']),
        callback('string_format',1,'number',0,0),
        callback('string_format',1,'stack',8,0xAABBCCDD,4),
        callback('box_int32',1,'stack',8,0xAABBCCDD,4),
        callback('state_callback',1,'actor',0xF0,p['other_statuses']),
        callback('state_callback',1,'statuses',0x10,p['other_active']),
        callback('state_callback',1,'actor',0xE4,30,4),
        callback('state_callback',1,'actor',0xF0,0),
        callback('unity_inequality',1,'actor',0x98,p['other_dead']),
        callback('unity_inequality',1,'actor',0x78,0),
        callback('game_object_set_active',2,'actor',0x98,p['other_dead']),
        callback('game_object_set_active',2,'object_class',0xE0,0,4),
        callback('game_object_set_active',2,'slot_2718bf0',0,p['other_object_class']),
        callback('class_init',1,'data',0x134,0x80000014,4),
        callback('component_game_object',2,'actor',0x50,p['other_data']),
        callback('refresh_character',1,'actor',0x50,p['other_data']),
        callback('refresh_view',1,'actor',0x180,0),
    ]
    for name in TARGETS:
        for plan in plans:
            if name=='Init' and plan['kind']=='component_game_object' and plan['ordinal']==2:continue
            cases.append(m.run(name,{'dead':'live','callbacks':[qualified(name,plan)]}))
        # Select exact data-publication barrier per method, then clear/replace.
        ordinal=3 if name=='InitWithNoReset' else 5
        for new in [0,p['other_data']]:cases.append(m.run(name,{'dead':'live','callbacks':[qualified(name,callback('barrier',ordinal,'actor',0x50,new))]}))
        cases.append(m.run(name,{'null':'data','dead':'live','callbacks':[qualified(name,callback('barrier',ordinal,'actor',0x50,p['other_data']))]}))
        # Wrong native site remains inert even when the kind/ordinal matches.
        wrong=qualified(name,callback('string_format',1,'actor',0x48,0));wrong['site']='0x3658bd'
        cases.append(m.run(name,{'dead':'live','callbacks':[wrong]}))
    sequences=[]
    for names in [('Init','InitWithNoReset','Init'),('InitWithNoReset','InitWithNoReset','Init'),('Init','Init','InitWithNoReset')]:
        rows=[]
        for i,name in enumerate(names):
            options={'dead':'live','id':19 if i==0 else 0xBEEF0000FFFFFF9C}
            rows.append(m.run(name,options,retained=i>0,allocation_index=i))
            if i:
                previous,current=rows[i-1]['final'],rows[i]['initial']
                assert all(previous['memory'][n]==current['memory'][n] for n in m.layout if n!='native_stack')
                assert all(previous[n]==current[n] for n in ['requests','native_entries','callbacks'])
        sequences.append({'steps':rows,'continuity':'physical records/history retained; external call stack supplied each invocation'})
    for name in TARGETS:
        first=m.run(name,{'failure':['state_callback',1],'dead':'live'})
        second=m.run(name,{'dead':'live','id':-100},retained=True,allocation_index=1)
        sequences.append({'steps':[first,second],'continuity':'stopped callback then new call; no rollback or callback effects at stopped boundary'})
        plan=qualified(name,callback('state_callback',1,'actor',0xE4,30,4))
        first=m.run(name,{'callbacks':[plan]})
        second=m.run(name,{'callbacks':[plan]},retained=True,allocation_index=1)
        assert len(first['final']['callbacks'])==1 and len(second['final']['callbacks'])==2
        sequences.append({'steps':[first,second],'continuity':'same supplied callback occurs at ordinal1 in each invocation; retained chronological logs do not suppress it'})
    baselines=[]
    for name in TARGETS:
        for options in [{'dead':'live'},{'dead':'destroyed','id':-100,'info_count':0},
                        {'dead':'live','callbacks':[qualified(name,callback('state_callback',1,'actor',0xF0,p['other_statuses']))]},
                        {'dead':'live','callbacks':[qualified(name,callback('game_object_set_active',2,'object_class',0xE0,0,4))]}]:
            baselines.append(m.run(name,options))
    stops=[]
    for bid,baseline in enumerate(baselines):
        for index,event in enumerate(baseline['events']):
            options=dict(baseline['options'],failure=[event['kind'],event['ordinal']])
            row=m.run(baseline['method'],options)
            assert row['events']==baseline['events'][:index+1] and row['final']==event['snapshot']
            stops.append({'baseline':bid,'prefix_length':index+1,'result':row})
    unsupported=set(m.instructions)-m.executed-TRAPS
    assert not unsupported,[hex(a) for a in sorted(unsupported)]
    allrows=cases+[r for sequence in sequences for r in sequence['steps']]+baselines+[r['result'] for r in stops]
    for row in allrows:
        planned=row['options'].get('callbacks',[])
        reached={(e['kind'],e['ordinal'],e['native_site']) for e in row['events'] if row['options'].get('failure')!=[e['kind'],e['ordinal']]}
        for plan in row['final']['callbacks'][len(row['initial']['callbacks']):]:assert (plan['kind'],plan['ordinal'],plan['site']) in reached and plan in planned
    report={'schema':'character_initializers_native_v1','build':BUILD,'input_pins':m.input_pins,'methods':m.methods,'bounds':m.bounds,
            'class_declarations_and_fields':m.classes,'character_field_offsets':m.fields,
            'native_header_bindings':m.header_bindings,'tmp_text_supplied_declaration':m.tmp_declaration,
            'derived_header_layouts':m.header_layouts,
            'iterator_constructor_excluded':{'declaration':m.iterator_constructor_decl,'separate_body_executed':False,
                                             'actual_publication_stores_in_initializers':[hex(a) for a in [0x3659D9,0x3659DC,0x365D36,0x365D39]],
                                             'folded_object_ctor_supplied':hex(0x33ED50),'folded_alias_declarations':3582,'alias_promoted':False},
            'runtime_layout':{'list_items':0x10,'list_size_signed_dword':0x18,'list_version_dword':0x1C,
                              'reference_array_elements':0x20,'status_array_element_width':4,
                              'tmp_slot':66,'tmp_function':0x558,'tmp_methodinfo':0x560,
                              'native_class_finished_word':0xE0,'boxed_id_stack_relative_to_entry_sp':8},
            'selected_operand_assertions':{hex(a):list(pin) for a,pin in CHECKS.items()},
            'call_sites':{hex(a):[i.mnemonic,i.op_str] for a,i in m.instructions.items() if i.mnemonic in ['call','jmp']},
            'metadata_bindings':{hex(a):r for a,r in m.bindings.items()},'metadata_flags':[hex(a) for a in sorted(m.flags)],
            'supplied_declarations':m.service_declarations,'supplied_services':dict((hex(a),n) for a,n in SERVICES.items()),
            'physical_layout':{n:{'pointer':p,'size':s} for n,(p,s) in m.layout.items()},
            'terminal_traps_excluded':[hex(a) for a in sorted(TRAPS)],
            'cases':cases,'sequences':sequences,'baselines':baselines,'stops':stops,
            'summary':{'cases':len(cases),'normal_returns':sum(r['returned'] for r in cases),
                       'native_stops':sum(not r['returned'] for r in cases),'sequences':len(sequences),'baselines':len(baselines),'stops':len(stops),
                       'decoded_instructions':len(m.instructions),'executed_nontrap_instructions':len(set(m.instructions)&m.executed),
                       'physical_records':len(m.layout),'independently_compared_rows':len(allrows)},
            'model':'Separate byte-addressed Python x64 operand/branch/call-stack model from initial complete physical bytes and entry registers, plus independent ordered initializer semantic checks; no observed trace/final inputs.',
            'scope':'Exactly two complete initializer bodies. Whole supplied runtime/metadata/class/barrier/Array.Clear/Unity/logging/boxing/formatting/TMP/Action/RefreshCharacter/RefreshView/allocation/folded System.Object ctor/StartCoroutine. Iterator ctor357700 is not entered; initializer publication stores are actual. No actual first yield, renderer, clone, wait, coroutine scheduling, exception unwinding, arbitrary reentrancy or concurrent list races.'}
    verify_model_primitives();verify_register_map_codec();verify_combined_codec()
    verify_history_codec();verify_memory_map_codec();verify_state_map_codec();verify_full_report_codec()
    encoded=encode_report(report)
    assert expand_report(encoded)==report
    return encoded


if __name__=='__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('game_root',type=Path);parser.add_argument('dumper_root',type=Path)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    report=audit(args.game_root,args.dumper_root)
    raw=(json.dumps(report,separators=(',',':'),ensure_ascii=True)+'\n').encode('utf-8')
    assert len(raw)<100*1024*1024
    args.output.write_bytes(raw)
    print(json.dumps({'summary':report['summary'],'bytes':len(raw),'sha256':hashlib.sha256(raw).hexdigest()}))
