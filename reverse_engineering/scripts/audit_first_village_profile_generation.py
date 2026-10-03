"""Pinned first-village profile/roster generation with explicit runtime services.

Native bytes stay private. This auditor never opens the game or changes saves.
"""
import argparse
import hashlib
import json
import struct
from pathlib import Path
import itertools

from audit_character_assets import BUILD
from audit_ascension_assets import COUNT_FIELDS, SCRIPT_LISTS, ASCENSION_LISTS, POOL_LISTS
from audit_report_snapshots import expand_snapshots, pool_snapshots


ROOT = Path(__file__).parents[1]


def pinned(path, digest):
    raw = Path(path).read_bytes()
    assert hashlib.sha256(raw).hexdigest().upper() == digest.upper(), path
    return raw


def load_inputs(game_root, dumper_root):
    import capstone
    import pefile
    lock = json.loads((ROOT / f'manifests/builds/{BUILD}.json').read_text(encoding='utf-8'))
    extraction = json.loads((ROOT / f'manifests/extractions/{BUILD}_il2cppdumper-v6.7.46.json').read_text(encoding='utf-8'))
    raw = pinned(game_root / 'GameAssembly.dll', lock['inputs']['game_assembly']['sha256'])
    meta = json.loads(pinned(dumper_root / 'script.json', extraction['outputs']['script_json']['sha256']).decode('utf-8-sig'))
    dump = pinned(dumper_root / 'dump.cs', extraction['outputs']['dump_cs']['sha256']).decode('utf-8-sig')
    pe = pefile.PE(data=raw, fast_load=False)
    cs = capstone.Cs(capstone.CS_ARCH_X86, capstone.CS_MODE_64)
    cs.detail = True
    return raw, meta, dump, pe, cs, lock


def caller_evidence(raw, meta, pe, cs, target=0x3DCE90):
    """Verify candidate direct calls from the containing unwind start.

    A zero direct-caller result does not exclude indirect/Unity calls. Pointer
    references are raw file-backed registrations, not invocation evidence.
    """
    aliases = {}
    for row in meta['ScriptMethod']:
        aliases.setdefault(row['Address'], []).append(row['Name'])
    calls = []
    candidates = 0
    base = pe.OPTIONAL_HEADER.ImageBase
    for section in pe.sections:
        if not section.Characteristics & 0x20000000:
            continue
        body = raw[section.PointerToRawData:section.PointerToRawData + section.SizeOfRawData]
        for index in range(len(body) - 4):
            if body[index] not in (0xE8, 0xE9):
                continue
            address = section.VirtualAddress + index
            if address + 5 + struct.unpack_from('<i', body, index + 1)[0] != target:
                continue
            candidates += 1
            unwind = next((e.struct for e in pe.DIRECTORY_ENTRY_EXCEPTION
                           if e.struct.BeginAddress <= address < e.struct.EndAddress), None)
            starts = [unwind.BeginAddress] if unwind else []
            starts += [a for a in aliases if section.VirtualAddress <= a <= address
                       and address - a < 0x1000]
            found = None
            for start in sorted(set(starts), reverse=True):
                data = pe.get_data(start, address + 5 - start)
                rows = list(cs.disasm(data, start))
                if rows and rows[-1].address == address and rows[-1].mnemonic in ('call', 'jmp'):
                    found = start
                    break
            if found is not None:
                calls.append({'site_rva': hex(address), 'decoded_from_rva': hex(found),
                              'caller_names': aliases.get(found, []), 'kind': rows[-1].mnemonic})
    pointers = []
    needle = struct.pack('<Q', base + target)
    for section in pe.sections:
        if section.Characteristics & 0x20000000:
            continue
        body = raw[section.PointerToRawData:section.PointerToRawData + section.SizeOfRawData]
        at = body.find(needle)
        while at >= 0:
            pointers.append(hex(section.VirtualAddress + at))
            at = body.find(needle, at + 1)
    return {'target_rva': hex(target), 'raw_relative_candidates': candidates,
            'verified_direct_callers': calls, 'file_backed_pointer_slots': pointers,
            'scope': 'Direct relative calls/jumps validated from a managed or unwind entry; indirect invocation and engine/editor callers are not excluded.'}


class NativeJoin:
    """One retained object graph; native callers with authored collection services."""
    METHODS = [
        'GameData.UpdateScriptCharactersFromPreviousLevels',
        'AscensionsData.AddCharactersToScript', 'RoguelikeStandard.GetCurrentAscension',
        'GameData.SetupCurrentAscension', 'AscensionsData.CopyData',
        'AscensionsData.ClearCurrentPickedScript', 'AscensionsData.SetupCharactersCount',
        'AscensionsData.SetupStartingCharacters', 'AscensionsData.GetCharactersCount',
        'AscensionsData.GetStartingtCharactersOfType', 'Gameplay.SetupCurrentVillageForStandard',
        'Gameplay.GetCurrentScript', 'RoguelikeStandard.GetGameMode',
        'RoguelikeStandard.GetStartingLevel', 'Gameplay.ResetSavedCharacters',
        'GameData.GetStartingtCharactersOfType',
    ]

    def __init__(self, inputs, assets_report, characters_report, copy_policy='field_faithful'):
        import capstone
        import unicorn
        from unicorn import x86_const as x
        assert unicorn.__version__ == '2.1.4'
        self.unicorn, self.x = unicorn, x
        self.raw, self.meta, self.dump, self.pe, self.cs, self.lock = inputs
        self.copy_policy = copy_policy
        self.base = self.pe.OPTIONAL_HEADER.ImageBase
        declarations = []
        for name in ('ascension_setup', 'gameplay_core'):
            declarations += json.loads((ROOT / f'targets/{name}.json').read_text(encoding='utf-8'))['functions']
        for name, rva, signature in (
            ('RoguelikeStandard.GetGameMode', 0x3712B0, 'int32_t RoguelikeStandard__GetGameMode (RoguelikeStandard_o* __this, const MethodInfo* method);'),
            ('RoguelikeStandard.GetStartingLevel', 0x3E9C80, 'int32_t RoguelikeStandard__GetStartingLevel (RoguelikeStandard_o* __this, const MethodInfo* method);'),
            ('Gameplay.ResetSavedCharacters', 0x37FDE0, 'void Gameplay__ResetSavedCharacters (Gameplay_o* __this, const MethodInfo* method);')):
            declarations.append({'name': name, 'metadata_name': name.replace('.', '$$', 1),
                                 'rva': hex(rva), 'signature': signature})
        self.entries = {}
        self.instructions = {}
        self.body_evidence = []
        for name in self.METHODS:
            target = next(r for r in declarations if r['name'] == name)
            start = int(target['rva'], 16)
            matching = [r for r in self.meta['ScriptMethod'] if r['Name'] == target['metadata_name']
                        and r['Address'] == start and r['Signature'] == target['signature']]
            assert len(matching) == 1, name
            end = min(r['Address'] for r in self.meta['ScriptMethod'] if r['Address'] > start)
            section = self.pe.get_section_by_rva(start)
            assert section and end <= section.VirtualAddress + section.SizeOfRawData
            body = self.pe.get_data(start, end - start)
            assert len(body) == end - start
            rows = list(self.cs.disasm(body, start))
            while rows and rows[-1].mnemonic == 'int3':
                rows.pop()
            assert rows and all(a.address + a.size == b.address for a, b in zip(rows, rows[1:])), name
            assert rows[-1].address + rows[-1].size <= end
            self.instructions.update({i.address: i for i in rows})
            self.entries[name] = start
            self.body_evidence.append({'name': name, 'rva': hex(start), 'end_rva': hex(rows[-1].address + rows[-1].size),
                                       'instruction_count': len(rows), 'body_sha256': hashlib.sha256(body[:rows[-1].address + rows[-1].size - start]).hexdigest(),
                                       'signature': target['signature']})
        self.uc = unicorn.Uc(unicorn.UC_ARCH_X86, unicorn.UC_MODE_64)
        self.uc.mem_map(self.base, (self.pe.OPTIONAL_HEADER.SizeOfImage + 4095) & ~4095)
        self.uc.mem_write(self.base, self.pe.get_memory_mapped_image())
        self.arena, self.stack, self.stop = 0x300000000, 0x400000000, 0x500000000
        self.uc.mem_map(self.arena, 0x4000000)
        self.uc.mem_map(self.stack, 0x20000)
        self.uc.mem_map(self.stop, 0x1000)
        self.cursor = self.arena + 0x1000
        self.collections, self.kinds, self.labels, self.script_objects = {}, {}, {}, set()
        self.events, self.draws, self.visited, self.choices, self.failure = [], [], set(), [], None
        self.stop_service = None
        self.record_events = True
        slots = {r['Address']: r['Name'] for key in ('ScriptMetadata', 'ScriptMetadataMethod') for r in self.meta[key]}
        self.names, self.metadata_names = {}, {}
        for ins in self.instructions.values():
            for operand in ins.operands:
                if operand.type != capstone.CS_OP_MEM or operand.mem.base != capstone.x86.X86_REG_RIP:
                    continue
                address = ins.address + ins.size + operand.mem.disp
                if address in slots:
                    if slots[address] not in self.names:
                        p = self.allocate(slots[address])
                        self.names[slots[address]] = p
                        self.metadata_names[p] = slots[address]
                        self.d(p + 0xE0, 1)
                    self.q(self.base + address, self.names[slots[address]])
                elif ins.mnemonic == 'cmp' and operand.size == 1:
                    self.uc.mem_write(self.base + address, b'\1')
        self.assets = {}
        for record in characters_report['records']:
            p = self.allocate(f"asset:{record['path_id']}")
            self.assets[record['path_id']] = p
            self.d(p + 0x130, record['type'])
            self.d(p + 0x134, record['startingAlignment'])
        self.asset_ids = {p: ident for ident, p in self.assets.items()}
        self.profiles = {}
        for row in assets_report['ascensions']:
            self.profiles[row['path_id']] = self.profile(row['data'], f"profile:{row['path_id']}")
        self.profile_ids = {p: ident for ident, p in self.profiles.items()}
        self.game, self.mode, self.mode_class, self.game_static, self.project_static, self.project, self.gameplay, self.gameplay_static = [
            self.allocate(name) for name in ('GameData', 'RoguelikeStandard', 'mode_class', 'game_static',
                                             'project_static', 'ProjectContext', 'Gameplay', 'gameplay_static')]
        groups = []
        for index, group in enumerate(assets_report['game_data']['data']['roguelikeStandardAscensions']):
            p = self.allocate(f'group:{index}')
            self.q(p + 0x10, self.array([self.profiles[r[1]] for r in group], f'group:{index}:profiles'))
            groups.append(p)
        self.q(self.game + 0x50, self.array(groups, 'roguelikeStandardAscensions'))
        self.temporary = self.profiles[21702]
        self.q(self.game + 0x78, self.temporary)
        self.q(self.game_static + 0x10, self.mode)
        self.q(self.mode, self.mode_class)
        self.q(self.mode_class + 0x1E8, self.base + self.entries['RoguelikeStandard.GetCurrentAscension'])
        self.q(self.mode_class + 0x178, self.base + self.entries['RoguelikeStandard.GetGameMode'])
        self.q(self.mode_class + 0x1C8, self.base + self.entries['RoguelikeStandard.GetStartingLevel'])
        self.q(self.names['GameData_TypeInfo'] + 0xB8, self.game_static)
        self.q(self.names['ProjectContext_TypeInfo'] + 0xB8, self.project_static)
        self.q(self.names['Gameplay_TypeInfo'] + 0xB8, self.gameplay_static)
        self.q(self.project_static, self.project)
        self.q(self.project + 0x20, self.game)
        self.q(self.gameplay_static, self.gameplay)
        for index in range(4):
            self.q(self.gameplay + 0x28 + index * 8, self.list([], f'current:{index}'))
            self.q(self.gameplay + 0x48 + index * 8, self.list([], f'saved:{index}'))
        self.uc.hook_add(unicorn.UC_HOOK_CODE, self.hook)

    def allocate(self, label):
        p = self.cursor
        self.cursor += 0x1000
        assert self.cursor < self.arena + 0x4000000
        self.uc.mem_write(p, b'\0' * 0x1000)
        self.labels[p] = label
        return p

    def save(self):
        return {'memory': bytes(self.uc.mem_read(self.arena, self.cursor - self.arena)),
                'cursor': self.cursor, 'collections': {p: list(v) for p, v in self.collections.items()},
                'kinds': dict(self.kinds), 'labels': dict(self.labels), 'scripts': set(self.script_objects)}

    def restore(self, saved):
        self.uc.mem_write(self.arena, saved['memory'])
        self.cursor = saved['cursor']
        self.collections = {p: list(v) for p, v in saved['collections'].items()}
        self.kinds, self.labels, self.script_objects = dict(saved['kinds']), dict(saved['labels']), set(saved['scripts'])

    def q(self, address, value):
        self.uc.mem_write(address, struct.pack('<Q', value))

    def d(self, address, value):
        self.uc.mem_write(address, struct.pack('<I', value & 0xffffffff))

    def rq(self, address):
        return struct.unpack('<Q', self.uc.mem_read(address, 8))[0]

    def rd(self, address):
        return struct.unpack('<I', self.uc.mem_read(address, 4))[0]

    def array(self, values, label='array'):
        p = self.allocate(label)
        self.collections[p] = list(values)
        self.kinds[p] = 'array'
        self.q(p + 0x18, len(values))
        for index, value in enumerate(values):
            self.q(p + 0x20 + index * 8, value)
        return p

    def list(self, values, label='list'):
        p = self.allocate(label)
        self.kinds[p] = 'list'
        self.write_list(p, values, 0)
        return p

    def write_list(self, p, values, version=None):
        assert len(values) <= 256
        self.collections[p] = list(values)
        self.kinds[p] = 'list'
        self.q(p + 0x10, p + 0x400)
        self.q(p + 0x418, 256)
        self.d(p + 0x18, len(values))
        self.d(p + 0x1C, self.rd(p + 0x1C) + 1 if version is None else version)
        for index, value in enumerate(values):
            self.q(p + 0x420 + index * 8, value)

    def pointer_refs(self, refs):
        assert all(r[0] == 0 and r[1] in self.assets for r in refs)
        return [self.assets[r[1]] for r in refs]

    def counts(self, records, label):
        values = []
        for index, record in enumerate(records):
            p = self.allocate(f'{label}:count:{index}')
            for offset, field in enumerate(COUNT_FIELDS):
                self.d(p + 0x10 + offset * 4, record[field])
            values.append(p)
        return self.list(values, label + ':counts')

    def script(self, data, label):
        p = self.allocate(label)
        self.script_objects.add(p)
        for index, field in enumerate(SCRIPT_LISTS):
            self.q(p + 0x10 + index * 8, self.list(self.pointer_refs(data[field]), label + ':' + field))
        self.q(p + 0x38, self.counts(data['characterCounts'], label))
        return p

    def profile(self, data, label):
        p = self.allocate(label)
        assert not data['possibleScriptsData'] or label != 'profile:21674'
        # Custom payloads are not selected in the first-village domain. Preserve
        # their identities as opaque nonnull records on other profile objects.
        self.q(p + 0x18, self.array([self.allocate(f'custom:{r[1]}') for r in data['possibleScriptsData']], label + ':custom'))
        self.q(p + 0x20, self.array([self.script(s, f'{label}:inline:{i}') for i, s in enumerate(data['possibleScripts'])], label + ':inline'))
        for index, field in enumerate(ASCENSION_LISTS):
            vals = self.pointer_refs(data[field])
            constructor = self.list if field in ('unlockedCharacters', 'alwaysInDeck') else self.array
            self.q(p + 0x28 + index * 8, constructor(vals, label + ':' + field))
        self.q(p + 0x60, self.script(data['currentPickedScript'], label + ':serialized-cache'))
        for index, field in enumerate(POOL_LISTS):
            self.q(p + 0x68 + index * 8, self.array(self.pointer_refs(data[field]), label + ':' + field))
        self.q(p + 0x88, self.counts(data['characterCounts'], label))
        # CopyData sees a real list identity; payload elements remain opaque to
        # this native boundary, which does not apply day rewards.
        self.q(p + 0x90, self.list([self.allocate(label + ':addition') for _ in data['cardAdditions']], label + ':additions'))
        return p

    def norm(self, values):
        return [self.asset_ids.get(p, self.labels.get(p, 'unknown')) for p in values]

    def snapshot(self):
        original = self.profiles[21674]
        original_script = self.collections[self.rq(original + 0x20)][0]
        picked = self.rq(self.temporary + 0x60)
        selected_counts = [] if not picked else [
            {field: self.rd(p + 0x10 + index * 4) for index, field in enumerate(COUNT_FIELDS)}
            for p in self.collections[self.rq(picked + 0x38)]]
        return {'source_starting': self.norm(self.collections[self.rq(original + 0x40)]),
                'source_inline': self.norm(self.collections[self.rq(original_script + 0x10)]),
                'temporary_starting': self.norm(self.collections[self.rq(self.temporary + 0x40)]),
                'temporary_cache': self.labels.get(picked) if picked else None,
                'selected_counts': selected_counts,
                'temporary_counts': self.norm(self.collections[self.rq(self.temporary + 0x88)]),
                'rosters': [self.norm(self.collections[self.rq(self.gameplay + 0x28 + i * 8)]) for i in range(4)],
                'saved_rosters': [self.norm(self.collections[self.rq(self.gameplay + 0x48 + i * 8)]) for i in range(4)]}

    def ret(self, value=0):
        sp = self.uc.reg_read(self.x.UC_X86_REG_RSP)
        self.uc.reg_write(self.x.UC_X86_REG_RAX, value)
        self.uc.reg_write(self.x.UC_X86_REG_RSP, sp + 8)
        self.uc.reg_write(self.x.UC_X86_REG_RIP, self.rq(sp))

    def service(self, name):
        self.events.append({'service': name, 'snapshot': self.snapshot() if self.record_events else None})
        if self.stop_service == len(self.events):
            self.failure = 'service:' + name
            self.uc.emu_stop()
            return False
        return True

    def copy_record(self, source):
        """Explicit field-faithful JSON-result service, not engine serialization."""
        if source == 0:
            return 0
        if self.copy_policy == 'shared_elements':
            return source
        p = self.allocate(self.labels[source] + ':field_copy')
        self.uc.mem_write(p, bytes(self.uc.mem_read(source, 0xA0)))
        if source in self.script_objects:
            self.script_objects.add(p)
            for index in range(6):
                old = self.rq(source + 0x10 + index * 8)
                vals = self.collections[old]
                if index == 5:
                    vals = [self.copy_record(v) for v in vals]
                self.q(p + 0x10 + index * 8, self.list(vals, self.labels[old] + ':field_copy'))
        return p

    def hook(self, _, address, size, __):
        x = self.x
        a = address - self.base
        args = [self.uc.reg_read(r) for r in (x.UC_X86_REG_RCX, x.UC_X86_REG_RDX, x.UC_X86_REG_R8, x.UC_X86_REG_R9)]
        rcx, rdx, r8, r9 = args
        if address == self.stop:
            self.uc.emu_stop()
            return
        if address in (self.stop + 0x100, self.stop + 0x110):
            if self.service('supplied_mode_getter'):
                self.ret(0)
            return
        if a in self.instructions:
            self.visited.add(a)
            if a == self.entries['AscensionsData.AddCharactersToScript']:
                self.native_calls.append({'profile_path_id': self.profile_ids[rcx],
                                          'unlocked_ids': self.norm(self.collections[self.rq(rcx + 0x28)]),
                                          'cumulative_input_ids': self.norm(self.collections[rdx])})
            return
        if a in (0x2B7D90, 0x2B7D80):
            self.failure = 'native_null' if a == 0x2B7D90 else 'native_index'
            self.uc.emu_stop()
            return
        known = {0x2B6FF0: 'barrier', 0x2B7D40: 'allocation', 0xB02160: 'empty_list_ctor',
                 0xB610A0: 'list_copy_ctor', 0xB53F50: 'add_range', 0xB55950: 'contains',
                 0x2EB0: 'add', 0xB16640: 'enumerator_ctor', 0x9693D0: 'move_next',
                 0x9674A0: 'enumerator_dispose', 0x33ED50: 'folded_noop', 0xB01F50: 'to_array',
                 0x602EE0: 'copy_array_elements', 0x6033B0: 'copy_object', 0x1C86600: 'rng',
                 0xB59E70: 'remove', 0xB22150: 'get_item', 0x1C822C0: 'unity_object_equality'}
        if a not in known:
            aliases = [r['Name'] for r in self.meta['ScriptMethod'] if r['Address'] == a]
            raise AssertionError(f'unhandled native/service RVA {a:x}, aliases {aliases[:3]}, return {self.rq(self.uc.reg_read(x.UC_X86_REG_RSP))-self.base:x}')
        if not self.service(known[a]):
            return
        if a == 0x2B6FF0:
            self.ret()
        elif a == 0x2B7D40:
            self.ret(self.list([], 'allocated'))
        elif a == 0xB02160:
            self.write_list(rcx, [], 0)
            self.ret()
        elif a == 0xB610A0:
            self.write_list(rcx, self.collections[rdx], 0)
            self.ret()
        elif a == 0xB53F50:
            self.write_list(rcx, self.collections[rcx] + self.collections[rdx])
            self.ret()
        elif a == 0xB55950:
            self.ret(int(rdx in self.collections[rcx]))
        elif a == 0x2EB0:
            self.write_list(rcx, self.collections[rcx] + [rdx])
            self.ret()
        elif a == 0xB16640:
            self.q(rcx, rdx)
            self.d(rcx + 8, 0)
            self.d(rcx + 12, self.rd(rdx + 0x1C))
            self.q(rcx + 16, 0)
            self.ret(rcx)
        elif a == 0x9693D0:
            values = self.collections[self.rq(rcx)]
            index = self.rd(rcx + 8)
            if index < len(values):
                self.q(rcx + 16, values[index])
                self.d(rcx + 8, index + 1)
                self.ret(1)
            else:
                self.ret()
        elif a in (0x9674A0, 0x33ED50):
            self.ret()
        elif a == 0xB01F50:
            self.ret(self.array(self.collections[rcx], self.labels[rcx] + ':snapshot'))
        elif a == 0x602EE0:
            self.ret(self.list([self.copy_record(v) for v in self.collections[rcx]], self.labels[rcx] + ':copy'))
        elif a == 0x6033B0:
            self.ret(self.copy_record(rcx))
        elif a == 0x1C86600:
            index = self.choices.pop(0) if self.choices else 0
            assert rcx == 0 and 0 <= index < rdx, (rcx, rdx, index)
            self.draws.append({'width': rdx, 'index': index, 'caller_return_rva': hex(self.rq(self.uc.reg_read(x.UC_X86_REG_RSP)) - self.base)})
            self.ret(index)
        elif a == 0xB59E70:
            vals = list(self.collections[rcx])
            found = rdx in vals
            if found:
                vals.remove(rdx)
                self.write_list(rcx, vals)
            self.ret(int(found))
        elif a == 0xB22150:
            self.ret(self.collections[rcx][rdx])
        elif a == 0x1C822C0:
            self.ret(int(rcx == rdx))

    def run(self, name, this, argument=0, choices=(), stop_service=None, record_events=True):
        self.events, self.draws, self.choices, self.failure = [], [], list(choices), None
        self.native_calls = []
        self.stop_service = stop_service
        self.record_events = record_events
        x = self.x
        sp = self.stack + 0x10008
        self.q(sp, self.stop)
        nonvolatiles = (x.UC_X86_REG_RBX, x.UC_X86_REG_RBP, x.UC_X86_REG_RSI, x.UC_X86_REG_RDI,
                        x.UC_X86_REG_R12, x.UC_X86_REG_R13, x.UC_X86_REG_R14, x.UC_X86_REG_R15)
        for index, r in enumerate(nonvolatiles):
            self.uc.reg_write(r, 0xABC000 + index)
        for r, value in ((x.UC_X86_REG_RSP, sp), (x.UC_X86_REG_RCX, this),
                         (x.UC_X86_REG_RDX, argument), (x.UC_X86_REG_R8, 0), (x.UC_X86_REG_R9, 0),
                         (x.UC_X86_REG_RAX, 0), (x.UC_X86_REG_R10, 0), (x.UC_X86_REG_R11, 0)):
            self.uc.reg_write(r, value)
        self.uc.emu_start(self.base + self.entries[name], self.stop, timeout=5_000_000, count=500000)
        if not self.failure:
            assert self.uc.reg_read(x.UC_X86_REG_RIP) == self.stop
            assert self.uc.reg_read(x.UC_X86_REG_RSP) == sp + 8
            assert all(self.uc.reg_read(r) == 0xABC000 + i for i, r in enumerate(nonvolatiles))
        result = {'method': name, 'returned': self.labels.get(self.uc.reg_read(x.UC_X86_REG_RAX), self.uc.reg_read(x.UC_X86_REG_RAX)),
                  'failure': self.failure, 'draws': list(self.draws), 'services': list(self.events),
                  'native_add_characters_calls': list(self.native_calls), 'final': self.snapshot()}
        if name == 'GameData.UpdateScriptCharactersFromPreviousLevels':
            result['final_inline_pools_by_profile'] = {
                str(ident): [self.norm(self.collections[self.rq(script + 0x10)])
                             for script in self.collections[self.rq(p + 0x20)]]
                for ident, p in self.profiles.items()}
        return result


def audit(game_root, dumper_root, inspect=False):
    inputs = load_inputs(game_root, dumper_root)
    raw, meta, dump, pe, cs, lock = inputs
    callers = caller_evidence(raw, meta, pe, cs)
    if inspect:
        return callers
    assets_report = json.loads((ROOT / f'reports/{BUILD}_ascension_assets_audit.json').read_text(encoding='utf-8'))
    characters_report = json.loads((ROOT / f'reports/{BUILD}_character_assets_audit.json').read_text(encoding='utf-8'))
    # Reparse original pinned assets using the existing complete-object audit;
    # tracked reports cannot substitute for the original configuration graph.
    from audit_ascension_assets import audit as asset_audit
    from audit_character_assets import audit as character_audit
    assert assets_report == json.loads(json.dumps(asset_audit(game_root, dumper_root)))
    assert characters_report == json.loads(json.dumps(character_audit(game_root, dumper_root)))
    asset_names = {r['path_id']: r['name'] for r in characters_report['records']}
    assert {ident: asset_names[ident] for ident in (21596, 21621, 21623, 21627)} == {
        21596: 'Minion', 21621: 'Hunter', 21623: 'Judge', 21627: 'Medium'}
    count_records = []
    for row in assets_report['ascensions']:
        for script in [row['data'], *row['data']['possibleScripts']]:
            count_records += script['characterCounts']
    for row in assets_report['custom_scripts']:
        count_records += row['data']['scriptInfo']['characterCounts']
    assert len(count_records) == 192
    excluded = [r for r in count_records if r['allCharCount'] in (4, 5) and
                (r['town'], r['outs'], r['minion'], r['demon']) == (r['allCharCount'] - 1, 0, 0, 1)]
    assert not excluded
    source = next(r['data'] for r in assets_report['ascensions'] if r['path_id'] == 21674)
    assert source['possibleScriptsData'] == [] and len(source['possibleScripts']) == 1
    original = [21614, 21626, 21621, 21618, 21620]
    added = [21627, 21623]
    assert [r[1] for r in source['possibleScripts'][0]['startingTownsfolks']] == original
    assert [r[1] for r in source['unlockedCharacters']] == added
    assert source['possibleScripts'][0]['startingMinions'] == [[0, 21596]]
    assert source['possibleScripts'][0]['characterCounts'][0] == dict(zip(COUNT_FIELDS, [5, 4, 0, 0, 1, 4, 0, 0, 1]))
    assert source['mustInlcude'] == [] and source['alwaysInDeck'] == []
    rows, roster_rows, stopped, copy_sensitivity = [], [], [], []
    runners = []
    for policy in ('field_faithful', 'shared_elements'):
        runner = NativeJoin(inputs, assets_report, characters_report, copy_policy=policy)
        runners.append(runner)
        fresh = runner.save()
        for repetitions in (0, 1, 2):
            runner.restore(fresh)
            steps = []
            def invoke(name, this, argument=0, choices=()):
                before = runner.save()
                result = runner.run(name, this, argument, choices)
                after = runner.save()
                if policy == 'field_faithful' and repetitions == 0:
                    stop_indices = range(1, len(result['services']) + 1)
                elif policy == 'field_faithful' and repetitions == 1 and name == 'GameData.UpdateScriptCharactersFromPreviousLevels':
                    first_add = next(i + 1 for i, event in enumerate(result['services']) if event['service'] == 'add')
                    stop_indices = sorted({1, first_add, len(result['services'])})
                else:
                    stop_indices = []
                for index in stop_indices:
                    runner.restore(before)
                    partial = runner.run(name, this, argument, choices, stop_service=index)
                    assert partial['services'] == result['services'][:index]
                    assert partial['final'] == result['services'][index - 1]['snapshot']
                    assert partial['failure'] == 'service:' + result['services'][index - 1]['service']
                    stopped.append({'method': name, 'stop_service_ordinal': index,
                                    'failure': partial['failure'], 'final': partial['final'],
                                    'verified_full_service_prefix_sha256': hashlib.sha256(json.dumps(partial['services'], sort_keys=True).encode('utf-8')).hexdigest()})
                runner.restore(after)
                # Restore return register needed by handoff callers after the
                # stopped replay family, whose last attempt overwrote it.
                if name == 'Gameplay.GetCurrentScript':
                    picked = runner.rq(runner.temporary + 0x60)
                    runner.uc.reg_write(runner.x.UC_X86_REG_RAX, runner.collections[runner.rq(picked + 0x38)][0])
                return result
            for _ in range(repetitions):
                result = invoke('GameData.UpdateScriptCharactersFromPreviousLevels', runner.game)
                assert result['failure'] == 'native_index'
                assert result['returned'] == 'profile:21695:inline'
                assert len(result['native_add_characters_calls']) == 22
                assert result['native_add_characters_calls'][-1]['profile_path_id'] == 21695
                assert result['final']['source_inline'] == original + added
                assert result['final']['source_starting'] == original
                steps.append(result)
            expected = original + added if repetitions else original
            steps.append(invoke('GameData.SetupCurrentAscension', runner.game))
            assert steps[-1]['failure'] is None
            assert steps[-1]['final']['temporary_starting'] == original
            assert runner.collections[runner.rq(runner.temporary + 0x68)] == runner.pointer_refs(source['townsfolks'])
            assert runner.collections[runner.rq(runner.temporary + 0x78)] == runner.pointer_refs(source['minions'])
            assert runner.rq(runner.temporary + 0x40) == runner.rq(runner.profiles[21674] + 0x40)
            steps.append(invoke('AscensionsData.ClearCurrentPickedScript', runner.temporary))
            assert steps[-1]['final']['temporary_cache'] is None
            steps.append(invoke('AscensionsData.SetupCharactersCount', runner.temporary, choices=[0]))
            assert steps[-1]['draws'] == [{'width': 1, 'index': 0, 'caller_return_rva': '0x3b2049'}]
            steps.append(invoke('AscensionsData.SetupStartingCharacters', runner.temporary))
            assert steps[-1]['final']['temporary_starting'] == expected
            assert steps[-1]['final']['source_starting'] == original
            assert runner.rq(runner.temporary + 0x40) != runner.rq(runner.profiles[21674] + 0x40)
            # Cached getters consume no second script-selection draw.
            typed = []
            for kind in (10, 20, 30, 100):
                result = runner.run('AscensionsData.GetStartingtCharactersOfType', runner.temporary, kind)
                assert result['failure'] is None and result['draws'] == []
                typed.append({'type': kind, 'asset_ids': runner.norm(runner.collections[runner.uc.reg_read(runner.x.UC_X86_REG_RAX)])})
            assert typed == [{'type': 10, 'asset_ids': expected}, {'type': 20, 'asset_ids': []},
                             {'type': 30, 'asset_ids': [21596]}, {'type': 100, 'asset_ids': []}]
            counts_result = invoke('Gameplay.GetCurrentScript', runner.gameplay)
            assert counts_result['failure'] is None
            counts = runner.uc.reg_read(runner.x.UC_X86_REG_RAX)
            assert {field: runner.rd(counts + 0x10 + i * 4) for i, field in enumerate(COUNT_FIELDS)} == source['possibleScripts'][0]['characterCounts'][0]
            runner.q(runner.gameplay_static + 0x30, counts)
            reset_result = invoke('Gameplay.ResetSavedCharacters', runner.gameplay)
            assert reset_result['failure'] is None
            assert reset_result['final']['saved_rosters'] == [expected, [], [21596], []]
            assert reset_result['final']['rosters'] == [[], [], [], []]
            steps += [counts_result, reset_result]
            rows.append({'copy_service': policy, 'supplied_accumulation_invocations': repetitions,
                         'recovery_after_native_failure_supplied': bool(repetitions), 'steps': steps, 'typed_handoff': typed})
            if policy != 'field_faithful' or repetitions == 2:
                continue
            roster_start = runner.save()
            # Original native Standard roster loops: Minion singleton draw,
            # followed by four ordered Villager draws without replacement.
            choices = itertools.product(*(range(width) for width in range(len(expected), len(expected) - 4, -1)))
            family = []
            for choice in choices:
                runner.restore(roster_start)
                result = runner.run('Gameplay.SetupCurrentVillageForStandard', runner.gameplay,
                                    choices=[0, *choice], record_events=False)
                assert result['failure'] is None
                assert [d['width'] for d in result['draws']] == [1, *range(len(expected), len(expected) - 4, -1)]
                rosters = result['final']['rosters']
                assert len(rosters[0]) == 4 and len(set(rosters[0])) == 4 and set(rosters[0]) <= set(expected)
                assert rosters[1:] == [[], [21596], []]
                family.append({'choices': list(choice), 'draws': result['draws'], 'rosters': rosters})
            unique = {tuple(sorted(r['rosters'][0])) for r in family}
            assert len(family) == (120 if repetitions == 0 else 840)
            assert len(unique) == (5 if repetitions == 0 else 35)
            roster_rows.append({'supplied_accumulation_invocations': repetitions, 'ordered_roster_count': len(family),
                                'unordered_roster_count': len(unique), 'cases': family})
        # The JSON/copy gateway is consequential when source mutation occurs
        # after copying. Keep that sensitivity separate from original order.
        runner.restore(fresh)
        copy_first = runner.run('GameData.SetupCurrentAscension', runner.game)
        late_accumulation = runner.run('GameData.UpdateScriptCharactersFromPreviousLevels', runner.game)
        assert late_accumulation['failure'] == 'native_index'
        runner.run('AscensionsData.ClearCurrentPickedScript', runner.temporary)
        runner.run('AscensionsData.SetupCharactersCount', runner.temporary, choices=[0])
        materialized = runner.run('AscensionsData.SetupStartingCharacters', runner.temporary)
        assert materialized['final']['temporary_starting'] == (original if policy == 'field_faithful' else original + added)
        copy_sensitivity.append({'copy_service': policy, 'source_mutation_after_copy': True,
                                 'steps': [copy_first, late_accumulation, materialized]})
    return {'schema_version': 1, 'build_id': BUILD, 'game_assembly_sha256': lock['inputs']['game_assembly']['sha256'],
            'caller_evidence': callers, 'native_bodies': runners[0].body_evidence,
            'asset_absence_finding': {'count_records': len(count_records), 'matching_n45_one_demon_no_outcast_minion': len(excluded),
                                     'scope': 'Pinned authored asset count records only; arbitrary runtime profile mutation, callbacks and trailer replacement excluded.'},
            'profile_path_id': 21674, 'selected_group': 0, 'selected_village': 0, 'ordinary_minion_path_id': 21596,
            'first_village_asset_names': {str(ident): asset_names[ident] for ident in [21596, *original, *added]},
            'source_profile': source, 'configured_group_profile_ids': [[r[1] for r in group] for group in assets_report['game_data']['data']['roguelikeStandardAscensions']],
            'retained_cases': rows, 'retained_case_count': len(rows), 'roster_families': roster_rows,
            'copy_order_sensitivity_cases': copy_sensitivity,
            'ordered_roster_case_count': sum(r['ordered_roster_count'] for r in roster_rows),
            'stopped_cases': stopped, 'stopped_case_count': len(stopped),
            'executed_instruction_count': len(set().union(*(r.visited for r in runners))),
            'scope': 'Native profile accumulation/selection/materialization, actual typed reset provider and Standard roster loops with explicit field-faithful or shared-element ClassConv outputs and collection/allocation/RNG/Unity-equality services. Mode object publication, engine JSON serialization, actual accumulation invocation/recovery, LoadCharacters, SetupDelay clear/callback ordering, shuffle/physical board construction and public visibility remain unestablished by this replay.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('game_root', type=Path)
    parser.add_argument('--dumper-root', type=Path, required=True)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--inspect', action='store_true')
    args = parser.parse_args()
    result = audit(args.game_root, args.dumper_root, inspect=args.inspect)
    if args.output:
        serialized = result if args.inspect else pool_snapshots(result)
        if not args.inspect:
            assert expand_snapshots(serialized) == result
        args.output.write_text(json.dumps(serialized, separators=(',', ':')) + '\n', encoding='utf-8')
        if not args.inspect:
            print(f"Verified {result['retained_case_count']} retained cases, {result['ordered_roster_case_count']} ordered roster cases, {result['stopped_case_count']} stopped service prefixes")
    else:
        print(json.dumps(result, indent=2))
