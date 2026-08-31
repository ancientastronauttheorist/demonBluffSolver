"""
Demon Bluff Memory Reader
Reads live game state directly from process memory.
Extracts current CharacterData roles, disguises, alignment, and state for all
cards. Current role can differ from a physical card's stable origin.

Usage:
    python memory_reader.py          # dump current board
    python memory_reader.py --watch  # continuously watch for changes
"""

import atexit
import ctypes
import struct
import sys
import os
import threading
import time
import argparse
from ctypes import wintypes

# Fix Windows console encoding
if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')
    os.system('')  # enable ANSI escape codes

kernel32 = ctypes.WinDLL('kernel32', use_last_error=True)
psapi = ctypes.WinDLL('psapi', use_last_error=True)

PROCESS_VM_READ = 0x0010
PROCESS_QUERY_INFORMATION = 0x0400

# IL2CPP offsets (from dump.cs / Il2CppDumper script.json analysis)
# Gameplay class
GAMEPLAY_TYPEINFO_RVA = 0x26F8140  # Il2CppClass* pointer address (RVA from GA base)
IL2CPP_CLASS_STATIC_FIELDS_OFFSET = 0xB8
IL2CPP_CLASS_NAME_OFFSET = 0x10
GAMEPLAY_INSTANCE_STATIC_OFFSET = 0x10
GAMEPLAY_CHARACTERS_OFFSET = 0x68
GAMEPLAY_SCORE_STATIC_OFFSET = 0x8   # static Score Score

# Score class field offsets
SCORE_COMPLETED_STAGES_OFFSET = 0x10
SCORE_POINTS_FOR_COMPLETING_OFFSET = 0x14
SCORE_COMPLETED_DAYS_OFFSET = 0x18
SCORE_MULTIPLIER_OFFSET = 0x1C
SCORE_KILLED_GOODS_OFFSET = 0x28
SCORE_TEMP_UNREVEALED_OFFSET = 0x2C
SCORE_UNREVEALED_CARDS_OFFSET = 0x30
SCORE_KILLED_EVILS_OFFSET = 0x34
SCORE_TEMP_KILLED_EVILS_OFFSET = 0x38
SCORE_POINT_PER_KILL_OFFSET = 0x3C
SCORE_POINTS_PER_UNREVEALED_OFFSET = 0x40

# Gameplay deck lists (List<CharacterData>)
GAMEPLAY_TOWNSFOLKS_OFFSET = 0x28
GAMEPLAY_OUTSIDERS_OFFSET = 0x30
GAMEPLAY_MINIONS_OFFSET = 0x38
GAMEPLAY_DEMONS_OFFSET = 0x40

# Characters class
CHARACTERS_LIST_OFFSET = 0x20

# List<T> internals
LIST_ITEMS_OFFSET = 0x10
LIST_SIZE_OFFSET = 0x18

# IL2CPP array
ARRAY_FIRST_ELEMENT_OFFSET = 0x20

# Character class field offsets
CHAR_DATAREF_OFFSET = 0x50       # current CharacterData* (not stable origin)
CHAR_BLUFF_OFFSET = 0x58         # CharacterData* (disguise)
CHAR_REGISTERAS_OFFSET = 0x60    # CharacterData* (registers as)
CHAR_STATE_OFFSET = 0xE4         # ECharacterState (int32)
CHAR_ALIGNMENT_OFFSET = 0xF8     # EAlignment (int32)
CHAR_ID_OFFSET = 0x118           # int32 (position)
CHAR_KILLED_HIDDEN_OFFSET = 0xEC # bool (killed by Lilis)
CHAR_REVEALED_OFFSET = 0xD8      # bool (card has been flipped)
CHAR_STATUSES_OFFSET = 0xF0      # CharacterStatuses object

# CharacterStatuses field offsets
CSTATUS_LIST_OFFSET = 0x10       # List<ECharacterStatus>

# ECharacterStatus enum
CHAR_STATUS = {
    0: 'None', 10: 'Corrupted', 15: 'Lying', 20: 'Mad',
    25: 'AppearTruthful', 26: 'AppearLying', 27: 'AppearDisguised',
    30: 'HealthyBluff', 35: 'BrokenAbility', 38: 'WorkingAbility',
    40: 'NoDamage', 45: 'CorruptionResistant',
    50: 'MessedUpByEvil', 55: 'KilledByEvil', 60: 'UnkillableByDemon',
    70: 'AlteredCharacter',
}

# Character clue/ability field offsets (Phase 2)
CHAR_RUNTIME_DATA_OFFSET = 0x70  # RuntimeCharacterData (polymorphic per role)
CHAR_ACTED_OFFSET = 0xA8         # Acted (speech bubble component)
CHAR_LEFT_ACT_OFFSET = 0xB0      # bool (left ability activated)
CHAR_USES_OFFSET = 0xDC          # int (ability use count / pickableUses)
CHAR_ACTED_INFOS_OFFSET = 0x148  # List<ActedInfo>
CHAR_SAVED_ACT_OFFSET = 0x198    # string (cached clue text)
CHAR_ACT_OFFSET = 0x1A1          # bool (ability activated flag)

# ActedInfo class field offsets
ACTED_INFO_DESC_OFFSET = 0x10    # string (formatted clue text)
ACTED_INFO_CHARS_OFFSET = 0x18   # List<Character> (referenced positions)

# EnlightenedRuntimeData.direction enum
EVIL_DIRECTION = {0: 'Equidistant', 10: 'CW', 20: 'CCW'}

# CharacterData class field offsets
CD_CHARACTER_ID_OFFSET = 0x18    # string (role name) -- STALE in multi-village!
CD_CACHED_PTR_OFFSET = 0x10      # IntPtr m_CachedPtr (Unity native object)
CD_NATIVE_NAME_OFFSET = 0x48     # char* name in Unity native object (RELIABLE)
CD_TYPE_OFFSET = 0x130           # ECharacterType (int32)
CD_ALIGNMENT_OFFSET = 0x134      # EAlignment (int32)

# Enum mappings
ALIGNMENT = {0: 'None', 10: 'Good', 20: 'Evil'}
STATE = {0: 'None', 5: 'Hidden', 10: 'Alive', 20: 'Dead', 30: 'Revealed'}
CHAR_TYPE = {0: 'None', 10: 'Villager', 20: 'Outcast', 30: 'Minion', 100: 'Demon'}

# Displayed roles that NEVER show a passive speech bubble.
# The game's savedAct field (offset 0x198) persists stale clue strings from a
# previous village until overwritten. For display roles with no passive clue,
# the string is always stale — null it out here so print_board and auto_card
# see a clean input. Matches NO_INFO_ROLES + ACTIVE_ONLY_ROLES in
# game_loop.py:1078,1312 (kept in sync by comment, not imported).
NO_PASSIVE_CLUE_DISPLAY_ROLES = {
    'wretch', 'bombardier', 'knight', 'doppelganger',
    'dreamer', 'druid', 'fortune teller', 'jester', 'judge',
    'slayer', 'plague doctor',
}

# Internal name → display name mapping
DISPLAY_NAMES = {
    'Juggler': 'Jester',
    'Lillith': 'Lilis',
    'Striga': 'Lilis',
    'Doppleganger': 'Doppelganger',
    'Mathematician': 'Lover',
    'RangedEmpath': 'Bard',
    'BountyHunter': 'Bounty Hunter',
    'Immortal': 'Knight',
    'Recluse': 'Wretch',
    'Skinwalker': 'Mutant',
    'Imp': 'Baa',
    'Baron': 'Chancellor',
    'Marionette': 'Twin Minion',
    'Illuzionist': 'Shaman',
    'Mezepheles': 'Puppeteer',
    'Athlete': 'Bard',
    'Acrobat': 'Acrobat',
    # The shipped public Judge CharacterData binds Judge2. Arbiter is a
    # separate unbound implementation and must not be conflated with it.
    'Judge2': 'Judge',
    'Noble': 'Noble',
    'Gossip': 'Poet',
    'Gambler': 'Gemcrafter',
    'Lookout': 'Medium',
    'Sapper': 'Sapper',
    'Archivist': 'Archivist',
    'Shugenja': 'Shugenja',
    'Tracker': 'Hunter',
    'Investigator': 'Oracle',
    'Librarian': 'Librarian',
    'Mutant': 'Mutant',
    'Puzzlemaster': 'Plague Doctor',
    'Cipher': 'Witch',
    'Scout': 'Scout',
    'Knitter': 'Knitter',
    'Witness': 'Witness',
    'Puppet': 'Puppet',
}


def clean_name(raw_name):
    """Strip numeric suffix and map internal→display name."""
    if not raw_name:
        return '?'
    # Strip _XXXXXXXX suffix
    parts = raw_name.rsplit('_', 1)
    if len(parts) == 2 and parts[1].isdigit():
        name = parts[0]
    else:
        name = raw_name
    return DISPLAY_NAMES.get(name, name)


# Validated against GameAssembly.dll with this fingerprint. Offsets in this file
# were derived from Il2CppDumper output of a matching DLL; any mismatch means
# the game updated and offsets are likely stale.
KNOWN_DLL_FINGERPRINT: dict = {"size": 44834304, "pe_timestamp": 1777936964}


def validate_dll_version(reader: 'MemoryReader'):
    """Check GameAssembly.dll fingerprint against the version these offsets target.

    Prints a loud warning if the DLL has changed (offsets almost certainly stale).
    """
    fp = reader.get_dll_fingerprint()
    if fp is None:
        print("WARNING: Could not read GameAssembly.dll fingerprint")
        return
    if fp['size'] != KNOWN_DLL_FINGERPRINT['size'] or fp['pe_timestamp'] != KNOWN_DLL_FINGERPRINT['pe_timestamp']:
        print("!!! WARNING: GameAssembly.dll changed from validated version! Offsets may be stale. !!!")
        print(f"  Validated: size={KNOWN_DLL_FINGERPRINT['size']}, timestamp={KNOWN_DLL_FINGERPRINT['pe_timestamp']}")
        print(f"  Current:   size={fp['size']}, timestamp={fp['pe_timestamp']}")
        print("  Re-run Il2CppDumper on GameAssembly.dll and update offsets (Gameplay_TypeInfo + Character fields).")


class MemoryReader:
    def __init__(self):
        self.handle = None
        self.ga_base = None
        self._ga_module_handle = None

    def open(self, pid=None):
        """Open the game process."""
        if pid is None:
            pid = self._find_pid()
        if pid is None:
            print("ERROR: Demon Bluff.exe not found. Is the game running?")
            return False

        self.handle = kernel32.OpenProcess(
            PROCESS_VM_READ | PROCESS_QUERY_INFORMATION, False, pid
        )
        if not self.handle:
            print(f"ERROR: Could not open process {pid}: {ctypes.get_last_error()}")
            return False

        self.ga_base = self._find_game_assembly()
        if not self.ga_base:
            print("ERROR: GameAssembly.dll not found in process")
            return False
        validate_dll_version(self)
        return True

    def close(self):
        if self.handle:
            kernel32.CloseHandle(self.handle)
            self.handle = None

    def _find_pid(self):
        """Find Demon Bluff process ID."""
        import subprocess
        result = subprocess.run(
            ['tasklist', '/FI', 'IMAGENAME eq Demon Bluff.exe', '/FO', 'CSV', '/NH'],
            capture_output=True, text=True
        )
        for line in result.stdout.strip().split('\n'):
            if 'Demon Bluff' in line:
                parts = line.strip('"').split('","')
                if len(parts) >= 2:
                    return int(parts[1])
        return None

    def _find_game_assembly(self):
        """Find GameAssembly.dll base address."""
        hMods = (ctypes.c_uint64 * 1024)()
        cbNeeded = wintypes.DWORD()
        psapi.EnumProcessModulesEx(
            self.handle, hMods, ctypes.sizeof(hMods),
            ctypes.byref(cbNeeded), 3
        )
        n_mods = cbNeeded.value // 8
        for i in range(n_mods):
            mod = hMods[i]
            name_buf = ctypes.create_unicode_buffer(260)
            psapi.GetModuleBaseNameW(
                self.handle, ctypes.c_void_p(mod), name_buf, 260
            )
            if 'GameAssembly' in name_buf.value:
                self._ga_module_handle = ctypes.c_void_p(mod)
                return mod
        return None

    def get_dll_fingerprint(self) -> dict | None:
        """Get GameAssembly.dll version fingerprint (path, file size, PE timestamp)."""
        if not self.handle or not self._ga_module_handle:
            return None
        # Get DLL file path via GetModuleFileNameExW
        path_buf = ctypes.create_unicode_buffer(512)
        psapi.GetModuleFileNameExW(
            self.handle, self._ga_module_handle, path_buf, 512
        )
        dll_path = path_buf.value
        if not dll_path:
            return None
        # Get file size from disk
        try:
            file_size = os.path.getsize(dll_path)
        except OSError:
            file_size = -1
        # Read PE timestamp from process memory
        # IMAGE_DOS_HEADER.e_lfanew at offset 0x3C from DLL base
        e_lfanew = self._read_i32(self.ga_base + 0x3C)
        pe_timestamp = 0
        if e_lfanew and e_lfanew > 0:
            # TimeDateStamp is at e_lfanew + 8 (past PE signature 4 bytes + Machine 2 + NumberOfSections 2)
            pe_timestamp = self._read_i32(self.ga_base + e_lfanew + 8)
            if pe_timestamp is None:
                pe_timestamp = 0
        return {"path": dll_path, "size": file_size, "pe_timestamp": pe_timestamp}

    def _read_ptr(self, addr):
        buf = ctypes.create_string_buffer(8)
        br = ctypes.c_size_t()
        if kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(addr), buf, 8, ctypes.byref(br)
        ):
            return struct.unpack('<Q', buf.raw[:8])[0]
        return None

    def _read_i32(self, addr):
        buf = ctypes.create_string_buffer(4)
        br = ctypes.c_size_t()
        if kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(addr), buf, 4, ctypes.byref(br)
        ):
            return struct.unpack('<i', buf.raw[:4])[0]
        return None

    def _read_bool(self, addr):
        buf = ctypes.create_string_buffer(1)
        br = ctypes.c_size_t()
        if kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(addr), buf, 1, ctypes.byref(br)
        ):
            return buf.raw[0] != 0
        return None

    def _read_native_name(self, cd_ptr):
        """Read the current role name from a CharacterData Unity object.

        The managed characterId field (0x18) is stale in multi-village.
        The native ScriptableObject stores the correct name at m_CachedPtr+0x48.
        """
        if not cd_ptr or cd_ptr < 0x10000:
            return None
        cached_ptr = self._read_ptr(cd_ptr + CD_CACHED_PTR_OFFSET)
        if not cached_ptr or cached_ptr < 0x10000:
            return None
        buf = ctypes.create_string_buffer(64)
        br = ctypes.c_size_t()
        if kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(cached_ptr + CD_NATIVE_NAME_OFFSET),
            buf, 64, ctypes.byref(br)
        ):
            end = buf.raw[:64].find(b'\x00')
            if end > 0:
                try:
                    return buf.raw[:end].decode('ascii')
                except (UnicodeDecodeError, ValueError):
                    pass
        return None

    def _read_cd_name(self, cd_ptr):
        """Read role name from CharacterData, preferring native name over characterId."""
        native = self._read_native_name(cd_ptr)
        if native:
            return clean_name(native)
        # Fallback to managed characterId
        name_ptr = self._read_ptr(cd_ptr + CD_CHARACTER_ID_OFFSET)
        raw = self._read_string(name_ptr)
        return clean_name(raw)

    def _read_string(self, str_ptr):
        if not str_ptr or str_ptr < 0x10000:
            return None
        length = self._read_i32(str_ptr + 0x10)
        if length is None or length <= 0 or length > 200:
            return None
        buf = ctypes.create_string_buffer(length * 2)
        br = ctypes.c_size_t()
        if kernel32.ReadProcessMemory(
            self.handle, ctypes.c_void_p(str_ptr + 0x14),
            buf, length * 2, ctypes.byref(br)
        ):
            return buf.raw[:length * 2].decode('utf-16-le', errors='replace')
        return None

    def _read_c_string(self, char_ptr, max_length=128):
        """Read a bounded native UTF-8/ASCII C string."""
        if not char_ptr or char_ptr < 0x10000:
            return None
        buf = ctypes.create_string_buffer(max_length)
        br = ctypes.c_size_t()
        if not kernel32.ReadProcessMemory(
            self.handle,
            ctypes.c_void_p(char_ptr),
            buf,
            max_length,
            ctypes.byref(br),
        ):
            return None
        raw = buf.raw[:br.value or max_length]
        end = raw.find(b'\x00')
        if end < 0:
            return None
        try:
            return raw[:end].decode('utf-8')
        except (UnicodeDecodeError, ValueError):
            return None

    def _read_object_class_name(self, object_ptr):
        """Read an IL2CPP object's exact managed class name.

        ``Il2CppObject.klass`` is the first pointer in every managed object;
        the class metadata's native ``name`` pointer is at +0x10 in this build.
        """
        if not object_ptr or object_ptr < 0x10000:
            return None
        klass_ptr = self._read_ptr(object_ptr)
        if not klass_ptr or klass_ptr < 0x10000:
            return None
        name_ptr = self._read_ptr(klass_ptr + IL2CPP_CLASS_NAME_OFFSET)
        return self._read_c_string(name_ptr)

    def _get_static_fields(self):
        """Get the Gameplay class static fields pointer."""
        il2cpp_class = self._read_ptr(self.ga_base + GAMEPLAY_TYPEINFO_RVA)
        if not il2cpp_class:
            return None
        return self._read_ptr(il2cpp_class + IL2CPP_CLASS_STATIC_FIELDS_OFFSET)

    def _get_gameplay_instance(self):
        """Follow the pointer chain to get Gameplay.Instance."""
        static_fields = self._get_static_fields()
        if not static_fields:
            return None
        return self._read_ptr(static_fields + GAMEPLAY_INSTANCE_STATIC_OFFSET)

    def read_score(self):
        """Read the Score object from Gameplay's static fields."""
        static_fields = self._get_static_fields()
        if not static_fields:
            return None
        score_ptr = self._read_ptr(static_fields + GAMEPLAY_SCORE_STATIC_OFFSET)
        if not score_ptr or score_ptr < 0x10000:
            return None
        return {
            'killedGoods': self._read_i32(score_ptr + SCORE_KILLED_GOODS_OFFSET),
            'completedStages': self._read_i32(score_ptr + SCORE_COMPLETED_STAGES_OFFSET),
            'tempUnrevealedCards': self._read_i32(score_ptr + SCORE_TEMP_UNREVEALED_OFFSET),
            'unrevealedCards': self._read_i32(score_ptr + SCORE_UNREVEALED_CARDS_OFFSET),
            'killedEvils': self._read_i32(score_ptr + SCORE_KILLED_EVILS_OFFSET),
            'tempKilledEvils': self._read_i32(score_ptr + SCORE_TEMP_KILLED_EVILS_OFFSET),
            'pointPerKill': self._read_i32(score_ptr + SCORE_POINT_PER_KILL_OFFSET),
            'pointsPerUnrevealed': self._read_i32(score_ptr + SCORE_POINTS_PER_UNREVEALED_OFFSET),
            'pointsForCompleting': self._read_i32(score_ptr + SCORE_POINTS_FOR_COMPLETING_OFFSET),
            'completedDays': self._read_i32(score_ptr + SCORE_COMPLETED_DAYS_OFFSET),
            'multiplier': self._read_i32(score_ptr + SCORE_MULTIPLIER_OFFSET),
        }

    def _read_statuses(self, char_ptr):
        """Read the CharacterStatuses list from a Character."""
        statuses_obj = self._read_ptr(char_ptr + CHAR_STATUSES_OFFSET)
        if not statuses_obj or statuses_obj < 0x10000:
            return []
        status_list = self._read_ptr(statuses_obj + CSTATUS_LIST_OFFSET)
        if not status_list or status_list < 0x10000:
            return []
        items_array = self._read_ptr(status_list + LIST_ITEMS_OFFSET)
        list_size = self._read_i32(status_list + LIST_SIZE_OFFSET)
        if not items_array or not list_size or list_size <= 0:
            return []
        result = []
        for i in range(list_size):
            val = self._read_i32(items_array + ARRAY_FIRST_ELEMENT_OFFSET + i * 4)
            if val is not None:
                result.append(CHAR_STATUS.get(val, f'?{val}'))
        return result

    def _read_acted_infos(self, char_ptr):
        """Read List<ActedInfo> — clue history for a character."""
        list_ptr = self._read_ptr(char_ptr + CHAR_ACTED_INFOS_OFFSET)
        if not list_ptr or list_ptr < 0x10000:
            return []
        items_array = self._read_ptr(list_ptr + LIST_ITEMS_OFFSET)
        list_size = self._read_i32(list_ptr + LIST_SIZE_OFFSET)
        if not items_array or not list_size or list_size <= 0:
            return []
        # A real game cannot approach this ceiling. Reject the whole list if a
        # stale offset/pointer produces a runaway size; never truncate a valid
        # chronological history because savedAct corresponds to its last item.
        if list_size > 4096:
            return []
        results = []
        # Native actedInfos is append-only and ResetAfterNight roles can exceed
        # ten uses. Preserve the entire chronological List order so index -1
        # remains the event that produced savedAct.
        for i in range(list_size):
            info_ptr = self._read_ptr(items_array + ARRAY_FIRST_ELEMENT_OFFSET + i * 8)
            if not info_ptr or info_ptr < 0x10000:
                # List<T> slots below _size are populated. Silently skipping a
                # bad pointer would splice the chronology and could preserve a
                # plausible newest savedAct while dropping earlier evidence.
                return []
            # Read desc string
            desc_ptr = self._read_ptr(info_ptr + ACTED_INFO_DESC_OFFSET)
            desc = self._read_string(desc_ptr) if desc_ptr else None
            # Read referenced character positions
            char_list_ptr = self._read_ptr(info_ptr + ACTED_INFO_CHARS_OFFSET)
            targets = []
            if char_list_ptr and char_list_ptr > 0x10000:
                ref_items = self._read_ptr(char_list_ptr + LIST_ITEMS_OFFSET)
                ref_size = self._read_i32(char_list_ptr + LIST_SIZE_OFFSET)
                if ref_items and ref_size and 0 < ref_size <= 20:
                    for j in range(ref_size):
                        ref_char = self._read_ptr(ref_items + ARRAY_FIRST_ELEMENT_OFFSET + j * 8)
                        if ref_char:
                            ref_id = self._read_i32(ref_char + CHAR_ID_OFFSET)
                            if ref_id is not None:
                                targets.append(ref_id)
            results.append({'desc': desc, 'targets': targets})
        return results

    def _read_saved_act(self, char_ptr):
        """Read the cached clue text string (savedAct at 0x198)."""
        str_ptr = self._read_ptr(char_ptr + CHAR_SAVED_ACT_OFFSET)
        if not str_ptr or str_ptr < 0x10000:
            return None
        return self._read_string(str_ptr)

    def _read_runtime_data(self, char_ptr, role_name=None):
        """Read RuntimeCharacterData using its exact IL2CPP object class.

        Shaman/other transforms can preserve a destination's runtime object
        while changing its true/displayed CharacterData. Dispatching from a
        role name would then reinterpret an int as a string pointer (or vice
        versa), so class metadata is the only safe layout discriminator.
        ``role_name`` remains an ignored compatibility argument for callers.
        """
        rd_ptr = self._read_ptr(char_ptr + CHAR_RUNTIME_DATA_OFFSET)
        if not rd_ptr or rd_ptr < 0x10000:
            return None
        runtime_class = self._read_object_class_name(rd_ptr)
        if runtime_class == 'EnlightenedRuntimeData':
            val = self._read_i32(rd_ptr + 0x10)
            return {'type': 'direction', 'direction': EVIL_DIRECTION.get(val, f'?{val}')}
        elif runtime_class == 'AlchemistRuntimeData':
            # Post-patch: this field is "# Corrupted in range at start of round (before cure)",
            # not "# cured" as before. Offset assumed unchanged — verify in-game on first use.
            val = self._read_i32(rd_ptr + 0x10)
            return {'type': 'corrupted_around', 'corrupted_around': val}
        elif runtime_class == 'BakerRuntimeData':
            name_ptr = self._read_ptr(rd_ptr + 0x10)
            name = self._read_string(name_ptr) if name_ptr else None
            return {'type': 'baker', 'original_role': clean_name(name) if name else None}
        return None

    def _read_ability_state(self, char_ptr):
        """Read ability usage state: uses count and act flag."""
        uses = self._read_i32(char_ptr + CHAR_USES_OFFSET)
        act = self._read_bool(char_ptr + CHAR_ACT_OFFSET)
        return {'uses': uses or 0, 'act': act or False}

    def read_board(self):
        """Read all cards on the current board."""
        gameplay = self._get_gameplay_instance()
        if not gameplay:
            return None

        characters_mgr = self._read_ptr(gameplay + GAMEPLAY_CHARACTERS_OFFSET)
        if not characters_mgr:
            return None

        char_list = self._read_ptr(characters_mgr + CHARACTERS_LIST_OFFSET)
        if not char_list:
            return None

        items_array = self._read_ptr(char_list + LIST_ITEMS_OFFSET)
        list_size = self._read_i32(char_list + LIST_SIZE_OFFSET)
        if not items_array or not list_size or list_size <= 0:
            return None

        cards = []
        for i in range(list_size):
            char_ptr = self._read_ptr(items_array + ARRAY_FIRST_ELEMENT_OFFSET + i * 8)
            if not char_ptr:
                continue

            # Read current CharacterData (prefer its native object name for
            # multi-village correctness). ``true_role`` remains a compatibility
            # alias for older callers; it does not mean stable physical origin.
            data_ref = self._read_ptr(char_ptr + CHAR_DATAREF_OFFSET)
            current_role = self._read_cd_name(data_ref) if data_ref else '?'

            # Read bluff/disguise
            bluff_ptr = self._read_ptr(char_ptr + CHAR_BLUFF_OFFSET)
            disguise = None
            if bluff_ptr and bluff_ptr > 0x10000:
                disguise = self._read_cd_name(bluff_ptr)

            # Read other fields
            alignment = self._read_i32(char_ptr + CHAR_ALIGNMENT_OFFSET)
            state = self._read_i32(char_ptr + CHAR_STATE_OFFSET)
            card_id = self._read_i32(char_ptr + CHAR_ID_OFFSET)
            killed_hidden = self._read_bool(char_ptr + CHAR_KILLED_HIDDEN_OFFSET)
            revealed = self._read_bool(char_ptr + CHAR_REVEALED_OFFSET)

            # Read statuses
            statuses = self._read_statuses(char_ptr)

            # Read clue/ability data
            saved_act = self._read_saved_act(char_ptr)
            acted_infos = self._read_acted_infos(char_ptr)
            ability_state = self._read_ability_state(char_ptr)
            runtime_data = self._read_runtime_data(char_ptr)

            # Stale clue filter: for displayed roles with no passive speech bubble,
            # savedAct is always stale (persists from previous village). Null it
            # unless the active ability has been used (uses>0, acted_infos
            # non-empty, or the act flag is set). Plague Doctor is stricter
            # because game-start setup can set the act flag before the active
            # check ability is used.
            display_role_lower = (disguise or current_role or '').lower().replace('_', ' ')
            is_active_unused = (
                ability_state['uses'] == 0
                and not acted_infos
                and not ability_state['act']
            )
            if display_role_lower == 'plague doctor':
                is_active_unused = ability_state['uses'] == 0 and not acted_infos
            if display_role_lower in NO_PASSIVE_CLUE_DISPLAY_ROLES and is_active_unused:
                saved_act = None

            cards.append({
                'position': card_id,
                'current_role': current_role,
                'true_role': current_role,
                'disguise': disguise,
                'alignment': ALIGNMENT.get(alignment, f'?{alignment}'),
                'is_evil': alignment == 20,
                'state': STATE.get(state, f'?{state}'),
                'killed_hidden': killed_hidden,
                'revealed': revealed,
                'statuses': statuses,
                'clue_text': saved_act,
                'acted_infos': acted_infos,
                'ability_used': ability_state['act'],
                'uses': ability_state['uses'],
                'runtime_data': runtime_data,
            })

        cards.sort(key=lambda c: c['position'])
        return cards

    def _read_character_data_list(self, list_ptr):
        """Read a List<CharacterData> and return role names."""
        if not list_ptr or list_ptr < 0x10000:
            return []
        items_array = self._read_ptr(list_ptr + LIST_ITEMS_OFFSET)
        list_size = self._read_i32(list_ptr + LIST_SIZE_OFFSET)
        if not items_array or not list_size or list_size <= 0:
            return []
        names = []
        for i in range(list_size):
            cd_ptr = self._read_ptr(items_array + ARRAY_FIRST_ELEMENT_OFFSET + i * 8)
            if not cd_ptr:
                continue
            names.append(self._read_cd_name(cd_ptr))
        return names

    def read_deck(self):
        """Read the current deck (pool of possible roles)."""
        gameplay = self._get_gameplay_instance()
        if not gameplay:
            return None
        deck = {}
        for faction, offset in [
            ('Villager', GAMEPLAY_TOWNSFOLKS_OFFSET),
            ('Outcast', GAMEPLAY_OUTSIDERS_OFFSET),
            ('Minion', GAMEPLAY_MINIONS_OFFSET),
            ('Demon', GAMEPLAY_DEMONS_OFFSET),
        ]:
            list_ptr = self._read_ptr(gameplay + offset)
            deck[faction] = self._read_character_data_list(list_ptr)
        return deck


class MemoryMonitor(threading.Thread):
    """Background thread that polls game memory and fires callbacks on state changes.

    Design decisions:
    - Owns a PRIVATE MemoryReader instance (never shared with main thread)
    - Copy-on-write: builds board OUTSIDE lock, swaps reference UNDER lock
    - Fires callbacks OUTSIDE lock to prevent deadlock
    - Each callback wrapped in try/except to prevent thread death
    - Error propagation: _error field checked by get_board()/wait_for()
    - pyautogui must ONLY be called from the main thread, never from callbacks

    Events:
        card_flipped: (position, old_state, new_state) — Hidden->Alive
        card_died: (position, old_state, new_state) — Alive->Dead or killed_hidden
        status_changed: (position, old_statuses, new_statuses)
        game_connected: (board) — read_board() returns non-None after None
        game_disconnected: () — read_board() returns None after non-None
    """

    def __init__(self, poll_interval=0.2, pid=None):
        super().__init__(daemon=True)
        self._poll_interval = poll_interval
        self._pid = pid
        self._callbacks: dict[str, list] = {}
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._board: list[dict] | None = None
        self._last_board: list[dict] | None = None
        self._error: Exception | None = None
        self._reader: MemoryReader | None = None
        self._consecutive_failures = 0
        self._MAX_FAILURES_BEFORE_RECONNECT = 3

        # Guard against free-threaded Python
        if hasattr(sys, '_is_gil_enabled') and not sys._is_gil_enabled():
            raise RuntimeError(
                "Free-threaded Python detected. MemoryMonitor requires GIL-enabled CPython."
            )

    def register(self, event: str, callback):
        """Register a callback for an event. Thread-safe."""
        if event not in self._callbacks:
            self._callbacks[event] = []
        self._callbacks[event].append(callback)

    def unregister(self, event: str, callback):
        """Remove a callback. Thread-safe."""
        if event in self._callbacks:
            self._callbacks[event] = [cb for cb in self._callbacks[event] if cb is not callback]

    def get_board(self) -> list[dict] | None:
        """Get the latest board snapshot. Thread-safe (microsecond lock)."""
        if self._error:
            raise RuntimeError(f"MemoryMonitor died: {self._error}") from self._error
        with self._lock:
            return self._board

    def wait_for(self, predicate, timeout: float, min_delay: float = 0) -> bool:
        """Block until predicate(board) is True, or timeout.

        Args:
            predicate: callable(list[dict] | None) -> bool
            timeout: max seconds to wait
            min_delay: minimum seconds before first check (for animation startup)

        Returns True if predicate was satisfied, False on timeout/shutdown.
        """
        if self._error:
            raise RuntimeError(f"MemoryMonitor died: {self._error}") from self._error

        if min_delay > 0:
            if self._stop_event.wait(min_delay):
                return False  # shutdown requested

        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            board = self.get_board()
            try:
                if predicate(board):
                    return True
            except Exception:
                pass  # predicate may fail on None board
            if self._stop_event.wait(0.1):
                return False  # shutdown requested
        return False

    def is_healthy(self) -> bool:
        """Check if the monitor thread is alive and error-free."""
        return self.is_alive() and self._error is None

    def stop(self):
        """Signal the monitor to stop. Non-blocking."""
        self._stop_event.set()
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass

    def run(self):
        """Main polling loop (runs in background thread)."""
        try:
            self._reader = MemoryReader()
            if not self._reader.open(self._pid):
                self._error = RuntimeError("Failed to open game process")
                return

            while not self._stop_event.is_set():
                try:
                    new_board = self._reader.read_board()
                except Exception as e:
                    new_board = None
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._MAX_FAILURES_BEFORE_RECONNECT:
                        self._reconnect()
                    self._stop_event.wait(self._poll_interval)
                    continue

                if new_board is None:
                    self._consecutive_failures += 1
                    if self._consecutive_failures >= self._MAX_FAILURES_BEFORE_RECONNECT:
                        self._reconnect()
                else:
                    self._consecutive_failures = 0

                # Compute diffs BEFORE taking lock
                events_to_fire = self._compute_diffs(self._last_board, new_board)

                # Swap board reference under lock (microseconds)
                with self._lock:
                    self._board = new_board

                self._last_board = new_board

                # Fire callbacks OUTSIDE lock
                for event_name, event_data in events_to_fire:
                    self._fire_callbacks(event_name, event_data)

                self._stop_event.wait(self._poll_interval)

        except Exception as e:
            self._error = e
        finally:
            if self._reader:
                try:
                    self._reader.close()
                except Exception:
                    pass

    def _reconnect(self):
        """Close and reopen the reader to recover from stale handles."""
        self._consecutive_failures = 0
        if self._reader:
            try:
                self._reader.close()
            except Exception:
                pass
        self._reader = MemoryReader()
        if not self._reader.open(self._pid):
            # Can't reconnect — will retry next poll cycle
            pass

    def _compute_diffs(self, old_board, new_board):
        """Compare old and new board states, return list of (event_name, event_data)."""
        events = []

        # game_connected / game_disconnected
        if old_board is None and new_board is not None:
            events.append(('game_connected', {'board': new_board}))
            return events  # don't diff further on first connect
        if old_board is not None and new_board is None:
            events.append(('game_disconnected', {}))
            return events
        if old_board is None or new_board is None:
            return events

        old_by_pos = {c['position']: c for c in old_board}
        new_by_pos = {c['position']: c for c in new_board}

        for pos, new_card in new_by_pos.items():
            old_card = old_by_pos.get(pos)
            if old_card is None:
                continue

            old_state = old_card['state']
            new_state = new_card['state']

            # card_flipped: Hidden -> Alive
            if old_state == 'Hidden' and new_state == 'Alive':
                events.append(('card_flipped', {
                    'position': pos, 'old_state': old_state, 'new_state': new_state
                }))

            # card_died: -> Dead, or killed_hidden changed
            if old_state != 'Dead' and new_state == 'Dead':
                events.append(('card_died', {
                    'position': pos, 'old_state': old_state, 'new_state': new_state
                }))
            elif not old_card.get('killed_hidden') and new_card.get('killed_hidden'):
                events.append(('card_died', {
                    'position': pos, 'old_state': old_state, 'new_state': 'killed_hidden'
                }))

            # status_changed
            old_statuses = frozenset(old_card.get('statuses', []))
            new_statuses = frozenset(new_card.get('statuses', []))
            if old_statuses != new_statuses:
                events.append(('status_changed', {
                    'position': pos,
                    'old_statuses': list(old_statuses),
                    'new_statuses': list(new_statuses),
                }))

        return events

    def _fire_callbacks(self, event_name, event_data):
        """Fire all registered callbacks for an event. Each wrapped in try/except."""
        for cb in self._callbacks.get(event_name, []):
            try:
                cb(event_data)
            except Exception as e:
                print(f"  [MemoryMonitor] callback error on {event_name}: {e}")


# Global monitor instance (started once per REPL session)
_monitor: MemoryMonitor | None = None


def get_monitor(poll_interval=0.2, pid=None) -> MemoryMonitor:
    """Get or create the global MemoryMonitor singleton."""
    global _monitor
    if _monitor is not None and _monitor.is_healthy():
        return _monitor
    if _monitor is not None:
        _monitor.stop()
    _monitor = MemoryMonitor(poll_interval=poll_interval, pid=pid)
    _monitor.start()
    atexit.register(_shutdown_monitor)
    return _monitor


def _shutdown_monitor():
    global _monitor
    if _monitor is not None:
        _monitor.stop()
        _monitor = None


def shutdown_monitor():
    """Public alias for _shutdown_monitor. Stops the monitor and clears the global."""
    _shutdown_monitor()


def restart_monitor(poll_interval=0.2, pid=None) -> MemoryMonitor:
    """Shut down the existing monitor and create a fresh one. Use between games for isolation."""
    shutdown_monitor()
    return get_monitor(poll_interval=poll_interval, pid=pid)


def print_deck(deck):
    """Pretty-print the current deck pool."""
    if not deck:
        print("No deck found.")
        return
    print()
    total = 0
    for faction in ['Villager', 'Outcast', 'Minion', 'Demon']:
        roles = deck.get(faction, [])
        total += len(roles)
        print(f"  {faction}s ({len(roles)}): {', '.join(roles)}")
    print(f"  Total: {total}")


def print_board(cards):
    """Pretty-print the board state."""
    if not cards:
        print("No cards found.")
        return

    print()
    print(f"{'POS':>3}  {'CURRENT ROLE':<18} {'SHOWS AS':<18} {'ALIGN':>5}  {'STATE':<8} {'FLIP':>4}  FLAGS")
    print("-" * 85)
    for c in cards:
        current_role = c.get('current_role') or c.get('true_role', '?')
        shows_as = c['disguise'] if c['disguise'] else current_role
        evil_marker = '*' if c['is_evil'] else ' '
        flipped = 'YES' if c.get('state') in ('Alive', 'Dead', 'Revealed') else 'NO'
        flags = []
        if c['killed_hidden']:
            flags.append('LILIS-KILL')
        statuses = c.get('statuses', [])
        for s in statuses:
            if s != 'None':
                flags.append(s)
        if c.get('state') == 'Hidden' and not c['killed_hidden']:
            flags.append('BLOCKED')
        flags_str = ', '.join(flags) if flags else ''
        print(
            f"#{c['position']:>2}{evil_marker} {current_role:<18} "
            f"{shows_as:<18} {c['alignment']:>5}  {c['state']:<8} {flipped:>4}  {flags_str}"
        )

    evils = [c for c in cards if c['is_evil']]
    print()
    print(f"Evil cards ({len(evils)}; CURRENT ROLE shown):")
    for c in evils:
        current_role = c.get('current_role') or c.get('true_role', '?')
        disguise_info = f" (disguised as {c['disguise']})" if c['disguise'] else ""
        print(f"  #{c['position']} CURRENT ROLE={current_role}{disguise_info}")

    # Show clue data for flipped cards
    clue_cards = [c for c in cards if c.get('clue_text') or c.get('acted_infos') or c.get('runtime_data')]
    if clue_cards:
        print()
        print("Clue data:")
        for c in clue_cards:
            current_role = c.get('current_role') or c.get('true_role', '?')
            parts = [f"  #{c['position']} CURRENT ROLE={current_role}:"]
            if c.get('clue_text'):
                parts.append(f"clue=\"{c['clue_text']}\"")
            if c.get('runtime_data'):
                rd = c['runtime_data']
                if rd.get('type') == 'direction':
                    parts.append(f"direction={rd['direction']}")
                elif rd.get('type') == 'corrupted_around':
                    parts.append(f"corrupted_around={rd['corrupted_around']}")
                elif rd.get('type') == 'cures':
                    parts.append(f"cures={rd['cures']}")
                elif rd.get('type') == 'baker':
                    parts.append(f"original={rd.get('original_role')}")
            if c.get('acted_infos'):
                for info in c['acted_infos']:
                    targets = info.get('targets', [])
                    if targets:
                        parts.append(f"targets={targets}")
            print(' '.join(parts))


def print_score(score):
    """Pretty-print the Score object."""
    if not score:
        print("No score found.")
        return
    print()
    print("=== SCORE CONFIG ===")
    print(f"  pointPerKill:         {score['pointPerKill']}")
    print(f"  pointsPerUnrevealed:  {score['pointsPerUnrevealed']}")
    print(f"  pointsForCompleting:  {score['pointsForCompleting']}")
    print(f"  multiplier:           {score['multiplier']}")
    print()
    print("=== CURRENT RUN ===")
    print(f"  completedStages:      {score['completedStages']}")
    print(f"  completedDays:        {score['completedDays']}")
    print(f"  killedEvils (total):  {score['killedEvils']}")
    print(f"  killedGoods (total):  {score['killedGoods']}")
    print(f"  unrevealedCards:      {score['unrevealedCards']}")
    print()
    print("=== THIS VILLAGE ===")
    print(f"  tempKilledEvils:      {score['tempKilledEvils']}")
    print(f"  tempUnrevealedCards:  {score['tempUnrevealedCards']}")
    print()
    # Estimated formula
    kill_pts = (score['killedEvils'] or 0) * (score['pointPerKill'] or 0)
    unrevealed_pts = (score['unrevealedCards'] or 0) * (score['pointsPerUnrevealed'] or 0)
    complete_pts = (score['completedDays'] or 0) * (score['pointsForCompleting'] or 0)
    mult = score['multiplier'] or 1
    estimated = (kill_pts + unrevealed_pts + complete_pts) * mult
    print(f"=== ESTIMATED TOTAL ===")
    print(f"  kills:      {score['killedEvils']} x {score['pointPerKill']} = {kill_pts}")
    print(f"  unrevealed: {score['unrevealedCards']} x {score['pointsPerUnrevealed']} = {unrevealed_pts}")
    print(f"  completed:  {score['completedDays']} x {score['pointsForCompleting']} = {complete_pts}")
    print(f"  subtotal:   {kill_pts + unrevealed_pts + complete_pts}")
    print(f"  multiplier: x{mult}")
    print(f"  estimated:  {estimated}")


def main():
    parser = argparse.ArgumentParser(description='Demon Bluff Memory Reader')
    parser.add_argument('--watch', action='store_true', help='Watch for changes')
    parser.add_argument('--deck', action='store_true', help='Show current deck pool')
    parser.add_argument('--score', action='store_true', help='Show score details')
    parser.add_argument('--interval', type=float, default=2.0, help='Watch interval')
    parser.add_argument('--pid', type=int, help='Process ID override')
    args = parser.parse_args()

    reader = MemoryReader()
    if not reader.open(args.pid):
        sys.exit(1)

    try:
        if args.score:
            score = reader.read_score()
            print_score(score)
        elif args.deck:
            deck = reader.read_deck()
            print_deck(deck)
        elif args.watch:
            prev = None
            while True:
                cards = reader.read_board()
                card_str = str([(c['position'], c['state']) for c in cards]) if cards else None
                if card_str != prev:
                    print(f"\n--- {time.strftime('%H:%M:%S')} ---")
                    print_board(cards)
                    prev = card_str
                time.sleep(args.interval)
        else:
            cards = reader.read_board()
            print_board(cards)
    except KeyboardInterrupt:
        pass
    finally:
        reader.close()


if __name__ == '__main__':
    main()
