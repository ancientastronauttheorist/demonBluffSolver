"""
Demon Bluff Memory Reader
Reads live game state directly from process memory.
Extracts true roles, disguises, alignment, and state for all cards.

Usage:
    python memory_reader.py          # dump current board
    python memory_reader.py --watch  # continuously watch for changes
"""

import ctypes
import struct
import sys
import os
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

# IL2CPP offsets (from dump.cs analysis)
# Gameplay class
GAMEPLAY_AWAKE_RVA = 0x34DBF0
GAMEPLAY_STATIC_CLASS_PTR_DISP = 0x0222A57C  # RIP-relative displacement from Awake+37
IL2CPP_CLASS_STATIC_FIELDS_OFFSET = 0xB8
GAMEPLAY_INSTANCE_STATIC_OFFSET = 0x8
GAMEPLAY_CHARACTERS_OFFSET = 0x60

# Gameplay deck lists (List<CharacterData>)
GAMEPLAY_TOWNSFOLKS_OFFSET = 0x20
GAMEPLAY_OUTSIDERS_OFFSET = 0x28
GAMEPLAY_MINIONS_OFFSET = 0x30
GAMEPLAY_DEMONS_OFFSET = 0x38

# Characters class
CHARACTERS_LIST_OFFSET = 0x20

# List<T> internals
LIST_ITEMS_OFFSET = 0x10
LIST_SIZE_OFFSET = 0x18

# IL2CPP array
ARRAY_FIRST_ELEMENT_OFFSET = 0x20

# Character class field offsets
CHAR_DATAREF_OFFSET = 0x50       # CharacterData* (true role)
CHAR_BLUFF_OFFSET = 0x58         # CharacterData* (disguise)
CHAR_REGISTERAS_OFFSET = 0x60    # CharacterData* (registers as)
CHAR_STATE_OFFSET = 0xC4         # ECharacterState (int32)
CHAR_ALIGNMENT_OFFSET = 0xD8     # EAlignment (int32)
CHAR_ID_OFFSET = 0xF8            # int32 (position)
CHAR_KILLED_HIDDEN_OFFSET = 0xCC # bool (killed by Lilis)

# CharacterData class field offsets
CD_CHARACTER_ID_OFFSET = 0x18    # string (role name) -- STALE in multi-village!
CD_CACHED_PTR_OFFSET = 0x10      # IntPtr m_CachedPtr (Unity native object)
CD_NATIVE_NAME_OFFSET = 0x48     # char* name in Unity native object (RELIABLE)
CD_TYPE_OFFSET = 0xF8            # ECharacterType (int32)
CD_ALIGNMENT_OFFSET = 0xFC       # EAlignment (int32)

# Enum mappings
ALIGNMENT = {0: 'None', 10: 'Good', 20: 'Evil'}
STATE = {0: 'None', 5: 'Hidden', 10: 'Alive', 20: 'Dead', 30: 'Revealed'}
CHAR_TYPE = {0: 'None', 10: 'Villager', 20: 'Outcast', 30: 'Minion', 100: 'Demon'}

# Internal name → display name mapping
DISPLAY_NAMES = {
    'Juggler': 'Jester',
    'Lillith': 'Lilis',
    'Striga': 'Lilis',
    'Doppleganger': 'Doppelganger',
    'Mathematician': 'Lover',
    'RangedEmpath': 'Bard',
    'BountyHunter': 'Hunter',
    'Immortal': 'Knight',
    'Recluse': 'Wretch',
    'Skinwalker': 'Baa',
    'Imp': 'Chancellor',
    'Marionette': 'Puppeteer',
    'Illuzionist': 'Witch',
    'Spy': 'Plague Doctor',
    'Mezepheles': 'Twin Minion',
    'Saint': 'Bombardier',
    'SaintVillager': 'Bombardier',
    'Athlete': 'Bard',
    'Acrobat': 'Acrobat',
    'Arbiter': 'Judge',
    'Noble': 'Noble',
    'Gossip': 'Gossip',
    'Gambler': 'Gemcrafter',
    'Lookout': 'Medium',
    'Sapper': 'Sapper',
    'Archivist': 'Archivist',
    'Shugenja': 'Shugenja',
    'Tracker': 'Tracker',
    'Investigator': 'Investigator',
    'Librarian': 'Librarian',
    'Mutant': 'Mutant',
    'Puzzlemaster': 'Puzzlemaster',
    'Cipher': 'Cipher',
    'Scout': 'Scout',
    'Knitter': 'Knitter',
    'Witness': 'Witness',
    'Baron': 'Baron',
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


class MemoryReader:
    def __init__(self):
        self.handle = None
        self.ga_base = None

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
                return mod
        return None

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
        """Read the true role name from a CharacterData's Unity native object.

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

    def _get_gameplay_instance(self):
        """Follow the pointer chain to get Gameplay.Instance."""
        awake_addr = self.ga_base + GAMEPLAY_AWAKE_RVA
        # RIP-relative address of Il2CppClass* pointer
        class_ptr_addr = awake_addr + 44 + GAMEPLAY_STATIC_CLASS_PTR_DISP
        il2cpp_class = self._read_ptr(class_ptr_addr)
        if not il2cpp_class:
            return None
        static_fields = self._read_ptr(il2cpp_class + IL2CPP_CLASS_STATIC_FIELDS_OFFSET)
        if not static_fields:
            return None
        return self._read_ptr(static_fields + GAMEPLAY_INSTANCE_STATIC_OFFSET)

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

            # Read true role (prefer native name for multi-village correctness)
            data_ref = self._read_ptr(char_ptr + CHAR_DATAREF_OFFSET)
            true_role = self._read_cd_name(data_ref) if data_ref else '?'

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

            cards.append({
                'position': card_id,
                'true_role': true_role,
                'disguise': disguise,
                'alignment': ALIGNMENT.get(alignment, f'?{alignment}'),
                'is_evil': alignment == 20,
                'state': STATE.get(state, f'?{state}'),
                'killed_hidden': killed_hidden,
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
    print(f"{'POS':>3}  {'TRUE ROLE':<18} {'SHOWS AS':<18} {'ALIGN':>5}  {'STATE':<8}")
    print("-" * 62)
    for c in cards:
        shows_as = c['disguise'] if c['disguise'] else c['true_role']
        evil_marker = '*' if c['is_evil'] else ' '
        skull = 'LILIS-KILL' if c['killed_hidden'] else ''
        print(
            f"#{c['position']:>2}{evil_marker} {c['true_role']:<18} "
            f"{shows_as:<18} {c['alignment']:>5}  {c['state']:<8} {skull}"
        )

    evils = [c for c in cards if c['is_evil']]
    print()
    print(f"Evil cards ({len(evils)}):")
    for c in evils:
        disguise_info = f" (disguised as {c['disguise']})" if c['disguise'] else ""
        print(f"  #{c['position']} {c['true_role']}{disguise_info}")


def main():
    parser = argparse.ArgumentParser(description='Demon Bluff Memory Reader')
    parser.add_argument('--watch', action='store_true', help='Watch for changes')
    parser.add_argument('--deck', action='store_true', help='Show current deck pool')
    parser.add_argument('--interval', type=float, default=2.0, help='Watch interval')
    parser.add_argument('--pid', type=int, help='Process ID override')
    args = parser.parse_args()

    reader = MemoryReader()
    if not reader.open(args.pid):
        sys.exit(1)

    try:
        if args.deck:
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
