from contextlib import redirect_stderr, redirect_stdout
import io
import json
import sys
import unittest
from unittest.mock import patch

import memory_reader
from game_loop import GameSession


class _MemoryMapReader(memory_reader.MemoryReader):
    def __init__(self):
        super().__init__()
        self.gameplay = 0x10000
        self.pointers = {}
        self.ints = {}
        self.names = {}
        self.fingerprint = dict(memory_reader.KNOWN_DLL_FINGERPRINT)

    def _get_gameplay_instance(self):
        return self.gameplay

    def get_dll_fingerprint(self):
        return dict(self.fingerprint)

    def _read_ptr(self, address):
        return self.pointers.get(address)

    def _read_i32(self, address):
        return self.ints.get(address)

    def _read_cd_name(self, pointer):
        return self.names.get(pointer, '?')


def _add_list(reader, list_ptr, items_ptr, occurrences):
    reader.ints[list_ptr + memory_reader.LIST_SIZE_OFFSET] = len(occurrences)
    if not occurrences:
        return
    reader.pointers[list_ptr + memory_reader.LIST_ITEMS_OFFSET] = items_ptr
    for index, pointer in enumerate(occurrences):
        reader.pointers[
            items_ptr
            + memory_reader.ARRAY_FIRST_ELEMENT_OFFSET
            + index * 8
        ] = pointer


def _complete_reader():
    reader = _MemoryMapReader()
    gameplay = reader.gameplay
    characters = 0x20000
    reader.pointers[
        gameplay + memory_reader.GAMEPLAY_CHARACTERS_OFFSET
    ] = characters

    scout = 0x80000
    witness = 0x81000
    confessor = 0x82000
    twin = 0x83000
    pooka = 0x84000
    reader.names.update({
        scout: 'Scout',
        witness: 'Witness',
        confessor: 'Confessor',
        twin: 'Twin Minion',
        pooka: 'Pooka',
    })

    pool_specs = (
        (
            memory_reader.CHARACTERS_UNIQUE_POOL_OFFSET,
            0x30000,
            0x40000,
            [witness, confessor],
        ),
        (
            memory_reader.CHARACTERS_DUPLICATES_POOL_OFFSET,
            0x31000,
            0x41000,
            [scout, scout, confessor],
        ),
        (
            memory_reader.CHARACTERS_BLUFF_MUST_INCLUDE_OFFSET,
            0x32000,
            0x42000,
            [],
        ),
    )
    for offset, list_ptr, items_ptr, occurrences in pool_specs:
        reader.pointers[characters + offset] = list_ptr
        _add_list(reader, list_ptr, items_ptr, occurrences)

    script_specs = (
        (
            memory_reader.GAMEPLAY_TOWNSFOLKS_OFFSET,
            0x33000,
            0x43000,
            [scout, confessor],
        ),
        (
            memory_reader.GAMEPLAY_OUTSIDERS_OFFSET,
            0x34000,
            0x44000,
            [],
        ),
        (
            memory_reader.GAMEPLAY_MINIONS_OFFSET,
            0x35000,
            0x45000,
            [twin],
        ),
        (
            memory_reader.GAMEPLAY_DEMONS_OFFSET,
            0x36000,
            0x46000,
            [pooka],
        ),
    )
    for offset, list_ptr, items_ptr, occurrences in script_specs:
        reader.pointers[gameplay + offset] = list_ptr
        _add_list(reader, list_ptr, items_ptr, occurrences)

    board_list = 0x37000
    board_items = 0x47000
    card_one = 0x70000
    card_two = 0x71000
    reader.pointers[
        characters + memory_reader.CHARACTERS_LIST_OFFSET
    ] = board_list
    _add_list(reader, board_list, board_items, [card_one, card_two])

    reader.pointers.update({
        card_one + memory_reader.CHAR_DATAREF_OFFSET: scout,
        card_one + memory_reader.CHAR_BLUFF_OFFSET: confessor,
        card_one + memory_reader.CHAR_REGISTERAS_OFFSET: 0,
        card_two + memory_reader.CHAR_DATAREF_OFFSET: twin,
        card_two + memory_reader.CHAR_BLUFF_OFFSET: 0,
        card_two + memory_reader.CHAR_REGISTERAS_OFFSET: witness,
    })
    reader.ints.update({
        card_one + memory_reader.CHAR_ID_OFFSET: 1,
        card_one + memory_reader.CHAR_ALIGNMENT_OFFSET: 10,
        card_one + memory_reader.CHAR_STATE_OFFSET: 5,
        card_two + memory_reader.CHAR_ID_OFFSET: 2,
        card_two + memory_reader.CHAR_ALIGNMENT_OFFSET: 20,
        card_two + memory_reader.CHAR_STATE_OFFSET: 10,
    })
    return reader


class PostmortemBluffSnapshotTests(unittest.TestCase):
    def test_strict_character_data_list_preserves_order_and_duplicates(self):
        reader = _MemoryMapReader()
        list_ptr = 0x20000
        items_ptr = 0x30000
        scout = 0x40000
        confessor = 0x41000
        reader.names.update({scout: 'Scout', confessor: 'Confessor'})
        _add_list(
            reader,
            list_ptr,
            items_ptr,
            [scout, scout, confessor, scout],
        )

        self.assertEqual(
            reader._read_character_data_list_strict(list_ptr, 'pool'),
            ['Scout', 'Scout', 'Confessor', 'Scout'],
        )

    def test_valid_empty_list_is_distinct_from_unreadable_and_native_null(self):
        reader = _MemoryMapReader()
        empty_list = 0x20000
        _add_list(reader, empty_list, 0x30000, [])

        self.assertEqual(
            reader._read_character_data_list_strict(empty_list, 'pool'),
            [],
        )
        with self.assertRaisesRegex(
            memory_reader.PostmortemBluffCaptureError,
            'unreadable',
        ):
            reader._read_character_data_list_strict(None, 'pool')
        with self.assertRaisesRegex(
            memory_reader.PostmortemBluffCaptureError,
            'native-null',
        ):
            reader._read_character_data_list_strict(0, 'pool')

    def test_partially_readable_list_returns_no_partial_data(self):
        reader = _MemoryMapReader()
        list_ptr = 0x20000
        items_ptr = 0x30000
        role = 0x40000
        reader.names[role] = 'Scout'
        _add_list(reader, list_ptr, items_ptr, [role, role])
        del reader.pointers[
            items_ptr + memory_reader.ARRAY_FIRST_ELEMENT_OFFSET + 8
        ]

        with self.assertRaisesRegex(
            memory_reader.PostmortemBluffCaptureError,
            r'pool\[1\].*unreadable',
        ):
            reader._read_character_data_list_strict(list_ptr, 'pool')

    def test_snapshot_has_exact_hidden_state_and_explicit_provenance(self):
        reader = _complete_reader()

        with patch.object(memory_reader.time, 'time', return_value=123.456):
            snapshot = reader.read_postmortem_bluff_snapshot()

        self.assertEqual(
            snapshot['schema'],
            memory_reader.POSTMORTEM_BLUFF_SNAPSHOT_SCHEMA,
        )
        self.assertEqual(snapshot['schema_version'], 1)
        self.assertEqual(
            snapshot['provenance'],
            {
                'intended_use': 'postmortem_only',
                'capture_phase': 'caller_asserted_postmortem',
                'source': 'read_only_process_memory',
                'captured_at_unix_ms': 123456,
                'build_fingerprint': dict(
                    memory_reader.KNOWN_DLL_FINGERPRINT
                ),
                'live_solver_input': False,
                'acquisition_order_captured': False,
                'snapshot_consistency': 'sequential_read_not_atomic',
                'pool_phase': 'settled_state_not_verified',
                'must_include_semantics': (
                    'remaining_at_capture_not_acquisition_time'
                ),
            },
        )
        self.assertEqual(
            snapshot['round_pools'],
            {
                'unique_pool': ['Witness', 'Confessor'],
                'duplicate_pool': ['Scout', 'Scout', 'Confessor'],
                'bluff_must_include_remaining_at_capture': [],
            },
        )
        self.assertEqual(
            snapshot['current_script'],
            {
                'villagers': ['Scout', 'Confessor'],
                'outcasts': [],
                'minions': ['Twin Minion'],
                'demons': ['Pooka'],
            },
        )
        self.assertEqual(
            snapshot['board_identity'],
            [
                {
                    'board_index': 0,
                    'position': 1,
                    'current_role': 'Scout',
                    'runtime_alignment': 'Good',
                    'state': 'Hidden',
                    'bluff': 'Confessor',
                    'register_as': None,
                },
                {
                    'board_index': 1,
                    'position': 2,
                    'current_role': 'Twin Minion',
                    'runtime_alignment': 'Evil',
                    'state': 'Alive',
                    'bluff': None,
                    'register_as': 'Witness',
                },
            ],
        )

    def test_snapshot_rejects_a_mismatched_build_fingerprint(self):
        reader = _complete_reader()
        reader.fingerprint['size'] += 1

        with self.assertRaisesRegex(
            memory_reader.PostmortemBluffCaptureError,
            'fingerprint does not match',
        ):
            reader.read_postmortem_bluff_snapshot()

    def test_named_postmortem_cli_prints_only_the_json_snapshot(self):
        snapshot = {
            'schema': memory_reader.POSTMORTEM_BLUFF_SNAPSHOT_SCHEMA,
            'schema_version': 1,
        }
        fake_reader = unittest.mock.Mock()
        fake_reader.open.return_value = True
        fake_reader.read_postmortem_bluff_snapshot.return_value = snapshot
        stdout = io.StringIO()
        stderr = io.StringIO()

        with (
            patch.object(memory_reader, 'MemoryReader', return_value=fake_reader),
            patch.object(
                sys,
                'argv',
                ['memory_reader.py', '--postmortem-bluff-snapshot-json'],
            ),
            redirect_stdout(stdout),
            redirect_stderr(stderr),
        ):
            memory_reader.main()

        self.assertEqual(json.loads(stdout.getvalue()), snapshot)
        self.assertEqual(stderr.getvalue(), '')
        fake_reader.close.assert_called_once_with()

    def test_normal_live_session_state_has_no_postmortem_capture_coupling(self):
        state = GameSession(4, 1).to_game_state().to_dict()

        self.assertNotIn('postmortem_bluff_snapshot', state)
        self.assertNotIn('round_pools', state)
        self.assertNotIn('current_script', state)
        self.assertNotIn('board_identity', state)
        self.assertNotIn('twin_recipient_bluff_context', state)


if __name__ == '__main__':
    unittest.main()
