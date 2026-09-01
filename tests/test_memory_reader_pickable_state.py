import unittest

import memory_reader as memory_module


class MemoryReaderPickableStateTests(unittest.TestCase):
    def test_pickable_state_preserves_one_zero_negative_and_none(self):
        character = 0x10000
        cases = (
            (1, True, True),
            (0, False, False),
            (-1, True, False),
            (None, None, None),
        )

        for uses, act, available in cases:
            with self.subTest(uses=uses, act=act):
                reader = memory_module.MemoryReader()
                reader._read_i32 = lambda address, value=uses: value
                reader._read_bool = lambda address, value=act: value

                state = reader._read_ability_state(character)

                self.assertEqual(state["pickable_uses_remaining"], uses)
                self.assertEqual(state["act_output_enabled"], act)
                self.assertEqual(state["pickable_available"], available)
                self.assertEqual(state["uses"], uses)
                self.assertEqual(state["ability_used"], act)

    def test_read_board_emits_exact_fields_and_preserves_unreadable_values(self):
        reader = self._board_reader(saved_act=None, acted_infos=[])
        reader._read_i32_values[
            reader._character + memory_module.CHAR_USES_OFFSET
        ] = None
        reader._read_bool_values[
            reader._character + memory_module.CHAR_ACT_OFFSET
        ] = None

        card = reader.read_board()[0]

        self.assertIsNone(card["pickable_uses_remaining"])
        self.assertIsNone(card["act_output_enabled"])
        self.assertIsNone(card["pickable_available"])
        self.assertIsNone(card["uses"])
        self.assertIsNone(card["ability_used"])

    def test_active_saved_act_without_history_is_cleared_regardless_of_state(self):
        reader = self._board_reader(
            saved_act="Is #2 Evil?: True",
            acted_infos=[],
            uses=1,
            act=True,
        )

        card = reader.read_board()[0]

        self.assertIsNone(card["clue_text"])

    def test_active_saved_act_must_match_newest_chronological_event(self):
        saved_act = "Is #2 Evil?: True"
        older_match = {"desc": saved_act, "targets": [2]}
        newer_mismatch = {"desc": "Is #3 Evil?: False", "targets": [3]}
        reader = self._board_reader(
            saved_act=saved_act,
            acted_infos=[older_match, newer_mismatch],
            uses=-1,
            act=True,
        )

        card = reader.read_board()[0]

        self.assertIsNone(card["clue_text"])

    def test_active_saved_act_matching_newest_event_survives_night_reset(self):
        saved_act = "Is #3 Evil?: False"
        reader = self._board_reader(
            saved_act=saved_act,
            acted_infos=[
                {"desc": "Is #2 Evil?: True", "targets": [2]},
                {"desc": saved_act, "targets": [3]},
            ],
            uses=1,
            act=True,
        )

        card = reader.read_board()[0]

        self.assertEqual(card["clue_text"], saved_act)
        self.assertTrue(card["pickable_available"])

    def test_unreadable_acted_history_is_not_collapsed_to_empty(self):
        reader = self._board_reader(saved_act=None, acted_infos=[])
        reader._read_acted_infos = (
            memory_module.MemoryReader._read_acted_infos.__get__(
                reader,
                memory_module.MemoryReader,
            )
        )
        acted_list_address = (
            reader._character + memory_module.CHAR_ACTED_INFOS_OFFSET
        )
        original_read_ptr = reader._read_ptr
        reader._read_ptr = lambda address: (
            None if address == acted_list_address else original_read_ptr(address)
        )

        card = reader.read_board()[0]

        self.assertIsNone(card["acted_infos"])
        self.assertIsNone(card["clue_text"])

    @staticmethod
    def _board_reader(
        *,
        saved_act,
        acted_infos,
        uses=0,
        act=False,
        current_role="Judge",
    ):
        reader = memory_module.MemoryReader()
        gameplay = 0x10000
        characters = 0x20000
        character_list = 0x30000
        items = 0x40000
        character = 0x50000
        character_data = 0x60000

        reader._character = character
        reader._read_i32_values = {
            character_list + memory_module.LIST_SIZE_OFFSET: 1,
            character + memory_module.CHAR_ALIGNMENT_OFFSET: 10,
            character + memory_module.CHAR_STATE_OFFSET: 10,
            character + memory_module.CHAR_ID_OFFSET: 1,
            character + memory_module.CHAR_USES_OFFSET: uses,
        }
        reader._read_bool_values = {
            character + memory_module.CHAR_KILLED_HIDDEN_OFFSET: False,
            character + memory_module.CHAR_REVEALED_OFFSET: True,
            character + memory_module.CHAR_ACT_OFFSET: act,
        }
        pointers = {
            gameplay + memory_module.GAMEPLAY_CHARACTERS_OFFSET: characters,
            characters + memory_module.CHARACTERS_LIST_OFFSET: character_list,
            character_list + memory_module.LIST_ITEMS_OFFSET: items,
            items + memory_module.ARRAY_FIRST_ELEMENT_OFFSET: character,
            character + memory_module.CHAR_DATAREF_OFFSET: character_data,
            character + memory_module.CHAR_BLUFF_OFFSET: 0,
        }

        reader._get_gameplay_instance = lambda: gameplay
        reader._read_ptr = lambda address: pointers.get(address, 0)
        reader._read_i32 = lambda address: reader._read_i32_values.get(address)
        reader._read_bool = lambda address: reader._read_bool_values.get(address)
        reader._read_cd_name = lambda pointer: current_role
        reader._read_statuses = lambda pointer: []
        reader._read_saved_act = lambda pointer: saved_act
        reader._read_acted_infos = lambda pointer: list(acted_infos)
        reader._read_runtime_data = lambda pointer: None
        return reader


if __name__ == "__main__":
    unittest.main()
