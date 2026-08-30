"""Focused live-capture regressions for the shipped Baker Day behavior."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import Mock, patch

from game_loop import (
    DecisionLog,
    GameSession,
    _parse_card_cli,
    _parse_clue_from_memory,
    dispatch,
)
from solver import BAKER_RULE_VERSION, CardInfo
from memory_reader import (
    CHAR_RUNTIME_DATA_OFFSET,
    IL2CPP_CLASS_NAME_OFFSET,
    MemoryReader,
)
from state_machine import GamePhase, GameStateMachine


def _memory_baker(
    clue,
    *,
    acted_infos=None,
    runtime_role="Baker",
) -> dict:
    if acted_infos is None:
        acted_infos = [{"desc": clue, "targets": []}]
    runtime_data = (
        None
        if runtime_role is ...
        else {"type": "baker", "original_role": runtime_role}
    )
    return {
        "position": 1,
        "true_role": "Baker",
        "disguise": "Baker",
        "state": "Alive",
        "clue_text": clue,
        "acted_infos": acted_infos,
        "runtime_data": runtime_data,
        "uses": 0,
        "ability_used": False,
    }


class BakerMemoryParsingTests(unittest.TestCase):
    def _parse_current(self, card: dict):
        return _parse_clue_from_memory(
            card,
            n_cards=5,
            baker_rule_version=BAKER_RULE_VERSION,
        )

    def test_exact_original_phrase_is_the_only_original_sentinel(self):
        clue = "I am the original Baker"
        parsed = self._parse_current(_memory_baker(clue))

        self.assertEqual(parsed.info_parsed, {"original_role": "original"})
        self.assertEqual(parsed.info_text, clue)

        for malformed in (
            "I am original Baker",
            "I am the original Baker!",
            "I was original Baker",
            "The original Baker",
            "I was an Baker",
            "I was a Alchemist",
        ):
            with self.subTest(malformed=malformed):
                self.assertIsNone(self._parse_current(_memory_baker(malformed)))

    def test_a_and_an_templates_capture_canonical_roles(self):
        cases = {
            "I was a Baker": "Baker",
            "I was an Alchemist": "Alchemist",
            "I was a Fortune Teller": "Fortune Teller",
            "I was a Twin Minion": "Twin Minion",
        }
        for clue, expected in cases.items():
            with self.subTest(clue=clue):
                parsed = self._parse_current(_memory_baker(clue))
                self.assertEqual(
                    parsed.info_parsed,
                    {"original_role": expected},
                )
                self.assertEqual(parsed.info_text, clue)

    def test_literal_baker_is_never_conflated_with_original(self):
        parsed = self._parse_current(_memory_baker("I was a Baker"))
        self.assertEqual(parsed.info_parsed["original_role"], "Baker")

        legacy_runtime = _parse_clue_from_memory(
            _memory_baker("", acted_infos=[], runtime_role="Baker"),
            n_cards=5,
        )
        self.assertEqual(
            legacy_runtime.info_parsed["original_role"],
            "Baker",
        )

    def test_current_capture_requires_latest_text_and_empty_refs(self):
        malformed = [
            _memory_baker("I was a Baker", acted_infos=[]),
            _memory_baker(
                "I was a Baker",
                acted_infos=[{"desc": "I was a Baker", "targets": []},
                             {"desc": "I was a Bard", "targets": []}],
            ),
            _memory_baker(
                "I was a Baker",
                acted_infos=[{"desc": "I was a Baker", "targets": [2]}],
            ),
            _memory_baker(
                "I was a Baker",
                acted_infos=[{"desc": "I was a Baker"}],
            ),
            _memory_baker(
                "I was a Baker",
                acted_infos=["not an acted-info record"],
            ),
        ]
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(self._parse_current(card))

    def test_current_capture_never_falls_back_to_runtime_data(self):
        for card in (
            _memory_baker("", acted_infos=[], runtime_role="Baker"),
            _memory_baker(
                "pending",
                acted_infos=[{"desc": "pending", "targets": []}],
                runtime_role="Bard",
            ),
            _memory_baker(
                "I was a Baker",
                acted_infos=[{"desc": "stale", "targets": []}],
                runtime_role="Bard",
            ),
        ):
            with self.subTest(card=card):
                self.assertIsNone(self._parse_current(card))

    def test_runtime_fallback_is_legacy_only(self):
        converted = _parse_clue_from_memory(
            _memory_baker("", acted_infos=[], runtime_role="Bard"),
            n_cards=5,
        )
        original = _parse_clue_from_memory(
            _memory_baker("", acted_infos=[], runtime_role=None),
            n_cards=5,
        )

        self.assertEqual(converted.info_parsed, {"original_role": "Bard"})
        self.assertEqual(original.info_parsed, {"original_role": "original"})

    def test_rambler_interrupted_baker_still_captures_shut_up_target(self):
        parsed = self._parse_current(
            _memory_baker(
                "#4 shut up!",
                acted_infos=[{"desc": "#4 shut up!", "targets": [4]}],
            )
        )

        self.assertEqual(parsed.info_parsed, {"shut_up_target": 4})
        self.assertEqual(parsed.info_text, "#4 shut up!")


class BakerManualCaptureTests(unittest.TestCase):
    def test_no_arg_is_explicit_original_but_literal_baker_stays_literal(self):
        session = GameSession(5, 1)

        self.assertEqual(
            _parse_card_cli(["baker", "1"], session=session).info_parsed,
            {"original_role": "original"},
        )
        self.assertEqual(
            _parse_card_cli(
                ["baker", "1", "original"], session=session,
            ).info_parsed,
            {"original_role": "original"},
        )
        self.assertEqual(
            _parse_card_cli(
                ["baker", "1", "Baker"], session=session,
            ).info_parsed,
            {"original_role": "Baker"},
        )
        self.assertEqual(
            _parse_card_cli(
                ["baker", "1", "fortune_teller"], session=session,
            ).info_parsed,
            {"original_role": "Fortune Teller"},
        )

    def test_empty_none_unknown_and_nonroles_are_rejected(self):
        session = GameSession(5, 1)
        for claim in ("", "none", "unknown", "?", "not_a_role"):
            with self.subTest(claim=claim):
                with self.assertRaises(ValueError):
                    _parse_card_cli(
                        ["baker", "1", claim],
                        session=session,
                    )

    def test_auto_card_surfaces_current_baker_recovery_instead_of_runtime_guess(self):
        session = GameSession(5, 1)
        session.reveal_order = [1]
        memory = _memory_baker("", acted_infos=[], runtime_role="Baker")

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [memory]

            def close(self):
                return None

        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("memory_reader.print_board"),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            redirect_stdout(StringIO()) as output,
        ):
            dispatch("auto_card", [], session)

        self.assertEqual(session.cards, [])
        self.assertIn("[RECOVERY]", output.getvalue())
        self.assertIn("Baker clue", output.getvalue())
        self.assertEqual(session.baker_rule_version, BAKER_RULE_VERSION)

    def test_auto_card_replaces_early_baker_placeholder_with_coherent_event(self):
        session = GameSession(5, 1)
        session.reveal_order = [1]
        session.add_card(CardInfo(1, "Baker"))
        clue = "I was a Baker"
        memory = _memory_baker(clue)

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [memory]

            def close(self):
                return None

        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("memory_reader.print_board"),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            redirect_stdout(StringIO()),
        ):
            dispatch("auto_card", [], session)

        self.assertEqual(
            session.cards[0].info_parsed,
            {"original_role": "Baker"},
        )
        self.assertEqual(session.cards[0].info_text, clue)
        self.assertEqual(session.baker_rule_version, BAKER_RULE_VERSION)

    def test_state_machine_pauses_on_malformed_current_baker_event(self):
        session = GameSession(1, 0)
        memory = _memory_baker("", acted_infos=[], runtime_role="Baker")

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return [memory]

            @staticmethod
            def wait_for(predicate, **_kwargs):
                return bool(predicate([memory]))

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine.phase = GamePhase.REVEALING
        machine._pending_reveal = (1,)

        with (
            patch("game_utils.all_game_card_coords", return_value={1: (10, 10)}),
            patch("template_match.safe_click_at"),
            patch("template_match.fast_click_at"),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_revealing()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("did not settle in memory", machine._needs_human_reason)
        self.assertEqual(session.cards, [])
        self.assertEqual(session.reveal_order, [1])
        self.assertEqual(session.baker_rule_version, BAKER_RULE_VERSION)


class RuntimeDataClassDiscriminationTests(unittest.TestCase):
    def test_il2cpp_object_class_name_uses_klass_then_native_name_pointer(self):
        reader = MemoryReader()
        object_ptr = 0x20000
        klass_ptr = 0x30000
        name_ptr = 0x40000
        reader._read_ptr = Mock(side_effect=lambda address: {
            object_ptr: klass_ptr,
            klass_ptr + IL2CPP_CLASS_NAME_OFFSET: name_ptr,
        }.get(address))
        reader._read_c_string = Mock(return_value="BakerRuntimeData")

        self.assertEqual(
            reader._read_object_class_name(object_ptr),
            "BakerRuntimeData",
        )
        reader._read_c_string.assert_called_once_with(name_ptr)

    def test_runtime_layout_comes_from_object_class_not_true_role(self):
        char_ptr = 0x10000
        runtime_ptr = 0x20000

        cases = [
            (
                "AlchemistRuntimeData",
                "Baker",
                {"type": "corrupted_around", "corrupted_around": 3},
            ),
            (
                "EnlightenedRuntimeData",
                "Baker",
                {"type": "direction", "direction": "CCW"},
            ),
        ]
        for runtime_class, misleading_role, expected in cases:
            with self.subTest(runtime_class=runtime_class):
                reader = MemoryReader()
                reader._read_ptr = Mock(
                    side_effect=lambda address: (
                        runtime_ptr
                        if address == char_ptr + CHAR_RUNTIME_DATA_OFFSET
                        else None
                    )
                )
                reader._read_object_class_name = Mock(
                    return_value=runtime_class,
                )
                reader._read_i32 = Mock(return_value=(3 if runtime_class.startswith("Alchemist") else 20))

                self.assertEqual(
                    reader._read_runtime_data(char_ptr, misleading_role),
                    expected,
                )

    def test_baker_runtime_reads_string_only_for_exact_baker_class(self):
        reader = MemoryReader()
        char_ptr = 0x10000
        runtime_ptr = 0x20000
        string_ptr = 0x30000
        reader._read_ptr = Mock(side_effect=lambda address: {
            char_ptr + CHAR_RUNTIME_DATA_OFFSET: runtime_ptr,
            runtime_ptr + 0x10: string_ptr,
        }.get(address))
        reader._read_object_class_name = Mock(return_value="BakerRuntimeData")
        reader._read_string = Mock(return_value="Baker")

        self.assertEqual(
            reader._read_runtime_data(char_ptr, "Alchemist"),
            {"type": "baker", "original_role": "Baker"},
        )
        reader._read_string.assert_called_once_with(string_ptr)

    def test_unknown_runtime_class_is_not_guessed_from_role(self):
        reader = MemoryReader()
        char_ptr = 0x10000
        runtime_ptr = 0x20000
        reader._read_ptr = Mock(return_value=runtime_ptr)
        reader._read_object_class_name = Mock(return_value="OtherRuntimeData")
        reader._read_i32 = Mock()
        reader._read_string = Mock()

        self.assertIsNone(reader._read_runtime_data(char_ptr, "Baker"))
        reader._read_i32.assert_not_called()
        reader._read_string.assert_not_called()


if __name__ == "__main__":
    unittest.main()
