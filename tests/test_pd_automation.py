"""Plague Doctor public-result parsing and CLI validation regressions."""

import unittest
from unittest.mock import Mock

from game_loop import (
    GameSession,
    _parse_pd_ability_result_from_memory,
    _parse_pd_check_args,
)
from solver import CardInfo
from state_machine import GamePhase, GameStateMachine


def _memory_card(clue: str, targets: list[int]) -> dict:
    return {
        "clue_text": clue,
        "acted_infos": [{"desc": clue, "targets": targets}],
    }


class PlagueDoctorMemoryResultTests(unittest.TestCase):
    def test_exact_corrupted_result_uses_public_text_and_cross_checks_refs(self):
        parsed, error = _parse_pd_ability_result_from_memory(
            _memory_card("#5 is Evil\n#2 is Corrupted", [2, 5]),
            ability_pos=1,
            expected_target=2,
            n_cards=6,
        )

        self.assertIsNone(error)
        self.assertEqual(
            parsed,
            {"target": 2, "is_corrupted": True, "evil_revealed": 5},
        )

    def test_exact_clean_result(self):
        parsed, error = _parse_pd_ability_result_from_memory(
            _memory_card("#4 is\nNot Corrupted", [4]),
            ability_pos=1,
            expected_target=4,
            n_cards=6,
        )

        self.assertIsNone(error)
        self.assertEqual(
            parsed,
            {"target": 4, "is_corrupted": False, "evil_revealed": None},
        )

    def test_self_clean_ignores_native_hidden_second_reference(self):
        parsed, error = _parse_pd_ability_result_from_memory(
            _memory_card("#1 is\nNot Corrupted", [1, 3]),
            ability_pos=1,
            expected_target=1,
            n_cards=6,
        )

        self.assertIsNone(error)
        self.assertEqual(
            parsed,
            {"target": 1, "is_corrupted": False, "evil_revealed": None},
        )

    def test_rejects_target_and_result_cross_check_mismatches(self):
        _, picked_error = _parse_pd_ability_result_from_memory(
            _memory_card("#4 is\nNot Corrupted", [3]),
            ability_pos=1,
            expected_target=4,
            n_cards=6,
        )
        self.assertIn("picked-target mismatch", picked_error)

        _, clue_error = _parse_pd_ability_result_from_memory(
            _memory_card("#3 is\nNot Corrupted", [4]),
            ability_pos=1,
            expected_target=4,
            n_cards=6,
        )
        self.assertIn("clue-target mismatch", clue_error)

        _, reveal_error = _parse_pd_ability_result_from_memory(
            _memory_card("#5 is Evil\n#4 is Corrupted", [4, 2]),
            ability_pos=1,
            expected_target=4,
            n_cards=6,
        )
        self.assertIn("revealed-position mismatch", reveal_error)

    def test_rejects_missing_refs_and_unknown_text(self):
        _, missing_error = _parse_pd_ability_result_from_memory(
            _memory_card("#4 is\nNot Corrupted", []),
            ability_pos=1,
            expected_target=4,
            n_cards=6,
        )
        self.assertIn("no recorded picked target", missing_error)

        _, text_error = _parse_pd_ability_result_from_memory(
            _memory_card("something changed", [4]),
            ability_pos=1,
            expected_target=4,
            n_cards=6,
        )
        self.assertIn("Unrecognized", text_error)

    def test_rejects_impossible_reference_shapes_self_corruption_and_ranges(self):
        bad_cases = [
            (
                _memory_card("#5 is Evil\n#1 is Corrupted", [1, 5]),
                1,
                1,
                "self-check",
            ),
            (
                _memory_card("#4 is\nNot Corrupted", [4, 5]),
                1,
                4,
                "only the picked",
            ),
            (
                _memory_card("#5 is Evil\n#4 is Corrupted", [4]),
                1,
                4,
                "exactly the picked and revealed",
            ),
            (
                _memory_card("#7 is Evil\n#4 is Corrupted", [4, 7]),
                1,
                4,
                "within 1..6",
            ),
        ]
        for card, ability_pos, target, expected_error in bad_cases:
            with self.subTest(card=card):
                parsed, error = _parse_pd_ability_result_from_memory(
                    card,
                    ability_pos=ability_pos,
                    expected_target=target,
                    n_cards=6,
                )
                self.assertIsNone(parsed)
                self.assertIn(expected_error, error)


class PlagueDoctorCliValidationTests(unittest.TestCase):
    def test_accepts_exact_clean_and_corrupted_shapes(self):
        roles = {2: "Plague_Doctor"}
        clean, clean_error = _parse_pd_check_args(
            ["2", "4", "clean"], 6, apparent_roles=roles
        )
        self.assertIsNone(clean_error)
        self.assertEqual(
            clean,
            {
                "pd_pos": 2,
                "target": 4,
                "is_corrupted": False,
                "evil_revealed": None,
            },
        )

        corrupted, corrupted_error = _parse_pd_check_args(
            ["2", "4", "corrupted", "6"], 6, apparent_roles=roles
        )
        self.assertIsNone(corrupted_error)
        self.assertEqual(corrupted["evil_revealed"], 6)
        self.assertTrue(corrupted["is_corrupted"])

    def test_rejects_malformed_range_self_and_duplicate_entries(self):
        bad_inputs = [
            (["2", "4"], (), "Usage"),
            (["0", "4", "clean"], (), "outside"),
            (["2", "7", "clean"], (), "outside"),
            (["2", "4", "clean", "5"], (), "must not include"),
            (["2", "4", "corrupted"], (), "requires exactly"),
            (["2", "2", "corrupted", "5"], (), "self-check"),
            (["2", "4", "corrupted", "7"], (), "outside"),
            (["2", "4", "clean"], (2,), "already recorded"),
        ]

        for args, used, expected in bad_inputs:
            with self.subTest(args=args, used=used):
                parsed, error = _parse_pd_check_args(args, 6, used)
                self.assertIsNone(parsed)
                self.assertIn(expected, error)

        for roles, expected in [({2: "Bard"}, "not an apparent"), ({}, "unrevealed")]:
            with self.subTest(roles=roles):
                parsed, error = _parse_pd_check_args(
                    ["2", "4", "clean"], 6, apparent_roles=roles
                )
                self.assertIsNone(parsed)
                self.assertIn(expected, error)

    def test_clear_result_restores_explicit_correction_path(self):
        session = GameSession(6, 2)
        session.cards.append(CardInfo(2, "Plague_Doctor"))
        session.add_pd_ability_result(2, 4, True, 6)

        self.assertIn(2, session.used_abilities)
        self.assertEqual(session.clear_pd_ability_result(2), 1)
        self.assertNotIn(2, session.used_abilities)
        self.assertEqual(session.pd_ability_results, [])
        self.assertEqual(session.clear_pd_ability_result(2), 0)

    def test_session_rejects_duplicate_and_non_pd_actor_results(self):
        session = GameSession(6, 2)
        session.cards.extend([CardInfo(2, "Plague_Doctor"), CardInfo(3, "Bard")])
        session.add_pd_ability_result(2, 4, False)

        with self.assertRaisesRegex(ValueError, "already has"):
            session.add_pd_ability_result(2, 5, False)
        with self.assertRaisesRegex(ValueError, "not an apparent"):
            session.add_pd_ability_result(3, 4, False)


class PlagueDoctorStateMachineTests(unittest.TestCase):
    def test_autonomous_phase_routes_pd_through_exact_session_automation(self):
        session = GameSession(6, 2)
        session.auto_use_ability = Mock(
            return_value={"success": True, "info_parsed": {}, "error": None}
        )
        machine = GameStateMachine(session=session, monitor=None)
        machine.phase = GamePhase.ABILITY_USE
        machine._pending_ability = (2, [4], "Plague Doctor", None)

        machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.SOLVING)
        action = session.auto_use_ability.call_args.args[0]
        self.assertEqual(action.action_type, "use_ability")
        self.assertEqual(action.position, 2)
        self.assertEqual(action.targets, [4])
        self.assertEqual(action.ability_name, "Plague Doctor")

    def test_autonomous_phase_pauses_with_pd_check_recovery_on_parse_failure(self):
        session = GameSession(6, 2)
        session.auto_use_ability = Mock(
            return_value={
                "success": False,
                "info_parsed": None,
                "error": "public result mismatch",
            }
        )
        machine = GameStateMachine(session=session, monitor=None)
        machine.phase = GamePhase.ABILITY_USE
        machine._pending_ability = (2, [4], "Plague_Doctor", None)

        machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("pd_check", machine._needs_human_reason)
        self.assertIn("public result mismatch", machine._needs_human_reason)


if __name__ == "__main__":
    unittest.main()
