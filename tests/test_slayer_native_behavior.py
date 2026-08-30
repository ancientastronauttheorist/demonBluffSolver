"""Focused regressions for Slayer's native register-and-reveal behavior."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from game_loop import DecisionLog, GameSession, dispatch


class SlayerBookkeepingTests(unittest.TestCase):
    def test_evil_kill_records_public_role_without_hp_loss(self):
        session = GameSession(3, 1)

        session.add_slayer_result(1, 2, True, revealed_role="Shaman")

        self.assertEqual(session.slayer_results, [{
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": True,
            "revealed_role": "Shaman",
        }])
        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_evil, [2])
        self.assertEqual(session.confirmed_good, [])
        self.assertEqual(session.executed_evil_roles, {2: "Shaman"})
        self.assertEqual(session.executed_good_roles, {})
        self.assertEqual(session.hp, 10)
        self.assertEqual(session.used_abilities, [1])

    def test_wretch_kill_is_good_role_evidence_and_costs_hp(self):
        session = GameSession(3, 1)
        session.hp = 7
        session.wrong_exec_cost = 5

        session.add_slayer_result(
            1,
            2,
            True,
            revealed_role="Wretch",
            was_corrupted=False,
        )

        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.confirmed_good, [2])
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.executed_good_roles, {2: "Wretch"})
        self.assertEqual(session.executed_good_corrupted, {2: False})
        self.assertEqual(session.hp, 2)

    def test_wretch_damage_clamps_and_duplicate_result_is_rejected(self):
        session = GameSession(3, 1)
        session.hp = 3
        session.wrong_exec_cost = 5
        session.add_slayer_result(1, 2, True, revealed_role="Wretch")

        self.assertEqual(session.hp, 0)
        with self.assertRaisesRegex(ValueError, "already has a recorded result"):
            session.add_slayer_result(1, 2, True, revealed_role="Wretch")
        self.assertEqual(session.hp, 0)
        self.assertEqual(len(session.slayer_results), 1)

    def test_failed_attempt_marks_ability_only(self):
        session = GameSession(3, 1)

        session.add_slayer_result(1, 2, False)

        self.assertEqual(session.slayer_results, [{
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": False,
        }])
        self.assertEqual(session.executed, [])
        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.confirmed_good, [])
        self.assertEqual(session.hp, 10)
        self.assertEqual(session.used_abilities, [1])

    def test_unknown_cli_outcome_does_not_consume_slayer_result(self):
        session = GameSession(3, 1)

        with redirect_stdout(StringIO()) as output:
            dispatch("slayer_result", ["1", "2", "kil"], session)

        self.assertIn("Unknown Slayer outcome", output.getvalue())
        self.assertEqual(session.slayer_results, [])
        self.assertEqual(session.used_abilities, [])
        self.assertEqual(session.executed, [])

    def test_non_wretch_good_kill_is_rejected_without_mutation(self):
        session = GameSession(3, 1)

        with self.assertRaisesRegex(ValueError, "only kill an Evil character"):
            session.add_slayer_result(1, 2, True, revealed_role="Knight")

        self.assertEqual(session.slayer_results, [])
        self.assertEqual(session.executed, [])
        self.assertEqual(session.hp, 10)

    def test_slayer_killed_baa_runs_deck_refresh_hook(self):
        session = GameSession(3, 1)

        with (
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_slayer_result"),
            patch("game_loop._baa_post_death_deck_refresh") as refresh_baa,
            redirect_stdout(StringIO()),
        ):
            dispatch("slayer_result", ["1", "2", "kill", "Baa"], session)

        refresh_baa.assert_called_once_with(session)
        self.assertEqual(session.executed_evil_roles, {2: "Baa"})


if __name__ == "__main__":
    unittest.main()
