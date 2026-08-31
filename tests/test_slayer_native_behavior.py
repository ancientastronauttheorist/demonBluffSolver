"""Focused regressions for Slayer's native register-and-reveal behavior."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from game_loop import CardInfo, DecisionLog, GameSession, dispatch


def _session() -> GameSession:
    session = GameSession(3, 1)
    session.cards = [CardInfo(1, "Slayer", info_parsed={})]
    return session


class SlayerBookkeepingTests(unittest.TestCase):
    def test_evil_kill_records_public_role_without_hp_loss(self):
        session = _session()

        session.add_slayer_result(1, 2, True, revealed_role="Shaman")

        self.assertEqual(session.slayer_results, [{
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": True,
            "revealed_role": "Shaman",
            "was_evil": True,
        }])
        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_evil, [2])
        self.assertEqual(session.confirmed_good, [])
        self.assertEqual(session.executed_evil_roles, {2: "Shaman"})
        self.assertEqual(session.executed_good_roles, {})
        self.assertEqual(session.hp, 10)
        self.assertEqual(session.used_abilities, [1])

    def test_wretch_kill_is_good_role_evidence_and_costs_hp(self):
        session = _session()
        session.hp = 7
        session.wrong_exec_cost = 5

        session.add_slayer_result(
            1,
            2,
            True,
            revealed_role="Wretch",
            was_corrupted=False,
            was_evil=False,
        )

        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.confirmed_good, [2])
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.executed_good_roles, {2: "Wretch"})
        self.assertEqual(session.executed_good_corrupted, {2: False})
        self.assertEqual(session.hp, 2)

    def test_wretch_damage_clamps_and_duplicate_result_is_rejected(self):
        session = _session()
        session.hp = 3
        session.wrong_exec_cost = 5
        session.add_slayer_result(
            1, 2, True, revealed_role="Wretch", was_evil=False
        )

        self.assertEqual(session.hp, 0)
        with self.assertRaisesRegex(ValueError, "already has a recorded result"):
            session.add_slayer_result(
                1, 2, True, revealed_role="Wretch", was_evil=False
            )
        self.assertEqual(session.hp, 0)
        self.assertEqual(len(session.slayer_results), 1)

    def test_failed_attempt_marks_ability_only(self):
        session = _session()

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
        session = _session()

        with redirect_stdout(StringIO()) as output:
            dispatch("slayer_result", ["1", "2", "kil"], session)

        self.assertIn("Unknown Slayer outcome", output.getvalue())
        self.assertEqual(session.slayer_results, [])
        self.assertEqual(session.used_abilities, [])
        self.assertEqual(session.executed, [])

    def test_transformed_good_role_defaults_to_neutral_public_death(self):
        session = _session()

        session.add_slayer_result(1, 2, True, revealed_role="Knight")

        self.assertEqual(session.slayer_results[0]["revealed_role"], "Knight")
        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_good, [])
        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.executed_good_roles, {})
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.hp, 10)

    def test_slayer_bypasses_corrupted_good_knight_extra_damage(self):
        session = _session()

        session.add_slayer_result(
            1,
            2,
            True,
            revealed_role="Knight",
            was_corrupted=True,
            was_evil=False,
        )

        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_good, [2])
        self.assertEqual(session.executed_good_roles, {2: "Knight"})
        self.assertEqual(session.executed_good_corrupted, {2: True})
        self.assertEqual(session.hp, 5)

    def test_transformed_runtime_evil_knight_keeps_original_role_untyped(self):
        session = _session()

        session.add_slayer_result(
            1,
            2,
            True,
            revealed_role="Knight",
            was_evil=True,
        )

        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_evil, [2])
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.hp, 10)

    def test_cli_accepts_alignment_and_status_in_either_order(self):
        for details in (["good", "corrupted"], ["corrupted", "good"]):
            with self.subTest(details=details):
                session = _session()
                with (
                    patch.object(session, "save"),
                    patch.object(DecisionLog, "log_slayer_result"),
                    redirect_stdout(StringIO()),
                ):
                    dispatch(
                        "slayer_result",
                        ["1", "2", "kill", "Knight", *details],
                        session,
                    )
                self.assertEqual(session.confirmed_good, [2])
                self.assertEqual(session.hp, 5)

    def test_cli_keeps_zero_delta_moved_kill_alignment_unknown(self):
        session = _session()
        session.minions = ["Twin Minion"]

        with (
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_slayer_result"),
            redirect_stdout(StringIO()),
        ):
            dispatch("slayer_result", ["1", "2", "kill", "Knight"], session)

        # No HP delta cannot distinguish runtime Evil from runtime Good with a
        # preserved NoDamage status after InitWithNoReset.
        self.assertEqual(session.slayer_results, [{
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": True,
            "revealed_role": "Knight",
        }])
        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_good, [])
        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.executed_good_roles, {})
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.hp, 10)

    def test_slayer_killed_baa_runs_deck_refresh_hook(self):
        session = _session()

        with (
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_slayer_result"),
            patch("game_loop._baa_post_death_deck_refresh") as refresh_baa,
            redirect_stdout(StringIO()),
        ):
            dispatch("slayer_result", ["1", "2", "kill", "Baa"], session)

        refresh_baa.assert_called_once_with(session)
        self.assertEqual(session.executed_evil_roles, {2: "Baa"})

    def test_invalid_actor_and_positions_reject_before_mutation(self):
        cases = [
            (0, 2, "within"),
            (1, 4, "within"),
            (2, 1, "not an apparent Slayer"),
        ]
        for slayer_pos, target_pos, message in cases:
            with self.subTest(slayer_pos=slayer_pos, target_pos=target_pos):
                session = _session()
                session.cards.append(CardInfo(2, "Knight", info_parsed={}))
                with self.assertRaisesRegex(ValueError, message):
                    session.add_slayer_result(
                        slayer_pos,
                        target_pos,
                        True,
                        revealed_role="Shaman",
                    )
                self.assertEqual(session.slayer_results, [])
                self.assertEqual(session.executed, [])
                self.assertEqual(session.used_abilities, [])


if __name__ == "__main__":
    unittest.main()
