"""Focused tests for post-execution Knight and HP bookkeeping."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from game_loop import (
    CardInfo,
    DecisionLog,
    GameSession,
    _clamped_post_damage_hp,
    _observed_knight_immunity,
    dispatch,
)
from state_machine import GamePhase, GameStateMachine


def _observed_card(**overrides):
    card = {
        "position": 1,
        "true_role": "Knight",
        "disguise": None,
        "is_evil": False,
        "state": "Alive",
        "statuses": [],
    }
    card.update(overrides)
    return card


class KnightObservationTests(unittest.TestCase):
    def test_clean_true_knight_survival_confirms_immunity(self):
        self.assertTrue(_observed_knight_immunity(_observed_card()))

    def test_corrupted_true_knight_survival_does_not_confirm_immunity(self):
        self.assertFalse(
            _observed_knight_immunity(
                _observed_card(statuses=["Corrupted"]),
            )
        )

    def test_healthy_bluff_precedes_corruption_for_true_knight(self):
        self.assertTrue(
            _observed_knight_immunity(
                _observed_card(statuses=["Corrupted", "HealthyBluff"]),
            )
        )

    def test_drunk_as_knight_is_not_assumed_immune(self):
        self.assertFalse(
            _observed_knight_immunity(
                _observed_card(true_role="Drunk", disguise="Knight"),
            )
        )

    def test_doppelganger_knight_requires_healthy_bluff(self):
        base = _observed_card(true_role="Doppelganger", disguise="Knight")
        self.assertFalse(_observed_knight_immunity(base))
        self.assertTrue(
            _observed_knight_immunity(
                {**base, "statuses": ["HealthyBluff"]},
            )
        )

    def test_dead_or_evil_card_is_not_a_confirmed_good_block(self):
        self.assertFalse(
            _observed_knight_immunity(_observed_card(state="Dead"))
        )
        self.assertFalse(
            _observed_knight_immunity(_observed_card(is_evil=True))
        )


class ExecutionBookkeepingTests(unittest.TestCase):
    def test_hp_damage_clamps_at_zero(self):
        self.assertEqual(_clamped_post_damage_hp(3, 9), 0)
        self.assertEqual(_clamped_post_damage_hp(10, 6), 4)

    def test_record_blocked_persists_good_without_marking_executed(self):
        session = GameSession(1, 0)
        with (
            patch.object(session, "save") as save,
            patch.object(DecisionLog, "log_custom") as log_custom,
        ):
            session.record_execution_blocked(1)

        self.assertEqual(session.confirmed_good, [1])
        self.assertEqual(session.executed, [])
        self.assertEqual(session.hp, 10)
        save.assert_called_once_with()
        log_custom.assert_called_once()

    def test_auto_observed_knight_block_is_successful_and_persisted(self):
        session = GameSession(1, 1)
        session.cards = [CardInfo(1, "Knight", info_parsed={})]
        result = SimpleNamespace(bombardier_positions=[])

        class FakeMonitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return False

            @staticmethod
            def get_board():
                return [_observed_card()]

        with (
            patch("game_utils.all_game_card_coords", return_value={1: (100, 100)}),
            patch("template_match.safe_click_at"),
            patch("mouse.click"),
            patch("game_loop.time.sleep"),
            patch.object(session, "save") as save,
            patch.object(DecisionLog, "log_custom") as log_custom,
        ):
            observed = session.auto_execute(1, result, monitor=FakeMonitor())

        self.assertTrue(observed["success"])
        self.assertTrue(observed["blocked"])
        self.assertIsNone(observed["error"])
        self.assertEqual(session.confirmed_good, [1])
        self.assertEqual(session.executed, [])
        self.assertEqual(session.hp, 10)
        save.assert_called_once_with()
        log_custom.assert_called_once()

    def test_auto_corrupted_knight_damage_is_stored_with_zero_clamp(self):
        session = GameSession(1, 1)
        session.hp = 3
        session.cards = [CardInfo(1, "Knight", info_parsed={})]
        result = SimpleNamespace(bombardier_positions=[])
        dead_knight = _observed_card(state="Dead", statuses=["Corrupted"])

        class FakeMonitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return True

            @staticmethod
            def get_board():
                return [dead_knight]

        with (
            patch("game_utils.all_game_card_coords", return_value={1: (100, 100)}),
            patch("template_match.safe_click_at"),
            patch("mouse.click"),
            patch("game_loop.time.sleep"),
            patch.object(session, "save") as save,
            patch.object(DecisionLog, "log_execution") as log_execution,
            redirect_stdout(StringIO()),
        ):
            observed = session.auto_execute(1, result, monitor=FakeMonitor())

        self.assertTrue(observed["success"])
        self.assertFalse(observed["blocked"])
        self.assertEqual(session.hp, 0)
        self.assertEqual(session.executed, [1])
        self.assertEqual(session.executed_good_corrupted, {1: True})
        save.assert_called_once_with()
        log_execution.assert_called_once()

    def test_explicit_blocked_refuses_live_dead_card(self):
        session = GameSession(1, 1)
        session.cards = [CardInfo(1, "Knight", info_parsed={})]
        dead_knight = _observed_card(state="Dead")

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save") as save,
            patch.object(DecisionLog, "log_custom") as log_custom,
            redirect_stdout(StringIO()) as output,
        ):
            reader_type.return_value.open.return_value = True
            reader_type.return_value.read_board.return_value = [dead_knight]
            dispatch("execute", ["1", "good", "blocked"], session)

        self.assertEqual(session.executed, [])
        self.assertEqual(session.confirmed_good, [])
        self.assertIn("explicit blocked outcome contradicts", output.getvalue())
        save.assert_not_called()
        log_custom.assert_not_called()

    def test_state_machine_continues_after_confirmed_knight_block(self):
        session = SimpleNamespace(
            cards=[],
            executed=[],
            used_abilities=[],
            blocked_positions=[],
            hp=10,
        )
        session.auto_execute = lambda *_args, **_kwargs: {
            "success": True,
            "blocked": True,
            "was_evil": False,
            "evil_role": None,
            "error": None,
        }
        machine = GameStateMachine(session=session, monitor=None)
        machine._pending_exec = (1, SimpleNamespace(), False)
        machine.phase = GamePhase.EXECUTING

        with redirect_stdout(StringIO()) as output:
            machine._do_executing()

        self.assertEqual(machine.phase, GamePhase.SOLVING)
        self.assertIn("confirmed Knight immunity", output.getvalue())
        self.assertNotIn("WRONG EXECUTION", output.getvalue())

    def test_manual_drunk_as_knight_recommends_six_and_clamps_to_zero(self):
        session = GameSession(1, 1)
        session.hp = 3
        session.cards = [CardInfo(1, "Knight", info_parsed={})]
        dead_drunk = _observed_card(
            true_role="Drunk",
            disguise="Knight",
            state="Dead",
        )

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution"),
            redirect_stdout(StringIO()) as output,
        ):
            reader_type.return_value.open.return_value = True
            reader_type.return_value.read_board.return_value = [dead_drunk]
            dispatch("execute", ["1", "good"], session)

        self.assertEqual(session.executed, [1])
        self.assertIn(1, session.confirmed_good)
        self.assertIn("Drunk, showing as Knight: -6", output.getvalue())
        self.assertIn("HP 3 -> 0", output.getvalue())

    def test_offline_apparent_knight_requires_explicit_outcome(self):
        session = GameSession(1, 1)
        session.cards = [CardInfo(1, "Knight", info_parsed={})]

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save") as save,
            redirect_stdout(StringIO()) as output,
        ):
            reader_type.return_value.open.return_value = False
            dispatch("execute", ["1", "good"], session)

        self.assertEqual(session.executed, [])
        self.assertEqual(session.confirmed_good, [])
        self.assertIn("Cannot classify apparent Knight", output.getvalue())
        save.assert_not_called()

    def test_offline_explicit_clean_knight_does_not_invent_exact_damage(self):
        session = GameSession(1, 1)
        session.hp = 3
        session.cards = [CardInfo(1, "Knight", info_parsed={})]

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution"),
            redirect_stdout(StringIO()) as output,
        ):
            reader_type.return_value.open.return_value = False
            dispatch("execute", ["1", "good", "clean"], session)

        self.assertEqual(session.executed, [1])
        self.assertEqual(session.confirmed_good, [1])
        self.assertIn("exact HP damage cannot be inferred", output.getvalue())
        self.assertIn("set_hp <current_hp>", output.getvalue())
        self.assertNotIn("HP 3 ->", output.getvalue())


if __name__ == "__main__":
    unittest.main()
