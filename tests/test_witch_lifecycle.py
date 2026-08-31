"""Focused current-build tests for public Witch quota and death lifecycle."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from game_loop import CardInfo, DecisionLog, GameSession, dispatch
from knowledge_base import get_card
from solver import DeckComposition, GameState, Scenario, SolverResult
from state_machine import GamePhase, GameStateMachine
from strategy import (
    _active_witch_blocked_positions,
    _remaining_evil_bounds,
    _witch_might_be_alive,
    recommend_action,
    recommend_reveal,
)


def _result(*scenarios: Scenario) -> SolverResult:
    return SolverResult([], [], [], len(scenarios), len(scenarios), list(scenarios))


def _state(
    *,
    executed=None,
    night_kills=None,
    blocked=None,
) -> GameState:
    return GameState(
        n_cards=2,
        n_evil=1,
        deck=DeckComposition(
            villagers=["Baker"],
            outcasts=[],
            minions=["Witch"],
            demons=[],
        ),
        cards=[CardInfo(1, "Baker", info_parsed={})],
        executed=list(executed or []),
        night_kills=list(night_kills or []),
        blocked_positions=list(blocked or []),
    )


def _dead_witch(position: int = 1) -> dict:
    return {
        "position": position,
        "true_role": "Witch",
        "disguise": "Baker",
        "is_evil": True,
        "state": "Dead",
        "statuses": [],
    }


class WitchDeathLifecycleTests(unittest.TestCase):
    def test_public_knowledge_marks_witch_as_start_ability(self):
        self.assertTrue(get_card("Witch").game_start_ability)

    def test_ordinary_known_witch_death_releases_without_auto_flip(self):
        session = GameSession(3, 1)
        session.minions = ["Witch"]
        session.blocked_positions = [1]
        session.reveal_order = [2, 3]

        session.mark_executed(2, was_evil=True, evil_role="Witch")

        self.assertEqual(session.blocked_positions, [])
        self.assertEqual(session.reveal_order, [2, 3])
        self.assertEqual(session.executed, [2])

    def test_either_duplicate_witch_death_releases_the_single_ordinary_quota(self):
        session = GameSession(3, 2)
        session.minions = ["Witch", "Witch"]
        session.blocked_positions = [1]
        session.reveal_order = [2, 3]

        session.mark_executed(2, was_evil=True, evil_role="Witch")

        self.assertTrue(session.is_witch_known_dead())
        self.assertEqual(session.blocked_positions, [])
        self.assertEqual(session.reveal_order, [2, 3])

    def test_auto_executing_self_blocked_witch_releases_without_auto_flip(self):
        session = GameSession(3, 1)
        session.minions = ["Witch"]
        session.blocked_positions = [1]
        session.reveal_order = [2, 3]
        result = SimpleNamespace(
            bombardier_positions=[],
            surviving_scenarios=[Scenario(evil_positions={1: "Witch"})],
        )

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return True

            @staticmethod
            def get_board():
                return [_dead_witch()]

        with (
            patch("game_utils.all_game_card_coords", return_value={1: (100, 100)}),
            patch("template_match.safe_click_at"),
            patch("mouse.click"),
            patch("game_loop.time.sleep"),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution"),
            redirect_stdout(StringIO()),
        ):
            observed = session.auto_execute(1, result, monitor=Monitor())

        self.assertTrue(observed["success"])
        self.assertEqual(session.blocked_positions, [])
        self.assertEqual(session.reveal_order, [2, 3])
        self.assertEqual(session.executed_evil_roles, {1: "Witch"})

    def test_slayer_witch_death_uses_same_release_transition(self):
        session = GameSession(3, 1)
        session.minions = ["Witch"]
        session.blocked_positions = [1]
        session.reveal_order = [2, 3]
        session.cards = [CardInfo(2, "Slayer", info_parsed={})]

        session.add_slayer_result(2, 1, True, revealed_role="Witch")

        self.assertEqual(session.blocked_positions, [])
        self.assertEqual(session.reveal_order, [2, 3])
        self.assertEqual(session.executed_evil_roles, {1: "Witch"})


class WitchNightLifecycleTests(unittest.TestCase):
    def _night_session(self) -> GameSession:
        session = GameSession(2, 1)
        session.minions = ["Witch"]
        session.demons = ["Lilis"]
        session.blocked_positions = [1]
        session.lilis_batch_index = 1
        session.pending_lilis_nights = 1
        return session

    def test_manual_evil_night_kill_releases_only_for_public_reprobe(self):
        session = self._night_session()

        with patch.object(session, "save"), redirect_stdout(StringIO()):
            dispatch("night_kill", ["2", "1"], session)

        self.assertEqual(session.blocked_positions, [])
        self.assertEqual(session.night_kills, [2])
        self.assertEqual(session.executed, [])
        self.assertNotIn(2, session.executed_evil_roles)

    def test_manual_good_night_kill_preserves_active_block_marker(self):
        session = self._night_session()

        with patch.object(session, "save"), redirect_stdout(StringIO()):
            dispatch("night_kill", ["2", "0"], session)

        self.assertEqual(session.blocked_positions, [1])
        self.assertEqual(session.night_kills, [2])
        self.assertEqual(session.executed, [])

    def test_state_machine_hidden_evil_alignment_is_not_imported(self):
        session = self._night_session()

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return [
                    {"position": 1, "state": "Hidden", "killed_hidden": False},
                    {
                        "position": 2,
                        "state": "Hidden",
                        "killed_hidden": True,
                        "is_evil": True,
                    },
                ]

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine.phase = GamePhase.NIGHT_RESOLVE
        with patch.object(session, "save") as save, redirect_stdout(StringIO()):
            machine._do_night_resolve()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertEqual(session.blocked_positions, [1])
        self.assertEqual(session.night_kills, [])
        self.assertEqual(session.executed, [])
        save.assert_not_called()

    def test_state_machine_good_night_kill_keeps_marker(self):
        session = self._night_session()

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return [
                    {"position": 1, "state": "Hidden", "killed_hidden": False},
                    {
                        "position": 2,
                        "state": "Hidden",
                        "killed_hidden": True,
                        "is_evil": False,
                    },
                ]

        machine = GameStateMachine(session=session, monitor=Monitor())
        with patch.object(session, "save"), redirect_stdout(StringIO()):
            machine._do_night_resolve()

        self.assertEqual(session.blocked_positions, [1])
        self.assertEqual(session.executed, [])


class WitchStateMachineRevealTests(unittest.TestCase):
    def _session(self) -> GameSession:
        session = GameSession(3, 1)
        session.minions = ["Witch"]
        session.reveal_order = [2, 3]
        return session

    def test_failed_arbitrary_reprobe_is_reclassified_as_block(self):
        session = self._session()

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return False

            @staticmethod
            def get_board():
                return [
                    {"position": 1, "state": "Hidden", "killed_hidden": False},
                    {"position": 2, "state": "Alive", "killed_hidden": False},
                    {"position": 3, "state": "Alive", "killed_hidden": False},
                ]

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine._pending_reveal = (1,)
        with (
            patch("game_utils.all_game_card_coords", return_value={1: (100, 100)}),
            patch("template_match.safe_click_at"),
            patch("template_match.fast_click_at"),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_revealing()

        self.assertEqual(machine.phase, GamePhase.SOLVING)
        self.assertEqual(session.blocked_positions, [1])
        self.assertEqual(session.reveal_order, [2, 3])

    def test_final_memory_board_wins_over_wait_timeout(self):
        session = self._session()
        session.blocked_positions = [1]

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return False

            @staticmethod
            def get_board():
                return [
                    {"position": 1, "state": "Alive", "killed_hidden": False},
                    {"position": 2, "state": "Alive", "killed_hidden": False},
                    {"position": 3, "state": "Alive", "killed_hidden": False},
                ]

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine._pending_reveal = (1,)
        with (
            patch("game_utils.all_game_card_coords", return_value={1: (100, 100)}),
            patch("template_match.safe_click_at"),
            patch("template_match.fast_click_at"),
            patch.object(session, "save"),
            patch.object(machine, "_auto_enter_single_card", return_value=True),
            redirect_stdout(StringIO()),
        ):
            machine._do_revealing()

        self.assertEqual(machine.phase, GamePhase.SOLVING)
        self.assertEqual(session.blocked_positions, [])
        self.assertEqual(session.reveal_order, [2, 3, 1])

    def test_flipping_uses_pending_delta_not_historical_batch_index(self):
        session = self._session()
        session.demons = ["Lilis"]
        session.lilis_batch_index = 1
        session.reveal_order = [1, 2]
        machine = GameStateMachine(session=session, monitor=None)
        machine.phase = GamePhase.FLIPPING

        with patch("game_loop.dispatch"), redirect_stdout(StringIO()):
            machine._do_flipping()

        self.assertEqual(machine.phase, GamePhase.FLIPPING)

    def test_fresh_pending_night_always_enters_night_resolution(self):
        session = self._session()
        session.demons = ["Lilis"]
        session.lilis_batch_index = 1
        session.reveal_order = [1, 2, 3]
        machine = GameStateMachine(session=session, monitor=None)
        machine.phase = GamePhase.FLIPPING

        def complete_batch(_command, _args, live_session):
            live_session.schedule_lilis_night()

        with patch("game_loop.dispatch", side_effect=complete_batch), redirect_stdout(StringIO()):
            machine._do_flipping()

        self.assertEqual(machine.phase, GamePhase.LILIS_NIGHT)


class WitchStrategyLifecycleTests(unittest.TestCase):
    def test_night_killed_witch_is_dead_for_liveness_and_bounds(self):
        state = _state(night_kills=[1], blocked=[2])
        result = _result(Scenario(evil_positions={1: "Witch"}))

        self.assertFalse(_witch_might_be_alive(state, result))
        self.assertEqual(_remaining_evil_bounds(state, result), (0, 0))
        self.assertEqual(_active_witch_blocked_positions(state, result), set())

    def test_night_killed_definite_evil_is_not_recommended_again(self):
        state = GameState(
            n_cards=3,
            n_evil=2,
            deck=DeckComposition(
                villagers=["Baker"],
                outcasts=[],
                minions=["Witch"],
                demons=["Pooka"],
            ),
            cards=[
                CardInfo(1, "Baker", info_parsed={}),
                CardInfo(2, "Baker", info_parsed={}),
                CardInfo(3, "Baker", info_parsed={}),
            ],
            night_kills=[1],
        )
        scenario = Scenario(evil_positions={1: "Witch", 2: "Pooka"})
        result = SolverResult([1, 2], [3], [], 1, 1, [scenario])

        action = recommend_action(state, result, used_abilities=[])

        self.assertEqual(action.action_type, "execute")
        self.assertEqual(action.position, 2)

    def test_legacy_stale_marker_does_not_block_after_witch_execution(self):
        state = _state(executed=[1], blocked=[2])
        result = _result(Scenario(evil_positions={1: "Witch"}))

        recommendation = recommend_reveal(state, result)

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.position, 2)

    def test_surviving_duplicate_does_not_preserve_quota_after_witch_death(self):
        state = _state(executed=[1], blocked=[2])
        state.n_evil = 2
        state.deck.minions = ["Witch", "Witch"]
        state.executed_evil_roles = {1: "Witch"}
        result = _result(Scenario(evil_positions={1: "Witch", 2: "Witch"}))

        self.assertTrue(_witch_might_be_alive(state, result))
        self.assertEqual(_active_witch_blocked_positions(state, result), set())
        recommendation = recommend_reveal(state, result)
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.position, 2)

    def test_active_marker_blocks_but_unobserved_final_seat_is_probeable(self):
        scenario = Scenario(evil_positions={1: "Witch"})
        active = _state(blocked=[2])
        unobserved = _state(blocked=[])

        self.assertIsNone(recommend_reveal(active, _result(scenario)))
        recommendation = recommend_reveal(unobserved, _result(scenario))
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.position, 2)


if __name__ == "__main__":
    unittest.main()
