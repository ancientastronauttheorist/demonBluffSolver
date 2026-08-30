"""Focused regression tests for memory-verified reveal bookkeeping."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import Mock, patch

from game_loop import (
    GameSession,
    _apply_flip_verification,
    _verify_flips,
    dispatch,
)
from solver import CardInfo
from state_machine import GamePhase, GameStateMachine


def _board(n_cards: int, hidden: set[int]) -> list[dict]:
    return [
        {
            "position": position,
            "state": "Hidden" if position in hidden else "Alive",
            "killed_hidden": False,
        }
        for position in range(1, n_cards + 1)
    ]


class _FakeSession:
    def __init__(
        self,
        reveal_order,
        *,
        blocked_positions=None,
        has_witch=True,
        witch_dead=False,
    ):
        self.reveal_order = list(reveal_order)
        self.blocked_positions = list(blocked_positions or [])
        self._has_witch = has_witch
        self._witch_dead = witch_dead
        self.save = Mock()

    def has_role_in_deck(self, role):
        return role == "Witch" and self._has_witch

    def is_witch_known_dead(self):
        return self._witch_dead


class RevealVerificationTests(unittest.TestCase):
    def test_memory_verification_preserves_current_baker_marker_without_upgrading_legacy(self):
        current = GameSession(3, 1)
        _apply_flip_verification(
            current,
            [1],
            {"flipped": [1], "blocked": [], "failed": [], "dead": []},
            persist=False,
        )
        self.assertEqual(current.reveal_order, [1])
        self.assertEqual(current.baker_rule_version, "baker_day_reveal_v1")

        downgraded = GameSession(3, 1)
        downgraded.add_card(CardInfo(1, "Bard"))
        _apply_flip_verification(
            downgraded,
            [2],
            {"flipped": [2], "blocked": [], "failed": [], "dead": []},
            persist=False,
        )
        self.assertEqual(downgraded.reveal_order, [1, 2])
        self.assertIsNone(downgraded.baker_rule_version)

    def test_batch_persists_arbitrary_non_max_block_and_true_reveal_order(self):
        session = _FakeSession(range(1, 9))

        verified = _verify_flips(_board(8, {1}), list(range(1, 9)), session)
        _apply_flip_verification(session, list(range(1, 9)), verified)

        self.assertEqual(verified["blocked"], [1])
        self.assertEqual(verified["failed"], [])
        self.assertEqual(session.reveal_order, [2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(session.blocked_positions, [1])
        session.save.assert_called_once_with()

    def test_block_is_global_and_has_no_target_identity_relation(self):
        # The verifier is intentionally given no target identity. The sole
        # hidden seat is legal block evidence even when it is Witch itself.
        session = _FakeSession([2, 3], blocked_positions=[])

        verified = _verify_flips(_board(3, {1}), [1], session)
        _apply_flip_verification(session, [1], verified)

        self.assertEqual(verified["blocked"], [1])
        self.assertEqual(session.blocked_positions, [1])
        self.assertNotIn(1, session.reveal_order)

    def test_multiple_hidden_cards_are_click_failures_not_witch_blocks(self):
        session = _FakeSession([1, 2, 3, 4])

        verified = _verify_flips(_board(4, {1, 4}), [1, 2, 3, 4], session)
        _apply_flip_verification(session, [1, 2, 3, 4], verified)

        self.assertEqual(verified["blocked"], [])
        self.assertEqual(verified["failed"], [1, 4])
        self.assertEqual(session.reveal_order, [2, 3])

    def test_known_dead_witch_cannot_explain_a_hidden_click(self):
        session = _FakeSession([1, 2, 3], witch_dead=True)

        verified = _verify_flips(_board(3, {1}), [1], session)
        _apply_flip_verification(session, [1], verified)

        self.assertEqual(verified["blocked"], [])
        self.assertEqual(verified["failed"], [1])
        self.assertNotIn(1, session.reveal_order)

    def test_verified_single_retry_clears_marker_and_appends_at_true_index(self):
        session = _FakeSession([2, 3], blocked_positions=[1])

        verified = _verify_flips(_board(3, set()), [1], session)
        _apply_flip_verification(session, [1], verified)

        self.assertEqual(verified["flipped"], [1])
        self.assertEqual(session.reveal_order, [2, 3, 1])
        self.assertEqual(session.blocked_positions, [])
        session.save.assert_called_once_with()

    def test_missing_memory_position_never_enters_reveal_order(self):
        session = _FakeSession([1, 2, 3])
        incomplete_board = _board(3, set())[:-1]

        verified = _verify_flips(incomplete_board, [3], session)
        _apply_flip_verification(session, [3], verified)

        self.assertEqual(verified["failed"], [3])
        self.assertNotIn(3, session.reveal_order)

    def test_killed_hidden_target_is_resolved_but_never_a_reveal(self):
        session = _FakeSession([1, 2, 3, 4])
        board = _board(4, set())
        board[3].update(state="Hidden", killed_hidden=True)

        verified = _verify_flips(board, [1, 2, 3, 4], session)
        _apply_flip_verification(session, [1, 2, 3, 4], verified)

        self.assertEqual(verified["flipped"], [1, 2, 3])
        self.assertEqual(verified["dead"], [4])
        self.assertNotIn(4, session.reveal_order)

    def test_dispatch_single_retry_mutates_only_from_final_memory_board(self):
        session = GameSession(3, 1)
        session.minions = ["Witch"]
        session.reveal_order = [2, 3]
        session.blocked_positions = [1]

        with (
            patch("game_utils.all_game_card_coords", return_value={1: (100, 100)}),
            patch("game_loop._click_flip_card"),
            patch("game_loop._read_board_once_for_flip", return_value=_board(3, set())),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            dispatch("flip", ["1"], session)

        self.assertEqual(session.reveal_order, [2, 3, 1])
        self.assertEqual(session.blocked_positions, [])

    def test_lilis_mode_without_lilis_rejects_before_any_click(self):
        session = GameSession(3, 1)

        with (
            patch("game_utils.all_game_card_coords") as coords,
            patch("game_loop._click_flip_card") as click,
            redirect_stdout(StringIO()) as output,
        ):
            dispatch("flip", ["--lilis"], session)

        self.assertIn("requires Lilis", output.getvalue())
        coords.assert_not_called()
        click.assert_not_called()
        self.assertEqual(session.reveal_order, [])

    def test_dispatch_single_killed_hidden_never_counts_as_reveal_or_night(self):
        session = GameSession(5, 1)
        session.demons = ["Lilis"]
        session.reveal_order = [1, 2, 3, 4]
        board = _board(5, set())
        board[4].update(state="Hidden", killed_hidden=True)

        with (
            patch("game_utils.all_game_card_coords", return_value={5: (500, 100)}),
            patch("game_loop._click_flip_card"),
            patch("game_loop._read_board_once_for_flip", return_value=board),
            patch.object(session, "save"),
            redirect_stdout(StringIO()) as output,
        ):
            dispatch("flip", ["5"], session)

        self.assertEqual(session.reveal_order, [1, 2, 3, 4])
        self.assertEqual(session.lilis_batch_index, 0)
        self.assertIn("resolved dead/hidden", output.getvalue())
        self.assertNotIn("Verified reveal", output.getvalue())
        self.assertNotIn("LILIS NIGHT", output.getvalue())

    def test_lilis_fourth_block_does_not_create_false_night_boundary(self):
        session = GameSession(4, 1)
        session.minions = ["Witch"]
        session.demons = ["Lilis"]

        class Monitor:
            def __init__(self):
                self.wait_for = Mock(return_value=False)

            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return _board(4, {4})

        monitor = Monitor()
        with (
            patch("game_utils.all_game_card_coords", return_value={
                1: (100, 100), 2: (200, 100), 3: (300, 100), 4: (400, 100)
            }),
            patch("game_loop._click_flip_card"),
            patch("memory_reader.get_monitor", return_value=monitor),
            patch("memory_reader.print_board"),
            patch("mouse.move"),
            patch("game_loop.time.sleep"),
            patch.object(session, "save"),
            redirect_stdout(StringIO()) as output,
        ):
            dispatch("flip", ["--lilis"], session)

        self.assertEqual(session.reveal_order, [1, 2, 3])
        self.assertEqual(session.blocked_positions, [4])
        self.assertEqual(session.lilis_batch_index, 0)
        self.assertEqual(monitor.wait_for.call_count, 1)
        self.assertIn("Lilis night did not trigger", output.getvalue())
        self.assertNotIn("Lilis deals 2 HP", output.getvalue())

    def test_manual_verified_fourth_reveal_increments_lilis_round(self):
        session = GameSession(4, 1)
        session.demons = ["Lilis"]
        session.reveal_order = [1, 2, 3]
        saved_states = []

        def capture_save():
            saved_states.append((list(session.reveal_order), session.pending_lilis_nights))

        with (
            patch("game_utils.all_game_card_coords", return_value={4: (400, 100)}),
            patch("game_loop._click_flip_card"),
            patch("game_loop._read_board_once_for_flip", return_value=_board(4, set())),
            patch("memory_reader.get_monitor") as get_monitor,
            patch("game_loop.time.sleep"),
            patch.object(session, "save", side_effect=capture_save),
            redirect_stdout(StringIO()),
        ):
            get_monitor.return_value.is_healthy.return_value = False
            dispatch("flip", ["4"], session)

        self.assertEqual(session.reveal_order, [1, 2, 3, 4])
        self.assertEqual(session.lilis_batch_index, 1)
        self.assertEqual(session.pending_lilis_nights, 1)
        self.assertEqual(saved_states, [([1, 2, 3, 4], 1)])

    def test_manual_witch_partial_retries_only_one_reveal_at_post_death_boundary(self):
        session = GameSession(4, 2)
        session.minions = ["Witch"]
        session.demons = ["Lilis"]

        boards = iter([_board(4, {4}), _board(4, set())])

        class Monitor:
            wait_for = Mock(return_value=False)

            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return next(boards)

        monitor = Monitor()
        coords = {
            position: (position * 100, 100)
            for position in range(1, 5)
        }
        with (
            patch("game_utils.all_game_card_coords", return_value=coords),
            patch("game_loop._click_flip_card", return_value=True) as click,
            patch("memory_reader.get_monitor", return_value=monitor),
            patch("memory_reader.print_board"),
            patch("mouse.move"),
            patch("game_loop.time.sleep"),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            dispatch("flip", ["--lilis"], session)
            self.assertEqual(session.reveal_order, [1, 2, 3])
            self.assertEqual(session.blocked_positions, [4])
            self.assertEqual(session.pending_lilis_nights, 0)

            session.mark_executed(1, was_evil=True, evil_role="Witch")
            session.mark_executed(2, was_evil=True, evil_role="Lilis")
            dispatch("flip", ["--lilis"], session)
            dispatch("night_no_kill", [], session)

        self.assertEqual(
            [call.args[0] for call in click.call_args_list],
            [1, 2, 3, 4, 4],
        )
        self.assertEqual(session.reveal_order, [1, 2, 3, 4])
        self.assertEqual(session.pending_lilis_nights, 0)
        self.assertEqual(session.lilis_nights_resolved, 1)
        self.assertEqual(session.hp, 10)

    def test_state_machine_failed_partial_retries_one_and_stops_at_active_night(self):
        session = GameSession(8, 1)
        session.demons = ["Lilis"]
        boards = iter([
            _board(8, {4, 5, 6, 7, 8}),
            _board(8, {5, 6, 7, 8}),
        ])

        class Monitor:
            wait_for = Mock(return_value=False)

            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return next(boards)

        coords = {
            position: (position * 100, 100)
            for position in range(1, 9)
        }
        machine = GameStateMachine(session=session, monitor=Monitor())
        machine.phase = GamePhase.FLIPPING
        with (
            patch("game_utils.all_game_card_coords", return_value=coords),
            patch("game_loop._click_flip_card", return_value=True) as click,
            patch("memory_reader.get_monitor", return_value=machine.monitor),
            patch("memory_reader.print_board"),
            patch("mouse.move"),
            patch("game_loop.time.sleep"),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_flipping()
            self.assertEqual(machine.phase, GamePhase.FLIPPING)
            self.assertEqual(session.reveal_order, [1, 2, 3])
            self.assertEqual(session.pending_lilis_nights, 0)
            machine._do_flipping()

        self.assertEqual(
            [call.args[0] for call in click.call_args_list],
            [1, 2, 3, 4, 4],
        )
        self.assertEqual(machine.phase, GamePhase.LILIS_NIGHT)
        self.assertEqual(session.reveal_order, [1, 2, 3, 4])
        self.assertEqual(session.pending_lilis_nights, 1)


if __name__ == "__main__":
    unittest.main()
