"""Focused live current-vs-origin safety regressions for Twin Minion."""

import unittest
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from game_loop import (
    CardInfo,
    DecisionLog,
    GameSession,
    _consensus_original_evil_role,
    _observed_current_role,
    _resolve_runtime_evil_origins,
    _validate_true_evils_against_session,
    dispatch,
)
from memory_reader import print_board
from solver import Scenario, SolverResult
from state_machine import GamePhase, GameStateMachine
from strategy import Action


def _result(*scenarios: Scenario) -> SolverResult:
    return SolverResult(
        definite_evil=[],
        definite_good=[],
        bombardier_positions=[],
        n_scenarios=len(scenarios),
        n_surviving=len(scenarios),
        surviving_scenarios=list(scenarios),
    )


class CurrentRoleReaderTests(unittest.TestCase):
    def test_current_role_precedes_legacy_true_role_alias(self):
        self.assertEqual(
            _observed_current_role({
                "current_role": "Baa",
                "true_role": "Twin Minion",
            }),
            "Baa",
        )
        self.assertEqual(
            _observed_current_role({"true_role": "Witch"}),
            "Witch",
        )

    def test_board_output_labels_current_role_only(self):
        card = {
            "position": 1,
            "current_role": "Baa",
            "true_role": "Twin Minion",
            "disguise": "Baker",
            "is_evil": True,
            "alignment": "Evil",
            "state": "Dead",
            "killed_hidden": False,
            "statuses": [],
        }

        with redirect_stdout(StringIO()) as output:
            print_board([card])

        rendered = output.getvalue()
        self.assertIn("CURRENT ROLE", rendered)
        self.assertIn("#1 CURRENT ROLE=Baa", rendered)
        self.assertNotIn("TRUE ROLE", rendered)


class CurrentDeathHookTests(unittest.TestCase):
    def test_twin_swapped_witch_hooks_follow_current_not_origin(self):
        old_witch_body = GameSession(3, 2)
        old_witch_body.minions = ["Witch", "Twin Minion"]
        old_witch_body.blocked_positions = [3]
        old_witch_body.mark_executed(
            1,
            was_evil=True,
            evil_role="Witch",
            true_role="Twin Minion",
        )

        self.assertEqual(old_witch_body.blocked_positions, [3])
        self.assertFalse(old_witch_body.is_witch_known_dead())
        self.assertEqual(old_witch_body.executed_evil_roles, {1: "Witch"})
        self.assertEqual(
            old_witch_body.executed_current_roles,
            {1: "Twin_Minion"},
        )

        current_witch_body = GameSession(3, 2)
        current_witch_body.minions = ["Witch", "Twin Minion"]
        current_witch_body.blocked_positions = [3]
        current_witch_body.mark_executed(
            2,
            was_evil=True,
            evil_role="Twin Minion",
            true_role="Witch",
        )

        self.assertEqual(current_witch_body.blocked_positions, [])
        self.assertTrue(current_witch_body.is_witch_known_dead())
        self.assertEqual(
            current_witch_body.executed_evil_roles,
            {2: "Twin_Minion"},
        )
        self.assertEqual(current_witch_body.executed_current_roles, {2: "Witch"})

    def test_shaman_origin_only_witch_death_stays_unknown(self):
        session = GameSession(3, 2)
        session.minions = ["Witch", "Shaman"]
        session.blocked_positions = [3]

        session.mark_executed(1, was_evil=True, evil_role="Witch")

        self.assertEqual(session.blocked_positions, [3])
        self.assertFalse(session.is_witch_known_dead())

    def test_twin_swapped_baa_hooks_follow_current_not_origin(self):
        old_baa_body = GameSession(2, 2)
        old_baa_body.minions = ["Twin Minion"]
        old_baa_body.demons = ["Baa"]
        current_baa_body = GameSession(2, 2)
        current_baa_body.minions = ["Twin Minion"]
        current_baa_body.demons = ["Baa"]

        with patch("game_loop._baa_post_death_deck_refresh") as refresh:
            old_baa_body.mark_executed(
                1,
                was_evil=True,
                evil_role="Baa",
                true_role="Twin Minion",
            )
            refresh.assert_not_called()
            current_baa_body.mark_executed(
                2,
                was_evil=True,
                evil_role="Twin Minion",
                true_role="Baa",
            )
            refresh.assert_called_once_with(current_baa_body)

    def test_manual_moved_baa_uses_explicit_current_role_once(self):
        session = GameSession(2, 2)
        session.minions = ["Twin Minion"]
        session.demons = ["Baa"]

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution"),
            patch("game_loop._baa_post_death_deck_refresh") as refresh,
            redirect_stdout(StringIO()),
        ):
            reader_type.return_value.open.return_value = False
            dispatch(
                "execute",
                ["2", "evil", "Twin_Minion", "current=Baa"],
                session,
            )

        refresh.assert_called_once_with(session)
        self.assertEqual(session.executed_evil_roles, {2: "Twin_Minion"})
        self.assertEqual(session.executed_current_roles, {2: "Baa"})

    def test_manual_mover_execution_without_current_role_refuses(self):
        session = GameSession(2, 1)
        session.minions = ["Twin Minion"]

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save") as save,
            patch.object(DecisionLog, "log_execution") as log_execution,
            redirect_stdout(StringIO()) as output,
        ):
            reader_type.return_value.open.return_value = False
            dispatch("execute", ["1", "evil", "Twin_Minion"], session)

        self.assertIn("exact public death role was unavailable", output.getvalue())
        self.assertEqual(session.executed, [])
        self.assertEqual(session.executed_evil_roles, {})
        save.assert_not_called()
        log_execution.assert_not_called()

    def test_good_mover_execution_accepts_current_equals_syntax(self):
        session = GameSession(2, 1)
        session.minions = ["Twin Minion"]

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution"),
            redirect_stdout(StringIO()),
        ):
            reader_type.return_value.open.return_value = False
            dispatch("execute", ["1", "good", "current=Knight", "clean"], session)

        self.assertEqual(session.executed, [1])
        self.assertEqual(session.executed_current_roles, {1: "Knight"})
        self.assertEqual(session.executed_good_roles, {1: "Knight"})

    def test_moved_terminal_execute_needs_no_alignment_guess(self):
        session = GameSession(2, 1)
        session.minions = ["Twin Minion"]
        session.hp = 7

        with (
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution") as log_execution,
            redirect_stdout(StringIO()),
        ):
            dispatch("execute", ["1", "current=Bombardier"], session)

        self.assertEqual(session.terminal_loss_role, "Bombardier")
        self.assertEqual(session.executed, [1])
        self.assertEqual(session.executed_current_roles, {1: "Bombardier"})
        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.confirmed_good, [])
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.hp, 7)
        log_execution.assert_called_once_with(1, None, None)

    def test_twin_lilis_liveness_follows_current_death_role(self):
        old_lilis_body = GameSession(2, 2)
        old_lilis_body.minions = ["Twin Minion"]
        old_lilis_body.demons = ["Lilis"]
        old_lilis_body.mark_executed(
            1,
            was_evil=True,
            evil_role="Lilis",
            true_role="Twin Minion",
        )
        self.assertTrue(old_lilis_body.is_lilis_alive())

        current_lilis_body = GameSession(2, 2)
        current_lilis_body.minions = ["Twin Minion"]
        current_lilis_body.demons = ["Lilis"]
        current_lilis_body.mark_executed(
            2,
            was_evil=True,
            evil_role="Twin Minion",
            true_role="Lilis",
        )
        self.assertFalse(current_lilis_body.is_lilis_alive())

    def test_hidden_good_victim_releases_witch_probe_when_data_can_move(self):
        session = GameSession(3, 2)
        session.minions = ["Witch", "Twin Minion"]
        session.demons = ["Lilis"]
        session.blocked_positions = [1]
        session.pending_lilis_nights = 1

        session.record_lilis_night_result([3], 0)

        self.assertEqual(session.blocked_positions, [])
        self.assertEqual(session.night_kills, [3])
        self.assertEqual(session.night_kill_evil_count, 0)


class MediumCurrentDeathTests(unittest.TestCase):
    def test_existing_placeholder_is_promoted_and_hook_is_idempotent(self):
        session = GameSession(3, 1)
        session.night_kills = [2]
        session.cards = [CardInfo(2, "Unknown", info_parsed={})]
        medium = CardInfo(
            1,
            "Medium",
            info_parsed={"good_position": 2, "good_role": "Baa"},
        )

        with patch("game_loop._baa_post_death_deck_refresh") as refresh:
            session.add_card(medium)
            session.add_card(medium)

        target = next(card for card in session.cards if card.position == 2)
        self.assertEqual(target.apparent_role, "Baa")
        self.assertEqual(session.revealed_night_current_roles, {2: "Baa"})
        refresh.assert_called_once_with(session)

    def test_conflicting_non_placeholder_is_rejected_atomically(self):
        session = GameSession(3, 1)
        session.night_kills = [2]
        session.cards = [CardInfo(2, "Knight", info_parsed={})]
        medium = CardInfo(
            1,
            "Medium",
            info_parsed={"good_position": 2, "good_role": "Baa"},
        )

        with self.assertRaisesRegex(ValueError, "conflicts"):
            session.add_card(medium)

        self.assertEqual(session.cards, [CardInfo(2, "Knight", info_parsed={})])
        self.assertEqual(session.revealed_night_current_roles, {})
        self.assertEqual(session.reveal_order, [])


class LilisShamanSafetyTests(unittest.TestCase):
    @staticmethod
    def _session() -> GameSession:
        session = GameSession(4, 2)
        session.minions = ["Shaman"]
        session.demons = ["Lilis"]
        return session

    def test_flip_refuses_before_any_click(self):
        session = self._session()

        with (
            patch("game_loop._click_flip_card") as click,
            redirect_stdout(StringIO()) as output,
        ):
            dispatch("flip", [], session)

        self.assertIn("Lilis+Shaman", output.getvalue())
        click.assert_not_called()
        self.assertEqual(session.reveal_order, [])
        self.assertEqual(session.pending_lilis_nights, 0)

    def test_state_machine_flipping_pauses_before_dispatch(self):
        session = self._session()
        machine = GameStateMachine(session=session, monitor=None)
        machine.phase = GamePhase.FLIPPING

        with patch("game_loop.dispatch") as live_dispatch:
            machine._do_flipping()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("0, 1, or 2", machine._needs_human_reason)
        live_dispatch.assert_not_called()

    def test_night_bookkeeping_refuses_without_mutation(self):
        session = self._session()
        session.pending_lilis_nights = 1

        with redirect_stdout(StringIO()) as output:
            dispatch("night_kill", ["2", "0"], session)

        self.assertIn("actor count", output.getvalue())
        self.assertEqual(session.pending_lilis_nights, 1)
        self.assertEqual(session.night_kills, [])
        self.assertEqual(session.hp, 10)


class SlayerMovedIdentityTests(unittest.TestCase):
    @staticmethod
    def _session() -> GameSession:
        session = GameSession(3, 1)
        session.minions = ["Twin Minion"]
        session.cards = [CardInfo(1, "Slayer", info_parsed={})]
        return session

    def test_runtime_good_current_twin_cannot_produce_kill(self):
        session = self._session()

        with self.assertRaisesRegex(ValueError, "physical Good registered alignment"):
            session.add_slayer_result(
                1,
                2,
                True,
                revealed_role="Twin Minion",
                was_evil=False,
            )

        self.assertEqual(session.slayer_results, [])
        self.assertEqual(session.executed, [])

    def test_runtime_evil_twin_body_can_reveal_received_good_role(self):
        session = self._session()

        session.add_slayer_result(
            1,
            2,
            True,
            revealed_role="Knight",
            was_evil=True,
        )

        self.assertEqual(session.slayer_results, [{
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": True,
            "revealed_role": "Knight",
            "was_evil": True,
        }])
        self.assertEqual(session.confirmed_evil, [2])
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.hp, 10)

    def test_moved_nonterminal_kill_can_remain_alignment_unknown(self):
        session = self._session()

        session.add_slayer_result(
            1,
            2,
            True,
            revealed_role="Knight",
        )

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

    def test_moved_terminal_bombardier_keeps_alignment_unresolved(self):
        session = self._session()
        session.hp = 7

        session.add_slayer_result(1, 2, True, revealed_role="Bombardier")

        self.assertEqual(session.terminal_loss_role, "Bombardier")
        self.assertEqual(session.slayer_results, [{
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": True,
            "revealed_role": "Bombardier",
        }])
        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.confirmed_good, [])
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.hp, 7)

    def test_possible_moved_bomb_auto_execution_refuses_before_click(self):
        session = self._session()
        session.outcasts = ["Bombardier"]
        result = _result(Scenario(evil_positions={2: "Twin Minion"}))
        result.bombardier_positions = [2]

        with (
            patch("mouse.click") as mouse_click,
            patch("template_match.safe_click_at") as safe_click,
        ):
            observed = session.auto_execute(2, result)

        self.assertFalse(observed["success"])
        self.assertIn("Bombardier protection", observed["error"])
        mouse_click.assert_not_called()
        safe_click.assert_not_called()

    def test_twin_stop_precedes_stale_slayer_recommendation(self):
        session = self._session()
        session.outcasts = ["Bombardier"]
        result = _result(Scenario(evil_positions={2: "Twin Minion"}))
        result.bombardier_positions = [2]
        action = Action(
            action_type="use_ability",
            position=1,
            targets=[2],
            ability_name="Slayer",
        )

        with (
            patch.object(session, "_solve", return_value=result),
            patch(
                "game_loop.print_recommendation",
                return_value=action,
            ) as recommendation,
            patch.object(session, "auto_use_ability") as auto_ability,
            patch.object(DecisionLog, "log_solver_output"),
            patch.object(DecisionLog, "log_recommendation"),
            redirect_stdout(StringIO()) as output,
        ):
            _, _, observed = session.auto_next()

        self.assertIsNone(observed)
        self.assertIn("LIVE TWIN SAFETY STOP", output.getvalue())
        recommendation.assert_not_called()
        auto_ability.assert_not_called()


class StableOriginRecoveryTests(unittest.TestCase):
    def test_empty_worlds_never_fall_back_to_current_role(self):
        self.assertIsNone(
            _consensus_original_evil_role(1, _result(), "Bombardier")
        )

    def test_post_win_known_executed_origins_resolve(self):
        session = GameSession(3, 2)
        session.minions = ["Twin Minion"]
        session.demons = ["Pooka"]
        session.executed = [1, 3]
        session.executed_evil_roles = {1: "Twin Minion", 3: "Pooka"}
        result = _result(
            Scenario(evil_positions={1: "Twin Minion", 3: "Pooka"})
        )

        resolved, errors = _resolve_runtime_evil_origins(
            {1, 3}, session, result
        )

        self.assertEqual(errors, [])
        self.assertEqual(resolved, {1: "Twin_Minion", 3: "Pooka"})

    def test_unresolved_executed_origin_is_rejected(self):
        session = GameSession(2, 1)
        session.minions = ["Twin Minion"]
        session.executed = [1]
        result = _result(Scenario(evil_positions={1: "Unknown"}))

        resolved, errors = _resolve_runtime_evil_origins({1}, session, result)

        self.assertEqual(resolved, {})
        self.assertTrue(any("Unknown" in error for error in errors))

    def test_exact_runtime_seat_filter_and_puppet_scalar(self):
        session = GameSession(3, 2)
        session.minions = ["Puppeteer"]
        result = _result(
            Scenario(evil_positions={1: "Puppeteer"}, puppet_position=3),
            Scenario(evil_positions={1: "Puppeteer", 2: "Pooka"}),
        )

        resolved, errors = _resolve_runtime_evil_origins(
            {1, 3}, session, result
        )

        self.assertEqual(errors, [])
        self.assertEqual(resolved, {1: "Puppeteer", 3: "Puppet"})

    def test_stable_twin_and_generated_puppet_overlap_refuses_auto_origin(self):
        session = GameSession(2, 1)
        session.minions = ["Twin Minion", "Puppeteer"]
        result = _result(
            Scenario(
                evil_positions={1: "Twin Minion"},
                puppet_position=1,
            )
        )

        resolved, errors = _resolve_runtime_evil_origins({1}, session, result)

        self.assertEqual(resolved, {})
        self.assertTrue(any("generated Puppet" in error for error in errors))

    def test_validator_consumes_authored_role_multiplicity(self):
        session = GameSession(4, 2)
        session.minions = ["Twin Minion"]
        session.demons = ["Pooka"]

        cleaned, errors = _validate_true_evils_against_session(
            {1: "Pooka", 2: "Pooka"}, session
        )

        self.assertEqual(cleaned, {})
        self.assertTrue(any("claimed 2 time" in error for error in errors))

        duplicate_pool = GameSession(4, 2)
        duplicate_pool.demons = ["Pooka", "Pooka"]
        cleaned, errors = _validate_true_evils_against_session(
            {1: "Pooka", 2: "Pooka"}, duplicate_pool
        )
        self.assertEqual(errors, [])
        self.assertEqual(cleaned, {1: "Pooka", 2: "Pooka"})

    def test_validator_allows_alive_evils_and_rejects_current_contamination(self):
        session = GameSession(4, 2)
        session.minions = ["Twin Minion"]
        session.demons = ["Pooka"]
        session.executed = [1]
        session.executed_evil_roles = {1: "Twin Minion"}

        cleaned, errors = _validate_true_evils_against_session(
            {1: "Twin Minion", 3: "Pooka"}, session
        )
        self.assertEqual(errors, [])
        self.assertEqual(cleaned, {1: "Twin Minion", 3: "Pooka"})

        cleaned, errors = _validate_true_evils_against_session(
            {1: "Twin Minion", 3: "Bombardier"}, session
        )
        self.assertEqual(cleaned, {})
        self.assertTrue(any("permits 0" in error for error in errors))

    def test_game_over_uses_structured_seats_and_never_current_as_origin(self):
        session = GameSession(2, 1)
        session.minions = ["Twin Minion"]
        board = [{
            "position": 1,
            "is_evil": True,
            "current_role": "Bombardier",
            "true_role": "Bombardier",
        }]
        scenario_result = _result(
            Scenario(evil_positions={1: "Twin Minion"})
        )

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "_solve", return_value=scenario_result),
            patch("game_loop._save_and_run_test") as save_case,
            patch.object(DecisionLog, "log_game_over"),
            patch("scorecard.record"),
            patch(
                "subprocess.run",
                return_value=SimpleNamespace(returncode=0, stderr=""),
            ),
            redirect_stdout(StringIO()) as output,
        ):
            reader_type.return_value.open.return_value = True
            reader_type.return_value.read_board.return_value = board
            dispatch("game_over", ["win", "twin_origin_case"], session)

        save_case.assert_called_once_with(
            "twin_origin_case",
            {1: "Twin_Minion"},
            "",
        )
        self.assertIn("1=Twin_Minion", output.getvalue())
        self.assertNotIn("1=Bombardier", output.getvalue())


class TwinLiveSolverStopTests(unittest.TestCase):
    def test_plain_twin_solver_stops_before_rust(self):
        session = GameSession(3, 1)
        session.minions = ["Twin Minion"]

        with patch("game_loop.rust_solve_to_objects") as rust_solve:
            result = session._solve(session.to_game_state())

        self.assertEqual(result.n_surviving, 0)
        self.assertIn("Twin Minion solving is paused", result.reasoning[0])
        rust_solve.assert_not_called()

    def test_twin_scout_hunter_next_produces_no_recommendation(self):
        session = GameSession(3, 1)
        session.villagers = ["Scout", "Hunter"]
        session.minions = ["Twin Minion"]
        session.cards = [
            CardInfo(1, "Scout", info_parsed={"evil_role": "Twin Minion", "distance": 1}),
            CardInfo(2, "Hunter", info_parsed={"distance": 1}),
        ]

        with (
            patch("game_loop.rust_solve_to_objects") as rust_solve,
            patch("game_loop.print_recommendation") as recommendation,
            patch.object(DecisionLog, "log_solver_output"),
            patch.object(DecisionLog, "log_recommendation") as log_recommendation,
            redirect_stdout(StringIO()) as output,
        ):
            action = session.next_action()

        self.assertIsNone(action)
        self.assertIn("LIVE TWIN SAFETY STOP", output.getvalue())
        rust_solve.assert_not_called()
        recommendation.assert_not_called()
        log_recommendation.assert_not_called()

    def test_plain_twin_auto_next_cannot_execute_or_use_ability(self):
        session = GameSession(2, 1)
        session.minions = ["Twin Minion"]

        with (
            patch("game_loop.rust_solve_to_objects") as rust_solve,
            patch("game_loop.print_recommendation") as recommendation,
            patch.object(session, "auto_execute") as auto_execute,
            patch.object(session, "auto_use_ability") as auto_ability,
            patch.object(DecisionLog, "log_solver_output"),
            patch.object(DecisionLog, "log_recommendation") as log_recommendation,
            redirect_stdout(StringIO()),
        ):
            action, result, observed = session.auto_next()

        self.assertIsNone(action)
        self.assertEqual(result.n_surviving, 0)
        self.assertIsNone(observed)
        rust_solve.assert_not_called()
        recommendation.assert_not_called()
        auto_execute.assert_not_called()
        auto_ability.assert_not_called()
        log_recommendation.assert_not_called()

    def test_twin_state_machine_pauses_before_strategy(self):
        session = GameSession(2, 1)
        session.minions = ["Twin Minion"]
        machine = GameStateMachine(session=session)
        machine.phase = GamePhase.SOLVING

        with (
            patch("game_loop.rust_solve_to_objects") as rust_solve,
            patch("strategy.recommend_action") as recommendation,
            redirect_stdout(StringIO()),
        ):
            machine._do_solving()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("ordered current-data trace", machine._needs_human_reason)
        rust_solve.assert_not_called()
        recommendation.assert_not_called()

    def test_non_twin_solver_still_calls_rust(self):
        session = GameSession(2, 1)
        session.minions = ["Minion"]
        expected = _result(Scenario(evil_positions={1: "Minion"}))

        with patch(
            "game_loop.rust_solve_to_objects",
            return_value=expected,
        ) as rust_solve:
            observed = session._solve(session.to_game_state())

        self.assertIs(observed, expected)
        rust_solve.assert_called_once()


class HonorRuleNightTests(unittest.TestCase):

    def test_state_machine_does_not_import_hidden_victim_alignment(self):
        session = GameSession(2, 1)
        session.demons = ["Lilis"]
        session.pending_lilis_nights = 1

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return [{
                    "position": 2,
                    "state": "Hidden",
                    "killed_hidden": True,
                    "is_evil": True,
                }]

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine.phase = GamePhase.NIGHT_RESOLVE
        with patch.object(session, "save") as save:
            machine._do_night_resolve()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("validation-only", machine._needs_human_reason)
        self.assertEqual(session.night_kills, [])
        self.assertEqual(session.night_kill_evil_count, 0)
        self.assertEqual(session.pending_lilis_nights, 1)
        self.assertEqual(session.hp, 10)
        save.assert_not_called()


if __name__ == "__main__":
    unittest.main()
