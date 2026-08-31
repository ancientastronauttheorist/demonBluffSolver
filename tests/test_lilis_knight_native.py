"""Focused current-build Lilis night and apparent-Knight regressions."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from unittest.mock import patch

import rust_solver
from game_loop import CardInfo, GameSession, dispatch
from solver import (
    Alignment,
    ChancellorTrace,
    DeckComposition,
    GameState,
    Scenario,
    ShamanTrace,
    SolverResult,
    effective_alignment,
    effective_role_at,
)
from state_machine import GamePhase, GameStateMachine
from strategy import (
    _execution_reveal_outcome,
    _knight_check_damage_profile,
    recommend_action,
)


def _night_session(
    *,
    n_cards: int = 4,
    pending_nights: int = 1,
    hp: int = 10,
) -> GameSession:
    session = GameSession(n_cards, 1)
    session.demons = ["Lilis"]
    session.lilis_batch_index = pending_nights
    session.pending_lilis_nights = pending_nights
    session.hp = hp
    return session


def _result(
    *scenarios: Scenario,
    bombardier_positions: list[int] | None = None,
) -> SolverResult:
    return SolverResult(
        definite_evil=[],
        definite_good=[],
        bombardier_positions=list(bombardier_positions or []),
        n_scenarios=len(scenarios),
        n_surviving=len(scenarios),
        surviving_scenarios=list(scenarios),
    )


class LilisNightBookkeepingTests(unittest.TestCase):
    @staticmethod
    def _reload(session: GameSession) -> GameSession:
        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "session.json")
            with redirect_stdout(StringIO()):
                session.save(path)
                return GameSession.load(path)

    def test_batched_catch_up_consumes_one_pending_night_per_victim(self):
        session = _night_session(pending_nights=2)

        with patch.object(session, "save"), redirect_stdout(StringIO()):
            dispatch("night_kill", ["2,3", "1"], session)

        self.assertEqual(session.night_kills, [2, 3])
        self.assertEqual(session.night_kill_evil_count, 1)
        self.assertEqual(session.lilis_nights_resolved, 2)
        self.assertEqual(session.hp, 6)
        self.assertEqual(session.executed, [])
        self.assertEqual(session.confirmed_evil, [])

    def test_manual_no_kill_clamps_hp_and_makes_no_identity_inference(self):
        session = _night_session(n_cards=1, hp=1)
        session.blocked_positions = [1]

        with patch.object(session, "save"), redirect_stdout(StringIO()) as output:
            dispatch("night_no_kill", [], session)

        self.assertEqual(session.hp, 0)
        self.assertEqual(session.lilis_nights_resolved, 1)
        self.assertEqual(session.confirmed_evil, [])
        self.assertIn("No Lilis position can be inferred", output.getvalue())

    def test_duplicate_and_no_pending_results_reject_without_mutation(self):
        session = _night_session(n_cards=3)
        with patch.object(session, "save"), redirect_stdout(StringIO()):
            dispatch("night_kill", ["2", "0"], session)

        expected = (
            list(session.night_kills),
            session.night_kill_evil_count,
            session.lilis_nights_resolved,
            session.hp,
        )
        with patch.object(session, "save") as save, redirect_stdout(StringIO()):
            dispatch("night_kill", ["2", "0"], session)
            dispatch("night_kill", ["3", "0"], session)
            dispatch("night_no_kill", [], session)

        self.assertEqual(
            (
                session.night_kills,
                session.night_kill_evil_count,
                session.lilis_nights_resolved,
                session.hp,
            ),
            expected,
        )
        save.assert_not_called()

    def test_whole_batch_is_validated_before_any_mutation(self):
        session = _night_session(pending_nights=2)
        session.reveal_order = [4]
        expected = (list(session.night_kills), session.lilis_nights_resolved, session.hp)

        with self.assertRaisesRegex(ValueError, "already revealed"):
            session.record_lilis_night_result([2, 4], 0)
        self.assertEqual(
            (session.night_kills, session.lilis_nights_resolved, session.hp),
            expected,
        )

        with self.assertRaisesRegex(ValueError, "unique"):
            session.record_lilis_night_result([2, 2], 0)
        with self.assertRaisesRegex(ValueError, "within"):
            session.record_lilis_night_result([5], 0)
        with self.assertRaisesRegex(ValueError, "evil-victim count"):
            session.record_lilis_night_result([2], 2)
        self.assertEqual(
            (session.night_kills, session.lilis_nights_resolved, session.hp),
            expected,
        )

    def test_resolution_scalar_persists_and_legacy_load_infers_only_kills(self):
        session = _night_session(n_cards=3, pending_nights=3)
        session.record_lilis_night_result([2], 0)
        session.record_lilis_night_result([], 0)
        session.executed = [1]
        session.executed_evil_roles = {1: "Lilis"}
        session.record_lilis_post_death_night()

        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "session.json")
            with redirect_stdout(StringIO()):
                session.save(path)
                loaded = GameSession.load(path)

        self.assertEqual(loaded.lilis_batch_index, 3)
        self.assertEqual(loaded.lilis_nights_resolved, 3)
        self.assertEqual(loaded.pending_lilis_nights, 0)
        self.assertEqual(loaded.night_kills, [2])
        self.assertEqual(loaded.hp, 6)

        legacy_state = GameState(
            n_cards=3,
            n_evil=1,
            deck=DeckComposition([], [], [], ["Lilis"]),
            cards=[],
            night_kills=[2],
            hp=6,
        )
        legacy = GameSession.from_game_state(legacy_state, lilis_batch_index=2)
        self.assertEqual(legacy.lilis_batch_index, 2)
        self.assertEqual(legacy.lilis_nights_resolved, 1)
        self.assertEqual(legacy.pending_lilis_nights, 0)
        legacy_machine = GameStateMachine(session=legacy, monitor=None)
        with patch.object(legacy_machine, "_run_loop"), redirect_stdout(StringIO()):
            legacy_machine.start()
        self.assertNotEqual(legacy_machine.phase, GamePhase.LILIS_NIGHT)

    def test_state_machine_pauses_for_public_count_then_manual_resolution_matches(self):
        manual = _night_session(n_cards=3, hp=7)
        automatic = _night_session(n_cards=3, hp=7)

        with patch.object(manual, "save"), redirect_stdout(StringIO()):
            dispatch("night_kill", ["2", "1"], manual)

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
                    {"position": 3, "state": "Hidden", "killed_hidden": False},
                ]

        machine = GameStateMachine(session=automatic, monitor=Monitor())
        machine.phase = GamePhase.NIGHT_RESOLVE
        with patch.object(automatic, "save") as auto_save, redirect_stdout(StringIO()):
            machine._do_night_resolve()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertEqual(automatic.night_kills, [])
        auto_save.assert_not_called()
        with patch.object(automatic, "save"), redirect_stdout(StringIO()):
            dispatch("night_kill", ["2", "1"], automatic)

        self.assertEqual(automatic.night_kills, manual.night_kills)
        self.assertEqual(
            automatic.night_kill_evil_count,
            manual.night_kill_evil_count,
        )
        self.assertEqual(
            automatic.lilis_nights_resolved,
            manual.lilis_nights_resolved,
        )
        self.assertEqual(automatic.hp, manual.hp)
        self.assertEqual(automatic.executed, [])

    def test_state_machine_no_kill_does_not_confirm_lone_hidden_position(self):
        session = _night_session(n_cards=1)
        session.blocked_positions = [1]

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return [
                    {"position": 1, "state": "Hidden", "killed_hidden": False},
                ]

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine.phase = GamePhase.NIGHT_RESOLVE
        with patch.object(session, "save"), redirect_stdout(StringIO()) as output:
            machine._do_night_resolve()

        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.lilis_nights_resolved, 1)
        self.assertEqual(session.hp, 8)
        self.assertIn("No identity inference", output.getvalue())

    def test_fourth_single_reveal_waits_for_delayed_night_result(self):
        session = _night_session(n_cards=4, pending_nights=0)
        session.reveal_order = [1, 2, 3]

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return True

            @staticmethod
            def get_board():
                return [
                    {"position": 4, "state": "Alive", "killed_hidden": False},
                ]

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine._pending_reveal = (4,)
        with (
            patch("game_utils.all_game_card_coords", return_value={4: (100, 100)}),
            patch("template_match.fast_click_at"),
            patch.object(machine, "_auto_enter_single_card", return_value=True),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_revealing()

        self.assertEqual(machine.phase, GamePhase.LILIS_NIGHT)
        self.assertEqual(session.lilis_batch_index, 1)
        self.assertEqual(session.lilis_nights_resolved, 0)
        self.assertEqual(session.pending_lilis_nights, 1)
        self.assertEqual(session.hp, 10)

    def test_manual_fourth_reveal_syncs_zero_damage_night_after_lilis_death(self):
        session = _night_session(n_cards=4, pending_nights=0)
        session.executed = [1]
        session.executed_evil_roles = {1: "Lilis"}
        session.reveal_order = [1, 2, 3]
        session.cards = [CardInfo(2, "Judge", info_parsed={})]
        session.used_abilities = [2]

        class Monitor:
            @staticmethod
            def is_healthy():
                return False

        with (
            patch("game_utils.all_game_card_coords", return_value={4: (100, 100)}),
            patch("game_loop._click_flip_card", return_value=True),
            patch(
                "game_loop._read_board_once_for_flip",
                return_value=[{
                    "position": 4,
                    "state": "Alive",
                    "killed_hidden": False,
                }],
            ),
            patch("memory_reader.get_monitor", return_value=Monitor()),
            patch("game_loop.time.sleep"),
            patch.object(session, "save"),
            redirect_stdout(StringIO()) as output,
        ):
            dispatch("flip", ["4"], session)
            dispatch("night_no_kill", [], session)

        self.assertEqual(session.reveal_order, [1, 2, 3, 4])
        self.assertEqual(session.lilis_batch_index, 1)
        self.assertEqual(session.lilis_nights_resolved, 1)
        self.assertEqual(session.pending_lilis_nights, 0)
        self.assertEqual(session.hp, 10)
        self.assertEqual(session.used_abilities, [])
        self.assertIn("persistent Night rule", output.getvalue())
        self.assertIn("no HP damage", output.getvalue())

    def test_state_machine_batches_and_resolves_post_death_night_before_continuing(self):
        session = _night_session(n_cards=8, pending_nights=0)
        session.executed = [1]
        session.executed_evil_roles = {1: "Lilis"}
        session.cards = [CardInfo(2, "Judge", info_parsed={})]
        session.used_abilities = [2]

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
                    {
                        "position": position,
                        "state": "Alive" if position in {2, 3, 4, 5} else "Hidden",
                        "killed_hidden": False,
                        "is_evil": False,
                    }
                    for position in range(1, 9)
                ]

        def complete_batch(_command, args, live_session):
            self.assertEqual(args, ["--lilis"])
            live_session.reveal_order.extend([2, 3, 4, 5])
            live_session.schedule_lilis_night()

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine.phase = GamePhase.FLIPPING
        with (
            patch("game_loop.dispatch", side_effect=complete_batch),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_flipping()
            self.assertEqual(machine.phase, GamePhase.LILIS_NIGHT)
            machine._do_lilis_night()
            self.assertEqual(machine.phase, GamePhase.NIGHT_RESOLVE)
            machine._do_night_resolve()

        self.assertEqual(machine.phase, GamePhase.FLIPPING)
        self.assertEqual(session.lilis_batch_index, 1)
        self.assertEqual(session.lilis_nights_resolved, 1)
        self.assertEqual(session.pending_lilis_nights, 0)
        self.assertEqual(session.hp, 10)
        self.assertEqual(session.used_abilities, [])
        self.assertEqual(session.night_kills, [])

    def test_duplicate_lilis_pauses_before_live_reveal_or_result_mutation(self):
        session = _night_session(n_cards=4, pending_nights=1)
        session.demons = ["Lilis", "Lilis"]
        machine = GameStateMachine(session=session, monitor=None)
        machine.phase = GamePhase.FLIPPING

        with patch("game_loop.dispatch") as live_dispatch, redirect_stdout(StringIO()):
            machine._do_flipping()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("Duplicate Lilis", machine._needs_human_reason)
        live_dispatch.assert_not_called()
        with self.assertRaisesRegex(ValueError, "duplicate Lilis"):
            session.record_lilis_night_result([], 0)
        self.assertEqual(session.lilis_nights_resolved, 0)
        self.assertEqual(session.hp, 10)

    def test_reload_routes_pending_active_kill_before_any_other_work(self):
        session = _night_session(n_cards=3, pending_nights=0)
        session.schedule_lilis_night()
        loaded = self._reload(session)

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
                    {"position": 3, "state": "Hidden", "killed_hidden": False},
                ]

        machine = GameStateMachine(session=loaded, monitor=Monitor())
        with patch.object(machine, "_run_loop"), redirect_stdout(StringIO()):
            machine.start()
        self.assertEqual(machine.phase, GamePhase.LILIS_NIGHT)
        self.assertEqual(loaded.pending_lilis_nights, 1)

        machine.phase = GamePhase.NIGHT_RESOLVE
        with patch.object(loaded, "save") as save, redirect_stdout(StringIO()):
            machine._do_night_resolve()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertEqual(loaded.night_kills, [])
        save.assert_not_called()
        with patch.object(loaded, "save"), redirect_stdout(StringIO()):
            dispatch("night_kill", ["2", "0"], loaded)

        self.assertEqual(loaded.night_kills, [2])
        self.assertEqual(loaded.pending_lilis_nights, 0)
        self.assertEqual(loaded.hp, 8)

    def test_reload_resume_routes_pending_active_no_kill(self):
        session = _night_session(n_cards=2, pending_nights=0)
        session.schedule_lilis_night()
        loaded = self._reload(session)

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return [
                    {
                        "position": position,
                        "state": "Hidden",
                        "killed_hidden": False,
                    }
                    for position in (1, 2)
                ]

        machine = GameStateMachine(session=loaded, monitor=Monitor())
        machine.phase = GamePhase.NEEDS_HUMAN
        with patch.object(machine, "_run_loop"), redirect_stdout(StringIO()):
            machine.resume()
        self.assertEqual(machine.phase, GamePhase.LILIS_NIGHT)

        machine.phase = GamePhase.NIGHT_RESOLVE
        with patch.object(loaded, "save"), redirect_stdout(StringIO()):
            machine._do_night_resolve()

        self.assertEqual(loaded.night_kills, [])
        self.assertEqual(loaded.pending_lilis_nights, 0)
        self.assertEqual(loaded.hp, 8)

    def test_reload_routes_pending_post_death_empty_night_at_zero_damage(self):
        session = _night_session(n_cards=2, pending_nights=0)
        session.executed = [1]
        session.executed_evil_roles = {1: "Lilis"}
        session.schedule_lilis_night()
        loaded = self._reload(session)

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def get_board():
                return [
                    {"position": 1, "state": "Dead", "killed_hidden": False},
                    {"position": 2, "state": "Hidden", "killed_hidden": False},
                ]

        machine = GameStateMachine(session=loaded, monitor=Monitor())
        with patch.object(machine, "_run_loop"), redirect_stdout(StringIO()):
            machine.start()
        self.assertEqual(machine.phase, GamePhase.LILIS_NIGHT)

        machine.phase = GamePhase.NIGHT_RESOLVE
        with patch.object(loaded, "save"), redirect_stdout(StringIO()):
            machine._do_night_resolve()

        self.assertEqual(loaded.pending_lilis_nights, 0)
        self.assertEqual(loaded.lilis_nights_resolved, 1)
        self.assertEqual(loaded.hp, 10)


class ApparentKnightStrategyTests(unittest.TestCase):
    def _state(self, *, hp: int = 10) -> GameState:
        return GameState(
            n_cards=2,
            n_evil=1,
            deck=DeckComposition(["Knight"], ["Gravedigger"], [], ["Pooka"]),
            cards=[CardInfo(1, "Knight", info_parsed={})],
            hp=hp,
            wrong_exec_cost=5,
        )

    @staticmethod
    def _generated(role: str, *, corrupted: bool = False) -> Scenario:
        return Scenario(
            evil_positions={2: "Pooka"},
            corrupted={1} if corrupted else set(),
            chancellor_conversion=1,
            chancellor_trace=ChancellorTrace([2], 1, role),
        )

    def test_generated_outcast_at_knight_is_a_damaging_branch(self):
        state = self._state()
        result = _result(
            Scenario(evil_positions={1: "Pooka"}),
            self._generated("Gravedigger"),
        )

        self.assertEqual(
            _knight_check_damage_profile(1, result, state),
            (0.5, 2.5, 5),
        )

    def test_generated_drunk_uses_exact_knight_damage_hook(self):
        state = self._state()
        result = _result(self._generated("Drunk", corrupted=True))

        self.assertEqual(
            _knight_check_damage_profile(1, result, state),
            (1.0, 6.0, 6),
        )

    def test_clean_doppelganger_as_knight_is_protected(self):
        state = self._state()
        result = _result(Scenario(
            evil_positions={2: "Pooka"},
            doppelganger_position=1,
        ))

        self.assertEqual(
            _knight_check_damage_profile(1, result, state),
            (0.0, 0.0, 0),
        )

    def test_corrupted_doppelganger_as_knight_is_killable_for_nine_damage(self):
        state = self._state()
        result = _result(Scenario(
            evil_positions={2: "Pooka"},
            corrupted={1},
            doppelganger_position=1,
        ))

        self.assertEqual(
            _knight_check_damage_profile(1, result, state),
            (1.0, 9.0, 9),
        )

    def test_shaman_copied_knight_outranks_preserved_evil_identity(self):
        state = self._state()
        scenario = Scenario(
            evil_positions={1: "Pooka"},
            shaman_trace=ShamanTrace(
                source_position=2,
                target_position=1,
                copied_role="Knight",
                target_previous_roles=["Pooka"],
            ),
        )

        self.assertEqual(effective_role_at(1, scenario, state), "Knight")
        self.assertEqual(effective_alignment(1, scenario, state), Alignment.EVIL)
        self.assertEqual(
            _execution_reveal_outcome(1, scenario, state),
            ("Knight", True, False, False),
        )
        self.assertEqual(
            _knight_check_damage_profile(1, _result(scenario), state),
            (0.0, 0.0, 0),
        )

    def test_rust_bridge_preserves_shaman_current_role_trace(self):
        class _State:
            @staticmethod
            def to_dict():
                return {"bridge_case": "shaman_knight"}

        payload = {
            "definite_evil": [1],
            "definite_good": [],
            "bombardier_positions": [],
            "n_scenarios": 1,
            "n_surviving": 1,
            "reasoning": [],
            "surviving_scenarios": [{
                "evil_positions": {"1": "Pooka"},
                "shaman_trace": {
                    "source_position": 2,
                    "target_position": 1,
                    "copied_role": "Knight",
                    "target_previous_roles": ["Pooka"],
                },
            }],
        }

        rust_solver.clear_solver_cache()
        with patch.object(rust_solver, "rust_solve", return_value=payload):
            bridged = rust_solver.rust_solve_to_objects(_State())

        trace = bridged.surviving_scenarios[0].shaman_trace
        self.assertEqual(trace.source_position, 2)
        self.assertEqual(trace.target_position, 1)
        self.assertEqual(trace.copied_role, "Knight")
        self.assertEqual(trace.target_previous_roles, ["Pooka"])

    def test_low_generated_outcast_risk_is_not_called_a_free_check(self):
        state = self._state()
        result = _result(
            Scenario(evil_positions={1: "Pooka"}),
            Scenario(evil_positions={1: "Pooka"}),
            Scenario(evil_positions={1: "Pooka"}),
            self._generated("Gravedigger"),
        )

        action = recommend_action(state, result, [])

        self.assertEqual((action.action_type, action.position), ("execute", 1))
        self.assertIn("25% damaging-outcome risk", action.reasoning)
        self.assertNotIn("free check", action.reasoning)

    def test_equal_probability_knights_prefer_the_free_native_branch(self):
        state = GameState(
            n_cards=2,
            n_evil=1,
            deck=DeckComposition(["Knight", "Knight"], [], [], ["Pooka"]),
            cards=[
                CardInfo(1, "Knight", info_parsed={}),
                CardInfo(2, "Knight", info_parsed={}),
            ],
            hp=10,
            wrong_exec_cost=5,
        )
        first_evil = Scenario(evil_positions={1: "Pooka"})
        second_evil = Scenario(evil_positions={2: "Pooka"}, corrupted={1})

        action = recommend_action(
            state,
            _result(first_evil, second_evil),
            [],
        )

        self.assertEqual((action.action_type, action.position), ("execute", 2))
        self.assertIn("Knight free check", action.reasoning)

    def test_possible_bombardier_is_excluded_from_knight_check(self):
        state = self._state()
        result = _result(
            Scenario(evil_positions={1: "Pooka"}),
            self._generated("Bombardier"),
            bombardier_positions=[1],
        )

        action = recommend_action(state, result, [])

        self.assertNotEqual((action.action_type, action.position), ("execute", 1))


if __name__ == "__main__":
    unittest.main()
