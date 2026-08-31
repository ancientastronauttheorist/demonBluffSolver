"""Native Bombardier terminal-loss parity across planning and live state."""

import tempfile
import unittest
from contextlib import redirect_stdout
from io import StringIO
from unittest.mock import patch

from game_loop import (
    CardInfo,
    DecisionLog,
    GameSession,
    _release_session_lock,
    dispatch,
)
from solver import (
    ChancellorTrace,
    DeckComposition,
    GameState,
    Scenario,
    ShamanTrace,
    SolverResult,
)
from state_machine import GamePhase, GameStateMachine
from strategy import (
    Action,
    _execution_observation_key,
    _execution_reveal_outcome,
    _find_forced_execution,
    _is_terminal_loss_role,
    _public_terminal_loss_position,
    _recommend_slayer,
    _scenario_terminal_loss_position,
    _shallow_lookahead,
    ordinary_execution_bombardier_positions,
    recommend_action,
)


def _state(n_cards=3, cards=None, **overrides):
    values = {
        "n_cards": n_cards,
        "n_evil": 1,
        "deck": DeckComposition([], [], ["Minion"], []),
        "cards": cards or [],
    }
    values.update(overrides)
    return GameState(**values)


def _result(*scenarios):
    worlds = list(scenarios)
    return SolverResult(
        definite_evil=[],
        definite_good=[],
        bombardier_positions=[],
        n_scenarios=len(worlds),
        n_surviving=len(worlds),
        surviving_scenarios=worlds,
    )


class BombardierPlanningTests(unittest.TestCase):
    def test_real_and_chancellor_current_bombardier_are_terminal(self):
        ordinary = _state(
            cards=[CardInfo(1, "Bombardier")],
            executed=[1],
        )
        ordinary_world = Scenario(evil_positions={})
        self.assertEqual(
            _scenario_terminal_loss_position(ordinary, ordinary_world), 1
        )
        self.assertEqual(
            recommend_action(ordinary, _result(ordinary_world), []).action_type,
            "loss",
        )

        generated = _state(executed=[2])
        generated_world = Scenario(
            evil_positions={},
            chancellor_trace=ChancellorTrace(
                original_positions=[3],
                added_outcast_position=2,
                added_outcast_role="Bombardier",
            ),
        )
        self.assertEqual(
            _scenario_terminal_loss_position(generated, generated_world), 2
        )

    def test_runtime_evil_shaman_current_bombardier_loses_before_alignment(self):
        state = _state(cards=[CardInfo(2, "Minion")])
        world = Scenario(
            evil_positions={2: "Minion"},
            shaman_trace=ShamanTrace(3, 2, "Bombardier"),
        )

        outcome = _execution_reveal_outcome(2, world, state)
        self.assertEqual(outcome[:2], ("Bombardier", True))
        self.assertEqual(
            _execution_observation_key(2, outcome, state),
            ("bombardier_loss",),
        )
        self.assertIsNone(_find_forced_execution(state, _result(world), [2]))

    def test_definite_evil_current_bombardier_is_never_recommended(self):
        state = _state(
            n_cards=2,
            cards=[CardInfo(1, "Minion"), CardInfo(2, "Bombardier")],
        )
        world = Scenario(
            evil_positions={1: "Minion"},
            shaman_trace=ShamanTrace(2, 1, "Bombardier"),
        )
        result = _result(world)
        result.definite_evil = [1]
        result.definite_good = [2]
        result.bombardier_positions = [1]

        action = recommend_action(state, result, [])
        self.assertFalse(
            action.action_type == "execute" and action.position == 1,
            action,
        )

    def test_opaque_aggregate_bomb_remains_unsafe_in_all_lookahead(self):
        state = _state(
            n_cards=2,
            cards=[CardInfo(1, "Minion")],
        )
        result = _result(Scenario(evil_positions={1: "Minion"}))
        # The aggregate collector is deliberately authoritative here even
        # though this reduced world lacks the ordered mover trace that made
        # the seat a possible current-data Bombardier.
        result.bombardier_positions = [1]

        self.assertIsNone(_find_forced_execution(state, result, [1]))
        self.assertIsNone(_shallow_lookahead(state, result, [1]))

        # With the other card also revealed, recommend_action reaches its
        # forced-execution candidate construction. It must still refuse #1,
        # rather than reviving the stale "confirmed safe Bombardier" path.
        state.cards.append(CardInfo(2, "Scout"))
        action = recommend_action(state, result, [])
        self.assertFalse(
            action.action_type == "execute" and action.position == 1,
            action,
        )

    def test_hidden_natural_good_bomb_is_root_unsafe_but_branch_local(self):
        state = GameState(
            n_cards=2,
            n_evil=1,
            deck=DeckComposition(
                ["Scout"], ["Bombardier"], ["Witch"], [],
            ),
            cards=[CardInfo(2, "Scout")],
            blocked_positions=[1],
            hp=10,
            wrong_exec_cost=5,
        )
        worlds = [
            Scenario(evil_positions={1: "Witch"}),
            Scenario(evil_positions={2: "Witch"}),
        ]
        result = _result(*worlds)
        # Rust now aggregates this anonymous natural-Outcast possibility. Keep
        # the regression valid for both the old empty collector and the new
        # precise collector, and prove neither form becomes permanently opaque.
        result.bombardier_positions = [1]

        self.assertEqual(
            ordinary_execution_bombardier_positions(state, result), {1}
        )
        self.assertEqual(_find_forced_execution(state, result, [1, 2]), 2)

        action = recommend_action(state, result, [])
        self.assertEqual((action.action_type, action.position), ("execute", 2))
        self.assertTrue(action.forced_safe)

        # After the #2-Good branch, #1 is runtime Evil in every surviving
        # world. A fresh aggregate no longer contains #1 because the absent
        # natural Good identity is then impossible.
        narrowed = _result(worlds[0])
        self.assertEqual(_find_forced_execution(state, narrowed, [1]), 1)

    def test_public_role_rules_out_anonymous_natural_bomb(self):
        state = GameState(
            n_cards=2,
            n_evil=1,
            deck=DeckComposition(
                ["Scout", "Hunter"], ["Bombardier"], ["Witch"], [],
            ),
            cards=[CardInfo(1, "Hunter"), CardInfo(2, "Scout")],
        )
        result = _result(
            Scenario(evil_positions={1: "Witch"}),
            Scenario(evil_positions={2: "Witch"}),
        )

        self.assertEqual(
            ordinary_execution_bombardier_positions(state, result), set()
        )

    def test_hidden_natural_good_bomb_remains_safe_for_slayer(self):
        state = GameState(
            n_cards=3,
            n_evil=1,
            deck=DeckComposition(
                ["Slayer", "Scout"], ["Bombardier"], ["Witch"], [],
            ),
            cards=[CardInfo(1, "Slayer"), CardInfo(3, "Scout")],
            blocked_positions=[2],
        )
        result = _result(
            Scenario(evil_positions={2: "Witch"}),
            Scenario(evil_positions={3: "Witch"}),
        )

        self.assertEqual(
            ordinary_execution_bombardier_positions(state, result), {2}
        )
        recommendation = _recommend_slayer(1, state, result)
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.targets, [2])

    def test_bluff_drunk_doppel_and_public_saint_are_not_terminal_roles(self):
        state = _state(cards=[CardInfo(2, "Bombardier")])
        worlds = [
            Scenario(evil_positions={2: "Minion"}),
            Scenario(evil_positions={}, drunk_position=2),
            Scenario(evil_positions={}, doppelganger_position=2),
        ]
        revealed = []
        for world in worlds:
            outcome = _execution_reveal_outcome(2, world, state)
            revealed.append(outcome[0])
            self.assertNotEqual(
                _execution_observation_key(2, outcome, state)[0],
                "bombardier_loss",
            )

        self.assertEqual(revealed, ["Minion", "Drunk", "Doppelganger"])
        self.assertFalse(_is_terminal_loss_role("Saint"))
        session = GameSession(1, 0)
        session.mark_executed(1, was_evil=False, true_role="Saint")
        self.assertIsNone(session.terminal_loss_role)
        self.assertEqual(session.executed_current_roles, {1: "Saint"})
        bluff_session = GameSession(1, 1)
        bluff_session.mark_executed(
            1,
            was_evil=True,
            evil_role="Minion",
            true_role="Minion",
        )
        self.assertIsNone(bluff_session.terminal_loss_role)
        self.assertEqual(bluff_session.executed_current_roles, {1: "Minion"})
        loaded = GameSession.from_game_state(
            _state(n_cards=1, n_evil=0, terminal_loss_role="Saint")
        )
        self.assertIsNone(loaded.terminal_loss_role)

    def test_night_kill_is_exempt_and_explicit_loss_precedes_hp_and_win(self):
        night = _state(
            n_cards=1,
            cards=[CardInfo(1, "Bombardier")],
            executed=[1],
            night_kills=[1],
            n_evil=0,
        )
        world = Scenario(evil_positions={})
        self.assertIsNone(_scenario_terminal_loss_position(night, world))
        self.assertEqual(recommend_action(night, _result(world), []).action_type, "win")

        terminal = _state(
            n_cards=1,
            n_evil=0,
            hp=0,
            terminal_loss_role="Bombardier",
        )
        action = recommend_action(terminal, _result(world), [])
        self.assertEqual(action.action_type, "loss")
        self.assertIn("Bombardier", action.reasoning)

    def test_public_death_evidence_precedes_zero_scenario_data_error(self):
        no_worlds = _result()
        executed = _state(
            n_cards=1,
            executed=[1],
            executed_good_roles={1: "Bombardier"},
        )
        self.assertEqual(_public_terminal_loss_position(executed), 1)
        self.assertEqual(
            recommend_action(executed, no_worlds, []).action_type,
            "loss",
        )

        slayer = _state(
            n_cards=1,
            executed=[1],
            slayer_results=[{
                "slayer_pos": 2,
                "target_pos": 1,
                "killed": True,
                "revealed_role": "Bombardier",
            }],
        )
        self.assertEqual(_public_terminal_loss_position(slayer), 1)
        self.assertEqual(
            recommend_action(slayer, no_worlds, []).action_type,
            "loss",
        )

        for safe_state in (
            _state(
                n_cards=1,
                executed=[1],
                executed_good_roles={1: "Saint"},
            ),
            _state(
                n_cards=1,
                executed=[1],
                night_kills=[1],
                executed_good_roles={1: "Bombardier"},
            ),
        ):
            self.assertIsNone(_public_terminal_loss_position(safe_state))
            self.assertEqual(
                recommend_action(safe_state, no_worlds, []).action_type,
                "error",
            )

    def test_slayer_public_saint_suppresses_inconsistent_bombardier_world(self):
        state = _state(
            n_cards=1,
            cards=[CardInfo(1, "Bombardier")],
            executed=[1],
            executed_good_roles={1: "Bombardier"},
            slayer_results=[{
                "slayer_pos": 2,
                "target_pos": 1,
                "killed": True,
                "revealed_role": "Saint",
            }],
        )
        world = Scenario(evil_positions={})

        self.assertIsNone(_public_terminal_loss_position(state))
        self.assertIsNone(_scenario_terminal_loss_position(state, world))

    def test_ordinary_current_role_is_authoritative_for_evil_death(self):
        world = Scenario(
            evil_positions={1: "Shaman"},
            shaman_trace=ShamanTrace(2, 1, "Bombardier"),
        )
        negative = _state(
            n_cards=2,
            executed=[1],
            confirmed_evil=[1],
            executed_evil_roles={1: "Shaman"},
            executed_current_roles={1: "Scout"},
        )
        self.assertIsNone(_public_terminal_loss_position(negative))
        self.assertIsNone(_scenario_terminal_loss_position(negative, world))

        positive = _state(
            n_cards=2,
            executed=[1],
            confirmed_evil=[1],
            executed_evil_roles={1: "Shaman"},
            executed_current_roles={1: "Bombardier"},
        )
        self.assertEqual(_public_terminal_loss_position(positive), 1)
        self.assertEqual(
            recommend_action(positive, _result(), []).action_type,
            "loss",
        )

    def test_slayer_planner_skips_runtime_evil_current_bombardier(self):
        state = _state(
            cards=[
                CardInfo(1, "Slayer"),
                CardInfo(2, "Minion"),
                CardInfo(3, "Bombardier"),
            ]
        )
        world = Scenario(
            evil_positions={2: "Minion"},
            shaman_trace=ShamanTrace(3, 2, "Bombardier"),
        )
        self.assertIsNone(_recommend_slayer(1, state, _result(world)))

    def test_slayer_planner_honors_aggregate_moved_bombardier_collector(self):
        state = _state(
            n_cards=2,
            cards=[CardInfo(1, "Slayer"), CardInfo(2, "Twin Minion")],
        )
        result = _result(Scenario(evil_positions={2: "Twin Minion"}))
        result.bombardier_positions = [2]

        # The reduced exact world lacks the earlier current-data trace, so its
        # local effective role is non-Bomb. The aggregate collector remains
        # authoritative and must suppress the Slayer top pick itself.
        self.assertIsNone(_recommend_slayer(1, state, result))


class BombardierLiveTests(unittest.TestCase):
    def test_session_marker_round_trips_and_slayer_sets_it_before_alignment(self):
        session = GameSession(3, 1)
        session.cards = [CardInfo(1, "Slayer")]
        session.add_slayer_result(
            1, 2, True, revealed_role="Bombardier", was_evil=True
        )

        self.assertEqual(session.terminal_loss_role, "Bombardier")
        self.assertEqual(session.confirmed_evil, [2])
        self.assertEqual(session.executed_evil_roles, {})
        self.assertEqual(session.executed_current_roles, {})
        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/session.json"
            session.save(path)
            loaded = GameSession.load(path)
            _release_session_lock()
        self.assertEqual(loaded.terminal_loss_role, "Bombardier")

        good_session = GameSession(3, 1)
        good_session.hp = 7
        good_session.cards = [CardInfo(1, "Slayer")]
        good_session.add_slayer_result(
            1,
            2,
            True,
            revealed_role="Bombardier",
            was_evil=False,
            was_corrupted=False,
        )
        self.assertEqual(good_session.terminal_loss_role, "Bombardier")
        # Explicit Good represents a visible positive HP delta. Native resource
        # handling happens before Bombardier's delayed terminal callback.
        self.assertEqual(good_session.hp, 2)

    def test_slayer_cli_accepts_terminal_bomb_without_runtime_alignment(self):
        session = GameSession(3, 1)
        session.hp = 7
        session.cards = [CardInfo(1, "Slayer")]

        with (
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_slayer_result"),
            redirect_stdout(StringIO()),
        ):
            dispatch(
                "slayer_result",
                ["1", "2", "kill", "Bombardier"],
                session,
            )

        self.assertEqual(session.terminal_loss_role, "Bombardier")
        self.assertEqual(session.slayer_results, [{
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": True,
            "revealed_role": "Bombardier",
        }])
        self.assertEqual(session.executed, [2])
        self.assertEqual(session.confirmed_evil, [])
        self.assertEqual(session.confirmed_good, [])
        self.assertEqual(session.hp, 7)

    def test_slayer_records_public_saint_names_without_terminal_alias(self):
        for public_role in ("Saint", "SaintVillager"):
            with self.subTest(public_role=public_role):
                session = GameSession(3, 1)
                session.hp = 7
                session.cards = [CardInfo(1, "Slayer")]
                with (
                    patch.object(session, "save"),
                    patch.object(DecisionLog, "log_slayer_result"),
                    redirect_stdout(StringIO()),
                ):
                    dispatch(
                        "slayer_result",
                        ["1", "2", "kill", public_role],
                        session,
                    )

                self.assertEqual(
                    session.slayer_results[-1]["revealed_role"],
                    public_role,
                )
                self.assertEqual(session.executed, [2])
                self.assertIsNone(session.terminal_loss_role)
                self.assertEqual(session.confirmed_evil, [])
                self.assertEqual(session.confirmed_good, [])
                self.assertEqual(session.hp, 7)

    def test_manual_evil_execution_keeps_original_role_and_exact_current_role(self):
        session = GameSession(2, 1)
        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution"),
            redirect_stdout(StringIO()),
        ):
            reader_type.return_value.open.return_value = False
            dispatch(
                "execute",
                ["2", "evil", "Minion", "current=Bombardier"],
                session,
            )

        self.assertEqual(session.executed_evil_roles, {2: "Minion"})
        self.assertEqual(session.executed_current_roles, {2: "Bombardier"})
        self.assertEqual(session.terminal_loss_role, "Bombardier")

    def test_manual_public_saint_names_never_set_terminal(self):
        for current_role in ("Saint", "SaintVillager"):
            with self.subTest(current_role=current_role):
                session = GameSession(1, 1)
                with (
                    patch("memory_reader.MemoryReader") as reader_type,
                    patch.object(session, "save"),
                    patch.object(DecisionLog, "log_execution"),
                    redirect_stdout(StringIO()),
                ):
                    reader_type.return_value.open.return_value = False
                    dispatch(
                        "execute",
                        ["1", "evil", "Minion", f"current={current_role}"],
                        session,
                    )

                self.assertIsNone(session.terminal_loss_role)
                self.assertEqual(
                    session.executed_current_roles,
                    {1: current_role},
                )

    def test_auto_execution_uses_world_for_original_evil_and_memory_for_current(self):
        session = GameSession(3, 1)
        session.hp = 7
        session.cards = [CardInfo(2, "Minion")]
        world = Scenario(
            evil_positions={2: "Minion"},
            shaman_trace=ShamanTrace(3, 2, "Bombardier"),
        )
        result = _result(world)
        dead = {
            "position": 2,
            "true_role": "Bombardier",
            "disguise": "Minion",
            "is_evil": True,
            "state": "Dead",
            "statuses": [],
        }

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return True

            @staticmethod
            def get_board():
                return [dead]

        with (
            patch("game_utils.all_game_card_coords", return_value={2: (100, 100)}),
            patch("template_match.safe_click_at"),
            patch("mouse.click"),
            patch("game_loop.time.sleep"),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution"),
            redirect_stdout(StringIO()),
        ):
            observed = session.auto_execute(2, result, monitor=Monitor())

        self.assertTrue(observed["success"])
        self.assertEqual(observed["evil_role"], "Minion")
        self.assertEqual(session.executed_evil_roles, {2: "Minion"})
        self.assertEqual(session.executed_current_roles, {2: "Bombardier"})
        self.assertEqual(session.terminal_loss_role, "Bombardier")
        self.assertEqual(session.hp, 7)
        self.assertTrue(observed["was_evil"])

    def test_auto_public_saint_never_sets_terminal(self):
        session = GameSession(1, 1)
        session.cards = [CardInfo(1, "Minion")]
        result = _result(Scenario(evil_positions={1: "Minion"}))
        dead = {
            "position": 1,
            "true_role": "Saint",
            "disguise": "Minion",
            "is_evil": True,
            "state": "Dead",
            "statuses": [],
        }

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return True

            @staticmethod
            def get_board():
                return [dead]

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
        self.assertIsNone(session.terminal_loss_role)
        self.assertEqual(session.executed_current_roles, {1: "Saint"})

    def test_auto_good_bombardier_pays_base_cost_before_terminal(self):
        session = GameSession(1, 0)
        session.hp = 7
        session.cards = [CardInfo(1, "Bombardier")]
        result = _result(Scenario(evil_positions={}))
        dead = {
            "position": 1,
            "true_role": "Bombardier",
            "disguise": None,
            "is_evil": False,
            "state": "Dead",
            "statuses": [],
        }

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return True

            @staticmethod
            def get_board():
                return [dead]

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
        self.assertEqual(session.terminal_loss_role, "Bombardier")
        self.assertFalse(observed["was_evil"])
        self.assertEqual(session.hp, 2)

    def test_auto_good_bombardier_no_damage_suppresses_cost_before_terminal(self):
        session = GameSession(1, 0)
        session.hp = 7
        session.cards = [CardInfo(1, "Bombardier")]
        result = _result(Scenario(evil_positions={}))
        dead = {
            "position": 1,
            "current_role": "Bombardier",
            "true_role": "Bombardier",
            "disguise": None,
            "is_evil": False,
            "state": "Dead",
            "statuses": ["NoDamage"],
        }

        class Monitor:
            @staticmethod
            def is_healthy():
                return True

            @staticmethod
            def wait_for(_predicate, timeout, min_delay):
                return True

            @staticmethod
            def get_board():
                return [dead]

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
        self.assertFalse(observed["was_evil"])
        self.assertEqual(session.terminal_loss_role, "Bombardier")
        self.assertEqual(session.hp, 7)

    def test_manual_offline_good_bombardier_requires_hp_sync(self):
        session = GameSession(1, 0)
        session.hp = 7
        session.cards = [CardInfo(1, "Bombardier")]

        with (
            patch("memory_reader.MemoryReader") as reader_type,
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_execution"),
            redirect_stdout(StringIO()) as output,
        ):
            reader_type.return_value.open.return_value = False
            dispatch(
                "execute",
                ["1", "good", "clean", "current=Bombardier"],
                session,
            )

        self.assertEqual(session.terminal_loss_role, "Bombardier")
        self.assertEqual(session.confirmed_good, [1])
        self.assertEqual(session.hp, 7)
        self.assertIn("HP outcome unresolved", output.getvalue())
        self.assertIn("set_hp <current_hp>", output.getvalue())

    def test_auto_execution_refuses_definite_evil_current_bombardier(self):
        session = GameSession(2, 1)
        result = _result(Scenario(evil_positions={1: "Minion"}))
        result.definite_evil = [1]
        result.bombardier_positions = [1]

        with (
            patch("mouse.click") as click,
            patch("template_match.safe_click_at") as click_at,
        ):
            observed = session.auto_execute(1, result, forced_safe=True)

        self.assertFalse(observed["success"])
        self.assertIn("Bombardier protection", observed["error"])
        click.assert_not_called()
        click_at.assert_not_called()

    def test_live_paths_refuse_hidden_anonymous_natural_bomb(self):
        session = GameSession(2, 1)
        session.villagers = ["Scout"]
        session.outcasts = ["Bombardier"]
        session.minions = ["Witch"]
        session.cards = [CardInfo(2, "Scout")]
        session.blocked_positions = [1]
        result = _result(
            Scenario(evil_positions={1: "Witch"}),
            Scenario(evil_positions={2: "Witch"}),
        )
        # Exercise the Python fallback independently of the Rust aggregate.
        self.assertEqual(result.bombardier_positions, [])

        with (
            patch("mouse.click") as click,
            patch("template_match.safe_click_at") as click_at,
        ):
            observed = session.auto_execute(1, result, forced_safe=True)

        self.assertFalse(observed["success"])
        self.assertIn("Bombardier protection", observed["error"])
        click.assert_not_called()
        click_at.assert_not_called()
        self.assertEqual(session.executed, [])

        stale_action = Action(
            "execute",
            position=1,
            reasoning="stale forced proof",
            forced_safe=True,
        )
        stale_action.confidence = 1.0
        with (
            patch.object(session, "_solve", return_value=result),
            patch("game_loop.print_recommendation", return_value=stale_action),
            patch.object(session, "auto_execute") as auto_execute,
            patch.object(DecisionLog, "log_solver_output"),
            patch.object(DecisionLog, "log_recommendation"),
            redirect_stdout(StringIO()),
        ):
            _, _, execution = session.auto_next()

        self.assertIsNone(execution)
        auto_execute.assert_not_called()

        machine = GameStateMachine(session=session)
        machine.phase = GamePhase.SOLVING
        with (
            patch.object(session, "_solve", return_value=result),
            patch("strategy.recommend_action", return_value=stale_action),
            redirect_stdout(StringIO()),
        ):
            machine._do_solving()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIsNone(machine._pending_exec)
        self.assertEqual(session.executed, [])

    def test_forced_safe_cannot_bypass_auto_or_state_machine_guards(self):
        session = GameSession(2, 1)
        result = _result(Scenario(evil_positions={1: "Shaman"}))
        result.definite_evil = [1]
        result.bombardier_positions = [1]
        action = Action(
            "execute",
            position=1,
            reasoning="stale forced proof",
            forced_safe=True,
        )
        action.confidence = 1.0

        with (
            patch.object(session, "_solve", return_value=result),
            patch("game_loop.print_recommendation", return_value=action),
            patch.object(session, "auto_execute") as auto_execute,
            patch.object(DecisionLog, "log_solver_output"),
            patch.object(DecisionLog, "log_recommendation"),
            redirect_stdout(StringIO()),
        ):
            _, _, execution = session.auto_next()

        self.assertIsNone(execution)
        auto_execute.assert_not_called()

        machine = GameStateMachine(session=session)
        machine.phase = GamePhase.SOLVING
        with (
            patch.object(session, "_solve", return_value=result),
            patch("strategy.recommend_action", return_value=action),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_solving()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIsNone(machine._pending_exec)

    def test_solve_diagnostics_stop_after_derived_terminal_loss(self):
        session = GameSession(2, 1)
        session.executed = [2]
        session.executed_current_roles = {2: "Bombardier"}
        result = _result()
        result.definite_evil = [1]
        result.bombardier_positions = [1]

        output = StringIO()
        with patch.object(session, "_solve", return_value=result), redirect_stdout(output):
            session.solve()

        text = output.getvalue()
        self.assertIn(">> TERMINAL LOSS", text)
        self.assertNotIn(">> EXECUTE", text)
        self.assertNotIn("NO VALID SCENARIOS", text)

    def test_state_machine_stops_after_auto_execution_and_preserves_hp(self):
        session = GameSession(2, 1)
        session.hp = 7

        def execute(*_args, **_kwargs):
            session.mark_executed(
                2,
                was_evil=True,
                evil_role="Minion",
                true_role="Bombardier",
            )
            return {
                "success": True,
                "blocked": False,
                "was_evil": True,
                "evil_role": "Minion",
                "error": None,
            }

        machine = GameStateMachine(session=session)
        machine._pending_exec = (2, _result(Scenario({2: "Minion"})), False)
        machine.phase = GamePhase.EXECUTING
        with (
            patch.object(session, "auto_execute", side_effect=execute),
            patch.object(session, "save") as save,
            redirect_stdout(StringIO()),
        ):
            machine._do_executing()

        self.assertEqual(machine.phase, GamePhase.GAME_OVER)
        self.assertEqual(machine._game_result, "loss")
        self.assertEqual(session.hp, 7)
        self.assertEqual(session.executed, [2])
        save.assert_called_once_with()

    def test_state_machine_persists_inferred_loss_and_game_over_keeps_loss(self):
        session = GameSession(1, 0)
        session.hp = 7
        session.executed = [1]
        session.executed_good_roles = {1: "Bombardier"}
        result = _result()
        machine = GameStateMachine(session=session)
        machine.phase = GamePhase.SOLVING

        with (
            patch.object(session, "_solve", return_value=result),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_solving()
            self.assertEqual(machine.phase, GamePhase.GAME_OVER)
            self.assertEqual(machine._game_result, "loss")
            self.assertEqual(session.terminal_loss_role, "Bombardier")
            machine._do_game_over()

        self.assertEqual(machine._game_result, "loss")
        self.assertEqual(session.hp, 7)
        self.assertEqual(machine.phase, GamePhase.POST_GAME)

    def test_state_machine_hp_loss_is_not_mislabeled_terminal(self):
        session = GameSession(1, 1)
        session.hp = 0
        machine = GameStateMachine(session=session)
        machine._game_result = "loss"

        output = StringIO()
        with patch.object(session, "save"), redirect_stdout(output):
            machine._do_game_over()

        self.assertEqual(machine._game_result, "loss")
        self.assertIn("HP depleted", output.getvalue())
        self.assertNotIn("HP retained", output.getvalue())


class BombardierAnalysisBridgeTests(unittest.TestCase):
    @staticmethod
    def _fake_result(*, definite_evil=None):
        world = Scenario(evil_positions={3: "Minion"} if definite_evil else {})
        result = _result(world)
        result.definite_evil = list(definite_evil or [])
        return result

    def test_final_marker_is_timed_and_analysis_result_is_loss(self):
        import decision_analysis
        import replay_analysis

        case = {
            "name": "terminal_timing",
            "n_cards": 1,
            "n_evil": 0,
            "deck": {"villagers": ["Bombardier"], "outcasts": [], "minions": [], "demons": []},
            "cards": [{"position": 1, "apparent_role": "Bombardier"}],
            "reveal_order": [1],
            "executed": [1],
            "confirmed_good": [1],
            "executed_good_roles": {"1": "Bombardier"},
            "executed_current_roles": {"1": "Bombardier"},
            "terminal_loss_role": "Bombardier",
        }

        decision_states = []

        def solve_decision(state):
            decision_states.append(state)
            return self._fake_result()

        with patch.object(
            decision_analysis, "rust_solve_to_objects", side_effect=solve_decision
        ):
            analysis = decision_analysis.analyze_game(case)

        self.assertEqual(analysis.result, "loss")
        self.assertTrue(decision_states)
        self.assertTrue(
            all(state.terminal_loss_role is None for state in decision_states)
        )

        replay_states = []

        def solve_replay(state):
            replay_states.append(state)
            return self._fake_result()

        with patch.object(replay_analysis, "quiet_solve", side_effect=solve_replay):
            replay_analysis.replay_case(case)

        self.assertGreaterEqual(len(replay_states), 2)
        self.assertIsNone(replay_states[0].terminal_loss_role)
        self.assertEqual(replay_states[-1].terminal_loss_role, "Bombardier")
        self.assertEqual(replay_states[-1].executed_current_roles, {1: "Bombardier"})

    def test_decision_analysis_classifies_unmarked_public_terminals_as_loss(self):
        import decision_analysis

        base = {
            "name": "unmarked_public_terminal",
            "n_cards": 2,
            "n_evil": 0,
            "deck": {
                "villagers": ["Slayer", "Bombardier"],
                "outcasts": [],
                "minions": [],
                "demons": [],
            },
            "cards": [],
            "executed": [2],
            "confirmed_good": [2],
        }
        cases = {
            "current_role": {
                "executed_current_roles": {"2": "Bombardier"},
            },
            "legacy_good_role": {
                "executed_good_roles": {"2": "Bombardier"},
            },
            "slayer_reveal": {
                "confirmed_good": [],
                "confirmed_evil": [2],
                "slayer_results": [{
                    "slayer_pos": 1,
                    "target_pos": 2,
                    "killed": True,
                    "revealed_role": "Bombardier",
                }],
            },
        }

        for label, evidence in cases.items():
            with self.subTest(label=label):
                case = dict(base)
                case.update(evidence)
                with patch.object(
                    decision_analysis,
                    "rust_solve_to_objects",
                    return_value=self._fake_result(),
                ):
                    analysis = decision_analysis.analyze_game(case)
                self.assertEqual(analysis.result, "loss")

    def test_runtime_evil_slayer_current_bomb_keeps_original_identity_channels(self):
        import decision_analysis
        import hindsight
        import replay_analysis

        case = {
            "name": "evil_shaman_current_bomb",
            "n_cards": 3,
            "n_evil": 2,
            "deck": {
                "villagers": ["Slayer", "Bombardier"],
                "outcasts": [],
                "minions": ["Shaman", "Minion"],
                "demons": [],
            },
            "cards": [
                {"position": 1, "apparent_role": "Slayer"},
                {"position": 2, "apparent_role": "Shaman"},
                {"position": 3, "apparent_role": "Minion"},
            ],
            "reveal_order": [1, 2, 3],
            "used_abilities": [1],
            "slayer_results": [{
                "slayer_pos": 1,
                "target_pos": 2,
                "killed": True,
                "revealed_role": "Bombardier",
            }],
            "executed": [2],
            "confirmed_evil": [2],
            "true_evil_positions": {"2": "Shaman", "3": "Minion"},
            "terminal_loss_role": "Bombardier",
        }

        decision_states = []
        with patch.object(
            decision_analysis,
            "rust_solve_to_objects",
            side_effect=lambda state: (
                decision_states.append(state) or self._fake_result(definite_evil=[3])
            ),
        ):
            decision_analysis.analyze_game(case)
        decision_state = decision_states[0]
        self.assertIn(2, decision_state.confirmed_evil)
        self.assertEqual(decision_state.executed_evil_roles, {})
        self.assertEqual(decision_state.executed_good_roles, {})
        self.assertEqual(decision_state.executed_current_roles, {})

        replay_states = []
        with patch.object(
            replay_analysis,
            "quiet_solve",
            side_effect=lambda state: (
                replay_states.append(state) or self._fake_result(definite_evil=[3])
            ),
        ):
            replay_analysis.replay_case(case)
        replay_state = replay_states[-1]
        self.assertIn(2, replay_state.confirmed_evil)
        self.assertEqual(replay_state.executed_evil_roles, {})
        self.assertEqual(replay_state.executed_good_roles, {})
        self.assertEqual(replay_state.executed_current_roles, {})

        hindsight_states = []
        with (
            # Isolate construction of the pre-applied state; the normal
            # public terminal short-circuit is covered separately below.
            patch.object(
                hindsight,
                "_public_terminal_loss_position",
                return_value=None,
            ),
            patch.object(
                hindsight,
                "rust_solve_to_objects",
                side_effect=lambda state: (
                    hindsight_states.append(state)
                    or self._fake_result(definite_evil=[3])
                ),
            ),
        ):
            hindsight.replay_hindsight(case)
        hindsight_state = hindsight_states[0]
        self.assertIn(2, hindsight_state.confirmed_evil)
        self.assertEqual(hindsight_state.executed_evil_roles, {})
        self.assertEqual(hindsight_state.executed_good_roles, {})
        self.assertEqual(hindsight_state.executed_current_roles, {})
        self.assertIsNone(hindsight_state.terminal_loss_role)

    def test_unresolved_slayer_bomb_alignment_stays_tristate_in_all_bridges(self):
        import decision_analysis
        import hindsight
        import replay_analysis

        case = {
            "name": "unresolved_slayer_bomb",
            "n_cards": 2,
            "n_evil": 1,
            "deck": {
                "villagers": ["Slayer", "Bombardier"],
                "outcasts": [],
                "minions": ["Shaman"],
                "demons": [],
            },
            "cards": [
                {"position": 1, "apparent_role": "Slayer"},
                {"position": 2, "apparent_role": "Shaman"},
            ],
            "reveal_order": [1, 2],
            "used_abilities": [1],
            "slayer_results": [{
                "slayer_pos": 1,
                "target_pos": 2,
                "killed": True,
                "revealed_role": "Bombardier",
            }],
            "executed": [2],
        }

        decision_states = []
        with patch.object(
            decision_analysis,
            "rust_solve_to_objects",
            side_effect=lambda state: (
                decision_states.append(state) or self._fake_result()
            ),
        ):
            analysis = decision_analysis.analyze_game(case)
        self.assertEqual(analysis.result, "loss")

        replay_states = []
        with patch.object(
            replay_analysis,
            "quiet_solve",
            side_effect=lambda state: (
                replay_states.append(state) or self._fake_result()
            ),
        ):
            replay_analysis.replay_case(case)

        hindsight_states = []
        with (
            patch.object(
                hindsight,
                "_public_terminal_loss_position",
                return_value=None,
            ),
            patch.object(
                hindsight,
                "rust_solve_to_objects",
                side_effect=lambda state: (
                    hindsight_states.append(state) or self._fake_result()
                ),
            ),
        ):
            hindsight.replay_hindsight(case)

        for state in (
            decision_states[0],
            replay_states[-1],
            hindsight_states[0],
        ):
            self.assertEqual(state.executed, [2])
            self.assertEqual(state.confirmed_good, [])
            self.assertEqual(state.confirmed_evil, [])
            self.assertEqual(state.executed_good_roles, {})
            self.assertEqual(state.executed_evil_roles, {})
            self.assertEqual(state.executed_current_roles, {})

    def test_replay_times_terminal_after_night_and_slayer_deaths(self):
        import replay_analysis

        case = {
            "name": "night_plus_slayer_terminal",
            "n_cards": 3,
            "n_evil": 1,
            "deck": {
                "villagers": ["Slayer", "Bombardier", "Scout"],
                "outcasts": [],
                "minions": ["Shaman"],
                "demons": [],
            },
            "cards": [
                {"position": 1, "apparent_role": "Slayer"},
                {"position": 2, "apparent_role": "Shaman"},
            ],
            "reveal_order": [1, 2],
            "used_abilities": [1],
            "slayer_results": [{
                "slayer_pos": 1,
                "target_pos": 2,
                "killed": True,
                "revealed_role": "Bombardier",
            }],
            "executed": [2, 3],
            "night_kills": [3],
            "confirmed_evil": [2],
            "terminal_loss_role": "Bombardier",
        }
        states = []
        with patch.object(
            replay_analysis,
            "quiet_solve",
            side_effect=lambda state: states.append(state) or self._fake_result(),
        ):
            replay_analysis.replay_case(case)

        self.assertEqual(len(states), 2)
        self.assertIsNone(states[0].terminal_loss_role)
        self.assertEqual(states[-1].terminal_loss_role, "Bombardier")
        self.assertEqual(states[-1].executed, [2])
        self.assertEqual(states[-1].night_kills, [3])

    def test_hindsight_last_evil_current_bomb_is_not_a_win(self):
        import hindsight

        case = {
            "name": "hindsight_terminal",
            "n_cards": 1,
            "n_evil": 1,
            "deck": {
                "villagers": ["Bombardier"],
                "outcasts": [],
                "minions": ["Shaman"],
                "demons": [],
            },
            "cards": [{"position": 1, "apparent_role": "Shaman"}],
            "reveal_order": [1],
            "true_evil_positions": {"1": "Shaman"},
            "executed_current_roles": {"1": "Bombardier"},
        }
        result = self._fake_result(definite_evil=[1])
        result.surviving_scenarios = [Scenario(evil_positions={1: "Shaman"})]
        result.bombardier_positions = []

        with patch.object(hindsight, "rust_solve_to_objects", return_value=result):
            analysis = hindsight.replay_hindsight(case)

        self.assertFalse(analysis.won)

    def test_hindsight_good_bomb_pays_base_cost_before_terminal(self):
        import hindsight

        case = {
            "name": "hindsight_good_bomb_hp",
            "n_cards": 2,
            "n_evil": 1,
            # Saved post-action HP: native resource handling paid 5 before
            # Bombardier's delayed terminal callback.
            "hp": 2,
            "wrong_exec_cost": 5,
            "deck": {
                "villagers": ["Bombardier"],
                "outcasts": [],
                "minions": ["Minion"],
                "demons": [],
            },
            "cards": [
                {"position": 1, "apparent_role": "Bombardier"},
                {"position": 2, "apparent_role": "Minion"},
            ],
            "reveal_order": [1, 2],
            "executed": [1],
            "confirmed_good": [1],
            "executed_good_roles": {"1": "Bombardier"},
            "executed_current_roles": {"1": "Bombardier"},
            "true_evil_positions": {"2": "Minion"},
        }
        result = self._fake_result(definite_evil=[1])
        result.bombardier_positions = []

        with patch.object(hindsight, "rust_solve_to_objects", return_value=result):
            analysis = hindsight.replay_hindsight(case)

        self.assertFalse(analysis.won)
        self.assertEqual(analysis.hp_start, 7)
        self.assertEqual(analysis.hp_end, 2)
        self.assertEqual(analysis.steps[-1].hp_cost, 5)

        output = StringIO()
        with redirect_stdout(output):
            hindsight.print_result(analysis)
        rendered = output.getvalue()
        self.assertIn("HP: 7->2", rendered)
        self.assertIn("TERMINAL (Bomb -5)", rendered)

    def test_hindsight_preapplied_slayer_bomb_precedes_last_evil_win(self):
        import hindsight

        case = {
            "name": "hindsight_slayer_terminal",
            "n_cards": 2,
            "n_evil": 1,
            "hp": 7,
            "deck": {
                "villagers": ["Slayer", "Bombardier"],
                "outcasts": [],
                "minions": ["Shaman"],
                "demons": [],
            },
            "cards": [
                {"position": 1, "apparent_role": "Slayer"},
                {"position": 2, "apparent_role": "Shaman"},
            ],
            "reveal_order": [1, 2],
            "slayer_results": [{
                "slayer_pos": 1,
                "target_pos": 2,
                "killed": True,
                "revealed_role": "Bombardier",
            }],
            "executed": [2],
            "confirmed_evil": [2],
            "true_evil_positions": {"2": "Shaman"},
        }

        with patch.object(hindsight, "rust_solve_to_objects") as solve:
            analysis = hindsight.replay_hindsight(case)

        solve.assert_not_called()
        self.assertFalse(analysis.won)
        self.assertEqual(analysis.hp_end, 7)
        self.assertEqual(analysis.evils_found, 1)
        self.assertEqual(analysis.steps[-1].evil_role, "Bombardier terminal")

    def test_hindsight_uses_scenario_exact_bombardier_collector(self):
        from hindsight import _pick_target

        result = self._fake_result(definite_evil=[1])
        probs = {1: 1.0}
        result.bombardier_positions = [1]
        self.assertIsNone(_pick_target(result, probs, {1}, set()))

        # An apparent Bombardier that is only an Evil bluff is absent from the
        # solver's current-role collector and remains an eligible correct kill.
        result.bombardier_positions = []
        self.assertEqual(_pick_target(result, probs, set(), set()), 1)


if __name__ == "__main__":
    unittest.main()
