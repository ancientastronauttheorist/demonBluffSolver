"""Focused live-capture regressions for the shipped Rambler2 redesign."""

import json
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from game_loop import (
    DecisionLog,
    GameSession,
    _parse_card_cli,
    _parse_clue_from_memory,
    _judge_observation_history,
    _release_session_lock,
    card_rambler_quote,
    card_shut_up,
    dispatch,
)
from solver import CardInfo, GameState
from state_machine import GamePhase, GameStateMachine
from strategy import Action


def _memory_card(
    position: int,
    role: str,
    clue: str,
    *,
    acted_infos=None,
    remaining: int | None = None,
) -> dict:
    acted_infos = list(acted_infos or [])
    if remaining is None:
        remaining = 0 if acted_infos else 1
    return {
        "position": position,
        "true_role": role,
        "disguise": role,
        "state": "Alive",
        "clue_text": clue,
        "acted_infos": acted_infos,
        "runtime_data": None,
        "pickable_uses_remaining": remaining,
        "act_output_enabled": True,
        "pickable_available": remaining > 0,
        "uses": remaining,
        "ability_used": True,
    }


class RamblerMemoryParsingTests(unittest.TestCase):
    def test_non_jester_saved_act_preserves_exact_public_target(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                1,
                "Lover",
                "#10 shut up!",
                acted_infos=[{"desc": "#10 shut up!", "targets": [10]}],
            ),
            n_cards=10,
        )

        self.assertEqual(parsed.apparent_role, "Lover")
        self.assertEqual(parsed.info_text, "#10 shut up!")
        self.assertEqual(parsed.info_parsed, {"shut_up_target": 10})

    def test_missing_saved_act_is_pending_instead_of_scanning_history(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                5,
                "Gemcrafter",
                "",
                acted_infos=[
                    {"desc": "old unrelated text", "targets": [2]},
                    {"desc": "#4 SHUT UP!", "targets": [4]},
                ],
            ),
            n_cards=9,
        )

        self.assertIsNone(parsed)

    def test_only_newest_acted_info_can_define_current_surface(self):
        clue = "1 Evil\nadjacent to me"
        parsed = _parse_clue_from_memory(
            _memory_card(
                5,
                "Lover",
                clue,
                acted_infos=[
                    {"desc": "#4 shut up!", "targets": [4]},
                    {"desc": clue, "targets": [4, 6]},
                ],
            ),
            n_cards=9,
        )

        self.assertEqual(parsed.info_text, clue)
        self.assertEqual(
            parsed.info_parsed,
            {"evil_adjacent": 1, "lover_variant": "public_current"},
        )

    def test_malformed_or_out_of_range_shut_up_never_falls_through(self):
        for clue in [
            "#0 shut up!",
            "#10 shut up!",
            "#X shut up!",
            "#4 shut up maybe",
            "4 shut up!",
        ]:
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_card(1, "Lover", clue),
                    n_cards=9,
                )
                self.assertIsNone(parsed)

    def test_reset_available_active_role_retains_prior_shut_up_evidence(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                3,
                "Jester",
                "#5\nshut up!",
                acted_infos=[{"desc": "#5\nshut up!", "targets": [5]}],
                remaining=1,
            ),
            n_cards=6,
        )

        self.assertEqual(
            parsed.info_parsed,
            {"jester_variant": "public_current", "shut_up_target": 5},
        )

    def test_current_rambler_quote_is_negative_interference_evidence(self):
        quote = "I once met a cabbage on the road."
        parsed = _parse_clue_from_memory(
            _memory_card(
                4,
                "Rambler",
                quote,
                acted_infos=[{"desc": quote, "targets": [3, 5]}],
            ),
            n_cards=7,
        )

        self.assertEqual(parsed, card_rambler_quote(4, quote))
        self.assertEqual(parsed.info_parsed, {"quote_observed": True})

    def test_rambler_quote_requires_native_circular_ref_order(self):
        cases = [
            (1, 1, [1, 1]),
            (1, 2, [2, 2]),
            (2, 2, [1, 1]),
            (4, 7, [3, 5]),
            (1, 7, [7, 2]),
            (7, 7, [6, 1]),
        ]
        for position, n_cards, refs in cases:
            with self.subTest(position=position, n_cards=n_cards):
                quote = f"quote {position}/{n_cards}"
                parsed = _parse_clue_from_memory(
                    _memory_card(
                        position,
                        "Rambler",
                        quote,
                        acted_infos=[{"desc": quote, "targets": refs}],
                    ),
                    n_cards=n_cards,
                )
                self.assertEqual(parsed, card_rambler_quote(position, quote))

    def test_shut_up_and_quote_require_latest_description_and_refs_agreement(self):
        malformed = [
            _memory_card(
                1,
                "Lover",
                "#4 shut up!",
                acted_infos=[{"desc": "#3 shut up!", "targets": [3]}],
            ),
            _memory_card(
                1,
                "Lover",
                "#4 shut up!",
                acted_infos=[{"desc": "#4 shut up!", "targets": [3]}],
            ),
            _memory_card(
                2,
                "Rambler",
                "quote",
                acted_infos=[{"desc": "different", "targets": [1, 3]}],
            ),
            _memory_card(
                2,
                "Rambler",
                "quote",
                acted_infos=[{"desc": "quote", "targets": [3, 1]}],
            ),
        ]
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=4)
                )

    def test_jester_rewritten_reference_is_not_saved_as_original_pick(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                3,
                "Jester",
                "#5\nshut up!",
                acted_infos=[{"desc": "#5\nshut up!", "targets": [5]}],
                remaining=0,
            ),
            n_cards=6,
        )

        self.assertEqual(
            parsed.info_parsed,
            {"jester_variant": "public_current", "shut_up_target": 5},
        )


class RamblerSessionCaptureTests(unittest.TestCase):
    def test_auto_card_does_not_consume_reset_available_retained_jester_event(self):
        session = GameSession(5, 1)
        session.cards.append(CardInfo(3, "Jester"))
        retained = _memory_card(
            3,
            "Jester",
            "#5\nshut up!",
            acted_infos=[{"desc": "#5\nshut up!", "targets": [5]}],
            remaining=1,
        )

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [retained]

            def close(self):
                return None

        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("memory_reader.print_board"),
            patch.object(session, "save"),
        ):
            dispatch("auto_card", [], session)

        self.assertEqual(session.cards[0].info_parsed, {})
        self.assertNotIn(3, session.used_abilities)

    def test_auto_card_replaces_early_passive_entry_and_keeps_reveal_order(self):
        session = GameSession(4, 1)
        session.add_card(CardInfo(1, "Lover", info_parsed={"evil_adjacent": 0}))
        memory = _memory_card(
            1,
            "Lover",
            "#4 shut up!",
            acted_infos=[{"desc": "#4 shut up!", "targets": [4]}],
        )

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
        ):
            dispatch("auto_card", [], session)

        self.assertEqual(session.cards[0].info_parsed, {"shut_up_target": 4})
        self.assertEqual(session.reveal_order, [1])

    def test_auto_card_replaces_early_rambler_no_info_with_visible_quote(self):
        session = GameSession(4, 1)
        session.add_card(CardInfo(2, "Rambler"))
        quote = "A late but verified Rambler quote"
        memory = _memory_card(
            2,
            "Rambler",
            quote,
            acted_infos=[{"desc": quote, "targets": [1, 3]}],
        )

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
        ):
            dispatch("auto_card", [], session)

        self.assertEqual(session.cards[0].info_parsed, {"quote_observed": True})
        self.assertEqual(session.cards[0].info_text, quote)
        self.assertEqual(session.reveal_order, [2])

    def test_state_machine_waits_for_settled_rambler_surface_before_advancing(self):
        session = GameSession(4, 1)
        quote = "A quote whose native event is still settling"
        incomplete = _memory_card(2, "Rambler", quote, acted_infos=[])
        settled = _memory_card(
            2,
            "Rambler",
            quote,
            acted_infos=[{"desc": quote, "targets": [1, 3]}],
        )

        class Monitor:
            def __init__(self):
                self.wait_calls = 0

            @staticmethod
            def is_healthy():
                return True

            def wait_for(self, predicate, timeout, min_delay):
                self.wait_calls += 1
                board = [incomplete] if self.wait_calls == 1 else [settled]
                return bool(predicate(board))

            @staticmethod
            def get_board():
                return [incomplete]

        monitor = Monitor()
        machine = GameStateMachine(session=session, monitor=monitor)
        machine.phase = GamePhase.REVEALING
        machine._pending_reveal = (2,)

        with (
            patch(
                "game_utils.all_game_card_coords",
                return_value={2: (200, 100)},
            ),
            patch("template_match.fast_click_at"),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            redirect_stdout(StringIO()),
        ):
            machine._do_revealing()

        self.assertEqual(monitor.wait_calls, 2)
        self.assertEqual(machine.phase, GamePhase.SOLVING)
        self.assertEqual(session.reveal_order, [2])
        self.assertEqual(session.cards[0], card_rambler_quote(2, quote))

    def test_state_machine_pauses_when_reveal_clue_never_settles(self):
        session = GameSession(4, 1)
        incomplete = _memory_card(
            2,
            "Rambler",
            "unsettled quote",
            acted_infos=[],
        )

        class Monitor:
            def __init__(self):
                self.wait_calls = 0

            @staticmethod
            def is_healthy():
                return True

            def wait_for(self, predicate, timeout, min_delay):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    return bool(predicate([incomplete]))
                return False

            @staticmethod
            def get_board():
                return [incomplete]

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine.phase = GamePhase.REVEALING
        machine._pending_reveal = (2,)

        with (
            patch(
                "game_utils.all_game_card_coords",
                return_value={2: (200, 100)},
            ),
            patch("template_match.fast_click_at"),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_revealing()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("did not settle", machine._needs_human_reason)
        self.assertEqual(session.reveal_order, [2])
        self.assertEqual(session.cards, [])

    def test_state_machine_night_boundary_pauses_with_pending_night_on_timeout(self):
        session = GameSession(4, 1)
        session.demons = ["Lilis"]
        session.reveal_order = [1, 2, 3]
        incomplete = _memory_card(
            4,
            "Rambler",
            "unsettled fourth quote",
            acted_infos=[],
        )

        class Monitor:
            def __init__(self):
                self.wait_calls = 0

            @staticmethod
            def is_healthy():
                return True

            def wait_for(self, predicate, timeout, min_delay):
                self.wait_calls += 1
                if self.wait_calls == 1:
                    return bool(predicate([incomplete]))
                return False

            @staticmethod
            def get_board():
                return [incomplete]

        machine = GameStateMachine(session=session, monitor=Monitor())
        machine.phase = GamePhase.REVEALING
        machine._pending_reveal = (4,)

        with (
            patch(
                "game_utils.all_game_card_coords",
                return_value={4: (400, 100)},
            ),
            patch("template_match.fast_click_at"),
            patch.object(session, "save"),
            redirect_stdout(StringIO()),
        ):
            machine._do_revealing()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("pending Night", machine._needs_human_reason)
        self.assertEqual(session.reveal_order, [1, 2, 3, 4])
        self.assertEqual(session.pending_lilis_nights, 1)
        self.assertEqual(session.cards, [])

    def test_interrupted_judge_auto_use_bypasses_strict_judge_result_parser(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(1, "Judge"))
        action = Action("use_ability", 1, [3], "Judge")
        memory = _memory_card(
            1,
            "Judge",
            "#2 shut up!",
            acted_infos=[{"desc": "#2 shut up!", "targets": [2]}],
            remaining=0,
        )
        before = _memory_card(
            1,
            "Judge",
            "",
            acted_infos=[],
            remaining=1,
        )

        class Reader:
            def __init__(self):
                self.reads = 0

            def open(self):
                return True

            def read_board(self):
                self.reads += 1
                return [before if self.reads == 1 else memory]

            def close(self):
                return None

        with (
            patch("template_match.safe_click_at"),
            patch("game_loop.time.sleep"),
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            patch.object(DecisionLog, "log_ability_used"),
        ):
            result = session.auto_use_ability(action)

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["info_parsed"], {"shut_up_target": 2})
        self.assertNotIn("target", result["info_parsed"])
        self.assertIn(1, session.used_abilities)

    def test_interrupted_later_judge_use_keeps_prior_round_observation(self):
        session = GameSession(5, 1)
        session.add_card(CardInfo(
            1,
            "Judge",
            info_parsed={"target": 3, "is_lying": False},
        ))
        session.reset_after_night_abilities()
        session.add_card(card_shut_up(1, "Judge", 2, "#2 shut up!"))

        self.assertEqual(
            session.cards[0].info_parsed,
            {
                "shut_up_target": 2,
                "observations": [{"target": 3, "is_lying": False}],
            },
        )

    def test_save_load_preserves_marker_target_quote_and_reveal_order(self):
        session = GameSession(4, 1)
        session.add_card(card_shut_up(1, "Lover", 4, "#4 shut up!"))
        session.add_card(card_rambler_quote(4, "A very interesting quote"))

        with tempfile.TemporaryDirectory() as temp_dir:
            path = str(Path(temp_dir) / "session.json")
            session.save(path)
            loaded = GameSession.load(path)
            _release_session_lock()

        self.assertEqual(loaded.rambler_rule_version, "rambler2_shut_up")
        self.assertEqual(
            loaded.rambler_shut_up_observations,
            [{"speaker_position": 1, "shut_up_target": 4}],
        )
        self.assertEqual(loaded.reveal_order, [1, 4])
        self.assertEqual(loaded.cards[0].info_parsed, {"shut_up_target": 4})
        self.assertEqual(loaded.cards[1].info_parsed, {"quote_observed": True})

    def test_manual_shut_up_rejects_out_of_board_target(self):
        session = GameSession(4, 1)
        with self.assertRaisesRegex(ValueError, "outside 1..4"):
            _parse_card_cli(["shut_up", "1", "Lover", "5"], session=session)
        self.assertEqual(session.cards, [])

    def test_manual_rambler_rejects_unknown_observation_token(self):
        with self.assertRaisesRegex(ValueError, "Unknown Rambler observation"):
            _parse_card_cli(["rambler", "1", "mystery"])

    def test_passive_same_event_correction_replaces_then_removes_ledger_record(self):
        session = GameSession(5, 1)
        session.add_card(card_shut_up(1, "Lover", 2, "#2 shut up!"))
        session.add_card(card_shut_up(3, "Baker", 4, "#4 shut up!"))

        session.add_card(card_shut_up(1, "Lover", 5, "#5 shut up!"))
        session.add_card(card_shut_up(1, "Lover", 5, "#5 shut up!"))
        self.assertEqual(
            session.rambler_shut_up_observations,
            [
                {"speaker_position": 1, "shut_up_target": 5},
                {"speaker_position": 3, "shut_up_target": 4},
            ],
        )

        session.add_card(CardInfo(
            1,
            "Lover",
            info_parsed={"evil_adjacent": 0},
        ))
        self.assertEqual(
            session.rambler_shut_up_observations,
            [{"speaker_position": 3, "shut_up_target": 4}],
        )

    def test_judge_same_round_normal_shutup_normal_corrects_current_round_only(self):
        session = GameSession(5, 1)
        session.add_card(CardInfo(
            2,
            "Judge",
            info_parsed={"target": 3, "is_lying": False},
        ))
        session.reset_after_night_abilities()
        session.add_card(CardInfo(
            2,
            "Judge",
            info_parsed={"target": 4, "is_lying": True},
        ))

        session.add_card(card_shut_up(2, "Judge", 1, "#1 shut up!"))

        self.assertEqual(
            session.rambler_shut_up_observations,
            [{"speaker_position": 2, "shut_up_target": 1}],
        )
        self.assertEqual(
            session.cards[0].info_parsed,
            {
                "shut_up_target": 1,
                "observations": [{"target": 3, "is_lying": False}],
            },
        )

        session.add_card(CardInfo(
            2,
            "Judge",
            info_parsed={"target": 5, "is_lying": True},
        ))
        self.assertNotIn("shut_up_target", session.cards[0].info_parsed)
        self.assertEqual(session.rambler_shut_up_observations, [])
        self.assertEqual(
            session.cards[0].info_parsed,
            {
                "target": 5,
                "is_lying": True,
                "observations": [
                    {"target": 3, "is_lying": False},
                    {"target": 5, "is_lying": True},
                ],
            },
        )

    def test_only_post_night_judge_event_appends_then_same_round_corrects(self):
        session = GameSession(5, 1)
        session.add_card(card_shut_up(2, "Judge", 1, "#1 shut up!"))
        session.reset_after_night_abilities()
        session.add_card(card_shut_up(2, "Judge", 1, "#1 shut up!"))
        self.assertEqual(
            session.rambler_shut_up_observations,
            [
                {"speaker_position": 2, "shut_up_target": 1},
                {"speaker_position": 2, "shut_up_target": 1},
            ],
        )

        session.add_card(card_shut_up(2, "Judge", 4, "#4 shut up!"))
        self.assertEqual(
            session.rambler_shut_up_observations,
            [
                {"speaker_position": 2, "shut_up_target": 1},
                {"speaker_position": 2, "shut_up_target": 4},
            ],
        )

        session.add_card(CardInfo(
            2,
            "Judge",
            info_parsed={"target": 5, "is_lying": False},
        ))
        self.assertEqual(
            session.rambler_shut_up_observations,
            [{"speaker_position": 2, "shut_up_target": 1}],
        )
        self.assertEqual(
            session.cards[0].info_parsed,
            {"target": 5, "is_lying": False},
        )

    def test_existing_card_malformed_late_callback_surfaces_recovery(self):
        session = GameSession(4, 1)
        session.add_card(CardInfo(
            1,
            "Lover",
            info_parsed={"evil_adjacent": 0},
        ))
        memory = _memory_card(
            1,
            "Lover",
            "#5 shut up!",
            acted_infos=[{"desc": "#5 shut up!", "targets": [5]}],
        )

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [memory]

            def close(self):
                return None

        output = StringIO()
        with (
            redirect_stdout(output),
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("memory_reader.print_board"),
        ):
            dispatch("auto_card", [], session)

        self.assertIn("[RECOVERY]", output.getvalue())
        self.assertIn("out-of-range", output.getvalue())
        self.assertEqual(
            session.cards[0].info_parsed,
            {"evil_adjacent": 0},
        )


class JudgeHistoryValidationTests(unittest.TestCase):
    def test_malformed_histories_raise_descriptive_value_errors(self):
        malformed = [
            ({"observations": "not-an-array"}, "must be an array"),
            ({"observations": [3]}, "must be an object"),
            ({"observations": [{"target": 2}]}, "both target and is_lying"),
            (
                {"observations": [{"target": True, "is_lying": False}]},
                "target must be an integer",
            ),
            (
                {"observations": [{"target": 6, "is_lying": False}]},
                "within 1..5",
            ),
            ({"target": 2}, "both target and is_lying"),
            ({"target": 2, "is_lying": 1}, "must be a boolean"),
        ]
        for info, expected in malformed:
            with self.subTest(info=info):
                with self.assertRaisesRegex(ValueError, expected):
                    _judge_observation_history(info, n_cards=5)

    def test_add_card_rejects_malformed_judge_atomically(self):
        session = GameSession(5, 1)
        with self.assertRaisesRegex(ValueError, "must be an object"):
            session.add_card(CardInfo(
                2,
                "Judge",
                info_parsed={"observations": ["bad"]},
            ))
        self.assertEqual(session.cards, [])
        self.assertEqual(session.reveal_order, [])


class RamblerCorpusRegressionTests(unittest.TestCase):
    def test_current_public_fixtures_are_versioned(self):
        cases = Path(__file__).parent / "cases_v2"
        asc82 = json.loads((cases / "asc82_v5.json").read_text())
        asc83 = json.loads((cases / "asc83_v7.json").read_text())

        self.assertEqual(asc82["rambler_rule_version"], "rambler2_shut_up")
        self.assertEqual(asc82["cards"][0]["info_parsed"], {"shut_up_target": 10})
        self.assertEqual(asc83["rambler_rule_version"], "rambler2_shut_up")
        self.assertEqual(
            asc82["rambler_shut_up_observations"],
            [{"speaker_position": 1, "shut_up_target": 10}],
        )
        self.assertEqual(
            asc83["rambler_shut_up_observations"],
            [
                {"speaker_position": 2, "shut_up_target": 1},
                {"speaker_position": 3, "shut_up_target": 4},
                {"speaker_position": 5, "shut_up_target": 4},
                {"speaker_position": 9, "shut_up_target": 1},
            ],
        )
        self.assertEqual(
            [
                card["position"]
                for card in asc83["cards"]
                if card["info_parsed"].get("quote_observed") is True
            ],
            [1, 4],
        )
        self.assertEqual(
            [
                card["info_parsed"]["shut_up_target"]
                for card in asc83["cards"]
                if "shut_up_target" in card["info_parsed"]
            ],
            [1, 4, 4, 1],
        )

        bridged = GameState.from_dict(asc83)
        self.assertEqual(
            bridged.rambler_rule_version,
            "rambler2_shut_up",
        )
        self.assertEqual(
            bridged.to_dict()["rambler_shut_up_observations"],
            asc83["rambler_shut_up_observations"],
        )

    def test_analysis_reconstructors_forward_version_and_event_history(self):
        import decision_analysis
        import hindsight
        import replay_analysis

        case_path = Path(__file__).parent / "cases_v2" / "asc83_v7.json"
        case = json.loads(case_path.read_text())
        expected_history = case["rambler_shut_up_observations"]

        replay_states = []

        class DummyResult:
            n_surviving = 1
            definite_evil = []
            definite_good = []

        def replay_solver(state):
            replay_states.append(state)
            return DummyResult()

        with patch.object(
            replay_analysis,
            "quiet_solve",
            side_effect=replay_solver,
        ):
            replay_analysis.replay_case(case)

        decision_states = []

        def unavailable_solver(state):
            decision_states.append(state)
            return None

        with patch.object(
            decision_analysis,
            "rust_solve_to_objects",
            side_effect=unavailable_solver,
        ):
            decision_analysis.analyze_game(case)

        hindsight_states = []

        def stopped_solver(state):
            hindsight_states.append(state)
            raise RuntimeError("stop after state capture")

        with patch.object(
            hindsight,
            "rust_solve_to_objects",
            side_effect=stopped_solver,
        ):
            hindsight.replay_hindsight(case)

        for state in [
            replay_states[0],
            decision_states[0],
            hindsight_states[0],
        ]:
            self.assertEqual(
                state.rambler_rule_version,
                "rambler2_shut_up",
            )
            self.assertEqual(
                state.rambler_shut_up_observations,
                expected_history,
            )


if __name__ == "__main__":
    unittest.main()
