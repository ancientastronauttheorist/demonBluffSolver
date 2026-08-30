"""Regression tests for post-execution evidence timing in analysis tools."""

import unittest
from unittest.mock import patch

import decision_analysis
import hindsight
import replay_analysis
from solver import SolverResult
from strategy import Action


def _solver_result():
    return SolverResult([], [], [], 1, 1, [])


def _case(n_cards=2):
    cards = [
        {"position": 1, "apparent_role": "Knight", "info_parsed": {}},
        {"position": 2, "apparent_role": "Scout", "info_parsed": {}},
    ]
    if n_cards == 3:
        cards.append({"position": 3, "apparent_role": "Hunter", "info_parsed": {}})
    return {
        "name": "incremental_execution_evidence",
        "n_cards": n_cards,
        "n_evil": 0,
        "deck": {
            "villagers": ["Knight", "Scout", "Hunter"],
            "outcasts": ["Drunk", "Doppelganger"],
            "minions": [],
            "demons": [],
        },
        "cards": cards,
        "reveal_order": list(range(1, n_cards + 1)),
        "executed": [1, 2],
        "confirmed_good": [1, 2],
        "executed_good_corrupted": {"1": False, "2": True},
        "executed_good_roles": {"1": "Drunk", "2": "Doppelganger"},
        "board_villager_count": n_cards,
        "board_outcast_count": 0,
        "board_count_provenance": "trusted_pre_start",
        "hp": 6,
        "wrong_exec_cost": 2,
    }


def _slayer_wretch_case():
    return {
        "name": "slayer_wretch_incremental_evidence",
        "n_cards": 3,
        "n_evil": 1,
        "deck": {
            "villagers": ["Slayer", "Hunter"],
            "outcasts": ["Wretch"],
            "minions": ["Shaman"],
            "demons": [],
        },
        "cards": [
            {"position": 1, "apparent_role": "Slayer", "info_parsed": {}},
            {"position": 2, "apparent_role": "Wretch", "info_parsed": {}},
            {"position": 3, "apparent_role": "Hunter", "info_parsed": {}},
        ],
        "reveal_order": [1, 2, 3],
        "used_abilities": [1],
        "slayer_results": [{
            "slayer_pos": 1,
            "target_pos": 2,
            "killed": True,
            "revealed_role": "Wretch",
        }],
        "executed": [2, 3],
        "confirmed_good": [2],
        "confirmed_evil": [3],
        "executed_good_corrupted": {"2": True},
        "executed_good_roles": {"2": "Wretch"},
        "executed_evil_roles": {"3": "Shaman"},
        "true_evil_positions": {"3": "Shaman"},
        "hp": 5,
        "wrong_exec_cost": 5,
    }


class IncrementalExecutionEvidenceTests(unittest.TestCase):
    def test_decision_analysis_applies_wretch_at_ability_and_skips_duplicate_execution(self):
        seen = []

        def solve(state):
            seen.append((
                list(state.executed),
                list(state.confirmed_good),
                dict(state.executed_good_roles),
                dict(state.executed_good_corrupted),
            ))
            return _solver_result()

        with (
            patch.object(decision_analysis, "rust_solve_to_objects", side_effect=solve),
            patch.object(decision_analysis, "evil_probabilities", return_value={}),
            patch.object(
                decision_analysis,
                "recommend_action",
                return_value=Action("execute", position=3),
            ),
            patch.object(decision_analysis, "_compute_confidence", return_value=1.0),
            patch.object(decision_analysis, "_actions_match", return_value=True),
        ):
            analysis = decision_analysis.analyze_game(_slayer_wretch_case())

        self.assertEqual(analysis.result, "win")
        self.assertEqual(len(seen), 2)
        self.assertEqual(seen[0], ([2], [2], {2: "Wretch"}, {2: True}))

    def test_replay_analysis_carries_wretch_evidence_at_slayer_step(self):
        seen = []

        def solve(state):
            seen.append((
                list(state.executed),
                list(state.confirmed_good),
                dict(state.executed_good_roles),
                dict(state.executed_good_corrupted),
            ))
            return _solver_result()

        with patch.object(replay_analysis, "quiet_solve", side_effect=solve):
            replay_analysis.replay_case(_slayer_wretch_case())

        self.assertEqual(seen[0][2:], ({}, {}))
        self.assertEqual(seen[1], ([2], [2], {2: "Wretch"}, {2: True}))
        self.assertEqual(len(seen), 3)

    def test_hindsight_preloads_slayer_wretch_public_evidence(self):
        seen = []

        def solve(state):
            seen.append((
                list(state.executed),
                list(state.confirmed_good),
                dict(state.executed_good_roles),
                dict(state.executed_good_corrupted),
            ))
            return _solver_result()

        with (
            patch.object(hindsight, "rust_solve_to_objects", side_effect=solve),
            patch.object(hindsight, "_pick_target", return_value=3),
        ):
            result = hindsight.replay_hindsight(_slayer_wretch_case())

        self.assertTrue(result.won)
        self.assertEqual(seen[0], ([2], [2], {2: "Wretch"}, {2: True}))
    def test_decision_analysis_adds_role_only_after_its_execution(self):
        seen_roles = []
        seen_provenance = []

        def solve(state):
            seen_roles.append(dict(state.executed_good_roles))
            seen_provenance.append(state.board_count_provenance)
            return _solver_result()

        with (
            patch.object(decision_analysis, "rust_solve_to_objects", side_effect=solve),
            patch.object(decision_analysis, "evil_probabilities", return_value={}),
            patch.object(
                decision_analysis,
                "recommend_action",
                return_value=Action("execute", position=1),
            ),
            patch.object(decision_analysis, "_compute_confidence", return_value=1.0),
            patch.object(decision_analysis, "_actions_match", return_value=True),
        ):
            decision_analysis.analyze_game(_case())

        self.assertEqual(seen_roles, [{}, {}, {1: "Drunk"}])
        self.assertEqual(seen_provenance, ["trusted_pre_start"] * 3)

    def test_replay_analysis_adds_role_at_the_matching_step(self):
        seen_roles = []
        seen_provenance = []

        def solve(state):
            seen_roles.append(dict(state.executed_good_roles))
            seen_provenance.append(state.board_count_provenance)
            return _solver_result()

        with patch.object(replay_analysis, "quiet_solve", side_effect=solve):
            replay_analysis.replay_case(_case())

        self.assertEqual(
            seen_roles,
            [{}, {1: "Drunk"}, {1: "Drunk", 2: "Doppelganger"}],
        )
        self.assertEqual(seen_provenance, ["trusted_pre_start"] * 3)

    def test_hindsight_does_not_preload_later_execution_roles(self):
        case = _case(n_cards=3)
        case.update({
            "n_evil": 1,
            "true_evil_positions": {"3": "Pooka"},
            "executed": [1, 2, 3],
            "confirmed_evil": [3],
            "executed_evil_roles": {"3": "Pooka"},
        })
        seen_roles = []
        seen_provenance = []

        def solve(state):
            seen_roles.append(dict(state.executed_good_roles))
            seen_provenance.append(state.board_count_provenance)
            return _solver_result()

        with (
            patch.object(hindsight, "rust_solve_to_objects", side_effect=solve),
            patch.object(hindsight, "_pick_target", side_effect=[1, 2, 3]),
        ):
            result = hindsight.replay_hindsight(case)

        self.assertTrue(result.won)
        self.assertEqual(
            seen_roles,
            [
                {},
                {1: "Drunk"},
                {1: "Drunk"},
                {1: "Drunk", 2: "Doppelganger"},
                {1: "Drunk", 2: "Doppelganger"},
                {1: "Drunk", 2: "Doppelganger"},
            ],
        )
        self.assertEqual(seen_provenance, ["trusted_pre_start"] * 6)


if __name__ == "__main__":
    unittest.main()
