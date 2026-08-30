"""Focused current-build tests for shipped Judge2 validation strategy."""

import unittest

from knowledge_base import get_card
from solver import CardInfo, DeckComposition, GameState, Scenario, SolverResult
from strategy import _judge_ground_truth, _recommend_judge, recommend_abilities


def _state(
    apparent_roles: list[str],
    *,
    executed: list[int] | None = None,
    night_kills: list[int] | None = None,
) -> GameState:
    return GameState(
        n_cards=len(apparent_roles),
        n_evil=1,
        deck=DeckComposition(
            villagers=["Judge", "Confessor", "Poet", "Baker"],
            outcasts=[],
            minions=["Minion"],
            demons=[],
        ),
        cards=[
            CardInfo(position, role)
            for position, role in enumerate(apparent_roles, start=1)
        ],
        executed=list(executed or []),
        night_kills=list(night_kills or []),
    )


def _result(scenarios: list[Scenario]) -> SolverResult:
    return SolverResult([], [], [], len(scenarios), len(scenarios), scenarios)


class JudgeNativeStrategyTests(unittest.TestCase):
    def test_public_knowledge_marks_reset_after_night_usage(self):
        judge = get_card("Judge")

        self.assertTrue(judge.activated_ability)
        self.assertTrue(judge.ability_resets_after_night)

    def test_confessor_appearance_overrides_actual_lie_status(self):
        state = _state(["Judge", "Confessor"])
        corrupted = Scenario(evil_positions={}, corrupted={2})
        evil = Scenario(evil_positions={2: "Minion"})

        self.assertFalse(_judge_ground_truth(2, corrupted, state))
        self.assertFalse(_judge_ground_truth(2, evil, state))

    def test_corrupted_judge_deterministically_inverts_observation(self):
        state = _state(["Judge", "Baker"])
        scenarios = [
            Scenario(evil_positions={}, corrupted={1}),
            Scenario(evil_positions={2: "Minion"}, corrupted={1}),
        ]

        rec = _recommend_judge(1, state, _result(scenarios), [2])

        self.assertIsNotNone(rec)
        self.assertEqual(rec.targets, [2])
        self.assertAlmostEqual(rec.score, 1.0)
        self.assertEqual(rec.warnings, [])
        self.assertIn("deterministic native Judge observations", rec.reasoning)

    def test_dead_target_is_native_legal_and_can_be_most_informative(self):
        state = _state(["Judge", "Baker", "Baker"], executed=[2])
        scenarios = [
            Scenario(evil_positions={3: "Minion"}),
            Scenario(evil_positions={2: "Minion"}),
        ]

        direct = _recommend_judge(1, state, _result(scenarios), [1, 2, 3])
        self.assertIsNotNone(direct)
        self.assertEqual(direct.targets, [2])
        self.assertAlmostEqual(direct.score, 1.0)

        public = next(
            rec
            for rec in recommend_abilities(state, _result(scenarios), [])
            if rec.ability_name == "Judge"
        )
        self.assertEqual(public.targets, [2])

    def test_night_killed_judge_is_not_recommended_as_an_actor(self):
        state = _state(["Judge", "Baker"], night_kills=[1])
        scenarios = [Scenario(evil_positions={2: "Minion"})]

        recommendations = recommend_abilities(state, _result(scenarios), [])

        self.assertFalse(any(rec.ability_name == "Judge" for rec in recommendations))

    def test_self_is_legal_but_last_on_an_exact_tie(self):
        state = _state(["Judge", "Baker"])
        scenario = Scenario(evil_positions={})

        rec = _recommend_judge(1, state, _result([scenario]), [1, 2])

        self.assertIsNotNone(rec)
        self.assertEqual(rec.targets, [2])

    def test_poet_is_not_excluded_from_native_candidates(self):
        state = _state(["Judge", "Poet", "Baker"])
        scenarios = [
            Scenario(evil_positions={2: "Minion"}),
            Scenario(evil_positions={3: "Minion"}),
        ]

        rec = _recommend_judge(1, state, _result(scenarios), [1, 2, 3])

        self.assertIsNotNone(rec)
        self.assertEqual(rec.targets, [2])
        self.assertAlmostEqual(rec.score, 1.0)


if __name__ == "__main__":
    unittest.main()
