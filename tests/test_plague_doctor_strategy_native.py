"""Focused current-build tests for Plague Doctor's native Day callback."""

import unittest

from solver import CardInfo, DeckComposition, GameState, Scenario, SolverResult
from strategy import (
    _pd_information_for_target,
    _pd_observation_likelihoods,
    _recommend_pd_ability,
    recommend_abilities,
)


def _state(apparent_roles: list[str], *, executed: list[int] | None = None) -> GameState:
    return GameState(
        n_cards=len(apparent_roles),
        n_evil=2,
        deck=DeckComposition(
            villagers=["Knight", "Bard", "Baker"],
            outcasts=["Plague Doctor", "Wretch", "Drunk"],
            minions=["Minion"],
            demons=["Pooka"],
        ),
        cards=[
            CardInfo(position, role)
            for position, role in enumerate(apparent_roles, start=1)
        ],
        executed=list(executed or []),
    )


class PlagueDoctorNativeLikelihoodTests(unittest.TestCase):
    def test_truthful_reveal_pool_includes_dead_evil_and_registered_wretch(self):
        state = _state(
            ["Plague_Doctor", "Knight", "Wretch", "Bard", "Baker"],
            executed=[2],
        )
        scenario = Scenario(evil_positions={2: "Pooka"}, corrupted={4})

        likelihoods = _pd_observation_likelihoods(4, 1, scenario, state)

        self.assertEqual(
            likelihoods,
            {("corrupted", 2): 0.5, ("corrupted", 3): 0.5},
        )
        self.assertAlmostEqual(sum(likelihoods.values()), 1.0)

    def test_bluff_inverts_clean_and_uniformly_names_runtime_good(self):
        state = _state(
            ["Plague_Doctor", "Knight", "Wretch", "Bard", "Baker"],
            executed=[2],
        )
        scenario = Scenario(evil_positions={1: "Minion", 2: "Pooka"})

        likelihoods = _pd_observation_likelihoods(4, 1, scenario, state)

        self.assertEqual(
            likelihoods,
            {("corrupted", 4): 0.5, ("corrupted", 5): 0.5},
        )
        self.assertAlmostEqual(sum(likelihoods.values()), 1.0)

    def test_ordinary_drunk_corrupted_status_is_reported(self):
        state = _state(["Plague_Doctor", "Knight", "Bard"])
        scenario = Scenario(
            evil_positions={2: "Pooka"},
            corrupted={3},
            drunk_position=3,
        )

        self.assertEqual(
            _pd_observation_likelihoods(3, 1, scenario, state),
            {("corrupted", 2): 1.0},
        )

    def test_self_check_is_always_visibly_clean(self):
        state = _state(["Plague_Doctor", "Knight", "Bard"])
        corrupted_good = Scenario(evil_positions={2: "Pooka"}, corrupted={1})
        evil_bluff = Scenario(evil_positions={1: "Minion", 2: "Pooka"})

        self.assertEqual(
            _pd_observation_likelihoods(1, 1, corrupted_good, state),
            {("clean",): 1.0},
        )
        self.assertEqual(
            _pd_observation_likelihoods(1, 1, evil_bluff, state),
            {("clean",): 1.0},
        )


class PlagueDoctorNativeRecommendationTests(unittest.TestCase):
    def test_random_reveal_support_carries_information_beyond_lowest_evil(self):
        state = _state(["Plague_Doctor", "Knight", "Bard", "Baker", "Knight"])
        scenarios = [
            Scenario(evil_positions={2: "Pooka", 3: "Minion"}, corrupted={5}),
            Scenario(evil_positions={2: "Pooka", 4: "Minion"}, corrupted={5}),
        ]

        information, expected_entropy, outcomes = _pd_information_for_target(
            5, 1, state, scenarios
        )

        self.assertAlmostEqual(information, 0.5)
        self.assertAlmostEqual(expected_entropy, 0.5)
        self.assertEqual(outcomes, 3)

    def test_recommender_can_choose_a_dead_target_and_keeps_self_last(self):
        state = _state(["Plague_Doctor", "Knight", "Bard"], executed=[2])
        scenarios = [
            Scenario(evil_positions={3: "Pooka"}, corrupted={2}),
            Scenario(evil_positions={3: "Pooka"}),
        ]
        result = SolverResult([], [], [], 2, 2, scenarios)

        direct = _recommend_pd_ability(1, [3, 2, 1], state, result)
        self.assertIsNotNone(direct)
        self.assertEqual(direct.targets, [2])
        self.assertAlmostEqual(direct.score, 1.0)
        self.assertIn("native PD observations", direct.reasoning)

        public = next(
            recommendation
            for recommendation in recommend_abilities(state, result, used_abilities=[])
            if recommendation.ability_name == "Plague Doctor"
        )
        self.assertEqual(public.targets, [2])


if __name__ == "__main__":
    unittest.main()
