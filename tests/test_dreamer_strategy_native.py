"""Focused current-build tests for the shipped public Dreamer strategy."""

import unittest

from solver import CardInfo, DeckComposition, GameState, Scenario, SolverResult
from strategy import (
    _dreamer_observation_likelihoods,
    _recommend_dreamer_ability,
    recommend_abilities,
)


def _state(apparent_roles: list[str]) -> GameState:
    return GameState(
        n_cards=len(apparent_roles),
        n_evil=0,
        deck=DeckComposition(
            villagers=["Dreamer", "Bard", "Knight", "Baker"],
            outcasts=[],
            minions=["Shaman"],
            demons=["Pooka", "Lilis"],
        ),
        cards=[
            CardInfo(position, role)
            for position, role in enumerate(apparent_roles, start=1)
        ],
    )


class DreamerNativeLikelihoodTests(unittest.TestCase):
    def test_truthful_fallback_is_per_board_entry_and_both_targets_can_match(self):
        state = _state(["Dreamer", "Bard", "Knight", "Baker", "Baker"])
        scenario = Scenario(evil_positions={})

        likelihoods = _dreamer_observation_likelihoods(
            [2, 3], 1, scenario, state
        )

        self.assertAlmostEqual(sum(likelihoods.values()), 1.0)
        self.assertEqual(
            likelihoods,
            {
                ("Baker", "Bard"): 0.25,
                ("Baker", "Knight"): 0.25,
                ("Bard", "Dreamer"): 0.125,
                ("Bard", "Knight"): 0.25,
                ("Dreamer", "Knight"): 0.125,
            },
        )
        # Native excludes only the anchor identity.  It does not exclude the
        # other selected card, so the two selected real roles have support.
        self.assertGreater(likelihoods[("Bard", "Knight")], 0.0)

    def test_truthful_anchor_prefers_other_targets_distinct_live_bluff(self):
        state = _state(["Dreamer", "Knight", "Bard"])
        scenario = Scenario(evil_positions={3: "Pooka"})

        likelihoods = _dreamer_observation_likelihoods(
            [2, 3], 1, scenario, state
        )

        self.assertEqual(
            likelihoods,
            {
                ("Bard", "Knight"): 0.5,
                ("Dreamer", "Pooka"): 0.25,
                ("Knight", "Pooka"): 0.25,
            },
        )
        self.assertAlmostEqual(sum(likelihoods.values()), 1.0)

    def test_truthful_wretch_is_deterministic_cabbage(self):
        state = _state(["Dreamer", "Wretch", "Knight"])

        likelihoods = _dreamer_observation_likelihoods(
            [2, 3], 1, Scenario(evil_positions={}), state
        )

        self.assertEqual(likelihoods, {("Cabbage",): 1.0})
        self.assertAlmostEqual(sum(likelihoods.values()), 1.0)

    def test_liar_fill_excludes_targets_and_draws_unique_roles_without_replacement(self):
        state = _state(["Dreamer", "Bard", "Knight", "Shaman", "Pooka"])
        scenario = Scenario(evil_positions={}, corrupted={1})

        likelihoods = _dreamer_observation_likelihoods(
            [2, 3], 1, scenario, state
        )

        self.assertAlmostEqual(sum(likelihoods.values()), 1.0)
        self.assertEqual(
            likelihoods,
            {
                ("Dreamer", "Shaman"): 1 / 3,
                ("Dreamer", "Pooka"): 1 / 3,
                ("Pooka", "Shaman"): 1 / 3,
            },
        )
        for observation in likelihoods:
            self.assertEqual(len(observation), len(set(observation)))
            self.assertNotIn("Bard", observation)
            self.assertNotIn("Knight", observation)

    def test_selected_bluff_can_cross_collide_with_other_target_real_role(self):
        state = _state(["Dreamer", "Bard", "Bard", "Knight"])
        scenario = Scenario(evil_positions={2: "Pooka"}, corrupted={1})

        likelihoods = _dreamer_observation_likelihoods(
            [2, 3], 1, scenario, state
        )

        # #2's Bard bluff is accepted before exclusions even though Bard is
        # #3's real identity.  Pooka and both target Bard identities are then
        # excluded from the random fill pool.
        self.assertEqual(
            likelihoods,
            {
                ("Bard", "Dreamer"): 0.5,
                ("Bard", "Knight"): 0.5,
            },
        )
        self.assertAlmostEqual(sum(likelihoods.values()), 1.0)
        self.assertTrue(all("Bard" in observation for observation in likelihoods))
        self.assertTrue(all("Pooka" not in observation for observation in likelihoods))


class DreamerNativeRecommendationTests(unittest.TestCase):
    def test_incomplete_board_does_not_invent_native_observations(self):
        state = _state(["Dreamer", "Bard", "Knight"])
        state.cards.pop()
        scenario = Scenario(evil_positions={})

        self.assertEqual(
            _dreamer_observation_likelihoods([1, 2], 1, scenario, state),
            {},
        )
        result = SolverResult(
            definite_evil=[],
            definite_good=[],
            bombardier_positions=[],
            n_scenarios=1,
            n_surviving=1,
            surviving_scenarios=[scenario],
        )
        self.assertIsNone(
            _recommend_dreamer_ability(1, [[1, 2]], state, result)
        )

    def test_mutual_information_selects_stable_pair_and_explains_likelihood(self):
        state = _state(["Dreamer", "Bard", "Knight", "Baker"])
        scenarios = [
            Scenario(evil_positions={2: "Pooka"}),
            Scenario(evil_positions={2: "Lilis"}),
        ]
        result = SolverResult(
            definite_evil=[],
            definite_good=[],
            bombardier_positions=[],
            n_scenarios=2,
            n_surviving=2,
            surviving_scenarios=scenarios,
        )

        # [2, 3] and [2, 4] carry the same information.  The explicit target
        # tuple tie-break must win independent of candidate iteration order.
        direct = _recommend_dreamer_ability(
            1, [[2, 4], [3, 4], [2, 3]], state, result
        )
        self.assertIsNotNone(direct)
        self.assertEqual(direct.targets, [2, 3])
        self.assertAlmostEqual(direct.score, 0.5)
        self.assertIn("Mutual information 0.500 bits", direct.reasoning)
        self.assertIn("expected posterior entropy 0.500 bits", direct.reasoning)

        public = next(
            recommendation
            for recommendation in recommend_abilities(state, result, used_abilities=[])
            if recommendation.ability_name == "Dreamer"
        )
        self.assertEqual(public.targets, [2, 3])
        self.assertAlmostEqual(public.score, 0.5)
        self.assertIn("native role-pair observations", public.reasoning)


if __name__ == "__main__":
    unittest.main()
