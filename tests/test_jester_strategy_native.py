"""Focused strategy tests for current public Jester / managed Juggler."""

import unittest
from unittest.mock import patch

from knowledge_base import get_card
from solver import (
    Alignment,
    CardInfo,
    DeckComposition,
    GameState,
    Scenario,
    ShamanTrace,
    SolverResult,
)
from strategy import (
    _jester_current_ground_truth,
    _jester_ground_truth,
    _jester_observation_likelihoods,
    _recommend_jester_ability,
    _jester_registered_alignment_at_observation,
    recommend_abilities,
)


def _state(
    apparent_roles: list[str],
    *,
    marker: str | None = "public_current",
    executed: list[int] | None = None,
    night_kills: list[int] | None = None,
    blocked: list[int] | None = None,
) -> GameState:
    cards = []
    for position, role in enumerate(apparent_roles, start=1):
        info = {}
        if role == "Jester" and marker is not None:
            info["jester_variant"] = marker
        cards.append(CardInfo(position, role, info_parsed=info))
    return GameState(
        n_cards=len(apparent_roles),
        n_evil=2,
        deck=DeckComposition(
            villagers=["Jester", "Baker", "Knight", "Bard"],
            outcasts=["Wretch", "Drunk", "Doppelganger"],
            minions=["Spy", "Minion", "Puppeteer", "Puppet"],
            demons=["Pooka"],
        ),
        cards=cards,
        executed=list(executed or []),
        night_kills=list(night_kills or []),
        blocked_positions=list(blocked or []),
    )


def _result(scenarios: list[Scenario]) -> SolverResult:
    return SolverResult([], [], [], len(scenarios), len(scenarios), scenarios)


class JesterNativeMetadataTests(unittest.TestCase):
    def test_jester_is_reset_after_night_active(self):
        jester = get_card("Jester")

        self.assertTrue(jester.activated_ability)
        self.assertTrue(jester.ability_resets_after_night)

    def test_gemcrafter_records_asset_reset_category_without_becoming_active(self):
        gemcrafter = get_card("Gemcrafter")

        self.assertFalse(gemcrafter.activated_ability)
        self.assertTrue(gemcrafter.ability_resets_after_night)


class JesterNativeRegistrationTests(unittest.TestCase):
    def test_register_as_precedes_runtime_alignment(self):
        state = _state([
            "Jester",
            "Knight",
            "Wretch",
            "Bard",
            "Bard",
            "Bard",
            "Bard",
        ])
        scenario = Scenario(
            evil_positions={2: "Spy", 7: "Minion"},
            puppet_position=4,
            drunk_position=5,
            doppelganger_position=6,
        )

        self.assertEqual(
            _jester_registered_alignment_at_observation(2, scenario, state),
            Alignment.GOOD,
        )
        self.assertEqual(
            _jester_registered_alignment_at_observation(3, scenario, state),
            Alignment.EVIL,
        )
        self.assertEqual(
            _jester_registered_alignment_at_observation(4, scenario, state),
            Alignment.EVIL,
        )
        self.assertEqual(
            _jester_registered_alignment_at_observation(5, scenario, state),
            Alignment.GOOD,
        )
        self.assertEqual(
            _jester_registered_alignment_at_observation(6, scenario, state),
            Alignment.GOOD,
        )
        self.assertEqual(
            _jester_registered_alignment_at_observation(7, scenario, state),
            Alignment.EVIL,
        )

    def test_current_count_does_not_reuse_legacy_effective_alignment(self):
        state = _state(["Jester", "Knight", "Wretch", "Bard", "Bard"])
        scenario = Scenario(
            evil_positions={2: "Spy"},
            puppet_position=4,
        )

        self.assertEqual(
            _jester_current_ground_truth([2, 3, 4], scenario, state),
            2,
        )
        self.assertEqual(
            _jester_ground_truth([2, 3, 4], scenario, state),
            3,
        )

    def test_unrepresented_hidden_register_as_fails_closed(self):
        state = GameState(
            n_cards=3,
            n_evil=1,
            deck=DeckComposition(["Jester"], ["Wretch"], ["Spy"], []),
            cards=[
                CardInfo(
                    1,
                    "Jester",
                    info_parsed={"jester_variant": "public_current"},
                )
            ],
        )

        self.assertEqual(
            _jester_observation_likelihoods(
                [1, 2, 3],
                1,
                Scenario(evil_positions={}),
                state,
            ),
            {},
        )

    def test_exact_three_distinct_targets_are_required(self):
        state = _state(["Jester", "Knight", "Wretch"])
        scenario = Scenario(evil_positions={})

        with self.assertRaisesRegex(ValueError, "exactly three distinct"):
            _jester_current_ground_truth([1, 1, 2], scenario, state)


class JesterNativeLikelihoodTests(unittest.TestCase):
    def test_truth_is_exact_and_bluff_is_uniform_over_other_counts(self):
        state = _state(["Jester", "Knight", "Wretch", "Bard"])
        truthful = Scenario(evil_positions={2: "Spy"})
        liar = Scenario(evil_positions={2: "Spy"}, corrupted={1})

        self.assertEqual(
            _jester_observation_likelihoods(
                [2, 3, 4], 1, truthful, state
            ),
            {(1,): 1.0},
        )
        false_counts = _jester_observation_likelihoods(
            [2, 3, 4], 1, liar, state
        )
        self.assertEqual(set(false_counts), {(0,), (2,), (3,)})
        for probability in false_counts.values():
            self.assertAlmostEqual(probability, 1 / 3)
        self.assertAlmostEqual(sum(false_counts.values()), 1.0)

    def test_runtime_evil_copied_jester_scores_truthful_real_then_false_raw(self):
        state = _state(["Jester", "Knight", "Wretch", "Bard"])
        scenario = Scenario(
            evil_positions={1: "Pooka"},
            shaman_trace=ShamanTrace(
                source_position=5,
                target_position=1,
                copied_role="Jester",
                target_previous_roles=["Pooka"],
            ),
        )

        likelihoods = _jester_observation_likelihoods(
            [2, 3, 4], 1, scenario, state
        )

        self.assertEqual(set(likelihoods), {(1, 0), (1, 2), (1, 3)})
        for probability in likelihoods.values():
            self.assertAlmostEqual(probability, 1 / 3)

    def test_ambiguous_copied_real_surface_fails_closed(self):
        state = _state(["Jester", "Knight", "Wretch", "Bard", "Bard"])
        scenario = Scenario(
            evil_positions={1: "Pooka"},
            shaman_trace=ShamanTrace(
                source_position=5,
                target_position=1,
                copied_role="Unknown",
                target_previous_roles=["Pooka"],
            ),
        )

        self.assertEqual(
            _jester_observation_likelihoods([2, 3, 4], 1, scenario, state),
            {},
        )
        self.assertIsNone(
            _recommend_jester_ability(
                1,
                [[2, 3, 4]],
                state,
                _result([scenario]),
            )
        )

    def test_recommender_scores_native_counts_and_can_select_self(self):
        state = _state(["Jester", "Wretch", "Knight", "Bard"])
        scenarios = [
            Scenario(evil_positions={}),
            Scenario(evil_positions={3: "Minion"}),
        ]

        recommendation = _recommend_jester_ability(
            1,
            [[1, 2, 3]],
            state,
            _result(scenarios),
        )

        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.targets, [1, 2, 3])
        self.assertAlmostEqual(recommendation.score, 1.0)
        self.assertIn("native Jester counts", recommendation.reasoning)


class JesterNativeCandidateTests(unittest.TestCase):
    def test_current_candidates_include_every_physical_lifecycle_seat(self):
        state = _state(
            ["Jester", "Knight", "Wretch", "Bard", "Judge", "Bard"],
            executed=[2],
            night_kills=[3],
            blocked=[4],
        )
        # Position 6 has no revealed CardInfo entry, but the physical object is
        # still a legal native picker target.
        state.cards = [card for card in state.cards if card.position != 6]
        result = _result([Scenario(evil_positions={})])

        with patch("strategy._recommend_jester_ability", return_value=None) as recommend:
            recommend_abilities(state, result, used_abilities=[])

        candidates = recommend.call_args.args[1]
        self.assertIn([1, 2, 3], candidates)
        self.assertIn([1, 4, 5], candidates)
        self.assertIn([2, 3, 6], candidates)

    def test_unmarked_jester_keeps_legacy_target_filtering(self):
        state = _state(
            ["Jester", "Knight", "Wretch", "Bard", "Baker"],
            marker=None,
            executed=[2],
        )
        result = _result([Scenario(evil_positions={})])

        with (
            patch("strategy._recommend_count_ability", return_value=None) as legacy,
            patch("strategy._recommend_jester_ability") as current,
        ):
            recommend_abilities(state, result, used_abilities=[])

        self.assertEqual(legacy.call_args.args[3], [[3, 4, 5]])
        current.assert_not_called()

    def test_unknown_version_does_not_fall_back_to_legacy(self):
        state = _state(
            ["Jester", "Knight", "Wretch", "Bard"],
            marker="future_variant",
        )
        result = _result([Scenario(evil_positions={})])

        with (
            patch("strategy._recommend_count_ability") as legacy,
            patch("strategy._recommend_jester_ability") as current,
        ):
            recommendations = recommend_abilities(
                state,
                result,
                used_abilities=[],
            )

        legacy.assert_not_called()
        current.assert_not_called()
        self.assertFalse(
            any(rec.ability_name == "Jester" for rec in recommendations)
        )


if __name__ == "__main__":
    unittest.main()
