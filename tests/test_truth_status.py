"""Focused parity tests for the Python CheckLying truth model."""

import unittest

from solver import (
    CardInfo,
    DeckComposition,
    GameState,
    Scenario,
    TruthStatus,
    truth_status,
)


def make_state(apparent_role: str = "Baker") -> GameState:
    return GameState(
        n_cards=1,
        deck=DeckComposition(villagers=[], outcasts=[], minions=[], demons=[]),
        cards=[CardInfo(position=1, apparent_role=apparent_role)],
    )


class TruthStatusTests(unittest.TestCase):
    def test_corrupted_confessor_lies_despite_cant_lie_role(self):
        scenario = Scenario(evil_positions={}, corrupted={1})

        self.assertEqual(
            truth_status(1, scenario, make_state("Confessor")), TruthStatus.LYING
        )

    def test_corruption_overrides_puppet_healthy_bluff(self):
        scenario = Scenario(evil_positions={}, puppet_position=1, corrupted={1})

        self.assertEqual(truth_status(1, scenario, make_state()), TruthStatus.LYING)

    def test_corruption_overrides_doppelganger_healthy_bluff(self):
        scenario = Scenario(
            evil_positions={}, doppelganger_position=1, corrupted={1}
        )

        self.assertEqual(truth_status(1, scenario, make_state()), TruthStatus.LYING)

    def test_clean_puppet_and_doppelganger_model_healthy_bluff(self):
        puppet = Scenario(evil_positions={}, puppet_position=1)
        doppelganger = Scenario(evil_positions={}, doppelganger_position=1)

        self.assertEqual(truth_status(1, puppet, make_state()), TruthStatus.TRUTHFUL)
        self.assertEqual(
            truth_status(1, doppelganger, make_state()), TruthStatus.TRUTHFUL
        )

    def test_drunk_models_non_null_bluff_without_healthy_bluff(self):
        scenario = Scenario(evil_positions={}, drunk_position=1)

        self.assertEqual(truth_status(1, scenario, make_state()), TruthStatus.LYING)

    def test_ordinary_evil_lies_even_when_appearing_as_confessor(self):
        scenario = Scenario(evil_positions={1: "Pooka"})

        self.assertEqual(
            truth_status(1, scenario, make_state("Confessor")), TruthStatus.LYING
        )

    def test_clean_good_character_is_truthful(self):
        scenario = Scenario(evil_positions={})

        self.assertEqual(
            truth_status(1, scenario, make_state("Baker")), TruthStatus.TRUTHFUL
        )


if __name__ == "__main__":
    unittest.main()
