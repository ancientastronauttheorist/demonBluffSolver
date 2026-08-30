"""Native execution-cost regressions for Python forced-safe lookahead."""

import unittest

from solver import CardInfo, DeckComposition, GameState, Scenario, SolverResult
from strategy import _find_forced_execution


def _state(*, hp: int, n_cards: int = 2) -> GameState:
    cards = [
        CardInfo(1, "Knight", info_parsed={}),
        CardInfo(2, "Hunter", info_parsed={}),
    ]
    if n_cards == 3:
        cards.append(CardInfo(3, "Baker", info_parsed={}))
    return GameState(
        n_cards=n_cards,
        deck=DeckComposition([], [], [], ["Pooka"]),
        cards=cards,
        n_evil=1,
        hp=hp,
        wrong_exec_cost=5,
    )


def _result(*scenarios: Scenario) -> SolverResult:
    return SolverResult(
        definite_evil=[],
        definite_good=[],
        bombardier_positions=[],
        n_scenarios=len(scenarios),
        n_surviving=len(scenarios),
        surviving_scenarios=list(scenarios),
    )


class ForcedExecutionTests(unittest.TestCase):
    def test_corrupted_knight_nine_damage_branch_is_not_forced_safe(self):
        state = _state(hp=6)
        result = _result(
            Scenario(evil_positions={1: "Pooka"}),
            Scenario(evil_positions={2: "Pooka"}, corrupted={1}),
        )

        # #1's corrupted-good branch costs 9 and is fatal. #2 remains safe:
        # its good branch costs 5, leaving 1 HP before the guaranteed evil.
        self.assertEqual(_find_forced_execution(state, result, [1, 2]), 2)

    def test_doppelganger_as_knight_models_healthy_bluff_protection(self):
        state = _state(hp=5)
        result = _result(
            Scenario(evil_positions={1: "Pooka"}),
            Scenario(evil_positions={2: "Pooka"}, doppelganger_position=1),
        )

        # #1 either kills the evil or is safely blocked as a good Doppelganger
        # with a HealthyBluff Knight role, after which #2 is forced evil.
        self.assertEqual(_find_forced_execution(state, result, [1, 2]), 1)

    def test_zero_hp_cannot_resolve_as_a_win(self):
        state = _state(hp=0)
        result = _result(Scenario(evil_positions={}))

        self.assertIsNone(_find_forced_execution(state, result, [1, 2]))

    def test_protected_identity_does_not_leak_hidden_real_role(self):
        state = _state(hp=5, n_cards=3)
        result = _result(
            Scenario(evil_positions={2: "Pooka"}),
            Scenario(evil_positions={3: "Pooka"}, doppelganger_position=1),
        )

        # Both #1 outcomes look exactly like a protected Knight. Keeping the
        # scenarios joined leaves #2/#3 ambiguous and either wrong pick fatal.
        self.assertIsNone(_find_forced_execution(state, result, [1, 2, 3]))


if __name__ == "__main__":
    unittest.main()
