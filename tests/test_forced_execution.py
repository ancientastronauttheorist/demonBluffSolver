"""Native execution-cost regressions for Python forced-safe lookahead."""

import unittest

from solver import (
    CardInfo,
    ChancellorTrace,
    DeckComposition,
    GameState,
    Scenario,
    ShamanTrace,
    SolverResult,
)
from strategy import _find_forced_execution, _shallow_lookahead


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

    def test_generated_bomb_candidate_is_recovered_only_after_branch_pruning(self):
        state = GameState(
            n_cards=2,
            deck=DeckComposition([], ["Bombardier"], [], ["Pooka"]),
            cards=[CardInfo(1, "Hunter"), CardInfo(2, "Hunter")],
            n_evil=1,
            hp=6,
            wrong_exec_cost=5,
        )
        result = _result(
            Scenario(
                evil_positions={1: "Pooka"},
                chancellor_trace=ChancellorTrace(
                    original_positions=[1],
                    added_outcast_position=2,
                    added_outcast_role="Bombardier",
                ),
            ),
            Scenario(evil_positions={2: "Pooka"}),
        )
        result.bombardier_positions = [2]

        # #2 is illegal at the root. Executing #1 distinguishes the worlds:
        # the Bomb world is already won, while the other branch proves #2 is
        # current Pooka and can safely finish the game.
        self.assertIsNone(_find_forced_execution(state, result, [2]))
        self.assertEqual(_find_forced_execution(state, result, [1, 2]), 1)

    def test_shaman_current_bomb_is_a_branch_local_role(self):
        state = GameState(
            n_cards=3,
            deck=DeckComposition([], ["Bombardier"], ["Shaman"], ["Pooka"]),
            cards=[
                CardInfo(1, "Hunter"),
                CardInfo(2, "Hunter"),
                CardInfo(3, "Scout"),
            ],
            n_evil=1,
            hp=6,
            wrong_exec_cost=5,
        )
        result = _result(
            Scenario(
                evil_positions={1: "Pooka"},
                shaman_trace=ShamanTrace(3, 2, "Bombardier"),
            ),
            Scenario(evil_positions={2: "Pooka"}),
        )
        result.bombardier_positions = [2, 3]

        self.assertEqual(_find_forced_execution(state, result, [1, 2]), 1)

    def test_shallow_reveal_can_remove_a_modeled_bomb_world(self):
        state = GameState(
            n_cards=3,
            deck=DeckComposition([], ["Bombardier"], [], ["Pooka"]),
            cards=[CardInfo(1, "Hunter"), CardInfo(2, "Hunter")],
            n_evil=1,
            hp=6,
            wrong_exec_cost=5,
        )
        result = _result(
            Scenario(
                evil_positions={1: "Pooka"},
                corrupted={3},
                chancellor_trace=ChancellorTrace(
                    original_positions=[1],
                    added_outcast_position=2,
                    added_outcast_role="Bombardier",
                ),
            ),
            Scenario(evil_positions={2: "Pooka"}),
        )
        result.bombardier_positions = [2]

        action = _shallow_lookahead(state, result, [1, 2])

        self.assertIsNotNone(action)
        self.assertEqual(action.action_type, "reveal")
        self.assertEqual(action.position, 3)

    def test_stable_twin_overlap_stays_opaque_after_modeled_bomb_prunes(self):
        state = GameState(
            n_cards=2,
            deck=DeckComposition(
                [], ["Bombardier"], ["Twin Minion"], ["Pooka"],
            ),
            cards=[CardInfo(1, "Hunter"), CardInfo(2, "Hunter")],
            n_evil=1,
            hp=6,
            wrong_exec_cost=5,
        )
        result = _result(
            Scenario(evil_positions={2: "Twin Minion"}),
            Scenario(
                evil_positions={1: "Pooka"},
                chancellor_trace=ChancellorTrace(
                    original_positions=[1],
                    added_outcast_position=2,
                    added_outcast_role="Bombardier",
                ),
            ),
        )
        result.bombardier_positions = [2]

        # The second world models current Bomb at #2, but the first world also
        # carries the pre-trace stable-Twin risk at that same seat. Executing
        # #1 must not make the opaque Twin path disappear.
        self.assertIsNone(_find_forced_execution(state, result, [1, 2]))


if __name__ == "__main__":
    unittest.main()
