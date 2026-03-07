import unittest

from solver import CardInfo, DeckComposition, GameState, Scenario, SolverResult
from strategy import recommend_action


class TestStrategyRecommendations(unittest.TestCase):
    def test_does_not_declare_win_when_scenarios_disagree(self):
        state = GameState(
            n_cards=2,
            deck=DeckComposition(villagers=[], outcasts=[], minions=["Minion"], demons=[]),
            cards=[],
            n_evil=1,
            executed=[1],
        )
        result = SolverResult(
            definite_evil=[],
            definite_good=[],
            bombardier_positions=[],
            n_scenarios=2,
            n_surviving=2,
            surviving_scenarios=[
                Scenario(evil_positions={}),
                Scenario(evil_positions={2: "Minion"}),
            ],
            reasoning=[],
        )

        action = recommend_action(state, result, used_abilities=[])

        self.assertNotEqual(action.action_type, "win")

    def test_prefers_informative_jester_over_reveal(self):
        state = GameState(
            n_cards=6,
            deck=DeckComposition(
                villagers=["Jester", "Confessor", "Confessor", "Confessor"],
                outcasts=["Bombardier"],
                minions=["Minion"],
                demons=["Baa"],
            ),
            cards=[
                CardInfo(1, "Jester"),
                CardInfo(2, "Confessor"),
                CardInfo(3, "Confessor"),
                CardInfo(4, "Confessor"),
            ],
            n_evil=2,
        )
        scenarios = [
            Scenario(evil_positions={5: "Minion", 6: "Baa"}),
            Scenario(evil_positions={2: "Minion", 5: "Baa"}),
            Scenario(evil_positions={3: "Minion", 5: "Baa"}),
            Scenario(evil_positions={4: "Minion", 5: "Baa"}),
            Scenario(evil_positions={2: "Minion", 3: "Baa"}),
            Scenario(evil_positions={2: "Minion", 4: "Baa"}),
            Scenario(evil_positions={3: "Minion", 4: "Baa"}),
        ]
        result = SolverResult(
            definite_evil=[],
            definite_good=[],
            bombardier_positions=[],
            n_scenarios=len(scenarios),
            n_surviving=len(scenarios),
            surviving_scenarios=scenarios,
            reasoning=[],
        )

        action = recommend_action(state, result, used_abilities=[])

        self.assertEqual(action.action_type, "use_ability")
        self.assertEqual(action.ability_name, "Jester")


if __name__ == "__main__":
    unittest.main()
