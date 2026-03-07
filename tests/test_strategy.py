import json
import unittest
from pathlib import Path

from solver import CardInfo, DeckComposition, GameState, Scenario, SolverResult, solve
from strategy import recommend_action


class TestStrategyRecommendations(unittest.TestCase):
    def _load_live_case_midstate(self) -> GameState:
        case_path = Path(__file__).parent / "cases" / "asc10_g10_live.json"
        state = GameState.from_dict(json.loads(case_path.read_text()))

        state.executed = [7]
        state.confirmed_evil = []
        state.confirmed_good = [7]
        state.executed_evil_roles = {}
        state.slayer_results = []
        state.pd_ability_results = [{
            "pd_pos": 6,
            "target": 1,
            "is_corrupted": True,
            "evil_revealed": 2,
        }]
        state.used_abilities = [6]
        state.hp = 6

        for card in state.cards:
            if card.position in {3, 5, 8}:
                card.info_parsed = {}

        return state

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

    def test_live_midgame_prefers_better_fortune_teller_pair(self):
        state = self._load_live_case_midstate()
        result = solve(state)

        action = recommend_action(state, result, used_abilities=state.used_abilities)

        self.assertEqual(action.action_type, "use_ability")
        self.assertEqual(action.ability_name, "Fortune Teller")
        self.assertEqual(action.targets, [1, 9])

    def test_corrupted_judge_is_not_treated_as_clean_liar(self):
        state = GameState(
            n_cards=4,
            deck=DeckComposition(
                villagers=["Judge", "Baker", "Baker"],
                outcasts=[],
                minions=["Minion"],
                demons=[],
            ),
            cards=[
                CardInfo(1, "Judge"),
                CardInfo(2, "Baker"),
                CardInfo(3, "Baker"),
            ],
            n_evil=1,
        )
        scenarios = [
            Scenario(evil_positions={2: "Minion"}, corrupted={1}),
            Scenario(evil_positions={4: "Minion"}, corrupted={1}),
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

        self.assertEqual(action.action_type, "reveal")
        self.assertEqual(action.position, 4)

    def test_definite_evil_disguised_as_knight_is_executed(self):
        case_path = Path(__file__).parent / "cases" / "asc11_g1_live.json"
        state = GameState.from_dict(json.loads(case_path.read_text()))

        state.executed = [5, 9]
        state.confirmed_evil = [5, 9]
        state.executed_evil_roles = {
            5: "Minion",
            9: "Twin_Minion",
        }
        state.hp = 10
        state.used_abilities = [8, 3, 9]

        result = solve(state)
        action = recommend_action(state, result, used_abilities=state.used_abilities)

        self.assertEqual(set(result.definite_evil), {5, 7, 9})
        self.assertEqual(action.action_type, "execute")
        self.assertEqual(action.position, 7)


if __name__ == "__main__":
    unittest.main()
