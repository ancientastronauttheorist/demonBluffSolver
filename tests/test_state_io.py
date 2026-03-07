import unittest

from solver import CardInfo, DeckComposition, GameState


class TestGameStateIO(unittest.TestCase):
    def test_round_trip_nested_deck(self):
        state = GameState(
            n_cards=5,
            deck=DeckComposition(
                villagers=["Confessor"],
                outcasts=["Bombardier"],
                minions=["Minion"],
                demons=["Baa"],
            ),
            cards=[CardInfo(2, "Confessor", info_parsed={"dizzy": False})],
            n_evil=2,
            executed=[4],
            confirmed_evil=[4],
            executed_evil_roles={4: "Minion"},
            hp=7,
            wrong_exec_cost=5,
        )

        data = state.to_dict()
        loaded = GameState.from_dict(data)

        self.assertEqual(loaded, state)

    def test_loads_legacy_flat_deck_shape(self):
        data = {
            "n_cards": 4,
            "n_evil": 1,
            "villagers": ["Confessor", "Enlightened"],
            "outcasts": [],
            "minions": ["Minion"],
            "demons": [],
            "cards": [{"position": 1, "apparent_role": "Confessor", "info_parsed": {"dizzy": False}}],
            "executed_evil_roles": {"3": "Minion"},
        }

        loaded = GameState.from_dict(data)

        self.assertEqual(loaded.deck.villagers, ["Confessor", "Enlightened"])
        self.assertEqual(loaded.executed_evil_roles, {3: "Minion"})


if __name__ == "__main__":
    unittest.main()
