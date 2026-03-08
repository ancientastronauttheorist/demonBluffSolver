import tempfile
import unittest

from game_loop import GameSession, _release_session_lock
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
            reveal_order=[2, 4],
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

    def test_game_session_save_load_round_trip_preserves_metadata(self):
        session = GameSession(5, 2)
        session.set_deck(
            villagers=["Confessor", "Bard"],
            outcasts=["Plague_Doctor"],
            minions=["Minion"],
            demons=["Pooka"],
        )
        session.board_villager_count = 2
        session.board_outcast_count = 1
        session.cards = [CardInfo(2, "Confessor", info_parsed={"dizzy": True})]
        session.used_abilities = [4]
        session.pd_ability_results = [{
            "pd_pos": 3,
            "target": 2,
            "is_corrupted": True,
            "evil_revealed": 5,
        }]

        with tempfile.TemporaryDirectory() as tmpdir:
            path = f"{tmpdir}/session.json"
            session.save(path)
            loaded = GameSession.load(path)
            _release_session_lock()

        self.assertEqual(loaded.villagers, session.villagers)
        self.assertEqual(loaded.outcasts, session.outcasts)
        self.assertEqual(loaded.minions, session.minions)
        self.assertEqual(loaded.demons, session.demons)
        self.assertEqual(loaded.board_villager_count, 2)
        self.assertEqual(loaded.board_outcast_count, 1)
        self.assertEqual(loaded.used_abilities, [4])
        self.assertEqual(loaded.pd_ability_results, session.pd_ability_results)

    def test_reveal_order_defaults_empty_for_legacy_data(self):
        data = {
            "n_cards": 4,
            "n_evil": 1,
            "villagers": ["Confessor"],
            "outcasts": [],
            "minions": ["Minion"],
            "demons": [],
            "cards": [],
        }
        loaded = GameState.from_dict(data)
        self.assertEqual(loaded.reveal_order, [])

    def test_reveal_order_round_trip(self):
        state = GameState(
            n_cards=5,
            deck=DeckComposition(
                villagers=["Confessor"],
                outcasts=[],
                minions=["Minion"],
                demons=["Baa"],
            ),
            cards=[CardInfo(3, "Confessor")],
            n_evil=1,
            reveal_order=[3, 1, 5],
        )
        data = state.to_dict()
        self.assertEqual(data["reveal_order"], [3, 1, 5])
        loaded = GameState.from_dict(data)
        self.assertEqual(loaded.reveal_order, [3, 1, 5])

    def test_session_add_card_populates_reveal_order(self):
        session = GameSession(5, 1)
        session.set_deck(["Confessor", "Bard"], [], ["Minion"], ["Baa"])
        session.add_card(CardInfo(3, "Confessor"))
        session.add_card(CardInfo(1, "Bard"))
        session.add_card(CardInfo(3, "Confessor"))  # re-read, should not duplicate
        self.assertEqual(session.reveal_order, [3, 1])


if __name__ == "__main__":
    unittest.main()
