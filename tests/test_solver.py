import unittest

from solver import DeckComposition, GameState, solve


class TestSolverSetupEffects(unittest.TestCase):
    def test_puppeteer_fresh_board_allows_hidden_puppet_target(self):
        state = GameState(
            n_cards=9,
            deck=DeckComposition(
                villagers=[
                    "Oracle",
                    "Lover",
                    "Druid",
                    "Enlightened",
                    "Alchemist",
                    "Bishop",
                ],
                outcasts=["Plague_Doctor"],
                minions=["Witch", "Puppeteer"],
                demons=["Lilis"],
            ),
            cards=[],
            n_evil=4,
            board_villager_count=5,
            board_outcast_count=1,
        )

        result = solve(state)

        self.assertGreater(result.n_surviving, 0)
        self.assertTrue(any(s.puppet_position is not None for s in result.surviving_scenarios))


if __name__ == "__main__":
    unittest.main()
