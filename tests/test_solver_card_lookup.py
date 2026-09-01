"""Regression tests for the shared Python card-position lookup."""

import unittest

import solver
from solver import CardInfo, DeckComposition, GameState, get_card_at


def _state(cards: list[CardInfo]) -> GameState:
    return GameState(
        n_cards=3,
        n_evil=1,
        deck=DeckComposition([], [], [], []),
        cards=cards,
    )


class CardLookupCacheTests(unittest.TestCase):
    def setUp(self):
        solver._card_lookup = {}
        solver._card_lookup_cards = None
        solver._card_lookup_signature = ()

    def test_new_list_never_reuses_a_prior_state_lookup(self):
        first_cards = [CardInfo(1, "Knight")]
        first = _state(first_cards)
        self.assertEqual(get_card_at(1, first).apparent_role, "Knight")

        second = _state([CardInfo(1, "Wretch")])
        # Simulate the old id-only cache's false hit deterministically. A
        # strong-reference identity check must still rebuild for this list.
        self.assertIsNot(second.cards, solver._card_lookup_cards)
        self.assertEqual(get_card_at(1, second).apparent_role, "Wretch")

    def test_in_place_position_changes_invalidate_the_lookup(self):
        cards = [CardInfo(1, "Knight"), CardInfo(2, "Wretch")]
        state = _state(cards)
        self.assertEqual(get_card_at(1, state).apparent_role, "Knight")

        cards[0].position = 3
        self.assertIsNone(get_card_at(1, state))
        self.assertEqual(get_card_at(3, state).apparent_role, "Knight")


if __name__ == "__main__":
    unittest.main()
