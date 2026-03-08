"""Regression tests for the Demon Bluff solver using known game sessions."""

import unittest
from solver import (
    GameState, DeckComposition, CardInfo, SolverResult, Scenario,
    _apply_post_corruption, circle_distance, circle_direction, adjacent_positions, solve,
    _check_scenario, _is_evil_in_scenario, _validate_bishop, _validate_poet, _validate_role_counts,
)


class TestCircleGeometry(unittest.TestCase):
    def test_distance(self):
        self.assertEqual(circle_distance(1, 4, 7), 3)
        self.assertEqual(circle_distance(1, 7, 7), 1)
        self.assertEqual(circle_distance(1, 1, 7), 0)
        self.assertEqual(circle_distance(3, 6, 8), 3)

    def test_direction(self):
        self.assertEqual(circle_direction(1, 3, 7), "CW")
        self.assertEqual(circle_direction(1, 7, 7), "CCW")
        self.assertEqual(circle_direction(1, 4, 7), "CW")  # 3 CW vs 4 CCW
        self.assertEqual(circle_direction(1, 5, 8), "Equidistant")  # 4 each way

    def test_adjacent(self):
        self.assertEqual(sorted(adjacent_positions(1, 7)), [2, 7])
        self.assertEqual(sorted(adjacent_positions(4, 7)), [3, 5])
        self.assertEqual(sorted(adjacent_positions(7, 7)), [1, 6])


class TestSyntheticSimple(unittest.TestCase):
    """Simple 5-card puzzle: 1 evil Minion.
    Cards: #1 Enlightened, #2 Lover, #3 Knitter, #4 Confessor, #5 Minion(disguised as Slayer).
    Minion at #5 → lies, disguises as Slayer."""

    def _make_state(self):
        deck = DeckComposition(
            villagers=["Enlightened", "Lover", "Knitter", "Confessor"],
            outcasts=[],
            minions=["Minion"],
            demons=[],
        )
        # Compute true values with Minion at #5:
        # #1 Enlightened: nearest evil = #5. Dir 1→5 on 5-card: CW dist 4, CCW dist 1 → CCW.
        #   Truthful → claims "CCW"
        # #2 Lover: adj = #1(Good), #3(Good). 0 evil adj. Truthful → claims 0.
        # #3 Knitter: evil pairs adj? Evil = {#5}. Only 1 evil, can't have pair. 0 pairs.
        #   Truthful → claims 0.
        # #4 Confessor: not evil, not corrupted → dizzy=False. Can't lie → claims False.
        # #5 Minion (disguised as Slayer): no info to validate.
        cards = [
            CardInfo(1, "Enlightened", info_parsed={"direction": "CCW"}),
            CardInfo(2, "Lover", info_parsed={"evil_adjacent": 0}),
            CardInfo(3, "Knitter", info_parsed={"evil_pairs": 0}),
            CardInfo(4, "Confessor", info_parsed={"dizzy": False}),
            CardInfo(5, "Slayer"),
        ]
        return GameState(n_cards=5, deck=deck, cards=cards, n_evil=1)

    def test_finds_evil(self):
        state = self._make_state()
        result = solve(state)
        self.assertIn(5, result.definite_evil,
                      f"Expected #5 evil. Reasoning: {result.reasoning}")
        self.assertEqual(len(result.definite_evil), 1)

    def test_good_positions(self):
        state = self._make_state()
        result = solve(state)
        for pos in [1, 2, 3, 4]:
            self.assertIn(pos, result.definite_good)


class TestSession8(unittest.TestCase):
    """Session 8: 7 cards, 2 evils (Chancellor + Pooka).
    Solution: #3=Pooka, #6=Chancellor.
    Pooka at #3 corrupts #2(Lover) and #4(Knitter).
    Key: Gemcrafter #3 says '#5 is Good' but Wretch registers as Evil → #3 is lying."""

    def _make_state(self):
        deck = DeckComposition(
            villagers=["Bard", "Knitter", "Gemcrafter", "Lover", "Alchemist"],
            outcasts=["Wretch", "Bombardier"],
            minions=["Chancellor"],
            demons=["Pooka"],
        )
        # True state: #3=Pooka, #6=Chancellor
        # Pooka at #3 corrupts adj #2 and #4
        # #1 Bard (truthful): dist to nearest corrupted. Corrupted={#2,#4}. #2 at dist 1. → 1
        # #2 Lover (corrupted→lying): adj #1(Good), #3(Evil). 1 evil adj. Lies: claims 0
        # #3 Gemcrafter (evil Pooka→lying): "#5 is Good". Wretch effective=Evil. Lie ✓
        # #4 Knitter (corrupted→lying): evil pairs. Effective evil={#3,#5(Wretch),#6}.
        #   Pairs: (#5,#6) adj. 1 pair. Lies: claims 0
        # #5 Wretch: no info
        # #6 Bombardier (evil Chancellor): no info
        # #7 Bombardier: no info
        cards = [
            CardInfo(1, "Bard", info_parsed={"corruption_distance": 1}),
            CardInfo(2, "Lover", info_parsed={"evil_adjacent": 0}),
            CardInfo(3, "Gemcrafter", info_parsed={"good_position": 5}),
            CardInfo(4, "Knitter", info_parsed={"evil_pairs": 0}),
            CardInfo(5, "Wretch"),
            CardInfo(6, "Bombardier"),
            CardInfo(7, "Bombardier"),
        ]
        return GameState(n_cards=7, deck=deck, cards=cards, n_evil=2)

    def test_pooka_found(self):
        """Gemcrafter calling Wretch 'Good' proves #3 is lying = evil."""
        state = self._make_state()
        result = solve(state)
        self.assertIn(3, result.definite_evil,
                      f"Expected #3 evil. Reasoning: {result.reasoning}")

    def test_bombardier_safe(self):
        """Real Bombardier (#7) should never be in definite_evil."""
        state = self._make_state()
        result = solve(state)
        self.assertNotIn(7, result.definite_evil)

    def test_correct_solution_survives(self):
        """The known solution (#3=Pooka, #6=Chancellor) must be in surviving scenarios."""
        state = self._make_state()
        result = solve(state)
        found = False
        for s in result.surviving_scenarios:
            if (s.evil_positions.get(3) == "Pooka" and
                    s.evil_positions.get(6) == "Chancellor"):
                found = True
                break
        self.assertTrue(found,
                        f"Expected solution not in survivors. "
                        f"Surviving: {[(dict(s.evil_positions), s.corrupted) for s in result.surviving_scenarios]}")


class TestSession11(unittest.TestCase):
    """Session 11: 8 cards, 3 evils (Puppeteer + Puppet + Pooka).
    Solution: #3=Puppeteer, #4=Puppet (converted from Enlightened), #5=Pooka.
    Pooka at #5 corrupts adj #6 (Slayer). #4 is evil, not corrupted."""

    def _make_state(self):
        deck = DeckComposition(
            villagers=["Knitter", "Scout", "Jester", "Slayer",
                       "Enlightened", "Confessor"],
            outcasts=["Plague Doctor"],
            minions=["Puppeteer"],
            demons=["Pooka"],
        )
        # True state: #3=Puppeteer, #4=Puppet, #5=Pooka
        # Pooka at #5 corrupts adj villagers: #4 is Puppet(evil)→skip, #6 Slayer→corrupted
        # Corrupted = {#6}
        #
        # #1 Confessor (truthful, can't lie): not evil, not corrupted → dizzy=False
        # #2 Knitter (truthful): evil pairs adj.
        #   Evil effective: {#3,#4,#5}. Pairs: (#3,#4) ✓, (#4,#5) ✓. 2 pairs. Claims 2.
        # #3 Enlightened (Puppeteer, evil→lying): nearest evil from #3.
        #   Other evil: #4 at dist 1 CW, #5 at dist 2 CW. Nearest: #4 CW.
        #   Truth = "CW". Lies: claims "CCW" (≠CW ✓)
        # #4 Enlightened (Puppet, evil but can't lie→truthful): nearest evil from #4.
        #   Other evil: #3 at dist 1 CCW, #5 at dist 1 CW. Equidistant.
        #   Truth = "Equidistant". Claims "Equidistant" ✓
        # #5 Scout (Pooka, evil→lying): "Pooka is N from nearest Evil"
        #   Pooka at #5. Nearest other evil: #4 at dist 1. Truth = 1.
        #   Lies: claims 2 (≠1 ✓)
        # #6 Slayer (corrupted→lying): no info ability for Slayer. Skip.
        # #7 Jester (truthful): picks [1,3,5], evil count.
        #   #1=Good, #3=Evil, #5=Evil. Count=2. Claims 2 ✓
        # #8 Plague Doctor: no standard info to validate
        cards = [
            CardInfo(1, "Confessor", info_parsed={"dizzy": False}),
            CardInfo(2, "Knitter", info_parsed={"evil_pairs": 2}),
            CardInfo(3, "Enlightened", info_parsed={"direction": "CCW"}),
            CardInfo(4, "Enlightened", info_parsed={"direction": "Equidistant"}),
            CardInfo(5, "Scout", info_parsed={
                "evil_role": "Pooka", "distance": 2}),
            CardInfo(6, "Slayer"),
            CardInfo(7, "Jester", info_parsed={
                "targets": [1, 3, 5], "evil_count": 2}),
            CardInfo(8, "Plague Doctor"),
        ]
        return GameState(n_cards=8, deck=deck, cards=cards, n_evil=3,
                        pd_corruption_target=6)  # PD corrupted #6 Slayer

    def test_finds_all_evils(self):
        state = self._make_state()
        result = solve(state)
        for pos in [3, 4, 5]:
            self.assertIn(pos, result.definite_evil,
                          f"Expected #{pos} evil. Surviving: {result.n_surviving}. "
                          f"Reasoning: {result.reasoning}")

    def test_good_positions(self):
        state = self._make_state()
        result = solve(state)
        for pos in [1, 2, 6, 7, 8]:
            self.assertIn(pos, result.definite_good,
                          f"Expected #{pos} good. Reasoning: {result.reasoning}")


class TestExecutedEvil(unittest.TestCase):
    """After executing one evil, solver should find remaining evil(s)."""

    def test_one_remaining(self):
        deck = DeckComposition(
            villagers=["Confessor", "Confessor", "Confessor"],
            outcasts=[],
            minions=["Minion"],
            demons=["Baa"],
        )
        # 5 cards, 2 evils. Minion at #3 already executed.
        # Remaining: Baa at #5 (disguised as Enlightened, lying)
        # Confessors at #1, #2, #4 all say dizzy=False → can't be evil
        # Only #5 can be Baa
        cards = [
            CardInfo(1, "Confessor", info_parsed={"dizzy": False}),
            CardInfo(2, "Confessor", info_parsed={"dizzy": False}),
            CardInfo(4, "Confessor", info_parsed={"dizzy": False}),
            CardInfo(5, "Enlightened"),
        ]
        state = GameState(
            n_cards=5, deck=deck, cards=cards, n_evil=2,
            executed=[3], confirmed_evil=[3],
        )
        result = solve(state)
        self.assertIn(5, result.definite_evil,
                      f"Expected #5 evil. Reasoning: {result.reasoning}")


class TestHiddenOutcastPresence(unittest.TestCase):
    @staticmethod
    def _revealed_villager_board():
        return [
            CardInfo(1, "Gemcrafter"),
            CardInfo(2, "Knitter"),
            CardInfo(3, "Architect"),
            CardInfo(4, "Knitter"),
        ]

    def test_doppelganger_forced_when_only_outcast_slot(self):
        deck = DeckComposition(
            villagers=["Gemcrafter", "Knitter", "Architect"],
            outcasts=["Doppelganger"],
            minions=["Minion"],
            demons=[],
        )
        state = GameState(
            n_cards=4,
            deck=deck,
            cards=self._revealed_villager_board(),
            n_evil=1,
            board_villager_count=3,
            board_outcast_count=1,
        )

        result = solve(state)

        self.assertGreater(result.n_surviving, 0)
        self.assertTrue(all(s.doppelganger_position is not None for s in result.surviving_scenarios))

    def test_doppelganger_absent_when_zero_outcast_slots(self):
        deck = DeckComposition(
            villagers=["Gemcrafter", "Knitter", "Architect"],
            outcasts=["Doppelganger"],
            minions=["Minion"],
            demons=[],
        )
        state = GameState(
            n_cards=4,
            deck=deck,
            cards=self._revealed_villager_board(),
            n_evil=1,
            board_villager_count=3,
            board_outcast_count=0,
        )

        result = solve(state)

        self.assertGreater(result.n_surviving, 0)
        self.assertTrue(all(s.doppelganger_position is None for s in result.surviving_scenarios))


class TestDuplicateRoleAllowance(unittest.TestCase):
    def test_absent_hidden_outcast_does_not_explain_duplicate_villager(self):
        deck = DeckComposition(
            villagers=["Jester"],
            outcasts=["Drunk"],
            minions=[],
            demons=[],
        )
        state = GameState(
            n_cards=2,
            deck=deck,
            cards=[CardInfo(1, "Jester"), CardInfo(2, "Jester")],
            n_evil=0,
            board_villager_count=2,
            board_outcast_count=0,
        )
        scenario = Scenario(evil_positions={}, drunk_position=None)

        self.assertFalse(_validate_role_counts(scenario, state))

    def test_present_hidden_outcast_can_explain_duplicate_villager(self):
        deck = DeckComposition(
            villagers=["Jester"],
            outcasts=["Drunk"],
            minions=[],
            demons=[],
        )
        state = GameState(
            n_cards=2,
            deck=deck,
            cards=[CardInfo(1, "Jester"), CardInfo(2, "Jester")],
            n_evil=0,
            board_villager_count=1,
            board_outcast_count=1,
        )
        scenario = Scenario(evil_positions={}, drunk_position=2)

        self.assertTrue(_validate_role_counts(scenario, state))


class TestHiddenOutcastValidation(unittest.TestCase):
    def _make_live_midstate(self):
        deck = DeckComposition(
            villagers=["Bishop", "Scout", "Hunter", "Alchemist", "Poet", "Jester", "Witness"],
            outcasts=["Drunk", "Plague_Doctor", "Doppelganger"],
            minions=["Minion", "Chancellor"],
            demons=["Baa"],
        )
        cards = [
            CardInfo(1, "Scout", info_parsed={"evil_role": "Baa", "distance": 2}),
            CardInfo(2, "Scout", info_parsed={"evil_role": "Minion", "distance": 3}),
            CardInfo(3, "Hunter", info_parsed={"distance": 1}),
            CardInfo(4, "Scout", info_parsed={"evil_role": "Chancellor", "distance": 1}),
            CardInfo(5, "Poet"),
            CardInfo(6, "Bishop", info_parsed={"targets": [4, 7, 8], "types": ["Minion", "Outcast", "Villager"]}),
            CardInfo(7, "Hunter", info_parsed={"distance": 2}),
            CardInfo(8, "Alchemist", info_parsed={"cured_count": 0}),
            CardInfo(9, "Jester", info_parsed={"targets": [1, 4, 5], "evil_count": 2}),
        ]
        return GameState(
            n_cards=9,
            deck=deck,
            cards=cards,
            n_evil=3,
            executed=[4, 2, 9],
            confirmed_evil=[4, 2],
            confirmed_good=[9],
            executed_evil_roles={4: "Minion", 2: "Chancellor"},
            hp=5,
            wrong_exec_cost=5,
            board_villager_count=5,
            board_outcast_count=1,
        )

    def test_hidden_outcasts_count_as_outcasts_for_bishop(self):
        state = self._make_live_midstate()
        scenario = Scenario(
            evil_positions={5: "Baa"},
            drunk_position=1,
            doppelganger_position=8,
            corrupted={1},
        )
        self.assertTrue(_check_scenario(scenario, state))

        result = solve(state)

        self.assertGreater(result.n_surviving, 0, f"Reasoning: {result.reasoning}")
        self.assertIn(5, result.definite_evil, f"Reasoning: {result.reasoning}")


class TestPoetRandomInfo(unittest.TestCase):
    def test_poet_random_clue_type_not_in_deck_still_validates(self):
        deck = DeckComposition(
            villagers=["Poet", "Hunter", "Lover", "Knitter"],
            outcasts=[],
            minions=[],
            demons=["Baa"],
        )
        state = GameState(
            n_cards=5,
            deck=deck,
            cards=[
                CardInfo(1, "Poet", info_parsed={"copied_role": "Gemcrafter", "good_position": 3}),
                CardInfo(2, "Hunter", info_parsed={"distance": 2}),
                CardInfo(3, "Lover", info_parsed={"evil_adjacent": 0}),
                CardInfo(4, "Knitter", info_parsed={"evil_pairs": 0}),
            ],
            n_evil=1,
        )
        scenario = Scenario(evil_positions={5: "Baa"})

        self.assertTrue(_validate_poet(state.cards[0], scenario, state))

        result = solve(state)
        self.assertIn(5, result.definite_evil, f"Reasoning: {result.reasoning}")

    def test_poet_bounty_hunter_style_clue_validates(self):
        deck = DeckComposition(
            villagers=["Poet", "Hunter", "Lover", "Knitter"],
            outcasts=[],
            minions=[],
            demons=["Baa"],
        )
        state = GameState(
            n_cards=5,
            deck=deck,
            cards=[CardInfo(1, "Poet", info_parsed={"copied_role": "Bounty Hunter", "evil_position": 5})],
            n_evil=1,
        )
        scenario = Scenario(evil_positions={5: "Baa"})

        self.assertTrue(_validate_poet(state.cards[0], scenario, state))


class TestWretchTypeRegistration(unittest.TestCase):
    def test_bishop_treats_wretch_as_minion_type(self):
        deck = DeckComposition(
            villagers=["Bishop", "Fortune_Teller"],
            outcasts=["Wretch"],
            minions=[],
            demons=[],
        )
        state = GameState(
            n_cards=3,
            deck=deck,
            cards=[
                CardInfo(1, "Bishop", info_parsed={"targets": [2, 3], "types": ["Minion", "Villager"]}),
                CardInfo(2, "Wretch"),
                CardInfo(3, "Fortune Teller"),
            ],
            n_evil=0,
        )
        scenario = Scenario(evil_positions={})

        self.assertTrue(_validate_bishop(state.cards[0], scenario, state))

    def test_baked_alchemist_cures_before_baker_conversion(self):
        deck = DeckComposition(
            villagers=["Alchemist", "Fortune_Teller", "Bishop", "Empress", "Knitter", "Baker"],
            outcasts=["Wretch"],
            minions=[],
            demons=["Pooka"],
        )
        state = GameState(
            n_cards=7,
            deck=deck,
            cards=[
                CardInfo(1, "Bishop", info_parsed={"targets": [3, 7], "types": ["Villager", "Minion"]}),
                CardInfo(2, "Baker", info_parsed={"original_role": "original"}),
                CardInfo(3, "Wretch"),
                CardInfo(4, "Knitter", info_parsed={"evil_pairs": 0}),
                CardInfo(5, "Empress", info_parsed={"targets": [2, 4, 7]}),
                CardInfo(6, "Baker", info_parsed={"original_role": "Alchemist"}),
                CardInfo(7, "Fortune Teller"),
            ],
            n_evil=1,
            board_villager_count=5,
            board_outcast_count=1,
        )
        raw_corrupted = {5}

        final_corrupted, alch_cures = _apply_post_corruption(
            raw_corrupted,
            {4: "Pooka"},
            state,
        )

        self.assertEqual(final_corrupted, set())
        self.assertEqual(alch_cures.get(6), 1)

    def test_live_baker_wretch_board_recovers_after_two_wrong_execs(self):
        deck = DeckComposition(
            villagers=["Alchemist", "Fortune_Teller", "Bishop", "Empress", "Knitter", "Baker"],
            outcasts=["Wretch"],
            minions=[],
            demons=["Pooka"],
        )
        state = GameState(
            n_cards=7,
            deck=deck,
            cards=[
                CardInfo(1, "Bishop", info_parsed={"targets": [3, 7], "types": ["Villager", "Minion"]}),
                CardInfo(2, "Baker", info_parsed={"original_role": "original"}),
                CardInfo(3, "Wretch"),
                CardInfo(4, "Knitter", info_parsed={"evil_pairs": 0}),
                CardInfo(5, "Empress", info_parsed={"targets": [2, 4, 7]}),
                CardInfo(6, "Baker", info_parsed={"original_role": "Alchemist"}),
                CardInfo(7, "Fortune Teller"),
            ],
            n_evil=1,
            executed=[6, 7],
            confirmed_good=[6, 7],
            hp=0,
            wrong_exec_cost=5,
            board_villager_count=5,
            board_outcast_count=1,
        )

        result = solve(state)

        self.assertGreater(result.n_surviving, 0, f"Reasoning: {result.reasoning}")
        self.assertIn(4, result.definite_evil, f"Reasoning: {result.reasoning}")
        self.assertIn(2, result.definite_good, f"Reasoning: {result.reasoning}")


if __name__ == "__main__":
    unittest.main()
