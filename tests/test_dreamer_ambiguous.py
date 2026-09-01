"""Unit tests for the public Dreamer "Among ... there is: A or B" parser.

Covers:
  - _parse_ambiguous_among helper (pure regex)
  - card_dreamer_ambiguous constructor (CardInfo shape)
  - _parse_clue_from_memory integration (new branch + fallback to standard form)

The strings below preserve historical fixtures; the exact shipped public
native shape is pinned independently by the reverse-engineering audit.
"""

import unittest
from unittest.mock import patch

from game_loop import (
    GameSession,
    _parse_ambiguous_among,
    _parse_clue_from_memory,
    card_druid,
    card_dreamer_ambiguous,
)
from solver import CardInfo, DeckComposition, GameState, Scenario, SolverResult
from strategy import recommend_abilities
from strategy import Action


class TestParseAmbiguousAmong(unittest.TestCase):
    def test_asc74_v7_pattern(self):
        clue = "Among\n#4, #9\nthere is:\nPooka or Rambler"
        self.assertEqual(
            _parse_ambiguous_among(clue),
            ([4, 9], ["Pooka", "Rambler"]),
        )

    def test_asc74_v2_pattern(self):
        clue = "Among\n#1, #8\nthere is:\nLilis or Knitter"
        self.assertEqual(
            _parse_ambiguous_among(clue),
            ([1, 8], ["Lilis", "Knitter"]),
        )

    def test_spaces_instead_of_newlines(self):
        clue = "Among #4, #9 there is: Pooka or Rambler"
        self.assertEqual(
            _parse_ambiguous_among(clue),
            ([4, 9], ["Pooka", "Rambler"]),
        )

    def test_trailing_period(self):
        clue = "Among #4, #9 there is: Pooka or Rambler."
        self.assertEqual(
            _parse_ambiguous_among(clue),
            ([4, 9], ["Pooka", "Rambler"]),
        )

    def test_no_colon(self):
        clue = "Among #4, #9 there is Pooka or Rambler"
        self.assertEqual(
            _parse_ambiguous_among(clue),
            ([4, 9], ["Pooka", "Rambler"]),
        )

    def test_three_targets(self):
        clue = "Among #1, #4, #9 there is: Pooka or Rambler"
        self.assertIsNone(_parse_ambiguous_among(clue))

    def test_single_target(self):
        clue = "Among #4 there is: Pooka or Rambler"
        self.assertIsNone(_parse_ambiguous_among(clue))

    def test_role_names_with_spaces(self):
        clue = "Among #4, #9 there is: Plague Doctor or Twin Minion"
        self.assertEqual(
            _parse_ambiguous_among(clue),
            ([4, 9], ["Plague Doctor", "Twin Minion"]),
        )

    def test_standard_dreamer_not_matched(self):
        self.assertIsNone(_parse_ambiguous_among("#5 could be: Pooka"))
        self.assertIsNone(_parse_ambiguous_among("#5 is Pooka"))

    def test_empty_or_none(self):
        self.assertIsNone(_parse_ambiguous_among(""))
        self.assertIsNone(_parse_ambiguous_among(None))

    def test_unrelated_clue_not_matched(self):
        # Oracle: "1 of #4, #9 is a Pooka" — no "or" + single role => no match.
        self.assertIsNone(
            _parse_ambiguous_among("1 of #4, #9 is a Pooka")
        )
        # Bishop-style types output — no "or" between roles.
        self.assertIsNone(
            _parse_ambiguous_among("#1 is a Villager, #2 is an Outcast")
        )


class TestCardDreamerAmbiguous(unittest.TestCase):
    def test_constructor_shape(self):
        ci = card_dreamer_ambiguous(7, [4, 9], ["Pooka", "Rambler"])
        self.assertEqual(ci.position, 7)
        self.assertEqual(ci.apparent_role, "Dreamer")
        self.assertEqual(
            ci.info_parsed,
            {
                "targets": [4, 9],
                "evil_role_options": ["Pooka", "Rambler"],
                "dreamer_variant": "public_current",
            },
        )

    def test_copies_lists_defensively(self):
        # Mutating caller lists after construction should not leak into CardInfo.
        targets = [4, 9]
        options = ["Pooka", "Rambler"]
        ci = card_dreamer_ambiguous(7, targets, options)
        targets.append(99)
        options.append("XXX")
        self.assertEqual(ci.info_parsed["targets"], [4, 9])
        self.assertEqual(ci.info_parsed["evil_role_options"], ["Pooka", "Rambler"])


class TestParseClueFromMemoryDreamer(unittest.TestCase):
    def _base(self, clue, targets=None):
        return {
            "position": 7,
            "true_role": "Dreamer",
            "disguise": "Dreamer",
            "clue_text": clue,
            "acted_infos": [{"desc": clue, "targets": targets or []}],
            "runtime_data": None,
            "pickable_uses_remaining": 0,
            "act_output_enabled": True,
            "pickable_available": False,
            "ability_used": True,
            "uses": 0,
        }

    def test_ambiguous_asc74_v7(self):
        card = self._base(
            "Among\n#4, #9\nthere is:\nPooka or Rambler",
            targets=[4, 9],
        )
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertEqual(ci.apparent_role, "Dreamer")
        self.assertEqual(
            ci.info_parsed,
            {
                "targets": [4, 9],
                "evil_role_options": ["Pooka", "Rambler"],
                "dreamer_variant": "public_current",
            },
        )

    def test_ambiguous_asc74_v2(self):
        card = self._base(
            "Among\n#1, #8\nthere is:\nLilis or Knitter",
            targets=[1, 8],
        )
        card["position"] = 9
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertEqual(
            ci.info_parsed,
            {
                "targets": [1, 8],
                "evil_role_options": ["Lilis", "Knitter"],
                "dreamer_variant": "public_current",
            },
        )

    def test_standard_dreamer_fallthrough(self):
        card = self._base("#5 could be: Pooka", targets=[5])
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertEqual(
            ci.info_parsed,
            {"target": 5, "evil_role": "Pooka"},
        )

    def test_drunk_disguised_as_dreamer(self):
        # Drunk may act as Dreamer; parser must key off apparent role, not true.
        card = {
            "position": 7,
            "true_role": "Drunk",
            "disguise": "Dreamer",
            "clue_text": "Among\n#4, #9\nthere is:\nPooka or Rambler",
            "acted_infos": [{
                "desc": "Among\n#4, #9\nthere is:\nPooka or Rambler",
                "targets": [4, 9],
            }],
            "runtime_data": None,
            "pickable_uses_remaining": 0,
            "act_output_enabled": True,
            "pickable_available": False,
            "ability_used": True,
            "uses": 0,
        }
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertEqual(ci.apparent_role, "Dreamer")
        self.assertEqual(ci.info_parsed["evil_role_options"], ["Pooka", "Rambler"])


class TestPublicDreamerStrategy(unittest.TestCase):
    def test_dreamer_recommendation_uses_two_targets(self):
        state = GameState(
            n_cards=3,
            n_evil=1,
            deck=DeckComposition(
                villagers=["Dreamer", "Bard"],
                outcasts=[],
                minions=["Minion"],
                demons=[],
            ),
            cards=[
                CardInfo(1, "Dreamer"),
                CardInfo(2, "Bard"),
                CardInfo(3, "Bard"),
            ],
        )
        result = SolverResult(
            definite_evil=[],
            definite_good=[],
            bombardier_positions=[],
            n_scenarios=2,
            n_surviving=2,
            surviving_scenarios=[
                Scenario(evil_positions={2: "Minion"}),
                Scenario(evil_positions={3: "Minion"}),
            ],
            reasoning=[],
        )

        dreamer = next(
            rec for rec in recommend_abilities(state, result, used_abilities=[])
            if rec.ability_name == "Dreamer"
        )

        self.assertCountEqual(dreamer.targets, [2, 3])
        self.assertEqual(len(dreamer.targets), 2)


class TestManualActiveAbilityBookkeeping(unittest.TestCase):
    def test_manual_dreamer_entry_marks_ability_used(self):
        session = GameSession(3, 1)
        session.add_card(card_dreamer_ambiguous(1, [2, 3], ["Minion", "Bard"]))
        self.assertIn(1, session.used_abilities)

    def test_manual_druid_none_entry_marks_ability_used(self):
        session = GameSession(4, 1)
        session.add_card(card_druid(1, [2, 3, 4], None))
        self.assertIn(1, session.used_abilities)


class TestDreamerAutoAbilityGuards(unittest.TestCase):
    def test_unknown_active_role_has_no_generic_automation_path(self):
        session = GameSession(3, 1)
        action = Action(
            "use_ability",
            position=1,
            targets=[2],
            ability_name="Unmapped Active",
        )

        result = session.auto_use_ability(action)

        self.assertFalse(result["success"])
        self.assertIn("no authenticated autonomous result path", result["error"])

    def test_dreamer_requires_two_targets(self):
        session = GameSession(3, 1)
        action = Action(
            "use_ability",
            position=1,
            targets=[2],
            ability_name="Dreamer",
        )

        result = session.auto_use_ability(action)

        self.assertFalse(result["success"])
        self.assertIn("exactly 2 distinct integer targets", result["error"])

    def test_dreamer_picker_accepts_unused_active_target(self):
        session = GameSession(3, 1)
        session.cards.extend([
            CardInfo(1, "Dreamer"),
            CardInfo(2, "Jester"),
        ])
        action = Action(
            "use_ability",
            position=1,
            targets=[2, 3],
            ability_name="Dreamer",
        )

        with patch("memory_reader.MemoryReader") as reader_type:
            reader_type.return_value.open.return_value = False
            result = session.auto_use_ability(action)

        self.assertFalse(result["success"])
        self.assertIn("Cannot open memory reader", result["error"])
        self.assertNotIn("unused active ability", result["error"])

    def test_dreamer_parser_rejects_text_reference_mismatch(self):
        clue = "Among\n#2, #3\nthere is:\nPooka or Rambler"
        card = {
            "position": 1,
            "true_role": "Dreamer",
            "clue_text": clue,
            "acted_infos": [{"desc": clue, "targets": [3, 4]}],
        }

        self.assertIsNone(_parse_clue_from_memory(card, n_cards=4))

    def test_auto_use_rejects_mismatched_memory_actor_before_click(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(1, "Dreamer"))
        before = {
            "position": 1,
            "true_role": "Jester",
            "clue_text": None,
            "acted_infos": [],
            "pickable_uses_remaining": 1,
        }

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [before]

            def close(self):
                return None

        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("template_match.safe_click_at") as click,
        ):
            result = session.auto_use_ability(
                Action("use_ability", 1, [2, 3], "Dreamer")
            )

        self.assertFalse(result["success"])
        self.assertIn("pre-click memory shows jester", result["error"])
        click.assert_not_called()

    def test_auto_use_rejects_new_refs_that_do_not_match_click_order(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(1, "Dreamer"))
        clue = "Among\n#2, #3\nthere is:\nPooka or Rambler"
        before = {
            "position": 1,
            "true_role": "Dreamer",
            "clue_text": None,
            "acted_infos": [],
            "pickable_uses_remaining": 1,
        }
        after = {
            **before,
            "clue_text": clue,
            "acted_infos": [{"desc": clue, "targets": [3, 2]}],
            "pickable_uses_remaining": 0,
        }

        class Reader:
            def __init__(self):
                self.reads = 0

            def open(self):
                return True

            def read_board(self):
                self.reads += 1
                return [before if self.reads == 1 else after]

            def close(self):
                return None

        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("template_match.safe_click_at"),
            patch("game_loop.time.sleep"),
        ):
            result = session.auto_use_ability(
                Action("use_ability", 1, [2, 3], "Dreamer")
            )

        self.assertFalse(result["success"])
        self.assertIn("do not match the clicked targets", result["error"])
        self.assertNotIn(1, session.used_abilities)


if __name__ == "__main__":
    unittest.main()
