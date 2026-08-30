"""Focused tests for the shipped public Dreamer clue formats."""

import unittest

from game_loop import (
    _parse_ambiguous_among,
    _parse_cabbage_between,
    _parse_card_cli,
    _parse_clue_from_memory,
    card_dreamer_ambiguous,
    card_dreamer_cabbage,
)


def _memory_card(clue: str, acted_targets=None) -> dict:
    return {
        "position": 5,
        "true_role": "Dreamer",
        "disguise": "Dreamer",
        "clue_text": clue,
        "acted_infos": [{"desc": clue, "targets": acted_targets or []}],
        "runtime_data": None,
        "ability_used": True,
        "uses": 0,
    }


class TestDreamerConstructors(unittest.TestCase):
    def test_ambiguous_requires_native_shape_and_preserves_text(self):
        clue = "Among\n#3, #9\nthere is:\nPuppeteer or Lover"
        card = card_dreamer_ambiguous(
            5,
            [3, 9],
            [" Puppeteer ", "Lover"],
            info_text=clue,
        )

        self.assertEqual(card.info_text, clue)
        self.assertEqual(
            card.info_parsed,
            {
                "targets": [3, 9],
                "evil_role_options": ["Puppeteer", "Lover"],
            },
        )

    def test_ambiguous_rejects_invalid_targets(self):
        invalid_targets = ([3], [3, 9, 10], [3, "9"], [3, True], None)
        for targets in invalid_targets:
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    card_dreamer_ambiguous(
                        5,
                        targets,
                        ["Puppeteer", "Lover"],
                    )

    def test_ambiguous_rejects_invalid_role_options(self):
        invalid_options = (
            ["Puppeteer"],
            ["Puppeteer", "Lover", "Pooka"],
            ["Puppeteer", ""],
            ["Puppeteer", "   "],
            ["Puppeteer", " puppeteer "],
            ["Twin Minion", "Twin_Minion"],
            ["Puppeteer", 7],
            None,
        )
        for options in invalid_options:
            with self.subTest(options=options):
                with self.assertRaises(ValueError):
                    card_dreamer_ambiguous(5, [3, 9], options)

    def test_cabbage_requires_two_integer_targets(self):
        clue = "Between #3, #9 there is: a Cabbage"
        card = card_dreamer_cabbage(5, [3, 9], info_text=clue)
        self.assertEqual(card.info_text, clue)
        self.assertEqual(
            card.info_parsed,
            {"targets": [3, 9], "cabbage": True},
        )

        for targets in ([3], [3, 9, 10], [3, "9"], None):
            with self.subTest(targets=targets):
                with self.assertRaises(ValueError):
                    card_dreamer_cabbage(5, targets)


class TestDreamerNativeParsers(unittest.TestCase):
    def test_ambiguous_native_multiline_and_space_flexible(self):
        self.assertEqual(
            _parse_ambiguous_among(
                "Among\n#3, #9\nthere is:\nPuppeteer or Lover"
            ),
            ([3, 9], ["Puppeteer", "Lover"]),
        )
        self.assertEqual(
            _parse_ambiguous_among(
                "  Among  #3 ,  #9  there is:  Plague Doctor or Twin Minion. "
            ),
            ([3, 9], ["Plague Doctor", "Twin Minion"]),
        )

    def test_ambiguous_rejects_non_native_cardinality_and_options(self):
        self.assertIsNone(
            _parse_ambiguous_among("Among #3 there is: Puppeteer or Lover")
        )
        self.assertIsNone(
            _parse_ambiguous_among(
                "Among #3, #9, #10 there is: Puppeteer or Lover"
            )
        )
        self.assertIsNone(
            _parse_ambiguous_among(
                "Among #3, #9 there is: Puppeteer or Puppeteer"
            )
        )
        self.assertIsNone(
            _parse_ambiguous_among(
                "Among #3, #9 there is: Puppeteer or Lover or Pooka"
            )
        )
        self.assertIsNone(
            _parse_ambiguous_among(
                "Among #3, #9 there is: Puppeteer"
            )
        )

    def test_cabbage_native_multiline_and_space_flexible(self):
        self.assertEqual(
            _parse_cabbage_between(
                "Between\n#3, #9\nthere is:\na Cabbage"
            ),
            [3, 9],
        )
        self.assertEqual(
            _parse_cabbage_between(
                " Between #3 , #9 there is: a Cabbage. "
            ),
            [3, 9],
        )

    def test_cabbage_rejects_other_shapes(self):
        self.assertIsNone(
            _parse_cabbage_between("Between #3 there is: a Cabbage")
        )
        self.assertIsNone(
            _parse_cabbage_between(
                "Between #3, #9, #10 there is: a Cabbage"
            )
        )
        self.assertIsNone(
            _parse_cabbage_between("Among #3, #9 there is: a Cabbage")
        )


class TestDreamerMemoryIngestion(unittest.TestCase):
    def test_ambiguous_uses_clue_ids_and_preserves_raw_text(self):
        clue = "Among\n#3, #9\nthere is:\nPuppeteer or Lover"
        parsed = _parse_clue_from_memory(
            _memory_card(clue, acted_targets=[1, 2])
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.info_text, clue)
        self.assertEqual(
            parsed.info_parsed,
            {
                "targets": [3, 9],
                "evil_role_options": ["Puppeteer", "Lover"],
            },
        )

    def test_cabbage_uses_clue_ids_and_preserves_raw_text(self):
        clue = "Between #3, #9 there is: a Cabbage"
        parsed = _parse_clue_from_memory(
            _memory_card(clue, acted_targets=[1, 2])
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.info_text, clue)
        self.assertEqual(
            parsed.info_parsed,
            {"targets": [3, 9], "cabbage": True},
        )

    def test_legacy_one_target_uses_its_clue_id(self):
        clue = "#7 could be:\nPooka"
        parsed = _parse_clue_from_memory(
            _memory_card(clue, acted_targets=[1])
        )

        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.info_text, clue)
        self.assertEqual(
            parsed.info_parsed,
            {"target": 7, "evil_role": "Pooka"},
        )

    def test_unbound_dreamer2_type_clue_is_not_legacy_misparsed(self):
        clue = "#3, 9:\nNone of them is\nVillager"
        parsed = _parse_clue_from_memory(
            _memory_card(clue, acted_targets=[3, 9])
        )

        self.assertIsNone(parsed)


class TestDreamerManualRouting(unittest.TestCase):
    def test_public_dreamer_is_two_target_shape(self):
        parsed = _parse_card_cli(
            ["dreamer", "5", "3,9", "Puppeteer,Lover"]
        )
        self.assertEqual(
            parsed.info_parsed,
            {
                "targets": [3, 9],
                "evil_role_options": ["Puppeteer", "Lover"],
            },
        )

        with self.assertRaises(ValueError):
            _parse_card_cli(
                ["dreamer", "5", "3,9", "Puppeteer,Lover,"]
            )

    def test_compatibility_and_explicit_old_aliases(self):
        for alias in ("dreamer2", "dreamer_ambiguous"):
            with self.subTest(alias=alias):
                ambiguous = _parse_card_cli(
                    [alias, "5", "3,9", "Puppeteer,Lover"]
                )
                self.assertEqual(ambiguous.info_parsed["targets"], [3, 9])

        for alias in ("dreamer_old", "dreamer1"):
            with self.subTest(alias=alias):
                old = _parse_card_cli([alias, "5", "3", "Pooka"])
                self.assertEqual(
                    old.info_parsed,
                    {"target": 3, "evil_role": "Pooka"},
                )

    def test_cabbage_manual_route(self):
        parsed = _parse_card_cli(["dreamer_cabbage", "5", "3,9"])
        self.assertEqual(
            parsed.info_parsed,
            {"targets": [3, 9], "cabbage": True},
        )


if __name__ == "__main__":
    unittest.main()
