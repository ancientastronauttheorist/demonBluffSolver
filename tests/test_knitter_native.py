"""Current-build native Knitter bridge regressions."""

import unittest

from game_loop import (
    GameSession,
    _knitter_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    card_knitter,
    card_poet_with_info,
)


def _memory_card(
    role: str,
    clue: str,
    targets: list[int] | None,
    *,
    position: int = 1,
    prior_infos: list[dict] | None = None,
) -> dict:
    infos = list(prior_infos or [])
    if targets is not None:
        infos.append({"desc": clue, "targets": list(targets)})
    return {
        "position": position,
        "true_role": role,
        "clue_text": clue,
        "acted_infos": infos,
    }


class KnitterConstructorTests(unittest.TestCase):
    def test_exact_native_text_has_three_canonical_forms(self):
        self.assertEqual(
            _knitter_native_text(0),
            "Evils are not adjacent to eachother",
        )
        self.assertEqual(
            _knitter_native_text(1),
            "There is only 1 pair of Evil",
        )
        self.assertEqual(
            _knitter_native_text(2),
            "There are 2 pairs of Evil",
        )
        self.assertEqual(
            _knitter_native_text(10),
            "There are 10 pairs of Evil",
        )

    def test_native_text_rejects_non_integer_and_negative_counts(self):
        for count in (-1, True, 1.5, "1"):
            with self.subTest(count=count), self.assertRaises(ValueError):
                _knitter_native_text(count)

    def test_unmarked_constructor_preserves_legacy_defaults(self):
        card = card_knitter(1, -1)

        self.assertEqual(card.info_text, "")
        self.assertEqual(card.info_parsed, {"evil_pairs": -1})

    def test_marked_constructor_synthesizes_current_text(self):
        card = card_knitter(
            1,
            2,
            knitter_variant="public_current",
        )

        self.assertEqual(card.info_text, "There are 2 pairs of Evil")
        self.assertEqual(
            card.info_parsed,
            {"evil_pairs": 2, "knitter_variant": "public_current"},
        )


class KnitterManualIngestionTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession(3, 1)

    def test_direct_cli_builds_canonical_current_observations(self):
        expected = (
            (0, "Evils are not adjacent to eachother"),
            (1, "There is only 1 pair of Evil"),
            (2, "There are 2 pairs of Evil"),
        )
        for count, text in expected:
            with self.subTest(count=count):
                card = _parse_card_cli(
                    ["knitter", "1", str(count)],
                    self.session,
                )
                self.assertEqual(card.info_text, text)
                self.assertEqual(
                    card.info_parsed,
                    {
                        "evil_pairs": count,
                        "knitter_variant": "public_current",
                    },
                )

    def test_direct_cli_rejects_missing_context_schema_bounds_and_type(self):
        invalid = (
            (["knitter"], self.session),
            (["knitter", "1"], self.session),
            (["knitter", "1", "1", "extra"], self.session),
            (["knitter", "0", "1"], self.session),
            (["knitter", "4", "1"], self.session),
            (["knitter", "1", "-1"], self.session),
            (["knitter", "1", "4"], self.session),
            (["knitter", "1", "one"], self.session),
            (["knitter", "1", "1"], None),
        )
        for args, session in invalid:
            with self.subTest(args=args, session=session), self.assertRaises(
                ValueError
            ):
                _parse_card_cli(args, session)

    def test_manual_poet_requires_board_and_synthesizes_exact_text(self):
        for count, text in (
            (0, "Evils are not adjacent to eachother"),
            (1, "There is only 1 pair of Evil"),
            (2, "There are 2 pairs of Evil"),
        ):
            with self.subTest(count=count):
                card = card_poet_with_info(
                    1,
                    "knitter",
                    [str(count)],
                    n_cards=3,
                )
                self.assertEqual(card.info_text, text)
                self.assertEqual(
                    card.info_parsed,
                    {
                        "evil_pairs": count,
                        "copied_role": "Knitter",
                        "poet_variant": "public_current",
                    },
                )

        with self.assertRaisesRegex(ValueError, "requires session board size"):
            card_poet_with_info(1, "knitter", ["1"])
        with self.assertRaises(ValueError):
            card_poet_with_info(4, "knitter", ["1"], n_cards=3)
        with self.assertRaises(ValueError):
            card_poet_with_info(1, "knitter", ["4"], n_cards=3)


class KnitterMemoryIngestionTests(unittest.TestCase):
    EXACT_CASES = (
        (0, "Evils are not adjacent to eachother"),
        (1, "There is only 1 pair of Evil"),
        (2, "There are 2 pairs of Evil"),
        (10, "There are 10 pairs of Evil"),
    )

    def test_direct_accepts_only_exact_current_sentences(self):
        for count, clue in self.EXACT_CASES:
            with self.subTest(count=count):
                parsed = _parse_clue_from_memory(
                    _memory_card("Knitter", clue, []),
                    n_cards=max(10, count),
                )

                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.info_text, clue)
                self.assertEqual(
                    parsed.info_parsed,
                    {
                        "evil_pairs": count,
                        "knitter_variant": "public_current",
                    },
                )

    def test_non_native_text_is_rejected(self):
        invalid = (
            "evils are not adjacent to eachother",
            "Evils are not adjacent to each other",
            "Evils  are not adjacent to eachother",
            "Evils are not adjacent to eachother!",
            "Evils are not adjacent to eachother ",
            " Evils are not adjacent to eachother",
            "Evils are not\nadjacent to eachother",
            "There Is only 1 pair of Evil",
            "There is only 1 pair of evil",
            "There is only 1 pair of Evil.",
            "There are 01 pairs of Evil",
            "There are 1 pairs of Evil",
            "There are 0 pairs of Evil",
            "There are 2 pair of Evil",
            "There are 2 pairs  of Evil",
            "There are 2 pairs of Evil trailing",
        )
        for clue in invalid:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Knitter", clue, []),
                        n_cards=3,
                    )
                )

    def test_direct_requires_newest_coherent_zero_ref_event(self):
        clue = "There is only 1 pair of Evil"
        malformed = (
            _memory_card("Knitter", clue, None),
            _memory_card("Knitter", clue, [2]),
            {
                **_memory_card("Knitter", clue, []),
                "acted_infos": [{"desc": clue, "targets": ["2"]}],
            },
            {
                **_memory_card("Knitter", clue, []),
                "acted_infos": [
                    {"desc": clue, "targets": []},
                    {"desc": "stale result", "targets": []},
                ],
            },
        )
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=3)
                )

    def test_direct_requires_current_board_actor_and_count(self):
        clue = "There is only 1 pair of Evil"
        invalid = (
            (_memory_card("Knitter", clue, [], position=0), 3),
            (_memory_card("Knitter", clue, [], position=4), 3),
            (_memory_card("Knitter", clue, []), None),
            (
                _memory_card(
                    "Knitter",
                    "There are 4 pairs of Evil",
                    [],
                ),
                3,
            ),
        )
        for card, n_cards in invalid:
            with self.subTest(card=card, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=n_cards)
                )

    def test_poet_uses_same_exact_text_and_zero_ref_contract(self):
        for count, clue in self.EXACT_CASES[:3]:
            with self.subTest(count=count):
                parsed = _parse_clue_from_memory(
                    _memory_card("Poet", clue, []),
                    n_cards=3,
                )
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.info_text, clue)
                self.assertEqual(
                    parsed.info_parsed,
                    {
                        "evil_pairs": count,
                        "copied_role": "Knitter",
                        "poet_variant": "public_current",
                    },
                )

        clue = "There is only 1 pair of Evil"
        for card, n_cards in (
            (_memory_card("Poet", clue, [2]), 3),
            (_memory_card("Poet", clue, [], position=0), 3),
            (_memory_card("Poet", clue, [], position=4), 3),
            (_memory_card("Poet", clue, []), None),
        ):
            with self.subTest(card=card, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=n_cards)
                )


if __name__ == "__main__":
    unittest.main()
