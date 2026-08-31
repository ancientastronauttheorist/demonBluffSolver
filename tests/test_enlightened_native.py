"""Current-build native Enlightened/Shugenja bridge regressions."""

import unittest

from game_loop import (
    GameSession,
    _enlightened_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    _parse_enlightened_native_text,
    card_enlightened,
    card_poet_with_info,
)
from solver import POET_VARIANT


_ABSENT = object()


def _memory_card(
    role: str,
    clue,
    targets=_ABSENT,
    *,
    position=1,
    runtime_data=_ABSENT,
    prior_infos: list | None = None,
) -> dict:
    infos = list(prior_infos or [])
    if targets is not _ABSENT:
        infos.append({"desc": clue, "targets": targets})
    card = {
        "position": position,
        "true_role": role,
        "clue_text": clue,
        "acted_infos": infos,
    }
    if runtime_data is not _ABSENT:
        card["runtime_data"] = runtime_data
    return card


class EnlightenedConstructorTests(unittest.TestCase):
    EXACT_CASES = (
        ("CW", "Closest Evil is:\nClockwise"),
        ("CCW", "Closest Evil is:\nCounter-clockwise"),
        ("Equidistant", "Closest Evil is equidistant"),
    )

    def test_exact_native_text_has_three_canonical_forms(self):
        for direction, text in self.EXACT_CASES:
            with self.subTest(direction=direction):
                self.assertEqual(_enlightened_native_text(direction), text)
                self.assertEqual(_parse_enlightened_native_text(text), direction)

    def test_native_text_helpers_reject_unknown_or_loose_values(self):
        for direction in ("Clockwise", "counter-clockwise", "Left", "", None):
            with self.subTest(direction=direction), self.assertRaises(ValueError):
                _enlightened_native_text(direction)

        for clue in (
            "closest Evil is:\nClockwise",
            "Closest Evil is: Clockwise",
            "Closest Evil is:\r\nClockwise",
            "Closest Evil is:\nclockwise",
            "Closest Evil is:\nClockwise.",
            "Closest Evil is:\nCounterclockwise",
            "Closest Evil is:\nCounter-clockwise ",
            "Closest Evil is Equidistant",
            "Closest Evil is equidistant!",
            " Closest Evil is equidistant",
            "",
            None,
        ):
            with self.subTest(clue=clue):
                self.assertIsNone(_parse_enlightened_native_text(clue))

    def test_unmarked_constructor_preserves_legacy_shape(self):
        card = card_enlightened(1, "Left")

        self.assertEqual(card.info_text, "")
        self.assertEqual(card.info_parsed, {"direction": "Left"})

    def test_marked_constructor_synthesizes_native_text_and_provenance(self):
        for direction, text in self.EXACT_CASES:
            with self.subTest(direction=direction):
                card = card_enlightened(
                    1,
                    direction,
                    enlightened_variant="public_current",
                )
                self.assertEqual(card.info_text, text)
                self.assertEqual(
                    card.info_parsed,
                    {
                        "direction": direction,
                        "enlightened_variant": "public_current",
                    },
                )

        with self.assertRaisesRegex(ValueError, "must match its direction"):
            card_enlightened(
                1,
                "CW",
                info_text="Closest Evil is equidistant",
                enlightened_variant="public_current",
            )


class EnlightenedManualIngestionTests(unittest.TestCase):
    EXACT_CASES = EnlightenedConstructorTests.EXACT_CASES

    def setUp(self):
        self.session = GameSession(3, 1)

    def test_direct_cli_builds_canonical_current_observations(self):
        for direction, text in self.EXACT_CASES:
            with self.subTest(direction=direction):
                card = _parse_card_cli(
                    ["enlightened", "2", direction],
                    self.session,
                )
                self.assertEqual(card.apparent_role, "Enlightened")
                self.assertEqual(card.info_text, text)
                self.assertEqual(
                    card.info_parsed,
                    {
                        "direction": direction,
                        "enlightened_variant": "public_current",
                    },
                )

    def test_direct_cli_canonicalizes_supported_human_spellings(self):
        aliases = (
            ("clockwise", "CW"),
            ("counter-clockwise", "CCW"),
            ("counterclockwise", "CCW"),
            ("equidistant", "Equidistant"),
        )
        for claim, expected in aliases:
            with self.subTest(claim=claim):
                card = _parse_card_cli(
                    ["enlightened", "1", claim],
                    self.session,
                )
                self.assertEqual(card.info_parsed["direction"], expected)
                self.assertEqual(
                    card.info_text,
                    _enlightened_native_text(expected),
                )

    def test_direct_cli_requires_context_exact_schema_and_valid_actor(self):
        invalid = (
            (["enlightened"], self.session),
            (["enlightened", "1"], self.session),
            (["enlightened", "1", "CW", "extra"], self.session),
            (["enlightened", "0", "CW"], self.session),
            (["enlightened", "4", "CW"], self.session),
            (["enlightened", "1", "Left"], self.session),
            (["enlightened", "1", "CW"], None),
        )
        for args, session in invalid:
            with self.subTest(args=args, session=session), self.assertRaises(
                ValueError
            ):
                _parse_card_cli(args, session)

    def test_poet_builds_exact_current_observations(self):
        for direction, text in self.EXACT_CASES:
            with self.subTest(direction=direction):
                card = card_poet_with_info(
                    2,
                    "enlightened",
                    [direction],
                    n_cards=3,
                )
                self.assertEqual(card.apparent_role, "Poet")
                self.assertEqual(card.info_text, text)
                self.assertEqual(
                    card.info_parsed,
                    {
                        "direction": direction,
                        "copied_role": "Enlightened",
                        "poet_variant": POET_VARIANT,
                    },
                )

    def test_poet_requires_context_exact_schema_and_valid_actor(self):
        invalid = (
            (1, [], 3),
            (1, ["CW", "extra"], 3),
            (0, ["CW"], 3),
            (4, ["CW"], 3),
            (1, ["Left"], 3),
            (1, ["CW"], None),
        )
        for position, args, n_cards in invalid:
            with self.subTest(position=position, args=args, n_cards=n_cards), self.assertRaises(
                ValueError
            ):
                card_poet_with_info(
                    position,
                    "enlightened",
                    args,
                    n_cards=n_cards,
                )


class EnlightenedMemoryIngestionTests(unittest.TestCase):
    EXACT_CASES = EnlightenedConstructorTests.EXACT_CASES

    def test_direct_accepts_public_and_managed_names_with_exact_text(self):
        for role in ("Enlightened", "Shugenja"):
            for direction, clue in self.EXACT_CASES:
                with self.subTest(role=role, direction=direction):
                    parsed = _parse_clue_from_memory(
                        _memory_card(role, clue, []),
                        n_cards=6,
                    )
                    self.assertIsNotNone(parsed)
                    self.assertEqual(parsed.apparent_role, "Enlightened")
                    self.assertEqual(parsed.info_text, clue)
                    self.assertEqual(
                        parsed.info_parsed,
                        {
                            "direction": direction,
                            "enlightened_variant": "public_current",
                        },
                    )

    def test_direct_runtime_data_must_match_when_available(self):
        for direction, clue in self.EXACT_CASES:
            with self.subTest(direction=direction, mode="matching"):
                parsed = _parse_clue_from_memory(
                    _memory_card(
                        "Shugenja",
                        clue,
                        [],
                        runtime_data={"type": "direction", "direction": direction},
                    ),
                    n_cards=6,
                )
                self.assertEqual(parsed.info_parsed["direction"], direction)

            for runtime_data in (
                {"type": "direction", "direction": "Left"},
                {"type": "direction", "direction": "CCW" if direction != "CCW" else "CW"},
                {"type": "baker", "direction": direction},
                {},
                "direction",
            ):
                with self.subTest(direction=direction, runtime_data=runtime_data):
                    self.assertIsNone(
                        _parse_clue_from_memory(
                            _memory_card(
                                "Shugenja",
                                clue,
                                [],
                                runtime_data=runtime_data,
                            ),
                            n_cards=6,
                        )
                    )

    def test_direct_requires_newest_coherent_zero_ref_event(self):
        clue = _enlightened_native_text("CW")
        malformed = (
            _memory_card("Enlightened", clue),
            _memory_card("Enlightened", clue, None),
            _memory_card("Enlightened", clue, [2]),
            _memory_card("Enlightened", clue, [True]),
            _memory_card("Enlightened", clue, ["2"]),
            {
                **_memory_card("Enlightened", clue, []),
                "acted_infos": [{"desc": clue}],
            },
            {
                **_memory_card("Enlightened", clue, []),
                "acted_infos": [{"desc": clue, "targets": ()}],
            },
            {
                **_memory_card("Enlightened", clue, []),
                "acted_infos": [
                    {"desc": clue, "targets": []},
                    {"desc": "stale result", "targets": []},
                ],
            },
            {
                **_memory_card("Enlightened", clue, []),
                "acted_infos": [{"desc": clue, "targets": []}, None],
            },
        )
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

        newest = _parse_clue_from_memory(
            _memory_card(
                "Enlightened",
                clue,
                [],
                prior_infos=[{"desc": "older result", "targets": [2]}],
            ),
            n_cards=6,
        )
        self.assertEqual(newest.info_parsed["direction"], "CW")

    def test_direct_requires_current_board_actor(self):
        clue = _enlightened_native_text("Equidistant")
        for position, n_cards in (
            (0, 6),
            (7, 6),
            (True, 6),
            ("1", 6),
            (1, None),
            (1, 0),
            (1, True),
        ):
            with self.subTest(position=position, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Enlightened", clue, [], position=position),
                        n_cards=n_cards,
                    )
                )

    def test_direct_rejects_every_loose_native_lookalike(self):
        lookalikes = (
            "closest Evil is:\nClockwise",
            "Closest  Evil is:\nClockwise",
            "Closest Evil is: Clockwise",
            "Closest Evil is:\r\nClockwise",
            "Closest Evil is:\nClockwise!",
            "Closest Evil is:\nCounterclockwise",
            "Closest Evil is:\nCounter-Clockwise",
            "Closest Evil is:\nCCW",
            "Closest Evil is Equidistant",
            "Closest Evil is:\nequidistant",
            "Closest Evil is equidistant.",
            "Closest Evil is equidistant trailing",
        )
        for clue in lookalikes:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Enlightened", clue, []),
                        n_cards=6,
                    )
                )

    def test_poet_uses_same_exact_text_runtime_and_zero_ref_contract(self):
        for direction, clue in self.EXACT_CASES:
            for runtime_data in (
                _ABSENT,
                {"type": "direction", "direction": direction},
            ):
                with self.subTest(direction=direction, runtime_data=runtime_data):
                    parsed = _parse_clue_from_memory(
                        _memory_card(
                            "Poet",
                            clue,
                            [],
                            runtime_data=runtime_data,
                        ),
                        n_cards=6,
                    )
                    self.assertIsNotNone(parsed)
                    self.assertEqual(parsed.info_text, clue)
                    self.assertEqual(
                        parsed.info_parsed,
                        {
                            "direction": direction,
                            "copied_role": "Enlightened",
                            "poet_variant": POET_VARIANT,
                        },
                    )

            for runtime_data in (
                {"type": "direction", "direction": "Left"},
                {"type": "direction", "direction": "CCW" if direction != "CCW" else "CW"},
                {"type": "baker", "direction": direction},
                {},
            ):
                with self.subTest(direction=direction, runtime_data=runtime_data):
                    self.assertIsNone(
                        _parse_clue_from_memory(
                            _memory_card(
                                "Poet",
                                clue,
                                [],
                                runtime_data=runtime_data,
                            ),
                            n_cards=6,
                        )
                    )

    def test_poet_rejects_stale_malformed_and_out_of_board_events(self):
        clue = _enlightened_native_text("CCW")
        malformed = (
            (_memory_card("Poet", clue), 6),
            (_memory_card("Poet", clue, [2]), 6),
            (_memory_card("Poet", clue, [True]), 6),
            (
                _memory_card(
                    "Poet",
                    clue,
                    [],
                    prior_infos=[{"desc": clue, "targets": []}],
                )
                | {
                    "acted_infos": [
                        {"desc": clue, "targets": []},
                        {"desc": "newer result", "targets": []},
                    ]
                },
                6,
            ),
            (_memory_card("Poet", clue, [], position=0), 6),
            (_memory_card("Poet", clue, [], position=7), 6),
            (_memory_card("Poet", clue, []), None),
        )
        for card, n_cards in malformed:
            with self.subTest(card=card, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=n_cards)
                )

    def test_readable_direction_runtime_rejects_nonmatching_poet_event(self):
        card = _memory_card(
            "Poet",
            "There are no minions",
            [],
            runtime_data={"type": "direction", "direction": "CW"},
        )

        self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))


if __name__ == "__main__":
    unittest.main()
