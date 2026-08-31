"""Current-build native Bard/Acrobat2 bridge regressions."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from game_loop import (
    DecisionLog,
    GameSession,
    _bard_native_text,
    _current_bard_refs,
    _parse_bard_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    _valid_current_bard_distance,
    card_bard,
    card_poet_with_info,
    dispatch,
)
from memory_reader import clean_name
from solver import BAKER_RULE_VERSION, CardInfo, POET_VARIANT


_ABSENT = object()


def _memory_card(
    role: str,
    clue,
    refs=_ABSENT,
    *,
    position=1,
    current_role=_ABSENT,
    disguise=_ABSENT,
    runtime_data=_ABSENT,
    prior_infos: list | None = None,
) -> dict:
    infos = list(prior_infos or [])
    if refs is not _ABSENT:
        infos.append({"desc": clue, "targets": refs})
    card = {
        "position": position,
        "true_role": role,
        "clue_text": clue,
        "acted_infos": infos,
    }
    if current_role is not _ABSENT:
        card["current_role"] = current_role
    if disguise is not _ABSENT:
        card["disguise"] = disguise
    if runtime_data is not _ABSENT:
        card["runtime_data"] = runtime_data
    return card


class BardConstructorTests(unittest.TestCase):
    def test_exact_native_builder_parser_and_range_helpers(self):
        cases = (
            (-1, "There are no Corrupted characters"),
            (1, "I am 1 card away from Corrupted character"),
            (2, "I am 2 cards away from Corrupted character"),
            (12, "I am 12 cards away from Corrupted character"),
        )
        for distance, clue in cases:
            with self.subTest(distance=distance):
                self.assertEqual(_bard_native_text(distance), clue)
                self.assertEqual(_parse_bard_native_text(clue), distance)

        self.assertTrue(_valid_current_bard_distance(-1, 6))
        self.assertTrue(_valid_current_bard_distance(3, 2))
        self.assertTrue(_valid_current_bard_distance(5, 10))
        self.assertFalse(_valid_current_bard_distance(4, 6))
        self.assertEqual(_current_bard_refs(1, -1, 6), [])
        self.assertEqual(_current_bard_refs(1, 0, 6), [])
        self.assertEqual(_current_bard_refs(1, 1, 6), [2, 6])
        self.assertEqual(_current_bard_refs(1, 3, 6), [4, 4])
        self.assertEqual(_current_bard_refs(1, 3, 2), [])

    def test_exact_parser_rejects_loose_and_malformed_lookalikes(self):
        lookalikes = (
            "there are no Corrupted characters",
            "There Are no Corrupted characters",
            "There are  no Corrupted characters",
            "There are no corrupted characters",
            "There are no Corrupted character",
            "There are no Corrupted characters.",
            "There are no Corrupted characters!",
            " There are no Corrupted characters",
            "There are no Corrupted characters ",
            "There are\nno Corrupted characters",
            "I am 0 cards away from Corrupted character",
            "I am 01 card away from Corrupted character",
            "I am 1 cards away from Corrupted character",
            "I am 2 card away from Corrupted character",
            "I am 2 cards away from corrupted character",
            "I am 2 cards away from Corrupted characters",
            "I am 2 cards away from Corrupted character.",
            "I am 2 cards  away from Corrupted character",
            "I am 2 cards away from Corrupted character trailing",
            "I am\n2 cards away from Corrupted character",
            "I am " + ("9" * 5000) + " cards away from Corrupted character",
            "",
            None,
        )
        for clue in lookalikes:
            with self.subTest(clue=clue):
                self.assertIsNone(_parse_bard_native_text(clue))

        for invalid in (-2, 0, True, False, "1", None):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _bard_native_text(invalid)

    def test_unmarked_constructor_and_archived_exceptions_remain_legacy(self):
        legacy = card_bard(2, 2)
        self.assertEqual(legacy.info_text, "")
        self.assertEqual(legacy.info_parsed, {"corruption_distance": 2})

        legacy_zero_convention = card_bard(2, 0)
        self.assertEqual(
            legacy_zero_convention.info_parsed,
            {"corruption_distance": -1},
        )
        self.assertNotIn("bard_variant", legacy_zero_convention.info_parsed)

        case_root = Path(__file__).parent / "cases"
        empty_case = json.loads(
            (case_root / "asc10_g8.json").read_text(encoding="utf-8")
        )
        empty = next(
            card for card in empty_case["cards"]
            if card["apparent_role"] == "Bard"
        )
        zero_case = json.loads(
            (case_root / "asc9_g3.json").read_text(encoding="utf-8")
        )
        zero = next(
            card for card in zero_case["cards"]
            if card.get("info_parsed", {}).get("corruption_distance") == 0
        )
        self.assertEqual(empty["info_parsed"], {})
        self.assertEqual(zero["info_parsed"], {"corruption_distance": 0})
        self.assertNotIn("bard_variant", empty["info_parsed"])
        self.assertNotIn("bard_variant", zero["info_parsed"])

    def test_marked_constructor_normalizes_sentinel_and_stamps_exact_schema(self):
        sentinel = card_bard(2, 0, bard_variant="public_current")
        self.assertEqual(sentinel.info_text, _bard_native_text(-1))
        self.assertEqual(
            sentinel.info_parsed,
            {
                "corruption_distance": -1,
                "bard_variant": "public_current",
            },
        )

        numeric = card_bard(2, 4, bard_variant="public_current")
        self.assertEqual(numeric.info_text, _bard_native_text(4))
        self.assertEqual(
            numeric.info_parsed,
            {
                "corruption_distance": 4,
                "bard_variant": "public_current",
            },
        )

    def test_marked_constructor_rejects_noncurrent_schema(self):
        invalid = (
            (0, 1),
            (True, 1),
            ("1", 1),
            (1, -2),
            (1, True),
            (1, False),
            (1, "1"),
        )
        for position, distance in invalid:
            with self.subTest(position=position, distance=distance), self.assertRaises(
                ValueError
            ):
                card_bard(position, distance, bard_variant="public_current")

        with self.assertRaises(ValueError):
            card_bard(1, 1, bard_variant="future")
        with self.assertRaisesRegex(ValueError, "must match"):
            card_bard(
                1,
                1,
                info_text=_bard_native_text(2),
                bard_variant="public_current",
            )
        with self.assertRaises(ValueError):
            card_bard(1, 1, info_text=None, bard_variant="public_current")

    def test_reader_maps_all_bard_managed_names_but_not_acrobat(self):
        self.assertEqual(clean_name("Acrobat2"), "Bard")
        self.assertEqual(clean_name("Acrobat2_12345"), "Bard")
        self.assertEqual(clean_name("RangedEmpath"), "Bard")
        self.assertEqual(clean_name("Athlete"), "Bard")
        self.assertEqual(clean_name("Acrobat"), "Acrobat")


class BardManualIngestionTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession(8, 1)

    def test_direct_cli_synthesizes_exact_current_text_and_marker(self):
        numeric = _parse_card_cli(["bard", "2", "4"], self.session)
        self.assertEqual(numeric.apparent_role, "Bard")
        self.assertEqual(numeric.info_text, _bard_native_text(4))
        self.assertEqual(
            numeric.info_parsed,
            {
                "corruption_distance": 4,
                "bard_variant": "public_current",
            },
        )

        for token in ("0", "-1"):
            with self.subTest(token=token):
                sentinel = _parse_card_cli(["bard", "2", token], self.session)
                self.assertEqual(sentinel.info_text, _bard_native_text(-1))
                self.assertEqual(sentinel.info_parsed["corruption_distance"], -1)

        tiny = _parse_card_cli(["bard", "1", "3"], GameSession(2, 1))
        self.assertEqual(tiny.info_parsed["corruption_distance"], 3)

    def test_direct_cli_requires_session_exact_arity_actor_and_native_range(self):
        invalid = (
            (["bard", "1", "1"], None),
            (["bard", "1"], self.session),
            (["bard", "1", "1", "extra"], self.session),
            (["bard", "0", "1"], self.session),
            (["bard", "9", "1"], self.session),
            (["bard", "1", "-2"], self.session),
            (["bard", "1", "5"], self.session),
            (["bard", "1", "not-a-distance"], self.session),
        )
        for args, session in invalid:
            with self.subTest(args=args), self.assertRaises((IndexError, ValueError)):
                _parse_card_cli(args, session)

    def test_poet_manual_stamps_only_poet_and_synthesizes_exact_text(self):
        for token, expected in (("0", -1), ("-1", -1), ("4", 4)):
            with self.subTest(token=token):
                card = card_poet_with_info(
                    2,
                    "bard",
                    [token],
                    n_cards=8,
                )
                self.assertEqual(card.apparent_role, "Poet")
                self.assertEqual(card.info_text, _bard_native_text(expected))
                self.assertEqual(
                    card.info_parsed,
                    {
                        "corruption_distance": expected,
                        "copied_role": "Bard",
                        "poet_variant": POET_VARIANT,
                    },
                )
                self.assertNotIn("bard_variant", card.info_parsed)

        tiny = card_poet_with_info(1, "Bard", ["3"], n_cards=2)
        self.assertEqual(tiny.info_text, _bard_native_text(3))

    def test_poet_manual_requires_board_actor_arity_sentinel_and_range(self):
        invalid = (
            (1, ["1"], None),
            (0, ["1"], 8),
            (9, ["1"], 8),
            (True, ["1"], 8),
            (1, [], 8),
            (1, ["1", "extra"], 8),
            (1, ["-2"], 8),
            (1, ["5"], 8),
            (1, ["not-a-distance"], 8),
        )
        for position, args, n_cards in invalid:
            with self.subTest(position=position, args=args), self.assertRaises(
                ValueError
            ):
                card_poet_with_info(
                    position,
                    "Bard",
                    args,
                    n_cards=n_cards,
                )


class BardMemoryIngestionTests(unittest.TestCase):
    def _assert_current(self, parsed, distance: int, *, poet: bool) -> None:
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.info_text, _bard_native_text(distance))
        if poet:
            self.assertEqual(
                parsed.info_parsed,
                {
                    "corruption_distance": distance,
                    "copied_role": "Bard",
                    "poet_variant": POET_VARIANT,
                },
            )
            self.assertNotIn("bard_variant", parsed.info_parsed)
        else:
            self.assertEqual(parsed.apparent_role, "Bard")
            self.assertEqual(
                parsed.info_parsed,
                {
                    "corruption_distance": distance,
                    "bard_variant": "public_current",
                },
            )

    def test_direct_accepts_public_current_and_legacy_managed_aliases(self):
        clue = _bard_native_text(1)
        refs = _current_bard_refs(1, 1, 6)
        cards = (
            _memory_card("Bard", clue, refs),
            _memory_card("Acrobat2", clue, refs),
            _memory_card("RangedEmpath", clue, refs),
            _memory_card("Athlete", clue, refs),
            _memory_card("Pooka", clue, refs, current_role="Acrobat2"),
            _memory_card(
                "Pooka",
                clue,
                refs,
                current_role="Pooka",
                disguise="Acrobat2",
            ),
        )
        for card in cards:
            with self.subTest(card=card):
                self._assert_current(
                    _parse_clue_from_memory(card, n_cards=6),
                    1,
                    poet=False,
                )

    def test_disguise_current_true_precedence_and_acrobat_distinction(self):
        clue = _bard_native_text(1)
        refs = _current_bard_refs(1, 1, 6)
        rejected = (
            _memory_card(
                "Acrobat2",
                clue,
                refs,
                current_role="Acrobat2",
                disguise="Pooka",
            ),
            _memory_card("Acrobat2", clue, refs, current_role="Pooka"),
            _memory_card("Acrobat", clue, refs),
            _memory_card(
                "Acrobat2",
                clue,
                refs,
                current_role="Acrobat2",
                disguise="Acrobat",
            ),
        )
        for card in rejected:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

    def test_direct_and_poet_accept_exact_geometry_for_full_current_domain(self):
        cases = (
            (6, 1, -1),
            (6, 1, 1),
            (6, 1, 2),
            (6, 1, 3),
            (8, 3, 4),
            (10, 4, 5),
            (2, 1, 1),
            (2, 1, 2),
            (2, 1, 3),
            (1, 1, -1),
            (1, 1, 1),
            (1, 1, 3),
        )
        for n_cards, position, distance in cases:
            clue = _bard_native_text(distance)
            refs = _current_bard_refs(position, distance, n_cards)
            for role, poet in (("Acrobat2", False), ("Poet", True)):
                with self.subTest(
                    role=role,
                    n_cards=n_cards,
                    position=position,
                    distance=distance,
                    refs=refs,
                ):
                    parsed = _parse_clue_from_memory(
                        _memory_card(role, clue, refs, position=position),
                        n_cards=n_cards,
                    )
                    self._assert_current(parsed, distance, poet=poet)

    def test_direct_and_poet_require_newest_exact_ordered_geometry(self):
        clue = _bard_native_text(1)
        expected = _current_bard_refs(1, 1, 6)
        malformed_refs = (
            _ABSENT,
            None,
            tuple(expected),
            [],
            list(reversed(expected)),
            expected[:1],
            expected + [2],
            [True, 6],
            ["2", 6],
            [0, 6],
            [2, 7],
        )
        for role in ("Acrobat2", "Poet"):
            for refs in malformed_refs:
                with self.subTest(role=role, refs=refs):
                    self.assertIsNone(
                        _parse_clue_from_memory(
                            _memory_card(role, clue, refs),
                            n_cards=6,
                        )
                    )

            coherent = _parse_clue_from_memory(
                _memory_card(
                    role,
                    clue,
                    expected,
                    prior_infos=[
                        {"desc": _bard_native_text(-1), "targets": []},
                    ],
                ),
                n_cards=6,
            )
            self._assert_current(coherent, 1, poet=role == "Poet")

            stale = _memory_card(role, clue, expected)
            stale["acted_infos"].append(
                {"desc": _bard_native_text(2), "targets": [3, 5]}
            )
            self.assertIsNone(_parse_clue_from_memory(stale, n_cards=6))

        opposite = _bard_native_text(3)
        for refs in ([4], [4, 4, 4]):
            with self.subTest(opposite_refs=refs):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Acrobat2", opposite, refs),
                        n_cards=6,
                    )
                )

        beyond_geometry = _bard_native_text(3)
        self.assertIsNone(
            _parse_clue_from_memory(
                _memory_card("Poet", beyond_geometry, [2, 2]),
                n_cards=2,
            )
        )

    def test_direct_and_poet_reject_every_exact_text_mutation(self):
        mutations = (
            "there are no Corrupted characters",
            "There are no corrupted characters",
            "There are no Corrupted characters.",
            " There are no Corrupted characters",
            "There are no Corrupted characters ",
            "I am 0 cards away from Corrupted character",
            "I am 01 card away from Corrupted character",
            "I am 1 cards away from Corrupted character",
            "I am 2 card away from Corrupted character",
            "I am 2 cards away from corrupted character",
            "I am 2 cards away from Corrupted character.",
            "I am 2 cards away from Corrupted character trailing",
        )
        for role in ("Acrobat2", "Poet"):
            for clue in mutations:
                with self.subTest(role=role, clue=clue):
                    self.assertIsNone(
                        _parse_clue_from_memory(
                            _memory_card(role, clue, []),
                            n_cards=6,
                        )
                    )

    def test_direct_and_poet_require_board_actor_and_current_claim_bounds(self):
        clue = _bard_native_text(1)
        refs = _current_bard_refs(1, 1, 6)
        for role in ("Acrobat2", "Poet"):
            for position, n_cards in (
                (0, 6),
                (7, 6),
                (True, 6),
                ("1", 6),
                (1, None),
                (1, 0),
                (1, True),
            ):
                with self.subTest(role=role, position=position, n_cards=n_cards):
                    self.assertIsNone(
                        _parse_clue_from_memory(
                            _memory_card(role, clue, refs, position=position),
                            n_cards=n_cards,
                        )
                    )

            too_far = _bard_native_text(4)
            self.assertIsNone(
                _parse_clue_from_memory(
                    _memory_card(role, too_far, [5, 3]),
                    n_cards=6,
                )
            )

    def test_direct_and_poet_ignore_unrelated_runtime_and_actor_status(self):
        distance = 2
        clue = _bard_native_text(distance)
        refs = _current_bard_refs(1, distance, 6)
        runtime_data = {"type": "direction", "direction": "CCW"}
        for role, poet in (("Acrobat2", False), ("Poet", True)):
            card = _memory_card(
                role,
                clue,
                refs,
                runtime_data=runtime_data,
            )
            card["state"] = "Dead"
            card["statuses"] = ["Corrupted"]
            parsed = _parse_clue_from_memory(card, n_cards=6)
            self._assert_current(parsed, distance, poet=poet)

    def test_rambler_and_baker_surfaces_win_and_remain_unmarked(self):
        interrupted = _parse_clue_from_memory(
            _memory_card("Acrobat2", "#4 shut up!", [4], position=5),
            n_cards=6,
        )
        self.assertEqual(interrupted.apparent_role, "Bard")
        self.assertEqual(interrupted.info_parsed, {"shut_up_target": 4})
        self.assertNotIn("bard_variant", interrupted.info_parsed)

        baker = _parse_clue_from_memory(
            _memory_card("Baker", "I was a Bard", [], position=2),
            n_cards=6,
            baker_rule_version=BAKER_RULE_VERSION,
        )
        self.assertEqual(baker.apparent_role, "Baker")
        self.assertEqual(baker.info_parsed, {"original_role": "Bard"})
        self.assertNotIn("bard_variant", baker.info_parsed)


class BardSessionCaptureTests(unittest.TestCase):
    @staticmethod
    def _run_auto_card(session: GameSession, memory: dict) -> None:
        memory = dict(memory)
        memory["state"] = "Revealed"

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [memory]

            def close(self):
                return None

        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("memory_reader.print_board"),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            redirect_stdout(StringIO()),
        ):
            dispatch("auto_card", [], session)

    def test_auto_card_replaces_only_empty_same_role_direct_placeholder(self):
        clue = _bard_native_text(2)
        memory = _memory_card(
            "Acrobat2",
            clue,
            _current_bard_refs(1, 2, 6),
        )

        current = GameSession(6, 1)
        current.add_card(CardInfo(1, "Bard"))
        self._run_auto_card(current, memory)
        self.assertEqual(current.cards[0].info_text, clue)
        self.assertEqual(
            current.cards[0].info_parsed,
            {
                "corruption_distance": 2,
                "bard_variant": "public_current",
            },
        )

        legacy = GameSession(6, 1)
        legacy.add_card(card_bard(1, 3))
        self._run_auto_card(legacy, memory)
        self.assertEqual(legacy.cards[0].info_text, "")
        self.assertEqual(
            legacy.cards[0].info_parsed,
            {"corruption_distance": 3},
        )


if __name__ == "__main__":
    unittest.main()
