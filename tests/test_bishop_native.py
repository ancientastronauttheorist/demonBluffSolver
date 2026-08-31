"""Current-build native Bishop bridge regressions."""

import unittest

from game_loop import (
    GameSession,
    _bishop_native_text,
    _parse_bishop_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    card_bishop,
    card_poet_with_info,
)
from solver import POET_VARIANT


_ABSENT = object()


def _memory_card(
    role: str,
    clue,
    refs=_ABSENT,
    *,
    position=1,
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
    if runtime_data is not _ABSENT:
        card["runtime_data"] = runtime_data
    return card


class BishopConstructorTests(unittest.TestCase):
    EXACT_CASES = (
        ([2], ["Villager"], "#2 is a Villager"),
        (
            [2, 4],
            ["Outcast", "Demon"],
            "Between\n#2, #4\nthere is:\nOutcast and Demon",
        ),
        (
            [2, 4, 6],
            ["Villager", "Villager", "Minion"],
            "Between\n#2, #4, #6\nthere is:\n"
            "Villager, Villager and Minion",
        ),
    )

    def test_exact_native_builder_and_parser_cover_all_counts(self):
        for targets, types, text in self.EXACT_CASES:
            with self.subTest(targets=targets):
                self.assertEqual(_bishop_native_text(targets, types), text)
                self.assertEqual(
                    _parse_bishop_native_text(text),
                    (targets, types),
                )

    def test_exact_parser_rejects_loose_or_malformed_lookalikes(self):
        lookalikes = (
            "#2 is a villager",
            "#2 is Villager",
            "#02 is a Villager",
            "#2  is a Villager",
            "#2 is a Villager.",
            " #2 is a Villager",
            "Between #2, #4 there is: Outcast and Demon",
            "Between\r\n#2, #4\r\nthere is:\r\nOutcast and Demon",
            "Between\n#4, #2\nthere is:\nOutcast and Demon",
            "Between\n#2,#4\nthere is:\nOutcast and Demon",
            "Between\n#2, #4\nthere is:\nOutcast, Demon",
            "Between\n#2, #4\nthere is:\nOutcast and demon",
            "Between\n#2, #4\nthere are:\nOutcast and Demon",
            "Between\n#2, #4, #6\nthere is:\n"
            "Villager, Villager, and Minion",
            "Between\n#2, #4, #6\nthere is:\n"
            "Villager and Villager and Minion",
            "Between\n#2, #4, #6\nthere is:\n"
            "Villager, Villager and Minion!",
            "",
            None,
        )
        for clue in lookalikes:
            with self.subTest(clue=clue):
                self.assertIsNone(_parse_bishop_native_text(clue))

    def test_builder_rejects_invalid_targets_and_types(self):
        invalid = (
            ((1,), ["Villager"]),
            ([], []),
            ([1, 2, 3, 4], ["Villager"] * 4),
            ([0], ["Villager"]),
            ([True], ["Villager"]),
            ([2, 2], ["Villager", "Demon"]),
            ([4, 2], ["Villager", "Demon"]),
            ([2], ("Villager",)),
            ([2], []),
            ([2], ["villager"]),
            ([2], ["Good"]),
            ([2, 4], ["Villager"]),
        )
        for targets, types in invalid:
            with self.subTest(targets=targets, types=types), self.assertRaises(
                ValueError
            ):
                _bishop_native_text(targets, types)

    def test_constructor_preserves_legacy_and_marks_only_direct_current(self):
        legacy = card_bishop(1, [4, 2], ["Demon", "Villager"])
        self.assertEqual(legacy.info_text, "")
        self.assertEqual(
            legacy.info_parsed,
            {"targets": [4, 2], "types": ["Demon", "Villager"]},
        )

        target_only = card_bishop(1, [3])
        self.assertEqual(target_only.info_parsed, {"targets": [3]})

        current = card_bishop(
            1,
            [2, 4],
            ["Villager", "Demon"],
            bishop_variant="public_current",
        )
        self.assertEqual(
            current.info_text,
            "Between\n#2, #4\nthere is:\nVillager and Demon",
        )
        self.assertEqual(current.info_parsed["bishop_variant"], "public_current")

        with self.assertRaisesRegex(ValueError, "must match"):
            card_bishop(
                1,
                [2],
                ["Villager"],
                info_text="#2 is a Demon",
                bishop_variant="public_current",
            )


class BishopManualIngestionTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession(6, 1)

    def test_direct_cli_is_strict_and_synthesizes_exact_current_text(self):
        cases = (
            (
                ["bishop", "1", "4", "demon"],
                [4],
                ["Demon"],
                "#4 is a Demon",
            ),
            (
                ["bishop", "2", "4,2", "outcast,VILLAGER"],
                [2, 4],
                ["Outcast", "Villager"],
                "Between\n#2, #4\nthere is:\nOutcast and Villager",
            ),
            (
                [
                    "bishop",
                    "6",
                    "6,2,4",
                    "Villager,Villager,Minion",
                ],
                [2, 4, 6],
                ["Villager", "Villager", "Minion"],
                "Between\n#2, #4, #6\nthere is:\n"
                "Villager, Villager and Minion",
            ),
        )
        for args, targets, types, text in cases:
            with self.subTest(args=args):
                parsed = _parse_card_cli(args, self.session)
                self.assertEqual(parsed.apparent_role, "Bishop")
                self.assertEqual(parsed.info_text, text)
                self.assertEqual(
                    parsed.info_parsed,
                    {
                        "targets": targets,
                        "types": types,
                        "bishop_variant": "public_current",
                    },
                )

    def test_direct_cli_rejects_missing_context_or_invalid_schema(self):
        invalid = (
            (["bishop", "1", "2", "Villager"], None),
            (["bishop", "1", "2"], self.session),
            (["bishop", "1", "2", "Villager", "extra"], self.session),
            (["bishop", "0", "2", "Villager"], self.session),
            (["bishop", "7", "2", "Villager"], self.session),
            (["bishop", "1", "", "Villager"], self.session),
            (["bishop", "1", "x", "Villager"], self.session),
            (["bishop", "1", "0", "Villager"], self.session),
            (["bishop", "1", "7", "Villager"], self.session),
            (["bishop", "1", "2,2", "Villager,Demon"], self.session),
            (
                ["bishop", "1", "2,3,4,5", "Villager,Outcast,Minion,Demon"],
                self.session,
            ),
            (["bishop", "1", "2,3", "Villager"], self.session),
            (["bishop", "1", "2", "Good"], self.session),
        )
        for args, session in invalid:
            with self.subTest(args=args, session=session), self.assertRaises(
                (IndexError, ValueError)
            ):
                _parse_card_cli(args, session)

    def test_poet_manual_entry_reuses_exact_builder_without_direct_marker(self):
        parsed = card_poet_with_info(
            5,
            "bishop",
            ["6,2,4", "demon,VILLAGER,villager"],
            n_cards=6,
        )
        self.assertEqual(
            parsed.info_text,
            "Between\n#2, #4, #6\nthere is:\nDemon, Villager and Villager",
        )
        self.assertEqual(
            parsed.info_parsed,
            {
                "targets": [2, 4, 6],
                "types": ["Demon", "Villager", "Villager"],
                "copied_role": "Bishop",
                "poet_variant": POET_VARIANT,
            },
        )
        self.assertNotIn("bishop_variant", parsed.info_parsed)

        cli = _parse_card_cli(
            ["poet", "5", "bishop", "4,2", "Minion,Outcast"],
            self.session,
        )
        self.assertEqual(
            cli.info_text,
            "Between\n#2, #4\nthere is:\nMinion and Outcast",
        )

    def test_poet_manual_entry_requires_board_and_exact_payload(self):
        invalid = (
            (1, ["2", "Villager"], None),
            (0, ["2", "Villager"], 6),
            (7, ["2", "Villager"], 6),
            (1, [], 6),
            (1, ["2"], 6),
            (1, ["2", "Villager", "extra"], 6),
            (1, ["0", "Villager"], 6),
            (1, ["7", "Villager"], 6),
            (1, ["2,2", "Villager,Demon"], 6),
            (1, ["2,3,4,5", "Villager,Outcast,Minion,Demon"], 6),
            (1, ["2,3", "Villager"], 6),
            (1, ["2", "Good"], 6),
        )
        for position, args, n_cards in invalid:
            with self.subTest(position=position, args=args), self.assertRaises(
                ValueError
            ):
                card_poet_with_info(
                    position,
                    "Bishop",
                    args,
                    n_cards=n_cards,
                )


class BishopMemoryIngestionTests(unittest.TestCase):
    def test_direct_accepts_all_counts_permuted_refs_and_type_multiplicity(self):
        cases = (
            ("#2 is a Villager", [2], [2], ["Villager"]),
            (
                "Between\n#2, #5\nthere is:\nOutcast and Demon",
                [5, 2],
                [2, 5],
                ["Outcast", "Demon"],
            ),
            (
                "Between\n#2, #4, #6\nthere is:\n"
                "Villager, Villager and Minion",
                [6, 2, 4],
                [2, 4, 6],
                ["Villager", "Villager", "Minion"],
            ),
        )
        for clue, refs, targets, types in cases:
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_card("Bishop", clue, refs),
                    n_cards=6,
                )
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.apparent_role, "Bishop")
                self.assertEqual(parsed.info_text, clue)
                self.assertEqual(
                    parsed.info_parsed,
                    {
                        "targets": targets,
                        "types": types,
                        "bishop_variant": "public_current",
                    },
                )

    def test_direct_accepts_current_role_and_disguise_precedence(self):
        clue = "Between\n#1, #4\nthere is:\nVillager and Demon"
        current_role = _memory_card("Pooka", clue, [4, 1])
        current_role["current_role"] = "Bishop"

        disguised = _memory_card("Pooka", clue, [1, 4])
        disguised["current_role"] = "Pooka"
        disguised["disguise"] = "Bishop"

        for card in (current_role, disguised):
            with self.subTest(card=card):
                parsed = _parse_clue_from_memory(card, n_cards=6)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.apparent_role, "Bishop")
                self.assertEqual(parsed.info_parsed["targets"], [1, 4])

    def test_direct_ignores_stale_unrelated_runtime_data(self):
        clue = "#3 is a Minion"
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Bishop",
                clue,
                [3],
                runtime_data={"type": "direction", "direction": "CCW"},
            ),
            n_cards=6,
        )
        self.assertEqual(parsed.info_parsed["types"], ["Minion"])

    def test_direct_uses_only_newest_coherent_exact_event(self):
        clue = "Between\n#2, #4\nthere is:\nVillager and Demon"
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Bishop",
                clue,
                [4, 2],
                prior_infos=[
                    {"desc": "#1 is a Minion", "targets": [1]},
                ],
            ),
            n_cards=6,
        )
        self.assertEqual(parsed.info_parsed["targets"], [2, 4])

        stale = _memory_card("Bishop", clue, [2, 4])
        stale["acted_infos"].append(
            {"desc": "newer unrelated result", "targets": [3]}
        )
        self.assertIsNone(_parse_clue_from_memory(stale, n_cards=6))

    def test_direct_rejects_incomplete_or_malformed_events_without_placeholder(self):
        clue = "Between\n#2, #4\nthere is:\nVillager and Demon"
        malformed = (
            _memory_card("Bishop", clue),
            _memory_card("Bishop", clue, None),
            _memory_card("Bishop", clue, (2, 4)),
            _memory_card("Bishop", clue, [2]),
            _memory_card("Bishop", clue, [2, 3]),
            _memory_card("Bishop", clue, [2, 2]),
            _memory_card("Bishop", clue, [True, 4]),
            _memory_card("Bishop", clue, ["2", 4]),
            _memory_card("Bishop", clue, [0, 4]),
            _memory_card("Bishop", clue, [2, 7]),
            {
                **_memory_card("Bishop", clue, [2, 4]),
                "acted_infos": [{"desc": clue}],
            },
            {
                **_memory_card("Bishop", clue, [2, 4]),
                "acted_infos": [None],
            },
            _memory_card("Bishop", "", _ABSENT),
            _memory_card("Bishop", None, _ABSENT),
        )
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

    def test_direct_requires_in_board_actor_and_targets(self):
        clue = "#2 is a Villager"
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
                        _memory_card(
                            "Bishop",
                            clue,
                            [2],
                            position=position,
                        ),
                        n_cards=n_cards,
                    )
                )

        self.assertIsNone(
            _parse_clue_from_memory(
                _memory_card("Bishop", "#7 is a Villager", [7]),
                n_cards=6,
            )
        )

    def test_direct_rejects_loose_text_even_when_refs_match(self):
        for clue in (
            "#2 is a villager",
            "#2 is a Villager.",
            "Between #2, #4 there is: Villager and Demon",
            "Between\n#4, #2\nthere is:\nVillager and Demon",
            "Between\n#2, #4\nthere is:\nVillager and demon",
            "Between\n#2, #4\nthere is:\nVillager and Demon trailing",
        ):
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Bishop", clue, [2, 4]),
                        n_cards=6,
                    )
                )

    def test_poet_uses_same_exact_parser_and_shuffled_ref_set(self):
        clue = (
            "Between\n#2, #4, #5\nthere is:\n"
            "Demon, Villager and Villager"
        )
        parsed = _parse_clue_from_memory(
            _memory_card("Poet", clue, [5, 2, 4], position=5),
            n_cards=6,
        )
        self.assertEqual(parsed.apparent_role, "Poet")
        self.assertEqual(parsed.info_text, clue)
        self.assertEqual(
            parsed.info_parsed,
            {
                "targets": [2, 4, 5],
                "types": ["Demon", "Villager", "Villager"],
                "copied_role": "Bishop",
                "poet_variant": POET_VARIANT,
            },
        )
        self.assertNotIn("bishop_variant", parsed.info_parsed)

    def test_poet_bishop_ignores_stale_enlightened_runtime(self):
        clue = "#3 is a Demon"
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Poet",
                clue,
                [3],
                runtime_data={"type": "direction", "direction": "CW"},
            ),
            n_cards=6,
        )
        self.assertEqual(parsed.info_parsed["copied_role"], "Bishop")
        self.assertEqual(parsed.info_parsed["types"], ["Demon"])

    def test_poet_requires_in_board_actor_and_newest_exact_event(self):
        clue = "Between\n#2, #4\nthere is:\nVillager and Demon"
        malformed = (
            (_memory_card("Poet", clue), 6),
            (_memory_card("Poet", clue, [2]), 6),
            (_memory_card("Poet", clue, [2, 2]), 6),
            (_memory_card("Poet", clue, [2, 7]), 6),
            (_memory_card("Poet", clue, [2, 4], position=0), 6),
            (_memory_card("Poet", clue, [2, 4], position=7), 6),
            (_memory_card("Poet", clue, [2, 4]), None),
        )
        stale = _memory_card("Poet", clue, [2, 4])
        stale["acted_infos"].append({"desc": "newer", "targets": [2, 4]})
        malformed += ((stale, 6),)

        for card, n_cards in malformed:
            with self.subTest(card=card, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=n_cards)
                )


if __name__ == "__main__":
    unittest.main()
