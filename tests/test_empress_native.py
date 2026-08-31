"""Current-build native Empress/Noble bridge regressions."""

import json
from pathlib import Path
import unittest

from game_loop import (
    GameSession,
    _empress_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    _parse_empress_native_text,
    card_empress,
    card_poet_with_info,
)
from solver import POET_VARIANT


_ABSENT = object()

_CURRENT_CLUE = "One is Evil:\n#2, #3 or #4"


def _memory_card(
    role: str,
    clue=_CURRENT_CLUE,
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


class EmpressConstructorTests(unittest.TestCase):
    def test_exact_native_builder_and_parser(self):
        self.assertEqual(_empress_native_text([2, 3, 4]), _CURRENT_CLUE)
        self.assertEqual(
            _parse_empress_native_text(_CURRENT_CLUE),
            [2, 3, 4],
        )

    def test_exact_parser_rejects_every_loose_or_malformed_lookalike(self):
        lookalikes = (
            "one is Evil:\n#2, #3 or #4",
            "One  is Evil:\n#2, #3 or #4",
            "One is evil:\n#2, #3 or #4",
            "One is Evil: #2, #3 or #4",
            "One is Evil:\r\n#2, #3 or #4",
            "One is Evil:\n#02, #3 or #4",
            "One is Evil:\n#2,#3 or #4",
            "One is Evil:\n#2,  #3 or #4",
            "One is Evil:\n#2, #3  or #4",
            "One is Evil:\n#2, #3 or  #4",
            "One is Evil:\n#4, #2 or #3",
            "One is Evil:\n#2, #2 or #4",
            "One is Evil:\n#2, #3 or #4.",
            "One is Evil:\n#2, #3 or #4!",
            "One is Evil:\n#2, #3 or #4 ",
            " One is Evil:\n#2, #3 or #4",
            "One is Evil:\n#2, #3 or #4 trailing",
            "",
            None,
        )
        for clue in lookalikes:
            with self.subTest(clue=clue):
                self.assertIsNone(_parse_empress_native_text(clue))

    def test_unmarked_constructor_preserves_every_legacy_shape(self):
        for targets in ([4, 2, 3], [1, 3]):
            with self.subTest(targets=targets):
                card = card_empress(1, targets)
                self.assertEqual(card.info_text, "")
                self.assertEqual(card.info_parsed, {"targets": targets})

    def test_asc65_v4_two_target_poet_stays_unversioned_legacy(self):
        fixture = json.loads(
            (Path(__file__).parent / "cases_v2" / "asc65_v4.json").read_text(
                encoding="utf-8"
            )
        )
        poet = next(
            card
            for card in fixture["cards"]
            if card["position"] == 2 and card["apparent_role"] == "Poet"
        )
        self.assertEqual(poet["info_text"], "")
        self.assertEqual(
            poet["info_parsed"],
            {"targets": [1, 3], "copied_role": "Empress"},
        )
        self.assertNotIn("poet_variant", poet["info_parsed"])
        self.assertNotIn("empress_variant", poet["info_parsed"])

    def test_marked_constructor_has_exact_current_schema(self):
        card = card_empress(
            1,
            [2, 3, 4],
            empress_variant="public_current",
        )
        self.assertEqual(card.info_text, _CURRENT_CLUE)
        self.assertEqual(
            card.info_parsed,
            {
                "targets": [2, 3, 4],
                "empress_variant": "public_current",
            },
        )

    def test_marked_constructor_rejects_noncurrent_schema(self):
        invalid_targets = (
            (),
            [],
            [2, 3],
            [1, 2, 3, 4],
            [2, 2, 4],
            [4, 2, 3],
            [0, 2, 3],
            [True, 2, 3],
            ["1", 2, 3],
        )
        for targets in invalid_targets:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                card_empress(
                    1,
                    targets,
                    empress_variant="public_current",
                )

        for position in (0, True, "1"):
            with self.subTest(position=position), self.assertRaises(ValueError):
                card_empress(
                    position,
                    [1, 2, 3],
                    empress_variant="public_current",
                )

        with self.assertRaises(ValueError):
            card_empress(1, [1, 2, 3], empress_variant="future")
        with self.assertRaisesRegex(ValueError, "must match"):
            card_empress(
                1,
                [1, 2, 3],
                info_text=_CURRENT_CLUE,
                empress_variant="public_current",
            )
        with self.assertRaises(ValueError):
            card_empress(
                1,
                [1, 2, 3],
                info_text=None,
                empress_variant="public_current",
            )


class EmpressManualIngestionTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession(6, 1)

    def test_direct_cli_stamps_and_canonicalizes_native_order(self):
        card = _parse_card_cli(["empress", "1", "4,2,3"], self.session)
        self.assertEqual(card.apparent_role, "Empress")
        self.assertEqual(card.info_text, _CURRENT_CLUE)
        self.assertEqual(
            card.info_parsed,
            {
                "targets": [2, 3, 4],
                "empress_variant": "public_current",
            },
        )

    def test_direct_cli_requires_board_and_exact_payload(self):
        invalid = (
            (["empress", "1", "2,3,4"], None),
            (["empress", "1"], self.session),
            (["empress", "1", "2,3,4", "extra"], self.session),
            (["empress", "0", "2,3,4"], self.session),
            (["empress", "7", "2,3,4"], self.session),
            (["empress", "1", ""], self.session),
            (["empress", "1", "2,x,4"], self.session),
            (["empress", "1", "2,3"], self.session),
            (["empress", "1", "2,3,4,5"], self.session),
            (["empress", "1", "2,2,4"], self.session),
            (["empress", "1", "0,2,3"], self.session),
            (["empress", "1", "2,3,7"], self.session),
        )
        for args, session in invalid:
            with self.subTest(args=args), self.assertRaises(
                (IndexError, ValueError)
            ):
                _parse_card_cli(args, session)

    def test_poet_manual_stamps_only_poet_and_canonicalizes_order(self):
        card = card_poet_with_info(
            5,
            "empress",
            ["4,2,3"],
            n_cards=6,
        )
        self.assertEqual(card.info_text, _CURRENT_CLUE)
        self.assertEqual(
            card.info_parsed,
            {
                "targets": [2, 3, 4],
                "copied_role": "Empress",
                "poet_variant": POET_VARIANT,
            },
        )
        self.assertNotIn("empress_variant", card.info_parsed)

    def test_poet_manual_requires_board_actor_and_exact_payload(self):
        invalid = (
            (1, ["2,3,4"], None),
            (0, ["2,3,4"], 6),
            (7, ["2,3,4"], 6),
            (1, [], 6),
            (1, ["2,3,4", "extra"], 6),
            (1, ["2,3"], 6),
            (1, ["2,2,4"], 6),
            (1, ["0,2,3"], 6),
            (1, ["2,3,7"], 6),
        )
        for position, args, n_cards in invalid:
            with self.subTest(position=position, args=args), self.assertRaises(
                ValueError
            ):
                card_poet_with_info(
                    position,
                    "Empress",
                    args,
                    n_cards=n_cards,
                )


class EmpressMemoryIngestionTests(unittest.TestCase):
    def test_direct_accepts_public_managed_current_and_disguise_surfaces(self):
        direct = _memory_card("Empress", refs=[2, 3, 4])
        managed = _memory_card("Noble", refs=[2, 3, 4])
        current = _memory_card("Pooka", refs=[2, 3, 4])
        current["current_role"] = "Noble"
        disguised = _memory_card("Pooka", refs=[2, 3, 4])
        disguised["current_role"] = "Pooka"
        disguised["disguise"] = "Empress"

        for card in (direct, managed, current, disguised):
            with self.subTest(card=card):
                parsed = _parse_clue_from_memory(card, n_cards=6)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.apparent_role, "Empress")
                self.assertEqual(parsed.info_text, _CURRENT_CLUE)
                self.assertEqual(
                    parsed.info_parsed,
                    {
                        "targets": [2, 3, 4],
                        "empress_variant": "public_current",
                    },
                )

        hidden = _memory_card("Empress", refs=[2, 3, 4])
        hidden["current_role"] = "Empress"
        hidden["disguise"] = "Pooka"
        self.assertIsNone(_parse_clue_from_memory(hidden, n_cards=6))

    def test_direct_allows_self_reference_and_ignores_runtime_data(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Empress",
                clue=_empress_native_text([1, 2, 3]),
                refs=[1, 2, 3],
                runtime_data={"type": "direction", "direction": "CW"},
            ),
            n_cards=6,
        )
        self.assertEqual(parsed.info_parsed["targets"], [1, 2, 3])
        self.assertNotIn("runtime_data", parsed.info_parsed)

    def test_direct_uses_only_newest_coherent_event(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Empress",
                refs=[2, 3, 4],
                prior_infos=[
                    {"desc": "older", "targets": [1, 2, 3]},
                ],
            ),
            n_cards=6,
        )
        self.assertEqual(parsed.info_parsed["targets"], [2, 3, 4])

        stale = _memory_card("Empress", refs=[2, 3, 4])
        stale["acted_infos"].append(
            {"desc": "newer unrelated result", "targets": [1, 2, 3]}
        )
        self.assertIsNone(_parse_clue_from_memory(stale, n_cards=6))

    def test_direct_rejects_incomplete_malformed_or_unordered_events(self):
        malformed = (
            _memory_card("Empress"),
            _memory_card("Empress", refs=None),
            _memory_card("Empress", refs=(2, 3, 4)),
            _memory_card("Empress", refs=[2, 3]),
            _memory_card("Empress", refs=[1, 2, 3, 4]),
            _memory_card("Empress", refs=[2, 2, 4]),
            _memory_card("Empress", refs=[4, 2, 3]),
            _memory_card("Empress", refs=[True, 3, 4]),
            _memory_card("Empress", refs=["2", 3, 4]),
            _memory_card(
                "Empress",
                clue="One is Evil:\n#0, #2 or #3",
                refs=[0, 2, 3],
            ),
            _memory_card(
                "Empress",
                clue=_empress_native_text([2, 3, 7]),
                refs=[2, 3, 7],
            ),
            {
                **_memory_card("Empress", refs=[2, 3, 4]),
                "acted_infos": [{"desc": _CURRENT_CLUE}],
            },
            {
                **_memory_card("Empress", refs=[2, 3, 4]),
                "acted_infos": [None],
            },
            _memory_card("Empress", clue="", refs=[2, 3, 4]),
            _memory_card("Empress", clue=None, refs=[2, 3, 4]),
        )
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

    def test_direct_requires_current_board_actor(self):
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
                            "Empress",
                            refs=[2, 3, 4],
                            position=position,
                        ),
                        n_cards=n_cards,
                    )
                )

    def test_direct_and_poet_reject_exact_text_mutations(self):
        mutations = (
            "one is Evil:\n#2, #3 or #4",
            "One  is Evil:\n#2, #3 or #4",
            "One is Evil: #2, #3 or #4",
            "One is Evil:\r\n#2, #3 or #4",
            "One is Evil:\n#02, #3 or #4",
            "One is Evil:\n#2,#3 or #4",
            "One is Evil:\n#2, #3 or #4.",
            "One is Evil:\n#2, #3 or #4 ",
            "One is Evil:\n#4, #2 or #3",
            "One is Evil:\n#2, #3 or #4 trailing",
        )
        for role in ("Empress", "Poet"):
            for clue in mutations:
                with self.subTest(role=role, clue=clue):
                    self.assertIsNone(
                        _parse_clue_from_memory(
                            _memory_card(role, clue=clue, refs=[2, 3, 4]),
                            n_cards=6,
                        )
                    )

    def test_poet_requires_identical_ordered_refs_and_current_board_actor(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Poet",
                refs=[2, 3, 4],
                position=1,
                runtime_data={"type": "direction", "direction": "CCW"},
            ),
            n_cards=6,
        )
        self.assertEqual(
            parsed.info_parsed,
            {
                "targets": [2, 3, 4],
                "copied_role": "Empress",
                "poet_variant": POET_VARIANT,
            },
        )
        self.assertNotIn("empress_variant", parsed.info_parsed)

        malformed = (
            (_memory_card("Poet"), 6),
            (_memory_card("Poet", refs=[2, 4, 3]), 6),
            (_memory_card("Poet", refs=[2, 2, 4]), 6),
            (_memory_card("Poet", refs=[2, 3, 7]), 6),
            (_memory_card("Poet", refs=[2, 3, 4], position=0), 6),
            (_memory_card("Poet", refs=[2, 3, 4], position=7), 6),
            (_memory_card("Poet", refs=[2, 3, 4]), None),
        )
        stale = _memory_card("Poet", refs=[2, 3, 4])
        stale["acted_infos"].append(
            {"desc": "newer unrelated result", "targets": [2, 3, 4]}
        )
        malformed += ((stale, 6),)

        for card, n_cards in malformed:
            with self.subTest(card=card, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=n_cards)
                )


if __name__ == "__main__":
    unittest.main()
