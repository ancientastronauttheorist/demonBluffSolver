"""Current-build native Gemcrafter/Archivist bridge regressions."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from game_loop import (
    DecisionLog,
    GameSession,
    _gemcrafter_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    _parse_gemcrafter_native_text,
    card_gemcrafter,
    card_poet_with_info,
    dispatch,
)
from memory_reader import clean_name
from solver import CardInfo, POET_VARIANT


_ABSENT = object()
_CURRENT_CLUE = "#2 is Good"


def _memory_card(
    role: str,
    clue=_CURRENT_CLUE,
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


class GemcrafterConstructorTests(unittest.TestCase):
    def test_exact_native_builder_and_parser(self):
        self.assertEqual(_gemcrafter_native_text(2), _CURRENT_CLUE)
        self.assertEqual(_parse_gemcrafter_native_text(_CURRENT_CLUE), 2)

    def test_exact_parser_rejects_every_loose_or_malformed_lookalike(self):
        lookalikes = (
            "#2 is good",
            "#2 Is Good",
            "#2  is Good",
            "#2 is  Good",
            "# 2 is Good",
            "#02 is Good",
            "#0 is Good",
            " #2 is Good",
            "#2 is Good ",
            "#2 is Good.",
            "#2 is Good!",
            "#2\nis Good",
            "#2 is\nGood",
            "#2 is Good\n",
            "#2 is Evil",
            "#2 is Good trailing",
            "#" + ("9" * 5000) + " is Good",
            "",
            None,
        )
        for clue in lookalikes:
            with self.subTest(clue=clue):
                self.assertIsNone(_parse_gemcrafter_native_text(clue))

    def test_unmarked_constructor_preserves_legacy_shape(self):
        legacy = card_gemcrafter(2, 2)
        self.assertEqual(legacy.apparent_role, "Gemcrafter")
        self.assertEqual(legacy.info_text, "")
        self.assertEqual(legacy.info_parsed, {"good_position": 2})

        historical_text = card_gemcrafter(
            1,
            3,
            info_text="archived text",
        )
        self.assertEqual(historical_text.info_text, "archived text")
        self.assertNotIn("gemcrafter_variant", historical_text.info_parsed)

    def test_marked_constructor_has_exact_schema_and_allows_self(self):
        current = card_gemcrafter(
            2,
            2,
            gemcrafter_variant="public_current",
        )
        self.assertEqual(current.info_text, _CURRENT_CLUE)
        self.assertEqual(
            current.info_parsed,
            {
                "good_position": 2,
                "gemcrafter_variant": "public_current",
            },
        )

    def test_marked_constructor_rejects_noncurrent_schema(self):
        for position, target in (
            (0, 2),
            (True, 2),
            ("1", 2),
            (1, 0),
            (1, -1),
            (1, True),
            (1, "2"),
        ):
            with self.subTest(position=position, target=target), self.assertRaises(
                ValueError
            ):
                card_gemcrafter(
                    position,
                    target,
                    gemcrafter_variant="public_current",
                )

        with self.assertRaises(ValueError):
            card_gemcrafter(1, 2, gemcrafter_variant="future")
        with self.assertRaisesRegex(ValueError, "must match"):
            card_gemcrafter(
                1,
                2,
                info_text="#3 is Good",
                gemcrafter_variant="public_current",
            )
        with self.assertRaises(ValueError):
            card_gemcrafter(
                1,
                2,
                info_text=None,
                gemcrafter_variant="public_current",
            )

    def test_reader_maps_current_and_legacy_managed_names(self):
        self.assertEqual(clean_name("Archivist"), "Gemcrafter")
        self.assertEqual(clean_name("Archivist_12345"), "Gemcrafter")
        self.assertEqual(clean_name("Gambler"), "Gemcrafter")

    def test_archived_direct_rambler_and_baker_shapes_remain_unmarked(self):
        case_root = Path(__file__).parent / "cases_v2"
        direct_case = json.loads(
            (case_root / "asc31_v2.json").read_text(encoding="utf-8")
        )
        direct = next(
            card
            for card in direct_case["cards"]
            if card["apparent_role"] == "Gemcrafter"
        )
        self.assertEqual(direct["info_parsed"], {"good_position": 2})

        rambler_case = json.loads(
            (case_root / "asc83_v7.json").read_text(encoding="utf-8")
        )
        interrupted = next(
            card
            for card in rambler_case["cards"]
            if card["position"] == 5
        )
        self.assertEqual(interrupted["apparent_role"], "Gemcrafter")
        self.assertEqual(interrupted["info_parsed"], {"shut_up_target": 4})

        baker_case = json.loads(
            (case_root / "asc47_v5.json").read_text(encoding="utf-8")
        )
        baker = next(
            card
            for card in baker_case["cards"]
            if card["position"] == 4
        )
        self.assertEqual(baker["apparent_role"], "Baker")
        self.assertEqual(baker["info_parsed"], {"original_role": "Gemcrafter"})
        for card in (direct, interrupted, baker):
            self.assertFalse(
                any(key.endswith("_variant") for key in card["info_parsed"])
            )


class GemcrafterManualIngestionTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession(6, 1)

    def test_direct_cli_stamps_exact_current_self_claim(self):
        card = _parse_card_cli(["gemcrafter", "2", "2"], self.session)
        self.assertEqual(card.apparent_role, "Gemcrafter")
        self.assertEqual(card.info_text, _CURRENT_CLUE)
        self.assertEqual(
            card.info_parsed,
            {
                "good_position": 2,
                "gemcrafter_variant": "public_current",
            },
        )

    def test_direct_cli_requires_board_exact_arity_and_bounds(self):
        invalid = (
            (["gemcrafter", "1", "2"], None),
            (["gemcrafter", "1"], self.session),
            (["gemcrafter", "1", "2", "extra"], self.session),
            (["gemcrafter", "0", "2"], self.session),
            (["gemcrafter", "7", "2"], self.session),
            (["gemcrafter", "1", "0"], self.session),
            (["gemcrafter", "1", "7"], self.session),
            (["gemcrafter", "1", "not-an-id"], self.session),
        )
        for args, session in invalid:
            with self.subTest(args=args), self.assertRaises(
                (IndexError, ValueError)
            ):
                _parse_card_cli(args, session)

    def test_poet_manual_synthesizes_text_and_stamps_only_poet(self):
        card = card_poet_with_info(
            2,
            "gem_crafter",
            ["2"],
            n_cards=6,
        )
        self.assertEqual(card.apparent_role, "Poet")
        self.assertEqual(card.info_text, _CURRENT_CLUE)
        self.assertEqual(
            card.info_parsed,
            {
                "good_position": 2,
                "copied_role": "Gemcrafter",
                "poet_variant": POET_VARIANT,
            },
        )
        self.assertNotIn("gemcrafter_variant", card.info_parsed)

    def test_poet_manual_requires_board_actor_exact_arity_and_bounds(self):
        invalid = (
            (1, ["2"], None),
            (0, ["2"], 6),
            (7, ["2"], 6),
            (True, ["2"], 6),
            (1, [], 6),
            (1, ["2", "extra"], 6),
            (1, ["0"], 6),
            (1, ["7"], 6),
            (1, ["not-an-id"], 6),
        )
        for position, args, n_cards in invalid:
            with self.subTest(position=position, args=args), self.assertRaises(
                ValueError
            ):
                card_poet_with_info(
                    position,
                    "Gemcrafter",
                    args,
                    n_cards=n_cards,
                )


class GemcrafterMemoryIngestionTests(unittest.TestCase):
    def test_direct_accepts_public_managed_legacy_and_display_surfaces(self):
        cards = (
            _memory_card("Gemcrafter", refs=[2]),
            _memory_card("Archivist", refs=[2]),
            _memory_card("Gambler", refs=[2]),
            _memory_card(
                "Pooka",
                refs=[2],
                current_role="Archivist",
            ),
            _memory_card(
                "Pooka",
                refs=[2],
                current_role="Pooka",
                disguise="Gemcrafter",
            ),
            _memory_card(
                "Pooka",
                refs=[2],
                current_role="Pooka",
                disguise="Archivist",
            ),
        )
        for card in cards:
            with self.subTest(card=card):
                parsed = _parse_clue_from_memory(card, n_cards=6)
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.apparent_role, "Gemcrafter")
                self.assertEqual(parsed.info_text, _CURRENT_CLUE)
                self.assertEqual(
                    parsed.info_parsed,
                    {
                        "good_position": 2,
                        "gemcrafter_variant": "public_current",
                    },
                )

    def test_disguise_then_current_then_true_role_precedence_is_preserved(self):
        disguised_away = _memory_card(
            "Archivist",
            refs=[2],
            current_role="Archivist",
            disguise="Pooka",
        )
        current_away = _memory_card(
            "Archivist",
            refs=[2],
            current_role="Pooka",
        )
        for card in (disguised_away, current_away):
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

    def test_direct_and_poet_allow_self_and_ignore_stale_runtime_data(self):
        clue = "#1 is Good"
        runtime_data = {"type": "direction", "direction": "CCW"}
        direct = _parse_clue_from_memory(
            _memory_card(
                "Archivist",
                clue,
                [1],
                runtime_data=runtime_data,
            ),
            n_cards=6,
        )
        poet = _parse_clue_from_memory(
            _memory_card(
                "Poet",
                clue,
                [1],
                runtime_data=runtime_data,
            ),
            n_cards=6,
        )
        self.assertEqual(direct.info_parsed["good_position"], 1)
        self.assertNotIn("runtime_data", direct.info_parsed)
        self.assertEqual(
            poet.info_parsed,
            {
                "good_position": 1,
                "copied_role": "Gemcrafter",
                "poet_variant": POET_VARIANT,
            },
        )
        self.assertNotIn("gemcrafter_variant", poet.info_parsed)

    def test_direct_uses_only_newest_coherent_exact_event(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Archivist",
                refs=[2],
                prior_infos=[{"desc": "#3 is Good", "targets": [3]}],
            ),
            n_cards=6,
        )
        self.assertEqual(parsed.info_parsed["good_position"], 2)

        stale = _memory_card("Archivist", refs=[2])
        stale["acted_infos"].append(
            {"desc": "#3 is Good", "targets": [3]}
        )
        self.assertIsNone(_parse_clue_from_memory(stale, n_cards=6))

    def test_direct_rejects_incomplete_malformed_or_mismatched_events(self):
        malformed = (
            _memory_card("Archivist"),
            _memory_card("Archivist", refs=None),
            _memory_card("Archivist", refs=()),
            _memory_card("Archivist", refs=[]),
            _memory_card("Archivist", refs=[2, 3]),
            _memory_card("Archivist", refs=[3]),
            _memory_card("Archivist", refs=[True]),
            _memory_card("Archivist", refs=["2"]),
            _memory_card("Archivist", refs=[0]),
            _memory_card("Archivist", refs=[7]),
            {
                **_memory_card("Archivist", refs=[2]),
                "acted_infos": [{"desc": _CURRENT_CLUE}],
            },
            {
                **_memory_card("Archivist", refs=[2]),
                "acted_infos": [None],
            },
            {
                **_memory_card("Archivist", refs=[2]),
                "acted_infos": "not-an-array",
            },
            _memory_card("Archivist", clue="", refs=[2]),
            _memory_card("Archivist", clue=None, refs=[2]),
        )
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

    def test_direct_requires_current_board_actor_and_target(self):
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
                            "Archivist",
                            refs=[2],
                            position=position,
                        ),
                        n_cards=n_cards,
                    )
                )

        self.assertIsNone(
            _parse_clue_from_memory(
                _memory_card("Archivist", "#7 is Good", [7]),
                n_cards=6,
            )
        )

    def test_direct_and_poet_reject_every_exact_text_mutation(self):
        mutations = (
            "#2 is good",
            "#2 Is Good",
            "#2  is Good",
            "#2 is  Good",
            "# 2 is Good",
            "#02 is Good",
            " #2 is Good",
            "#2 is Good ",
            "#2 is Good.",
            "#2 is Good!",
            "#2\nis Good",
            "#2 is\nGood",
            "#2 is Good trailing",
        )
        for role in ("Archivist", "Poet"):
            for clue in mutations:
                with self.subTest(role=role, clue=clue):
                    self.assertIsNone(
                        _parse_clue_from_memory(
                            _memory_card(role, clue, [2]),
                            n_cards=6,
                        )
                    )

    def test_poet_requires_newest_identical_ref_and_in_board_actor(self):
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Poet",
                refs=[2],
                prior_infos=[{"desc": "older", "targets": [4]}],
            ),
            n_cards=6,
        )
        self.assertEqual(parsed.info_parsed["copied_role"], "Gemcrafter")

        malformed = (
            (_memory_card("Poet"), 6),
            (_memory_card("Poet", refs=[]), 6),
            (_memory_card("Poet", refs=[3]), 6),
            (_memory_card("Poet", refs=[2, 3]), 6),
            (_memory_card("Poet", refs=[True]), 6),
            (_memory_card("Poet", refs=["2"]), 6),
            (_memory_card("Poet", refs=[7]), 6),
            (_memory_card("Poet", refs=[2], position=0), 6),
            (_memory_card("Poet", refs=[2], position=7), 6),
            (_memory_card("Poet", refs=[2], position=True), 6),
            (_memory_card("Poet", refs=[2]), None),
        )
        stale = _memory_card("Poet", refs=[2])
        stale["acted_infos"].append(
            {"desc": "newer unrelated", "targets": [2]}
        )
        malformed += ((stale, 6),)

        for card, n_cards in malformed:
            with self.subTest(card=card, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=n_cards)
                )

    def test_rambler_interruption_wins_before_gemcrafter_parsing(self):
        clue = "#4 shut up!"
        parsed = _parse_clue_from_memory(
            _memory_card("Archivist", clue, [4], position=5),
            n_cards=6,
        )
        self.assertEqual(parsed.apparent_role, "Gemcrafter")
        self.assertEqual(parsed.info_text, clue)
        self.assertEqual(parsed.info_parsed, {"shut_up_target": 4})
        self.assertNotIn("gemcrafter_variant", parsed.info_parsed)


class GemcrafterSessionCaptureTests(unittest.TestCase):
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

    def test_auto_card_replaces_only_empty_direct_placeholder(self):
        session = GameSession(6, 1)
        session.add_card(CardInfo(1, "Gemcrafter"))
        self._run_auto_card(
            session,
            _memory_card("Archivist", refs=[2]),
        )
        self.assertEqual(session.cards[0].info_text, _CURRENT_CLUE)
        self.assertEqual(
            session.cards[0].info_parsed,
            {
                "good_position": 2,
                "gemcrafter_variant": "public_current",
            },
        )

        legacy = GameSession(6, 1)
        legacy.add_card(card_gemcrafter(1, 3))
        self._run_auto_card(
            legacy,
            _memory_card("Archivist", refs=[2]),
        )
        self.assertEqual(legacy.cards[0].info_text, "")
        self.assertEqual(legacy.cards[0].info_parsed, {"good_position": 3})

    def test_passive_reset_metadata_does_not_reactivate_gemcrafter(self):
        session = GameSession(6, 1)
        session.cards.extend([
            CardInfo(1, "Gemcrafter"),
            CardInfo(2, "Judge"),
        ])
        session.used_abilities = [1, 2]

        self.assertEqual(session.reset_after_night_abilities(), [2])
        self.assertEqual(session.used_abilities, [1])


if __name__ == "__main__":
    unittest.main()
