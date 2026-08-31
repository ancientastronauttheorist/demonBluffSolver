"""Current-build native Confessor reader and bridge regressions."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

import memory_reader as memory_module
from game_loop import (
    DecisionLog,
    GameSession,
    _canonical_confessor_claim,
    _confessor_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    _parse_confessor_native_text,
    card_confessor,
    dispatch,
)
from solver import BAKER_RULE_VERSION, CardInfo


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
    **extra,
) -> dict:
    infos = list(prior_infos or [])
    if refs is not _ABSENT:
        infos.append({"desc": clue, "targets": refs})
    card = {
        "position": position,
        "true_role": role,
        "clue_text": clue,
        "acted_infos": infos,
        **extra,
    }
    if current_role is not _ABSENT:
        card["current_role"] = current_role
    if disguise is not _ABSENT:
        card["disguise"] = disguise
    if runtime_data is not _ABSENT:
        card["runtime_data"] = runtime_data
    return card


class ConfessorConstructorTests(unittest.TestCase):
    def test_exact_native_builder_and_parser(self):
        for dizzy, clue in ((False, "I am Good"), (True, "I am dizzy")):
            with self.subTest(dizzy=dizzy):
                self.assertEqual(_confessor_native_text(dizzy), clue)
                self.assertIs(_parse_confessor_native_text(clue), dizzy)

        for invalid in (0, 1, None, "true"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                _confessor_native_text(invalid)

    def test_exact_parser_rejects_all_loose_lookalikes(self):
        mutations = (
            "i am Good",
            "I Am Good",
            "I am good",
            "I am  Good",
            "I am Good ",
            " I am Good",
            "I am Good.",
            "I am Good!",
            "I am\nGood",
            "I am Good trailing",
            "I am Dizzy",
            "I am  dizzy",
            "I am dizzy ",
            " I am dizzy",
            "I am dizzy.",
            "I am dizzy!",
            "I am\ndizzy",
            "not dizzy at all",
            "good grief",
            "cleanliness",
            "dirty",
            "",
            None,
        )
        for clue in mutations:
            with self.subTest(clue=clue):
                self.assertIsNone(_parse_confessor_native_text(clue))

    def test_unmarked_constructor_and_archive_shapes_remain_unchanged(self):
        for dizzy in (False, True):
            with self.subTest(dizzy=dizzy):
                legacy = card_confessor(1, dizzy)
                self.assertEqual(legacy.info_text, "")
                self.assertEqual(legacy.info_parsed, {"dizzy": dizzy})
                self.assertNotIn("confessor_variant", legacy.info_parsed)

        case_root = Path(__file__).parent
        empty_case = json.loads(
            (case_root / "cases_v2" / "asc35_v2.json").read_text(
                encoding="utf-8"
            )
        )
        empty = next(
            card
            for card in empty_case["cards"]
            if card["apparent_role"] == "Confessor"
        )
        baker_case = json.loads(
            (case_root / "cases" / "asc13_v4.json").read_text(
                encoding="utf-8"
            )
        )
        baker = next(
            card
            for card in baker_case["cards"]
            if card.get("info_parsed", {}).get("original_role") == "Confessor"
        )
        self.assertEqual(empty["info_parsed"], {})
        self.assertEqual(baker["apparent_role"], "Baker")
        self.assertEqual(baker["info_parsed"], {"original_role": "Confessor"})

        dirty = CardInfo(1, "Confessor", info_parsed={"dirty": True})
        missing = CardInfo(1, "Confessor")
        self.assertEqual(dirty.info_parsed, {"dirty": True})
        self.assertEqual(missing.info_parsed, {})

    def test_marked_constructor_has_exact_schema_and_rejects_malformed_values(self):
        for dizzy in (False, True):
            with self.subTest(dizzy=dizzy):
                current = card_confessor(
                    2,
                    dizzy,
                    confessor_variant="public_current",
                )
                self.assertEqual(current.info_text, _confessor_native_text(dizzy))
                self.assertEqual(
                    current.info_parsed,
                    {
                        "dizzy": dizzy,
                        "confessor_variant": "public_current",
                    },
                )

        for position, dizzy in (
            (0, True),
            (True, True),
            ("1", True),
            (1, 0),
            (1, 1),
            (1, "true"),
        ):
            with self.subTest(position=position, dizzy=dizzy), self.assertRaises(
                ValueError
            ):
                card_confessor(
                    position,
                    dizzy,
                    confessor_variant="public_current",
                )

        with self.assertRaises(ValueError):
            card_confessor(1, True, confessor_variant="future")
        with self.assertRaisesRegex(ValueError, "must match"):
            card_confessor(
                1,
                True,
                info_text="I am Good",
                confessor_variant="public_current",
            )
        with self.assertRaises(ValueError):
            card_confessor(
                1,
                True,
                info_text=None,
                confessor_variant="public_current",
            )

    def test_manual_claim_canonicalizer_is_explicit_not_false_by_default(self):
        for token in ("dizzy", "dirty", "true", "1", "yes"):
            with self.subTest(token=token):
                self.assertTrue(_canonical_confessor_claim(token))
        for token in ("Good", "clean", "false", "0", "no"):
            with self.subTest(token=token):
                self.assertFalse(_canonical_confessor_claim(token))
        for token in ("unknown", "not dizzy", "good grief", "", None):
            with self.subTest(token=token), self.assertRaises(ValueError):
                _canonical_confessor_claim(token)

    def test_memory_reader_preserves_null_vs_populated_empty_reference_list(self):
        def read_history(character_refs_pointer: int | None):
            reader = memory_module.MemoryReader()
            character = 0x10000
            acted_list = 0x20000
            items = 0x30000
            info = 0x40000
            desc = 0x50000
            pointers = {
                character + memory_module.CHAR_ACTED_INFOS_OFFSET: acted_list,
                acted_list + memory_module.LIST_ITEMS_OFFSET: items,
                items + memory_module.ARRAY_FIRST_ELEMENT_OFFSET: info,
                info + memory_module.ACTED_INFO_DESC_OFFSET: desc,
                info + memory_module.ACTED_INFO_CHARS_OFFSET: character_refs_pointer,
            }
            if character_refs_pointer:
                pointers[
                    character_refs_pointer + memory_module.LIST_ITEMS_OFFSET
                ] = 0x70000
            reader._read_ptr = lambda address: pointers.get(address, 0)
            reader._read_i32 = lambda address: (
                1
                if address == acted_list + memory_module.LIST_SIZE_OFFSET
                else 0
            )
            reader._read_string = lambda pointer: "I am Good"
            return reader._read_acted_infos(character)

        null_refs = read_history(0)
        empty_refs = read_history(0x60000)
        unreadable_refs = read_history(None)
        self.assertIsNone(null_refs[0]["targets"])
        self.assertEqual(empty_refs[0]["targets"], [])
        self.assertEqual(unreadable_refs, [])


class ConfessorManualIngestionTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession(6, 1)

    def test_direct_cli_synthesizes_exact_current_text_and_schema(self):
        for token, dizzy in (("Good", False), ("dizzy", True), ("clean", False)):
            with self.subTest(token=token):
                card = _parse_card_cli(
                    ["confessor", "2", token],
                    self.session,
                )
                self.assertEqual(card.info_text, _confessor_native_text(dizzy))
                self.assertEqual(
                    card.info_parsed,
                    {
                        "dizzy": dizzy,
                        "confessor_variant": "public_current",
                    },
                )

    def test_direct_cli_requires_session_actor_exact_arity_and_known_result(self):
        invalid = (
            (["confessor", "1", "Good"], None),
            (["confessor", "1"], self.session),
            (["confessor", "1", "Good", "extra"], self.session),
            (["confessor", "0", "Good"], self.session),
            (["confessor", "7", "Good"], self.session),
            (["confessor", "1", "unknown"], self.session),
            (["confessor", "1", "not-dizzy"], self.session),
        )
        for args, session in invalid:
            with self.subTest(args=args), self.assertRaises((IndexError, ValueError)):
                _parse_card_cli(args, session)


class ConfessorMemoryIngestionTests(unittest.TestCase):
    def _assert_current(self, parsed: CardInfo, dizzy: bool) -> None:
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.apparent_role, "Confessor")
        self.assertEqual(parsed.info_text, _confessor_native_text(dizzy))
        self.assertEqual(
            parsed.info_parsed,
            {
                "dizzy": dizzy,
                "confessor_variant": "public_current",
            },
        )

    def test_direct_accepts_both_exact_native_results(self):
        for dizzy in (False, True):
            with self.subTest(dizzy=dizzy):
                parsed = _parse_clue_from_memory(
                    _memory_card(
                        "Confessor",
                        _confessor_native_text(dizzy),
                        None,
                    ),
                    n_cards=6,
                )
                self._assert_current(parsed, dizzy)

    def test_direct_requires_newest_coherent_event_with_native_null_refs(self):
        clue = "I am Good"
        parsed = _parse_clue_from_memory(
            _memory_card(
                "Confessor",
                clue,
                None,
                prior_infos=[{"desc": "I am dizzy", "targets": None}],
            ),
            n_cards=6,
        )
        self._assert_current(parsed, False)

        malformed = (
            _memory_card("Confessor", clue),
            _memory_card("Confessor", clue, []),
            _memory_card("Confessor", clue, ()),
            _memory_card("Confessor", clue, [1]),
            _memory_card("Confessor", clue, True),
            {
                **_memory_card("Confessor", clue, None),
                "acted_infos": [{"desc": clue}],
            },
            {
                **_memory_card("Confessor", clue, None),
                "acted_infos": [None],
            },
            {
                **_memory_card("Confessor", clue, None),
                "acted_infos": "not-a-list",
            },
        )
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

        stale = _memory_card("Confessor", clue, None)
        stale["acted_infos"].append(
            {"desc": "I am dizzy", "targets": None}
        )
        self.assertIsNone(_parse_clue_from_memory(stale, n_cards=6))

    def test_direct_rejects_every_text_mutation_even_with_null_refs(self):
        mutations = (
            "i am Good",
            "I am good",
            "I am Good ",
            "I am Good.",
            "I am\nGood",
            "good grief",
            "I am Dizzy",
            "I am dizzy ",
            "I am dizzy.",
            "not dizzy at all",
            "dirty",
        )
        for clue in mutations:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Confessor", clue, None),
                        n_cards=6,
                    )
                )

    def test_direct_requires_board_actor_and_preserves_display_precedence(self):
        clue = "I am Good"
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
                            "Confessor",
                            clue,
                            None,
                            position=position,
                        ),
                        n_cards=n_cards,
                    )
                )

        accepted = (
            _memory_card(
                "Spy",
                clue,
                None,
                current_role="Confessor",
            ),
            _memory_card(
                "Spy",
                clue,
                None,
                current_role="Spy",
                disguise="Confessor",
            ),
        )
        for card in accepted:
            with self.subTest(card=card):
                self._assert_current(
                    _parse_clue_from_memory(card, n_cards=6),
                    False,
                )

        rejected = (
            _memory_card(
                "Confessor",
                clue,
                None,
                current_role="Confessor",
                disguise="Spy",
            ),
            _memory_card(
                "Confessor",
                clue,
                None,
                current_role="Spy",
            ),
        )
        for card in rejected:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

    def test_hidden_alignment_status_runtime_and_state_never_infer_or_override(self):
        cases = (
            (
                False,
                {
                    "alignment": "Evil",
                    "is_evil": True,
                    "statuses": ["Corrupted", "AppearLying"],
                    "state": "Dead",
                },
            ),
            (
                True,
                {
                    "alignment": "Good",
                    "is_evil": False,
                    "statuses": ["AppearTruthful", "HealthyBluff"],
                    "state": "Hidden",
                },
            ),
        )
        for dizzy, hidden in cases:
            with self.subTest(dizzy=dizzy):
                card = _memory_card(
                    "Confessor",
                    _confessor_native_text(dizzy),
                    None,
                    runtime_data={"type": "direction", "direction": "CCW"},
                    **hidden,
                )
                self._assert_current(
                    _parse_clue_from_memory(card, n_cards=6),
                    dizzy,
                )

        no_public_event = _memory_card(
            "Confessor",
            "",
            alignment="Evil",
            is_evil=True,
            statuses=["Corrupted", "AppearTruthful"],
        )
        placeholder = _parse_clue_from_memory(no_public_event, n_cards=6)
        self.assertEqual(placeholder.info_parsed, {})
        self.assertNotIn("confessor_variant", placeholder.info_parsed)

    def test_rambler_and_baker_precedence_remain_unmarked(self):
        interrupted = _parse_clue_from_memory(
            _memory_card("Confessor", "#4 shut up!", [4], position=2),
            n_cards=6,
        )
        self.assertEqual(interrupted.apparent_role, "Confessor")
        self.assertEqual(interrupted.info_parsed, {"shut_up_target": 4})
        self.assertNotIn("confessor_variant", interrupted.info_parsed)

        baker = _parse_clue_from_memory(
            _memory_card("Baker", "I was a Confessor", [], position=2),
            n_cards=6,
            baker_rule_version=BAKER_RULE_VERSION,
        )
        self.assertEqual(baker.apparent_role, "Baker")
        self.assertEqual(baker.info_parsed, {"original_role": "Confessor"})
        self.assertNotIn("confessor_variant", baker.info_parsed)


class ConfessorSessionCaptureTests(unittest.TestCase):
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

    def test_auto_card_replaces_only_empty_same_role_placeholder(self):
        memory = _memory_card("Confessor", "I am dizzy", None)

        current = GameSession(6, 1)
        current.add_card(CardInfo(1, "Confessor"))
        self._run_auto_card(current, memory)
        self.assertEqual(current.cards[0].info_text, "I am dizzy")
        self.assertEqual(
            current.cards[0].info_parsed,
            {"dizzy": True, "confessor_variant": "public_current"},
        )

        legacy = GameSession(6, 1)
        legacy.add_card(card_confessor(1, False))
        self._run_auto_card(legacy, memory)
        self.assertEqual(legacy.cards[0].info_text, "")
        self.assertEqual(legacy.cards[0].info_parsed, {"dizzy": False})


if __name__ == "__main__":
    unittest.main()
