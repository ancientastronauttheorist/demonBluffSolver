"""Current-build native Poet/Gossip ingestion regressions."""

from contextlib import redirect_stdout
from io import StringIO
import unittest
from unittest.mock import patch

from game_loop import (
    DecisionLog,
    GameSession,
    _parse_card_cli,
    _parse_clue_from_memory,
    card_bounty_hunter,
    card_poet_with_info,
    dispatch,
)
from solver import CardInfo, POET_PROVIDER_ROLES, POET_VARIANT


def _memory_poet(
    clue: str,
    targets: list[int] | None = None,
    *,
    runtime_data: dict | None = None,
    prior_infos: list[dict] | None = None,
) -> dict:
    infos = list(prior_infos or [])
    if targets is not None:
        infos.append({"desc": clue, "targets": list(targets)})
    card = {
        "position": 1,
        "true_role": "Poet",
        "clue_text": clue,
        "acted_infos": infos,
    }
    if runtime_data is not None:
        card["runtime_data"] = dict(runtime_data)
    return card


class PoetManualIngestionTests(unittest.TestCase):
    def test_native_provider_whitelist_preserves_constructor_order(self):
        self.assertEqual(
            POET_PROVIDER_ROLES,
            (
                "Lover",
                "Scout",
                "Oracle",
                "Bounty Hunter",
                "Medium",
                "Knitter",
                "Hunter",
                "Enlightened",
                "Empress",
                "Bishop",
                "Gemcrafter",
                "Bard",
            ),
        )

    def test_every_manual_native_provider_is_canonicalized_and_stamped(self):
        cases = [
            ("lover", ["2"], "Lover"),
            ("SCOUT", ["Pooka", "2"], "Scout"),
            ("oracle", ["2,3", "Witch"], "Oracle"),
            ("bounty_hunter", ["4"], "Bounty Hunter"),
            ("medium", ["2", "Scout"], "Medium"),
            ("knitter", ["1"], "Knitter"),
            ("hunter", ["2"], "Hunter"),
            ("enlightened", ["CW"], "Enlightened"),
            ("empress", ["2,3,4"], "Empress"),
            ("bishop", ["2,3,4", "Villager,Outcast,Minion"], "Bishop"),
            ("gem_crafter", ["2"], "Gemcrafter"),
            ("bard", ["1"], "Bard"),
        ]

        for provider, args, canonical in cases:
            with self.subTest(provider=provider):
                card = card_poet_with_info(1, provider, args)
                self.assertEqual(card.apparent_role, "Poet")
                self.assertEqual(card.info_parsed["copied_role"], canonical)
                self.assertEqual(card.info_parsed["poet_variant"], POET_VARIANT)

    def test_manual_provider_payloads_are_exact_and_canonical(self):
        cases = [
            ("lover", ["2"], {"evil_adjacent": 2}),
            ("scout", ["pooka", "2"], {"evil_role": "Pooka", "distance": 2}),
            (
                "oracle",
                ["2,3", "witch"],
                {"targets": [2, 3], "minion_role": "Witch"},
            ),
            ("bounty_hunter", ["4"], {"evil_position": 4}),
            (
                "medium",
                ["2", "scout"],
                {"good_position": 2, "good_role": "Scout"},
            ),
            ("knitter", ["1"], {"evil_pairs": 1}),
            ("hunter", ["2"], {"distance": 2}),
            (
                "enlightened",
                ["counter-clockwise"],
                {"direction": "CCW"},
            ),
            ("empress", ["2,3,4"], {"targets": [2, 3, 4]}),
            (
                "bishop",
                ["2,3,4", "villager,outcast,minion"],
                {
                    "targets": [2, 3, 4],
                    "types": ["Villager", "Outcast", "Minion"],
                },
            ),
            ("gem_crafter", ["2"], {"good_position": 2}),
            ("bard", ["0"], {"corruption_distance": -1}),
        ]

        for provider, args, expected in cases:
            with self.subTest(provider=provider):
                card = card_poet_with_info(1, provider, args)
                expected = dict(expected)
                expected["copied_role"] = card.info_parsed["copied_role"]
                expected["poet_variant"] = POET_VARIANT
                self.assertEqual(card.info_parsed, expected)

    def test_manual_current_payloads_fail_early_when_not_schema_safe(self):
        cases = [
            ("lover", ["3"]),
            ("lover", ["1", "extra"]),
            ("scout", ["lover", "2"]),
            ("scout", ["pooka", "0"]),
            ("scout", ["pooka", "-1"]),
            ("oracle", ["2,2", "witch"]),
            ("oracle", ["2,3", "pooka"]),
            ("bounty_hunter", ["0"]),
            ("medium", ["2", "not_a_role"]),
            ("knitter", ["-1"]),
            ("hunter", ["-1"]),
            ("hunter", ["0"]),
            ("enlightened", ["left"]),
            ("empress", ["2,3"]),
            ("empress", ["2,2,3"]),
            ("bishop", ["2"]),
            ("bishop", ["2,3", "villager"]),
            ("bishop", ["2", "unknown"]),
            ("gemcrafter", ["0"]),
            ("bard", ["-2"]),
        ]

        for provider, args in cases:
            with self.subTest(provider=provider, args=args), self.assertRaises(
                ValueError
            ):
                card_poet_with_info(1, provider, args)

    def test_manual_current_payloads_use_live_board_bounds(self):
        cases = [
            ("bounty_hunter", ["6"]),
            ("scout", ["pooka", "6"]),
            ("oracle", ["2,6", "witch"]),
            ("medium", ["6", "scout"]),
            ("knitter", ["6"]),
            ("hunter", ["6"]),
            ("empress", ["2,3,6"]),
            ("bishop", ["6", "villager"]),
            ("gemcrafter", ["6"]),
            ("bard", ["6"]),
        ]

        for provider, args in cases:
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                card_poet_with_info(1, provider, args, n_cards=5)

        with self.assertRaises(ValueError):
            card_poet_with_info(6, "lover", ["1"], n_cards=5)

        session = GameSession(5, 1)
        with self.assertRaises(ValueError):
            _parse_card_cli(["bounty_hunter", "1", "6"], session=session)

    def test_bounty_hunter_builder_is_current_and_validates_target(self):
        card = card_bounty_hunter(1, 4)
        self.assertEqual(
            card.info_parsed,
            {
                "evil_position": 4,
                "copied_role": "Bounty Hunter",
                "poet_variant": POET_VARIANT,
            },
        )
        for invalid in (0, -1, True, "4"):
            with self.subTest(invalid=invalid), self.assertRaises(ValueError):
                card_bounty_hunter(1, invalid)

    def test_obsolete_and_unknown_manual_providers_are_rejected(self):
        for provider in (
            "Architect",
            "Confessor",
            "Baker",
            "Fortune Teller",
            "Unknown",
        ):
            with self.subTest(provider=provider), self.assertRaises(ValueError):
                card_poet_with_info(1, provider, ["1"])


class PoetMemoryIngestionTests(unittest.TestCase):
    def test_all_unambiguous_native_provider_surfaces_are_stamped(self):
        cases = [
            ("2 Evils\nadjacent to me", [], "Lover"),
            ("Pooka is 2 cards away from closest Evil", [], "Scout"),
            ("#2 or #3 is a Witch", [2, 3], "Oracle"),
            ("#4\nis Evil", [], "Bounty Hunter"),
            ("#2 is a real Scout", [2], "Medium"),
            ("There is only 1 pair of Evil", [], "Knitter"),
            ("I am 2 cards away from closest Evil", [], "Hunter"),
            ("Closest Evil is:\nCounter-clockwise", [], "Enlightened"),
            ("One is Evil:\n#2, #3 or #4", [2, 3, 4], "Empress"),
            (
                "Between\n#2, #3, #4\nthere is:\nVillager, Outcast and Minion",
                [2, 3, 4],
                "Bishop",
            ),
            ("#2 is Good", [2], "Gemcrafter"),
            ("I am 1 card away from Corrupted character", [], "Bard"),
        ]

        for clue, targets, provider in cases:
            with self.subTest(provider=provider, clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_poet(clue, targets),
                    n_cards=6,
                )
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.info_parsed["copied_role"], provider)
                self.assertEqual(
                    parsed.info_parsed["poet_variant"],
                    POET_VARIANT,
                )

    def test_enlightened_runtime_data_keeps_poet_identity_and_provenance(self):
        parsed = _parse_clue_from_memory(
            _memory_poet(
                "Closest Evil is:\nCounter-clockwise",
                [],
                runtime_data={"type": "direction", "direction": "CCW"},
            ),
            n_cards=6,
        )

        self.assertEqual(parsed.apparent_role, "Poet")
        self.assertEqual(
            parsed.info_parsed,
            {
                "direction": "CCW",
                "copied_role": "Enlightened",
                "poet_variant": POET_VARIANT,
            },
        )

        mismatched = _parse_clue_from_memory(
            _memory_poet(
                "Closest Evil is:\nCounter-clockwise",
                [],
                runtime_data={"type": "direction", "direction": "CW"},
            ),
            n_cards=6,
        )
        self.assertIsNone(mismatched)

    def test_lover_accepts_only_exact_native_count_wording(self):
        cases = [
            ("NO Evils\nadjacent to me", 0),
            ("1 Evil\nadjacent to me", 1),
            ("2 Evils\nadjacent to me", 2),
        ]
        for clue, count in cases:
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_poet(clue, []),
                    n_cards=6,
                )
                self.assertEqual(parsed.info_parsed["copied_role"], "Lover")
                self.assertEqual(parsed.info_parsed["evil_adjacent"], count)

        for unsupported in (
            "2 of my neighbors are Evil",
            "0 Evils adjacent to me",
            "3 Evils adjacent to me",
            "NO Evils adjacent to me, allegedly",
        ):
            with self.subTest(unsupported=unsupported):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(unsupported, []),
                        n_cards=6,
                    )
                )

    def test_text_only_providers_require_newest_exact_zero_ref_event(self):
        cases = [
            ("NO Evils\nadjacent to me", "Lover"),
            ("Pooka is\n2 cards away\nfrom closest Evil", "Scout"),
            ("Evils are not adjacent to eachother", "Knitter"),
            ("I am 2 cards away from closest Evil", "Hunter"),
            ("Closest Evil is equidistant", "Enlightened"),
            ("There are no Corrupted characters", "Bard"),
        ]

        for clue, provider in cases:
            with self.subTest(provider=provider, mode="valid"):
                parsed = _parse_clue_from_memory(
                    _memory_poet(clue, []),
                    n_cards=6,
                )
                self.assertEqual(parsed.info_parsed["copied_role"], provider)

            with self.subTest(provider=provider, mode="nonempty_refs"):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(clue, [2]),
                        n_cards=6,
                    )
                )

            with self.subTest(provider=provider, mode="stale_latest_desc"):
                stale = _memory_poet(clue, [])
                stale["acted_infos"][-1]["desc"] = f"{clue} stale"
                self.assertIsNone(
                    _parse_clue_from_memory(stale, n_cards=6)
                )

            with self.subTest(provider=provider, mode="trailing_text"):
                trailing = f"{clue} trailing"
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(trailing, []),
                        n_cards=6,
                    )
                )

    def test_remaining_native_singular_plural_variants_are_exact(self):
        cases = [
            ("1 Evil\nadjacent to me", "Lover", "evil_adjacent", 1),
            ("There is only 1 pair of Evil", "Knitter", "evil_pairs", 1),
            ("There are 2 pairs of Evil", "Knitter", "evil_pairs", 2),
            ("I am 1 card away from closest Evil", "Hunter", "distance", 1),
            ("Pooka is\n1 card away\nfrom closest Evil", "Scout", "distance", 1),
            (
                "I am 2 cards away from Corrupted character",
                "Bard",
                "corruption_distance",
                2,
            ),
            ("Closest Evil is:\nClockwise", "Enlightened", "direction", "CW"),
        ]
        for clue, provider, field, expected in cases:
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_poet(clue, []),
                    n_cards=6,
                )
                self.assertEqual(parsed.info_parsed["copied_role"], provider)
                self.assertEqual(parsed.info_parsed[field], expected)

        malformed_grammar = [
            "1 Evils adjacent to me",
            "2 Evil adjacent to me",
            "I am 1 cards away from closest Evil",
            "I am 2 card away from closest Evil",
            "Pooka is 1 cards away from closest Evil",
            "Pooka is 2 card away from closest Evil",
            "I am 1 cards away from Corrupted character",
            "I am 2 card away from Corrupted character",
        ]
        for clue in malformed_grammar:
            with self.subTest(malformed=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(clue, []),
                        n_cards=6,
                    )
                )

    def test_bounty_hunter_uses_latest_zero_ref_event_only(self):
        clue = "#4\nis Evil"
        parsed = _parse_clue_from_memory(
            _memory_poet(
                clue,
                [],
                prior_infos=[{"desc": "#2 is Good", "targets": [2]}],
            ),
            n_cards=6,
        )
        self.assertEqual(parsed.info_parsed["copied_role"], "Bounty Hunter")
        self.assertEqual(parsed.info_parsed["evil_position"], 4)
        self.assertEqual(parsed.info_text, clue)

        malformed = [
            _memory_poet(clue, [4]),
            _memory_poet(clue),
            {
                **_memory_poet(clue, []),
                "acted_infos": [
                    {"desc": clue, "targets": []},
                    {"desc": "#5 is Evil", "targets": []},
                ],
            },
            {
                **_memory_poet(clue, []),
                "acted_infos": [{"desc": "#4 is Evil!", "targets": []}],
            },
        ]
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=6)
                )

    def test_bishop_refs_are_a_set_and_types_are_a_multiset(self):
        clue = (
            "Between\n#2, #3, #4\nthere is:\n"
            "Villager, Villager and Minion"
        )
        parsed = _parse_clue_from_memory(
            _memory_poet(clue, [4, 2, 3]),
            n_cards=6,
        )

        self.assertEqual(parsed.info_parsed["copied_role"], "Bishop")
        self.assertEqual(parsed.info_parsed["targets"], [2, 3, 4])
        self.assertEqual(
            parsed.info_parsed["types"],
            ["Villager", "Villager", "Minion"],
        )

        for bad_clue, refs in (
            (
                "Between #3, #2, #4 there is: Villager, Outcast and Minion",
                [2, 3, 4],
            ),
            (
                "Between #2, #3, #4 there is: Villager, Outcast and Minion",
                [2, 3, 3],
            ),
            (
                "Between #2, #3, #4 there is: Villager and Minion",
                [2, 3, 4],
            ),
        ):
            with self.subTest(clue=bad_clue, refs=refs):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(bad_clue, refs),
                        n_cards=6,
                    )
                )

    def test_medium_exact_normal_and_drunk_reveal_forms(self):
        for clue, role in (
            ("#2 is a real Scout", "Scout"),
            ("#2 is actually a\nDrunk", "Drunk"),
        ):
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_poet(clue, [2]),
                    n_cards=6,
                )
                self.assertEqual(parsed.info_parsed["copied_role"], "Medium")
                self.assertEqual(parsed.info_parsed["good_position"], 2)
                self.assertEqual(parsed.info_parsed["good_role"], role)

        for clue, refs in (
            ("#2 is a real Scout trailing text", [2]),
            ("#2 is actually a Drunk", [3]),
            ("#2 is actually a Future Role", [2]),
        ):
            with self.subTest(clue=clue, refs=refs):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(clue, refs),
                        n_cards=6,
                    )
                )

    def test_obsolete_and_fortune_surfaces_are_not_current_poet_results(self):
        cases = [
            ("Left", []),
            ("I am dizzy", []),
            ("I was a Baker", []),
            ("Is #2 or #3 Evil?: True", [2, 3]),
        ]
        for clue, targets in cases:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(clue, targets),
                        n_cards=6,
                    )
                )

    def test_no_info_and_shut_up_observations_remain_unmarked(self):
        no_info = _parse_clue_from_memory(
            _memory_poet(""),
            n_cards=6,
        )
        self.assertEqual(no_info.info_parsed, {})

        shut_up_text = "#2 shut up!"
        shut_up = _parse_clue_from_memory(
            _memory_poet(shut_up_text, [2]),
            n_cards=6,
        )
        self.assertEqual(shut_up.info_parsed, {"shut_up_target": 2})
        self.assertNotIn("poet_variant", shut_up.info_parsed)

    def test_auto_card_replaces_only_an_empty_poet_placeholder(self):
        clue = "#4 is Evil"
        memory = _memory_poet(clue, [])
        memory["state"] = "Revealed"

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [memory]

            def close(self):
                return None

        session = GameSession(5, 1)
        session.add_card(CardInfo(1, "Poet"))
        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("memory_reader.print_board"),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            redirect_stdout(StringIO()),
        ):
            dispatch("auto_card", [], session)

        self.assertEqual(
            session.cards[0].info_parsed,
            {
                "evil_position": 4,
                "copied_role": "Bounty Hunter",
                "poet_variant": POET_VARIANT,
            },
        )

        legacy = GameSession(5, 1)
        legacy_payload = {"copied_role": "Architect", "side": "Left"}
        legacy.add_card(CardInfo(1, "Poet", info_parsed=legacy_payload))
        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("memory_reader.print_board"),
            patch.object(legacy, "save"),
            patch.object(DecisionLog, "log_card"),
            redirect_stdout(StringIO()),
        ):
            dispatch("auto_card", [], legacy)

        self.assertEqual(legacy.cards[0].info_parsed, legacy_payload)

    def test_exact_native_refs_are_required_for_structured_providers(self):
        mismatches = [
            ("#2 or #3 is a Witch", [2]),
            ("One is Evil: #2, #3 or #4", [2, 4, 3]),
            ("#2 is Good", [3]),
            ("#2 is a real Scout", [3]),
        ]
        for clue, targets in mismatches:
            with self.subTest(clue=clue, targets=targets):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(clue, targets),
                        n_cards=6,
                    )
                )

        # Bishop is the exception: native shuffles references independently
        # after sorting the IDs embedded in the public sentence.
        bishop = _parse_clue_from_memory(
            _memory_poet(
                "Between #2, #3 there is: Villager and Outcast",
                [3, 2],
            ),
            n_cards=6,
        )
        self.assertEqual(bishop.info_parsed["copied_role"], "Bishop")
        self.assertEqual(bishop.info_parsed["targets"], [2, 3])

    def test_anchored_structured_text_and_unsupported_sentinels_stay_manual(self):
        cases = [
            ("#2 or #3 is a Witch trailing", [2, 3]),
            ("One is Evil: #2, #3 or #4 trailing", [2, 3, 4]),
            ("#2 is Good trailing", [2]),
            ("There are NO Minions", []),
            ("There is only 1 Evil", []),
            ("There are 7 pairs of Evil", []),
            ("Pooka is 7 cards away from closest Evil", []),
            ("I am 7 cards away from closest Evil", []),
            ("I am 7 cards away from Corrupted character", []),
        ]
        for clue, targets in cases:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_poet(clue, targets),
                        n_cards=6,
                    )
                )


if __name__ == "__main__":
    unittest.main()
