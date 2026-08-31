"""Current-build native Scout/Hunter bridge regressions."""

import unittest

from game_loop import (
    GameSession,
    _parse_card_cli,
    _parse_clue_from_memory,
    card_hunter,
    card_scout,
)


CURRENT = "public_current"


def _refs(position: int, distance: int, n_cards: int) -> list[int]:
    if distance == 0:
        return []
    return [
        ((position - 1 + distance) % n_cards) + 1,
        ((position - 1 - distance) % n_cards) + 1,
    ]


def _memory_card(
    role: str,
    position: int,
    clue: str,
    refs: list[int] | None,
    *,
    prior_infos: list[dict] | None = None,
) -> dict:
    infos = list(prior_infos or [])
    if refs is not None:
        infos.append({"desc": clue, "targets": list(refs)})
    return {
        "position": position,
        "true_role": role,
        "clue_text": clue,
        "acted_infos": infos,
    }


class CurrentManualEntryTests(unittest.TestCase):
    def test_bare_builders_keep_legacy_payloads(self):
        self.assertEqual(
            card_scout(1, "Pooka", 2).info_parsed,
            {"evil_role": "Pooka", "distance": 2},
        )
        self.assertEqual(
            card_hunter(1, 2).info_parsed,
            {"distance": 2},
        )

    def test_direct_cli_canonicalizes_and_stamps_current_payloads(self):
        session = GameSession(6, 2)

        scout = _parse_card_cli(
            ["scout", "1", "lover", "3"],
            session=session,
        )
        self.assertEqual(
            scout.info_parsed,
            {
                "evil_role": "Lover",
                "distance": 3,
                "scout_variant": CURRENT,
            },
        )

        sentinel = _parse_card_cli(
            ["scout", "1", "one_evil"],
            session=session,
        )
        self.assertEqual(
            sentinel.info_parsed,
            {"one_evil": True, "scout_variant": CURRENT},
        )

        hunter = _parse_card_cli(
            ["hunter", "2", "5"],
            session=session,
        )
        self.assertEqual(
            hunter.info_parsed,
            {"distance": 5, "hunter_variant": CURRENT},
        )

    def test_direct_cli_enforces_native_distance_unions(self):
        six = GameSession(6, 2)
        for distance in (1, 2, 3):
            with self.subTest(role="Scout", distance=distance):
                parsed = _parse_card_cli(
                    ["scout", "1", "Pooka", str(distance)],
                    session=six,
                )
                self.assertEqual(parsed.info_parsed["distance"], distance)
        for distance in (0, 4, 5, 6):
            with self.subTest(role="Scout", distance=distance), self.assertRaises(
                ValueError
            ):
                _parse_card_cli(
                    ["scout", "1", "Pooka", str(distance)],
                    session=six,
                )

        for distance in (1, 2, 3, 5):
            with self.subTest(role="Hunter", distance=distance):
                parsed = _parse_card_cli(
                    ["hunter", "1", str(distance)],
                    session=six,
                )
                self.assertEqual(parsed.info_parsed["distance"], distance)
        for distance in (0, 4, 6):
            with self.subTest(role="Hunter", distance=distance), self.assertRaises(
                ValueError
            ):
                _parse_card_cli(
                    ["hunter", "1", str(distance)],
                    session=six,
                )

        one = GameSession(1, 1)
        self.assertEqual(
            _parse_card_cli(["hunter", "1", "0"], session=one).info_parsed,
            {"distance": 0, "hunter_variant": CURRENT},
        )
        with self.assertRaises(ValueError):
            _parse_card_cli(["hunter", "1", "1"], session=one)

    def test_direct_cli_rejects_unknown_roles_and_missing_board_context(self):
        session = GameSession(6, 2)
        with self.assertRaises(ValueError):
            _parse_card_cli(
                ["scout", "1", "not_a_role", "2"],
                session=session,
            )
        with self.assertRaises(ValueError):
            _parse_card_cli(["scout", "1", "Pooka", "2"])
        with self.assertRaises(ValueError):
            _parse_card_cli(["hunter", "1", "2"])


class CurrentScoutMemoryTests(unittest.TestCase):
    def test_numeric_and_sentinel_forms_are_canonical_and_stamped(self):
        clue = "Lover is 3 cards away from closest Evil"
        parsed = _parse_clue_from_memory(
            _memory_card("Scout", 2, clue, []),
            n_cards=6,
        )
        self.assertEqual(parsed.info_text, clue)
        self.assertEqual(
            parsed.info_parsed,
            {
                "evil_role": "Lover",
                "distance": 3,
                "scout_variant": CURRENT,
            },
        )

        sentinel_text = "There is only 1 Evil"
        sentinel = _parse_clue_from_memory(
            _memory_card("Scout", 2, sentinel_text, []),
            n_cards=6,
        )
        self.assertEqual(
            sentinel.info_parsed,
            {"one_evil": True, "scout_variant": CURRENT},
        )

    def test_numeric_union_includes_bluff_three_and_truthful_half_circle(self):
        three_on_four = "Pooka is 3 cards away from closest Evil"
        self.assertIsNotNone(
            _parse_clue_from_memory(
                _memory_card("Scout", 1, three_on_four, []),
                n_cards=4,
            )
        )

        four_on_eight = "Pooka is 4 cards away from closest Evil"
        self.assertIsNotNone(
            _parse_clue_from_memory(
                _memory_card("Scout", 1, four_on_eight, []),
                n_cards=8,
            )
        )

        four_on_six = "Pooka is 4 cards away from closest Evil"
        self.assertIsNone(
            _parse_clue_from_memory(
                _memory_card("Scout", 1, four_on_six, []),
                n_cards=6,
            )
        )

    def test_requires_newest_coherent_zero_ref_event(self):
        clue = "Pooka is 2 cards away from closest Evil"
        prior = [{"desc": "old", "targets": [3]}]
        self.assertIsNotNone(
            _parse_clue_from_memory(
                _memory_card("Scout", 1, clue, [], prior_infos=prior),
                n_cards=6,
            )
        )

        malformed = [
            _memory_card("Scout", 1, clue, None),
            _memory_card("Scout", 1, clue, [2]),
            {
                **_memory_card("Scout", 1, clue, []),
                "acted_infos": [{"desc": f"{clue} stale", "targets": []}],
            },
        ]
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=6)
                )

    def test_rejects_non_native_grammar_and_unknown_roles(self):
        invalid = [
            "Pooka is 1 cards away from closest Evil",
            "Pooka is 2 card away from closest Evil",
            "Pooka is 2 cards away from closest Evil trailing",
            "Pooka is 2 cards away from closest Evil.",
            "Unknown Future Role is 2 cards away from closest Evil",
            "There is only 1 Evil!",
        ]
        for clue in invalid:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Scout", 1, clue, []),
                        n_cards=6,
                    )
                )


class CurrentHunterMemoryTests(unittest.TestCase):
    def test_native_refs_are_forward_then_backward_and_keep_duplicates(self):
        cases = [
            (1, "I am 1 card away from closest Evil", [2, 6]),
            (3, "I am 3 cards away from closest Evil", [4, 4]),
            (5, "I am 5 cards away from closest Evil", [6, 2]),
        ]
        for distance, clue, expected_refs in cases:
            with self.subTest(distance=distance):
                self.assertEqual(_refs(1, distance, 6), expected_refs)
                parsed = _parse_clue_from_memory(
                    _memory_card("Hunter", 1, clue, expected_refs),
                    n_cards=6,
                )
                self.assertEqual(
                    parsed.info_parsed,
                    {"distance": distance, "hunter_variant": CURRENT},
                )
                self.assertEqual(parsed.info_text, clue)

    def test_single_card_board_uses_zero_and_no_refs(self):
        clue = "I am 0 cards away from closest Evil"
        parsed = _parse_clue_from_memory(
            _memory_card("Hunter", 1, clue, []),
            n_cards=1,
        )
        self.assertEqual(
            parsed.info_parsed,
            {"distance": 0, "hunter_variant": CURRENT},
        )

        wrong_plurality = "I am 0 card away from closest Evil"
        self.assertIsNone(
            _parse_clue_from_memory(
                _memory_card("Hunter", 1, wrong_plurality, []),
                n_cards=1,
            )
        )

    def test_rejects_wrong_ref_shape_order_and_unreachable_distance(self):
        clue = "I am 2 cards away from closest Evil"
        expected = _refs(1, 2, 6)
        for refs in (None, [], expected[:1], list(reversed(expected)), expected + [2]):
            with self.subTest(refs=refs):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Hunter", 1, clue, refs),
                        n_cards=6,
                    )
                )

        unreachable = "I am 4 cards away from closest Evil"
        self.assertIsNone(
            _parse_clue_from_memory(
                _memory_card("Hunter", 1, unreachable, _refs(1, 4, 6)),
                n_cards=6,
            )
        )

    def test_rejects_non_native_grammar_and_stale_event(self):
        invalid = [
            "I am 1 cards away from closest Evil",
            "I am 2 card away from closest Evil",
            "I am 2 cards away from closest Evil trailing",
            "I am 2 cards away from closest Evil.",
            "nearest evil is 2 away",
        ]
        for clue in invalid:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Hunter", 1, clue, _refs(1, 2, 6)),
                        n_cards=6,
                    )
                )

        clue = "I am 2 cards away from closest Evil"
        stale = _memory_card("Hunter", 1, clue, _refs(1, 2, 6))
        stale["acted_infos"][-1]["desc"] = f"{clue} stale"
        self.assertIsNone(_parse_clue_from_memory(stale, n_cards=6))


if __name__ == "__main__":
    unittest.main()
