"""Current-build native Lover/Empath bridge regressions."""

import unittest

from game_loop import (
    GameSession,
    _current_lover_refs,
    _parse_card_cli,
    _parse_clue_from_memory,
    card_lover,
    card_poet_with_info,
)


CURRENT = "public_current"


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


class CurrentLoverManualEntryTests(unittest.TestCase):
    def test_bare_builder_preserves_legacy_payload(self):
        card = card_lover(1, 2)
        self.assertEqual(card.info_parsed, {"evil_adjacent": 2})
        self.assertEqual(card.info_text, "")

    def test_direct_cli_stamps_exact_native_text(self):
        expected = {
            0: "NO Evils\nadjacent to me",
            1: "1 Evil\nadjacent to me",
            2: "2 Evils\nadjacent to me",
        }
        for count, text in expected.items():
            with self.subTest(count=count):
                card = _parse_card_cli(
                    ["lover", "3", str(count)],
                    session=GameSession(6, 2),
                )
                self.assertEqual(card.info_text, text)
                self.assertEqual(
                    card.info_parsed,
                    {"evil_adjacent": count, "lover_variant": CURRENT},
                )

    def test_direct_cli_rejects_unsafe_current_payloads(self):
        session = GameSession(6, 2)
        invalid = [
            ["lover", "0", "1"],
            ["lover", "7", "1"],
            ["lover", "1", "-1"],
            ["lover", "1", "3"],
            ["lover", "1", "one"],
            ["lover", "1", "1", "extra"],
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                _parse_card_cli(args, session=session)
        with self.assertRaises(ValueError):
            _parse_card_cli(["lover", "1", "1"])

    def test_poet_manual_entry_uses_the_same_exact_surface(self):
        card = card_poet_with_info(2, "lover", ["0"], n_cards=4)
        self.assertEqual(card.info_text, "NO Evils\nadjacent to me")
        self.assertEqual(
            card.info_parsed,
            {
                "evil_adjacent": 0,
                "copied_role": "Lover",
                "poet_variant": CURRENT,
            },
        )


class CurrentLoverMemoryTests(unittest.TestCase):
    def test_exact_native_results_are_stamped_with_physical_neighbors(self):
        cases = [
            ("NO Evils\nadjacent to me", 0),
            ("1 Evil\nadjacent to me", 1),
            ("2 Evils\nadjacent to me", 2),
        ]
        for clue, count in cases:
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_card(
                        "Lover",
                        3,
                        clue,
                        _current_lover_refs(3, 6),
                        prior_infos=[{"desc": "old", "targets": [1]}],
                    ),
                    n_cards=6,
                )
                self.assertEqual(parsed.info_text, clue)
                self.assertEqual(
                    parsed.info_parsed,
                    {"evil_adjacent": count, "lover_variant": CURRENT},
                )

    def test_reference_order_and_newest_event_must_match(self):
        clue = "1 Evil\nadjacent to me"
        expected = _current_lover_refs(3, 6)
        malformed = [
            None,
            [],
            list(reversed(expected)),
            expected[:1],
            expected + [4],
            [0, 4],
            [2, 7],
        ]
        for refs in malformed:
            with self.subTest(refs=refs):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Lover", 3, clue, refs),
                        n_cards=6,
                    )
                )

        stale = _memory_card("Lover", 3, clue, expected)
        stale["acted_infos"][-1]["desc"] = f"{clue} stale"
        self.assertIsNone(_parse_clue_from_memory(stale, n_cards=6))

    def test_tiny_boards_preserve_duplicate_native_references(self):
        cases = [
            (1, 1, [1, 1]),
            (1, 2, [2, 2]),
            (2, 2, [1, 1]),
        ]
        for position, n_cards, refs in cases:
            with self.subTest(position=position, n_cards=n_cards):
                self.assertEqual(_current_lover_refs(position, n_cards), refs)
                parsed = _parse_clue_from_memory(
                    _memory_card(
                        "Lover",
                        position,
                        "2 Evils\nadjacent to me",
                        refs,
                    ),
                    n_cards=n_cards,
                )
                self.assertIsNotNone(parsed)

    def test_non_native_text_and_missing_board_context_fail_closed(self):
        malformed = [
            "No Evils\nadjacent to me",
            "0 Evils\nadjacent to me",
            "1 Evils\nadjacent to me",
            "2 Evil\nadjacent to me",
            "3 Evils\nadjacent to me",
            "1 Evil adjacent to me",
            "1 Evil\nadjacent to me.",
            "1 of my neighbors is Evil",
        ]
        refs = _current_lover_refs(3, 6)
        for clue in malformed:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Lover", 3, clue, refs),
                        n_cards=6,
                    )
                )

        valid = _memory_card(
            "Lover",
            3,
            "1 Evil\nadjacent to me",
            refs,
        )
        self.assertIsNone(_parse_clue_from_memory(valid))


if __name__ == "__main__":
    unittest.main()
