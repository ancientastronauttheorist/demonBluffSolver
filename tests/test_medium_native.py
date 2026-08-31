"""Current-build native Medium/Lookout bridge regressions."""

import unittest

from game_loop import (
    GameSession,
    _medium_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    card_medium,
)
from solver import CardInfo


def _memory_medium(
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
        "true_role": "Medium",
        "clue_text": clue,
        "acted_infos": infos,
    }


class MediumConstructorTests(unittest.TestCase):
    def test_exact_native_text_uses_only_the_drunk_special_branch(self):
        self.assertEqual(_medium_native_text(2, "Scout"), "#2 is a real\nScout")
        self.assertEqual(
            _medium_native_text(2, "Drunk"),
            "#2 is actually a\nDrunk",
        )

    def test_unmarked_constructor_preserves_legacy_defaults(self):
        card = card_medium(1, 2, "legacy role spelling")

        self.assertEqual(card.info_text, "")
        self.assertEqual(
            card.info_parsed,
            {"good_position": 2, "good_role": "legacy role spelling"},
        )

    def test_marked_constructor_synthesizes_exact_text(self):
        card = card_medium(
            1,
            2,
            "Scout",
            medium_variant="public_current",
        )

        self.assertEqual(card.info_text, "#2 is a real\nScout")
        self.assertEqual(card.info_parsed["medium_variant"], "public_current")


class MediumManualIngestionTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession(3, 1)

    def test_manual_current_entry_is_canonical_stamped_and_exact(self):
        scout = _parse_card_cli(
            ["medium", "1", "2", "scout"],
            self.session,
        )
        drunk = _parse_card_cli(
            ["medium", "1", "2", "drunk"],
            self.session,
        )

        self.assertEqual(
            scout.info_parsed,
            {
                "good_position": 2,
                "good_role": "Scout",
                "medium_variant": "public_current",
            },
        )
        self.assertEqual(scout.info_text, "#2 is a real\nScout")
        self.assertEqual(drunk.info_text, "#2 is actually a\nDrunk")

    def test_real_resolves_only_an_existing_canonical_target(self):
        self.session.cards = [CardInfo(2, "Scout", info_parsed={})]

        card = _parse_card_cli(
            ["medium", "1", "2", "real"],
            self.session,
        )

        self.assertEqual(card.info_parsed["good_role"], "Scout")
        self.assertEqual(card.info_text, "#2 is a real\nScout")

    def test_manual_current_entry_rejects_missing_context_and_bad_schema(self):
        invalid = (
            (["medium", "1", "2"], self.session),
            (["medium", "1", "2", "scout", "extra"], self.session),
            (["medium", "0", "2", "scout"], self.session),
            (["medium", "4", "2", "scout"], self.session),
            (["medium", "1", "0", "scout"], self.session),
            (["medium", "1", "4", "scout"], self.session),
            (["medium", "1", "2", "future role"], self.session),
            (["medium", "1", "2", "scout"], None),
        )
        for args, session in invalid:
            with self.subTest(args=args, session=session), self.assertRaises(
                ValueError
            ):
                _parse_card_cli(args, session)

    def test_real_hard_fails_when_target_is_missing_or_unknown(self):
        with self.assertRaisesRegex(ValueError, "no current card entry"):
            _parse_card_cli(["medium", "1", "2", "real"], self.session)

        self.session.cards = [CardInfo(2, "Unknown", info_parsed={})]
        with self.assertRaisesRegex(ValueError, "must be canonical"):
            _parse_card_cli(["medium", "1", "2", "real"], self.session)


class MediumMemoryIngestionTests(unittest.TestCase):
    def test_exact_normal_and_drunk_results_are_current_and_canonical(self):
        for clue, role in (
            ("#2 is a real\nScout", "Scout"),
            ("#2 is actually a\nDrunk", "Drunk"),
        ):
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_medium(clue, [2]),
                    n_cards=3,
                )

                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.info_text, clue)
                self.assertEqual(
                    parsed.info_parsed,
                    {
                        "good_position": 2,
                        "good_role": role,
                        "medium_variant": "public_current",
                    },
                )

    def test_non_native_text_surfaces_are_rejected(self):
        invalid = (
            "#2 is a real Scout",
            "#2 is a real  \nScout",
            "#2 is a real\r\nScout",
            "#2 is a real\nscout",
            "#2 is a real\nScout!",
            "#2 is a real\nScout ",
            " #2 is a real\nScout",
            "# 2 is a real\nScout",
            "#02 is a real\nScout",
            "#2 Is a real\nScout",
            "#2 is a real\nDrunk",
            "#2 is actually a\nScout",
            "#2 is a real\nFuture Role",
        )
        for clue in invalid:
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_medium(clue, [2]),
                        n_cards=3,
                    )
                )

    def test_only_the_newest_exact_one_ref_event_is_accepted(self):
        clue = "#2 is a real\nScout"
        malformed = (
            _memory_medium(clue, None),
            _memory_medium(clue, []),
            _memory_medium(clue, [3]),
            _memory_medium(clue, [2, 3]),
            {
                **_memory_medium(clue, [2]),
                "acted_infos": [{"desc": clue, "targets": ["2"]}],
            },
            {
                **_memory_medium(clue, [2]),
                "acted_infos": [
                    {"desc": clue, "targets": [2]},
                    {"desc": "stale result", "targets": [2]},
                ],
            },
        )
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=3)
                )

    def test_actor_target_and_board_context_must_be_current(self):
        clue = "#2 is a real\nScout"
        invalid = (
            (_memory_medium(clue, [2], position=0), 3),
            (_memory_medium(clue, [2], position=4), 3),
            (_memory_medium("#4 is a real\nScout", [4]), 3),
            (_memory_medium(clue, [2]), None),
        )
        for card, n_cards in invalid:
            with self.subTest(card=card, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(card, n_cards=n_cards)
                )

    def test_current_marker_preserves_night_death_role_side_effect(self):
        clue = "#2 is a real\nScout"
        parsed = _parse_clue_from_memory(
            _memory_medium(clue, [2]),
            n_cards=3,
        )
        session = GameSession(3, 1)
        session.night_kills = [2]

        session.add_card(parsed)

        target = next(card for card in session.cards if card.position == 2)
        self.assertEqual(target.apparent_role, "Scout")
        self.assertEqual(session.revealed_night_current_roles, {2: "Scout"})


if __name__ == "__main__":
    unittest.main()
