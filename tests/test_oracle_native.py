"""Current-build native Oracle bridge regressions."""

import unittest

from game_loop import (
    GameSession,
    _parse_card_cli,
    _parse_clue_from_memory,
    card_oracle,
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


class CurrentOracleManualEntryTests(unittest.TestCase):
    def test_bare_builder_keeps_legacy_payload(self):
        card = card_oracle(1, [2, 3], "Witch")
        self.assertEqual(
            card.info_parsed,
            {"targets": [2, 3], "minion_role": "Witch"},
        )
        self.assertEqual(card.info_text, "")

    def test_direct_cli_canonicalizes_and_stamps_positive_and_sentinel(self):
        session = GameSession(6, 2)
        positive = _parse_card_cli(
            ["oracle", "1", "2,3", "twin_minion"],
            session=session,
        )
        self.assertEqual(
            positive.info_parsed,
            {
                "targets": [2, 3],
                "minion_role": "Twin Minion",
                "oracle_variant": CURRENT,
            },
        )
        self.assertEqual(positive.info_text, "#2 or #3 is a Twin Minion")

        sentinel = _parse_card_cli(
            ["oracle", "1", "no_minions"],
            session=session,
        )
        self.assertEqual(
            sentinel.info_parsed,
            {"no_minions": True, "oracle_variant": CURRENT},
        )
        self.assertEqual(sentinel.info_text, "There are no minions")

    def test_direct_cli_allows_native_duplicate_references(self):
        parsed = _parse_card_cli(
            ["oracle", "2", "2,2", "Witch"],
            session=GameSession(6, 2),
        )
        self.assertEqual(parsed.info_parsed["targets"], [2, 2])
        self.assertEqual(parsed.info_text, "#2 or #2 is a Witch")

    def test_direct_cli_rejects_unsafe_schema_and_roles(self):
        session = GameSession(6, 2)
        invalid = [
            ["oracle", "0", "2,3", "Witch"],
            ["oracle", "7", "2,3", "Witch"],
            ["oracle", "1", "2", "Witch"],
            ["oracle", "1", "2,3,4", "Witch"],
            ["oracle", "1", "3,2", "Witch"],
            ["oracle", "1", "0,2", "Witch"],
            ["oracle", "1", "2,7", "Witch"],
            ["oracle", "1", "two,3", "Witch"],
            ["oracle", "1", "2,3", "Pooka"],
            ["oracle", "1", "2,3", "Lover"],
            ["oracle", "1", "2,3", "not_a_role"],
        ]
        for args in invalid:
            with self.subTest(args=args), self.assertRaises(ValueError):
                _parse_card_cli(args, session=session)

        with self.assertRaises(ValueError):
            _parse_card_cli(["oracle", "1", "2,3", "Witch"])


class CurrentOracleMemoryTests(unittest.TestCase):
    def test_positive_and_sentinel_are_exact_and_stamped(self):
        clue = "#2 or #3 is a Witch"
        positive = _parse_clue_from_memory(
            _memory_card("Oracle", 1, clue, [2, 3]),
            n_cards=6,
        )
        self.assertEqual(positive.info_text, clue)
        self.assertEqual(
            positive.info_parsed,
            {
                "targets": [2, 3],
                "minion_role": "Witch",
                "oracle_variant": CURRENT,
            },
        )

        sentinel_text = "There are no minions"
        sentinel = _parse_clue_from_memory(
            _memory_card("Oracle", 1, sentinel_text, []),
            n_cards=6,
        )
        self.assertEqual(sentinel.info_text, sentinel_text)
        self.assertEqual(
            sentinel.info_parsed,
            {"no_minions": True, "oracle_variant": CURRENT},
        )

    def test_positive_accepts_self_and_duplicate_references(self):
        for position, clue, refs in (
            (2, "#2 or #4 is a Witch", [2, 4]),
            (2, "#2 or #2 is a Witch", [2, 2]),
        ):
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_card("Oracle", position, clue, refs),
                    n_cards=6,
                )
                self.assertIsNotNone(parsed)
                self.assertEqual(parsed.info_parsed["targets"], refs)

    def test_requires_newest_coherent_event_and_exact_references(self):
        clue = "#2 or #3 is a Witch"
        prior = [{"desc": "old", "targets": [5, 6]}]
        self.assertIsNotNone(
            _parse_clue_from_memory(
                _memory_card("Oracle", 1, clue, [2, 3], prior_infos=prior),
                n_cards=6,
            )
        )

        malformed = [
            _memory_card("Oracle", 1, clue, None),
            _memory_card("Oracle", 1, clue, []),
            _memory_card("Oracle", 1, clue, [2]),
            _memory_card("Oracle", 1, clue, [3, 2]),
            _memory_card("Oracle", 1, clue, [2, 3, 4]),
            _memory_card("Oracle", 1, clue, [2, 7]),
            {
                **_memory_card("Oracle", 1, clue, [2, 3]),
                "acted_infos": [{"desc": f"{clue} stale", "targets": [2, 3]}],
            },
            {
                **_memory_card("Oracle", 1, clue, [2, 3]),
                "acted_infos": [{"desc": clue, "targets": [2, "3"]}],
            },
        ]
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

        self.assertIsNone(
            _parse_clue_from_memory(
                _memory_card("Oracle", 1, clue, [2, 3]),
            )
        )

    def test_sentinel_requires_exact_text_and_zero_references(self):
        for refs in ([1], [1, 2]):
            with self.subTest(refs=refs):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Oracle", 1, "There are no minions", refs),
                        n_cards=6,
                    )
                )

        for clue in (
            "There are NO minions",
            "There are no minions.",
            "there are no minions",
            "There are no minions trailing",
        ):
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Oracle", 1, clue, []),
                        n_cards=6,
                    )
                )

    def test_positive_requires_exact_canonical_native_sentence(self):
        invalid = [
            "#3 or #2 is a Witch",
            "#2 or #3 is a witch",
            "#2 or #3 is a Witch.",
            "#2 or #3 is a Pooka",
            "#2 or #3 is a Lover",
            "#2 or #3 is a Future Minion",
            " #2 or #3 is a Witch",
        ]
        for clue in invalid:
            refs = [3, 2] if clue.startswith("#3") else [2, 3]
            with self.subTest(clue=clue):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_card("Oracle", 1, clue, refs),
                        n_cards=6,
                    )
                )


class CurrentPoetOracleMemoryTests(unittest.TestCase):
    def test_positive_sentinel_and_duplicates_use_oracle_provider(self):
        cases = [
            ("#2 or #3 is a Witch", [2, 3], "Witch"),
            ("#2 or #2 is a Witch", [2, 2], "Witch"),
        ]
        for clue, refs, minion in cases:
            with self.subTest(clue=clue):
                parsed = _parse_clue_from_memory(
                    _memory_card("Poet", 1, clue, refs),
                    n_cards=6,
                )
                self.assertEqual(parsed.info_text, clue)
                self.assertEqual(parsed.info_parsed["copied_role"], "Oracle")
                self.assertEqual(parsed.info_parsed["targets"], refs)
                self.assertEqual(parsed.info_parsed["minion_role"], minion)

        sentinel = _parse_clue_from_memory(
            _memory_card("Poet", 1, "There are no minions", []),
            n_cards=6,
        )
        self.assertEqual(sentinel.info_parsed["copied_role"], "Oracle")
        self.assertTrue(sentinel.info_parsed["no_minions"])

    def test_poet_oracle_rejects_stale_or_mismatched_events(self):
        clue = "#2 or #3 is a Witch"
        invalid = [
            _memory_card("Poet", 1, clue, [3, 2]),
            _memory_card("Poet", 1, f"{clue}.", [2, 3]),
            {
                **_memory_card("Poet", 1, clue, [2, 3]),
                "acted_infos": [{"desc": "old", "targets": [2, 3]}],
            },
        ]
        for card in invalid:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))


if __name__ == "__main__":
    unittest.main()
