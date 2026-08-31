"""Current-build public Druid / managed Librarian bridge regressions."""

from contextlib import redirect_stdout
from io import StringIO
import json
from pathlib import Path
import unittest
from unittest.mock import patch

from game_loop import (
    DecisionLog,
    GameSession,
    _canonical_druid_outcast,
    _druid_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    _parse_druid_native_text,
    _validate_current_druid_targets,
    card_druid,
    dispatch,
)
from memory_reader import clean_name
from solver import (
    BAKER_RULE_VERSION,
    CardInfo,
    DeckComposition,
    GameState,
    Scenario,
    SolverResult,
)
from strategy import (
    _druid_bluff_false_outcasts,
    _druid_observation_likelihoods,
    recommend_abilities,
)


_ABSENT = object()


def _memory_druid(
    clue,
    refs=_ABSENT,
    *,
    role="Librarian",
    position=2,
    prior_infos=None,
    current_role=_ABSENT,
    disguise=_ABSENT,
    **extra,
):
    infos = list(prior_infos or [])
    if refs is not _ABSENT:
        infos.append({"desc": clue, "targets": refs})
    card = {
        "position": position,
        "true_role": role,
        "clue_text": clue,
        "acted_infos": infos,
        "runtime_data": None,
        "ability_used": False,
        "uses": 0,
        **extra,
    }
    if current_role is not _ABSENT:
        card["current_role"] = current_role
    if disguise is not _ABSENT:
        card["disguise"] = disguise
    return card


class DruidNativeTextTests(unittest.TestCase):
    def test_builder_sorts_display_ids_but_preserves_click_order_payload(self):
        for found, clue in (
            (None, "Among #1, #2, #3\nthere are NO Outcasts"),
            ("Plague_Doctor", "Among #1, #2, #3\nthere is: Plague Doctor"),
        ):
            with self.subTest(found=found):
                self.assertEqual(_druid_native_text([3, 1, 2], found), clue)
                parsed = _parse_druid_native_text(clue)
                self.assertEqual(parsed, ([1, 2, 3], found))
                card = card_druid(
                    2,
                    [3, 1, 2],
                    found,
                    druid_variant="public_current",
                )
                self.assertEqual(card.info_text, clue)
                self.assertEqual(
                    card.info_parsed,
                    {
                        "targets": [3, 1, 2],
                        "found_outcast": found,
                        "druid_variant": "public_current",
                    },
                )

    def test_exact_parser_rejects_text_and_role_mutations(self):
        mutations = (
            "among #1, #2, #3\nthere are NO Outcasts",
            "Among #1,#2, #3\nthere are NO Outcasts",
            "Among #1, #2, #3 there are NO Outcasts",
            "Among #1, #2, #3\nthere are no Outcasts",
            "Among #1, #2, #3\nthere are NO Outcasts.",
            "Among #1, #2, #3\nthere is: plague doctor",
            "Among #1, #2, #3\nthere is: Plague_Doctor",
            "Among #1, #2, #3\nthere was: Bombardier",
            "Among #1, #2, #3\nthere is: Lover",
            "Among #2, #1, #3\nthere is: Bombardier",
            "Among #01, #2, #3\nthere is: Bombardier",
            "Among #1, #1, #3\nthere is: Bombardier",
            "Among #1, #2, #3\nthere is: Bombardier ",
            "Among #1, #2, #3\nthere is: Bombardier\n",
            "",
            None,
        )
        for clue in mutations:
            with self.subTest(clue=clue):
                self.assertIsNone(_parse_druid_native_text(clue))

    def test_current_validation_is_closed_but_unmarked_archive_stays_permissive(self):
        self.assertEqual(_canonical_druid_outcast("plague_doctor"), "Plague_Doctor")
        self.assertIsNone(_canonical_druid_outcast(None))
        self.assertEqual(
            _validate_current_druid_targets([3, 1, 2], n_cards=3),
            [3, 1, 2],
        )
        legacy = card_druid(1, [2], "Plague_Doctor")
        self.assertEqual(legacy.info_text, "")
        self.assertEqual(
            legacy.info_parsed,
            {"targets": [2], "found_outcast": "Plague_Doctor"},
        )
        self.assertNotIn("druid_variant", legacy.info_parsed)

        invalid_targets = (
            [1, 2],
            [1, 2, 2],
            [0, 2, 3],
            [True, 2, 3],
            [1, 2, "3"],
            (1, 2, 3),
            [1, 2, 4],
        )
        for targets in invalid_targets:
            with self.subTest(targets=targets), self.assertRaises(ValueError):
                _validate_current_druid_targets(targets, n_cards=3)
        for found in ("Lover", "Spy", "Definitely Fake", "", 1):
            with self.subTest(found=found), self.assertRaises(ValueError):
                _canonical_druid_outcast(found)
        with self.assertRaises(ValueError):
            card_druid(1, [1, 2, 3], None, druid_variant="future")
        with self.assertRaises(ValueError):
            card_druid(
                1,
                [1, 2, 3],
                None,
                info_text="Among #1, #2, #3\nthere is: Bombardier",
                druid_variant="public_current",
            )

    def test_reader_aliases_match_current_and_obsolete_managed_bindings(self):
        self.assertEqual(clean_name("Librarian"), "Druid")
        self.assertEqual(clean_name("Librarian_12345"), "Druid")
        self.assertEqual(clean_name("RangedEmpath"), "Druid")
        self.assertEqual(clean_name("Acrobat2"), "Bard")
        self.assertEqual(clean_name("Acrobat"), "Bard")


class DruidMemoryIngestionTests(unittest.TestCase):
    @staticmethod
    def _assert_current(card, refs, found):
        assert card is not None
        expected = _druid_native_text(refs, found)
        if card.info_text != expected:
            raise AssertionError((card.info_text, expected))
        if card.info_parsed != {
            "targets": refs,
            "found_outcast": found,
            "druid_variant": "public_current",
        }:
            raise AssertionError(card.info_parsed)

    def test_exact_result_authenticates_before_uses_or_act_settles(self):
        for found in (None, "Bombardier", "Plague_Doctor"):
            refs = [3, 1, 4]
            clue = _druid_native_text(refs, found)
            for role in ("Druid", "Librarian", "RangedEmpath"):
                with self.subTest(found=found, role=role):
                    parsed = _parse_clue_from_memory(
                        _memory_druid(clue, refs, role=role),
                        n_cards=6,
                    )
                    self._assert_current(parsed, refs, found)

    def test_newest_coherent_event_owns_click_order_and_stale_events_reject(self):
        refs = [5, 1, 3]
        clue = _druid_native_text(refs, "Drunk")
        parsed = _parse_clue_from_memory(
            _memory_druid(
                clue,
                refs,
                prior_infos=[
                    {
                        "desc": "Among #1, #2, #3\nthere are NO Outcasts",
                        "targets": [1, 2, 3],
                    }
                ],
            ),
            n_cards=6,
        )
        self._assert_current(parsed, refs, "Drunk")

        stale = _memory_druid(clue, refs)
        stale["acted_infos"].append(
            {"desc": "newer unrelated event", "targets": [1, 2, 3]}
        )
        self.assertIsNone(_parse_clue_from_memory(stale, n_cards=6))

    def test_event_refs_must_be_exact_click_order_permutation_of_text_ids(self):
        clue = _druid_native_text([1, 2, 3], None)
        for refs in ([3, 2, 1], [2, 1, 3], [1, 3, 2]):
            with self.subTest(refs=refs):
                parsed = _parse_clue_from_memory(
                    _memory_druid(clue, refs),
                    n_cards=6,
                )
                self._assert_current(parsed, refs, None)

        invalid = (
            _ABSENT,
            None,
            [],
            [1, 2],
            [1, 2, 2],
            [1, 2, 4],
            [1, 2, True],
            (1, 2, 3),
            "1,2,3",
        )
        for refs in invalid:
            with self.subTest(refs=refs):
                card = _memory_druid(clue, refs)
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=3))

    def test_actor_board_bounds_and_self_selection(self):
        refs = [2, 4, 1]
        clue = _druid_native_text(refs, "Rambler")
        parsed = _parse_clue_from_memory(
            _memory_druid(clue, refs, position=2),
            n_cards=4,
        )
        self._assert_current(parsed, refs, "Rambler")

        for position, n_cards in (
            (0, 4),
            (5, 4),
            (True, 4),
            ("2", 4),
            (2, None),
            (2, 0),
            (2, True),
        ):
            with self.subTest(position=position, n_cards=n_cards):
                self.assertIsNone(
                    _parse_clue_from_memory(
                        _memory_druid(clue, refs, position=position),
                        n_cards=n_cards,
                    )
                )

    def test_display_precedence_and_unrelated_runtime_are_public_only(self):
        refs = [1, 2, 3]
        clue = _druid_native_text(refs, None)
        accepted = (
            _memory_druid(
                clue,
                refs,
                role="Pooka",
                current_role="Librarian",
                alignment="Evil",
                statuses=["Corrupted"],
                runtime_data={"type": "direction", "direction": "CCW"},
            ),
            _memory_druid(
                clue,
                refs,
                role="Pooka",
                current_role="Pooka",
                disguise="Librarian",
            ),
        )
        for card in accepted:
            with self.subTest(card=card):
                self._assert_current(
                    _parse_clue_from_memory(card, n_cards=6), refs, None
                )

        rejected = (
            _memory_druid(
                clue,
                refs,
                role="Librarian",
                current_role="Librarian",
                disguise="Bard",
            ),
            _memory_druid(
                clue,
                refs,
                role="Librarian",
                current_role="Bard",
            ),
        )
        for card in rejected:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

    def test_passive_empty_is_unmarked_but_malformed_public_data_fails_closed(self):
        for card in (
            _memory_druid(""),
            _memory_druid("", None),
        ):
            with self.subTest(card=card):
                parsed = _parse_clue_from_memory(card, n_cards=6)
                self.assertEqual(parsed.info_parsed, {})
                self.assertNotIn("druid_variant", parsed.info_parsed)

        malformed = (
            _memory_druid("", []),
            _memory_druid("Among #1, #2, #3 there are NO Outcasts", [1, 2, 3]),
            {
                **_memory_druid(
                    "Among #1, #2, #3\nthere are NO Outcasts",
                    [1, 2, 3],
                ),
                "acted_infos": [{"desc": [], "targets": None}],
            },
        )
        for card in malformed:
            with self.subTest(card=card):
                self.assertIsNone(_parse_clue_from_memory(card, n_cards=6))

    def test_rambler_and_baker_surfaces_keep_precedence(self):
        shut_up = _parse_clue_from_memory(
            _memory_druid("#3 shut up!", [3]),
            n_cards=6,
        )
        self.assertEqual(shut_up.apparent_role, "Druid")
        self.assertEqual(shut_up.info_parsed, {"shut_up_target": 3})
        self.assertNotIn("druid_variant", shut_up.info_parsed)

        baker = {
            "position": 2,
            "true_role": "Baker",
            "clue_text": "I was a Druid",
            "acted_infos": [
                {"desc": "I was a Druid", "targets": []}
            ],
        }
        parsed = _parse_clue_from_memory(
            baker,
            n_cards=6,
            baker_rule_version=BAKER_RULE_VERSION,
        )
        self.assertEqual(parsed.apparent_role, "Baker")
        self.assertEqual(parsed.info_parsed, {"original_role": "Druid"})


class DruidManualAndCaptureTests(unittest.TestCase):
    def setUp(self):
        self.session = GameSession(6, 1)

    def test_manual_entry_is_strict_and_synthesizes_current_provenance(self):
        positive = _parse_card_cli(
            ["druid", "2", "4,2,1", "Plague_Doctor"],
            self.session,
        )
        self.assertEqual(
            positive.info_parsed,
            {
                "targets": [4, 2, 1],
                "found_outcast": "Plague_Doctor",
                "druid_variant": "public_current",
            },
        )
        self.assertEqual(
            positive.info_text,
            "Among #1, #2, #4\nthere is: Plague Doctor",
        )
        none = _parse_card_cli(
            ["druid", "2", "2,5,6", "none"],
            self.session,
        )
        self.assertEqual(none.info_parsed["targets"], [2, 5, 6])
        self.assertIsNone(none.info_parsed["found_outcast"])

    def test_manual_rejects_missing_context_arity_bounds_duplicates_and_roles(self):
        invalid = (
            (["druid", "2", "1,2,3", "none"], None),
            (["druid", "2", "1,2,3"], self.session),
            (["druid", "2", "1,2,3", "none", "extra"], self.session),
            (["druid", "0", "1,2,3", "none"], self.session),
            (["druid", "7", "1,2,3", "none"], self.session),
            (["druid", "2", "1,2", "none"], self.session),
            (["druid", "2", "1,2,2", "none"], self.session),
            (["druid", "2", "1,2,7", "none"], self.session),
            (["druid", "2", "1,x,3", "none"], self.session),
            (["druid", "2", "1,2,3", "Lover"], self.session),
            (["druid", "2", "1,2,3", "DefinitelyFake"], self.session),
        )
        for args, session in invalid:
            with self.subTest(args=args), self.assertRaises((IndexError, ValueError)):
                _parse_card_cli(args, session)

    @staticmethod
    def _run_auto_card(session, memory):
        memory = {**memory, "state": "Revealed"}

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

    def test_auto_card_replaces_empty_or_prior_current_but_not_unmarked_result(self):
        first_refs = [3, 1, 2]
        first = _memory_druid(_druid_native_text(first_refs, None), first_refs)

        empty = GameSession(6, 1)
        empty.add_card(CardInfo(2, "Druid"))
        self._run_auto_card(empty, first)
        self.assertEqual(empty.cards[0].info_parsed["targets"], first_refs)
        self.assertEqual(
            empty.cards[0].info_parsed["druid_variant"],
            "public_current",
        )

        reset = GameSession(6, 1)
        reset.add_card(
            card_druid(
                2,
                first_refs,
                None,
                druid_variant="public_current",
            )
        )
        second_refs = [6, 2, 4]
        second = _memory_druid(
            _druid_native_text(second_refs, "Wretch"),
            second_refs,
        )
        self._run_auto_card(reset, second)
        self.assertEqual(reset.cards[0].info_parsed["targets"], second_refs)
        self.assertEqual(reset.cards[0].info_parsed["found_outcast"], "Wretch")

        legacy = GameSession(6, 1)
        legacy.add_card(card_druid(2, [1, 2, 3], "Bombardier"))
        self._run_auto_card(legacy, first)
        self.assertEqual(
            legacy.cards[0].info_parsed,
            {"targets": [1, 2, 3], "found_outcast": "Bombardier"},
        )


class DruidArchiveCompatibilityTests(unittest.TestCase):
    def test_all_archived_direct_shapes_remain_unmarked_and_counted(self):
        root = Path(__file__).parent
        totals = {"cases": [0, 0], "cases_v2": [0, 0]}
        self_targets = []
        for group in totals:
            for path in (root / group).glob("*.json"):
                data = json.loads(path.read_text(encoding="utf-8"))
                for card in data.get("cards", []):
                    if card.get("apparent_role") != "Druid":
                        continue
                    totals[group][0] += 1
                    info = card.get("info_parsed") or {}
                    self.assertEqual(card.get("info_text"), "")
                    self.assertNotIn("druid_variant", info)
                    if info:
                        totals[group][1] += 1
                        self.assertEqual(set(info), {"targets", "found_outcast"})
                        if card["position"] in info["targets"]:
                            self_targets.append((group, path.name, card["position"]))
        self.assertEqual(totals, {"cases": [42, 30], "cases_v2": [118, 79]})
        self.assertEqual(self_targets, [("cases_v2", "asc32_v7.json", 6)])


class DruidStrategyTests(unittest.TestCase):
    @staticmethod
    def _state(*, outcasts=None):
        return GameState(
            n_cards=6,
            deck=DeckComposition(
                ["Druid", "Lover", "Scout"],
                outcasts or ["Bombardier", "Drunk", "Wretch"],
                [],
                [],
            ),
            cards=[
                CardInfo(1, "Druid"),
                CardInfo(2, "Bombardier"),
                CardInfo(3, "Drunk"),
                CardInfo(4, "Lover"),
                CardInfo(5, "Wretch"),
                CardInfo(6, "Scout"),
            ],
        )

    def test_truth_uniformly_names_physical_registered_outcast_candidates(self):
        state = self._state()
        likelihoods = _druid_observation_likelihoods(
            [2, 3, 4],
            1,
            Scenario({}),
            state,
        )
        self.assertEqual(
            likelihoods,
            {
                ("outcast", "Bombardier"): 0.5,
                ("outcast", "Drunk"): 0.5,
            },
        )
        self.assertEqual(
            _druid_observation_likelihoods(
                [1, 4, 5],
                1,
                Scenario({}),
                state,
            ),
            {("none",): 1.0},
        )

    def test_spy_and_wretch_register_as_do_not_count(self):
        state = self._state()
        spy = Scenario({6: "Spy"})
        self.assertEqual(
            _druid_observation_likelihoods([1, 5, 6], 1, spy, state),
            {("none",): 1.0},
        )

        doppelganger = Scenario({}, doppelganger_position=6)
        self.assertEqual(
            _druid_observation_likelihoods(
                [1, 4, 6],
                1,
                doppelganger,
                state,
            ),
            {("outcast", "Doppelganger"): 1.0},
        )

    def test_bluff_complement_and_false_ladder_multiplicity(self):
        state = self._state(
            outcasts=["Drunk", "Drunk", "Wretch", "Bombardier"]
        )
        liar = Scenario({}, corrupted={1})
        self.assertEqual(
            _druid_observation_likelihoods([2, 4, 6], 1, liar, state),
            {("none",): 1.0},
        )
        self.assertEqual(
            _druid_bluff_false_outcasts(state),
            ("Drunk", "Drunk", "Wretch"),
        )
        false_positive = _druid_observation_likelihoods(
            [1, 4, 6], 1, liar, state
        )
        self.assertAlmostEqual(false_positive[("outcast", "Drunk")], 2 / 3)
        self.assertAlmostEqual(false_positive[("outcast", "Wretch")], 1 / 3)

    def test_unknown_visible_identity_fails_closed(self):
        state = GameState(
            3,
            DeckComposition(["Druid"], ["Drunk"], [], []),
            [CardInfo(1, "Druid")],
        )
        self.assertEqual(
            _druid_observation_likelihoods([1, 2, 3], 1, Scenario({}), state),
            {},
        )

    def test_recommendation_candidates_include_every_board_lifecycle_seat(self):
        state = self._state()
        state.executed = [2]
        state.night_kills = [3]
        state.blocked_positions = [4]
        result = SolverResult([], [], [], 1, 1, [Scenario({})])
        with patch("strategy._recommend_druid_ability", return_value=None) as recommend:
            recommend_abilities(state, result, used_abilities=[])
        candidates = recommend.call_args.args[1]
        self.assertIn([1, 2, 3], candidates)
        self.assertIn([1, 4, 5], candidates)
        self.assertIn([2, 3, 6], candidates)


if __name__ == "__main__":
    unittest.main()
