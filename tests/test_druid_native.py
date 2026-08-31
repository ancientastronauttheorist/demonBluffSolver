"""Current-build public Druid / managed Librarian bridge regressions."""

from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from game_loop import (
    DecisionLog,
    GameSession,
    _ORDERED_CALLBACK_LEDGER_VARIANT,
    _canonical_druid_outcast,
    _druid_callback_ledger,
    _druid_native_text,
    _parse_card_cli,
    _parse_clue_from_memory,
    _parse_druid_result_from_memory,
    _parse_druid_native_text,
    _validate_current_druid_targets,
    card_druid,
    dispatch,
)
from knowledge_base import get_card
from memory_reader import clean_name
from solver import (
    BAKER_RULE_VERSION,
    CardInfo,
    DeckComposition,
    GameState,
    Scenario,
    SolverResult,
)
from state_machine import GamePhase, GameStateMachine
from strategy import (
    Action,
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


def _druid_event(refs, found=None):
    return {
        "desc": _druid_native_text(list(refs), found),
        "targets": list(refs),
    }


def _memory_druid_history(events, *, position=2, role="Librarian", **extra):
    events = [copy.deepcopy(event) for event in events]
    clue = events[-1]["desc"] if events else ""
    return {
        "position": position,
        "true_role": role,
        "clue_text": clue,
        "acted_infos": [{"desc": "", "targets": None}] + events,
        "runtime_data": None,
        "ability_used": bool(events),
        "uses": 0,
        **extra,
    }


def _strict_session(*, reveals=(1, 2, 3), nights=0):
    session = GameSession(6, 1)
    session.reveal_order = list(reveals)
    session.lilis_nights_resolved = nights
    return session


def _parse_and_add(session, memory, *, expected_targets=None):
    parsed, error = _parse_druid_result_from_memory(
        memory,
        n_cards=session.n_cards,
        expected_targets=expected_targets,
    )
    if error is not None:
        raise AssertionError(error)
    if parsed is None:
        raise AssertionError("Druid callback is still pending")
    session.add_card(parsed)
    return next(card for card in session.cards if card.position == parsed.position)


def _session_snapshot(session):
    return json.dumps(
        {
            "state": session.to_game_state().to_dict(),
            "used_abilities": list(session.used_abilities),
            "lilis_batch_index": session.lilis_batch_index,
            "lilis_nights_resolved": session.lilis_nights_resolved,
            "pending_lilis_nights": session.pending_lilis_nights,
            "druid_reset_generations": session.druid_reset_generations,
            "druid_pending_activations": session.druid_pending_activations,
        },
        sort_keys=True,
    )


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
            _memory_druid("#3\nshut up!", [3]),
            n_cards=6,
        )
        self.assertEqual(shut_up.apparent_role, "Druid")
        self.assertEqual(
            shut_up.info_parsed,
            {
                "shut_up_target": 3,
                "druid_variant": "public_current",
            },
        )
        for mutation in (
            "#3 shut up!",
            "#3\nShut up!",
            "#3\nshut up",
            "#3\r\nshut up!",
        ):
            with self.subTest(mutation=mutation):
                self.assertIsNone(_parse_clue_from_memory(
                    _memory_druid(mutation, [3]),
                    n_cards=6,
                ))

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

    def test_manual_interruption_is_rejected_without_raw_provenance(self):
        for args in (
            ["druid", "2", "shut_up", "5"],
            ["shut_up", "2", "Druid", "5"],
        ):
            with self.subTest(args=args), self.assertRaisesRegex(
                ValueError,
                "authenticated raw",
            ):
                _parse_card_cli(args, self.session)

        scalar = CardInfo(
            2,
            "Druid",
            info_text="#5\nshut up!",
            info_parsed={
                "shut_up_target": 5,
                "druid_variant": "public_current",
            },
        )
        with self.assertRaisesRegex(ValueError, "authenticated raw"):
            self.session.add_card(scalar)
        self.assertEqual(self.session.cards, [])
        self.assertEqual(self.session.rambler_shut_up_observations, [])
        self.assertEqual(self.session.reveal_order, [])

    def test_manual_compatibility_reveal_never_invents_a_ledger_boundary(self):
        self.assertEqual(self.session.reveal_order, [])
        self.assertIsNotNone(self.session.baker_rule_version)
        card = _parse_card_cli(
            ["druid", "2", "4,2,1", "none"],
            self.session,
        )
        self.session.add_card(card)
        stored = self.session.cards[0]
        self.assertEqual(stored.info_parsed, card.info_parsed)
        self.assertNotIn("callback_events", stored.info_parsed)
        self.assertEqual(self.session.reveal_order, [2])
        self.assertIsNone(self.session.baker_rule_version)

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

        output = StringIO()
        with (
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch("memory_reader.print_board"),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            redirect_stdout(output),
        ):
            dispatch("auto_card", [], session)
        return output.getvalue()

    def test_auto_card_builds_ledger_for_empty_placeholder_but_preserves_legacy(self):
        refs = [3, 1, 2]
        memory = _memory_druid(_druid_native_text(refs, None), refs)

        current = _strict_session()
        current.cards = [CardInfo(2, "Druid")]
        output = self._run_auto_card(current, memory)
        self.assertIn("updated #2 Druid", output)
        stored = current.cards[0]
        self.assertEqual(
            stored.info_parsed["callback_ledger_variant"],
            _ORDERED_CALLBACK_LEDGER_VARIANT,
        )
        self.assertEqual(stored.info_parsed["targets"], refs)

        legacy = _strict_session()
        legacy.cards = [card_druid(2, [1, 2, 3], "Bombardier")]
        self._run_auto_card(legacy, memory)
        self.assertEqual(
            legacy.cards[0].info_parsed,
            {"targets": [1, 2, 3], "found_outcast": "Bombardier"},
        )

    def test_auto_card_surfaces_malformed_druid_as_recovery(self):
        session = _strict_session()
        session.cards = [CardInfo(2, "Druid")]
        malformed = _memory_druid(
            "Among #1, #2, #3\nthere are no Outcasts",
            [3, 1, 2],
        )
        output = self._run_auto_card(session, malformed)
        self.assertIn("[RECOVERY]", output)
        self.assertIn("Unrecognized Druid acted-info text", output)
        self.assertEqual(session.cards[0].info_parsed, {})

    def test_auto_card_stale_after_reset_waits_then_appended_suffix_updates(self):
        session = _strict_session()
        session.cards = [CardInfo(2, "Druid")]
        first = _druid_event([3, 1, 2], None)
        self._run_auto_card(session, _memory_druid_history([first]))
        first_info = copy.deepcopy(session.cards[0].info_parsed)

        session.reset_after_night_abilities()
        stale_output = self._run_auto_card(
            session,
            _memory_druid_history([first]),
        )
        self.assertIn("Entered 0 cards", stale_output)
        self.assertEqual(session.cards[0].info_parsed, first_info)
        self.assertNotIn(2, session.used_abilities)

        session.reveal_order.append(4)
        second = _druid_event([4, 1, 2], "Wretch")
        appended_output = self._run_auto_card(
            session,
            _memory_druid_history([first, second]),
        )
        self.assertIn("updated #2 Druid", appended_output)
        events = session.cards[0].info_parsed["callback_events"]
        self.assertEqual([event["activation_id"] for event in events], [1, 2])
        self.assertEqual(
            [event["settled_reveal_count"] for event in events],
            [3, 4],
        )
        self.assertEqual(events[-1]["reset_generation"], 1)
        self.assertEqual(
            events[-1]["activation_evidence"],
            "session_reset_generation",
        )


class DruidOrderedCallbackLedgerTests(unittest.TestCase):
    @staticmethod
    def _pending_session(expected_targets=(3, 1, 2), *, boundary=3):
        session = _strict_session(reveals=tuple(range(1, boundary + 1)))
        session.cards = [CardInfo(2, "Druid")]
        session.druid_reset_generations[2] = 0
        session.druid_pending_activations[2] = {
            "activation_id": 1,
            "expected_targets": list(expected_targets),
            "prior_callback_count": 0,
            "reset_generation": 0,
            "settled_reveal_count": boundary,
        }
        return session

    def test_metadata_initial_single_callback_and_exact_fields(self):
        definition = get_card("Druid")
        self.assertTrue(definition.activated_ability)
        self.assertTrue(definition.ability_resets_after_night)

        session = _strict_session()
        stored = _parse_and_add(
            session,
            _memory_druid_history([_druid_event([3, 1, 2], None)]),
        )
        self.assertEqual(
            set(stored.info_parsed),
            {
                "druid_variant",
                "callback_ledger_variant",
                "callback_events",
                "targets",
                "found_outcast",
            },
        )
        event = stored.info_parsed["callback_events"][0]
        self.assertEqual(
            event,
            {
                "activation_id": 1,
                "activation_evidence": "single_callback_suffix",
                "callback_index": 0,
                "dispatch_path": "either",
                "event_kind": "druid_result",
                "reset_generation": 0,
                "settled_reveal_count": 3,
                "text": _druid_native_text([3, 1, 2], None),
                "references": [3, 1, 2],
                "targets": [3, 1, 2],
                "found_outcast": None,
            },
        )

    def test_discovery_after_prior_nights_uses_global_generation_not_zero(self):
        session = _strict_session(nights=3)
        stored = _parse_and_add(
            session,
            _memory_druid_history([_druid_event([3, 1, 2], "Drunk")]),
        )
        event = stored.info_parsed["callback_events"][0]
        self.assertEqual(event["reset_generation"], 3)
        self.assertEqual(
            event["activation_evidence"],
            "session_reset_generation",
        )
        self.assertEqual(session.druid_reset_generations, {2: 3})

        ambiguous = _strict_session(nights=3)
        with self.assertRaisesRegex(ValueError, "exactly one raw callback"):
            _parse_and_add(
                ambiguous,
                _memory_druid_history([
                    _druid_event([3, 1, 2], None),
                    _druid_event([3, 1, 2], "Wretch"),
                ]),
            )

        night_mapped = _strict_session()
        night_mapped.cards = [CardInfo(2, "Druid")]
        night_mapped.reset_after_night_abilities()
        self.assertEqual(night_mapped.druid_reset_generations, {2: 1})
        with self.assertRaisesRegex(ValueError, "exactly one raw callback"):
            _parse_and_add(
                night_mapped,
                _memory_druid_history([
                    _druid_event([3, 1, 2], None),
                    _druid_event([3, 1, 2], "Wretch"),
                ]),
            )

    def test_foreign_real_callbacks_are_opaque_including_outcast_and_among(self):
        foreign_events = (
            {"desc": "#4 is Outcast", "targets": [4]},
            {
                "desc": "Among #1, #2 there is: Lover or Scout",
                "targets": [1, 2],
            },
            {
                "desc": "Among #1, #2, #3 there are 2 Evils",
                "targets": [1, 2, 3],
            },
        )
        for foreign in foreign_events:
            with self.subTest(foreign=foreign):
                session = self._pending_session()
                stored = _parse_and_add(
                    session,
                    _memory_druid_history([
                        foreign,
                        _druid_event([3, 1, 2], "Wretch"),
                    ]),
                    expected_targets=[3, 1, 2],
                )
                events = stored.info_parsed["callback_events"]
                self.assertEqual(
                    [event["event_kind"] for event in events],
                    ["opaque_real", "druid_result"],
                )
                self.assertEqual(
                    [event["dispatch_path"] for event in events],
                    ["real", "raw"],
                )
                self.assertTrue(all(
                    event["activation_evidence"] == "auto_use_click"
                    for event in events
                ))

    def test_true_druid_family_near_misses_are_not_opaque(self):
        for text in (
            "Among #1, #2, #3 there was: Wretch",
            "Among #1, #2, #3 there were zero Outcasts",
        ):
            with self.subTest(text=text):
                _, error = _parse_druid_result_from_memory(
                    _memory_druid_history([
                        {"desc": text, "targets": [1, 2, 3]},
                        _druid_event([3, 1, 2], None),
                    ]),
                    n_cards=6,
                )
                self.assertIn("Unrecognized Druid", error)

    def test_dual_druid_results_share_targets_and_boundary(self):
        session = self._pending_session()
        stored = _parse_and_add(
            session,
            _memory_druid_history([
                _druid_event([3, 1, 2], None),
                _druid_event([3, 1, 2], "Wretch"),
            ]),
            expected_targets=[3, 1, 2],
        )
        events = stored.info_parsed["callback_events"]
        self.assertEqual([event["callback_index"] for event in events], [0, 1])
        self.assertEqual(
            [event["settled_reveal_count"] for event in events],
            [3, 3],
        )
        self.assertEqual(
            [event["dispatch_path"] for event in events],
            ["real", "raw"],
        )

    def test_dual_interruption_rewrites_both_and_global_evidence_has_both(self):
        interruption = {"desc": "#5\nshut up!", "targets": [5]}
        session = self._pending_session()
        stored = _parse_and_add(
            session,
            _memory_druid_history([interruption, interruption]),
        )
        events = stored.info_parsed["callback_events"]
        self.assertEqual(
            [event["event_kind"] for event in events],
            ["rambler_interruption", "rambler_interruption"],
        )
        self.assertEqual(stored.info_parsed["shut_up_target"], 5)
        self.assertEqual(
            session.rambler_shut_up_observations,
            [
                {"speaker_position": 2, "shut_up_target": 5},
                {"speaker_position": 2, "shut_up_target": 5},
            ],
        )

        parsed, error = _parse_druid_result_from_memory(
            _memory_druid_history([interruption, interruption]),
            n_cards=6,
        )
        self.assertIsNone(error)
        status, status_error = __import__("game_loop")._classify_druid_auto_capture(
            stored,
            parsed,
            n_cards=6,
            reveal_order=session.reveal_order,
            baker_rule_version=session.baker_rule_version,
            rambler_observations=session.rambler_shut_up_observations,
        )
        self.assertEqual((status, status_error), ("stale", None))

    def test_stale_capture_rejects_missing_or_extra_global_rambler_rows(self):
        normal_session = _strict_session()
        normal_memory = _memory_druid_history([
            _druid_event([3, 1, 2], None),
        ])
        normal = _parse_and_add(normal_session, normal_memory)
        normal_session.rambler_shut_up_observations.append({
            "speaker_position": 2,
            "shut_up_target": 5,
        })
        parsed, error = _parse_druid_result_from_memory(
            normal_memory,
            n_cards=6,
        )
        self.assertIsNone(error)
        status, status_error = __import__(
            "game_loop"
        )._classify_druid_auto_capture(
            normal,
            parsed,
            n_cards=6,
            reveal_order=normal_session.reveal_order,
            baker_rule_version=normal_session.baker_rule_version,
            rambler_observations=(
                normal_session.rambler_shut_up_observations
            ),
        )
        self.assertEqual(status, "error")
        self.assertIn("exact same-speaker", status_error)
        with self.assertRaisesRegex(ValueError, "exact same-speaker"):
            normal_session.add_card(parsed)

        interrupted_session = _strict_session()
        interrupted_memory = _memory_druid_history([
            {"desc": "#5\nshut up!", "targets": [5]},
        ])
        interrupted = _parse_and_add(
            interrupted_session,
            interrupted_memory,
        )
        interrupted_session.rambler_shut_up_observations.clear()
        parsed, error = _parse_druid_result_from_memory(
            interrupted_memory,
            n_cards=6,
        )
        self.assertIsNone(error)
        status, status_error = __import__(
            "game_loop"
        )._classify_druid_auto_capture(
            interrupted,
            parsed,
            n_cards=6,
            reveal_order=interrupted_session.reveal_order,
            baker_rule_version=interrupted_session.baker_rule_version,
            rambler_observations=(
                interrupted_session.rambler_shut_up_observations
            ),
        )
        self.assertEqual(status, "error")
        self.assertIn("exact same-speaker", status_error)

        identical = CardInfo(
            2,
            "Druid",
            info_text=interrupted.info_text,
            info_parsed=copy.deepcopy(interrupted.info_parsed),
        )
        with self.assertRaisesRegex(ValueError, "exact same-speaker"):
            interrupted_session.add_card(identical)

    def test_mixed_rambler_dual_dispatch_fails_closed(self):
        normal = _druid_event([3, 1, 2], None)
        interruption = {"desc": "#5\nshut up!", "targets": [5]}
        for events in ([normal, interruption], [interruption, normal]):
            with self.subTest(events=events):
                session = self._pending_session()
                before = (
                    copy.deepcopy(session.cards),
                    copy.deepcopy(session.rambler_shut_up_observations),
                    copy.deepcopy(session.druid_pending_activations),
                    copy.deepcopy(session.druid_reset_generations),
                )
                parsed, error = _parse_druid_result_from_memory(
                    _memory_druid_history(events),
                    n_cards=6,
                )
                self.assertIsNone(error)
                with self.assertRaisesRegex(ValueError, "Both callbacks"):
                    session.add_card(parsed)
                self.assertEqual(session.cards, before[0])
                self.assertEqual(
                    session.rambler_shut_up_observations,
                    before[1],
                )
                self.assertEqual(session.druid_pending_activations, before[2])
                self.assertEqual(session.druid_reset_generations, before[3])

    def test_delayed_second_callback_upgrades_entire_activation(self):
        session = _strict_session()
        first = _druid_event([3, 1, 2], None)
        _parse_and_add(session, _memory_druid_history([first]))
        self.assertIn(2, session.used_abilities)

        second = _druid_event([3, 1, 2], "Wretch")
        stored = _parse_and_add(
            session,
            _memory_druid_history([first, second]),
        )
        events = stored.info_parsed["callback_events"]
        self.assertEqual(
            [event["activation_evidence"] for event in events],
            ["same_activation_extension", "same_activation_extension"],
        )
        self.assertEqual(
            [event["dispatch_path"] for event in events],
            ["real", "raw"],
        )
        self.assertEqual(
            [event["settled_reveal_count"] for event in events],
            [3, 3],
        )

    def test_interruption_after_older_normal_has_own_settled_boundary(self):
        session = _strict_session()
        first = _druid_event([3, 1, 2], None)
        _parse_and_add(session, _memory_druid_history([first]))
        session.reset_after_night_abilities()
        session.reveal_order.append(4)

        interrupted = {"desc": "#5\nshut up!", "targets": [5]}
        stored = _parse_and_add(
            session,
            _memory_druid_history([first, interrupted]),
        )
        events = stored.info_parsed["callback_events"]
        self.assertEqual([event["activation_id"] for event in events], [1, 2])
        self.assertEqual(
            [event["settled_reveal_count"] for event in events],
            [3, 4],
        )
        self.assertEqual(stored.info_parsed["shut_up_target"], 5)
        self.assertEqual(
            session.rambler_shut_up_observations,
            [{"speaker_position": 2, "shut_up_target": 5}],
        )

    def test_skipped_unused_generations_allow_one_callback_but_not_two(self):
        first = _druid_event([3, 1, 2], None)
        second = _druid_event([3, 1, 2], "Wretch")

        single = _strict_session()
        _parse_and_add(single, _memory_druid_history([first]))
        for _ in range(3):
            single.reset_after_night_abilities()
        stored = _parse_and_add(
            single,
            _memory_druid_history([first, second]),
        )
        events = stored.info_parsed["callback_events"]
        self.assertEqual(events[-1]["reset_generation"], 3)
        self.assertEqual(events[-1]["activation_id"], 2)

        ambiguous = _strict_session()
        _parse_and_add(ambiguous, _memory_druid_history([first]))
        for _ in range(3):
            ambiguous.reset_after_night_abilities()
        with self.assertRaisesRegex(ValueError, "two-callback.*ambiguous"):
            _parse_and_add(
                ambiguous,
                _memory_druid_history([
                    first,
                    second,
                    _druid_event([3, 1, 2], "Drunk"),
                ]),
            )

    def test_clicked_trio_validates_every_normal_callback_atomically(self):
        session = self._pending_session()
        memory = _memory_druid_history([
            _druid_event([4, 1, 2], None),
            _druid_event([3, 1, 2], "Wretch"),
        ])
        parsed, error = _parse_druid_result_from_memory(
            memory,
            n_cards=6,
            expected_targets=[3, 1, 2],
        )
        self.assertIsNone(error)
        caller_before = parsed.to_dict()
        session_before = {
            "cards": [card.to_dict() for card in session.cards],
            "rambler": copy.deepcopy(session.rambler_shut_up_observations),
            "pending": copy.deepcopy(session.druid_pending_activations),
            "generation": copy.deepcopy(session.druid_reset_generations),
            "used": list(session.used_abilities),
            "reveals": list(session.reveal_order),
        }
        with self.assertRaisesRegex(ValueError, "click token"):
            session.add_card(parsed)
        self.assertEqual(parsed.to_dict(), caller_before)
        self.assertEqual(
            [card.to_dict() for card in session.cards],
            session_before["cards"],
        )
        self.assertEqual(
            session.rambler_shut_up_observations,
            session_before["rambler"],
        )
        self.assertEqual(
            session.druid_pending_activations,
            session_before["pending"],
        )
        self.assertEqual(
            session.druid_reset_generations,
            session_before["generation"],
        )
        self.assertEqual(session.used_abilities, session_before["used"])
        self.assertEqual(session.reveal_order, session_before["reveals"])

    def test_impossible_boundaries_generations_and_actor_prefix_reject(self):
        session = _strict_session()
        stored = _parse_and_add(
            session,
            _memory_druid_history([_druid_event([3, 1, 2], None)]),
        )
        base = stored.info_parsed
        malformed = []

        zero = copy.deepcopy(base)
        zero["callback_events"][0]["settled_reveal_count"] = 0
        malformed.append((zero, "settled_reveal_count"))

        actor_absent = copy.deepcopy(base)
        actor_absent["callback_events"][0]["settled_reveal_count"] = 1
        malformed.append((actor_absent, "absent"))

        for info, message in malformed:
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                _druid_callback_ledger(
                    info,
                    actor_position=2,
                    n_cards=6,
                    reveal_order=[1, 2, 3],
                    baker_rule_version=BAKER_RULE_VERSION,
                )

        session.reset_after_night_abilities()
        second = _parse_and_add(
            session,
            _memory_druid_history([
                _druid_event([3, 1, 2], None),
                _druid_event([3, 1, 2], "Wretch"),
            ]),
        )
        duplicate_generation = copy.deepcopy(second.info_parsed)
        duplicate_generation["callback_events"][1]["reset_generation"] = 0
        with self.assertRaisesRegex(ValueError, "increase"):
            _druid_callback_ledger(
                duplicate_generation,
                actor_position=2,
                n_cards=6,
                reveal_order=[1, 2, 3],
                baker_rule_version=BAKER_RULE_VERSION,
            )

    def test_reset_generation_persists_and_advances_for_every_completed_night(self):
        session = _strict_session()
        session.cards = [CardInfo(2, "Druid")]
        self.assertEqual(
            session.reset_after_night_abilities(completed_nights=2),
            [],
        )
        self.assertEqual(session.druid_reset_generations, {2: 2})

        session.druid_pending_activations[2] = {
            "activation_id": 1,
            "expected_targets": [3, 1, 2],
            "prior_callback_count": 0,
            "reset_generation": 2,
            "settled_reveal_count": 3,
        }
        with tempfile.TemporaryDirectory() as directory:
            path = str(Path(directory) / "session.json")
            with redirect_stdout(StringIO()):
                session.save(path)
                loaded = GameSession.load(path)
        self.assertEqual(loaded.druid_reset_generations, {2: 2})
        self.assertEqual(
            loaded.druid_pending_activations,
            session.druid_pending_activations,
        )

    def test_pending_click_blocks_direct_and_production_nights_atomically(self):
        def with_pending():
            session = _strict_session()
            session.cards = [CardInfo(2, "Druid")]
            session.used_abilities = [2]
            session.druid_reset_generations = {2: 0}
            session.druid_pending_activations = {
                2: {
                    "activation_id": 1,
                    "expected_targets": [3, 1, 2],
                    "prior_callback_count": 0,
                    "reset_generation": 0,
                    "settled_reveal_count": 3,
                },
            }
            return session

        direct = with_pending()
        before = _session_snapshot(direct)
        with self.assertRaisesRegex(ValueError, "run auto_card"):
            direct.reset_after_night_abilities()
        self.assertEqual(_session_snapshot(direct), before)

        live_lilis = with_pending()
        live_lilis.demons = ["Lilis"]
        live_lilis.pending_lilis_nights = 1
        before = _session_snapshot(live_lilis)
        with self.assertRaisesRegex(ValueError, "run auto_card"):
            live_lilis.record_lilis_night_result([], 0)
        self.assertEqual(_session_snapshot(live_lilis), before)

        dead_lilis = with_pending()
        dead_lilis.demons = ["Lilis"]
        dead_lilis.executed = [6]
        dead_lilis.executed_current_roles = {6: "Lilis"}
        dead_lilis.pending_lilis_nights = 1
        before = _session_snapshot(dead_lilis)
        with self.assertRaisesRegex(ValueError, "run auto_card"):
            dead_lilis.record_lilis_post_death_night()
        self.assertEqual(_session_snapshot(dead_lilis), before)

    def test_production_night_generation_honors_global_floor(self):
        session = _strict_session(nights=3)
        session.cards = [CardInfo(2, "Druid")]
        session.demons = ["Lilis"]
        session.pending_lilis_nights = 1
        result = session.record_lilis_night_result([], 0)
        self.assertEqual(result["resolved_events"], 1)
        self.assertEqual(session.lilis_nights_resolved, 4)
        self.assertEqual(session.druid_reset_generations, {2: 4})

        direct = _strict_session()
        direct.cards = [CardInfo(2, "Druid")]
        direct.druid_reset_generations = {2: 2}
        direct.reset_after_night_abilities(completed_nights=3)
        self.assertEqual(direct.druid_reset_generations, {2: 5})


class DruidRepeatableAutomationTests(unittest.TestCase):
    class Reader:
        def __init__(self, snapshots):
            self.snapshots = list(snapshots)
            self.index = 0

        def open(self):
            return True

        def read_board(self):
            snapshot = self.snapshots[min(
                self.index,
                len(self.snapshots) - 1,
            )]
            self.index += 1
            return [copy.deepcopy(snapshot)]

        def close(self):
            return None

    def test_auto_use_quiesces_and_groups_real_then_raw_with_click_token(self):
        session = _strict_session()
        session.cards = [CardInfo(2, "Druid")]
        targets = [3, 1, 2]
        passive = _memory_druid_history([])
        first = _memory_druid_history([_druid_event(targets, None)])
        dual = _memory_druid_history([
            _druid_event(targets, None),
            _druid_event(targets, "Wretch"),
        ])
        reader = self.Reader([passive, first, dual, dual, dual])

        with (
            patch("template_match.safe_click_at") as safe_click,
            patch("game_loop.time.sleep"),
            patch("memory_reader.MemoryReader", return_value=reader),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            patch.object(DecisionLog, "log_ability_used"),
        ):
            result = session.auto_use_ability(
                Action("use_ability", 2, targets, "Druid")
            )

        self.assertTrue(result["success"], result["error"])
        self.assertGreaterEqual(safe_click.call_count, 4)
        events = session.cards[0].info_parsed["callback_events"]
        self.assertEqual(len(events), 2)
        self.assertTrue(all(
            event["activation_evidence"] == "auto_use_click"
            for event in events
        ))
        self.assertNotIn(2, session.druid_pending_activations)

    def test_callback_visible_after_stable_window_is_reconciled_as_read_race(self):
        session = _strict_session()
        session.cards = [CardInfo(2, "Druid")]
        targets = [3, 1, 2]
        passive = _memory_druid_history([])
        first = _memory_druid_history([_druid_event(targets, None)])
        reader = self.Reader([passive, first, first, first, first, first])

        # Native real/raw dispatch is synchronous and both callbacks use
        # ShowActedDelayed(0.0). This deliberately simulates only a memory-read
        # visibility race that outlives the two stable 0.15s reads, not a native
        # delayed callback. The later complete raw history must remain safely
        # recoverable before the next solve.
        with (
            patch("template_match.safe_click_at"),
            patch("game_loop.time.sleep"),
            patch("memory_reader.MemoryReader", return_value=reader),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            patch.object(DecisionLog, "log_ability_used"),
        ):
            result = session.auto_use_ability(
                Action("use_ability", 2, targets, "Druid")
            )

        self.assertTrue(result["success"], result["error"])
        initial = session.cards[0].info_parsed["callback_events"]
        self.assertEqual(len(initial), 1)
        self.assertEqual(initial[0]["dispatch_path"], "either")
        self.assertEqual(initial[0]["activation_evidence"], "auto_use_click")

        dual = _memory_druid_history([
            _druid_event(targets, None),
            _druid_event(targets, "Wretch"),
        ])
        output = DruidManualAndCaptureTests._run_auto_card(session, dual)
        self.assertIn("updated #2 Druid", output)
        events = session.cards[0].info_parsed["callback_events"]
        self.assertEqual(
            [event["activation_evidence"] for event in events],
            ["same_activation_extension", "same_activation_extension"],
        )
        self.assertEqual(
            [event["dispatch_path"] for event in events],
            ["real", "raw"],
        )
        self.assertEqual(
            [event["settled_reveal_count"] for event in events],
            [3, 3],
        )

    def test_scalar_resume_stops_before_reader_or_click_with_honest_restart(self):
        session = _strict_session()
        session.cards = [card_druid(
            2,
            [3, 1, 2],
            None,
            druid_variant="public_current",
        )]
        with (
            patch("template_match.safe_click_at") as safe_click,
            patch("memory_reader.MemoryReader") as memory_reader,
        ):
            result = session.auto_use_ability(
                Action("use_ability", 2, [3, 1, 2], "Druid")
            )
        self.assertFalse(result["success"])
        self.assertIn("cannot be resumed", result["error"])
        self.assertIn("restart", result["error"])
        safe_click.assert_not_called()
        memory_reader.assert_not_called()

    def test_state_machine_recovery_does_not_claim_manual_resume_is_safe(self):
        session = _strict_session()
        machine = GameStateMachine(session=session, monitor=None)
        machine.phase = GamePhase.ABILITY_USE
        machine._pending_ability = (2, [3, 1, 2], "Druid", None)
        with (
            patch.object(
                session,
                "auto_use_ability",
                return_value={
                    "success": False,
                    "info_parsed": None,
                    "error": "test mismatch",
                },
            ),
            patch.object(machine, "_pause") as pause,
        ):
            machine._do_ability_use()
        message = pause.call_args.args[0]
        self.assertIn("card druid <actor>", message)
        self.assertIn("cannot resume ResetAfterNight history", message)
        self.assertIn("cannot be entered manually", message)
        self.assertIn("restart", message)
        self.assertNotIn("then 'resume'", message)

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
