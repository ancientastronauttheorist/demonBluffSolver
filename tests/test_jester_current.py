"""Adversarial current-build Jester / managed Juggler bridge tests.

These tests deliberately distinguish the strict current schema from the
unmarked archive-compatible ``targets``/``evil_count`` shape.  Native anchors
live in ``reverse_engineering/notes/roles/gameplay_role_jester.md``:

* exact text sorts display IDs while ActedInfo references retain click order;
* ResetAfterNight keeps an append-only callback history;
* real dispatch precedes raw/bluff dispatch; and
* Rambler may replace either callback independently before it is appended.
"""

from contextlib import redirect_stdout
import copy
from io import StringIO
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import game_loop
from game_loop import DecisionLog, GameSession, _parse_clue_from_memory, dispatch
from knowledge_base import get_card
from solver import CardInfo
from state_machine import GamePhase, GameStateMachine
from strategy import Action


JESTER_VARIANT = "public_current"
LEDGER_VARIANT = "ordered_callbacks_v1"
_ABSENT = object()


def _jester_text(references, evil_count):
    first, second, third = sorted(references)
    if evil_count == 1:
        result = "There is 1 Evil"
    else:
        result = f"There are {evil_count} Evils"
    return f"Among:\n#{first}, #{second}, #{third}:\n{result}"


def _jester_event(references, evil_count):
    return {
        "desc": _jester_text(references, evil_count),
        "targets": list(references),
    }


def _interruption_event(target):
    return {"desc": f"#{target}\nshut up!", "targets": [target]}


def _memory_jester(
    events,
    *,
    position=1,
    role="Jester",
    clue=_ABSENT,
    remaining=_ABSENT,
    state="Alive",
    **extra,
):
    copied_events = None if events is None else copy.deepcopy(events)
    if clue is _ABSENT:
        clue = copied_events[-1]["desc"] if copied_events else ""
    if remaining is _ABSENT:
        remaining = 0 if copied_events else 1
    return {
        "position": position,
        "true_role": role,
        "current_role": role,
        "disguise": role,
        "state": state,
        "clue_text": clue,
        "acted_infos": copied_events,
        "runtime_data": None,
        "pickable_uses_remaining": remaining,
        "act_output_enabled": True,
        "pickable_available": (
            remaining > 0 if type(remaining) is int else None
        ),
        # Deprecated aliases remain fixture coverage only. Production must use
        # the explicit fields above.
        "uses": remaining,
        "ability_used": True,
        **extra,
    }


def _strict_session(*, reveals=(1, 2, 3, 4, 5, 6), nights=0):
    session = GameSession(6, 2)
    session.reveal_order = list(reveals)
    session.lilis_nights_resolved = nights
    session.cards = [CardInfo(1, "Jester")]
    return session


def _parse(memory, session):
    return _parse_clue_from_memory(
        memory,
        n_cards=session.n_cards,
        baker_rule_version=session.baker_rule_version,
        fortune_teller_rule_version=session.fortune_teller_rule_version,
    )


def _parse_and_add(session, memory, *, mark_active_result=True):
    parsed = _parse(memory, session)
    if parsed is None:
        raise AssertionError("strict current Jester memory did not parse")
    session.add_card(parsed, mark_active_result=mark_active_result)
    return next(card for card in session.cards if card.position == parsed.position)


def _pending_token(targets, *, activation_id=1, prior_count=0,
                   generation=0, boundary=6):
    return {
        "activation_id": activation_id,
        "expected_targets": list(targets),
        "prior_callback_count": prior_count,
        "reset_generation": generation,
        "settled_reveal_count": boundary,
    }


def _install_pending(session, targets, *, activation_id=1, prior_count=0,
                     generation=0, boundary=6):
    # These attributes are part of the strict-current persistence contract.
    # Assignment keeps later assertions focused when running against an older
    # production implementation that has not introduced them yet.
    session.jester_reset_generations = {1: generation}
    session.jester_pending_activations = {
        1: _pending_token(
            targets,
            activation_id=activation_id,
            prior_count=prior_count,
            generation=generation,
            boundary=boundary,
        )
    }


def _strict_event_payload(
    raw_event,
    *,
    activation_id,
    evidence,
    callback_index,
    dispatch_path,
    generation,
    boundary,
):
    desc = raw_event["desc"]
    references = list(raw_event["targets"])
    if desc == f"#{references[0]}\nshut up!" and len(references) == 1:
        return {
            "activation_id": activation_id,
            "activation_evidence": evidence,
            "callback_index": callback_index,
            "dispatch_path": dispatch_path,
            "event_kind": "rambler_interruption",
            "reset_generation": generation,
            "settled_reveal_count": boundary,
            "text": desc,
            "references": references,
            "shut_up_target": references[0],
        }
    for evil_count in range(4):
        if desc == _jester_text(references, evil_count):
            return {
                "activation_id": activation_id,
                "activation_evidence": evidence,
                "callback_index": callback_index,
                "dispatch_path": dispatch_path,
                "event_kind": "jester_result",
                "reset_generation": generation,
                "settled_reveal_count": boundary,
                "text": desc,
                "references": references,
                "targets": references,
                "evil_count": evil_count,
            }
    raise AssertionError(f"not a strict Jester event: {raw_event!r}")


def _strict_card(raw_events, groups):
    """Build already-authenticated state for preflight/reload tests.

    ``groups`` contains ``(start, end, activation_id, evidence, generation,
    boundary)`` entries. One-record groups are ``either``; two-record groups
    are ``real`` then ``raw``.
    """
    events = []
    for start, end, activation_id, evidence, generation, boundary in groups:
        size = end - start
        for relative, raw_event in enumerate(raw_events[start:end]):
            path = "either" if size == 1 else ("real" if relative == 0 else "raw")
            events.append(_strict_event_payload(
                raw_event,
                activation_id=activation_id,
                evidence=evidence,
                callback_index=relative,
                dispatch_path=path,
                generation=generation,
                boundary=boundary,
            ))
    latest = events[-1]
    info = {
        "jester_variant": JESTER_VARIANT,
        "callback_ledger_variant": LEDGER_VARIANT,
        "callback_events": events,
    }
    if latest["event_kind"] == "jester_result":
        info["targets"] = list(latest["targets"])
        info["evil_count"] = latest["evil_count"]
    elif latest["event_kind"] == "rambler_interruption":
        info["shut_up_target"] = latest["shut_up_target"]
    return CardInfo(1, "Jester", info_text=latest["text"], info_parsed=info)


class _Reader:
    def __init__(self, snapshots):
        self.snapshots = [copy.deepcopy(snapshot) for snapshot in snapshots]
        self.index = 0

    def open(self):
        return True

    def read_board(self):
        snapshot = self.snapshots[min(self.index, len(self.snapshots) - 1)]
        self.index += 1
        return [copy.deepcopy(snapshot)]

    def close(self):
        return None


class _NoResultMonitor:
    """Healthy monitor that never reports a post-click result."""

    def __init__(self, baseline):
        self.baseline = copy.deepcopy(baseline)

    def is_healthy(self):
        return True

    def get_board(self):
        return [copy.deepcopy(self.baseline)]

    def wait_for(self, predicate, timeout, min_delay):
        return False


def _run_auto_card(session, memory):
    with (
        patch("memory_reader.MemoryReader", return_value=_Reader([memory])),
        patch("memory_reader.print_board"),
        patch.object(session, "save"),
        patch.object(DecisionLog, "log_card"),
        redirect_stdout(StringIO()) as output,
    ):
        dispatch("auto_card", [], session)
    return output.getvalue()


class JesterCurrentTextTests(unittest.TestCase):
    def test_exact_singular_plural_and_sorted_text_preserve_click_order_refs(self):
        references = [3, 1, 2]
        for evil_count in range(4):
            with self.subTest(evil_count=evil_count):
                session = _strict_session()
                event = _jester_event(references, evil_count)
                stored = _parse_and_add(session, _memory_jester([event]))

                self.assertEqual(stored.info_text, _jester_text(references, evil_count))
                self.assertEqual(
                    set(stored.info_parsed),
                    {
                        "jester_variant",
                        "callback_ledger_variant",
                        "callback_events",
                        "targets",
                        "evil_count",
                    },
                )
                self.assertEqual(stored.info_parsed["jester_variant"], JESTER_VARIANT)
                self.assertEqual(
                    stored.info_parsed["callback_ledger_variant"],
                    LEDGER_VARIANT,
                )
                self.assertEqual(stored.info_parsed["targets"], references)
                self.assertEqual(stored.info_parsed["evil_count"], evil_count)
                self.assertEqual(
                    stored.info_parsed["callback_events"],
                    [_strict_event_payload(
                        event,
                        activation_id=1,
                        evidence="single_callback_suffix",
                        callback_index=0,
                        dispatch_path="either",
                        generation=0,
                        boundary=6,
                    )],
                )

    def test_exact_public_grammar_rejects_every_near_miss(self):
        references = [3, 1, 2]
        mutations = (
            "Among:\n#1, #2, #3:\nThere is 0 Evil",
            "Among:\n#1, #2, #3:\nThere are 1 Evils",
            "Among #1, #2, #3: There are 2 Evils",
            "Among:\n#3, #1, #2:\nThere are 2 Evils",
            "Among:\n#1, #2, #4:\nThere are 2 Evils",
            "Among:\n#1, #2, #3\nThere are 2 Evils",
            "Among:\n#1, #2, #3:\nThere are 2 Evils.",
            "among:\n#1, #2, #3:\nthere are 2 evils",
        )
        for text in mutations:
            with self.subTest(text=text):
                event = {"desc": text, "targets": references}
                session = _strict_session()
                self.assertIsNone(_parse(_memory_jester([event]), session))

    def test_normal_result_requires_three_in_board_integer_reference_ids(self):
        malformed = (
            [1, 2],
            [1, 2, 3, 4],
            [0, 1, 2],
            [1, 2, 7],
            [1, 2, True],
        )
        for references in malformed:
            with self.subTest(references=references):
                text = "Among:\n#1, #2, #3:\nThere are 2 Evils"
                session = _strict_session()
                self.assertIsNone(_parse(
                    _memory_jester([{"desc": text, "targets": references}]),
                    session,
                ))

    def test_duplicate_display_ids_bind_to_distinct_pending_physical_clicks(self):
        physical_targets = [3, 1, 2]
        reference_ids = [2, 2, 5]
        session = _strict_session()
        _install_pending(session, physical_targets)

        stored = _parse_and_add(
            session,
            _memory_jester([_jester_event(reference_ids, 2)], remaining=0),
        )

        callback = stored.info_parsed["callback_events"][0]
        self.assertEqual(callback["targets"], physical_targets)
        self.assertEqual(callback["references"], reference_ids)
        self.assertEqual(stored.info_parsed["targets"], physical_targets)

    def test_rambler_requires_exact_newline_text_and_one_matching_reference(self):
        session = _strict_session()
        stored = _parse_and_add(
            session,
            _memory_jester([_interruption_event(5)]),
        )
        self.assertEqual(
            set(stored.info_parsed),
            {
                "jester_variant",
                "callback_ledger_variant",
                "callback_events",
                "shut_up_target",
            },
        )
        self.assertEqual(stored.info_parsed["shut_up_target"], 5)
        self.assertNotIn("targets", stored.info_parsed)
        self.assertNotIn("evil_count", stored.info_parsed)

        malformed = (
            {"desc": "#5 shut up!", "targets": [5]},
            {"desc": "#5\nShut up!", "targets": [5]},
            {"desc": "#5\nshut up!.", "targets": [5]},
            {"desc": "#5\nshut up!", "targets": []},
            {"desc": "#5\nshut up!", "targets": [4]},
            {"desc": "#5\nshut up!", "targets": [5, 4]},
        )
        for event in malformed:
            with self.subTest(event=event):
                self.assertIsNone(_parse(
                    _memory_jester([event]),
                    _strict_session(),
                ))


class JesterCurrentLedgerTests(unittest.TestCase):
    def _pending_session(self, targets=(3, 1, 2)):
        session = _strict_session()
        _install_pending(session, targets)
        return session

    def test_real_only_opaque_callback_waits_for_raw_public_result(self):
        foreign_real = {"desc": "#4 is Outcast", "targets": [4]}
        session = self._pending_session()
        self.assertIsNone(_parse(_memory_jester([foreign_real]), session))
        self.assertEqual(session.cards[0].info_parsed, {})
        self.assertIn(1, session.jester_pending_activations)

    def test_single_public_callback_is_either_even_with_click_provenance(self):
        targets = [3, 1, 2]
        session = self._pending_session(targets)
        stored = _parse_and_add(
            session,
            _memory_jester([_jester_event(targets, 2)]),
        )
        self.assertIn("callback_events", stored.info_parsed)
        event = stored.info_parsed["callback_events"][0]
        self.assertEqual(event["dispatch_path"], "either")
        self.assertEqual(event["activation_evidence"], "auto_use_click")
        self.assertEqual(event["activation_id"], 1)
        self.assertNotIn(1, session.jester_pending_activations)

    def test_real_then_raw_callbacks_share_activation_and_newest_alias(self):
        targets = [3, 1, 2]
        session = self._pending_session(targets)
        raw_events = [
            _jester_event(targets, 1),
            _jester_event(targets, 3),
        ]
        stored = _parse_and_add(session, _memory_jester(raw_events, remaining=-1))
        self.assertIn("callback_events", stored.info_parsed)
        events = stored.info_parsed["callback_events"]

        self.assertEqual([event["dispatch_path"] for event in events], ["real", "raw"])
        self.assertEqual([event["callback_index"] for event in events], [0, 1])
        self.assertEqual([event["activation_id"] for event in events], [1, 1])
        self.assertEqual(
            [event["activation_evidence"] for event in events],
            ["auto_use_click", "auto_use_click"],
        )
        self.assertEqual(stored.info_parsed["targets"], targets)
        self.assertEqual(stored.info_parsed["evil_count"], 3)
        self.assertEqual(stored.info_text, raw_events[-1]["desc"])

    def test_foreign_real_then_raw_jester_is_preserved_as_opaque_real(self):
        targets = [3, 1, 2]
        session = self._pending_session(targets)
        foreign = {"desc": "#4 is Outcast", "targets": [4]}
        raw = _jester_event(targets, 2)
        stored = _parse_and_add(
            session,
            _memory_jester([foreign, raw], remaining=-1),
        )
        self.assertIn("callback_events", stored.info_parsed)
        events = stored.info_parsed["callback_events"]
        self.assertEqual(
            [event["event_kind"] for event in events],
            ["opaque_real", "jester_result"],
        )
        self.assertEqual([event["dispatch_path"] for event in events], ["real", "raw"])
        self.assertEqual(events[0]["text"], foreign["desc"])
        self.assertEqual(events[0]["references"], foreign["targets"])

    def test_rambler_replaces_real_and_raw_independently_in_mixed_groups(self):
        targets = [3, 1, 2]
        cases = (
            (
                [_jester_event(targets, 1), _interruption_event(5)],
                ["jester_result", "rambler_interruption"],
                {"shut_up_target": 5},
            ),
            (
                [_interruption_event(5), _jester_event(targets, 2)],
                ["rambler_interruption", "jester_result"],
                {"targets": targets, "evil_count": 2},
            ),
            (
                [_interruption_event(4), _interruption_event(5)],
                ["rambler_interruption", "rambler_interruption"],
                {"shut_up_target": 5},
            ),
        )
        for raw_events, kinds, newest_alias in cases:
            with self.subTest(kinds=kinds):
                session = self._pending_session(targets)
                stored = _parse_and_add(
                    session,
                    _memory_jester(raw_events, remaining=-1),
                )
                self.assertIn("callback_events", stored.info_parsed)
                events = stored.info_parsed["callback_events"]
                self.assertEqual([event["event_kind"] for event in events], kinds)
                self.assertEqual(
                    [event["dispatch_path"] for event in events],
                    ["real", "raw"],
                )
                for key, value in newest_alias.items():
                    self.assertEqual(stored.info_parsed[key], value)
                expected_rambler = [
                    {
                        "speaker_position": 1,
                        "shut_up_target": event["targets"][0],
                    }
                    for event in raw_events
                    if len(event["targets"]) == 1
                    and event["desc"] == f"#{event['targets'][0]}\nshut up!"
                ]
                self.assertEqual(
                    session.rambler_shut_up_observations,
                    expected_rambler,
                )

    def test_append_only_multi_night_ledger_retains_every_result(self):
        events = [
            _jester_event([3, 1, 2], 0),
            _jester_event([4, 2, 1], 1),
            _jester_event([6, 2, 5], 3),
        ]
        session = _strict_session()
        first = _parse_and_add(session, _memory_jester(events[:1]))
        self.assertIn("callback_events", first.info_parsed)
        self.assertEqual(len(first.info_parsed["callback_events"]), 1)
        self.assertIn(1, session.used_abilities)

        reset = session.reset_after_night_abilities()
        self.assertEqual(reset, [1])
        self.assertNotIn(1, session.used_abilities)
        second = _parse_and_add(session, _memory_jester(events[:2]))

        session.reset_after_night_abilities(completed_nights=2)
        third = _parse_and_add(session, _memory_jester(events, remaining=0))
        ledger = third.info_parsed["callback_events"]
        self.assertEqual([event["activation_id"] for event in ledger], [1, 2, 3])
        self.assertEqual([event["reset_generation"] for event in ledger], [0, 1, 3])
        self.assertEqual(
            [event["activation_evidence"] for event in ledger],
            [
                "single_callback_suffix",
                "session_reset_generation",
                "session_reset_generation",
            ],
        )
        self.assertEqual(session.jester_reset_generations, {1: 3})
        self.assertEqual(second.info_parsed["callback_events"][-1]["evil_count"], 1)

    def test_delayed_second_callback_upgrades_same_activation_atomically(self):
        targets = [3, 1, 2]
        raw_events = [
            _jester_event(targets, 1),
            _jester_event(targets, 2),
        ]
        session = self._pending_session(targets)
        initial = _parse_and_add(
            session,
            _memory_jester(raw_events[:1], remaining=0),
        )
        self.assertIn("callback_events", initial.info_parsed)
        self.assertEqual(
            initial.info_parsed["callback_events"][0]["dispatch_path"],
            "either",
        )

        upgraded = _parse_and_add(
            session,
            _memory_jester(raw_events, remaining=-1),
        )
        self.assertIn("callback_events", upgraded.info_parsed)
        ledger = upgraded.info_parsed["callback_events"]
        self.assertEqual([event["activation_id"] for event in ledger], [1, 1])
        self.assertEqual([event["dispatch_path"] for event in ledger], ["real", "raw"])
        self.assertEqual(
            [event["activation_evidence"] for event in ledger],
            ["same_activation_extension", "same_activation_extension"],
        )
        self.assertEqual(
            [event["settled_reveal_count"] for event in ledger],
            [6, 6],
        )

    def test_stale_same_length_and_unreadable_history_never_extend_ledger(self):
        first = _jester_event([3, 1, 2], 1)
        changed = _jester_event([3, 1, 2], 2)
        session = _strict_session()
        stored = _parse_and_add(session, _memory_jester([first]))
        before = json.dumps(stored.info_parsed, sort_keys=True)

        stale = _parse(_memory_jester([first], remaining=0), session)
        self.assertIsNotNone(stale)
        session.add_card(stale)
        self.assertEqual(json.dumps(session.cards[0].info_parsed, sort_keys=True), before)

        mutation = _parse(_memory_jester([changed], remaining=0), session)
        self.assertIsNotNone(mutation)
        with self.assertRaisesRegex(ValueError, "prefix|same-length|history"):
            session.add_card(mutation)
        self.assertEqual(json.dumps(session.cards[0].info_parsed, sort_keys=True), before)

        self.assertIsNone(_parse(
            _memory_jester(None, clue=first["desc"], remaining=None),
            session,
        ))
        self.assertEqual(json.dumps(session.cards[0].info_parsed, sort_keys=True), before)

    def test_readable_empty_history_is_a_shorter_prefix_not_stale(self):
        session = _strict_session()
        _parse_and_add(
            session,
            _memory_jester([_jester_event([3, 1, 2], 1)], remaining=0),
        )
        empty = _parse(_memory_jester([], remaining=1), session)
        self.assertIsNotNone(empty)

        with self.assertRaisesRegex(ValueError, "prefix"):
            session.add_card(empty)


class JesterCurrentLifecycleTests(unittest.TestCase):
    def test_metadata_and_generation_advance_for_every_completed_night(self):
        definition = get_card("Jester")
        self.assertTrue(definition.activated_ability)
        self.assertTrue(definition.ability_resets_after_night)

        session = _strict_session(nights=2)
        self.assertTrue(hasattr(session, "jester_reset_generations"))
        self.assertTrue(hasattr(session, "jester_pending_activations"))
        self.assertEqual(session.jester_reset_generations, {})
        self.assertEqual(session.jester_pending_activations, {})
        # A Jester first observed after two completed Nights inherits that
        # generation before the next reset advances it.  Seed the established
        # per-actor baseline explicitly so this test isolates reset arithmetic
        # rather than discovery-time initialization.
        session.jester_reset_generations = {1: 2}
        session.used_abilities = [1]
        reset = session.reset_after_night_abilities(completed_nights=3)
        self.assertEqual(reset, [1])
        self.assertEqual(session.jester_reset_generations, {1: 5})

    def test_pending_click_blocks_night_reset_atomically(self):
        session = _strict_session()
        _install_pending(session, [3, 1, 2])
        session.used_abilities = [1]
        before = json.dumps({
            "used": session.used_abilities,
            "generations": session.jester_reset_generations,
            "pending": session.jester_pending_activations,
        }, sort_keys=True)

        with self.assertRaisesRegex(ValueError, "Jester.*auto_card|auto_card.*Jester"):
            session.reset_after_night_abilities()
        after = json.dumps({
            "used": session.used_abilities,
            "generations": session.jester_reset_generations,
            "pending": session.jester_pending_activations,
        }, sort_keys=True)
        self.assertEqual(after, before)

    def test_reload_preserves_generation_and_pending_click_provenance(self):
        session = _strict_session(nights=2)
        _install_pending(
            session,
            [3, 1, 2],
            activation_id=2,
            prior_count=1,
            generation=2,
            boundary=6,
        )
        strict = _strict_card(
            [_jester_event([3, 1, 2], 1)],
            [(0, 1, 1, "session_reset_generation", 1, 6)],
        )
        session.cards = [strict]

        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp) / "session.json")
            with redirect_stdout(StringIO()):
                session.save(path)
                loaded = GameSession.load(path)

        self.assertTrue(hasattr(loaded, "jester_reset_generations"))
        self.assertTrue(hasattr(loaded, "jester_pending_activations"))
        self.assertEqual(loaded.jester_reset_generations, {1: 2})
        self.assertEqual(
            loaded.jester_pending_activations,
            session.jester_pending_activations,
        )
        loaded_card = next(card for card in loaded.cards if card.position == 1)
        self.assertEqual(loaded_card.info_parsed, strict.info_parsed)

        with (
            patch("template_match.safe_click_at") as click,
            patch("memory_reader.MemoryReader") as reader,
        ):
            result = loaded.auto_use_ability(
                Action("use_ability", 1, [3, 1, 2], "Jester")
            )
        self.assertFalse(result["success"])
        self.assertIn("pending", result["error"].lower())
        self.assertIn("auto_card", result["error"])
        click.assert_not_called()
        reader.assert_not_called()


class JesterCurrentAutomationTests(unittest.TestCase):
    def _auto_use(self, session, snapshots, *, targets=(3, 1, 2), **patches):
        reader = _Reader(snapshots)
        with (
            patch("template_match.safe_click_at") as click,
            patch("game_loop.time.sleep"),
            patch("memory_reader.MemoryReader", return_value=reader),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            patch.object(DecisionLog, "log_ability_used"),
        ):
            result = session.auto_use_ability(
                Action("use_ability", 1, list(targets), "Jester")
            )
        return result, reader, click

    def test_auto_use_quiesces_real_then_raw_before_recording(self):
        targets = [3, 1, 2]
        baseline = _memory_jester([], remaining=1)
        first = _memory_jester([_jester_event(targets, 1)], remaining=0)
        dual = _memory_jester([
            _jester_event(targets, 1),
            _jester_event(targets, 3),
        ], remaining=-1)
        session = _strict_session()

        result, reader, click = self._auto_use(
            session,
            [baseline, first, dual, dual, dual],
            targets=targets,
        )
        self.assertTrue(result["success"], result["error"])
        self.assertGreaterEqual(click.call_count, 4)
        self.assertGreaterEqual(reader.index, 4)
        self.assertIn("callback_events", session.cards[0].info_parsed)
        ledger = session.cards[0].info_parsed["callback_events"]
        self.assertEqual(len(ledger), 2)
        self.assertEqual([event["dispatch_path"] for event in ledger], ["real", "raw"])
        self.assertTrue(all(
            event["activation_evidence"] == "auto_use_click"
            for event in ledger
        ))
        self.assertIn(1, session.used_abilities)
        self.assertNotIn(1, session.jester_pending_activations)

    def test_final_actor_identity_change_rejects_without_add_or_mark(self):
        targets = [3, 1, 2]
        baseline = _memory_jester([], remaining=1)
        changed_actor = _memory_jester(
            [_jester_event(targets, 2)],
            role="Dreamer",
            remaining=0,
        )
        session = _strict_session()
        with (
            patch.object(session, "add_card") as add,
            patch.object(session, "mark_ability_used") as mark,
        ):
            result, _, _ = self._auto_use(
                session,
                [baseline, changed_actor, changed_actor],
                targets=targets,
            )
        self.assertFalse(result["success"])
        add.assert_not_called()
        mark.assert_not_called()
        self.assertEqual(session.cards[0].info_parsed, {})

    def test_parse_and_add_complete_before_marking_used(self):
        targets = [3, 1, 2]
        baseline = _memory_jester([], remaining=1)
        resolved = _memory_jester([_jester_event(targets, 2)], remaining=0)
        session = _strict_session()
        order = []
        original_add = session.add_card
        original_mark = session.mark_ability_used

        def add(card, **kwargs):
            order.append("add")
            return original_add(card, **kwargs)

        def mark(position):
            order.append("mark")
            return original_mark(position)

        with (
            patch.object(session, "add_card", side_effect=add),
            patch.object(session, "mark_ability_used", side_effect=mark),
        ):
            result, _, _ = self._auto_use(
                session,
                [baseline, resolved, resolved, resolved],
                targets=targets,
            )
        self.assertTrue(result["success"], result["error"])
        self.assertEqual(order[-2:], ["add", "mark"])

        rejected = _strict_session()
        with (
            patch.object(rejected, "add_card", side_effect=ValueError("rejected")) as add,
            patch.object(rejected, "mark_ability_used") as mark,
        ):
            failure, _, _ = self._auto_use(
                rejected,
                [baseline, resolved, resolved, resolved],
                targets=targets,
            )
        self.assertFalse(failure["success"])
        self.assertIn("rejected", failure["error"])
        add.assert_called_once()
        mark.assert_not_called()

    def test_unreadable_stale_and_same_length_histories_never_resolve(self):
        targets = [3, 1, 2]
        old = _jester_event(targets, 1)
        changed = _jester_event(targets, 2)
        cases = (
            (
                _memory_jester(None, clue="", remaining=1),
                [_memory_jester([old], remaining=0)],
                "unreadable",
            ),
            (
                _memory_jester([old], remaining=1),
                [_memory_jester([old], remaining=0)],
                "stale",
            ),
            (
                _memory_jester([old], remaining=1),
                [_memory_jester([changed], remaining=0)],
                "same-length mutation",
            ),
        )
        for baseline, candidates, label in cases:
            with self.subTest(label=label):
                session = _strict_session()
                with (
                    patch.object(session, "add_card") as add,
                    patch.object(session, "mark_ability_used") as mark,
                ):
                    result, _, _ = self._auto_use(
                        session,
                        [baseline, *candidates],
                        targets=targets,
                    )
                self.assertFalse(result["success"])
                add.assert_not_called()
                mark.assert_not_called()

    def test_preflight_rejects_native_history_shorter_than_persisted_ledger(self):
        targets = [3, 1, 2]
        raw_events = [
            _jester_event(targets, 1),
            _jester_event(targets, 2),
        ]
        session = _strict_session(nights=1)
        session.cards = [_strict_card(
            raw_events,
            [
                (0, 1, 1, "single_callback_suffix", 0, 6),
                (1, 2, 2, "session_reset_generation", 1, 6),
            ],
        )]
        session.jester_reset_generations = {1: 2}
        session.jester_pending_activations = {}
        baseline = _memory_jester(raw_events[:1], remaining=1)

        result, _, click = self._auto_use(
            session,
            [baseline],
            targets=targets,
        )
        self.assertFalse(result["success"])
        click.assert_not_called()
        self.assertRegex(result["error"], "shorter|prefix|history")

    def test_auto_card_ignores_post_night_stale_history_then_appends_new_suffix(self):
        first = _jester_event([3, 1, 2], 1)
        second = _jester_event([4, 2, 1], 3)
        session = _strict_session()
        stored = _parse_and_add(session, _memory_jester([first]))
        first_payload = copy.deepcopy(stored.info_parsed)
        session.reset_after_night_abilities()

        stale_output = _run_auto_card(
            session,
            _memory_jester([first], remaining=1),
        )
        self.assertIn("Entered 0 cards", stale_output)
        self.assertEqual(session.cards[0].info_parsed, first_payload)
        self.assertNotIn(1, session.used_abilities)

        appended_output = _run_auto_card(
            session,
            _memory_jester([first, second], remaining=0),
        )
        self.assertIn("updated #1 Jester", appended_output)
        self.assertIn("callback_events", session.cards[0].info_parsed)
        ledger = session.cards[0].info_parsed["callback_events"]
        self.assertEqual([event["activation_id"] for event in ledger], [1, 2])
        self.assertEqual([event["reset_generation"] for event in ledger], [0, 1])
        self.assertIn(1, session.used_abilities)

    def test_auto_card_promotes_current_marker_only_placeholder(self):
        session = _strict_session()
        session.cards = [CardInfo(
            1,
            "Jester",
            info_parsed={"jester_variant": JESTER_VARIANT},
        )]
        event = _jester_event([3, 1, 2], 2)

        output = _run_auto_card(
            session,
            _memory_jester([event], remaining=0),
        )

        self.assertIn("updated #1 Jester", output)
        self.assertEqual(
            session.cards[0].info_parsed["callback_ledger_variant"],
            LEDGER_VARIANT,
        )
        self.assertIn(1, session.used_abilities)

    def test_reset_available_same_length_mutation_surfaces_recovery(self):
        first = _jester_event([3, 1, 2], 1)
        session = _strict_session()
        stored = _parse_and_add(session, _memory_jester([first]))
        original = copy.deepcopy(stored.info_parsed)
        session.reset_after_night_abilities()
        mutated = _jester_event([3, 1, 2], 2)

        output = _run_auto_card(
            session,
            _memory_jester([mutated], remaining=1),
        )

        self.assertIn("RECOVERY", output)
        self.assertIn("prefix", output)
        self.assertEqual(session.cards[0].info_parsed, original)
        self.assertNotIn(1, session.used_abilities)

    def test_reset_available_unowned_history_keeps_current_marker(self):
        session = _strict_session(nights=1)
        session.cards = [CardInfo(
            1,
            "Jester",
            info_parsed={"jester_variant": JESTER_VARIANT},
        )]

        output = _run_auto_card(
            session,
            _memory_jester(
                [_jester_event([3, 1, 2], 2)],
                remaining=1,
            ),
        )

        self.assertNotIn("callback_events", session.cards[0].info_parsed)
        self.assertNotIn(1, session.used_abilities)
        self.assertNotIn("RECOVERY", output)

    def test_pending_auto_card_capture_marks_used_before_counter_decrements(self):
        session = _strict_session()
        session.cards = [CardInfo(
            1,
            "Jester",
            info_parsed={"jester_variant": JESTER_VARIANT},
        )]
        _install_pending(session, [3, 1, 2])

        _run_auto_card(
            session,
            _memory_jester(
                [_jester_event([3, 1, 2], 1)],
                remaining=1,
            ),
        )

        self.assertNotIn(1, session.jester_pending_activations)
        self.assertIn(1, session.used_abilities)

    def test_actor_click_exception_retains_persisted_pending_token(self):
        session = _strict_session()
        baseline = _memory_jester([], remaining=1)
        reader = _Reader([baseline])
        with (
            patch("memory_reader.MemoryReader", return_value=reader),
            patch("template_match.safe_click_at", side_effect=RuntimeError("click")),
            patch("game_loop.time.sleep"),
            patch.object(session, "save"),
        ):
            result = session.auto_use_ability(
                Action("use_ability", 1, [3, 1, 2], "Jester")
            )

        self.assertFalse(result["success"])
        self.assertIn(1, session.jester_pending_activations)
        self.assertIn("retained", result["error"])

    def test_partial_current_schema_cannot_fall_through_legacy_jester_parser(self):
        memory = _memory_jester([{
            "desc": "2 are evil",
            "targets": [3, 1, 2],
        }])
        del memory["pickable_uses_remaining"]

        self.assertIsNone(_parse(memory, _strict_session()))

    def test_state_machine_routes_jester_through_strict_session_automation(self):
        session = _strict_session()
        machine = GameStateMachine(
            session=session,
            monitor=_NoResultMonitor(_memory_jester([], remaining=1)),
        )
        machine.phase = GamePhase.ABILITY_USE
        machine._pending_ability = (1, [3, 1, 2], "Jester", None)
        with (
            patch.object(
                session,
                "auto_use_ability",
                return_value={
                    "success": True,
                    "info_parsed": {"strict": True},
                    "error": None,
                },
            ) as use,
            patch.object(machine, "_snapshot_counts"),
            patch("template_match.safe_click_at"),
            patch("template_match.fast_click_at"),
            patch("time.sleep"),
        ):
            machine._do_ability_use()

        use.assert_called_once()
        action = use.call_args.args[0]
        self.assertEqual(action.position, 1)
        self.assertEqual(action.targets, [3, 1, 2])
        self.assertEqual(action.ability_name, "Jester")
        self.assertEqual(machine.phase, GamePhase.SOLVING)


if __name__ == "__main__":
    unittest.main()
