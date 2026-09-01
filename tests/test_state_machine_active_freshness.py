"""Fresh-event gates for the generic Dreamer state-machine ability path.

Current Jester now routes through GameSession's ordered callback ledger; its
state-machine integration lives in test_jester_current.py.
"""

import copy
import unittest
from unittest.mock import patch

from game_loop import DecisionLog, GameSession
from solver import CardInfo
from state_machine import GamePhase, GameStateMachine


def _memory_card(role, events, *, clue=None, remaining=1, **extra):
    events = copy.deepcopy(events)
    if clue is None:
        clue = events[-1]["desc"] if events else None
    return {
        "position": 1,
        "true_role": role,
        "current_role": role,
        "disguise": role,
        "clue_text": clue,
        "acted_infos": events,
        "pickable_uses_remaining": remaining,
        "act_output_enabled": True,
        "pickable_available": (
            remaining > 0 if type(remaining) is int else None
        ),
        "uses": remaining,
        "ability_used": True,
        **extra,
    }


class _Monitor:
    def __init__(self, baseline, candidates, log=None):
        self.baseline = copy.deepcopy(baseline)
        self.candidates = [copy.deepcopy(card) for card in candidates]
        self.log = log if log is not None else []

    def is_healthy(self):
        return True

    def get_board(self):
        self.log.append("baseline")
        return [copy.deepcopy(self.baseline)]

    def wait_for(self, predicate, timeout, min_delay):
        self.log.append(("wait", timeout, min_delay))
        for card in self.candidates:
            if predicate([copy.deepcopy(card)]):
                return True
        return False


class GenericActiveFreshnessTests(unittest.TestCase):
    @staticmethod
    def _machine(role, monitor):
        session = GameSession(4, 1)
        session.cards = [CardInfo(1, role)]
        if role == "Jester":
            session.reveal_order = [1]
        machine = GameStateMachine(session=session, monitor=monitor)
        machine.phase = GamePhase.ABILITY_USE
        targets = [2, 3] if role == "Dreamer" else [2, 3, 4]
        machine._pending_ability = (1, targets, role, None)
        return session, machine

    def test_dreamer_snapshots_before_click_and_stores_only_append(self):
        for role in ("Dreamer",):
            with self.subTest(role=role):
                targets = [2, 3] if role == "Dreamer" else [2, 3, 4]
                old = {"desc": "old result", "targets": list(targets)}
                new = {"desc": "new result", "targets": list(targets)}
                log = []
                baseline = _memory_card(role, [old], remaining=1)
                resolved = _memory_card(
                    role,
                    [old, new],
                    remaining=0 if role == "Dreamer" else 1,
                )
                monitor = _Monitor(baseline, [resolved], log)
                session, machine = self._machine(role, monitor)
                parsed = CardInfo(
                    1,
                    role,
                    info_text="new result",
                    info_parsed={"verified": True},
                )
                calls = []

                with (
                    patch(
                        "template_match.safe_click_at",
                        side_effect=lambda *_args: log.append("actor_click"),
                    ),
                    patch("template_match.fast_click_at"),
                    patch("time.sleep"),
                    patch(
                        "game_loop._parse_clue_from_memory",
                        return_value=parsed,
                    ) as parse,
                    patch.object(
                        session,
                        "add_card",
                        side_effect=lambda _card, **_kwargs: calls.append("add"),
                    ),
                    patch.object(
                        session,
                        "mark_ability_used",
                        side_effect=lambda _pos: calls.append("mark"),
                    ),
                    patch.object(session, "save"),
                    patch.object(DecisionLog, "log_card"),
                ):
                    machine._do_ability_use()

                self.assertLess(log.index("baseline"), log.index("actor_click"))
                self.assertEqual(calls, ["add", "mark"])
                self.assertEqual(machine.phase, GamePhase.SOLVING)
                self.assertEqual(
                    parse.call_args.args[0]["acted_infos"],
                    [old, new],
                )

    def test_retained_history_flags_and_counter_decrease_do_not_resolve(self):
        old = {"desc": "retained", "targets": [2, 3]}
        baseline = _memory_card("Dreamer", [old], remaining=1)
        stale = _memory_card(
            "Dreamer",
            [old],
            remaining=0,
            ability_used=True,
            act_output_enabled=True,
        )
        session, machine = self._machine(
            "Dreamer",
            _Monitor(baseline, [stale]),
        )

        with (
            patch("template_match.safe_click_at"),
            patch("template_match.fast_click_at"),
            patch("time.sleep"),
            patch("game_loop._parse_clue_from_memory") as parse,
            patch.object(session, "add_card") as add,
            patch.object(session, "mark_ability_used") as mark,
        ):
            machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("counter decreased", machine._needs_human_reason)
        self.assertIn("no coherent new acted-info", machine._needs_human_reason)
        parse.assert_not_called()
        add.assert_not_called()
        mark.assert_not_called()

    def test_saved_text_mismatch_and_nonprefix_history_do_not_resolve(self):
        old = {"desc": "old", "targets": [2, 3]}
        new = {"desc": "new", "targets": [2, 3]}
        bad_cases = (
            _memory_card("Dreamer", [old, new], clue="old", remaining=0),
            _memory_card(
                "Dreamer",
                [{"desc": "replacement", "targets": [2, 3]}, new],
                remaining=0,
            ),
        )
        for candidate in bad_cases:
            with self.subTest(candidate=candidate):
                baseline = _memory_card("Dreamer", [old], remaining=1)
                session, machine = self._machine(
                    "Dreamer",
                    _Monitor(baseline, [candidate]),
                )
                with (
                    patch("template_match.safe_click_at"),
                    patch("template_match.fast_click_at"),
                    patch("time.sleep"),
                    patch("game_loop._parse_clue_from_memory") as parse,
                    patch.object(session, "add_card") as add,
                    patch.object(session, "mark_ability_used") as mark,
                ):
                    machine._do_ability_use()
                self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
                parse.assert_not_called()
                add.assert_not_called()
                mark.assert_not_called()

    def test_memory_actor_identity_must_match_before_click(self):
        baseline = _memory_card("Jester", [], clue="", remaining=1)
        session, machine = self._machine(
            "Dreamer",
            _Monitor(baseline, []),
        )

        with (
            patch("template_match.safe_click_at") as click,
            patch.object(session, "mark_ability_used") as mark,
        ):
            machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("pre-click memory shows jester", machine._needs_human_reason)
        click.assert_not_called()
        mark.assert_not_called()

    def test_new_event_references_must_match_clicked_targets(self):
        old = {"desc": "old", "targets": [2, 3]}
        mismatched = {"desc": "new", "targets": [3, 4]}
        session, machine = self._machine(
            "Dreamer",
            _Monitor(
                _memory_card("Dreamer", [old], remaining=1),
                [_memory_card("Dreamer", [old, mismatched], remaining=0)],
            ),
        )

        with (
            patch("template_match.safe_click_at"),
            patch("template_match.fast_click_at"),
            patch("time.sleep"),
            patch("game_loop._parse_clue_from_memory") as parse,
            patch.object(session, "add_card") as add,
            patch.object(session, "mark_ability_used") as mark,
        ):
            machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        parse.assert_not_called()
        add.assert_not_called()
        mark.assert_not_called()

    def test_jester_requires_exactly_three_distinct_targets_before_click(self):
        baseline = _memory_card("Jester", [], clue="", remaining=1)
        for targets in ([2, 3], [2, 3, 4, 1], [2, 2, 3]):
            with self.subTest(targets=targets):
                session, machine = self._machine(
                    "Jester",
                    _Monitor(baseline, []),
                )
                machine._pending_ability = (1, targets, "Jester", None)
                with patch("template_match.safe_click_at") as click:
                    machine._do_ability_use()
                self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
                click.assert_not_called()

    def test_unreadable_baseline_fails_before_actor_click(self):
        baseline = _memory_card("Jester", [], clue=None)
        baseline["acted_infos"] = None
        session, machine = self._machine(
            "Jester",
            _Monitor(baseline, []),
        )

        with (
            patch("template_match.safe_click_at") as click,
            patch("template_match.fast_click_at"),
            patch("time.sleep"),
            patch.object(session, "mark_ability_used") as mark,
        ):
            machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("pre-click acted-info history", machine._needs_human_reason)
        click.assert_not_called()
        mark.assert_not_called()

    def test_unreadable_or_spent_budget_fails_before_actor_click(self):
        for remaining in (None, 0, -1):
            with self.subTest(remaining=remaining):
                baseline = _memory_card(
                    "Jester",
                    [],
                    clue="",
                    remaining=remaining,
                )
                session, machine = self._machine(
                    "Jester",
                    _Monitor(baseline, []),
                )
                with (
                    patch("template_match.safe_click_at") as click,
                    patch.object(session, "mark_ability_used") as mark,
                ):
                    machine._do_ability_use()
                self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
                click.assert_not_called()
                mark.assert_not_called()

    def test_parse_failure_does_not_mark_ability_used(self):
        old = {"desc": "old", "targets": [2, 3]}
        new = {"desc": "new", "targets": [2, 3]}
        session, machine = self._machine(
            "Dreamer",
            _Monitor(
                _memory_card("Dreamer", [old], remaining=1),
                [_memory_card("Dreamer", [old, new], remaining=0)],
            ),
        )

        with (
            patch("template_match.safe_click_at"),
            patch("template_match.fast_click_at"),
            patch("time.sleep"),
            patch("game_loop._parse_clue_from_memory", return_value=None),
            patch.object(session, "add_card") as add,
            patch.object(session, "mark_ability_used") as mark,
        ):
            machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        add.assert_not_called()
        mark.assert_not_called()

    def test_add_failure_does_not_mark_ability_used(self):
        old = {"desc": "old", "targets": [2, 3]}
        new = {"desc": "new", "targets": [2, 3]}
        session, machine = self._machine(
            "Dreamer",
            _Monitor(
                _memory_card("Dreamer", [old], remaining=1),
                [_memory_card("Dreamer", [old, new], remaining=0)],
            ),
        )
        parsed = CardInfo(1, "Dreamer", info_parsed={"verified": True})

        with (
            patch("template_match.safe_click_at"),
            patch("template_match.fast_click_at"),
            patch("time.sleep"),
            patch(
                "game_loop._parse_clue_from_memory",
                return_value=parsed,
            ),
            patch.object(
                session,
                "add_card",
                side_effect=ValueError("rejected result"),
            ) as add,
            patch.object(session, "mark_ability_used") as mark,
        ):
            machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.NEEDS_HUMAN)
        self.assertIn("rejected result", machine._needs_human_reason)
        add.assert_called_once_with(parsed, mark_active_result=False)
        mark.assert_not_called()


if __name__ == "__main__":
    unittest.main()
