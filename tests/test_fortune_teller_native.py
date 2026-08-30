"""Current-build Fortune Teller solver/live boundary regressions."""

import unittest
from unittest.mock import patch

import memory_reader as memory_module
from game_loop import (
    DecisionLog,
    GameSession,
    _parse_clue_from_memory,
    _parse_fortune_teller_result_from_memory,
    card_fortune_teller,
    card_shut_up,
)
from knowledge_base import get_card
from solver import (
    CardInfo,
    DeckComposition,
    FORTUNE_TELLER_RULE_VERSION,
    GameState,
    Scenario,
    SolverResult,
)
from state_machine import GamePhase, GameStateMachine
from strategy import Action, recommend_abilities


def _event(first: int, second: int, has_evil: bool) -> dict:
    text = (
        f"Is #{first} or #{second} Evil?: "
        f"{'True' if has_evil else 'False'}"
    )
    return {"desc": text, "targets": [first, second]}


def _memory_card(
    events: list[dict],
    *,
    position: int = 1,
    clue: str | None = None,
) -> dict:
    if clue is None:
        clue = events[-1]["desc"] if events else ""
    return {
        "position": position,
        "true_role": "Fortune Teller",
        "clue_text": clue,
        "acted_infos": [dict(event) for event in events],
        "uses": 1 if events else 0,
        "ability_used": bool(events),
    }


class FortuneTellerMemoryTests(unittest.TestCase):
    def test_exact_native_result_accepts_reverse_click_order_but_stores_sorted_refs(self):
        parsed, error = _parse_fortune_teller_result_from_memory(
            _memory_card([_event(2, 5, True)]),
            expected_targets=[5, 2],
            n_cards=6,
        )

        self.assertIsNone(error)
        self.assertEqual(parsed.info_text, "Is #2 or #5 Evil?: True")
        self.assertEqual(parsed.info_parsed["targets"], [2, 5])
        self.assertTrue(parsed.info_parsed["has_evil"])
        self.assertEqual(
            parsed.info_parsed["observations"],
            [{
                "targets": [2, 5],
                "has_evil": True,
                "text": "Is #2 or #5 Evil?: True",
            }],
        )

    def test_reset_history_uses_newest_event_and_preserves_every_normal_result(self):
        events = [_event(1, 3, False), _event(2, 5, True)]
        parsed, error = _parse_fortune_teller_result_from_memory(
            _memory_card(events),
            expected_targets=[2, 5],
            n_cards=6,
        )

        self.assertIsNone(error)
        self.assertEqual(parsed.info_parsed["targets"], [2, 5])
        self.assertTrue(parsed.info_parsed["has_evil"])
        self.assertEqual(
            parsed.info_parsed["observations"],
            [
                {
                    "targets": [1, 3],
                    "has_evil": False,
                    "text": "Is #1 or #3 Evil?: False",
                },
                {
                    "targets": [2, 5],
                    "has_evil": True,
                    "text": "Is #2 or #5 Evil?: True",
                },
            ],
        )

    def test_rejects_malformed_refs_speech_order_and_savedact(self):
        malformed = [
            (_memory_card([{"desc": "Is #2 or #3 Evil?: False", "targets": [2]}]), "exactly two"),
            (_memory_card([{"desc": "Is #2 or #2 Evil?: False", "targets": [2, 2]}]), "distinct"),
            (_memory_card([{"desc": "Is #3 or #2 Evil?: False", "targets": [3, 2]}]), "ascending"),
            (_memory_card([{"desc": "Is #2 or #7 Evil?: False", "targets": [2, 7]}]), "within"),
            (_memory_card([{"desc": "Is #2 or #4 Evil?: True", "targets": [2, 3]}]), "target mismatch"),
            (_memory_card([{"desc": "is #2 or #3 Evil?: True", "targets": [2, 3]}]), "Unrecognized"),
            (_memory_card([_event(2, 3, True)], clue="Is #2 or #3 Evil?: False"), "savedAct"),
        ]
        for card, expected_error in malformed:
            with self.subTest(card=card):
                parsed, error = _parse_fortune_teller_result_from_memory(
                    card,
                    n_cards=6,
                )
                self.assertIsNone(parsed)
                self.assertIn(expected_error, error)

    def test_strict_auto_card_is_version_gated_and_legacy_reverse_order_survives(self):
        memory = _memory_card([
            {"desc": "Is #5 or #1 Evil?: True", "targets": [5, 1]},
        ])

        current = _parse_clue_from_memory(
            memory,
            n_cards=6,
            fortune_teller_rule_version=FORTUNE_TELLER_RULE_VERSION,
        )
        legacy = _parse_clue_from_memory(memory, n_cards=6)

        self.assertIsNone(current)
        self.assertEqual(legacy.info_parsed, {
            "targets": [5, 1],
            "has_evil": True,
        })

    def test_memory_reader_keeps_more_than_ten_events_in_list_order(self):
        reader = memory_module.MemoryReader()
        character = 0x10000
        acted_list = 0x20000
        items = 0x30000
        info_ptrs = [0x40000 + index * 0x100 for index in range(11)]
        desc_ptrs = [0x50000 + index * 0x100 for index in range(11)]

        pointers = {
            character + memory_module.CHAR_ACTED_INFOS_OFFSET: acted_list,
            acted_list + memory_module.LIST_ITEMS_OFFSET: items,
        }
        for index, (info_ptr, desc_ptr) in enumerate(zip(info_ptrs, desc_ptrs)):
            pointers[
                items + memory_module.ARRAY_FIRST_ELEMENT_OFFSET + index * 8
            ] = info_ptr
            pointers[info_ptr + memory_module.ACTED_INFO_DESC_OFFSET] = desc_ptr
            pointers[info_ptr + memory_module.ACTED_INFO_CHARS_OFFSET] = 0

        reader._read_ptr = lambda address: pointers.get(address, 0)
        reader._read_i32 = lambda address: (
            11 if address == acted_list + memory_module.LIST_SIZE_OFFSET else 0
        )
        reader._read_string = lambda pointer: f"event-{desc_ptrs.index(pointer)}"

        history = reader._read_acted_infos(character)

        self.assertEqual(len(history), 11)
        self.assertEqual(history[0]["desc"], "event-0")
        self.assertEqual(history[-1]["desc"], "event-10")

    def test_memory_reader_rejects_runaway_history_size_without_truncating(self):
        reader = memory_module.MemoryReader()
        character = 0x10000
        acted_list = 0x20000
        items = 0x30000
        pointers = {
            character + memory_module.CHAR_ACTED_INFOS_OFFSET: acted_list,
            acted_list + memory_module.LIST_ITEMS_OFFSET: items,
        }
        reader._read_ptr = lambda address: pointers.get(address, 0)
        reader._read_i32 = lambda address: 4097

        self.assertEqual(reader._read_acted_infos(character), [])

    def test_memory_reader_rejects_null_slot_inside_reported_history(self):
        reader = memory_module.MemoryReader()
        character = 0x10000
        acted_list = 0x20000
        items = 0x30000
        first_info = 0x40000
        first_desc = 0x50000
        pointers = {
            character + memory_module.CHAR_ACTED_INFOS_OFFSET: acted_list,
            acted_list + memory_module.LIST_ITEMS_OFFSET: items,
            items + memory_module.ARRAY_FIRST_ELEMENT_OFFSET: first_info,
            items + memory_module.ARRAY_FIRST_ELEMENT_OFFSET + 8: 0,
            first_info + memory_module.ACTED_INFO_DESC_OFFSET: first_desc,
            first_info + memory_module.ACTED_INFO_CHARS_OFFSET: 0,
        }
        reader._read_ptr = lambda address: pointers.get(address, 0)
        reader._read_i32 = lambda address: (
            3 if address == acted_list + memory_module.LIST_SIZE_OFFSET else 0
        )
        reader._read_string = lambda pointer: "first event"

        self.assertEqual(reader._read_acted_infos(character), [])


class FortuneTellerSessionTests(unittest.TestCase):
    def test_manual_current_result_is_normalized_and_reuse_appends_history(self):
        session = GameSession(5, 1)
        session.add_card(card_fortune_teller(1, [4, 2], False))

        first = session.cards[0]
        self.assertEqual(first.info_parsed["targets"], [2, 4])
        self.assertEqual(first.info_text, "Is #2 or #4 Evil?: False")
        self.assertEqual(session.reset_after_night_abilities(), [1])

        session.add_card(card_fortune_teller(1, [3, 5], True))
        latest = session.cards[0]
        self.assertEqual(latest.info_parsed["targets"], [3, 5])
        self.assertEqual(
            latest.info_parsed["observations"],
            [
                {
                    "targets": [2, 4],
                    "has_evil": False,
                    "text": "Is #2 or #4 Evil?: False",
                },
                {
                    "targets": [3, 5],
                    "has_evil": True,
                    "text": "Is #3 or #5 Evil?: True",
                },
            ],
        )

    def test_explicit_empty_current_history_is_rejected_atomically(self):
        malformed_infos = [
            {
                "targets": [1, 3],
                "has_evil": False,
                "observations": [],
            },
        ]
        for info in malformed_infos:
            with self.subTest(info=info):
                session = GameSession(5, 1)
                malformed = CardInfo(
                    2,
                    "Fortune Teller",
                    info_text="Is #1 or #3 Evil?: False",
                    info_parsed=info,
                )

                with self.assertRaises(ValueError):
                    session.add_card(malformed)

                self.assertEqual(session.cards, [])
                self.assertEqual(session.reveal_order, [])
                self.assertEqual(session.used_abilities, [])

    def test_bare_empty_current_history_is_rejected_atomically(self):
        session = GameSession(5, 1)
        malformed = CardInfo(
            2,
            "Fortune Teller",
            info_parsed={"observations": []},
        )

        with self.assertRaisesRegex(ValueError, "empty observations require"):
            session.add_card(malformed)

        self.assertEqual(session.cards, [])
        self.assertEqual(session.reveal_order, [])
        self.assertEqual(session.used_abilities, [])

    def test_normal_result_and_rambler_interruption_are_rejected_atomically(self):
        session = GameSession(5, 1)
        malformed = CardInfo(
            2,
            "Fortune Teller",
            info_parsed={
                "targets": [3, 1],
                "has_evil": False,
                "shut_up_target": 4,
            },
        )

        with self.assertRaisesRegex(ValueError, "cannot combine"):
            session.add_card(malformed)

        self.assertEqual(session.cards, [])
        self.assertEqual(session.reveal_order, [])
        self.assertEqual(session.used_abilities, [])
        self.assertEqual(malformed.info_parsed["targets"], [3, 1])
        self.assertNotIn("observations", malformed.info_parsed)
        self.assertFalse(malformed.info_text)

    def test_rambler_interruption_keeps_prior_normal_history_and_next_use_appends(self):
        session = GameSession(5, 1)
        session.add_card(card_fortune_teller(1, [2, 4], False))
        session.reset_after_night_abilities()

        session.add_card(card_shut_up(
            1,
            "Fortune Teller",
            3,
            info_text="#3 shut up!",
        ))
        interrupted = session.cards[0]
        self.assertNotIn("targets", interrupted.info_parsed)
        self.assertEqual(interrupted.info_parsed["shut_up_target"], 3)
        self.assertEqual(len(interrupted.info_parsed["observations"]), 1)
        self.assertEqual(
            session.rambler_shut_up_observations,
            [{"speaker_position": 1, "shut_up_target": 3}],
        )

        session.reset_after_night_abilities()
        session.add_card(card_fortune_teller(1, [3, 5], True))
        self.assertEqual(len(session.cards[0].info_parsed["observations"]), 2)
        self.assertEqual(
            session.rambler_shut_up_observations,
            [{"speaker_position": 1, "shut_up_target": 3}],
        )

    def test_public_knowledge_and_night_reset_include_fortune_teller(self):
        definition = get_card("Fortune Teller")
        self.assertTrue(definition.activated_ability)
        self.assertTrue(definition.ability_resets_after_night)

        session = GameSession(4, 1)
        session.cards.extend([CardInfo(1, "Fortune Teller"), CardInfo(2, "Slayer")])
        session.used_abilities = [1, 2]
        self.assertEqual(session.reset_after_night_abilities(), [1])
        self.assertEqual(session.used_abilities, [2])


class FortuneTellerAutomationTests(unittest.TestCase):
    def test_self_and_unused_active_target_pass_preflight_and_record_result(self):
        session = GameSession(4, 1)
        session.cards.extend([
            CardInfo(1, "Fortune Teller"),
            CardInfo(2, "Dreamer"),
        ])
        action = Action("use_ability", 1, [1, 2], "Fortune Teller")
        before = _memory_card([], position=1)
        after = _memory_card([_event(1, 2, False)], position=1)

        class Reader:
            def __init__(self):
                self.reads = 0

            def open(self):
                return True

            def read_board(self):
                self.reads += 1
                return [before if self.reads == 1 else after]

            def close(self):
                return None

        with (
            patch("template_match.safe_click_at"),
            patch("game_loop.time.sleep"),
            patch("memory_reader.MemoryReader", return_value=Reader()),
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_card"),
            patch.object(DecisionLog, "log_ability_used"),
        ):
            result = session.auto_use_ability(action)

        self.assertTrue(result["success"], result["error"])
        self.assertEqual(result["info_parsed"]["targets"], [1, 2])
        self.assertIn(1, session.used_abilities)

    def test_preflight_requires_exactly_two_distinct_targets(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(1, "Fortune Teller"))

        for targets in [[], [2], [2, 2], [2, 3, 4]]:
            with self.subTest(targets=targets):
                result = session.auto_use_ability(
                    Action("use_ability", 1, targets, "Fortune Teller")
                )
                self.assertIn("exactly 2 distinct", result["error"])

    def test_unchanged_old_latest_event_is_not_accepted_after_reset(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(1, "Fortune Teller"))
        stale = _memory_card([_event(2, 3, False)])

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [stale]

            def close(self):
                return None

        with (
            patch("template_match.safe_click_at"),
            patch("game_loop.time.sleep"),
            patch("memory_reader.MemoryReader", return_value=Reader()),
        ):
            result = session.auto_use_ability(
                Action("use_ability", 1, [2, 3], "Fortune Teller")
            )

        self.assertFalse(result["success"])
        self.assertIn("new or changed latest", result["error"])
        self.assertNotIn(1, session.used_abilities)

    def test_prior_same_pair_fails_before_click_when_pre_history_is_unreadable(self):
        session = GameSession(4, 1)
        session.add_card(card_fortune_teller(1, [2, 3], False))
        session.reset_after_night_abilities()
        stale = _memory_card([_event(2, 3, False)])
        unreadable = _memory_card([], clue=stale["clue_text"])

        class Reader:
            def __init__(self):
                self.reads = 0

            def open(self):
                return True

            def read_board(self):
                self.reads += 1
                return [unreadable if self.reads == 1 else stale]

            def close(self):
                return None

        reader = Reader()
        with (
            patch("template_match.safe_click_at") as click,
            patch("memory_reader.MemoryReader", return_value=reader),
        ):
            result = session.auto_use_ability(
                Action("use_ability", 1, [2, 3], "Fortune Teller")
            )

        self.assertFalse(result["success"])
        self.assertIn("prior active evidence", result["error"])
        self.assertIn("no readable newest", result["error"])
        click.assert_not_called()
        self.assertEqual(reader.reads, 1)
        self.assertNotIn(1, session.used_abilities)

    def test_prior_same_pair_fails_before_click_on_stale_shorter_history(self):
        session = GameSession(4, 1)
        session.add_card(card_fortune_teller(1, [2, 3], False))
        session.reset_after_night_abilities()
        session.add_card(card_fortune_teller(1, [2, 3], False))
        session.reset_after_night_abilities()
        event = _event(2, 3, False)
        stale_short = _memory_card([event])
        recovered_old = _memory_card([event, event])

        class Reader:
            def __init__(self):
                self.reads = 0

            def open(self):
                return True

            def read_board(self):
                self.reads += 1
                return [stale_short if self.reads == 1 else recovered_old]

            def close(self):
                return None

        reader = Reader()
        with (
            patch("template_match.safe_click_at") as click,
            patch("memory_reader.MemoryReader", return_value=reader),
        ):
            result = session.auto_use_ability(
                Action("use_ability", 1, [2, 3], "Fortune Teller")
            )

        self.assertFalse(result["success"])
        self.assertIn("shorter than the local minimum", result["error"])
        click.assert_not_called()
        self.assertEqual(reader.reads, 1)
        self.assertNotIn(1, session.used_abilities)


class FortuneTellerStateMachineTests(unittest.TestCase):
    def test_autonomous_phase_routes_legal_picker_targets_through_strict_session(self):
        session = GameSession(4, 1)
        session.cards.extend([
            CardInfo(1, "Fortune Teller"),
            CardInfo(2, "Dreamer"),
        ])
        machine = GameStateMachine(session=session, monitor=None)
        machine.phase = GamePhase.ABILITY_USE
        machine._pending_ability = (1, [1, 2], "Fortune Teller", None)

        with (
            patch.object(
                session,
                "auto_use_ability",
                return_value={
                    "success": True,
                    "info_parsed": {"targets": [1, 2], "has_evil": False},
                    "error": None,
                },
            ) as auto_use,
            patch.object(
                machine,
                "_has_active_ability",
                side_effect=AssertionError("generic target filter ran"),
            ),
        ):
            machine._do_ability_use()

        self.assertEqual(machine.phase, GamePhase.SOLVING)
        action = auto_use.call_args.args[0]
        self.assertEqual(action.action_type, "use_ability")
        self.assertEqual(action.position, 1)
        self.assertEqual(action.targets, [1, 2])
        self.assertEqual(action.ability_name, "Fortune Teller")


class FortuneTellerStrategyTests(unittest.TestCase):
    def test_candidate_surface_includes_self_dead_hidden_and_unused_active(self):
        state = GameState(
            n_cards=4,
            n_evil=1,
            deck=DeckComposition(
                villagers=["Fortune Teller", "Dreamer"],
                outcasts=[],
                minions=["Minion"],
                demons=[],
            ),
            cards=[
                CardInfo(1, "Fortune Teller"),
                CardInfo(2, "Dreamer"),
            ],
            executed=[3],
        )
        scenarios = [Scenario(evil_positions={4: "Minion"})]
        result = SolverResult([], [], [], 1, 1, scenarios)

        with patch("strategy._recommend_boolean_ability", return_value=None) as recommend:
            recommend_abilities(state, result, used_abilities=[2])

        candidates = recommend.call_args.args[3]
        self.assertEqual(
            candidates,
            [[1, 2], [1, 3], [1, 4], [2, 3], [2, 4], [3, 4]],
        )


if __name__ == "__main__":
    unittest.main()
