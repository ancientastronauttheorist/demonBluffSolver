"""Judge2 public-result parsing and autonomous-use boundary regressions."""

import unittest
from unittest.mock import patch

from game_loop import (
    DecisionLog,
    GameSession,
    _parse_clue_from_memory,
    _parse_judge_result_from_memory,
)
from solver import CardInfo
from strategy import Action


def _memory_card(
    clue: str,
    targets: list[int],
    *,
    position: int = 2,
    infos: int = 1,
) -> dict:
    acted_infos = [
        {"desc": clue, "targets": list(targets)}
        for _ in range(infos)
    ]
    return {
        "position": position,
        "true_role": "Judge",
        "clue_text": clue,
        "acted_infos": acted_infos,
        "uses": 1,
        "ability_used": True,
    }


class JudgeMemoryResultTests(unittest.TestCase):
    def test_exact_truth_and_lying_sentences(self):
        for clue, expected in [
            ("#4 is\nsaying Truth", False),
            ("#4 is\nLying", True),
        ]:
            with self.subTest(clue=clue):
                parsed, error = _parse_judge_result_from_memory(
                    _memory_card(clue, [4]),
                    expected_target=4,
                    n_cards=6,
                )

                self.assertIsNone(error)
                self.assertEqual(parsed.apparent_role, "Judge")
                self.assertEqual(
                    parsed.info_parsed,
                    {"target": 4, "is_lying": expected},
                )
                self.assertEqual(parsed.info_text, clue)

    def test_exact_parser_accepts_native_self_check_shape(self):
        parsed, error = _parse_judge_result_from_memory(
            _memory_card("#2 is\nsaying Truth", [2]),
            expected_target=2,
            n_cards=6,
        )

        self.assertIsNone(error)
        self.assertEqual(parsed.info_parsed["target"], 2)
        self.assertFalse(parsed.info_parsed["is_lying"])

    def test_rejects_unknown_text_instead_of_defaulting_to_truth(self):
        for clue in [
            "#4 is honest",
            "#4 might be Lying",
            "#4 is saying Truth somehow",
            "",
        ]:
            with self.subTest(clue=clue):
                parsed, error = _parse_judge_result_from_memory(
                    _memory_card(clue, [4]),
                    expected_target=4,
                    n_cards=6,
                )
                self.assertIsNone(parsed)
                self.assertIn("Unrecognized Judge", error)

    def test_rejects_reference_shape_target_cross_checks_and_ranges(self):
        cases = [
            (_memory_card("#4 is\nLying", [], infos=0), 4, "acted-info"),
            (_memory_card("#4 is\nLying", []), 4, "exactly one picked"),
            (_memory_card("#4 is\nLying", [4, 5]), 4, "exactly one picked"),
            (_memory_card("#4 is\nLying", [3]), 4, "target mismatch"),
            (_memory_card("#3 is\nLying", [4]), 4, "target mismatch"),
            (_memory_card("#7 is\nLying", [7]), 7, "outside"),
        ]
        for card, target, expected_error in cases:
            with self.subTest(card=card, target=target):
                parsed, error = _parse_judge_result_from_memory(
                    card,
                    expected_target=target,
                    n_cards=6,
                )
                self.assertIsNone(parsed)
                self.assertIn(expected_error, error)

    def test_reset_after_night_history_uses_latest_result_and_keeps_all_evidence(self):
        card = _memory_card("#5 is\nLying", [5])
        card["acted_infos"] = [
            {"desc": "#3 is\nsaying Truth", "targets": [3]},
            {"desc": "#5 is\nLying", "targets": [5]},
        ]

        parsed, error = _parse_judge_result_from_memory(
            card,
            expected_target=5,
            n_cards=6,
        )

        self.assertIsNone(error)
        self.assertEqual(
            parsed.info_parsed,
            {
                "target": 5,
                "is_lying": True,
                "observations": [
                    {"target": 3, "is_lying": False},
                    {"target": 5, "is_lying": True},
                ],
            },
        )

    def test_auto_card_uses_same_strict_native_contract(self):
        valid = _parse_clue_from_memory(
            _memory_card("#5 is\nsaying Truth", [5])
        )
        invalid = _parse_clue_from_memory(
            _memory_card("#5 seems truthful", [5])
        )

        self.assertEqual(
            valid.info_parsed,
            {"target": 5, "is_lying": False},
        )
        self.assertIsNone(invalid)
        self.assertIsNone(_parse_clue_from_memory(
            _memory_card("#7 is\nLying", [7]),
            n_cards=6,
        ))


class JudgeAutonomousUseTests(unittest.TestCase):
    def test_self_target_passes_preflight_and_records_exact_result(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(2, "Judge"))
        action = Action(
            "use_ability",
            position=2,
            targets=[2],
            ability_name="Judge",
        )
        memory_result = _memory_card("#2 is\nsaying Truth", [2])
        before = _memory_card("", [], position=2, infos=0)

        class Reader:
            def __init__(self):
                self.reads = 0

            def open(self):
                return True

            def read_board(self):
                self.reads += 1
                return [before if self.reads == 1 else memory_result]

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
        self.assertEqual(
            result["info_parsed"],
            {"target": 2, "is_lying": False},
        )
        self.assertIn(2, session.used_abilities)

    def test_preflight_requires_one_target_and_apparent_judge_actor(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(2, "Bard"))

        wrong_count = session.auto_use_ability(
            Action("use_ability", 2, [], "Judge")
        )
        wrong_actor = session.auto_use_ability(
            Action("use_ability", 2, [3], "Judge")
        )

        self.assertIn("exactly 1 target", wrong_count["error"])
        self.assertIn("not an apparent Judge", wrong_actor["error"])

    def test_preflight_allows_target_with_unused_active_ability(self):
        session = GameSession(4, 1)
        session.cards.extend([
            CardInfo(1, "Judge"),
            CardInfo(2, "Dreamer"),
        ])
        action = Action("use_ability", 1, [2], "Judge")
        before = _memory_card("", [], position=1, infos=0)

        class Reader:
            def open(self):
                return True

            def read_board(self):
                return [before]

            def close(self):
                return None

        # Reaching the activation click proves the unused-active target did not
        # trip the generic preflight guard. Stop there; result parsing is
        # covered independently.
        with (
            patch(
                "template_match.safe_click_at",
                side_effect=RuntimeError("sentinel after preflight"),
            ),
            patch("memory_reader.MemoryReader", return_value=Reader()),
        ):
            result = session.auto_use_ability(action)

        self.assertIn("sentinel after preflight", result["error"])
        self.assertNotIn("unused active ability", result["error"])

    def test_resettable_judge_rejects_unchanged_stale_latest_event(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(2, "Judge"))
        action = Action("use_ability", 2, [3], "Judge")
        stale = _memory_card("#3 is\nsaying Truth", [3])

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
            result = session.auto_use_ability(action)

        self.assertFalse(result["success"])
        self.assertIn("new or changed latest", result["error"])
        self.assertNotIn(2, session.used_abilities)
        self.assertEqual(session.cards[0].info_parsed, {})

    def test_prior_same_target_fails_before_click_when_pre_history_is_unreadable(self):
        session = GameSession(4, 1)
        session.add_card(CardInfo(
            2,
            "Judge",
            info_text="#3 is\nsaying Truth",
            info_parsed={"target": 3, "is_lying": False},
        ))
        session.reset_after_night_abilities()
        stale = _memory_card("#3 is\nsaying Truth", [3])
        unreadable = _memory_card("#3 is\nsaying Truth", [3], infos=0)

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
                Action("use_ability", 2, [3], "Judge")
            )

        self.assertFalse(result["success"])
        self.assertIn("prior active evidence", result["error"])
        self.assertIn("no readable newest", result["error"])
        click.assert_not_called()
        self.assertEqual(reader.reads, 1)
        self.assertNotIn(2, session.used_abilities)

    def test_prior_same_target_fails_before_click_on_stale_shorter_history(self):
        session = GameSession(4, 1)
        prior = CardInfo(
            1,
            "Judge",
            info_parsed={"target": 3, "is_lying": False},
        )
        session.add_card(prior)
        session.reset_after_night_abilities()
        session.add_card(CardInfo(
            1,
            "Judge",
            info_parsed={"target": 3, "is_lying": False},
        ))
        session.reset_after_night_abilities()
        clue = "#3 is\nsaying Truth"
        stale_short = _memory_card(clue, [3], position=1, infos=1)
        recovered_old = _memory_card(clue, [3], position=1, infos=2)

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
                Action("use_ability", 1, [3], "Judge")
            )

        self.assertFalse(result["success"])
        self.assertIn("shorter than the local minimum", result["error"])
        click.assert_not_called()
        self.assertEqual(reader.reads, 1)
        self.assertNotIn(1, session.used_abilities)

    def test_resettable_judge_accepts_changed_latest_event(self):
        session = GameSession(4, 1)
        session.cards.append(CardInfo(2, "Judge"))
        action = Action("use_ability", 2, [3], "Judge")
        before = _memory_card("#1 is\nsaying Truth", [1])
        after = _memory_card("#3 is\nLying", [3])

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
        self.assertEqual(
            result["info_parsed"],
            {"target": 3, "is_lying": True},
        )


class JudgeResetAfterNightTests(unittest.TestCase):
    def test_night_resets_only_apparent_judges(self):
        session = GameSession(4, 1)
        session.cards.extend([
            CardInfo(1, "Judge", info_parsed={"target": 2, "is_lying": False}),
            CardInfo(2, "Slayer"),
            CardInfo(3, "Judge"),
        ])
        session.used_abilities = [1, 2]

        reset = session.reset_after_night_abilities()

        self.assertEqual(reset, [1])
        self.assertEqual(session.used_abilities, [2])

    def test_reused_judge_retains_all_round_observations(self):
        session = GameSession(5, 1)
        session.add_card(CardInfo(
            1,
            "Judge",
            info_parsed={"target": 2, "is_lying": False},
        ))
        session.reset_after_night_abilities()
        session.add_card(CardInfo(
            1,
            "Judge",
            info_parsed={"target": 4, "is_lying": True},
        ))

        self.assertEqual(
            session.cards[0].info_parsed,
            {
                "target": 4,
                "is_lying": True,
                "observations": [
                    {"target": 2, "is_lying": False},
                    {"target": 4, "is_lying": True},
                ],
            },
        )
        self.assertEqual(session.used_abilities, [1])

    def test_same_round_reentry_corrects_instead_of_appending(self):
        session = GameSession(5, 1)
        session.add_card(CardInfo(
            1,
            "Judge",
            info_parsed={"target": 2, "is_lying": False},
        ))
        session.add_card(CardInfo(
            1,
            "Judge",
            info_parsed={"target": 3, "is_lying": True},
        ))

        self.assertEqual(
            session.cards[0].info_parsed,
            {"target": 3, "is_lying": True},
        )


if __name__ == "__main__":
    unittest.main()
