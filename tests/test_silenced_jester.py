"""Unit tests for Rambler2-interrupted Jester parsing.

Covers:
  - card_jester_silenced constructor (CardInfo shape)
  - _parse_clue_from_memory interruption branch ("#X shut up!")
  - Normal "N evil" Jester clue still parses as non-silenced
  - Drunk-disguised-as-Jester silenced flow keys off apparent role

Rambler2 rewrites the emitted ActedInfo to description "#R shut up!" and
references [R]. Those references are not Jester's original picks. The parser
therefore keeps the global shut_up_target plus an explicit silenced marker,
but never invents the discarded Jester target trio.
"""

import unittest

from game_loop import (
    _parse_clue_from_memory,
    card_jester_silenced,
)


def _base_jester(clue, targets=None, position=3, true_role="Jester",
                 disguise="Jester"):
    return {
        "position": position,
        "true_role": true_role,
        "disguise": disguise,
        "clue_text": clue,
        "acted_infos": [{"desc": clue, "targets": targets or []}],
        "runtime_data": None,
        "ability_used": True,
        "uses": 1,
    }


class TestCardJesterSilenced(unittest.TestCase):
    def test_constructor_shape(self):
        ci = card_jester_silenced(3, [1, 2, 4])
        self.assertEqual(ci.position, 3)
        self.assertEqual(ci.apparent_role, "Jester")
        self.assertEqual(
            ci.info_parsed,
            {"targets": [1, 2, 4], "silenced": True},
        )

    def test_constructor_preserves_shut_up_target(self):
        ci = card_jester_silenced(3, [1, 2, 4], shut_up_target=5)
        self.assertEqual(ci.info_parsed["shut_up_target"], 5)

    def test_copies_targets_defensively(self):
        targets = [1, 2, 4]
        ci = card_jester_silenced(3, targets)
        targets.append(99)
        self.assertEqual(ci.info_parsed["targets"], [1, 2, 4])


class TestParseClueFromMemorySilencedJester(unittest.TestCase):
    def test_silenced_by_rambler_shut_up(self):
        card = _base_jester("#5 shut up!", targets=[5])
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertEqual(ci.apparent_role, "Jester")
        self.assertEqual(ci.info_parsed.get("silenced"), True)
        self.assertNotIn("targets", ci.info_parsed)
        self.assertEqual(ci.info_parsed.get("shut_up_target"), 5)
        # Must NOT have an evil_count key (the validator keys off silenced first).
        self.assertNotIn("evil_count", ci.info_parsed)

    def test_silenced_case_insensitive(self):
        card = _base_jester("#5 SHUT UP!", targets=[5])
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertEqual(ci.info_parsed.get("silenced"), True)
        self.assertEqual(ci.info_parsed.get("shut_up_target"), 5)

    def test_saved_act_empty_requires_manual_recovery(self):
        card = _base_jester("", targets=[5])
        card["acted_infos"][0]["desc"] = "#5 shut up!"
        ci = _parse_clue_from_memory(card)
        self.assertIsNone(ci)

    def test_empty_result_does_not_invent_rambler_interference(self):
        card = _base_jester("", targets=[1, 2, 4])
        self.assertIsNone(_parse_clue_from_memory(card))

    def test_normal_jester_n_evil_preserved(self):
        # Real "2 of them are evil" clue must NOT be misclassified as silenced.
        card = _base_jester("2 are evil", targets=[1, 2, 4])
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertNotIn("silenced", ci.info_parsed)
        self.assertEqual(ci.info_parsed.get("evil_count"), 2)
        self.assertEqual(ci.info_parsed.get("targets"), [1, 2, 4])

    def test_normal_jester_none_evil_preserved(self):
        card = _base_jester("none of them are evil", targets=[1, 2, 4])
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        # "none" branch -> evil_count=0, not silenced.
        self.assertNotIn("silenced", ci.info_parsed)
        self.assertEqual(ci.info_parsed.get("evil_count"), 0)

    def test_drunk_disguised_as_jester_silenced(self):
        # Parser must key off apparent role (disguise), not true_role.
        card = {
            "position": 3,
            "true_role": "Drunk",
            "disguise": "Jester",
            "clue_text": "#5 shut up!",
            "acted_infos": [{"desc": "#5 shut up!", "targets": [5]}],
            "runtime_data": None,
            "ability_used": True,
            "uses": 1,
        }
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertEqual(ci.apparent_role, "Jester")
        self.assertEqual(ci.info_parsed.get("silenced"), True)
        self.assertNotIn("targets", ci.info_parsed)
        self.assertEqual(ci.info_parsed.get("shut_up_target"), 5)


class TestShutUpCluesForNonJester(unittest.TestCase):
    def test_lover_shut_up_is_not_adjacent_evil_count(self):
        card = {
            "position": 1,
            "true_role": "Lover",
            "disguise": "Lover",
            "clue_text": "#10 shut up!",
            "acted_infos": [{"desc": "#10 shut up!", "targets": [10]}],
            "runtime_data": None,
            "ability_used": False,
            "uses": 0,
        }
        ci = _parse_clue_from_memory(card)
        self.assertIsNotNone(ci)
        self.assertEqual(ci.apparent_role, "Lover")
        self.assertEqual(ci.info_parsed, {"shut_up_target": 10})


if __name__ == "__main__":
    unittest.main()
