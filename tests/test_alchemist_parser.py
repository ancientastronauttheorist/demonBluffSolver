import unittest

from game_loop import _parse_clue_from_memory


class TestAlchemistParser(unittest.TestCase):
    def _base(self, clue):
        return {
            "position": 4,
            "true_role": "Alchemist",
            "disguise": "Alchemist",
            "clue_text": clue,
            "acted_infos": [],
            "runtime_data": None,
            "ability_used": False,
            "uses": 0,
        }

    def test_no_one_was_corrupted_means_zero(self):
        ci = _parse_clue_from_memory(
            self._base("NO one was Corrupted around me")
        )
        self.assertIsNotNone(ci)
        self.assertEqual(ci.info_parsed, {"corrupted_count": 0})

    def test_no_corruption_means_zero(self):
        ci = _parse_clue_from_memory(
            self._base("There was no Corruption around me")
        )
        self.assertIsNotNone(ci)
        self.assertEqual(ci.info_parsed, {"corrupted_count": 0})

    def test_numeric_corruption_still_parses(self):
        ci = _parse_clue_from_memory(
            self._base("There was 2 Corruption around me")
        )
        self.assertIsNotNone(ci)
        self.assertEqual(ci.info_parsed, {"corrupted_count": 2})


if __name__ == "__main__":
    unittest.main()
