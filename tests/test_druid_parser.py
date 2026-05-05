import unittest

from game_loop import _parse_clue_from_memory


def _base_druid(clue, targets=None):
    return {
        "position": 2,
        "true_role": "Druid",
        "disguise": "Druid",
        "clue_text": clue,
        "acted_infos": [{"desc": "...", "targets": targets or [1, 3, 4]}],
        "runtime_data": None,
        "ability_used": True,
        "uses": 1,
    }


class TestDruidParser(unittest.TestCase):
    def test_singular_there_is_outcast(self):
        ci = _parse_clue_from_memory(
            _base_druid("Among #1, #3, #4\nthere is: Bombardier")
        )
        self.assertIsNotNone(ci)
        self.assertEqual(ci.apparent_role, "Druid")
        self.assertEqual(
            ci.info_parsed,
            {"targets": [1, 3, 4], "found_outcast": "Bombardier"},
        )

    def test_space_in_outcast_name_is_normalized(self):
        ci = _parse_clue_from_memory(
            _base_druid("Among #1, #3, #4 there is: Plague Doctor")
        )
        self.assertIsNotNone(ci)
        self.assertEqual(ci.info_parsed["found_outcast"], "Plague_Doctor")

    def test_no_outcasts(self):
        ci = _parse_clue_from_memory(
            _base_druid("Among #1, #3, #4 there are no Outcasts")
        )
        self.assertIsNotNone(ci)
        self.assertEqual(
            ci.info_parsed,
            {"targets": [1, 3, 4], "found_outcast": None},
        )


if __name__ == "__main__":
    unittest.main()
