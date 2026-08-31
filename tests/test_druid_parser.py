import unittest

from game_loop import _parse_clue_from_memory


def _base_druid(clue, targets=None):
    refs = targets if targets is not None else [1, 3, 4]
    return {
        "position": 2,
        "true_role": "Librarian",
        "disguise": None,
        "clue_text": clue,
        "acted_infos": [{"desc": clue, "targets": refs}],
        "runtime_data": None,
        "ability_used": False,
        "uses": 0,
    }


class TestDruidParser(unittest.TestCase):
    def test_singular_there_is_outcast(self):
        ci = _parse_clue_from_memory(
            _base_druid("Among #1, #3, #4\nthere is: Bombardier"),
            n_cards=6,
        )
        self.assertIsNotNone(ci)
        self.assertEqual(ci.apparent_role, "Druid")
        self.assertEqual(
            ci.info_parsed,
            {
                "targets": [1, 3, 4],
                "found_outcast": "Bombardier",
                "druid_variant": "public_current",
            },
        )

    def test_space_in_outcast_name_is_normalized(self):
        ci = _parse_clue_from_memory(
            _base_druid("Among #1, #3, #4\nthere is: Plague Doctor"),
            n_cards=6,
        )
        self.assertIsNotNone(ci)
        self.assertEqual(ci.info_parsed["found_outcast"], "Plague_Doctor")

    def test_no_outcasts(self):
        ci = _parse_clue_from_memory(
            _base_druid("Among #1, #3, #4\nthere are NO Outcasts"),
            n_cards=6,
        )
        self.assertIsNotNone(ci)
        self.assertEqual(
            ci.info_parsed,
            {
                "targets": [1, 3, 4],
                "found_outcast": None,
                "druid_variant": "public_current",
            },
        )


if __name__ == "__main__":
    unittest.main()
