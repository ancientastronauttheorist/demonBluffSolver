import unittest
from pathlib import Path

import ui


class TestUI(unittest.TestCase):
    def test_find_target_button_prefers_template_match_for_next_dialog(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "loop_next_board_final_try.jpg"

        button = ui.find_target_button("next", str(path))

        self.assertIsNotNone(button)
        self.assertEqual(button.label, "next")
        self.assertGreaterEqual(button.x, 1260)
        self.assertLessEqual(button.x, 1300)
        self.assertGreaterEqual(button.y, 850)
        self.assertLessEqual(button.y, 880)
        self.assertGreater(button.confidence, 0.95)

    def test_find_target_button_finds_cancel_in_ability_prompt(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "live_after_jester4_arrow.jpg"

        button = ui.find_target_button("cancel", str(path))

        self.assertIsNotNone(button)
        self.assertEqual(button.label, "cancel")
        self.assertGreaterEqual(button.x, 1260)
        self.assertLessEqual(button.x, 1300)
        self.assertGreaterEqual(button.y, 850)
        self.assertLessEqual(button.y, 880)
        self.assertGreater(button.confidence, 0.75)


if __name__ == "__main__":
    unittest.main()
