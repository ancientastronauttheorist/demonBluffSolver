import unittest
from pathlib import Path

from game_utils import find_execute_button_center


class TestGameUtils(unittest.TestCase):
    def test_find_execute_button_center_ignores_settings_icons(self):
        path = Path(__file__).resolve().parent.parent / "screenshots" / "live_after_pause_close.jpg"

        pos = find_execute_button_center(str(path))

        self.assertIsNotNone(pos)
        x, y = pos
        self.assertGreaterEqual(x, 2120)
        self.assertLessEqual(x, 2300)
        self.assertGreaterEqual(y, 1120)
        self.assertLessEqual(y, 1245)


if __name__ == "__main__":
    unittest.main()
