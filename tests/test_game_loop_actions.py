import unittest
from unittest.mock import patch

from game_loop import GameSession


class TestGameLoopActions(unittest.TestCase):
    @patch("game_utils.reveal_card")
    @patch("card_vision.resolved_board_seat_center")
    @patch("game_utils.take_game_screenshot")
    def test_reveal_uses_game_numbered_seat_center(self, mock_screenshot, mock_center, mock_reveal):
        session = GameSession(8, 2)
        mock_screenshot.return_value = "board.jpg"
        mock_center.return_value = (930, 364)

        session.reveal(7)

        mock_screenshot.assert_called_once_with("_card_detect")
        mock_center.assert_called_once_with("board.jpg", 7, 8)
        mock_reveal.assert_called_once_with((930, 364))


if __name__ == "__main__":
    unittest.main()
