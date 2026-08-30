import unittest
from collections import Counter
from contextlib import redirect_stdout
from io import StringIO
from types import SimpleNamespace
from unittest.mock import patch

from game_loop import (
    DecisionLog,
    GameSession,
    _baa_hides_outcast,
    _baa_post_death_deck_refresh,
    dispatch,
)


class TestBaaHidesOutcast(unittest.TestCase):
    """Baa-in-deck renders exactly one outcast as a face-down eye-symbol in the
    deck-pool view. _baa_hides_outcast captures that explained-mismatch shape
    so _cmd_read_deck reports MATCH instead of halting. Confirmed asc78 v2
    (Doppelganger hidden in a Baa+Puppeteer+Doppelganger pool, 2026-04-20).
    """

    _FULL_MR = Counter({
        "lover": 1, "witness": 1, "judge": 1, "druid": 1, "enlightened": 1,
        "fortune_teller": 1, "jester": 1, "doppelganger": 1,
        "puppeteer": 1, "baa": 1,
    })

    def test_asc78v2_doppelganger_hidden(self):
        only_mr = Counter({"doppelganger": 1})
        only_cv = Counter()
        self.assertTrue(_baa_hides_outcast(only_cv, only_mr, self._FULL_MR, 1))

    def test_drunk_hidden_counts_as_outcast(self):
        mr = Counter({"baker": 1, "oracle": 1, "drunk": 1, "shaman": 1, "baa": 1})
        self.assertTrue(_baa_hides_outcast(Counter(), Counter({"drunk": 1}), mr, 1))

    def test_duplicate_hidden_role_is_not_lost_to_set_comparison(self):
        mr = Counter({"baa": 1, "baker": 1, "drunk": 2})
        cv = Counter({"baa": 1, "baker": 1, "drunk": 1})
        self.assertTrue(_baa_hides_outcast(cv - mr, mr - cv, mr, 1))

    def test_no_baa_means_no_allowance(self):
        mr = Counter({"baker": 1, "drunk": 1, "shaman": 1, "lilis": 1})
        self.assertFalse(_baa_hides_outcast(Counter(), Counter({"drunk": 1}), mr, 1))

    def test_hidden_role_must_be_outcast(self):
        mr = Counter({"baker": 1, "oracle": 1, "shaman": 1, "baa": 1})
        self.assertFalse(_baa_hides_outcast(Counter(), Counter({"baker": 1}), mr, 1))

    def test_only_cv_nonempty_halts(self):
        self.assertFalse(_baa_hides_outcast(Counter({"judge": 1}),
                                            Counter({"doppelganger": 1}),
                                            self._FULL_MR, 1))

    def test_more_than_one_missing_halts(self):
        self.assertFalse(_baa_hides_outcast(Counter(),
                                            Counter({"doppelganger": 1, "drunk": 1}),
                                            self._FULL_MR, 2))

    def test_zero_unclassified_halts(self):
        self.assertFalse(_baa_hides_outcast(Counter(), Counter({"doppelganger": 1}),
                                            self._FULL_MR, 0))

    def test_baa_death_refresh_does_not_mark_a_board_card_revealed(self):
        session = GameSession(3, 1)
        session.reveal_order = [1]

        with redirect_stdout(StringIO()) as output:
            _baa_post_death_deck_refresh(session)

        self.assertEqual(session.reveal_order, [1])
        self.assertIn("deck view", output.getvalue())
        self.assertIn("no board card", output.getvalue())

    def test_manual_deck_keeps_hud_outcast_count_for_baa(self):
        session = GameSession(4, 2)
        args = ["V=Baker", "O=Drunk", "M=Shaman", "D=Baa", "nv=2", "no=1"]

        with (
            patch.object(session, "save"),
            patch.object(DecisionLog, "log_deck"),
            redirect_stdout(StringIO()) as output,
        ):
            dispatch("deck", args, session)

        text = output.getvalue()
        self.assertEqual(session.board_outcast_count, 1)
        self.assertIn("one existing Outcast", text)
        self.assertIn("Use the HUD no= exactly as shown", text)
        self.assertNotIn("+1 fake", text)

    def test_start_village_keeps_hud_outcast_count_for_baa(self):
        deck_output = (
            "Villagers (1): Baker\n"
            "Outcasts (1): Drunk\n"
            "Minions (1): Shaman\n"
            "Demons (1): Baa\n"
        )

        with (
            patch("subprocess.run", return_value=SimpleNamespace(
                returncode=0, stdout=deck_output, stderr="",
            )),
            patch.object(GameSession, "save"),
            patch.object(DecisionLog, "start_game"),
            patch.object(DecisionLog, "log_deck"),
            redirect_stdout(StringIO()) as output,
        ):
            session = dispatch("start_village", ["4", "2", "nv=2", "no=1"], None)

        text = output.getvalue()
        self.assertEqual(session.board_outcast_count, 1)
        self.assertIn("one existing Outcast", text)
        self.assertIn("Use the HUD no= exactly as shown", text)
        self.assertNotIn("+1 fake", text)


if __name__ == "__main__":
    unittest.main()
