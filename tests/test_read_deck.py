import unittest

from game_loop import _baa_hides_outcast


class TestBaaHidesOutcast(unittest.TestCase):
    """Baa-in-deck renders exactly one outcast as a face-down eye-symbol in the
    deck-pool view. _baa_hides_outcast captures that explained-mismatch shape
    so _cmd_read_deck reports MATCH instead of halting. Confirmed asc78 v2
    (Doppelganger hidden in a Baa+Puppeteer+Doppelganger pool, 2026-04-20).
    """

    _FULL_MR = {"lover", "witness", "judge", "druid", "enlightened",
                "fortune_teller", "jester", "doppelganger", "puppeteer", "baa"}

    def test_asc78v2_doppelganger_hidden(self):
        only_mr = {"doppelganger"}
        only_cv = set()
        self.assertTrue(_baa_hides_outcast(only_cv, only_mr, self._FULL_MR, 1))

    def test_drunk_hidden_counts_as_outcast(self):
        mr = {"baker", "oracle", "drunk", "shaman", "baa"}
        self.assertTrue(_baa_hides_outcast(set(), {"drunk"}, mr, 1))

    def test_no_baa_means_no_allowance(self):
        mr = {"baker", "drunk", "shaman", "lilis"}
        self.assertFalse(_baa_hides_outcast(set(), {"drunk"}, mr, 1))

    def test_hidden_role_must_be_outcast(self):
        mr = {"baker", "oracle", "shaman", "baa"}
        self.assertFalse(_baa_hides_outcast(set(), {"baker"}, mr, 1))

    def test_only_cv_nonempty_halts(self):
        self.assertFalse(_baa_hides_outcast({"judge"}, {"doppelganger"},
                                            self._FULL_MR, 1))

    def test_more_than_one_missing_halts(self):
        self.assertFalse(_baa_hides_outcast(set(), {"doppelganger", "drunk"},
                                            self._FULL_MR, 2))

    def test_zero_unclassified_halts(self):
        self.assertFalse(_baa_hides_outcast(set(), {"doppelganger"},
                                            self._FULL_MR, 0))


if __name__ == "__main__":
    unittest.main()
