"""Unit test for reveal_order cleanup after flake-failed flip clicks.

Regression for asc78_v6 halt (2026-04-21): a fresh `flip` appended all
click-attempted positions to reveal_order without verifying they actually
revealed. When #1 failed to register, reveal_order still recorded [1..N],
corrupting the Baker-chain validator's seed. After a subsequent wrong
execute, the scenario space collapsed to 0 with the bad ordering.

The fix: after the in-memory verification step, strip failed positions from
session.reveal_order so the subsequent `flip <pos>` retry lands them at
their true reveal index.
"""

import unittest


class _FakeSession:
    def __init__(self, reveal_order):
        self.reveal_order = list(reveal_order)
        self._saved = False

    def save(self):
        self._saved = True


def _apply_fix(session, verify_result):
    """Exact behaviour of the edited block in _cmd_flip — extracted so the
    logic is testable without spinning up the game or monkeypatching."""
    failed = verify_result.get("failed", [])
    if not failed:
        return False
    removed = False
    for p in failed:
        if p in session.reveal_order:
            session.reveal_order.remove(p)
            removed = True
    if removed:
        session.save()
    return removed


class TestRevealOrderCleanup(unittest.TestCase):
    def test_asc78_v6_shape_single_flake(self):
        """#1 flake: flip appended [1..8], verification fails #1 only."""
        session = _FakeSession([1, 2, 3, 4, 5, 6, 7, 8])
        _apply_fix(session, {"flipped": [2, 3, 4, 5, 6, 7, 8], "failed": [1], "blocked": []})
        self.assertEqual(session.reveal_order, [2, 3, 4, 5, 6, 7, 8])
        self.assertTrue(session._saved)

    def test_multiple_flakes(self):
        session = _FakeSession([1, 2, 3, 4, 5, 6, 7, 8])
        _apply_fix(session, {"flipped": [2, 3, 5, 6, 7, 8], "failed": [1, 4], "blocked": []})
        self.assertEqual(session.reveal_order, [2, 3, 5, 6, 7, 8])

    def test_clean_flip_is_noop(self):
        session = _FakeSession([1, 2, 3, 4, 5, 6, 7, 8])
        _apply_fix(session, {"flipped": [1, 2, 3, 4, 5, 6, 7, 8], "failed": [], "blocked": []})
        self.assertEqual(session.reveal_order, [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertFalse(session._saved)

    def test_witch_block_not_treated_as_flake(self):
        """Verify stores Witch-blocked positions under 'blocked', not 'failed'.
        The fix must not strip blocked positions (they're tracked separately
        via session.blocked_positions)."""
        session = _FakeSession([1, 2, 3, 4, 5, 6, 7])
        _apply_fix(session, {"flipped": [1, 2, 3, 4, 5, 6], "failed": [], "blocked": [7]})
        self.assertEqual(session.reveal_order, [1, 2, 3, 4, 5, 6, 7])

    def test_retry_appends_at_true_index(self):
        """After cleanup, a subsequent flip <pos> retry appends at end —
        reflecting the true reveal index (last, not first)."""
        session = _FakeSession([1, 2, 3, 4, 5, 6, 7, 8])
        _apply_fix(session, {"flipped": [2, 3, 4, 5, 6, 7, 8], "failed": [1], "blocked": []})
        # Simulate the retry's "append if not in reveal_order" logic:
        if 1 not in session.reveal_order:
            session.reveal_order.append(1)
        self.assertEqual(session.reveal_order, [2, 3, 4, 5, 6, 7, 8, 1])


if __name__ == "__main__":
    unittest.main()
