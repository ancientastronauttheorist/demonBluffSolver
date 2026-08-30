"""Tests for per-role wrong-execution HP cost lookup.

Added asc78_v6 (2026-04-21): confirmed empirically that wrong-executing a
Drunk costs only 2 HP, not the default 5. The `wrong_exec_cost_for` helper
in knowledge_base.py centralises the override table so both the auto_exec
path and the CLI `execute` warning agree.
"""

import unittest

from knowledge_base import (
    DEFAULT_WRONG_EXEC_COST,
    KNIGHT_BLUFF_EXTRA_DAMAGE,
    WRONG_EXEC_COST_OVERRIDES,
    execution_cost_for,
    wrong_exec_cost_for,
)


class WrongExecCostTests(unittest.TestCase):
    def test_default_is_five(self):
        self.assertEqual(DEFAULT_WRONG_EXEC_COST, 5)

    def test_drunk_override_registered(self):
        self.assertIn("Drunk", WRONG_EXEC_COST_OVERRIDES)
        self.assertEqual(WRONG_EXEC_COST_OVERRIDES["Drunk"], 2)

    def test_unknown_role_returns_default(self):
        self.assertEqual(wrong_exec_cost_for("Villager"), 5)

    def test_drunk_returns_two(self):
        self.assertEqual(wrong_exec_cost_for("Drunk"), 2)

    def test_none_role_returns_default(self):
        self.assertEqual(wrong_exec_cost_for(None), 5)

    def test_custom_default_for_unknown_role(self):
        self.assertEqual(wrong_exec_cost_for("Villager", default=10), 10)

    def test_custom_default_does_not_override_drunk(self):
        # Override table wins over the caller-supplied default.
        self.assertEqual(wrong_exec_cost_for("Drunk", default=10), 2)

    def test_none_role_with_custom_default(self):
        self.assertEqual(wrong_exec_cost_for(None, default=7), 7)

    def test_multiword_role_not_overridden(self):
        # Plague Doctor is a real role name with a space — currently no
        # override, so the default should apply. This guards against
        # accidental whitespace-munging in the lookup path.
        self.assertEqual(wrong_exec_cost_for("Plague Doctor"), 5)
        self.assertEqual(wrong_exec_cost_for("Plague Doctor", default=3), 3)

    def test_knight_bluff_extra_is_four(self):
        self.assertEqual(KNIGHT_BLUFF_EXTRA_DAMAGE, 4)

    def test_corrupted_true_knight_costs_nine(self):
        self.assertEqual(
            execution_cost_for(
                "Knight",
                apparent_role="Knight",
                was_corrupted=True,
                was_killable=True,
            ),
            9,
        )

    def test_statused_drunk_as_knight_costs_six(self):
        self.assertEqual(
            execution_cost_for(
                "Drunk",
                apparent_role="Knight",
                was_corrupted=True,
                was_killable=True,
            ),
            6,
        )

    def test_resistant_drunk_as_knight_costs_two(self):
        self.assertEqual(
            execution_cost_for(
                "Drunk",
                apparent_role="Knight",
                was_corrupted=False,
                was_killable=True,
            ),
            2,
        )

    def test_drunk_non_knight_keeps_base_override(self):
        self.assertEqual(
            execution_cost_for(
                "Drunk",
                apparent_role="Bard",
                was_killable=True,
            ),
            2,
        )

    def test_apparent_knight_without_killable_or_corrupted_signal_has_no_extra(self):
        self.assertEqual(
            execution_cost_for("Knight", apparent_role="knight"),
            5,
        )

    def test_clean_killable_non_drunk_knight_bluff_has_no_extra(self):
        self.assertEqual(
            execution_cost_for(
                "Doppelganger",
                apparent_role="Knight",
                was_killable=True,
            ),
            5,
        )

    def test_blocked_or_evil_execution_costs_zero(self):
        self.assertEqual(
            execution_cost_for(
                "Knight",
                apparent_role="Knight",
                was_corrupted=True,
                execution_blocked=True,
            ),
            0,
        )
        self.assertEqual(
            execution_cost_for(
                "Drunk",
                apparent_role="Knight",
                was_killable=True,
                was_evil=True,
            ),
            0,
        )

    def test_apparent_role_normalizes_underscores_and_case(self):
        self.assertEqual(
            execution_cost_for(
                "Drunk",
                apparent_role="  KNIGHT  ",
                was_corrupted=True,
                was_killable=True,
            ),
            6,
        )


if __name__ == "__main__":
    unittest.main()
