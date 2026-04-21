"""Tests for per-role wrong-execution HP cost lookup.

Added asc78_v6 (2026-04-21): confirmed empirically that wrong-executing a
Drunk costs only 2 HP, not the default 5. The `wrong_exec_cost_for` helper
in knowledge_base.py centralises the override table so both the auto_exec
path and the CLI `execute` warning agree.
"""

import unittest

from knowledge_base import (
    DEFAULT_WRONG_EXEC_COST,
    WRONG_EXEC_COST_OVERRIDES,
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


if __name__ == "__main__":
    unittest.main()
