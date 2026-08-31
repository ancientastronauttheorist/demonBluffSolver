import unittest
from copy import deepcopy
from unittest.mock import patch

import rust_solver
from solver import (
    Scenario,
    TwinNeighborSide,
    TwinStartKind,
)


class _DummyState:
    def to_dict(self):
        return {"n_cards": 7, "n_evil": 1}


def _solver_response(twin_trace_marker=...):
    scenario = {"evil_positions": {"1": "Twin Minion"}}
    if twin_trace_marker is not ...:
        scenario["twin_trace"] = twin_trace_marker
    return {
        "definite_evil": [],
        "definite_good": [],
        "bombardier_positions": [],
        "n_scenarios": 1,
        "n_surviving": 1,
        "surviving_scenarios": [scenario],
        "reasoning": [],
    }


class TwinTraceBridgeTests(unittest.TestCase):
    def setUp(self):
        rust_solver.clear_solver_cache()

    def _bridge(self, trace_marker=...):
        with patch.object(
            rust_solver,
            "rust_solve",
            return_value=_solver_response(trace_marker),
        ):
            return rust_solver.rust_solve_to_objects(_DummyState())

    def test_legacy_missing_trace_is_none_and_field_is_appended(self):
        result = self._bridge()
        self.assertIsNone(result.surviving_scenarios[0].twin_trace)

        positional = Scenario(
            {}, None, set(), None, None, None, {}, 3, set(), None, None
        )
        self.assertIsNone(positional.twin_trace)

    def test_exact_no_demon_trace_parses_and_round_trips(self):
        raw = {
            "actor_position": 2,
            "outcome": {"kind": "no_demon"},
        }
        trace = self._bridge(raw).surviving_scenarios[0].twin_trace

        self.assertEqual(trace.actor_position, 2)
        self.assertIs(trace.outcome.kind, TwinStartKind.NO_DEMON)
        self.assertEqual(trace.to_dict(), raw)
        self.assertEqual(rust_solver._parse_twin_trace(trace.to_dict(), 7), trace)

    def test_exact_swap_trace_parses_enums_and_round_trips(self):
        raw = {
            "actor_position": 7,
            "outcome": {
                "kind": "swap",
                "demon_occurrence_index": 2,
                "demon_anchor_position": 3,
                "neighbor_side": "next",
                "neighbor_position": 4,
                "neighbor_pre_swap_role": "Bombardier",
            },
        }
        trace = self._bridge(raw).surviving_scenarios[0].twin_trace

        self.assertEqual(trace.actor_position, 7)
        self.assertIs(trace.outcome.kind, TwinStartKind.SWAP)
        self.assertIs(trace.outcome.neighbor_side, TwinNeighborSide.NEXT)
        self.assertEqual(trace.outcome.neighbor_pre_swap_role, "Bombardier")
        self.assertEqual(trace.to_dict(), raw)
        self.assertEqual(rust_solver._parse_twin_trace(trace.to_dict(), 7), trace)

    def test_malformed_trace_payloads_fail_closed(self):
        valid_swap = {
            "actor_position": 7,
            "outcome": {
                "kind": "swap",
                "demon_occurrence_index": 0,
                "demon_anchor_position": 3,
                "neighbor_side": "previous",
                "neighbor_position": 4,
                "neighbor_pre_swap_role": "Scout",
            },
        }

        def changed(path, value=...):
            payload = deepcopy(valid_swap)
            target = payload
            for key in path[:-1]:
                target = target[key]
            if value is ...:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            return payload

        no_demon_extra = {
            "actor_position": 2,
            "outcome": {"kind": "no_demon", "neighbor_position": 1},
        }
        top_extra = deepcopy(valid_swap)
        top_extra["unexpected"] = True
        swap_extra = deepcopy(valid_swap)
        swap_extra["outcome"]["unexpected"] = True

        self.assertEqual(
            rust_solver._parse_twin_trace(valid_swap, 7)
            .outcome.demon_occurrence_index,
            0,
        )

        malformed = [
            ("top_not_dict", True),
            ("top_missing_actor", {"outcome": {"kind": "no_demon"}}),
            ("top_missing_outcome", {"actor_position": 2}),
            ("top_extra_key", top_extra),
            ("outcome_not_dict", {"actor_position": 2, "outcome": None}),
            ("actor_bool", changed(("actor_position",), True)),
            ("actor_negative", changed(("actor_position",), -1)),
            ("actor_zero", changed(("actor_position",), 0)),
            ("actor_beyond_board", changed(("actor_position",), 8)),
            ("missing_kind", changed(("outcome", "kind"))),
            ("unknown_kind", changed(("outcome", "kind"), "mystery")),
            ("kind_not_string", changed(("outcome", "kind"), True)),
            ("no_demon_extra_field", no_demon_extra),
            ("swap_missing_key", changed(("outcome", "neighbor_position"))),
            ("swap_extra_key", swap_extra),
            ("index_bool", changed(("outcome", "demon_occurrence_index"), False)),
            ("index_negative", changed(("outcome", "demon_occurrence_index"), -1)),
            ("index_out_of_u8", changed(("outcome", "demon_occurrence_index"), 256)),
            ("index_string", changed(("outcome", "demon_occurrence_index"), "0")),
            ("anchor_bool", changed(("outcome", "demon_anchor_position"), True)),
            ("anchor_negative", changed(("outcome", "demon_anchor_position"), -1)),
            ("anchor_zero", changed(("outcome", "demon_anchor_position"), 0)),
            ("anchor_beyond_board", changed(("outcome", "demon_anchor_position"), 8)),
            ("neighbor_zero", changed(("outcome", "neighbor_position"), 0)),
            ("neighbor_beyond_board", changed(("outcome", "neighbor_position"), 8)),
            ("side_unknown", changed(("outcome", "neighbor_side"), "left")),
            ("side_null", changed(("outcome", "neighbor_side"), None)),
            ("role_null", changed(("outcome", "neighbor_pre_swap_role"), None)),
            ("role_empty", changed(("outcome", "neighbor_pre_swap_role"), "")),
            ("role_whitespace", changed(("outcome", "neighbor_pre_swap_role"), "   ")),
        ]

        for name, payload in malformed:
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    rust_solver._parse_twin_trace(payload, 7)


if __name__ == "__main__":
    unittest.main()
