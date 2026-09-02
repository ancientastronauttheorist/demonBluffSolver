from copy import deepcopy
from dataclasses import fields
import unittest
from unittest.mock import patch

import rust_solver
from solver import (
    BluffAcquisitionSourceKind,
    Scenario,
)


class _DummyState:
    def to_dict(self):
        return {
            "n_cards": 5,
            "n_evil": 2,
            "twin_recipient_bluff_context": {
                "rule_version": "twin_recipient_bluff_native_v1",
                "recipient_position": 3,
                "acquisition_ordinal": 4,
                "duplicate_pool": ["Scout", "Scout", "Confessor"],
                "unique_pool": ["Witness", "Confessor"],
                "bluff_must_include_at_recipient": [],
            },
        }


def _solver_response(trace_marker=...):
    scenario = {
        "evil_positions": {"1": "Twin Minion", "2": "Lilis"},
    }
    if trace_marker is not ...:
        scenario["twin_recipient_bluff_trace"] = trace_marker
    return {
        "definite_evil": [],
        "definite_good": [],
        "bombardier_positions": [],
        "n_scenarios": 1,
        "n_surviving": 1,
        "surviving_scenarios": [scenario],
        "reasoning": [],
    }


class TwinRecipientBluffBridgeTests(unittest.TestCase):
    def setUp(self):
        rust_solver.clear_solver_cache()

    def _bridge(self, trace_marker=...):
        rust_solver.clear_solver_cache()
        with patch.object(
            rust_solver,
            "rust_solve",
            return_value=_solver_response(trace_marker),
        ):
            return rust_solver.rust_solve_to_objects(_DummyState())

    def test_legacy_missing_trace_is_none_and_field_is_appended(self):
        scenario = self._bridge().surviving_scenarios[0]

        self.assertIsNone(scenario.twin_recipient_bluff_trace)
        self.assertEqual(
            fields(Scenario)[-1].name,
            "twin_recipient_bluff_trace",
        )

    def test_every_tagged_source_parses_and_round_trips(self):
        expected_kinds = {
            "duplicate_pool": BluffAcquisitionSourceKind.DUPLICATE_POOL,
            "unique_pool": BluffAcquisitionSourceKind.UNIQUE_POOL,
            "bluff_must_include": (
                BluffAcquisitionSourceKind.BLUFF_MUST_INCLUDE
            ),
        }
        for occurrence_index, (raw_kind, expected_kind) in enumerate(
            expected_kinds.items()
        ):
            with self.subTest(kind=raw_kind):
                raw = {
                    "recipient_position": 3,
                    "acquisition_ordinal": 4,
                    "bluff_role": "Confessor",
                    "source": {
                        "kind": raw_kind,
                        "occurrence_index": occurrence_index,
                    },
                }

                trace = (
                    self._bridge(raw)
                    .surviving_scenarios[0]
                    .twin_recipient_bluff_trace
                )

                self.assertEqual(trace.recipient_position, 3)
                self.assertEqual(trace.acquisition_ordinal, 4)
                self.assertEqual(trace.bluff_role, "Confessor")
                self.assertIs(trace.source.kind, expected_kind)
                self.assertEqual(
                    trace.source.occurrence_index,
                    occurrence_index,
                )
                self.assertEqual(trace.to_dict(), raw)
                self.assertEqual(
                    rust_solver._parse_twin_recipient_bluff_trace(raw, 5),
                    trace,
                )

    def test_u16_boundary_values_are_preserved(self):
        raw = {
            "recipient_position": 5,
            "acquisition_ordinal": 65535,
            "bluff_role": "Witness",
            "source": {
                "kind": "unique_pool",
                "occurrence_index": 65535,
            },
        }

        trace = rust_solver._parse_twin_recipient_bluff_trace(raw, 5)

        self.assertEqual(trace.acquisition_ordinal, 65535)
        self.assertEqual(trace.source.occurrence_index, 65535)
        self.assertEqual(trace.to_dict(), raw)

    def test_malformed_trace_payloads_fail_closed(self):
        valid = {
            "recipient_position": 3,
            "acquisition_ordinal": 4,
            "bluff_role": "Confessor",
            "source": {
                "kind": "duplicate_pool",
                "occurrence_index": 1,
            },
        }

        def changed(path, value=...):
            payload = deepcopy(valid)
            target = payload
            for key in path[:-1]:
                target = target[key]
            if value is ...:
                del target[path[-1]]
            else:
                target[path[-1]] = value
            return payload

        top_extra = deepcopy(valid)
        top_extra["unexpected"] = True
        source_extra = deepcopy(valid)
        source_extra["source"]["unexpected"] = True

        malformed = [
            ("top_not_dict", True),
            ("missing_recipient", changed(("recipient_position",))),
            ("missing_ordinal", changed(("acquisition_ordinal",))),
            ("missing_role", changed(("bluff_role",))),
            ("missing_source", changed(("source",))),
            ("top_extra", top_extra),
            ("recipient_bool", changed(("recipient_position",), True)),
            ("recipient_zero", changed(("recipient_position",), 0)),
            ("recipient_beyond_board", changed(("recipient_position",), 6)),
            ("ordinal_bool", changed(("acquisition_ordinal",), False)),
            ("ordinal_negative", changed(("acquisition_ordinal",), -1)),
            ("ordinal_over_u16", changed(("acquisition_ordinal",), 65536)),
            ("ordinal_string", changed(("acquisition_ordinal",), "4")),
            ("role_null", changed(("bluff_role",), None)),
            ("role_empty", changed(("bluff_role",), "")),
            ("role_whitespace", changed(("bluff_role",), "   ")),
            ("source_not_dict", changed(("source",), [])),
            ("source_missing_kind", changed(("source", "kind"))),
            (
                "source_missing_occurrence",
                changed(("source", "occurrence_index")),
            ),
            ("source_extra", source_extra),
            ("kind_bool", changed(("source", "kind"), True)),
            ("kind_unknown", changed(("source", "kind"), "script_pool")),
            (
                "occurrence_bool",
                changed(("source", "occurrence_index"), False),
            ),
            (
                "occurrence_negative",
                changed(("source", "occurrence_index"), -1),
            ),
            (
                "occurrence_over_u16",
                changed(("source", "occurrence_index"), 65536),
            ),
            (
                "occurrence_string",
                changed(("source", "occurrence_index"), "1"),
            ),
        ]

        for name, payload in malformed:
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    rust_solver._parse_twin_recipient_bluff_trace(payload, 5)


if __name__ == "__main__":
    unittest.main()
