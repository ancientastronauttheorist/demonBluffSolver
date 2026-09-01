import unittest
from copy import deepcopy
from unittest.mock import patch

import rust_solver
from solver import (
    CardInfo,
    DeckComposition,
    GameState,
    PuppeteerNeighborSide,
    PuppeteerStartKind,
    Scenario,
    TwinNeighborSide,
    TwinStartKind,
    TwinStartOutcome,
    TwinTrace,
    effective_role_at,
    puppet_erased_role_at,
)


class _DummyState:
    def to_dict(self):
        return {"n_cards": 5, "n_evil": 2}


def _solver_response(trace_marker=..., role_map_marker=...):
    scenario = {
        "evil_positions": {"1": "Puppeteer", "2": "Twin Minion"},
    }
    if trace_marker is not ...:
        scenario["puppeteer_trace"] = trace_marker
    if role_map_marker is not ...:
        scenario["pre_twin_current_roles"] = role_map_marker
    return {
        "definite_evil": [],
        "definite_good": [],
        "bombardier_positions": [],
        "n_scenarios": 1,
        "n_surviving": 1,
        "surviving_scenarios": [scenario],
        "reasoning": [],
    }


class PuppeteerTraceBridgeTests(unittest.TestCase):
    def setUp(self):
        rust_solver.clear_solver_cache()

    def _bridge(self, trace_marker=..., role_map_marker=...):
        with patch.object(
            rust_solver,
            "rust_solve",
            return_value=_solver_response(trace_marker, role_map_marker),
        ):
            return rust_solver.rust_solve_to_objects(_DummyState())

    def test_legacy_missing_trace_and_role_map_preserve_defaults(self):
        scenario = self._bridge().surviving_scenarios[0]

        self.assertIsNone(scenario.puppeteer_trace)
        self.assertEqual(scenario.pre_twin_current_roles, {})

    def test_exact_no_candidate_trace_parses_and_round_trips(self):
        raw = {
            "actor_position": 1,
            "outcome": {"kind": "no_candidate"},
        }
        trace = self._bridge(raw).surviving_scenarios[0].puppeteer_trace

        self.assertEqual(trace.actor_position, 1)
        self.assertIs(trace.outcome.kind, PuppeteerStartKind.NO_CANDIDATE)
        self.assertEqual(trace.to_dict(), raw)
        self.assertEqual(rust_solver._parse_puppeteer_trace(raw, 5), trace)

    def test_exact_conversion_and_pre_twin_map_parse_and_round_trip(self):
        raw = {
            "actor_position": 4,
            "outcome": {
                "kind": "converted",
                "candidate_occurrence_index": 1,
                "neighbor_side": "next",
                "target_position": 5,
                "erased_villager_role": "Scout",
            },
        }
        raw_roles = {
            "1": "Pooka",
            "2": "Twin Minion",
            "3": "Baker",
            "4": "Puppeteer",
            "5": "Scout",
        }
        scenario = self._bridge(raw, raw_roles).surviving_scenarios[0]
        trace = scenario.puppeteer_trace

        self.assertEqual(trace.actor_position, 4)
        self.assertIs(trace.outcome.kind, PuppeteerStartKind.CONVERTED)
        self.assertIs(trace.outcome.neighbor_side, PuppeteerNeighborSide.NEXT)
        self.assertEqual(trace.outcome.erased_villager_role, "Scout")
        self.assertEqual(trace.to_dict(), raw)
        self.assertEqual(scenario.pre_twin_current_roles, {
            1: "Pooka",
            2: "Twin Minion",
            3: "Baker",
            4: "Puppeteer",
            5: "Scout",
        })

    def test_exact_overlap_replays_twin_before_puppeteer(self):
        state = GameState(
            n_cards=5,
            deck=DeckComposition(
                villagers=["Baker", "Scout"],
                outcasts=[],
                minions=["Puppeteer", "Twin Minion"],
                demons=["Pooka"],
            ),
            cards=[
                CardInfo(1, "Baker"),
                CardInfo(2, "Twin Minion"),
                CardInfo(3, "Pooka"),
                CardInfo(4, "Scout"),
                CardInfo(5, "Puppeteer"),
            ],
        )
        twin_trace = TwinTrace(
            actor_position=4,
            outcome=TwinStartOutcome(
                kind=TwinStartKind.SWAP,
                demon_occurrence_index=0,
                demon_anchor_position=3,
                neighbor_side=TwinNeighborSide.PREVIOUS,
                neighbor_position=2,
                neighbor_pre_swap_role="Scout",
            ),
        )
        converted = rust_solver._parse_puppeteer_trace({
            "actor_position": 5,
            "outcome": {
                "kind": "converted",
                "candidate_occurrence_index": 0,
                "neighbor_side": "previous",
                "target_position": 4,
                "erased_villager_role": "Scout",
            },
        }, 5)
        scenario = Scenario(
            evil_positions={3: "Pooka", 4: "Twin Minion", 5: "Puppeteer"},
            puppet_position=4,
            twin_trace=twin_trace,
            pre_twin_current_roles={
                1: "Baker",
                2: "Scout",
                3: "Pooka",
                4: "Twin Minion",
                5: "Puppeteer",
            },
            puppeteer_trace=converted,
        )

        self.assertEqual(effective_role_at(4, scenario, state), "Puppet")
        self.assertEqual(puppet_erased_role_at(4, scenario), "Scout")
        self.assertEqual(effective_role_at(2, scenario, state), "Twin Minion")

    def test_exact_replay_uses_relocated_puppeteer_actor(self):
        state = GameState(
            n_cards=5,
            deck=DeckComposition(
                villagers=["Baker", "Scout"],
                outcasts=[],
                minions=["Puppeteer", "Twin Minion"],
                demons=["Pooka"],
            ),
            cards=[
                CardInfo(1, "Baker"),
                CardInfo(2, "Twin Minion"),
                CardInfo(3, "Pooka"),
                CardInfo(4, "Puppeteer"),
                CardInfo(5, "Scout"),
            ],
        )
        twin_trace = TwinTrace(
            actor_position=4,
            outcome=TwinStartOutcome(
                kind=TwinStartKind.SWAP,
                demon_occurrence_index=0,
                demon_anchor_position=3,
                neighbor_side=TwinNeighborSide.PREVIOUS,
                neighbor_position=2,
                neighbor_pre_swap_role="Puppeteer",
            ),
        )
        converted = rust_solver._parse_puppeteer_trace({
            "actor_position": 4,
            "outcome": {
                "kind": "converted",
                "candidate_occurrence_index": 0,
                "neighbor_side": "next",
                "target_position": 5,
                "erased_villager_role": "Scout",
            },
        }, 5)
        scenario = Scenario(
            evil_positions={2: "Puppeteer", 3: "Pooka", 4: "Twin Minion"},
            puppet_position=5,
            twin_trace=twin_trace,
            pre_twin_current_roles={
                1: "Baker",
                2: "Puppeteer",
                3: "Pooka",
                4: "Twin Minion",
                5: "Scout",
            },
            puppeteer_trace=converted,
        )

        self.assertEqual(effective_role_at(2, scenario, state), "Twin Minion")
        self.assertEqual(effective_role_at(4, scenario, state), "Puppeteer")
        self.assertEqual(effective_role_at(5, scenario, state), "Puppet")
        self.assertEqual(puppet_erased_role_at(5, scenario), "Scout")

    def test_exact_no_candidate_ignores_legacy_scalar_overlay(self):
        state = GameState(
            n_cards=1,
            deck=DeckComposition([], [], ["Puppeteer"], []),
            cards=[CardInfo(1, "Puppeteer")],
        )
        trace = rust_solver._parse_puppeteer_trace({
            "actor_position": 1,
            "outcome": {"kind": "no_candidate"},
        }, 1)
        scenario = Scenario(
            evil_positions={1: "Puppeteer"},
            puppet_position=1,
            pre_twin_current_roles={1: "Puppeteer"},
            puppeteer_trace=trace,
        )

        self.assertEqual(effective_role_at(1, scenario, state), "Puppeteer")
        self.assertIsNone(puppet_erased_role_at(1, scenario))

    def test_malformed_trace_payloads_fail_closed(self):
        valid = {
            "actor_position": 4,
            "outcome": {
                "kind": "converted",
                "candidate_occurrence_index": 0,
                "neighbor_side": "previous",
                "target_position": 3,
                "erased_villager_role": "Baker",
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
        outcome_extra = deepcopy(valid)
        outcome_extra["outcome"]["unexpected"] = True
        no_candidate_extra = {
            "actor_position": 1,
            "outcome": {"kind": "no_candidate", "target_position": 2},
        }

        malformed = [
            ("top_not_dict", True),
            ("top_missing_actor", {"outcome": {"kind": "no_candidate"}}),
            ("top_extra", top_extra),
            ("actor_zero", changed(("actor_position",), 0)),
            ("actor_beyond_board", changed(("actor_position",), 6)),
            ("actor_bool", changed(("actor_position",), True)),
            ("outcome_not_dict", {"actor_position": 1, "outcome": None}),
            ("missing_kind", changed(("outcome", "kind"))),
            ("unknown_kind", changed(("outcome", "kind"), "mystery")),
            ("no_candidate_extra", no_candidate_extra),
            ("converted_missing", changed(("outcome", "target_position"))),
            ("converted_extra", outcome_extra),
            ("index_bool", changed(("outcome", "candidate_occurrence_index"), False)),
            ("index_negative", changed(("outcome", "candidate_occurrence_index"), -1)),
            ("side_unknown", changed(("outcome", "neighbor_side"), "left")),
            ("target_zero", changed(("outcome", "target_position"), 0)),
            ("target_beyond_board", changed(("outcome", "target_position"), 6)),
            ("role_null", changed(("outcome", "erased_villager_role"), None)),
            ("role_empty", changed(("outcome", "erased_villager_role"), "")),
            ("role_whitespace", changed(("outcome", "erased_villager_role"), "   ")),
        ]

        for name, payload in malformed:
            with self.subTest(name=name):
                with self.assertRaises((TypeError, ValueError)):
                    rust_solver._parse_puppeteer_trace(payload, 5)

    def test_malformed_pre_twin_maps_fail_closed(self):
        malformed = [
            True,
            [],
            {1: "Scout"},
            {"position": "Scout"},
            {"0": "Scout"},
            {"6": "Scout"},
            {"1": None},
            {"1": ""},
            {"1": "   "},
            {"1": "Scout"},
            {
                "1": "Scout",
                "01": "Baker",
                "2": "Scout",
                "3": "Pooka",
                "4": "Twin Minion",
                "5": "Puppeteer",
            },
        ]

        for payload in malformed:
            with self.subTest(payload=payload):
                with self.assertRaises((TypeError, ValueError)):
                    rust_solver._parse_pre_twin_current_roles(payload, 5)


if __name__ == "__main__":
    unittest.main()
