"""Bridge and strategy regressions for native Chancellor trace projection."""

import unittest
from unittest.mock import patch

import rust_solver
from solver import (
    Alignment,
    CardInfo,
    ChancellorTrace,
    DeckComposition,
    GameState,
    Scenario,
    SolverResult,
    TruthStatus,
    effective_alignment,
    truth_status,
)
from strategy import (
    _corruption_risk,
    _compute_position_fingerprint,
    _dreamer_effective_role,
    _druid_ground_truth,
    _execution_reveal_outcome,
    _execution_observation_key,
    _pd_ground_truth,
    _recommend_slayer,
    _wretch_kill_probability,
)


class _DummyState:
    def to_dict(self):
        return {"n_cards": 4, "bridge_test": "chancellor-trace"}


class ChancellorTraceTests(unittest.TestCase):
    def tearDown(self):
        rust_solver.clear_solver_cache()

    def test_rust_bridge_preserves_grouped_trace_and_status_provenance(self):
        payload = {
            "definite_evil": [1],
            "definite_good": [2, 3, 4],
            "bombardier_positions": [],
            "n_scenarios": 2,
            "n_surviving": 1,
            "reasoning": [],
            "surviving_scenarios": [
                {
                    "evil_positions": {"1": "Chancellor"},
                    "puppet_position": None,
                    "corrupted": [],
                    "pd_corrupted": None,
                    "doppelganger_position": None,
                    "drunk_position": None,
                    "alchemist_cures": {"3": 1},
                    "messed_up_by_evil": [2],
                    "chancellor_trace": {
                        "original_positions": [1, 3, 4],
                        "added_outcast_position": 2,
                        "added_outcast_role": "Plague Doctor",
                    },
                    "chancellor_conversion": 2,
                }
            ],
        }
        rust_solver.clear_solver_cache()

        with patch.object(rust_solver, "rust_solve", return_value=payload):
            result = rust_solver.rust_solve_to_objects(_DummyState())

        self.assertIsNotNone(result)
        scenario = result.surviving_scenarios[0]
        self.assertEqual(scenario.messed_up_by_evil, {2})
        self.assertEqual(scenario.alchemist_cures, {3: 1})
        self.assertEqual(scenario.chancellor_conversion, 2)
        self.assertEqual(
            scenario.chancellor_trace,
            ChancellorTrace(
                original_positions=[1, 3, 4],
                added_outcast_position=2,
                added_outcast_role="Plague Doctor",
            ),
        )

    def test_generated_role_drives_execution_and_dreamer_projection(self):
        state = GameState(
            n_cards=3,
            deck=DeckComposition([], ["Plague_Doctor"], ["Chancellor"], []),
            cards=[
                CardInfo(1, "Baker"),
                CardInfo(3, "Baker"),
            ],
            n_evil=1,
        )
        scenario = Scenario(
            evil_positions={1: "Chancellor"},
            chancellor_trace=ChancellorTrace(
                original_positions=[1, 3],
                added_outcast_position=2,
                added_outcast_role="Plague Doctor",
            ),
            chancellor_conversion=2,
        )

        self.assertEqual(
            _execution_reveal_outcome(2, scenario, state),
            ("Plague Doctor", False, False, False),
        )
        self.assertEqual(_dreamer_effective_role(2, scenario, state), "Plague Doctor")

    def test_legacy_bridge_and_positional_scenario_abi(self):
        payload = {
            "definite_evil": [],
            "definite_good": [1],
            "bombardier_positions": [],
            "n_scenarios": 1,
            "n_surviving": 1,
            "surviving_scenarios": [{
                "evil_positions": {},
                "puppet_position": None,
                "corrupted": [],
                "pd_corrupted": None,
                "doppelganger_position": None,
                "drunk_position": None,
                "alchemist_cures": {},
                "chancellor_conversion": 1,
            }],
        }
        with patch.object(rust_solver, "rust_solve", return_value=payload):
            bridged = rust_solver.rust_solve_to_objects(_DummyState())
        self.assertEqual(bridged.surviving_scenarios[0].chancellor_conversion, 1)
        self.assertIsNone(bridged.surviving_scenarios[0].chancellor_trace)

        positional = Scenario({}, None, set(), None, None, None, {}, 3)
        self.assertEqual(positional.chancellor_conversion, 3)
        self.assertEqual(positional.messed_up_by_evil, set())
        self.assertIsNone(positional.chancellor_trace)

    def test_generated_wretch_projects_alignment_druid_and_slayer_risk(self):
        state = GameState(
            n_cards=3,
            deck=DeckComposition([], ["Wretch"], ["Chancellor"], []),
            cards=[CardInfo(1, "Slayer"), CardInfo(3, "Baker")],
            n_evil=1,
        )
        scenario = Scenario(
            evil_positions={3: "Chancellor"},
            chancellor_trace=ChancellorTrace([3], 2, "Wretch"),
            chancellor_conversion=2,
        )
        result = SolverResult([], [1, 2], [], 1, 1, [scenario])

        self.assertEqual(effective_alignment(2, scenario, state), Alignment.EVIL)
        self.assertEqual(_druid_ground_truth([2], scenario, state), "none")
        self.assertEqual(_wretch_kill_probability(2, state, result), 1.0)

    def test_python_druid_projects_trace_and_hidden_outcasts(self):
        state = GameState(
            n_cards=3,
            deck=DeckComposition([], ["Plague_Doctor", "Drunk", "Doppelganger"], [], []),
            cards=[CardInfo(1, "Druid")],
        )
        generated = Scenario(
            {},
            chancellor_trace=ChancellorTrace([3], 2, "Plague_Doctor"),
            chancellor_conversion=2,
        )
        self.assertEqual(_druid_ground_truth([2], generated, state), "Plague Doctor")
        self.assertEqual(
            _druid_ground_truth([2], Scenario({}, doppelganger_position=2), state),
            "Doppelganger",
        )
        self.assertEqual(
            _druid_ground_truth([2], Scenario({}, drunk_position=2), state),
            "Drunk",
        )

    def test_reveal_fingerprint_splits_visible_generated_roles_only(self):
        state = GameState(
            n_cards=3,
            deck=DeckComposition([], ["Plague_Doctor", "Bombardier"], ["Chancellor"], []),
            cards=[],
        )
        trace = lambda role: Scenario(
            evil_positions={3: "Chancellor"},
            chancellor_trace=ChancellorTrace([3], 2, role),
            chancellor_conversion=2,
        )
        self.assertNotEqual(
            _compute_position_fingerprint(2, trace("Plague_Doctor"), state),
            _compute_position_fingerprint(2, trace("Bombardier"), state),
        )
        self.assertEqual(
            _compute_position_fingerprint(2, trace("Doppelganger"), state),
            _compute_position_fingerprint(2, trace("Drunk"), state),
        )

    def test_slayer_can_target_probable_evil_showing_as_bombardier(self):
        state = GameState(
            n_cards=3,
            deck=DeckComposition([], ["Bombardier"], ["Chancellor"], []),
            cards=[CardInfo(1, "Slayer")],
        )
        scenarios = [
            Scenario(evil_positions={2: "Chancellor"}) for _ in range(4)
        ] + [Scenario(evil_positions={3: "Chancellor"})]
        result = SolverResult([], [], [2], 5, 5, scenarios)
        recommendation = _recommend_slayer(1, state, result)
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.targets, [2])

    def test_slayer_can_target_probable_evil_showing_as_knight(self):
        state = GameState(
            n_cards=3,
            deck=DeckComposition(["Slayer", "Knight"], [], ["Chancellor"], []),
            cards=[CardInfo(1, "Slayer"), CardInfo(2, "Knight")],
        )
        scenarios = [
            Scenario(evil_positions={2: "Chancellor"}) for _ in range(4)
        ] + [Scenario(evil_positions={3: "Chancellor"})]
        result = SolverResult([], [], [], 5, 5, scenarios)
        recommendation = _recommend_slayer(1, state, result)
        self.assertIsNotNone(recommendation)
        self.assertEqual(recommendation.targets, [2])

    def test_resistant_generated_drunk_keeps_truth_pd_risk_and_damage_surfaces(self):
        state = GameState(
            n_cards=1,
            deck=DeckComposition([], ["Drunk"], ["Chancellor"], []),
            cards=[CardInfo(1, "Knight")],
            hp=5,
            wrong_exec_cost=5,
        )
        resistant = Scenario(
            {},
            chancellor_trace=ChancellorTrace([1], 1, "Drunk"),
            chancellor_conversion=1,
        )
        statused = Scenario(
            {},
            corrupted={1},
            drunk_position=1,
            chancellor_trace=ChancellorTrace([1], 1, "Drunk"),
            chancellor_conversion=1,
        )
        result = SolverResult([], [], [], 1, 1, [resistant])

        self.assertEqual(truth_status(1, resistant, state), TruthStatus.LYING)
        self.assertEqual(_pd_ground_truth(1, resistant, state), (False, None))
        self.assertEqual(_pd_ground_truth(1, statused, state), (False, None))
        self.assertEqual(_corruption_risk(1, result, state), 1.0)

        resistant_outcome = _execution_reveal_outcome(1, resistant, state)
        statused_outcome = _execution_reveal_outcome(1, statused, state)
        self.assertEqual(resistant_outcome, ("Drunk", False, False, False))
        self.assertEqual(statused_outcome, ("Drunk", False, False, True))
        self.assertEqual(
            _execution_observation_key(1, resistant_outcome, state),
            ("killed", "Drunk", False, False, 2),
        )
        self.assertEqual(
            _execution_observation_key(1, statused_outcome, state),
            ("killed", "Drunk", False, False, 6),
        )
        self.assertNotEqual(
            _compute_position_fingerprint(1, resistant, state),
            _compute_position_fingerprint(1, statused, state),
        )

        ordinary_corrupted = Scenario({}, corrupted={1})
        self.assertEqual(_pd_ground_truth(1, ordinary_corrupted, state), (True, None))


if __name__ == "__main__":
    unittest.main()
