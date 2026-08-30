"""Native Witness/Chancellor live-parser and reveal-strategy regressions."""

import unittest

from game_loop import _parse_clue_from_memory
from solver import ChancellorTrace, DeckComposition, GameState, Scenario
from strategy import _compute_position_fingerprint, _witness_observation_support


def _memory_witness(clue: str, targets: list[int]) -> dict:
    return {
        "position": 1,
        "true_role": "Witness",
        "disguise": "Witness",
        "clue_text": clue,
        "acted_infos": [{"desc": clue, "targets": targets}],
        "runtime_data": None,
        "ability_used": False,
        "uses": 0,
    }


class WitnessMemoryParserTests(unittest.TestCase):
    def test_exact_positive_surface_requires_matching_single_reference(self):
        parsed = _parse_clue_from_memory(
            _memory_witness("#3 was affected by an Evil", [3]),
            n_cards=4,
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.info_parsed, {"affected_position": 3})

        self.assertIsNone(_parse_clue_from_memory(
            _memory_witness("#3 was affected by an Evil", [2]),
            n_cards=4,
        ))
        self.assertIsNone(_parse_clue_from_memory(
            _memory_witness("#3 was affected by an Evil", []),
            n_cards=4,
        ))
        self.assertIsNone(_parse_clue_from_memory(
            _memory_witness("#3 was affected by an Evil", [3, 4]),
            n_cards=4,
        ))

    def test_exact_no_surface_maps_empty_references_to_zero(self):
        parsed = _parse_clue_from_memory(
            _memory_witness("NO character was affected by an Evil", []),
            n_cards=4,
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.info_parsed, {"affected_position": 0})

        self.assertIsNone(_parse_clue_from_memory(
            _memory_witness("NO character was affected by an Evil", [2]),
            n_cards=4,
        ))

        desc_only = _memory_witness("", [])
        desc_only["acted_infos"][0]["desc"] = (
            "NO character was affected by an Evil"
        )
        parsed = _parse_clue_from_memory(desc_only, n_cards=4)
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.info_parsed, {"affected_position": 0})

    def test_target_only_legacy_fallback_is_bounded(self):
        parsed = _parse_clue_from_memory(
            _memory_witness("", [4]),
            n_cards=4,
        )
        self.assertIsNotNone(parsed)
        self.assertEqual(parsed.info_parsed, {"affected_position": 4})
        self.assertIsNone(_parse_clue_from_memory(
            _memory_witness("", [5]),
            n_cards=4,
        ))
        self.assertIsNone(_parse_clue_from_memory(
            _memory_witness("Witness text changed unexpectedly", [4]),
            n_cards=4,
        ))


class WitnessRevealFingerprintTests(unittest.TestCase):
    def setUp(self):
        self.state = GameState(
            n_cards=4,
            deck=DeckComposition(["Witness"], ["Wretch"], ["Chancellor"], []),
            cards=[],
        )

    def test_truthful_uses_markers_and_includes_night_dead_cards(self):
        scenario = Scenario(
            evil_positions={4: "Chancellor"},
            messed_up_by_evil={2},
            chancellor_trace=ChancellorTrace(
                [4], 3, "Wretch", affected_anchor_positions=[1]
            ),
        )
        self.state.night_kills = [4]

        # Attempted anchor #1 and erased Villager #3 are provenance only.
        self.assertEqual(_witness_observation_support(1, scenario, self.state), (2, 4))

    def test_liar_uses_full_board_complement_and_self_is_legal(self):
        scenario = Scenario(
            evil_positions={1: "Chancellor"},
            messed_up_by_evil={2, 4},
        )
        self.assertEqual(_witness_observation_support(1, scenario, self.state), (1, 3))

        scenario.messed_up_by_evil = {1, 2, 3, 4}
        self.assertEqual(_witness_observation_support(1, scenario, self.state), (0,))

    def test_truthful_no_marker_uses_no_result(self):
        scenario = Scenario(evil_positions={4: "Chancellor"})
        self.assertEqual(_witness_observation_support(1, scenario, self.state), (0,))

    def test_fingerprint_uses_surviving_markers_not_trace_history(self):
        common = dict(
            evil_positions={4: "Chancellor"},
            messed_up_by_evil={3},
        )
        first = Scenario(
            **common,
            chancellor_trace=ChancellorTrace(
                [1], 2, "Wretch", affected_anchor_positions=[1]
            ),
        )
        second = Scenario(
            **common,
            chancellor_trace=ChancellorTrace(
                [2], 2, "Wretch", affected_anchor_positions=[4]
            ),
        )
        changed_marker = Scenario(
            evil_positions={4: "Chancellor"},
            messed_up_by_evil={2},
            chancellor_trace=second.chancellor_trace,
        )

        self.assertNotEqual(
            first.chancellor_original_villager_positions(),
            second.chancellor_original_villager_positions(),
        )
        self.assertEqual(
            _compute_position_fingerprint(1, first, self.state),
            _compute_position_fingerprint(1, second, self.state),
        )
        self.assertNotEqual(
            _compute_position_fingerprint(1, second, self.state),
            _compute_position_fingerprint(1, changed_marker, self.state),
        )


if __name__ == "__main__":
    unittest.main()
