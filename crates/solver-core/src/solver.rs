/// Main solver entry point.

use rayon::prelude::*;
use crate::knowledge_base::normalize_role;
use crate::scenario::{
    build_scenarios, scenario_allows_anonymous_natural_outcast_role_at,
};
use crate::types::{GameState, Scenario, SolverResult};
use crate::validators::{
    current_data_role_at, known_evil_role, twin_may_have_replaced_current_data_at,
};

pub(crate) fn twin_origin_may_hold_bombardier(
    twin_origin: u8,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if scenario.twin_trace.is_some() || scenario.puppet_position == Some(twin_origin) {
        return false;
    }

    if !known_evil_role(twin_origin, scenario, state)
        .is_some_and(|role| normalize_role(role) == "twinminion")
    {
        return false;
    }

    // Twin receives the old current data from the selected Demon's adjacent
    // occurrence, never from its own runtime body. The endpoint's final role
    // cannot identify that old data: after the swap it carries Twin and may
    // reveal an unrelated Minion bluff. Until TwinTrace records the swap, any
    // geometrically feasible endpoint could have supplied authored Bombardier
    // data. An exact TwinTrace bypasses this coarse quarantine above.
    (1..=state.n_cards)
        .filter(|&source| source != twin_origin)
        .any(|source| twin_may_have_replaced_current_data_at(source, scenario, state))
}

pub(crate) fn hidden_ordinary_good_may_hold_bombardier(
    position: u8,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if scenario.twin_trace.is_some()
        && current_data_role_at(position, scenario, state).is_some()
    {
        return false;
    }

    let hidden_and_alive = !state.cards.iter().any(|card| card.position == position)
        && !state.executed.contains(&position)
        && !state.night_kills.contains(&position);

    hidden_and_alive
        && scenario_allows_anonymous_natural_outcast_role_at(
            position,
            "Bombardier",
            scenario,
            state,
        )
}

fn collect_bombardier_positions(
    state: &GameState,
    surviving: &[Scenario],
) -> Vec<u8> {
    let authored_roles = state.deck.all_roles();
    let authored_twin_and_bombardier = ["twinminion", "bombardier"]
        .iter()
        .all(|expected| {
            authored_roles
                .iter()
                .any(|role| normalize_role(role) == *expected)
        });

    (1..=state.n_cards)
        .filter(|&position| {
            let modeled_current_bombardier = surviving.iter().any(|scenario| {
                current_data_role_at(position, scenario, state)
                    .is_some_and(|role| normalize_role(&role) == "bombardier")
            });
            let hidden_ordinary_good_may_be_bombardier = surviving.iter().any(|scenario| {
                hidden_ordinary_good_may_hold_bombardier(position, scenario, state)
            });
            let pre_trace_twin_may_hold_bombardier = authored_twin_and_bombardier
                && surviving
                    .iter()
                    .any(|scenario| twin_origin_may_hold_bombardier(position, scenario, state));
            modeled_current_bombardier
                || hidden_ordinary_good_may_be_bombardier
                || pre_trace_twin_may_hold_bombardier
        })
        .collect()
}
use crate::validators::check_scenario;

/// Run the solver on the current game state.
pub fn solve(state: &GameState) -> SolverResult {
    let scenarios = build_scenarios(state);
    let n_scenarios = scenarios.len();

    let surviving: Vec<_> = scenarios
        .into_par_iter()
        .filter(|s| check_scenario(s, state))
        .collect();
    let n_surviving = surviving.len();

    let n = state.n_cards;
    let mut definite_evil = Vec::new();
    let mut definite_good = Vec::new();
    let mut bombardier_positions = Vec::new();

    if n_surviving > 0 {
        for pos in 1..=n {
            let all_evil = surviving.iter().all(|s| s.is_evil(pos));
            let all_good = surviving.iter().all(|s| !s.is_evil(pos));
            if all_evil { definite_evil.push(pos); }
            if all_good { definite_good.push(pos); }
        }
        bombardier_positions = collect_bombardier_positions(
            state,
            &surviving,
        );
    }

    definite_evil.sort_unstable();
    definite_good.sort_unstable();
    bombardier_positions.sort_unstable();

    SolverResult {
        definite_evil,
        definite_good,
        bombardier_positions,
        n_scenarios,
        n_surviving,
        surviving_scenarios: surviving,
        reasoning: Vec::new(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::strategy::execution::pick_execution_target;
    use crate::types::{
        CardInfo, ChancellorTrace, GameState, Scenario, ShamanTrace, TwinNeighborSide,
        TwinStartOutcome, TwinTrace,
    };
    use serde_json::json;
    use std::collections::HashSet;

    #[test]
    fn asc84_v5_keeps_drunk_bard_corruption_scenario() {
        let state = GameState::from_json(&json!({
            "n_cards": 9,
            "n_evil": 3,
            "cards": [
                {"position": 1, "apparent_role": "Scout", "info_parsed": {"evil_role": "Shaman", "distance": 2}},
                {"position": 2, "apparent_role": "Baker", "info_parsed": {"original_role": "original"}},
                {"position": 3, "apparent_role": "Enlightened", "info_parsed": {"direction": "CW"}},
                {"position": 4, "apparent_role": "Baker", "info_parsed": {"original_role": "Witness"}},
                {"position": 5, "apparent_role": "Medium", "info_parsed": {"good_position": 4, "good_role": "Baker"}},
                {"position": 7, "apparent_role": "Oracle", "info_parsed": {"targets": [2, 4], "minion_role": "Minion"}},
                {"position": 8, "apparent_role": "Scout", "info_parsed": {"evil_role": "Minion", "distance": 2}},
                {"position": 9, "apparent_role": "Bard", "info_parsed": {"corruption_distance": 1}}
            ],
            "night_kills": [6],
            "night_kill_evil_count": 0,
            "hp": 6,
            "wrong_exec_cost": 5,
            "board_villager_count": 5,
            "board_outcast_count": 1,
            "reveal_order": [1, 2, 3, 4, 6, 9, 5, 7, 8],
            "deck": {
                "villagers": ["Witness", "Oracle", "Scout", "Medium", "Bard", "Baker", "Enlightened"],
                "outcasts": ["Drunk"],
                "minions": ["Minion", "Shaman"],
                "demons": ["Lilis"]
            }
        })).unwrap();

        let result = solve(&state);
        assert!(result.n_surviving > 0);
        assert!(!result.definite_evil.contains(&1));

        let true_branch = result.surviving_scenarios.iter().any(|s| {
            s.drunk_position == Some(1)
                && s.corrupted.contains(&1)
                && s.evil_positions.get(&3).is_some_and(|r| r == "Minion")
                && s.evil_positions.get(&7).is_some_and(|r| r == "Shaman")
                && s.evil_positions.get(&8).is_some_and(|r| r == "Lilis")
        });
        assert!(true_branch);
    }

    #[test]
    fn generated_bombardier_is_always_excluded_from_good_targets() {
        let mut state = GameState::default();
        state.n_cards = 2;
        let mut generated = Scenario::default();
        generated.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![2],
            added_outcast_position: 1,
            added_outcast_role: "Bombardier".to_string(),
            affected_anchor_positions: vec![],
        });
        let mut evil = Scenario::default();
        evil.evil_positions.insert(1, "Lilis".to_string());

        assert_eq!(
            collect_bombardier_positions(&state, &[generated, evil]),
            vec![1],
        );
        assert!(collect_bombardier_positions(&state, &[]).is_empty());
    }

    #[test]
    fn current_bombardier_collection_includes_runtime_evil_shaman_copy() {
        let state = GameState {
            n_cards: 2,
            cards: vec![CardInfo {
                position: 1,
                apparent_role: "Bombardier".to_string(),
                ..CardInfo::default()
            }],
            ..GameState::default()
        };

        let mut shaman_current = Scenario::default();
        shaman_current
            .evil_positions
            .insert(1, "Pooka".to_string());
        shaman_current.shaman_trace = Some(ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Bombardier".to_string(),
            target_previous_roles: vec!["Pooka".to_string()],
        });
        assert_eq!(
            collect_bombardier_positions(&state, &[shaman_current]),
            // The copied runtime-Evil body and its still-surviving physical
            // Villager source are both fatal current-role targets.
            vec![1, 2],
        );

        let mut bluff_only = Scenario::default();
        bluff_only.evil_positions.insert(1, "Pooka".to_string());
        assert!(collect_bombardier_positions(&state, &[bluff_only]).is_empty());

        let mut drunk_display = Scenario::default();
        drunk_display.drunk_position = Some(1);
        assert!(collect_bombardier_positions(&state, &[drunk_display]).is_empty());

        let mut doppel_display = Scenario::default();
        doppel_display.doppelganger_position = Some(1);
        assert!(collect_bombardier_positions(&state, &[doppel_display]).is_empty());
    }

    #[test]
    fn hidden_good_bombardier_is_collected_without_a_scenario_role() {
        let state = GameState {
            n_cards: 2,
            n_evil: 1,
            cards: vec![CardInfo {
                position: 2,
                apparent_role: "Scout".to_string(),
                ..CardInfo::default()
            }],
            blocked_positions: vec![1],
            deck: crate::types::DeckComposition {
                villagers: vec!["Scout".to_string()],
                outcasts: vec!["Bombardier".to_string()],
                minions: vec!["Witch".to_string()],
                ..crate::types::DeckComposition::default()
            },
            ..GameState::default()
        };

        let result = solve(&state);
        assert!(result.surviving_scenarios.iter().any(|scenario| {
            scenario.evil_positions.get(&1)
                .is_some_and(|role| normalize_role(role) == "witch")
        }));
        assert!(result.surviving_scenarios.iter().any(|scenario| {
            scenario.evil_positions.get(&2)
                .is_some_and(|role| normalize_role(role) == "witch")
                && current_data_role_at(1, scenario, &state).is_none()
        }));
        assert_eq!(result.bombardier_positions, vec![1]);
    }

    #[test]
    fn pre_trace_twin_origin_is_a_possible_current_bombardier_target() {
        let mut state = GameState {
            n_cards: 3,
            cards: vec![CardInfo {
                position: 3,
                apparent_role: "Bombardier".to_string(),
                ..CardInfo::default()
            }],
            ..GameState::default()
        };
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];

        let mut stable_twin = Scenario::default();
        stable_twin
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        stable_twin
            .evil_positions
            .insert(2, "Pooka".to_string());

        let possible_bombardiers =
            collect_bombardier_positions(&state, &[stable_twin.clone()]);
        assert_eq!(possible_bombardiers, vec![1, 3]);

        let mut puppet_overlay = stable_twin.clone();
        puppet_overlay.puppet_position = Some(1);
        assert_eq!(collect_bombardier_positions(&state, &[puppet_overlay]), vec![3]);

        let result = SolverResult {
            definite_evil: vec![1, 2],
            definite_good: vec![3],
            bombardier_positions: possible_bombardiers,
            n_scenarios: 1,
            n_surviving: 1,
            surviving_scenarios: vec![stable_twin.clone()],
            reasoning: vec![],
        };
        assert_eq!(
            pick_execution_target(&state, &result, &HashSet::new())
                .map(|pick| pick.position),
            Some(2),
        );

        state.deck.outcasts.clear();
        assert_eq!(
            collect_bombardier_positions(&state, &[stable_twin.clone()]),
            vec![3],
        );

        state.deck.outcasts.push("Bombardier".to_string());
        state.deck.minions.clear();
        assert_eq!(
            collect_bombardier_positions(&state, &[stable_twin]),
            vec![3],
        );
    }

    #[test]
    fn revealed_twin_neighbor_role_cannot_disprove_former_bombardier_data() {
        let mut state = GameState {
            n_cards: 4,
            cards: vec![
                CardInfo {
                    position: 1,
                    apparent_role: "Bombardier".to_string(),
                    ..CardInfo::default()
                },
                CardInfo {
                    position: 2,
                    apparent_role: "Knight".to_string(),
                    ..CardInfo::default()
                },
                CardInfo {
                    position: 3,
                    apparent_role: "Dreamer".to_string(),
                    ..CardInfo::default()
                },
                CardInfo {
                    position: 4,
                    apparent_role: "Medium".to_string(),
                    ..CardInfo::default()
                },
            ],
            ..GameState::default()
        };
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];

        let mut scenario = Scenario::default();
        scenario
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());

        assert_eq!(collect_bombardier_positions(&state, &[scenario]), vec![1]);
    }

    #[test]
    fn traced_twin_uses_only_exact_current_bombardier_data() {
        let mut state = GameState {
            n_cards: 3,
            cards: vec![
                CardInfo {
                    position: 1,
                    apparent_role: "Scout".to_string(),
                    ..CardInfo::default()
                },
                CardInfo {
                    position: 2,
                    apparent_role: "Knight".to_string(),
                    ..CardInfo::default()
                },
                CardInfo {
                    position: 3,
                    apparent_role: "Pooka".to_string(),
                    ..CardInfo::default()
                },
            ],
            ..GameState::default()
        };
        state.deck.villagers = vec!["Scout".to_string(), "Knight".to_string()];
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];

        let trace = |neighbor_pre_swap_role: &str| TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: neighbor_pre_swap_role.to_string(),
            },
        };
        let mut scenario = Scenario::default();
        scenario
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.twin_trace = Some(trace("Bombardier"));

        assert!(!twin_origin_may_hold_bombardier(1, &scenario, &state));
        assert_eq!(collect_bombardier_positions(&state, &[scenario.clone()]), vec![1]);

        // Exact current Twin data at a hidden recipient is not an anonymous
        // natural-Outcast slot. The actor's exact Bombardier data remains the
        // only lethal current-data seat.
        state.cards.retain(|card| card.position != 2);
        assert!(!hidden_ordinary_good_may_hold_bombardier(
            2, &scenario, &state,
        ));
        assert_eq!(collect_bombardier_positions(&state, &[scenario.clone()]), vec![1]);

        // Authored Bombardier alone cannot re-enable the opaque Twin-origin
        // quarantine once an exact trace says the actor received Scout data.
        scenario.twin_trace = Some(trace("Scout"));
        assert!(collect_bombardier_positions(&state, &[scenario.clone()]).is_empty());

        // The same legacy world without a trace retains the conservative
        // quarantine because its neighbor's former current data is unknown.
        scenario.twin_trace = None;
        assert!(twin_origin_may_hold_bombardier(1, &scenario, &state));
        assert!(hidden_ordinary_good_may_hold_bombardier(
            2, &scenario, &state,
        ));
        assert_eq!(collect_bombardier_positions(&state, &[scenario]), vec![1, 2]);
    }

    #[test]
    fn exact_twin_self_swap_does_not_trigger_opaque_bomb_quarantine() {
        let mut state = GameState {
            n_cards: 3,
            cards: vec![
                CardInfo {
                    position: 1,
                    apparent_role: "Scout".to_string(),
                    ..CardInfo::default()
                },
                CardInfo {
                    position: 2,
                    apparent_role: "Pooka".to_string(),
                    ..CardInfo::default()
                },
                CardInfo {
                    position: 3,
                    apparent_role: "Knight".to_string(),
                    ..CardInfo::default()
                },
            ],
            ..GameState::default()
        };
        state.deck.villagers = vec!["Scout".to_string(), "Knight".to_string()];
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];

        let mut scenario = Scenario::default();
        scenario
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        scenario.evil_positions.insert(2, "Pooka".to_string());
        scenario.twin_trace = Some(TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 2,
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 1,
                neighbor_pre_swap_role: "Twin Minion".to_string(),
            },
        });

        assert!(!twin_origin_may_hold_bombardier(1, &scenario, &state));
        assert!(collect_bombardier_positions(&state, &[scenario.clone()]).is_empty());

        // Without the exact self-swap, #3 remains a feasible former source of
        // authored Bombardier data and conservatively quarantines Twin #1.
        scenario.twin_trace = None;
        assert!(twin_origin_may_hold_bombardier(1, &scenario, &state));
        assert_eq!(collect_bombardier_positions(&state, &[scenario]), vec![1]);
    }
}
