/// Main solver entry point.

use rayon::prelude::*;
use crate::knowledge_base::normalize_role;
use crate::scenario::build_scenarios;
use crate::types::{GameState, Scenario, SolverResult};

fn collect_bombardier_positions(
    state: &GameState,
    surviving: &[Scenario],
    definite_evil: &[u8],
) -> Vec<u8> {
    let mut positions: Vec<u8> = state.cards.iter()
        .filter(|card| {
            normalize_role(&card.apparent_role) == "bombardier"
                && !definite_evil.contains(&card.position)
        })
        .map(|card| card.position)
        .collect();

    for pos in 1..=state.n_cards {
        if definite_evil.contains(&pos) {
            continue;
        }
        if surviving.iter().any(|scenario| {
            !scenario.is_evil(pos)
                && scenario.chancellor_added_outcast_position() == Some(pos)
                && scenario
                    .chancellor_added_outcast_role()
                    .is_some_and(|role| normalize_role(role) == "bombardier")
        }) {
            positions.push(pos);
        }
    }
    positions.sort_unstable();
    positions.dedup();
    positions
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
            &definite_evil,
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
    use crate::types::{ChancellorTrace, GameState, Scenario};
    use serde_json::json;

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
            collect_bombardier_positions(&state, &[generated, evil], &[]),
            vec![1],
        );
        assert!(collect_bombardier_positions(&state, &[], &[1]).is_empty());
    }
}
