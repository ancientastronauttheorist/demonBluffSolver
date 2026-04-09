/// Main solver entry point.

use rayon::prelude::*;
use crate::scenario::build_scenarios;
use crate::types::{GameState, SolverResult};
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
        for card in &state.cards {
            if card.apparent_role == "Bombardier" && !definite_evil.contains(&card.position) {
                bombardier_positions.push(card.position);
            }
        }
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
