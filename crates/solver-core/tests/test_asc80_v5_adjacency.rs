//! Regression test for asc80_v5: adjacency-to-confirmed-evil tiebreaker.
//!
//! At step 6 of the replay (after executing #3=Pooka and #9=Minion), exactly
//! 3 scenarios remain. Positions #2 and #7 tie at 33% p_evil. Before the
//! adjacency tiebreaker was added, the solver picked #7 (lower corruption
//! risk), costing a wrong execution on the true Baker at #7.
//!
//! With the adjacency bonus, #2 now wins: it is adjacent to confirmed-evil
//! #3 on the 9-card circle, whereas #7 has no adjacent confirmed evils.

use solver_core::solver::solve;
use solver_core::strategy::execution::pick_execution_target;
use solver_core::types::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

#[test]
fn asc80_v5_adjacency_picks_pos_2_at_step_6() {
    // Load the case JSON
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../tests/cases_v2/asc80_v5.json");
    let content = std::fs::read_to_string(&path)
        .expect("failed to read asc80_v5.json");
    let value: serde_json::Value =
        serde_json::from_str(&content).expect("failed to parse asc80_v5.json");

    // Build GameState from the full case, then override the execution-loop
    // fields to the "after executed=[3,9]" checkpoint (step 5 → step 6 boundary).
    let mut state = GameState::from_json(&value).expect("parse GameState");

    // Override: we're at the point where only #3 (Pooka) and #9 (Minion) have
    // been executed. Slayer fired at #3 (survived) and PD #4 checked #3 (clean)
    // — both of those abilities are retained as in the case.
    state.executed = vec![3, 9];
    state.confirmed_evil = vec![3, 9];
    state.confirmed_good = vec![];
    state.executed_evil_roles = HashMap::from([
        (3u8, "Pooka".to_string()),
        (9u8, "Minion".to_string()),
    ]);
    state.executed_good_corrupted = HashMap::new();

    // HP before the wrong exec #7 that cost 5 HP — i.e. 10HP at this branch point.
    state.hp = 10;
    state.wrong_exec_cost = 5;
    state.night_kills = vec![];
    state.night_kill_evil_count = 0;

    // Slayer and PD ability results carry over as in the original case.
    // (state.slayer_results and state.pd_ability_results already populated from JSON)

    let result = solve(&state);
    assert!(result.n_surviving > 0, "expected surviving scenarios, got 0");

    // Sanity: both #2 and #7 should still be uncertain candidates.
    let pos2_prob = result.surviving_scenarios.iter()
        .filter(|s| s.is_evil(2)).count() as f64 / result.n_surviving as f64;
    let pos7_prob = result.surviving_scenarios.iter()
        .filter(|s| s.is_evil(7)).count() as f64 / result.n_surviving as f64;
    println!(
        "asc80_v5 step-6 branch: n_surviving={}, p(#2)={:.3}, p(#7)={:.3}",
        result.n_surviving, pos2_prob, pos7_prob
    );
    assert!(pos2_prob > 0.0, "#2 should have non-zero evil prob at this branch");

    let pick = pick_execution_target(&state, &result, &HashSet::new())
        .expect("expected an execution pick");

    // Primary assertion: adjacency tiebreaker makes #2 the top pick over #7.
    assert_eq!(
        pick.position, 2,
        "expected #2 (adjacent to confirmed evil #3), got #{}",
        pick.position
    );
}
