//! Forced execution DFS: find an execution that guarantees a win
//! across all reveal branches under the current HP budget.
//!
//! Ported from strategy.py:187-315.

use std::collections::{BTreeSet, HashMap, HashSet};
use crate::types::{GameState, Scenario, SolverResult};
use super::{evil_probabilities, EXECUTION_IMMUNE_ROLES, get_card_role};

/// Observed outcome if `pos` is executed in a given scenario.
/// Returns (revealed_role, was_evil, was_corrupted).
/// Ported from strategy.py:187-211.
pub fn execution_reveal_outcome(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
) -> (String, bool, bool) {
    // Evil position
    if let Some(role) = scenario.evil_positions.get(&pos) {
        return (role.clone(), true, false);
    }

    // Puppet (evil but separate from evil_positions)
    if scenario.puppet_position == Some(pos) {
        return ("Puppet".to_string(), true, false);
    }

    // Drunk (disguised as villager)
    if scenario.drunk_position == Some(pos) {
        return (
            "Drunk".to_string(),
            false,
            scenario.corrupted.contains(&pos),
        );
    }

    // Doppelganger (disguised as villager)
    if scenario.doppelganger_position == Some(pos) {
        return (
            "Doppelganger".to_string(),
            false,
            scenario.corrupted.contains(&pos),
        );
    }

    // Normal card — use apparent role
    let role = get_card_role(pos, state)
        .unwrap_or("Unknown")
        .to_string();
    (role, false, scenario.corrupted.contains(&pos))
}

/// Maximum scenarios for the DFS to avoid combinatorial explosion.
const MAX_SCENARIOS_FOR_DFS: usize = 500;
/// Maximum recursion depth to prevent stack overflow.
const MAX_DFS_DEPTH: usize = 8;

/// Return a position whose execution guarantees a win across all branches
/// under the current HP budget. Returns None if no forced win exists.
///
/// Ported from strategy.py:214-315.
pub fn find_forced_execution(
    state: &GameState,
    result: &SolverResult,
    candidate_positions: &[u8],
) -> Option<u8> {
    let scenarios = &result.surviving_scenarios;
    if scenarios.is_empty() || candidate_positions.is_empty() {
        return None;
    }
    // Skip DFS on very large scenario sets to avoid combinatorial explosion
    if scenarios.len() > MAX_SCENARIOS_FOR_DFS {
        return None;
    }

    // Order candidates by evil probability (highest first), bombardier last
    let probs = evil_probabilities(state, result);
    let bomb_set: HashSet<u8> = result.bombardier_positions.iter().copied().collect();
    let mut ordered_candidates: Vec<u8> = candidate_positions.to_vec();
    ordered_candidates.sort_by(|a, b| {
        let pa = probs.get(a).copied().unwrap_or(0.0);
        let pb = probs.get(b).copied().unwrap_or(0.0);
        pb.partial_cmp(&pa).unwrap()
            .then(bomb_set.contains(a).cmp(&bomb_set.contains(b)))
            .then(a.cmp(b))
    });

    let executed_set: HashSet<u8> = state.executed.iter().copied().collect();
    let all_indices: Vec<usize> = (0..scenarios.len()).collect();

    // Memoization table
    let mut memo: HashMap<(Vec<usize>, BTreeSet<u8>, i32), (bool, Option<u8>)> = HashMap::new();

    let (success, pos) = can_force(
        &all_indices,
        &BTreeSet::new(),
        state.hp,
        &ordered_candidates,
        &executed_set,
        scenarios,
        state,
        &mut memo,
        0,
    );

    if success { pos } else { None }
}

/// Recursive DFS with memoization. Checks if all evil can be eliminated
/// through a sequence of executions starting from the given scenario subset.
#[allow(clippy::too_many_arguments)]
fn can_force(
    indices: &[usize],
    executed_now: &BTreeSet<u8>,
    hp: i32,
    ordered_candidates: &[u8],
    global_executed: &HashSet<u8>,
    scenarios: &[Scenario],
    state: &GameState,
    memo: &mut HashMap<(Vec<usize>, BTreeSet<u8>, i32), (bool, Option<u8>)>,
    depth: usize,
) -> (bool, Option<u8>) {
    if depth > MAX_DFS_DEPTH {
        return (false, None);
    }
    let key = (indices.to_vec(), executed_now.clone(), hp);
    if let Some(&cached) = memo.get(&key) {
        return cached;
    }

    // Check if all evils are gone in all scenarios
    let all_done = indices.iter().all(|&idx| {
        let s = &scenarios[idx];
        !(1..=state.n_cards).any(|pos| {
            !global_executed.contains(&pos)
                && !executed_now.contains(&pos)
                && s.is_evil(pos)
        })
    });
    if all_done {
        memo.insert(key, (true, None));
        return (true, None);
    }

    // Try each candidate position
    let available: Vec<u8> = ordered_candidates.iter()
        .filter(|&&p| !executed_now.contains(&p) && !global_executed.contains(&p))
        .copied()
        .collect();

    if available.is_empty() {
        memo.insert(key, (false, None));
        return (false, None);
    }

    for pos in &available {
        // Partition scenarios by execution outcome
        let mut branches: HashMap<(String, bool, bool), Vec<usize>> = HashMap::new();
        for &idx in indices {
            let outcome = execution_reveal_outcome(*pos, &scenarios[idx], state);
            branches.entry(outcome).or_default().push(idx);
        }

        let mut branch_ok = true;
        for ((role, was_evil, was_corrupted), branch_indices) in &branches {
            let next_hp = if *was_evil {
                hp // Correct execution, no HP cost
            } else if role == "Bombardier" {
                // Executing good Bombardier = instant game loss
                branch_ok = false;
                break;
            } else if EXECUTION_IMMUNE_ROLES.contains(&role.as_str()) && !was_corrupted {
                // Knight immunity blocks — no HP cost, confirms good
                hp
            } else if role == "Drunk" {
                // Drunk-as-Knight costs 6 HP, regular Drunk costs 2 HP
                let apparent = get_card_role(*pos, state).unwrap_or("");
                if apparent == "Knight" {
                    hp - 6
                } else {
                    hp - 2
                }
            } else {
                hp - state.wrong_exec_cost
            };

            if next_hp <= 0 {
                branch_ok = false;
                break;
            }

            let mut sorted_branch: Vec<usize> = branch_indices.clone();
            sorted_branch.sort();

            let mut next_executed = executed_now.clone();
            // Knight immunity: position NOT removed (stays on board)
            if !(EXECUTION_IMMUNE_ROLES.contains(&role.as_str()) && !was_corrupted) {
                next_executed.insert(*pos);
            }

            let (can_win, _) = can_force(
                &sorted_branch,
                &next_executed,
                next_hp,
                ordered_candidates,
                global_executed,
                scenarios,
                state,
                memo,
                depth + 1,
            );

            if !can_win {
                branch_ok = false;
                break;
            }
        }

        if branch_ok {
            let key = (indices.to_vec(), executed_now.clone(), hp);
            memo.insert(key, (true, Some(*pos)));
            return (true, Some(*pos));
        }
    }

    let key = (indices.to_vec(), executed_now.clone(), hp);
    memo.insert(key, (false, None));
    (false, None)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use std::collections::HashSet;

    fn make_scenario(evil: &[(u8, &str)]) -> Scenario {
        Scenario {
            evil_positions: evil.iter().map(|(p, r)| (*p, r.to_string())).collect(),
            puppet_position: None,
            corrupted: HashSet::new(),
            pd_corrupted: None,
            doppelganger_position: None,
            drunk_position: None,
            alchemist_cures: HashMap::new(),
            chancellor_conversion: None,
        }
    }

    #[test]
    fn test_forced_definite_evil() {
        // Position 1 is evil in all scenarios — should be forced
        let state = GameState {
            n_cards: 3,
            hp: 10,
            wrong_exec_cost: 5,
            executed: vec![],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![1],
            definite_good: vec![2, 3],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(1, "Pooka")]),
            ],
            reasoning: vec![],
        };
        let pos = find_forced_execution(&state, &result, &[1, 2, 3]);
        assert_eq!(pos, Some(1));
    }

    #[test]
    fn test_forced_two_step() {
        // 2 evil among positions 1-3, HP=10, cost=5: can afford 1 wrong exec
        // Scenarios: {1,2} or {1,3} — executing 1 always hits, then 50/50
        let state = GameState {
            n_cards: 3,
            hp: 10,
            wrong_exec_cost: 5,
            executed: vec![],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![1],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka"), (2, "Witch")]),
                make_scenario(&[(1, "Pooka"), (3, "Witch")]),
            ],
            reasoning: vec![],
        };
        // Execute 1 (guaranteed evil), then 50/50 on 2 or 3
        // With HP=10 and cost=5, can afford 1 wrong exec → forced win exists
        let pos = find_forced_execution(&state, &result, &[1, 2, 3]);
        assert_eq!(pos, Some(1));
    }

    #[test]
    fn test_forced_no_budget() {
        // HP too low for any wrong exec
        let state = GameState {
            n_cards: 3,
            hp: 4,
            wrong_exec_cost: 5,
            executed: vec![],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(2, "Pooka")]),
            ],
            reasoning: vec![],
        };
        // 50/50, can't afford mistake → no forced execution
        let pos = find_forced_execution(&state, &result, &[1, 2, 3]);
        assert_eq!(pos, None);
    }

    #[test]
    fn test_bombardier_blocks_forced() {
        // Bombardier at position 2 blocks the path even if it's evil
        let state = GameState {
            n_cards: 3,
            hp: 10,
            wrong_exec_cost: 5,
            executed: vec![],
            cards: vec![
                CardInfo { position: 2, apparent_role: "Bombardier".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![2],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka")]),
                // Scenario where #2 is Bombardier disguise (evil)
                make_scenario(&[(2, "Pooka")]),
            ],
            reasoning: vec![],
        };
        // Executing 1: branch where evil → win, branch where good → cost 5, then #2 left
        // Executing 2: branch where evil → win, branch where good Bombardier → instant loss
        // Position 1 should be chosen (safe) — can force via 1 then 2 in surviving branch
        let pos = find_forced_execution(&state, &result, &[1, 2, 3]);
        assert_eq!(pos, Some(1));
    }
}
