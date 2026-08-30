//! Forced execution DFS: find an execution that guarantees a win
//! across all reveal branches under the current HP budget.
//!
//! Ported from strategy.py:187-315.

use std::collections::{BTreeSet, HashMap, HashSet};
use crate::types::{GameState, Scenario, SolverResult};
use super::{
    apply_execution_damage, evil_probabilities, execution_consequence,
    execution_terminal_outcome, get_card_role, ExecutionConsequence,
    ExecutionTerminalOutcome,
};

/// Observed outcome if `pos` is executed in a given scenario.
/// Returns (revealed_role, was_evil, observed_corrupted, active_corrupted).
/// Ported from strategy.py:187-211.
pub fn execution_reveal_outcome(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
) -> (String, bool, bool, bool) {
    // Evil position
    if let Some(role) = scenario.evil_positions.get(&pos) {
        return (role.clone(), true, false, false);
    }

    // Puppet (evil but separate from evil_positions)
    if scenario.puppet_position == Some(pos) {
        return ("Puppet".to_string(), true, false, false);
    }

    // Drunk (disguised as villager)
    if scenario.drunk_position == Some(pos) {
        return (
            "Drunk".to_string(),
            false,
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
            scenario.corrupted.contains(&pos),
        );
    }

    if scenario.chancellor_added_outcast_position() == Some(pos) {
        if let Some(role) = scenario.chancellor_added_outcast_role() {
            let active_corrupted = scenario.corrupted.contains(&pos);
            let observed_corrupted = if crate::knowledge_base::normalize_role(role) == "drunk" {
                false
            } else {
                active_corrupted
            };
            return (role.to_string(), false, observed_corrupted, active_corrupted);
        }
    }

    // Normal card — use apparent role
    let role = get_card_role(pos, state)
        .unwrap_or("Unknown")
        .to_string();
    let corrupted = scenario.corrupted.contains(&pos);
    (role, false, corrupted, corrupted)
}

/// Canonical execution result visible to the player and therefore safe to use
/// as a DFS branch key. Protected true-Knight and Doppelganger-as-Knight
/// outcomes intentionally collapse: both leave the same face-up Knight alive.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum ExecutionObservation {
    Protected,
    Killed {
        revealed_role: String,
        was_evil: bool,
        was_corrupted: bool,
        hp_damage: i32,
    },
    BombardierLoss,
}

fn execution_observation(
    revealed_role: String,
    apparent_role: &str,
    was_evil: bool,
    observed_corrupted: bool,
    active_corrupted: bool,
    default_wrong_exec_cost: i32,
) -> ExecutionObservation {
    match execution_consequence(
        &revealed_role,
        apparent_role,
        was_evil,
        active_corrupted,
        default_wrong_exec_cost,
    ) {
        ExecutionConsequence::Protected => ExecutionObservation::Protected,
        ExecutionConsequence::Killed { hp_damage } => ExecutionObservation::Killed {
            revealed_role,
            was_evil,
            was_corrupted: observed_corrupted,
            hp_damage,
        },
        ExecutionConsequence::BombardierLoss => ExecutionObservation::BombardierLoss,
    }
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

    // Check terminal conditions in native order: HP loss precedes evil-count win.
    let all_done = indices.iter().all(|&idx| {
        let s = &scenarios[idx];
        !(1..=state.n_cards).any(|pos| {
            !global_executed.contains(&pos)
                && !executed_now.contains(&pos)
                && s.is_evil(pos)
        })
    });
    match execution_terminal_outcome(hp, all_done) {
        ExecutionTerminalOutcome::HpLoss => {
            memo.insert(key, (false, None));
            return (false, None);
        }
        ExecutionTerminalOutcome::Win => {
            memo.insert(key, (true, None));
            return (true, None);
        }
        ExecutionTerminalOutcome::Continue => {}
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
        // Partition by observable execution outcome. Hidden real identity must
        // not split two otherwise identical protected-Knight branches.
        let mut branches: HashMap<ExecutionObservation, Vec<usize>> = HashMap::new();
        let apparent_role = get_card_role(*pos, state).unwrap_or("");
        for &idx in indices {
            let (role, was_evil, observed_corrupted, active_corrupted) =
                execution_reveal_outcome(*pos, &scenarios[idx], state);
            let observation = execution_observation(
                role,
                apparent_role,
                was_evil,
                observed_corrupted,
                active_corrupted,
                state.wrong_exec_cost,
            );
            branches.entry(observation).or_default().push(idx);
        }

        let mut branch_ok = true;
        for (observation, branch_indices) in &branches {
            let next_hp = match observation {
                ExecutionObservation::BombardierLoss => {
                    branch_ok = false;
                    break;
                }
                ExecutionObservation::Protected => hp,
                ExecutionObservation::Killed { hp_damage, .. } => {
                    apply_execution_damage(hp, *hp_damage)
                }
            };

            if next_hp <= 0 {
                branch_ok = false;
                break;
            }

            let mut sorted_branch: Vec<usize> = branch_indices.clone();
            sorted_branch.sort();

            let mut next_executed = executed_now.clone();
            // This set means "resolved/unavailable" inside the planner. A
            // protected target remains physically alive but is confirmed good
            // in this branch and must not be attempted repeatedly.
            next_executed.insert(*pos);

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
            messed_up_by_evil: HashSet::new(),
            chancellor_trace: None,
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

    #[test]
    fn corrupted_knight_uses_nine_hp_in_forced_search() {
        let state = GameState {
            n_cards: 2,
            hp: 9,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 1, apparent_role: "Knight".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let first = make_scenario(&[(1, "Pooka")]);
        let mut second = make_scenario(&[(2, "Pooka")]);
        second.corrupted.insert(1);
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![first, second],
            reasoning: vec![],
        };

        // #1 is fatal in the corrupted-Knight branch. #2 is the only forcing move.
        assert_eq!(find_forced_execution(&state, &result, &[1, 2]), Some(2));
    }

    #[test]
    fn clean_doppelganger_as_knight_is_a_protected_information_branch() {
        let state = GameState {
            n_cards: 2,
            hp: 5,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 1, apparent_role: "Knight".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let first = make_scenario(&[(1, "Pooka")]);
        let mut second = make_scenario(&[(2, "Pooka")]);
        second.doppelganger_position = Some(1);
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![first, second],
            reasoning: vec![],
        };

        // Executing #1 either kills Pooka or is blocked and identifies #2 as evil.
        assert_eq!(find_forced_execution(&state, &result, &[1, 2]), Some(1));
    }

    #[test]
    fn forced_search_does_not_turn_zero_hp_into_a_win() {
        let state = GameState { n_cards: 1, ..GameState::default() };
        let scenarios = vec![make_scenario(&[])];
        let mut memo = HashMap::new();

        let outcome = can_force(
            &[0],
            &BTreeSet::new(),
            0,
            &[],
            &HashSet::new(),
            &scenarios,
            &state,
            &mut memo,
            0,
        );

        assert_eq!(outcome, (false, None));
    }

    #[test]
    fn protected_outcome_does_not_reveal_knight_vs_doppelganger_identity() {
        let state = GameState {
            n_cards: 3,
            hp: 5,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 1, apparent_role: "Knight".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
                CardInfo { position: 3, apparent_role: "Baker".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let first = make_scenario(&[(2, "Pooka")]);
        let mut second = make_scenario(&[(3, "Pooka")]);
        second.doppelganger_position = Some(1);
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![first, second],
            reasoning: vec![],
        };

        // Executing #1 is protected in both worlds and reveals no distinction.
        // With only 5 HP, neither remaining 50/50 target is safely forced.
        assert_eq!(find_forced_execution(&state, &result, &[1, 2, 3]), None);
    }

    #[test]
    fn drunk_observation_keeps_active_status_damage_separate_from_clean_evidence() {
        let state = GameState {
            n_cards: 2,
            hp: 5,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 1, apparent_role: "Knight".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let mut statused = make_scenario(&[(2, "Pooka")]);
        statused.drunk_position = Some(1);
        statused.corrupted.insert(1);
        let mut resistant = statused.clone();
        resistant.corrupted.clear();
        resistant.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![2],
            added_outcast_position: 1,
            added_outcast_role: "Drunk".to_string(),
        });

        assert_eq!(
            execution_reveal_outcome(1, &statused, &state),
            ("Drunk".to_string(), false, false, true),
        );
        assert_eq!(
            execution_reveal_outcome(1, &resistant, &state),
            ("Drunk".to_string(), false, false, false),
        );
        let statused_observation = execution_observation(
            "Drunk".to_string(), "Knight", false, false, true, 5,
        );
        let resistant_observation = execution_observation(
            "Drunk".to_string(), "Knight", false, false, false, 5,
        );
        assert!(matches!(
            &statused_observation,
            ExecutionObservation::Killed { hp_damage: 6, was_corrupted: false, .. }
        ));
        assert!(matches!(
            &resistant_observation,
            ExecutionObservation::Killed { hp_damage: 2, was_corrupted: false, .. }
        ));
        assert_ne!(statused_observation, resistant_observation);

        let result = SolverResult {
            definite_evil: vec![2],
            definite_good: vec![1],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![statused, resistant],
            reasoning: vec![],
        };
        // The statused branch dies at HP 5, so #1 cannot be a forced-safe move
        // even though the resistant branch would survive on the same evidence.
        assert_eq!(find_forced_execution(&state, &result, &[1]), None);
    }
}
