//! Forced execution DFS: find an execution that guarantees a win
//! across all reveal branches under the current HP budget.
//!
//! Ported from strategy.py:187-315.

use std::collections::{BTreeSet, HashMap, HashSet};
use crate::knowledge_base::normalize_role;
use crate::solver::{
    hidden_ordinary_good_may_hold_bombardier, twin_origin_may_hold_bombardier,
};
use crate::types::{GameState, Scenario, SolverResult};
use crate::validators::effective_role_at;
use super::{
    apply_execution_damage, evil_probabilities, execution_consequence,
    execution_terminal_outcome, get_card_role, is_terminal_loss_role,
    public_terminal_loss_position, scenario_terminal_loss_position,
    ExecutionConsequence, ExecutionTerminalOutcome,
};

/// Observed outcome if `pos` is executed in a given scenario.
/// Returns (revealed_role, was_evil, observed_corrupted, active_corrupted).
/// Ported from strategy.py:187-211.
pub fn execution_reveal_outcome(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
) -> (String, bool, bool, bool) {
    let role = effective_role_at(pos, scenario, state)
        .or_else(|| get_card_role(pos, state).map(str::to_string))
        .unwrap_or_else(|| "Unknown".to_string());
    let was_evil = scenario.is_evil(pos);
    if was_evil {
        // Runtime Evil determines the correct-execution outcome, while Shaman's
        // copied current role is what KillAndReveal publicly exposes.
        return (role, true, false, false);
    }

    let active_corrupted = scenario.corrupted.contains(&pos);
    let observed_corrupted = normalize_role(&role) != "drunk" && active_corrupted;
    (role, false, observed_corrupted, active_corrupted)
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

    let executed_set: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter()).copied().collect();
    let all_indices: Vec<usize> = (0..scenarios.len()).collect();
    // Exact modeled Bombardier seats are unsafe only while a surviving world
    // at the current DFS node still places Bombardier there. A prior public
    // outcome can remove those worlds and make the seat legal later. Opaque
    // pre-TwinTrace movement has no corresponding Scenario role, so every
    // aggregate Bomb seat supported by a Twin-origin world stays permanently
    // unsafe, even if another world models Bombardier at the same seat.
    let authored_roles = state.deck.all_roles();
    let authored_twin_and_bombardier = ["twinminion", "bombardier"]
        .iter()
        .all(|expected| {
            authored_roles
                .iter()
                .any(|role| normalize_role(role) == *expected)
        });
    let permanently_unsafe_bombardier_positions: HashSet<u8> = result
        .bombardier_positions
        .iter()
        .copied()
        .filter(|position| {
            authored_twin_and_bombardier
                && scenarios.iter().any(|scenario| {
                    twin_origin_may_hold_bombardier(*position, scenario, state)
                })
        })
        .collect();

    // Memoization table
    let mut memo: HashMap<(Vec<usize>, BTreeSet<u8>, i32), (bool, Option<u8>)> = HashMap::new();

    let (success, pos) = can_force(
        &all_indices,
        &BTreeSet::new(),
        state.hp,
        &ordered_candidates,
        &executed_set,
        &permanently_unsafe_bombardier_positions,
        scenarios,
        state,
        &mut memo,
        0,
    );

    if success { pos } else { None }
}

fn any_scenario_has_bombardier_risk(
    position: u8,
    indices: &[usize],
    scenarios: &[Scenario],
    state: &GameState,
) -> bool {
    indices.iter().any(|&idx| {
        let scenario = &scenarios[idx];
        effective_role_at(position, scenario, state)
            .is_some_and(|role| normalize_role(&role) == "bombardier")
            || hidden_ordinary_good_may_hold_bombardier(position, scenario, state)
    })
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
    permanently_unsafe_bombardier_positions: &HashSet<u8>,
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

    // Check terminal conditions in native order: Bombardier loss, HP loss,
    // then evil-count win. Existing public terminal state is authoritative;
    // otherwise evaluate each remaining exact world.
    let bombardier_loss = state
        .terminal_loss_role
        .as_deref()
        .is_some_and(is_terminal_loss_role)
        || public_terminal_loss_position(state).is_some()
        || indices.iter().any(|&idx| {
            scenario_terminal_loss_position(state, &scenarios[idx]).is_some()
        });
    let all_done = indices.iter().all(|&idx| {
        let s = &scenarios[idx];
        !(1..=state.n_cards).any(|pos| {
            !global_executed.contains(&pos)
                && !executed_now.contains(&pos)
                && s.is_evil(pos)
        })
    });
    match execution_terminal_outcome(bombardier_loss, hp, all_done) {
        ExecutionTerminalOutcome::BombardierLoss => {
            memo.insert(key, (false, None));
            return (false, None);
        }
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
        .filter(|&&p| {
            !executed_now.contains(&p)
                && !global_executed.contains(&p)
                && !permanently_unsafe_bombardier_positions.contains(&p)
                && !any_scenario_has_bombardier_risk(p, indices, scenarios, state)
        })
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
                permanently_unsafe_bombardier_positions,
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
    use crate::strategy::execution::pick_execution_target;
    use crate::strategy::ExecutionReason;
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
            shaman_trace: None,
            chancellor_trace: None,
            chancellor_conversion: None,
            twin_trace: None,
            pre_twin_current_roles: HashMap::new(),
            puppeteer_trace: None,
        }
    }

    #[test]
    fn forced_search_treats_night_killed_evil_as_terminally_dead() {
        let state = GameState {
            n_cards: 1,
            night_kills: vec![1],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![], definite_good: vec![], bombardier_positions: vec![],
            n_scenarios: 1, n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[(1, "Witch")])], reasoning: vec![],
        };
        assert_eq!(find_forced_execution(&state, &result, &[1]), None);
    }

    #[test]
    fn forced_search_never_selects_night_killed_candidate() {
        let state = GameState {
            n_cards: 2,
            night_kills: vec![1],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![], definite_good: vec![], bombardier_positions: vec![],
            n_scenarios: 1, n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[
                (1, "Pooka"), (2, "Witch"),
            ])],
            reasoning: vec![],
        };
        assert_eq!(find_forced_execution(&state, &result, &[1, 2]), Some(2));
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
    fn scenario_bombardier_can_become_legal_after_branch_pruning() {
        // #2 is Bombardier in the first world and Pooka in the second. It is
        // illegal at the root, but executing #1 publicly distinguishes the
        // worlds; the surviving second-world child may then execute #2.
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
        // The first move is #1, never the root-possible Bombardier at #2.
        let pos = find_forced_execution(&state, &result, &[1, 2, 3]);
        assert_eq!(pos, Some(1));
        assert_eq!(find_forced_execution(&state, &result, &[2]), None);

        let pick = pick_execution_target(&state, &result, &HashSet::new())
            .expect("root strategy should preserve the branch-safe forcing move");
        assert_eq!(pick.position, 1);
        assert!(matches!(pick.reason, ExecutionReason::ForcedExecution));
    }

    #[test]
    fn hidden_good_bombardier_blocks_root_but_not_proven_evil_child() {
        // #1 is hidden: it may be ordinary Good Bombardier while #2 is Witch,
        // although ordinary hidden Good identities have no Scenario role.
        // Executing revealed #2 first distinguishes the worlds; in the child
        // where #2 was Good, #1 is proven Witch and becomes safe to execute.
        let state = GameState {
            n_cards: 2,
            hp: 10,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 2, apparent_role: "Scout".to_string(), ..CardInfo::default() },
            ],
            blocked_positions: vec![1],
            deck: DeckComposition {
                villagers: vec!["Scout".to_string()],
                outcasts: vec!["Bombardier".to_string()],
                minions: vec!["Witch".to_string()],
                ..DeckComposition::default()
            },
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![1],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Witch")]),
                make_scenario(&[(2, "Witch")]),
            ],
            reasoning: vec![],
        };

        assert_eq!(find_forced_execution(&state, &result, &[1]), None);
        assert_eq!(find_forced_execution(&state, &result, &[1, 2]), Some(2));
    }

    #[test]
    fn pretrace_twin_bombardier_overlap_remains_permanently_unsafe() {
        // #2 is exact Bombardier in the first world and a stable Twin origin
        // in the second. Pruning the exact Bomb world must not clear the
        // unrepresented possibility that Twin moved Bomb data onto #2.
        let state = GameState {
            n_cards: 3,
            hp: 10,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 2, apparent_role: "Bombardier".to_string(), ..CardInfo::default() },
            ],
            deck: DeckComposition {
                outcasts: vec!["Bombardier".to_string()],
                minions: vec!["Twin Minion".to_string()],
                demons: vec!["Pooka".to_string()],
                ..DeckComposition::default()
            },
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
                make_scenario(&[(2, "Twin Minion"), (3, "Pooka")]),
            ],
            reasoning: vec![],
        };

        assert_eq!(find_forced_execution(&state, &result, &[1, 2, 3]), None);
        assert_eq!(find_forced_execution(&state, &result, &[2]), None);
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
            affected_anchor_positions: vec![],
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
