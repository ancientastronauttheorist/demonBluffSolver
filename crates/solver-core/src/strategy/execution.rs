//! Execution decision tree: pick the best position to execute.
//!
//! Ported from strategy.py:1263-1620 (execution sub-tree only).
//! Steps 4-5 (abilities/reveals) are skipped because the simulation
//! pre-loads all reveals and abilities.

use std::cmp::Ordering;
use std::collections::HashSet;
use crate::knowledge_base::normalize_role;
use crate::types::{GameState, SolverResult};
#[allow(unused_imports)]
use super::{
    ExecutionPick, ExecutionReason,
    evil_probabilities, remaining_evil_bounds, corruption_risk,
    unrevealed_positions, tiebreak_score,
    execution_terminal_outcome, get_card_role, ExecutionTerminalOutcome,
    EXECUTION_IMMUNE_ROLES,
};
use super::forced::find_forced_execution;

/// Threshold: skip Knight check if a non-Knight has >= this evil probability.
const KNIGHT_CHECK_THRESHOLD: f64 = 0.65;

fn compare_probability_then_position(a: &(u8, f64), b: &(u8, f64)) -> Ordering {
    b.1.total_cmp(&a.1)
        .then_with(|| a.0.cmp(&b.0))
}

fn compare_active_candidates(
    a: &(u8, f64),
    b: &(u8, f64),
    state: &GameState,
    result: &SolverResult,
) -> Ordering {
    let ta = tiebreak_score(a.0, state, result);
    let tb = tiebreak_score(b.0, state, result);
    b.1.total_cmp(&a.1)
        .then_with(|| tb.0.total_cmp(&ta.0))
        .then_with(|| tb.1.total_cmp(&ta.1))
        .then_with(|| tb.2.total_cmp(&ta.2))
        .then_with(|| tb.3.total_cmp(&ta.3))
        .then_with(|| a.0.cmp(&b.0))
}

fn has_active_witch_block_evidence(state: &GameState, result: &SolverResult) -> bool {
    if state.executed_evil_roles.values()
        .any(|role| normalize_role(role) == "witch")
    {
        return false;
    }
    let unrevealed: HashSet<u8> = unrevealed_positions(state).into_iter().collect();
    if !state.blocked_positions.iter().any(|position| unrevealed.contains(position)) {
        return false;
    }

    let dead: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter()).copied().collect();
    result.surviving_scenarios.iter().any(|scenario| {
        let witch_positions: Vec<u8> = scenario.evil_positions.iter()
            .filter_map(|(&position, role)| {
                (normalize_role(role) == "witch").then_some(position)
            }).collect();
        let has_live = witch_positions.iter().any(|position| !dead.contains(position));
        let has_dead = witch_positions.iter().any(|position| dead.contains(position));
        // The shipped first-match Start adds one scalar block. Any real Witch
        // death reduces it, even if an ordinary duplicate Witch remains alive.
        has_live && !has_dead
    })
}

/// Pick the next execution target given solver results.
///
/// Returns None if:
/// - 0 surviving scenarios (error state)
/// - All evil already executed (win state)
/// - No valid target exists (stalemate)
///
/// The caller should re-solve after each execution and call this again.
///
/// `immunity_blocked` contains positions where Knight immunity was already
/// observed this round — avoids infinite loops re-targeting immune Knights.
pub fn pick_execution_target(
    state: &GameState,
    result: &SolverResult,
    immunity_blocked: &HashSet<u8>,
) -> Option<ExecutionPick> {
    // 1. Error: 0 scenarios
    if result.n_surviving == 0 {
        return None;
    }

    // 2. Terminal check. Native resolution loses on depleted HP before it
    // considers the evil-count win condition.
    let (_, max_remaining) = remaining_evil_bounds(state, result);
    match execution_terminal_outcome(state.hp, max_remaining == 0) {
        ExecutionTerminalOutcome::HpLoss | ExecutionTerminalOutcome::Win => return None,
        ExecutionTerminalOutcome::Continue => {}
    }

    let probs = evil_probabilities(state, result);
    // Dead positions (already-executed + night-killed by Lilis) cannot be the target
    // of a future execute action. Treat them uniformly as "executed" for selection.
    let executed: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter())
        .copied().collect();
    let bomb_set: HashSet<u8> = result.bombardier_positions.iter().copied().collect();

    // 3. Execute definite evil (skip Bombardier)
    for &pos in &result.definite_evil {
        if !executed.contains(&pos)
            && !bomb_set.contains(&pos)
            && !immunity_blocked.contains(&pos)
        {
            return Some(ExecutionPick {
                position: pos,
                reason: ExecutionReason::DefiniteEvil,
            });
        }
    }

    // 3.5. Knight free check
    // Executing an uncertain Knight is free info:
    //   - Real Knight (uncorrupted): blocked, confirms good, 0 HP
    //   - Evil disguised as Knight: evil dies
    //   - Corrupted Knight: loses immunity, costs HP (risky)
    // Gate: skip if a non-Knight candidate has >= 65% evil probability
    let non_knight_positions: HashSet<u8> = state.cards.iter()
        .filter(|c| {
            !EXECUTION_IMMUNE_ROLES.contains(&c.apparent_role.as_str())
                && !executed.contains(&c.position)
        })
        .map(|c| c.position)
        .collect();

    let best_non_knight_prob = non_knight_positions.iter()
        .map(|p| probs.get(p).copied().unwrap_or(0.0))
        .fold(0.0_f64, f64::max);

    if best_non_knight_prob < KNIGHT_CHECK_THRESHOLD {
        let mut knight_checks: Vec<(u8, f64, f64)> = Vec::new();
        for card in &state.cards {
            if EXECUTION_IMMUNE_ROLES.contains(&card.apparent_role.as_str())
                && !executed.contains(&card.position)
                && !immunity_blocked.contains(&card.position)
                && !result.definite_good.contains(&card.position)
                && !result.definite_evil.contains(&card.position)
            {
                let corr_risk = corruption_risk(card.position, state, result);
                let evil_prob = probs.get(&card.position).copied().unwrap_or(0.0);
                knight_checks.push((card.position, evil_prob, corr_risk));
            }
        }

        knight_checks.sort_by(|a, b| {
            b.1.total_cmp(&a.1)
                .then_with(|| a.2.total_cmp(&b.2))
                .then_with(|| a.0.cmp(&b.0))
        });

        if let Some(&(kpos, evil_prob, corr_risk)) = knight_checks.first() {
            if corr_risk == 0.0 {
                // Truly free: 0% corruption → execution blocked or kills evil
                return Some(ExecutionPick {
                    position: kpos,
                    reason: ExecutionReason::KnightFreeCheck {
                        evil_prob,
                        corruption_risk: corr_risk,
                    },
                });
            } else if corr_risk < 0.3 {
                // Mostly free: small corruption risk
                let corrupted_knight_cost = state.wrong_exec_cost + 4;
                let expected_cost = corr_risk * (1.0 - evil_prob) * corrupted_knight_cost as f64;
                if state.hp > corrupted_knight_cost
                    && expected_cost < state.wrong_exec_cost as f64 * 0.3
                {
                    return Some(ExecutionPick {
                        position: kpos,
                        reason: ExecutionReason::KnightFreeCheck {
                            evil_prob,
                            corruption_risk: corr_risk,
                        },
                    });
                }
            }
        }
    }

    // 5.5a. Forced execution (E5): DFS proof that execution guarantees win
    let mut all_uncertain: Vec<u8> = probs.iter()
        .filter(|(p, prob)| **prob > 0.0 && !executed.contains(p))
        .map(|(p, _)| *p)
        .collect();
    all_uncertain.sort_unstable();

    if !all_uncertain.is_empty() {
        if let Some(forced_pos) = find_forced_execution(state, result, &all_uncertain) {
            if !immunity_blocked.contains(&forced_pos) {
                return Some(ExecutionPick {
                    position: forced_pos,
                    reason: ExecutionReason::ForcedExecution,
                });
            }
        }
    }

    // 6. Probabilistic execution
    let _wrong_exec_budget = if state.wrong_exec_cost > 0 {
        state.hp / state.wrong_exec_cost
    } else {
        99
    };

    // Wretch positions: always wrong exec, no upside
    let wretch_positions: HashSet<u8> = state.cards.iter()
        .filter(|c| {
            c.apparent_role == "Wretch"
                && !result.definite_evil.contains(&c.position)
        })
        .map(|c| c.position)
        .collect();

    // Bombardier candidates (excluded from normal selection)
    let bombardier_candidates: Vec<(u8, f64)> = result.bombardier_positions.iter()
        .filter(|&&p| !executed.contains(&p) && probs.get(&p).copied().unwrap_or(0.0) > 0.0)
        .map(|&p| (p, probs[&p]))
        .collect();

    // Active candidates: exclude Bombardier, Wretch, immunity-blocked
    let mut active_probs: Vec<(u8, f64)> = probs.iter()
        .filter(|(p, _)| {
            !executed.contains(p)
                && !bomb_set.contains(p)
                && !wretch_positions.contains(p)
                && !immunity_blocked.contains(p)
        })
        .map(|(&p, &prob)| (p, prob))
        .collect();

    if !active_probs.is_empty() {
        // E4: Sort by (p_evil, tiebreak) for stable 50/50 resolution.
        // tiebreak_score returns (adjacency_bonus, corruption_penalty,
        // role_consistency, witch_boost) — higher is better on every term.
        active_probs.sort_by(|a, b| compare_active_candidates(a, b, state, result));

        let best_pos = active_probs[0].0;
        let best_prob = active_probs[0].1;

        // Bombardier-disguise override: if every non-Bombardier candidate is 0% evil
        // but a Bombardier candidate has non-zero evil probability, the remaining evil
        // MUST be a Bombardier disguise.
        if best_prob == 0.0 && !bombardier_candidates.is_empty() {
            if let Some(&(bomb_pos, bomb_prob)) = bombardier_candidates.iter()
                .max_by(|a, b| {
                    a.1.total_cmp(&b.1)
                        .then_with(|| b.0.cmp(&a.0))
                })
            {
                if bomb_prob > 0.0 {
                    return Some(ExecutionPick {
                        position: bomb_pos,
                        reason: ExecutionReason::BombardierDisguiseOverride { p_evil: bomb_prob },
                    });
                }
            }
        }

        // Witch hunting needs an unresolved current marker, not stale marker
        // list length retained by a historical fixture.
        if has_active_witch_block_evidence(state, result) {
            let mut best_witch: Option<(u8, f64)> = None;
            for &(p, _) in &active_probs {
                let witch_count = result.surviving_scenarios.iter()
                    .filter(|s| s.evil_positions.get(&p).map(|r| r.as_str()) == Some("Witch"))
                    .count();
                if witch_count > 0 {
                    let witch_prob = witch_count as f64 / result.n_surviving as f64;
                    if best_witch.map_or(true, |(_, bp)| witch_prob > bp) {
                        best_witch = Some((p, witch_prob));
                    }
                }
            }
            if let Some((wp, witch_prob)) = best_witch {
                let evil_prob = probs.get(&wp).copied().unwrap_or(0.0);
                if evil_prob > 0.5 && witch_prob > 0.3 {
                    return Some(ExecutionPick {
                        position: wp,
                        reason: ExecutionReason::WitchHunting { witch_prob },
                    });
                }
            }
        }

        // E3: HP-aware confidence thresholds — in the live game these gate
        // execution, but the simulation always follows the solver's top pick
        // (Core Rule 1: "Even for probabilistic executions: execute the
        // solver's top pick."). So thresholds are tracked for diagnostics
        // but never block execution here.

        return Some(ExecutionPick {
            position: best_pos,
            reason: ExecutionReason::Probabilistic { p_evil: best_prob },
        });
    }

    // 6c. Bombardier safety fallback: all high-prob candidates are Bombardier,
    // prefer a non-Bombardier uncertain position (wrong exec = HP cost, not game loss)
    if !bombardier_candidates.is_empty() {
        // Include Wretch — wrong exec on Wretch = HP cost, not game loss
        let mut safety_probs: Vec<(u8, f64)> = probs.iter()
            .filter(|(p, prob)| {
                !executed.contains(p)
                    && !bomb_set.contains(p)
                    && !immunity_blocked.contains(p)
                    && **prob > 0.0
            })
            .map(|(&p, &prob)| (p, prob))
            .collect();

        if !safety_probs.is_empty() {
            safety_probs.sort_by(compare_probability_then_position);
            let (safe_pos, safe_prob) = safety_probs[0];
            return Some(ExecutionPick {
                position: safe_pos,
                reason: ExecutionReason::BombardierSafetyFallback { p_evil: safe_prob },
            });
        }
    }

    // 7. No valid target
    None
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::*;
    use std::collections::{HashMap, HashSet};

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
        }
    }

    #[test]
    fn witch_hunting_uses_only_unresolved_current_block_evidence() {
        let result = SolverResult {
            definite_evil: vec![], definite_good: vec![], bombardier_positions: vec![],
            n_scenarios: 1, n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[(2, "Witch")])], reasoning: vec![],
        };
        let active = GameState {
            n_cards: 3,
            cards: vec![
                CardInfo { position: 1, ..CardInfo::default() },
                CardInfo { position: 2, ..CardInfo::default() },
            ],
            blocked_positions: vec![3],
            ..GameState::default()
        };
        assert!(has_active_witch_block_evidence(&active, &result));

        let duplicate_result = SolverResult {
            surviving_scenarios: vec![make_scenario(&[(1, "Witch"), (2, "Witch")])],
            ..result.clone()
        };
        assert!(has_active_witch_block_evidence(&active, &duplicate_result));
        let one_duplicate_dead = GameState { executed: vec![2], ..active.clone() };
        assert!(!has_active_witch_block_evidence(
            &one_duplicate_dead,
            &duplicate_result,
        ));

        // Exact public death evidence wins even if a collapsed scenario keeps
        // only a different live Witch identity and omits the dead one.
        let public_known_death = GameState {
            executed: vec![1],
            executed_evil_roles: HashMap::from([(1, "Witch".to_string())]),
            ..active.clone()
        };
        assert!(!has_active_witch_block_evidence(
            &public_known_death,
            &result,
        ));

        let already_revealed_marker = GameState {
            cards: vec![
                active.cards[0].clone(), active.cards[1].clone(),
                CardInfo { position: 3, ..CardInfo::default() },
            ],
            ..active.clone()
        };
        assert!(!has_active_witch_block_evidence(&already_revealed_marker, &result));

        let no_observed_block = GameState { blocked_positions: vec![], ..active.clone() };
        assert!(!has_active_witch_block_evidence(&no_observed_block, &result));

        let dead_witch = GameState { night_kills: vec![2], ..active };
        assert!(!has_active_witch_block_evidence(&dead_witch, &result));
    }

    #[test]
    fn test_pick_definite_evil() {
        let state = GameState {
            n_cards: 5,
            hp: 10,
            wrong_exec_cost: 5,
            executed: vec![],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![3],
            definite_good: vec![1, 2, 4, 5],
            bombardier_positions: vec![],
            n_scenarios: 1,
            n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[(3, "Pooka")])],
            reasoning: vec![],
        };
        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        assert_eq!(pick.position, 3);
        assert!(matches!(pick.reason, ExecutionReason::DefiniteEvil));
    }

    #[test]
    fn test_pick_always_executes_best_candidate() {
        // Even with low HP (budget=0), simulation always executes best candidate
        // (Core Rule 1: always follow solver's top pick)
        let state = GameState {
            n_cards: 5,
            hp: 4,
            wrong_exec_cost: 5,
            executed: vec![],
            cards: vec![
                CardInfo { position: 1, apparent_role: "Baker".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
                CardInfo { position: 3, apparent_role: "Oracle".to_string(), ..CardInfo::default() },
                CardInfo { position: 4, apparent_role: "Knitter".to_string(), ..CardInfo::default() },
                CardInfo { position: 5, apparent_role: "Medium".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 3,
            n_surviving: 3,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(2, "Pooka")]),
            ],
            reasoning: vec![],
        };
        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        // Position 1 has 67% probability — always executed regardless of budget
        assert_eq!(pick.position, 1);
    }

    #[test]
    fn test_pick_probabilistic_with_budget() {
        // HP=6, cost=5: budget=1. Evil at #1 with 80% prob → should execute
        let state = GameState {
            n_cards: 5,
            hp: 6,
            wrong_exec_cost: 5,
            executed: vec![],
            cards: vec![
                CardInfo { position: 1, apparent_role: "Baker".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
                CardInfo { position: 3, apparent_role: "Oracle".to_string(), ..CardInfo::default() },
                CardInfo { position: 4, apparent_role: "Knitter".to_string(), ..CardInfo::default() },
                CardInfo { position: 5, apparent_role: "Medium".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 5,
            n_surviving: 5,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(2, "Pooka")]),
            ],
            reasoning: vec![],
        };
        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        assert_eq!(pick.position, 1);
        // With budget=1 and 80% confidence, this should go through
        // (though it might be ForcedExecution if DFS proves it)
    }

    #[test]
    fn test_pick_skips_bombardier() {
        let state = GameState {
            n_cards: 3,
            hp: 10,
            wrong_exec_cost: 5,
            executed: vec![],
            cards: vec![
                CardInfo { position: 1, apparent_role: "Bombardier".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
                CardInfo { position: 3, apparent_role: "Oracle".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![1],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(2, "Pooka")]),
            ],
            reasoning: vec![],
        };
        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        // Should pick #2 (non-Bombardier) even though #1 has same probability
        assert_eq!(pick.position, 2);
    }

    #[test]
    fn equal_knight_checks_choose_lowest_position() {
        let state = GameState {
            n_cards: 2,
            hp: 4,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 2, apparent_role: "Knight".to_string(), ..CardInfo::default() },
                CardInfo { position: 1, apparent_role: "Knight".to_string(), ..CardInfo::default() },
            ],
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

        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        assert_eq!(pick.position, 1);
        assert!(matches!(pick.reason, ExecutionReason::KnightFreeCheck { .. }));
    }

    #[test]
    fn equal_knight_probabilities_prefer_the_lower_corruption_risk() {
        let state = GameState {
            n_cards: 2,
            hp: 4,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 1, apparent_role: "Knight".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Knight".to_string(), ..CardInfo::default() },
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

        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        assert_eq!(pick.position, 2);
        assert!(matches!(
            pick.reason,
            ExecutionReason::KnightFreeCheck { corruption_risk: 0.0, .. }
        ));
    }

    #[test]
    fn equal_probabilistic_candidates_choose_lowest_position() {
        let state = GameState {
            n_cards: 2,
            hp: 4,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 1, apparent_role: "Baker".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
            ],
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

        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        assert_eq!(pick.position, 1);
        assert!(matches!(pick.reason, ExecutionReason::Probabilistic { .. }));
    }

    #[test]
    fn active_candidate_comparator_is_independent_of_input_order() {
        let state = GameState {
            n_cards: 2,
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

        for mut candidates in [vec![(2, 0.5), (1, 0.5)], vec![(1, 0.5), (2, 0.5)]] {
            candidates.sort_by(|a, b| compare_active_candidates(a, b, &state, &result));
            assert_eq!(candidates, vec![(1, 0.5), (2, 0.5)]);
        }
    }

    #[test]
    fn equal_bombardier_overrides_choose_lowest_position() {
        let state = GameState {
            n_cards: 3,
            hp: 4,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 1, apparent_role: "Bombardier".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Bombardier".to_string(), ..CardInfo::default() },
                CardInfo { position: 3, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![3],
            bombardier_positions: vec![1, 2],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(2, "Pooka")]),
            ],
            reasoning: vec![],
        };

        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        assert_eq!(pick.position, 1);
        assert!(matches!(
            pick.reason,
            ExecutionReason::BombardierDisguiseOverride { .. }
        ));
    }

    #[test]
    fn equal_bombardier_safety_candidates_choose_lowest_position() {
        let state = GameState {
            n_cards: 3,
            hp: 4,
            wrong_exec_cost: 5,
            cards: vec![
                CardInfo { position: 1, apparent_role: "Wretch".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Wretch".to_string(), ..CardInfo::default() },
                CardInfo { position: 3, apparent_role: "Bombardier".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![3],
            n_scenarios: 3,
            n_surviving: 3,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka")]),
                make_scenario(&[(2, "Pooka")]),
                make_scenario(&[(3, "Pooka")]),
            ],
            reasoning: vec![],
        };

        let pick = pick_execution_target(&state, &result, &HashSet::new()).unwrap();
        assert_eq!(pick.position, 1);
        assert!(matches!(
            pick.reason,
            ExecutionReason::BombardierSafetyFallback { .. }
        ));
    }

    #[test]
    fn probability_comparator_is_independent_of_input_order() {
        for mut candidates in [vec![(2, 0.5), (1, 0.5)], vec![(1, 0.5), (2, 0.5)]] {
            candidates.sort_by(compare_probability_then_position);
            assert_eq!(candidates, vec![(1, 0.5), (2, 0.5)]);
        }
    }

    #[test]
    fn test_win_returns_none() {
        let state = GameState {
            n_cards: 3,
            hp: 10,
            wrong_exec_cost: 5,
            executed: vec![1],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![2, 3],
            bombardier_positions: vec![],
            n_scenarios: 1,
            n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[(1, "Pooka")])],
            reasoning: vec![],
        };
        // All evil executed → None (win)
        let pick = pick_execution_target(&state, &result, &HashSet::new());
        assert!(pick.is_none());
    }

    #[test]
    fn test_immunity_blocked_skipped() {
        let state = GameState {
            n_cards: 3,
            hp: 10,
            wrong_exec_cost: 5,
            executed: vec![],
            cards: vec![
                CardInfo { position: 1, apparent_role: "Baker".to_string(), ..CardInfo::default() },
                CardInfo { position: 2, apparent_role: "Hunter".to_string(), ..CardInfo::default() },
            ],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![1],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 1,
            n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[(1, "Pooka")])],
            reasoning: vec![],
        };
        // Block position 1 — should fall through to probabilistic
        let mut blocked = HashSet::new();
        blocked.insert(1);
        let pick = pick_execution_target(&state, &result, &blocked);
        // Position 1 is blocked, but it's the only evil → would still pick 2 or None
        // Since 2 has 0% evil probability with budget >= 2, it should still pick 2
        // Actually with only 1 scenario where #1 is evil, #2 has 0% → no valid target
        assert!(pick.is_none() || pick.unwrap().position != 1);
    }
}
