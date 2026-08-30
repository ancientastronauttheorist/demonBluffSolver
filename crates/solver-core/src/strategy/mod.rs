//! Strategy module: execution decision logic ported from strategy.py.
//!
//! This contains only the execution sub-tree of the full Python strategy.
//! Ability timing, reveal entropy, and shallow lookahead stay in Python
//! because the simulation test pre-loads all reveals and abilities.

pub mod execution;
pub mod forced;

use std::collections::{HashMap, HashSet};
use crate::types::{GameState, SolverResult};
use crate::geometry::adjacent_positions;
use crate::knowledge_base::normalize_role;
use crate::validators::effective_role_at;

/// Result of pick_execution_target().
#[derive(Debug, Clone)]
pub struct ExecutionPick {
    pub position: u8,
    pub reason: ExecutionReason,
}

/// Why this position was chosen for execution.
#[derive(Debug, Clone)]
pub enum ExecutionReason {
    DefiniteEvil,
    KnightFreeCheck { evil_prob: f64, corruption_risk: f64 },
    ForcedExecution,
    Probabilistic { p_evil: f64 },
    BombardierDisguiseOverride { p_evil: f64 },
    WitchHunting { witch_prob: f64 },
    BombardierSafetyFallback { p_evil: f64 },
}

// ── Helper functions ──

/// Per-position probability of being evil across surviving scenarios.
/// Ported from strategy.py:91-102.
pub fn evil_probabilities(state: &GameState, result: &SolverResult) -> HashMap<u8, f64> {
    if result.n_surviving == 0 {
        return HashMap::new();
    }
    let mut probs = HashMap::new();
    let executed: HashSet<u8> = state.executed.iter().copied().collect();
    for pos in 1..=state.n_cards {
        if executed.contains(&pos) {
            continue;
        }
        let count = result.surviving_scenarios.iter()
            .filter(|s| s.is_evil(pos))
            .count();
        probs.insert(pos, count as f64 / result.n_surviving as f64);
    }
    probs
}

/// Min/max evil characters still alive across surviving scenarios.
/// Ported from strategy.py:154-167.
pub fn remaining_evil_bounds(state: &GameState, result: &SolverResult) -> (usize, usize) {
    if result.surviving_scenarios.is_empty() {
        return (0, 0);
    }
    let executed: HashSet<u8> = state.executed.iter().copied().collect();
    let counts: Vec<usize> = result.surviving_scenarios.iter().map(|s| {
        let mut count = s.evil_positions.keys()
            .filter(|p| !executed.contains(p))
            .count();
        if let Some(pp) = s.puppet_position {
            if !executed.contains(&pp) {
                count += 1;
            }
        }
        count
    }).collect();
    (*counts.iter().min().unwrap(), *counts.iter().max().unwrap())
}

/// Probability that a position is unsafe to treat as a clean execution/clue
/// surface. Drunk is intrinsically lying and killable even when an inherited
/// Alchemist resistance prevents its generic Corrupted status bit.
/// Ported from strategy.py:179-184.
pub fn corruption_risk(pos: u8, state: &GameState, result: &SolverResult) -> f64 {
    if result.n_surviving == 0 {
        return 0.0;
    }
    let count = result.surviving_scenarios.iter()
        .filter(|scenario| {
            scenario.corrupted.contains(&pos)
                || effective_role_at(pos, scenario, state)
                    .is_some_and(|role| normalize_role(&role) == "drunk")
        })
        .count();
    count as f64 / result.n_surviving as f64
}

/// Check if Witch could be alive in any surviving scenario.
/// Ported from strategy.py:170-176.
pub fn witch_might_be_alive(state: &GameState, result: &SolverResult) -> bool {
    let executed: HashSet<u8> = state.executed.iter().copied().collect();
    result.surviving_scenarios.iter().any(|s| {
        s.evil_positions.iter().any(|(pos, role)| {
            role == "Witch" && !executed.contains(pos)
        })
    })
}

/// Positions not yet revealed (no CardInfo) and not dead.
/// Ported from strategy.py:105-110.
pub fn unrevealed_positions(state: &GameState) -> Vec<u8> {
    let revealed: HashSet<u8> = state.cards.iter().map(|c| c.position).collect();
    let dead: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter())
        .copied().collect();
    (1..=state.n_cards)
        .filter(|p| !revealed.contains(p) && !dead.contains(p))
        .collect()
}

/// Tiebreaker score for positions with similar p_evil.
/// Returns (adjacency_bonus, corruption_penalty, role_consistency, witch_boost).
/// Higher is better for all terms. `adjacency_bonus` is the primary tiebreaker:
/// a position adjacent on the circle to any already-confirmed-evil position is
/// weak evidence that it too is evil (evil often clusters via Puppeteer/Poisoner
/// adjacency constraints and corruption spread patterns).
/// Ported from strategy.py:487-523, extended with adjacency (Option 2 prototype).
pub fn tiebreak_score(pos: u8, state: &GameState, result: &SolverResult) -> (f64, f64, f64, f64) {
    // 0. Adjacency to any confirmed evil (primary tiebreak).
    // Confirmed evils = state.confirmed_evil ∪ keys(executed_evil_roles).
    // Both are maintained through the live/sim loop as execution results arrive.
    let mut confirmed_evil_set: HashSet<u8> = state.confirmed_evil.iter().copied().collect();
    confirmed_evil_set.extend(state.executed_evil_roles.keys().copied());

    let adj = adjacent_positions(pos, state.n_cards);
    let adjacency_bonus = if adj.iter().any(|p| confirmed_evil_set.contains(p)) {
        1.0
    } else {
        0.0
    };

    // 1. Corruption risk (lower = safer, negate for "higher is better")
    let corr = corruption_risk(pos, state, result);
    let corruption_penalty = -corr;

    // 2. Role consistency: count distinct evil roles this position could be
    let mut evil_roles: HashSet<&str> = HashSet::new();
    for s in &result.surviving_scenarios {
        if let Some(role) = s.evil_positions.get(&pos) {
            evil_roles.insert(role.as_str());
        } else if s.puppet_position == Some(pos) {
            evil_roles.insert("Puppet");
        }
    }
    let role_consistency = if evil_roles.is_empty() {
        0.0
    } else {
        1.0 / evil_roles.len() as f64
    };

    // 3. Witch likelihood
    let witch_count = result.surviving_scenarios.iter()
        .filter(|s| s.evil_positions.get(&pos).map(|r| r.as_str()) == Some("Witch"))
        .count();
    let witch_boost = witch_count as f64 / result.n_surviving.max(1) as f64;

    (adjacency_bonus, corruption_penalty, role_consistency, witch_boost)
}

/// Get apparent role of a card at position.
pub fn get_card_role(pos: u8, state: &GameState) -> Option<&str> {
    state.cards.iter()
        .find(|c| c.position == pos)
        .map(|c| c.apparent_role.as_str())
}

/// Native execution result for one resolved target.
///
/// The revealed role is the target's real role after execution, while the
/// apparent role is the role shown before execution. Keeping both identities is
/// required to distinguish a protected Doppelganger-as-Knight from a killable
/// Drunk-as-Knight.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionConsequence {
    /// The execution was blocked; the character remains alive and HP is unchanged.
    Protected,
    /// The character dies and this much HP is removed from the player.
    Killed { hp_damage: i32 },
    /// Executing the real Bombardier/Saint is an immediate terminal loss.
    BombardierLoss,
}

/// Resolve native-static execution protection and HP damage for one target.
pub fn execution_consequence(
    revealed_role: &str,
    apparent_role: &str,
    was_evil: bool,
    was_corrupted: bool,
    default_wrong_exec_cost: i32,
) -> ExecutionConsequence {
    // Correct executions always kill and never damage the player, regardless
    // of the role used as the target's disguise.
    if was_evil {
        return ExecutionConsequence::Killed { hp_damage: 0 };
    }

    if revealed_role == "Bombardier" {
        return ExecutionConsequence::BombardierLoss;
    }

    let true_clean_knight = revealed_role == "Knight" && !was_corrupted;
    // HealthyBluff makes Doppelganger delegate killability to its Knight bluff.
    // Knight checks HealthyBluff before Corrupted, so this stays protected even
    // in a hand-built scenario that carries both statuses.
    let healthy_bluff_doppelganger_as_knight =
        revealed_role == "Doppelganger" && apparent_role == "Knight";
    if true_clean_knight || healthy_bluff_doppelganger_as_knight {
        return ExecutionConsequence::Protected;
    }

    let base_damage = if revealed_role == "Drunk" {
        2
    } else {
        default_wrong_exec_cost
    };
    // Knight's separate damage hook checks the active Corrupted status. PD and
    // execution bookkeeping still report Drunk as clean, so callers must pass
    // the role-effect status rather than that persisted observation here.
    let knight_extra = if apparent_role == "Knight" && was_corrupted {
        4
    } else {
        0
    };

    ExecutionConsequence::Killed {
        hp_damage: base_damage.wrapping_add(knight_extra),
    }
}

/// Apply `CurrentMaxValue.Reduce`'s native lower-clamped subtraction.
pub fn apply_execution_damage(current_hp: i32, hp_damage: i32) -> i32 {
    current_hp.wrapping_sub(hp_damage).max(0)
}

/// Terminal outcome relevant to execution planning after Saint/Bombardier has
/// already been handled by [`ExecutionConsequence::BombardierLoss`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTerminalOutcome {
    Continue,
    HpLoss,
    Win,
}

/// Apply the native terminal precedence: depleted HP loses before evil-count win.
pub fn execution_terminal_outcome(
    current_hp: i32,
    all_evils_gone: bool,
) -> ExecutionTerminalOutcome {
    if current_hp <= 0 {
        ExecutionTerminalOutcome::HpLoss
    } else if all_evils_gone {
        ExecutionTerminalOutcome::Win
    } else {
        ExecutionTerminalOutcome::Continue
    }
}

/// Roles that grant execution immunity when Good and uncorrupted.
pub const EXECUTION_IMMUNE_ROLES: &[&str] = &["Knight"];

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
            shaman_trace: None,
            chancellor_trace: None,
            chancellor_conversion: None,
        }
    }

    #[test]
    fn test_evil_probabilities_basic() {
        let state = GameState {
            n_cards: 3,
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
        let probs = evil_probabilities(&state, &result);
        assert_eq!(probs[&1], 0.5);
        assert_eq!(probs[&2], 0.5);
        assert_eq!(probs[&3], 0.0);
    }

    #[test]
    fn test_remaining_evil_bounds() {
        let state = GameState {
            n_cards: 5,
            executed: vec![1],
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                make_scenario(&[(1, "Pooka"), (3, "Witch")]),   // 1 is executed → 1 remaining
                make_scenario(&[(1, "Pooka"), (4, "Chancellor"), (5, "Minion")]),  // 1 executed → 2 remaining
            ],
            reasoning: vec![],
        };
        let (min, max) = remaining_evil_bounds(&state, &result);
        assert_eq!(min, 1);
        assert_eq!(max, 2);
    }

    #[test]
    fn execution_consequence_distinguishes_real_and_apparent_knights() {
        assert_eq!(
            execution_consequence("Knight", "Knight", false, false, 5),
            ExecutionConsequence::Protected,
        );
        assert_eq!(
            execution_consequence("Knight", "Knight", false, true, 5),
            ExecutionConsequence::Killed { hp_damage: 9 },
        );
        assert_eq!(
            execution_consequence("Drunk", "Knight", false, true, 5),
            ExecutionConsequence::Killed { hp_damage: 6 },
        );
        assert_eq!(
            execution_consequence("Drunk", "Knight", false, false, 5),
            ExecutionConsequence::Killed { hp_damage: 2 },
        );
        assert_eq!(
            execution_consequence("Doppelganger", "Knight", false, false, 5),
            ExecutionConsequence::Protected,
        );
        assert_eq!(
            execution_consequence("Doppelganger", "Knight", false, true, 5),
            ExecutionConsequence::Protected,
        );
        assert_eq!(
            execution_consequence("Pooka", "Knight", true, false, 5),
            ExecutionConsequence::Killed { hp_damage: 0 },
        );
    }

    #[test]
    fn hp_loss_precedes_evil_count_win() {
        assert_eq!(
            execution_terminal_outcome(0, true),
            ExecutionTerminalOutcome::HpLoss,
        );
        assert_eq!(
            execution_terminal_outcome(1, true),
            ExecutionTerminalOutcome::Win,
        );
        assert_eq!(
            execution_terminal_outcome(1, false),
            ExecutionTerminalOutcome::Continue,
        );
    }

    #[test]
    fn execution_damage_clamps_only_the_lower_bound() {
        assert_eq!(apply_execution_damage(4, 5), 0);
        assert_eq!(apply_execution_damage(0, 5), 0);
        assert_eq!(apply_execution_damage(4, -2), 6);
    }

    #[test]
    fn test_tiebreak_score() {
        // Minimal state: 5-card circle, no confirmed evils — adjacency term == 0
        // for every position, so legacy (corr, consist, witch) behavior is unchanged.
        let state = GameState {
            n_cards: 5,
            ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 2,
            n_surviving: 2,
            surviving_scenarios: vec![
                {
                    let mut s = make_scenario(&[(1, "Witch")]);
                    s.corrupted.insert(2);
                    s
                },
                make_scenario(&[(2, "Pooka")]),
            ],
            reasoning: vec![],
        };
        // Position 1: Witch in 1/2 scenarios, not corrupted, 1 evil role
        let (adj, corr, consist, witch) = tiebreak_score(1, &state, &result);
        assert_eq!(adj, 0.0); // no confirmed evil on board
        assert_eq!(corr, 0.0); // not corrupted in any
        assert_eq!(consist, 1.0); // 1 role
        assert_eq!(witch, 0.5); // Witch in 1/2

        // Position 2: corrupted in 1/2, Pooka (not Witch)
        let (adj2, corr2, consist2, witch2) = tiebreak_score(2, &state, &result);
        assert_eq!(adj2, 0.0);
        assert_eq!(corr2, -0.5); // corrupted in 1/2
        assert_eq!(consist2, 1.0); // 1 role
        assert_eq!(witch2, 0.0); // not Witch
    }

    #[test]
    fn test_tiebreak_score_adjacency_bonus() {
        // 9-card circle; #3 is a confirmed (already-executed) evil.
        // Expect: #2 and #4 get adjacency_bonus=1.0; #7 gets 0.0.
        let mut state = GameState {
            n_cards: 9,
            ..GameState::default()
        };
        state.confirmed_evil = vec![3];
        state.executed_evil_roles.insert(3, "Pooka".to_string());

        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 3,
            n_surviving: 3,
            surviving_scenarios: vec![
                make_scenario(&[(2, "Poisoner")]),
                make_scenario(&[(4, "Poisoner")]),
                make_scenario(&[(7, "Poisoner")]),
            ],
            reasoning: vec![],
        };

        let (adj2, _, _, _) = tiebreak_score(2, &state, &result);
        let (adj4, _, _, _) = tiebreak_score(4, &state, &result);
        let (adj7, _, _, _) = tiebreak_score(7, &state, &result);
        assert_eq!(adj2, 1.0); // #2 adjacent to confirmed evil #3
        assert_eq!(adj4, 1.0); // #4 adjacent to confirmed evil #3
        assert_eq!(adj7, 0.0); // #7 not adjacent to any confirmed evil
    }

    #[test]
    fn test_tiebreak_score_adjacency_wraparound() {
        // 9-card circle; #1 confirmed evil. Wrap: #9 and #2 adjacent; #5 not.
        let mut state = GameState {
            n_cards: 9,
            ..GameState::default()
        };
        state.confirmed_evil = vec![1];

        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 1,
            n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[(5, "Pooka")])],
            reasoning: vec![],
        };
        assert_eq!(tiebreak_score(9, &state, &result).0, 1.0);
        assert_eq!(tiebreak_score(2, &state, &result).0, 1.0);
        assert_eq!(tiebreak_score(5, &state, &result).0, 0.0);
    }

    #[test]
    fn resistant_generated_drunk_is_never_a_zero_risk_knight_check() {
        let state = GameState {
            n_cards: 1,
            cards: vec![CardInfo {
                position: 1,
                apparent_role: "Knight".to_string(),
                ..CardInfo::default()
            }],
            ..GameState::default()
        };
        let mut scenario = make_scenario(&[]);
        scenario.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![1],
            added_outcast_position: 1,
            added_outcast_role: "Drunk".to_string(),
        });
        assert!(scenario.corrupted.is_empty());
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: 1,
            n_surviving: 1,
            surviving_scenarios: vec![scenario],
            reasoning: vec![],
        };

        assert_eq!(corruption_risk(1, &state, &result), 1.0);
    }
}
