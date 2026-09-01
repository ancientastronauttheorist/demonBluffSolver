//! Strategy module: execution decision logic ported from strategy.py.
//!
//! This contains only the execution sub-tree of the full Python strategy.
//! Ability timing, reveal entropy, and shallow lookahead stay in Python
//! because the simulation test pre-loads all reveals and abilities.

pub mod execution;
pub mod forced;

use std::collections::{HashMap, HashSet};
use crate::types::{GameState, Scenario, SolverResult};
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
    let dead: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter()).copied().collect();
    for pos in 1..=state.n_cards {
        if dead.contains(&pos) {
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
    let dead: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter()).copied().collect();
    let counts: Vec<usize> = result.surviving_scenarios.iter().map(|s| {
        let mut count = s.evil_positions.keys()
            .filter(|p| !dead.contains(p))
            .count();
        if let Some(pp) = s.puppet_position {
            if !dead.contains(&pp) && !s.evil_positions.contains_key(&pp) {
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
    let dead: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter()).copied().collect();
    result.surviving_scenarios.iter().any(|s| {
        s.evil_positions.iter().any(|(pos, role)| {
            normalize_role(role) == "witch" && !dead.contains(pos)
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
    /// Killing a current-role public Bombardier is an immediate terminal loss.
    BombardierLoss,
}

/// Only canonical public CharacterData Bombardier has the managed Saint death
/// hook. Public CharacterData named Saint binds a different managed role and
/// must never alias this predicate.
pub fn is_terminal_loss_role(role: &str) -> bool {
    normalize_role(role) == "bombardier"
}

/// Return an already-dead qualifying position in one exact solver world.
/// Night kills are exempt. A wrong execution's public revealed current role is
/// stronger evidence than its pre-death appearance; other worlds use the
/// scenario-exact effective current role (including generated/Shaman roles).
pub fn scenario_terminal_loss_position(
    state: &GameState,
    scenario: &Scenario,
) -> Option<u8> {
    state.executed.iter().copied().find(|position| {
        if state.night_kills.contains(position) {
            return false;
        }
        public_death_role_at(state, *position)
            .map(str::to_string)
            .or_else(|| effective_role_at(*position, scenario, state))
            .as_deref()
            .is_some_and(is_terminal_loss_role)
    })
}

fn public_death_role_at(state: &GameState, position: u8) -> Option<&str> {
    state
        .slayer_results
        .iter()
        .rev()
        .find(|result| result.killed && result.target_pos == position)
        .and_then(|result| result.revealed_role.as_deref())
        .or_else(|| {
            state
                .executed_current_roles
                .get(&position)
                .map(String::as_str)
        })
        .or_else(|| {
            state
                .executed_good_roles
                .get(&position)
                .map(String::as_str)
        })
}

/// Return a qualifying death established entirely by public persisted
/// evidence, even when constraint validation currently has zero worlds.
pub fn public_terminal_loss_position(state: &GameState) -> Option<u8> {
    state.executed.iter().copied().find(|position| {
        if state.night_kills.contains(position) {
            return false;
        }
        public_death_role_at(state, *position)
            .is_some_and(is_terminal_loss_role)
    })
}

/// Fail closed when any surviving world has already reached the native
/// Bombardier terminal. Public death evidence normally makes all worlds agree;
/// a mixed result still cannot safely recommend another in-game action.
pub fn has_terminal_role_loss(state: &GameState, result: &SolverResult) -> bool {
    state
        .terminal_loss_role
        .as_deref()
        .is_some_and(is_terminal_loss_role)
        || public_terminal_loss_position(state).is_some()
        || result
            .surviving_scenarios
            .iter()
            .any(|scenario| scenario_terminal_loss_position(state, scenario).is_some())
}

/// Resolve native-static execution protection and HP damage for one target.
pub fn execution_consequence(
    revealed_role: &str,
    apparent_role: &str,
    was_evil: bool,
    was_corrupted: bool,
    default_wrong_exec_cost: i32,
) -> ExecutionConsequence {
    let revealed_role_norm = normalize_role(revealed_role);
    let apparent_role_norm = normalize_role(apparent_role);

    // Native checks the killed Character's current data before runtime
    // alignment. A Shaman-copied Bombardier therefore loses even when the
    // preserved physical Character is Evil. Bluff-only appearances, Drunk,
    // and Doppelganger reveal different current roles and remain safe.
    if is_terminal_loss_role(&revealed_role_norm) {
        return ExecutionConsequence::BombardierLoss;
    }

    // Other correct executions kill and never damage the player, regardless
    // of the role used as the target's disguise.
    if was_evil {
        return ExecutionConsequence::Killed { hp_damage: 0 };
    }

    let true_clean_knight = revealed_role_norm == "knight" && !was_corrupted;
    // A normally modeled clean Doppelganger acquired HealthyBluff while taking
    // its Knight bluff. A corrupted Doppelganger never acquired that status,
    // so its real role remains killable. Native HealthyBluff would still have
    // absolute precedence in an external both-status state Scenario cannot
    // represent.
    let healthy_bluff_doppelganger_as_knight =
        matches!(revealed_role_norm.as_str(), "doppelganger" | "doppleganger")
            && apparent_role_norm == "knight"
            && !was_corrupted;
    if true_clean_knight || healthy_bluff_doppelganger_as_knight {
        return ExecutionConsequence::Protected;
    }

    let base_damage = if revealed_role_norm == "drunk" {
        2
    } else {
        default_wrong_exec_cost
    };
    // Knight's separate damage hook checks the active Corrupted status. PD and
    // execution bookkeeping still report Drunk as clean, so callers must pass
    // the role-effect status rather than that persisted observation here.
    let knight_extra = if apparent_role_norm == "knight" && was_corrupted {
        4
    } else {
        0
    };

    ExecutionConsequence::Killed {
        hp_damage: base_damage.wrapping_add(knight_extra),
    }
}

/// Resolve one solver world's execution through the same real/apparent role
/// surface used by forced search. This keeps generated Outcasts and disguisers
/// from drifting away from native Knight protection and damage rules.
pub fn scenario_execution_consequence(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
) -> ExecutionConsequence {
    let (revealed_role, was_evil, _, active_corrupted) =
        forced::execution_reveal_outcome(pos, scenario, state);
    let apparent_role = get_card_role(pos, state).unwrap_or(&revealed_role);
    execution_consequence(
        &revealed_role,
        apparent_role,
        was_evil,
        active_corrupted,
        state.wrong_exec_cost,
    )
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ExecutionDamageProfile {
    /// Probability of HP damage or an immediate Bombardier loss.
    pub risk: f64,
    /// Mean HP damage. Infinite when any world is a terminal Bombardier loss.
    pub expected_damage: f64,
    /// Largest finite HP hit among surviving worlds.
    pub max_damage: i32,
    pub terminal_risk: bool,
}

/// Scenario-exact player risk for executing one position.
pub fn execution_damage_profile(
    pos: u8,
    state: &GameState,
    result: &SolverResult,
) -> ExecutionDamageProfile {
    let total = result.surviving_scenarios.len();
    if total == 0 {
        return ExecutionDamageProfile {
            risk: 0.0,
            expected_damage: 0.0,
            max_damage: 0,
            terminal_risk: false,
        };
    }

    let mut damaging = 0usize;
    let mut total_damage = 0i64;
    let mut max_damage = 0i32;
    let mut terminal_risk = false;
    for scenario in &result.surviving_scenarios {
        match scenario_execution_consequence(pos, scenario, state) {
            ExecutionConsequence::Protected | ExecutionConsequence::Killed { hp_damage: 0 } => {}
            ExecutionConsequence::Killed { hp_damage } => {
                damaging += 1;
                total_damage += i64::from(hp_damage);
                max_damage = max_damage.max(hp_damage);
            }
            ExecutionConsequence::BombardierLoss => {
                damaging += 1;
                terminal_risk = true;
            }
        }
    }

    ExecutionDamageProfile {
        risk: damaging as f64 / total as f64,
        expected_damage: if terminal_risk {
            f64::INFINITY
        } else {
            total_damage as f64 / total as f64
        },
        max_damage,
        terminal_risk,
    }
}

/// Apply `CurrentMaxValue.Reduce`'s native lower-clamped subtraction.
pub fn apply_execution_damage(current_hp: i32, hp_damage: i32) -> i32 {
    current_hp.wrapping_sub(hp_damage).max(0)
}

/// Terminal outcome for execution planning after applying the resolved
/// current-role death and HP state.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ExecutionTerminalOutcome {
    BombardierLoss,
    Continue,
    HpLoss,
    Win,
}

/// Apply native terminal precedence: Bombardier, depleted HP, evil-count win.
pub fn execution_terminal_outcome(
    bombardier_loss: bool,
    current_hp: i32,
    all_evils_gone: bool,
) -> ExecutionTerminalOutcome {
    if bombardier_loss {
        ExecutionTerminalOutcome::BombardierLoss
    } else if current_hp <= 0 {
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
            twin_trace: None,
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
    fn evil_probabilities_omit_night_killed_positions() {
        let state = GameState {
            n_cards: 3, night_kills: vec![2], ..GameState::default()
        };
        let result = SolverResult {
            definite_evil: vec![], definite_good: vec![], bombardier_positions: vec![],
            n_scenarios: 1, n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[(2, "Witch"), (3, "Pooka")])],
            reasoning: vec![],
        };
        let probs = evil_probabilities(&state, &result);
        assert!(!probs.contains_key(&2));
        assert_eq!(probs[&3], 1.0);
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
    fn remaining_evil_bounds_excludes_executed_and_night_killed_evils() {
        let state = GameState {
            n_cards: 4,
            executed: vec![1],
            night_kills: vec![3],
            ..GameState::default()
        };
        let mut scenario = make_scenario(&[(1, "Pooka"), (3, "Witch")]);
        scenario.puppet_position = Some(4);
        let result = SolverResult {
            definite_evil: vec![], definite_good: vec![], bombardier_positions: vec![],
            n_scenarios: 1, n_surviving: 1,
            surviving_scenarios: vec![scenario], reasoning: vec![],
        };

        assert_eq!(remaining_evil_bounds(&state, &result), (1, 1));
        let state = GameState { night_kills: vec![3, 4], ..state };
        assert_eq!(remaining_evil_bounds(&state, &result), (0, 0));
    }

    #[test]
    fn remaining_evil_bounds_counts_puppet_by_physical_position() {
        let state = GameState {
            n_cards: 4,
            ..GameState::default()
        };
        let bounds_for = |scenario| {
            let result = SolverResult {
                definite_evil: vec![],
                definite_good: vec![],
                bombardier_positions: vec![],
                n_scenarios: 1,
                n_surviving: 1,
                surviving_scenarios: vec![scenario],
                reasoning: vec![],
            };
            remaining_evil_bounds(&state, &result)
        };

        let mut ordinary = make_scenario(&[(1, "Puppeteer"), (2, "Puppet")]);
        ordinary.puppet_position = Some(2);
        assert_eq!(bounds_for(ordinary), (2, 2));

        let mut twin_overlap = make_scenario(&[(1, "Puppeteer"), (2, "Twin Minion")]);
        twin_overlap.puppet_position = Some(2);
        assert_eq!(bounds_for(twin_overlap), (2, 2));

        let mut separate = make_scenario(&[(1, "Puppeteer"), (2, "Pooka")]);
        separate.puppet_position = Some(3);
        assert_eq!(bounds_for(separate), (3, 3));
    }

    #[test]
    fn witch_liveness_tracks_each_named_witch_across_death_surfaces() {
        let result = SolverResult {
            definite_evil: vec![], definite_good: vec![], bombardier_positions: vec![],
            n_scenarios: 1, n_surviving: 1,
            surviving_scenarios: vec![make_scenario(&[(2, "Witch"), (4, "Witch")])],
            reasoning: vec![],
        };
        let one_executed = GameState {
            n_cards: 4, executed: vec![2], ..GameState::default()
        };
        assert!(witch_might_be_alive(&one_executed, &result));

        let split_deaths = GameState {
            night_kills: vec![4], ..one_executed.clone()
        };
        assert!(!witch_might_be_alive(&split_deaths, &result));

        let both_night_killed = GameState {
            executed: vec![], night_kills: vec![2, 4], ..one_executed
        };
        assert!(!witch_might_be_alive(&both_night_killed, &result));
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
            ExecutionConsequence::Killed { hp_damage: 9 },
        );
        assert_eq!(
            execution_consequence("Pooka", "Knight", true, false, 5),
            ExecutionConsequence::Killed { hp_damage: 0 },
        );
    }

    #[test]
    fn shaman_copied_knight_reveals_current_role_but_keeps_evil_execution() {
        let state = GameState {
            n_cards: 2,
            wrong_exec_cost: 5,
            cards: vec![CardInfo {
                position: 1,
                apparent_role: "Knight".to_string(),
                ..CardInfo::default()
            }],
            ..GameState::default()
        };
        let mut scenario = make_scenario(&[(1, "Pooka")]);
        scenario.shaman_trace = Some(ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Knight".to_string(),
            target_previous_roles: vec!["Pooka".to_string()],
        });

        assert_eq!(
            forced::execution_reveal_outcome(1, &scenario, &state),
            ("Knight".to_string(), true, false, false)
        );
        assert_eq!(
            scenario_execution_consequence(1, &scenario, &state),
            ExecutionConsequence::Killed { hp_damage: 0 }
        );
    }

    #[test]
    fn bombardier_terminal_uses_exact_current_role_before_runtime_alignment() {
        assert_eq!(
            execution_consequence("Bombardier", "Bombardier", false, false, 5),
            ExecutionConsequence::BombardierLoss,
        );
        assert_eq!(
            execution_consequence("Bombardier", "Minion", true, false, 5),
            ExecutionConsequence::BombardierLoss,
        );
        // Public CharacterData Saint is distinct from the managed Saint class
        // that implements public Bombardier.
        assert_eq!(
            execution_consequence("Saint", "Saint", false, false, 5),
            ExecutionConsequence::Killed { hp_damage: 5 },
        );

        let state = GameState {
            n_cards: 2,
            cards: vec![CardInfo {
                position: 1,
                apparent_role: "Bombardier".to_string(),
                ..CardInfo::default()
            }],
            ..GameState::default()
        };

        let bluff_only = make_scenario(&[(1, "Pooka")]);
        assert_eq!(
            scenario_execution_consequence(1, &bluff_only, &state),
            ExecutionConsequence::Killed { hp_damage: 0 },
        );

        let mut drunk_display = make_scenario(&[]);
        drunk_display.drunk_position = Some(1);
        assert!(matches!(
            scenario_execution_consequence(1, &drunk_display, &state),
            ExecutionConsequence::Killed { .. }
        ));

        let mut doppel_display = make_scenario(&[]);
        doppel_display.doppelganger_position = Some(1);
        assert!(matches!(
            scenario_execution_consequence(1, &doppel_display, &state),
            ExecutionConsequence::Killed { .. }
        ));

        let mut shaman_current = make_scenario(&[(1, "Pooka")]);
        shaman_current.shaman_trace = Some(ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Bombardier".to_string(),
            target_previous_roles: vec!["Pooka".to_string()],
        });
        assert_eq!(
            scenario_execution_consequence(1, &shaman_current, &state),
            ExecutionConsequence::BombardierLoss,
        );
    }

    #[test]
    fn prior_bombardier_death_uses_generated_roles_and_exempts_night() {
        let ordinary_state = GameState {
            n_cards: 2,
            cards: vec![CardInfo {
                position: 1,
                apparent_role: "Bombardier".to_string(),
                ..CardInfo::default()
            }],
            executed: vec![1],
            ..GameState::default()
        };
        let ordinary = make_scenario(&[]);
        assert_eq!(
            scenario_terminal_loss_position(&ordinary_state, &ordinary),
            Some(1),
        );

        let night_state = GameState {
            night_kills: vec![1],
            ..ordinary_state.clone()
        };
        assert_eq!(
            scenario_terminal_loss_position(&night_state, &ordinary),
            None,
        );

        let generated_state = GameState {
            n_cards: 3,
            executed: vec![2],
            ..GameState::default()
        };
        let mut generated = make_scenario(&[]);
        generated.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 2,
            added_outcast_role: "Bombardier".to_string(),
            affected_anchor_positions: vec![],
        });
        assert_eq!(
            scenario_terminal_loss_position(&generated_state, &generated),
            Some(2),
        );

        let mut shaman_current = make_scenario(&[(2, "Pooka")]);
        shaman_current.shaman_trace = Some(ShamanTrace {
            source_position: 3,
            target_position: 2,
            copied_role: "Bombardier".to_string(),
            target_previous_roles: vec!["Pooka".to_string()],
        });
        assert_eq!(
            scenario_terminal_loss_position(&generated_state, &shaman_current),
            Some(2),
        );

        let public_good = GameState {
            n_cards: 1,
            executed: vec![1],
            executed_good_roles: HashMap::from([(1, "Bombardier".to_string())]),
            ..GameState::default()
        };
        assert_eq!(public_terminal_loss_position(&public_good), Some(1));

        let public_slayer = GameState {
            n_cards: 1,
            executed: vec![1],
            slayer_results: vec![SlayerResult {
                slayer_pos: 2,
                target_pos: 1,
                killed: true,
                revealed_role: Some("Bombardier".to_string()),
                was_evil: None,
            }],
            ..GameState::default()
        };
        assert_eq!(public_terminal_loss_position(&public_slayer), Some(1));

        let public_evil_current_bomb = GameState {
            confirmed_evil: vec![1],
            executed_evil_roles: HashMap::from([(1, "Shaman".to_string())]),
            executed_current_roles: HashMap::from([(1, "Bombardier".to_string())]),
            ..public_good.clone()
        };
        assert_eq!(
            public_terminal_loss_position(&public_evil_current_bomb),
            Some(1),
        );

        let public_non_bomb_overrides_legacy_and_scenario = GameState {
            confirmed_evil: vec![1],
            executed_evil_roles: HashMap::from([(1, "Shaman".to_string())]),
            executed_current_roles: HashMap::from([(1, "Scout".to_string())]),
            ..public_good.clone()
        };
        assert_eq!(
            public_terminal_loss_position(&public_non_bomb_overrides_legacy_and_scenario),
            None,
        );
        assert_eq!(
            scenario_terminal_loss_position(
                &public_non_bomb_overrides_legacy_and_scenario,
                &ordinary,
            ),
            None,
        );

        let public_saint = GameState {
            executed_good_roles: HashMap::from([(1, "Saint".to_string())]),
            ..public_good.clone()
        };
        assert_eq!(public_terminal_loss_position(&public_saint), None);

        let slayer_saint_overrides_scenario_bomb = GameState {
            cards: vec![CardInfo {
                position: 1,
                apparent_role: "Bombardier".to_string(),
                ..CardInfo::default()
            }],
            slayer_results: vec![SlayerResult {
                slayer_pos: 2,
                target_pos: 1,
                killed: true,
                revealed_role: Some("Saint".to_string()),
                was_evil: None,
            }],
            ..public_good.clone()
        };
        assert_eq!(
            scenario_terminal_loss_position(
                &slayer_saint_overrides_scenario_bomb,
                &ordinary,
            ),
            None,
        );

        let public_night = GameState {
            night_kills: vec![1],
            ..public_good
        };
        assert_eq!(public_terminal_loss_position(&public_night), None);
    }

    #[test]
    fn knight_damage_profile_uses_real_generated_and_disguised_identities() {
        let state = GameState {
            n_cards: 2,
            wrong_exec_cost: 5,
            cards: vec![CardInfo {
                position: 1,
                apparent_role: "Knight".to_string(),
                ..CardInfo::default()
            }],
            ..GameState::default()
        };
        let clean_knight = make_scenario(&[]);
        let mut corrupted_knight = clean_knight.clone();
        corrupted_knight.corrupted.insert(1);
        let mut drunk = clean_knight.clone();
        drunk.drunk_position = Some(1);
        drunk.corrupted.insert(1);
        let mut doppel = clean_knight.clone();
        doppel.doppelganger_position = Some(1);
        let mut corrupted_doppel = doppel.clone();
        corrupted_doppel.corrupted.insert(1);
        let mut generated_wretch = clean_knight.clone();
        generated_wretch.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![2],
            added_outcast_position: 1,
            added_outcast_role: "Wretch".to_string(),
            affected_anchor_positions: vec![],
        });
        let evil = make_scenario(&[(1, "Pooka")]);

        assert_eq!(
            scenario_execution_consequence(1, &clean_knight, &state),
            ExecutionConsequence::Protected,
        );
        assert_eq!(
            scenario_execution_consequence(1, &corrupted_knight, &state),
            ExecutionConsequence::Killed { hp_damage: 9 },
        );
        assert_eq!(
            scenario_execution_consequence(1, &drunk, &state),
            ExecutionConsequence::Killed { hp_damage: 6 },
        );
        assert_eq!(
            scenario_execution_consequence(1, &doppel, &state),
            ExecutionConsequence::Protected,
        );
        assert_eq!(
            scenario_execution_consequence(1, &corrupted_doppel, &state),
            ExecutionConsequence::Killed { hp_damage: 9 },
        );
        assert_eq!(
            scenario_execution_consequence(1, &generated_wretch, &state),
            ExecutionConsequence::Killed { hp_damage: 5 },
        );
        assert_eq!(
            scenario_execution_consequence(1, &evil, &state),
            ExecutionConsequence::Killed { hp_damage: 0 },
        );

        let scenarios = vec![
            clean_knight,
            corrupted_knight,
            drunk,
            doppel,
            corrupted_doppel,
            generated_wretch,
            evil,
        ];
        let result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![],
            n_scenarios: scenarios.len(),
            n_surviving: scenarios.len(),
            surviving_scenarios: scenarios,
            reasoning: vec![],
        };
        let profile = execution_damage_profile(1, &state, &result);
        assert_eq!(profile.risk, 4.0 / 7.0);
        assert_eq!(profile.expected_damage, 29.0 / 7.0);
        assert_eq!(profile.max_damage, 9);
        assert!(!profile.terminal_risk);

        let mut generated_bombardier = make_scenario(&[]);
        generated_bombardier.chancellor_trace = Some(ChancellorTrace {
            original_positions: vec![2],
            added_outcast_position: 1,
            added_outcast_role: "Bombardier".to_string(),
            affected_anchor_positions: vec![],
        });
        let bomb_result = SolverResult {
            definite_evil: vec![],
            definite_good: vec![],
            bombardier_positions: vec![1],
            n_scenarios: 1,
            n_surviving: 1,
            surviving_scenarios: vec![generated_bombardier],
            reasoning: vec![],
        };
        let bomb_profile = execution_damage_profile(1, &state, &bomb_result);
        assert_eq!(bomb_profile.risk, 1.0);
        assert!(bomb_profile.expected_damage.is_infinite());
        assert!(bomb_profile.terminal_risk);
    }

    #[test]
    fn hp_loss_precedes_evil_count_win() {
        assert_eq!(
            execution_terminal_outcome(false, 0, true),
            ExecutionTerminalOutcome::HpLoss,
        );
        assert_eq!(
            execution_terminal_outcome(false, 1, true),
            ExecutionTerminalOutcome::Win,
        );
        assert_eq!(
            execution_terminal_outcome(false, 1, false),
            ExecutionTerminalOutcome::Continue,
        );
    }

    #[test]
    fn bombardier_loss_precedes_hp_loss_and_evil_count_win() {
        assert_eq!(
            execution_terminal_outcome(true, 0, true),
            ExecutionTerminalOutcome::BombardierLoss,
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
            affected_anchor_positions: vec![],
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
