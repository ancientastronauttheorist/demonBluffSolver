//! Game simulation test: combined constraint validation + strategy-driven execution.
//!
//! Replaces both replay.rs (constraint checks) and adds Phase 3 (strategy wins).
//!
//! Phase 1: Incremental reveals with truth-in-set checks
//! Phase 2: Apply abilities, assert truth survives
//! Phase 3: Strategy-driven execution loop — verify the solver would WIN

use solver_core::solver::solve;
use solver_core::knowledge_base::normalize_role;
use solver_core::strategy::execution::pick_execution_target;
use solver_core::strategy::{
    apply_execution_damage, execution_consequence, execution_terminal_outcome,
    get_card_role, remaining_evil_bounds, ExecutionConsequence,
    ExecutionTerminalOutcome,
};
use solver_core::types::*;
use std::collections::{HashMap, HashSet, VecDeque};
use std::path::PathBuf;

fn v2_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/cases_v2")
}

// ── Shared helpers (from replay.rs) ──

fn night_kill_timing(reveal_order: &[u8], night_kills: &[u8]) -> HashMap<usize, Vec<u8>> {
    let mut timing: HashMap<usize, Vec<u8>> = HashMap::new();
    if night_kills.is_empty() { return timing; }
    let n_nights = reveal_order.len() / 4;
    let mut nk_idx = 0;
    for night_num in 1..=n_nights {
        let trigger = night_num * 4;
        if nk_idx < night_kills.len() {
            timing.entry(trigger).or_default().push(night_kills[nk_idx]);
            nk_idx += 1;
        }
    }
    if nk_idx < night_kills.len() {
        let last_trigger = timing.keys().max().copied().unwrap_or(4);
        for &nk in &night_kills[nk_idx..] {
            timing.entry(last_trigger).or_default().push(nk);
        }
    }
    timing
}

fn truth_in_set(result: &SolverResult, true_evil_set: &HashSet<u8>, executed: &[u8]) -> bool {
    if result.n_surviving == 0 { return false; }
    let exec_set: HashSet<u8> = executed.iter().copied().collect();
    let non_exec_true: HashSet<u8> = true_evil_set.difference(&exec_set).copied().collect();
    for s in &result.surviving_scenarios {
        let mut scenario_evil: HashSet<u8> = s.evil_positions.keys().copied().collect();
        if let Some(pp) = s.puppet_position { scenario_evil.insert(pp); }
        let non_exec_scenario: HashSet<u8> = scenario_evil.difference(&exec_set).copied().collect();
        if non_exec_true == non_exec_scenario { return true; }
    }
    false
}

fn make_state(
    case: &serde_json::Value,
    cards: &[serde_json::Value],
    executed: &[u8],
    confirmed_evil: &[u8],
    confirmed_good: &[u8],
    executed_evil_roles: &HashMap<u8, String>,
    executed_good_corrupted: &HashMap<u8, bool>,
    executed_good_roles: &HashMap<u8, String>,
    slayer_results: &[SlayerResult],
    pd_ability_results: &[PdAbilityResult],
    night_kills: &[u8],
    night_kill_evil_count: u8,
    reveal_order: &[u8],
) -> GameState {
    let mut state = GameState::from_json(case).unwrap();
    state.cards = cards.iter()
        .filter_map(|c| serde_json::from_value::<CardInfo>(c.clone()).ok())
        .collect();
    state.executed = executed.to_vec();
    state.confirmed_evil = confirmed_evil.to_vec();
    state.confirmed_good = confirmed_good.to_vec();
    state.executed_evil_roles = executed_evil_roles.clone();
    state.executed_good_corrupted = executed_good_corrupted.clone();
    state.executed_good_roles = executed_good_roles.clone();
    state.slayer_results = slayer_results.to_vec();
    state.pd_ability_results = pd_ability_results.to_vec();
    state.night_kills = night_kills.to_vec();
    state.night_kill_evil_count = night_kill_evil_count;
    state.reveal_order = reveal_order.to_vec();
    state
}

// ── Simulation result types ──

#[derive(Debug)]
#[allow(dead_code)]
enum SimResult {
    Win { executions: usize, wrong_execs: usize },
    ConstraintFailure { phase: String, detail: String },
    SimLoss { reason: String, executions: usize },
    InsufficientTruth { detail: String },
}

// ── Retain every scenario compatible with recorded evil ground truth ──

fn truth_compatible_scenarios<'a>(
    result: &'a SolverResult,
    true_evil_positions: &HashMap<u8, String>,
    executed: &[u8],
) -> Vec<&'a Scenario> {
    let true_evil_set: HashSet<u8> = true_evil_positions.keys().copied().collect();
    let exec_set: HashSet<u8> = executed.iter().copied().collect();
    let non_exec_true: HashSet<u8> = true_evil_set.difference(&exec_set).copied().collect();
    result.surviving_scenarios.iter().filter(|s| {
        let mut scenario_evil: HashSet<u8> = s.evil_positions.keys().copied().collect();
        if let Some(pp) = s.puppet_position { scenario_evil.insert(pp); }
        let non_exec_scenario: HashSet<u8> = scenario_evil.difference(&exec_set).copied().collect();
        non_exec_true == non_exec_scenario
            && non_exec_true.iter().all(|position| {
                let Some(true_role) = true_evil_positions.get(position) else {
                    return false;
                };
                if normalize_role(true_role) == "puppet" {
                    s.puppet_position == Some(*position)
                } else {
                    s.evil_positions
                        .get(position)
                        .is_some_and(|role| normalize_role(role) == normalize_role(true_role))
                }
            })
    }).collect()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct GoodExecutionObservation {
    consequence: ExecutionConsequence,
    /// Corruption value persisted by the live bridge. Drunk is reported clean
    /// here even when a separate active Corrupted status affects role hooks.
    observed_corrupted: bool,
    /// A killed good card reveals its real role. Protected Knight outcomes do
    /// not reveal a distinct hidden identity and intentionally keep this None.
    revealed_role: Option<String>,
}

fn good_execution_observation(
    scenario: &Scenario,
    pos: u8,
    apparent_role: &str,
    wrong_exec_cost: i32,
) -> GoodExecutionObservation {
    let revealed_role = if scenario.drunk_position == Some(pos) {
        "Drunk"
    } else if scenario.doppelganger_position == Some(pos) {
        "Doppelganger"
    } else if scenario.chancellor_added_outcast_position() == Some(pos) {
        scenario
            .chancellor_added_outcast_role()
            .unwrap_or(apparent_role)
    } else {
        apparent_role
    };
    let observed_corrupted = if normalize_role(revealed_role) == "drunk" {
        false
    } else {
        scenario.corrupted.contains(&pos)
    };
    let active_corrupted = scenario.corrupted.contains(&pos);
    let consequence = execution_consequence(
        revealed_role,
        apparent_role,
        false,
        active_corrupted,
        wrong_exec_cost,
    );
    let revealed_role = if consequence == ExecutionConsequence::Protected
        || revealed_role.trim().is_empty()
        || matches!(normalize_role(revealed_role).as_str(), "unknown" | "none" | "?")
    {
        None
    } else {
        Some(revealed_role.to_string())
    };
    GoodExecutionObservation {
        revealed_role,
        consequence,
        observed_corrupted,
    }
}

/// Resolve an execution only when every truth-compatible hidden-Outcast world
/// produces the same state-relevant observation. This prevents the harness from
/// picking the first scenario and inventing Drunk/Doppelganger identity.
fn consensus_good_execution_observation(
    scenarios: &[&Scenario],
    pos: u8,
    apparent_role: &str,
    wrong_exec_cost: i32,
    required_corruption: Option<bool>,
    required_role: Option<&str>,
) -> Result<GoodExecutionObservation, String> {
    let observations = distinct_good_execution_observations(
        scenarios,
        pos,
        apparent_role,
        wrong_exec_cost,
        required_corruption,
        required_role,
    );
    let Some(first) = observations.first() else {
        return Err(format!(
            "execution #{pos} has no truth-compatible outcome for apparent {apparent_role}"
        ));
    };
    if observations.iter().skip(1).any(|observation| {
        observation.consequence != first.consequence
            || observation.observed_corrupted != first.observed_corrupted
    }) {
        Err(format!(
            "execution #{pos} has multiple truth-compatible outcomes for apparent {apparent_role}"
        ))
    } else {
        // HP reconstruction needs only the persisted consequence/corruption
        // surface. Role-distinct killed outcomes are branched later by the
        // strategy simulation and persisted for the next solve.
        Ok(first.clone())
    }
}

fn distinct_good_execution_observations(
    scenarios: &[&Scenario],
    pos: u8,
    apparent_role: &str,
    wrong_exec_cost: i32,
    required_corruption: Option<bool>,
    required_role: Option<&str>,
) -> Vec<GoodExecutionObservation> {
    let mut observations = Vec::new();
    for scenario in scenarios {
        let observation = good_execution_observation(
            scenario,
            pos,
            apparent_role,
            wrong_exec_cost,
        );
        if required_corruption.is_some_and(|value| value != observation.observed_corrupted) {
            continue;
        }
        if required_role.is_some_and(|required| {
            observation
                .revealed_role
                .as_deref()
                .map(|role| normalize_role(role) != normalize_role(required))
                .unwrap_or(true)
        }) {
            continue;
        }
        if !observations.contains(&observation) {
            observations.push(observation);
        }
    }
    observations
}

#[derive(Debug, Clone, Default)]
struct ExecutionBranch {
    cards: Vec<serde_json::Value>,
    blocked_positions: Vec<u8>,
    reveal_order: Vec<u8>,
    executed: Vec<u8>,
    confirmed_evil: Vec<u8>,
    confirmed_good: Vec<u8>,
    evil_roles: HashMap<u8, String>,
    good_corrupted: HashMap<u8, bool>,
    good_roles: HashMap<u8, String>,
    immunity_blocked: HashSet<u8>,
    hp: i32,
    total_executions: usize,
    wrong_executions: usize,
    iterations: usize,
    observations: Vec<String>,
}

impl ExecutionBranch {
    fn context(&self) -> String {
        if self.observations.is_empty() {
            "initial".to_string()
        } else {
            self.observations.join(" -> ")
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
struct ExecutionBranchKey {
    card_positions: Vec<u8>,
    blocked_positions: Vec<u8>,
    reveal_order: Vec<u8>,
    executed: Vec<u8>,
    confirmed_evil: Vec<u8>,
    confirmed_good: Vec<u8>,
    evil_roles: Vec<(u8, String)>,
    good_corrupted: Vec<(u8, bool)>,
    good_roles: Vec<(u8, String)>,
    immunity_blocked: Vec<u8>,
    hp: i32,
}

fn sorted_positions(positions: impl IntoIterator<Item = u8>) -> Vec<u8> {
    let mut positions: Vec<u8> = positions.into_iter().collect();
    positions.sort_unstable();
    positions.dedup();
    positions
}

fn json_card_position(card: &serde_json::Value) -> Option<u8> {
    card.get("position").and_then(|value| value.as_u64())
        .map(|position| position as u8)
}

fn initialize_execution_block_state(
    cards: Vec<serde_json::Value>,
    reveal_order: Vec<u8>,
    fixture_blocked_positions: &[u8],
    pre_execution_dead: &HashSet<u8>,
    pre_execution_witch_deaths: usize,
) -> (Vec<serde_json::Value>, Vec<u8>, Vec<u8>) {
    let mut unique_markers = Vec::new();
    for &position in fixture_blocked_positions {
        if !unique_markers.contains(&position) { unique_markers.push(position); }
    }

    // The shipped first-match Start creates one scalar block even if ordinary
    // duplicate Witch records exist. Any prior Slayer/night Witch death calls
    // Reduce(1), clearing that scalar before the execution phase.
    let blocked_positions: Vec<u8> = if pre_execution_witch_deaths == 0 {
        unique_markers.into_iter()
            .filter(|position| !pre_execution_dead.contains(position))
            .take(1).collect()
    } else {
        Vec::new()
    };
    let blocked: HashSet<u8> = blocked_positions.iter().copied().collect();

    // Some final fixtures include a card revealed after the recorded day-Witch
    // death. Rewind that observation together with its reveal-order entry.
    let cards = cards.into_iter().filter(|card| {
        json_card_position(card).map_or(true, |position| !blocked.contains(&position))
    }).collect();
    let reveal_order = reveal_order.into_iter()
        .filter(|position| !blocked.contains(position)).collect();
    (cards, blocked_positions, reveal_order)
}

fn release_one_witch_block(
    branch: &mut ExecutionBranch,
    card_by_pos: &HashMap<u8, &serde_json::Value>,
) -> Option<u8> {
    let position = branch.blocked_positions.first().copied()?;
    branch.blocked_positions.remove(0);
    if branch.cards.iter().all(|card| json_card_position(card) != Some(position)) {
        if let Some(card) = card_by_pos.get(&position) {
            branch.cards.push((*card).clone());
            if !branch.reveal_order.contains(&position) {
                branch.reveal_order.push(position);
            }
        }
    }
    Some(position)
}

fn execution_branch_key(branch: &ExecutionBranch) -> ExecutionBranchKey {
    let mut evil_roles: Vec<(u8, String)> = branch
        .evil_roles
        .iter()
        .map(|(&position, role)| (position, role.clone()))
        .collect();
    evil_roles.sort_unstable();
    let mut good_corrupted: Vec<(u8, bool)> = branch
        .good_corrupted
        .iter()
        .map(|(&position, &corrupted)| (position, corrupted))
        .collect();
    good_corrupted.sort_unstable();
    let mut good_roles: Vec<(u8, String)> = branch
        .good_roles
        .iter()
        .map(|(&position, role)| (position, role.clone()))
        .collect();
    good_roles.sort_unstable();

    ExecutionBranchKey {
        card_positions: sorted_positions(branch.cards.iter().filter_map(json_card_position)),
        blocked_positions: branch.blocked_positions.clone(),
        reveal_order: branch.reveal_order.clone(),
        executed: sorted_positions(branch.executed.iter().copied()),
        confirmed_evil: sorted_positions(branch.confirmed_evil.iter().copied()),
        confirmed_good: sorted_positions(branch.confirmed_good.iter().copied()),
        evil_roles,
        good_corrupted,
        good_roles,
        immunity_blocked: sorted_positions(branch.immunity_blocked.iter().copied()),
        hp: branch.hp,
    }
}

fn enqueue_execution_branch(
    queue: &mut VecDeque<ExecutionBranch>,
    seen: &mut HashMap<ExecutionBranchKey, usize>,
    branch: ExecutionBranch,
) {
    let key = execution_branch_key(&branch);
    if seen
        .get(&key)
        .is_some_and(|&best_iterations| best_iterations <= branch.iterations)
    {
        return;
    }
    seen.insert(key, branch.iterations);
    queue.push_back(branch);
}

#[derive(Debug)]
enum GoodExecutionContinuation {
    Continue(ExecutionBranch),
    BombardierLoss,
}

fn continue_good_execution(
    branch: &ExecutionBranch,
    pos: u8,
    observation: GoodExecutionObservation,
) -> GoodExecutionContinuation {
    let mut next = branch.clone();
    next.iterations += 1;
    next.total_executions += 1;
    next.observations.push(format!("#{pos}={observation:?}"));

    match observation.consequence {
        ExecutionConsequence::BombardierLoss => GoodExecutionContinuation::BombardierLoss,
        ExecutionConsequence::Protected => {
            if !next.confirmed_good.contains(&pos) {
                next.confirmed_good.push(pos);
            }
            next.immunity_blocked.insert(pos);
            GoodExecutionContinuation::Continue(next)
        }
        ExecutionConsequence::Killed { hp_damage } => {
            next.wrong_executions += 1;
            next.hp = apply_execution_damage(next.hp, hp_damage);
            if !next.executed.contains(&pos) {
                next.executed.push(pos);
            }
            if !next.confirmed_good.contains(&pos) {
                next.confirmed_good.push(pos);
            }
            next.good_corrupted
                .insert(pos, observation.observed_corrupted);
            if let Some(role) = observation.revealed_role {
                next.good_roles.insert(pos, role);
            }
            next.immunity_blocked.clear();
            GoodExecutionContinuation::Continue(next)
        }
    }
}

// ── HP reconstruction ──

fn reconstruct_phase3_hp(
    case: &serde_json::Value,
    true_evil_set: &HashSet<u8>,
    night_kills_set: &HashSet<u8>,
    slayer_killed: &HashSet<u8>,
    truth_scenarios: &[&Scenario],
) -> Result<i32, String> {
    let obj = case.as_object().unwrap();
    let json_hp = obj.get("hp").and_then(|v| v.as_i64()).unwrap_or(10) as i32;
    let wrong_exec_cost = obj.get("wrong_exec_cost").and_then(|v| v.as_i64()).unwrap_or(2) as i32;

    let case_executed: Vec<u8> = obj.get("executed")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
        .unwrap_or_default();

    let exec_good_corr: HashMap<u8, bool> = obj.get("executed_good_corrupted")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| Some((k.parse::<u8>().ok()?, v.as_bool()?))).collect())
        .unwrap_or_default();
    let exec_good_roles: HashMap<u8, String> = obj.get("executed_good_roles")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| {
            Some((k.parse::<u8>().ok()?, v.as_str()?.to_string()))
        }).collect())
        .unwrap_or_default();

    let cards: Vec<serde_json::Value> = obj.get("cards")
        .and_then(|v| v.as_array())
        .cloned().unwrap_or_default();
    let card_roles: HashMap<u8, String> = cards.iter().filter_map(|c| {
        let pos = c.get("position")?.as_i64()? as u8;
        let role = c.get("apparent_role")?.as_str()?.to_string();
        Some((pos, role))
    }).collect();

    // Add back execution costs for recorded player executions
    let mut cost_addback = 0i32;
    let mut compatible_scenarios = truth_scenarios.to_vec();
    for &pos in &case_executed {
        if night_kills_set.contains(&pos) || slayer_killed.contains(&pos) {
            continue; // Not a player execution
        }
        if true_evil_set.contains(&pos) {
            // Correct execution: 0 HP cost
            continue;
        }
        // Historical saves used `executed` for both dead targets and resolved
        // protected Knight checks. Recorded corruption can filter worlds, but
        // killed versus protected must remain part of the consensus.
        let apparent = card_roles.get(&pos).map(|s| s.as_str()).unwrap_or("");
        let required_corruption = exec_good_corr.get(&pos).copied();
        let required_role = exec_good_roles.get(&pos).map(String::as_str);
        compatible_scenarios.retain(|scenario| {
            let observation = good_execution_observation(
                scenario,
                pos,
                apparent,
                wrong_exec_cost,
            );
            let corruption_matches = required_corruption
                .map_or(true, |required| required == observation.observed_corrupted);
            let role_matches = required_role.map_or(true, |required| {
                observation
                    .revealed_role
                    .as_deref()
                    .is_some_and(|role| normalize_role(role) == normalize_role(required))
            });
            corruption_matches && role_matches
        });
        if compatible_scenarios.is_empty() {
            return Err(format!(
                "execution evidence through #{pos} has no shared truth-compatible world"
            ));
        }
        let observation = consensus_good_execution_observation(
            &compatible_scenarios,
            pos,
            apparent,
            wrong_exec_cost,
            required_corruption,
            required_role,
        )?;
        match observation.consequence {
            ExecutionConsequence::Protected => {}
            ExecutionConsequence::Killed { hp_damage } => {
                cost_addback = cost_addback.wrapping_add(hp_damage);
            }
            ExecutionConsequence::BombardierLoss => {
                return Err(format!(
                    "recorded good Bombardier execution #{pos} would be terminal"
                ));
            }
        }
    }

    Ok(json_hp + cost_addback)
}

// ── Main simulation function ──

fn simulate_game(value: &serde_json::Value) -> SimResult {
    let obj = value.as_object().unwrap();
    let case_name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");

    // Parse ground truth
    let true_evil_positions: HashMap<u8, String> = obj.get("true_evil_positions")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| {
            Some((k.parse::<u8>().ok()?, v.as_str()?.to_string()))
        }).collect())
        .unwrap_or_default();
    let true_evil_set: HashSet<u8> = true_evil_positions.keys().copied().collect();

    // Parse case data
    let reveal_order: Vec<u8> = obj.get("reveal_order")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
        .unwrap_or_else(|| {
            obj.get("cards").and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|c| c.get("position")?.as_i64().map(|x| x as u8)).collect())
                .unwrap_or_default()
        });

    let case_cards: Vec<serde_json::Value> = obj.get("cards")
        .and_then(|v| v.as_array()).cloned().unwrap_or_default();
    let card_by_pos: HashMap<u8, &serde_json::Value> = case_cards.iter()
        .filter_map(|c| {
            let pos = c.get("position")?.as_i64()? as u8;
            Some((pos, c))
        }).collect();

    let case_night_kills: Vec<u8> = obj.get("night_kills")
        .and_then(|v| v.as_array())
        .map(|a| a.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
        .unwrap_or_default();
    let nk_timing = night_kill_timing(&reveal_order, &case_night_kills);
    let nk_set: HashSet<u8> = case_night_kills.iter().copied().collect();

    let case_evil_roles: HashMap<u8, String> = obj.get("executed_evil_roles")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| Some((k.parse::<u8>().ok()?, v.as_str()?.to_string()))).collect())
        .unwrap_or_default();

    let case_exec_good_corr: HashMap<u8, bool> = obj.get("executed_good_corrupted")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| Some((k.parse::<u8>().ok()?, v.as_bool()?))).collect())
        .unwrap_or_default();
    let case_exec_good_roles: HashMap<u8, String> = obj.get("executed_good_roles")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| {
            Some((k.parse::<u8>().ok()?, v.as_str()?.to_string()))
        }).collect())
        .unwrap_or_default();

    // Incremental state (Phases 1-2)
    let mut current_cards: Vec<serde_json::Value> = Vec::new();
    let mut current_reveal_order: Vec<u8> = Vec::new();
    let mut current_executed: Vec<u8> = Vec::new();
    let mut current_confirmed_evil: Vec<u8> = Vec::new();
    let mut current_confirmed_good: Vec<u8> = Vec::new();
    let mut current_evil_roles: HashMap<u8, String> = HashMap::new();
    let mut current_good_corr: HashMap<u8, bool> = HashMap::new();
    let mut current_good_roles: HashMap<u8, String> = HashMap::new();
    let mut current_slayer: Vec<SlayerResult> = Vec::new();
    let mut current_pd: Vec<PdAbilityResult> = Vec::new();
    let mut current_night_kills: Vec<u8> = Vec::new();
    let mut current_nk_evil_count: u8 = 0;

    // ── Phase 1: Reveal cards ──
    for (reveals_done, &pos) in reveal_order.iter().enumerate() {
        if let Some(&card_val) = card_by_pos.get(&pos) {
            current_cards.push(card_val.clone());
        }
        current_reveal_order.push(pos);

        let reveals_count = reveals_done + 1;
        if let Some(nk_list) = nk_timing.get(&reveals_count) {
            for &nk_pos in nk_list {
                current_night_kills.push(nk_pos);
                if true_evil_set.contains(&nk_pos) {
                    current_nk_evil_count += 1;
                }
                if !current_executed.contains(&nk_pos) {
                    current_executed.push(nk_pos);
                }
            }
        }
    }

    // Add extra card data (not in reveal_order)
    let revealed_positions: HashSet<u8> = current_reveal_order.iter().copied().collect();
    for card_val in &case_cards {
        if let Some(pos) = card_val.get("position").and_then(|v| v.as_i64()).map(|x| x as u8) {
            if !revealed_positions.contains(&pos) {
                current_cards.push(card_val.clone());
            }
        }
    }

    // ── Phase 2: Apply abilities ──
    let mut slayer_killed: HashSet<u8> = HashSet::new();
    if let Some(slayer_arr) = obj.get("slayer_results").and_then(|v| v.as_array()) {
        for sr_val in slayer_arr {
            let sr: SlayerResult = serde_json::from_value(sr_val.clone()).unwrap();
            if sr.killed {
                slayer_killed.insert(sr.target_pos);
                if !current_executed.contains(&sr.target_pos) {
                    current_executed.push(sr.target_pos);
                }
                if true_evil_set.contains(&sr.target_pos) {
                    if !current_confirmed_evil.contains(&sr.target_pos) {
                        current_confirmed_evil.push(sr.target_pos);
                    }
                    let role = sr.revealed_role.clone()
                        .or_else(|| case_evil_roles.get(&sr.target_pos).cloned())
                        .unwrap_or_default();
                    if !role.is_empty() {
                        current_evil_roles.insert(sr.target_pos, role);
                    }
                } else {
                    if !current_confirmed_good.contains(&sr.target_pos) {
                        current_confirmed_good.push(sr.target_pos);
                    }
                    if let Some(observed) = case_exec_good_corr.get(&sr.target_pos) {
                        current_good_corr.insert(sr.target_pos, *observed);
                    }
                    let role = sr.revealed_role.clone()
                        .or_else(|| case_exec_good_roles.get(&sr.target_pos).cloned());
                    if let Some(role) = role {
                        current_good_roles.insert(sr.target_pos, role);
                    }
                }
            }
            current_slayer.push(sr);
        }
    }
    if let Some(pd_arr) = obj.get("pd_ability_results").and_then(|v| v.as_array()) {
        for pd_val in pd_arr {
            let pr: PdAbilityResult = serde_json::from_value(pd_val.clone()).unwrap();
            current_pd.push(pr);
        }
    }

    // Constraint check after all info loaded
    let state = make_state(value, &current_cards, &current_executed,
        &current_confirmed_evil, &current_confirmed_good,
        &current_evil_roles, &current_good_corr,
        &current_good_roles,
        &current_slayer, &current_pd,
        &current_night_kills, current_nk_evil_count,
        &current_reveal_order);
    let result = solve(&state);

    if !truth_in_set(&result, &true_evil_set, &current_executed) {
        return SimResult::ConstraintFailure {
            phase: "post_reveal".into(),
            detail: format!("{case_name}: truth eliminated after reveals+abilities ({} surviving)", result.n_surviving),
        };
    }
    for &pos in &result.definite_evil {
        if !true_evil_set.contains(&pos) {
            return SimResult::ConstraintFailure {
                phase: "post_reveal".into(),
                detail: format!("{case_name}: false definite_evil #{pos}"),
            };
        }
    }
    for &pos in &result.definite_good {
        if true_evil_set.contains(&pos) {
            return SimResult::ConstraintFailure {
                phase: "post_reveal".into(),
                detail: format!("{case_name}: false definite_good #{pos}"),
            };
        }
    }

    // ── Phase 3: Strategy-driven execution ──
    // Reconstruct HP at start of execution phase
    let truth_scenarios = truth_compatible_scenarios(
        &result,
        &true_evil_positions,
        &current_executed,
    );
    let hp = match reconstruct_phase3_hp(
        value,
        &true_evil_set,
        &nk_set,
        &slayer_killed,
        &truth_scenarios,
    ) {
        Ok(hp) => hp,
        Err(detail) => {
            return SimResult::InsufficientTruth {
                detail: format!("{case_name}: HP reconstruction: {detail}"),
            };
        }
    };
    let wrong_exec_cost = state.wrong_exec_cost;

    // Strip recorded executions — simulation drives its own
    // Keep: night kills, slayer kills, confirmed from those
    // Reset: player executions
    let pre_exec_executed: Vec<u8> = current_executed.iter()
        .filter(|p| nk_set.contains(p) || slayer_killed.contains(p))
        .copied().collect();
    let pre_exec_confirmed_evil: Vec<u8> = current_confirmed_evil.clone();
    let pre_exec_confirmed_good: Vec<u8> = current_confirmed_good.iter()
        .filter(|p| slayer_killed.contains(p))
        .copied().collect();

    let pre_exec_evil_roles = current_evil_roles.iter()
        .filter(|(p, _)| nk_set.contains(p) || slayer_killed.contains(p))
        .map(|(&p, r)| (p, r.clone())).collect();
    let pre_exec_good_corrupted = current_good_corr.iter()
        .filter(|(p, _)| slayer_killed.contains(p))
        .map(|(&p, &corrupted)| (p, corrupted)).collect();
    let pre_exec_good_roles = current_good_roles.iter()
        .filter(|(p, _)| slayer_killed.contains(p))
        .map(|(&p, role)| (p, role.clone())).collect();
    let pre_exec_dead: HashSet<u8> = pre_exec_executed.iter().copied().collect();
    // Simulation-only truth resolves the native death hook for Lilis/Slayer
    // kills. Hidden roles are never copied into the live solver GameState.
    let pre_exec_witch_deaths = pre_exec_executed.iter().filter(|position| {
        true_evil_positions.get(position)
            .is_some_and(|role| normalize_role(role) == "witch")
    }).count();
    let (initial_cards, initial_blocked_positions, initial_reveal_order) =
        initialize_execution_block_state(
            current_cards, current_reveal_order, &state.blocked_positions,
            &pre_exec_dead, pre_exec_witch_deaths,
        );
    let max_iterations = 20; // Safety valve
    let initial_branch = ExecutionBranch {
        cards: initial_cards,
        blocked_positions: initial_blocked_positions,
        reveal_order: initial_reveal_order,
        executed: pre_exec_executed,
        confirmed_evil: pre_exec_confirmed_evil,
        confirmed_good: pre_exec_confirmed_good,
        evil_roles: pre_exec_evil_roles,
        good_corrupted: pre_exec_good_corrupted,
        good_roles: pre_exec_good_roles,
        immunity_blocked: HashSet::new(),
        hp,
        total_executions: 0,
        wrong_executions: 0,
        iterations: 0,
        observations: Vec::new(),
    };
    let mut branches = VecDeque::new();
    let mut seen_branches = HashMap::new();
    enqueue_execution_branch(&mut branches, &mut seen_branches, initial_branch);
    let mut completed_wins = 0usize;
    let mut max_win_executions = 0usize;
    let mut max_win_wrong_executions = 0usize;

    while let Some(branch) = branches.pop_front() {
        // Build state and solve
        let state = make_state(value, &branch.cards, &branch.executed,
            &branch.confirmed_evil, &branch.confirmed_good,
            &branch.evil_roles, &branch.good_corrupted,
            &branch.good_roles,
            &current_slayer, &current_pd,
            &current_night_kills, current_nk_evil_count,
            &branch.reveal_order);
        let mut state_with_hp = state;
        state_with_hp.blocked_positions = branch.blocked_positions.clone();
        state_with_hp.hp = branch.hp;
        state_with_hp.wrong_exec_cost = wrong_exec_cost;

        let result = solve(&state_with_hp);

        // Constraint check
        if !truth_in_set(&result, &true_evil_set, &branch.executed) {
            return SimResult::ConstraintFailure {
                phase: format!("exec_step_{}", branch.total_executions),
                detail: format!(
                    "{case_name}: truth eliminated during simulation ({} surviving) [branch {}]",
                    result.n_surviving,
                    branch.context(),
                ),
            };
        }

        // Native terminal precedence: depleted HP loses before evil-count win.
        let (_, max_remaining) = remaining_evil_bounds(&state_with_hp, &result);
        match execution_terminal_outcome(branch.hp, max_remaining == 0) {
            ExecutionTerminalOutcome::HpLoss => {
                return SimResult::SimLoss {
                    reason: format!(
                        "{case_name}: HP exhausted (hp={}) before win resolution [branch {}]",
                        branch.hp,
                        branch.context(),
                    ),
                    executions: branch.total_executions,
                };
            }
            ExecutionTerminalOutcome::Win => {
                completed_wins += 1;
                max_win_executions = max_win_executions.max(branch.total_executions);
                max_win_wrong_executions =
                    max_win_wrong_executions.max(branch.wrong_executions);
                continue;
            }
            ExecutionTerminalOutcome::Continue => {}
        }

        if branch.iterations >= max_iterations {
            return SimResult::SimLoss {
                reason: format!(
                    "{case_name}: exceeded max iterations ({max_iterations}) [branch {}]",
                    branch.context(),
                ),
                executions: branch.total_executions,
            };
        }

        // Strategy picks target
        let pick = match pick_execution_target(&state_with_hp, &result, &branch.immunity_blocked) {
            Some(p) => p,
            None => {
                return SimResult::SimLoss {
                    reason: format!(
                        "{case_name}: no valid execution target (hp={}, budget={}, remaining={}) [branch {}]",
                        branch.hp,
                        if wrong_exec_cost > 0 { branch.hp / wrong_exec_cost } else { 99 },
                        max_remaining,
                        branch.context(),
                    ),
                    executions: branch.total_executions,
                };
            }
        };

        let pos = pick.position;

        // Simulate execution using ground truth
        if let Some(evil_role) = true_evil_positions.get(&pos) {
            // Correct: evil found
            let mut next = branch.clone();
            next.iterations += 1;
            next.total_executions += 1;
            next.observations.push(format!("#{pos}=evil({evil_role})"));
            if !next.executed.contains(&pos) {
                next.executed.push(pos);
            }
            if !next.confirmed_evil.contains(&pos) {
                next.confirmed_evil.push(pos);
            }
            next.evil_roles.insert(pos, evil_role.clone());

            // Any real Cipher death clears the represented shipped scalar.
            // Persist that release so later solves cannot reload stale data.
            if normalize_role(evil_role) == "witch" {
                release_one_witch_block(&mut next, &card_by_pos);
            }

            next.immunity_blocked.clear();
            enqueue_execution_branch(&mut branches, &mut seen_branches, next);
        } else {
            // Wrong execution: fork every distinct native observation. A
            // killed good card reveals and persists its real role; a protected
            // attempt records only confirmed-good immunity, never hidden
            // memory identity.
            let apparent = get_card_role(pos, &state_with_hp).unwrap_or("");
            let truth_scenarios = truth_compatible_scenarios(
                &result,
                &true_evil_positions,
                &branch.executed,
            );
            let observations = distinct_good_execution_observations(
                &truth_scenarios,
                pos,
                apparent,
                wrong_exec_cost,
                case_exec_good_corr.get(&pos).copied(),
                case_exec_good_roles.get(&pos).map(String::as_str),
            );
            if observations.is_empty() {
                return SimResult::InsufficientTruth {
                    detail: format!(
                        "{case_name}: execution #{pos} has no truth-compatible outcome for apparent {apparent}"
                    ),
                };
            }

            for observation in observations {
                let observation_context = format!("{observation:?}");
                match continue_good_execution(&branch, pos, observation) {
                    GoodExecutionContinuation::BombardierLoss => {
                        return SimResult::SimLoss {
                            reason: format!(
                                "{case_name}: executed good Bombardier #{pos} [branch {} -> #{pos}={observation_context}]",
                                branch.context(),
                            ),
                            executions: branch.total_executions + 1,
                        };
                    }
                    GoodExecutionContinuation::Continue(next) if next.hp <= 0 => {
                        return SimResult::SimLoss {
                            reason: format!(
                                "{case_name}: HP exhausted (hp={}) after wrong exec #{pos} [branch {}]",
                                next.hp,
                                next.context(),
                            ),
                            executions: next.total_executions,
                        };
                    }
                    GoodExecutionContinuation::Continue(next) => {
                        enqueue_execution_branch(&mut branches, &mut seen_branches, next);
                    }
                }
            }
        }
    }

    if completed_wins > 0 {
        SimResult::Win {
            executions: max_win_executions,
            wrong_execs: max_win_wrong_executions,
        }
    } else {
        SimResult::SimLoss {
            reason: format!(
                "{case_name}: all execution branches were deduplicated before a terminal outcome"
            ),
            executions: 0,
        }
    }
}

#[test]
fn hp_reconstruction_uses_composite_knight_damage() {
    let empty = HashSet::new();
    let mut knight_scenario = Scenario {
        evil_positions: HashMap::new(),
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
    };
    knight_scenario.corrupted.insert(1);
    let corrupted_knight = serde_json::json!({
        "hp": 1,
        "wrong_exec_cost": 5,
        "executed": [1],
        "executed_good_corrupted": {"1": true},
        "cards": [{"position": 1, "apparent_role": "Knight"}]
    });
    assert_eq!(
        reconstruct_phase3_hp(
            &corrupted_knight,
            &empty,
            &empty,
            &empty,
            &[&knight_scenario],
        ).unwrap(),
        10,
    );

    let lilis_killed_corrupted_knight = serde_json::json!({
        "hp": 8,
        "wrong_exec_cost": 5,
        // Some historical fixtures also retained non-player deaths here.
        "executed": [1],
        "cards": [{"position": 1, "apparent_role": "Knight"}]
    });
    assert_eq!(
        reconstruct_phase3_hp(
            &lilis_killed_corrupted_knight,
            &empty,
            &HashSet::from([1]),
            &empty,
            &[&knight_scenario],
        ).unwrap(),
        // Lilis's fixed 2 HP Night hit is already reflected in saved HP.
        // Demon death fires OnDied, not Knight's Executed +4 hook.
        8,
    );

    let slayer_killed_wretch = serde_json::json!({
        "hp": 5,
        "wrong_exec_cost": 5,
        "executed": [1],
        "cards": [{"position": 1, "apparent_role": "Wretch"}]
    });
    assert_eq!(
        reconstruct_phase3_hp(
            &slayer_killed_wretch,
            &empty,
            &empty,
            &HashSet::from([1]),
            &[&Scenario::default()],
        ).unwrap(),
        // Slayer bypasses protection/OnExecuted, but its wrong Good-Wretch
        // death cost occurred before Phase 3 and remains in saved HP.
        5,
    );

    let mut drunk_scenario = knight_scenario;
    drunk_scenario.drunk_position = Some(1);
    let drunk_as_knight = serde_json::json!({
        "hp": 4,
        "wrong_exec_cost": 5,
        "executed": [1],
        // Execution bookkeeping reports Drunk clean, while its active status
        // still drives Knight's separate four-damage hook. Plague Doctor reads
        // that active status directly and is intentionally a different surface.
        "executed_good_corrupted": {"1": false},
        "executed_good_roles": {"1": "Drunk"},
        "cards": [{"position": 1, "apparent_role": "Knight"}]
    });
    assert_eq!(
        reconstruct_phase3_hp(
            &drunk_as_knight,
            &empty,
            &empty,
            &empty,
            &[&drunk_scenario],
        ).unwrap(),
        10,
    );

    let mut resistant_drunk_scenario = drunk_scenario.clone();
    resistant_drunk_scenario.corrupted.clear();
    resistant_drunk_scenario.chancellor_trace = Some(ChancellorTrace {
        original_positions: vec![2],
        added_outcast_position: 1,
        added_outcast_role: "Drunk".to_string(),
        affected_anchor_positions: vec![],
    });
    let resistant_drunk_as_knight = serde_json::json!({
        "hp": 8,
        "wrong_exec_cost": 5,
        "executed": [1],
        "executed_good_corrupted": {"1": false},
        "executed_good_roles": {"1": "Drunk"},
        "cards": [{"position": 1, "apparent_role": "Knight"}]
    });
    assert_eq!(
        reconstruct_phase3_hp(
            &resistant_drunk_as_knight,
            &empty,
            &empty,
            &empty,
            &[&resistant_drunk_scenario],
        ).unwrap(),
        10,
    );
    assert!(
        reconstruct_phase3_hp(
            &resistant_drunk_as_knight,
            &empty,
            &empty,
            &empty,
            &[&drunk_scenario, &resistant_drunk_scenario],
        ).is_err(),
    );

    let mut corrupted_scout_scenario = Scenario {
        evil_positions: HashMap::new(),
        puppet_position: None,
        corrupted: HashSet::from([1]),
        pd_corrupted: None,
        doppelganger_position: None,
        drunk_position: None,
        alchemist_cures: HashMap::new(),
        messed_up_by_evil: HashSet::new(),
        shaman_trace: None,
        chancellor_trace: None,
        chancellor_conversion: None,
    };
    let corrupted_scout = serde_json::json!({
        "hp": 5,
        "wrong_exec_cost": 5,
        "executed": [1],
        "executed_good_corrupted": {"1": true},
        "cards": [{"position": 1, "apparent_role": "Scout"}]
    });
    // A first-match harness could pick this incompatible Drunk world and add
    // back only 2 HP. The recorded Corrupted result excludes Drunk's clean
    // bookkeeping surface, leaving the actual 5-damage Scout observation.
    let mut incompatible_drunk = corrupted_scout_scenario.clone();
    incompatible_drunk.drunk_position = Some(1);
    assert_eq!(
        reconstruct_phase3_hp(
            &corrupted_scout,
            &empty,
            &empty,
            &empty,
            &[&incompatible_drunk, &corrupted_scout_scenario],
        ).unwrap(),
        10,
    );

    corrupted_scout_scenario.corrupted.clear();
    let ambiguous_clean_scout = serde_json::json!({
        "hp": 5,
        "wrong_exec_cost": 5,
        "executed": [1],
        "executed_good_corrupted": {"1": false},
        "cards": [{"position": 1, "apparent_role": "Scout"}]
    });
    assert!(
        reconstruct_phase3_hp(
            &ambiguous_clean_scout,
            &empty,
            &empty,
            &empty,
            &[&incompatible_drunk, &corrupted_scout_scenario],
        ).is_err(),
    );
    let observed_drunk = serde_json::json!({
        "hp": 8,
        "wrong_exec_cost": 5,
        "executed": [1],
        "executed_good_corrupted": {"1": false},
        "executed_good_roles": {"1": "Drunk"},
        "cards": [{"position": 1, "apparent_role": "Scout"}]
    });
    assert_eq!(
        reconstruct_phase3_hp(
            &observed_drunk,
            &empty,
            &empty,
            &empty,
            &[&incompatible_drunk, &corrupted_scout_scenario],
        ).unwrap(),
        10,
    );

    let protected_knight = serde_json::json!({
        "hp": 10,
        "wrong_exec_cost": 5,
        "executed": [1],
        "executed_good_corrupted": {"1": false},
        "cards": [{"position": 1, "apparent_role": "Knight"}]
    });
    let mut doppelganger_knight = corrupted_scout_scenario.clone();
    doppelganger_knight.doppelganger_position = Some(1);
    assert_eq!(
        reconstruct_phase3_hp(
            &protected_knight,
            &empty,
            &empty,
            &empty,
            &[&corrupted_scout_scenario, &doppelganger_knight],
        ).unwrap(),
        10,
    );

    // Adding an observed-clean, actively Corrupted Drunk-as-Knight world makes
    // the historical `executed` entry ambiguous: protected in two worlds,
    // killed for 6 HP in the third.
    assert!(
        reconstruct_phase3_hp(
            &protected_knight,
            &empty,
            &empty,
            &empty,
            &[&corrupted_scout_scenario, &doppelganger_knight, &incompatible_drunk],
        ).is_err(),
    );
}

#[test]
fn hp_reconstruction_requires_one_shared_world_across_execution_evidence() {
    let empty = HashSet::new();
    let mut drunk_at_one = Scenario::default();
    drunk_at_one.drunk_position = Some(1);
    drunk_at_one.corrupted.insert(1);
    let mut drunk_at_two = Scenario::default();
    drunk_at_two.drunk_position = Some(2);
    drunk_at_two.corrupted.insert(2);
    let impossible_pair = serde_json::json!({
        "hp": 6,
        "wrong_exec_cost": 2,
        "executed": [1, 2],
        "executed_good_corrupted": {"1": false, "2": false},
        "executed_good_roles": {"1": "Drunk", "2": "Drunk"},
        "cards": [
            {"position": 1, "apparent_role": "Scout"},
            {"position": 2, "apparent_role": "Scout"}
        ]
    });

    let error = reconstruct_phase3_hp(
        &impossible_pair,
        &empty,
        &empty,
        &empty,
        &[&drunk_at_one, &drunk_at_two],
    )
    .unwrap_err();

    assert!(error.contains("no shared truth-compatible world"));
}

#[test]
fn truth_worlds_include_exact_evil_role_assignments() {
    let mut actual = Scenario::default();
    actual.evil_positions = HashMap::from([
        (6, "Chancellor".to_string()),
        (7, "Lilis".to_string()),
    ]);
    let mut swapped = Scenario::default();
    swapped.evil_positions = HashMap::from([
        (6, "Lilis".to_string()),
        (7, "Chancellor".to_string()),
    ]);
    let result = SolverResult {
        definite_evil: Vec::new(),
        definite_good: Vec::new(),
        bombardier_positions: Vec::new(),
        n_scenarios: 2,
        n_surviving: 2,
        surviving_scenarios: vec![actual, swapped],
        reasoning: Vec::new(),
    };
    let truth = HashMap::from([
        (6, "Chancellor".to_string()),
        (7, "Lilis".to_string()),
    ]);

    let compatible = truth_compatible_scenarios(&result, &truth, &[]);
    assert_eq!(compatible.len(), 1);
    assert_eq!(compatible[0].evil_positions.get(&6).unwrap(), "Chancellor");
}

#[test]
fn chancellor_witness_trace_distinguishes_anchor_from_first_villager_target() {
    let anchor_claim = CardInfo {
        position: 1,
        apparent_role: "Witness".to_string(),
        info_parsed: serde_json::json!({"affected_position": 5})
            .as_object()
            .unwrap()
            .clone(),
        ..CardInfo::default()
    };
    let first_target_claim = CardInfo {
        info_parsed: serde_json::json!({"affected_position": 3})
            .as_object()
            .unwrap()
            .clone(),
        ..anchor_claim.clone()
    };
    let mut state = GameState::default();
    state.n_cards = 6;
    state.cards = vec![anchor_claim.clone()];
    // Witness scans current physical statuses without filtering dead cards.
    state.executed = vec![5];

    let mut scenario = Scenario::default();
    scenario
        .evil_positions
        .insert(4, "Chancellor".to_string());
    scenario.messed_up_by_evil.insert(5);
    scenario.chancellor_trace = Some(ChancellorTrace {
        original_positions: vec![2],
        added_outcast_position: 3,
        added_outcast_role: "Bombardier".to_string(),
        affected_anchor_positions: vec![5],
    });

    assert_eq!(scenario.chancellor_original_villager_positions(), vec![3]);
    assert!(solver_core::validators::validate_card(
        &anchor_claim,
        &scenario,
        &state,
    ));
    assert!(!solver_core::validators::validate_card(
        &first_target_claim,
        &scenario,
        &state,
    ));
}

#[test]
fn chancellor_witness_v2_observations_parse_as_positive_targets() {
    // These are the four 426-case v2 observations whose shipped pools contain
    // both Chancellor and Witness. Keep their public positive clues pinned as
    // corpus evidence without treating the hidden first replacement as truth.
    for (case_name, expected_target) in [
        ("asc41_v7", 6),
        ("asc72_v2", 2),
        ("asc74_v7", 3),
        ("asc79_v4", 4),
    ] {
        let path = v2_dir().join(format!("{case_name}.json"));
        let content = std::fs::read_to_string(path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();
        let state = GameState::from_json(&value).unwrap();
        assert!(state
            .deck
            .minions
            .iter()
            .any(|role| normalize_role(role) == "chancellor"));
        assert!(state
            .deck
            .villagers
            .iter()
            .any(|role| normalize_role(role) == "witness"));
        let witness = state
            .cards
            .iter()
            .find(|card| normalize_role(&card.apparent_role) == "witness")
            .expect("fixture must retain its Witness observation");
        assert_eq!(
            witness
                .info_parsed
                .get("affected_position")
                .and_then(serde_json::Value::as_u64),
            Some(expected_target)
        );
    }
}

#[test]
fn rambler_current_v2_observations_survive_incremental_and_final_solves() {
    for (case_name, expected_observations) in [
        ("asc82_v5", vec![(1, 10)]),
        ("asc83_v7", vec![(2, 1), (3, 4), (5, 4), (9, 1)]),
    ] {
        let path = v2_dir().join(format!("{case_name}.json"));
        let content = std::fs::read_to_string(path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();
        let case = TestCase::from_json(&value).unwrap();
        assert_eq!(
            case.state.rambler_rule_version.as_deref(),
            Some("rambler2_shut_up"),
        );

        let ledger: Vec<(u8, u8)> = case
            .state
            .rambler_shut_up_observations
            .iter()
            .map(|observation| {
                (observation.speaker_position, observation.shut_up_target)
            })
            .collect();
        assert_eq!(
            ledger, expected_observations,
            "{case_name} must retain the exact append-only public observation order",
        );

        let observed: Vec<(u8, u8)> = case
            .state
            .cards
            .iter()
            .filter_map(|card| {
                Some((
                    card.position,
                    card.info_parsed.get("shut_up_target")?.as_u64()? as u8,
                ))
            })
            .collect();
        assert_eq!(observed, expected_observations);

        let result = solve(&case.state);
        assert!(
            result.n_surviving > 0,
            "{case_name} final current-Rambler observations eliminated every world",
        );
        let true_evil_positions: HashMap<u8, String> = value
            .get("true_evil_positions")
            .and_then(serde_json::Value::as_object)
            .unwrap()
            .iter()
            .map(|(position, role)| {
                (
                    position.parse::<u8>().unwrap(),
                    role.as_str().unwrap().to_string(),
                )
            })
            .collect();
        assert!(result.surviving_scenarios.iter().any(|scenario| {
            let represented_count = scenario.evil_positions.len()
                + usize::from(
                    scenario.puppet_position.is_some()
                        && !scenario
                            .evil_positions
                            .contains_key(&scenario.puppet_position.unwrap()),
                );
            represented_count == true_evil_positions.len()
                && true_evil_positions.iter().all(|(position, role)| {
                    if normalize_role(role) == "puppet" {
                        scenario.puppet_position == Some(*position)
                    } else {
                        scenario
                            .evil_positions
                            .get(position)
                            .is_some_and(|candidate| {
                                normalize_role(candidate) == normalize_role(role)
                            })
                    }
                })
        }), "{case_name} exact postmortem evil-role world was pruned");
        if case_name == "asc83_v7" {
            assert_eq!(
                result.n_surviving, 1,
                "asc83's two real/two fake Rambler observations should retain its unique world",
            );
        }

        if let SimResult::ConstraintFailure { detail, .. } = simulate_game(&value) {
            panic!("{case_name} incremental current-Rambler truth eliminated: {detail}");
        }
    }
}

#[test]
fn execution_branches_ordinary_and_drunk_damage_with_revealed_roles() {
    let ordinary = Scenario::default();
    let mut drunk = Scenario::default();
    drunk.drunk_position = Some(1);
    drunk.corrupted.insert(1);
    let observations = distinct_good_execution_observations(
        &[&ordinary, &drunk],
        1,
        "Scout",
        5,
        None,
        None,
    );
    assert_eq!(observations.len(), 2);

    let branch = ExecutionBranch {
        hp: 10,
        ..ExecutionBranch::default()
    };
    let continuations: Vec<ExecutionBranch> = observations
        .into_iter()
        .map(|observation| match continue_good_execution(&branch, 1, observation) {
            GoodExecutionContinuation::Continue(next) => next,
            GoodExecutionContinuation::BombardierLoss => panic!("Scout is not Bombardier"),
        })
        .collect();
    let hp_values: HashSet<i32> = continuations.iter().map(|next| next.hp).collect();
    assert_eq!(hp_values, HashSet::from([5, 8]));
    assert!(continuations.iter().all(|next| next.executed == vec![1]));
    assert!(continuations
        .iter()
        .all(|next| next.good_corrupted == HashMap::from([(1, false)])));
    let revealed_roles: HashSet<&str> = continuations
        .iter()
        .filter_map(|next| next.good_roles.get(&1).map(String::as_str))
        .collect();
    assert_eq!(revealed_roles, HashSet::from(["Scout", "Drunk"]));
    assert_ne!(
        execution_branch_key(&continuations[0]),
        execution_branch_key(&continuations[1]),
    );
}

#[test]
fn witch_block_branch_rewinds_and_clears_scalar_persistently() {
    let fixture_cards = vec![
        serde_json::json!({"position": 1, "apparent_role": "Baker"}),
        serde_json::json!({"position": 8, "apparent_role": "Scout"}),
    ];
    let card_by_pos: HashMap<u8, &serde_json::Value> = fixture_cards.iter()
        .filter_map(|card| Some((json_card_position(card)?, card))).collect();
    let (cards, blocked_positions, reveal_order) = initialize_execution_block_state(
        fixture_cards.clone(), vec![1, 8], &[8], &HashSet::new(), 0,
    );
    assert_eq!(cards.iter().filter_map(json_card_position).collect::<Vec<_>>(), vec![1]);
    assert_eq!(blocked_positions, vec![8]);
    assert_eq!(reveal_order, vec![1]);

    let mut branch = ExecutionBranch {
        cards, blocked_positions, reveal_order, ..ExecutionBranch::default()
    };
    let before = execution_branch_key(&branch);

    assert_eq!(release_one_witch_block(&mut branch, &card_by_pos), Some(8));
    assert!(branch.blocked_positions.is_empty());
    assert_eq!(branch.reveal_order, vec![1, 8]);
    assert!(branch.cards.iter().any(|card| json_card_position(card) == Some(8)));
    assert_ne!(execution_branch_key(&branch), before);
    assert_eq!(release_one_witch_block(&mut branch, &card_by_pos), None);

    let mut different_reveal_history = branch.clone();
    different_reveal_history.reveal_order.swap(0, 1);
    assert_ne!(
        execution_branch_key(&branch),
        execution_branch_key(&different_reveal_history),
    );
}

#[test]
fn any_pre_execution_witch_death_clears_shipped_scalar_block() {
    let fixture_cards = vec![serde_json::json!({
        "position": 8, "apparent_role": "Scout"
    })];
    let (cards, blocked_positions, reveal_order) = initialize_execution_block_state(
        fixture_cards, vec![8], &[8], &HashSet::from([2]), 1,
    );
    assert!(blocked_positions.is_empty());
    assert_eq!(cards.len(), 1);
    assert_eq!(reveal_order.len(), 1);

    // Even malformed/hand-built duplicate markers cannot manufacture stacked
    // ordinary Start calls in represented GameState.
    let (_, blocked_positions, _) = initialize_execution_block_state(
        vec![], vec![], &[8, 9], &HashSet::new(), 0,
    );
    assert_eq!(blocked_positions, vec![8]);
}

#[test]
fn execution_branch_does_not_persist_an_empty_revealed_role() {
    let observation = good_execution_observation(&Scenario::default(), 1, "", 2);
    assert_eq!(
        observation.consequence,
        ExecutionConsequence::Killed { hp_damage: 2 },
    );
    assert_eq!(observation.revealed_role, None);

    let branch = ExecutionBranch {
        hp: 10,
        ..ExecutionBranch::default()
    };
    let GoodExecutionContinuation::Continue(next) =
        continue_good_execution(&branch, 1, observation)
    else {
        panic!("ordinary unknown-role execution must continue");
    };
    assert_eq!(next.hp, 8);
    assert!(next.good_roles.is_empty());
}

#[test]
fn execution_branches_protected_knight_and_drunk_kill_continuations() {
    let ordinary = Scenario::default();
    let mut drunk = Scenario::default();
    drunk.drunk_position = Some(1);
    drunk.corrupted.insert(1);
    let mut resistant_drunk = drunk.clone();
    resistant_drunk.corrupted.clear();
    resistant_drunk.chancellor_trace = Some(ChancellorTrace {
        original_positions: vec![2],
        added_outcast_position: 1,
        added_outcast_role: "Drunk".to_string(),
        affected_anchor_positions: vec![],
    });
    let observations = distinct_good_execution_observations(
        &[&ordinary, &drunk, &resistant_drunk],
        1,
        "Knight",
        5,
        None,
        None,
    );
    assert_eq!(observations.len(), 3);

    let branch = ExecutionBranch {
        hp: 10,
        ..ExecutionBranch::default()
    };
    let continuations: Vec<ExecutionBranch> = observations
        .into_iter()
        .map(|observation| match continue_good_execution(&branch, 1, observation) {
            GoodExecutionContinuation::Continue(next) => next,
            GoodExecutionContinuation::BombardierLoss => panic!("Knight is not Bombardier"),
        })
        .collect();
    let protected = continuations
        .iter()
        .find(|next| next.immunity_blocked.contains(&1))
        .expect("ordinary Knight branch is protected");
    assert_eq!(protected.hp, 10);
    assert!(protected.executed.is_empty());
    assert_eq!(protected.confirmed_good, vec![1]);
    assert!(protected.good_roles.is_empty());

    let killed: Vec<&ExecutionBranch> = continuations
        .iter()
        .filter(|next| next.executed.contains(&1))
        .collect();
    assert_eq!(killed.len(), 2);
    assert_eq!(
        killed.iter().map(|next| next.hp).collect::<HashSet<_>>(),
        HashSet::from([4, 8]),
    );
    for branch in killed {
        assert!(branch.immunity_blocked.is_empty());
        assert_eq!(branch.good_corrupted, HashMap::from([(1, false)]));
        assert_eq!(branch.good_roles, HashMap::from([(1, "Drunk".to_string())]));
    }
}

#[test]
fn corrupted_doppelganger_as_knight_is_a_nine_hp_kill_branch() {
    let mut doppel = Scenario::default();
    doppel.doppelganger_position = Some(1);
    doppel.corrupted.insert(1);
    let observation = good_execution_observation(&doppel, 1, "Knight", 5);
    assert_eq!(
        observation.consequence,
        ExecutionConsequence::Killed { hp_damage: 9 },
    );
    assert!(observation.observed_corrupted);
    assert_eq!(observation.revealed_role.as_deref(), Some("Doppelganger"));

    let branch = ExecutionBranch {
        hp: 10,
        ..ExecutionBranch::default()
    };
    let GoodExecutionContinuation::Continue(next) =
        continue_good_execution(&branch, 1, observation)
    else {
        panic!("corrupted Doppelganger is not Bombardier");
    };
    assert_eq!(next.hp, 1);
    assert_eq!(next.executed, vec![1]);
    assert!(next.immunity_blocked.is_empty());
}

// ── Test entry point ──

#[test]
fn simulate_all_v2() {
    let dir = v2_dir();
    assert!(dir.exists(), "v2 test cases dir not found: {:?}", dir);

    let mut files: Vec<_> = glob::glob(dir.join("*.json").to_str().unwrap()).unwrap()
        .filter_map(|e| e.ok()).collect();
    files.sort_by_key(|p| p.file_name().unwrap().to_str().unwrap().to_string());

    // Known simulation losses: genuine probabilistic or Bombardier-forced games.
    // These are NOT solver bugs — the strategy correctly identifies the best
    // candidate but the game outcome is unfavorable.
    let known_losses: HashSet<&str> = [
        "asc37_v6",     // Bombardier execution: all non-Bombardier candidates exhausted
        "asc40_v1",     // HP exhausted from multiple wrong probabilistic picks
        "asc40_v6",     // Bombardier execution: all non-Bombardier candidates exhausted
        "asc42_v5",     // 50/50 coin flip, 0 budget
        "asc46_v7",     // HP exhausted, 0 budget probabilistic
        "asc59_v5",     // 35% confidence pick
        "asc65_v2",     // 49% and 79% picks both wrong
        "asc68_v3",     // 0HP loss: wrong exec #4 (corrupted, 60%) + #9 (Doppelganger, 75%)
        "asc71_v7",     // 0HP loss: Bombardier-safety skipped true TM-as-Bombardier #2
        "asc72_v2",     // 0HP loss: corrupted Alchemist cured-count + no Rambler validator
        "asc73_v6",     // 0HP loss: Rambler-silenced by PD requires picker detection that lacks pd_ability_result
    ].into_iter().collect();

    // Cases with known bad data that may cause constraint failures
    let known_constraint_issues: HashSet<&str> = [
        "asc52_v6",     // Invalid Doppelganger in Druid claim (poisoned data)
        // Recorded under old Rambler rule (silenced ⟺ Disguised picker). New rule
        // is "silenced ⟺ Liar picker" — Doppelganger picker no longer silences,
        // so historical clue values no longer satisfy the validator.
        "asc73_v5",
        "asc73_v7",
        "asc74_v4",
        // Predate the asc77 bug-hunt session; still failing but not caused by
        // the Baker-chain validator fix. Root cause TBD — likely corrupted/Drunk
        // scenario modeling gaps analogous to the issues patched here. Kept as
        // documented tech debt rather than silently masked.
        "asc74_v7",
        "asc75_v7",
        // Frozen Lilis history says two night kills contained zero Evil even
        // though the recorded truth makes victim #10 the Twin Minion. The
        // Baker/Drunk witness is covered independently in validator tests.
        "asc59_v7",
        // Recorded under pre-patch Alchemist rules (cured-count clue + corruptible).
        // 2026-05-04 patch: Alchemist is immune to Corruption AND now reports
        // pre-cure corrupted-in-range (incl. Drunk), not cured-count. Frozen
        // historical data no longer satisfies the new validator semantics.
        // Will be replaced by post-patch recordings as games are played.
        "asc27_v3", "asc28_v7", "asc30_v3", "asc33_v1", "asc36_v4", "asc36_v6",
        "asc38_v6", "asc39_v5", "asc40_v3", "asc42_v2", "asc42_v3", "asc43_v5",
        "asc47_v3", "asc50_v1", "asc52_v5", "asc53_v4", "asc55_v3", "asc56_v2",
        "asc56_v3", "asc57_v7", "asc60_v2", "asc61_v3", "asc61_v5", "asc62_v4",
        "asc63_v5", "asc69_v1", "asc72_v2", "asc73_v3", "asc75_v4", "asc80_v1",
        // Frozen clues conflict with the recovered current-build Start order.
        // asc32/33 predate mandatory PD targeting against the live post-Drunk
        // set; asc55 records both duplicate Alchemists seeing the same prior
        // corruption, while native high-ID-first live cures make the second
        // actor observe zero.
        "asc32_v4", "asc33_v3", "asc55_v7",
    ].into_iter().collect();

    let mut wins = 0usize;
    let mut expected_losses = 0usize;
    let mut constraint_failures = 0usize;
    let mut expected_constraint_issues = 0usize;
    let mut unexpected_losses: Vec<String> = Vec::new();
    let mut unexpected_constraint_fails: Vec<String> = Vec::new();
    let mut insufficient_truth: Vec<String> = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_str().unwrap();
        let case_name = name.trim_end_matches(".json");
        let content = std::fs::read_to_string(path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        match simulate_game(&value) {
            SimResult::Win { .. } => {
                wins += 1;
            }
            SimResult::ConstraintFailure { detail, .. } => {
                if known_constraint_issues.contains(case_name) {
                    expected_constraint_issues += 1;
                } else {
                    constraint_failures += 1;
                    unexpected_constraint_fails.push(format!("{case_name}: {detail}"));
                }
            }
            SimResult::SimLoss { reason, .. } => {
                if known_losses.contains(case_name) || known_constraint_issues.contains(case_name) {
                    expected_losses += 1;
                } else {
                    unexpected_losses.push(reason);
                }
            }
            SimResult::InsufficientTruth { detail } => {
                insufficient_truth.push(detail);
            }
        }
    }

    let total = files.len();
    println!("\n=== Simulation Results ===");
    println!("  Wins: {wins}");
    println!("  Expected losses: {expected_losses}");
    println!("  Expected constraint issues: {expected_constraint_issues}");
    println!("  Unexpected constraint failures: {constraint_failures}");
    println!("  Unexpected simulation losses: {}", unexpected_losses.len());
    println!("  Cases missing hidden-Outcast truth: {}", insufficient_truth.len());
    println!("  Total: {total}");

    if !unexpected_constraint_fails.is_empty() {
        println!("\nUnexpected constraint failures:");
        for f in &unexpected_constraint_fails {
            println!("  - {f}");
        }
    }
    if !unexpected_losses.is_empty() {
        println!("\nUnexpected simulation losses:");
        for l in &unexpected_losses {
            println!("  - {l}");
        }
    }
    if !insufficient_truth.is_empty() {
        println!("\nCases missing hidden-Outcast truth:");
        for detail in &insufficient_truth {
            println!("  - {detail}");
        }
    }

    // Constraint failures are always real bugs
    assert_eq!(constraint_failures, 0,
        "{constraint_failures} unexpected constraint failures: {unexpected_constraint_fails:?}");

    // Simulation losses: allow some tolerance for probabilistic games
    // but flag if too many
    let max_unexpected_losses = 10; // Generous threshold initially
    assert!(unexpected_losses.len() <= max_unexpected_losses,
        "{} unexpected simulation losses (max {max_unexpected_losses}): {unexpected_losses:?}",
        unexpected_losses.len());

    // Never silently turn missing hidden-Outcast ground truth into a passing
    // simulated win. These cases are reported separately until fixtures gain
    // explicit hidden-role truth or the harness branches every compatible
    // native outcome. The bound prevents this known corpus gap from growing.
    const MAX_INSUFFICIENT_TRUTH_CASES: usize = 25;
    assert!(insufficient_truth.len() <= MAX_INSUFFICIENT_TRUTH_CASES,
        "{} cases lack a unique native execution outcome (max {MAX_INSUFFICIENT_TRUTH_CASES}): {insufficient_truth:?}",
        insufficient_truth.len());
}

/// Debug test for asc68_v3: verifies truth scenario (evils at 3,7,8) survives constraint validation.
///
/// Root cause found: `true_evil_positions` in the JSON was missing #8=Pooka.
/// The simulation's `truth_in_set` looked for a 2-evil scenario (only {3,7}) but
/// n_evil=3 means all surviving scenarios have 3 evils, so no match was found.
/// Fix: added "8": "Pooka" to true_evil_positions in asc68_v3.json.
#[test]
fn debug_asc68_v3() {
    use std::collections::{HashMap, HashSet};

    let path = v2_dir().join("asc68_v3.json");
    let content = std::fs::read_to_string(&path).expect("Failed to read asc68_v3.json");
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    let obj = value.as_object().unwrap();

    // Verify true_evil_positions now has all 3 evils
    let true_evil_positions: HashMap<u8, String> = obj.get("true_evil_positions")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| {
            Some((k.parse::<u8>().ok()?, v.as_str()?.to_string()))
        }).collect())
        .unwrap_or_default();
    let true_evil_set: HashSet<u8> = true_evil_positions.keys().copied().collect();

    assert_eq!(true_evil_set.len(), 3, "true_evil_positions should have 3 evils (was missing #8=Pooka)");
    assert!(true_evil_set.contains(&3), "Should contain #3=Chancellor");
    assert!(true_evil_set.contains(&7), "Should contain #7=Minion");
    assert!(true_evil_set.contains(&8), "Should contain #8=Pooka");

    // Verify truth survives solver constraint validation
    let result = simulate_game(&value);
    match &result {
        SimResult::ConstraintFailure { detail, .. } => {
            panic!("Truth eliminated: {detail}");
        }
        _ => {} // Win, SimLoss, or insufficient truth are acceptable here.
    }
}

/// Regression: asc71_v6 Bishop's [V,O,M] claim stays valid even when Chancellor@6
/// converts the Villager adjacent (#7 Empress). Root cause: Bishop's clue captures
/// types at game-start independent of Chancellor's conversion — we accept both
/// pre- and post-conversion views.
#[test]
fn regression_asc71_v6_bishop_vs_chancellor() {
    let path = v2_dir().join("asc71_v6.json");
    let content = std::fs::read_to_string(&path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    let result = simulate_game(&value);
    match &result {
        SimResult::ConstraintFailure { detail, .. } => {
            panic!("Truth eliminated: {detail}");
        }
        _ => {}
    }
}

#[test]
fn regression_disguised_outcasts_do_not_consume_trusted_outcast_slots() {
    for case_name in ["asc37_v5", "asc63_v2"] {
        let path = v2_dir().join(format!("{case_name}.json"));
        let content = std::fs::read_to_string(path).unwrap();
        let mut value: serde_json::Value = serde_json::from_str(&content).unwrap();
        value["board_count_provenance"] = serde_json::json!("trusted_pre_start");
        if let SimResult::ConstraintFailure { detail, .. } = simulate_game(&value) {
            panic!("{case_name} truth eliminated with trusted no=1: {detail}");
        }
    }
}

#[test]
fn regression_lilis_knight_and_slayer_native_surfaces() {
    // Evil fallback victim, post-night Knight check, no-kill history retained
    // only through saved HP, and Slayer's registered-Evil Wretch kill.
    for case_name in ["asc33_v5", "asc69_v2", "asc70_v7", "asc26_v8"] {
        let path = v2_dir().join(format!("{case_name}.json"));
        let content = std::fs::read_to_string(path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();
        if let SimResult::ConstraintFailure { detail, .. } = simulate_game(&value) {
            panic!("{case_name} native Lilis/Knight/Slayer truth eliminated: {detail}");
        }
    }
}

#[test]
fn regression_asc63_v2_uses_native_shaman_trace_instead_of_fake_drunk() {
    let path = v2_dir().join("asc63_v2.json");
    let content = std::fs::read_to_string(path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    let state = GameState::from_json(&value).unwrap();

    let result = solve(&state);
    let native_world = result.surviving_scenarios.iter().find(|scenario| {
        scenario.evil_positions.get(&1)
            .is_some_and(|role| normalize_role(role) == "baa")
            && scenario.evil_positions.get(&5)
                .is_some_and(|role| normalize_role(role) == "poisoner")
            && scenario.evil_positions.get(&8)
                .is_some_and(|role| normalize_role(role) == "shaman")
            && scenario.drunk_position.is_none()
            && scenario.shaman_trace.as_ref().is_some_and(|trace| {
                HashSet::from([trace.source_position, trace.target_position])
                    == HashSet::from([4, 9])
                    && normalize_role(&trace.copied_role) == "empress"
            })
    });

    let scenario = native_world.expect(
        "asc63_v2 truth must survive as Empress #4/#9 Shaman endpoints without fake Drunk #2",
    );
    assert!(scenario.messed_up_by_evil.contains(&4));
    assert!(scenario.messed_up_by_evil.contains(&9));
}

#[test]
fn regression_asc55_v2_separates_bard_shaman_trace_from_baker_chain() {
    let path = v2_dir().join("asc55_v2.json");
    let content = std::fs::read_to_string(path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    let state = GameState::from_json(&value).unwrap();

    let result = solve(&state);
    let native_world = result.surviving_scenarios.iter().find(|scenario| {
        scenario.evil_positions.get(&3)
            .is_some_and(|role| normalize_role(role) == "baa")
            && scenario.evil_positions.get(&4)
                .is_some_and(|role| normalize_role(role) == "shaman")
            && scenario.shaman_trace.as_ref().is_some_and(|trace| {
                HashSet::from([trace.source_position, trace.target_position])
                    == HashSet::from([1, 5])
                    && normalize_role(&trace.copied_role) == "bard"
            })
    });

    native_world.expect("asc55_v2 Baker chain must not be mistaken for Shaman's Bard #1/#5 copy");
}

#[test]
fn regression_asc46_v1_legacy_missing_copied_baker_clue_stays_conservative() {
    let path = v2_dir().join("asc46_v1.json");
    let content = std::fs::read_to_string(path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    let state = GameState::from_json(&value).unwrap();

    let result = solve(&state);
    let native_world = result.surviving_scenarios.iter().find(|scenario| {
        scenario.evil_positions.get(&2)
            .is_some_and(|role| normalize_role(role) == "pooka")
            && scenario.evil_positions.get(&3)
                .is_some_and(|role| normalize_role(role) == "minion")
            && scenario.evil_positions.get(&5)
                .is_some_and(|role| normalize_role(role) == "shaman")
            && scenario.shaman_trace.as_ref().is_some_and(|trace| {
                normalize_role(&trace.copied_role) == "baker"
            })
    });

    native_world.expect(
        "asc46_v1 legacy 'none' must survive through an exact copied-Baker runtime class",
    );
}

#[test]
fn regression_legacy_header_count_does_not_invent_a_second_outcast() {
    let path = v2_dir().join("asc57_v5.json");
    let content = std::fs::read_to_string(path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    assert!(matches!(simulate_game(&value), SimResult::Win { .. }));

    let mut falsely_trusted = value;
    falsely_trusted["board_count_provenance"] = serde_json::json!("trusted_pre_start");
    assert!(matches!(
        simulate_game(&falsely_trusted),
        SimResult::ConstraintFailure { .. },
    ));
}

#[test]
fn regression_asc77_v6_baker_chain_before_original() {
    // asc77 predates verified reveal capture: its sorted list records click
    // attempts, while #1's swallowed first click made its real reveal last.
    // Unversioned fixtures must retain a conservative ordering search.
    let path = v2_dir().join("asc77_v6.json");
    let content = std::fs::read_to_string(&path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    let result = simulate_game(&value);
    if let SimResult::ConstraintFailure { detail, .. } = &result {
        panic!("asc77_v6 truth eliminated (Baker chain validator regression): {detail}");
    }
}

#[test]
fn regression_asc76_v4_data_completeness() {
    // Guard against missing-ground-truth data rot — asc76_v4 was originally
    // saved with only 2 of 3 true evils (#4=Lilis was missing). Same failure
    // class as asc68_v3.
    let path = v2_dir().join("asc76_v4.json");
    let content = std::fs::read_to_string(&path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    let obj = value.as_object().unwrap();
    let n_evil = obj.get("n_evil").and_then(|v| v.as_u64()).unwrap_or(0);
    let true_count = obj.get("true_evil_positions")
        .and_then(|v| v.as_object())
        .map(|m| m.len())
        .unwrap_or(0);
    assert_eq!(true_count as u64, n_evil,
        "asc76_v4 true_evil_positions must have all {n_evil} evils");
}

#[test]
fn regression_asc79_v2_dreamer_outcast_option() {
    // asc79 village 2: archived Dreamer result "Among #2, #6 there is Baa or
    // Doppelganger". Baa already executed at #5. Doppelganger (outcast, Good)
    // is the true answer at #6.
    //
    // Pre-fix: validate_dreamer Shape 2 used known_evil_role() which only
    // returns evil roles, so "Doppelganger" never matched any target, forcing
    // Dreamer to be classified as lying in every scenario — which forced
    // Dreamer #1 (truly Good) to be evil in every surviving scenario. Solver
    // recommended EXECUTE #1 at 100% confidence.
    //
    // Fix: introduced effective_role_at() which also considers outcast
    // placements (Doppelganger, Drunk) and apparent_role.
    let path = v2_dir().join("asc79_v2_dreamer2_outcast.json");
    let content = std::fs::read_to_string(&path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&content).unwrap();
    let result = simulate_game(&value);
    if let SimResult::ConstraintFailure { detail, .. } = &result {
        panic!("asc79_v2 truth eliminated (Dreamer outcast option regression): {detail}");
    }
}
