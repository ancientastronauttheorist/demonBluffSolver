//! Replay test: step-by-step validation matching Python's test_replay.py.
//! 3 phases: reveal cards, apply abilities, execute.

use solver_core::solver::solve;
use solver_core::types::*;
use std::collections::{HashMap, HashSet};
use std::path::PathBuf;

fn v2_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../tests/cases_v2")
}

/// Determine night kill timing (which reveal count triggers which kills).
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
    // Overflow: remaining kills go to last trigger
    if nk_idx < night_kills.len() {
        let last_trigger = timing.keys().max().copied().unwrap_or(4);
        for &nk in &night_kills[nk_idx..] {
            timing.entry(last_trigger).or_default().push(nk);
        }
    }
    timing
}

/// Check if the true evil assignment exists in surviving scenarios.
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

/// Build a GameState from incremental replay state.
fn make_state(
    case: &serde_json::Value,
    cards: &[serde_json::Value],
    executed: &[u8],
    confirmed_evil: &[u8],
    confirmed_good: &[u8],
    executed_evil_roles: &HashMap<u8, String>,
    executed_good_corrupted: &HashMap<u8, bool>,
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
    state.slayer_results = slayer_results.to_vec();
    state.pd_ability_results = pd_ability_results.to_vec();
    state.night_kills = night_kills.to_vec();
    state.night_kill_evil_count = night_kill_evil_count;
    state.reveal_order = reveal_order.to_vec();
    state
}

/// Replay a single test case. Returns (passed, n_surviving at final step).
fn replay_game(value: &serde_json::Value) -> (bool, usize) {
    let obj = value.as_object().unwrap();
    let case_name = obj.get("name").and_then(|v| v.as_str()).unwrap_or("unknown");

    // Parse true evil positions
    let true_evil_set: HashSet<u8> = obj.get("true_evil_positions")
        .and_then(|v| v.as_object())
        .map(|m| m.keys().filter_map(|k| k.parse::<u8>().ok()).collect())
        .unwrap_or_default();

    // Get reveal order
    let reveal_order: Vec<u8> = obj.get("reveal_order")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
        .unwrap_or_else(|| {
            obj.get("cards").and_then(|v| v.as_array())
                .map(|arr| arr.iter().filter_map(|c| c.get("position")?.as_i64().map(|x| x as u8)).collect())
                .unwrap_or_default()
        });

    let case_cards: Vec<serde_json::Value> = obj.get("cards")
        .and_then(|v| v.as_array())
        .cloned().unwrap_or_default();
    let card_by_pos: HashMap<u8, &serde_json::Value> = case_cards.iter()
        .filter_map(|c| {
            let pos = c.get("position")?.as_i64()? as u8;
            Some((pos, c))
        }).collect();

    let case_night_kills: Vec<u8> = obj.get("night_kills")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
        .unwrap_or_default();
    let nk_timing = night_kill_timing(&reveal_order, &case_night_kills);

    let case_confirmed_evil: HashSet<u8> = obj.get("confirmed_evil")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
        .unwrap_or_default();

    let case_evil_roles: HashMap<u8, String> = obj.get("executed_evil_roles")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| {
            Some((k.parse::<u8>().ok()?, v.as_str()?.to_string()))
        }).collect())
        .unwrap_or_default();

    let case_confirmed_good: HashSet<u8> = obj.get("confirmed_good")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
        .unwrap_or_default();

    let case_exec_good_corr: HashMap<u8, bool> = obj.get("executed_good_corrupted")
        .and_then(|v| v.as_object())
        .map(|m| m.iter().filter_map(|(k, v)| {
            Some((k.parse::<u8>().ok()?, v.as_bool()?))
        }).collect())
        .unwrap_or_default();

    // Incremental state
    let mut current_cards: Vec<serde_json::Value> = Vec::new();
    let mut current_reveal_order: Vec<u8> = Vec::new();
    let mut current_executed: Vec<u8> = Vec::new();
    let mut current_confirmed_evil: Vec<u8> = Vec::new();
    let mut current_confirmed_good: Vec<u8> = Vec::new();
    let mut current_evil_roles: HashMap<u8, String> = HashMap::new();
    let mut current_good_corr: HashMap<u8, bool> = HashMap::new();
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

        // Process night kills at this reveal count
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

    // Add card data for positions not in reveal_order (e.g., night-killed with Medium data)
    let revealed_positions: HashSet<u8> = current_reveal_order.iter().copied().collect();
    for card_val in &case_cards {
        if let Some(pos) = card_val.get("position").and_then(|v| v.as_i64()).map(|x| x as u8) {
            if !revealed_positions.contains(&pos) {
                current_cards.push(card_val.clone());
            }
        }
    }

    // ── Phase 2: Apply abilities ──
    // Slayer results
    if let Some(slayer_arr) = obj.get("slayer_results").and_then(|v| v.as_array()) {
        for sr_val in slayer_arr {
            let sr: SlayerResult = serde_json::from_value(sr_val.clone()).unwrap();
            if sr.killed {
                if !current_executed.contains(&sr.target_pos) {
                    current_executed.push(sr.target_pos);
                }
                if true_evil_set.contains(&sr.target_pos) {
                    if !current_confirmed_evil.contains(&sr.target_pos) {
                        current_confirmed_evil.push(sr.target_pos);
                    }
                    let role = sr.evil_role.clone()
                        .or_else(|| case_evil_roles.get(&sr.target_pos).cloned())
                        .unwrap_or_default();
                    if !role.is_empty() {
                        current_evil_roles.insert(sr.target_pos, role);
                    }
                } else {
                    if !current_confirmed_good.contains(&sr.target_pos) {
                        current_confirmed_good.push(sr.target_pos);
                    }
                }
            }
            current_slayer.push(sr);
        }
    }

    // PD results
    if let Some(pd_arr) = obj.get("pd_ability_results").and_then(|v| v.as_array()) {
        for pd_val in pd_arr {
            let pr: PdAbilityResult = serde_json::from_value(pd_val.clone()).unwrap();
            current_pd.push(pr);
        }
    }

    // Run solver with all info
    let state = make_state(value, &current_cards, &current_executed,
        &current_confirmed_evil, &current_confirmed_good,
        &current_evil_roles, &current_good_corr,
        &current_slayer, &current_pd,
        &current_night_kills, current_nk_evil_count,
        &current_reveal_order);

    let result = solve(&state);
    if !truth_in_set(&result, &true_evil_set, &current_executed) {
        eprintln!("FAIL {case_name}: truth eliminated after reveals+abilities ({} surviving)", result.n_surviving);
        return (false, result.n_surviving);
    }

    // Check definite_evil/good correctness
    for &pos in &result.definite_evil {
        if !true_evil_set.contains(&pos) {
            eprintln!("FAIL {case_name}: false definite_evil #{pos}");
            return (false, result.n_surviving);
        }
    }
    for &pos in &result.definite_good {
        if true_evil_set.contains(&pos) {
            eprintln!("FAIL {case_name}: false definite_good #{pos}");
            return (false, result.n_surviving);
        }
    }

    // ── Phase 3: Execute one by one ──
    let case_executed: Vec<u8> = obj.get("executed")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_i64().map(|x| x as u8)).collect())
        .unwrap_or_default();

    let nk_set: HashSet<u8> = case_night_kills.iter().copied().collect();
    let exec_steps: Vec<u8> = case_executed.iter()
        .filter(|&&p| !nk_set.contains(&p))
        .copied().collect();

    let mut last_surviving = result.n_surviving;

    for &pos in &exec_steps {
        if current_executed.contains(&pos) { continue; }
        current_executed.push(pos);

        let was_evil = case_confirmed_evil.contains(&pos) || case_evil_roles.contains_key(&pos);
        if was_evil {
            if !current_confirmed_evil.contains(&pos) {
                current_confirmed_evil.push(pos);
            }
            if let Some(role) = case_evil_roles.get(&pos) {
                current_evil_roles.insert(pos, role.clone());
            }
        } else {
            if !current_confirmed_good.contains(&pos) {
                current_confirmed_good.push(pos);
            }
            if let Some(&was_corr) = case_exec_good_corr.get(&pos) {
                current_good_corr.insert(pos, was_corr);
            }
        }

        let state = make_state(value, &current_cards, &current_executed,
            &current_confirmed_evil, &current_confirmed_good,
            &current_evil_roles, &current_good_corr,
            &current_slayer, &current_pd,
            &current_night_kills, current_nk_evil_count,
            &current_reveal_order);

        let result = solve(&state);
        last_surviving = result.n_surviving;

        if !truth_in_set(&result, &true_evil_set, &current_executed) {
            eprintln!("FAIL {case_name}: truth eliminated after exec #{pos} ({} surviving)", result.n_surviving);
            return (false, result.n_surviving);
        }
        for &ep in &result.definite_evil {
            if !true_evil_set.contains(&ep) {
                eprintln!("FAIL {case_name}: false definite_evil #{ep} after exec #{pos}");
                return (false, result.n_surviving);
            }
        }
        for &gp in &result.definite_good {
            if true_evil_set.contains(&gp) {
                eprintln!("FAIL {case_name}: false definite_good #{gp} after exec #{pos}");
                return (false, result.n_surviving);
            }
        }
    }

    (true, last_surviving)
}

#[test]
fn replay_all_v2() {
    let dir = v2_dir();
    assert!(dir.exists(), "v2 test cases dir not found: {:?}", dir);

    let mut files: Vec<_> = glob::glob(dir.join("*.json").to_str().unwrap()).unwrap()
        .filter_map(|e| e.ok())
        .collect();
    files.sort_by_key(|p| p.file_name().unwrap().to_str().unwrap().to_string());

    let mut passed = 0;
    let mut failed = 0;
    let mut failures: Vec<String> = Vec::new();

    for path in &files {
        let name = path.file_name().unwrap().to_str().unwrap();
        let content = std::fs::read_to_string(path).unwrap();
        let value: serde_json::Value = serde_json::from_str(&content).unwrap();

        let (ok, _surviving) = replay_game(&value);
        if ok {
            passed += 1;
        } else {
            failed += 1;
            failures.push(name.to_string());
        }
    }

    println!("\nReplay results: {passed} passed, {failed} failed, {} total", passed + failed);
    if !failures.is_empty() {
        println!("Failures:");
        for f in &failures {
            println!("  - {f}");
        }
    }
    assert_eq!(failed, 0, "{failed} tests failed: {:?}", failures);
}
