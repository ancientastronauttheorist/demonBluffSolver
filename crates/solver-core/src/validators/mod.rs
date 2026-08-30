/// Card info validators — check if a card's claimed info is consistent with a scenario.

mod helpers;
pub use helpers::*;

use std::collections::{HashMap, HashSet};
use crate::geometry::{circle_distance, circle_direction, adjacent_positions};
use crate::knowledge_base::{self, get_card, normalize_role, Faction};
use crate::types::{CardInfo, GameState, Scenario};

/// Validate a single card's info against a scenario using the appropriate role validator.
pub fn validate_card(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let role = normalize_role(&card.apparent_role);
    match role.as_str() {
        "enlightened" => validate_enlightened(card, scenario, state),
        "knitter" => validate_knitter(card, scenario, state),
        "confessor" => validate_confessor(card, scenario, state),
        "gemcrafter" => validate_gemcrafter(card, scenario, state),
        "lover" => validate_lover(card, scenario, state),
        "scout" => validate_scout(card, scenario, state),
        "bard" => validate_bard(card, scenario, state),
        "fortuneteller" => validate_fortune_teller(card, scenario, state),
        "oracle" => validate_oracle(card, scenario, state),
        "medium" => validate_medium(card, scenario, state),
        "hunter" => validate_hunter(card, scenario, state),
        "architect" => validate_architect(card, scenario, state),
        "empress" => validate_empress(card, scenario, state),
        "witness" => validate_witness(card, scenario, state),
        "jester" => validate_jester(card, scenario, state),
        "dreamer" => validate_dreamer(card, scenario, state),
        "judge" => validate_judge(card, scenario, state),
        "alchemist" => validate_alchemist(card, scenario, state),
        "druid" => validate_druid(card, scenario, state),
        "bishop" => validate_bishop(card, scenario, state),
        "bountyhunter" => validate_bounty_hunter(card, scenario, state),
        "baker" => validate_baker(card, scenario, state),
        "poet" => validate_poet(card, scenario, state),
        "rambler" => validate_rambler(card, scenario, state),
        _ => true,
    }
}

fn matches_executed_good_corruption(
    scenario: &Scenario,
    pos: u8,
    was_corrupted: bool,
) -> bool {
    // Execution bookkeeping reports Drunk clean even when its active
    // Corrupted status still affects role hooks. Plague Doctor does not share
    // this projection: its native callback reads the active status directly.
    if scenario.drunk_position == Some(pos)
        || (scenario.chancellor_added_outcast_position() == Some(pos)
            && scenario
                .chancellor_added_outcast_role()
                .is_some_and(|role| normalize_role(role) == "drunk"))
    {
        return !was_corrupted;
    }
    scenario.corrupted.contains(&pos) == was_corrupted
}

fn matches_executed_good_role(
    scenario: &Scenario,
    state: &GameState,
    pos: u8,
    observed_role: &str,
) -> bool {
    effective_role_at(pos, scenario, state)
        .as_deref()
        .is_some_and(|role| roles_equal(role, observed_role))
}

/// Check if all revealed cards + ability results + structural constraints are consistent.
pub fn check_scenario(scenario: &Scenario, state: &GameState) -> bool {
    // Check observed corruption status of executed good cards.
    for (&pos, &was_corrupted) in &state.executed_good_corrupted {
        if !matches_executed_good_corruption(scenario, pos, was_corrupted) {
            return false;
        }
    }
    for (&pos, observed_role) in &state.executed_good_roles {
        if !matches_executed_good_role(scenario, state, pos, observed_role) {
            return false;
        }
    }

    // Structural: role counts
    if !validate_role_counts(scenario, state) { return false; }

    // Patch 2026-05-05 Rambler redesign: adjacent truthful characters say
    // "#R shut up!" instead of giving their own clue.
    if !validate_rambler_shut_ups(scenario, state) { return false; }

    // Card info validators
    for card in &state.cards {
        if state.executed.contains(&card.position) {
            if state.confirmed_evil.contains(&card.position)
                && !state.executed_evil_roles.contains_key(&card.position)
            {
                continue; // Skip unknown executed evils
            }
        }
        if !validate_card(card, scenario, state) {
            return false;
        }
    }

    // Slayer results
    if !validate_slayer_results(scenario, state) { return false; }

    // PD ability results
    if !validate_pd_ability(scenario, state) { return false; }

    // Lilis night kill constraint
    if !state.night_kills.is_empty() {
        let evil_in_nk = state.night_kills.iter()
            .filter(|p| scenario.evil_positions.contains_key(p))
            .count() as u8;
        if evil_in_nk != state.night_kill_evil_count { return false; }

        // Knight immunity check
        if state.deck.villagers.iter().any(|v| v == "Knight") {
            let knight_revealed = state.cards.iter().any(|card| {
                !scenario.evil_positions.contains_key(&card.position)
                    && effective_role_at(card.position, scenario, state)
                        .is_some_and(|role| normalize_role(&role) == "knight")
            });
            if !knight_revealed {
                let revealed: HashSet<u8> = state.cards.iter().map(|c| c.position).collect();
                let valid = (1..=state.n_cards).any(|p| {
                    let intrinsically_killable = effective_role_at(p, scenario, state)
                        .is_some_and(|role| normalize_role(&role) == "drunk");
                    !scenario.evil_positions.contains_key(&p)
                        && !revealed.contains(&p)
                        && !(state.night_kills.contains(&p)
                            && !scenario.corrupted.contains(&p)
                            && !intrinsically_killable)
                });
                let pool_gt_board = state.board_villager_count
                    .map_or(false, |bvc| state.deck.villagers.len() as u8 > bvc);
                if !valid && !pool_gt_board { return false; }
            }
        }
    }

    true
}

// ── Individual validators ──

fn validate_enlightened(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let raw = match info_str(&card.info_parsed, "direction") {
        Some(s) => s,
        None => return true,
    };
    let claimed = match raw.to_lowercase().as_str() {
        "cw" => "CW", "ccw" => "CCW",
        "equidistant" | "equal" => "Equidistant",
        _ => raw,
    };
    let pos = card.position;
    let n = state.n_cards;
    let truth = truth_status(pos, scenario, state);

    let evil_positions: Vec<u8> = (1..=n)
        .filter(|&p| effective_alignment(p, scenario, state) == EffectiveAlignment::Evil && p != pos)
        .collect();
    if evil_positions.is_empty() { return true; }

    let min_dist = evil_positions.iter().map(|&ep| circle_distance(pos, ep, n)).min().unwrap();
    let closest: Vec<u8> = evil_positions.iter()
        .filter(|&&ep| circle_distance(pos, ep, n) == min_dist)
        .copied().collect();

    let real_answer = if closest.len() >= 2 {
        let dirs: HashSet<&str> = closest.iter()
            .map(|&ep| circle_direction(pos, ep, n).as_str())
            .collect();
        if dirs.contains("CW") && dirs.contains("CCW") { "Equidistant" }
        else { circle_direction(pos, closest[0], n).as_str() }
    } else {
        circle_direction(pos, closest[0], n).as_str()
    };

    if truth == TruthStatus::Truthful { claimed == real_answer }
    else { claimed != real_answer }
}

fn validate_knitter(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let claimed = match info_i64(&card.info_parsed, "evil_pairs") {
        Some(v) => v,
        None => return true,
    };
    let n = state.n_cards;
    let truth = truth_status(card.position, scenario, state);

    let evil_set: HashSet<u8> = (1..=n)
        .filter(|&p| effective_alignment(p, scenario, state) == EffectiveAlignment::Evil)
        .collect();
    let mut pairs = 0i64;
    for &p in &evil_set {
        for adj in adjacent_positions(p, n) {
            if evil_set.contains(&adj) && adj > p { pairs += 1; }
        }
    }

    if truth == TruthStatus::Truthful { claimed == pairs }
    else { claimed != pairs }
}

fn validate_confessor(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let claimed_dizzy = info_bool(&card.info_parsed, "dizzy")
        .or_else(|| info_bool(&card.info_parsed, "dirty"));
    let claimed_dizzy = match claimed_dizzy {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let is_evil = is_evil_in_board_state(pos, scenario, state);
    let is_corrupted = scenario.corrupted.contains(&pos);
    let actually_dizzy = is_evil || is_corrupted;
    claimed_dizzy == actually_dizzy
}

fn validate_gemcrafter(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let claimed_pos = match info_pos(&card.info_parsed, "good_position") {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let truth = truth_status(pos, scenario, state);
    let actual_good = effective_alignment(claimed_pos, scenario, state) == EffectiveAlignment::Good;

    if truth == TruthStatus::Truthful {
        if !actual_good { return false; }
        if claimed_pos == pos {
            let other_good = (1..=state.n_cards).any(|p|
                p != pos && effective_alignment(p, scenario, state) == EffectiveAlignment::Good);
            if other_good { return false; }
        }
        true
    } else {
        !actual_good
    }
}

fn validate_lover(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let claimed = match info_i64(&card.info_parsed, "evil_adjacent") {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let n = state.n_cards;
    let truth = truth_status(pos, scenario, state);
    let actual: i64 = adjacent_positions(pos, n).iter()
        .filter(|&&a| effective_alignment(a, scenario, state) == EffectiveAlignment::Evil)
        .count() as i64;

    if truth == TruthStatus::Truthful { claimed == actual }
    else { claimed != actual }
}

fn validate_scout(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let evil_role = match info_str(&card.info_parsed, "evil_role") {
        Some(s) => s,
        None => return true,
    };
    let claimed_dist = match info_i64(&card.info_parsed, "distance") {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let n = state.n_cards;
    let truth = truth_status(pos, scenario, state);

    let target_pos = (1..=n).find(|&p| {
        known_evil_role(p, scenario, state).map_or(false, |r| roles_equal(r, evil_role))
    });
    let target_pos = match target_pos { Some(p) => p, None => return true };

    let other_evil: Vec<u8> = (1..=n)
        .filter(|&p| p != target_pos && effective_alignment(p, scenario, state) == EffectiveAlignment::Evil)
        .collect();
    if other_evil.is_empty() { return true; }

    let actual = other_evil.iter().map(|&ep| circle_distance(target_pos, ep, n) as i64).min().unwrap();

    if truth == TruthStatus::Truthful { claimed_dist == actual }
    else { claimed_dist != actual }
}

fn validate_bard(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let claimed = match info_i64(&card.info_parsed, "corruption_distance") {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let n = state.n_cards;
    let truth = truth_status(pos, scenario, state);
    let no_corrupted = scenario.corrupted.is_empty();

    if claimed == -1 {
        return if truth == TruthStatus::Truthful { no_corrupted } else { !no_corrupted };
    }
    if no_corrupted {
        return if truth == TruthStatus::Truthful { false } else { true };
    }
    let actual = scenario.corrupted.iter()
        .map(|&c| circle_distance(pos, c, n) as i64)
        .min().unwrap();

    if truth == TruthStatus::Truthful { claimed == actual }
    else { claimed != actual }
}

fn validate_fortune_teller(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let targets = match info_targets(&card.info_parsed, "targets") {
        Some(t) => t,
        None => return true,
    };
    let claimed_evil = match info_bool(&card.info_parsed, "has_evil") {
        Some(v) => v,
        None => return true,
    };
    let truth = truth_status(card.position, scenario, state);
    let actual_evil = targets.iter().any(|&t| effective_alignment(t, scenario, state) == EffectiveAlignment::Evil);

    if truth == TruthStatus::Truthful { claimed_evil == actual_evil }
    else { claimed_evil != actual_evil }
}

fn validate_oracle(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let targets = match info_targets(&card.info_parsed, "targets") {
        Some(t) => t,
        None => return true,
    };
    let minion_role = match info_str(&card.info_parsed, "minion_role") {
        Some(s) => s,
        None => return true,
    };
    let truth = truth_status(card.position, scenario, state);

    let target_matches_definite = |t: u8| -> bool {
        known_evil_role(t, scenario, state).map_or(false, |r| roles_equal(r, minion_role))
    };
    let target_matches_possible = |t: u8| -> bool {
        if target_matches_definite(t) { return true; }
        if effective_role_at(t, scenario, state)
            .is_some_and(|role| roles_equal(&role, "Wretch"))
        {
            return get_card(minion_role).map_or(false, |cd| cd.faction == Faction::Minion);
        }
        false
    };

    if truth == TruthStatus::Truthful {
        targets.iter().any(|&t| target_matches_possible(t))
    } else {
        // Lying Oracle: both targets must be Good (skip for Poet)
        if card.apparent_role != "Poet" {
            if !targets.iter().all(|&t| !is_evil_in_board_state(t, scenario, state)) {
                return false;
            }
        }
        let actual_definite = targets.iter().any(|&t| target_matches_definite(t));
        !actual_definite
    }
}

fn validate_medium(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let claimed_pos = match info_pos(&card.info_parsed, "good_position") {
        Some(v) => v,
        None => return true,
    };
    let claimed_role = match info_str(&card.info_parsed, "good_role") {
        Some(s) => s,
        None => return true,
    };
    let pos = card.position;
    let truth = truth_status(pos, scenario, state);

    let is_good = effective_alignment(claimed_pos, scenario, state) == EffectiveAlignment::Good;
    let actual_role = get_real_role(claimed_pos, scenario, state);
    let mut actual_match = is_good && actual_role.replace(' ', "_") == claimed_role.replace(' ', "_");

    // Baker conversion: Medium may see original role
    if !actual_match && is_good && actual_role == "Baker" {
        if let Some(target_card) = state.card_at(claimed_pos) {
            if let Some(orig) = info_str(&target_card.info_parsed, "original_role") {
                if orig.replace(' ', "_") == claimed_role.replace(' ', "_") {
                    actual_match = true;
                }
            }
        }
    }
    // Night-killed: actual_role is "Unknown", accept valid good role
    if actual_role == "Unknown" && is_good {
        let norm_claimed = claimed_role.replace(' ', "_");
        let valid_good: Vec<String> = state.deck.villagers.iter()
            .chain(state.deck.outcasts.iter())
            .map(|r| r.replace(' ', "_"))
            .collect();
        if valid_good.contains(&norm_claimed) {
            actual_match = true;
        }
    }

    if truth == TruthStatus::Truthful {
        actual_match
    } else {
        let target_is_evil = is_evil_in_board_state(claimed_pos, scenario, state);
        let target_is_drunk = scenario.drunk_position == Some(claimed_pos);
        let target_is_dopp = scenario.doppelganger_position == Some(claimed_pos);
        if !(target_is_evil || target_is_drunk || target_is_dopp) { return false; }
        !actual_match
    }
}

fn validate_hunter(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let claimed = match info_i64(&card.info_parsed, "distance") {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let n = state.n_cards;
    let truth = truth_status(pos, scenario, state);

    let evil_positions: Vec<u8> = (1..=n)
        .filter(|&p| p != pos && effective_alignment(p, scenario, state) == EffectiveAlignment::Evil)
        .collect();
    if evil_positions.is_empty() { return true; }

    let actual = evil_positions.iter().map(|&ep| circle_distance(pos, ep, n) as i64).min().unwrap();

    if truth == TruthStatus::Truthful { claimed == actual }
    else { claimed != actual }
}

fn validate_architect(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let raw = match info_str(&card.info_parsed, "side") {
        Some(s) => s,
        None => return true,
    };
    let claimed = match raw.to_lowercase().as_str() {
        "left" => "Left", "right" => "Right", "equal" => "Equal",
        "cw" => "Left", "ccw" => "Right",
        _ => raw,
    };
    let n = state.n_cards;
    let truth = truth_status(card.position, scenario, state);

    let (left_set, right_set, both_set) = architect_sides(n);
    let mut left_count = 0i32;
    let mut right_count = 0i32;
    for p in 1..=n {
        if effective_alignment(p, scenario, state) == EffectiveAlignment::Evil {
            if both_set.contains(&p) {
                left_count += 1;
                right_count += 1;
            } else if left_set.contains(&p) {
                left_count += 1;
            } else if right_set.contains(&p) {
                right_count += 1;
            }
        }
    }
    let actual = if left_count > right_count { "Left" }
        else if right_count > left_count { "Right" }
        else { "Equal" };

    if truth == TruthStatus::Truthful { claimed == actual }
    else { claimed != actual }
}

fn architect_sides(n: u8) -> (HashSet<u8>, HashSet<u8>, HashSet<u8>) {
    let half = n / 2;
    let mut both = HashSet::new();
    both.insert(n); // Top center
    if n % 2 == 0 { both.insert(half); } // Bottom center for even

    let mut right = HashSet::new();
    let right_end = if n % 2 == 0 { half - 1 } else { half };
    for p in 1..=right_end { right.insert(p); }

    let mut left = HashSet::new();
    let left_start = if n % 2 == 0 { half + 1 } else { half + 1 };
    for p in left_start..n { left.insert(p); }

    (left, right, both)
}

fn validate_empress(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let targets = match info_targets(&card.info_parsed, "targets") {
        Some(t) => t,
        None => return true,
    };
    let pos = card.position;
    let truth = truth_status(pos, scenario, state);

    // Self-targeting blocked unless Puppet
    if targets.contains(&pos) && scenario.puppet_position != Some(pos) { return false; }

    let evil_count: i64 = targets.iter()
        .filter(|&&t| effective_alignment(t, scenario, state) == EffectiveAlignment::Evil)
        .count() as i64;

    if truth == TruthStatus::Truthful { evil_count == 1 }
    else { evil_count != 1 }
}

fn validate_witness(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let claimed_pos = match info_pos(&card.info_parsed, "affected_position") {
        Some(v) => v,
        None => return true,
    };
    let truth = truth_status(card.position, scenario, state);

    // `MessedUpByEvil` is its own persistent native status. Alchemist can cure
    // Corrupted without removing this marker, and Chancellor adds it to the
    // selected Outcast anchor without adding Corrupted at all.
    let mut affected = scenario.messed_up_by_evil.clone();
    for &nk in &state.night_kills { affected.insert(nk); }

    if claimed_pos == 0 {
        if truth == TruthStatus::Truthful { affected.is_empty() }
        else { !affected.is_empty() }
    } else {
        let actually_affected = affected.contains(&claimed_pos);
        if truth == TruthStatus::Truthful { actually_affected }
        else { !actually_affected }
    }
}

fn validate_jester(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    // Silenced Jester (e.g. by Rambler): ability fired but no evil_count was
    // emitted. Explicitly a no-op constraint — previously this case arrived
    // here via a missing `evil_count` key and returned true by accident, which
    // silently turned a stale/empty Jester into "any placement is fine" and
    // mis-sized the scenario space (asc78_v6 halt).
    if info_bool(&card.info_parsed, "silenced") == Some(true) {
        return true;
    }
    let targets = match info_targets(&card.info_parsed, "targets") {
        Some(t) => t,
        None => return true,
    };
    let claimed = match info_i64(&card.info_parsed, "evil_count") {
        Some(v) => v,
        None => return true,
    };
    let truth = truth_status(card.position, scenario, state);
    let actual: i64 = targets.iter()
        .filter(|&&t| effective_alignment(t, scenario, state) == EffectiveAlignment::Evil)
        .count() as i64;

    if truth == TruthStatus::Truthful { claimed == actual }
    else { claimed != actual }
}

#[derive(Debug)]
enum DreamerObservation {
    RolePair {
        targets: [u8; 2],
        options: [String; 2],
        current_build: bool,
    },
    Cabbage {
        targets: [u8; 2],
    },
}

#[derive(Debug)]
struct DreamerIdentity {
    real: Option<String>,
    bluff: Option<String>,
}

/// Parse strict public-Dreamer observation shapes. Unlike the shared
/// `info_targets` / `info_str_array` helpers, this parser must not silently
/// discard malformed elements: the native picker always records two integer
/// targets, and the role-pair formatter always records two distinct, non-empty
/// role names. A `public_current` marker opts role pairs into the build-pinned
/// native validator; archived unmarked pairs retain their historical model.
fn parse_dreamer_observation(
    info: &serde_json::Map<String, serde_json::Value>,
) -> Result<Option<DreamerObservation>, ()> {
    let has_targets = info.contains_key("targets");
    let has_options = info.contains_key("evil_role_options");
    let has_cabbage = info.contains_key("cabbage");
    let current_build = match info.get("dreamer_variant") {
        None => false,
        Some(serde_json::Value::String(variant)) if variant == "public_current" => true,
        Some(_) => return Err(()),
    };

    if !has_targets && !has_options && !has_cabbage {
        return if current_build { Err(()) } else { Ok(None) };
    }
    if !has_targets || has_options == has_cabbage {
        return Err(());
    }

    let raw_targets = info
        .get("targets")
        .and_then(|value| value.as_array())
        .ok_or(())?;
    if raw_targets.len() != 2 {
        return Err(());
    }
    let mut targets = [0_u8; 2];
    for (index, value) in raw_targets.iter().enumerate() {
        let target = value
            .as_u64()
            .and_then(|value| u8::try_from(value).ok())
            .ok_or(())?;
        if target == 0 {
            return Err(());
        }
        targets[index] = target;
    }

    if has_cabbage {
        if info.get("cabbage").and_then(|value| value.as_bool()) != Some(true) {
            return Err(());
        }
        return Ok(Some(DreamerObservation::Cabbage { targets }));
    }

    let raw_options = info
        .get("evil_role_options")
        .and_then(|value| value.as_array())
        .ok_or(())?;
    if raw_options.len() != 2 {
        return Err(());
    }
    let mut options = [String::new(), String::new()];
    for (index, value) in raw_options.iter().enumerate() {
        let option = value
            .as_str()
            .map(str::trim)
            .filter(|role| !role.is_empty())
            .ok_or(())?;
        options[index] = option.to_string();
    }
    if roles_equal(&options[0], &options[1]) {
        return Err(());
    }

    Ok(Some(DreamerObservation::RolePair {
        targets,
        options,
        current_build,
    }))
}

fn dreamer_concrete_role(role: &str) -> Option<String> {
    let role = role.trim();
    if role.is_empty() || normalize_role(role) == "unknown" {
        None
    } else {
        Some(role.to_string())
    }
}

fn dreamer_identity_at(pos: u8, scenario: &Scenario, state: &GameState) -> DreamerIdentity {
    let real = effective_role_at(pos, scenario, state)
        .as_deref()
        .and_then(dreamer_concrete_role);
    let apparent = state
        .card_at(pos)
        .and_then(|card| dreamer_concrete_role(&card.apparent_role));
    let bluff = apparent.filter(|apparent| {
        real.as_deref()
            .map(|real| !roles_equal(apparent, real))
            // A known apparent identity at a position whose real identity is
            // unresolved is still useful positive bluff evidence. The board
            // is marked incomplete below, so it cannot cause a hard reject.
            .unwrap_or(true)
    });

    DreamerIdentity { real, bluff }
}

fn dreamer_board_projection(
    scenario: &Scenario,
    state: &GameState,
) -> (Vec<DreamerIdentity>, bool) {
    let mut entries = Vec::with_capacity(state.n_cards as usize);
    let mut complete = state.n_cards > 0;
    for pos in 1..=state.n_cards {
        let identity = dreamer_identity_at(pos, scenario, state);
        complete &= state.card_at(pos).is_some() && identity.real.is_some();
        entries.push(identity);
    }
    (entries, complete)
}

fn dreamer_push_unique(roles: &mut Vec<String>, role: &str) {
    if !roles.iter().any(|known| roles_equal(known, role)) {
        roles.push(role.to_string());
    }
}

fn dreamer_pair_matches(options: &[String; 2], first: &str, second: &str) -> bool {
    (roles_equal(&options[0], first) && roles_equal(&options[1], second))
        || (roles_equal(&options[0], second) && roles_equal(&options[1], first))
}

fn dreamer_legacy_pair_has_underlying_match(
    targets: &[u8; 2],
    options: &[String; 2],
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    targets.iter().any(|&target| {
        effective_role_at(target, scenario, state).is_some_and(|role| {
            options
                .iter()
                .any(|option| roles_equal(option, &role))
        })
    })
}

fn dreamer_targets_include_wretch(
    targets: &[u8; 2],
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    targets.iter().any(|&target| {
        effective_role_at(target, scenario, state).is_some_and(|role| roles_equal(&role, "Wretch"))
    })
}

fn dreamer_truthful_pair_supported(
    targets: &[u8; 2],
    options: &[String; 2],
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    // Real Wretch takes the native Cabbage short-circuit before role-pair
    // generation. A truthful Wretch observation can therefore never be a
    // normal pair.
    if dreamer_targets_include_wretch(targets, scenario, state) {
        return false;
    }

    let target_identities = [
        dreamer_identity_at(targets[0], scenario, state),
        dreamer_identity_at(targets[1], scenario, state),
    ];
    let (board, board_complete) = dreamer_board_projection(scenario, state);

    // Build f530404b0f3f_807de4a83df4 has 46 CharacterData assets and every
    // `usuallyDisguised` flag is false. The native authored-role pool between
    // the target-bluff and board-entry branches is unreachable in this build.
    for anchor_index in 0..2 {
        let other_index = 1 - anchor_index;
        let Some(anchor_real) = target_identities[anchor_index].real.as_deref() else {
            continue;
        };

        // The other selected character's bluff wins whenever it is distinct
        // from the real anchor. Only when that candidate is absent/colliding
        // does native code fall through to a board entry.
        if let Some(other_bluff) = target_identities[other_index]
            .bluff
            .as_deref()
            .filter(|bluff| !roles_equal(bluff, anchor_real))
        {
            if dreamer_pair_matches(options, anchor_real, other_bluff) {
                return true;
            }
            continue;
        }

        for entry in &board {
            let Some(entry_real) = entry.real.as_deref() else {
                continue;
            };
            if roles_equal(entry_real, anchor_real)
                || entry
                    .bluff
                    .as_deref()
                    .is_some_and(|bluff| roles_equal(bluff, anchor_real))
            {
                continue;
            }
            if dreamer_pair_matches(options, anchor_real, entry_real) {
                return true;
            }
        }
    }

    if board_complete {
        return false;
    }

    // Hidden/blocked cards leave `Gameplay.CurrentCharacters` richer than the
    // solver projection. Preserve an unknown target as a possible anchor. For
    // a known anchor, however, preserve native priority: an already-known,
    // distinct bluff on the other target fixes the second role and forbids an
    // unseen board fallback after that exact pair has failed.
    for anchor_index in 0..2 {
        let other_index = 1 - anchor_index;
        let Some(anchor_real) = target_identities[anchor_index].real.as_deref() else {
            return true;
        };
        if !options
            .iter()
            .any(|option| roles_equal(option, anchor_real))
        {
            continue;
        }
        let has_priority_bluff = target_identities[other_index]
            .bluff
            .as_deref()
            .is_some_and(|bluff| !roles_equal(bluff, anchor_real));
        if !has_priority_bluff {
            return true;
        }
    }
    false
}

fn dreamer_liar_pair_supported(
    targets: &[u8; 2],
    options: &[String; 2],
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let target_identities = [
        dreamer_identity_at(targets[0], scenario, state),
        dreamer_identity_at(targets[1], scenario, state),
    ];
    let (board, board_complete) = dreamer_board_projection(scenario, state);

    // Native starts with the two targets' bluff identities, preserving only
    // the first occurrence of each CharacterData identity. It does not reject
    // a bluff merely because it equals the *other* target's real identity.
    let mut initial_bluffs = Vec::new();
    for identity in &target_identities {
        if let Some(bluff) = identity.bluff.as_deref() {
            dreamer_push_unique(&mut initial_bluffs, bluff);
        }
    }

    // See the build-pinned `usuallyDisguised` note in the truthful path. With
    // that authored pool empty, every missing output comes from the helper.
    let mut excluded = Vec::new();
    for identity in &target_identities {
        if let Some(real) = identity.real.as_deref() {
            dreamer_push_unique(&mut excluded, real);
        }
        if let Some(bluff) = identity.bluff.as_deref() {
            dreamer_push_unique(&mut excluded, bluff);
        }
    }

    let mut helper_pool = Vec::new();
    for entry in &board {
        for role in [entry.real.as_deref(), entry.bluff.as_deref()]
            .into_iter()
            .flatten()
        {
            if !excluded.iter().any(|known| roles_equal(known, role)) {
                dreamer_push_unique(&mut helper_pool, role);
            }
        }
    }

    let exact_support = match initial_bluffs.as_slice() {
        [first, second] => dreamer_pair_matches(options, first, second),
        [initial] => helper_pool
            .iter()
            .any(|helper| dreamer_pair_matches(options, initial, helper)),
        [] => helper_pool.iter().enumerate().any(|(index, first)| {
            helper_pool
                .iter()
                .skip(index + 1)
                .any(|second| dreamer_pair_matches(options, first, second))
        }),
        _ => false,
    };
    if exact_support || board_complete {
        return exact_support;
    }

    // Conservative incomplete-board fallback. Known initial bluffs are
    // mandatory. Any remaining option may be supplied by an unseen helper
    // identity, but the helper excludes both selected real and bluff
    // identities. This keeps the native cross-target bluff/real collision
    // reachable without admitting an arbitrary match to a selected real role.
    if initial_bluffs.len() == 2 {
        return dreamer_pair_matches(options, &initial_bluffs[0], &initial_bluffs[1]);
    }
    if !initial_bluffs
        .iter()
        .all(|initial| options.iter().any(|option| roles_equal(option, initial)))
    {
        return false;
    }
    options.iter().all(|option| {
        initial_bluffs
            .iter()
            .any(|initial| roles_equal(option, initial))
            || !excluded.iter().any(|role| roles_equal(option, role))
    })
}

fn validate_dreamer(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    match parse_dreamer_observation(&card.info_parsed) {
        Err(()) => return false,
        Ok(Some(DreamerObservation::Cabbage { targets })) => {
            if targets.iter().any(|&target| target > state.n_cards) {
                return false;
            }
            return truth_status(card.position, scenario, state) == TruthStatus::Truthful
                && dreamer_targets_include_wretch(&targets, scenario, state);
        }
        Ok(Some(DreamerObservation::RolePair {
            targets,
            options,
            current_build,
        })) => {
            if targets.iter().any(|&target| target > state.n_cards) {
                return false;
            }
            let truth = truth_status(card.position, scenario, state);
            if !current_build {
                let any_match =
                    dreamer_legacy_pair_has_underlying_match(&targets, &options, scenario, state);
                return match truth {
                    TruthStatus::Truthful => any_match,
                    TruthStatus::Lying => !any_match,
                };
            }
            return match truth {
                TruthStatus::Truthful => {
                    dreamer_truthful_pair_supported(&targets, &options, scenario, state)
                }
                TruthStatus::Lying => {
                    dreamer_liar_pair_supported(&targets, &options, scenario, state)
                }
            };
        }
        Ok(None) => {}
    }

    // Shape 1 (original Dreamer1): {"target": pos, "evil_role": role}
    let target = match info_pos(&card.info_parsed, "target") {
        Some(v) => v,
        None => return true,
    };
    let claimed_role = match info_str(&card.info_parsed, "evil_role") {
        Some(s) => s,
        None => return true,
    };
    let truth = truth_status(card.position, scenario, state);
    let target_is_evil = is_evil_in_board_state(target, scenario, state);
    let target_is_wretch =
        effective_role_at(target, scenario, state).is_some_and(|role| roles_equal(&role, "Wretch"));

    // Gather all known evil roles on the board
    let evil_roles: Vec<&str> = (1..=state.n_cards)
        .filter_map(|p| {
            let r = known_evil_role(p, scenario, state)?;
            if r == "Puppet" || r == "Unknown" {
                None
            } else {
                Some(r)
            }
        })
        .collect();

    if truth == TruthStatus::Truthful {
        if target_is_wretch {
            return claimed_role.eq_ignore_ascii_case("cabbage");
        }
        if target_is_evil {
            let actual = known_evil_role(target, scenario, state).unwrap_or("");
            return roles_equal(claimed_role, actual);
        }
        evil_roles.iter().any(|r| roles_equal(r, claimed_role))
    } else {
        if target_is_wretch {
            return !claimed_role.eq_ignore_ascii_case("cabbage");
        }
        if target_is_evil {
            let actual = known_evil_role(target, scenario, state).unwrap_or("");
            return !roles_equal(claimed_role, actual);
        }
        true
    }
}

fn validate_judge(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let target = match info_pos(&card.info_parsed, "target") {
        Some(v) => v,
        None => return true,
    };
    let claimed_lying = match info_bool(&card.info_parsed, "is_lying") {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let truth = truth_status(pos, scenario, state);

    // Corrupted Judge's ability doesn't work reliably
    if truth == TruthStatus::Lying && !is_evil_in_board_state(pos, scenario, state) {
        // Corrupted (not evil) Judge — skip validation
        return true;
    }

    let target_truth = truth_status(target, scenario, state);
    let actually_lying = target_truth == TruthStatus::Lying;

    if truth == TruthStatus::Truthful { claimed_lying == actually_lying }
    else { claimed_lying != actually_lying }
}

fn validate_alchemist(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    // Post-patch: the clue is "# of Corrupted characters around me [Range 2] at
    // the start of the Round (before the Cure)", stored as info_parsed["corrupted_count"].
    // Legacy test cases use "cured_count" — accept both during migration.
    let claimed = match info_i64(&card.info_parsed, "corrupted_count")
        .or_else(|| info_i64(&card.info_parsed, "cured_count"))
    {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let truth = truth_status(pos, scenario, state);
    // A visible Alchemist can be an evil/Drunk/Doppelganger/Puppet bluff. A
    // missing native Start actor has no real cure counter; zero would invent a
    // constraint and discard valid disguise worlds.
    let actual = match scenario.alchemist_cures.get(&pos).copied() {
        Some(value) => value as i64,
        None => return true,
    };

    if truth == TruthStatus::Truthful { claimed == actual }
    else { claimed != actual }
}

fn validate_druid(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let targets = match info_targets(&card.info_parsed, "targets") {
        Some(t) => t,
        None => return true,
    };
    let found_outcast = card.info_parsed.get("found_outcast")
        .and_then(|v| v.as_str());
    let truth = truth_status(card.position, scenario, state);

    let mut actual_outcasts: Vec<String> = Vec::new();
    let mut has_unrevealed_good_target = false;

    for &t in &targets {
        if is_evil_in_board_state(t, scenario, state) { continue; }
        if scenario.chancellor_added_outcast_position() == Some(t) {
            if let Some(role) = scenario.chancellor_added_outcast_role() {
                if normalize_role(role) != "wretch" {
                    actual_outcasts.push(role.to_string());
                }
                continue;
            }
        }
        if scenario.doppelganger_position == Some(t) {
            actual_outcasts.push("Doppelganger".to_string());
            continue;
        }
        if scenario.drunk_position == Some(t) {
            actual_outcasts.push("Drunk".to_string());
            continue;
        }
        if let Some(c) = state.card_at(t) {
            if let Some(cd) = get_card(&c.apparent_role) {
                if cd.faction == Faction::Outcast {
                    if c.apparent_role == "Wretch" { continue; } // Wretch registers as Evil
                    actual_outcasts.push(c.apparent_role.clone());
                }
            }
        } else {
            has_unrevealed_good_target = true;
        }
    }

    let has_outcast = !actual_outcasts.is_empty();

    if truth == TruthStatus::Truthful {
        if let Some(fo) = found_outcast {
            // Operator precedence matches Python: (has_outcast && found in list) || unrevealed
            (has_outcast && actual_outcasts.iter().any(|a| roles_equal(a, fo)))
                || has_unrevealed_good_target
        } else {
            !has_outcast
        }
    } else {
        if found_outcast.is_some() {
            !has_outcast
        } else {
            has_outcast || has_unrevealed_good_target
        }
    }
}

fn validate_bishop(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let targets = match info_targets(&card.info_parsed, "targets") {
        Some(t) => t,
        None => return true,
    };
    let claimed_types: Option<Vec<String>> = card.info_parsed.get("types")
        .and_then(|v| v.as_array())
        .map(|arr| arr.iter().filter_map(|v| v.as_str().map(String::from)).collect());
    let truth = truth_status(card.position, scenario, state);

    if let Some(ref ct) = claimed_types {
        // Bishop's clue reflects a snapshot of its 3 targets at game start.
        // Chancellor reinitializes and later reveals the added Outcast role;
        // frozen observations span both its real-data type and its eventual
        // register-as role surface, so retain both timing views.
        let mut sorted_claimed = ct.clone();
        sorted_claimed.sort();

        let try_view = |include_conv: bool| -> Option<bool> {
            let mut actual: Vec<String> = Vec::new();
            for &t in &targets {
                match get_position_type_ex(t, scenario, state, include_conv) {
                    Some(tp) => actual.push(tp.to_string()),
                    None => return None, // Unrevealed target — skip this view
                }
            }
            actual.sort();
            Some(actual == sorted_claimed)
        };

        let post = try_view(true);
        let pre = try_view(false);
        if post.is_some() || pre.is_some() {
            let any_match = post.unwrap_or(false) || pre.unwrap_or(false);
            return if truth == TruthStatus::Truthful { any_match } else { !any_match };
        }
    }

    // Weak fallback
    let has_evil = targets.iter().any(|&t| effective_alignment(t, scenario, state) == EffectiveAlignment::Evil);
    if truth == TruthStatus::Truthful { has_evil } else { true }
}

fn validate_bounty_hunter(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let evil_pos = match info_pos(&card.info_parsed, "evil_position") {
        Some(v) => v,
        None => return true,
    };
    let truth = truth_status(card.position, scenario, state);
    let actually_evil = effective_alignment(evil_pos, scenario, state) == EffectiveAlignment::Evil;

    if truth == TruthStatus::Truthful { actually_evil }
    else { !actually_evil }
}

fn validate_baker(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    if scenario
        .shaman_trace
        .as_ref()
        .is_some_and(|trace| normalize_role(&trace.copied_role) == "baker")
    {
        // InitWithNoReset preserves the destination's prior runtimeData, and
        // the native Baker composition has not yet been recovered. Treat all
        // Baker clue text as opaque in copied-Baker worlds rather than pruning
        // from a synthesized runtime provenance.
        return true;
    }
    let claimed = match info_str(&card.info_parsed, "original_role") {
        Some(s) => s,
        None => return true,
    };
    let pos = card.position;
    let truth = truth_status(pos, scenario, state);

    if claimed.eq_ignore_ascii_case("original") {
        if truth == TruthStatus::Lying {
            // Original Baker is immune to corruption, and corrupted converted
            // Bakers lie about their role but do NOT claim "I am the original Baker".
            // Thus any lying good position (including a resistant Drunk) claiming
            // "original" is impossible.
            if !is_evil_in_board_state(pos, scenario, state) {
                return false;
            }
        }
        return true;
    }

    let claimed_card = match get_card(claimed) {
        Some(c) => c,
        None => return true,
    };
    let is_villager = claimed_card.faction == Faction::Villager;

    if truth == TruthStatus::Truthful {
        let deck_norm: HashSet<String> = state.deck.villagers.iter().map(|v| normalize_role(v)).collect();
        is_villager && deck_norm.contains(&normalize_role(claimed_card.name))
    } else {
        true // Lying Baker can claim anything
    }
}

fn validate_poet(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let copied_role = match info_str(&card.info_parsed, "copied_role") {
        Some(s) => s,
        None => return true,
    };
    let norm = normalize_role(copied_role);
    // Delegate to the copied role's validator
    match norm.as_str() {
        "enlightened" => validate_enlightened(card, scenario, state),
        "knitter" => validate_knitter(card, scenario, state),
        "confessor" => validate_confessor(card, scenario, state),
        "gemcrafter" => validate_gemcrafter(card, scenario, state),
        "lover" => validate_lover(card, scenario, state),
        "scout" => validate_scout(card, scenario, state),
        "bard" => validate_bard(card, scenario, state),
        "fortuneteller" | "fortune teller" | "fortune_teller" => validate_fortune_teller(card, scenario, state),
        "oracle" => validate_oracle(card, scenario, state),
        "medium" => validate_medium(card, scenario, state),
        "hunter" => validate_hunter(card, scenario, state),
        "architect" => validate_architect(card, scenario, state),
        "empress" => validate_empress(card, scenario, state),
        "witness" => validate_witness(card, scenario, state),
        "jester" => validate_jester(card, scenario, state),
        "dreamer" => validate_dreamer(card, scenario, state),
        "judge" => validate_judge(card, scenario, state),
        "alchemist" => validate_alchemist(card, scenario, state),
        "druid" => validate_druid(card, scenario, state),
        "bishop" => validate_bishop(card, scenario, state),
        "bountyhunter" | "bounty hunter" | "bounty_hunter" => validate_bounty_hunter(card, scenario, state),
        "baker" => validate_baker(card, scenario, state),
        _ => true,
    }
}

/// Rambler's passive card validator is a no-op after the redesign.
/// The active constraint is enforced globally by `validate_rambler_shut_ups()`.
fn validate_rambler(_card: &CardInfo, _scenario: &Scenario, _state: &GameState) -> bool {
    true
}

// ── Special ability validators ──

fn is_truthful_rambler_act_surface(pos: u8, scenario: &Scenario, state: &GameState) -> bool {
    if truth_status(pos, scenario, state) != TruthStatus::Truthful {
        return false;
    }

    let underlying_rambler = effective_role_at(pos, scenario, state)
        .map_or(false, |role| normalize_role(&role) == "rambler");
    if underlying_rambler {
        return true;
    }

    let apparent_rambler = state
        .card_at(pos)
        .map(|card| normalize_role(&card.apparent_role) == "rambler")
        .unwrap_or(false);
    let modeled_healthy_bluff_holder = scenario.puppet_position == Some(pos)
        || scenario.doppelganger_position == Some(pos);

    apparent_rambler && modeled_healthy_bluff_holder
}

fn card_has_normal_clue(card: &CardInfo) -> bool {
    if card.info_parsed.is_empty() || info_pos(&card.info_parsed, "shut_up_target").is_some() {
        return false;
    }
    let role = normalize_role(&card.apparent_role);
    if role == "rambler" {
        return false;
    }
    if role == "jester" && info_bool(&card.info_parsed, "silenced") == Some(true) {
        return false;
    }
    true
}

fn validate_rambler_shut_ups(scenario: &Scenario, state: &GameState) -> bool {
    if !state.cards.iter().any(|card| info_pos(&card.info_parsed, "shut_up_target").is_some()) {
        return true;
    }

    for card in &state.cards {
        let adjacent_ramblers: Vec<u8> = adjacent_positions(card.position, state.n_cards)
            .into_iter()
            .filter(|&p| is_truthful_rambler_act_surface(p, scenario, state))
            .collect();
        let truthful =
            truth_appearance_status(card.position, scenario, state) == TruthStatus::Truthful;

        if let Some(target) = info_pos(&card.info_parsed, "shut_up_target") {
            let actual = adjacent_ramblers.contains(&target);
            if truthful {
                if !actual { return false; }
            } else if actual {
                return false;
            }
            continue;
        }

        if truthful && card_has_normal_clue(card) && !adjacent_ramblers.is_empty() {
            return false;
        }
    }
    true
}

fn validate_slayer_results(scenario: &Scenario, state: &GameState) -> bool {
    for result in &state.slayer_results {
        let slayer_pos = result.slayer_pos;
        let target_pos = result.target_pos;
        let killed = result.killed;

        let slayer_evil_role = known_evil_role(slayer_pos, scenario, state);
        let slayer_is_evil = slayer_evil_role.is_some();
        let slayer_is_puppet = slayer_evil_role == Some("Puppet");
        let slayer_lies = truth_status(slayer_pos, scenario, state) == TruthStatus::Lying;
        let target_is_evil = effective_alignment(target_pos, scenario, state) == EffectiveAlignment::Evil;

        if killed {
            if slayer_is_evil && !slayer_is_puppet { return false; }
            if slayer_lies { return false; }
            if !target_is_evil { return false; }
            if let Some(revealed_role) = result.revealed_role.as_deref() {
                let role_matches = effective_role_at(target_pos, scenario, state)
                    .is_some_and(|actual| {
                        normalize_role(&actual) == normalize_role(revealed_role)
                    });
                if !role_matches { return false; }
            }
        } else {
            let slayer_works = (!slayer_is_evil || slayer_is_puppet) && !slayer_lies;
            if slayer_works && target_is_evil { return false; }
        }
    }
    true
}

fn validate_pd_ability(scenario: &Scenario, state: &GameState) -> bool {
    let mut used_pd_actors = HashSet::new();
    for result in &state.pd_ability_results {
        let pd_pos = result.pd_pos;
        let target = result.target;
        let claimed_corrupted = result.is_corrupted;
        let evil_revealed = result.evil_revealed;

        if pd_pos == 0
            || pd_pos > state.n_cards
            || target == 0
            || target > state.n_cards
        {
            return false;
        }
        if !used_pd_actors.insert(pd_pos) {
            // The shipped ability is once-use. Repeated evidence for one
            // apparent actor is malformed rather than a second observation.
            return false;
        }
        let Some(actor) = state.card_at(pd_pos) else {
            return false;
        };
        if !knowledge_base::is_plague_doctor(&actor.apparent_role) {
            // Validate the visible role, not the underlying scenario role:
            // an Evil character bluffing as PD is a legal lying actor.
            return false;
        }

        let pd_lies = truth_status(pd_pos, scenario, state) == TruthStatus::Lying;
        let actual_corrupted = scenario.corrupted.contains(&target);

        // ConjourInfo always formats a self-check as clean, even when the
        // truthful/bluff callback already drew a hidden result pointer.
        let expected_corrupted = target != pd_pos
            && if pd_lies {
                !actual_corrupted
            } else {
                actual_corrupted
            };
        if claimed_corrupted != expected_corrupted {
            return false;
        }

        if !claimed_corrupted {
            // The visible clean branch has no revealed character.
            if evil_revealed.is_some() {
                return false;
            }
            continue;
        }

        let Some(revealed) = evil_revealed else {
            return false;
        };
        if revealed == 0 || revealed > state.n_cards {
            return false;
        }

        // Truth chooses uniformly from runtime/apparent Evil; Bluff chooses
        // uniformly from runtime/apparent Good and falsely labels it Evil.
        let expected_alignment = if pd_lies {
            EffectiveAlignment::Good
        } else {
            EffectiveAlignment::Evil
        };
        if effective_alignment(revealed, scenario, state) != expected_alignment {
            return false;
        }
    }
    true
}

// ── Role count validation ──

fn validate_role_counts(scenario: &Scenario, state: &GameState) -> bool {
    let mut good_villager_counts: HashMap<String, i32> = HashMap::new();
    let mut good_outcast_counts: HashMap<String, i32> = HashMap::new();
    let mut counted_good_villager_positions = HashSet::new();

    for card in &state.cards {
        let pos = card.position;
        if known_evil_role(pos, scenario, state).is_some() { continue; }
        let generated_role = scenario
            .chancellor_added_outcast_position()
            .filter(|position| *position == pos)
            .and_then(|_| scenario.chancellor_added_outcast_role());
        let role = generated_role
            .filter(|role| !matches!(normalize_role(role).as_str(), "drunk" | "doppelganger"))
            .unwrap_or(&card.apparent_role);
        if knowledge_base::is_villager_role(role) {
            *good_villager_counts.entry(normalize_role(role)).or_insert(0) += 1;
            counted_good_villager_positions.insert(pos);
        } else if state.deck.outcasts.iter().any(|o| o == role || o.replace('_', " ") == role) {
            *good_outcast_counts.entry(normalize_role(role)).or_insert(0) += 1;
        }
    }

    let actual_shaman = (1..=state.n_cards).any(|position| {
        known_evil_role(position, scenario, state)
            .is_some_and(|role| normalize_role(role) == "shaman")
    });
    if actual_shaman != scenario.shaman_trace.is_some() {
        return false;
    }

    // Project the final duplicate back through Shaman's one-way overwrite so
    // deck multiplicity is checked against the identities that existed before
    // its Start action. Hidden endpoints are still real selected identities.
    // Solver-equivalent erased roles share one trace, so keep one count map per
    // viable prior identity and accept the trace if any map fits the deck.
    let mut initial_good_villager_count_variants = vec![good_villager_counts.clone()];
    if let Some(trace) = scenario.shaman_trace.as_ref() {
        let copied = normalize_role(&trace.copied_role);
        let previous_roles: Vec<String> = trace
            .target_previous_roles
            .iter()
            .map(|role| normalize_role(role))
            .collect();
        let previous_role_set: HashSet<&str> =
            previous_roles.iter().map(String::as_str).collect();
        let previous_was_alchemist = previous_roles
            .first()
            .is_some_and(|role| role == "alchemist");
        if trace.source_position == trace.target_position
            || trace.source_position == 0
            || trace.target_position == 0
            || trace.source_position > state.n_cards
            || trace.target_position > state.n_cards
            || previous_roles.is_empty()
            || previous_role_set.len() != previous_roles.len()
            || previous_roles
                .iter()
                .any(|role| (role == "alchemist") != previous_was_alchemist)
            || !state
                .deck
                .villagers
                .iter()
                .any(|role| normalize_role(role) == copied)
            || previous_roles.iter().any(|previous| {
                !state
                    .deck
                    .villagers
                    .iter()
                    .any(|role| normalize_role(role) == *previous)
            })
            || known_evil_role(trace.source_position, scenario, state).is_some()
            || known_evil_role(trace.target_position, scenario, state).is_some()
            || scenario.puppet_position == Some(trace.source_position)
            || scenario.puppet_position == Some(trace.target_position)
            || scenario.doppelganger_position == Some(trace.source_position)
            || scenario.doppelganger_position == Some(trace.target_position)
            || scenario.drunk_position == Some(trace.source_position)
            || scenario.drunk_position == Some(trace.target_position)
            || scenario.chancellor_added_outcast_position() == Some(trace.source_position)
            || scenario.chancellor_added_outcast_position() == Some(trace.target_position)
        {
            return false;
        }

        for endpoint in [trace.source_position, trace.target_position] {
            if counted_good_villager_positions.contains(&endpoint) {
                let observed = state
                    .card_at(endpoint)
                    .map(|card| normalize_role(&card.apparent_role));
                if observed.as_deref() != Some(copied.as_str()) {
                    return false;
                }
            }
        }

        let mut base_initial_counts = good_villager_counts.clone();
        if counted_good_villager_positions.contains(&trace.target_position) {
            let Some(count) = base_initial_counts.get_mut(&copied) else {
                return false;
            };
            *count -= 1;
            if *count == 0 {
                base_initial_counts.remove(&copied);
            }
        }
        if !counted_good_villager_positions.contains(&trace.source_position) {
            *base_initial_counts.entry(copied).or_insert(0) += 1;
        }
        initial_good_villager_count_variants = previous_roles
            .into_iter()
            .map(|previous| {
                let mut counts = base_initial_counts.clone();
                *counts.entry(previous).or_insert(0) += 1;
                counts
            })
            .collect();
    }

    // Disguiser count
    let mut disguiser_positions = HashSet::new();
    if let Some(dp) = scenario.doppelganger_position {
        if !scenario.evil_positions.contains_key(&dp) { disguiser_positions.insert(dp); }
    }
    if let Some(dp) = scenario.drunk_position {
        if !scenario.evil_positions.contains_key(&dp) { disguiser_positions.insert(dp); }
    }
    if let (Some(position), Some(role)) = (
        scenario.chancellor_added_outcast_position(),
        scenario.chancellor_added_outcast_role(),
    ) {
        if matches!(normalize_role(role).as_str(), "drunk" | "doppelganger")
            && !scenario.evil_positions.contains_key(&position)
        {
            disguiser_positions.insert(position);
        }
    }
    let n_disguisers = disguiser_positions.len() as i32;

    // Check villager excess
    let deck_v_counts: HashMap<String, i32> = {
        let mut m = HashMap::new();
        for v in &state.deck.villagers {
            *m.entry(normalize_role(v)).or_insert(0) += 1;
        }
        m
    };
    let has_baker_in_deck = state.deck.villagers.iter().any(|v| normalize_role(v) == "baker");

    let any_initial_role_multiset_fits = initial_good_villager_count_variants
        .iter()
        .any(|counts| {
            let mut total_excess = 0i32;
            for (role, &count) in counts {
                if role == "baker" && has_baker_in_deck { continue; }
                let deck_count = deck_v_counts.get(role).copied().unwrap_or(0);
                if count > deck_count { total_excess += count - deck_count; }
            }
            total_excess <= n_disguisers
        });
    if !any_initial_role_multiset_fits { return false; }

    // Check outcast counts
    let deck_o_counts: HashMap<String, i32> = {
        let mut m = HashMap::new();
        for o in &state.deck.outcasts { *m.entry(normalize_role(o)).or_insert(0) += 1; }
        m
    };
    for (role, &count) in &good_outcast_counts {
        if count > deck_o_counts.get(&normalize_role(role)).copied().unwrap_or(0) {
            return false;
        }
    }

    // Board outcast count ceiling
    if let Some(boc) = state.board_outcast_count {
        let chancellor_allowance = i32::from(
            scenario.chancellor_added_outcast_position().is_some()
        );
        let total_good_outcasts: i32 = good_outcast_counts.values().sum();
        if total_good_outcasts > boc as i32 + chancellor_allowance { return false; }
    }

    // Board villager count ceiling
    if let Some(bvc) = state.board_villager_count {
        let total_good_villagers: i32 = good_villager_counts.values().sum();
        if total_good_villagers > bvc as i32 + n_disguisers { return false; }
    }

    let shaman_copied_baker = scenario
        .shaman_trace
        .as_ref()
        .is_some_and(|trace| normalize_role(&trace.copied_role) == "baker");
    if shaman_copied_baker {
        // Copied Baker immediately acts with preserved destination runtimeData.
        // Until that native composition is modeled, Baker-derived uniqueness
        // and chain-existence pruning would be stronger than our evidence.
        return true;
    }

    // Baker original-role uniqueness
    let deck_villagers_norm: HashSet<String> = state.deck.villagers.iter().map(|v| normalize_role(v)).collect();
    let mut baker_claimed_counts: HashMap<String, i32> = HashMap::new();
    for card in &state.cards {
        if normalize_role(&card.apparent_role) != "baker" { continue; }
        let pos = card.position;
        if known_evil_role(pos, scenario, state).is_some() { continue; }
        if scenario.puppet_position == Some(pos) { continue; }
        let claimed = match info_str(&card.info_parsed, "original_role") {
            Some(s) => s,
            None => continue,
        };
        if claimed.is_empty() || claimed.eq_ignore_ascii_case("original") { continue; }
        let truth = truth_status(pos, scenario, state);
        if truth != TruthStatus::Truthful { continue; }
        let claimed_cd = match get_card(claimed) {
            Some(c) if c.faction == Faction::Villager => c,
            _ => continue,
        };
        if !deck_villagers_norm.contains(&normalize_role(claimed_cd.name)) { continue; }
        *baker_claimed_counts.entry(normalize_role(claimed)).or_insert(0) += 1;
    }

    for (norm_role, &baker_count) in &baker_claimed_counts {
        if norm_role == "baker" { continue; }
        let mut non_baker_real = 0i32;
        for card_c in &state.cards {
            if normalize_role(&card_c.apparent_role) != *norm_role { continue; }
            let pos_c = card_c.position;
            if known_evil_role(pos_c, scenario, state).is_some() { continue; }
            if scenario.puppet_position == Some(pos_c) { continue; }
            if scenario.drunk_position == Some(pos_c) || scenario.doppelganger_position == Some(pos_c) { continue; }
            non_baker_real += 1;
        }
        let deck_count = deck_v_counts.get(norm_role).copied().unwrap_or(0);
        let copied_role_allowance =
            i32::from(scenario.shaman_trace.as_ref().is_some_and(|trace| {
                let copied = normalize_role(&trace.copied_role);
                copied == *norm_role
            }));
        if non_baker_real + baker_count > deck_count + copied_role_allowance {
            return false;
        }

        // At most one ordinary Baker chain conversion can produce a given
        // "I was <role>" claim. Copied-Baker worlds returned above because
        // their preserved runtime provenance is still opaque.
        // Safe for current v2 suite (no deck has duplicate villagers, so the
        // existing deck_count check already enforces this implicitly). Adds
        // stricter behavior for hypothetical duplicate-villager decks.
        if baker_count > 1 { return false; }
    }

    // Baker conversion chain existence for fully modeled runtime provenance.
    // Copied-Baker worlds already returned above; merely listing or placing
    // Shaman does not disable this check.
    // NOTE (asc77 v6): we previously required chain-converted Bakers to reveal
    // AFTER the original, but the in-game conversion chain pre-seeds at game
    // start — chain Bakers can reveal in any order (observed: chain-Baker at #1
    // revealed before original at #6). Keep the existence requirement (an
    // original Baker must be somewhere in the truthful-Baker set) but drop the
    // reveal-ordering constraint.
    if !baker_claimed_counts.is_empty() {
        // If any truthful Baker claims "I was <role>", at least one truthful
        // Baker must claim to be the original (or it must be possible that
        // the original Baker is an unrevealed/night-killed/puppet position
        // where we can't observe its claim).
        let mut has_chain_claim = false;
        let mut has_original_claim = false;
        let mut has_unobservable_baker_slot = false;

        for card in &state.cards {
            if normalize_role(&card.apparent_role) != "baker" { continue; }
            let pos = card.position;
            if known_evil_role(pos, scenario, state).is_some() { continue; }
            if scenario.puppet_position == Some(pos) { continue; }
            let truth = truth_status(pos, scenario, state);
            if truth != TruthStatus::Truthful { continue; }
            let claimed = info_str(&card.info_parsed, "original_role").unwrap_or("");
            if claimed.eq_ignore_ascii_case("original") || claimed.eq_ignore_ascii_case("baker") {
                has_original_claim = true;
            } else if !claimed.is_empty() {
                has_chain_claim = true;
            }
        }
        // Also allow the original Baker to be a hidden (unrevealed) or
        // night-killed good position — we can't see its claim in those cases.
        for card in &state.cards {
            if !state.blocked_positions.contains(&card.position)
               && !state.night_kills.contains(&card.position) { continue; }
            // Hidden/night-killed Good positions could be the original Baker
            if known_evil_role(card.position, scenario, state).is_none() {
                has_unobservable_baker_slot = true;
                break;
            }
        }

        if has_chain_claim && !has_original_claim && !has_unobservable_baker_slot {
            return false;
        }
    }

    true
}

#[cfg(test)]
mod tests {
    //! Validator unit tests. See tests/simulation.rs for integration tests.
    use super::*;
    use serde_json::json;

    fn make_card(pos: u8, role: &str, info: serde_json::Value) -> CardInfo {
        CardInfo {
            position: pos,
            apparent_role: role.to_string(),
            info_text: String::new(),
            info_parsed: info.as_object().unwrap().clone(),
        }
    }

    fn empty_scenario() -> Scenario {
        Scenario {
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
        }
    }

    fn base_state(n_cards: u8, cards: Vec<CardInfo>) -> GameState {
        let mut s = GameState::default();
        s.n_cards = n_cards;
        s.cards = cards;
        s
    }

    fn dreamer_state(
        observation: serde_json::Value,
        first_apparent: &str,
        second_apparent: &str,
        fallback_apparent: &str,
    ) -> (CardInfo, GameState) {
        let dreamer = make_card(1, "Dreamer", observation);
        let state = base_state(
            4,
            vec![
                dreamer.clone(),
                make_card(2, first_apparent, json!({})),
                make_card(3, second_apparent, json!({})),
                make_card(4, fallback_apparent, json!({})),
            ],
        );
        (dreamer, state)
    }

    #[test]
    fn silenced_jester_is_vacuous_constraint() {
        // Silenced Jester (no evil_count) must return true unconditionally,
        // regardless of whether the scenario places evils inside targets or not.
        let jester = make_card(
            3,
            "Jester",
            json!({"targets": [1, 2, 4], "silenced": true}),
        );
        let state = base_state(7, vec![jester.clone()]);

        // Empty scenario
        let s1 = empty_scenario();
        assert!(validate_jester(&jester, &s1, &state));

        // Scenario with evils inside the target set
        let mut s2 = empty_scenario();
        s2.evil_positions.insert(1, "Pooka".to_string());
        s2.evil_positions.insert(2, "Witch".to_string());
        assert!(validate_jester(&jester, &s2, &state));

        // Scenario with evils outside the target set
        let mut s3 = empty_scenario();
        s3.evil_positions.insert(5, "Pooka".to_string());
        s3.evil_positions.insert(6, "Witch".to_string());
        assert!(validate_jester(&jester, &s3, &state));
    }

    #[test]
    fn witness_uses_persistent_messed_up_status_not_live_corruption() {
        let witness = make_card(
            1,
            "Witness",
            json!({"affected_position": 2}),
        );
        let state = base_state(3, vec![witness.clone(), make_card(2, "Baker", json!({}))]);

        let mut cured = empty_scenario();
        cured.messed_up_by_evil.insert(2);
        assert!(validate_witness(&witness, &cured, &state));

        let mut plague_doctor_only = empty_scenario();
        plague_doctor_only.corrupted.insert(2);
        plague_doctor_only.pd_corrupted = Some(2);
        assert!(!validate_witness(&witness, &plague_doctor_only, &state));
    }

    #[test]
    fn non_silenced_jester_still_validates_evil_count() {
        // With evil_count present, validator must do real work.
        let jester = make_card(
            3,
            "Jester",
            json!({"targets": [1, 2, 4], "evil_count": 1}),
        );
        let state = base_state(7, vec![jester.clone()]);

        let mut truthy = empty_scenario();
        truthy.evil_positions.insert(1, "Pooka".to_string());
        // Exactly 1 evil among targets — truthful Jester ok
        assert!(validate_jester(&jester, &truthy, &state));

        let mut falsy = empty_scenario();
        falsy.evil_positions.insert(1, "Pooka".to_string());
        falsy.evil_positions.insert(2, "Witch".to_string());
        // 2 evils among targets, claimed 1, Jester not evil -> inconsistent
        assert!(!validate_jester(&jester, &falsy, &state));
    }

    #[test]
    fn dreamer_honest_one_match_has_positive_native_support() {
        let (dreamer, state) = dreamer_state(
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Baker", "Scout"]
            }),
            "Baker",
            "Knight",
            "Scout",
        );
        assert!(validate_dreamer(&dreamer, &empty_scenario(), &state));

        // If the other target has a distinct bluff, native code uses that
        // bluff before considering the board fallback.
        let mut other_has_priority_bluff = empty_scenario();
        other_has_priority_bluff
            .evil_positions
            .insert(3, "Pooka".to_string());
        assert!(!validate_dreamer(
            &dreamer,
            &other_has_priority_bluff,
            &state,
        ));

        // Missing non-target board data must not erase that already-known
        // priority. An unseen board role is reachable only when the native
        // path actually falls through to the board-entry branch.
        let mut incomplete_state = state.clone();
        incomplete_state.cards.retain(|card| card.position != 4);
        assert!(!validate_dreamer(
            &dreamer,
            &other_has_priority_bluff,
            &incomplete_state,
        ));
    }

    #[test]
    fn dreamer_honest_both_options_can_match_via_board_entry_fallback() {
        let (dreamer, state) = dreamer_state(
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Baker", "Knight"]
            }),
            "Baker",
            "Knight",
            "Scout",
        );

        // With no target bluff, #2 can be the anchor and #3 itself can supply
        // the board-entry fallback. Both displayed roles are then among the
        // selected characters, and that observation is genuinely reachable.
        assert!(validate_dreamer(&dreamer, &empty_scenario(), &state));
    }

    #[test]
    fn dreamer_liar_zero_match_pair_comes_from_unique_helper_pool() {
        let (dreamer, state) = dreamer_state(
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Dreamer", "Scout"]
            }),
            "Baker",
            "Knight",
            "Scout",
        );
        let mut liar = empty_scenario();
        liar.corrupted.insert(1);

        assert!(validate_dreamer(&dreamer, &liar, &state));
    }

    #[test]
    fn dreamer_liar_allows_cross_target_bluff_real_collision() {
        let (dreamer, state) = dreamer_state(
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Baker", "Scout"]
            }),
            "Baker",
            "Baker",
            "Scout",
        );
        let mut liar = empty_scenario();
        liar.corrupted.insert(1);
        liar.evil_positions.insert(2, "Pooka".to_string());

        // #2's Baker bluff is inserted before helper exclusions are built,
        // even though it equals #3's real Baker identity.
        assert!(validate_dreamer(&dreamer, &liar, &state));
    }

    #[test]
    fn dreamer_liar_rejects_arbitrary_target_role_match() {
        let (dreamer, state) = dreamer_state(
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Baker", "Scout"]
            }),
            "Baker",
            "Knight",
            "Scout",
        );
        let mut liar = empty_scenario();
        liar.corrupted.insert(1);

        // Baker is a selected real identity and is not an initial target
        // bluff, so the helper explicitly excludes it.
        assert!(!validate_dreamer(&dreamer, &liar, &state));
    }

    #[test]
    fn dreamer_unversioned_role_pair_preserves_archived_match_semantics() {
        let unversioned = make_card(
            1,
            "Dreamer",
            json!({
                "targets": [2, 3],
                "evil_role_options": ["Alchemist", "Gravedigger"]
            }),
        );
        let current = make_card(
            1,
            "Dreamer",
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Alchemist", "Gravedigger"]
            }),
        );
        let state = base_state(
            4,
            vec![
                unversioned.clone(),
                make_card(2, "Druid", json!({})),
                make_card(3, "Alchemist", json!({})),
                make_card(4, "Baker", json!({})),
            ],
        );
        let mut selected_pooka = empty_scenario();
        selected_pooka
            .evil_positions
            .insert(3, "Pooka".to_string());

        // Archived asc81_v1-style data compares options only with the two
        // underlying selected roles. Druid/Pooka matches neither displayed
        // option, so the unversioned observation supports only a liar.
        assert!(!validate_dreamer(&unversioned, &selected_pooka, &state));
        selected_pooka.corrupted.insert(1);
        assert!(validate_dreamer(&unversioned, &selected_pooka, &state));

        // The same marked observation is not reachable under the exact native
        // helper pool: Gravedigger is absent from this complete board.
        assert!(!validate_dreamer(&current, &selected_pooka, &state));
    }

    #[test]
    fn dreamer_cabbage_requires_truthful_actor_and_real_wretch_target() {
        let (cabbage, state) = dreamer_state(
            json!({"targets": [2, 3], "cabbage": true}),
            "Wretch",
            "Knight",
            "Scout",
        );
        assert!(validate_dreamer(&cabbage, &empty_scenario(), &state));

        let marked_cabbage = make_card(
            1,
            "Dreamer",
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "cabbage": true
            }),
        );
        assert!(validate_dreamer(
            &marked_cabbage,
            &empty_scenario(),
            &state,
        ));

        let pair = make_card(
            1,
            "Dreamer",
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Wretch", "Scout"]
            }),
        );
        assert!(!validate_dreamer(&pair, &empty_scenario(), &state));

        let mut liar = empty_scenario();
        liar.corrupted.insert(1);
        assert!(!validate_dreamer(&cabbage, &liar, &state));

        let (no_wretch, no_wretch_state) = dreamer_state(
            json!({"targets": [2, 3], "cabbage": true}),
            "Baker",
            "Knight",
            "Scout",
        );
        assert!(!validate_dreamer(
            &no_wretch,
            &empty_scenario(),
            &no_wretch_state,
        ));
    }

    #[test]
    fn dreamer_new_shape_rejects_malformed_arrays_without_legacy_fallback() {
        let malformed = [
            json!({"dreamer_variant": "public_current"}),
            json!({"dreamer_variant": "legacy"}),
            json!({"dreamer_variant": 7}),
            json!({
                "dreamer_variant": "legacy",
                "targets": [2, 3],
                "evil_role_options": ["Baker", "Scout"]
            }),
            json!({
                "dreamer_variant": "legacy",
                "targets": [2, 3],
                "cabbage": true
            }),
            json!({
                "dreamer_variant": "public_current",
                "targets": [2],
                "evil_role_options": ["Baker", "Scout"]
            }),
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 5],
                "evil_role_options": ["Baker", "Scout"]
            }),
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Baker"]
            }),
            json!({"targets": [2], "evil_role_options": ["Baker", "Scout"]}),
            json!({"targets": [2, 3, 4], "evil_role_options": ["Baker", "Scout"]}),
            json!({"targets": [2, "3"], "evil_role_options": ["Baker", "Scout"]}),
            json!({"targets": [0, 3], "evil_role_options": ["Baker", "Scout"]}),
            json!({"targets": [2, 5], "evil_role_options": ["Baker", "Scout"]}),
            json!({"targets": [2, 3], "evil_role_options": ["Baker"]}),
            json!({"targets": [2, 3], "evil_role_options": ["Baker", " "]}),
            json!({"targets": [2, 3], "evil_role_options": ["Twin Minion", "Twin_Minion"]}),
            json!({"targets": [2, 3], "evil_role_options": ["Baker", 7]}),
            json!({"targets": [2, 3]}),
            json!({"evil_role_options": ["Baker", "Scout"]}),
            json!({"targets": [2, 3], "cabbage": false}),
            json!({
                "targets": [2, 3],
                "evil_role_options": ["Baker", "Scout"],
                "cabbage": true
            }),
            // Even valid historical keys must not hide malformed current
            // observation keys.
            json!({
                "target": 2,
                "evil_role": "Pooka",
                "targets": [2],
                "evil_role_options": ["Baker", "Scout"]
            }),
        ];
        for info in malformed {
            let (dreamer, state) = dreamer_state(info, "Baker", "Knight", "Scout");
            assert!(!validate_dreamer(&dreamer, &empty_scenario(), &state));
        }

        let (historical, historical_state) = dreamer_state(
            json!({"target": 2, "evil_role": "Pooka"}),
            "Baker",
            "Knight",
            "Scout",
        );
        let mut historical_scenario = empty_scenario();
        historical_scenario
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(validate_dreamer(
            &historical,
            &historical_scenario,
            &historical_state,
        ));
    }

    #[test]
    fn dreamer_observation_order_is_irrelevant() {
        let (_, state) = dreamer_state(json!({}), "Baker", "Knight", "Scout");
        let forward = make_card(
            1,
            "Dreamer",
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Baker", "Knight"]
            }),
        );
        let reversed = make_card(
            1,
            "Dreamer",
            json!({
                "dreamer_variant": "public_current",
                "targets": [3, 2],
                "evil_role_options": ["Knight", "Baker"]
            }),
        );

        assert!(validate_dreamer(&forward, &empty_scenario(), &state));
        assert!(validate_dreamer(&reversed, &empty_scenario(), &state));

        let mut liar = empty_scenario();
        liar.corrupted.insert(1);
        liar.evil_positions.insert(2, "Pooka".to_string());
        let (_, collision_state) = dreamer_state(json!({}), "Baker", "Baker", "Scout");
        let liar_forward = make_card(
            1,
            "Dreamer",
            json!({
                "dreamer_variant": "public_current",
                "targets": [2, 3],
                "evil_role_options": ["Baker", "Scout"]
            }),
        );
        let liar_reversed = make_card(
            1,
            "Dreamer",
            json!({
                "dreamer_variant": "public_current",
                "targets": [3, 2],
                "evil_role_options": ["Scout", "Baker"]
            }),
        );
        assert!(validate_dreamer(&liar_forward, &liar, &collision_state));
        assert!(validate_dreamer(&liar_reversed, &liar, &collision_state));
    }

    #[test]
    fn truthful_adjacent_card_must_shut_up_for_real_rambler() {
        let rambler = make_card(1, "Rambler", json!({"silenced": false}));
        let scout_normal = make_card(2, "Scout", json!({"evil_role": "Pooka", "distance": 1}));
        let baker_shut_up = make_card(5, "Baker", json!({"shut_up_target": 1}));
        let state = base_state(5, vec![rambler, scout_normal, baker_shut_up]);

        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn truthful_shut_up_target_must_be_adjacent_real_rambler() {
        let rambler = make_card(1, "Rambler", json!({"silenced": false}));
        let scout = make_card(2, "Scout", json!({"shut_up_target": 1}));
        let state = base_state(5, vec![rambler, scout]);

        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));

        let mut fake_rambler = empty_scenario();
        fake_rambler.evil_positions.insert(1, "Puppeteer".to_string());
        assert!(!validate_rambler_shut_ups(&fake_rambler, &state));
    }

    #[test]
    fn confessor_disguise_uses_apparent_truth_for_rambler() {
        let rambler = make_card(1, "Rambler", json!({"silenced": false}));
        let confessor = make_card(2, "Confessor", json!({"shut_up_target": 1}));
        let state = base_state(5, vec![rambler, confessor]);

        let mut evil_confessor = empty_scenario();
        evil_confessor
            .evil_positions
            .insert(2, "Puppeteer".to_string());

        assert_eq!(truth_status(2, &evil_confessor, &state), TruthStatus::Lying);
        assert_eq!(
            truth_appearance_status(2, &evil_confessor, &state),
            TruthStatus::Truthful
        );
        assert!(validate_rambler_shut_ups(&evil_confessor, &state));

        let mut corrupted_confessor = empty_scenario();
        corrupted_confessor.corrupted.insert(2);
        assert_eq!(
            truth_status(2, &corrupted_confessor, &state),
            TruthStatus::Lying
        );
        assert!(validate_rambler_shut_ups(&corrupted_confessor, &state));
    }

    #[test]
    fn rambler_act_surface_follows_actual_dispatch_truth() {
        let rambler = make_card(1, "Rambler", json!({"silenced": false}));
        let state = base_state(1, vec![rambler]);

        assert!(is_truthful_rambler_act_surface(1, &empty_scenario(), &state));

        let mut corrupted_real = empty_scenario();
        corrupted_real.corrupted.insert(1);
        assert!(!is_truthful_rambler_act_surface(1, &corrupted_real, &state));

        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(1);
        assert!(is_truthful_rambler_act_surface(1, &puppet, &state));
        puppet.corrupted.insert(1);
        assert!(!is_truthful_rambler_act_surface(1, &puppet, &state));

        let mut doppelganger = empty_scenario();
        doppelganger.doppelganger_position = Some(1);
        assert!(is_truthful_rambler_act_surface(1, &doppelganger, &state));
        doppelganger.corrupted.insert(1);
        assert!(!is_truthful_rambler_act_surface(1, &doppelganger, &state));

        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(1);
        assert!(!is_truthful_rambler_act_surface(1, &drunk, &state));

        let mut evil = empty_scenario();
        evil.evil_positions.insert(1, "Pooka".to_string());
        assert!(!is_truthful_rambler_act_surface(1, &evil, &state));
    }

    #[test]
    fn liar_shut_up_target_must_be_false() {
        let rambler = make_card(1, "Rambler", json!({"silenced": false}));
        let baker = make_card(2, "Baker", json!({"shut_up_target": 1}));
        let state = base_state(5, vec![rambler, baker]);

        let mut lying_next_to_real_rambler = empty_scenario();
        lying_next_to_real_rambler.evil_positions.insert(2, "Puppeteer".to_string());
        assert!(!validate_rambler_shut_ups(&lying_next_to_real_rambler, &state));

        let mut lying_to_fake_rambler = empty_scenario();
        lying_to_fake_rambler.evil_positions.insert(1, "Twin_Minion".to_string());
        lying_to_fake_rambler.evil_positions.insert(2, "Puppeteer".to_string());
        assert!(validate_rambler_shut_ups(&lying_to_fake_rambler, &state));
    }

    #[test]
    fn ordinary_drunk_status_is_visible_to_pd_but_execution_projects_clean() {
        let pd = make_card(7, "Plague_Doctor", json!({}));
        let druid = make_card(6, "Druid", json!({"targets": [1, 2, 7], "found_outcast": null}));
        let bard = make_card(5, "Bard", json!({"corruption_distance": 1}));
        let mut state = base_state(7, vec![druid, bard.clone(), pd]);
        state.pd_ability_results.push(crate::types::PdAbilityResult {
            pd_pos: 7,
            target: 6,
            is_corrupted: true,
            evil_revealed: Some(2),
        });

        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Pooka".to_string());
        scenario.drunk_position = Some(6);
        scenario.corrupted.insert(6);

        assert_eq!(truth_status(6, &scenario, &state), TruthStatus::Lying);
        assert!(scenario.corrupted.contains(&6));
        assert!(validate_bard(&bard, &scenario, &state));
        assert!(validate_pd_ability(&scenario, &state));

        assert!(matches_executed_good_corruption(&scenario, 6, false));
        assert!(!matches_executed_good_corruption(&scenario, 6, true));
    }

    #[test]
    fn resistant_generated_drunk_keeps_native_actor_and_target_surfaces() {
        let mut scenario = empty_scenario();
        scenario.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 1,
            added_outcast_role: "Drunk".to_string(),
        });
        scenario.evil_positions.insert(2, "Pooka".to_string());
        assert!(scenario.corrupted.is_empty());

        let mut slayer_state = base_state(3, vec![make_card(1, "Slayer", json!({}))]);
        slayer_state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: true,
            revealed_role: None,
        });
        assert!(!validate_slayer_results(&scenario, &slayer_state));
        slayer_state.slayer_results[0].killed = false;
        assert!(validate_slayer_results(&scenario, &slayer_state));

        let mut pd_state = base_state(3, vec![make_card(1, "Plague_Doctor", json!({}))]);
        pd_state.pd_ability_results.push(crate::types::PdAbilityResult {
            pd_pos: 1,
            target: 3,
            is_corrupted: true,
            evil_revealed: Some(3),
        });
        assert!(validate_pd_ability(&scenario, &pd_state));
        pd_state.pd_ability_results[0].is_corrupted = false;
        pd_state.pd_ability_results[0].evil_revealed = None;
        assert!(!validate_pd_ability(&scenario, &pd_state));

        let baker = make_card(1, "Baker", json!({"original_role": "original"}));
        let baker_state = base_state(3, vec![baker.clone()]);
        assert!(!validate_baker(&baker, &scenario, &baker_state));
        assert!(validate_baker(&baker, &empty_scenario(), &baker_state));

        assert!(matches_executed_good_corruption(&scenario, 1, false));
        assert!(!matches_executed_good_corruption(&scenario, 1, true));
    }

    #[test]
    fn slayer_revealed_role_pins_native_wretch_and_evil_kills() {
        let mut state = base_state(
            3,
            vec![
                make_card(1, "Slayer", json!({})),
                make_card(2, "Wretch", json!({})),
            ],
        );
        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: true,
            revealed_role: Some("Wretch".to_string()),
        });

        let good_wretch = empty_scenario();
        assert!(validate_slayer_results(&good_wretch, &state));
        state.slayer_results[0].revealed_role = Some("Shaman".to_string());
        assert!(!validate_slayer_results(&good_wretch, &state));

        let mut evil = empty_scenario();
        evil.evil_positions.insert(2, "Shaman".to_string());
        assert!(validate_slayer_results(&evil, &state));
        state.slayer_results[0].revealed_role = Some("Wretch".to_string());
        assert!(!validate_slayer_results(&evil, &state));

        let mut generated_state = base_state(3, vec![make_card(1, "Slayer", json!({}))]);
        generated_state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: true,
            revealed_role: Some("Wretch".to_string()),
        });
        let mut generated_wretch = empty_scenario();
        generated_wretch.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 2,
            added_outcast_role: "Wretch".to_string(),
        });
        generated_wretch.evil_positions.insert(3, "Chancellor".to_string());
        assert!(validate_slayer_results(&generated_wretch, &generated_state));
    }

    #[test]
    fn resistant_generated_drunk_is_clean_but_active_corruption_is_not() {
        let mut state = base_state(3, vec![make_card(3, "Plague_Doctor", json!({}))]);
        state.pd_ability_results.push(crate::types::PdAbilityResult {
            pd_pos: 3,
            target: 1,
            is_corrupted: false,
            evil_revealed: None,
        });
        let mut drunk = empty_scenario();
        drunk.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![2],
            added_outcast_position: 1,
            added_outcast_role: "Drunk".to_string(),
        });
        assert!(validate_pd_ability(&drunk, &state));
        state.pd_ability_results[0].is_corrupted = true;
        assert!(!validate_pd_ability(&drunk, &state));

        let mut scout = empty_scenario();
        scout.evil_positions.insert(2, "Pooka".to_string());
        scout.corrupted.insert(1);
        state.pd_ability_results[0].evil_revealed = Some(2);
        assert!(validate_pd_ability(&scout, &state));
    }

    #[test]
    fn pd_enforces_native_truth_bluff_and_visible_result_shapes() {
        let mut state = base_state(5, vec![make_card(1, "Plague_Doctor", json!({}))]);
        state.pd_ability_results.push(crate::types::PdAbilityResult {
            pd_pos: 1,
            target: 4,
            is_corrupted: true,
            evil_revealed: Some(2),
        });

        let mut truthful = empty_scenario();
        truthful.evil_positions.insert(2, "Pooka".to_string());
        truthful.corrupted.insert(4);
        assert!(validate_pd_ability(&truthful, &state));

        state.pd_ability_results[0].evil_revealed = None;
        assert!(!validate_pd_ability(&truthful, &state));
        state.pd_ability_results[0].evil_revealed = Some(5);
        assert!(!validate_pd_ability(&truthful, &state));
        state.pd_ability_results[0].is_corrupted = false;
        state.pd_ability_results[0].evil_revealed = Some(2);
        assert!(!validate_pd_ability(&truthful, &state));

        let mut bluff = truthful.clone();
        bluff.evil_positions.insert(1, "Minion".to_string());
        bluff.corrupted.remove(&4);
        state.pd_ability_results[0].is_corrupted = true;
        state.pd_ability_results[0].evil_revealed = Some(5);
        assert!(validate_pd_ability(&bluff, &state));
        state.pd_ability_results[0].evil_revealed = Some(2);
        assert!(!validate_pd_ability(&bluff, &state));

        bluff.corrupted.insert(4);
        state.pd_ability_results[0].is_corrupted = false;
        state.pd_ability_results[0].evil_revealed = None;
        assert!(validate_pd_ability(&bluff, &state));
    }

    #[test]
    fn pd_rejects_non_pd_actors_and_duplicate_once_use_evidence() {
        let mut state = base_state(4, vec![make_card(1, "Bard", json!({}))]);
        state.pd_ability_results.push(crate::types::PdAbilityResult {
            pd_pos: 1,
            target: 3,
            is_corrupted: false,
            evil_revealed: None,
        });
        let scenario = empty_scenario();
        assert!(!validate_pd_ability(&scenario, &state));

        state.cards[0].apparent_role = "Plague_Doctor".to_string();
        assert!(validate_pd_ability(&scenario, &state));
        state.pd_ability_results.push(crate::types::PdAbilityResult {
            pd_pos: 1,
            target: 4,
            is_corrupted: false,
            evil_revealed: None,
        });
        assert!(!validate_pd_ability(&scenario, &state));
    }

    #[test]
    fn pd_self_check_is_visibly_clean_regardless_of_truth_or_status() {
        let mut state = base_state(3, vec![make_card(1, "Plague_Doctor", json!({}))]);
        state.pd_ability_results.push(crate::types::PdAbilityResult {
            pd_pos: 1,
            target: 1,
            is_corrupted: false,
            evil_revealed: None,
        });

        let truthful_clean = empty_scenario();
        assert!(validate_pd_ability(&truthful_clean, &state));

        let mut corrupted_good = empty_scenario();
        corrupted_good.corrupted.insert(1);
        assert!(validate_pd_ability(&corrupted_good, &state));

        let mut bluff = empty_scenario();
        bluff.evil_positions.insert(1, "Minion".to_string());
        assert!(validate_pd_ability(&bluff, &state));

        state.pd_ability_results[0].is_corrupted = true;
        state.pd_ability_results[0].evil_revealed = Some(2);
        assert!(!validate_pd_ability(&truthful_clean, &state));
        assert!(!validate_pd_ability(&corrupted_good, &state));
        assert!(!validate_pd_ability(&bluff, &state));
    }

    #[test]
    fn pd_truthful_reveal_pool_uses_wretch_registered_alignment() {
        let mut state = base_state(
            3,
            vec![
                make_card(1, "Plague_Doctor", json!({})),
                make_card(3, "Wretch", json!({})),
            ],
        );
        state.pd_ability_results.push(crate::types::PdAbilityResult {
            pd_pos: 1,
            target: 2,
            is_corrupted: true,
            evil_revealed: Some(3),
        });
        let mut scenario = empty_scenario();
        scenario.corrupted.insert(2);

        assert_eq!(
            effective_alignment(3, &scenario, &state),
            EffectiveAlignment::Evil,
        );
        assert!(validate_pd_ability(&scenario, &state));
    }

    #[test]
    fn resistant_generated_drunk_can_fill_a_night_killed_knight_identity() {
        let mut state = base_state(1, vec![]);
        state.deck.villagers = vec!["Knight".to_string()];
        state.night_kills = vec![1];
        state.night_kill_evil_count = 0;
        state.board_villager_count = Some(1);

        let clean_unknown = empty_scenario();
        assert!(!check_scenario(&clean_unknown, &state));

        let mut generated_drunk = empty_scenario();
        generated_drunk.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![1],
            added_outcast_position: 1,
            added_outcast_role: "Drunk".to_string(),
        });
        assert!(check_scenario(&generated_drunk, &state));
    }

    #[test]
    fn hidden_generated_drunk_counts_its_apparent_villager_and_one_disguise() {
        let mut state = base_state(3, vec![
            make_card(1, "Baker", json!({})),
            make_card(2, "Knight", json!({})),
            make_card(3, "Knight", json!({})),
        ]);
        state.deck.villagers = vec!["Knight".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string()];
        let mut generated_drunk = empty_scenario();
        generated_drunk.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![2],
            added_outcast_position: 1,
            added_outcast_role: "Drunk".to_string(),
        });

        assert!(!validate_role_counts(&generated_drunk, &state));
    }

    #[test]
    fn shaman_trace_accepts_any_viable_role_in_its_erased_identity_class() {
        let mut state = base_state(
            4,
            vec![
                make_card(1, "Scout", json!({})),
                make_card(2, "Scout", json!({})),
                make_card(3, "Witness", json!({})),
                make_card(4, "Baker", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Scout".to_string(),
            "Witness".to_string(),
            "Judge".to_string(),
        ];
        state.deck.minions = vec!["Shaman".to_string()];
        state.board_villager_count = Some(3);

        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Shaman".to_string());
        scenario.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 1,
            target_position: 2,
            copied_role: "Scout".to_string(),
            target_previous_roles: vec!["Scout".to_string(), "Judge".to_string()],
        });

        assert!(validate_role_counts(&scenario, &state));

        scenario
            .shaman_trace
            .as_mut()
            .unwrap()
            .target_previous_roles = vec!["Scout".to_string()];
        assert!(
            !validate_role_counts(&scenario, &state),
            "an equivalence class with no deck-compatible erased role must fail"
        );

        scenario
            .shaman_trace
            .as_mut()
            .unwrap()
            .target_previous_roles = vec!["Judge".to_string()];

        state.board_villager_count = Some(2);
        assert!(
            !validate_role_counts(&scenario, &state),
            "Shaman changes a role identity, not the physical Villager count"
        );
    }

    #[test]
    fn undealt_shaman_pool_entry_grants_no_duplicate_allowance() {
        let mut state = base_state(
            2,
            vec![
                make_card(1, "Scout", json!({})),
                make_card(2, "Scout", json!({})),
            ],
        );
        state.deck.villagers = vec!["Scout".to_string(), "Witness".to_string()];
        state.deck.minions = vec!["Shaman".to_string()];

        assert!(!validate_role_counts(&empty_scenario(), &state));
    }

    #[test]
    fn generated_wretch_uses_every_special_registration_surface() {
        let fortune_teller = make_card(
            1,
            "Fortune_Teller",
            json!({"targets": [2], "has_evil": true}),
        );
        let oracle = make_card(
            1,
            "Oracle",
            json!({"targets": [2], "minion_role": "Minion"}),
        );
        let dreamer = make_card(
            1,
            "Dreamer",
            json!({"target": 2, "evil_role": "cabbage"}),
        );
        let state = base_state(3, vec![fortune_teller.clone()]);
        let mut scenario = empty_scenario();
        scenario.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 2,
            added_outcast_role: "Wretch".to_string(),
        });

        assert_eq!(
            effective_alignment(2, &scenario, &state),
            EffectiveAlignment::Evil,
        );
        assert_eq!(get_position_type(2, &scenario, &state), Some("Minion"));
        assert!(validate_fortune_teller(&fortune_teller, &scenario, &state));
        assert!(validate_oracle(&oracle, &scenario, &state));
        assert!(validate_dreamer(&dreamer, &scenario, &state));
    }

    #[test]
    fn druid_projects_trace_hidden_outcasts_and_legacy_fallback() {
        let pd_druid = make_card(
            1,
            "Druid",
            json!({"targets": [2], "found_outcast": "Plague Doctor"}),
        );
        let none_druid = make_card(
            1,
            "Druid",
            json!({"targets": [2], "found_outcast": null}),
        );
        let mut state = base_state(3, vec![pd_druid.clone()]);

        let mut generated_pd = empty_scenario();
        generated_pd.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 2,
            added_outcast_role: "Plague_Doctor".to_string(),
        });
        assert!(validate_druid(&pd_druid, &generated_pd, &state));

        let mut generated_wretch = generated_pd.clone();
        generated_wretch
            .chancellor_trace
            .as_mut()
            .unwrap()
            .added_outcast_role = "Wretch".to_string();
        assert!(validate_druid(&none_druid, &generated_wretch, &state));

        state.cards.push(make_card(2, "Plague_Doctor", json!({})));
        let mut legacy = empty_scenario();
        legacy.chancellor_conversion = Some(2);
        assert!(validate_druid(&pd_druid, &legacy, &state));

        state.cards.retain(|card| card.position != 2);
        let mut doppelganger = empty_scenario();
        doppelganger.doppelganger_position = Some(2);
        let dopp_druid = make_card(
            1,
            "Druid",
            json!({"targets": [2], "found_outcast": "Doppelganger"}),
        );
        assert!(validate_druid(&dopp_druid, &doppelganger, &state));

        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(2);
        let drunk_druid = make_card(
            1,
            "Druid",
            json!({"targets": [2], "found_outcast": "Drunk"}),
        );
        assert!(validate_druid(&drunk_druid, &drunk, &state));
    }

    #[test]
    fn observed_executed_good_role_filters_role_distinct_trace_worlds() {
        let mut state = base_state(3, vec![]);
        state.executed = vec![2];
        state.confirmed_good = vec![2];
        state.executed_good_roles.insert(2, "Plague Doctor".to_string());

        let trace = |role: &str| {
            let mut scenario = empty_scenario();
            scenario.chancellor_trace = Some(crate::types::ChancellorTrace {
                original_positions: vec![3],
                added_outcast_position: 2,
                added_outcast_role: role.to_string(),
            });
            scenario
        };
        assert!(matches_executed_good_role(
            &trace("Plague_Doctor"),
            &state,
            2,
            "Plague Doctor",
        ));
        assert!(!matches_executed_good_role(
            &trace("Rambler"),
            &state,
            2,
            "Plague Doctor",
        ));
    }
}
