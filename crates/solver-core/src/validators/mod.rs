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
        _ => true,
    }
}

/// Check if all revealed cards + ability results + structural constraints are consistent.
pub fn check_scenario(scenario: &Scenario, state: &GameState) -> bool {
    // Check observed corruption status of executed good cards
    for (&pos, &was_corrupted) in &state.executed_good_corrupted {
        if was_corrupted && !scenario.corrupted.contains(&pos) { return false; }
        if !was_corrupted && scenario.corrupted.contains(&pos) { return false; }
    }

    // Structural: role counts
    if !validate_role_counts(scenario, state) { return false; }

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
            let knight_revealed = state.cards.iter().any(|c|
                c.apparent_role == "Knight" && !scenario.evil_positions.contains_key(&c.position));
            if !knight_revealed {
                let revealed: HashSet<u8> = state.cards.iter().map(|c| c.position).collect();
                let valid = (1..=state.n_cards).any(|p|
                    !scenario.evil_positions.contains_key(&p)
                    && !revealed.contains(&p)
                    && !(state.night_kills.contains(&p) && !scenario.corrupted.contains(&p))
                );
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
        known_evil_role(p, scenario, state).map_or(false, |r| r == evil_role)
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
        known_evil_role(t, scenario, state).map_or(false, |r| r == minion_role)
    };
    let target_matches_possible = |t: u8| -> bool {
        if target_matches_definite(t) { return true; }
        if let Some(c) = state.card_at(t) {
            if c.apparent_role == "Wretch" {
                return get_card(minion_role).map_or(false, |cd| cd.faction == Faction::Minion);
            }
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

    // Build affected set: evil-sourced corruption + puppet + night kills
    let mut evil_corrupted = scenario.corrupted.clone();
    if let Some(pd) = scenario.pd_corrupted { evil_corrupted.remove(&pd); }
    if let Some(dp) = scenario.drunk_position { evil_corrupted.remove(&dp); }
    for c in &state.cards {
        if c.apparent_role == "Drunk" { evil_corrupted.remove(&c.position); }
    }
    let mut affected = evil_corrupted;
    if let Some(pp) = scenario.puppet_position { affected.insert(pp); }
    for (&pos, role) in &state.executed_evil_roles {
        if role == "Puppet" { affected.insert(pos); }
    }
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

fn validate_dreamer(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
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
    let target_is_wretch = state.card_at(target).map_or(false, |c| c.apparent_role == "Wretch");

    // Gather all known evil roles on the board
    let evil_roles: Vec<&str> = (1..=state.n_cards)
        .filter_map(|p| {
            let r = known_evil_role(p, scenario, state)?;
            if r == "Puppet" || r == "Unknown" { None } else { Some(r) }
        })
        .collect();

    if truth == TruthStatus::Truthful {
        if target_is_wretch { return claimed_role.eq_ignore_ascii_case("cabbage"); }
        if target_is_evil {
            let actual = known_evil_role(target, scenario, state).unwrap_or("");
            return claimed_role == actual;
        }
        evil_roles.contains(&claimed_role)
    } else {
        if target_is_wretch { return !claimed_role.eq_ignore_ascii_case("cabbage"); }
        if target_is_evil {
            let actual = known_evil_role(target, scenario, state).unwrap_or("");
            return claimed_role != actual;
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
    let claimed = match info_i64(&card.info_parsed, "cured_count") {
        Some(v) => v,
        None => return true,
    };
    let pos = card.position;
    let truth = truth_status(pos, scenario, state);
    let actual = scenario.alchemist_cures.get(&pos).copied().unwrap_or(0) as i64;

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
            (has_outcast && actual_outcasts.iter().any(|a| a == fo)) || has_unrevealed_good_target
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
        let mut actual_types: Vec<String> = Vec::new();
        let mut incomplete = false;
        for &t in &targets {
            if let Some(tp) = get_position_type(t, scenario, state) {
                actual_types.push(tp.to_string());
            } else {
                incomplete = true;
                break;
            }
        }
        if !incomplete {
            let mut sorted_actual = actual_types.clone();
            sorted_actual.sort();
            let mut sorted_claimed = ct.clone();
            sorted_claimed.sort();
            let types_match = sorted_actual == sorted_claimed;
            return if truth == TruthStatus::Truthful { types_match } else { !types_match };
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
    let claimed = match info_str(&card.info_parsed, "original_role") {
        Some(s) => s,
        None => return true,
    };
    let pos = card.position;
    let truth = truth_status(pos, scenario, state);

    if claimed.eq_ignore_ascii_case("original") {
        if truth == TruthStatus::Lying {
            if !is_evil_in_board_state(pos, scenario, state) && scenario.corrupted.contains(&pos) {
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

// ── Special ability validators ──

fn validate_slayer_results(scenario: &Scenario, state: &GameState) -> bool {
    for result in &state.slayer_results {
        let slayer_pos = result.slayer_pos;
        let target_pos = result.target_pos;
        let killed = result.killed;

        let slayer_evil_role = known_evil_role(slayer_pos, scenario, state);
        let slayer_is_evil = slayer_evil_role.is_some();
        let slayer_is_puppet = slayer_evil_role == Some("Puppet");
        let slayer_is_corrupted = scenario.corrupted.contains(&slayer_pos);
        let target_is_evil = effective_alignment(target_pos, scenario, state) == EffectiveAlignment::Evil;

        if killed {
            if slayer_is_evil && !slayer_is_puppet { return false; }
            if slayer_is_corrupted { return false; }
            if !target_is_evil { return false; }
        } else {
            let slayer_works = (!slayer_is_evil || slayer_is_puppet) && !slayer_is_corrupted;
            if slayer_works && target_is_evil { return false; }
        }
    }
    true
}

fn validate_pd_ability(scenario: &Scenario, state: &GameState) -> bool {
    for result in &state.pd_ability_results {
        let pd_pos = result.pd_pos;
        let target = result.target;
        let claimed_corrupted = result.is_corrupted;
        let evil_revealed = result.evil_revealed;

        let pd_is_evil = is_evil_in_board_state(pd_pos, scenario, state);
        let actual_corrupted = scenario.corrupted.contains(&target);

        if pd_is_evil {
            if claimed_corrupted == actual_corrupted { return false; }
            if claimed_corrupted {
                if let Some(er) = evil_revealed {
                    if effective_alignment(er, scenario, state) == EffectiveAlignment::Evil {
                        return false;
                    }
                }
            }
        } else {
            if claimed_corrupted != actual_corrupted { return false; }
            if claimed_corrupted {
                if let Some(er) = evil_revealed {
                    if effective_alignment(er, scenario, state) != EffectiveAlignment::Evil {
                        return false;
                    }
                }
            }
        }
    }
    true
}

// ── Role count validation ──

fn validate_role_counts(scenario: &Scenario, state: &GameState) -> bool {
    let mut good_villager_counts: HashMap<String, i32> = HashMap::new();
    let mut good_outcast_counts: HashMap<String, i32> = HashMap::new();

    for card in &state.cards {
        let pos = card.position;
        if known_evil_role(pos, scenario, state).is_some() { continue; }
        let role = &card.apparent_role;
        if knowledge_base::is_villager_role(role) {
            *good_villager_counts.entry(role.clone()).or_insert(0) += 1;
        } else if state.deck.outcasts.iter().any(|o| o == role || o.replace('_', " ") == *role) {
            *good_outcast_counts.entry(role.clone()).or_insert(0) += 1;
        }
    }

    // Disguiser count
    let mut n_disguisers = 0i32;
    if let Some(dp) = scenario.doppelganger_position {
        if !scenario.evil_positions.contains_key(&dp) { n_disguisers += 1; }
    }
    if let Some(dp) = scenario.drunk_position {
        if !scenario.evil_positions.contains_key(&dp) { n_disguisers += 1; }
    }

    // Check villager excess
    let deck_v_counts: HashMap<String, i32> = {
        let mut m = HashMap::new();
        for v in &state.deck.villagers {
            *m.entry(normalize_role(v)).or_insert(0) += 1;
        }
        m
    };
    let has_baker_in_deck = state.deck.villagers.iter().any(|v| normalize_role(v) == "baker");

    let mut total_excess = 0i32;
    for (role, &count) in &good_villager_counts {
        if normalize_role(role) == "baker" && has_baker_in_deck { continue; }
        let deck_count = deck_v_counts.get(&normalize_role(role)).copied().unwrap_or(0);
        if count > deck_count { total_excess += count - deck_count; }
    }

    // Shaman allowance
    let mut shaman_allowance = 0i32;
    if state.deck.minions.iter().any(|m| m == "Shaman") {
        for p in 1..=state.n_cards {
            if known_evil_role(p, scenario, state) == Some("Shaman") {
                shaman_allowance = 1;
                break;
            }
        }
    }

    // Shaman requires duplicate
    if shaman_allowance > 0 {
        let has_dup = good_villager_counts.values().any(|&c| c >= 2);
        if !has_dup {
            let revealed: HashSet<u8> = state.cards.iter().map(|c| c.position).collect();
            let unrevealed_good = (1..=state.n_cards)
                .filter(|&p| !revealed.contains(&p) && known_evil_role(p, scenario, state).is_none())
                .count();
            if unrevealed_good == 0 { return false; }
        }
    }

    if total_excess > n_disguisers + shaman_allowance { return false; }

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
        let chancellor_allowance = if state.deck.minions.iter().any(|m| m == "Chancellor") { 1 } else { 0 };
        let total_good_outcasts: i32 = good_outcast_counts.values().sum();
        if total_good_outcasts > boc as i32 + chancellor_allowance { return false; }
    }

    // Board villager count ceiling
    if let Some(bvc) = state.board_villager_count {
        let total_good_villagers: i32 = good_villager_counts.values().sum();
        if total_good_villagers > bvc as i32 + n_disguisers + shaman_allowance { return false; }
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
        if non_baker_real + baker_count > deck_count + shaman_allowance { return false; }
    }

    // Baker conversion chain ordering
    if !state.reveal_order.is_empty() && !baker_claimed_counts.is_empty() {
        let reveal_idx: HashMap<u8, usize> = state.reveal_order.iter().enumerate()
            .map(|(i, &p)| (p, i)).collect();

        // Find earliest good original Baker
        let mut original_baker_idx: Option<usize> = None;
        for card in &state.cards {
            if normalize_role(&card.apparent_role) != "baker" { continue; }
            let pos = card.position;
            if known_evil_role(pos, scenario, state).is_some() { continue; }
            if scenario.puppet_position == Some(pos) { continue; }
            let truth = truth_status(pos, scenario, state);
            if truth != TruthStatus::Truthful { continue; }
            let claimed = info_str(&card.info_parsed, "original_role").unwrap_or("");
            if claimed.eq_ignore_ascii_case("original") || claimed.eq_ignore_ascii_case("baker") {
                if let Some(&idx) = reveal_idx.get(&pos) {
                    original_baker_idx = Some(idx);
                    break;
                }
            }
        }

        match original_baker_idx {
            None => return false, // No original Baker — no conversion possible
            Some(orig_idx) => {
                for card in &state.cards {
                    if normalize_role(&card.apparent_role) != "baker" { continue; }
                    let pos = card.position;
                    if known_evil_role(pos, scenario, state).is_some() { continue; }
                    if scenario.puppet_position == Some(pos) { continue; }
                    let claimed = info_str(&card.info_parsed, "original_role").unwrap_or("");
                    if claimed.is_empty() || claimed.eq_ignore_ascii_case("original") { continue; }
                    let truth = truth_status(pos, scenario, state);
                    if truth != TruthStatus::Truthful { continue; }
                    if let Some(&idx) = reveal_idx.get(&pos) {
                        if idx < orig_idx { return false; }
                    }
                }
            }
        }
    }

    true
}
