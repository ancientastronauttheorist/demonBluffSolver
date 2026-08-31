/// Card info validators — check if a card's claimed info is consistent with a scenario.

mod baker;
mod disguisers;
mod helpers;
pub use helpers::*;

use baker::{
    baker_history_can_erase_role, medium_uses_baker_history,
    validate_baker_history,
};
use disguisers::validate_clean_doppel_source_support;

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
    let exact_match = effective_role_at(pos, scenario, state)
        .as_deref()
        .is_some_and(|role| roles_equal(role, observed_role));
    exact_match || twin_can_explain_current_role_mismatch(pos, observed_role, scenario, state)
}

fn twin_can_explain_current_role_mismatch(
    pos: u8,
    observed_role: &str,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let authored_roles = state.deck.all_roles();
    let deck_has = |expected: &str| {
        authored_roles
            .iter()
            .any(|role| normalize_role(role) == expected)
    };
    if !deck_has("twinminion") {
        return false;
    }

    if normalize_role(observed_role) == "twinminion" {
        return true;
    }

    let stable_twin = known_evil_role(pos, scenario, state)
        .is_some_and(|role| normalize_role(role) == "twinminion");
    if !stable_twin {
        return false;
    }

    // The original Twin can receive any authored current CharacterData from
    // its selected neighbor. Later start actions can overwrite that received
    // data with another authored role (Shaman), or with generated Puppet data
    // when Puppeteer is authored. Do not admit arbitrary roles absent from
    // those pre-trace current-data surfaces.
    let observed = normalize_role(observed_role);
    deck_has(&observed) || (observed == "puppet" && deck_has("puppeteer"))
}

fn valid_executed_current_role_entry(
    state: &GameState,
    pos: u8,
    observed_role: &str,
) -> bool {
    if pos == 0
        || pos > state.n_cards
        || !state.executed.contains(&pos)
        || state.night_kills.contains(&pos)
        || state
            .slayer_results
            .iter()
            .any(|result| result.killed && result.target_pos == pos)
    {
        return false;
    }

    let normalized = normalize_role(observed_role);
    !matches!(normalized.as_str(), "" | "unknown" | "?" | "none" | "null")
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
    for (&pos, observed_role) in &state.executed_current_roles {
        if !valid_executed_current_role_entry(state, pos, observed_role)
            || !matches_executed_good_role(scenario, state, pos, observed_role)
        {
            return false;
        }
    }

    // Shipped Cipher/Witch contributes one global hidden-card block. A blocked
    // seat is whichever hidden card the player happened to click once that
    // quota was reached; its position does not identify the Witch. Historical
    // fixtures retain that marker after a Witch death.
    if !validate_witch_block_evidence(scenario, state) { return false; }

    // Structural: role counts
    if !validate_role_counts(scenario, state) { return false; }

    // Audited delayed Reveal: a clean Doppelganger needs at least one
    // physically surviving, real-bluffable Good Villager source.
    if !validate_clean_doppel_source_support(scenario, state) { return false; }

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

    // Public successful Lilis deaths. No-kill Nights and their ordering are
    // not represented by GameState, so this validates only exact facts carried
    // by `night_kills` rather than inferring chronology from final reveal data.
    if !validate_lilis_night_kills(scenario, state) { return false; }

    validate_baker_history(scenario, state)
}

fn validate_lilis_night_kills(scenario: &Scenario, state: &GameState) -> bool {
    if state.night_kills.is_empty() {
        return state.night_kill_evil_count == 0;
    }

    // Striga/Lilis is the only shipped producer of this evidence. Historical
    // events remain valid after Lilis is executed, so deck membership is the
    // safe actor constraint; final-state liveness is not.
    let deck_lilis_count = state.deck.demons.iter()
        .filter(|role| normalize_role(role) == "lilis")
        .count();
    if deck_lilis_count == 0 {
        return false;
    }

    // The public pool can contain duplicate Lilis records even when role-count
    // selection put only one physical Lilis on this board. Ordinary Start
    // protects exactly one physical match, so a successful Lilis victim needs
    // two possible physical Lilis actors in this scenario, not merely two pool
    // records. Historical confirmed-Evil deaths without a public role leave an
    // unbound role slot; conservatively allow that slot to be another Lilis,
    // bounded by the authored pool and trusted board Demon count.
    let named_lilis_positions: HashSet<u8> = scenario.evil_positions.iter()
        .filter_map(|(&position, role)| {
            (normalize_role(role) == "lilis").then_some(position)
        })
        .collect();
    let mut untyped_evil_positions: HashSet<u8> = scenario.evil_positions.iter()
        .filter_map(|(&position, role)| {
            (normalize_role(role) == "unknown").then_some(position)
        })
        .collect();
    for &position in &state.confirmed_evil {
        let dead = state.executed.contains(&position) || state.night_kills.contains(&position);
        let public_role_is_untyped = state.executed_evil_roles.get(&position)
            .map_or(true, |role| normalize_role(role) == "unknown");
        if dead && public_role_is_untyped && !named_lilis_positions.contains(&position) {
            untyped_evil_positions.insert(position);
        }
    }
    let mut possible_physical_lilis = named_lilis_positions.len()
        .saturating_add(untyped_evil_positions.len())
        .min(deck_lilis_count);
    if let Some(board_demon_count) = state.board_demon_count {
        possible_physical_lilis = possible_physical_lilis.min(board_demon_count as usize);
    }
    if state.n_evil > 0 {
        possible_physical_lilis = possible_physical_lilis.min(state.n_evil as usize);
    }

    let named_lilis_victims = state.night_kills.iter()
        .filter(|position| {
            effective_role_at(**position, scenario, state)
                .is_some_and(|role| normalize_role(&role) == "lilis")
                && !twin_may_have_replaced_current_data_at(**position, scenario, state)
        })
        .count();
    if named_lilis_victims > 0 && named_lilis_victims >= possible_physical_lilis {
        // Ordinary Start protects one physical same-asset Lilis for the whole
        // game. Other Lilis deaths do not remove that actor's status, so at
        // most physical_count - 1 named Lilis can be successful Night victims.
        return false;
    }

    let unique_victims: HashSet<u8> = state.night_kills.iter().copied().collect();
    if unique_victims.len() != state.night_kills.len()
        || unique_victims.iter().any(|position| *position == 0 || *position > state.n_cards)
    {
        return false;
    }

    let evil_victims = state.night_kills.iter()
        .filter(|position| scenario.is_evil(**position))
        .count() as u8;
    if evil_victims != state.night_kill_evil_count {
        return false;
    }

    for &position in &state.night_kills {
        let role = effective_role_at(position, scenario, state);
        let twin_may_have_replaced_current_data =
            twin_may_have_replaced_current_data_at(position, scenario, state);

        if scenario.is_evil(position) {
            // Runtime-Evil Knight and every ordinary Evil role are killable;
            // Evil victims arise only from Lilis's unaligned fallback pass.
            continue;
        }

        let corrupted = scenario.corrupted.contains(&position);
        if role.as_deref().is_some_and(|role| normalize_role(role) == "knight")
            && !corrupted
            && !twin_may_have_replaced_current_data
        {
            // Delayed demon death asks the current real role for protection.
            // A clean Good Knight aborts the death and produces no night_kill.
            return false;
        }

        let apparent_knight = state.card_at(position)
            .is_some_and(|card| normalize_role(&card.apparent_role) == "knight");
        let effective_doppelganger = role.as_deref().is_some_and(|role| {
            matches!(normalize_role(role).as_str(), "doppelganger" | "doppleganger")
        });
        if effective_doppelganger
            && apparent_knight
            && !corrupted
            && !twin_may_have_replaced_current_data
        {
            // A clean ordinary or Chancellor-generated Doppelganger acquired
            // HealthyBluff and delegates protection to its Knight bluff.
            return false;
        }
    }

    // Preserve the legacy hidden-identity check: if the deck contains Knight
    // but no known Good Knight is placed, at least one compatible unobserved
    // home (or an omitted pool identity) must remain. This does not invent a
    // particular victim role.
    let knight_identity_may_be_erased = scenario.shaman_trace.as_ref().is_some_and(|trace| {
        trace.target_previous_roles.iter()
            .any(|role| normalize_role(role) == "knight")
    })
        || untyped_historical_evil_may_be_start_eraser(scenario, state)
        || scenario.chancellor_trace.is_some()
        || scenario.chancellor_conversion.is_some()
        || scenario.puppet_position.is_some()
        || scenario.drunk_position.is_some()
        || scenario.doppelganger_position.is_some()
        || baker_history_can_erase_role(scenario, state, "Knight");
    if !knight_identity_may_be_erased
        && state.deck.villagers.iter().any(|role| normalize_role(role) == "knight")
    {
        let knight_revealed = state.cards.iter().any(|card| {
            !scenario.is_evil(card.position)
                && effective_role_at(card.position, scenario, state)
                    .is_some_and(|role| normalize_role(&role) == "knight")
        });
        if !knight_revealed {
            let revealed: HashSet<u8> = state.cards.iter().map(|card| card.position).collect();
            let valid = (1..=state.n_cards).any(|position| {
                let intrinsically_killable = effective_role_at(position, scenario, state)
                    .is_some_and(|role| normalize_role(&role) == "drunk");
                !scenario.is_evil(position)
                    && !revealed.contains(&position)
                    && !(state.night_kills.contains(&position)
                        && !scenario.corrupted.contains(&position)
                        && !intrinsically_killable)
            });
            let pool_gt_board = state.board_villager_count
                .is_some_and(|count| state.deck.villagers.len() as u8 > count);
            if !valid && !pool_gt_board {
                return false;
            }
        }
    }

    true
}

pub(crate) fn twin_may_have_replaced_current_data_at(
    position: u8,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if state.n_cards < 2
        || !state
            .deck
            .minions
            .iter()
            .any(|role| normalize_role(role) == "twinminion")
    {
        return false;
    }

    // The unmodeled endpoint `n` cannot be Twin's own runtime body `m`.
    // Historical deaths can leave an untyped Evil origin, so retain it as a
    // possible Twin without allowing the same unknown seat to also supply the
    // distinct Demon anchor below.
    let possible_twin_origins: Vec<u8> = (1..=state.n_cards)
        .filter(|candidate| *candidate != position)
        .filter(|candidate| {
            known_evil_role(*candidate, scenario, state).is_some_and(|role| {
                matches!(normalize_role(role).as_str(), "twinminion" | "unknown")
            })
        })
        .collect();
    if possible_twin_origins.is_empty() {
        return false;
    }

    let authored_demon_exists = !state.deck.demons.is_empty();
    let chancellor_may_have_relocated_demon_data = authored_demon_exists
        && (scenario.chancellor_trace.is_some() || scenario.chancellor_conversion.is_some());

    adjacent_positions(position, state.n_cards)
        .into_iter()
        .filter(|anchor| *anchor != position)
        .any(|anchor| {
            let modeled_registered_or_real_demon = effective_role_at(anchor, scenario, state)
                .as_deref()
                .and_then(get_card)
                .is_some_and(|card| card.faction == Faction::Demon);
            let unknown_evil_could_be_demon = authored_demon_exists
                && known_evil_role(anchor, scenario, state)
                    .is_some_and(|role| normalize_role(role) == "unknown");
            let possible_demon_anchor = modeled_registered_or_real_demon
                || unknown_evil_could_be_demon
                || chancellor_may_have_relocated_demon_data;
            possible_demon_anchor
                && possible_twin_origins
                    .iter()
                    .any(|twin_origin| *twin_origin != anchor)
        })
}

fn untyped_historical_evil_may_be_start_eraser(
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let dead: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter())
        .copied()
        .collect();
    let has_untyped_historical_evil = state.confirmed_evil.iter().any(|position| {
        dead.contains(position)
            && state.executed_evil_roles.get(position)
                .is_none_or(|role| normalize_role(role) == "unknown")
    });
    if !has_untyped_historical_evil {
        return false;
    }

    // Shaman and Chancellor are Minions in this fingerprint. A trusted board
    // Minion count can prove that every physical Minion slot is already bound
    // to a typed history/current placement, in which case the untyped death
    // cannot be either Start eraser even if that role appears in the pool.
    let known_minion_positions: HashSet<u8> = scenario.evil_positions.iter()
        .chain(state.executed_evil_roles.iter())
        .filter_map(|(&position, role)| {
            get_card(role)
                .is_some_and(|card| card.faction == Faction::Minion)
                .then_some(position)
        })
        .collect();
    if state.board_minion_count
        .is_some_and(|count| known_minion_positions.len() >= count as usize)
    {
        return false;
    }

    ["shaman", "chancellor"].into_iter().any(|eraser| {
        let authored = state.deck.evil_roles().iter()
            .filter(|role| normalize_role(role) == eraser)
            .count();
        let represented_positions: HashSet<u8> = scenario.evil_positions.iter()
            .chain(state.executed_evil_roles.iter())
            .filter_map(|(&position, role)| {
                (normalize_role(role) == eraser).then_some(position)
            })
            .collect();
        authored > represented_positions.len()
    })
}

fn validate_witch_block_evidence(scenario: &Scenario, state: &GameState) -> bool {
    let blocked_count = state.blocked_positions.iter().copied()
        .collect::<HashSet<_>>().len();
    if blocked_count == 0 { return true; }
    // ManageCharacters stops after the first ordinary asset match. Duplicate
    // Witch records therefore do not stack Start increments in represented
    // GameState; independent successful Cipher.Start calls are not modeled.
    if blocked_count > 1 { return false; }

    let dead: HashSet<u8> = state.executed.iter()
        .chain(state.night_kills.iter()).copied().collect();
    let named_witch_positions: Vec<u8> = scenario.evil_positions.iter()
        .filter_map(|(&position, role)| {
            (normalize_role(role) == "witch").then_some(position)
        }).collect();
    let live_named_witches = named_witch_positions.iter()
        .filter(|position| !dead.contains(position)).count();
    let dead_named_witches = named_witch_positions.len() - live_named_witches;

    // An executed confirmed Evil may have no observed role. Scenario
    // generation represents that seat as `Unknown` after removing one role
    // identity from the deck, so preserve the possibility that the missing
    // identity was Witch. This is historical evidence only: an Unknown live
    // seat never supplies current Witch quota.
    let dead_unknown_slots = scenario.evil_positions.iter()
        .filter(|(position, role)| {
            dead.contains(position) && normalize_role(role) == "unknown"
        }).count();
    let deck_witch_count = state.deck.minions.iter()
        .filter(|role| normalize_role(role) == "witch").count();
    let unrepresented_witch_count = deck_witch_count
        .saturating_sub(named_witch_positions.len());
    let possible_unknown_dead_witches = dead_unknown_slots
        .min(unrepresented_witch_count);
    let has_historical_witch = dead_named_witches > 0 || possible_unknown_dead_witches > 0;

    // A live named Witch explains a current scalar block. Any real Witch death
    // calls Reduce(1), including either member of an ordinary duplicate pair,
    // so a marker alongside a dead Witch is historical rather than stacked.
    live_named_witches > 0 || has_historical_witch
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

const FORTUNE_TELLER_CURRENT_RULE: &str = "fortune_teller_native_v1";

fn fortune_teller_claim_matches(
    actor: u8,
    targets: [u8; 2],
    claimed_evil: bool,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let actual_evil = targets.iter().any(|&target| {
        effective_alignment(target, scenario, state) == EffectiveAlignment::Evil
    });
    let truthful_claim = truth_status(actor, scenario, state) == TruthStatus::Truthful;
    claimed_evil == (actual_evil == truthful_claim)
}

fn fortune_teller_native_targets(
    info: &serde_json::Map<String, serde_json::Value>,
    state: &GameState,
) -> Option<[u8; 2]> {
    let values = info.get("targets")?.as_array()?;
    if values.len() != 2 { return None; }
    let first = u8::try_from(values[0].as_u64()?).ok()?;
    let second = u8::try_from(values[1].as_u64()?).ok()?;
    if first == 0 || second > state.n_cards || first >= second { return None; }
    Some([first, second])
}

fn fortune_teller_native_observation_matches(
    actor: u8,
    observation: &serde_json::Map<String, serde_json::Value>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let Some(targets) = fortune_teller_native_targets(observation, state) else {
        return false;
    };
    let Some(claimed_evil) = observation.get("has_evil").and_then(|value| value.as_bool()) else {
        return false;
    };
    let expected_text = format!(
        "Is #{} or #{} Evil?: {}",
        targets[0], targets[1], if claimed_evil { "True" } else { "False" },
    );
    if observation.get("text").and_then(|value| value.as_str()) != Some(expected_text.as_str()) {
        return false;
    }
    fortune_teller_claim_matches(actor, targets, claimed_evil, scenario, state)
}

fn validate_fortune_teller(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    if state.fortune_teller_rule_version.as_deref() == Some(FORTUNE_TELLER_CURRENT_RULE) {
        let has_targets = card.info_parsed.contains_key("targets");
        let has_result = card.info_parsed.contains_key("has_evil");
        if has_targets != has_result || card.position == 0 || card.position > state.n_cards {
            return false;
        }

        let interruption = match card.info_parsed.get("shut_up_target") {
            None => false,
            Some(value) => {
                let Some(position) = value.as_u64() else { return false; };
                if position == 0 || position > u64::from(state.n_cards) { return false; }
                true
            }
        };
        if interruption && has_targets { return false; }

        let Some(observation_value) = card.info_parsed.get("observations") else {
            // A current unused active card carries no result fields. Any
            // current scalar result or interruption must have an explicit
            // chronological ledger, even when that ledger is empty.
            return !has_targets && !interruption;
        };
        let Some(observations) = observation_value.as_array() else { return false; };
        if observations.is_empty() { return interruption && !has_targets; }

        if !observations.iter().all(|value| {
            value.as_object().is_some_and(|observation| {
                fortune_teller_native_observation_matches(
                    card.position, observation, scenario, state,
                )
            })
        }) {
            return false;
        }

        if has_targets {
            let Some(latest) = observations.last().and_then(|value| value.as_object()) else {
                return false;
            };
            let Some(alias_targets) = fortune_teller_native_targets(&card.info_parsed, state) else {
                return false;
            };
            let Some(latest_targets) = fortune_teller_native_targets(latest, state) else {
                return false;
            };
            let alias_result = card.info_parsed.get("has_evil").and_then(|value| value.as_bool());
            let latest_result = latest.get("has_evil").and_then(|value| value.as_bool());
            let latest_text = latest.get("text").and_then(|value| value.as_str());
            return alias_targets == latest_targets
                && alias_result == latest_result
                && latest_text == Some(card.info_text.as_str());
        }

        return interruption;
    }

    // Archived fixtures contain only the scalar Boolean/pair and may preserve
    // click order rather than native ascending reference order. Keep that
    // historical predicate unchanged, including its conservative shape.
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

    // A truthful Medium can observe a Baker predecessor before conversion or
    // through BakerRuntimeData afterward. The shared history validator owns
    // that temporal fact; never infer it from the target's own (possibly
    // lying) speech.
    if !actual_match
        && is_good
        && truth == TruthStatus::Truthful
        && medium_uses_baker_history(claimed_pos, scenario, state)
    {
        actual_match = true;
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
    // Corrupted without removing this marker. Chancellor attempts it only on
    // the later real-Outcast anchor; reinitializing the first Villager target
    // does not mark that card.
    let mut affected = scenario.messed_up_by_evil.clone();
    // In this fingerprint every successful delayed demon kill installs status
    // 50. The only shipped resistance producer is Alchemist.OnInit, which
    // resists Corrupted (40), so no current night-kill history can reject 50.
    for &nk in &state.night_kills { affected.insert(nk); }

    if claimed_pos == 0 {
        let marked_count = (1..=state.n_cards)
            .filter(|position| affected.contains(position))
            .count();
        if truth == TruthStatus::Truthful { marked_count == 0 }
        else { marked_count == state.n_cards as usize }
    } else if claimed_pos > state.n_cards {
        false
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

fn judge_observation_matches(
    pos: u8,
    target: u8,
    claimed_lying: bool,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let truth = truth_status(pos, scenario, state);
    // Shipped Judge2 queries CheckLyingAppearance.  BluffAct, including the
    // path used by a corrupted Good Judge, deterministically inverts that same
    // predicate rather than producing an unconstrained result.
    let target_truth = truth_appearance_status(target, scenario, state);
    let actually_lying = target_truth == TruthStatus::Lying;

    if truth == TruthStatus::Truthful { claimed_lying == actually_lying }
    else { claimed_lying != actually_lying }
}

fn validate_judge(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    // Judge resets after Night. New state can retain every fired result under
    // `observations`; legacy fixtures keep the single top-level shape. Once
    // evidence is present its shape is strict so malformed records cannot
    // silently erase a constraint.
    if let Some(value) = card.info_parsed.get("observations") {
        let observations = match value.as_array() {
            Some(observations) => observations,
            None => return false,
        };
        if !observations.is_empty() {
            if card.position == 0 || card.position > state.n_cards {
                return false;
            }
            return observations.iter().all(|value| {
                let observation = match value.as_object() {
                    Some(observation) => observation,
                    None => return false,
                };
                let target = match info_pos(observation, "target") {
                    Some(target) if target > 0 && target <= state.n_cards => target,
                    _ => return false,
                };
                let claimed_lying = match info_bool(observation, "is_lying") {
                    Some(claimed_lying) => claimed_lying,
                    None => return false,
                };
                judge_observation_matches(
                    card.position,
                    target,
                    claimed_lying,
                    scenario,
                    state,
                )
            });
        }
    }

    let has_target = card.info_parsed.contains_key("target");
    let has_result = card.info_parsed.contains_key("is_lying");
    if !has_target && !has_result {
        return true;
    }
    if !has_target || !has_result || card.position == 0 || card.position > state.n_cards {
        return false;
    }

    let target = match info_pos(&card.info_parsed, "target") {
        Some(target) if target > 0 && target <= state.n_cards => target,
        _ => return false,
    };
    let claimed_lying = match info_bool(&card.info_parsed, "is_lying") {
        Some(claimed_lying) => claimed_lying,
        None => return false,
    };

    judge_observation_matches(
        card.position,
        target,
        claimed_lying,
        scenario,
        state,
    )
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
    if normalize_role(&card.apparent_role) == "baker" {
        // Ordinary Baker cards are validated together by the native Day/reveal
        // history. This scalar entry remains only for Poet's copied-role
        // delegation below.
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

const RAMBLER_MATCHES_TRUTHFUL: u8 = 1;
const RAMBLER_MATCHES_LYING: u8 = 2;
const RAMBLER_CURRENT_RULE: &str = "rambler2_shut_up";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct RamblerSourceAlternative {
    matchers: u8,
    /// This alternative assigns the physical source an anonymous natural
    /// Rambler identity and therefore consumes one real pool occurrence.
    anonymous_natural: bool,
}

#[derive(Debug, Default)]
struct RamblerSourceSupport {
    /// Matchers that must have been installed in this represented scenario.
    definite: u8,
    /// Complete matcher sets from viable identity/bluff assignments grouped
    /// into this scenario. Each entry is one alternative, not an additive
    /// union of unrelated anonymous-role worlds.
    possibilities: Vec<RamblerSourceAlternative>,
}

fn add_rambler_matcher(mask: &mut u8, act: bool) {
    *mask |= if act {
        RAMBLER_MATCHES_TRUTHFUL
    } else {
        RAMBLER_MATCHES_LYING
    };
}

fn rambler_source_support(pos: u8, scenario: &Scenario, state: &GameState) -> RamblerSourceSupport {
    let truth = truth_status(pos, scenario, state);
    let runtime_evil = is_evil_in_board_state(pos, scenario, state);
    let effective_role = state
        .executed_good_roles
        .get(&pos)
        .cloned()
        .or_else(|| effective_role_at(pos, scenario, state));
    let real_rambler = effective_role
        .as_deref()
        .is_some_and(|role| normalize_role(role) == "rambler");

    // These are the explicit non-null-bluff surfaces represented by Scenario.
    // Ordinary Evil roles also receive a bluff in the shipped setup. Shaman's
    // InitWithNoReset preserves that pointer, which matters when its copied
    // real role is Rambler on an Evil-aligned destination.
    let has_modeled_bluff = runtime_evil
        || scenario.doppelganger_position == Some(pos)
        || scenario.drunk_position == Some(pos)
        || effective_role.as_deref().is_some_and(|role| {
            matches!(normalize_role(role).as_str(), "doppelganger" | "drunk")
        });
    let apparent_rambler = state
        .card_at(pos)
        .is_some_and(|card| normalize_role(&card.apparent_role) == "rambler");
    // A normal Good Rambler displays its real data; equality alone does not
    // prove a second bluff-role dispatch. A represented disguiser does.
    let known_rambler_bluff = apparent_rambler && (!real_rambler || has_modeled_bluff);

    let mut definite = 0;
    if real_rambler {
        // Native Character.Act matrix:
        // truthful => real Act;
        // lying non-Evil => real BluffAct;
        // lying Evil with a non-null bluff => real Act, otherwise BluffAct.
        let real_act = truth == TruthStatus::Truthful || (runtime_evil && has_modeled_bluff);
        add_rambler_matcher(&mut definite, real_act);
    }
    if known_rambler_bluff {
        add_rambler_matcher(&mut definite, truth == TruthStatus::Truthful);
    }

    let mut possibilities = vec![RamblerSourceAlternative {
        matchers: definite,
        anonymous_natural: false,
    }];
    let deck_has_rambler = state
        .deck
        .villagers
        .iter()
        .chain(state.deck.outcasts.iter())
        .any(|role| normalize_role(role) == "rambler");

    // A hidden modeled disguiser has a non-null but unobserved bluff identity.
    // It can supply a Rambler bluff surface without consuming the natural
    // Rambler occurrence; asc83 demonstrates simultaneous real and fake
    // displayed Ramblers from a one-Rambler pool.
    if state.card_at(pos).is_none()
        && has_modeled_bluff
        && !known_rambler_bluff
        && deck_has_rambler
    {
        let mut with_rambler_bluff = definite;
        add_rambler_matcher(
            &mut with_rambler_bluff,
            truth == TruthStatus::Truthful,
        );
        possibilities.push(RamblerSourceAlternative {
            matchers: with_rambler_bluff,
            anonymous_natural: false,
        });
    }

    // Ordinary hidden Villager/Outcast assignments are intentionally grouped
    // in Scenario. Reuse scenario generation's exact pool/header accounting to
    // ask whether this particular physical seat can be the natural Rambler.
    if !real_rambler
        && crate::scenario::scenario_allows_anonymous_natural_outcast_role_at(
            pos, "Rambler", scenario, state,
        )
    {
        let mut natural_rambler = 0;
        add_rambler_matcher(
            &mut natural_rambler,
            truth == TruthStatus::Truthful,
        );
        possibilities.push(RamblerSourceAlternative {
            matchers: natural_rambler,
            anonymous_natural: true,
        });
    }

    let mut unique_possibilities = Vec::new();
    for possibility in possibilities {
        if !unique_possibilities.contains(&possibility) {
            unique_possibilities.push(possibility);
        }
    }
    RamblerSourceSupport {
        definite,
        possibilities: unique_possibilities,
    }
}

fn card_has_normal_clue(card: &CardInfo, current_rules: bool) -> bool {
    if card.info_parsed.is_empty() {
        return false;
    }
    let role = normalize_role(&card.apparent_role);
    if role == "rambler" {
        // Current capture writes `quote_observed=true`; the versioned
        // `silenced=false` form is retained only for an early live-parser
        // compatibility window. False/empty quote flags and the old
        // silencing metadata are not evidence that a Day action completed.
        return info_bool(&card.info_parsed, "quote_observed") == Some(true)
            || (current_rules && info_bool(&card.info_parsed, "silenced") == Some(false));
    }
    if role == "jester" && info_bool(&card.info_parsed, "silenced") == Some(true) {
        return false;
    }
    // `shut_up_target` is only the latest-value compatibility alias. Active
    // roles such as Judge can retain an earlier/later normal observation in
    // the same object, and that evidence still proves a callback did not
    // replace that action. A scalar by itself carries no such negative fact.
    card.info_parsed.iter().any(|(key, value)| match key.as_str() {
        "shut_up_target" | "silenced" | "silenced_by" | "quote_observed" => false,
        // An empty Judge history is the initialized no-info shape, not an
        // observed uninterrupted action. Non-empty history remains evidence
        // even when a later/earlier shut-up alias coexists on the same card.
        "observations" => value
            .as_array()
            .is_some_and(|observations| !observations.is_empty()),
        _ => true,
    })
}

fn rambler_required_sources_are_jointly_possible(
    required_by_source: &HashMap<u8, u8>,
    forbidden_by_source: &HashMap<u8, u8>,
    forbidden_anonymous_natural_sources: &HashSet<u8>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let mut requirements: Vec<(u8, Vec<RamblerSourceAlternative>)> = required_by_source
        .iter()
        .map(|(&source, &required_matchers)| {
            let support = rambler_source_support(source, scenario, state);
            let possibilities = support
                .possibilities
                .into_iter()
                .filter(|possibility| {
                    possibility.matchers & required_matchers == required_matchers
                        && possibility.matchers
                            & forbidden_by_source.get(&source).copied().unwrap_or(0)
                            == 0
                })
                .collect();
            (source, possibilities)
        })
        .collect();
    requirements.sort_unstable_by_key(|(source, _)| *source);
    if requirements
        .iter()
        .any(|(_, possibilities)| possibilities.is_empty())
    {
        return false;
    }

    fn search(
        index: usize,
        requirements: &[(u8, Vec<RamblerSourceAlternative>)],
        anonymous_natural_sources: &mut HashSet<u8>,
        forbidden_anonymous_natural_sources: &HashSet<u8>,
        scenario: &Scenario,
        state: &GameState,
    ) -> bool {
        if index == requirements.len() {
            return crate::scenario::scenario_allows_anonymous_natural_outcast_role_assignments(
                anonymous_natural_sources,
                "Rambler",
                forbidden_anonymous_natural_sources,
                scenario,
                state,
            );
        }

        let (source, possibilities) = &requirements[index];
        for possibility in possibilities {
            let inserted = possibility.anonymous_natural
                && anonymous_natural_sources.insert(*source);
            if search(
                index + 1,
                requirements,
                anonymous_natural_sources,
                forbidden_anonymous_natural_sources,
                scenario,
                state,
            ) {
                return true;
            }
            if inserted {
                anonymous_natural_sources.remove(source);
            }
        }
        false
    }

    search(
        0,
        &requirements,
        &mut HashSet::new(),
        forbidden_anonymous_natural_sources,
        scenario,
        state,
    )
}

fn validate_rambler_shut_ups(scenario: &Scenario, state: &GameState) -> bool {
    let current_rules = state.rambler_rule_version.as_deref() == Some(RAMBLER_CURRENT_RULE);
    let mut observations: Vec<(u8, u8)> = Vec::new();
    for observation in &state.rambler_shut_up_observations {
        if observation.speaker_position == 0
            || observation.speaker_position > state.n_cards
            || observation.shut_up_target == 0
            || observation.shut_up_target > state.n_cards
        {
            return false;
        }
        let public_pair = (observation.speaker_position, observation.shut_up_target);
        if !observations.contains(&public_pair) {
            observations.push(public_pair);
        }
    }
    let mut forbidden_by_source: HashMap<u8, u8> = HashMap::new();
    for card in &state.cards {
        if let Some(raw_source) = card.info_parsed.get("shut_up_target") {
            let Some(source) = raw_source
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .filter(|source| *source > 0 && *source <= state.n_cards)
            else {
                // Presence means the capture intended to assert a public
                // position. Never silently discard malformed scalar evidence
                // or wrap a signed/wide integer through `as u8`.
                return false;
            };
            if card.position == 0 || card.position > state.n_cards {
                return false;
            }
            let latest_alias = (card.position, source);
            if !observations.contains(&latest_alias) {
                observations.push(latest_alias);
            }
        }
    }
    let has_positive = !observations.is_empty();
    // Frozen fixtures without a version predate the redesign, so clue absence
    // is not evidence. A positive public replacement is self-provenance and is
    // still validated exactly for backward compatibility.
    if !current_rules && !has_positive {
        return true;
    }

    let mut required_by_source: HashMap<u8, u8> = HashMap::new();
    for (speaker, source) in observations {
        let Some(card) = state.card_at(speaker) else {
            return false;
        };
        if !adjacent_positions(speaker, state.n_cards).contains(&source) {
            return false;
        }
        let required_matcher =
            if truth_appearance_status(card.position, scenario, state) == TruthStatus::Truthful {
                RAMBLER_MATCHES_TRUTHFUL
            } else {
                RAMBLER_MATCHES_LYING
            };
        *required_by_source.entry(source).or_insert(0) |= required_matcher;
    }

    if current_rules {
        for card in &state.cards {
            if !card_has_normal_clue(card, true) {
                continue;
            }
            if card.position == 0 || card.position > state.n_cards {
                return false;
            }
            let appearance_truthful =
                truth_appearance_status(card.position, scenario, state) == TruthStatus::Truthful;
            let required_matcher = if appearance_truthful {
                RAMBLER_MATCHES_TRUTHFUL
            } else {
                RAMBLER_MATCHES_LYING
            };
            for source in adjacent_positions(card.position, state.n_cards) {
                *forbidden_by_source.entry(source).or_insert(0) |= required_matcher;
            }
        }
    }
    for (&source, &forbidden_matchers) in &forbidden_by_source {
        if rambler_source_support(source, scenario, state).definite & forbidden_matchers != 0 {
            return false;
        }
    }
    let forbidden_anonymous_natural_sources: HashSet<u8> = forbidden_by_source
        .iter()
        .filter_map(|(&source, &forbidden_matchers)| {
            let natural_matcher =
                if truth_status(source, scenario, state) == TruthStatus::Truthful {
                    RAMBLER_MATCHES_TRUTHFUL
                } else {
                    RAMBLER_MATCHES_LYING
                };
            (forbidden_matchers & natural_matcher != 0).then_some(source)
        })
        .collect();

    rambler_required_sources_are_jointly_possible(
        &required_by_source,
        &forbidden_by_source,
        &forbidden_anonymous_natural_sources,
        scenario,
        state,
    )
}

fn validate_slayer_results(scenario: &Scenario, state: &GameState) -> bool {
    let mut used_actors = HashSet::new();
    for result in &state.slayer_results {
        let slayer_pos = result.slayer_pos;
        let target_pos = result.target_pos;
        let killed = result.killed;

        if slayer_pos == 0
            || slayer_pos > state.n_cards
            || target_pos == 0
            || target_pos > state.n_cards
            || !used_actors.insert(slayer_pos)
        {
            return false;
        }
        let Some(actor) = state.card_at(slayer_pos) else {
            return false;
        };
        if normalize_role(&actor.apparent_role) != "slayer" {
            return false;
        }
        if !killed && (result.revealed_role.is_some() || result.was_evil.is_some()) {
            // The disabled/failure callback reports only the selected target;
            // neither current role nor physical runtime alignment is public.
            return false;
        }

        let slayer_evil_role = known_evil_role(slayer_pos, scenario, state);
        let slayer_is_evil = slayer_evil_role.is_some();
        let slayer_is_puppet = slayer_evil_role == Some("Puppet");
        let slayer_lies = truth_status(slayer_pos, scenario, state) == TruthStatus::Lying;
        let target_is_physically_evil = is_evil_in_board_state(target_pos, scenario, state);
        if result
            .was_evil
            .is_some_and(|observed| observed != target_is_physically_evil)
        {
            return false;
        }

        // Slayer's target predicate is registered alignment, not the physical
        // runtime alignment used by HP/objective bookkeeping. Native
        // GetRegisterAlignment prefers a live role override (public Wretch)
        // and otherwise falls back to physical runtime alignment. In
        // particular, authored Evil alignment on current Twin data is not a
        // register override and cannot make a runtime-Good body killable.
        let modeled_registered_evil =
            effective_alignment(target_pos, scenario, state) == EffectiveAlignment::Evil;
        let target_is_registered_evil = result
            .revealed_role
            .as_deref()
            .map(|role| target_is_physically_evil || normalize_role(role) == "wretch")
            .unwrap_or(modeled_registered_evil);

        if killed {
            if slayer_is_evil && !slayer_is_puppet { return false; }
            if slayer_lies { return false; }
            if !target_is_registered_evil { return false; }
            if let Some(revealed_role) = result.revealed_role.as_deref() {
                let role_matches = effective_role_at(target_pos, scenario, state)
                    .is_some_and(|actual| {
                        normalize_role(&actual) == normalize_role(revealed_role)
                    });
                // Until Scenario carries Twin's exact two-seat data swap, the
                // only defensible mismatch is one where current Twin data is
                // observed at the recipient, or the stable Twin actor reveals
                // the role received in exchange. Other mismatches still prune.
                let twin_can_explain_mismatch = twin_can_explain_current_role_mismatch(
                    target_pos,
                    revealed_role,
                    scenario,
                    state,
                );
                if !role_matches && !twin_can_explain_mismatch { return false; }
            }
        } else {
            let slayer_works = (!slayer_is_evil || slayer_is_puppet) && !slayer_lies;
            // A failed result has no current-role reveal, so retain only the
            // registered-alignment contradiction the existing model can
            // represent. A reachable runtime-Good Wretch endpoint is the one
            // pre-TwinTrace exception: Wretch data can move to the Evil Twin,
            // leaving current Twin/base registration on the Good Wretch body.
            let modeled_good_wretch_can_receive_twin = !target_is_physically_evil
                && effective_role_at(target_pos, scenario, state)
                    .is_some_and(|role| normalize_role(&role) == "wretch")
                && twin_may_have_replaced_current_data_at(target_pos, scenario, state);
            if slayer_works
                && modeled_registered_evil
                && !modeled_good_wretch_can_receive_twin
            {
                return false;
            }
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
    let shaman_endpoints = scenario.shaman_trace.as_ref().map(|trace| {
        HashSet::from([trace.source_position, trace.target_position])
    });

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
            counted_good_villager_positions.insert(pos);
            // Baker Day rewrites and both Shaman endpoints are projected back
            // to their one shared initial-role witness by `baker::history`.
            // Counting their final appearances here would either invent
            // duplicate Baker assets or commit to the erased Shaman identity.
            if normalize_role(role) != "baker"
                && !shaman_endpoints
                    .as_ref()
                    .is_some_and(|positions| positions.contains(&pos))
            {
                *good_villager_counts.entry(normalize_role(role)).or_insert(0) += 1;
            }
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

    // Structural Shaman trace checks remain here. Its initial role multiset is
    // validated jointly with Baker conversions, since either endpoint may be
    // overwritten again during a later Day reveal.
    if let Some(trace) = scenario.shaman_trace.as_ref() {
        let copied = normalize_role(&trace.copied_role);
        let previous_roles: Vec<String> = trace
            .target_previous_roles
            .iter()
            .map(|role| normalize_role(role))
            .collect();
        let previous_role_set: HashSet<&str> =
            previous_roles.iter().map(String::as_str).collect();
        let previous_runtime_class = trace
            .target_previous_roles
            .first()
            .map(|role| knowledge_base::shaman_erased_role_class(&trace.copied_role, role));
        if trace.source_position == trace.target_position
            || trace.source_position == 0
            || trace.target_position == 0
            || trace.source_position > state.n_cards
            || trace.target_position > state.n_cards
            || previous_roles.is_empty()
            || previous_role_set.len() != previous_roles.len()
            || trace.target_previous_roles.iter().any(|role| {
                Some(knowledge_base::shaman_erased_role_class(&trace.copied_role, role))
                    != previous_runtime_class
            })
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
                if observed.as_deref() != Some("baker")
                    && observed.as_deref() != Some(copied.as_str())
                {
                    return false;
                }
            }
        }
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
    let mut total_excess = 0i32;
    for (role, &count) in &good_villager_counts {
        let deck_count = deck_v_counts.get(role).copied().unwrap_or(0);
        if count > deck_count {
            total_excess += count - deck_count;
        }
    }
    let any_initial_role_multiset_fits = total_excess <= n_disguisers;
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
        let total_good_villagers = counted_good_villager_positions.len() as i32;
        if total_good_villagers > bvc as i32 + n_disguisers { return false; }
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
            twin_trace: None,
        }
    }

    fn base_state(n_cards: u8, cards: Vec<CardInfo>) -> GameState {
        let mut s = GameState::default();
        s.n_cards = n_cards;
        s.cards = cards;
        s
    }

    #[test]
    fn witch_block_marker_is_arbitrary_and_allows_self_block() {
        let mut state = base_state(7, vec![]);
        state.blocked_positions = vec![1];

        let mut witch_elsewhere = empty_scenario();
        witch_elsewhere.evil_positions.insert(7, "Witch".to_string());
        assert!(validate_witch_block_evidence(&witch_elsewhere, &state));

        let mut witch_on_blocked_seat = empty_scenario();
        witch_on_blocked_seat.evil_positions.insert(1, "Witch".to_string());
        assert!(validate_witch_block_evidence(&witch_on_blocked_seat, &state));
    }

    #[test]
    fn current_witch_block_marker_rejects_no_witch_scenario() {
        let mut state = base_state(3, vec![make_card(2, "Baker", json!({}))]);
        state.blocked_positions = vec![1];
        assert!(!validate_witch_block_evidence(&empty_scenario(), &state));
    }

    #[test]
    fn retained_witch_marker_is_historical_after_execution_or_night_death() {
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(3, "Witch".to_string());

        let mut executed = base_state(3, vec![]);
        executed.blocked_positions = vec![1];
        executed.executed = vec![3];
        assert!(validate_witch_block_evidence(&scenario, &executed));

        let mut night_killed = base_state(3, vec![]);
        night_killed.blocked_positions = vec![1];
        night_killed.night_kills = vec![3];
        assert!(validate_witch_block_evidence(&scenario, &night_killed));
    }

    #[test]
    fn unknown_executed_evil_can_preserve_historical_witch_identity() {
        let mut state = base_state(3, vec![]);
        state.blocked_positions = vec![1];
        state.executed = vec![3];
        state.confirmed_evil = vec![3];
        state.deck.minions = vec!["Witch".to_string(), "Minion".to_string()];

        let mut unknown_executed = empty_scenario();
        unknown_executed.evil_positions.insert(3, "Unknown".to_string());
        assert!(validate_witch_block_evidence(&unknown_executed, &state));

        state.deck.minions = vec!["Minion".to_string()];
        assert!(!validate_witch_block_evidence(&unknown_executed, &state));
    }

    #[test]
    fn witch_block_is_scalar_even_with_duplicate_witch_records() {
        let mut state = base_state(6, vec![]);
        state.blocked_positions = vec![1, 2];

        let mut two_witches = empty_scenario();
        two_witches.evil_positions.insert(6, "Witch".to_string());
        two_witches.evil_positions.insert(5, "Witch".to_string());
        assert!(!validate_witch_block_evidence(&two_witches, &state));

        state.blocked_positions = vec![1];
        assert!(validate_witch_block_evidence(&two_witches, &state));

        state.executed = vec![5];
        assert!(validate_witch_block_evidence(&two_witches, &state));

        state.night_kills = vec![6];
        assert!(validate_witch_block_evidence(&two_witches, &state));
    }

    #[test]
    fn judge_truthful_callback_reports_target_lying_appearance() {
        let says_lying = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": true}),
        );
        let says_truthful = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": false}),
        );
        let state = base_state(
            2,
            vec![says_lying.clone(), make_card(2, "Baker", json!({}))],
        );
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Minion".to_string());

        assert!(validate_judge(&says_lying, &scenario, &state));
        assert!(!validate_judge(&says_truthful, &scenario, &state));
    }

    #[test]
    fn corrupted_good_judge_deterministically_inverts_callback() {
        let says_truthful = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": false}),
        );
        let says_lying = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": true}),
        );
        let state = base_state(
            2,
            vec![says_truthful.clone(), make_card(2, "Baker", json!({}))],
        );
        let mut scenario = empty_scenario();
        scenario.corrupted.insert(1);
        scenario.evil_positions.insert(2, "Minion".to_string());

        assert!(validate_judge(&says_truthful, &scenario, &state));
        assert!(!validate_judge(&says_lying, &scenario, &state));
    }

    #[test]
    fn judge_uses_confessor_truthful_appearance_override() {
        let says_truthful = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": false}),
        );
        let says_lying = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": true}),
        );
        let state = base_state(
            2,
            vec![
                says_truthful.clone(),
                make_card(2, "Confessor", json!({})),
            ],
        );
        let mut corrupted = empty_scenario();
        corrupted.corrupted.insert(2);

        assert_eq!(truth_status(2, &corrupted, &state), TruthStatus::Lying);
        assert_eq!(
            truth_appearance_status(2, &corrupted, &state),
            TruthStatus::Truthful,
        );
        assert!(validate_judge(&says_truthful, &corrupted, &state));
        assert!(!validate_judge(&says_lying, &corrupted, &state));

        let mut evil = empty_scenario();
        evil.evil_positions.insert(2, "Minion".to_string());
        assert!(validate_judge(&says_truthful, &evil, &state));
        assert!(!validate_judge(&says_lying, &evil, &state));
    }

    #[test]
    fn judge_no_info_is_vacuous_but_partial_or_out_of_range_evidence_rejects() {
        let no_info = make_card(1, "Judge", json!({}));
        let empty_observations = make_card(1, "Judge", json!({"observations": []}));
        let target_only = make_card(1, "Judge", json!({"target": 1}));
        let result_only = make_card(1, "Judge", json!({"is_lying": true}));
        let out_of_range = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": false}),
        );
        let invalid_actor = make_card(
            2,
            "Judge",
            json!({"target": 1, "is_lying": false}),
        );
        let state = base_state(1, vec![no_info.clone()]);
        let mut scenario = empty_scenario();
        scenario.corrupted.insert(1);

        assert!(validate_judge(&no_info, &scenario, &state));
        assert!(validate_judge(&empty_observations, &scenario, &state));
        assert!(!validate_judge(&target_only, &scenario, &state));
        assert!(!validate_judge(&result_only, &scenario, &state));
        assert!(!validate_judge(&out_of_range, &scenario, &state));
        assert!(!validate_judge(&invalid_actor, &scenario, &state));
    }

    #[test]
    fn judge_rejects_malformed_repeated_observation_evidence() {
        let state = base_state(2, vec![make_card(1, "Judge", json!({}))]);
        let scenario = empty_scenario();
        let malformed = [
            json!({"observations": "not-an-array"}),
            json!({"observations": [false]}),
            json!({"observations": [{}]}),
            json!({"observations": [{"target": "2", "is_lying": false}]}),
            json!({"observations": [{"target": 2, "is_lying": "false"}]}),
            json!({"observations": [{"target": 0, "is_lying": false}]}),
            json!({"observations": [{"target": 3, "is_lying": false}]}),
        ];

        for info in malformed {
            let judge = make_card(1, "Judge", info);
            assert!(!validate_judge(&judge, &scenario, &state));
        }
    }

    #[test]
    fn judge_validates_every_reset_after_night_observation() {
        let repeated = make_card(
            1,
            "Judge",
            json!({
                "observations": [
                    {"target": 2, "is_lying": false},
                    {"target": 3, "is_lying": true}
                ]
            }),
        );
        let state = base_state(
            3,
            vec![
                repeated.clone(),
                make_card(2, "Baker", json!({})),
                make_card(3, "Baker", json!({})),
            ],
        );
        let mut compatible = empty_scenario();
        compatible.evil_positions.insert(3, "Minion".to_string());
        assert!(validate_judge(&repeated, &compatible, &state));

        let mut contradicts_first = empty_scenario();
        contradicts_first
            .evil_positions
            .insert(2, "Minion".to_string());
        assert!(!validate_judge(&repeated, &contradicts_first, &state));
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
    fn witness_uses_chancellor_anchor_not_first_villager_target() {
        let anchor_claim = make_card(
            1,
            "Witness",
            json!({"affected_position": 2}),
        );
        let first_target_claim = make_card(
            1,
            "Witness",
            json!({"affected_position": 3}),
        );
        let state = base_state(5, vec![anchor_claim.clone()]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Chancellor".to_string());
        scenario.messed_up_by_evil.insert(2);
        scenario.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![1],
            added_outcast_position: 3,
            added_outcast_role: "Bombardier".to_string(),
            affected_anchor_positions: vec![2],
        });

        assert_eq!(scenario.chancellor_original_villager_positions(), vec![3]);
        assert!(validate_witness(&anchor_claim, &scenario, &state));
        assert!(!validate_witness(&first_target_claim, &scenario, &state));
    }

    #[test]
    fn witness_no_claim_uses_truthful_empty_and_lying_full_marker_sets() {
        let no_claim = make_card(
            1,
            "Witness",
            json!({"affected_position": 0}),
        );
        let state = base_state(3, vec![no_claim.clone()]);

        assert!(validate_witness(&no_claim, &empty_scenario(), &state));

        let mut partial = empty_scenario();
        partial.messed_up_by_evil.insert(2);
        assert!(!validate_witness(&no_claim, &partial, &state));

        partial.corrupted.insert(1);
        assert!(!validate_witness(&no_claim, &partial, &state));

        partial.messed_up_by_evil.extend([1, 3]);
        assert!(validate_witness(&no_claim, &partial, &state));

        let invalid_claim = make_card(
            1,
            "Witness",
            json!({"affected_position": 4}),
        );
        assert!(!validate_witness(&invalid_claim, &partial, &state));
    }

    #[test]
    fn witness_positive_lie_chooses_unmarked_and_dead_markers_remain_visible() {
        let marked_claim = make_card(
            1,
            "Witness",
            json!({"affected_position": 2}),
        );
        let unmarked_claim = make_card(
            1,
            "Witness",
            json!({"affected_position": 3}),
        );
        let mut state = base_state(3, vec![marked_claim.clone()]);
        state.executed.push(2);

        let mut scenario = empty_scenario();
        scenario.messed_up_by_evil.insert(2);
        assert!(validate_witness(&marked_claim, &scenario, &state));

        scenario.corrupted.insert(1);
        assert!(!validate_witness(&marked_claim, &scenario, &state));
        assert!(validate_witness(&unmarked_claim, &scenario, &state));
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
    fn current_negative_evidence_rejects_a_normal_adjacent_clue() {
        let rambler = make_card(1, "Rambler", json!({"quote_observed": true}));
        let scout = make_card(2, "Scout", json!({"evil_role": "Pooka", "distance": 1}));
        let mut state = base_state(5, vec![rambler, scout]);
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());

        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));

        // Unversioned historical clues predate deterministic replacement, so
        // their absence remains non-evidence.
        state.rambler_rule_version = None;
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn legacy_positive_is_validated_without_activating_current_negative_evidence() {
        let mut state = base_state(
            3,
            vec![
                make_card(1, "Rambler", json!({"quote_observed": true})),
                make_card(2, "Scout", json!({"shut_up_target": 1})),
                make_card(3, "Baker", json!({"original_role": "original"})),
            ],
        );

        // Positive replacements are self-provenancing even in an old save,
        // but the neighboring normal clue predates reliable absence capture.
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));

        for non_exact in [" rambler2_shut_up", "RAMBLER2_SHUT_UP", "rambler2_shut_up "] {
            state.rambler_rule_version = Some(non_exact.to_string());
            assert!(
                validate_rambler_shut_ups(&empty_scenario(), &state),
                "non-exact marker {non_exact:?} must remain legacy-compatible",
            );
        }

        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn rambler_latest_alias_does_not_hide_coexisting_judge_evidence() {
        let source = make_card(1, "Rambler", json!({}));
        let judge = make_card(
            2,
            "Judge",
            json!({
                "target": 4,
                "target_was_truthful": true,
                "shut_up_target": 1
            }),
        );
        let mut state = base_state(5, vec![source, judge]);
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());

        // The positive alias proves the Rambler callback, while the retained
        // normal Judge result claims an un-interrupted action at the same
        // appearance. Both public facts must be checked, so this is impossible.
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn empty_judge_and_rambler_metadata_are_not_normal_clue_evidence() {
        let source = make_card(1, "Rambler", json!({}));
        let judge = make_card(
            2,
            "Judge",
            json!({
                "observations": [],
                "silenced_by": null,
                "quote_observed": false
            }),
        );
        let mut state = base_state(5, vec![source, judge]);
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());

        assert!(!card_has_normal_clue(&state.cards[1], true));
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));

        state.cards[1].info_parsed = json!({
            "observations": [{"target": 4, "is_lying": false}],
            "silenced_by": 1
        })
        .as_object()
        .unwrap()
        .clone();
        assert!(card_has_normal_clue(&state.cards[1], true));
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));

        let metadata_only_rambler = make_card(
            3,
            "Rambler",
            json!({"silenced": true, "silenced_by": 2, "quote_observed": false}),
        );
        assert!(!card_has_normal_clue(&metadata_only_rambler, true));
    }

    #[test]
    fn rambler_history_survives_a_later_normal_role_result() {
        let source = make_card(1, "Rambler", json!({}));
        let judge = make_card(
            2,
            "Judge",
            json!({"target": 4, "target_was_truthful": true}),
        );
        let mut state = base_state(5, vec![source, judge]);
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        state.rambler_shut_up_observations = vec![crate::types::RamblerShutUpObservation {
            speaker_position: 2,
            shut_up_target: 1,
        }];

        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));

        // With no later normal observation, the same historical interruption
        // is sufficient positive evidence even though the latest alias is gone.
        state.cards[1].info_parsed.clear();
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn rambler_history_and_latest_alias_merge_all_observed_sources() {
        let mut state = base_state(
            3,
            vec![
                make_card(1, "Rambler", json!({})),
                make_card(2, "Scout", json!({"shut_up_target": 3})),
                make_card(3, "Rambler", json!({})),
            ],
        );
        state.rambler_shut_up_observations = vec![
            crate::types::RamblerShutUpObservation {
                speaker_position: 2,
                shut_up_target: 1,
            },
            // Exact duplicates are harmless validation-wise and remain legal
            // append-only history records.
            crate::types::RamblerShutUpObservation {
                speaker_position: 2,
                shut_up_target: 1,
            },
        ];

        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));

        state.rambler_shut_up_observations[0].speaker_position = 4;
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn truthful_shut_up_target_requires_an_adjacent_act_surface() {
        let rambler = make_card(1, "Rambler", json!({"quote_observed": true}));
        let scout = make_card(2, "Scout", json!({"shut_up_target": 1}));
        let state = base_state(5, vec![rambler, scout]);

        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));

        let mut fake_rambler = empty_scenario();
        fake_rambler.evil_positions.insert(1, "Puppeteer".to_string());
        assert!(!validate_rambler_shut_ups(&fake_rambler, &state));

        let non_adjacent = base_state(
            5,
            vec![
                make_card(1, "Rambler", json!({"quote_observed": true})),
                make_card(3, "Scout", json!({"shut_up_target": 1})),
            ],
        );
        assert!(!validate_rambler_shut_ups(
            &empty_scenario(),
            &non_adjacent,
        ));
    }

    #[test]
    fn confessor_disguise_uses_apparent_truth_for_rambler() {
        let rambler = make_card(1, "Rambler", json!({"quote_observed": true}));
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
    fn rambler_source_modes_follow_native_real_and_bluff_dispatch() {
        let rambler = make_card(1, "Rambler", json!({"quote_observed": true}));
        let state = base_state(1, vec![rambler]);

        assert_eq!(
            rambler_source_support(1, &empty_scenario(), &state).definite,
            RAMBLER_MATCHES_TRUTHFUL,
        );

        let mut corrupted_real = empty_scenario();
        corrupted_real.corrupted.insert(1);
        assert_eq!(
            rambler_source_support(1, &corrupted_real, &state).definite,
            RAMBLER_MATCHES_LYING,
        );

        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(1);
        assert_eq!(
            rambler_source_support(1, &puppet, &state).definite,
            RAMBLER_MATCHES_TRUTHFUL,
        );
        puppet.corrupted.insert(1);
        assert_eq!(
            rambler_source_support(1, &puppet, &state).definite,
            RAMBLER_MATCHES_LYING,
        );

        let mut doppelganger = empty_scenario();
        doppelganger.doppelganger_position = Some(1);
        assert_eq!(
            rambler_source_support(1, &doppelganger, &state).definite,
            RAMBLER_MATCHES_TRUTHFUL,
        );
        doppelganger.corrupted.insert(1);
        assert_eq!(
            rambler_source_support(1, &doppelganger, &state).definite,
            RAMBLER_MATCHES_LYING,
        );

        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(1);
        assert_eq!(
            rambler_source_support(1, &drunk, &state).definite,
            RAMBLER_MATCHES_LYING,
        );

        let mut evil = empty_scenario();
        evil.evil_positions.insert(1, "Pooka".to_string());
        assert_eq!(
            rambler_source_support(1, &evil, &state).definite,
            RAMBLER_MATCHES_LYING,
        );
    }

    #[test]
    fn lying_target_requires_a_rambler_bluffact_surface() {
        let rambler = make_card(1, "Rambler", json!({"quote_observed": true}));
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
    fn hidden_natural_rambler_uses_pool_multiplicity_and_header_capacity() {
        let target = make_card(2, "Scout", json!({"shut_up_target": 1}));
        let mut state = base_state(5, vec![target.clone()]);
        state.deck.outcasts = vec!["Rambler".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));

        state.deck.outcasts.clear();
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));

        state.deck.outcasts = vec!["Rambler".to_string()];
        state.board_outcast_count = Some(0);
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));

        state.board_outcast_count = Some(1);
        state.cards.push(make_card(4, "Rambler", json!({"quote_observed": true})));
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));

        state.deck.outcasts.push("Rambler".to_string());
        state.board_outcast_count = Some(2);
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn multiple_hidden_rambler_sources_share_one_joint_natural_pool_budget() {
        let mut state = base_state(
            6,
            vec![
                make_card(2, "Scout", json!({"shut_up_target": 1})),
                make_card(5, "Baker", json!({"shut_up_target": 4})),
            ],
        );
        state.deck.outcasts = vec!["Rambler".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;

        // Independent existentials would incorrectly reuse the only natural
        // Rambler at both named physical sources.
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));

        state.deck.outcasts.push("Rambler".to_string());
        state.board_outcast_count = Some(2);
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));

        // A represented Evil Rambler bluff is a separate physical install and
        // does not consume a second natural pool occurrence.
        state.deck.outcasts.pop();
        state.board_outcast_count = Some(1);
        state.cards.push(make_card(4, "Rambler", json!({})));
        let mut one_natural_one_fake = empty_scenario();
        one_natural_one_fake
            .evil_positions
            .insert(4, "Twin_Minion".to_string());
        one_natural_one_fake
            .evil_positions
            .insert(5, "Puppeteer".to_string());
        assert!(validate_rambler_shut_ups(&one_natural_one_fake, &state));
    }

    #[test]
    fn forced_anonymous_rambler_source_respects_other_normal_neighbor_evidence() {
        let mut state = base_state(
            5,
            vec![
                make_card(2, "Scout", json!({"shut_up_target": 1})),
                make_card(5, "Baker", json!({"original_role": "Poet"})),
            ],
        );
        state.deck.outcasts = vec!["Rambler".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());

        // #2's positive replacement forces hidden #1 to have installed an Act
        // matcher, which would also have replaced truthful adjacent #5's
        // retained Baker output. The anonymous alternative must honor both.
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn current_normal_clue_rejects_a_forced_hidden_rambler_without_any_positive() {
        let mut state = base_state(
            3,
            vec![make_card(
                1,
                "Baker",
                json!({"original_role": "original"}),
            )],
        );
        state.deck.villagers = vec!["Baker".to_string()];
        state.deck.outcasts = vec!["Rambler".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());

        // Both hidden seats neighbor #1. The exact O=1 header forces the sole
        // natural Rambler into one of them, where its Act callback would have
        // replaced the retained Baker clue.
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn hidden_rambler_completion_preserves_mandatory_plague_doctor_occupancy() {
        let mut state = base_state(
            3,
            vec![make_card(2, "Scout", json!({"shut_up_target": 1}))],
        );
        state.deck.outcasts = vec!["Plague_Doctor".to_string(), "Rambler".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        state.pd_corruption_target = Some(3);
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());

        // The known Start target proves that the sole ordinary-Outcast seat
        // was Plague Doctor. It cannot simultaneously be hidden Rambler #1.
        assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn malformed_rambler_positions_are_rejected_without_wrapping_or_panicking() {
        for malformed in [
            json!(-255),
            json!(0),
            json!(4),
            json!(257),
            json!("1"),
            json!(null),
            json!(1.0),
            json!(1.5),
            json!(true),
            json!([]),
            json!({}),
        ] {
            let mut state = base_state(
                3,
                vec![
                    make_card(1, "Rambler", json!({})),
                    make_card(2, "Scout", json!({"shut_up_target": malformed})),
                ],
            );
            state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
            assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
        }

        let mut zero_speaker = base_state(3, vec![make_card(0, "Scout", json!({}))]);
        zero_speaker.rambler_shut_up_observations =
            vec![crate::types::RamblerShutUpObservation {
                speaker_position: 0,
                shut_up_target: 1,
            }];
        assert!(!validate_rambler_shut_ups(
            &empty_scenario(),
            &zero_speaker,
        ));

        let mut out_of_range_source = base_state(3, vec![make_card(2, "Scout", json!({}))]);
        out_of_range_source.rambler_shut_up_observations =
            vec![crate::types::RamblerShutUpObservation {
                speaker_position: 2,
                shut_up_target: 4,
            }];
        assert!(!validate_rambler_shut_ups(
            &empty_scenario(),
            &out_of_range_source,
        ));

        let mut zero_cards = base_state(0, vec![]);
        zero_cards.rambler_shut_up_observations =
            vec![crate::types::RamblerShutUpObservation {
                speaker_position: 1,
                shut_up_target: 1,
            }];
        assert!(!validate_rambler_shut_ups(
            &empty_scenario(),
            &zero_cards,
        ));
    }

    #[test]
    fn shaman_copied_good_rambler_reserves_its_natural_pool_occurrence() {
        let mut state = base_state(
            5,
            vec![make_card(2, "Scout", json!({"shut_up_target": 1}))],
        );
        state.deck.outcasts = vec!["Rambler".to_string()];
        state.board_outcast_count = Some(2);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;

        let mut copied = empty_scenario();
        copied.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 3,
            target_position: 4,
            copied_role: "Rambler".to_string(),
            target_previous_roles: vec!["Baker".to_string()],
        });
        assert!(!validate_rambler_shut_ups(&copied, &state));

        state.deck.outcasts.push("Rambler".to_string());
        assert!(validate_rambler_shut_ups(&copied, &state));
    }

    #[test]
    fn hidden_disguised_rambler_uses_source_mode_without_consuming_pool_copy() {
        let target = make_card(2, "Baker", json!({"shut_up_target": 1}));
        let mut state = base_state(5, vec![target]);
        state.deck.outcasts = vec!["Rambler".to_string()];

        let mut source_and_target_evil = empty_scenario();
        source_and_target_evil
            .evil_positions
            .insert(1, "Twin_Minion".to_string());
        source_and_target_evil
            .evil_positions
            .insert(2, "Puppeteer".to_string());
        assert!(validate_rambler_shut_ups(&source_and_target_evil, &state));

        source_and_target_evil.evil_positions.remove(&2);
        assert!(!validate_rambler_shut_ups(&source_and_target_evil, &state));
    }

    #[test]
    fn shaman_copied_evil_rambler_can_install_real_and_stale_bluff_matchers() {
        let targets = vec![
            make_card(2, "Scout", json!({"shut_up_target": 1})),
            make_card(5, "Baker", json!({"shut_up_target": 1})),
        ];
        let mut state = base_state(5, targets);
        state.deck.outcasts = vec!["Rambler".to_string()];

        let mut copied = empty_scenario();
        copied.evil_positions.insert(1, "Pooka".to_string());
        copied.evil_positions.insert(5, "Puppeteer".to_string());
        copied.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 1,
            target_position: 3,
            copied_role: "Rambler".to_string(),
            target_previous_roles: vec!["Baker".to_string()],
        });

        // With a non-Rambler displayed bluff, the Evil real role installs Act
        // only, so one physical source cannot explain both appearance modes.
        state.cards.push(make_card(1, "Baker", json!({})));
        assert!(!validate_rambler_shut_ups(&copied, &state));

        // A stale Rambler bluff dispatches BluffAct after the real Act and the
        // physical card installs both matchers.
        state.cards.retain(|card| card.position != 1);
        state.cards.push(make_card(1, "Rambler", json!({"quote_observed": true})));
        assert!(validate_rambler_shut_ups(&copied, &state));
    }

    #[test]
    fn persistent_callbacks_survive_source_death_and_any_final_writer_is_possible() {
        let mut state = base_state(
            3,
            vec![
                make_card(1, "Rambler", json!({})),
                make_card(2, "Scout", json!({"shut_up_target": 1})),
                make_card(3, "Rambler", json!({})),
            ],
        );
        state.executed = vec![1];
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));

        state.cards[1].info_parsed = json!({"shut_up_target": 3})
            .as_object()
            .unwrap()
            .clone();
        // Cross-card same-delay install order is scheduler-owned and not
        // represented, so either matching adjacent callback may be last.
        assert!(validate_rambler_shut_ups(&empty_scenario(), &state));
    }

    #[test]
    fn rambler_quote_and_legacy_unsilenced_marker_are_negative_evidence_only_when_current() {
        for info in [json!({"quote_observed": true}), json!({"silenced": false})] {
            let source = make_card(1, "Rambler", json!({"quote_observed": true}));
            let target = make_card(2, "Rambler", info);
            let mut state = base_state(3, vec![source, target]);
            assert!(validate_rambler_shut_ups(&empty_scenario(), &state));
            state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
            assert!(!validate_rambler_shut_ups(&empty_scenario(), &state));
        }
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
            affected_anchor_positions: vec![],
        });
        scenario.evil_positions.insert(2, "Pooka".to_string());
        assert!(scenario.corrupted.is_empty());

        let mut slayer_state = base_state(3, vec![make_card(1, "Slayer", json!({}))]);
        slayer_state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: true,
            revealed_role: None,
            was_evil: None,
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
        // The shared history (rather than this scalar dispatcher) owns Baker
        // truth and the legacy parser's original/Baker ambiguity.
        assert!(validate_baker(&baker, &scenario, &baker_state));
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
            was_evil: None,
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
            was_evil: None,
        });
        let mut generated_wretch = empty_scenario();
        generated_wretch.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 2,
            added_outcast_role: "Wretch".to_string(),
            affected_anchor_positions: vec![],
        });
        generated_wretch.evil_positions.insert(3, "Chancellor".to_string());
        assert!(validate_slayer_results(&generated_wretch, &generated_state));
    }

    #[test]
    fn slayer_uses_registered_alignment_and_bypasses_knight_protection_only_for_evil() {
        let mut state = base_state(
            2,
            vec![
                make_card(1, "Slayer", json!({})),
                make_card(2, "Knight", json!({})),
            ],
        );
        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: false,
            revealed_role: None,
            was_evil: None,
        });

        let clean_knight = empty_scenario();
        assert!(validate_slayer_results(&clean_knight, &state));
        state.slayer_results[0].killed = true;
        state.slayer_results[0].revealed_role = Some("Knight".to_string());
        assert!(!validate_slayer_results(&clean_knight, &state));

        let mut corrupted_knight = clean_knight.clone();
        corrupted_knight.corrupted.insert(2);
        assert!(!validate_slayer_results(&corrupted_knight, &state));

        let mut evil_disguise = empty_scenario();
        evil_disguise.evil_positions.insert(2, "Shaman".to_string());
        state.slayer_results[0].revealed_role = Some("Shaman".to_string());
        assert!(validate_slayer_results(&evil_disguise, &state));
    }

    #[test]
    fn slayer_runtime_alignment_evidence_is_independent_of_registered_alignment() {
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
            was_evil: Some(false),
        });

        // Wretch is physically Good but registered Evil, so the kill and the
        // public Good HP outcome are jointly consistent.
        assert!(validate_slayer_results(&empty_scenario(), &state));
        state.slayer_results[0].was_evil = Some(true);
        assert!(!validate_slayer_results(&empty_scenario(), &state));

        let mut physical_evil = empty_scenario();
        physical_evil.evil_positions.insert(2, "Shaman".to_string());
        state.slayer_results[0].revealed_role = Some("Shaman".to_string());
        assert!(validate_slayer_results(&physical_evil, &state));
        state.slayer_results[0].was_evil = Some(false);
        assert!(!validate_slayer_results(&physical_evil, &state));
    }

    #[test]
    fn slayer_unknown_runtime_alignment_keeps_each_registered_kill_branch() {
        let mut state = base_state(
            4,
            vec![
                make_card(1, "Slayer", json!({})),
                make_card(2, "Wretch", json!({})),
            ],
        );
        state.deck.villagers = vec!["Baker".to_string()];
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: true,
            revealed_role: Some("Wretch".to_string()),
            was_evil: None,
        });

        // A no-damage public outcome does not distinguish a runtime-Good
        // Wretch from a runtime-Evil Twin body holding current Wretch data.
        assert!(validate_slayer_results(&empty_scenario(), &state));
        let mut stable_twin = empty_scenario();
        stable_twin
            .evil_positions
            .insert(2, "Twin Minion".to_string());
        assert!(validate_slayer_results(&stable_twin, &state));

        // The same unknown alignment preserves a stable runtime-Evil Twin
        // carrying an ordinary authored Good role.
        state.slayer_results[0].revealed_role = Some("Baker".to_string());
        assert!(validate_slayer_results(&stable_twin, &state));

        // Current Twin data itself supplies no registered-Evil override. A
        // runtime-Good recipient cannot produce a kill, while a runtime-Evil
        // recipient remains a possible registered-Evil branch.
        state.slayer_results[0].revealed_role = Some("Twin Minion".to_string());
        assert!(!validate_slayer_results(&empty_scenario(), &state));
        let mut runtime_evil_recipient = empty_scenario();
        runtime_evil_recipient
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(validate_slayer_results(&runtime_evil_recipient, &state));
    }

    #[test]
    fn slayer_twin_role_mismatches_preserve_physical_registration_boundary() {
        let mut state = base_state(
            3,
            vec![
                make_card(1, "Slayer", json!({})),
                make_card(2, "Baker", json!({})),
            ],
        );
        state.deck.villagers = vec!["Baker".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: true,
            revealed_role: Some("Twin Minion".to_string()),
            was_evil: Some(false),
        });

        // A runtime-Good recipient can hold current Twin data, but Twin has no
        // register override. Authored Evil alignment on the current data does
        // not make the physical Good body killable.
        assert!(!validate_slayer_results(&empty_scenario(), &state));

        let mut runtime_evil_neighbor = empty_scenario();
        runtime_evil_neighbor
            .evil_positions
            .insert(2, "Pooka".to_string());
        state.slayer_results[0].was_evil = Some(true);
        assert!(validate_slayer_results(&runtime_evil_neighbor, &state));

        let mut stable_twin = empty_scenario();
        stable_twin
            .evil_positions
            .insert(2, "Twin Minion".to_string());
        state.slayer_results[0].revealed_role = Some("Baker".to_string());
        state.slayer_results[0].was_evil = Some(true);
        assert!(validate_slayer_results(&stable_twin, &state));

        // Merely having Twin in the deck must not waive an unrelated current
        // role mismatch on a non-Twin stable Evil.
        let mut unrelated_evil = empty_scenario();
        unrelated_evil.evil_positions.insert(2, "Pooka".to_string());
        state.slayer_results[0].revealed_role = Some("Shaman".to_string());
        assert!(!validate_slayer_results(&unrelated_evil, &state));

        // Without authored Twin support, a Good Baker cannot reveal Twin or
        // enter Slayer's registered-Evil kill branch.
        state.deck.minions.clear();
        state.slayer_results[0].revealed_role = Some("Twin Minion".to_string());
        state.slayer_results[0].was_evil = Some(false);
        assert!(!validate_slayer_results(&empty_scenario(), &state));
    }

    #[test]
    fn failed_slayer_results_expose_neither_role_nor_runtime_alignment() {
        let mut state = base_state(
            4,
            vec![
                make_card(1, "Slayer", json!({})),
                make_card(2, "Knight", json!({})),
            ],
        );
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: false,
            revealed_role: None,
            was_evil: Some(false),
        });
        assert!(!validate_slayer_results(&empty_scenario(), &state));

        state.slayer_results[0].was_evil = None;
        // A runtime-Good recipient holding current Twin data still registers
        // Good, so the failed branch remains possible before TwinTrace exists.
        assert!(validate_slayer_results(&empty_scenario(), &state));

        // Conversely, the stable runtime-Evil Twin body remains registered
        // Evil even while carrying a received Good current role.
        let mut stable_twin = empty_scenario();
        stable_twin
            .evil_positions
            .insert(2, "Twin Minion".to_string());
        stable_twin.evil_positions.insert(3, "Pooka".to_string());
        assert!(!validate_slayer_results(&stable_twin, &state));

        state.cards[1].apparent_role = "Wretch".to_string();
        assert!(!validate_slayer_results(&empty_scenario(), &state));
    }

    #[test]
    fn failed_slayer_allows_only_a_reachable_good_wretch_twin_recipient() {
        let mut state = base_state(
            5,
            vec![
                make_card(1, "Slayer", json!({})),
                make_card(2, "Wretch", json!({})),
            ],
        );
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: false,
            revealed_role: None,
            was_evil: None,
        });

        let mut reachable = empty_scenario();
        reachable.evil_positions.insert(3, "Pooka".to_string());
        reachable
            .evil_positions
            .insert(5, "Twin Minion".to_string());
        assert!(twin_may_have_replaced_current_data_at(2, &reachable, &state));
        assert!(validate_slayer_results(&reachable, &state));

        // With no registered/real Demon adjacent to #2, its Wretch data cannot
        // have moved away and the working Slayer failure is contradictory.
        let mut unreachable = empty_scenario();
        unreachable.evil_positions.insert(4, "Pooka".to_string());
        unreachable
            .evil_positions
            .insert(5, "Twin Minion".to_string());
        assert!(!twin_may_have_replaced_current_data_at(2, &unreachable, &state));
        assert!(!validate_slayer_results(&unreachable, &state));

        // Physical runtime Evil remains registered Evil regardless of the
        // unmodeled current-data swap and can never take the failure branch.
        let mut physical_evil = reachable;
        physical_evil
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(!validate_slayer_results(&physical_evil, &state));
    }

    #[test]
    fn shaman_copied_knight_precedes_preserved_evil_identity_for_role_surfaces() {
        let mut state = base_state(
            3,
            vec![
                make_card(1, "Slayer", json!({})),
                make_card(2, "Knight", json!({})),
                make_card(3, "Knight", json!({})),
            ],
        );
        state.deck.demons = vec!["Lilis".to_string()];
        state.night_kills = vec![2];
        state.night_kill_evil_count = 1;

        let mut copied_knight = empty_scenario();
        copied_knight.evil_positions.insert(2, "Pooka".to_string());
        copied_knight.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 3,
            target_position: 2,
            copied_role: "Knight".to_string(),
            target_previous_roles: vec!["Pooka".to_string()],
        });

        assert_eq!(
            effective_role_at(2, &copied_knight, &state).as_deref(),
            Some("Knight")
        );
        assert_eq!(get_real_role(2, &copied_knight, &state), "Knight");
        assert_eq!(
            effective_alignment(2, &copied_knight, &state),
            EffectiveAlignment::Evil
        );
        assert!(validate_lilis_night_kills(&copied_knight, &state));

        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: true,
            revealed_role: Some("Knight".to_string()),
            was_evil: None,
        });
        assert!(validate_slayer_results(&copied_knight, &state));
        state.slayer_results[0].revealed_role = Some("Pooka".to_string());
        assert!(!validate_slayer_results(&copied_knight, &state));
    }

    #[test]
    fn slayer_rejects_malformed_actor_reuse_and_failed_reveal_shape() {
        let mut state = base_state(
            2,
            vec![
                make_card(1, "Slayer", json!({})),
                make_card(2, "Knight", json!({})),
            ],
        );
        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: false,
            revealed_role: Some("Knight".to_string()),
            was_evil: None,
        });
        assert!(!validate_slayer_results(&empty_scenario(), &state));

        state.slayer_results[0].revealed_role = None;
        state.slayer_results.push(state.slayer_results[0].clone());
        assert!(!validate_slayer_results(&empty_scenario(), &state));

        state.slayer_results.truncate(1);
        state.cards[0].apparent_role = "Bard".to_string();
        assert!(!validate_slayer_results(&empty_scenario(), &state));
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
            affected_anchor_positions: vec![],
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
        state.deck.demons = vec!["Lilis".to_string()];
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
            affected_anchor_positions: vec![],
        });
        assert!(check_scenario(&generated_drunk, &state));
    }

    #[test]
    fn lilis_successful_death_rejects_self_and_protected_knights() {
        let mut state = base_state(3, vec![make_card(1, "Knight", json!({}))]);
        state.deck.villagers = vec!["Knight".to_string()];
        state.deck.demons = vec!["Lilis".to_string()];
        state.night_kills = vec![1];

        let clean_knight = empty_scenario();
        assert!(!validate_lilis_night_kills(&clean_knight, &state));

        let mut corrupted_knight = clean_knight.clone();
        corrupted_knight.corrupted.insert(1);
        assert!(validate_lilis_night_kills(&corrupted_knight, &state));

        let mut evil_disguise = empty_scenario();
        evil_disguise.evil_positions.insert(1, "Shaman".to_string());
        state.night_kill_evil_count = 1;
        assert!(validate_lilis_night_kills(&evil_disguise, &state));

        let mut lilis_self = empty_scenario();
        lilis_self.evil_positions.insert(1, "Lilis".to_string());
        assert!(!validate_lilis_night_kills(&lilis_self, &state));
    }

    #[test]
    fn lilis_self_protection_uses_scenario_physical_multiplicity() {
        let mut state = base_state(2, vec![]);
        state.n_evil = 1;
        state.deck.demons = vec!["Lilis".to_string(), "Lilis".to_string()];
        state.board_demon_count = Some(1);
        state.night_kills = vec![1];
        state.night_kill_evil_count = 1;

        let mut one_physical_lilis = empty_scenario();
        one_physical_lilis.evil_positions.insert(1, "Lilis".to_string());
        assert!(!validate_lilis_night_kills(&one_physical_lilis, &state));

        state.n_evil = 2;
        state.board_demon_count = Some(2);
        let mut two_physical_lilis = one_physical_lilis.clone();
        two_physical_lilis.evil_positions.insert(2, "Lilis".to_string());
        assert!(validate_lilis_night_kills(&two_physical_lilis, &state));

        state.night_kills = vec![1, 2];
        state.night_kill_evil_count = 2;
        assert!(!validate_lilis_night_kills(&two_physical_lilis, &state));

        state.night_kills = vec![1];
        state.night_kill_evil_count = 1;
        state.executed = vec![2];
        state.confirmed_evil = vec![2];
        assert!(validate_lilis_night_kills(&one_physical_lilis, &state));
    }

    #[test]
    fn lilis_twin_waiver_requires_a_reachable_demon_neighbor_endpoint() {
        let mut state = base_state(4, vec![]);
        state.n_evil = 3;
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Lilis".to_string(), "Pooka".to_string()];
        state.night_kills = vec![2];
        state.night_kill_evil_count = 1;

        let mut reachable = empty_scenario();
        reachable
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        reachable.evil_positions.insert(2, "Lilis".to_string());
        reachable.evil_positions.insert(3, "Pooka".to_string());

        // Pooka #3 can select adjacent #2 as Twin's recipient endpoint. Lilis
        // data and its later self-protection move to the original Twin body,
        // leaving stable-Lilis #2 as killable current Twin data.
        assert!(twin_may_have_replaced_current_data_at(2, &reachable, &state));
        assert!(validate_lilis_night_kills(&reachable, &state));

        let mut untyped_twin_origin = reachable.clone();
        untyped_twin_origin
            .evil_positions
            .insert(1, "Unknown".to_string());
        assert!(twin_may_have_replaced_current_data_at(
            2,
            &untyped_twin_origin,
            &state,
        ));

        let mut one_untyped_seat = empty_scenario();
        one_untyped_seat
            .evil_positions
            .insert(3, "Unknown".to_string());
        assert!(!twin_may_have_replaced_current_data_at(
            2,
            &one_untyped_seat,
            &state,
        ));

        let mut unreachable = reachable.clone();
        unreachable.evil_positions.remove(&3);
        unreachable.evil_positions.insert(4, "Pooka".to_string());
        assert!(!twin_may_have_replaced_current_data_at(2, &unreachable, &state));
        assert!(!validate_lilis_night_kills(&unreachable, &state));

        // A lone Lilis cannot choose itself as its own adjacent endpoint. On a
        // two-card Twin/Lilis board both previous/next occurrences are Twin.
        state.n_cards = 2;
        state.n_evil = 2;
        state.deck.demons = vec!["Lilis".to_string()];
        let mut twin_and_lilis = empty_scenario();
        twin_and_lilis
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        twin_and_lilis
            .evil_positions
            .insert(2, "Lilis".to_string());
        assert!(!twin_may_have_replaced_current_data_at(
            2,
            &twin_and_lilis,
            &state,
        ));
        assert!(!validate_lilis_night_kills(&twin_and_lilis, &state));
    }

    #[test]
    fn lilis_twin_waiver_applies_to_current_role_knight_protection_only() {
        let mut state = base_state(4, vec![make_card(2, "Knight", json!({}))]);
        state.n_evil = 3;
        state.deck.villagers = vec!["Knight".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Lilis".to_string(), "Pooka".to_string()];
        state.night_kills = vec![2];

        let mut reachable = empty_scenario();
        reachable
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        reachable.evil_positions.insert(3, "Pooka".to_string());
        reachable.evil_positions.insert(4, "Lilis".to_string());
        assert!(validate_lilis_night_kills(&reachable, &state));

        // Moving the non-Twin Demon away makes #2 unreachable, so its stable
        // clean Knight protection must still reject the observed death.
        let mut unreachable = reachable.clone();
        unreachable.evil_positions.remove(&3);
        assert!(!validate_lilis_night_kills(&unreachable, &state));

        // The same endpoint rule applies to HealthyBluff Doppel-as-Knight.
        let mut doppel = reachable;
        doppel.doppelganger_position = Some(2);
        assert!(validate_lilis_night_kills(&doppel, &state));
    }

    #[test]
    fn lilis_death_distinguishes_drunk_and_healthy_bluff_knight_surfaces() {
        let mut hidden_drunk_state = base_state(1, vec![]);
        hidden_drunk_state.deck.villagers = vec!["Knight".to_string()];
        hidden_drunk_state.deck.demons = vec!["Lilis".to_string()];
        hidden_drunk_state.night_kills = vec![1];
        hidden_drunk_state.board_villager_count = Some(1);
        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(1);
        assert!(validate_lilis_night_kills(&drunk, &hidden_drunk_state));

        let mut doppel_state = base_state(2, vec![make_card(1, "Knight", json!({}))]);
        doppel_state.deck.villagers = vec!["Knight".to_string()];
        doppel_state.deck.demons = vec!["Lilis".to_string()];
        doppel_state.night_kills = vec![1];
        let mut doppel = empty_scenario();
        doppel.doppelganger_position = Some(1);
        assert!(!validate_lilis_night_kills(&doppel, &doppel_state));
        doppel.corrupted.insert(1);
        assert!(validate_lilis_night_kills(&doppel, &doppel_state));

        let mut generated_doppel = empty_scenario();
        generated_doppel.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![2],
            added_outcast_position: 1,
            added_outcast_role: "Doppelganger".to_string(),
            affected_anchor_positions: vec![],
        });
        assert!(!validate_lilis_night_kills(&generated_doppel, &doppel_state));
        generated_doppel.corrupted.insert(1);
        assert!(validate_lilis_night_kills(&generated_doppel, &doppel_state));
    }

    #[test]
    fn lilis_death_counts_puppet_and_rejects_malformed_evidence() {
        let mut state = base_state(2, vec![]);
        state.deck.demons = vec!["Lilis".to_string()];
        state.night_kills = vec![1];
        state.night_kill_evil_count = 1;
        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(1);
        assert!(validate_lilis_night_kills(&puppet, &state));

        state.deck.demons.clear();
        assert!(!validate_lilis_night_kills(&puppet, &state));
        state.deck.demons.push("Lilis".to_string());
        state.night_kills = vec![1, 1];
        assert!(!validate_lilis_night_kills(&puppet, &state));
        state.night_kills = vec![3];
        assert!(!validate_lilis_night_kills(&puppet, &state));
    }

    #[test]
    fn lilis_death_checks_shaman_copied_knight_identity_directly() {
        let mut state = base_state(3, vec![make_card(2, "Knight", json!({}))]);
        state.deck.villagers = vec!["Knight".to_string()];
        state.deck.minions = vec!["Shaman".to_string()];
        state.deck.demons = vec!["Lilis".to_string()];
        state.night_kills = vec![2];

        let mut copied_knight = empty_scenario();
        copied_knight.evil_positions.insert(3, "Shaman".to_string());
        copied_knight.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 1,
            target_position: 2,
            copied_role: "Knight".to_string(),
            target_previous_roles: vec!["Baker".to_string()],
        });
        assert!(!validate_lilis_night_kills(&copied_knight, &state));
        copied_knight.corrupted.insert(2);
        assert!(validate_lilis_night_kills(&copied_knight, &state));
    }

    #[test]
    fn hidden_knight_identity_check_allows_native_erasure_histories() {
        let mut state = base_state(3, vec![make_card(2, "Baker", json!({}))]);
        state.deck.villagers = vec!["Knight".to_string()];
        state.deck.minions = vec!["Shaman".to_string()];
        state.deck.demons = vec!["Lilis".to_string()];
        state.night_kills = vec![1];
        state.board_villager_count = Some(1);

        let mut no_erasure = empty_scenario();
        no_erasure.evil_positions.insert(3, "Shaman".to_string());
        assert!(!validate_lilis_night_kills(&no_erasure, &state));

        let mut shaman_erased_knight = empty_scenario();
        shaman_erased_knight.evil_positions.insert(3, "Shaman".to_string());
        shaman_erased_knight.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Baker".to_string(),
            target_previous_roles: vec!["Knight".to_string()],
        });
        assert!(validate_lilis_night_kills(&shaman_erased_knight, &state));

        let mut puppet_erased_knight = empty_scenario();
        puppet_erased_knight.puppet_position = Some(2);
        puppet_erased_knight.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_lilis_night_kills(&puppet_erased_knight, &state));

        let mut chancellor_erased_knight = empty_scenario();
        chancellor_erased_knight.evil_positions.insert(3, "Chancellor".to_string());
        chancellor_erased_knight.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 2,
            added_outcast_role: "Wretch".to_string(),
            affected_anchor_positions: vec![],
        });
        assert!(validate_lilis_night_kills(&chancellor_erased_knight, &state));
    }

    #[test]
    fn untyped_historical_start_eraser_preserves_only_viable_hidden_knight_worlds() {
        let mut state = base_state(
            4,
            vec![
                make_card(2, "Baker", json!({})),
                make_card(3, "Baker", json!({})),
                make_card(4, "Baker", json!({})),
            ],
        );
        state.deck.villagers = vec!["Knight".to_string()];
        state.deck.minions = vec!["Shaman".to_string()];
        state.deck.demons = vec!["Lilis".to_string()];
        state.night_kills = vec![1];
        state.executed = vec![3];
        state.confirmed_evil = vec![3];
        state.board_villager_count = Some(1);
        state.board_minion_count = Some(1);

        let no_current_eraser = empty_scenario();
        assert!(validate_lilis_night_kills(&no_current_eraser, &state));

        let mut chancellor_history = state.clone();
        chancellor_history.deck.minions = vec!["Chancellor".to_string()];
        assert!(validate_lilis_night_kills(
            &no_current_eraser,
            &chancellor_history,
        ));

        let mut typed_history = state.clone();
        typed_history
            .executed_evil_roles
            .insert(3, "Poisoner".to_string());
        assert!(!validate_lilis_night_kills(
            &no_current_eraser,
            &typed_history,
        ));

        let mut eraser_already_represented = empty_scenario();
        eraser_already_represented
            .evil_positions
            .insert(4, "Shaman".to_string());
        assert!(!validate_lilis_night_kills(
            &eraser_already_represented,
            &state,
        ));

        let mut minion_slots_full = state.clone();
        minion_slots_full.deck.minions = vec!["Shaman".to_string(), "Poisoner".to_string()];
        let mut current_poisoner = empty_scenario();
        current_poisoner
            .evil_positions
            .insert(4, "Poisoner".to_string());
        assert!(!validate_lilis_night_kills(
            &current_poisoner,
            &minion_slots_full,
        ));
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
            affected_anchor_positions: vec![],
        });

        assert!(!check_scenario(&generated_drunk, &state));
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
            "Baker".to_string(),
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

        assert!(check_scenario(&scenario, &state));

        scenario
            .shaman_trace
            .as_mut()
            .unwrap()
            .target_previous_roles = vec!["Scout".to_string()];
        assert!(
            !check_scenario(&scenario, &state),
            "an equivalence class with no deck-compatible erased role must fail"
        );

        scenario
            .shaman_trace
            .as_mut()
            .unwrap()
            .target_previous_roles = vec!["Judge".to_string()];

        state.board_villager_count = Some(2);
        assert!(
            !check_scenario(&scenario, &state),
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
    fn legacy_asc59_three_baker_drunk_history_survives_unrelated_lilis_data() {
        let path = std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../tests/cases_v2/asc59_v7.json");
        let value: serde_json::Value = serde_json::from_str(
            &std::fs::read_to_string(path).unwrap(),
        )
        .unwrap();
        let state = GameState::from_json(&value).unwrap();
        let witness = crate::scenario::build_scenarios(&state)
            .into_iter()
            .find(|scenario| {
                scenario
                    .evil_positions
                    .get(&1)
                    .is_some_and(|role| normalize_role(role) == "lilis")
                    && scenario
                        .evil_positions
                        .get(&3)
                        .is_some_and(|role| normalize_role(role) == "poisoner")
                    && scenario
                        .evil_positions
                        .get(&10)
                        .is_some_and(|role| normalize_role(role) == "twinminion")
                    && scenario.drunk_position == Some(8)
                    && scenario.corrupted.contains(&2)
                    && scenario.corrupted.contains(&8)
                    && validate_role_counts(scenario, &state)
                    && validate_baker_history(scenario, &state)
                    && state
                        .cards
                        .iter()
                        .all(|card| validate_card(card, scenario, &state))
            })
            .expect("asc59_v7 must retain its legacy three-Baker/Drunk witness");

        // The frozen case records two Lilis victims but zero evil kills even
        // though #10 is the true Twin Minion. Keep that independent stale
        // surface out of this Baker regression rather than weakening Lilis.
        assert!(!validate_lilis_night_kills(&witness, &state));
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
            affected_anchor_positions: vec![],
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

    fn current_fortune_teller(
        info: serde_json::Value,
        info_text: &str,
        n_cards: u8,
    ) -> (CardInfo, GameState) {
        let mut card = make_card(1, "Fortune Teller", info);
        card.info_text = info_text.to_string();
        let mut state = base_state(n_cards, vec![card.clone()]);
        state.fortune_teller_rule_version = Some(FORTUNE_TELLER_CURRENT_RULE.to_string());
        (card, state)
    }

    #[test]
    fn current_fortune_teller_validates_every_observation_and_latest_alias() {
        let (card, state) = current_fortune_teller(
            json!({
                "targets": [4, 5],
                "has_evil": false,
                "observations": [
                    {"targets": [2, 3], "has_evil": true,
                     "text": "Is #2 or #3 Evil?: True"},
                    {"targets": [4, 5], "has_evil": false,
                     "text": "Is #4 or #5 Evil?: False"}
                ]
            }),
            "Is #4 or #5 Evil?: False",
            5,
        );
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Pooka".to_string());
        assert!(validate_fortune_teller(&card, &scenario, &state));

        scenario.evil_positions.insert(4, "Minion".to_string());
        assert!(!validate_fortune_teller(&card, &scenario, &state));
    }

    #[test]
    fn current_fortune_teller_registered_alignment_and_lie_complement_are_exact() {
        let (card, state) = current_fortune_teller(
            json!({
                "targets": [2, 3], "has_evil": true,
                "observations": [{
                    "targets": [2, 3], "has_evil": true,
                    "text": "Is #2 or #3 Evil?: True"
                }]
            }),
            "Is #2 or #3 Evil?: True",
            3,
        );

        let mut wretch = empty_scenario();
        wretch.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 2,
            added_outcast_role: "Wretch".to_string(),
            affected_anchor_positions: vec![],
        });
        assert!(validate_fortune_teller(&card, &wretch, &state));

        let mut ordinary_drunk = empty_scenario();
        ordinary_drunk.drunk_position = Some(2);
        ordinary_drunk.corrupted.insert(2);
        assert!(!validate_fortune_teller(&card, &ordinary_drunk, &state));

        let mut lying_actor = empty_scenario();
        lying_actor.corrupted.insert(1);
        assert!(validate_fortune_teller(&card, &lying_actor, &state));
    }

    #[test]
    fn current_fortune_teller_rejects_malformed_native_shapes() {
        for (targets, n_cards) in [
            (json!([2]), 3),
            (json!([2, 2]), 3),
            (json!([3, 2]), 3),
            (json!([2, 4]), 3),
        ] {
            let (card, state) = current_fortune_teller(
                json!({
                    "targets": targets, "has_evil": false,
                    "observations": [{
                        "targets": targets, "has_evil": false,
                        "text": "Is #2 or #3 Evil?: False"
                    }]
                }),
                "Is #2 or #3 Evil?: False",
                n_cards,
            );
            assert!(!validate_fortune_teller(&card, &empty_scenario(), &state));
        }

        for info in [
            json!({"targets": [2, 3], "has_evil": false}),
            json!({"targets": [2, 3], "has_evil": false, "observations": []}),
            json!({
                "targets": [2, 3], "has_evil": false,
                "observations": [{
                    "targets": [2, 3], "has_evil": false,
                    "text": "is #2 or #3 Evil?: False"
                }]
            }),
            json!({
                "targets": [2, 3], "has_evil": false,
                "observations": [{
                    "targets": [2, 3], "has_evil": true,
                    "text": "Is #2 or #3 Evil?: True"
                }]
            }),
            json!({"shut_up_target": "3"}),
            json!({"shut_up_target": 0}),
            json!({"shut_up_target": 4}),
        ] {
            let (card, state) = current_fortune_teller(
                info,
                "Is #2 or #3 Evil?: False",
                3,
            );
            assert!(!validate_fortune_teller(&card, &empty_scenario(), &state));
        }
    }

    #[test]
    fn current_fortune_teller_allows_prior_history_when_latest_use_was_interrupted() {
        let (card, state) = current_fortune_teller(
            json!({
                "shut_up_target": 3,
                "observations": [{
                    "targets": [2, 3], "has_evil": false,
                    "text": "Is #2 or #3 Evil?: False"
                }]
            }),
            "#3 shut up!",
            3,
        );
        assert!(validate_fortune_teller(&card, &empty_scenario(), &state));

        let mut contradiction = empty_scenario();
        contradiction.evil_positions.insert(2, "Pooka".to_string());
        assert!(!validate_fortune_teller(&card, &contradiction, &state));
    }

    #[test]
    fn legacy_fortune_teller_preserves_reverse_order_and_scalar_shape() {
        let card = make_card(
            1,
            "Fortune Teller",
            json!({"targets": [5, 1], "has_evil": true}),
        );
        let state = base_state(5, vec![card.clone()]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(5, "Pooka".to_string());
        assert!(validate_fortune_teller(&card, &scenario, &state));
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
            affected_anchor_positions: vec![],
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
                affected_anchor_positions: vec![],
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

    #[test]
    fn twin_current_role_waiver_covers_only_the_two_unmodeled_swap_endpoints() {
        let mut state = base_state(3, vec![make_card(2, "Baker", json!({}))]);
        state.deck.villagers = vec!["Baker".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];

        // A physical Good recipient can publicly reveal the current Twin data
        // that the pre-trace role model still sees as its stable Good role.
        assert!(matches_executed_good_role(
            &empty_scenario(),
            &state,
            2,
            "Twin Minion",
        ));

        // Conversely, the stable Twin actor can publicly reveal the role it
        // received from the selected neighbor.
        let mut stable_twin = empty_scenario();
        stable_twin
            .evil_positions
            .insert(2, "Twin Minion".to_string());
        assert!(matches_executed_good_role(
            &stable_twin,
            &state,
            2,
            "Baker",
        ));

        // An arbitrary role absent from every authored/post-Twin generation
        // surface is not a valid endpoint just because this is the Twin seat.
        assert!(!matches_executed_good_role(
            &stable_twin,
            &state,
            2,
            "Shaman",
        ));

        // Puppeteer acts after Twin and can replace a received Villager with
        // generated Puppet data on the original runtime-Evil Twin body.
        state.deck.minions.push("Puppeteer".to_string());
        assert!(matches_executed_good_role(
            &stable_twin,
            &state,
            2,
            "Puppet",
        ));

        // Twin's presence does not waive a mismatch at an unrelated stable
        // Evil seat when neither endpoint surface names Twin.
        let mut unrelated_evil = empty_scenario();
        unrelated_evil
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(!matches_executed_good_role(
            &unrelated_evil,
            &state,
            2,
            "Shaman",
        ));

        // An observed Twin mismatch needs an authored Twin in the pool.
        state.deck.minions.clear();
        assert!(!matches_executed_good_role(
            &empty_scenario(),
            &state,
            2,
            "Twin Minion",
        ));
    }

    #[test]
    fn observed_evil_current_role_prunes_incompatible_shaman_trace() {
        let mut state = base_state(3, vec![]);
        state.n_evil = 1;
        state.deck.villagers = vec!["Scout".to_string(), "Bombardier".to_string()];
        state.deck.minions = vec!["Shaman".to_string()];
        state.executed = vec![2];
        state.executed_current_roles.insert(2, "Scout".to_string());

        let trace = |copied_role: &str| {
            let mut scenario = empty_scenario();
            scenario.evil_positions.insert(3, "Shaman".to_string());
            scenario.shaman_trace = Some(crate::types::ShamanTrace {
                source_position: 1,
                target_position: 2,
                copied_role: copied_role.to_string(),
                target_previous_roles: vec!["Bombardier".to_string()],
            });
            scenario
        };

        let compatible = trace("Scout");
        assert!(matches_executed_good_role(
            &compatible,
            &state,
            2,
            "Scout",
        ));
        assert!(validate_witch_block_evidence(&compatible, &state));
        assert!(validate_role_counts(&compatible, &state));
        assert!(validate_clean_doppel_source_support(&compatible, &state));
        assert!(validate_rambler_shut_ups(&compatible, &state));
        assert!(validate_slayer_results(&compatible, &state));
        assert!(validate_pd_ability(&compatible, &state));
        assert!(validate_lilis_night_kills(&compatible, &state));
        assert!(validate_baker_history(&compatible, &state));
        assert!(check_scenario(&compatible, &state));
        assert!(!check_scenario(&trace("Bombardier"), &state));
    }

    #[test]
    fn executed_current_role_entries_require_an_ordinary_exact_death() {
        let mut state = base_state(3, vec![]);
        state.executed = vec![2];
        assert!(valid_executed_current_role_entry(&state, 2, "Scout"));

        state.executed.clear();
        assert!(!valid_executed_current_role_entry(&state, 2, "Scout"));

        state.executed = vec![2];
        state.night_kills = vec![2];
        assert!(!valid_executed_current_role_entry(&state, 2, "Scout"));

        state.night_kills.clear();
        state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 1,
            target_pos: 2,
            killed: true,
            revealed_role: Some("Scout".to_string()),
            was_evil: None,
        });
        assert!(!valid_executed_current_role_entry(&state, 2, "Scout"));

        state.slayer_results.clear();
        assert!(!valid_executed_current_role_entry(&state, 0, "Scout"));
        assert!(!valid_executed_current_role_entry(&state, 4, "Scout"));
        for unknown in ["", "unknown", "?", "none", "null"] {
            assert!(!valid_executed_current_role_entry(&state, 2, unknown));
        }
    }
}
