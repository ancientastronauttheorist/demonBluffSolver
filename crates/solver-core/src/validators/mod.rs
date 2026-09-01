/// Card info validators — check if a card's claimed info is consistent with a scenario.

mod baker;
mod disguisers;
mod helpers;
pub use helpers::*;

use baker::{
    baker_history_can_erase_role, baker_history_supports_pre_day_role,
    baker_spy_conversion_timelines, medium_uses_baker_history, validate_baker_history,
    BakerSpyTimeline, BAKER_CURRENT_RULE,
};
use disguisers::validate_clean_doppel_source_support;

use std::collections::{HashMap, HashSet};
use crate::geometry::{circle_distance, circle_direction, adjacent_positions, Direction};
use crate::knowledge_base::{self, get_card, normalize_role, Faction};
use crate::shaman::{enumerate_shaman_traces, role_after_shaman};
use crate::twin::{
    distinct_swap_has_unsupported_public_action_evidence, enumerate_twin_traces,
    role_after_twin,
};
use crate::types::{BoardCountProvenance, CardInfo, GameState, Scenario};

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
    let exact_match = current_data_role_at(pos, scenario, state)
        .as_deref()
        .is_some_and(|role| roles_equal(role, observed_role));
    exact_match || twin_can_explain_current_role_mismatch(pos, observed_role, scenario, state)
}

fn legacy_current_evil_role_at(
    pos: u8,
    scenario: &Scenario,
    state: &GameState,
) -> Option<String> {
    if scenario.puppet_position == Some(pos) {
        return current_data_role_at(pos, scenario, state);
    }
    known_evil_role(pos, scenario, state).map(str::to_string)
}

fn twin_can_explain_current_role_mismatch(
    pos: u8,
    observed_role: &str,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if scenario.twin_trace.is_some() {
        return false;
    }

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
    // its selected neighbor. Later Shaman writes can replace that received
    // data with another authored role. Generated Puppet is different: it is a
    // post-Twin full Init and must match the scenario's explicit overlay.
    let observed = normalize_role(observed_role);
    observed != "puppet" && deck_has(&observed)
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

fn executed_evil_origin_is_unresolved(
    position: u8,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    state.executed.contains(&position)
        && state.confirmed_evil.contains(&position)
        && stable_evil_origin_role_at(position, scenario, state)
            .is_some_and(|role| normalize_role(role) == "unknown")
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

    // Every audited native-current registered-alignment provider observes one
    // physical natural-Wretch assignment. The same joint search also retains
    // explicit Wretch registerAs labels, Medium's stored Spy/raw-bluff surfaces,
    // and the shared Baker/Spy chronology instead of solving those dimensions
    // independently.
    if !validate_current_hidden_surface_consistency(scenario, state) {
        return false;
    }

    // Card info validators
    for card in &state.cards {
        if executed_evil_origin_is_unresolved(card.position, scenario, state) {
            continue; // Skip only a genuinely unresolved executed Evil.
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

    match exact_twin_shaman_post_twin_roles(scenario, state) {
        Some(_) => true,
        None if is_exact_twin_shaman_claim(scenario) => false,
        None => validate_baker_history(scenario, state),
    }
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
        let branch_role_is_untyped = scenario.puppet_position != Some(position)
            && stable_evil_origin_role_at(position, scenario, state)
                .is_none_or(|role| normalize_role(role) == "unknown");
        if dead && branch_role_is_untyped && !named_lilis_positions.contains(&position) {
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
    if scenario.twin_trace.is_some()
        || state.n_cards < 2
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
            let modeled_registered_or_real_demon = current_data_role_at(anchor, scenario, state)
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
            && scenario.puppet_position != Some(*position)
            && stable_evil_origin_role_at(*position, scenario, state)
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
    match current_passive_payload_source(
        card,
        ENLIGHTENED_CURRENT_VARIANT_FIELD,
        "Enlightened",
    ) {
        Ok(Some(source)) => return validate_current_enlightened(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

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
    match current_passive_payload_source(card, KNITTER_CURRENT_VARIANT_FIELD, "Knitter") {
        Ok(Some(source)) => return validate_current_knitter(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

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
    match current_passive_payload_source(
        card,
        CONFESSOR_CURRENT_VARIANT_FIELD,
        "Confessor",
    ) {
        Ok(Some(CurrentPassivePayloadSource::Direct)) => {
            return validate_current_confessor(card, scenario, state);
        }
        // Confessor is not one of the twelve providers constructed by current
        // Poet/Gossip. Keep obsolete unmarked Poet captures on the legacy path,
        // but never admit a provenance-marked current Poet surface here.
        Ok(Some(CurrentPassivePayloadSource::Poet)) | Err(()) => return false,
        Ok(None) => {}
    }

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
    match current_passive_payload_source(
        card,
        GEMCRAFTER_CURRENT_VARIANT_FIELD,
        "Gemcrafter",
    ) {
        Ok(Some(source)) => return validate_current_gemcrafter(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

    // Preserve unmarked archived observations on the legacy scalar predicate.
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
    match current_passive_payload_source(card, LOVER_CURRENT_VARIANT_FIELD, "Lover") {
        Ok(Some(source)) => return validate_current_lover(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

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

const LOVER_CURRENT_VARIANT_FIELD: &str = "lover_variant";
const SCOUT_CURRENT_VARIANT_FIELD: &str = "scout_variant";
const HUNTER_CURRENT_VARIANT_FIELD: &str = "hunter_variant";
const ORACLE_CURRENT_VARIANT_FIELD: &str = "oracle_variant";
const MEDIUM_CURRENT_VARIANT_FIELD: &str = "medium_variant";
const KNITTER_CURRENT_VARIANT_FIELD: &str = "knitter_variant";
const ENLIGHTENED_CURRENT_VARIANT_FIELD: &str = "enlightened_variant";
const EMPRESS_CURRENT_VARIANT_FIELD: &str = "empress_variant";
const BISHOP_CURRENT_VARIANT_FIELD: &str = "bishop_variant";
const GEMCRAFTER_CURRENT_VARIANT_FIELD: &str = "gemcrafter_variant";
const BARD_CURRENT_VARIANT_FIELD: &str = "bard_variant";
const CONFESSOR_CURRENT_VARIANT_FIELD: &str = "confessor_variant";
const DRUID_CURRENT_VARIANT_FIELD: &str = "druid_variant";
const JESTER_CURRENT_VARIANT_FIELD: &str = "jester_variant";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentPassivePayloadSource {
    Direct,
    Poet,
}

/// Distinguish frozen, unmarked observations from the exact current bridge
/// schema. Once either provenance marker is present, malformed, mixed, or
/// future provenance fails closed instead of falling back to legacy behavior.
fn current_passive_payload_source(
    card: &CardInfo,
    direct_variant_field: &str,
    copied_role: &str,
) -> Result<Option<CurrentPassivePayloadSource>, ()> {
    let direct_variant = card.info_parsed.get(direct_variant_field);
    let poet_variant = card.info_parsed.get("poet_variant");

    match (direct_variant, poet_variant) {
        (None, None)
            if !card.info_parsed.contains_key(LOVER_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(SCOUT_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(HUNTER_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(ORACLE_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(MEDIUM_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(KNITTER_CURRENT_VARIANT_FIELD)
                && !card
                    .info_parsed
                    .contains_key(ENLIGHTENED_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(EMPRESS_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(BISHOP_CURRENT_VARIANT_FIELD)
                && !card
                    .info_parsed
                    .contains_key(GEMCRAFTER_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(BARD_CURRENT_VARIANT_FIELD)
                && !card
                    .info_parsed
                    .contains_key(CONFESSOR_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(DRUID_CURRENT_VARIANT_FIELD)
                && !card.info_parsed.contains_key(JESTER_CURRENT_VARIANT_FIELD) =>
        {
            Ok(None)
        }
        (Some(value), None)
            if value.as_str() == Some(POET_CURRENT_VARIANT)
                && roles_equal(&card.apparent_role, copied_role) =>
        {
            Ok(Some(CurrentPassivePayloadSource::Direct))
        }
        (None, Some(value))
            if value.as_str() == Some(POET_CURRENT_VARIANT)
                && roles_equal(&card.apparent_role, "Poet")
                && card
                    .info_parsed
                    .get("copied_role")
                    .and_then(serde_json::Value::as_str)
                    == Some(copied_role) =>
        {
            Ok(Some(CurrentPassivePayloadSource::Poet))
        }
        _ => Err(()),
    }
}

fn current_confessor_claim_text(dizzy: bool) -> &'static str {
    if dizzy {
        "I am dizzy"
    } else {
        "I am Good"
    }
}

fn parse_current_confessor_claim(card: &CardInfo, state: &GameState) -> Option<bool> {
    if card.position == 0
        || card.position > state.n_cards
        || card.apparent_role != "Confessor"
        || card.info_parsed.len() != 2
        || card
            .info_parsed
            .get(CONFESSOR_CURRENT_VARIANT_FIELD)
            .and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }
    let dizzy = card.info_parsed.get("dizzy")?.as_bool()?;
    (card.info_text == current_confessor_claim_text(dizzy)).then_some(dizzy)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentConfessorSupport {
    anonymous_wretches: AnonymousWretchConstraints,
    register_as: Option<(u8, String)>,
    raw_bluff: Option<(u8, String)>,
    forbidden_raw_bluff: Option<(u8, String)>,
    baker_spy_timeline: BakerSpyTimeline,
}

fn current_confessor_actual_dizzy(
    actor: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<bool> {
    let current_role =
        current_data_role_at_observation(actor, actor, timeline, scenario, state)?;
    // Confessor checks the current real dataRef role before either corruption
    // or registered alignment. Native Spy (and a managed role derived from it)
    // therefore reports Good even when the physical Character is otherwise
    // Evil or Corrupted. Spy is the only represented member of that native
    // class family in the public solver model.
    if roles_equal(&current_role, "Spy") {
        return Some(false);
    }
    if scenario.corrupted.contains(&actor) {
        return Some(true);
    }
    Some(
        registered_alignment_at_observation(actor, actor, timeline, scenario, state)?
            == EffectiveAlignment::Evil,
    )
}

fn current_confessor_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentConfessorSupport> {
    let Some(claimed_dizzy) = parse_current_confessor_claim(card, state) else {
        return Vec::new();
    };
    // An untyped Evil may have been a corruption/status writer, identity mover,
    // or bluff producer. A final role overlay cannot reconstruct the Day-time
    // charRef surface sampled by Confessor.
    if current_has_unresolved_start_identity(scenario, state) {
        return Vec::new();
    }

    let anonymous_candidates = anonymous_natural_wretch_candidates(scenario, state);
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(card.position, state) {
            continue;
        }
        let Some(current_role) = current_data_role_at_observation(
            card.position,
            card.position,
            &timeline,
            scenario,
            state,
        ) else {
            continue;
        };
        if current_confessor_actual_dizzy(card.position, &timeline, scenario, state)
            != Some(claimed_dizzy)
        {
            continue;
        }
        let raw_bluff_holder = current_medium_raw_bluff_holder_at(
            card.position,
            card.position,
            &timeline,
            scenario,
            state,
        );

        if roles_equal(&current_role, "Confessor") {
            // A proven non-null raw pointer dispatches after the real callback.
            // The newest event can still have Confessor's exact shape only
            // when that later callback is also Confessor (its real/bluff
            // methods compute the same predicate). `Possible` must remain
            // unconstrained here because the current model cannot distinguish
            // an absent pointer from an unobserved one; the separate raw branch
            // below represents the positive Confessor assignment.
            let surviving_raw_bluff = (raw_bluff_holder
                == CurrentMediumRawBluffHolder::Proven)
                .then(|| (card.position, normalize_role("Confessor")));
            let support = CurrentConfessorSupport {
                anonymous_wretches: AnonymousWretchConstraints::empty(),
                register_as: None,
                raw_bluff: surviving_raw_bluff,
                // Real GetInfo and a later raw-Confessor GetBluffInfo compute
                // the same charRef predicate and text. The later callback may
                // overwrite the first, but it cannot contradict this claim.
                forbidden_raw_bluff: None,
                baker_spy_timeline: timeline.clone(),
            };
            if !supports.contains(&support) {
                supports.push(support);
            }
        }

        if raw_bluff_holder == CurrentMediumRawBluffHolder::Impossible {
            continue;
        }

        let mut anonymous_wretches = AnonymousWretchConstraints::empty();
        if anonymous_candidates.contains(&card.position) {
            // Natural Wretch has no raw bluff selector. A raw Confessor callback
            // excludes that grouped identity unless a represented mover already
            // supplied the non-null pointer.
            anonymous_wretches.forbidden.insert(card.position);
            if !anonymous_wretch_assignment_possible(
                &anonymous_wretches.required,
                &anonymous_wretches.forbidden,
                scenario,
                state,
            ) {
                continue;
            }
        }

        let register_as = if current_spy_register_as_surface_at_observation(
            card.position,
            card.position,
            &timeline,
            scenario,
            state,
        ) == Some(true)
        {
            if !current_medium_spy_register_as_label_allowed("Confessor", state) {
                continue;
            }
            Some((card.position, normalize_role("Confessor")))
        } else {
            None
        };
        let support = CurrentConfessorSupport {
            anonymous_wretches,
            register_as,
            raw_bluff: Some((card.position, normalize_role("Confessor"))),
            forbidden_raw_bluff: None,
            baker_spy_timeline: timeline,
        };
        if !supports.contains(&support) {
            supports.push(support);
        }
    }
    supports
}

fn validate_current_confessor(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    !current_confessor_supports(card, scenario, state).is_empty()
}

fn current_gemcrafter_claim_text(target: u8) -> String {
    format!("#{target} is Good")
}

fn parse_current_gemcrafter_target(
    card: &CardInfo,
    source: CurrentPassivePayloadSource,
    state: &GameState,
) -> Option<u8> {
    if card.position == 0 || card.position > state.n_cards {
        return None;
    }
    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => {
            if card.apparent_role != "Gemcrafter" {
                return None;
            }
            (GEMCRAFTER_CURRENT_VARIANT_FIELD, 1)
        }
        CurrentPassivePayloadSource::Poet => {
            if card.apparent_role != "Poet"
                || info.get("copied_role").and_then(serde_json::Value::as_str)
                    != Some("Gemcrafter")
            {
                return None;
            }
            ("poet_variant", 2)
        }
    };
    if info.len() != fixed_fields + 1
        || info.get(variant_field).and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }
    let target = poet_position_value(info.get("good_position"), state.n_cards)?;
    (card.info_text == current_gemcrafter_claim_text(target)).then_some(target)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentGemcrafterSupport {
    anonymous_wretches: AnonymousWretchConstraints,
    baker_spy_timeline: BakerSpyTimeline,
}

fn current_gemcrafter_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentGemcrafterSupport> {
    let Some(target) = parse_current_gemcrafter_target(card, source, state) else {
        return Vec::new();
    };
    // An untyped executed Evil may have run any Start writer. A late role
    // overlay cannot reconstruct the registered-alignment surface Archivist
    // sampled synchronously at this observation.
    if current_has_unresolved_start_identity(scenario, state) {
        return Vec::new();
    }

    let truth = truth_status(card.position, scenario, state);
    let candidates = anonymous_natural_wretch_candidates(scenario, state);
    let mut supports = Vec::new();

    fn enumerate(
        index: usize,
        candidates: &[u8],
        required: &mut HashSet<u8>,
        forbidden: &mut HashSet<u8>,
        known_evil: &HashSet<u8>,
        target: u8,
        actor: u8,
        truth: TruthStatus,
        timeline: &BakerSpyTimeline,
        scenario: &Scenario,
        state: &GameState,
        supports: &mut Vec<CurrentGemcrafterSupport>,
    ) {
        if index == candidates.len() {
            if !anonymous_wretch_assignment_possible(required, forbidden, scenario, state) {
                return;
            }
            let mut registered_evil = known_evil.clone();
            registered_evil.extend(required.iter().copied());
            let selects_evil = truth == TruthStatus::Lying;
            let target_is_evil = registered_evil.contains(&target);
            if target_is_evil != selects_evil {
                return;
            }

            // Native builds the whole selected-alignment pool first. It calls
            // Remove(charRef) only when that original pool has more than one
            // occurrence. Consequently a self target survives exactly when it
            // is the sole member of the selected pool.
            let selected_pool_count = (1..=state.n_cards)
                .filter(|position| registered_evil.contains(position) == selects_evil)
                .count();
            if target == actor && selected_pool_count > 1 {
                return;
            }

            let support = CurrentGemcrafterSupport {
                anonymous_wretches: AnonymousWretchConstraints {
                    required: required.clone(),
                    forbidden: forbidden.clone(),
                },
                baker_spy_timeline: timeline.clone(),
            };
            if !supports.contains(&support) {
                supports.push(support);
            }
            return;
        }

        let position = candidates[index];
        forbidden.insert(position);
        enumerate(
            index + 1,
            candidates,
            required,
            forbidden,
            known_evil,
            target,
            actor,
            truth,
            timeline,
            scenario,
            state,
            supports,
        );
        forbidden.remove(&position);

        required.insert(position);
        enumerate(
            index + 1,
            candidates,
            required,
            forbidden,
            known_evil,
            target,
            actor,
            truth,
            timeline,
            scenario,
            state,
            supports,
        );
        required.remove(&position);
    }

    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(card.position, state) {
            continue;
        }
        let mut known_evil = HashSet::new();
        let mut complete = true;
        for position in 1..=state.n_cards {
            let Some(alignment) = registered_alignment_at_observation(
                position,
                card.position,
                &timeline,
                scenario,
                state,
            ) else {
                complete = false;
                break;
            };
            if alignment == EffectiveAlignment::Evil {
                known_evil.insert(position);
            }
        }
        if !complete {
            continue;
        }
        enumerate(
            0,
            &candidates,
            &mut HashSet::new(),
            &mut HashSet::new(),
            &known_evil,
            target,
            card.position,
            truth,
            &timeline,
            scenario,
            state,
            &mut supports,
        );
    }
    supports
}

fn validate_current_gemcrafter(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_gemcrafter_supports(card, scenario, state, source).is_empty()
}

fn current_lover_claim_text(claimed: i64) -> Option<&'static str> {
    match claimed {
        0 => Some("NO Evils\nadjacent to me"),
        1 => Some("1 Evil\nadjacent to me"),
        2 => Some("2 Evils\nadjacent to me"),
        _ => None,
    }
}

fn parse_current_lover_claim(
    card: &CardInfo,
    source: CurrentPassivePayloadSource,
    state: &GameState,
) -> Option<i64> {
    if card.position == 0 || card.position > state.n_cards {
        return None;
    }
    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => {
            if !roles_equal(&card.apparent_role, "Lover") {
                return None;
            }
            (LOVER_CURRENT_VARIANT_FIELD, 1)
        }
        CurrentPassivePayloadSource::Poet => {
            if !roles_equal(&card.apparent_role, "Poet")
                || info.get("copied_role").and_then(serde_json::Value::as_str)
                    != Some("Lover")
            {
                return None;
            }
            ("poet_variant", 2)
        }
    };
    if info.len() != fixed_fields + 1
        || info.get(variant_field).and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }
    let claimed = info.get("evil_adjacent")?.as_i64()?;
    let expected_text = current_lover_claim_text(claimed)?;
    (card.info_text == expected_text).then_some(claimed)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentLoverSupport {
    actual: i64,
    anonymous_wretches: AnonymousWretchConstraints,
    baker_spy_timeline: BakerSpyTimeline,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum BakerSpyObservationPhase {
    Unaffected,
    BeforeConversion,
    PendingRegisterAsReset,
    Reset,
}

fn baker_spy_observation_phase(
    position: u8,
    observation: u8,
    timeline: &BakerSpyTimeline,
    state: &GameState,
) -> Option<BakerSpyObservationPhase> {
    if !timeline.contains_position(position) {
        return Some(BakerSpyObservationPhase::Unaffected);
    }
    if !timeline.converted_at_observation(position, observation, state)? {
        return Some(BakerSpyObservationPhase::BeforeConversion);
    }
    if timeline.registered_evil_at_observation(position, observation, state)? {
        Some(BakerSpyObservationPhase::Reset)
    } else {
        Some(BakerSpyObservationPhase::PendingRegisterAsReset)
    }
}

fn current_data_role_at_observation(
    position: u8,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<String> {
    match baker_spy_observation_phase(position, observation, timeline, state)? {
        BakerSpyObservationPhase::BeforeConversion => Some("Spy".to_string()),
        BakerSpyObservationPhase::PendingRegisterAsReset | BakerSpyObservationPhase::Reset => {
            Some("Baker".to_string())
        }
        BakerSpyObservationPhase::Unaffected => current_data_role_at(position, scenario, state),
    }
}

fn current_spy_register_as_surface_at_observation(
    position: u8,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<bool> {
    match baker_spy_observation_phase(position, observation, timeline, state)? {
        BakerSpyObservationPhase::BeforeConversion
        | BakerSpyObservationPhase::PendingRegisterAsReset => Some(true),
        BakerSpyObservationPhase::Reset => Some(false),
        BakerSpyObservationPhase::Unaffected => {
            Some(current_spy_register_as_surface_at(position, scenario, state))
        }
    }
}

fn registered_alignment_at_observation(
    position: u8,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<EffectiveAlignment> {
    match baker_spy_observation_phase(position, observation, timeline, state)? {
        BakerSpyObservationPhase::BeforeConversion
        | BakerSpyObservationPhase::PendingRegisterAsReset => Some(EffectiveAlignment::Good),
        BakerSpyObservationPhase::Reset => Some(EffectiveAlignment::Evil),
        BakerSpyObservationPhase::Unaffected => {
            if current_spy_register_as_surface_at(position, scenario, state) {
                Some(EffectiveAlignment::Good)
            } else {
                Some(registered_alignment_at(position, scenario, state))
            }
        }
    }
}

fn current_enlightened_claim_text(direction: Direction) -> &'static str {
    match direction {
        Direction::CW => "Closest Evil is:\nClockwise",
        Direction::CCW => "Closest Evil is:\nCounter-clockwise",
        Direction::Equidistant => "Closest Evil is equidistant",
    }
}

fn parse_current_enlightened_claim(
    card: &CardInfo,
    source: CurrentPassivePayloadSource,
    state: &GameState,
) -> Option<Direction> {
    if card.position == 0 || card.position > state.n_cards {
        return None;
    }
    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => {
            if card.apparent_role != "Enlightened" {
                return None;
            }
            (ENLIGHTENED_CURRENT_VARIANT_FIELD, 1)
        }
        CurrentPassivePayloadSource::Poet => {
            if card.apparent_role != "Poet"
                || info.get("copied_role").and_then(serde_json::Value::as_str)
                    != Some("Enlightened")
            {
                return None;
            }
            ("poet_variant", 2)
        }
    };
    if info.len() != fixed_fields + 1
        || info.get(variant_field).and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }

    let claimed = match info.get("direction")?.as_str()? {
        "CW" => Direction::CW,
        "CCW" => Direction::CCW,
        "Equidistant" => Direction::Equidistant,
        _ => return None,
    };
    (card.info_text == current_enlightened_claim_text(claimed)).then_some(claimed)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentEnlightenedSupport {
    actual: Direction,
    anonymous_wretches: AnonymousWretchConstraints,
    baker_spy_timeline: BakerSpyTimeline,
}

/// Native Shugenja orders every current character after the actor in each
/// direction, then compares the first registered-Evil distance. The actor is
/// never a candidate; equal distances and two exhausted scans are Either.
fn current_enlightened_direction(
    actor: u8,
    n_cards: u8,
    registered_evil: &HashSet<u8>,
) -> Direction {
    if n_cards == 0 {
        return Direction::Equidistant;
    }
    let mut clockwise = None;
    let mut counterclockwise = None;
    for &position in registered_evil {
        if position == actor || position == 0 || position > n_cards {
            continue;
        }
        // Native Shugenja names the forward CurrentCharacters scan
        // Counter-clockwise and the reverse scan Clockwise. Solver position
        // IDs increase in forward order, so public CW uses decreasing IDs.
        let cw = (i16::from(actor) - i16::from(position))
            .rem_euclid(i16::from(n_cards)) as u8;
        let ccw = (i16::from(position) - i16::from(actor))
            .rem_euclid(i16::from(n_cards)) as u8;
        clockwise = Some(clockwise.map_or(cw, |known: u8| known.min(cw)));
        counterclockwise = Some(counterclockwise.map_or(ccw, |known: u8| known.min(ccw)));
    }

    match (clockwise, counterclockwise) {
        (Some(cw), Some(ccw)) => match cw.cmp(&ccw) {
            std::cmp::Ordering::Less => Direction::CW,
            std::cmp::Ordering::Greater => Direction::CCW,
            std::cmp::Ordering::Equal => Direction::Equidistant,
        },
        _ => Direction::Equidistant,
    }
}

fn current_enlightened_worlds(
    observation: u8,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentEnlightenedSupport> {
    let candidates = anonymous_natural_wretch_candidates(scenario, state);
    let mut worlds = Vec::new();

    fn enumerate(
        index: usize,
        candidates: &[u8],
        required: &mut HashSet<u8>,
        forbidden: &mut HashSet<u8>,
        known_evil: &HashSet<u8>,
        observation: u8,
        baker_spy_timeline: &BakerSpyTimeline,
        scenario: &Scenario,
        state: &GameState,
        worlds: &mut Vec<CurrentEnlightenedSupport>,
    ) {
        if index == candidates.len() {
            if !anonymous_wretch_assignment_possible(required, forbidden, scenario, state) {
                return;
            }
            let mut registered_evil = known_evil.clone();
            registered_evil.extend(required.iter().copied());
            worlds.push(CurrentEnlightenedSupport {
                actual: current_enlightened_direction(
                    observation,
                    state.n_cards,
                    &registered_evil,
                ),
                anonymous_wretches: AnonymousWretchConstraints {
                    required: required.clone(),
                    forbidden: forbidden.clone(),
                },
                baker_spy_timeline: baker_spy_timeline.clone(),
            });
            return;
        }

        let position = candidates[index];
        forbidden.insert(position);
        enumerate(
            index + 1,
            candidates,
            required,
            forbidden,
            known_evil,
            observation,
            baker_spy_timeline,
            scenario,
            state,
            worlds,
        );
        forbidden.remove(&position);

        required.insert(position);
        enumerate(
            index + 1,
            candidates,
            required,
            forbidden,
            known_evil,
            observation,
            baker_spy_timeline,
            scenario,
            state,
            worlds,
        );
        required.remove(&position);
    }

    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(observation, state) {
            continue;
        }
        let mut known_evil = HashSet::new();
        let mut complete = true;
        for position in 1..=state.n_cards {
            let Some(alignment) = registered_alignment_at_observation(
                position,
                observation,
                &timeline,
                scenario,
                state,
            ) else {
                complete = false;
                break;
            };
            if alignment == EffectiveAlignment::Evil {
                known_evil.insert(position);
            }
        }
        if !complete {
            continue;
        }
        enumerate(
            0,
            &candidates,
            &mut HashSet::new(),
            &mut HashSet::new(),
            &known_evil,
            observation,
            &timeline,
            scenario,
            state,
            &mut worlds,
        );
    }
    worlds
}

fn current_enlightened_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentEnlightenedSupport> {
    let Some(claimed) = parse_current_enlightened_claim(card, source, state) else {
        return Vec::new();
    };
    let worlds = current_enlightened_worlds(card.position, scenario, state);
    match truth_status(card.position, scenario, state) {
        TruthStatus::Truthful => worlds
            .into_iter()
            .filter(|world| world.actual == claimed)
            .collect(),
        TruthStatus::Lying => worlds
            .into_iter()
            .filter(|world| world.actual != claimed)
            .collect(),
    }
}

fn validate_current_enlightened(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_enlightened_supports(card, scenario, state, source).is_empty()
}

fn current_lover_actual_supports(
    actor: u8,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentLoverSupport> {
    baker_spy_conversion_timelines(scenario, state)
        .into_iter()
        .flat_map(|timeline| {
            current_lover_actual_supports_for_timeline(actor, scenario, state, &timeline)
        })
        .collect()
}

fn current_lover_actual_supports_for_timeline(
    actor: u8,
    scenario: &Scenario,
    state: &GameState,
    timeline: &BakerSpyTimeline,
) -> Vec<CurrentLoverSupport> {
    if !timeline.supports_observation(actor, state) {
        return Vec::new();
    }
    let adjacent = adjacent_positions(actor, state.n_cards);
    let mut known_count = 0;
    for &position in &adjacent {
        let Some(alignment) =
            registered_alignment_at_observation(position, actor, timeline, scenario, state)
        else {
            return Vec::new();
        };
        known_count += i64::from(alignment == EffectiveAlignment::Evil);
    }
    let anonymous_wretches: HashSet<u8> = anonymous_natural_wretch_candidates(scenario, state)
        .into_iter()
        .collect();
    let mut adjacent_candidates: Vec<u8> = adjacent
        .iter()
        .copied()
        .filter(|position| anonymous_wretches.contains(position))
        .collect();
    adjacent_candidates.sort_unstable();
    adjacent_candidates.dedup();

    if adjacent_candidates.is_empty() {
        return vec![CurrentLoverSupport {
            actual: known_count,
            anonymous_wretches: AnonymousWretchConstraints::empty(),
            baker_spy_timeline: timeline.clone(),
        }];
    }

    let mut supports = Vec::new();
    for mask in 0..(1usize << adjacent_candidates.len()) {
        let mut required = HashSet::new();
        let mut forbidden = HashSet::new();
        for (index, &position) in adjacent_candidates.iter().enumerate() {
            if mask & (1usize << index) == 0 {
                forbidden.insert(position);
            } else {
                required.insert(position);
            }
        }
        if !anonymous_wretch_assignment_possible(
            &required,
            &forbidden,
            scenario,
            state,
        ) {
            continue;
        }
        let actual = known_count
            + adjacent
                .iter()
                .filter(|position| required.contains(position))
                .count() as i64;
        let support = CurrentLoverSupport {
            actual,
            anonymous_wretches: AnonymousWretchConstraints {
                required,
                forbidden,
            },
            baker_spy_timeline: timeline.clone(),
        };
        if !supports.contains(&support) {
            supports.push(support);
        }
    }
    supports
}

#[cfg(test)]
fn current_lover_possible_actual_counts(
    actor: u8,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<i64> {
    let mut counts = Vec::new();
    for support in current_lover_actual_supports(actor, scenario, state) {
        if !counts.contains(&support.actual) {
            counts.push(support.actual);
        }
    }
    counts
}

fn current_authored_evil_slots(
    scenario: &Scenario,
    state: &GameState,
) -> Option<u8> {
    // The HUD objective counts the generated Puppet, while native passive
    // bluff domains based on CurrentScript Minion+Demon do not. A registered-
    // Evil Wretch is runtime Good and therefore needs no corresponding
    // adjustment here. Exact generated scenarios retain every stable authored
    // role in `evil_positions`; an ordinary Puppet occupies its own map entry,
    // while a post-Twin Puppet overlay shares the Twin's physical position.
    // Recover the authored CurrentScript domain directly whenever that physical
    // union matches an accepted board total. Do not derive it from deck lists:
    // those are role pools and may contain undealt Evil identities.
    let separate_puppet = scenario
        .puppet_position
        .is_some_and(|position| !scenario.evil_positions.contains_key(&position));
    let represented_total = scenario.evil_positions.len() + usize::from(separate_puppet);
    let has_puppeteer = scenario
        .evil_positions
        .values()
        .any(|role| normalize_role(role) == "puppeteer");
    let exact_trusted_total = represented_total == state.n_evil as usize;
    let exact_legacy_authored_total =
        state.board_count_provenance == BoardCountProvenance::LegacyUnknown
            && scenario.puppet_position.is_some()
            && has_puppeteer
            && represented_total == state.n_evil as usize + 1;
    if exact_trusted_total || exact_legacy_authored_total {
        let authored = scenario
            .evil_positions
            .values()
            .filter(|role| normalize_role(role) != "puppet")
            .count();
        return u8::try_from(authored).ok();
    }
    state
        .n_evil
        .checked_sub(u8::from(scenario.puppet_position.is_some()))
}

fn validate_current_lover(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_lover_supports(card, scenario, state, source).is_empty()
}

fn current_lover_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentLoverSupport> {
    let Some(claimed) = parse_current_lover_claim(card, source, state) else {
        return Vec::new();
    };
    let supports = current_lover_actual_supports(card.position, scenario, state);

    match truth_status(card.position, scenario, state) {
        TruthStatus::Truthful => supports
            .into_iter()
            .filter(|support| support.actual == claimed)
            .collect(),
        TruthStatus::Lying => {
            let Some(authored_evil_slots) =
                current_authored_evil_slots(scenario, state)
            else {
                return Vec::new();
            };
            if claimed > i64::from(authored_evil_slots.min(2)) {
                return Vec::new();
            }
            supports
                .into_iter()
                .filter(|support| support.actual != claimed)
                .collect()
        }
    }
}

fn current_knitter_claim_text(claimed: i64) -> Option<String> {
    match claimed {
        0 => Some("Evils are not adjacent to eachother".to_string()),
        1 => Some("There is only 1 pair of Evil".to_string()),
        value if value >= 2 => Some(format!("There are {value} pairs of Evil")),
        _ => None,
    }
}

fn parse_current_knitter_claim(
    card: &CardInfo,
    source: CurrentPassivePayloadSource,
    state: &GameState,
) -> Option<i64> {
    if card.position == 0 || card.position > state.n_cards {
        return None;
    }

    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => {
            if card.apparent_role != "Knitter" {
                return None;
            }
            (KNITTER_CURRENT_VARIANT_FIELD, 1)
        }
        CurrentPassivePayloadSource::Poet => {
            if card.apparent_role != "Poet"
                || info.get("copied_role").and_then(serde_json::Value::as_str) != Some("Knitter")
            {
                return None;
            }
            ("poet_variant", 2)
        }
    };
    if info.len() != fixed_fields + 1
        || info.get(variant_field).and_then(serde_json::Value::as_str) != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }

    let claimed = info.get("evil_pairs")?.as_i64()?;
    if !(0..=i64::from(state.n_cards)).contains(&claimed) {
        return None;
    }
    let expected_text = current_knitter_claim_text(claimed)?;
    (card.info_text == expected_text).then_some(claimed)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentKnitterSupport {
    actual: i64,
    required_anonymous_wretches: HashSet<u8>,
    forbidden_anonymous_wretches: HashSet<u8>,
    baker_spy_timeline: BakerSpyTimeline,
}

/// Native GetRegisterAlignment is registerAs-first. Spy's stored Villager
/// registerAs therefore overrides a runtime-Evil body; absent that cache,
/// runtime Evil remains registered Evil. Wretch supplies a Minion registerAs
/// on a runtime-Good body. A missing current identity is the grouped ordinary-
/// Good surface on which a natural Wretch may still be hiding.
fn current_knitter_known_registered_evil_at(
    position: u8,
    observation: u8,
    baker_spy_timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<bool> {
    registered_alignment_at_observation(
        position,
        observation,
        baker_spy_timeline,
        scenario,
        state,
    )
    .map(|alignment| alignment == EffectiveAlignment::Evil)
}

fn current_knitter_pair_count(n_cards: u8, registered_evil: &HashSet<u8>) -> i64 {
    if n_cards == 0 {
        return 0;
    }
    (1..=n_cards)
        .filter(|position| {
            let next = if *position == n_cards {
                1
            } else {
                *position + 1
            };
            registered_evil.contains(position) && registered_evil.contains(&next)
        })
        .count() as i64
}

fn current_knitter_worlds(
    observation: u8,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentKnitterSupport> {
    let candidates = anonymous_natural_wretch_candidates(scenario, state);
    let mut worlds = Vec::new();

    fn enumerate(
        index: usize,
        candidates: &[u8],
        required: &mut HashSet<u8>,
        forbidden: &mut HashSet<u8>,
        known_evil: &HashSet<u8>,
        baker_spy_timeline: &BakerSpyTimeline,
        scenario: &Scenario,
        state: &GameState,
        worlds: &mut Vec<CurrentKnitterSupport>,
    ) {
        if index == candidates.len() {
            if !anonymous_wretch_assignment_possible(required, forbidden, scenario, state) {
                return;
            }
            let mut registered_evil = known_evil.clone();
            registered_evil.extend(required.iter().copied());
            worlds.push(CurrentKnitterSupport {
                actual: current_knitter_pair_count(state.n_cards, &registered_evil),
                required_anonymous_wretches: required.clone(),
                forbidden_anonymous_wretches: forbidden.clone(),
                baker_spy_timeline: baker_spy_timeline.clone(),
            });
            return;
        }

        let position = candidates[index];
        forbidden.insert(position);
        enumerate(
            index + 1,
            candidates,
            required,
            forbidden,
            known_evil,
            baker_spy_timeline,
            scenario,
            state,
            worlds,
        );
        forbidden.remove(&position);

        required.insert(position);
        enumerate(
            index + 1,
            candidates,
            required,
            forbidden,
            known_evil,
            baker_spy_timeline,
            scenario,
            state,
            worlds,
        );
        required.remove(&position);
    }

    for baker_spy_timeline in baker_spy_conversion_timelines(scenario, state) {
        let mut known_evil = HashSet::new();
        for position in 1..=state.n_cards {
            if current_knitter_known_registered_evil_at(
                position,
                observation,
                &baker_spy_timeline,
                scenario,
                state,
            ) == Some(true)
            {
                known_evil.insert(position);
            }
        }
        enumerate(
            0,
            &candidates,
            &mut HashSet::new(),
            &mut HashSet::new(),
            &known_evil,
            &baker_spy_timeline,
            scenario,
            state,
            &mut worlds,
        );
    }
    worlds
}

fn current_knitter_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentKnitterSupport> {
    let Some(claimed) = parse_current_knitter_claim(card, source, state) else {
        return Vec::new();
    };
    let worlds = current_knitter_worlds(card.position, scenario, state);
    match truth_status(card.position, scenario, state) {
        TruthStatus::Truthful => worlds
            .into_iter()
            .filter(|world| world.actual == claimed)
            .collect(),
        TruthStatus::Lying => {
            let Some(authored_evil_slots) = current_authored_evil_slots(scenario, state) else {
                return Vec::new();
            };
            let native_upper_exclusive = i64::from(authored_evil_slots.max(2));
            worlds
                .into_iter()
                .filter(|world| claimed < native_upper_exclusive && claimed != world.actual)
                .collect()
        }
    }
}

fn validate_current_knitter(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_knitter_supports(card, scenario, state, source).is_empty()
}

#[cfg(test)]
fn validate_current_knitter_consistency(scenario: &Scenario, state: &GameState) -> bool {
    let mut observations = Vec::new();
    for card in &state.cards {
        if executed_evil_origin_is_unresolved(card.position, scenario, state) {
            continue;
        }
        let Ok(Some(source)) =
            current_passive_payload_source(card, KNITTER_CURRENT_VARIANT_FIELD, "Knitter")
        else {
            continue;
        };
        let supports = current_knitter_supports(card, scenario, state, source);
        if supports.is_empty() {
            return false;
        }
        observations.push(supports);
    }

    fn search(
        index: usize,
        observations: &[Vec<CurrentKnitterSupport>],
        required_wretches: &HashSet<u8>,
        forbidden_wretches: &HashSet<u8>,
        baker_spy_timeline: Option<&BakerSpyTimeline>,
        scenario: &Scenario,
        state: &GameState,
    ) -> bool {
        if index == observations.len() {
            return true;
        }
        for support in &observations[index] {
            if baker_spy_timeline
                .is_some_and(|selected| selected != &support.baker_spy_timeline)
            {
                continue;
            }
            let mut required = required_wretches.clone();
            required.extend(&support.required_anonymous_wretches);
            let mut forbidden = forbidden_wretches.clone();
            forbidden.extend(&support.forbidden_anonymous_wretches);
            if !required.is_disjoint(&forbidden)
                || !anonymous_wretch_assignment_possible(&required, &forbidden, scenario, state)
            {
                continue;
            }
            if search(
                index + 1,
                observations,
                &required,
                &forbidden,
                Some(&support.baker_spy_timeline),
                scenario,
                state,
            ) {
                return true;
            }
        }
        false
    }

    search(
        0,
        &observations,
        &HashSet::new(),
        &HashSet::new(),
        None,
        scenario,
        state,
    )
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentScoutClaim<'a> {
    Numeric { evil_role: &'a str, distance: i64 },
    OneEvil,
}

fn current_scout_distance_in_native_union(distance: i64, n_cards: u8) -> bool {
    // Truthful distance is the shortest circular distance. Bluff distance is
    // independently sampled from 1..=3, even on a smaller board.
    distance > 0 && distance <= i64::from((n_cards / 2).max(3))
}

fn parse_current_scout_claim<'a>(
    info: &'a serde_json::Map<String, serde_json::Value>,
    source: CurrentPassivePayloadSource,
    n_cards: u8,
) -> Option<CurrentScoutClaim<'a>> {
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => (SCOUT_CURRENT_VARIANT_FIELD, 1),
        CurrentPassivePayloadSource::Poet => ("poet_variant", 2),
    };
    if info.get(variant_field).and_then(serde_json::Value::as_str)
        != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }
    if source == CurrentPassivePayloadSource::Poet
        && info.get("copied_role").and_then(serde_json::Value::as_str) != Some("Scout")
    {
        return None;
    }

    if info.len() == fixed_fields + 1
        && info.get("one_evil").and_then(serde_json::Value::as_bool) == Some(true)
    {
        return Some(CurrentScoutClaim::OneEvil);
    }

    if info.len() != fixed_fields + 2 {
        return None;
    }
    let evil_role = poet_canonical_role(info, "evil_role")?;
    let distance = info.get("distance")?.as_i64()?;
    current_scout_distance_in_native_union(distance, n_cards)
        .then_some(CurrentScoutClaim::Numeric { evil_role, distance })
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CurrentScoutLabelSupport {
    Direct,
    RegisterAs(String),
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct AnonymousWretchConstraints {
    required: HashSet<u8>,
    forbidden: HashSet<u8>,
}

impl AnonymousWretchConstraints {
    fn empty() -> Self {
        Self {
            required: HashSet::new(),
            forbidden: HashSet::new(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentRegisterAsSupport {
    register_as: Option<(u8, String)>,
    anonymous_wretches: AnonymousWretchConstraints,
    baker_spy_timeline: BakerSpyTimeline,
}

fn canonical_minion_role(role: &str) -> bool {
    get_card(role).is_some_and(|card| card.faction == Faction::Minion && card.name == role)
}

fn current_wretch_register_as_label_allowed(role: &str, state: &GameState) -> bool {
    if !canonical_minion_role(role) {
        return false;
    }
    state.deck.minions.is_empty()
        || state.deck.minions.iter().any(|authored| {
            get_card(authored).is_some_and(|card| card.faction == Faction::Minion)
                && roles_equal(authored, role)
        })
}

fn current_scout_label_support(
    target: u8,
    observation: u8,
    claimed_role: &str,
    truth: TruthStatus,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<CurrentScoutLabelSupport> {
    let Some(data_role) =
        current_data_role_at_observation(target, observation, timeline, scenario, state)
    else {
        return None;
    };

    if truth == TruthStatus::Truthful
        && current_spy_register_as_surface_at_observation(
            target,
            observation,
            timeline,
            scenario,
            state,
        )?
    {
        return current_medium_spy_register_as_label_allowed(claimed_role, state)
            .then(|| CurrentScoutLabelSupport::RegisterAs(normalize_role(claimed_role)));
    }

    // Truthful Scout names Character.GetRegisterAs(). Wretch is the one
    // modeled current data identity whose register-as role differs: it samples
    // a canonical authored Minion. When the bridge lacks current script Minion
    // metadata, the solver conservatively approximates the native all-ascension
    // fallback with any canonical Minion. Bluff Scout names dataRef directly.
    if truth == TruthStatus::Truthful && roles_equal(&data_role, "Wretch") {
        return current_wretch_register_as_label_allowed(claimed_role, state)
            .then(|| CurrentScoutLabelSupport::RegisterAs(normalize_role(claimed_role)));
    }

    roles_equal(&data_role, claimed_role).then_some(CurrentScoutLabelSupport::Direct)
}

fn current_known_registered_distance(
    anchor: u8,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<Option<i64>> {
    let mut distance = None;
    for position in 1..=state.n_cards {
        if position == anchor {
            continue;
        }
        if registered_alignment_at_observation(
            position,
            observation,
            timeline,
            scenario,
            state,
        )? == EffectiveAlignment::Evil
        {
            let candidate = i64::from(circle_distance(anchor, position, state.n_cards));
            distance = Some(distance.map_or(candidate, |known: i64| known.min(candidate)));
        }
    }
    Some(distance)
}

fn anonymous_natural_wretch_candidates(scenario: &Scenario, state: &GameState) -> Vec<u8> {
    (1..=state.n_cards)
        // Exact Twin/Shaman/generated data is already represented by the
        // ordinary registered-alignment helper, not by this grouped identity.
        .filter(|&position| current_data_role_at(position, scenario, state).is_none())
        .filter(|&position| {
            crate::scenario::scenario_allows_anonymous_natural_outcast_role_at(
                position,
                "Wretch",
                scenario,
                state,
            )
        })
        .collect()
}

fn anonymous_wretch_assignment_possible(
    required: &HashSet<u8>,
    forbidden: &HashSet<u8>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    crate::scenario::scenario_allows_anonymous_natural_outcast_role_assignments(
        required,
        "Wretch",
        forbidden,
        scenario,
        state,
    )
}

fn registered_distance_equal_anonymous_wretch_supports(
    anchor: u8,
    claimed: i64,
    known_distance: Option<i64>,
    anonymous_wretch_candidates: &[u8],
    scenario: &Scenario,
    state: &GameState,
) -> Vec<AnonymousWretchConstraints> {
    if known_distance.is_some_and(|distance| distance < claimed) {
        return Vec::new();
    }

    let forbidden_closer: HashSet<u8> = anonymous_wretch_candidates
        .iter()
        .copied()
        .filter(|position| i64::from(circle_distance(anchor, *position, state.n_cards)) < claimed)
        .collect();
    if known_distance == Some(claimed) {
        return anonymous_wretch_assignment_possible(
            &HashSet::new(), &forbidden_closer, scenario, state,
        )
        .then_some(AnonymousWretchConstraints {
            required: HashSet::new(),
            forbidden: forbidden_closer,
        })
        .into_iter()
        .collect();
    }

    anonymous_wretch_candidates
        .iter()
        .copied()
        .filter(|position| i64::from(circle_distance(anchor, *position, state.n_cards)) == claimed)
        .filter_map(|position| {
            let required = HashSet::from([position]);
            anonymous_wretch_assignment_possible(
                &required, &forbidden_closer, scenario, state,
            )
            .then(|| AnonymousWretchConstraints {
                required,
                forbidden: forbidden_closer.clone(),
            })
        })
        .collect()
}

fn registered_distance_different_anonymous_wretch_supports(
    anchor: u8,
    claimed: i64,
    known_distance: Option<i64>,
    anonymous_wretch_candidates: &[u8],
    scenario: &Scenario,
    state: &GameState,
) -> Vec<AnonymousWretchConstraints> {
    if known_distance.is_some_and(|distance| distance < claimed) {
        return vec![AnonymousWretchConstraints::empty()];
    }

    // A closer Wretch makes the native nearest distance false even if another
    // Wretch is simultaneously forced at the claimed distance.
    let mut supports: Vec<AnonymousWretchConstraints> = anonymous_wretch_candidates
        .iter()
        .copied()
        .filter(|position| i64::from(circle_distance(anchor, *position, state.n_cards)) < claimed)
        .filter_map(|position| {
            let required = HashSet::from([position]);
            anonymous_wretch_assignment_possible(
                &required, &HashSet::new(), scenario, state,
            )
            .then(|| AnonymousWretchConstraints {
                required,
                forbidden: HashSet::new(),
            })
        })
        .collect();

    if known_distance != Some(claimed) {
        let forbidden_equal: HashSet<u8> = anonymous_wretch_candidates
            .iter()
            .copied()
            .filter(|position| {
                i64::from(circle_distance(anchor, *position, state.n_cards)) == claimed
            })
            .collect();
        if anonymous_wretch_assignment_possible(
            &HashSet::new(), &forbidden_equal, scenario, state,
        ) {
            supports.push(AnonymousWretchConstraints {
                required: HashSet::new(),
                forbidden: forbidden_equal,
            });
        }
    }
    supports
}

fn current_scout_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentRegisterAsSupport> {
    if card.position == 0 || card.position > state.n_cards {
        return Vec::new();
    }
    let Some(claim) = parse_current_scout_claim(&card.info_parsed, source, state.n_cards) else {
        return Vec::new();
    };
    let truth = truth_status(card.position, scenario, state);
    let anonymous_wretches = anonymous_natural_wretch_candidates(scenario, state);
    let mut all_supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(card.position, state) {
            continue;
        }
        match claim {
            CurrentScoutClaim::OneEvil => {
                if truth == TruthStatus::Lying {
                    continue;
                }
                let forbidden: HashSet<u8> = anonymous_wretches.iter().copied().collect();
                let supported = (1..=state.n_cards)
                    .filter(|&target| is_runtime_evil_at(target, scenario, state))
                    .any(|target| {
                        current_known_registered_distance(
                            target,
                            card.position,
                            &timeline,
                            scenario,
                            state,
                        ) == Some(None)
                            && anonymous_wretch_assignment_possible(
                                &HashSet::new(),
                                &forbidden,
                                scenario,
                                state,
                            )
                    });
                if supported {
                    all_supports.push(CurrentRegisterAsSupport {
                        register_as: None,
                        anonymous_wretches: AnonymousWretchConstraints {
                            required: HashSet::new(),
                            forbidden,
                        },
                        baker_spy_timeline: timeline,
                    });
                }
            }
            CurrentScoutClaim::Numeric { evil_role, distance } => {
                if truth == TruthStatus::Lying && !(1..=3).contains(&distance) {
                    continue;
                }
                for target in (1..=state.n_cards)
                    .filter(|&target| is_runtime_evil_at(target, scenario, state))
                {
                    let Some(label_support) = current_scout_label_support(
                        target,
                        card.position,
                        evil_role,
                        truth,
                        &timeline,
                        scenario,
                        state,
                    ) else {
                        continue;
                    };
                    let Some(known_distance) = current_known_registered_distance(
                        target,
                        card.position,
                        &timeline,
                        scenario,
                        state,
                    ) else {
                        continue;
                    };
                    let wretch_supports = match truth {
                        TruthStatus::Truthful => {
                            registered_distance_equal_anonymous_wretch_supports(
                                target,
                                distance,
                                known_distance,
                                &anonymous_wretches,
                                scenario,
                                state,
                            )
                        }
                        TruthStatus::Lying => {
                            registered_distance_different_anonymous_wretch_supports(
                                target,
                                distance,
                                known_distance,
                                &anonymous_wretches,
                                scenario,
                                state,
                            )
                        }
                    };
                    let register_as = match label_support {
                        CurrentScoutLabelSupport::Direct => None,
                        CurrentScoutLabelSupport::RegisterAs(role) => Some((target, role)),
                    };
                    for anonymous_wretches in wretch_supports {
                        let support = CurrentRegisterAsSupport {
                            register_as: register_as.clone(),
                            anonymous_wretches,
                            baker_spy_timeline: timeline.clone(),
                        };
                        if !all_supports.contains(&support) {
                            all_supports.push(support);
                        }
                    }
                }
            }
        }
    }
    all_supports
}

fn validate_current_scout(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_scout_supports(card, scenario, state, source).is_empty()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentOracleClaim<'a> {
    Positive {
        targets: [u8; 2],
        minion_role: &'a str,
    },
    NoMinions,
}

fn parse_current_oracle_claim<'a>(
    card: &'a CardInfo,
    source: CurrentPassivePayloadSource,
    n_cards: u8,
) -> Option<CurrentOracleClaim<'a>> {
    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => (ORACLE_CURRENT_VARIANT_FIELD, 1),
        CurrentPassivePayloadSource::Poet => ("poet_variant", 2),
    };
    if info.get(variant_field).and_then(serde_json::Value::as_str)
        != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }
    if source == CurrentPassivePayloadSource::Poet
        && info.get("copied_role").and_then(serde_json::Value::as_str) != Some("Oracle")
    {
        return None;
    }

    if info.len() == fixed_fields + 1
        && info.get("no_minions").and_then(serde_json::Value::as_bool) == Some(true)
        && card.info_text == "There are no minions"
    {
        return Some(CurrentOracleClaim::NoMinions);
    }
    if info.len() != fixed_fields + 2 {
        return None;
    }

    let values = info.get("targets")?.as_array()?;
    if values.len() != 2 {
        return None;
    }
    let first = poet_position_value(values.first(), n_cards)?;
    let second = poet_position_value(values.get(1), n_cards)?;
    if first > second {
        return None;
    }
    let minion_role = poet_canonical_role(info, "minion_role")?;
    if !canonical_minion_role(minion_role)
        || card.info_text != format!("#{first} or #{second} is a {minion_role}")
    {
        return None;
    }

    Some(CurrentOracleClaim::Positive {
        targets: [first, second],
        minion_role,
    })
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentOracleMinionSupport {
    required_anonymous_wretch: Option<u8>,
    register_as: Option<(u8, String)>,
}

fn current_oracle_minion_target_support(
    target: u8,
    observation: u8,
    minion_role: &str,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<CurrentOracleMinionSupport> {
    if current_spy_register_as_surface_at_observation(
        target,
        observation,
        timeline,
        scenario,
        state,
    )? {
        return None;
    }
    match current_data_role_at_observation(target, observation, timeline, scenario, state) {
        Some(data_role) if roles_equal(&data_role, "Wretch") => {
            current_wretch_register_as_label_allowed(minion_role, state).then(|| {
                CurrentOracleMinionSupport {
                    required_anonymous_wretch: None,
                    register_as: Some((target, normalize_role(minion_role))),
                }
            })
        }
        Some(data_role)
            if get_card(&data_role).is_some_and(|card| card.faction == Faction::Minion)
                && roles_equal(&data_role, minion_role) =>
        {
            Some(CurrentOracleMinionSupport {
                required_anonymous_wretch: None,
                register_as: None,
            })
        }
        Some(_) => None,
        None => current_wretch_register_as_label_allowed(minion_role, state).then_some(
            CurrentOracleMinionSupport {
                required_anonymous_wretch: Some(target),
                register_as: Some((target, normalize_role(minion_role))),
            },
        ),
    }
}

/// Return the anonymous seat that must be forbidden from holding Wretch data,
/// or `None` when this is already a modeled registered-Good target.
fn current_oracle_good_target_forbidden_wretch(
    target: u8,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<Option<u8>> {
    let data_role = current_data_role_at_observation(target, observation, timeline, scenario, state);
    if registered_alignment_at_observation(
        target,
        observation,
        timeline,
        scenario,
        state,
    )? != EffectiveAlignment::Good
        || data_role.as_deref().is_some_and(|role| roles_equal(role, "Wretch"))
    {
        return None;
    }
    Some(data_role.is_none().then_some(target))
}

fn current_oracle_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentRegisterAsSupport> {
    if card.position == 0 || card.position > state.n_cards {
        return Vec::new();
    }
    let Some(claim) = parse_current_oracle_claim(card, source, state.n_cards) else {
        return Vec::new();
    };
    let truth = truth_status(card.position, scenario, state);
    let anonymous_wretches = anonymous_natural_wretch_candidates(scenario, state);
    let mut all_supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(card.position, state) {
            continue;
        }
        match claim {
            CurrentOracleClaim::NoMinions => {
                if truth == TruthStatus::Lying {
                    continue;
                }
                let mut known_current_minion = false;
                let mut coherent = true;
                for position in 1..=state.n_cards {
                    let spy_register_as = current_spy_register_as_surface_at_observation(
                        position,
                        card.position,
                        &timeline,
                        scenario,
                        state,
                    );
                    let Some(spy_register_as) = spy_register_as else {
                        coherent = false;
                        break;
                    };
                    if spy_register_as {
                        continue;
                    }
                    if current_data_role_at_observation(
                        position,
                        card.position,
                        &timeline,
                        scenario,
                        state,
                    )
                    .is_some_and(|role| {
                        roles_equal(&role, "Wretch")
                            || get_card(&role)
                                .is_some_and(|card| card.faction == Faction::Minion)
                    }) {
                        known_current_minion = true;
                        break;
                    }
                }
                if !coherent || known_current_minion {
                    continue;
                }
                let forbidden: HashSet<u8> = anonymous_wretches.iter().copied().collect();
                if anonymous_wretch_assignment_possible(
                    &HashSet::new(),
                    &forbidden,
                    scenario,
                    state,
                ) {
                    all_supports.push(CurrentRegisterAsSupport {
                        register_as: None,
                        anonymous_wretches: AnonymousWretchConstraints {
                            required: HashSet::new(),
                            forbidden,
                        },
                        baker_spy_timeline: timeline,
                    });
                }
            }
            CurrentOracleClaim::Positive {
                targets,
                minion_role,
            } if truth == TruthStatus::Lying => {
                if targets[0] == targets[1]
                    || !current_wretch_register_as_label_allowed(minion_role, state)
                {
                    continue;
                }
                let mut forbidden = HashSet::new();
                let mut supported = true;
                for target in targets {
                    let Some(anonymous_forbidden) =
                        current_oracle_good_target_forbidden_wretch(
                            target,
                            card.position,
                            &timeline,
                            scenario,
                            state,
                        )
                    else {
                        supported = false;
                        break;
                    };
                    if let Some(position) = anonymous_forbidden {
                        forbidden.insert(position);
                    }
                }
                if supported
                    && anonymous_wretch_assignment_possible(
                        &HashSet::new(),
                        &forbidden,
                        scenario,
                        state,
                    )
                {
                    all_supports.push(CurrentRegisterAsSupport {
                        register_as: None,
                        anonymous_wretches: AnonymousWretchConstraints {
                            required: HashSet::new(),
                            forbidden,
                        },
                        baker_spy_timeline: timeline,
                    });
                }
            }
            CurrentOracleClaim::Positive {
                targets,
                minion_role,
            } => {
                let orientations = if targets[0] == targets[1] {
                    vec![(targets[0], targets[1])]
                } else {
                    vec![(targets[0], targets[1]), (targets[1], targets[0])]
                };
                for (minion_target, good_target) in orientations {
                    let Some(minion_support) = current_oracle_minion_target_support(
                        minion_target,
                        card.position,
                        minion_role,
                        &timeline,
                        scenario,
                        state,
                    ) else {
                        continue;
                    };
                    let Some(anonymous_good_forbidden) =
                        current_oracle_good_target_forbidden_wretch(
                            good_target,
                            card.position,
                            &timeline,
                            scenario,
                            state,
                        )
                    else {
                        continue;
                    };
                    let required: HashSet<u8> = minion_support
                        .required_anonymous_wretch
                        .into_iter()
                        .collect();
                    let forbidden: HashSet<u8> = anonymous_good_forbidden.into_iter().collect();
                    if !anonymous_wretch_assignment_possible(
                        &required,
                        &forbidden,
                        scenario,
                        state,
                    ) {
                        continue;
                    }
                    let support = CurrentRegisterAsSupport {
                        register_as: minion_support.register_as,
                        anonymous_wretches: AnonymousWretchConstraints {
                            required,
                            forbidden,
                        },
                        baker_spy_timeline: timeline.clone(),
                    };
                    if !all_supports.contains(&support) {
                        all_supports.push(support);
                    }
                }
            }
        }
    }
    all_supports
}

fn validate_current_oracle(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_oracle_supports(card, scenario, state, source).is_empty()
}

#[cfg(test)]
fn validate_current_register_as_consistency(
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let mut current_cards = Vec::new();
    for card in &state.cards {
        if executed_evil_origin_is_unresolved(card.position, scenario, state) {
            continue;
        }
        let Ok(Some(source)) = current_passive_payload_source(
            card,
            SCOUT_CURRENT_VARIANT_FIELD,
            "Scout",
        ) else {
            if let Ok(Some(source)) = current_passive_payload_source(
                card,
                ORACLE_CURRENT_VARIANT_FIELD,
                "Oracle",
            ) {
                current_cards.push((card, source, true));
            }
            continue;
        };
        current_cards.push((card, source, false));
    }
    // One observation cannot disagree with itself, and its ordinary validator
    // owns all schema, candidate, and distance checks. Avoid replaying the
    // anonymous-Outcast allocator on the common one-observation path.
    if current_cards.len() <= 1 {
        return true;
    }

    let mut observations = Vec::new();
    for (card, source, is_oracle) in current_cards {
        let supports = if is_oracle {
            current_oracle_supports(card, scenario, state, source)
        } else {
            current_scout_supports(card, scenario, state, source)
        };
        if supports.is_empty() {
            return false;
        }
        observations.push(supports);
    }

    fn search(
        index: usize,
        observations: &[Vec<CurrentRegisterAsSupport>],
        selected_register_as: &mut HashMap<u8, String>,
    ) -> bool {
        if index == observations.len() {
            return true;
        }
        for support in &observations[index] {
            let Some((target, role)) = support.register_as.as_ref() else {
                if search(index + 1, observations, selected_register_as) {
                    return true;
                }
                continue;
            };
            if selected_register_as
                .get(target)
                .is_some_and(|selected| selected != role)
            {
                continue;
            }
            let inserted = selected_register_as.insert(*target, role.clone()).is_none();
            if search(index + 1, observations, selected_register_as) {
                return true;
            }
            if inserted {
                selected_register_as.remove(target);
            }
        }
        false
    }

    // This focused helper is retained for explicit-label unit isolation.
    // check_scenario joins both labels and anonymous natural-Wretch seats
    // across all current passive providers in
    // validate_current_hidden_surface_consistency.
    search(0, &observations, &mut HashMap::new())
}

fn validate_scout(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    match current_passive_payload_source(card, SCOUT_CURRENT_VARIANT_FIELD, "Scout") {
        Ok(Some(source)) => return validate_current_scout(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

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
        legacy_current_evil_role_at(p, scenario, state)
            .is_some_and(|role| roles_equal(&role, evil_role))
    });
    let target_pos = match target_pos {
        Some(p) => p,
        None => {
            // A historical executed Evil can retain an Unknown origin role,
            // so its authored identity may still be the named role. Without
            // such an unresolved seat, absence is a definite false statement:
            // truthful Scout rejects it and a lying Scout accepts it.
            let unresolved_evil_role = (1..=n).any(|p| {
                known_evil_role(p, scenario, state)
                    .is_some_and(|role| normalize_role(role) == "unknown")
            });
            return unresolved_evil_role || truth == TruthStatus::Lying;
        }
    };

    let other_evil: Vec<u8> = (1..=n)
        .filter(|&p| p != target_pos && effective_alignment(p, scenario, state) == EffectiveAlignment::Evil)
        .collect();
    if other_evil.is_empty() {
        // Historical direct-Scout captures used distance zero for the native
        // one-Evil sentinel. Current provenance-marked direct and Poet payloads
        // take the explicit `one_evil` path before this legacy numeric helper,
        // so only that legacy encoding is true here; an actual positive-distance
        // sentence remains false and is inverted for a lying source.
        return if claimed_dist == 0 {
            truth == TruthStatus::Truthful
        } else {
            truth == TruthStatus::Lying
        };
    }

    let actual = other_evil.iter().map(|&ep| circle_distance(target_pos, ep, n) as i64).min().unwrap();

    if truth == TruthStatus::Truthful { claimed_dist == actual }
    else { claimed_dist != actual }
}

fn validate_bard(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    match current_passive_payload_source(card, BARD_CURRENT_VARIANT_FIELD, "Bard") {
        Ok(Some(source)) => return validate_current_bard(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

    // Preserve archived observations byte-for-byte on their permissive
    // scalar predicate. Fresh direct and Poet observations use the closed
    // native schema and output domain above.
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

fn current_bard_claim_text(internal_distance: i64) -> Option<String> {
    match internal_distance {
        0 => Some("There are no Corrupted characters".to_string()),
        1 => Some("I am 1 card away from Corrupted character".to_string()),
        distance if distance > 1 => {
            Some(format!("I am {distance} cards away from Corrupted character"))
        }
        _ => None,
    }
}

fn parse_current_bard_claim(
    card: &CardInfo,
    source: CurrentPassivePayloadSource,
    state: &GameState,
) -> Option<i64> {
    if card.position == 0 || card.position > state.n_cards {
        return None;
    }
    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => {
            if card.apparent_role != "Bard" {
                return None;
            }
            (BARD_CURRENT_VARIANT_FIELD, 1)
        }
        CurrentPassivePayloadSource::Poet => {
            if card.apparent_role != "Poet"
                || info.get("copied_role").and_then(serde_json::Value::as_str)
                    != Some("Bard")
            {
                return None;
            }
            ("poet_variant", 2)
        }
    };
    if info.len() != fixed_fields + 1
        || info.get(variant_field).and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }

    let serialized = info.get("corruption_distance")?.as_i64()?;
    let internal = match serialized {
        -1 => 0,
        distance if distance > 0 => distance,
        _ => return None,
    };
    // Truth can reach half the physical circle. Bluff uses the fixed native
    // {0,1,2,3} domain even on tiny boards, so current payloads accept the
    // union rather than deriving their range only from board geometry.
    let maximum = i64::from(state.n_cards / 2).max(3);
    if internal > maximum
        || current_bard_claim_text(internal).as_deref() != Some(card.info_text.as_str())
    {
        return None;
    }
    Some(internal)
}

fn current_bard_actual_distance(actor: u8, scenario: &Scenario, state: &GameState) -> i64 {
    scenario
        .corrupted
        .iter()
        .copied()
        .filter(|position| *position != actor && *position > 0 && *position <= state.n_cards)
        .map(|position| i64::from(circle_distance(actor, position, state.n_cards)))
        .min()
        .unwrap_or(0)
}

fn current_bard_claim_supported(claimed: i64, actual: i64, truth: TruthStatus) -> bool {
    match truth {
        TruthStatus::Truthful => claimed == actual,
        TruthStatus::Lying => (0..=3).contains(&claimed) && claimed != actual,
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentBardSupport {
    anonymous_wretches: AnonymousWretchConstraints,
    raw_bluff: Option<(u8, String)>,
    forbidden_raw_bluff: Option<(u8, String)>,
    baker_spy_timeline: BakerSpyTimeline,
}

fn current_bard_raw_provider_truth(
    actor: u8,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> TruthStatus {
    if scenario.corrupted.contains(&actor) {
        return TruthStatus::Lying;
    }
    let current_role = current_data_role_at_observation(
        actor,
        observation,
        timeline,
        scenario,
        state,
    );
    let healthy_bluff = scenario.puppet_position == Some(actor)
        || scenario.doppelganger_position == Some(actor)
        || current_role
            .as_deref()
            .is_some_and(|role| roles_equal(role, "Doppelganger"));
    if healthy_bluff {
        TruthStatus::Truthful
    } else {
        // Reaching the apparent Bard/Poet through a raw bluff proves the
        // non-null pointer which makes an otherwise clean runtime-Good body
        // lie. Runtime Evil also dispatches the bluff role through BluffAct.
        TruthStatus::Lying
    }
}

fn current_bard_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentBardSupport> {
    let Some(claimed) = parse_current_bard_claim(card, source, state) else {
        return Vec::new();
    };
    // An unresolved Evil identity may have been any corruption/status writer
    // or current-data mover. A later executed-role overlay cannot reconstruct
    // the Start history which Bard samples synchronously on Day.
    if current_has_unresolved_start_identity(scenario, state) {
        return Vec::new();
    }

    let actual = current_bard_actual_distance(card.position, scenario, state);
    let provider_role = match source {
        CurrentPassivePayloadSource::Direct => "Bard",
        CurrentPassivePayloadSource::Poet => "Poet",
    };
    let anonymous_candidates = anonymous_natural_wretch_candidates(scenario, state);
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(card.position, state) {
            continue;
        }
        let current_role = current_data_role_at_observation(
            card.position,
            card.position,
            &timeline,
            scenario,
            state,
        );
        let raw_bluff_holder = current_medium_raw_bluff_holder_at(
            card.position,
            card.position,
            &timeline,
            scenario,
            state,
        );
        // Character.Act dispatches a runtime-Evil current role through Act
        // when a non-null bluff role follows it. The real Bard/Poet callback
        // is therefore truthful and occurs first. A final Bard-shaped newest
        // event from that real callback additionally proves the later bluff
        // role was not Bard/Poet; that exact raw role would run BluffAct and
        // overwrite it with the lying provider result.
        let runtime_evil_real_truth = is_runtime_evil_at(card.position, scenario, state)
            && raw_bluff_holder != CurrentMediumRawBluffHolder::Impossible;
        let real_provider_truth = if runtime_evil_real_truth {
            TruthStatus::Truthful
        } else {
            truth_status(card.position, scenario, state)
        };
        if current_role
            .as_deref()
            .is_some_and(|role| roles_equal(role, provider_role))
            && current_bard_claim_supported(claimed, actual, real_provider_truth)
        {
            let support = CurrentBardSupport {
                anonymous_wretches: AnonymousWretchConstraints::empty(),
                raw_bluff: None,
                forbidden_raw_bluff: runtime_evil_real_truth
                    .then(|| (card.position, normalize_role(provider_role))),
                baker_spy_timeline: timeline.clone(),
            };
            if !supports.contains(&support) {
                supports.push(support);
            }
        }

        if raw_bluff_holder == CurrentMediumRawBluffHolder::Impossible
            || !current_bard_claim_supported(
                claimed,
                actual,
                current_bard_raw_provider_truth(
                    card.position,
                    card.position,
                    &timeline,
                    scenario,
                    state,
                ),
            )
        {
            continue;
        }

        let mut anonymous_wretches = AnonymousWretchConstraints::empty();
        if anonymous_candidates.contains(&card.position) {
            // Natural Wretch has a base-null raw bluff. A Bard/Poet bluff
            // surface at that physical seat therefore excludes that grouped
            // identity in the same assignment shared by all current providers.
            anonymous_wretches.forbidden.insert(card.position);
            if !anonymous_wretch_assignment_possible(
                &anonymous_wretches.required,
                &anonymous_wretches.forbidden,
                scenario,
                state,
            ) {
                continue;
            }
        }
        let support = CurrentBardSupport {
            anonymous_wretches,
            raw_bluff: Some((card.position, normalize_role(provider_role))),
            forbidden_raw_bluff: None,
            baker_spy_timeline: timeline,
        };
        if !supports.contains(&support) {
            supports.push(support);
        }
    }
    supports
}

fn validate_current_bard(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_bard_supports(card, scenario, state, source).is_empty()
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
    match current_passive_payload_source(card, ORACLE_CURRENT_VARIANT_FIELD, "Oracle") {
        Ok(Some(source)) => return validate_current_oracle(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

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
        legacy_current_evil_role_at(t, scenario, state)
            .is_some_and(|role| roles_equal(&role, minion_role))
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

fn current_medium_text(target: u8, role: &str) -> String {
    if roles_equal(role, "Drunk") {
        format!("#{target} is actually a\nDrunk")
    } else {
        format!("#{target} is a real\n{role}")
    }
}

fn parse_current_medium_claim<'a>(
    card: &'a CardInfo,
    source: CurrentPassivePayloadSource,
    state: &GameState,
) -> Option<(u8, &'a str)> {
    if card.position == 0 || card.position > state.n_cards {
        return None;
    }
    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => (MEDIUM_CURRENT_VARIANT_FIELD, 1),
        CurrentPassivePayloadSource::Poet => ("poet_variant", 2),
    };
    if info.len() != fixed_fields + 2
        || info.get(variant_field).and_then(serde_json::Value::as_str) != Some(POET_CURRENT_VARIANT)
        || (source == CurrentPassivePayloadSource::Poet
            && info.get("copied_role").and_then(serde_json::Value::as_str) != Some("Medium"))
    {
        return None;
    }
    let target = poet_position_value(info.get("good_position"), state.n_cards)?;
    let role = poet_canonical_role(info, "good_role")?;
    (card.info_text == current_medium_text(target, role)).then_some((target, role))
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentMediumRawBluffHolder {
    Impossible,
    Possible,
    Proven,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentMediumSupport {
    required_anonymous_wretches: HashSet<u8>,
    forbidden_anonymous_wretches: HashSet<u8>,
    register_as: Option<(u8, String)>,
    raw_bluff: Option<(u8, String)>,
    baker_spy_timeline: BakerSpyTimeline,
}

fn empty_current_medium_support(timeline: &BakerSpyTimeline) -> CurrentMediumSupport {
    CurrentMediumSupport {
        required_anonymous_wretches: HashSet::new(),
        forbidden_anonymous_wretches: HashSet::new(),
        register_as: None,
        raw_bluff: None,
        baker_spy_timeline: timeline.clone(),
    }
}

fn current_spy_register_as_surface_at(
    position: u8,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let exact_executed_spy_overlay = scenario
        .evil_positions
        .get(&position)
        .is_some_and(|role| normalize_role(role) == "unknown")
        && state
            .executed_evil_roles
            .get(&position)
            .is_some_and(|role| normalize_role(role) == "spy");

    exact_executed_spy_overlay
        || stable_evil_origin_role_at(position, scenario, state)
        .is_some_and(|role| normalize_role(role) == "spy")
        || current_data_role_at(position, scenario, state)
            .is_some_and(|role| normalize_role(&role) == "spy")
}

fn current_medium_spy_register_as_label_allowed(role: &str, state: &GameState) -> bool {
    get_card(role).is_some_and(|card| {
        card.name == role
            && card.faction == Faction::Villager
            && state
                .deck
                .villagers
                .iter()
                .any(|authored| roles_equal(authored, role))
    })
}

fn current_medium_mover_history_possible_at(
    position: u8,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let twin = scenario.twin_trace.as_ref().is_some_and(|trace| {
        if trace.actor_position == position {
            return true;
        }
        matches!(
            &trace.outcome,
            crate::types::TwinStartOutcome::Swap {
                neighbor_position,
                ..
            } if *neighbor_position == position
        )
    });
    let shaman = scenario.shaman_trace.as_ref().is_some_and(|trace| {
        trace.source_position == position || trace.target_position == position
    });
    let chancellor = scenario.chancellor_added_outcast_position() == Some(position)
        || scenario
            .chancellor_original_villager_positions()
            .contains(&position);
    let baker = current_data_role_at(position, scenario, state)
        .is_some_and(|role| roles_equal(&role, "Baker"))
        && medium_uses_baker_history(position, scenario, state);
    twin || shaman || chancellor || baker
}

fn current_medium_known_role_matches(
    target: u8,
    data_role: &str,
    claimed_role: &str,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    roles_equal(data_role, claimed_role)
        || (roles_equal(data_role, "Baker")
            && medium_uses_baker_history(target, scenario, state)
            && validate_baker_history(scenario, state))
}

fn current_medium_anonymous_role_possible(
    target: u8,
    claimed_role: &str,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let Some(card) = get_card(claimed_role) else {
        return false;
    };
    match card.faction {
        Faction::Villager => state
            .deck
            .villagers
            .iter()
            .any(|authored| roles_equal(authored, claimed_role)),
        Faction::Outcast if !roles_equal(claimed_role, "Wretch") => {
            crate::scenario::scenario_allows_anonymous_natural_outcast_role_at(
                target,
                claimed_role,
                scenario,
                state,
            )
        }
        Faction::Minion | Faction::Demon => {
            current_medium_mover_history_possible_at(target, scenario, state)
        }
        Faction::Outcast => false,
    }
}

fn current_medium_truth_target_support(
    target: u8,
    observation: u8,
    claimed_role: &str,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<CurrentMediumSupport> {
    let mut support = empty_current_medium_support(timeline);

    // Spy is the important register-as-first exception to the shared helper's
    // runtime-Evil shortcut. A stable Spy keeps its physical cache; a represented
    // seat whose current provider is Spy can execute Spy Start and cache its own
    // Villager (for example Shaman's immediate copied Start). This is not a
    // generic claim that Twin moves another Character's physical registerAs.
    if current_spy_register_as_surface_at_observation(
        target,
        observation,
        timeline,
        scenario,
        state,
    )? {
        if !current_medium_spy_register_as_label_allowed(claimed_role, state) {
            return None;
        }
        support.register_as = Some((target, normalize_role(claimed_role)));
        return Some(support);
    }

    if is_runtime_evil_at(target, scenario, state) {
        return None;
    }
    match current_data_role_at_observation(target, observation, timeline, scenario, state) {
        Some(data_role) if roles_equal(&data_role, "Wretch") => None,
        Some(data_role) => {
            current_medium_known_role_matches(target, &data_role, claimed_role, scenario, state)
                .then_some(support)
        }
        None => {
            if !current_medium_anonymous_role_possible(target, claimed_role, scenario, state) {
                return None;
            }
            support.forbidden_anonymous_wretches.insert(target);
            Some(support)
        }
    }
}

fn current_medium_require_registered_evil(
    position: u8,
    observation: u8,
    support: &mut CurrentMediumSupport,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let Some(spy_register_as) = current_spy_register_as_surface_at_observation(
        position,
        observation,
        timeline,
        scenario,
        state,
    ) else {
        return false;
    };
    if spy_register_as {
        return false;
    }
    if is_runtime_evil_at(position, scenario, state) {
        return true;
    }
    match current_data_role_at_observation(position, observation, timeline, scenario, state) {
        Some(role) => roles_equal(&role, "Wretch"),
        None => {
            support.required_anonymous_wretches.insert(position);
            true
        }
    }
}

fn current_medium_raw_bluff_holder_at(
    position: u8,
    actor: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> CurrentMediumRawBluffHolder {
    let acquisition_surface = match baker_spy_observation_phase(position, actor, timeline, state) {
        Some(BakerSpyObservationPhase::BeforeConversion) => {
            CurrentMediumRawBluffHolder::Proven
        }
        Some(
            BakerSpyObservationPhase::PendingRegisterAsReset | BakerSpyObservationPhase::Reset,
        ) => {
            // Baker's synchronous InitWithNoReset clears Character.bluff. The
            // later internal Reveal cannot repopulate it because Baker's base
            // selector is null; stale registerAs/bluffRole do not change that.
            return CurrentMediumRawBluffHolder::Impossible;
        }
        Some(BakerSpyObservationPhase::Unaffected) => {
            if is_runtime_evil_at(position, scenario, state)
                || scenario.puppet_position == Some(position)
                || scenario.drunk_position == Some(position)
                || scenario.doppelganger_position == Some(position)
                || current_spy_register_as_surface_at(position, scenario, state)
            {
                CurrentMediumRawBluffHolder::Proven
            } else if current_data_role_at(position, scenario, state)
                .as_deref()
                .is_some_and(|role| {
                    matches!(
                        normalize_role(role).as_str(),
                        "drunk" | "doppelganger" | "mutant"
                    ) || get_card(role).is_some_and(|card| {
                        matches!(card.faction, Faction::Minion | Faction::Demon)
                    })
                })
            {
                CurrentMediumRawBluffHolder::Proven
            } else if current_medium_mover_history_possible_at(position, scenario, state)
                || current_data_role_at(position, scenario, state).is_none()
            {
                CurrentMediumRawBluffHolder::Possible
            } else {
                CurrentMediumRawBluffHolder::Impossible
            }
        }
        None => return CurrentMediumRawBluffHolder::Impossible,
    };

    if acquisition_surface == CurrentMediumRawBluffHolder::Impossible || position == actor {
        return acquisition_surface;
    }

    // A non-self raw bluff must already have been acquired when the Medium's
    // own Reveal reaches BluffAct. Current live states certify their complete
    // click order with the same provenance used by Baker chronology. Archived
    // and manually entered orders remain conservative because they cannot
    // prove which acquisition ran first.
    if state.baker_rule_version.as_deref() == Some(BAKER_CURRENT_RULE) {
        let actor_index = state.reveal_order.iter().position(|seen| *seen == actor);
        let holder_index = state
            .reveal_order
            .iter()
            .position(|seen| *seen == position);
        match (actor_index, holder_index) {
            (Some(actor_index), Some(holder_index)) if holder_index < actor_index => {
                return acquisition_surface;
            }
            (Some(_), Some(_) | None) => return CurrentMediumRawBluffHolder::Impossible,
            (None, _) => {}
        }
    }
    acquisition_surface
}

fn current_medium_truth_supports(
    actor: u8,
    target: u8,
    claimed_role: &str,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentMediumSupport> {
    let Some(mut support) = current_medium_truth_target_support(
        target,
        actor,
        claimed_role,
        timeline,
        scenario,
        state,
    )
    else {
        return Vec::new();
    };

    // Native removes the actor exactly once only when the fresh registered-
    // Good pool contains more than one occurrence. A self result therefore
    // proves that every other physical occurrence registered Evil in the same
    // hidden-Outcast assignment.
    if target == actor {
        for position in 1..=state.n_cards {
            if position != actor
                && !current_medium_require_registered_evil(
                    position,
                    actor,
                    &mut support,
                    timeline,
                    scenario,
                    state,
                )
            {
                return Vec::new();
            }
        }
    }

    if !support
        .required_anonymous_wretches
        .is_disjoint(&support.forbidden_anonymous_wretches)
        || !anonymous_wretch_assignment_possible(
            &support.required_anonymous_wretches,
            &support.forbidden_anonymous_wretches,
            scenario,
            state,
        )
    {
        return Vec::new();
    }
    vec![support]
}

fn current_medium_bluff_supports(
    actor: u8,
    target: u8,
    claimed_role: &str,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentMediumSupport> {
    if current_medium_raw_bluff_holder_at(target, actor, timeline, scenario, state)
        == CurrentMediumRawBluffHolder::Impossible
    {
        return Vec::new();
    }

    // Bluff first builds the raw non-null-holder pool without the actor. It
    // falls back to the full holder pool only when that first pool is empty.
    // Because an unrepresented raw pointer is a real possibility, self is
    // sound only when every other seat is proved base-null with no mover or
    // reveal-history ambiguity.
    if target == actor
        && (1..=state.n_cards).any(|position| {
            position != actor
                && current_medium_raw_bluff_holder_at(
                    position,
                    actor,
                    timeline,
                    scenario,
                    state,
                )
                    != CurrentMediumRawBluffHolder::Impossible
        })
    {
        return Vec::new();
    }

    let mut support = empty_current_medium_support(timeline);
    if current_data_role_at_observation(target, actor, timeline, scenario, state).is_none()
        && !current_medium_mover_history_possible_at(target, scenario, state)
    {
        // An anonymous natural Wretch has base-null bluff data. Without a
        // represented mover/reveal history that could preserve an earlier raw
        // pointer, selecting this anonymous holder proves it is not Wretch.
        support.forbidden_anonymous_wretches.insert(target);
        if !anonymous_wretch_assignment_possible(
            &support.required_anonymous_wretches,
            &support.forbidden_anonymous_wretches,
            scenario,
            state,
        ) {
            return Vec::new();
        }
    }
    let normalized = normalize_role(claimed_role);
    // Scenario has neither the persistent Character.bluff pointer nor its value.
    // Once a holder surface is supported, every canonical raw CharacterData label
    // remains possible; deck/faction narrowing would invent certainty across
    // delayed Reveal, Twin/Shaman writes, and stale persisted bluff compositions.
    support.raw_bluff = Some((target, normalized.clone()));
    if current_spy_register_as_surface_at_observation(
        target,
        actor,
        timeline,
        scenario,
        state,
    ) == Some(true)
    {
        if !current_medium_spy_register_as_label_allowed(claimed_role, state) {
            return Vec::new();
        }
        // Spy's raw bluff and register-as use the same cached Villager record.
        support.register_as = Some((target, normalized));
    }
    vec![support]
}

fn current_medium_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentMediumSupport> {
    let Some((target, claimed_role)) = parse_current_medium_claim(card, source, state) else {
        return Vec::new();
    };
    let truth = truth_status(card.position, scenario, state);
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(card.position, state) {
            continue;
        }
        supports.extend(match truth {
            TruthStatus::Truthful => current_medium_truth_supports(
                card.position,
                target,
                claimed_role,
                &timeline,
                scenario,
                state,
            ),
            TruthStatus::Lying => current_medium_bluff_supports(
                card.position,
                target,
                claimed_role,
                &timeline,
                scenario,
                state,
            ),
        });
    }
    supports
}

fn validate_current_medium(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_medium_supports(card, scenario, state, source).is_empty()
}

#[cfg(test)]
fn validate_current_medium_consistency(scenario: &Scenario, state: &GameState) -> bool {
    let mut observations = Vec::new();
    for card in &state.cards {
        if executed_evil_origin_is_unresolved(card.position, scenario, state) {
            continue;
        }
        let Ok(Some(source)) =
            current_passive_payload_source(card, MEDIUM_CURRENT_VARIANT_FIELD, "Medium")
        else {
            continue;
        };
        let supports = current_medium_supports(card, scenario, state, source);
        if supports.is_empty() {
            return false;
        }
        observations.push(supports);
    }
    if observations.len() <= 1 {
        return true;
    }

    fn search(
        index: usize,
        observations: &[Vec<CurrentMediumSupport>],
        required_wretches: &HashSet<u8>,
        forbidden_wretches: &HashSet<u8>,
        register_as: &HashMap<u8, String>,
        raw_bluffs: &HashMap<u8, String>,
        scenario: &Scenario,
        state: &GameState,
    ) -> bool {
        if index == observations.len() {
            return true;
        }
        for support in &observations[index] {
            let mut required = required_wretches.clone();
            required.extend(&support.required_anonymous_wretches);
            let mut forbidden = forbidden_wretches.clone();
            forbidden.extend(&support.forbidden_anonymous_wretches);
            if !required.is_disjoint(&forbidden)
                || !anonymous_wretch_assignment_possible(&required, &forbidden, scenario, state)
            {
                continue;
            }

            let mut selected_register_as = register_as.clone();
            if let Some((position, role)) = support.register_as.as_ref() {
                if selected_register_as
                    .get(position)
                    .is_some_and(|selected| selected != role)
                {
                    continue;
                }
                selected_register_as.insert(*position, role.clone());
            }
            let mut selected_raw_bluffs = raw_bluffs.clone();
            if let Some((position, role)) = support.raw_bluff.as_ref() {
                if selected_raw_bluffs
                    .get(position)
                    .is_some_and(|selected| selected != role)
                {
                    continue;
                }
                selected_raw_bluffs.insert(*position, role.clone());
            }
            if search(
                index + 1,
                observations,
                &required,
                &forbidden,
                &selected_register_as,
                &selected_raw_bluffs,
                scenario,
                state,
            ) {
                return true;
            }
        }
        false
    }

    search(
        0,
        &observations,
        &HashSet::new(),
        &HashSet::new(),
        &HashMap::new(),
        &HashMap::new(),
        scenario,
        state,
    )
}

fn validate_medium(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    match current_passive_payload_source(card, MEDIUM_CURRENT_VARIANT_FIELD, "Medium") {
        Ok(Some(source)) => return validate_current_medium(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

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

fn current_hunter_distance_in_native_union(distance: i64, n_cards: u8) -> bool {
    if n_cards == 0 || distance < 0 {
        return false;
    }
    let distance = distance as u64;
    (n_cards == 1 && distance == 0)
        || (distance >= 1 && distance <= u64::from(n_cards / 2))
        || distance == u64::from(n_cards - 1)
}

fn parse_current_hunter_distance(
    info: &serde_json::Map<String, serde_json::Value>,
    source: CurrentPassivePayloadSource,
    n_cards: u8,
) -> Option<i64> {
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => (HUNTER_CURRENT_VARIANT_FIELD, 1),
        CurrentPassivePayloadSource::Poet => ("poet_variant", 2),
    };
    if info.len() != fixed_fields + 1
        || info.get(variant_field).and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }
    if source == CurrentPassivePayloadSource::Poet
        && info.get("copied_role").and_then(serde_json::Value::as_str) != Some("Hunter")
    {
        return None;
    }
    let distance = info.get("distance")?.as_i64()?;
    current_hunter_distance_in_native_union(distance, n_cards).then_some(distance)
}

fn validate_current_hunter(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_hunter_supports(card, scenario, state, source).is_empty()
}

fn current_hunter_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentRegisterAsSupport> {
    if card.position == 0 || card.position > state.n_cards {
        return Vec::new();
    }
    let Some(claimed) = parse_current_hunter_distance(&card.info_parsed, source, state.n_cards)
    else {
        return Vec::new();
    };

    let anonymous_wretches = anonymous_natural_wretch_candidates(scenario, state);
    let truth = truth_status(card.position, scenario, state);
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(card.position, state) {
            continue;
        }
        let Some(known_distance) = current_known_registered_distance(
            card.position,
            card.position,
            &timeline,
            scenario,
            state,
        ) else {
            continue;
        };
        let known_distance = known_distance.or(Some(i64::from(state.n_cards - 1)));
        let wretch_supports = match truth {
            TruthStatus::Truthful => registered_distance_equal_anonymous_wretch_supports(
                card.position,
                claimed,
                known_distance,
                &anonymous_wretches,
                scenario,
                state,
            ),
            TruthStatus::Lying => {
                let maximum_bluff = i64::from(state.n_cards / 2);
                if !(1..=maximum_bluff).contains(&claimed) {
                    continue;
                }
                registered_distance_different_anonymous_wretch_supports(
                    card.position,
                    claimed,
                    known_distance,
                    &anonymous_wretches,
                    scenario,
                    state,
                )
            }
        };
        supports.extend(wretch_supports.into_iter().map(|anonymous_wretches| {
            CurrentRegisterAsSupport {
                register_as: None,
                anonymous_wretches,
                baker_spy_timeline: timeline.clone(),
            }
        }));
    }
    supports
}

fn validate_hunter(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    match current_passive_payload_source(card, HUNTER_CURRENT_VARIANT_FIELD, "Hunter") {
        Ok(Some(source)) => return validate_current_hunter(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentEmpressSupport {
    anonymous_wretches: AnonymousWretchConstraints,
    baker_spy_timeline: BakerSpyTimeline,
}

fn current_empress_claim_text(targets: &[u8]) -> Option<String> {
    let [first, second, third] = targets else {
        return None;
    };
    Some(format!(
        "One is Evil:\n#{first}, #{second} or #{third}"
    ))
}

fn parse_current_empress_targets(
    card: &CardInfo,
    source: CurrentPassivePayloadSource,
    state: &GameState,
) -> Option<Vec<u8>> {
    if card.position == 0 || card.position > state.n_cards {
        return None;
    }
    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => {
            if card.apparent_role != "Empress" {
                return None;
            }
            (EMPRESS_CURRENT_VARIANT_FIELD, 1)
        }
        CurrentPassivePayloadSource::Poet => {
            if card.apparent_role != "Poet"
                || info.get("copied_role").and_then(serde_json::Value::as_str)
                    != Some("Empress")
            {
                return None;
            }
            ("poet_variant", 2)
        }
    };
    if info.len() != fixed_fields + 1
        || info.get(variant_field).and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }
    let targets = poet_targets(info, state.n_cards, 3, 3)?;
    if targets.windows(2).any(|pair| pair[0] >= pair[1]) {
        return None;
    }
    if current_empress_claim_text(&targets).as_deref() != Some(card.info_text.as_str()) {
        return None;
    }
    Some(targets)
}

fn current_has_unresolved_start_identity(
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    (1..=state.n_cards).any(|position| {
        let public_origin_was_unresolved = state.executed.contains(&position)
            && state.confirmed_evil.contains(&position)
            && state
                .executed_evil_roles
                .get(&position)
                .is_none_or(|public_role| normalize_role(public_role) == "unknown");
        if public_origin_was_unresolved && scenario.puppet_position == Some(position) {
            // Generated Puppet has no stable Evil origin role by definition,
            // but its erased Villager identity is missing unless the exact
            // ordered Puppeteer trace retained it.
            return scenario.puppeteer_trace.is_none();
        }
        let Some(role) = stable_evil_origin_role_at(position, scenario, state) else {
            return false;
        };
        let role = normalize_role(role);
        if role == "unknown" {
            return true;
        }

        public_origin_was_unresolved
            && match role.as_str() {
                // The ordered Twin slice proves completeness with an explicit
                // trace. Legacy fallback must not become trusted merely because
                // construction inferred the actor's stable role.
                "twinminion" => scenario.twin_trace.is_none(),
                // These writers still have deliberately partial general Start
                // models (erased Puppet identity and copied-Start side effects).
                "puppeteer" => scenario.puppeteer_trace.is_none(),
                "shaman" => true,
                _ => false,
            }
    })
}

fn current_empress_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentEmpressSupport> {
    let Some(targets) = parse_current_empress_targets(card, source, state) else {
        return Vec::new();
    };
    // An untyped executed Evil could have been Spy, Chancellor, Twin, Shaman,
    // or Puppeteer and therefore could have changed the Start state observed
    // by Empress. Scenario.evil_positions is the authority for the replay that
    // built this world; a later state-map role overlay cannot retroactively
    // repair an `Unknown` Start history.
    if current_has_unresolved_start_identity(scenario, state) {
        return Vec::new();
    }

    let truth = truth_status(card.position, scenario, state);
    let anonymous_wretches: HashSet<u8> = anonymous_natural_wretch_candidates(scenario, state)
        .into_iter()
        .collect();
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(card.position, state) {
            continue;
        }

        let mut known_evil = HashSet::new();
        let mut optional_wretches = Vec::new();
        let mut complete = true;
        for &target in &targets {
            let Some(alignment) = registered_alignment_at_observation(
                target,
                card.position,
                &timeline,
                scenario,
                state,
            ) else {
                complete = false;
                break;
            };
            if alignment == EffectiveAlignment::Evil {
                known_evil.insert(target);
            } else if anonymous_wretches.contains(&target) {
                optional_wretches.push(target);
            }
        }
        if !complete {
            continue;
        }

        for mask in 0..(1usize << optional_wretches.len()) {
            let mut required = HashSet::new();
            let mut forbidden = HashSet::new();
            for (index, &position) in optional_wretches.iter().enumerate() {
                if mask & (1usize << index) == 0 {
                    forbidden.insert(position);
                } else {
                    required.insert(position);
                }
            }
            let evil_count = known_evil.len() + required.len();
            let actor_is_selected_evil = known_evil.contains(&card.position)
                || required.contains(&card.position);
            let native_selection = match truth {
                // Truth chooses two registered-Good characters after removing
                // charRef exactly, then independently one registered Evil.
                TruthStatus::Truthful => {
                    evil_count == 1
                        && (!targets.contains(&card.position) || actor_is_selected_evil)
                }
                // Bluff chooses three registered-Good characters from that
                // same actor-excluded pool.
                TruthStatus::Lying => {
                    evil_count == 0 && !targets.contains(&card.position)
                }
            };
            if !native_selection
                || !anonymous_wretch_assignment_possible(
                    &required,
                    &forbidden,
                    scenario,
                    state,
                )
            {
                continue;
            }
            let support = CurrentEmpressSupport {
                anonymous_wretches: AnonymousWretchConstraints {
                    required,
                    forbidden,
                },
                baker_spy_timeline: timeline.clone(),
            };
            if !supports.contains(&support) {
                supports.push(support);
            }
        }
    }
    supports
}

fn validate_current_empress(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_empress_supports(card, scenario, state, source).is_empty()
}

fn validate_empress(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    match current_passive_payload_source(card, EMPRESS_CURRENT_VARIANT_FIELD, "Empress") {
        Ok(Some(source)) => return validate_current_empress(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

    // Preserve unmarked archived observations on the legacy scalar predicate.
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
    match current_passive_payload_source(card, JESTER_CURRENT_VARIANT_FIELD, "Jester") {
        Ok(Some(CurrentPassivePayloadSource::Direct)) => {
            // A marker-only card is the exact current shell before its first
            // successful picker completion. Once callback evidence exists,
            // require the authenticated append-only ledger; marked scalar or
            // interrupted compatibility payloads cannot safely reconstruct
            // real/raw chronology.
            if card.info_parsed.len() == 1 {
                return card.info_text.is_empty();
            }
            return validate_current_jester(card, scenario, state);
        }
        // Jester is absent from current Poet's closed provider list.
        Ok(Some(CurrentPassivePayloadSource::Poet)) | Err(()) => return false,
        Ok(None) => {}
    }

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
    let evil_roles: Vec<String> = (1..=state.n_cards)
        .filter_map(|p| {
            let role = legacy_current_evil_role_at(p, scenario, state)?;
            let normalized = normalize_role(&role);
            if matches!(normalized.as_str(), "puppet" | "unknown") {
                None
            } else {
                Some(role)
            }
        })
        .collect();

    if truth == TruthStatus::Truthful {
        if target_is_wretch {
            return claimed_role.eq_ignore_ascii_case("cabbage");
        }
        if target_is_evil {
            let actual = legacy_current_evil_role_at(target, scenario, state)
                .unwrap_or_default();
            return roles_equal(claimed_role, &actual);
        }
        evil_roles.iter().any(|r| roles_equal(r, claimed_role))
    } else {
        if target_is_wretch {
            return !claimed_role.eq_ignore_ascii_case("cabbage");
        }
        if target_is_evil {
            let actual = legacy_current_evil_role_at(target, scenario, state)
                .unwrap_or_default();
            return !roles_equal(claimed_role, &actual);
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
    match current_passive_payload_source(card, DRUID_CURRENT_VARIANT_FIELD, "Druid") {
        Ok(Some(CurrentPassivePayloadSource::Direct)) => {
            return validate_current_druid(card, scenario, state);
        }
        // Druid is not one of the current Poet providers. Archived unmarked
        // Poet captures still delegate to the legacy predicate below.
        Ok(Some(CurrentPassivePayloadSource::Poet)) | Err(()) => return false,
        Ok(None) => {}
    }

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

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentDruidClaim {
    targets: [u8; 3],
    found_outcast: Option<String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CurrentDruidPayload {
    /// Compatibility for the first marker-gated bridge, which serialized only
    /// the newest result before Druid's ResetAfterNight metadata was audited.
    Scalar(CurrentDruidClaim),
    Ledger(Vec<CurrentDruidCallbackEvent>),
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentDruidDispatchPath {
    Either,
    Real,
    Raw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentDruidActivationEvidence {
    SingleCallbackSuffix,
    AutoUseClick,
    SessionResetGeneration,
    SameActivationExtension,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CurrentDruidCallbackKind {
    Result(CurrentDruidClaim),
    RamblerInterruption { target: u8 },
    OpaqueReal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentDruidCallbackEvent {
    activation_id: usize,
    callback_index: usize,
    dispatch_path: CurrentDruidDispatchPath,
    kind: CurrentDruidCallbackKind,
    settled_reveal_count: usize,
    reset_generation: usize,
    activation_evidence: CurrentDruidActivationEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentDruidObservationBoundary {
    FinalCompatibility,
    SettledRevealCount(usize),
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CurrentDruidTargetSurface {
    KnownOutcast(String),
    KnownNonOutcast,
    AnonymousGood,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentDruidClueSupport {
    anonymous_type_options: HashMap<u8, u8>,
    anonymous_outcast_roles: HashMap<u8, String>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentDruidSupport {
    anonymous_wretches: AnonymousWretchConstraints,
    anonymous_type_options: HashMap<u8, u8>,
    anonymous_outcast_roles: HashMap<u8, String>,
    register_as: Option<(u8, String)>,
    raw_bluff: Option<(u8, String)>,
    forbidden_raw_bluff: Option<(u8, String)>,
    baker_spy_timeline: BakerSpyTimeline,
    callbacks: Vec<CurrentDruidResolvedCallback>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentDruidResolvedPath {
    Real,
    Raw,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CurrentDruidResolvedCallback {
    activation_id: usize,
    callback_index: usize,
    path: CurrentDruidResolvedPath,
    opaque: bool,
}

fn current_druid_payload_role_token(display_name: &str) -> String {
    display_name.replace(' ', "_")
}

fn current_druid_claim_text(targets: &[u8; 3], found_outcast: Option<&str>) -> String {
    let mut sorted = *targets;
    sorted.sort_unstable();
    match found_outcast {
        Some(role) => {
            let display_name = get_card(role).map_or(role, |card| card.name);
            format!(
                "Among #{}, #{}, #{}\nthere is: {display_name}",
                sorted[0], sorted[1], sorted[2]
            )
        }
        None => format!(
            "Among #{}, #{}, #{}\nthere are NO Outcasts",
            sorted[0], sorted[1], sorted[2]
        ),
    }
}

fn parse_current_druid_claim_fields(
    info: &serde_json::Map<String, serde_json::Value>,
    state: &GameState,
) -> Option<CurrentDruidClaim> {
    let targets = info
        .get("targets")?
        .as_array()?
        .iter()
        .map(|value| u8::try_from(value.as_u64()?).ok())
        .collect::<Option<Vec<_>>>()?;
    let targets: [u8; 3] = targets.try_into().ok()?;
    if targets
        .iter()
        .any(|target| *target == 0 || *target > state.n_cards)
        || targets[0] == targets[1]
        || targets[0] == targets[2]
        || targets[1] == targets[2]
    {
        return None;
    }

    let found_outcast = match info.get("found_outcast")? {
        serde_json::Value::Null => None,
        serde_json::Value::String(role) => {
            let role_data = get_card(role)?;
            if role_data.faction != Faction::Outcast
                || current_druid_payload_role_token(role_data.name) != *role
            {
                return None;
            }
            Some(role_data.name.to_string())
        }
        _ => return None,
    };
    Some(CurrentDruidClaim {
        targets,
        found_outcast,
    })
}

fn parse_current_druid_claim(card: &CardInfo, state: &GameState) -> Option<CurrentDruidClaim> {
    if card.position == 0
        || card.position > state.n_cards
        || card.apparent_role != "Druid"
        || card.info_parsed.len() != 3
        || card
            .info_parsed
            .get(DRUID_CURRENT_VARIANT_FIELD)
            .and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }
    let claim = parse_current_druid_claim_fields(&card.info_parsed, state)?;
    (card.info_text == current_druid_claim_text(&claim.targets, claim.found_outcast.as_deref()))
        .then_some(claim)
}

fn current_druid_has_exact_fields(
    info: &serde_json::Map<String, serde_json::Value>,
    fields: &[&str],
) -> bool {
    info.len() == fields.len() && fields.iter().all(|field| info.contains_key(*field))
}

fn parse_current_druid_references(
    value: &serde_json::Value,
    state: &GameState,
) -> Option<Option<Vec<u8>>> {
    if value.is_null() {
        return Some(None);
    }
    value
        .as_array()?
        .iter()
        .map(|value| {
            value
                .as_u64()
                .and_then(|position| u8::try_from(position).ok())
                .filter(|position| *position > 0 && *position <= state.n_cards)
        })
        .collect::<Option<Vec<_>>>()
        .map(Some)
}

fn parse_current_druid_dispatch_path(value: &serde_json::Value) -> Option<CurrentDruidDispatchPath> {
    match value.as_str()? {
        "either" => Some(CurrentDruidDispatchPath::Either),
        "real" => Some(CurrentDruidDispatchPath::Real),
        "raw" => Some(CurrentDruidDispatchPath::Raw),
        _ => None,
    }
}

fn parse_current_druid_activation_evidence(
    value: &serde_json::Value,
) -> Option<CurrentDruidActivationEvidence> {
    match value.as_str()? {
        "single_callback_suffix" => Some(CurrentDruidActivationEvidence::SingleCallbackSuffix),
        "auto_use_click" => Some(CurrentDruidActivationEvidence::AutoUseClick),
        "session_reset_generation" => {
            Some(CurrentDruidActivationEvidence::SessionResetGeneration)
        }
        "same_activation_extension" => {
            Some(CurrentDruidActivationEvidence::SameActivationExtension)
        }
        _ => None,
    }
}

fn current_druid_text_contains_word(text: &str, word: &str) -> bool {
    text.match_indices(word).any(|(index, _)| {
        let before = text[..index].chars().next_back();
        let after = text[index + word.len()..].chars().next();
        before.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
            && after.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
    })
}

fn current_druid_is_python_regex_whitespace(character: char) -> bool {
    // Python's Unicode `\s` follows str.isspace(), which additionally treats
    // these four C0 information separators as whitespace. Rust's Unicode
    // White_Space predicate deliberately omits them.
    character.is_whitespace() || matches!(character, '\u{001C}'..='\u{001F}')
}

fn current_druid_trim_python_regex_whitespace(text: &str) -> &str {
    text.trim_start_matches(current_druid_is_python_regex_whitespace)
}

fn current_druid_has_result_clause(text: &str) -> bool {
    let mut search_from = 0;
    while let Some(relative_index) = text[search_from..].find("there") {
        let index = search_from + relative_index;
        let before = text[..index].chars().next_back();
        let after_there = &text[index + "there".len()..];
        let there_is_a_word = before
            .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
            && after_there
                .chars()
                .next()
                .is_some_and(current_druid_is_python_regex_whitespace);
        if there_is_a_word {
            let after_there = current_druid_trim_python_regex_whitespace(after_there);
            for verb in ["is", "was"] {
                if let Some(after_verb) = after_there.strip_prefix(verb) {
                    let boundary = after_verb
                        .chars()
                        .next()
                        .is_none_or(|character| {
                            !character.is_ascii_alphanumeric() && character != '_'
                        });
                    if boundary
                        && current_druid_trim_python_regex_whitespace(after_verb)
                            .starts_with(':')
                    {
                        return true;
                    }
                }
            }
            for verb in ["are", "were"] {
                if let Some(after_verb) = after_there.strip_prefix(verb) {
                    let boundary = after_verb
                        .chars()
                        .next()
                        .is_none_or(|character| {
                            !character.is_ascii_alphanumeric() && character != '_'
                        });
                    if boundary
                        && (current_druid_text_contains_word(after_verb, "outcast")
                            || current_druid_text_contains_word(after_verb, "outcasts"))
                    {
                        return true;
                    }
                }
            }
        }
        search_from = index + "there".len();
    }
    false
}

fn current_druid_is_python_decimal_digit(character: char) -> bool {
    // Python 3.13's Unicode `\d` is General_Category=Decimal_Number (Nd),
    // not Rust's broader `char::is_numeric`. These are the Unicode 15.1 Nd
    // blocks used by the bridge; every listed block has ten code points.
    const TEN_DIGIT_BLOCK_STARTS: &[u32] = &[
        0x30, 0x660, 0x6F0, 0x7C0, 0x966, 0x9E6, 0xA66, 0xAE6, 0xB66, 0xBE6,
        0xC66, 0xCE6, 0xD66, 0xDE6, 0xE50, 0xED0, 0xF20, 0x1040, 0x1090,
        0x17E0, 0x1810, 0x1946, 0x19D0, 0x1A80, 0x1A90, 0x1B50, 0x1BB0,
        0x1C40, 0x1C50, 0xA620, 0xA8D0, 0xA900, 0xA9D0, 0xA9F0, 0xAA50,
        0xABF0, 0xFF10, 0x104A0, 0x10D30, 0x11066, 0x110F0, 0x11136,
        0x111D0, 0x112F0, 0x11450, 0x114D0, 0x11650, 0x116C0, 0x11730,
        0x118E0, 0x11950, 0x11C50, 0x11D50, 0x11DA0, 0x11F50, 0x16A60,
        0x16AC0, 0x16B50, 0x1E140, 0x1E2F0, 0x1E4F0, 0x1E950, 0x1FBF0,
    ];
    let code_point = u32::from(character);
    (0x1D7CE..=0x1D7FF).contains(&code_point)
        || TEN_DIGIT_BLOCK_STARTS
            .iter()
            .any(|start| (*start..=*start + 9).contains(&code_point))
}

fn current_druid_python_ignorecase_fold(text: &str) -> String {
    // Python's Unicode IGNORECASE adds these four code points to ASCII
    // letter matching. The bridge's Druid-family regexes are ASCII literals,
    // so applying these folds plus ASCII lowercase reproduces their relevant
    // case-insensitive surface without broad compatibility normalization.
    text.chars()
        .map(|character| match character {
            '\u{0130}' | '\u{0131}' => 'i',
            '\u{017F}' => 's',
            '\u{212A}' => 'k',
            _ => character.to_ascii_lowercase(),
        })
        .collect()
}

fn current_druid_opaque_text_is_ambiguous(text: &str) -> bool {
    let normalized = current_druid_python_ignorecase_fold(
        current_druid_trim_python_regex_whitespace(text),
    );
    let druid_prefix = normalized
        .strip_prefix("among")
        .is_some_and(|rest| {
            rest.chars()
                .next()
                .is_some_and(current_druid_is_python_regex_whitespace)
                && current_druid_trim_python_regex_whitespace(rest).starts_with('#')
        });
    let displayed_ids = normalized
        .split('#')
        .skip(1)
        .filter(|suffix| {
            current_druid_trim_python_regex_whitespace(suffix)
                .chars()
                .next()
                .is_some_and(current_druid_is_python_decimal_digit)
        })
        .count();
    let druid_family =
        druid_prefix && displayed_ids == 3 && current_druid_has_result_clause(&normalized);
    let after_hash = normalized
        .strip_prefix('#')
        .map(current_druid_trim_python_regex_whitespace)
        .filter(|suffix| {
            suffix
                .chars()
                .next()
                .is_some_and(current_druid_is_python_decimal_digit)
        });
    // Match the bridge's non-DOTALL guard: a later-line foreign callback that
    // happens to contain "shut" is not a malformed one-line shut-up result.
    let shut_up_family = after_hash.is_some_and(|suffix| {
        current_druid_text_contains_word(suffix.split('\n').next().unwrap_or(suffix), "shut")
    });
    druid_family || shut_up_family
}

fn parse_current_druid_callback_event(
    value: &serde_json::Value,
    state: &GameState,
) -> Option<CurrentDruidCallbackEvent> {
    const COMMON_FIELDS: &[&str] = &[
        "activation_id",
        "callback_index",
        "dispatch_path",
        "event_kind",
        "text",
        "references",
        "settled_reveal_count",
        "reset_generation",
        "activation_evidence",
    ];
    const RESULT_FIELDS: &[&str] = &[
        "activation_id",
        "callback_index",
        "dispatch_path",
        "event_kind",
        "text",
        "references",
        "settled_reveal_count",
        "reset_generation",
        "activation_evidence",
        "targets",
        "found_outcast",
    ];
    const INTERRUPTION_FIELDS: &[&str] = &[
        "activation_id",
        "callback_index",
        "dispatch_path",
        "event_kind",
        "text",
        "references",
        "settled_reveal_count",
        "reset_generation",
        "activation_evidence",
        "shut_up_target",
    ];

    let event = value.as_object()?;
    let activation_id = event
        .get("activation_id")?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)?;
    let callback_index = event
        .get("callback_index")?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())?;
    let dispatch_path = parse_current_druid_dispatch_path(event.get("dispatch_path")?)?;
    let settled_reveal_count = event
        .get("settled_reveal_count")?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)?;
    let reset_generation = event
        .get("reset_generation")?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())?;
    let activation_evidence =
        parse_current_druid_activation_evidence(event.get("activation_evidence")?)?;
    let references = parse_current_druid_references(event.get("references")?, state)?;
    let text = event.get("text")?.as_str()?;

    let kind = match event.get("event_kind")?.as_str()? {
        "druid_result" => {
            if !current_druid_has_exact_fields(event, RESULT_FIELDS) {
                return None;
            }
            let claim = parse_current_druid_claim_fields(event, state)?;
            if references.as_deref() != Some(claim.targets.as_slice())
                || text
                    != current_druid_claim_text(
                        &claim.targets,
                        claim.found_outcast.as_deref(),
                    )
            {
                return None;
            }
            CurrentDruidCallbackKind::Result(claim)
        }
        "rambler_interruption" => {
            if !current_druid_has_exact_fields(event, INTERRUPTION_FIELDS) {
                return None;
            }
            let target = event
                .get("shut_up_target")?
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .filter(|value| *value > 0 && *value <= state.n_cards)?;
            if references.as_deref() != Some([target].as_slice())
                || text != format!("#{target}\nshut up!")
            {
                return None;
            }
            CurrentDruidCallbackKind::RamblerInterruption { target }
        }
        "opaque_real" => {
            if !current_druid_has_exact_fields(event, COMMON_FIELDS)
                || text.is_empty()
                || current_druid_opaque_text_is_ambiguous(text)
            {
                return None;
            }
            CurrentDruidCallbackKind::OpaqueReal
        }
        _ => return None,
    };

    Some(CurrentDruidCallbackEvent {
        activation_id,
        callback_index,
        dispatch_path,
        kind,
        settled_reveal_count,
        reset_generation,
        activation_evidence,
    })
}

fn parse_current_druid_payload(card: &CardInfo, state: &GameState) -> Option<CurrentDruidPayload> {
    if !card.info_parsed.contains_key("callback_ledger_variant")
        && !card.info_parsed.contains_key("callback_events")
    {
        return parse_current_druid_claim(card, state).map(CurrentDruidPayload::Scalar);
    }
    if card.position == 0
        || card.position > state.n_cards
        || card.apparent_role != "Druid"
        || card
            .info_parsed
            .get(DRUID_CURRENT_VARIANT_FIELD)
            .and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
        || card
            .info_parsed
            .get("callback_ledger_variant")
            .and_then(serde_json::Value::as_str)
            != Some("ordered_callbacks_v1")
        || state.baker_rule_version.as_deref() != Some(BAKER_CURRENT_RULE)
    {
        return None;
    }
    let values = card.info_parsed.get("callback_events")?.as_array()?;
    if values.is_empty() {
        return None;
    }
    let mut seen_reveals = HashSet::new();
    if state.reveal_order.len() > usize::from(state.n_cards)
        || state.reveal_order.iter().any(|position| {
            *position == 0 || *position > state.n_cards || !seen_reveals.insert(*position)
        })
    {
        return None;
    }

    let events = values
        .iter()
        .map(|value| parse_current_druid_callback_event(value, state))
        .collect::<Option<Vec<_>>>()?;
    let mut groups: Vec<&[CurrentDruidCallbackEvent]> = Vec::new();
    let mut start = 0;
    let mut previous_boundary = 0;
    let mut previous_generation = 0;
    while start < events.len() {
        let activation_id = events[start].activation_id;
        if activation_id != groups.len() + 1 || events[start].callback_index != 0 {
            return None;
        }
        let mut end = start + 1;
        while end < events.len() && events[end].activation_id == activation_id {
            end += 1;
        }
        let group = &events[start..end];
        let first = &group[0];
        if first.settled_reveal_count < previous_boundary
            || first.settled_reveal_count > state.reveal_order.len()
            || !state.reveal_order[..first.settled_reveal_count].contains(&card.position)
            || !groups.is_empty() && first.reset_generation <= previous_generation
            || group.iter().enumerate().any(|(index, event)| {
                event.callback_index != index
                    || event.settled_reveal_count != first.settled_reveal_count
                    || event.reset_generation != first.reset_generation
                    || event.activation_evidence != first.activation_evidence
            })
        {
            return None;
        }
        match group {
            [only]
                if only.dispatch_path == CurrentDruidDispatchPath::Either
                    && only.activation_evidence
                        != CurrentDruidActivationEvidence::SameActivationExtension => {}
            [real, raw]
                if real.dispatch_path == CurrentDruidDispatchPath::Real
                    && raw.dispatch_path == CurrentDruidDispatchPath::Raw
                    && real.activation_evidence
                        != CurrentDruidActivationEvidence::SingleCallbackSuffix => {}
            _ => return None,
        }
        let evidence_is_reachable = match first.activation_evidence {
            CurrentDruidActivationEvidence::SingleCallbackSuffix => {
                groups.is_empty() && first.reset_generation == 0 && group.len() == 1
            }
            CurrentDruidActivationEvidence::AutoUseClick => true,
            CurrentDruidActivationEvidence::SessionResetGeneration => {
                if groups.is_empty() {
                    first.reset_generation > 0 && group.len() == 1
                } else {
                    group.len() == 1
                        || first.reset_generation.checked_sub(previous_generation) == Some(1)
                }
            }
            CurrentDruidActivationEvidence::SameActivationExtension => {
                group.len() == 2
                    && !matches!(group[0].kind, CurrentDruidCallbackKind::OpaqueReal)
            }
        };
        if first.activation_evidence == CurrentDruidActivationEvidence::SingleCallbackSuffix
            && group.len() != 1
            || first.activation_evidence
                == CurrentDruidActivationEvidence::SameActivationExtension
                && group.len() != 2
            || !evidence_is_reachable
            || matches!(group[0].kind, CurrentDruidCallbackKind::OpaqueReal)
                && (group.len() != 2
                    || !matches!(group[1].kind, CurrentDruidCallbackKind::Result(_) | CurrentDruidCallbackKind::RamblerInterruption { .. }))
            || group
                .iter()
                .skip(1)
                .any(|event| matches!(event.kind, CurrentDruidCallbackKind::OpaqueReal))
            || group.len() == 2
                && matches!(group[0].kind, CurrentDruidCallbackKind::RamblerInterruption { .. })
                    != matches!(group[1].kind, CurrentDruidCallbackKind::RamblerInterruption { .. })
            || matches!(
                (&group[0].kind, group.get(1).map(|event| &event.kind)),
                (
                    CurrentDruidCallbackKind::RamblerInterruption { target: left },
                    Some(CurrentDruidCallbackKind::RamblerInterruption { target: right }),
                ) if left != right
            )
            || matches!(
                (&group[0].kind, group.get(1).map(|event| &event.kind)),
                (
                    CurrentDruidCallbackKind::Result(left),
                    Some(CurrentDruidCallbackKind::Result(right)),
                ) if left.targets != right.targets
            )
        {
            return None;
        }
        previous_boundary = first.settled_reveal_count;
        previous_generation = first.reset_generation;
        groups.push(group);
        start = end;
    }

    let latest = events.last()?;
    const COMMON_TOP_FIELDS: &[&str] = &[
        DRUID_CURRENT_VARIANT_FIELD,
        "callback_ledger_variant",
        "callback_events",
    ];
    match &latest.kind {
        CurrentDruidCallbackKind::Result(claim) => {
            const RESULT_TOP_FIELDS: &[&str] = &[
                DRUID_CURRENT_VARIANT_FIELD,
                "callback_ledger_variant",
                "callback_events",
                "targets",
                "found_outcast",
            ];
            if !current_druid_has_exact_fields(&card.info_parsed, RESULT_TOP_FIELDS)
                || parse_current_druid_claim_fields(&card.info_parsed, state)? != *claim
                || card.info_text
                    != current_druid_claim_text(
                        &claim.targets,
                        claim.found_outcast.as_deref(),
                    )
            {
                return None;
            }
        }
        CurrentDruidCallbackKind::RamblerInterruption { target } => {
            const INTERRUPTION_TOP_FIELDS: &[&str] = &[
                DRUID_CURRENT_VARIANT_FIELD,
                "callback_ledger_variant",
                "callback_events",
                "shut_up_target",
            ];
            if !current_druid_has_exact_fields(&card.info_parsed, INTERRUPTION_TOP_FIELDS)
                || card
                    .info_parsed
                    .get("shut_up_target")
                    .and_then(serde_json::Value::as_u64)
                    != Some(u64::from(*target))
                || card.info_text != format!("#{target}\nshut up!")
            {
                return None;
            }
        }
        CurrentDruidCallbackKind::OpaqueReal => {
            let _ = COMMON_TOP_FIELDS;
            return None;
        }
    }

    let ledger_interruptions = events
        .iter()
        .filter_map(|event| match event.kind {
            CurrentDruidCallbackKind::RamblerInterruption { target } => Some(target),
            _ => None,
        })
        .collect::<Vec<_>>();
    let public_interruptions = state
        .rambler_shut_up_observations
        .iter()
        .filter_map(|observation| {
            (observation.speaker_position == card.position).then_some(observation.shut_up_target)
        })
        .collect::<Vec<_>>();
    if ledger_interruptions != public_interruptions
        || !ledger_interruptions.is_empty()
            && state.rambler_rule_version.as_deref() != Some(RAMBLER_CURRENT_RULE)
    {
        return None;
    }

    Some(CurrentDruidPayload::Ledger(events))
}

fn current_druid_converted_at_boundary(
    position: u8,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    state: &GameState,
) -> Option<bool> {
    match boundary {
        CurrentDruidObservationBoundary::FinalCompatibility => {
            Some(timeline.contains_position(position))
        }
        CurrentDruidObservationBoundary::SettledRevealCount(count) => {
            timeline.converted_before_settled_reveal_count(position, count, state)
        }
    }
}

fn current_data_role_at_druid_observation(
    position: u8,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<String> {
    if current_druid_converted_at_boundary(position, boundary, timeline, state)? {
        Some("Baker".to_string())
    } else if timeline.contains_position(position) {
        Some("Spy".to_string())
    } else {
        current_data_role_at(position, scenario, state)
    }
}

fn current_spy_register_as_surface_at_druid_observation(
    position: u8,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<bool> {
    if timeline.contains_position(position) {
        Some(!current_druid_converted_at_boundary(
            position, boundary, timeline, state,
        )?)
    } else {
        Some(current_spy_register_as_surface_at(
            position, scenario, state,
        ))
    }
}

fn current_raw_bluff_holder_at_druid_observation(
    position: u8,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> CurrentMediumRawBluffHolder {
    if current_druid_converted_at_boundary(position, boundary, timeline, state) != Some(false) {
        CurrentMediumRawBluffHolder::Impossible
    } else {
        current_medium_raw_bluff_holder_at(position, position, timeline, scenario, state)
    }
}

fn current_druid_target_surface_at_observation(
    position: u8,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<CurrentDruidTargetSurface> {
    if current_spy_register_as_surface_at_druid_observation(
        position, boundary, timeline, scenario, state,
    )? {
        return Some(CurrentDruidTargetSurface::KnownNonOutcast);
    }

    let role =
        match current_data_role_at_druid_observation(position, boundary, timeline, scenario, state)
        {
            Some(role) if normalize_role(&role) == "unknown" => return None,
            Some(role) => role,
            None if scenario.is_evil(position) => return None,
            None => return Some(CurrentDruidTargetSurface::AnonymousGood),
        };
    if roles_equal(&role, "Wretch") {
        return Some(CurrentDruidTargetSurface::KnownNonOutcast);
    }

    let in_minion_pool = state
        .deck
        .minions
        .iter()
        .any(|candidate| roles_equal(candidate, &role));
    let in_demon_pool = state
        .deck
        .demons
        .iter()
        .any(|candidate| roles_equal(candidate, &role));
    let faction = match (in_minion_pool, in_demon_pool) {
        (true, false) => Faction::Minion,
        (false, true) => Faction::Demon,
        (true, true) => return None,
        (false, false) => get_card(&role)?.faction,
    };
    if faction == Faction::Outcast {
        Some(CurrentDruidTargetSurface::KnownOutcast(
            get_card(&role)?.name.to_string(),
        ))
    } else {
        Some(CurrentDruidTargetSurface::KnownNonOutcast)
    }
}

fn current_druid_role_is_non_bluffable_outcast(role: &str) -> bool {
    matches!(
        normalize_role(role).as_str(),
        "drunk" | "wretch" | "doppelganger"
    )
}

fn current_druid_false_role_support(state: &GameState) -> HashSet<String> {
    let canonical_outcasts = |roles: &[String]| {
        roles
            .iter()
            .filter_map(|role| {
                get_card(role).and_then(|card| {
                    (card.faction == Faction::Outcast).then(|| card.name.to_string())
                })
            })
            .collect::<HashSet<_>>()
    };

    // The first native ladder rung is the current script's exact
    // non-bluffable Outcast records. Preserve set membership here; duplicate
    // records affect probability, not whether an observed role is possible.
    let script_non_bluffable = canonical_outcasts(&state.deck.outcasts)
        .into_iter()
        .filter(|role| current_druid_role_is_non_bluffable_outcast(role))
        .collect::<HashSet<_>>();
    if !script_non_bluffable.is_empty() {
        return script_non_bluffable;
    }

    // Current shipped all-ascension non-bluffable Outcasts. This is the exact
    // public asset set represented by the solver fingerprint; retain the later
    // ladder rungs explicitly so a future asset-table change fails visibly.
    let all_ascension_non_bluffable = ["Drunk", "Wretch", "Doppelganger"]
        .into_iter()
        .filter(|role| {
            get_card(role).is_some_and(|card| card.faction == Faction::Outcast)
        })
        .map(str::to_string)
        .collect::<HashSet<_>>();
    if !all_ascension_non_bluffable.is_empty() {
        return all_ascension_non_bluffable;
    }

    let any_script_outcast = canonical_outcasts(&state.deck.outcasts);
    if !any_script_outcast.is_empty() {
        any_script_outcast
    } else {
        HashSet::from(["Drunk".to_string()])
    }
}

fn current_druid_clue_supports(
    claim: &CurrentDruidClaim,
    truth: TruthStatus,
    surfaces: &[CurrentDruidTargetSurface; 3],
    state: &GameState,
) -> Vec<CurrentDruidClueSupport> {
    let known_outcasts: Vec<&str> = surfaces
        .iter()
        .filter_map(|surface| match surface {
            CurrentDruidTargetSurface::KnownOutcast(role) => Some(role.as_str()),
            CurrentDruidTargetSurface::KnownNonOutcast
            | CurrentDruidTargetSurface::AnonymousGood => None,
        })
        .collect();
    let anonymous: Vec<u8> = claim
        .targets
        .iter()
        .copied()
        .zip(surfaces.iter())
        .filter_map(|(position, surface)| {
            matches!(surface, CurrentDruidTargetSurface::AnonymousGood)
                .then_some(position)
        })
        .collect();

    let empty = || CurrentDruidClueSupport {
        anonymous_type_options: HashMap::new(),
        anonymous_outcast_roles: HashMap::new(),
    };
    match (truth, claim.found_outcast.as_deref()) {
        (TruthStatus::Truthful, Some(claimed)) => {
            // Wretch's live register-as is Minion, so no physical Wretch
            // occurrence belongs to Druid's selected Outcast pool.
            if roles_equal(claimed, "Wretch") {
                return Vec::new();
            }
            let mut supports = Vec::new();
            if known_outcasts.iter().any(|actual| *actual == claimed) {
                supports.push(empty());
            }
            for position in anonymous {
                supports.push(CurrentDruidClueSupport {
                    anonymous_type_options: HashMap::from([(
                        position,
                        BishopType::Outcast.bit(),
                    )]),
                    anonymous_outcast_roles: HashMap::from([(
                        position,
                        claimed.to_string(),
                    )]),
                });
            }
            supports
        }
        (TruthStatus::Truthful, None) => {
            if !known_outcasts.is_empty() {
                return Vec::new();
            }
            vec![CurrentDruidClueSupport {
                anonymous_type_options: anonymous
                    .into_iter()
                    .map(|position| {
                        (
                            position,
                            BishopType::Villager.bit() | BishopType::Minion.bit(),
                        )
                    })
                    .collect(),
                anonymous_outcast_roles: HashMap::new(),
            }]
        }
        (TruthStatus::Lying, None) => {
            if !known_outcasts.is_empty() {
                return vec![empty()];
            }
            anonymous
                .into_iter()
                .map(|position| CurrentDruidClueSupport {
                    anonymous_type_options: HashMap::from([(
                        position,
                        BishopType::Outcast.bit(),
                    )]),
                    anonymous_outcast_roles: HashMap::new(),
                })
                .collect()
        }
        (TruthStatus::Lying, Some(claimed)) => {
            if !known_outcasts.is_empty()
                || !current_druid_false_role_support(state).contains(claimed)
            {
                return Vec::new();
            }
            vec![CurrentDruidClueSupport {
                anonymous_type_options: anonymous
                    .into_iter()
                    .map(|position| {
                        (
                            position,
                            BishopType::Villager.bit() | BishopType::Minion.bit(),
                        )
                    })
                    .collect(),
                anonymous_outcast_roles: HashMap::new(),
            }]
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn current_druid_append_supports(
    supports: &mut Vec<CurrentDruidSupport>,
    claim: &CurrentDruidClaim,
    truth: TruthStatus,
    surfaces: &[CurrentDruidTargetSurface; 3],
    anonymous_wretches: &AnonymousWretchConstraints,
    register_as: Option<(u8, String)>,
    raw_bluff: Option<(u8, String)>,
    forbidden_raw_bluff: Option<(u8, String)>,
    timeline: &BakerSpyTimeline,
    callbacks: &[CurrentDruidResolvedCallback],
    scenario: &Scenario,
    state: &GameState,
) {
    for clue in current_druid_clue_supports(claim, truth, surfaces, state) {
        if !current_hidden_anonymous_assignment_possible(
            &clue.anonymous_type_options,
            &clue.anonymous_outcast_roles,
            &anonymous_wretches.required,
            &anonymous_wretches.forbidden,
            scenario,
            state,
        ) {
            continue;
        }
        let support = CurrentDruidSupport {
            anonymous_wretches: anonymous_wretches.clone(),
            anonymous_type_options: clue.anonymous_type_options,
            anonymous_outcast_roles: clue.anonymous_outcast_roles,
            register_as: register_as.clone(),
            raw_bluff: raw_bluff.clone(),
            forbidden_raw_bluff: forbidden_raw_bluff.clone(),
            baker_spy_timeline: timeline.clone(),
            callbacks: callbacks.to_vec(),
        };
        if !supports.contains(&support) {
            supports.push(support);
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentDruidProviderSelection {
    ScalarCompatibility,
    Real {
        callback: CurrentDruidResolvedCallback,
        raw_druid: CurrentDruidRawConstraint,
    },
    Raw {
        callback: CurrentDruidResolvedCallback,
        allow_real_druid: bool,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentDruidRawConstraint {
    Unconstrained,
    Required,
    Forbidden,
}

fn current_druid_result_supports(
    card: &CardInfo,
    claim: &CurrentDruidClaim,
    boundary: CurrentDruidObservationBoundary,
    selection: CurrentDruidProviderSelection,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentDruidSupport> {
    let anonymous_wretch_candidates: HashSet<u8> =
        anonymous_natural_wretch_candidates(scenario, state)
            .into_iter()
            .collect();
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if matches!(boundary, CurrentDruidObservationBoundary::SettledRevealCount(count) if !timeline.supports_settled_reveal_count(count, state))
        {
            continue;
        }
        let surfaces = claim
            .targets
            .iter()
            .map(|position| {
                current_druid_target_surface_at_observation(
                    *position, boundary, &timeline, scenario, state,
                )
            })
            .collect::<Option<Vec<_>>>();
        let Some(surfaces) = surfaces.and_then(|values| values.try_into().ok()) else {
            continue;
        };
        let current_role = current_data_role_at_druid_observation(
            card.position,
            boundary,
            &timeline,
            scenario,
            state,
        );
        let raw_holder = current_raw_bluff_holder_at_druid_observation(
            card.position,
            boundary,
            &timeline,
            scenario,
            state,
        );
        let real_druid_callback = current_role
            .as_deref()
            .is_some_and(|role| roles_equal(role, "Druid"));

        let runtime_evil_real_truth = is_runtime_evil_at(card.position, scenario, state)
            && raw_holder != CurrentMediumRawBluffHolder::Impossible;
        if !matches!(selection, CurrentDruidProviderSelection::Raw { .. })
            && real_druid_callback
        {
            let truth = if runtime_evil_real_truth {
                TruthStatus::Truthful
            } else {
                truth_status(card.position, scenario, state)
            };
            let (
                anonymous_wretches,
                register_as,
                raw_bluff,
                forbidden_raw_bluff,
                callbacks,
            ) = match selection {
                CurrentDruidProviderSelection::ScalarCompatibility => (
                    AnonymousWretchConstraints::empty(),
                    None,
                    None,
                    runtime_evil_real_truth
                        .then(|| (card.position, normalize_role("Druid"))),
                    Vec::new(),
                ),
                CurrentDruidProviderSelection::Real {
                    callback,
                    raw_druid,
                } => {
                    let mut anonymous_wretches = AnonymousWretchConstraints::empty();
                    let mut register_as = None;
                    let mut raw_bluff = None;
                    let mut forbidden_raw_bluff = None;
                    match raw_druid {
                        CurrentDruidRawConstraint::Unconstrained => {}
                        CurrentDruidRawConstraint::Forbidden => {
                            forbidden_raw_bluff =
                                Some((card.position, normalize_role("Druid")));
                        }
                        CurrentDruidRawConstraint::Required => {
                            if raw_holder == CurrentMediumRawBluffHolder::Impossible {
                                continue;
                            }
                            if anonymous_wretch_candidates.contains(&card.position) {
                                anonymous_wretches.forbidden.insert(card.position);
                            }
                            if current_spy_register_as_surface_at_druid_observation(
                                card.position,
                                boundary,
                                &timeline,
                                scenario,
                                state,
                            ) == Some(true)
                            {
                                if !current_medium_spy_register_as_label_allowed("Druid", state) {
                                    continue;
                                }
                                register_as =
                                    Some((card.position, normalize_role("Druid")));
                            }
                            raw_bluff = Some((card.position, normalize_role("Druid")));
                        }
                    }
                    (
                        anonymous_wretches,
                        register_as,
                        raw_bluff,
                        forbidden_raw_bluff,
                        vec![callback],
                    )
                }
                CurrentDruidProviderSelection::Raw { .. } => unreachable!(),
            };
            current_druid_append_supports(
                &mut supports,
                &claim,
                truth,
                &surfaces,
                &anonymous_wretches,
                register_as,
                raw_bluff,
                forbidden_raw_bluff,
                &timeline,
                &callbacks,
                scenario,
                state,
            );
        }

        if matches!(selection, CurrentDruidProviderSelection::Real { .. })
            || matches!(selection, CurrentDruidProviderSelection::Raw { allow_real_druid: false, .. })
                && !current_role
                    .as_deref()
                    .is_some_and(current_druid_role_has_no_day_callback)
            || raw_holder == CurrentMediumRawBluffHolder::Impossible
        {
            continue;
        }
        let mut anonymous_wretches = AnonymousWretchConstraints::empty();
        if anonymous_wretch_candidates.contains(&card.position) {
            // A natural Wretch has a base-null raw selector. A current Druid
            // callback at that grouped seat therefore excludes Wretch.
            anonymous_wretches.forbidden.insert(card.position);
        }
        let register_as = if current_spy_register_as_surface_at_druid_observation(
            card.position,
            boundary,
            &timeline,
            scenario,
            state,
        ) == Some(true)
        {
            if !current_medium_spy_register_as_label_allowed("Druid", state) {
                continue;
            }
            Some((card.position, normalize_role("Druid")))
        } else {
            None
        };
        let callbacks = match selection {
            CurrentDruidProviderSelection::ScalarCompatibility => Vec::new(),
            CurrentDruidProviderSelection::Raw { callback, .. } => vec![callback],
            CurrentDruidProviderSelection::Real { .. } => unreachable!(),
        };
        current_druid_append_supports(
            &mut supports,
            &claim,
            current_bard_raw_provider_truth(
                card.position,
                card.position,
                &timeline,
                scenario,
                state,
            ),
            &surfaces,
            &anonymous_wretches,
            register_as,
            Some((card.position, normalize_role("Druid"))),
            None,
            &timeline,
            &callbacks,
            scenario,
            state,
        );
    }
    supports
}

fn current_druid_callback_witness(
    event: &CurrentDruidCallbackEvent,
    path: CurrentDruidResolvedPath,
) -> CurrentDruidResolvedCallback {
    CurrentDruidResolvedCallback {
        activation_id: event.activation_id,
        callback_index: event.callback_index,
        path,
        opaque: matches!(event.kind, CurrentDruidCallbackKind::OpaqueReal),
    }
}

fn current_druid_role_can_emit_day_callback(role: &str) -> bool {
    // A real callback record needs a concrete Day-output producer. Start-,
    // Night-, death-, and status-only roles cannot manufacture the first
    // record in a real-then-raw Druid activation. Keep this current-build
    // boundary closed instead of treating every non-Druid current role as an
    // opaque producer.
    matches!(
        normalize_role(role).as_str(),
        "alchemist"
            | "architect"
            | "baker"
            | "bard"
            | "bishop"
            | "bountyhunter"
            | "confessor"
            | "dreamer"
            | "druid"
            | "empress"
            | "enlightened"
            | "fortuneteller"
            | "gemcrafter"
            | "hunter"
            | "jester"
            | "judge"
            | "knitter"
            | "lover"
            | "medium"
            | "oracle"
            | "plaguedoctor"
            | "poet"
            | "rambler"
            | "scout"
            | "slayer"
            | "witness"
    )
}

fn current_druid_role_has_no_day_callback(role: &str) -> bool {
    // This is intentionally a closed companion set. An unknown role proves
    // neither that a real callback exists nor that it is absent.
    matches!(
        normalize_role(role).as_str(),
        "baa"
            | "bombardier"
            | "chancellor"
            | "demon"
            | "doppelganger"
            | "drunk"
            | "knight"
            | "lilis"
            | "minion"
            | "mutant"
            | "poisoner"
            | "pooka"
            | "puppet"
            | "puppeteer"
            | "saint"
            | "saintvillager"
            | "shaman"
            | "spy"
            | "twinminion"
            | "villager"
            | "witch"
            | "wretch"
    )
}

fn current_druid_non_result_supports(
    card: &CardInfo,
    event: &CurrentDruidCallbackEvent,
    path: CurrentDruidResolvedPath,
    raw_druid: CurrentDruidRawConstraint,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentDruidSupport> {
    let anonymous_wretch_candidates: HashSet<u8> =
        anonymous_natural_wretch_candidates(scenario, state)
            .into_iter()
            .collect();
    let boundary = CurrentDruidObservationBoundary::SettledRevealCount(
        event.settled_reveal_count,
    );
    let callback = current_druid_callback_witness(event, path);
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_settled_reveal_count(event.settled_reveal_count, state) {
            continue;
        }
        let current_role = current_data_role_at_druid_observation(
            card.position,
            boundary,
            &timeline,
            scenario,
            state,
        );
        if path == CurrentDruidResolvedPath::Real {
            let Some(current_role) = current_role.as_deref() else {
                continue;
            };
            if !current_druid_role_can_emit_day_callback(current_role) {
                continue;
            }
            if matches!(event.kind, CurrentDruidCallbackKind::OpaqueReal)
                && roles_equal(current_role, "Druid")
            {
                continue;
            }
            let mut anonymous_wretches = AnonymousWretchConstraints::empty();
            let mut register_as = None;
            let mut raw_bluff = None;
            let mut forbidden_raw_bluff = None;
            match raw_druid {
                CurrentDruidRawConstraint::Unconstrained => {}
                CurrentDruidRawConstraint::Forbidden => {
                    forbidden_raw_bluff = Some((card.position, normalize_role("Druid")));
                }
                CurrentDruidRawConstraint::Required => {
                    if current_raw_bluff_holder_at_druid_observation(
                        card.position,
                        boundary,
                        &timeline,
                        scenario,
                        state,
                    ) == CurrentMediumRawBluffHolder::Impossible
                    {
                        continue;
                    }
                    if anonymous_wretch_candidates.contains(&card.position) {
                        anonymous_wretches.forbidden.insert(card.position);
                        if !anonymous_wretch_assignment_possible(
                            &anonymous_wretches.required,
                            &anonymous_wretches.forbidden,
                            scenario,
                            state,
                        ) {
                            continue;
                        }
                    }
                    if current_spy_register_as_surface_at_druid_observation(
                        card.position,
                        boundary,
                        &timeline,
                        scenario,
                        state,
                    ) == Some(true)
                    {
                        if !current_medium_spy_register_as_label_allowed("Druid", state) {
                            continue;
                        }
                        register_as = Some((card.position, normalize_role("Druid")));
                    }
                    raw_bluff = Some((card.position, normalize_role("Druid")));
                }
            }
            let support = CurrentDruidSupport {
                anonymous_wretches,
                anonymous_type_options: HashMap::new(),
                anonymous_outcast_roles: HashMap::new(),
                register_as,
                raw_bluff,
                forbidden_raw_bluff,
                baker_spy_timeline: timeline,
                callbacks: vec![callback],
            };
            if !supports.contains(&support) {
                supports.push(support);
            }
            continue;
        }

        if matches!(event.kind, CurrentDruidCallbackKind::OpaqueReal)
            || event.callback_index == 0
                && !current_role
                    .as_deref()
                    .is_some_and(current_druid_role_has_no_day_callback)
            || current_raw_bluff_holder_at_druid_observation(
                card.position,
                boundary,
                &timeline,
                scenario,
                state,
            ) == CurrentMediumRawBluffHolder::Impossible
        {
            continue;
        }
        let mut anonymous_wretches = AnonymousWretchConstraints::empty();
        if anonymous_wretch_candidates.contains(&card.position) {
            anonymous_wretches.forbidden.insert(card.position);
            if !anonymous_wretch_assignment_possible(
                &anonymous_wretches.required,
                &anonymous_wretches.forbidden,
                scenario,
                state,
            ) {
                continue;
            }
        }
        let register_as = if current_spy_register_as_surface_at_druid_observation(
            card.position,
            boundary,
            &timeline,
            scenario,
            state,
        ) == Some(true)
        {
            if !current_medium_spy_register_as_label_allowed("Druid", state) {
                continue;
            }
            Some((card.position, normalize_role("Druid")))
        } else {
            None
        };
        let support = CurrentDruidSupport {
            anonymous_wretches,
            anonymous_type_options: HashMap::new(),
            anonymous_outcast_roles: HashMap::new(),
            register_as,
            raw_bluff: Some((card.position, normalize_role("Druid"))),
            forbidden_raw_bluff: None,
            baker_spy_timeline: timeline,
            callbacks: vec![callback],
        };
        if !supports.contains(&support) {
            supports.push(support);
        }
    }
    supports
}

fn current_druid_event_supports(
    card: &CardInfo,
    event: &CurrentDruidCallbackEvent,
    path: CurrentDruidResolvedPath,
    raw_druid: CurrentDruidRawConstraint,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentDruidSupport> {
    match &event.kind {
        CurrentDruidCallbackKind::Result(claim) => current_druid_result_supports(
            card,
            claim,
            CurrentDruidObservationBoundary::SettledRevealCount(
                event.settled_reveal_count,
            ),
            match path {
                CurrentDruidResolvedPath::Real => CurrentDruidProviderSelection::Real {
                    callback: current_druid_callback_witness(event, path),
                    raw_druid,
                },
                CurrentDruidResolvedPath::Raw => CurrentDruidProviderSelection::Raw {
                    callback: current_druid_callback_witness(event, path),
                    allow_real_druid: event.callback_index > 0,
                },
            },
            scenario,
            state,
        ),
        CurrentDruidCallbackKind::RamblerInterruption { .. }
        | CurrentDruidCallbackKind::OpaqueReal => current_druid_non_result_supports(
            card,
            event,
            path,
            raw_druid,
            scenario,
            state,
        ),
    }
}

fn merge_current_druid_supports(
    left: &CurrentDruidSupport,
    right: &CurrentDruidSupport,
    scenario: &Scenario,
    state: &GameState,
) -> Option<CurrentDruidSupport> {
    if left.baker_spy_timeline != right.baker_spy_timeline {
        return None;
    }
    let mut merged = left.clone();
    merged
        .anonymous_wretches
        .required
        .extend(&right.anonymous_wretches.required);
    merged
        .anonymous_wretches
        .forbidden
        .extend(&right.anonymous_wretches.forbidden);
    if !merged
        .anonymous_wretches
        .required
        .is_disjoint(&merged.anonymous_wretches.forbidden)
    {
        return None;
    }

    for (&position, &options) in &right.anonymous_type_options {
        if let Some(selected) = merged.anonymous_type_options.get_mut(&position) {
            *selected &= options;
            if *selected == 0 {
                return None;
            }
        } else if options == 0 {
            return None;
        } else {
            merged.anonymous_type_options.insert(position, options);
        }
    }
    for (&position, role) in &right.anonymous_outcast_roles {
        if merged
            .anonymous_outcast_roles
            .get(&position)
            .is_some_and(|selected| selected != role)
        {
            return None;
        }
        merged
            .anonymous_outcast_roles
            .insert(position, role.clone());
    }

    for (selected, incoming) in [
        (&mut merged.register_as, &right.register_as),
        (&mut merged.raw_bluff, &right.raw_bluff),
        (&mut merged.forbidden_raw_bluff, &right.forbidden_raw_bluff),
    ] {
        if selected
            .as_ref()
            .zip(incoming.as_ref())
            .is_some_and(|(known, candidate)| known != candidate)
        {
            return None;
        }
        if selected.is_none() {
            *selected = incoming.clone();
        }
    }
    if merged
        .raw_bluff
        .as_ref()
        .zip(merged.forbidden_raw_bluff.as_ref())
        .is_some_and(|(required, forbidden)| required == forbidden)
        || !current_hidden_anonymous_assignment_possible(
            &merged.anonymous_type_options,
            &merged.anonymous_outcast_roles,
            &merged.anonymous_wretches.required,
            &merged.anonymous_wretches.forbidden,
            scenario,
            state,
        )
    {
        return None;
    }
    if merged.callbacks.iter().any(|selected| {
        right.callbacks.iter().any(|incoming| {
            selected.activation_id == incoming.activation_id
                && selected.callback_index == incoming.callback_index
        })
    }) || merged
        .callbacks
        .last()
        .zip(right.callbacks.first())
        .is_some_and(|(selected, incoming)| {
            (selected.activation_id, selected.callback_index)
                >= (incoming.activation_id, incoming.callback_index)
        })
    {
        return None;
    }
    merged.callbacks.extend(&right.callbacks);
    Some(merged)
}



fn current_druid_group_supports(
    card: &CardInfo,
    group: &[CurrentDruidCallbackEvent],
    allow_pending_raw_callback: bool,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentDruidSupport> {
    match group {
        [event] if event.dispatch_path == CurrentDruidDispatchPath::Either => {
            let mut supports = current_druid_event_supports(
                card,
                event,
                CurrentDruidResolvedPath::Real,
                CurrentDruidRawConstraint::Forbidden,
                scenario,
                state,
            );
            // Only the terminal activation can be sampled after its real
            // callback but before the zero-delay raw-Druid callback appends.
            // Any later activation proves that an earlier callback group was
            // already immutable and complete.
            if allow_pending_raw_callback {
                for support in current_druid_event_supports(
                    card,
                    event,
                    CurrentDruidResolvedPath::Real,
                    CurrentDruidRawConstraint::Required,
                    scenario,
                    state,
                ) {
                    if !supports.contains(&support) {
                        supports.push(support);
                    }
                }
            }
            for support in current_druid_event_supports(
                card,
                event,
                CurrentDruidResolvedPath::Raw,
                CurrentDruidRawConstraint::Unconstrained,
                scenario,
                state,
            ) {
                if !supports.contains(&support) {
                    supports.push(support);
                }
            }
            supports
        }
        [real, raw]
            if real.dispatch_path == CurrentDruidDispatchPath::Real
                && raw.dispatch_path == CurrentDruidDispatchPath::Raw =>
        {
            let real_supports = current_druid_event_supports(
                card,
                real,
                CurrentDruidResolvedPath::Real,
                CurrentDruidRawConstraint::Unconstrained,
                scenario,
                state,
            );
            let raw_supports = current_druid_event_supports(
                card,
                raw,
                CurrentDruidResolvedPath::Raw,
                CurrentDruidRawConstraint::Unconstrained,
                scenario,
                state,
            );
            let mut combined = Vec::new();
            for selected in &real_supports {
                for support in &raw_supports {
                    if let Some(merged) =
                        merge_current_druid_supports(selected, support, scenario, state)
                    {
                        if !combined.contains(&merged) {
                            combined.push(merged);
                        }
                    }
                }
            }
            combined
        }
        _ => Vec::new(),
    }
}

fn current_druid_supports_for_payload(
    card: &CardInfo,
    payload: &CurrentDruidPayload,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentDruidSupport> {
    // An opaque executed Evil may have been any Start status/data writer.
    // Active-use sampling cannot reconstruct that world from a later role map.
    if current_has_unresolved_start_identity(scenario, state) {
        return Vec::new();
    }
    match payload {
        CurrentDruidPayload::Scalar(claim) => current_druid_result_supports(
            card,
            claim,
            CurrentDruidObservationBoundary::FinalCompatibility,
            CurrentDruidProviderSelection::ScalarCompatibility,
            scenario,
            state,
        ),
        CurrentDruidPayload::Ledger(events) => {
            let mut combined = Vec::new();
            let mut start = 0;
            while start < events.len() {
                let activation_id = events[start].activation_id;
                let mut end = start + 1;
                while end < events.len() && events[end].activation_id == activation_id {
                    end += 1;
                }
                let supports = current_druid_group_supports(
                    card,
                    &events[start..end],
                    end == events.len(),
                    scenario,
                    state,
                );
                if supports.is_empty() {
                    return Vec::new();
                }
                if start == 0 {
                    combined = supports;
                } else {
                    let mut next = Vec::new();
                    for selected in &combined {
                        for support in &supports {
                            if let Some(merged) = merge_current_druid_supports(
                                selected,
                                support,
                                scenario,
                                state,
                            ) {
                                if !next.contains(&merged) {
                                    next.push(merged);
                                }
                            }
                        }
                    }
                    if next.is_empty() {
                        return Vec::new();
                    }
                    combined = next;
                }
                start = end;
            }
            combined
        }
    }
}

fn validate_current_druid(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let Some(payload) = parse_current_druid_payload(card, state) else {
        return false;
    };
    !current_druid_supports_for_payload(card, &payload, scenario, state).is_empty()
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentJesterClaim {
    targets: [u8; 3],
    evil_count: i64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
enum CurrentJesterCallbackKind {
    Result(CurrentJesterClaim),
    RamblerInterruption { target: u8 },
    OpaqueReal,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentJesterCallbackEvent {
    activation_id: usize,
    callback_index: usize,
    dispatch_path: CurrentDruidDispatchPath,
    kind: CurrentJesterCallbackKind,
    references: Option<Vec<u8>>,
    settled_reveal_count: usize,
    reset_generation: usize,
    activation_evidence: CurrentDruidActivationEvidence,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CurrentJesterResolvedCallback {
    actor: u8,
    activation_id: usize,
    callback_index: usize,
    path: CurrentDruidResolvedPath,
    boundary: CurrentRamblerBoundary,
    truth: TruthStatus,
    interruption_target: Option<u8>,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentJesterSupport {
    anonymous_wretches: AnonymousWretchConstraints,
    register_as: Option<(u8, String)>,
    raw_bluff: Option<(u8, String)>,
    forbidden_raw_bluff: Option<(u8, String)>,
    baker_spy_timeline: BakerSpyTimeline,
    callbacks: Vec<CurrentJesterResolvedCallback>,
}

fn current_jester_claim_text(references: &[u8; 3], evil_count: i64) -> Option<String> {
    if !(0..=3).contains(&evil_count) {
        return None;
    }
    let mut sorted = *references;
    sorted.sort_unstable();
    let result = if evil_count == 1 {
        "There is 1 Evil".to_string()
    } else {
        format!("There are {evil_count} Evils")
    };
    Some(format!(
        "Among:\n#{}, #{}, #{}:\n{result}",
        sorted[0], sorted[1], sorted[2]
    ))
}

fn parse_current_jester_claim_fields(
    info: &serde_json::Map<String, serde_json::Value>,
    state: &GameState,
) -> Option<CurrentJesterClaim> {
    let targets = info
        .get("targets")?
        .as_array()?
        .iter()
        .map(|value| u8::try_from(value.as_u64()?).ok())
        .collect::<Option<Vec<_>>>()?;
    let targets: [u8; 3] = targets.try_into().ok()?;
    if targets
        .iter()
        .any(|target| *target == 0 || *target > state.n_cards)
        || targets[0] == targets[1]
        || targets[0] == targets[2]
        || targets[1] == targets[2]
    {
        return None;
    }
    let evil_count = info.get("evil_count")?.as_i64()?;
    (0..=3).contains(&evil_count).then_some(CurrentJesterClaim {
        targets,
        evil_count,
    })
}

fn current_jester_opaque_text_is_ambiguous(text: &str) -> bool {
    let normalized = current_druid_python_ignorecase_fold(text);
    let trimmed = current_druid_trim_python_regex_whitespace(&normalized);
    let near_jester = trimmed.strip_prefix("among").is_some_and(|suffix| {
        current_druid_trim_python_regex_whitespace(suffix).starts_with(':')
            && (current_druid_text_contains_word(trimmed, "evil")
                || current_druid_text_contains_word(trimmed, "evils"))
    });
    let shut_up = normalized.match_indices("shut").any(|(index, _)| {
        let before = normalized[..index].chars().next_back();
        let suffix = &normalized[index + "shut".len()..];
        let after_shut = current_druid_trim_python_regex_whitespace(suffix);
        let separated = suffix
            .chars()
            .next()
            .is_some_and(current_druid_is_python_regex_whitespace);
        let up_boundary = after_shut.strip_prefix("up").is_some_and(|after| {
            after
                .chars()
                .next()
                .is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
        });
        before.is_none_or(|character| !character.is_ascii_alphanumeric() && character != '_')
            && separated
            && up_boundary
    });
    near_jester || shut_up
}

fn parse_current_jester_callback_event(
    value: &serde_json::Value,
    state: &GameState,
) -> Option<CurrentJesterCallbackEvent> {
    const COMMON_FIELDS: &[&str] = &[
        "activation_id",
        "callback_index",
        "dispatch_path",
        "event_kind",
        "text",
        "references",
        "settled_reveal_count",
        "reset_generation",
        "activation_evidence",
    ];
    const RESULT_FIELDS: &[&str] = &[
        "activation_id",
        "callback_index",
        "dispatch_path",
        "event_kind",
        "text",
        "references",
        "settled_reveal_count",
        "reset_generation",
        "activation_evidence",
        "targets",
        "evil_count",
    ];
    const INTERRUPTION_FIELDS: &[&str] = &[
        "activation_id",
        "callback_index",
        "dispatch_path",
        "event_kind",
        "text",
        "references",
        "settled_reveal_count",
        "reset_generation",
        "activation_evidence",
        "shut_up_target",
    ];

    let event = value.as_object()?;
    let activation_id = event
        .get("activation_id")?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)?;
    let callback_index = event
        .get("callback_index")?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())?;
    let dispatch_path = parse_current_druid_dispatch_path(event.get("dispatch_path")?)?;
    let settled_reveal_count = event
        .get("settled_reveal_count")?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())
        .filter(|value| *value > 0)?;
    let reset_generation = event
        .get("reset_generation")?
        .as_u64()
        .and_then(|value| usize::try_from(value).ok())?;
    let activation_evidence =
        parse_current_druid_activation_evidence(event.get("activation_evidence")?)?;
    let references = parse_current_druid_references(event.get("references")?, state)?;
    let text = event.get("text")?.as_str()?;

    let kind = match event.get("event_kind")?.as_str()? {
        "jester_result" => {
            if !current_druid_has_exact_fields(event, RESULT_FIELDS) {
                return None;
            }
            let claim = parse_current_jester_claim_fields(event, state)?;
            let display_references: [u8; 3] = references.clone()?.try_into().ok()?;
            if current_jester_claim_text(&display_references, claim.evil_count).as_deref()
                != Some(text)
            {
                return None;
            }
            CurrentJesterCallbackKind::Result(claim)
        }
        "rambler_interruption" => {
            if !current_druid_has_exact_fields(event, INTERRUPTION_FIELDS) {
                return None;
            }
            let target = event
                .get("shut_up_target")?
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .filter(|value| *value > 0 && *value <= state.n_cards)?;
            if references.as_deref() != Some([target].as_slice())
                || text != format!("#{target}\nshut up!")
            {
                return None;
            }
            CurrentJesterCallbackKind::RamblerInterruption { target }
        }
        "opaque_real" => {
            if !current_druid_has_exact_fields(event, COMMON_FIELDS)
                || text.is_empty()
                || current_jester_opaque_text_is_ambiguous(text)
            {
                return None;
            }
            CurrentJesterCallbackKind::OpaqueReal
        }
        _ => return None,
    };

    Some(CurrentJesterCallbackEvent {
        activation_id,
        callback_index,
        dispatch_path,
        kind,
        references,
        settled_reveal_count,
        reset_generation,
        activation_evidence,
    })
}

fn parse_current_jester_payload(
    card: &CardInfo,
    state: &GameState,
) -> Option<Vec<CurrentJesterCallbackEvent>> {
    if card.position == 0
        || card.position > state.n_cards
        || !roles_equal(&card.apparent_role, "Jester")
        || card
            .info_parsed
            .get(JESTER_CURRENT_VARIANT_FIELD)
            .and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
        || card
            .info_parsed
            .get("callback_ledger_variant")
            .and_then(serde_json::Value::as_str)
            != Some("ordered_callbacks_v1")
        || state.baker_rule_version.as_deref() != Some(BAKER_CURRENT_RULE)
    {
        return None;
    }
    let values = card.info_parsed.get("callback_events")?.as_array()?;
    if values.is_empty() {
        return None;
    }
    let mut seen_reveals = HashSet::new();
    if state.reveal_order.len() > usize::from(state.n_cards)
        || state.reveal_order.iter().any(|position| {
            *position == 0 || *position > state.n_cards || !seen_reveals.insert(*position)
        })
    {
        return None;
    }

    let events = values
        .iter()
        .map(|value| parse_current_jester_callback_event(value, state))
        .collect::<Option<Vec<_>>>()?;
    let mut group_count = 0;
    let mut start = 0;
    let mut previous_boundary = 0;
    let mut previous_generation = 0;
    while start < events.len() {
        let activation_id = events[start].activation_id;
        if activation_id != group_count + 1 || events[start].callback_index != 0 {
            return None;
        }
        let mut end = start + 1;
        while end < events.len() && events[end].activation_id == activation_id {
            end += 1;
        }
        let group = &events[start..end];
        let first = &group[0];
        if first.settled_reveal_count < previous_boundary
            || first.settled_reveal_count > state.reveal_order.len()
            || !state.reveal_order[..first.settled_reveal_count].contains(&card.position)
            || group_count > 0 && first.reset_generation <= previous_generation
            || group.iter().enumerate().any(|(index, event)| {
                event.callback_index != index
                    || event.settled_reveal_count != first.settled_reveal_count
                    || event.reset_generation != first.reset_generation
                    || event.activation_evidence != first.activation_evidence
            })
        {
            return None;
        }
        match group {
            [only]
                if only.dispatch_path == CurrentDruidDispatchPath::Either
                    && only.activation_evidence
                        != CurrentDruidActivationEvidence::SameActivationExtension => {}
            [real, raw]
                if real.dispatch_path == CurrentDruidDispatchPath::Real
                    && raw.dispatch_path == CurrentDruidDispatchPath::Raw
                    && real.activation_evidence
                        != CurrentDruidActivationEvidence::SingleCallbackSuffix => {}
            _ => return None,
        }
        let evidence_is_reachable = match first.activation_evidence {
            CurrentDruidActivationEvidence::SingleCallbackSuffix => {
                group_count == 0 && first.reset_generation == 0 && group.len() == 1
            }
            CurrentDruidActivationEvidence::AutoUseClick => true,
            CurrentDruidActivationEvidence::SessionResetGeneration => {
                if group_count == 0 {
                    first.reset_generation > 0 && group.len() == 1
                } else {
                    group.len() == 1
                        || first.reset_generation.checked_sub(previous_generation) == Some(1)
                }
            }
            CurrentDruidActivationEvidence::SameActivationExtension => {
                group.len() == 2 && !matches!(group[0].kind, CurrentJesterCallbackKind::OpaqueReal)
            }
        };
        if !evidence_is_reachable
            || matches!(group[0].kind, CurrentJesterCallbackKind::OpaqueReal)
                && (group.len() != 2
                    || !matches!(
                        group[1].kind,
                        CurrentJesterCallbackKind::Result(_)
                            | CurrentJesterCallbackKind::RamblerInterruption { .. }
                    ))
            || group
                .iter()
                .skip(1)
                .any(|event| matches!(event.kind, CurrentJesterCallbackKind::OpaqueReal))
            || matches!(
                (&group[0].kind, group.get(1).map(|event| &event.kind)),
                (
                    CurrentJesterCallbackKind::Result(left),
                    Some(CurrentJesterCallbackKind::Result(right)),
                ) if left.targets != right.targets || group[0].references != group[1].references
            )
        {
            return None;
        }
        previous_boundary = first.settled_reveal_count;
        previous_generation = first.reset_generation;
        group_count += 1;
        start = end;
    }

    let latest = events.last()?;
    match &latest.kind {
        CurrentJesterCallbackKind::Result(claim) => {
            const RESULT_TOP_FIELDS: &[&str] = &[
                JESTER_CURRENT_VARIANT_FIELD,
                "callback_ledger_variant",
                "callback_events",
                "targets",
                "evil_count",
            ];
            if !current_druid_has_exact_fields(&card.info_parsed, RESULT_TOP_FIELDS)
                || parse_current_jester_claim_fields(&card.info_parsed, state)? != *claim
                || latest.references.as_deref().and_then(|references| {
                    <[u8; 3]>::try_from(references).ok()
                }).and_then(|references| current_jester_claim_text(&references, claim.evil_count))
                    .as_deref() != Some(card.info_text.as_str())
            {
                return None;
            }
        }
        CurrentJesterCallbackKind::RamblerInterruption { target } => {
            const INTERRUPTION_TOP_FIELDS: &[&str] = &[
                JESTER_CURRENT_VARIANT_FIELD,
                "callback_ledger_variant",
                "callback_events",
                "shut_up_target",
            ];
            if !current_druid_has_exact_fields(&card.info_parsed, INTERRUPTION_TOP_FIELDS)
                || card
                    .info_parsed
                    .get("shut_up_target")
                    .and_then(serde_json::Value::as_u64)
                    != Some(u64::from(*target))
                || card.info_text != format!("#{target}\nshut up!")
            {
                return None;
            }
        }
        CurrentJesterCallbackKind::OpaqueReal => return None,
    }

    let ledger_interruptions = events
        .iter()
        .filter_map(|event| match event.kind {
            CurrentJesterCallbackKind::RamblerInterruption { target } => Some(target),
            _ => None,
        })
        .collect::<Vec<_>>();
    let public_interruptions = state
        .rambler_shut_up_observations
        .iter()
        .filter_map(|observation| {
            (observation.speaker_position == card.position).then_some(observation.shut_up_target)
        })
        .collect::<Vec<_>>();
    if ledger_interruptions != public_interruptions
        || !ledger_interruptions.is_empty()
            && state.rambler_rule_version.as_deref() != Some(RAMBLER_CURRENT_RULE)
    {
        return None;
    }
    Some(events)
}

fn current_jester_raw_provider_truth(
    actor: u8,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> TruthStatus {
    if scenario.corrupted.contains(&actor) {
        return TruthStatus::Lying;
    }
    let healthy_bluff = scenario.puppet_position == Some(actor)
        || scenario.doppelganger_position == Some(actor)
        || current_data_role_at_druid_observation(actor, boundary, timeline, scenario, state)
            .as_deref()
            .is_some_and(|role| roles_equal(role, "Doppelganger"));
    if healthy_bluff {
        TruthStatus::Truthful
    } else {
        TruthStatus::Lying
    }
}

fn current_jester_real_provider_truth(
    actor: u8,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> TruthStatus {
    if is_runtime_evil_at(actor, scenario, state)
        && current_raw_bluff_holder_at_druid_observation(actor, boundary, timeline, scenario, state)
            != CurrentMediumRawBluffHolder::Impossible
    {
        TruthStatus::Truthful
    } else {
        truth_status(actor, scenario, state)
    }
}

fn current_jester_registered_alignment_at_boundary(
    position: u8,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<EffectiveAlignment> {
    // GetRegisterAlignment reads a live registerAs record before runtime
    // alignment. Stable Spy therefore counts Good until its delayed Baker
    // reset has settled, even on a runtime-Evil body.
    if current_spy_register_as_surface_at_druid_observation(
        position, boundary, timeline, scenario, state,
    )? {
        return Some(EffectiveAlignment::Good);
    }
    let role =
        current_data_role_at_druid_observation(position, boundary, timeline, scenario, state);
    if role
        .as_deref()
        .is_some_and(|role| normalize_role(role) == "unknown")
    {
        return None;
    }
    if role
        .as_deref()
        .is_some_and(|role| roles_equal(role, "Wretch"))
        || is_runtime_evil_at(position, scenario, state)
    {
        Some(EffectiveAlignment::Evil)
    } else {
        Some(EffectiveAlignment::Good)
    }
}

fn current_jester_claim_supports(
    claim: &CurrentJesterClaim,
    truth: TruthStatus,
    boundary: CurrentDruidObservationBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<AnonymousWretchConstraints> {
    let anonymous_candidates: HashSet<u8> = anonymous_natural_wretch_candidates(scenario, state)
        .into_iter()
        .collect();
    let mut fixed_evil = 0_i64;
    let mut unresolved = Vec::new();
    for &target in &claim.targets {
        let Some(alignment) = current_jester_registered_alignment_at_boundary(
            target, boundary, timeline, scenario, state,
        ) else {
            return Vec::new();
        };
        if alignment == EffectiveAlignment::Evil {
            fixed_evil += 1;
        } else if anonymous_candidates.contains(&target) {
            unresolved.push(target);
        }
    }

    fn enumerate(
        index: usize,
        candidates: &[u8],
        fixed_evil: i64,
        claim: &CurrentJesterClaim,
        truth: TruthStatus,
        required: &mut HashSet<u8>,
        forbidden: &mut HashSet<u8>,
        scenario: &Scenario,
        state: &GameState,
        supports: &mut Vec<AnonymousWretchConstraints>,
    ) {
        if index == candidates.len() {
            let actual = fixed_evil + required.len() as i64;
            let supported = match truth {
                TruthStatus::Truthful => claim.evil_count == actual,
                TruthStatus::Lying => claim.evil_count != actual,
            };
            if supported
                && anonymous_wretch_assignment_possible(required, forbidden, scenario, state)
            {
                let candidate = AnonymousWretchConstraints {
                    required: required.clone(),
                    forbidden: forbidden.clone(),
                };
                if !supports.contains(&candidate) {
                    supports.push(candidate);
                }
            }
            return;
        }

        let position = candidates[index];
        forbidden.insert(position);
        enumerate(
            index + 1,
            candidates,
            fixed_evil,
            claim,
            truth,
            required,
            forbidden,
            scenario,
            state,
            supports,
        );
        forbidden.remove(&position);

        required.insert(position);
        enumerate(
            index + 1,
            candidates,
            fixed_evil,
            claim,
            truth,
            required,
            forbidden,
            scenario,
            state,
            supports,
        );
        required.remove(&position);
    }

    let mut supports = Vec::new();
    enumerate(
        0,
        &unresolved,
        fixed_evil,
        claim,
        truth,
        &mut HashSet::new(),
        &mut HashSet::new(),
        scenario,
        state,
        &mut supports,
    );
    supports
}

fn current_jester_callback_witness(
    card: &CardInfo,
    event: &CurrentJesterCallbackEvent,
    path: CurrentDruidResolvedPath,
    truth: TruthStatus,
) -> CurrentJesterResolvedCallback {
    CurrentJesterResolvedCallback {
        actor: card.position,
        activation_id: event.activation_id,
        callback_index: event.callback_index,
        path,
        boundary: CurrentRamblerBoundary::SettledRevealCount(event.settled_reveal_count),
        truth,
        interruption_target: match event.kind {
            CurrentJesterCallbackKind::RamblerInterruption { target } => Some(target),
            _ => None,
        },
    }
}

#[allow(clippy::too_many_arguments)]
fn current_jester_append_event_supports(
    supports: &mut Vec<CurrentJesterSupport>,
    card: &CardInfo,
    event: &CurrentJesterCallbackEvent,
    path: CurrentDruidResolvedPath,
    truth: TruthStatus,
    register_as: Option<(u8, String)>,
    raw_bluff: Option<(u8, String)>,
    forbidden_raw_bluff: Option<(u8, String)>,
    actor_wretch_constraints: &AnonymousWretchConstraints,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) {
    let claim_supports = match &event.kind {
        CurrentJesterCallbackKind::Result(claim) => current_jester_claim_supports(
            claim,
            truth,
            CurrentDruidObservationBoundary::SettledRevealCount(event.settled_reveal_count),
            timeline,
            scenario,
            state,
        ),
        CurrentJesterCallbackKind::RamblerInterruption { .. }
        | CurrentJesterCallbackKind::OpaqueReal => {
            vec![AnonymousWretchConstraints::empty()]
        }
    };
    for mut anonymous_wretches in claim_supports {
        anonymous_wretches
            .required
            .extend(&actor_wretch_constraints.required);
        anonymous_wretches
            .forbidden
            .extend(&actor_wretch_constraints.forbidden);
        if !anonymous_wretches
            .required
            .is_disjoint(&anonymous_wretches.forbidden)
            || !anonymous_wretch_assignment_possible(
                &anonymous_wretches.required,
                &anonymous_wretches.forbidden,
                scenario,
                state,
            )
        {
            continue;
        }
        let support = CurrentJesterSupport {
            anonymous_wretches,
            register_as: register_as.clone(),
            raw_bluff: raw_bluff.clone(),
            forbidden_raw_bluff: forbidden_raw_bluff.clone(),
            baker_spy_timeline: timeline.clone(),
            callbacks: vec![current_jester_callback_witness(card, event, path, truth)],
        };
        if !supports.contains(&support) {
            supports.push(support);
        }
    }
}

fn current_jester_event_supports(
    card: &CardInfo,
    event: &CurrentJesterCallbackEvent,
    path: CurrentDruidResolvedPath,
    raw_jester: CurrentDruidRawConstraint,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentJesterSupport> {
    let boundary = CurrentDruidObservationBoundary::SettledRevealCount(event.settled_reveal_count);
    let anonymous_candidates: HashSet<u8> = anonymous_natural_wretch_candidates(scenario, state)
        .into_iter()
        .collect();
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_settled_reveal_count(event.settled_reveal_count, state) {
            continue;
        }
        let current_role = current_data_role_at_druid_observation(
            card.position,
            boundary,
            &timeline,
            scenario,
            state,
        );
        let raw_holder = current_raw_bluff_holder_at_druid_observation(
            card.position,
            boundary,
            &timeline,
            scenario,
            state,
        );
        let mut actor_wretches = AnonymousWretchConstraints::empty();
        let mut register_as = None;
        let mut raw_bluff = None;
        let mut forbidden_raw_bluff = None;

        let truth = match path {
            CurrentDruidResolvedPath::Real => {
                let Some(role) = current_role.as_deref() else {
                    continue;
                };
                let valid_real_provider = match event.kind {
                    CurrentJesterCallbackKind::OpaqueReal => {
                        current_druid_role_can_emit_day_callback(role)
                            && !roles_equal(role, "Jester")
                    }
                    CurrentJesterCallbackKind::Result(_)
                    | CurrentJesterCallbackKind::RamblerInterruption { .. } => {
                        roles_equal(role, "Jester")
                    }
                };
                if !valid_real_provider {
                    continue;
                }
                match raw_jester {
                    CurrentDruidRawConstraint::Unconstrained => {}
                    CurrentDruidRawConstraint::Forbidden => {
                        forbidden_raw_bluff = Some((card.position, normalize_role("Jester")));
                    }
                    CurrentDruidRawConstraint::Required => {
                        if raw_holder == CurrentMediumRawBluffHolder::Impossible {
                            continue;
                        }
                        if anonymous_candidates.contains(&card.position) {
                            actor_wretches.forbidden.insert(card.position);
                        }
                        if current_spy_register_as_surface_at_druid_observation(
                            card.position,
                            boundary,
                            &timeline,
                            scenario,
                            state,
                        ) == Some(true)
                        {
                            if !current_medium_spy_register_as_label_allowed("Jester", state) {
                                continue;
                            }
                            register_as = Some((card.position, normalize_role("Jester")));
                        }
                        raw_bluff = Some((card.position, normalize_role("Jester")));
                    }
                }
                current_jester_real_provider_truth(
                    card.position,
                    boundary,
                    &timeline,
                    scenario,
                    state,
                )
            }
            CurrentDruidResolvedPath::Raw => {
                if matches!(event.kind, CurrentJesterCallbackKind::OpaqueReal)
                    || event.callback_index == 0
                        && !current_role
                            .as_deref()
                            .is_some_and(current_druid_role_has_no_day_callback)
                    || raw_holder == CurrentMediumRawBluffHolder::Impossible
                {
                    continue;
                }
                if anonymous_candidates.contains(&card.position) {
                    actor_wretches.forbidden.insert(card.position);
                }
                if current_spy_register_as_surface_at_druid_observation(
                    card.position,
                    boundary,
                    &timeline,
                    scenario,
                    state,
                ) == Some(true)
                {
                    if !current_medium_spy_register_as_label_allowed("Jester", state) {
                        continue;
                    }
                    register_as = Some((card.position, normalize_role("Jester")));
                }
                raw_bluff = Some((card.position, normalize_role("Jester")));
                current_jester_raw_provider_truth(
                    card.position,
                    boundary,
                    &timeline,
                    scenario,
                    state,
                )
            }
        };

        current_jester_append_event_supports(
            &mut supports,
            card,
            event,
            path,
            truth,
            register_as,
            raw_bluff,
            forbidden_raw_bluff,
            &actor_wretches,
            &timeline,
            scenario,
            state,
        );
    }
    supports
}

fn merge_current_jester_supports(
    left: &CurrentJesterSupport,
    right: &CurrentJesterSupport,
    scenario: &Scenario,
    state: &GameState,
) -> Option<CurrentJesterSupport> {
    if left.baker_spy_timeline != right.baker_spy_timeline {
        return None;
    }
    let mut merged = left.clone();
    merged
        .anonymous_wretches
        .required
        .extend(&right.anonymous_wretches.required);
    merged
        .anonymous_wretches
        .forbidden
        .extend(&right.anonymous_wretches.forbidden);
    if !merged
        .anonymous_wretches
        .required
        .is_disjoint(&merged.anonymous_wretches.forbidden)
    {
        return None;
    }
    for (selected, incoming) in [
        (&mut merged.register_as, &right.register_as),
        (&mut merged.raw_bluff, &right.raw_bluff),
        (&mut merged.forbidden_raw_bluff, &right.forbidden_raw_bluff),
    ] {
        if selected
            .as_ref()
            .zip(incoming.as_ref())
            .is_some_and(|(known, candidate)| known != candidate)
        {
            return None;
        }
        if selected.is_none() {
            *selected = incoming.clone();
        }
    }
    if merged
        .raw_bluff
        .as_ref()
        .zip(merged.forbidden_raw_bluff.as_ref())
        .is_some_and(|(required, forbidden)| required == forbidden)
        || !anonymous_wretch_assignment_possible(
            &merged.anonymous_wretches.required,
            &merged.anonymous_wretches.forbidden,
            scenario,
            state,
        )
        || merged.callbacks.iter().any(|selected| {
            right.callbacks.iter().any(|incoming| {
                selected.actor == incoming.actor
                    && selected.activation_id == incoming.activation_id
                    && selected.callback_index == incoming.callback_index
            })
        })
        || merged.callbacks.last().zip(right.callbacks.first()).is_some_and(
            |(selected, incoming)| {
                (selected.actor, selected.activation_id, selected.callback_index)
                    >= (incoming.actor, incoming.activation_id, incoming.callback_index)
            },
        )
    {
        return None;
    }
    merged.callbacks.extend(&right.callbacks);
    Some(merged)
}

fn current_jester_group_supports(
    card: &CardInfo,
    group: &[CurrentJesterCallbackEvent],
    allow_pending_raw_callback: bool,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentJesterSupport> {
    match group {
        [event] if event.dispatch_path == CurrentDruidDispatchPath::Either => {
            let mut supports = current_jester_event_supports(
                card,
                event,
                CurrentDruidResolvedPath::Real,
                CurrentDruidRawConstraint::Forbidden,
                scenario,
                state,
            );
            if allow_pending_raw_callback {
                for support in current_jester_event_supports(
                    card,
                    event,
                    CurrentDruidResolvedPath::Real,
                    CurrentDruidRawConstraint::Required,
                    scenario,
                    state,
                ) {
                    if !supports.contains(&support) {
                        supports.push(support);
                    }
                }
            }
            for support in current_jester_event_supports(
                card,
                event,
                CurrentDruidResolvedPath::Raw,
                CurrentDruidRawConstraint::Unconstrained,
                scenario,
                state,
            ) {
                if !supports.contains(&support) {
                    supports.push(support);
                }
            }
            supports
        }
        [real, raw]
            if real.dispatch_path == CurrentDruidDispatchPath::Real
                && raw.dispatch_path == CurrentDruidDispatchPath::Raw =>
        {
            let real_supports = current_jester_event_supports(
                card,
                real,
                CurrentDruidResolvedPath::Real,
                CurrentDruidRawConstraint::Unconstrained,
                scenario,
                state,
            );
            let raw_supports = current_jester_event_supports(
                card,
                raw,
                CurrentDruidResolvedPath::Raw,
                CurrentDruidRawConstraint::Unconstrained,
                scenario,
                state,
            );
            let mut combined = Vec::new();
            for selected in &real_supports {
                for support in &raw_supports {
                    if let Some(merged) = merge_current_jester_supports(selected, support, scenario, state) {
                        if !combined.contains(&merged) {
                            combined.push(merged);
                        }
                    }
                }
            }
            combined
        }
        _ => Vec::new(),
    }
}

fn current_jester_supports_for_payload(
    card: &CardInfo,
    events: &[CurrentJesterCallbackEvent],
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentJesterSupport> {
    if current_has_unresolved_start_identity(scenario, state) {
        return Vec::new();
    }
    let mut combined = Vec::new();
    let mut start = 0;
    while start < events.len() {
        let activation_id = events[start].activation_id;
        let mut end = start + 1;
        while end < events.len() && events[end].activation_id == activation_id {
            end += 1;
        }
        let supports = current_jester_group_supports(
            card,
            &events[start..end],
            end == events.len(),
            scenario,
            state,
        );
        if supports.is_empty() {
            return Vec::new();
        }
        if start == 0 {
            combined = supports;
        } else {
            let mut next = Vec::new();
            for selected in &combined {
                for support in &supports {
                    if let Some(merged) = merge_current_jester_supports(selected, support, scenario, state) {
                        if !next.contains(&merged) {
                            next.push(merged);
                        }
                    }
                }
            }
            if next.is_empty() {
                return Vec::new();
            }
            combined = next;
        }
        start = end;
    }
    combined
}

fn validate_current_jester(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let Some(events) = parse_current_jester_payload(card, state) else {
        return false;
    };
    !current_jester_supports_for_payload(card, &events, scenario, state).is_empty()
}


#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
enum BishopType {
    Villager,
    Outcast,
    Minion,
    Demon,
}

impl BishopType {
    fn parse(value: &str) -> Option<Self> {
        match value {
            "Villager" => Some(Self::Villager),
            "Outcast" => Some(Self::Outcast),
            "Minion" => Some(Self::Minion),
            "Demon" => Some(Self::Demon),
            _ => None,
        }
    }

    fn as_str(self) -> &'static str {
        match self {
            Self::Villager => "Villager",
            Self::Outcast => "Outcast",
            Self::Minion => "Minion",
            Self::Demon => "Demon",
        }
    }

    fn bit(self) -> u8 {
        match self {
            Self::Villager => 1,
            Self::Outcast => 2,
            Self::Minion => 4,
            Self::Demon => 8,
        }
    }
}

const BISHOP_ANONYMOUS_GOOD_TYPES: u8 = 1 | 2 | 4;

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentBishopClaim {
    targets: Vec<u8>,
    types: Vec<BishopType>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentBishopTypeSurface {
    Known(BishopType),
    AnonymousGood,
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentBishopSupport {
    anonymous_type_options: HashMap<u8, u8>,
    baker_spy_timeline: BakerSpyTimeline,
}

fn current_bishop_claim_text(targets: &[u8], types: &[BishopType]) -> Option<String> {
    if !(1..=3).contains(&targets.len())
        || targets.len() != types.len()
        || targets.windows(2).any(|pair| pair[0] >= pair[1])
    {
        return None;
    }

    match targets.len() {
        1 => Some(format!("#{} is a {}", targets[0], types[0].as_str())),
        2 => Some(format!(
            "Between\n#{}, #{}\nthere is:\n{} and {}",
            targets[0],
            targets[1],
            types[0].as_str(),
            types[1].as_str(),
        )),
        3 => Some(format!(
            "Between\n#{}, #{}, #{}\nthere is:\n{}, {} and {}",
            targets[0],
            targets[1],
            targets[2],
            types[0].as_str(),
            types[1].as_str(),
            types[2].as_str(),
        )),
        _ => None,
    }
}

fn parse_current_bishop_claim(
    card: &CardInfo,
    source: CurrentPassivePayloadSource,
    state: &GameState,
) -> Option<CurrentBishopClaim> {
    if card.position == 0 || card.position > state.n_cards {
        return None;
    }
    let info = &card.info_parsed;
    let (variant_field, fixed_fields) = match source {
        CurrentPassivePayloadSource::Direct => {
            if card.apparent_role != "Bishop" {
                return None;
            }
            (BISHOP_CURRENT_VARIANT_FIELD, 1)
        }
        CurrentPassivePayloadSource::Poet => {
            if card.apparent_role != "Poet"
                || info.get("copied_role").and_then(serde_json::Value::as_str)
                    != Some("Bishop")
            {
                return None;
            }
            ("poet_variant", 2)
        }
    };
    if info.len() != fixed_fields + 2
        || info.get(variant_field).and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
    {
        return None;
    }

    let targets = poet_targets(info, state.n_cards, 1, 3)?;
    if targets.windows(2).any(|pair| pair[0] >= pair[1]) {
        return None;
    }
    let types: Option<Vec<BishopType>> = info
        .get("types")?
        .as_array()?
        .iter()
        .map(|value| BishopType::parse(value.as_str()?))
        .collect();
    let types = types?;
    if types.len() != targets.len() {
        return None;
    }
    let expected_text = current_bishop_claim_text(&targets, &types)?;
    (card.info_text == expected_text).then_some(CurrentBishopClaim { targets, types })
}

fn current_bishop_type_surface_at_observation(
    position: u8,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<CurrentBishopTypeSurface> {
    if current_spy_register_as_surface_at_observation(
        position,
        observation,
        timeline,
        scenario,
        state,
    )? {
        return Some(CurrentBishopTypeSurface::Known(BishopType::Villager));
    }
    let role = current_data_role_at_observation(
        position,
        observation,
        timeline,
        scenario,
        state,
    );
    let role = match role {
        // Scenario construction currently groups an untyped executed Evil as
        // `Unknown`. Its missing identity could have changed Start history
        // (Spy, Chancellor, Twin, Shaman, Puppeteer, ...), so Bishop cannot
        // safely resolve that world from a validator-local faction guess.
        None if scenario.is_evil(position) => return None,
        None => return Some(CurrentBishopTypeSurface::AnonymousGood),
        Some(role) => role,
    };
    let role = if normalize_role(&role) == "unknown" {
        match state.executed_evil_roles.get(&position) {
            Some(exact) if normalize_role(exact) != "unknown" => exact.clone(),
            _ if scenario.is_evil(position) => return None,
            _ => role,
        }
    } else {
        role
    };

    // GetCharacterData is register-as first. Spy's live registered Villager
    // and Wretch's registered Minion therefore override their real types.
    if roles_equal(&role, "Wretch") {
        return Some(CurrentBishopTypeSurface::Known(BishopType::Minion));
    }
    let in_minion_pool = state
        .deck
        .minions
        .iter()
        .any(|candidate| roles_equal(candidate, &role));
    let in_demon_pool = state
        .deck
        .demons
        .iter()
        .any(|candidate| roles_equal(candidate, &role));
    let role_type = match (in_minion_pool, in_demon_pool) {
        (true, false) => BishopType::Minion,
        (false, true) => BishopType::Demon,
        (true, true) => return None,
        (false, false) => match get_card(&role)?.faction {
            Faction::Villager => BishopType::Villager,
            Faction::Outcast => BishopType::Outcast,
            Faction::Minion => BishopType::Minion,
            Faction::Demon => BishopType::Demon,
        },
    };
    Some(CurrentBishopTypeSurface::Known(role_type))
}

fn current_bishop_anonymous_assignment_possible(
    type_options: &HashMap<u8, u8>,
    required_wretches: &HashSet<u8>,
    forbidden_wretches: &HashSet<u8>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if type_options.is_empty() {
        return anonymous_wretch_assignment_possible(
            required_wretches,
            forbidden_wretches,
            scenario,
            state,
        );
    }

    let mut candidates: Vec<(u8, u8)> = type_options
        .iter()
        .map(|(&position, &options)| {
            let mut options = options & BISHOP_ANONYMOUS_GOOD_TYPES;
            if required_wretches.contains(&position) {
                options &= BishopType::Minion.bit();
            }
            if forbidden_wretches.contains(&position) {
                options &= !BishopType::Minion.bit();
            }
            (position, options)
        })
        .collect();
    candidates.sort_unstable_by_key(|(_, options)| options.count_ones());
    if candidates.iter().any(|(_, options)| *options == 0) {
        return false;
    }
    let good_positions: HashSet<u8> = candidates.iter().map(|(position, _)| *position).collect();
    if required_wretches
        .iter()
        .any(|position| !good_positions.contains(position))
        || forbidden_wretches
            .iter()
            .any(|position| !good_positions.contains(position))
    {
        return false;
    }

    #[allow(clippy::too_many_arguments)]
    fn search(
        index: usize,
        candidates: &[(u8, u8)],
        villagers: &mut HashSet<u8>,
        outcasts: &mut HashSet<u8>,
        wretches: &mut HashSet<u8>,
        required_wretches: &HashSet<u8>,
        forbidden_wretches: &HashSet<u8>,
        scenario: &Scenario,
        state: &GameState,
    ) -> bool {
        if index == candidates.len() {
            let good_possible = if candidates.is_empty() {
                anonymous_wretch_assignment_possible(
                    required_wretches,
                    forbidden_wretches,
                    scenario,
                    state,
                )
            } else {
                crate::scenario::scenario_allows_anonymous_good_type_assignment(
                    villagers,
                    outcasts,
                    wretches,
                    scenario,
                    state,
                )
            };
            return good_possible;
        }
        let (position, options) = candidates[index];
        for role_type in [
            BishopType::Villager,
            BishopType::Outcast,
            BishopType::Minion,
        ] {
            if options & role_type.bit() == 0 {
                continue;
            }
            match role_type {
                BishopType::Villager => {
                    villagers.insert(position);
                }
                BishopType::Outcast => {
                    outcasts.insert(position);
                }
                BishopType::Minion => {
                    wretches.insert(position);
                }
                BishopType::Demon => unreachable!(),
            }
            if search(
                index + 1,
                candidates,
                villagers,
                outcasts,
                wretches,
                required_wretches,
                forbidden_wretches,
                scenario,
                state,
            ) {
                return true;
            }
            match role_type {
                BishopType::Villager => {
                    villagers.remove(&position);
                }
                BishopType::Outcast => {
                    outcasts.remove(&position);
                }
                BishopType::Minion => {
                    wretches.remove(&position);
                }
                BishopType::Demon => unreachable!(),
            }
        }
        false
    }

    search(
        0,
        &candidates,
        &mut HashSet::new(),
        &mut HashSet::new(),
        &mut HashSet::new(),
        required_wretches,
        forbidden_wretches,
        scenario,
        state,
    )
}

fn current_hidden_anonymous_assignment_possible(
    type_options: &HashMap<u8, u8>,
    exact_outcast_roles: &HashMap<u8, String>,
    required_wretches: &HashSet<u8>,
    forbidden_wretches: &HashSet<u8>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if !current_bishop_anonymous_assignment_possible(
        type_options,
        required_wretches,
        forbidden_wretches,
        scenario,
        state,
    ) {
        return false;
    }

    let mut roles: HashMap<String, (String, HashSet<u8>)> = HashMap::new();
    for (&position, role) in exact_outcast_roles {
        if type_options
            .get(&position)
            .is_none_or(|options| options & BishopType::Outcast.bit() == 0)
            || roles_equal(role, "Wretch")
        {
            return false;
        }
        let Some(card) = get_card(role) else {
            return false;
        };
        if card.faction != Faction::Outcast || card.name != role {
            return false;
        }
        roles
            .entry(normalize_role(role))
            .or_insert_with(|| (role.clone(), HashSet::new()))
            .1
            .insert(position);
    }

    roles.into_values().all(|(role, positions)| {
        crate::scenario::scenario_allows_anonymous_natural_outcast_role_assignments(
            &positions,
            &role,
            &HashSet::new(),
            scenario,
            state,
        )
    })
}

fn current_bishop_type_permutations(types: &[BishopType]) -> Vec<Vec<BishopType>> {
    fn enumerate(
        index: usize,
        values: &mut Vec<BishopType>,
        permutations: &mut Vec<Vec<BishopType>>,
    ) {
        if index == values.len() {
            if !permutations.contains(values) {
                permutations.push(values.clone());
            }
            return;
        }
        for swap in index..values.len() {
            values.swap(index, swap);
            enumerate(index + 1, values, permutations);
            values.swap(index, swap);
        }
    }

    let mut values = types.to_vec();
    let mut permutations = Vec::new();
    enumerate(0, &mut values, &mut permutations);
    permutations
}

fn current_bishop_surfaces_for_timeline(
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<(HashMap<u8, CurrentBishopTypeSurface>, HashMap<u8, u8>)> {
    if !timeline.supports_observation(observation, state) {
        return None;
    }
    let mut surfaces = HashMap::new();
    let mut anonymous = HashMap::new();
    for position in 1..=state.n_cards {
        let surface = current_bishop_type_surface_at_observation(
            position,
            observation,
            timeline,
            scenario,
            state,
        )?;
        match surface {
            CurrentBishopTypeSurface::AnonymousGood => {
                anonymous.insert(position, BISHOP_ANONYMOUS_GOOD_TYPES);
            }
            CurrentBishopTypeSurface::Known(_) => {}
        }
        surfaces.insert(position, surface);
    }
    Some((surfaces, anonymous))
}

fn current_bishop_truth_supports_for_timeline(
    claim: &CurrentBishopClaim,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentBishopSupport> {
    let mut distinct = claim.types.clone();
    distinct.sort_unstable();
    distinct.dedup();
    if distinct.len() != claim.types.len() {
        return Vec::new();
    }
    let has_villager = distinct.contains(&BishopType::Villager);
    let has_outcast = distinct.contains(&BishopType::Outcast);
    let has_minion = distinct.contains(&BishopType::Minion);
    let has_demon = distinct.contains(&BishopType::Demon);
    if has_minion == has_demon {
        return Vec::new();
    }

    let Some((surfaces, mut base_options)) = current_bishop_surfaces_for_timeline(
        observation,
        timeline,
        scenario,
        state,
    ) else {
        return Vec::new();
    };
    for surface in surfaces.values() {
        let CurrentBishopTypeSurface::Known(role_type) = surface else {
            continue;
        };
        if (*role_type == BishopType::Villager && !has_villager)
            || (*role_type == BishopType::Outcast && !has_outcast)
            || (*role_type == BishopType::Minion && has_demon)
        {
            return Vec::new();
        }
    }
    let allowed_anonymous = u8::from(has_villager) * BishopType::Villager.bit()
        | u8::from(has_outcast) * BishopType::Outcast.bit()
        | u8::from(has_minion) * BishopType::Minion.bit()
        | u8::from(has_demon) * BishopType::Demon.bit();
    for options in base_options.values_mut() {
        *options &= allowed_anonymous;
        if *options == 0 {
            return Vec::new();
        }
    }

    let mut supports = Vec::new();
    for permutation in current_bishop_type_permutations(&claim.types) {
        let mut anonymous_type_options = base_options.clone();
        let mut compatible = true;
        for (&target, &claimed_type) in claim.targets.iter().zip(permutation.iter()) {
            match surfaces.get(&target) {
                Some(CurrentBishopTypeSurface::Known(actual)) => {
                    compatible &= *actual == claimed_type;
                }
                Some(CurrentBishopTypeSurface::AnonymousGood) => {
                    anonymous_type_options.insert(target, claimed_type.bit());
                }
                None => compatible = false,
            }
        }
        if !compatible
            || !current_bishop_anonymous_assignment_possible(
                &anonymous_type_options,
                &HashSet::new(),
                &HashSet::new(),
                scenario,
                state,
            )
        {
            continue;
        }
        let support = CurrentBishopSupport {
            anonymous_type_options,
            baker_spy_timeline: timeline.clone(),
        };
        if !supports.contains(&support) {
            supports.push(support);
        }
    }
    supports
}

fn current_bishop_authored_bluff_good_types(state: &GameState) -> Option<(bool, bool)> {
    if state.board_count_provenance
        != crate::types::BoardCountProvenance::TrustedPreStart
    {
        return None;
    }
    let has_town = state.board_villager_count? > 0;
    let has_outcast = state.board_outcast_count? > 0;
    Some((has_town, has_outcast))
}

fn current_bishop_authored_minion_present(
    scenario: &Scenario,
    state: &GameState,
) -> Option<bool> {
    if state.board_count_provenance
        == crate::types::BoardCountProvenance::TrustedPreStart
    {
        if let Some(count) = state.board_minion_count {
            return Some(count > 0);
        }
    }

    let mut represented = scenario.evil_positions.clone();
    for (&position, role) in &state.executed_evil_roles {
        represented
            .entry(position)
            .and_modify(|known| {
                if normalize_role(known) == "unknown" {
                    *known = role.clone();
                }
            })
            .or_insert_with(|| role.clone());
    }

    let mut has_minion = false;
    for role in represented.values() {
        let normalized = normalize_role(role);
        if normalized == "unknown" {
            return None;
        }
        // Puppet is generated at Start and is absent from CurrentScript's
        // authored Minion list.
        if normalized == "puppet" {
            continue;
        }
        let in_minion_pool = state
            .deck
            .minions
            .iter()
            .any(|candidate| roles_equal(candidate, role));
        let in_demon_pool = state
            .deck
            .demons
            .iter()
            .any(|candidate| roles_equal(candidate, role));
        match (in_minion_pool, in_demon_pool) {
            (true, false) => has_minion = true,
            (false, true) => {}
            (true, true) => return None,
            (false, false) => match get_card(role).map(|card| card.faction) {
                Some(Faction::Minion) => has_minion = true,
                Some(Faction::Demon | Faction::Villager | Faction::Outcast) => {}
                None => return None,
            },
        }
    }
    Some(has_minion)
}

fn current_bishop_bluff_supports_for_timeline(
    claim: &CurrentBishopClaim,
    observation: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentBishopSupport> {
    let Some((has_town, has_outcast)) = current_bishop_authored_bluff_good_types(state) else {
        return Vec::new();
    };
    let has_minion = claim.types.contains(&BishopType::Minion);
    let has_demon = claim.types.contains(&BishopType::Demon);
    if has_minion == has_demon
        || current_bishop_authored_minion_present(scenario, state) != Some(has_minion)
    {
        return Vec::new();
    }
    let expected_targets = if has_outcast { 3 } else { 2 };
    let mut authored_types = vec![if has_minion {
        BishopType::Minion
    } else {
        BishopType::Demon
    }];
    if has_outcast {
        authored_types.push(BishopType::Outcast);
    }
    if has_town {
        authored_types.push(BishopType::Villager);
    }
    let mut claimed_types = claim.types.clone();
    authored_types.sort_unstable();
    claimed_types.sort_unstable();
    if claim.targets.len() != expected_targets || claimed_types != authored_types {
        return Vec::new();
    }

    let Some((surfaces, mut anonymous_type_options)) = current_bishop_surfaces_for_timeline(
        observation,
        timeline,
        scenario,
        state,
    ) else {
        return Vec::new();
    };
    for &target in &claim.targets {
        match surfaces.get(&target) {
            Some(CurrentBishopTypeSurface::Known(BishopType::Villager)) => {}
            Some(CurrentBishopTypeSurface::AnonymousGood) => {
                anonymous_type_options.insert(target, BishopType::Villager.bit());
            }
            _ => return Vec::new(),
        }
    }
    if !current_bishop_anonymous_assignment_possible(
        &anonymous_type_options,
        &HashSet::new(),
        &HashSet::new(),
        scenario,
        state,
    ) {
        return Vec::new();
    }
    vec![CurrentBishopSupport {
        anonymous_type_options,
        baker_spy_timeline: timeline.clone(),
    }]
}

fn current_bishop_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> Vec<CurrentBishopSupport> {
    let Some(claim) = parse_current_bishop_claim(card, source, state) else {
        return Vec::new();
    };
    baker_spy_conversion_timelines(scenario, state)
        .into_iter()
        .flat_map(|timeline| match truth_status(card.position, scenario, state) {
            TruthStatus::Truthful => current_bishop_truth_supports_for_timeline(
                &claim,
                card.position,
                &timeline,
                scenario,
                state,
            ),
            TruthStatus::Lying => current_bishop_bluff_supports_for_timeline(
                &claim,
                card.position,
                &timeline,
                scenario,
                state,
            ),
        })
        .collect()
}

fn validate_current_bishop(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
    source: CurrentPassivePayloadSource,
) -> bool {
    !current_bishop_supports(card, scenario, state, source).is_empty()
}

fn validate_bishop(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    match current_passive_payload_source(card, BISHOP_CURRENT_VARIANT_FIELD, "Bishop") {
        Ok(Some(source)) => return validate_current_bishop(card, scenario, state, source),
        Err(()) => return false,
        Ok(None) => {}
    }

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

fn parse_current_bounty_hunter_target(card: &CardInfo, state: &GameState) -> Option<u8> {
    let info = &card.info_parsed;
    if !roles_equal(&card.apparent_role, "Poet")
        || !poet_has_exact_fields(info, &["evil_position"])
        || info
            .get("poet_variant")
            .and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
        || info
            .get("copied_role")
            .and_then(serde_json::Value::as_str)
            != Some("Bounty Hunter")
    {
        return None;
    }
    let target = poet_position_value(info.get("evil_position"), state.n_cards)?;
    (card.position > 0
        && card.position <= state.n_cards
        && card.info_text == format!("#{target}\nis Evil"))
    .then_some(target)
}

fn validate_current_bounty_hunter(
    actor: u8,
    target: u8,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    !current_bounty_hunter_hidden_supports_for_target(actor, target, scenario, state).is_empty()
}

#[cfg(test)]
fn validate_current_bounty_hunter_wretch_consistency(
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let anonymous_wretches: HashSet<u8> =
        anonymous_natural_wretch_candidates(scenario, state)
            .into_iter()
            .collect();
    let mut required = HashSet::new();
    let mut forbidden = HashSet::new();

    for card in &state.cards {
        if executed_evil_origin_is_unresolved(card.position, scenario, state) {
            continue;
        }
        if card
            .info_parsed
            .get("poet_variant")
            .and_then(serde_json::Value::as_str)
            != Some(POET_CURRENT_VARIANT)
            || card
                .info_parsed
                .get("copied_role")
                .and_then(serde_json::Value::as_str)
                != Some("Bounty Hunter")
        {
            continue;
        }
        let Some(target) = parse_current_bounty_hunter_target(card, state) else {
            // Individual schema validation remains fail-closed in the card
            // pass below; malformed or dormant direct-role payloads do not
            // contribute hidden-identity constraints here.
            continue;
        };
        if registered_alignment_at(target, scenario, state) == EffectiveAlignment::Evil
            || !anonymous_wretches.contains(&target)
        {
            continue;
        }
        match truth_status(card.position, scenario, state) {
            TruthStatus::Truthful => {
                required.insert(target);
            }
            TruthStatus::Lying => {
                forbidden.insert(target);
            }
        }
    }

    (required.is_empty() && forbidden.is_empty())
        || anonymous_wretch_assignment_possible(&required, &forbidden, scenario, state)
}

#[derive(Debug, Clone, PartialEq, Eq)]
struct CurrentHiddenSurfaceSupport {
    anonymous_wretches: AnonymousWretchConstraints,
    bishop_type_options: HashMap<u8, u8>,
    anonymous_outcast_roles: HashMap<u8, String>,
    register_as: Option<(u8, String)>,
    raw_bluff: Option<(u8, String)>,
    forbidden_raw_bluff: Option<(u8, String)>,
    baker_spy_timeline: Option<BakerSpyTimeline>,
    jester_callbacks: Vec<CurrentJesterResolvedCallback>,
}

fn current_bounty_hunter_hidden_supports_for_target(
    actor: u8,
    target: u8,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentHiddenSurfaceSupport> {
    let anonymous_wretch_candidate =
        anonymous_natural_wretch_candidates(scenario, state).contains(&target);
    let truth = truth_status(actor, scenario, state);
    let mut supports = Vec::new();
    for timeline in baker_spy_conversion_timelines(scenario, state) {
        if !timeline.supports_observation(actor, state) {
            continue;
        }
        let Some(known_alignment) = registered_alignment_at_observation(
            target,
            actor,
            &timeline,
            scenario,
            state,
        ) else {
            continue;
        };
        let anonymous_wretches = match truth {
            TruthStatus::Truthful if known_alignment == EffectiveAlignment::Evil => {
                AnonymousWretchConstraints::empty()
            }
            TruthStatus::Truthful if anonymous_wretch_candidate => AnonymousWretchConstraints {
                required: HashSet::from([target]),
                forbidden: HashSet::new(),
            },
            TruthStatus::Truthful => continue,
            TruthStatus::Lying if known_alignment == EffectiveAlignment::Evil => continue,
            TruthStatus::Lying if anonymous_wretch_candidate => AnonymousWretchConstraints {
                required: HashSet::new(),
                forbidden: HashSet::from([target]),
            },
            TruthStatus::Lying => AnonymousWretchConstraints::empty(),
        };
        if anonymous_wretch_assignment_possible(
            &anonymous_wretches.required,
            &anonymous_wretches.forbidden,
            scenario,
            state,
        ) {
            supports.push(CurrentHiddenSurfaceSupport {
                anonymous_wretches,
                bishop_type_options: HashMap::new(),
                anonymous_outcast_roles: HashMap::new(),
                register_as: None,
                raw_bluff: None,
                forbidden_raw_bluff: None,
                baker_spy_timeline: Some(timeline),
                jester_callbacks: Vec::new(),
            });
        }
    }
    supports
}

fn current_bounty_hunter_hidden_supports(
    card: &CardInfo,
    scenario: &Scenario,
    state: &GameState,
) -> Vec<CurrentHiddenSurfaceSupport> {
    let Some(target) = parse_current_bounty_hunter_target(card, state) else {
        return Vec::new();
    };
    current_bounty_hunter_hidden_supports_for_target(
        card.position,
        target,
        scenario,
        state,
    )
}

fn validate_current_hidden_surface_consistency(
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let has_current_full_pool_provider = state.cards.iter().any(|card| {
        card.info_parsed.contains_key(EMPRESS_CURRENT_VARIANT_FIELD)
            || card
                .info_parsed
                .contains_key(GEMCRAFTER_CURRENT_VARIANT_FIELD)
            || card.info_parsed.contains_key(BARD_CURRENT_VARIANT_FIELD)
            || card
                .info_parsed
                .contains_key(CONFESSOR_CURRENT_VARIANT_FIELD)
            || card.info_parsed.contains_key(DRUID_CURRENT_VARIANT_FIELD)
            || (card.info_parsed.contains_key(JESTER_CURRENT_VARIANT_FIELD)
                && card.info_parsed.contains_key("callback_events"))
            || (card.info_parsed.contains_key("poet_variant")
                && card
                    .info_parsed
                    .get("copied_role")
                    .and_then(serde_json::Value::as_str)
                    .is_some_and(|role| matches!(role, "Empress" | "Gemcrafter" | "Bard")))
    });
    if has_current_full_pool_provider && current_has_unresolved_start_identity(scenario, state) {
        return false;
    }

    let mut observations: Vec<Vec<CurrentHiddenSurfaceSupport>> = Vec::new();
    for card in &state.cards {
        if executed_evil_origin_is_unresolved(card.position, scenario, state) {
            continue;
        }

        let apparent = normalize_role(&card.apparent_role);
        let copied = card
            .info_parsed
            .get("copied_role")
            .and_then(serde_json::Value::as_str);
        let supports = if apparent == "lover"
            || (apparent == "poet" && copied == Some("Lover"))
        {
            match current_passive_payload_source(card, LOVER_CURRENT_VARIANT_FIELD, "Lover") {
                Ok(Some(source)) => Some(
                    current_lover_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: None,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "scout"
            || (apparent == "poet" && copied == Some("Scout"))
        {
            match current_passive_payload_source(card, SCOUT_CURRENT_VARIANT_FIELD, "Scout") {
                Ok(Some(source)) => Some(
                    current_scout_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: support.register_as,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "oracle"
            || (apparent == "poet" && copied == Some("Oracle"))
        {
            match current_passive_payload_source(card, ORACLE_CURRENT_VARIANT_FIELD, "Oracle") {
                Ok(Some(source)) => Some(
                    current_oracle_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: support.register_as,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "hunter"
            || (apparent == "poet" && copied == Some("Hunter"))
        {
            match current_passive_payload_source(card, HUNTER_CURRENT_VARIANT_FIELD, "Hunter") {
                Ok(Some(source)) => Some(
                    current_hunter_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: None,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "enlightened"
            || (apparent == "poet" && copied == Some("Enlightened"))
        {
            match current_passive_payload_source(
                card,
                ENLIGHTENED_CURRENT_VARIANT_FIELD,
                "Enlightened",
            ) {
                Ok(Some(source)) => Some(
                    current_enlightened_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: None,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "empress"
            || (apparent == "poet" && copied == Some("Empress"))
        {
            match current_passive_payload_source(
                card,
                EMPRESS_CURRENT_VARIANT_FIELD,
                "Empress",
            ) {
                Ok(Some(source)) => Some(
                    current_empress_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: None,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "gemcrafter"
            || (apparent == "poet" && copied == Some("Gemcrafter"))
        {
            match current_passive_payload_source(
                card,
                GEMCRAFTER_CURRENT_VARIANT_FIELD,
                "Gemcrafter",
            ) {
                Ok(Some(source)) => Some(
                    current_gemcrafter_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: None,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "confessor" {
            match current_passive_payload_source(
                card,
                CONFESSOR_CURRENT_VARIANT_FIELD,
                "Confessor",
            ) {
                Ok(Some(CurrentPassivePayloadSource::Direct)) => Some(
                    current_confessor_supports(card, scenario, state)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: support.register_as,
                            raw_bluff: support.raw_bluff,
                            forbidden_raw_bluff: support.forbidden_raw_bluff,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(Some(CurrentPassivePayloadSource::Poet)) | Err(()) => Some(Vec::new()),
                Ok(None) => None,
            }
        } else if apparent == "druid" {
            match current_passive_payload_source(card, DRUID_CURRENT_VARIANT_FIELD, "Druid") {
                Ok(Some(CurrentPassivePayloadSource::Direct)) => {
                    match parse_current_druid_payload(card, state) {
                        Some(payload) => Some(
                            current_druid_supports_for_payload(card, &payload, scenario, state)
                                .into_iter()
                                .map(|support| CurrentHiddenSurfaceSupport {
                                    anonymous_wretches: support.anonymous_wretches,
                                    bishop_type_options: support.anonymous_type_options,
                                    anonymous_outcast_roles: support.anonymous_outcast_roles,
                                    register_as: support.register_as,
                                    raw_bluff: support.raw_bluff,
                                    forbidden_raw_bluff: support.forbidden_raw_bluff,
                                    baker_spy_timeline: Some(support.baker_spy_timeline),
                                    jester_callbacks: Vec::new(),
                                })
                                .collect(),
                        ),
                        None => Some(Vec::new()),
                    }
                }
                Ok(Some(CurrentPassivePayloadSource::Poet)) | Err(()) => Some(Vec::new()),
                Ok(None) => None,
            }
        } else if apparent == "jester" {
            match current_passive_payload_source(card, JESTER_CURRENT_VARIANT_FIELD, "Jester") {
                Ok(Some(CurrentPassivePayloadSource::Direct)) if card.info_parsed.len() == 1 => {
                    None
                }
                Ok(Some(CurrentPassivePayloadSource::Direct)) => {
                    match parse_current_jester_payload(card, state) {
                        Some(events) => Some(
                            current_jester_supports_for_payload(card, &events, scenario, state)
                                .into_iter()
                                .map(|support| CurrentHiddenSurfaceSupport {
                                    anonymous_wretches: support.anonymous_wretches,
                                    bishop_type_options: HashMap::new(),
                                    anonymous_outcast_roles: HashMap::new(),
                                    register_as: support.register_as,
                                    raw_bluff: support.raw_bluff,
                                    forbidden_raw_bluff: support.forbidden_raw_bluff,
                                    baker_spy_timeline: Some(support.baker_spy_timeline),
                                    jester_callbacks: support.callbacks,
                                })
                                .collect(),
                        ),
                        None => Some(Vec::new()),
                    }
                }
                Ok(Some(CurrentPassivePayloadSource::Poet)) | Err(()) => Some(Vec::new()),
                Ok(None) => None,
            }
        } else if apparent == "bard"
            || (apparent == "poet" && copied == Some("Bard"))
        {
            match current_passive_payload_source(card, BARD_CURRENT_VARIANT_FIELD, "Bard") {
                Ok(Some(source)) => Some(
                    current_bard_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: support.anonymous_wretches,
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: None,
                            raw_bluff: support.raw_bluff,
                            forbidden_raw_bluff: support.forbidden_raw_bluff,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "bishop"
            || (apparent == "poet" && copied == Some("Bishop"))
        {
            match current_passive_payload_source(card, BISHOP_CURRENT_VARIANT_FIELD, "Bishop") {
                Ok(Some(source)) => Some(
                    current_bishop_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: AnonymousWretchConstraints::empty(),
                            bishop_type_options: support.anonymous_type_options,
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: None,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "medium"
            || (apparent == "poet" && copied == Some("Medium"))
        {
            match current_passive_payload_source(card, MEDIUM_CURRENT_VARIANT_FIELD, "Medium") {
                Ok(Some(source)) => Some(
                    current_medium_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: AnonymousWretchConstraints {
                                required: support.required_anonymous_wretches,
                                forbidden: support.forbidden_anonymous_wretches,
                            },
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: support.register_as,
                            raw_bluff: support.raw_bluff,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "knitter"
            || (apparent == "poet" && copied == Some("Knitter"))
        {
            match current_passive_payload_source(card, KNITTER_CURRENT_VARIANT_FIELD, "Knitter") {
                Ok(Some(source)) => Some(
                    current_knitter_supports(card, scenario, state, source)
                        .into_iter()
                        .map(|support| CurrentHiddenSurfaceSupport {
                            anonymous_wretches: AnonymousWretchConstraints {
                                required: support.required_anonymous_wretches,
                                forbidden: support.forbidden_anonymous_wretches,
                            },
                            bishop_type_options: HashMap::new(),
                            anonymous_outcast_roles: HashMap::new(),
                            register_as: None,
                            raw_bluff: None,
                            forbidden_raw_bluff: None,
                            baker_spy_timeline: Some(support.baker_spy_timeline),
                            jester_callbacks: Vec::new(),
                        })
                        .collect(),
                ),
                Ok(None) => None,
                Err(()) => Some(Vec::new()),
            }
        } else if apparent == "poet" && copied == Some("Bounty Hunter") {
            (card
                .info_parsed
                .get("poet_variant")
                .and_then(serde_json::Value::as_str)
                == Some(POET_CURRENT_VARIANT))
            .then(|| current_bounty_hunter_hidden_supports(card, scenario, state))
        } else {
            None
        };

        if let Some(supports) = supports {
            if supports.is_empty() {
                return false;
            }
            observations.push(supports);
        }
    }

    fn search(
        index: usize,
        observations: &[Vec<CurrentHiddenSurfaceSupport>],
        required: &HashSet<u8>,
        forbidden: &HashSet<u8>,
        bishop_type_options: &HashMap<u8, u8>,
        anonymous_outcast_roles: &HashMap<u8, String>,
        register_as: &HashMap<u8, String>,
        raw_bluffs: &HashMap<u8, String>,
        forbidden_raw_bluffs: &HashMap<u8, HashSet<String>>,
        baker_spy_timeline: Option<&BakerSpyTimeline>,
        jester_callbacks: &[CurrentJesterResolvedCallback],
        scenario: &Scenario,
        state: &GameState,
    ) -> bool {
        if index == observations.len() {
            if state.rambler_rule_version.as_deref() == Some(RAMBLER_CURRENT_RULE)
                && state
                    .cards
                    .iter()
                    .any(|card| is_ordered_current_druid(card) || is_ordered_current_jester(card))
            {
                let Some(timeline) = baker_spy_timeline else {
                    return false;
                };
                return current_rambler_timeline_jointly_possible(
                    timeline,
                    register_as,
                    raw_bluffs,
                    forbidden_raw_bluffs,
                    anonymous_outcast_roles,
                    jester_callbacks,
                    scenario,
                    state,
                );
            }
            return true;
        }
        for support in &observations[index] {
            let mut next_required = required.clone();
            next_required.extend(&support.anonymous_wretches.required);
            let mut next_forbidden = forbidden.clone();
            next_forbidden.extend(&support.anonymous_wretches.forbidden);
            if !next_required.is_disjoint(&next_forbidden) {
                continue;
            }

            let mut next_bishop_type_options = bishop_type_options.clone();
            let mut bishop_types_compatible = true;
            for (&position, &options) in &support.bishop_type_options {
                if let Some(selected) = next_bishop_type_options.get_mut(&position) {
                    *selected &= options;
                    bishop_types_compatible &= *selected != 0;
                } else {
                    next_bishop_type_options.insert(position, options);
                    bishop_types_compatible &= options != 0;
                }
            }
            if !bishop_types_compatible {
                continue;
            }

            let mut next_anonymous_outcast_roles = anonymous_outcast_roles.clone();
            let mut outcast_roles_compatible = true;
            for (&position, role) in &support.anonymous_outcast_roles {
                if next_anonymous_outcast_roles
                    .get(&position)
                    .is_some_and(|selected| selected != role)
                {
                    outcast_roles_compatible = false;
                    break;
                }
                next_anonymous_outcast_roles.insert(position, role.clone());
            }
            if !outcast_roles_compatible
                || !current_hidden_anonymous_assignment_possible(
                    &next_bishop_type_options,
                    &next_anonymous_outcast_roles,
                    &next_required,
                    &next_forbidden,
                    scenario,
                    state,
                )
            {
                continue;
            }

            let mut next_register_as = register_as.clone();
            if let Some((position, role)) = support.register_as.as_ref() {
                if next_register_as
                    .get(position)
                    .is_some_and(|selected| selected != role)
                {
                    continue;
                }
                next_register_as.insert(*position, role.clone());
            }
            let mut next_raw_bluffs = raw_bluffs.clone();
            if let Some((position, role)) = support.raw_bluff.as_ref() {
                if next_raw_bluffs
                    .get(position)
                    .is_some_and(|selected| selected != role)
                    || forbidden_raw_bluffs
                        .get(position)
                        .is_some_and(|forbidden| forbidden.contains(role))
                {
                    continue;
                }
                next_raw_bluffs.insert(*position, role.clone());
            }
            let mut next_forbidden_raw_bluffs = forbidden_raw_bluffs.clone();
            if let Some((position, role)) = support.forbidden_raw_bluff.as_ref() {
                if next_raw_bluffs
                    .get(position)
                    .is_some_and(|selected| selected == role)
                {
                    continue;
                }
                next_forbidden_raw_bluffs
                    .entry(*position)
                    .or_default()
                    .insert(role.clone());
            }
            if let Some(timeline) = support.baker_spy_timeline.as_ref() {
                if baker_spy_timeline.is_some_and(|selected| selected != timeline) {
                    continue;
                }
            }
            let next_timeline = support
                .baker_spy_timeline
                .as_ref()
                .or(baker_spy_timeline);
            let mut next_jester_callbacks = jester_callbacks.to_vec();
            if support.jester_callbacks.iter().any(|incoming| {
                next_jester_callbacks.iter().any(|selected| {
                    selected.actor == incoming.actor
                        && selected.activation_id == incoming.activation_id
                        && selected.callback_index == incoming.callback_index
                })
            }) {
                continue;
            }
            next_jester_callbacks.extend(&support.jester_callbacks);
            if search(
                index + 1,
                observations,
                &next_required,
                &next_forbidden,
                &next_bishop_type_options,
                &next_anonymous_outcast_roles,
                &next_register_as,
                &next_raw_bluffs,
                &next_forbidden_raw_bluffs,
                next_timeline,
                &next_jester_callbacks,
                scenario,
                state,
            ) {
                return true;
            }
        }
        false
    }

    search(
        0,
        &observations,
        &HashSet::new(),
        &HashSet::new(),
        &HashMap::new(),
        &HashMap::new(),
        &HashMap::new(),
        &HashMap::new(),
        &HashMap::new(),
        None,
        &[],
        scenario,
        state,
    )
}

fn validate_bounty_hunter(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    if card.info_parsed.contains_key("poet_variant") {
        let Some(target) = parse_current_bounty_hunter_target(card, state) else {
            return false;
        };
        return validate_current_bounty_hunter(card.position, target, scenario, state);
    }

    // Preserve every unmarked historical Poet/Bounty Hunter observation on
    // its legacy scalar predicate. Current schema and generation reachability
    // are marker-gated above.
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

const POET_CURRENT_VARIANT: &str = "public_current";

fn poet_has_exact_fields(
    info: &serde_json::Map<String, serde_json::Value>,
    provider_fields: &[&str],
) -> bool {
    info.len() == provider_fields.len() + 2
        && info.contains_key("poet_variant")
        && info.contains_key("copied_role")
        && provider_fields.iter().all(|field| info.contains_key(*field))
}

fn poet_position_value(
    value: Option<&serde_json::Value>,
    n_cards: u8,
) -> Option<u8> {
    value?
        .as_u64()
        .and_then(|position| u8::try_from(position).ok())
        .filter(|position| *position > 0 && *position <= n_cards)
}

fn poet_targets(
    info: &serde_json::Map<String, serde_json::Value>,
    n_cards: u8,
    minimum: usize,
    maximum: usize,
) -> Option<Vec<u8>> {
    let values = info.get("targets")?.as_array()?;
    if !(minimum..=maximum).contains(&values.len()) {
        return None;
    }
    let targets: Option<Vec<u8>> = values
        .iter()
        .map(|value| poet_position_value(Some(value), n_cards))
        .collect();
    let targets = targets?;
    (targets.iter().copied().collect::<HashSet<_>>().len() == targets.len())
        .then_some(targets)
}

fn poet_canonical_role<'a>(
    info: &'a serde_json::Map<String, serde_json::Value>,
    field: &str,
) -> Option<&'a str> {
    let role = info.get(field)?.as_str()?;
    get_card(role)
        .is_some_and(|card| card.name == role)
        .then_some(role)
}

/// Validate the exact bridge-owned payload for the audited public Gossip
/// selector. Provider-specific sentinels are accepted only when they have an
/// explicit exact current schema; partial current payloads always fail closed.
fn validate_current_poet_payload(card: &CardInfo, state: &GameState, copied_role: &str) -> bool {
    if card.position == 0 || card.position > state.n_cards {
        return false;
    }
    let info = &card.info_parsed;
    if info
        .get("poet_variant")
        .and_then(serde_json::Value::as_str)
        != Some(POET_CURRENT_VARIANT)
        || info
            .get("copied_role")
            .and_then(serde_json::Value::as_str)
            != Some(copied_role)
    {
        return false;
    }
    match copied_role {
        "Lover" => parse_current_lover_claim(
            card,
            CurrentPassivePayloadSource::Poet,
            state,
        )
        .is_some(),
        "Scout" => parse_current_scout_claim(
            info,
            CurrentPassivePayloadSource::Poet,
            state.n_cards,
        )
        .is_some(),
        "Oracle" => parse_current_oracle_claim(
            card,
            CurrentPassivePayloadSource::Poet,
            state.n_cards,
        )
        .is_some(),
        "Bounty Hunter" => parse_current_bounty_hunter_target(card, state).is_some(),
        "Medium" => {
            parse_current_medium_claim(card, CurrentPassivePayloadSource::Poet, state).is_some()
        }
        "Knitter" => {
            parse_current_knitter_claim(card, CurrentPassivePayloadSource::Poet, state).is_some()
        }
        "Hunter" => parse_current_hunter_distance(
            info,
            CurrentPassivePayloadSource::Poet,
            state.n_cards,
        )
        .is_some(),
        "Enlightened" => parse_current_enlightened_claim(
            card,
            CurrentPassivePayloadSource::Poet,
            state,
        )
        .is_some(),
        "Empress" => parse_current_empress_targets(
            card,
            CurrentPassivePayloadSource::Poet,
            state,
        )
        .is_some(),
        "Bishop" => parse_current_bishop_claim(
            card,
            CurrentPassivePayloadSource::Poet,
            state,
        )
        .is_some(),
        "Gemcrafter" => parse_current_gemcrafter_target(
            card,
            CurrentPassivePayloadSource::Poet,
            state,
        )
        .is_some(),
        "Bard" => parse_current_bard_claim(
            card,
            CurrentPassivePayloadSource::Poet,
            state,
        )
        .is_some(),
        _ => false,
    }
}

fn validate_poet(card: &CardInfo, scenario: &Scenario, state: &GameState) -> bool {
    let current_build = match card.info_parsed.get("poet_variant") {
        None => false,
        Some(serde_json::Value::String(variant)) if variant == POET_CURRENT_VARIANT => true,
        Some(_) => return false,
    };
    let copied_role = match info_str(&card.info_parsed, "copied_role") {
        Some(s) => s,
        None => return !current_build,
    };
    if current_build && !validate_current_poet_payload(card, state, copied_role) {
        return false;
    }
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

fn is_ordered_current_druid(card: &CardInfo) -> bool {
    normalize_role(&card.apparent_role) == "druid"
        && card
            .info_parsed
            .get(DRUID_CURRENT_VARIANT_FIELD)
            .and_then(serde_json::Value::as_str)
            == Some(POET_CURRENT_VARIANT)
        && card
            .info_parsed
            .get("callback_ledger_variant")
            .and_then(serde_json::Value::as_str)
            == Some("ordered_callbacks_v1")
}

fn is_ordered_current_jester(card: &CardInfo) -> bool {
    normalize_role(&card.apparent_role) == "jester"
        && card
            .info_parsed
            .get(JESTER_CURRENT_VARIANT_FIELD)
            .and_then(serde_json::Value::as_str)
            == Some(POET_CURRENT_VARIANT)
        && card
            .info_parsed
            .get("callback_ledger_variant")
            .and_then(serde_json::Value::as_str)
            == Some("ordered_callbacks_v1")
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentRamblerBoundary {
    Final,
    SettledRevealCount(usize),
}

impl CurrentRamblerBoundary {
    fn druid_boundary(self) -> CurrentDruidObservationBoundary {
        match self {
            Self::Final => CurrentDruidObservationBoundary::FinalCompatibility,
            Self::SettledRevealCount(count) => {
                CurrentDruidObservationBoundary::SettledRevealCount(count)
            }
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct CurrentRamblerConstraint {
    source: u8,
    boundary: CurrentRamblerBoundary,
    invocation: Option<(u8, usize, usize)>,
    required: u8,
    forbidden: u8,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum CurrentRamblerIdentity {
    Existing,
    RawBluff,
    AnonymousNatural,
}

fn push_current_rambler_constraint(
    constraints: &mut Vec<CurrentRamblerConstraint>,
    source: u8,
    boundary: CurrentRamblerBoundary,
    required: u8,
    forbidden: u8,
    invocation: Option<(u8, usize, usize)>,
) {
    if let Some(existing) = constraints.iter_mut().find(|constraint| {
        constraint.source == source
            && constraint.boundary == boundary
            && constraint.invocation == invocation
    }) {
        existing.required |= required;
        existing.forbidden |= forbidden;
    } else {
        constraints.push(CurrentRamblerConstraint {
            source,
            boundary,
            invocation,
            required,
            forbidden,
        });
    }
}

fn current_role_has_raw_bluff_selector(role: &str) -> bool {
    matches!(
        normalize_role(role).as_str(),
        "drunk" | "doppelganger" | "mutant"
    ) || get_card(role).is_some_and(|card| {
        matches!(card.faction, Faction::Minion | Faction::Demon)
    })
}

fn current_pre_writer_raw_holder_possible(
    position: u8,
    previous_role: &str,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    is_runtime_evil_at(position, scenario, state)
        || scenario.drunk_position == Some(position)
        || scenario.doppelganger_position == Some(position)
        || current_role_has_raw_bluff_selector(previous_role)
}

fn current_pre_writer_raw_confessor_unresolved(
    position: u8,
    previous_role: &str,
    had_raw_holder: bool,
    stale_raw_label_preserved: bool,
    selected_register_as: &HashMap<u8, String>,
    selected_raw_bluffs: &HashMap<u8, String>,
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    if !had_raw_holder {
        return false;
    }

    // Spy's registerAs and raw bluff share one cached Villager record. Other
    // represented raw selectors deliberately retain the model's any-canonical
    // label policy. A joined stale label can exclude historical Confessor only
    // when the intervening writer has a null selector; a new selector may have
    // overwritten bluffRole after Confessor installed its physical status.
    let cached_spy = stable_evil_origin_role_at(position, scenario, state)
        .is_some_and(|role| roles_equal(role, "Spy"))
        || roles_equal(previous_role, "Spy");
    if cached_spy && !current_medium_spy_register_as_label_allowed("Confessor", state) {
        return false;
    }

    let registered = cached_spy
        .then(|| selected_register_as.get(&position))
        .flatten();
    let raw = stale_raw_label_preserved
        .then(|| selected_raw_bluffs.get(&position))
        .flatten();
    let exact_historical_label = match (registered, raw) {
        (Some(registered), Some(raw)) if !roles_equal(registered, raw) => return true,
        (Some(registered), _) => Some(registered),
        (None, Some(raw)) => Some(raw),
        (None, None) => None,
    };
    exact_historical_label
        .map(|role| roles_equal(role, "Confessor"))
        .unwrap_or(true)
}

fn current_rambler_speaker_matcher_at_with_hidden_labels(
    speaker: u8,
    boundary: CurrentRamblerBoundary,
    timeline: &BakerSpyTimeline,
    selected_register_as: &HashMap<u8, String>,
    selected_raw_bluffs: &HashMap<u8, String>,
    scenario: &Scenario,
    state: &GameState,
) -> Option<u8> {
    let start_role = current_data_role_at_druid_observation(
        speaker,
        CurrentDruidObservationBoundary::SettledRevealCount(0),
        timeline,
        scenario,
        state,
    );
    let current_role = current_data_role_at_druid_observation(
        speaker,
        boundary.druid_boundary(),
        timeline,
        scenario,
        state,
    );
    if current_role
        .as_deref()
        .is_some_and(|role| normalize_role(role) == "unknown")
    {
        return None;
    }
    if start_role
        .as_deref()
        .is_some_and(|role| normalize_role(role) == "unknown")
    {
        return None;
    }

    // Both exact Shaman copied-Confessor endpoints receive the physical
    // AppearTruthfull status during their delayed internal Reveal. A later
    // Baker write uses InitWithNoReset and therefore preserves it.
    if shaman_copied_confessor_status_at(speaker, scenario) {
        return Some(RAMBLER_MATCHES_TRUTHFUL);
    }

    // Confessor's Init writes AppearTruthfull onto the physical Character and
    // InitWithNoReset preserves it. Scenario has no general appearance-status
    // history (nor the exact-resistance branch), so a presentation/current-data
    // mismatch or a later loss/gain of Confessor data cannot be reconstructed
    // from role names. Fail closed instead of inventing a temporal appearance
    // transition. No shipped role currently supplies a represented
    // AppearLying status.
    let start_confessor = start_role
        .as_deref()
        .is_some_and(|role| roles_equal(role, "Confessor"));
    let current_confessor = current_role
        .as_deref()
        .is_some_and(|role| roles_equal(role, "Confessor"));
    let shaman_may_preserve_confessor = scenario.shaman_trace.as_ref().is_some_and(|trace| {
        trace.target_position == speaker
            && trace
                .target_previous_roles
                .iter()
                .any(|role| roles_equal(role, "Confessor"))
    });
    let twin_may_misproject_confessor = scenario.twin_trace.as_ref().is_some_and(|trace| {
        matches!(
            &trace.outcome,
            crate::types::TwinStartOutcome::Swap {
                neighbor_position,
                neighbor_pre_swap_role,
                ..
            } if roles_equal(neighbor_pre_swap_role, "Confessor")
                && (trace.actor_position == speaker || *neighbor_position == speaker)
        )
    });
    let baker_may_preserve_confessor = !current_confessor
        && baker_history_supports_pre_day_role(scenario, state, speaker, "Confessor");
    let baker_may_preserve_raw_confessor = timeline.contains_position(speaker)
        && current_pre_writer_raw_confessor_unresolved(
            speaker,
            "Spy",
            true,
            true,
            selected_register_as,
            selected_raw_bluffs,
            scenario,
            state,
        );
    let twin_may_preserve_raw_confessor = scenario.twin_trace.as_ref().is_some_and(|trace| {
        matches!(
            &trace.outcome,
            crate::types::TwinStartOutcome::Swap {
                neighbor_position,
                neighbor_pre_swap_role,
                ..
            } if trace.actor_position == speaker
                && current_pre_writer_raw_confessor_unresolved(
                    speaker,
                    "Twin Minion",
                    true,
                    !current_role_has_raw_bluff_selector(neighbor_pre_swap_role),
                    selected_register_as,
                    selected_raw_bluffs,
                    scenario,
                    state,
                )
                || *neighbor_position == speaker
                    && current_pre_writer_raw_confessor_unresolved(
                        speaker,
                        neighbor_pre_swap_role,
                        current_pre_writer_raw_holder_possible(
                            speaker,
                            neighbor_pre_swap_role,
                            scenario,
                            state,
                        ),
                        false,
                        selected_register_as,
                        selected_raw_bluffs,
                        scenario,
                        state,
                    )
        )
    });
    let shaman_may_preserve_raw_confessor =
        scenario.shaman_trace.as_ref().is_some_and(|trace| {
            trace.target_position == speaker
                && trace.target_previous_roles.iter().any(|role| {
                    current_pre_writer_raw_confessor_unresolved(
                        speaker,
                        role,
                        current_pre_writer_raw_holder_possible(
                            speaker,
                            role,
                            scenario,
                            state,
                        ),
                        !current_role_has_raw_bluff_selector(&trace.copied_role),
                        selected_register_as,
                        selected_raw_bluffs,
                        scenario,
                        state,
                    )
                })
        });
    if shaman_may_preserve_confessor
        || twin_may_misproject_confessor
        || baker_may_preserve_confessor
        || baker_may_preserve_raw_confessor
        || twin_may_preserve_raw_confessor
        || shaman_may_preserve_raw_confessor
    {
        return None;
    }
    let apparent_confessor = state
        .card_at(speaker)
        .is_some_and(|card| roles_equal(&card.apparent_role, "Confessor"));
    if start_confessor != current_confessor || apparent_confessor != current_confessor {
        return None;
    }

    let truthful_appearance = current_confessor
        || truth_status(speaker, scenario, state) == TruthStatus::Truthful;
    Some(if truthful_appearance {
        RAMBLER_MATCHES_TRUTHFUL
    } else {
        RAMBLER_MATCHES_LYING
    })
}

#[cfg(test)]
fn current_rambler_speaker_matcher_at(
    speaker: u8,
    boundary: CurrentRamblerBoundary,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<u8> {
    current_rambler_speaker_matcher_at_with_hidden_labels(
        speaker,
        boundary,
        timeline,
        &HashMap::new(),
        &HashMap::new(),
        scenario,
        state,
    )
}

fn current_rambler_installed_matchers(
    source: u8,
    identity: CurrentRamblerIdentity,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> Option<u8> {
    // Rambler installs its closure during the initial AfterRoundStart pass.
    // The closure survives source death, reveal, and Init/InitWithNoReset, so
    // its captured truthful/lying mode is fixed here and must not be rebuilt
    // from the source's later role/bluff surface at each Druid activation.
    let start_boundary = CurrentDruidObservationBoundary::SettledRevealCount(0);
    let current_role = current_data_role_at_druid_observation(
        source,
        start_boundary,
        timeline,
        scenario,
        state,
    );
    if current_role
        .as_deref()
        .is_some_and(|role| normalize_role(role) == "unknown")
    {
        return None;
    }
    let raw_holder = current_rambler_start_raw_bluff_holder(source, timeline, scenario, state);
    let real_rambler = match identity {
        CurrentRamblerIdentity::AnonymousNatural => {
            if current_role.is_some() {
                return None;
            }
            true
        }
        CurrentRamblerIdentity::Existing | CurrentRamblerIdentity::RawBluff => current_role
            .as_deref()
            .is_some_and(|role| roles_equal(role, "Rambler")),
    };
    let raw_rambler = identity == CurrentRamblerIdentity::RawBluff
        && raw_holder != CurrentMediumRawBluffHolder::Impossible;
    let truth = truth_status(source, scenario, state);
    let runtime_evil = is_runtime_evil_at(source, scenario, state);
    let mut matchers = 0;
    if real_rambler {
        let real_act = truth == TruthStatus::Truthful
            || runtime_evil && raw_holder != CurrentMediumRawBluffHolder::Impossible;
        add_rambler_matcher(&mut matchers, real_act);
    }
    if raw_rambler {
        add_rambler_matcher(&mut matchers, truth == TruthStatus::Truthful);
    }
    Some(matchers)
}

fn current_rambler_start_raw_bluff_holder(
    source: u8,
    timeline: &BakerSpyTimeline,
    scenario: &Scenario,
    state: &GameState,
) -> CurrentMediumRawBluffHolder {
    if timeline.contains_position(source) {
        // A stable Spy acquired its ordinary Evil bluff during initial Reveal,
        // before any player-triggered Baker conversion represented by this
        // timeline. The later synchronous clear cannot uninstall a Rambler
        // closure that this bluff already installed at AfterRoundStart.
        CurrentMediumRawBluffHolder::Proven
    } else {
        current_medium_raw_bluff_holder_at(source, source, timeline, scenario, state)
    }
}

fn current_rambler_constraints_for_timeline(
    timeline: &BakerSpyTimeline,
    selected_register_as: &HashMap<u8, String>,
    selected_raw_bluffs: &HashMap<u8, String>,
    jester_callbacks: &[CurrentJesterResolvedCallback],
    scenario: &Scenario,
    state: &GameState,
) -> Option<Vec<CurrentRamblerConstraint>> {
    let mut constraints = Vec::new();
    let ordered_druid_positions: HashSet<u8> = state
        .cards
        .iter()
        .filter(|card| is_ordered_current_druid(card))
        .map(|card| card.position)
        .collect();
    let ordered_jester_positions: HashSet<u8> = state
        .cards
        .iter()
        .filter(|card| is_ordered_current_jester(card))
        .map(|card| card.position)
        .collect();

    let mut public_interruptions = Vec::new();
    for observation in &state.rambler_shut_up_observations {
        if observation.speaker_position == 0
            || observation.speaker_position > state.n_cards
            || observation.shut_up_target == 0
            || observation.shut_up_target > state.n_cards
        {
            return None;
        }
        if !ordered_druid_positions.contains(&observation.speaker_position)
            && !ordered_jester_positions.contains(&observation.speaker_position)
        {
            let pair = (observation.speaker_position, observation.shut_up_target);
            if !public_interruptions.contains(&pair) {
                public_interruptions.push(pair);
            }
        }
    }
    for card in &state.cards {
        if ordered_druid_positions.contains(&card.position)
            || ordered_jester_positions.contains(&card.position)
        {
            continue;
        }
        if let Some(value) = card.info_parsed.get("shut_up_target") {
            let target = value
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .filter(|target| *target > 0 && *target <= state.n_cards)?;
            let pair = (card.position, target);
            if !public_interruptions.contains(&pair) {
                public_interruptions.push(pair);
            }
        }
    }
    for (speaker, source) in public_interruptions {
        if !adjacent_positions(speaker, state.n_cards).contains(&source) {
            return None;
        }
        let matcher = current_rambler_speaker_matcher_at_with_hidden_labels(
            speaker,
            CurrentRamblerBoundary::Final,
            timeline,
            selected_register_as,
            selected_raw_bluffs,
            scenario,
            state,
        )?;
        push_current_rambler_constraint(
            &mut constraints,
            source,
            CurrentRamblerBoundary::Final,
            matcher,
            0,
            None,
        );
    }

    for card in &state.cards {
        if is_ordered_current_druid(card) {
            let CurrentDruidPayload::Ledger(events) = parse_current_druid_payload(card, state)?
            else {
                return None;
            };
            for event in events {
                let boundary = CurrentRamblerBoundary::SettledRevealCount(
                    event.settled_reveal_count,
                );
                if !timeline.supports_settled_reveal_count(
                    event.settled_reveal_count,
                    state,
                ) {
                    return None;
                }
                let matcher = current_rambler_speaker_matcher_at_with_hidden_labels(
                    card.position,
                    boundary,
                    timeline,
                    selected_register_as,
                    selected_raw_bluffs,
                    scenario,
                    state,
                )?;
                match event.kind {
                    CurrentDruidCallbackKind::RamblerInterruption { target } => {
                        if !adjacent_positions(card.position, state.n_cards).contains(&target) {
                            return None;
                        }
                        push_current_rambler_constraint(
                            &mut constraints,
                            target,
                            boundary,
                            matcher,
                            0,
                            None,
                        );
                    }
                    CurrentDruidCallbackKind::Result(_)
                    | CurrentDruidCallbackKind::OpaqueReal => {
                        for source in adjacent_positions(card.position, state.n_cards) {
                            push_current_rambler_constraint(
                                &mut constraints,
                                source,
                                boundary,
                                0,
                                matcher,
                                None,
                            );
                        }
                    }
                }
            }
        } else if is_ordered_current_jester(card) {
            continue;
        } else if card_has_normal_clue(card, true) {
            let matcher = current_rambler_speaker_matcher_at_with_hidden_labels(
                card.position,
                CurrentRamblerBoundary::Final,
                timeline,
                selected_register_as,
                selected_raw_bluffs,
                scenario,
                state,
            )?;
            for source in adjacent_positions(card.position, state.n_cards) {
                push_current_rambler_constraint(
                    &mut constraints,
                    source,
                    CurrentRamblerBoundary::Final,
                    0,
                    matcher,
                    None,
                );
            }
        }
    }

    // Unlike Druid, Jester's two callbacks can independently survive or be
    // replaced. Preserve the resolved real/raw truth surface and an invocation
    // key per callback so a sibling negative cannot be collapsed into the
    // interruption fact for a different native result object. Rambler source
    // identity remains shared by the outer joint search.
    for callback in jester_callbacks {
        if !timeline.supports_settled_reveal_count(
            match callback.boundary {
                CurrentRamblerBoundary::SettledRevealCount(count) => count,
                CurrentRamblerBoundary::Final => return None,
            },
            state,
        ) {
            return None;
        }
        let matcher = if callback.truth == TruthStatus::Truthful {
            RAMBLER_MATCHES_TRUTHFUL
        } else {
            RAMBLER_MATCHES_LYING
        };
        let invocation = Some((
            callback.actor,
            callback.activation_id,
            callback.callback_index,
        ));
        if let Some(target) = callback.interruption_target {
            if !adjacent_positions(callback.actor, state.n_cards).contains(&target) {
                return None;
            }
            push_current_rambler_constraint(
                &mut constraints,
                target,
                callback.boundary,
                matcher,
                0,
                invocation,
            );
        } else {
            for source in adjacent_positions(callback.actor, state.n_cards) {
                push_current_rambler_constraint(
                    &mut constraints,
                    source,
                    callback.boundary,
                    0,
                    matcher,
                    invocation,
                );
            }
        }
    }
    Some(constraints)
}

fn current_rambler_timeline_jointly_possible(
    timeline: &BakerSpyTimeline,
    selected_register_as: &HashMap<u8, String>,
    selected_raw_bluffs: &HashMap<u8, String>,
    forbidden_raw_bluffs: &HashMap<u8, HashSet<String>>,
    selected_anonymous_outcasts: &HashMap<u8, String>,
    jester_callbacks: &[CurrentJesterResolvedCallback],
    scenario: &Scenario,
    state: &GameState,
) -> bool {
    let Some(constraints) = current_rambler_constraints_for_timeline(
        timeline,
        selected_register_as,
        selected_raw_bluffs,
        jester_callbacks,
        scenario,
        state,
    )
    else {
        return false;
    };
    let mut by_source: HashMap<u8, Vec<CurrentRamblerConstraint>> = HashMap::new();
    for constraint in constraints {
        by_source.entry(constraint.source).or_default().push(constraint);
    }
    let deck_has_rambler = state
        .deck
        .villagers
        .iter()
        .chain(state.deck.outcasts.iter())
        .any(|role| normalize_role(role) == "rambler");
    let mut alternatives = Vec::new();
    for (&source, constraints) in &by_source {
        let start_role = current_data_role_at_druid_observation(
            source,
            CurrentDruidObservationBoundary::SettledRevealCount(0),
            timeline,
            scenario,
            state,
        );
        let start_raw_holder =
            current_rambler_start_raw_bluff_holder(source, timeline, scenario, state);
        // Spy stores registerAs and its raw bluff as the same cached Villager
        // CharacterData.  Medium/other hidden-surface evidence therefore pins
        // the identity of an already-installed Rambler closure too; these
        // cannot be solved as independent existential labels.
        let cached_spy = start_role
            .as_deref()
            .is_some_and(|role| roles_equal(role, "Spy"));
        let cached_registered_rambler = cached_spy
            && selected_register_as
                .get(&source)
                .is_some_and(|role| roles_equal(role, "Rambler"));
        let apparent_rambler = state
            .card_at(source)
            .is_some_and(|card| normalize_role(&card.apparent_role) == "rambler");
        let known_raw_rambler = cached_registered_rambler
            || apparent_rambler
                && (!start_role
                    .as_deref()
                    .is_some_and(|role| roles_equal(role, "Rambler"))
                    || start_raw_holder != CurrentMediumRawBluffHolder::Impossible);
        let raw_surface_possible = deck_has_rambler
            && start_raw_holder != CurrentMediumRawBluffHolder::Impossible
            && (apparent_rambler
                || state.card_at(source).is_none()
                || timeline.contains_position(source)
                || cached_registered_rambler
                || selected_raw_bluffs
                    .get(&source)
                    .is_some_and(|role| roles_equal(role, "Rambler")));
        let natural_possible = crate::scenario::scenario_allows_anonymous_natural_outcast_role_at(
            source, "Rambler", scenario, state,
        );
        let mut source_alternatives = Vec::new();
        for identity in [
            CurrentRamblerIdentity::Existing,
            CurrentRamblerIdentity::RawBluff,
            CurrentRamblerIdentity::AnonymousNatural,
        ] {
            if identity == CurrentRamblerIdentity::Existing && known_raw_rambler
                || identity == CurrentRamblerIdentity::RawBluff && !raw_surface_possible
                || identity == CurrentRamblerIdentity::AnonymousNatural && !natural_possible
            {
                continue;
            }
            if identity == CurrentRamblerIdentity::RawBluff {
                if selected_raw_bluffs
                    .get(&source)
                    .is_some_and(|role| !roles_equal(role, "Rambler"))
                    || cached_spy
                        && selected_register_as
                            .get(&source)
                            .is_some_and(|role| !roles_equal(role, "Rambler"))
                    || forbidden_raw_bluffs
                        .get(&source)
                        .is_some_and(|roles| roles.contains(&normalize_role("Rambler")))
                {
                    continue;
                }
            } else if selected_raw_bluffs
                .get(&source)
                .is_some_and(|role| roles_equal(role, "Rambler"))
                || cached_registered_rambler
            {
                continue;
            }
            if identity == CurrentRamblerIdentity::AnonymousNatural {
                if selected_anonymous_outcasts
                    .get(&source)
                    .is_some_and(|role| !roles_equal(role, "Rambler"))
                {
                    continue;
                }
            } else if selected_anonymous_outcasts
                .get(&source)
                .is_some_and(|role| roles_equal(role, "Rambler"))
            {
                continue;
            }
            let supported = current_rambler_installed_matchers(
                source, identity, timeline, scenario, state,
            )
            .is_some_and(|matchers| {
                constraints.iter().all(|constraint| {
                    matchers & constraint.required == constraint.required
                        && matchers & constraint.forbidden == 0
                })
            });
            if supported {
                source_alternatives.push(identity);
            }
        }
        if source_alternatives.is_empty() {
            return false;
        }
        alternatives.push((source, source_alternatives));
    }
    alternatives.sort_unstable_by_key(|(source, _)| *source);

    fn search(
        index: usize,
        alternatives: &[(u8, Vec<CurrentRamblerIdentity>)],
        selected_natural: &mut HashSet<u8>,
        forbidden_natural: &mut HashSet<u8>,
        scenario: &Scenario,
        state: &GameState,
    ) -> bool {
        if index == alternatives.len() {
            return crate::scenario::scenario_allows_anonymous_natural_outcast_role_assignments(
                selected_natural,
                "Rambler",
                forbidden_natural,
                scenario,
                state,
            );
        }
        let (source, identities) = &alternatives[index];
        for identity in identities {
            let natural = *identity == CurrentRamblerIdentity::AnonymousNatural;
            let inserted_natural = natural && selected_natural.insert(*source);
            let inserted_forbidden = !natural
                && identities.contains(&CurrentRamblerIdentity::AnonymousNatural)
                && forbidden_natural.insert(*source);
            if natural && forbidden_natural.contains(source) {
                if inserted_natural {
                    selected_natural.remove(source);
                }
                continue;
            }
            if search(
                index + 1,
                alternatives,
                selected_natural,
                forbidden_natural,
                scenario,
                state,
            ) {
                return true;
            }
            if inserted_natural {
                selected_natural.remove(source);
            }
            if inserted_forbidden {
                forbidden_natural.remove(source);
            }
        }
        false
    }

    let mut selected_natural = selected_anonymous_outcasts
        .iter()
        .filter_map(|(&position, role)| roles_equal(role, "Rambler").then_some(position))
        .collect::<HashSet<_>>();
    let mut forbidden_natural = selected_anonymous_outcasts
        .iter()
        .filter_map(|(&position, role)| (!roles_equal(role, "Rambler")).then_some(position))
        .collect::<HashSet<_>>();
    search(
        0,
        &alternatives,
        &mut selected_natural,
        &mut forbidden_natural,
        scenario,
        state,
    )
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
    if is_ordered_current_jester(card) {
        return card
            .info_parsed
            .get("callback_events")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|events| {
                events.iter().any(|event| {
                    event.get("event_kind").and_then(serde_json::Value::as_str)
                        == Some("jester_result")
                })
            });
    }
    if role == "jester"
        && card
            .info_parsed
            .get(JESTER_CURRENT_VARIANT_FIELD)
            .and_then(serde_json::Value::as_str)
            == Some(POET_CURRENT_VARIANT)
    {
        // The exact marker-only current shell is an unused active ability, not
        // evidence that a normal result escaped Rambler replacement.
        return false;
    }
    if role == "druid"
        && card
            .info_parsed
            .get(DRUID_CURRENT_VARIANT_FIELD)
            .and_then(serde_json::Value::as_str)
            == Some(POET_CURRENT_VARIANT)
        && card
            .info_parsed
            .get("callback_ledger_variant")
            .and_then(serde_json::Value::as_str)
            == Some("ordered_callbacks_v1")
    {
        return card
            .info_parsed
            .get("callback_events")
            .and_then(serde_json::Value::as_array)
            .is_some_and(|events| {
                events.iter().any(|event| {
                    event
                        .get("event_kind")
                        .and_then(serde_json::Value::as_str)
                        == Some("druid_result")
                })
            });
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
    if current_rules
        && state
            .cards
            .iter()
            .any(|card| is_ordered_current_druid(card) || is_ordered_current_jester(card))
    {
        // Ordered Druid/Jester events need their settled Baker/Spy boundary. Their
        // positive and negative Rambler facts, together with every unrelated
        // card's current Rambler facts, are evaluated once at the base of the
        // hidden-surface DFS. Keep this early layer to public shape/geometry
        // only so it cannot choose a different final-state identity world.
        for observation in &state.rambler_shut_up_observations {
            if observation.speaker_position == 0
                || observation.speaker_position > state.n_cards
                || observation.shut_up_target == 0
                || observation.shut_up_target > state.n_cards
                || !adjacent_positions(observation.speaker_position, state.n_cards)
                    .contains(&observation.shut_up_target)
            {
                return false;
            }
        }
        for card in &state.cards {
            let Some(value) = card.info_parsed.get("shut_up_target") else {
                continue;
            };
            let Some(source) = value
                .as_u64()
                .and_then(|value| u8::try_from(value).ok())
                .filter(|source| *source > 0 && *source <= state.n_cards)
            else {
                return false;
            };
            if card.position == 0
                || card.position > state.n_cards
                || !adjacent_positions(card.position, state.n_cards).contains(&source)
            {
                return false;
            }
        }
        return true;
    }
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
        let slayer_is_puppet = scenario.puppet_position == Some(slayer_pos)
            || slayer_evil_role
                .is_some_and(|role| normalize_role(role) == "puppet");
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
                let role_matches = current_data_role_at(target_pos, scenario, state)
                    .is_some_and(|actual| {
                        normalize_role(&actual) == normalize_role(revealed_role)
                    });
                // Trace-less scenarios retain the narrow two-endpoint
                // compatibility waiver. Exact TwinTrace current data is
                // authoritative and cannot enter that waiver.
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
            // trace-less exception: Wretch data can move to the Evil Twin,
            // leaving current Twin/base registration on the Good Wretch body.
            let modeled_good_wretch_can_receive_twin = !target_is_physically_evil
                && current_data_role_at(target_pos, scenario, state)
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

fn is_exact_twin_shaman_villager_role(role: &str) -> bool {
    matches!(normalize_role(role).as_str(), "scout" | "witness")
}

fn is_exact_twin_shaman_claim(scenario: &Scenario) -> bool {
    !scenario.pre_twin_current_roles.is_empty()
        && scenario.twin_trace.is_some()
        && scenario.shaman_trace.is_some()
        && scenario.puppeteer_trace.is_none()
}

/// Validate the complete role-only claim emitted by the exact Twin -> Shaman
/// kernel and return its post-Twin map.
///
/// Generic Shaman worlds cannot use a stable Evil seat as an endpoint. This
/// narrow claim proves the exception from first principles: one complete
/// pre-Twin map, one reachable Twin trace, and one Shaman trace enumerated from
/// that exact post-swap Villager pool. Forged, partial, or hazard-bearing maps
/// fail closed rather than leaking into the legacy validator rules.
fn exact_twin_shaman_post_twin_roles(
    scenario: &Scenario,
    state: &GameState,
) -> Option<HashMap<u8, String>> {
    let twin_trace = scenario.twin_trace.as_ref()?;
    let shaman_trace = scenario.shaman_trace.as_ref()?;
    let (Some(villager_count), Some(demon_count)) =
        (state.board_villager_count, state.board_demon_count)
    else {
        return None;
    };

    if state.n_cards == 0
        || state.board_count_provenance != BoardCountProvenance::TrustedPreStart
        || state.board_outcast_count != Some(0)
        || state.board_minion_count != Some(2)
        || !state.deck.outcasts.is_empty()
        || state.deck.minions.len() != 2
        || state
            .deck
            .minions
            .iter()
            .filter(|role| normalize_role(role) == "twinminion")
            .count()
            != 1
        || state
            .deck
            .minions
            .iter()
            .filter(|role| normalize_role(role) == "shaman")
            .count()
            != 1
        || state.deck.demons.iter().any(|role| normalize_role(role) != "lilis")
        || state
            .deck
            .villagers
            .iter()
            .any(|role| !is_exact_twin_shaman_villager_role(role))
        || state.pd_corruption_target.is_some()
        || villager_count < 2
        || demon_count as usize != state.deck.demons.len()
        || villager_count as usize > state.deck.villagers.len()
        || villager_count as usize + demon_count as usize + 2 != state.n_cards as usize
        || scenario.puppet_position.is_some()
        || scenario.doppelganger_position.is_some()
        || scenario.drunk_position.is_some()
        || scenario.chancellor_trace.is_some()
        || scenario.chancellor_conversion.is_some()
        || scenario.puppeteer_trace.is_some()
        || !scenario.corrupted.is_empty()
        || scenario.pd_corrupted.is_some()
        || !scenario.alchemist_cures.is_empty()
        || scenario.pre_twin_current_roles.len() != state.n_cards as usize
        || state.cards.iter().any(|card| normalize_role(&card.apparent_role) == "baker")
        || state
            .executed_current_roles
            .values()
            .chain(state.executed_good_roles.values())
            .any(|role| normalize_role(role) == "baker")
    {
        return None;
    }

    let twin_count = scenario
        .evil_positions
        .values()
        .filter(|role| normalize_role(role) == "twinminion")
        .count();
    let shaman_count = scenario
        .evil_positions
        .values()
        .filter(|role| normalize_role(role) == "shaman")
        .count();
    let placed_demons = scenario
        .evil_positions
        .values()
        .filter(|role| normalize_role(role) == "lilis")
        .count();
    if twin_count != 1
        || shaman_count != 1
        || placed_demons != demon_count as usize
        || scenario.evil_positions.len() != 2 + demon_count as usize
        || scenario.evil_positions.len() != state.n_evil as usize
        || scenario
            .evil_positions
            .keys()
            .any(|position| *position == 0 || *position > state.n_cards)
        || scenario.evil_positions.values().any(|role| {
            !matches!(
                normalize_role(role).as_str(),
                "twinminion" | "shaman" | "lilis"
            )
        })
        || state.executed_evil_roles.iter().any(|(position, role)| {
            !scenario.evil_positions.get(position).is_some_and(|placed| {
                normalize_role(placed) == normalize_role(role)
            })
        })
        || state
            .confirmed_evil
            .iter()
            .any(|position| !scenario.evil_positions.contains_key(position))
        || state
            .confirmed_good
            .iter()
            .any(|position| scenario.evil_positions.contains_key(position))
        || state
            .executed_good_roles
            .keys()
            .chain(state.executed_good_corrupted.keys())
            .any(|position| scenario.evil_positions.contains_key(position))
    {
        return None;
    }

    if (1..=state.n_cards).any(|position| {
        !scenario.pre_twin_current_roles.contains_key(&position)
            || scenario
                .evil_positions
                .get(&position)
                .is_some_and(|stable_role| {
                    !scenario
                        .pre_twin_current_roles
                        .get(&position)
                        .is_some_and(|current_role| {
                            normalize_role(current_role) == normalize_role(stable_role)
                        })
                })
    }) {
        return None;
    }

    let mut remaining_villagers: HashMap<String, usize> = HashMap::new();
    for role in &state.deck.villagers {
        *remaining_villagers.entry(normalize_role(role)).or_insert(0) += 1;
    }
    let mut represented_villagers = 0usize;
    for position in 1..=state.n_cards {
        if scenario.evil_positions.contains_key(&position) {
            continue;
        }
        let role = scenario.pre_twin_current_roles.get(&position)?;
        if !is_exact_twin_shaman_villager_role(role) {
            return None;
        }
        let remaining = remaining_villagers.get_mut(&normalize_role(role))?;
        if *remaining == 0 {
            return None;
        }
        *remaining -= 1;
        represented_villagers += 1;
    }
    if represented_villagers != villager_count as usize {
        return None;
    }

    let current_order: Vec<u8> = (1..=state.n_cards).rev().collect();
    if !enumerate_twin_traces(
        &scenario.pre_twin_current_roles,
        &current_order,
        &current_order,
    )
    .contains(twin_trace)
    {
        return None;
    }
    if distinct_swap_has_unsupported_public_action_evidence(state, twin_trace) {
        return None;
    }
    let post_twin_current_roles: Option<HashMap<u8, String>> = (1..=state.n_cards)
        .map(|position| {
            role_after_twin(
                position,
                &scenario.pre_twin_current_roles,
                twin_trace,
            )
            .map(|role| (position, role))
        })
        .collect();
    let post_twin_current_roles = post_twin_current_roles?;
    if !enumerate_shaman_traces(
        &post_twin_current_roles,
        &current_order,
        state.n_cards,
    )?
    .contains(shaman_trace)
        || scenario.messed_up_by_evil
            != HashSet::from([
                shaman_trace.source_position,
                shaman_trace.target_position,
            ])
    {
        return None;
    }

    let final_current_roles: Option<HashMap<u8, String>> = (1..=state.n_cards)
        .map(|position| {
            role_after_shaman(position, &post_twin_current_roles, shaman_trace)
                .map(|role| (position, role))
        })
        .collect();
    let final_current_roles = final_current_roles?;
    let mut unpredictable_card_positions: HashSet<u8> =
        scenario.evil_positions.keys().copied().collect();
    if let crate::types::TwinStartOutcome::Swap {
        neighbor_position,
        ..
    } = twin_trace.outcome
    {
        if neighbor_position != twin_trace.actor_position {
            unpredictable_card_positions.insert(neighbor_position);
        }
    }
    if state.cards.iter().any(|card| {
        !unpredictable_card_positions.contains(&card.position)
            && !final_current_roles.get(&card.position).is_some_and(|role| {
                normalize_role(role) == normalize_role(&card.apparent_role)
            })
    }) {
        return None;
    }
    Some(post_twin_current_roles)
}

fn validate_role_counts(scenario: &Scenario, state: &GameState) -> bool {
    match exact_twin_shaman_post_twin_roles(scenario, state) {
        Some(_) => return true,
        None if is_exact_twin_shaman_claim(scenario) => return false,
        None => {}
    }

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
            pre_twin_current_roles: HashMap::new(),
            puppeteer_trace: None,
        }
    }

    fn base_state(n_cards: u8, cards: Vec<CardInfo>) -> GameState {
        let mut s = GameState::default();
        s.n_cards = n_cards;
        s.cards = cards;
        s
    }

    fn current_oracle_text(payload: &serde_json::Value) -> String {
        if payload.get("no_minions").and_then(serde_json::Value::as_bool) == Some(true) {
            return "There are no minions".to_string();
        }
        let Some(targets) = payload.get("targets").and_then(serde_json::Value::as_array) else {
            return String::new();
        };
        let Some([first, second]) = targets
            .iter()
            .map(serde_json::Value::as_u64)
            .collect::<Option<Vec<_>>>()
            .and_then(|targets| <[u64; 2]>::try_from(targets).ok())
        else {
            return String::new();
        };
        let Some(role) = payload.get("minion_role").and_then(serde_json::Value::as_str) else {
            return String::new();
        };
        format!("#{first} or #{second} is a {role}")
    }

    fn current_poet(provider: &str, payload: serde_json::Value) -> CardInfo {
        let info_text = match provider {
            "Lover" => payload
                .get("evil_adjacent")
                .and_then(serde_json::Value::as_i64)
                .and_then(current_lover_claim_text)
                .map(str::to_string),
            "Oracle" => Some(current_oracle_text(&payload)),
            "Bounty Hunter" => payload
                .get("evil_position")
                .and_then(serde_json::Value::as_u64)
                .map(|target| format!("#{target}\nis Evil")),
            "Medium" => payload
                .get("good_position")
                .and_then(serde_json::Value::as_u64)
                .zip(payload.get("good_role").and_then(serde_json::Value::as_str))
                .map(|(target, role)| current_medium_text(target as u8, role)),
            "Knitter" => payload
                .get("evil_pairs")
                .and_then(serde_json::Value::as_i64)
                .and_then(current_knitter_claim_text),
            "Enlightened" => payload
                .get("direction")
                .and_then(serde_json::Value::as_str)
                .and_then(|direction| match direction {
                    "CW" => Some(Direction::CW),
                    "CCW" => Some(Direction::CCW),
                    "Equidistant" => Some(Direction::Equidistant),
                    _ => None,
                })
                .map(current_enlightened_claim_text)
                .map(str::to_string),
            "Empress" => payload
                .get("targets")
                .and_then(serde_json::Value::as_array)
                .and_then(|targets| {
                    targets
                        .iter()
                        .map(|target| {
                            target
                                .as_u64()
                                .and_then(|target| u8::try_from(target).ok())
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .and_then(|targets| current_empress_claim_text(&targets)),
            "Gemcrafter" => payload
                .get("good_position")
                .and_then(serde_json::Value::as_u64)
                .and_then(|target| u8::try_from(target).ok())
                .map(current_gemcrafter_claim_text),
            "Bishop" => payload
                .get("targets")
                .and_then(serde_json::Value::as_array)
                .and_then(|targets| {
                    targets
                        .iter()
                        .map(|target| {
                            target
                                .as_u64()
                                .and_then(|target| u8::try_from(target).ok())
                        })
                        .collect::<Option<Vec<_>>>()
                })
                .zip(
                    payload
                        .get("types")
                        .and_then(serde_json::Value::as_array)
                        .and_then(|types| {
                            types
                                .iter()
                                .map(|role_type| {
                                    role_type.as_str().and_then(BishopType::parse)
                                })
                                .collect::<Option<Vec<_>>>()
                        }),
                )
                .and_then(|(targets, types)| current_bishop_claim_text(&targets, &types)),
            "Bard" => payload
                .get("corruption_distance")
                .and_then(serde_json::Value::as_i64)
                .and_then(|distance| {
                    current_bard_claim_text(if distance == -1 { 0 } else { distance })
                }),
            _ => None,
        };
        let mut info = payload.as_object().unwrap().clone();
        info.insert(
            "poet_variant".to_string(),
            serde_json::Value::String("public_current".to_string()),
        );
        info.insert(
            "copied_role".to_string(),
            serde_json::Value::String(provider.to_string()),
        );
        let mut card = make_card(1, "Poet", serde_json::Value::Object(info));
        if let Some(info_text) = info_text {
            card.info_text = info_text;
        }
        card
    }

    fn current_lover(pos: u8, claimed: serde_json::Value) -> CardInfo {
        let info_text = claimed
            .as_i64()
            .and_then(current_lover_claim_text)
            .unwrap_or_default()
            .to_string();
        let mut card = make_card(
            pos,
            "Lover",
            json!({
                "lover_variant": "public_current",
                "evil_adjacent": claimed,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn current_bard(pos: u8, claimed: serde_json::Value) -> CardInfo {
        let info_text = claimed
            .as_i64()
            .and_then(|distance| {
                current_bard_claim_text(if distance == -1 { 0 } else { distance })
            })
            .unwrap_or_default();
        let mut card = make_card(
            pos,
            "Bard",
            json!({
                "bard_variant": "public_current",
                "corruption_distance": claimed,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn current_confessor(pos: u8, dizzy: serde_json::Value) -> CardInfo {
        let info_text = dizzy
            .as_bool()
            .map(current_confessor_claim_text)
            .unwrap_or_default()
            .to_string();
        let mut card = make_card(
            pos,
            "Confessor",
            json!({
                "confessor_variant": "public_current",
                "dizzy": dizzy,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn current_druid(
        pos: u8,
        targets: [u8; 3],
        found_outcast: Option<&str>,
    ) -> CardInfo {
        let mut card = make_card(
            pos,
            "Druid",
            json!({
                "targets": targets,
                "found_outcast": found_outcast,
                "druid_variant": "public_current",
            }),
        );
        card.info_text = current_druid_claim_text(&targets, found_outcast);
        card
    }

    #[allow(clippy::too_many_arguments)]
    fn current_druid_result_event(
        activation_id: usize,
        callback_index: usize,
        dispatch_path: &str,
        targets: [u8; 3],
        found_outcast: Option<&str>,
        settled_reveal_count: usize,
        reset_generation: usize,
        activation_evidence: &str,
    ) -> serde_json::Value {
        json!({
            "activation_id": activation_id,
            "callback_index": callback_index,
            "dispatch_path": dispatch_path,
            "event_kind": "druid_result",
            "targets": targets,
            "found_outcast": found_outcast,
            "text": current_druid_claim_text(&targets, found_outcast),
            "references": targets,
            "settled_reveal_count": settled_reveal_count,
            "reset_generation": reset_generation,
            "activation_evidence": activation_evidence,
        })
    }

    fn current_druid_history(pos: u8, observations: &[([u8; 3], Option<&str>, usize)]) -> CardInfo {
        let (targets, found_outcast, _) = observations.last().copied().unwrap();
        let history = observations
            .iter()
            .enumerate()
            .map(|(index, (targets, found_outcast, settled_reveal_count))| {
                current_druid_result_event(
                    index + 1,
                    0,
                    "either",
                    *targets,
                    *found_outcast,
                    *settled_reveal_count,
                    index,
                    if index == 0 {
                        "single_callback_suffix"
                    } else {
                        "session_reset_generation"
                    },
                )
            })
            .collect::<Vec<_>>();
        let mut card = make_card(
            pos,
            "Druid",
            json!({
                "targets": targets,
                "found_outcast": found_outcast,
                "druid_variant": "public_current",
                "callback_ledger_variant": "ordered_callbacks_v1",
                "callback_events": history,
            }),
        );
        card.info_text = current_druid_claim_text(&targets, found_outcast);
        card
    }

    fn current_druid_interruption_event(
        activation_id: usize,
        callback_index: usize,
        dispatch_path: &str,
        target: u8,
        settled_reveal_count: usize,
        reset_generation: usize,
        activation_evidence: &str,
    ) -> serde_json::Value {
        json!({
            "activation_id": activation_id,
            "callback_index": callback_index,
            "dispatch_path": dispatch_path,
            "event_kind": "rambler_interruption",
            "text": format!("#{target}\nshut up!"),
            "references": [target],
            "shut_up_target": target,
            "settled_reveal_count": settled_reveal_count,
            "reset_generation": reset_generation,
            "activation_evidence": activation_evidence,
        })
    }

    fn current_druid_ledger(pos: u8, events: Vec<serde_json::Value>) -> CardInfo {
        let latest = events.last().unwrap();
        let mut info = serde_json::Map::from_iter([
            (
                "druid_variant".to_string(),
                json!(POET_CURRENT_VARIANT),
            ),
            (
                "callback_ledger_variant".to_string(),
                json!("ordered_callbacks_v1"),
            ),
            ("callback_events".to_string(), json!(events)),
        ]);
        match latest["event_kind"].as_str().unwrap() {
            "druid_result" => {
                info.insert("targets".to_string(), latest["targets"].clone());
                info.insert(
                    "found_outcast".to_string(),
                    latest["found_outcast"].clone(),
                );
            }
            "rambler_interruption" => {
                info.insert(
                    "shut_up_target".to_string(),
                    latest["shut_up_target"].clone(),
                );
            }
            _ => panic!("opaque callback cannot be the latest public alias"),
        }
        let mut card = make_card(pos, "Druid", serde_json::Value::Object(info));
        card.info_text = latest["text"].as_str().unwrap().to_string();
        card
    }

    #[allow(clippy::too_many_arguments)]
    fn current_jester_result_event_with_references(
        activation_id: usize,
        callback_index: usize,
        dispatch_path: &str,
        targets: [u8; 3],
        references: [u8; 3],
        evil_count: i64,
        settled_reveal_count: usize,
        reset_generation: usize,
        activation_evidence: &str,
    ) -> serde_json::Value {
        json!({
            "activation_id": activation_id,
            "callback_index": callback_index,
            "dispatch_path": dispatch_path,
            "event_kind": "jester_result",
            "targets": targets,
            "evil_count": evil_count,
            "text": current_jester_claim_text(&references, evil_count).unwrap(),
            "references": references,
            "settled_reveal_count": settled_reveal_count,
            "reset_generation": reset_generation,
            "activation_evidence": activation_evidence,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn current_jester_result_event(
        activation_id: usize,
        callback_index: usize,
        dispatch_path: &str,
        targets: [u8; 3],
        evil_count: i64,
        settled_reveal_count: usize,
        reset_generation: usize,
        activation_evidence: &str,
    ) -> serde_json::Value {
        current_jester_result_event_with_references(
            activation_id,
            callback_index,
            dispatch_path,
            targets,
            targets,
            evil_count,
            settled_reveal_count,
            reset_generation,
            activation_evidence,
        )
    }

    fn current_jester_ledger(pos: u8, events: Vec<serde_json::Value>) -> CardInfo {
        let latest = events.last().unwrap();
        let mut info = serde_json::Map::from_iter([
            (
                JESTER_CURRENT_VARIANT_FIELD.to_string(),
                json!(POET_CURRENT_VARIANT),
            ),
            (
                "callback_ledger_variant".to_string(),
                json!("ordered_callbacks_v1"),
            ),
            ("callback_events".to_string(), json!(events)),
        ]);
        match latest["event_kind"].as_str().unwrap() {
            "jester_result" => {
                info.insert("targets".to_string(), latest["targets"].clone());
                info.insert("evil_count".to_string(), latest["evil_count"].clone());
            }
            "rambler_interruption" => {
                info.insert(
                    "shut_up_target".to_string(),
                    latest["shut_up_target"].clone(),
                );
            }
            _ => panic!("opaque callback cannot be the latest public alias"),
        }
        let mut card = make_card(pos, "Jester", serde_json::Value::Object(info));
        card.info_text = latest["text"].as_str().unwrap().to_string();
        card
    }

    fn current_druid_rambler_timeline_state(druid: CardInfo) -> (GameState, Scenario) {
        let interruptions = druid.info_parsed["callback_events"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|event| {
                (event["event_kind"] == "rambler_interruption").then(|| {
                    crate::types::RamblerShutUpObservation {
                        speaker_position: druid.position,
                        shut_up_target: event["shut_up_target"].as_u64().unwrap() as u8,
                    }
                })
            })
            .collect();
        let mut state = base_state(
            4,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                druid,
                make_card(4, "Pooka", json!({})),
            ],
        );
        state.deck.villagers = vec!["Baker".to_string(), "Druid".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string(), "Rambler".to_string()];
        state.deck.minions = vec!["Spy".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        // Druid acts once before Baker #1 converts stable Spy #2, then may act
        // again after that conversion has fully settled.
        state.reveal_order = vec![3, 1, 2, 4];
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        state.rambler_shut_up_observations = interruptions;

        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Spy".to_string());
        scenario.evil_positions.insert(4, "Pooka".to_string());
        scenario.corrupted.insert(3);
        (state, scenario)
    }

    fn current_scout(pos: u8, payload: serde_json::Value) -> CardInfo {
        let mut info = payload.as_object().unwrap().clone();
        info.insert(
            SCOUT_CURRENT_VARIANT_FIELD.to_string(),
            serde_json::Value::String(POET_CURRENT_VARIANT.to_string()),
        );
        make_card(pos, "Scout", serde_json::Value::Object(info))
    }

    fn current_hunter(pos: u8, distance: serde_json::Value) -> CardInfo {
        make_card(
            pos,
            "Hunter",
            json!({
                "hunter_variant": "public_current",
                "distance": distance,
            }),
        )
    }

    fn current_oracle(pos: u8, payload: serde_json::Value) -> CardInfo {
        let info_text = current_oracle_text(&payload);
        let mut info = payload.as_object().unwrap().clone();
        info.insert(
            ORACLE_CURRENT_VARIANT_FIELD.to_string(),
            serde_json::Value::String(POET_CURRENT_VARIANT.to_string()),
        );
        let mut card = make_card(pos, "Oracle", serde_json::Value::Object(info));
        card.info_text = info_text;
        card
    }

    fn current_medium(pos: u8, target: serde_json::Value, role: serde_json::Value) -> CardInfo {
        let info_text = target
            .as_u64()
            .and_then(|target| u8::try_from(target).ok())
            .zip(role.as_str())
            .map(|(target, role)| current_medium_text(target, role))
            .unwrap_or_default();
        let mut card = make_card(
            pos,
            "Medium",
            json!({
                "medium_variant": "public_current",
                "good_position": target,
                "good_role": role,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn current_knitter(pos: u8, claimed: serde_json::Value) -> CardInfo {
        let info_text = claimed
            .as_i64()
            .and_then(current_knitter_claim_text)
            .unwrap_or_default();
        let mut card = make_card(
            pos,
            "Knitter",
            json!({
                "knitter_variant": "public_current",
                "evil_pairs": claimed,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn current_enlightened(pos: u8, direction: serde_json::Value) -> CardInfo {
        let info_text = direction
            .as_str()
            .and_then(|direction| match direction {
                "CW" => Some(Direction::CW),
                "CCW" => Some(Direction::CCW),
                "Equidistant" => Some(Direction::Equidistant),
                _ => None,
            })
            .map(current_enlightened_claim_text)
            .unwrap_or_default()
            .to_string();
        let mut card = make_card(
            pos,
            "Enlightened",
            json!({
                "enlightened_variant": "public_current",
                "direction": direction,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn current_empress(pos: u8, targets: Vec<u8>) -> CardInfo {
        let info_text = current_empress_claim_text(&targets).unwrap_or_default();
        let mut card = make_card(
            pos,
            "Empress",
            json!({
                "empress_variant": "public_current",
                "targets": targets,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn current_gemcrafter(pos: u8, target: serde_json::Value) -> CardInfo {
        let info_text = target
            .as_u64()
            .and_then(|target| u8::try_from(target).ok())
            .map(current_gemcrafter_claim_text)
            .unwrap_or_default();
        let mut card = make_card(
            pos,
            "Gemcrafter",
            json!({
                "gemcrafter_variant": "public_current",
                "good_position": target,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn current_bishop(pos: u8, targets: Vec<u8>, types: Vec<&str>) -> CardInfo {
        let parsed_types: Vec<BishopType> = types
            .iter()
            .filter_map(|role_type| BishopType::parse(role_type))
            .collect();
        let info_text = current_bishop_claim_text(&targets, &parsed_types).unwrap_or_default();
        let mut card = make_card(
            pos,
            "Bishop",
            json!({
                "bishop_variant": "public_current",
                "targets": targets,
                "types": types,
            }),
        );
        card.info_text = info_text;
        card
    }

    fn set_current_bishop_authored_good_counts(state: &mut GameState, town: u8, outs: u8) {
        state.board_villager_count = Some(town);
        state.board_outcast_count = Some(outs);
        state.board_count_provenance =
            crate::types::BoardCountProvenance::TrustedPreStart;
    }

    #[test]
    fn current_enlightened_schema_text_poet_and_legacy_are_exact() {
        let state = base_state(5, vec![]);
        let scenario = empty_scenario();
        for (name, direction, text) in [
            ("CW", Direction::CW, "Closest Evil is:\nClockwise"),
            (
                "CCW",
                Direction::CCW,
                "Closest Evil is:\nCounter-clockwise",
            ),
            (
                "Equidistant",
                Direction::Equidistant,
                "Closest Evil is equidistant",
            ),
        ] {
            let direct = current_enlightened(1, json!(name));
            assert_eq!(direct.info_text, text);
            assert_eq!(
                parse_current_enlightened_claim(
                    &direct,
                    CurrentPassivePayloadSource::Direct,
                    &state,
                ),
                Some(direction),
            );
            let poet = current_poet("Enlightened", json!({"direction": name}));
            assert_eq!(poet.info_text, text);
            assert!(validate_current_poet_payload(
                &poet,
                &state,
                "Enlightened",
            ));
        }

        for direction in [json!("cw"), json!("Clockwise"), json!("Equal"), json!(10)] {
            assert!(!validate_enlightened(
                &current_enlightened(1, direction),
                &scenario,
                &state,
            ));
        }
        let mut wrong_text = current_enlightened(1, json!("Equidistant"));
        wrong_text.info_text = "Closest Evil is: Equidistant".to_string();
        assert!(!validate_enlightened(&wrong_text, &scenario, &state));

        let mut extra = current_enlightened(1, json!("Equidistant"));
        extra.info_parsed.insert("targets".to_string(), json!([]));
        assert!(!validate_enlightened(&extra, &scenario, &state));
        let mut noncanonical = current_enlightened(1, json!("Equidistant"));
        noncanonical.apparent_role = "enlightened".to_string();
        assert!(!validate_enlightened(&noncanonical, &scenario, &state));
        for position in [0, 6] {
            assert!(!validate_enlightened(
                &current_enlightened(position, json!("Equidistant")),
                &scenario,
                &state,
            ));
        }
        for info in [
            json!({"enlightened_variant": "future", "direction": "Equidistant"}),
            json!({"enlightened_variant": 7, "direction": "Equidistant"}),
            json!({
                "enlightened_variant": "public_current",
                "poet_variant": "public_current",
                "direction": "Equidistant",
            }),
            json!({"knitter_variant": "public_current", "direction": "Equidistant"}),
        ] {
            let mut malformed = make_card(1, "Enlightened", info);
            malformed.info_text = current_enlightened_claim_text(Direction::Equidistant).to_string();
            assert!(!validate_enlightened(&malformed, &scenario, &state));
        }

        let poet_truth = current_poet("Enlightened", json!({"direction": "CW"}));
        let poet_false = current_poet("Enlightened", json!({"direction": "CCW"}));
        let poet_state = base_state(5, vec![poet_truth.clone()]);
        let mut poet_world = empty_scenario();
        poet_world.evil_positions.insert(5, "Pooka".to_string());
        assert!(validate_poet(&poet_truth, &poet_world, &poet_state));
        assert!(!validate_poet(&poet_false, &poet_world, &poet_state));
        poet_world.corrupted.insert(1);
        assert!(!validate_poet(&poet_truth, &poet_world, &poet_state));
        assert!(validate_poet(&poet_false, &poet_world, &poet_state));

        let legacy_direct = make_card(1, "Enlightened", json!({"direction": "cw"}));
        let legacy_poet = make_card(
            1,
            "Poet",
            json!({"copied_role": "Enlightened", "direction": "equal"}),
        );
        assert!(validate_enlightened(&legacy_direct, &scenario, &state));
        assert!(validate_poet(&legacy_poet, &scenario, &state));
    }

    #[test]
    fn current_enlightened_poet_anchors_geometry_on_the_poet_position() {
        let mut poet_truth = current_poet("Enlightened", json!({"direction": "CW"}));
        poet_truth.position = 4;
        let mut poet_false = current_poet("Enlightened", json!({"direction": "CCW"}));
        poet_false.position = 4;
        let state = base_state(6, vec![poet_truth.clone()]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(3, "Pooka".to_string());

        // From Poet #4, Evil #3 is one decreasing-ID step: public Clockwise.
        // Anchoring on the helper's old/default #1 instead would yield CCW.
        assert!(validate_poet(&poet_truth, &scenario, &state));
        assert!(!validate_poet(&poet_false, &scenario, &state));
    }

    #[test]
    fn current_enlightened_uses_full_circle_registered_geometry_and_all_lifecycle_seats() {
        assert_eq!(
            current_enlightened_direction(1, 5, &HashSet::from([2])),
            Direction::CCW,
        );
        assert_eq!(
            current_enlightened_direction(1, 5, &HashSet::from([5])),
            Direction::CW,
        );
        assert_eq!(
            current_enlightened_direction(1, 6, &HashSet::from([3, 5])),
            Direction::Equidistant,
        );
        assert_eq!(
            current_enlightened_direction(1, 4, &HashSet::from([3])),
            Direction::Equidistant,
        );
        assert_eq!(
            current_enlightened_direction(1, 1, &HashSet::from([1])),
            Direction::Equidistant,
        );
        assert_eq!(
            current_enlightened_direction(1, 2, &HashSet::from([2])),
            Direction::Equidistant,
        );
        assert_eq!(
            current_enlightened_direction(1, 5, &HashSet::from([1])),
            Direction::Equidistant,
        );

        // Archived clean truth asc27_v1: #6 is the nearest true Evil to
        // Enlightened #7, one decreasing-ID step, so the public answer is CW.
        let asc27 = current_enlightened(7, json!("CW"));
        let asc27_state = base_state(7, vec![asc27.clone()]);
        let mut asc27_world = empty_scenario();
        asc27_world.evil_positions.insert(6, "Pooka".to_string());
        assert!(validate_enlightened(&asc27, &asc27_world, &asc27_state));

        let equidistant = current_enlightened(1, json!("Equidistant"));
        let mut lifecycle_state = base_state(7, vec![equidistant.clone()]);
        lifecycle_state.executed = vec![2];
        lifecycle_state.night_kills = vec![7];
        let mut lifecycle = empty_scenario();
        lifecycle.evil_positions.insert(2, "Witch".to_string());
        lifecycle.evil_positions.insert(7, "Pooka".to_string());
        assert!(validate_enlightened(
            &equidistant,
            &lifecycle,
            &lifecycle_state,
        ));

        let spy_state = base_state(
            5,
            vec![
                current_enlightened(1, json!("CW")),
                make_card(2, "Spy", json!({})),
            ],
        );
        let mut spy = empty_scenario();
        spy.evil_positions.insert(2, "Spy".to_string());
        spy.evil_positions.insert(4, "Pooka".to_string());
        assert!(validate_enlightened(
            &current_enlightened(1, json!("CW")),
            &spy,
            &spy_state,
        ));
        assert!(!validate_enlightened(
            &current_enlightened(1, json!("CCW")),
            &spy,
            &spy_state,
        ));
    }

    #[test]
    fn current_enlightened_bluff_supports_each_and_only_false_direction() {
        let state = base_state(5, vec![]);

        let mut clockwise = empty_scenario();
        clockwise.corrupted.insert(1);
        clockwise.evil_positions.insert(5, "Pooka".to_string());
        assert!(!validate_enlightened(
            &current_enlightened(1, json!("CW")),
            &clockwise,
            &state,
        ));
        assert!(validate_enlightened(
            &current_enlightened(1, json!("CCW")),
            &clockwise,
            &state,
        ));
        assert!(validate_enlightened(
            &current_enlightened(1, json!("Equidistant")),
            &clockwise,
            &state,
        ));

        let mut counterclockwise = empty_scenario();
        counterclockwise.corrupted.insert(1);
        counterclockwise.evil_positions.insert(2, "Pooka".to_string());
        assert!(!validate_enlightened(
            &current_enlightened(1, json!("CCW")),
            &counterclockwise,
            &state,
        ));
        assert!(validate_enlightened(
            &current_enlightened(1, json!("CW")),
            &counterclockwise,
            &state,
        ));
        assert!(validate_enlightened(
            &current_enlightened(1, json!("Equidistant")),
            &counterclockwise,
            &state,
        ));

        let mut no_evil = empty_scenario();
        no_evil.corrupted.insert(1);
        assert!(validate_enlightened(
            &current_enlightened(1, json!("CW")),
            &no_evil,
            &state,
        ));
        assert!(validate_enlightened(
            &current_enlightened(1, json!("CCW")),
            &no_evil,
            &state,
        ));
        assert!(!validate_enlightened(
            &current_enlightened(1, json!("Equidistant")),
            &no_evil,
            &state,
        ));
    }

    #[test]
    fn current_enlightened_shares_one_anonymous_wretch_world() {
        let enlightened = current_enlightened(1, json!("CCW"));
        let mut bounty = current_poet("Bounty Hunter", json!({"evil_position": 7}));
        bounty.position = 2;
        let mut state = base_state(8, vec![enlightened.clone(), bounty.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(5, "Pooka".to_string());

        assert!(validate_enlightened(&enlightened, &scenario, &state));
        assert!(validate_poet(&bounty, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        let mut compatible = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        compatible.position = 2;
        state.cards[1] = compatible.clone();
        assert!(validate_poet(&compatible, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
    }

    #[test]
    fn current_enlightened_observations_share_one_monotonic_baker_spy_timeline() {
        let first = current_enlightened(3, json!("CCW"));
        let second = current_enlightened(5, json!("CCW"));
        let mut state = base_state(
            8,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Scout", json!({})),
                first.clone(),
                make_card(4, "Baker", json!({"original_role": "Spy"})),
                second.clone(),
                make_card(6, "Lover", json!({})),
                make_card(7, "Bard", json!({})),
                make_card(8, "Pooka", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Baker".to_string(),
            "Scout".to_string(),
            "Enlightened".to_string(),
            "Enlightened".to_string(),
            "Lover".to_string(),
            "Bard".to_string(),
        ];
        state.deck.minions = vec!["Spy".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.n_evil = 2;
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Spy".to_string());
        scenario.evil_positions.insert(8, "Pooka".to_string());

        // #3 says the converted Spy has reset to registered Evil. Later #5
        // says it still has stale Good registerAs. Each clue is independently
        // reachable, but the delayed reset cannot move backwards.
        state.reveal_order = vec![1, 3, 2, 5, 6, 7, 4, 8];
        assert!(validate_enlightened(&first, &scenario, &state));
        assert!(validate_enlightened(&second, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        // Stale Good first, then registered Evil, has one monotonic boundary.
        state.reveal_order = vec![1, 5, 2, 3, 6, 7, 4, 8];
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
    }

    #[test]
    fn current_empress_schema_poet_parity_and_legacy_are_marker_gated() {
        let direct = current_empress(1, vec![2, 3, 4]);
        let mut poet = current_poet("Empress", json!({"targets": [2, 3, 4]}));
        poet.position = 1;
        let mut state = base_state(5, vec![direct.clone()]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Pooka".to_string());

        assert_eq!(direct.info_text, "One is Evil:\n#2, #3 or #4");
        assert_eq!(poet.info_text, direct.info_text);
        assert_eq!(
            parse_current_empress_targets(
                &direct,
                CurrentPassivePayloadSource::Direct,
                &state,
            ),
            Some(vec![2, 3, 4]),
        );
        assert!(validate_empress(&direct, &scenario, &state));
        state.cards[0] = poet.clone();
        assert!(validate_current_poet_payload(&poet, &state, "Empress"));
        assert!(validate_poet(&poet, &scenario, &state));

        for payload in [
            json!({"empress_variant": "public_current", "targets": [2, 3]}),
            json!({"empress_variant": "public_current", "targets": [2, 2, 4]}),
            json!({"empress_variant": "public_current", "targets": [3, 2, 4]}),
            json!({"empress_variant": "public_current", "targets": [0, 2, 4]}),
            json!({"empress_variant": "public_current", "targets": [2, 3, 6]}),
            json!({"empress_variant": "public_current", "targets": [2, true, 4]}),
            json!({"empress_variant": "future", "targets": [2, 3, 4]}),
            json!({"empress_variant": "public_current", "targets": [2, 3, 4], "extra": true}),
            json!({"empress_variant": "public_current", "poet_variant": "public_current", "targets": [2, 3, 4]}),
        ] {
            let malformed = make_card(1, "Empress", payload);
            assert!(!validate_empress(&malformed, &scenario, &state));
        }
        for text in [
            "One is Evil: #2, #3 or #4",
            "One is Evil:\n#2, #3, #4",
            "One is Evil:\n#2, #3 or #4.",
            "One is Evil:\n#2, #4 or #3",
            "One is Evil:\n#2, #3 or #4 ",
        ] {
            let mut wrong_text = direct.clone();
            wrong_text.info_text = text.to_string();
            assert!(!validate_empress(&wrong_text, &scenario, &state));
        }
        poet.info_text.push(' ');
        state.cards[0] = poet.clone();
        assert!(!validate_current_poet_payload(&poet, &state, "Empress"));
        assert!(!validate_poet(&poet, &scenario, &state));

        let legacy_two = make_card(1, "Empress", json!({"targets": [2, 4]}));
        assert!(validate_empress(&legacy_two, &scenario, &state));
        assert!(validate_empress(
            &make_card(1, "Empress", json!({})),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_empress_builds_native_truth_and_bluff_pools_with_actor_removal() {
        let truth = current_empress(1, vec![2, 3, 4]);
        let mut state = base_state(4, vec![truth.clone()]);
        let mut one_evil = empty_scenario();
        one_evil.evil_positions.insert(4, "Pooka".to_string());
        state.executed = vec![4];
        state.night_kills = vec![2];
        assert!(validate_empress(&truth, &one_evil, &state));

        let good_actor_self = current_empress(1, vec![1, 2, 4]);
        assert!(!validate_empress(&good_actor_self, &one_evil, &state));

        let puppet_self = current_empress(1, vec![1, 2, 3]);
        let mut truthful_puppet = empty_scenario();
        truthful_puppet.puppet_position = Some(1);
        assert!(validate_empress(&puppet_self, &truthful_puppet, &state));

        let bluff = current_empress(1, vec![2, 3, 4]);
        let mut lying = empty_scenario();
        lying.corrupted.insert(1);
        assert!(validate_empress(&bluff, &lying, &state));
        lying.evil_positions.insert(4, "Pooka".to_string());
        assert!(!validate_empress(&bluff, &lying, &state));
        assert!(!validate_empress(
            &current_empress(1, vec![1, 2, 3]),
            &lying,
            &state,
        ));

        let three = base_state(3, vec![]);
        let mut small_truth = empty_scenario();
        small_truth.evil_positions.insert(3, "Pooka".to_string());
        assert!(!validate_empress(
            &current_empress(1, vec![1, 2, 3]),
            &small_truth,
            &three,
        ));
        let mut small_bluff = empty_scenario();
        small_bluff.corrupted.insert(1);
        assert!(!validate_empress(
            &current_empress(1, vec![1, 2, 3]),
            &small_bluff,
            &three,
        ));
        small_truth = empty_scenario();
        small_truth.puppet_position = Some(1);
        assert!(validate_empress(
            &current_empress(1, vec![1, 2, 3]),
            &small_truth,
            &three,
        ));
    }

    #[test]
    fn current_empress_self_selection_has_direct_and_poet_charref_parity() {
        let direct_self = current_empress(1, vec![1, 2, 3]);
        let mut poet_self = current_poet("Empress", json!({"targets": [1, 2, 3]}));
        poet_self.position = 1;

        // A physical Spy can display Empress as its bluff. Its Villager
        // registerAs overrides runtime Evil, but native bluff selection still
        // removes charRef from the registered-Good pool, so self is invalid.
        let mut stable_spy = empty_scenario();
        stable_spy.evil_positions.insert(1, "Spy".to_string());
        let direct_state = base_state(3, vec![direct_self.clone()]);
        let poet_state = base_state(3, vec![poet_self.clone()]);
        assert!(!validate_empress(&direct_self, &stable_spy, &direct_state));
        assert!(!validate_poet(&poet_self, &stable_spy, &poet_state));

        // Clean Puppet's HealthyBluff makes the provider truthful, and its
        // registered-Evil charRef remains eligible for the independent Evil
        // draw. Corruption overrides HealthyBluff and makes the same self-ref
        // impossible for both direct and Poet bluff paths.
        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(1);
        assert!(validate_empress(&direct_self, &puppet, &direct_state));
        assert!(validate_poet(&poet_self, &puppet, &poet_state));
        puppet.corrupted.insert(1);
        assert!(!validate_empress(&direct_self, &puppet, &direct_state));
        assert!(!validate_poet(&poet_self, &puppet, &poet_state));
    }

    #[test]
    fn current_empress_shares_anonymous_wretch_and_baker_spy_worlds() {
        let empress = current_empress(1, vec![2, 3, 4]);
        let mut bounty = current_poet("Bounty Hunter", json!({"evil_position": 2}));
        bounty.position = 5;
        let mut state = base_state(
            6,
            vec![
                empress.clone(),
                make_card(4, "Pooka", json!({})),
                bounty.clone(),
            ],
        );
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance =
            crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Pooka".to_string());
        assert!(validate_empress(&empress, &scenario, &state));
        assert!(validate_poet(&bounty, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
        bounty.info_parsed.insert("evil_position".to_string(), json!(6));
        bounty.info_text = "#6\nis Evil".to_string();
        state.cards[2] = bounty;
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        let mut stale_bounty = current_poet("Bounty Hunter", json!({"evil_position": 4}));
        stale_bounty.position = 3;
        let reset_empress = current_empress(5, vec![2, 4, 6]);
        let mut timeline_state = base_state(
            8,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Scout", json!({})),
                stale_bounty.clone(),
                make_card(4, "Baker", json!({"original_role": "Spy"})),
                reset_empress.clone(),
                make_card(6, "Scout", json!({})),
                make_card(7, "Bard", json!({})),
                make_card(8, "Pooka", json!({})),
            ],
        );
        timeline_state.deck.villagers = vec![
            "Baker".to_string(),
            "Scout".to_string(),
            "Poet".to_string(),
            "Empress".to_string(),
            "Scout".to_string(),
            "Bard".to_string(),
        ];
        timeline_state.deck.minions = vec!["Spy".to_string()];
        timeline_state.deck.demons = vec!["Pooka".to_string()];
        timeline_state.n_evil = 2;
        timeline_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        let mut timeline = empty_scenario();
        timeline.evil_positions.insert(4, "Spy".to_string());
        timeline.evil_positions.insert(8, "Pooka".to_string());
        timeline.corrupted.insert(3);
        // A filler reveal after the converting Baker lets Empress choose an
        // early reset, while the later Bounty clue can independently choose a
        // late reset. No single monotonic boundary supports both witnesses.
        timeline_state.reveal_order = vec![1, 2, 5, 3, 6, 7, 4, 8];
        assert!(validate_empress(
            &reset_empress,
            &timeline,
            &timeline_state,
        ));
        assert!(validate_poet(
            &stale_bounty,
            &timeline,
            &timeline_state,
        ));
        assert!(!validate_current_hidden_surface_consistency(
            &timeline,
            &timeline_state,
        ));
        timeline_state.reveal_order = vec![1, 3, 2, 5, 6, 7, 4, 8];
        assert!(validate_current_hidden_surface_consistency(
            &timeline,
            &timeline_state,
        ));
    }

    #[test]
    fn current_empress_fails_closed_on_unreplayed_unknown_start_identity() {
        let empress = current_empress(1, vec![2, 3, 4]);
        let mut state = base_state(4, vec![empress.clone()]);
        let mut unknown = empty_scenario();
        unknown.evil_positions.insert(4, "Unknown".to_string());
        assert!(!validate_empress(&empress, &unknown, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &unknown, &state,
        ));

        state.executed_evil_roles.insert(4, "Pooka".to_string());
        assert!(!validate_empress(&empress, &unknown, &state));

        unknown.evil_positions.insert(4, "Pooka".to_string());
        assert!(validate_empress(&empress, &unknown, &state));
    }

    #[test]
    fn current_gemcrafter_schema_text_poet_and_legacy_are_exact() {
        let direct = current_gemcrafter(1, json!(2));
        let mut poet = current_poet("Gemcrafter", json!({"good_position": 2}));
        poet.position = 1;
        let mut state = base_state(4, vec![direct.clone()]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Pooka".to_string());

        assert_eq!(direct.info_text, "#2 is Good");
        assert_eq!(poet.info_text, direct.info_text);
        assert_eq!(
            parse_current_gemcrafter_target(
                &direct,
                CurrentPassivePayloadSource::Direct,
                &state,
            ),
            Some(2),
        );
        assert!(validate_gemcrafter(&direct, &scenario, &state));
        state.cards[0] = poet.clone();
        assert!(validate_current_poet_payload(
            &poet,
            &state,
            "Gemcrafter",
        ));
        assert!(validate_poet(&poet, &scenario, &state));

        for target in [json!(0), json!(-1), json!(5), json!(256), json!(true), json!("2")] {
            assert!(!validate_gemcrafter(
                &current_gemcrafter(1, target),
                &scenario,
                &state,
            ));
        }
        for position in [0, 5] {
            assert!(!validate_gemcrafter(
                &current_gemcrafter(position, json!(2)),
                &scenario,
                &state,
            ));
        }
        for payload in [
            json!({"gemcrafter_variant": "future", "good_position": 2}),
            json!({"gemcrafter_variant": 1, "good_position": 2}),
            json!({"gemcrafter_variant": "public_current", "good_position": 2, "extra": true}),
            json!({
                "gemcrafter_variant": "public_current",
                "poet_variant": "public_current",
                "good_position": 2,
            }),
            json!({"empress_variant": "public_current", "good_position": 2}),
        ] {
            let mut malformed = make_card(1, "Gemcrafter", payload);
            malformed.info_text = "#2 is Good".to_string();
            assert!(!validate_gemcrafter(&malformed, &scenario, &state));
        }
        let mut noncanonical = direct.clone();
        noncanonical.apparent_role = "gemcrafter".to_string();
        assert!(!validate_gemcrafter(&noncanonical, &scenario, &state));
        for text in [
            "#2 is good",
            "#2\nis Good",
            "#2 is Good.",
            "#3 is Good",
            "#2 is Good ",
        ] {
            let mut wrong = direct.clone();
            wrong.info_text = text.to_string();
            assert!(!validate_gemcrafter(&wrong, &scenario, &state));
        }
        poet.info_text.push(' ');
        state.cards[0] = poet.clone();
        assert!(!validate_current_poet_payload(
            &poet,
            &state,
            "Gemcrafter",
        ));
        assert!(!validate_poet(&poet, &scenario, &state));

        // Missing provenance retains the frozen scalar behavior, including a
        // Rambler-replaced clue and unmarked Poet delegation.
        assert!(validate_gemcrafter(
            &make_card(1, "Gemcrafter", json!({})),
            &scenario,
            &state,
        ));
        assert!(validate_gemcrafter(
            &make_card(1, "Gemcrafter", json!({"shut_up_target": 3})),
            &scenario,
            &state,
        ));
        assert!(validate_gemcrafter(
            &make_card(1, "Gemcrafter", json!({"good_position": 2})),
            &scenario,
            &state,
        ));
        let legacy_poet = make_card(
            1,
            "Poet",
            json!({"copied_role": "Gemcrafter", "good_position": 2}),
        );
        assert!(validate_poet(&legacy_poet, &scenario, &state));
    }

    #[test]
    fn current_gemcrafter_applies_native_pool_wide_conditional_self_removal() {
        let direct_self = current_gemcrafter(1, json!(1));
        let mut poet_self = current_poet("Gemcrafter", json!({"good_position": 1}));
        poet_self.position = 1;
        let direct_state = base_state(3, vec![direct_self.clone()]);
        let poet_state = base_state(3, vec![poet_self.clone()]);

        let mut sole_good = empty_scenario();
        sole_good.evil_positions.insert(2, "Witch".to_string());
        sole_good.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_gemcrafter(
            &direct_self,
            &sole_good,
            &direct_state,
        ));
        assert!(validate_poet(&poet_self, &sole_good, &poet_state));
        sole_good.evil_positions.remove(&2);
        assert!(!validate_gemcrafter(
            &direct_self,
            &sole_good,
            &direct_state,
        ));
        assert!(!validate_poet(&poet_self, &sole_good, &poet_state));

        let mut sole_evil = empty_scenario();
        sole_evil.evil_positions.insert(1, "Pooka".to_string());
        assert!(validate_gemcrafter(
            &direct_self,
            &sole_evil,
            &direct_state,
        ));
        assert!(validate_poet(&poet_self, &sole_evil, &poet_state));
        sole_evil.evil_positions.insert(2, "Witch".to_string());
        assert!(!validate_gemcrafter(
            &direct_self,
            &sole_evil,
            &direct_state,
        ));
        assert!(!validate_poet(&poet_self, &sole_evil, &poet_state));

        let target_evil = current_gemcrafter(1, json!(3));
        let mut poet_target_evil = current_poet("Gemcrafter", json!({"good_position": 3}));
        poet_target_evil.position = 1;
        let mut corrupted_good = empty_scenario();
        corrupted_good.corrupted.insert(1);
        corrupted_good.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_gemcrafter(
            &target_evil,
            &corrupted_good,
            &direct_state,
        ));
        assert!(validate_poet(
            &poet_target_evil,
            &corrupted_good,
            &poet_state,
        ));
        assert!(!validate_gemcrafter(
            &direct_self,
            &corrupted_good,
            &direct_state,
        ));
        assert!(!validate_poet(
            &poet_self,
            &corrupted_good,
            &poet_state,
        ));

        // Clean Puppet is runtime/registered Evil but HealthyBluff makes its
        // Gemcrafter provider truthful. It can select a Good occurrence and
        // cannot select its Evil self from that truth pool.
        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(1);
        assert!(validate_gemcrafter(
            &current_gemcrafter(1, json!(2)),
            &puppet,
            &direct_state,
        ));
        let mut poet_target_good = current_poet("Gemcrafter", json!({"good_position": 2}));
        poet_target_good.position = 1;
        assert!(validate_poet(&poet_target_good, &puppet, &poet_state));
        assert!(!validate_gemcrafter(
            &direct_self,
            &puppet,
            &direct_state,
        ));
        assert!(!validate_poet(&poet_self, &puppet, &poet_state));
    }

    #[test]
    fn current_gemcrafter_uses_all_lifecycle_seats_and_current_identity_writers() {
        let mut state = base_state(4, vec![current_gemcrafter(1, json!(2))]);
        state.executed = vec![2];
        state.night_kills = vec![3];
        state.blocked_positions = vec![4];
        let scenario = empty_scenario();
        for target in [2, 3, 4] {
            assert!(validate_gemcrafter(
                &current_gemcrafter(1, json!(target)),
                &scenario,
                &state,
            ));
        }

        let spy_state = base_state(
            3,
            vec![
                current_gemcrafter(1, json!(2)),
                make_card(2, "Spy", json!({})),
            ],
        );
        let mut stable_spy = empty_scenario();
        stable_spy.evil_positions.insert(2, "Spy".to_string());
        stable_spy.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_gemcrafter(
            &current_gemcrafter(1, json!(2)),
            &stable_spy,
            &spy_state,
        ));
        stable_spy.corrupted.insert(1);
        assert!(!validate_gemcrafter(
            &current_gemcrafter(1, json!(2)),
            &stable_spy,
            &spy_state,
        ));

        let mut explicit_state = base_state(
            4,
            vec![
                current_gemcrafter(1, json!(2)),
                make_card(2, "Wretch", json!({})),
            ],
        );
        explicit_state.deck.outcasts = vec!["Wretch".to_string()];
        assert!(!validate_gemcrafter(
            &current_gemcrafter(1, json!(2)),
            &scenario,
            &explicit_state,
        ));
        let mut lying = scenario.clone();
        lying.corrupted.insert(1);
        assert!(validate_gemcrafter(
            &current_gemcrafter(1, json!(2)),
            &lying,
            &explicit_state,
        ));

        let mut moved_state = base_state(
            4,
            vec![
                current_gemcrafter(1, json!(3)),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Wretch", json!({})),
            ],
        );
        moved_state.deck.outcasts = vec!["Wretch".to_string()];
        let mut moved = empty_scenario();
        moved.evil_positions.insert(2, "Twin Minion".to_string());
        moved.evil_positions.insert(4, "Pooka".to_string());
        moved.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 2,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 4,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 3,
                neighbor_pre_swap_role: "Wretch".to_string(),
            },
        });
        // Wretch data moved onto physical Evil #2 remains registered Evil;
        // physical Good #3 now carrying Twin data remains registered Good.
        assert!(validate_gemcrafter(
            &current_gemcrafter(1, json!(3)),
            &moved,
            &moved_state,
        ));
        assert!(!validate_gemcrafter(
            &current_gemcrafter(1, json!(2)),
            &moved,
            &moved_state,
        ));

        let mut copied_state = base_state(
            4,
            vec![
                current_gemcrafter(1, json!(2)),
                make_card(2, "Scout", json!({})),
            ],
        );
        copied_state.deck.outcasts = vec!["Wretch".to_string()];
        let mut copied = empty_scenario();
        copied.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 4,
            target_position: 2,
            copied_role: "Wretch".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });
        assert!(!validate_gemcrafter(
            &current_gemcrafter(1, json!(2)),
            &copied,
            &copied_state,
        ));
        copied.corrupted.insert(1);
        assert!(validate_gemcrafter(
            &current_gemcrafter(1, json!(2)),
            &copied,
            &copied_state,
        ));
    }

    #[test]
    fn current_gemcrafter_shares_anonymous_wretch_and_baker_spy_worlds() {
        let gemcrafter = current_gemcrafter(1, json!(2));
        let mut bounty = current_poet("Bounty Hunter", json!({"evil_position": 2}));
        bounty.position = 4;
        let mut state = base_state(5, vec![gemcrafter.clone(), bounty.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance =
            crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(5, "Pooka".to_string());
        assert!(validate_gemcrafter(&gemcrafter, &scenario, &state));
        assert!(validate_poet(&bounty, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        bounty.info_parsed.insert("evil_position".to_string(), json!(3));
        bounty.info_text = "#3\nis Evil".to_string();
        state.cards[1] = bounty;
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        let mut reset_bounty = current_poet("Bounty Hunter", json!({"evil_position": 4}));
        reset_bounty.position = 3;
        let stale_gemcrafter = current_gemcrafter(5, json!(4));
        let mut timeline_state = base_state(
            8,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Scout", json!({})),
                reset_bounty.clone(),
                make_card(4, "Baker", json!({"original_role": "Spy"})),
                stale_gemcrafter.clone(),
                make_card(6, "Scout", json!({})),
                make_card(7, "Bard", json!({})),
                make_card(8, "Pooka", json!({})),
            ],
        );
        timeline_state.deck.villagers = vec![
            "Baker".to_string(),
            "Scout".to_string(),
            "Poet".to_string(),
            "Gemcrafter".to_string(),
            "Scout".to_string(),
            "Bard".to_string(),
        ];
        timeline_state.deck.minions = vec!["Spy".to_string()];
        timeline_state.deck.demons = vec!["Pooka".to_string()];
        timeline_state.n_evil = 2;
        timeline_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        let mut timeline = empty_scenario();
        timeline.evil_positions.insert(4, "Spy".to_string());
        timeline.evil_positions.insert(8, "Pooka".to_string());

        // Bounty can independently choose an early delayed reset while the
        // later Gemcrafter can independently choose stale Spy registerAs, but
        // no one monotonic native reset boundary supports that ordering.
        timeline_state.reveal_order = vec![1, 2, 3, 5, 6, 7, 4, 8];
        assert!(validate_poet(
            &reset_bounty,
            &timeline,
            &timeline_state,
        ));
        assert!(validate_gemcrafter(
            &stale_gemcrafter,
            &timeline,
            &timeline_state,
        ));
        assert!(!validate_current_hidden_surface_consistency(
            &timeline,
            &timeline_state,
        ));

        timeline_state.reveal_order = vec![1, 2, 5, 3, 6, 7, 4, 8];
        assert!(validate_current_hidden_surface_consistency(
            &timeline,
            &timeline_state,
        ));
    }

    #[test]
    fn current_gemcrafter_fails_closed_on_unreplayed_unknown_start_identity() {
        let direct = current_gemcrafter(1, json!(2));
        let mut poet = current_poet("Gemcrafter", json!({"good_position": 2}));
        poet.position = 1;
        let mut state = base_state(3, vec![direct.clone(), poet.clone()]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(3, "Unknown".to_string());
        assert!(!validate_gemcrafter(&direct, &scenario, &state));
        assert!(!validate_poet(&poet, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        // A late public execution role cannot replay an opaque Start history.
        state.executed_evil_roles.insert(3, "Pooka".to_string());
        assert!(!validate_gemcrafter(&direct, &scenario, &state));

        scenario.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_gemcrafter(&direct, &scenario, &state));
        assert!(validate_poet(&poet, &scenario, &state));
    }

    #[test]
    fn executed_evil_card_is_skipped_only_while_scenario_origin_is_unknown() {
        let card = make_card(1, "Gemcrafter", json!({"good_position": 2}));
        let mut state = base_state(2, vec![card.clone()]);
        state.executed = vec![1];
        state.confirmed_evil = vec![1];
        assert!(state.executed_evil_roles.is_empty());

        let mut unresolved = empty_scenario();
        unresolved
            .evil_positions
            .insert(1, "Unknown".to_string());
        assert!(executed_evil_origin_is_unresolved(1, &unresolved, &state));
        assert!(check_scenario(&unresolved, &state));

        let mut exact = unresolved.clone();
        exact.evil_positions.insert(1, "Pooka".to_string());
        assert!(!executed_evil_origin_is_unresolved(1, &exact, &state));
        assert!(!validate_gemcrafter(&card, &exact, &state));
        assert!(!check_scenario(&exact, &state));
    }

    #[test]
    fn inferred_complex_start_writers_remain_fail_closed_until_replayed() {
        let mut state = base_state(1, vec![]);
        state.executed = vec![1];
        state.confirmed_evil = vec![1];
        let mut scenario = empty_scenario();

        scenario
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        assert!(current_has_unresolved_start_identity(&scenario, &state));
        scenario.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::NoDemon,
        });
        assert!(!current_has_unresolved_start_identity(&scenario, &state));

        scenario.twin_trace = None;
        for role in ["Puppeteer", "Shaman"] {
            scenario.evil_positions.insert(1, role.to_string());
            assert!(current_has_unresolved_start_identity(&scenario, &state));
        }

        scenario.evil_positions.insert(1, "Puppet".to_string());
        scenario.puppet_position = Some(1);
        assert!(current_has_unresolved_start_identity(&scenario, &state));
        state
            .executed_evil_roles
            .insert(1, "Unknown".to_string());
        assert_eq!(stable_evil_origin_role_at(1, &scenario, &state), None);
        assert_eq!(known_evil_role(1, &scenario, &state), Some("Puppet"));
        assert!(!executed_evil_origin_is_unresolved(1, &scenario, &state));

        scenario.puppet_position = None;
        state.executed_evil_roles.clear();
        scenario.evil_positions.insert(1, "Pooka".to_string());
        assert!(!current_has_unresolved_start_identity(&scenario, &state));
    }

    #[test]
    fn current_bishop_schema_text_poet_and_legacy_are_exact() {
        let state = base_state(6, vec![]);
        let scenario = empty_scenario();
        for (targets, types, text) in [
            (vec![2], vec!["Minion"], "#2 is a Minion"),
            (
                vec![2, 4],
                vec!["Villager", "Demon"],
                "Between\n#2, #4\nthere is:\nVillager and Demon",
            ),
            (
                vec![2, 4, 6],
                vec!["Minion", "Outcast", "Villager"],
                "Between\n#2, #4, #6\nthere is:\nMinion, Outcast and Villager",
            ),
        ] {
            let direct = current_bishop(1, targets.clone(), types.clone());
            assert_eq!(direct.info_text, text);
            assert!(parse_current_bishop_claim(
                &direct,
                CurrentPassivePayloadSource::Direct,
                &state,
            )
            .is_some());
            let poet = current_poet(
                "Bishop",
                json!({"targets": targets, "types": types}),
            );
            assert_eq!(poet.info_text, text);
            assert!(validate_current_poet_payload(&poet, &state, "Bishop"));
        }

        for payload in [
            json!({"bishop_variant": "public_current", "targets": [3, 2], "types": ["Villager", "Minion"]}),
            json!({"bishop_variant": "public_current", "targets": [2, 2], "types": ["Villager", "Minion"]}),
            json!({"bishop_variant": "public_current", "targets": [-1], "types": ["Villager"]}),
            json!({"bishop_variant": "public_current", "targets": [256], "types": ["Villager"]}),
            json!({"bishop_variant": "public_current", "targets": [2], "types": ["villager"]}),
            json!({"bishop_variant": "public_current", "targets": [2], "types": ["Villager"], "extra": true}),
            json!({"bishop_variant": "future", "targets": [2], "types": ["Villager"]}),
            json!({"bishop_variant": "public_current", "poet_variant": "public_current", "targets": [2], "types": ["Villager"]}),
        ] {
            let mut malformed = make_card(1, "Bishop", payload);
            malformed.info_text = "#2 is a Villager".to_string();
            assert!(!validate_bishop(&malformed, &scenario, &state));
        }
        let mut wrong_text = current_bishop(1, vec![2, 4], vec!["Villager", "Demon"]);
        wrong_text.info_text = "Between #2, #4 there is: Villager and Demon".to_string();
        assert!(!validate_bishop(&wrong_text, &scenario, &state));
        let mut wrong_actor = current_bishop(0, vec![2], vec!["Villager"]);
        assert!(!validate_bishop(&wrong_actor, &scenario, &state));
        wrong_actor.position = 7;
        assert!(!validate_bishop(&wrong_actor, &scenario, &state));

        let legacy_missing = make_card(1, "Bishop", json!({}));
        assert!(validate_bishop(&legacy_missing, &scenario, &state));
        let legacy_poet = make_card(
            1,
            "Poet",
            json!({"copied_role": "Bishop", "targets": [2]}),
        );
        let mut legacy_scenario = scenario.clone();
        legacy_scenario.evil_positions.insert(2, "Pooka".to_string());
        assert!(validate_poet(&legacy_poet, &legacy_scenario, &state));
    }

    #[test]
    fn current_bishop_truth_selects_each_available_category_and_prefers_minion() {
        let truth = current_bishop(
            1,
            vec![1, 3, 4],
            vec!["Minion", "Villager", "Outcast"],
        );
        let state = base_state(
            4,
            vec![
                truth.clone(),
                make_card(2, "Scout", json!({})),
                make_card(3, "Bombardier", json!({})),
                make_card(4, "Witch", json!({})),
            ],
        );
        let mut minion = empty_scenario();
        minion.evil_positions.insert(4, "Witch".to_string());
        assert!(validate_bishop(&truth, &minion, &state));

        let missing_outcast = current_bishop(1, vec![1, 4], vec!["Villager", "Minion"]);
        assert!(!validate_bishop(&missing_outcast, &minion, &state));
        let demon_while_minion_exists =
            current_bishop(1, vec![1, 4], vec!["Villager", "Demon"]);
        assert!(!validate_bishop(
            &demon_while_minion_exists,
            &minion,
            &state,
        ));
        let duplicate_category = current_bishop(
            1,
            vec![1, 2, 4],
            vec!["Villager", "Villager", "Minion"],
        );
        assert!(!validate_bishop(&duplicate_category, &minion, &state));

        let demon_truth = current_bishop(
            1,
            vec![1, 3, 4],
            vec!["Demon", "Villager", "Outcast"],
        );
        let mut demon = empty_scenario();
        demon.evil_positions.insert(4, "Pooka".to_string());
        assert!(validate_bishop(&demon_truth, &demon, &state));
    }

    #[test]
    fn current_bishop_includes_self_dead_hidden_and_tiny_board_occurrences() {
        let self_claim = current_bishop(1, vec![1], vec!["Minion"]);
        let singleton = base_state(1, vec![self_claim.clone()]);
        let mut healthy_puppet = empty_scenario();
        healthy_puppet.puppet_position = Some(1);
        assert!(validate_bishop(
            &self_claim,
            &healthy_puppet,
            &singleton,
        ));

        let dead_claim = current_bishop(1, vec![1, 2], vec!["Villager", "Demon"]);
        let mut dead_state = base_state(
            2,
            vec![dead_claim.clone(), make_card(2, "Pooka", json!({}))],
        );
        dead_state.executed = vec![2];
        let mut dead_demon = empty_scenario();
        dead_demon.evil_positions.insert(2, "Pooka".to_string());
        assert!(validate_bishop(&dead_claim, &dead_demon, &dead_state));
        dead_state.executed.clear();
        dead_state.night_kills = vec![2];
        assert!(validate_bishop(&dead_claim, &dead_demon, &dead_state));

        let hidden_claim = current_bishop(1, vec![1, 2], vec!["Villager", "Minion"]);
        let mut hidden_state = base_state(
            3,
            vec![hidden_claim.clone(), make_card(3, "Pooka", json!({}))],
        );
        hidden_state.deck.villagers = vec!["Bishop".to_string()];
        hidden_state.deck.outcasts = vec!["Wretch".to_string()];
        hidden_state.deck.demons = vec!["Pooka".to_string()];
        set_current_bishop_authored_good_counts(&mut hidden_state, 1, 1);
        let mut hidden_wretch = empty_scenario();
        hidden_wretch
            .evil_positions
            .insert(3, "Pooka".to_string());
        assert!(validate_bishop(
            &hidden_claim,
            &hidden_wretch,
            &hidden_state,
        ));
        let wrong_fallback = current_bishop(1, vec![1, 3], vec!["Villager", "Demon"]);
        assert!(!validate_bishop(
            &wrong_fallback,
            &hidden_wretch,
            &hidden_state,
        ));
    }

    #[test]
    fn current_bishop_bluff_uses_authored_domain_and_only_live_villager_refs() {
        let bluff = current_bishop(1, vec![2, 3], vec!["Villager", "Minion"]);
        let mut state = base_state(
            5,
            vec![
                bluff.clone(),
                make_card(2, "Scout", json!({})),
                make_card(3, "Lover", json!({})),
                make_card(4, "Oracle", json!({})),
                make_card(5, "Witch", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Bishop".to_string(),
            "Scout".to_string(),
            "Lover".to_string(),
            "Oracle".to_string(),
        ];
        state.deck.minions = vec!["Witch".to_string()];
        set_current_bishop_authored_good_counts(&mut state, 4, 0);
        let mut minion = empty_scenario();
        minion.corrupted.insert(1);
        minion.evil_positions.insert(5, "Witch".to_string());
        assert!(validate_bishop(&bluff, &minion, &state));
        let wrong_evil_token = current_bishop(1, vec![2, 3], vec!["Villager", "Demon"]);
        assert!(!validate_bishop(&wrong_evil_token, &minion, &state));
        let evil_ref = current_bishop(1, vec![2, 5], vec!["Villager", "Minion"]);
        assert!(!validate_bishop(&evil_ref, &minion, &state));

        state.cards.push(make_card(6, "Bombardier", json!({})));
        state.n_cards = 6;
        state.deck.outcasts = vec!["Bombardier".to_string()];
        set_current_bishop_authored_good_counts(&mut state, 4, 1);
        let three_refs = current_bishop(
            1,
            vec![1, 2, 3],
            vec!["Outcast", "Villager", "Minion"],
        );
        assert!(validate_bishop(&three_refs, &minion, &state));
        assert!(!validate_bishop(&bluff, &minion, &state));

        let mut demon = minion.clone();
        demon.evil_positions.insert(5, "Pooka".to_string());
        state.deck.demons = vec!["Pooka".to_string()];
        let demon_types = current_bishop(
            1,
            vec![1, 2, 3],
            vec!["Villager", "Demon", "Outcast"],
        );
        assert!(validate_bishop(&demon_types, &demon, &state));
        let candidate_pool_minion = current_bishop(
            1,
            vec![1, 2, 3],
            vec!["Villager", "Minion", "Outcast"],
        );
        assert!(!validate_bishop(
            &candidate_pool_minion,
            &demon,
            &state,
        ));

        let mut trusted_minion_state = state.clone();
        trusted_minion_state.board_minion_count = Some(1);
        let mut trusted_minion_world = demon.clone();
        trusted_minion_world
            .evil_positions
            .insert(4, "Witch".to_string());
        assert!(validate_bishop(
            &candidate_pool_minion,
            &trusted_minion_world,
            &trusted_minion_state,
        ));
        assert!(!validate_bishop(
            &demon_types,
            &trusted_minion_world,
            &trusted_minion_state,
        ));

        let mut untyped = demon.clone();
        untyped.evil_positions.insert(5, "Unknown".to_string());
        // An untyped executed Evil is not merely an unknown Minion/Demon
        // label: its missing role may have changed Start history. Until
        // scenario construction replays each concrete identity, current
        // Bishop must fail closed instead of branching locally.
        assert!(!validate_bishop(&demon_types, &untyped, &state));
        assert!(!validate_bishop(&candidate_pool_minion, &untyped, &state));
        let mut typed_death_state = state.clone();
        typed_death_state
            .executed_evil_roles
            .insert(5, "Pooka".to_string());
        assert!(validate_bishop(
            &demon_types,
            &untyped,
            &typed_death_state,
        ));
        assert!(!validate_bishop(
            &candidate_pool_minion,
            &untyped,
            &typed_death_state,
        ));

        let mut spy_state = state.clone();
        spy_state.deck.minions = vec!["Witch".to_string(), "Spy".to_string()];
        let mut dealt_spy = demon.clone();
        dealt_spy.evil_positions.insert(5, "Spy".to_string());
        assert!(validate_bishop(
            &candidate_pool_minion,
            &dealt_spy,
            &spy_state,
        ));
    }

    #[test]
    fn current_bishop_projects_spy_puppet_twin_and_shaman_current_data() {
        let spy_claim = current_bishop(1, vec![2, 3], vec!["Villager", "Demon"]);
        let spy_state = base_state(
            3,
            vec![
                spy_claim.clone(),
                make_card(2, "Scout", json!({})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        let mut spy = empty_scenario();
        spy.evil_positions.insert(2, "Spy".to_string());
        spy.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_bishop(&spy_claim, &spy, &spy_state));

        // Public typed-death evidence can refine the scenario builder's
        // grouped Unknown evil seat. Physical Spy registration still takes
        // precedence over its Minion current-data type in that overlay shape.
        let mut typed_spy_state = spy_state.clone();
        typed_spy_state.executed.push(2);
        typed_spy_state
            .executed_evil_roles
            .insert(2, "Spy".to_string());
        let mut typed_spy = spy.clone();
        typed_spy
            .evil_positions
            .insert(2, "Unknown".to_string());
        assert!(validate_bishop(
            &spy_claim,
            &typed_spy,
            &typed_spy_state,
        ));

        let puppet_claim = current_bishop(1, vec![1, 2], vec!["Villager", "Minion"]);
        let puppet_state = base_state(
            3,
            vec![
                puppet_claim.clone(),
                make_card(2, "Scout", json!({})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(2);
        puppet.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_bishop(&puppet_claim, &puppet, &puppet_state));

        let twin_claim = current_bishop(4, vec![1, 2], vec!["Villager", "Minion"]);
        let mut twin_state = base_state(
            4,
            vec![
                make_card(1, "Scout", json!({})),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Pooka", json!({})),
                twin_claim.clone(),
            ],
        );
        twin_state.deck.villagers = vec!["Scout".to_string(), "Bishop".to_string()];
        twin_state.deck.minions = vec!["Twin Minion".to_string()];
        twin_state.deck.demons = vec!["Pooka".to_string()];
        let mut twin = empty_scenario();
        twin.evil_positions.insert(1, "Twin Minion".to_string());
        twin.evil_positions.insert(3, "Pooka".to_string());
        twin.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Scout".to_string(),
            },
        });
        assert!(validate_bishop(&twin_claim, &twin, &twin_state));

        // InitWithNoReset preserves the physical Spy's live Villager
        // registerAs even after Twin moves Twin data onto that body. Bishop's
        // GetCharacterData projection must consult that cache before current
        // data for both truthful category selection and bluff target choice.
        let cached_spy_truth = current_bishop(4, vec![2, 3], vec!["Villager", "Demon"]);
        let mut cached_spy_state = base_state(
            4,
            vec![
                make_card(1, "Spy", json!({})),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Pooka", json!({})),
                cached_spy_truth.clone(),
            ],
        );
        cached_spy_state.deck.villagers = vec!["Bishop".to_string()];
        cached_spy_state.deck.minions =
            vec!["Twin Minion".to_string(), "Spy".to_string()];
        cached_spy_state.deck.demons = vec!["Pooka".to_string()];
        set_current_bishop_authored_good_counts(&mut cached_spy_state, 1, 0);
        let mut cached_spy = empty_scenario();
        cached_spy
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        cached_spy.evil_positions.insert(2, "Spy".to_string());
        cached_spy.evil_positions.insert(3, "Pooka".to_string());
        cached_spy.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Spy".to_string(),
            },
        });
        assert!(validate_bishop(
            &cached_spy_truth,
            &cached_spy,
            &cached_spy_state,
        ));

        let cached_spy_bluff = current_bishop(4, vec![1, 2], vec!["Villager", "Minion"]);
        cached_spy_state.cards[3] = cached_spy_bluff.clone();
        cached_spy.corrupted.insert(4);
        assert!(validate_bishop(
            &cached_spy_bluff,
            &cached_spy,
            &cached_spy_state,
        ));

        let shaman_claim = current_bishop(1, vec![2, 4], vec!["Villager", "Minion"]);
        let shaman_state = base_state(
            4,
            vec![
                shaman_claim.clone(),
                make_card(2, "Baker", json!({})),
                make_card(3, "Scout", json!({})),
                make_card(4, "Shaman", json!({})),
            ],
        );
        let mut shaman = empty_scenario();
        shaman.evil_positions.insert(4, "Shaman".to_string());
        shaman.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 3,
            target_position: 2,
            copied_role: "Scout".to_string(),
            target_previous_roles: vec!["Baker".to_string()],
        });
        assert!(validate_bishop(&shaman_claim, &shaman, &shaman_state));
    }

    #[test]
    fn current_bishop_shares_anonymous_wretch_and_baker_spy_worlds() {
        let bishop = current_bishop(1, vec![2, 5], vec!["Villager", "Minion"]);
        let mut bounty = current_poet("Bounty Hunter", json!({"evil_position": 2}));
        bounty.position = 4;
        let mut state = base_state(
            6,
            vec![
                bishop.clone(),
                make_card(3, "Pooka", json!({})),
                bounty.clone(),
                make_card(5, "Witch", json!({})),
            ],
        );
        state.deck.villagers = vec!["Bishop".to_string(), "Scout".to_string()];
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.minions = vec!["Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        set_current_bishop_authored_good_counts(&mut state, 2, 1);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.evil_positions.insert(5, "Witch".to_string());
        assert!(validate_bishop(&bishop, &scenario, &state));
        assert!(validate_poet(&bounty, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        bounty = current_poet("Bounty Hunter", json!({"evil_position": 6}));
        bounty.position = 4;
        state.cards[2] = bounty.clone();
        assert!(validate_poet(&bounty, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        let timeline_claim = current_bishop(3, vec![3, 8], vec!["Villager", "Demon"]);
        let mut timeline_state = base_state(
            8,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Scout", json!({})),
                timeline_claim.clone(),
                make_card(4, "Baker", json!({"original_role": "Spy"})),
                make_card(5, "Lover", json!({})),
                make_card(6, "Oracle", json!({})),
                make_card(7, "Bard", json!({})),
                make_card(8, "Pooka", json!({})),
            ],
        );
        timeline_state.deck.villagers = vec![
            "Baker".to_string(),
            "Scout".to_string(),
            "Bishop".to_string(),
            "Lover".to_string(),
            "Oracle".to_string(),
            "Bard".to_string(),
        ];
        timeline_state.deck.minions = vec!["Spy".to_string()];
        timeline_state.deck.demons = vec!["Pooka".to_string()];
        timeline_state.n_evil = 2;
        timeline_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        timeline_state.reveal_order = vec![1, 3, 2, 5, 6, 7, 4, 8];
        set_current_bishop_authored_good_counts(&mut timeline_state, 6, 0);
        let mut timeline = empty_scenario();
        timeline.evil_positions.insert(4, "Spy".to_string());
        timeline.evil_positions.insert(8, "Pooka".to_string());
        assert!(validate_bishop(&timeline_claim, &timeline, &timeline_state));
        timeline_state.reveal_order.retain(|position| *position != 3);
        assert!(!validate_bishop(&timeline_claim, &timeline, &timeline_state));
    }

    #[test]
    fn current_confessor_schema_text_provenance_and_archive_fallback_are_exact() {
        let scenario = empty_scenario();
        let good = current_confessor(1, json!(false));
        let state = base_state(3, vec![good.clone()]);
        assert!(validate_confessor(&good, &scenario, &state));

        let dizzy = current_confessor(1, json!(true));
        let mut corrupted = empty_scenario();
        corrupted.corrupted.insert(1);
        let dizzy_state = base_state(3, vec![dizzy.clone()]);
        assert!(validate_confessor(&dizzy, &corrupted, &dizzy_state));

        for mutation in [
            "wrong_text",
            "extra_ref",
            "future",
            "mixed",
            "wrong_role",
            "missing",
        ] {
            let mut malformed = current_confessor(1, json!(false));
            match mutation {
                "wrong_text" => malformed.info_text.push('.'),
                // Rust CardInfo does not serialize ActedInfo references. Any
                // bridge attempt to wrap the native null refs is nevertheless
                // rejected as an extra current-schema field.
                "extra_ref" => {
                    malformed
                        .info_parsed
                        .insert("targets".to_string(), json!([]));
                }
                "future" => {
                    malformed
                        .info_parsed
                        .insert("confessor_variant".to_string(), json!("future"));
                }
                "mixed" => {
                    malformed
                        .info_parsed
                        .insert("poet_variant".to_string(), json!("public_current"));
                }
                "wrong_role" => malformed.apparent_role = "confessor".to_string(),
                "missing" => {
                    malformed.info_parsed.remove("dizzy");
                }
                _ => unreachable!(),
            }
            let malformed_state = base_state(3, vec![malformed.clone()]);
            assert!(
                !validate_confessor(&malformed, &scenario, &malformed_state),
                "{mutation}"
            );
        }
        for malformed_claim in [json!(0), json!(1), json!("true"), json!(null)] {
            let malformed = current_confessor(1, malformed_claim);
            let malformed_state = base_state(3, vec![malformed.clone()]);
            assert!(!validate_confessor(
                &malformed,
                &scenario,
                &malformed_state,
            ));
        }
        for position in [0, 4] {
            let malformed = current_confessor(position, json!(false));
            let malformed_state = base_state(3, vec![malformed.clone()]);
            assert!(!validate_confessor(
                &malformed,
                &scenario,
                &malformed_state,
            ));
        }

        let mut current_poet = make_card(
            1,
            "Poet",
            json!({
                "poet_variant": "public_current",
                "copied_role": "Confessor",
                "dizzy": false,
            }),
        );
        current_poet.info_text = "I am Good".to_string();
        let current_poet_state = base_state(3, vec![current_poet.clone()]);
        assert!(!validate_poet(
            &current_poet,
            &scenario,
            &current_poet_state,
        ));

        // Frozen direct aliases, missing scalars, and obsolete Poet captures
        // stay byte-for-byte on the prior permissive predicate.
        let dirty = make_card(1, "Confessor", json!({"dirty": false}));
        let missing = make_card(1, "Confessor", json!({}));
        let archived_poet = make_card(
            1,
            "Poet",
            json!({"copied_role": "Confessor", "dizzy": false}),
        );
        let legacy_state = base_state(
            3,
            vec![dirty.clone(), missing.clone(), archived_poet.clone()],
        );
        assert!(validate_confessor(&dirty, &scenario, &legacy_state));
        assert!(validate_confessor(&missing, &scenario, &legacy_state));
        assert!(validate_poet(&archived_poet, &scenario, &legacy_state));
    }

    #[test]
    fn current_confessor_uses_corruption_registered_alignment_and_spy_exception() {
        let good = current_confessor(1, json!(false));
        let dizzy = current_confessor(1, json!(true));
        let mut state = base_state(3, vec![good.clone()]);
        state.deck.villagers = vec!["Confessor".to_string()];
        assert!(validate_confessor(&good, &empty_scenario(), &state));
        assert!(!validate_confessor(&dizzy, &empty_scenario(), &state));

        let mut corrupted = empty_scenario();
        corrupted.corrupted.insert(1);
        state.cards[0] = dizzy.clone();
        assert!(validate_confessor(&dizzy, &corrupted, &state));
        state.cards[0] = good.clone();
        assert!(!validate_confessor(&good, &corrupted, &state));

        let mut runtime_evil = empty_scenario();
        runtime_evil.evil_positions.insert(1, "Pooka".to_string());
        state.cards[0] = dizzy.clone();
        state.deck.demons = vec!["Pooka".to_string()];
        assert!(validate_confessor(&dizzy, &runtime_evil, &state));

        let mut spy = empty_scenario();
        spy.evil_positions.insert(1, "Spy".to_string());
        spy.corrupted.insert(1);
        state.cards[0] = good.clone();
        state.deck.minions = vec!["Spy".to_string()];
        // Current real Spy data wins before both corruption and registered
        // alignment in Confessor's native predicate.
        assert!(validate_confessor(&good, &spy, &state));
        state.cards[0] = dizzy.clone();
        assert!(!validate_confessor(&dizzy, &spy, &state));

        let mut wretch = empty_scenario();
        wretch.chancellor_trace = Some(crate::types::ChancellorTrace {
            original_positions: vec![3],
            added_outcast_position: 1,
            added_outcast_role: "Wretch".to_string(),
            affected_anchor_positions: vec![2],
        });
        state.cards[0] = dizzy.clone();
        state.deck.outcasts = vec!["Wretch".to_string()];
        // Wretch is runtime Good but its live registerAs is Evil. A represented
        // mover can preserve the raw Confessor provider while exposing that
        // registered-alignment surface.
        assert!(validate_confessor(&dizzy, &wretch, &state));
    }

    #[test]
    fn current_confessor_handles_puppet_drunk_and_doppelganger_surfaces() {
        let good = current_confessor(1, json!(false));
        let dizzy = current_confessor(1, json!(true));
        let mut state = base_state(3, vec![dizzy.clone()]);
        state.deck.villagers = vec!["Confessor".to_string()];

        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(1);
        assert!(validate_confessor(&dizzy, &puppet, &state));
        state.cards[0] = good.clone();
        assert!(!validate_confessor(&good, &puppet, &state));

        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(1);
        assert!(validate_confessor(&good, &drunk, &state));
        state.cards[0] = dizzy.clone();
        assert!(!validate_confessor(&dizzy, &drunk, &state));
        drunk.corrupted.insert(1);
        assert!(validate_confessor(&dizzy, &drunk, &state));

        let mut doppelganger = empty_scenario();
        doppelganger.doppelganger_position = Some(1);
        state.cards[0] = good.clone();
        assert!(validate_confessor(&good, &doppelganger, &state));
        doppelganger.corrupted.insert(1);
        state.cards[0] = dizzy.clone();
        assert!(validate_confessor(&dizzy, &doppelganger, &state));
    }

    #[test]
    fn current_confessor_follows_exact_twin_and_shaman_current_data() {
        let dizzy = current_confessor(1, json!(true));
        let mut state = base_state(4, vec![dizzy.clone()]);
        state.deck.villagers = vec!["Confessor".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        let mut twin = empty_scenario();
        twin.evil_positions.insert(1, "Twin Minion".to_string());
        twin.evil_positions.insert(3, "Pooka".to_string());
        twin.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Confessor".to_string(),
            },
        });
        assert!(validate_confessor(&dizzy, &twin, &state));

        // The Good neighbour now carries Twin data. A preserved/raw Confessor
        // callback evaluates the neighbour's Good registered alignment.
        let good_neighbor = current_confessor(2, json!(false));
        state.cards = vec![good_neighbor.clone()];
        assert!(validate_confessor(&good_neighbor, &twin, &state));

        let copied = current_confessor(2, json!(false));
        state.cards = vec![copied.clone()];
        let mut shaman = empty_scenario();
        shaman.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 4,
            target_position: 2,
            copied_role: "Confessor".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });
        assert!(validate_confessor(&copied, &shaman, &state));
        shaman.corrupted.insert(2);
        let copied_dizzy = current_confessor(2, json!(true));
        state.cards[0] = copied_dizzy.clone();
        assert!(validate_confessor(&copied_dizzy, &shaman, &state));
    }

    #[test]
    fn current_confessor_baker_conversion_clears_spy_raw_provider() {
        let observation = current_confessor(2, json!(false));
        let mut before_state = base_state(3, vec![observation.clone()]);
        before_state.deck.villagers = vec!["Confessor".to_string()];
        before_state.deck.minions = vec!["Spy".to_string()];
        let mut before = empty_scenario();
        before.evil_positions.insert(2, "Spy".to_string());
        assert!(validate_confessor(&observation, &before, &before_state));

        let mut converted_state = base_state(
            3,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        converted_state.deck.villagers =
            vec!["Baker".to_string(), "Confessor".to_string()];
        converted_state.deck.minions = vec!["Spy".to_string()];
        converted_state.deck.demons = vec!["Pooka".to_string()];
        converted_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        converted_state.reveal_order = vec![1, 2, 3];
        let mut converted = empty_scenario();
        converted.evil_positions.insert(2, "Spy".to_string());
        converted.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_baker_history(&converted, &converted_state));
        assert!(current_confessor_supports(&observation, &converted, &converted_state).is_empty());
    }

    #[test]
    fn current_confessor_raw_identity_and_identical_callback_order_join_globally() {
        let confessor = current_confessor(1, json!(true));
        let mut medium = current_medium(4, json!(1), json!("Judge"));
        let mut state = base_state(5, vec![confessor.clone(), medium.clone()]);
        state.deck.villagers = vec!["Confessor".to_string(), "Medium".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        let mut raw_only = empty_scenario();
        raw_only.evil_positions.insert(1, "Pooka".to_string());
        raw_only.corrupted.insert(4);
        assert!(validate_confessor(&confessor, &raw_only, &state));
        assert!(validate_medium(&medium, &raw_only, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &raw_only,
            &state,
        ));
        medium = current_medium(4, json!(1), json!("Confessor"));
        state.cards[1] = medium.clone();
        assert!(validate_medium(&medium, &raw_only, &state));
        assert!(validate_current_hidden_surface_consistency(
            &raw_only,
            &state,
        ));

        let mut real_and_raw = empty_scenario();
        real_and_raw
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        real_and_raw.evil_positions.insert(3, "Pooka".to_string());
        real_and_raw.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Confessor".to_string(),
            },
        });
        real_and_raw.corrupted.insert(4);
        state.deck.minions = vec!["Twin Minion".to_string()];
        let supports = current_confessor_supports(&confessor, &real_and_raw, &state);
        assert!(supports.iter().all(|support| {
            support
                .raw_bluff
                .as_ref()
                .is_some_and(|(_, role)| role == "confessor")
        }));
        assert!(supports
            .iter()
            .all(|support| support.forbidden_raw_bluff.is_none()));
        // The raw callback is proven at this runtime-Evil Twin endpoint. A
        // different provider would overwrite the real Confessor callback, so
        // the global assignment must reject a Medium clue pinning Judge here.
        let pinned_judge = current_medium(4, json!(1), json!("Judge"));
        state.cards[1] = pinned_judge;
        assert!(!validate_current_hidden_surface_consistency(
            &real_and_raw,
            &state,
        ));

        // Raw Confessor runs later but computes the identical native predicate,
        // leaving the exact newest payload unchanged.
        state.cards[1] = current_medium(4, json!(1), json!("Confessor"));
        assert!(validate_current_hidden_surface_consistency(
            &real_and_raw,
            &state,
        ));
    }

    #[test]
    fn current_confessor_rejects_unresolved_start_identity() {
        let direct = current_confessor(1, json!(false));
        let state = base_state(3, vec![direct.clone()]);
        let mut unresolved = empty_scenario();
        unresolved.evil_positions.insert(2, "Unknown".to_string());
        assert!(!validate_confessor(&direct, &unresolved, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &unresolved,
            &state,
        ));

        unresolved.evil_positions.insert(2, "Witch".to_string());
        assert!(validate_confessor(&direct, &unresolved, &state));
    }

    #[test]
    fn current_bard_schema_text_source_and_archive_fallback_are_exact() {
        let scenario = empty_scenario();
        for claim in [json!(-1), json!(1), json!(2), json!(3)] {
            let card = current_bard(1, claim);
            let state = base_state(6, vec![card.clone()]);
            assert_eq!(
                validate_bard(&card, &scenario, &state),
                card.info_parsed["corruption_distance"] == json!(-1),
            );
        }

        let mut singular_scenario = empty_scenario();
        singular_scenario.corrupted.insert(2);
        let singular = current_bard(1, json!(1));
        let singular_state = base_state(6, vec![singular.clone()]);
        assert!(validate_bard(&singular, &singular_scenario, &singular_state));

        let mut plural_scenario = empty_scenario();
        plural_scenario.corrupted.insert(3);
        let plural = current_bard(1, json!(2));
        let plural_state = base_state(6, vec![plural.clone()]);
        assert!(validate_bard(&plural, &plural_scenario, &plural_state));

        let poet = current_poet("Bard", json!({"corruption_distance": 2}));
        let poet_state = base_state(6, vec![poet.clone()]);
        assert!(validate_poet(&poet, &plural_scenario, &poet_state));

        for mutation in ["wrong_text", "extra", "future", "mixed", "wrong_role"] {
            let mut malformed = current_bard(1, json!(2));
            match mutation {
                "wrong_text" => malformed.info_text.push('.'),
                "extra" => {
                    malformed.info_parsed.insert("extra".to_string(), json!(true));
                }
                "future" => {
                    malformed
                        .info_parsed
                        .insert("bard_variant".to_string(), json!("future"));
                }
                "mixed" => {
                    malformed
                        .info_parsed
                        .insert("poet_variant".to_string(), json!("public_current"));
                }
                "wrong_role" => malformed.apparent_role = "bard".to_string(),
                _ => unreachable!(),
            }
            let state = base_state(6, vec![malformed.clone()]);
            assert!(
                !validate_bard(&malformed, &plural_scenario, &state),
                "{mutation}"
            );
        }

        for claim in [json!(-2), json!(0), json!(4), json!(true), json!("2")] {
            let malformed = current_bard(1, claim);
            let state = base_state(6, vec![malformed.clone()]);
            assert!(!validate_bard(&malformed, &plural_scenario, &state));
        }
        for position in [0, 7] {
            let mut malformed = current_bard(position, json!(-1));
            malformed.position = position;
            let state = base_state(6, vec![malformed.clone()]);
            assert!(!validate_bard(&malformed, &scenario, &state));
        }

        // Frozen captures retain their original fail-open/mismatch behavior,
        // including a missing scalar and the old zero encoding.
        let missing = make_card(1, "Bard", json!({}));
        let missing_state = base_state(2, vec![missing.clone()]);
        assert!(validate_bard(&missing, &scenario, &missing_state));
        let zero = make_card(1, "Bard", json!({"corruption_distance": 0}));
        let mut legacy_world = empty_scenario();
        legacy_world.evil_positions.insert(1, "Pooka".to_string());
        legacy_world.corrupted.insert(2);
        let zero_state = base_state(2, vec![zero.clone()]);
        assert!(validate_bard(&zero, &legacy_world, &zero_state));
    }

    #[test]
    fn current_bard_truth_uses_other_physical_corruption_and_full_lifecycle() {
        let far = current_bard(1, json!(4));
        let mut state = base_state(8, vec![far.clone()]);
        state.executed = vec![5];
        state.night_kills = vec![5];
        state.blocked_positions = vec![5];
        let mut scenario = empty_scenario();
        scenario.corrupted.insert(5);
        assert!(validate_bard(&far, &scenario, &state));

        let tie = current_bard(1, json!(2));
        state.cards = vec![tie.clone()];
        scenario.corrupted = HashSet::from([3, 7]);
        assert!(validate_bard(&tie, &scenario, &state));

        let wrap = current_bard(1, json!(1));
        state.cards = vec![wrap.clone()];
        scenario.corrupted = HashSet::from([8]);
        assert!(validate_bard(&wrap, &scenario, &state));

        // Self is removed from the distance scan, but its Corrupted status
        // still routes the provider through BluffAct. Native removes truth 0
        // from its fixed bluff domain.
        let false_one = current_bard(1, json!(1));
        state.cards = vec![false_one.clone()];
        scenario.corrupted = HashSet::from([1]);
        assert!(validate_bard(&false_one, &scenario, &state));
        let actual_zero = current_bard(1, json!(-1));
        state.cards = vec![actual_zero.clone()];
        assert!(!validate_bard(&actual_zero, &scenario, &state));
    }

    #[test]
    fn current_bard_bluff_domain_is_fixed_and_supports_tiny_boards() {
        let mut far_world = empty_scenario();
        far_world.corrupted = HashSet::from([1, 5]);
        for claim in [-1, 1, 2, 3] {
            let card = current_bard(1, json!(claim));
            let state = base_state(8, vec![card.clone()]);
            assert!(validate_bard(&card, &far_world, &state), "claim {claim}");
        }
        let truth_value = current_bard(1, json!(4));
        let truth_value_state = base_state(8, vec![truth_value.clone()]);
        assert!(!validate_bard(&truth_value, &far_world, &truth_value_state));

        let mut singleton_world = empty_scenario();
        singleton_world.corrupted.insert(1);
        for claim in [1, 2, 3] {
            let card = current_bard(1, json!(claim));
            let state = base_state(1, vec![card.clone()]);
            assert!(validate_bard(&card, &singleton_world, &state));
        }
        let singleton_truth = current_bard(1, json!(-1));
        let singleton_state = base_state(1, vec![singleton_truth.clone()]);
        assert!(!validate_bard(
            &singleton_truth,
            &singleton_world,
            &singleton_state,
        ));

        let mut pair_world = empty_scenario();
        pair_world.corrupted = HashSet::from([1, 2]);
        for claim in [-1, 2, 3] {
            let card = current_bard(1, json!(claim));
            let state = base_state(2, vec![card.clone()]);
            assert!(validate_bard(&card, &pair_world, &state));
        }
        let pair_truth = current_bard(1, json!(1));
        let pair_state = base_state(2, vec![pair_truth.clone()]);
        assert!(!validate_bard(&pair_truth, &pair_world, &pair_state));
    }

    #[test]
    fn current_bard_puppet_and_exact_twin_data_follow_physical_truth_state() {
        let truthful_puppet = current_bard(2, json!(2));
        let mut state = base_state(5, vec![truthful_puppet.clone()]);
        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(2);
        puppet.corrupted.insert(5);
        assert!(validate_bard(&truthful_puppet, &puppet, &state));
        let false_puppet = current_bard(2, json!(1));
        state.cards = vec![false_puppet.clone()];
        assert!(!validate_bard(&false_puppet, &puppet, &state));
        puppet.corrupted.insert(2);
        assert!(validate_bard(&false_puppet, &puppet, &state));
        state.cards = vec![truthful_puppet.clone()];
        assert!(!validate_bard(&truthful_puppet, &puppet, &state));

        let moved_bard = current_bard(1, json!(1));
        let moved_state = base_state(4, vec![moved_bard.clone()]);
        let mut moved = empty_scenario();
        moved.evil_positions.insert(1, "Twin Minion".to_string());
        moved.evil_positions.insert(3, "Pooka".to_string());
        moved.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Bard".to_string(),
            },
        });
        assert!(validate_bard(&moved_bard, &moved, &moved_state));
        let real_truth = current_bard(1, json!(-1));
        let real_truth_state = base_state(4, vec![real_truth.clone()]);
        assert!(validate_bard(&real_truth, &moved, &real_truth_state));

        let copied_bard = current_bard(2, json!(-1));
        let copied_state = base_state(4, vec![copied_bard.clone()]);
        let mut copied = empty_scenario();
        copied.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 4,
            target_position: 2,
            copied_role: "Bard".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });
        assert!(validate_bard(&copied_bard, &copied, &copied_state));
    }

    #[test]
    fn current_bard_runtime_evil_current_data_respects_callback_order() {
        let truth = current_bard(1, json!(3));
        let mut medium = current_medium(4, json!(1), json!("Bard"));
        let mut state = base_state(6, vec![truth.clone(), medium.clone()]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(1, "Twin Minion".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Bard".to_string(),
            },
        });
        scenario.corrupted.insert(4);

        // Runtime Evil dispatches the real current-data Bard through Act
        // first. A later raw Bard would overwrite that truth with BluffAct,
        // so the truthful newest event proves the raw pointer is not Bard.
        assert!(validate_bard(&truth, &scenario, &state));
        assert!(validate_medium(&medium, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));
        medium = current_medium(4, json!(1), json!("Judge"));
        state.cards[1] = medium.clone();
        assert!(validate_medium(&medium, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));

        // Conversely, a false Bard result can only be the later raw Bard
        // BluffAct. It binds that same pointer to Bard in the global world.
        let lie = current_bard(1, json!(1));
        state.cards[0] = lie.clone();
        assert!(validate_bard(&lie, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));
        medium = current_medium(4, json!(1), json!("Bard"));
        state.cards[1] = medium;
        assert!(validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_bard_raw_bluff_identity_joins_medium_support() {
        let bard = current_bard(1, json!(1));
        let medium = current_medium(4, json!(1), json!("Judge"));
        let mut state = base_state(6, vec![bard.clone(), medium.clone()]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(1, "Pooka".to_string());
        scenario.corrupted = HashSet::from([3, 4]);

        // Stable Evil #1 reaches the apparent Bard through its non-null raw
        // bluff. Lying Medium can select the same pointer independently, but
        // one physical pointer cannot simultaneously be Bard and Judge.
        assert!(validate_bard(&bard, &scenario, &state));
        assert!(validate_medium(&medium, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));

        let matching_medium = current_medium(4, json!(1), json!("Bard"));
        state.cards[1] = matching_medium.clone();
        assert!(validate_medium(&matching_medium, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_bard_rejects_baker_cleared_spy_provider_surface() {
        let bard_observation = current_bard(2, json!(2));
        let mut before_state = base_state(3, vec![bard_observation.clone()]);
        before_state.deck.minions = vec!["Spy".to_string()];
        let mut before = empty_scenario();
        before.evil_positions.insert(2, "Spy".to_string());
        before.corrupted.insert(3);
        // Before any Baker conversion, Spy's acquired Bard raw bluff reaches
        // Bard.BluffAct and supports the false distance.
        assert!(validate_bard(&bard_observation, &before, &before_state));

        let mut converted_state = base_state(
            3,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        converted_state.deck.villagers = vec!["Baker".to_string(), "Bard".to_string()];
        converted_state.deck.minions = vec!["Spy".to_string()];
        converted_state.deck.demons = vec!["Pooka".to_string()];
        converted_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        converted_state.reveal_order = vec![1, 2, 3];
        let mut converted = empty_scenario();
        converted.evil_positions.insert(2, "Spy".to_string());
        converted.evil_positions.insert(3, "Pooka".to_string());
        converted.corrupted.insert(3);
        assert!(validate_baker_history(&converted, &converted_state));
        // The strict observation object is deliberately separate from the
        // final Baker CardInfo. Every exact history has already synchronously
        // cleared Spy's raw bluff before #2 can observe Day, so no strict Bard
        // provider support remains.
        assert!(current_bard_supports(
            &bard_observation,
            &converted,
            &converted_state,
            CurrentPassivePayloadSource::Direct,
        )
        .is_empty());
    }

    #[test]
    fn current_bard_rejects_unresolved_start_identity_direct_and_poet() {
        let direct = current_bard(1, json!(-1));
        let mut poet = current_poet("Bard", json!({"corruption_distance": -1}));
        poet.position = 3;
        let state = base_state(3, vec![direct.clone(), poet.clone()]);
        let mut unresolved = empty_scenario();
        unresolved.evil_positions.insert(2, "Unknown".to_string());
        assert!(!validate_bard(&direct, &unresolved, &state));
        assert!(!validate_poet(&poet, &unresolved, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &unresolved,
            &state,
        ));

        let mut concrete = unresolved;
        concrete.evil_positions.insert(2, "Witch".to_string());
        assert!(validate_bard(&direct, &concrete, &state));
        assert!(validate_poet(&poet, &concrete, &state));
    }

    #[test]
    fn current_poet_schema_accepts_every_complete_native_provider_payload() {
        let state = base_state(6, vec![]);
        for (provider, payload) in [
            ("Lover", json!({"evil_adjacent": 1})),
            ("Scout", json!({"evil_role": "Pooka", "distance": 2})),
            (
                "Oracle",
                json!({"targets": [2, 3], "minion_role": "Witch"}),
            ),
            ("Bounty Hunter", json!({"evil_position": 4})),
            (
                "Medium",
                json!({"good_position": 2, "good_role": "Scout"}),
            ),
            ("Knitter", json!({"evil_pairs": 1})),
            ("Hunter", json!({"distance": 2})),
            ("Enlightened", json!({"direction": "CCW"})),
            ("Empress", json!({"targets": [2, 3, 4]})),
            (
                "Bishop",
                json!({
                    "targets": [2, 3, 4],
                    "types": ["Villager", "Villager", "Minion"],
                }),
            ),
            ("Gemcrafter", json!({"good_position": 2})),
            ("Bard", json!({"corruption_distance": -1})),
        ] {
            let poet = current_poet(provider, payload);
            assert!(
                validate_current_poet_payload(&poet, &state, provider),
                "current Poet provider {provider} must have a complete schema"
            );
        }

        let scout_with_good_current_data = current_poet(
            "Scout",
            json!({"evil_role": "Scout", "distance": 2}),
        );
        assert!(validate_current_poet_payload(
            &scout_with_good_current_data,
            &state,
            "Scout",
        ));

        let scout_sentinel = current_poet("Scout", json!({"one_evil": true}));
        assert!(validate_current_poet_payload(
            &scout_sentinel,
            &state,
            "Scout",
        ));

        let hunter_no_other_evil = current_poet("Hunter", json!({"distance": 5}));
        assert!(validate_current_poet_payload(
            &hunter_no_other_evil,
            &state,
            "Hunter",
        ));

        let duplicate_oracle = current_poet(
            "Oracle",
            json!({"targets": [2, 2], "minion_role": "Twin Minion"}),
        );
        assert!(validate_current_poet_payload(
            &duplicate_oracle,
            &state,
            "Oracle",
        ));
    }

    #[test]
    fn current_poet_schema_requires_exact_keys_and_canonical_provider_names() {
        let state = base_state(6, vec![]);
        let valid = [
            ("Lover", json!({"evil_adjacent": 1}), vec!["evil_adjacent"]),
            (
                "Scout",
                json!({"evil_role": "Pooka", "distance": 2}),
                vec!["evil_role", "distance"],
            ),
            (
                "Oracle",
                json!({"targets": [2, 3], "minion_role": "Witch"}),
                vec!["targets", "minion_role"],
            ),
            (
                "Bounty Hunter",
                json!({"evil_position": 4}),
                vec!["evil_position"],
            ),
            (
                "Medium",
                json!({"good_position": 2, "good_role": "Scout"}),
                vec!["good_position", "good_role"],
            ),
            ("Knitter", json!({"evil_pairs": 1}), vec!["evil_pairs"]),
            ("Hunter", json!({"distance": 2}), vec!["distance"]),
            (
                "Enlightened",
                json!({"direction": "CCW"}),
                vec!["direction"],
            ),
            (
                "Empress",
                json!({"targets": [2, 3, 4]}),
                vec!["targets"],
            ),
            (
                "Bishop",
                json!({
                    "targets": [2, 3, 4],
                    "types": ["Villager", "Outcast", "Minion"],
                }),
                vec!["targets", "types"],
            ),
            (
                "Gemcrafter",
                json!({"good_position": 2}),
                vec!["good_position"],
            ),
            (
                "Bard",
                json!({"corruption_distance": -1}),
                vec!["corruption_distance"],
            ),
        ];

        for (provider, payload, required_fields) in valid {
            for field in required_fields {
                let mut missing = current_poet(provider, payload.clone());
                missing.info_parsed.remove(field);
                assert!(
                    !validate_current_poet_payload(&missing, &state, provider),
                    "{provider} must require {field}"
                );
            }

            let mut extra = current_poet(provider, payload);
            extra.info_parsed.insert("unexpected".to_string(), json!(true));
            assert!(
                !validate_current_poet_payload(&extra, &state, provider),
                "{provider} must reject extra payload keys"
            );
        }

        for (provider, payload) in [
            ("lover", json!({"evil_adjacent": 1})),
            ("SCOUT", json!({"evil_role": "Pooka", "distance": 2})),
            ("Bounty_Hunter", json!({"evil_position": 4})),
            ("Gem Crafter", json!({"good_position": 2})),
        ] {
            let poet = current_poet(provider, payload);
            assert!(!validate_poet(&poet, &empty_scenario(), &state));
        }
    }

    #[test]
    fn current_poet_schema_rejects_wrapping_or_malformed_positions_and_targets() {
        let state = base_state(6, vec![]);

        for position in [0, 7] {
            let mut poet = current_poet("Bounty Hunter", json!({"evil_position": 2}));
            poet.position = position;
            assert!(!validate_current_poet_payload(
                &poet,
                &state,
                "Bounty Hunter"
            ));
        }

        for value in [json!(0), json!(-1), json!(7), json!(256), json!(true), json!("2")] {
            for (provider, field) in [
                ("Bounty Hunter", "evil_position"),
                ("Gemcrafter", "good_position"),
            ] {
                let mut payload = serde_json::Map::new();
                payload.insert(field.to_string(), value.clone());
                let poet = current_poet(provider, serde_json::Value::Object(payload));
                assert!(
                    !validate_current_poet_payload(&poet, &state, provider),
                    "{provider} must reject {field}={value}"
                );
            }
        }

        for targets in [
            json!([2]),
            json!([0, 2]),
            json!([2, 7]),
            json!([2, -1]),
            json!([2, true]),
            json!([2, "3"]),
            json!([2, 3, 4]),
        ] {
            let oracle = current_poet(
                "Oracle",
                json!({"targets": targets, "minion_role": "Witch"}),
            );
            assert!(!validate_current_poet_payload(&oracle, &state, "Oracle"));
        }

        for targets in [json!([2, 3]), json!([2, 2, 4]), json!([2, 3, 7])] {
            let empress = current_poet("Empress", json!({"targets": targets}));
            assert!(!validate_current_poet_payload(&empress, &state, "Empress"));
        }

        for payload in [
            json!({"targets": [], "types": []}),
            json!({"targets": [2, 2], "types": ["Villager", "Minion"]}),
            json!({"targets": [2, 3], "types": ["Villager"]}),
            json!({"targets": [2, 3, 4, 5], "types": ["Villager", "Outcast", "Minion", "Demon"]}),
            json!({"targets": [2, true], "types": ["Villager", "Minion"]}),
        ] {
            let bishop = current_poet("Bishop", payload);
            assert!(!validate_current_poet_payload(&bishop, &state, "Bishop"));
        }
    }

    #[test]
    fn current_poet_schema_rejects_noncanonical_roles_types_and_scalar_ranges() {
        let state = base_state(6, vec![]);
        for payload in [
            json!({"evil_role": "pooka", "distance": 2}),
            json!({"evil_role": "Future Demon", "distance": 2}),
            json!({"evil_role": 7, "distance": 2}),
            json!({"evil_role": "Pooka", "distance": 0}),
            json!({"evil_role": "Pooka", "distance": -1}),
            json!({"evil_role": "Pooka", "distance": 7}),
            json!({"evil_role": "Pooka", "distance": true}),
        ] {
            let scout = current_poet("Scout", payload);
            assert!(!validate_current_poet_payload(&scout, &state, "Scout"));
        }

        for minion_role in [json!("witch"), json!("Pooka"), json!("Future Minion"), json!(7)] {
            let oracle = current_poet(
                "Oracle",
                json!({"targets": [2, 3], "minion_role": minion_role}),
            );
            assert!(!validate_current_poet_payload(&oracle, &state, "Oracle"));
        }

        for good_role in [json!("scout"), json!("Future Villager"), json!(7)] {
            let medium = current_poet(
                "Medium",
                json!({"good_position": 2, "good_role": good_role}),
            );
            assert!(!validate_current_poet_payload(&medium, &state, "Medium"));
        }

        for direction in [json!("cw"), json!("Clockwise"), json!("Equal"), json!(7)] {
            let enlightened = current_poet("Enlightened", json!({"direction": direction}));
            assert!(!validate_current_poet_payload(
                &enlightened,
                &state,
                "Enlightened"
            ));
        }

        for role_types in [
            json!(["villager"]),
            json!(["Good"]),
            json!(["Villager", 7]),
        ] {
            let targets = if role_types.as_array().unwrap().len() == 1 {
                json!([2])
            } else {
                json!([2, 3])
            };
            let bishop = current_poet(
                "Bishop",
                json!({"targets": targets, "types": role_types}),
            );
            assert!(!validate_current_poet_payload(&bishop, &state, "Bishop"));
        }

        for (provider, field, value) in [
            ("Lover", "evil_adjacent", json!(-1)),
            ("Lover", "evil_adjacent", json!(3)),
            ("Knitter", "evil_pairs", json!(-1)),
            ("Knitter", "evil_pairs", json!(7)),
            ("Hunter", "distance", json!(0)),
            ("Hunter", "distance", json!(-1)),
            ("Hunter", "distance", json!(4)),
            ("Hunter", "distance", json!(7)),
            ("Bard", "corruption_distance", json!(-2)),
            ("Bard", "corruption_distance", json!(0)),
            ("Bard", "corruption_distance", json!(7)),
        ] {
            let mut payload = serde_json::Map::new();
            payload.insert(field.to_string(), value);
            let poet = current_poet(provider, serde_json::Value::Object(payload));
            assert!(!validate_current_poet_payload(&poet, &state, provider));
        }
    }

    #[test]
    fn current_poet_rejects_obsolete_unknown_and_malformed_provenance() {
        let state = base_state(3, vec![]);
        let scenario = empty_scenario();

        for provider in [
            "Architect",
            "Fortune Teller",
            "Baker",
            "Confessor",
            "Future Oracle",
        ] {
            let poet = make_card(
                1,
                "Poet",
                json!({
                    "poet_variant": "public_current",
                    "copied_role": provider,
                }),
            );
            assert!(
                !validate_poet(&poet, &scenario, &state),
                "current Poet provider {provider} must be rejected"
            );
        }

        for info in [
            json!({"poet_variant": "public_current"}),
            json!({"poet_variant": "public_current", "copied_role": 7}),
            json!({"poet_variant": "legacy", "copied_role": "Scout"}),
            json!({"poet_variant": "future", "copied_role": "Scout"}),
            json!({"poet_variant": 1, "copied_role": "Scout"}),
        ] {
            let poet = make_card(1, "Poet", info);
            assert!(!validate_poet(&poet, &scenario, &state));
        }
    }

    #[test]
    fn unmarked_poet_preserves_archived_provider_and_empty_fallbacks() {
        let state = base_state(3, vec![]);
        let scenario = empty_scenario();
        let archived = [
            json!({"copied_role": "Architect", "side": "Equal"}),
            json!({
                "copied_role": "Fortune Teller",
                "targets": [2, 3],
                "has_evil": false,
            }),
            json!({"copied_role": "Future Oracle"}),
            json!({
                "copied_role": "Oracle",
                "sentinel": "There are NO Minions",
            }),
            json!({
                "copied_role": "Hunter",
                "sentinel": "There is only 1 Evil",
            }),
            json!({"shut_up_target": 2}),
            json!({"no_info": true}),
            json!({}),
        ];

        for info in archived {
            let poet = make_card(1, "Poet", info);
            assert!(validate_poet(&poet, &scenario, &state));
        }

        for info in [
            json!({
                "poet_variant": "public_current",
                "copied_role": "Oracle",
                "sentinel": "There are NO Minions",
            }),
            json!({
                "poet_variant": "public_current",
                "copied_role": "Hunter",
                "sentinel": "There is only 1 Evil",
            }),
        ] {
            let poet = make_card(1, "Poet", info);
            assert!(!validate_poet(&poet, &scenario, &state));
        }
    }

    #[test]
    fn current_poet_provider_still_delegates_truth_and_inverse() {
        let poet = current_poet("Bounty Hunter", json!({"evil_position": 2}));
        let state = base_state(3, vec![poet.clone()]);
        let mut matching = empty_scenario();
        matching.evil_positions.insert(2, "Pooka".to_string());

        assert!(validate_poet(&poet, &matching, &state));
        assert!(!validate_poet(&poet, &empty_scenario(), &state));

        let mut lying_matching = empty_scenario();
        lying_matching.corrupted.insert(1);
        assert!(validate_poet(&poet, &lying_matching, &state));
        lying_matching
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(!validate_poet(&poet, &lying_matching, &state));
    }

    #[test]
    fn current_medium_schema_text_and_source_are_exact_and_fail_closed() {
        let direct = current_medium(1, json!(2), json!("Scout"));
        let poet = current_poet("Medium", json!({"good_position": 2, "good_role": "Scout"}));
        let target = make_card(2, "Scout", json!({}));
        let mut direct_state = base_state(2, vec![direct.clone(), target.clone()]);
        direct_state.deck.villagers = vec!["Medium".to_string(), "Scout".to_string()];
        let mut poet_state = base_state(2, vec![poet.clone(), target]);
        poet_state.deck.villagers = vec!["Poet".to_string(), "Scout".to_string()];

        assert_eq!(direct.info_text, "#2 is a real\nScout");
        assert_eq!(
            parse_current_medium_claim(&direct, CurrentPassivePayloadSource::Direct, &direct_state,),
            Some((2, "Scout")),
        );
        assert!(validate_medium(&direct, &empty_scenario(), &direct_state,));
        assert!(validate_poet(&poet, &empty_scenario(), &poet_state));

        let drunk = current_medium(1, json!(2), json!("Drunk"));
        assert_eq!(drunk.info_text, "#2 is actually a\nDrunk");
        for text in [
            "",
            "#2 is a real Scout",
            "#2 is a real\nscout",
            "#2 is a real\nScout.",
            "#2 is actually a\nScout",
        ] {
            let mut malformed = direct.clone();
            malformed.info_text = text.to_string();
            assert!(!validate_medium(
                &malformed,
                &empty_scenario(),
                &direct_state,
            ));
        }
        let mut wrong_drunk_text = drunk;
        wrong_drunk_text.info_text = "#2 is a real\nDrunk".to_string();
        assert!(!validate_medium(
            &wrong_drunk_text,
            &empty_scenario(),
            &direct_state,
        ));

        for value in [
            json!(0),
            json!(-1),
            json!(3),
            json!(256),
            json!(true),
            json!("2"),
        ] {
            let malformed = current_medium(1, value, json!("Scout"));
            assert!(!validate_medium(
                &malformed,
                &empty_scenario(),
                &direct_state,
            ));
        }
        for role in [json!("scout"), json!("Future Villager"), json!(7)] {
            let malformed = current_medium(1, json!(2), role);
            assert!(!validate_medium(
                &malformed,
                &empty_scenario(),
                &direct_state,
            ));
        }
        let mut extra = direct.clone();
        extra.info_parsed.insert("future".to_string(), json!(true));
        assert!(!validate_medium(&extra, &empty_scenario(), &direct_state));
        let future = make_card(
            1,
            "Medium",
            json!({
                "medium_variant": "future",
                "good_position": 2,
                "good_role": "Scout",
            }),
        );
        assert!(!validate_medium(&future, &empty_scenario(), &direct_state,));
    }

    #[test]
    fn current_medium_truth_removes_actor_except_for_sole_good_fallback() {
        let self_claim = current_medium(1, json!(1), json!("Medium"));
        let scout = make_card(2, "Scout", json!({}));
        let mut state = base_state(2, vec![self_claim.clone(), scout]);
        state.deck.villagers = vec!["Medium".to_string(), "Scout".to_string()];
        assert!(!validate_medium(&self_claim, &empty_scenario(), &state,));

        let mut sole_good = empty_scenario();
        sole_good.evil_positions.insert(2, "Pooka".to_string());
        assert!(validate_medium(&self_claim, &sole_good, &state));

        let other_claim = current_medium(1, json!(2), json!("Scout"));
        state.cards[0] = other_claim.clone();
        assert!(validate_medium(&other_claim, &empty_scenario(), &state,));
    }

    #[test]
    fn current_medium_direct_and_poet_use_identical_provider_semantics() {
        let direct = current_medium(1, json!(2), json!("Scout"));
        let poet = current_poet("Medium", json!({"good_position": 2, "good_role": "Scout"}));
        let target = make_card(2, "Scout", json!({}));
        let mut direct_state = base_state(2, vec![direct.clone(), target.clone()]);
        direct_state.deck.villagers = vec!["Medium".to_string(), "Scout".to_string()];
        let mut poet_state = base_state(2, vec![poet.clone(), target]);
        poet_state.deck.villagers = vec!["Poet".to_string(), "Scout".to_string()];

        assert!(validate_medium(&direct, &empty_scenario(), &direct_state));
        assert!(validate_poet(&poet, &empty_scenario(), &poet_state));

        let mut lying = empty_scenario();
        lying.corrupted.insert(1);
        assert!(!validate_medium(&direct, &lying, &direct_state));
        assert!(!validate_poet(&poet, &lying, &poet_state));
        lying.evil_positions.insert(2, "Pooka".to_string());
        assert!(validate_medium(&direct, &lying, &direct_state));
        assert!(validate_poet(&poet, &lying, &poet_state));
    }

    #[test]
    fn current_medium_truth_uses_spy_register_as_and_moved_current_data() {
        let spy_claim = current_medium(1, json!(2), json!("Scout"));
        let spy_display = make_card(2, "Judge", json!({}));
        let mut spy_state = base_state(2, vec![spy_claim.clone(), spy_display]);
        spy_state.deck.villagers = vec!["Medium".to_string(), "Scout".to_string()];
        spy_state.deck.minions = vec!["Spy".to_string()];
        let mut spy = empty_scenario();
        spy.evil_positions.insert(2, "Spy".to_string());
        assert!(validate_medium(&spy_claim, &spy, &spy_state));

        let wrong_spy_label = current_medium(1, json!(2), json!("Pooka"));
        assert!(!validate_medium(&wrong_spy_label, &spy, &spy_state,));

        let moved_claim = current_medium(1, json!(3), json!("Twin Minion"));
        let twin = make_card(2, "Twin Minion", json!({}));
        let moved_target = make_card(3, "Scout", json!({}));
        let mut moved_state = base_state(3, vec![moved_claim.clone(), twin, moved_target]);
        moved_state.deck.villagers = vec!["Medium".to_string(), "Scout".to_string()];
        moved_state.deck.minions = vec!["Twin Minion".to_string()];
        moved_state.deck.demons = vec!["Pooka".to_string()];
        let mut moved = empty_scenario();
        moved.evil_positions.insert(2, "Twin Minion".to_string());
        moved.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 2,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 1,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 3,
                neighbor_pre_swap_role: "Scout".to_string(),
            },
        });
        assert!(validate_medium(&moved_claim, &moved, &moved_state));

        let copied_spy_claim = current_medium(1, json!(3), json!("Scout"));
        let mut copied_spy_state = base_state(
            3,
            vec![
                copied_spy_claim.clone(),
                make_card(2, "Judge", json!({})),
                make_card(3, "Scout", json!({})),
            ],
        );
        copied_spy_state.deck.villagers = vec!["Medium".to_string(), "Scout".to_string()];
        let mut copied_spy = empty_scenario();
        copied_spy.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 3,
            copied_role: "Spy".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });
        assert!(validate_medium(&copied_spy_claim, &copied_spy, &copied_spy_state));

        let wretch_claim = current_medium(1, json!(2), json!("Wretch"));
        let mut wretch_state = base_state(2, vec![wretch_claim.clone(), make_card(2, "Wretch", json!({}))]);
        wretch_state.deck.villagers = vec!["Medium".to_string()];
        wretch_state.deck.outcasts = vec!["Wretch".to_string()];
        assert!(!validate_medium(&wretch_claim, &empty_scenario(), &wretch_state));
    }

    #[test]
    fn current_medium_truth_observations_share_one_anonymous_wretch_world() {
        let self_claim = current_medium(1, json!(1), json!("Medium"));
        let mut poet_claim =
            current_poet("Medium", json!({"good_position": 2, "good_role": "Scout"}));
        poet_claim.position = 3;
        let mut state = base_state(3, vec![self_claim.clone(), poet_claim.clone()]);
        state.deck.villagers = vec![
            "Medium".to_string(),
            "Poet".to_string(),
            "Scout".to_string(),
        ];
        state.deck.outcasts = vec!["Wretch".to_string()];
        let mut scenario = empty_scenario();
        scenario.puppet_position = Some(3);

        assert!(validate_medium(&self_claim, &scenario, &state));
        assert!(validate_poet(&poet_claim, &scenario, &state));
        assert!(!validate_current_medium_consistency(&scenario, &state));
    }

    #[test]
    fn current_medium_bluff_respects_nonself_priority_and_self_fallback() {
        let self_claim = current_medium(1, json!(1), json!("Scout"));
        let mut state = base_state(1, vec![self_claim.clone()]);
        state.deck.villagers = vec!["Medium".to_string(), "Scout".to_string()];
        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(1);
        assert!(validate_medium(&self_claim, &drunk, &state));

        state.n_cards = 2;
        state.cards.push(make_card(2, "Scout", json!({})));
        let mut other_holder = drunk.clone();
        other_holder.evil_positions.insert(2, "Pooka".to_string());
        assert!(!validate_medium(&self_claim, &other_holder, &state));

        state.reveal_order = vec![1, 2];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        assert!(validate_medium(&self_claim, &other_holder, &state));

        let nonself = current_medium(1, json!(2), json!("Judge"));
        state.cards[0] = nonself.clone();
        assert!(!validate_medium(&nonself, &other_holder, &state));
        state.reveal_order = vec![2, 1];
        assert!(validate_medium(&nonself, &other_holder, &state));
        state.cards.truncate(1);
        state.reveal_order = vec![1];
        assert!(!validate_medium(&nonself, &other_holder, &state));
    }

    #[test]
    fn current_medium_bluff_keeps_unmodeled_mover_surfaces_conservative() {
        let claim = current_medium(1, json!(2), json!("Judge"));
        let target = make_card(2, "Scout", json!({}));
        let third = make_card(3, "Scout", json!({}));
        let mut state = base_state(3, vec![claim.clone(), target, third]);
        state.deck.villagers = vec![
            "Medium".to_string(),
            "Scout".to_string(),
            "Judge".to_string(),
        ];
        let mut lying = empty_scenario();
        lying.corrupted.insert(1);
        assert!(!validate_medium(&claim, &lying, &state));

        lying.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 3,
            copied_role: "Scout".to_string(),
            target_previous_roles: vec!["Judge".to_string()],
        });
        assert!(validate_medium(&claim, &lying, &state));
    }

    #[test]
    fn current_medium_baker_history_compares_exact_claim_for_direct_and_poet() {
        let baker_one = make_card(1, "Baker", json!({"original_role": "Knight"}));
        let baker_two = make_card(2, "Baker", json!({"original_role": "original"}));
        let direct = current_medium(3, json!(1), json!("Baker"));
        let mut direct_state = base_state(3, vec![baker_one.clone(), baker_two.clone(), direct]);
        direct_state.deck.villagers = vec![
            "Baker".to_string(),
            "Knight".to_string(),
            "Medium".to_string(),
        ];
        direct_state.reveal_order = vec![2, 3, 1];
        direct_state.baker_rule_version = Some("baker_day_reveal_v1".to_string());
        assert!(validate_baker_history(&empty_scenario(), &direct_state));
        assert!(check_scenario(&empty_scenario(), &direct_state));

        direct_state.cards[2] = current_medium(3, json!(1), json!("Knight"));
        assert!(!validate_baker_history(&empty_scenario(), &direct_state));
        assert!(!check_scenario(&empty_scenario(), &direct_state));
        direct_state.reveal_order = vec![3, 2, 1];
        assert!(validate_baker_history(&empty_scenario(), &direct_state));

        let mut poet = current_poet("Medium", json!({"good_position": 1, "good_role": "Baker"}));
        poet.position = 3;
        let mut poet_state = base_state(3, vec![baker_one, baker_two, poet]);
        poet_state.deck.villagers = vec![
            "Baker".to_string(),
            "Knight".to_string(),
            "Poet".to_string(),
        ];
        poet_state.reveal_order = vec![2, 3, 1];
        poet_state.baker_rule_version = Some("baker_day_reveal_v1".to_string());
        assert!(validate_baker_history(&empty_scenario(), &poet_state));
        assert!(check_scenario(&empty_scenario(), &poet_state));

        let mut wrong_poet =
            current_poet("Medium", json!({"good_position": 1, "good_role": "Knight"}));
        wrong_poet.position = 3;
        poet_state.cards[2] = wrong_poet;
        assert!(!validate_baker_history(&empty_scenario(), &poet_state));
        assert!(!check_scenario(&empty_scenario(), &poet_state));

        let mut impossible_poet =
            current_poet("Medium", json!({"good_position": 1, "good_role": "Pooka"}));
        impossible_poet.position = 3;
        poet_state.cards[2] = impossible_poet;
        poet_state.reveal_order.clear();
        poet_state.baker_rule_version = None;
        assert!(!validate_baker_history(&empty_scenario(), &poet_state));
    }

    #[test]
    fn unmarked_medium_keeps_legacy_missing_and_textless_behavior() {
        let missing = make_card(1, "Medium", json!({}));
        let target = make_card(2, "Scout", json!({}));
        let state = base_state(2, vec![missing.clone(), target.clone()]);
        assert!(validate_medium(&missing, &empty_scenario(), &state));

        let historical = make_card(
            1,
            "Medium",
            json!({"good_position": 2, "good_role": "Scout"}),
        );
        let state = base_state(2, vec![historical.clone(), target]);
        assert!(historical.info_text.is_empty());
        assert!(validate_medium(&historical, &empty_scenario(), &state,));
    }

    #[test]
    fn current_poet_bounty_hunter_schema_is_exact_text_bound_and_fail_closed() {
        let valid = current_poet("Bounty Hunter", json!({"evil_position": 2}));
        let state = base_state(3, vec![valid.clone()]);
        let scenario = empty_scenario();

        assert_eq!(valid.info_text, "#2\nis Evil");
        assert_eq!(
            parse_current_bounty_hunter_target(&valid, &state),
            Some(2),
        );
        assert!(validate_current_poet_payload(
            &valid,
            &state,
            "Bounty Hunter",
        ));

        for text in ["", "#2 is Evil", "#2\nis evil", "#2\nis Evil!"] {
            let mut malformed = valid.clone();
            malformed.info_text = text.to_string();
            assert!(parse_current_bounty_hunter_target(&malformed, &state).is_none());
            assert!(!validate_poet(&malformed, &scenario, &state));
        }

        let mut extra = valid.clone();
        extra.info_parsed.insert("future".to_string(), json!(true));
        assert!(!validate_poet(&extra, &scenario, &state));

        let mut wrong_type = valid.clone();
        wrong_type
            .info_parsed
            .insert("evil_position".to_string(), json!(true));
        assert!(!validate_poet(&wrong_type, &scenario, &state));

        let mut future = valid.clone();
        future
            .info_parsed
            .insert("poet_variant".to_string(), json!("future"));
        assert!(!validate_poet(&future, &scenario, &state));

        let mut dormant_direct = valid;
        dormant_direct.apparent_role = "Bounty Hunter".to_string();
        assert!(!validate_bounty_hunter(
            &dormant_direct,
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_poet_bounty_hunter_uses_explicit_wretch_registration() {
        let poet = current_poet("Bounty Hunter", json!({"evil_position": 2}));
        let mut state = base_state(
            3,
            vec![poet.clone(), make_card(2, "Wretch", json!({}))],
        );
        let truthful = empty_scenario();

        assert!(validate_poet(&poet, &truthful, &state));
        let mut lying = truthful.clone();
        lying.corrupted.insert(1);
        assert!(!validate_poet(&poet, &lying, &state));

        state.cards[1] = make_card(2, "Scout", json!({}));
        assert!(!validate_poet(&poet, &truthful, &state));
        assert!(validate_poet(&poet, &lying, &state));
    }

    #[test]
    fn current_poet_bounty_hunter_resolves_anonymous_wretch_at_named_target() {
        let poet = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        let mut state = base_state(
            5,
            vec![poet.clone(), make_card(2, "Scout", json!({}))],
        );
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut truthful = empty_scenario();
        truthful.evil_positions.insert(5, "Pooka".to_string());
        let mut lying = truthful.clone();
        lying.corrupted.insert(1);

        // With anonymous seats #3 and #4, the one Wretch can either make the
        // named target registered Evil or leave it registered Good.
        assert!(validate_poet(&poet, &truthful, &state));
        assert!(validate_poet(&poet, &lying, &state));

        // Once #4 has known Good data, the Wretch is forced onto #3. Truth
        // remains reachable, while the bluff cannot select that target.
        state.cards.push(make_card(4, "Scout", json!({})));
        assert!(validate_poet(&poet, &truthful, &state));
        assert!(!validate_poet(&poet, &lying, &state));

        state.board_outcast_count = Some(0);
        assert!(!validate_poet(&poet, &truthful, &state));
        assert!(validate_poet(&poet, &lying, &state));
    }

    #[test]
    fn current_poet_bounty_hunter_rejects_two_truths_consuming_one_wretch() {
        let first = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        let mut second = current_poet("Bounty Hunter", json!({"evil_position": 4}));
        second.position = 2;
        let mut state = base_state(5, vec![first.clone(), second.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(5, "Pooka".to_string());

        assert!(validate_poet(&first, &scenario, &state));
        assert!(validate_poet(&second, &scenario, &state));
        assert!(!validate_current_bounty_hunter_wretch_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_poet_bounty_hunter_rejects_truth_and_lie_on_same_anonymous_wretch() {
        let first = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        let mut second = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        second.position = 2;
        let mut state = base_state(5, vec![first.clone(), second.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(5, "Pooka".to_string());
        scenario.corrupted.insert(2);

        assert!(validate_poet(&first, &scenario, &state));
        assert!(validate_poet(&second, &scenario, &state));
        assert!(!validate_current_bounty_hunter_wretch_consistency(
            &scenario,
            &state,
        ));

        state.executed.push(2);
        state.confirmed_evil.push(2);
        assert!(validate_current_bounty_hunter_wretch_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_poet_bounty_hunter_accepts_one_joint_wretch_assignment() {
        let first = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        let mut second = current_poet("Bounty Hunter", json!({"evil_position": 4}));
        second.position = 2;
        let mut state = base_state(5, vec![first.clone(), second.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(5, "Pooka".to_string());
        scenario.corrupted.insert(2);

        assert!(validate_poet(&first, &scenario, &state));
        assert!(validate_poet(&second, &scenario, &state));
        assert!(validate_current_bounty_hunter_wretch_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_poet_bounty_hunter_allows_self_puppet_and_dead_targets() {
        let self_claim = current_poet("Bounty Hunter", json!({"evil_position": 1}));
        let singleton = base_state(1, vec![self_claim.clone()]);
        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(1);

        assert!(validate_poet(&self_claim, &puppet, &singleton));
        puppet.corrupted.insert(1);
        assert!(!validate_poet(&self_claim, &puppet, &singleton));

        let mut lying_good_self = empty_scenario();
        lying_good_self.corrupted.insert(1);
        assert!(validate_poet(
            &self_claim,
            &lying_good_self,
            &singleton,
        ));

        let dead_claim = current_poet("Bounty Hunter", json!({"evil_position": 2}));
        let mut dead_state = base_state(3, vec![dead_claim.clone()]);
        dead_state.executed.push(2);
        let mut dead_evil = empty_scenario();
        dead_evil.evil_positions.insert(2, "Pooka".to_string());
        assert!(validate_poet(&dead_claim, &dead_evil, &dead_state));
    }

    #[test]
    fn unmarked_poet_bounty_hunter_preserves_legacy_predicate() {
        let legacy = make_card(
            1,
            "Poet",
            json!({
                "copied_role": "Bounty Hunter",
                "evil_position": 2,
            }),
        );
        let state = base_state(3, vec![legacy.clone()]);
        let mut evil_target = empty_scenario();
        evil_target.evil_positions.insert(2, "Pooka".to_string());

        assert!(legacy.info_text.is_empty());
        assert!(validate_poet(&legacy, &evil_target, &state));
        assert!(!validate_poet(&legacy, &empty_scenario(), &state));

        let mut lying_good = empty_scenario();
        lying_good.corrupted.insert(1);
        assert!(validate_poet(&legacy, &lying_good, &state));
        lying_good
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(!validate_poet(&legacy, &lying_good, &state));

        let missing_claim = make_card(
            1,
            "Poet",
            json!({"copied_role": "Bounty Hunter"}),
        );
        assert!(validate_poet(&missing_claim, &empty_scenario(), &state));
    }

    #[test]
    fn current_lover_schema_is_exact_text_bound_and_fail_closed() {
        let state = base_state(3, vec![]);
        let scenario = empty_scenario();

        for (claimed, text) in [
            (0, "NO Evils\nadjacent to me"),
            (1, "1 Evil\nadjacent to me"),
            (2, "2 Evils\nadjacent to me"),
        ] {
            let direct = current_lover(1, json!(claimed));
            assert_eq!(direct.info_text, text);
            assert_eq!(
                parse_current_lover_claim(
                    &direct,
                    CurrentPassivePayloadSource::Direct,
                    &state,
                ),
                Some(claimed),
            );

            let poet = current_poet("Lover", json!({"evil_adjacent": claimed}));
            assert_eq!(poet.info_text, text);
            assert!(validate_current_poet_payload(&poet, &state, "Lover"));
        }

        for text in [
            "No Evils\nadjacent to me",
            "NO Evil\nadjacent to me",
            "NO Evils adjacent to me",
            "",
        ] {
            let mut wrong_text = current_lover(1, json!(0));
            wrong_text.info_text = text.to_string();
            assert!(!validate_lover(&wrong_text, &scenario, &state));
        }

        for claimed in [json!(-1), json!(3), json!(true), json!("1")] {
            assert!(!validate_lover(
                &current_lover(1, claimed),
                &scenario,
                &state,
            ));
        }

        let mut extra = current_lover(1, json!(0));
        extra.info_parsed.insert("unexpected".to_string(), json!(true));
        assert!(!validate_lover(&extra, &scenario, &state));

        for position in [0, 4] {
            assert!(!validate_lover(
                &current_lover(position, json!(0)),
                &scenario,
                &state,
            ));
        }

        for info in [
            json!({"lover_variant": "future", "evil_adjacent": 0}),
            json!({"lover_variant": 7, "evil_adjacent": 0}),
            json!({
                "lover_variant": "public_current",
                "poet_variant": "public_current",
                "evil_adjacent": 0,
            }),
            json!({"scout_variant": "public_current", "evil_adjacent": 0}),
        ] {
            let mut malformed = make_card(1, "Lover", info);
            malformed.info_text = "NO Evils\nadjacent to me".to_string();
            assert!(!validate_lover(&malformed, &scenario, &state));
        }

        // Marker absence preserves the archived permissive predicate.
        let archived = make_card(1, "Lover", json!({"evil_adjacent": 1}));
        let mut one_adjacent_evil = empty_scenario();
        one_adjacent_evil
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(validate_lover(&archived, &one_adjacent_evil, &state));
        assert!(validate_lover(
            &make_card(1, "Lover", json!({})),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_lover_truth_and_bluff_use_exact_authored_domain() {
        let mut state = base_state(5, vec![]);
        state.n_evil = 2;
        state.deck.minions = vec!["Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Pooka".to_string());

        assert!(validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(0)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));

        scenario.corrupted.insert(1);
        assert!(validate_lover(
            &current_lover(1, json!(0)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));
        assert!(validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));

        let mut no_authored_slots = base_state(
            5,
            vec![make_card(2, "Wretch", json!({}))],
        );
        no_authored_slots.deck.outcasts = vec!["Wretch".to_string()];
        let mut registered_evil_only = empty_scenario();
        registered_evil_only.corrupted.insert(1);
        assert!(validate_lover(
            &current_lover(1, json!(0)),
            &registered_evil_only,
            &no_authored_slots,
        ));
        no_authored_slots.cards.clear();
        no_authored_slots.deck.outcasts.clear();
        assert!(!validate_lover(
            &current_lover(1, json!(0)),
            &registered_evil_only,
            &no_authored_slots,
        ));
    }

    #[test]
    fn current_poet_lover_delegates_to_exact_truth_and_inverse() {
        let poet = current_poet("Lover", json!({"evil_adjacent": 1}));
        let mut state = base_state(4, vec![poet.clone()]);
        state.n_evil = 1;
        state.deck.demons = vec!["Pooka".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Pooka".to_string());

        assert!(validate_poet(&poet, &scenario, &state));
        scenario.corrupted.insert(1);
        assert!(!validate_poet(&poet, &scenario, &state));

        let lying = current_poet("Lover", json!({"evil_adjacent": 0}));
        state.cards = vec![lying.clone()];
        assert!(validate_poet(&lying, &scenario, &state));

        let mut wrong_text = lying;
        wrong_text.info_text = "0 Evils\nadjacent to me".to_string();
        assert!(!validate_current_poet_payload(
            &wrong_text,
            &state,
            "Lover",
        ));
        assert!(!validate_poet(&wrong_text, &scenario, &state));
    }

    #[test]
    fn current_lover_resolves_anonymous_wretch_assignments_jointly() {
        let mut state = base_state(4, vec![current_lover(1, json!(1))]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Pooka".to_string());

        // The one anonymous Wretch can be adjacent at #2 or non-adjacent at
        // #3, so this grouped Scenario supports exact native counts 2 and 1.
        assert!(validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));
        assert!(validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(0)),
            &scenario,
            &state,
        ));

        state.cards.push(make_card(3, "Scout", json!({})));
        assert!(!validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));
        assert!(validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));

        state.board_outcast_count = Some(0);
        assert!(validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_lover_undealt_evil_roles_do_not_expand_authored_bluff_domain() {
        let mut state = base_state(5, vec![]);
        state.n_evil = 1;
        state.deck.minions = vec!["Witch".to_string(), "Poisoner".to_string()];
        state.deck.demons = vec!["Pooka".to_string(), "Lilis".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.corrupted.insert(1);

        assert!(validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(0)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_lover_generated_puppet_is_subtracted_from_hud_authored_slots() {
        let mut state = base_state(5, vec![]);
        state.n_evil = 2;
        state.deck.minions = vec!["Puppeteer".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Puppeteer".to_string());
        scenario.puppet_position = Some(2);
        scenario.corrupted.insert(1);

        assert!(validate_lover(
            &current_lover(1, json!(0)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));

        state.deck.demons.push("Pooka".to_string());
        assert!(!validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_lover_wretch_does_not_reduce_hud_authored_slots() {
        let mut state = base_state(
            6,
            vec![make_card(2, "Wretch", json!({}))],
        );
        state.n_evil = 2;
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.minions = vec!["Witch".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(3, "Witch".to_string());
        scenario.evil_positions.insert(4, "Pooka".to_string());
        scenario.corrupted.insert(1);

        assert_eq!(
            current_lover_possible_actual_counts(1, &scenario, &state),
            vec![1],
        );
        assert!(validate_lover(
            &current_lover(1, json!(0)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));
        assert!(validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_lover_counts_previous_and_next_occurrences_on_tiny_boards() {
        let state = base_state(2, vec![current_lover(1, json!(2))]);
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Pooka".to_string());
        assert_eq!(
            current_lover_possible_actual_counts(1, &scenario, &state),
            vec![2],
        );
        assert!(validate_lover(
            &current_lover(1, json!(2)),
            &scenario,
            &state,
        ));
        assert!(!validate_lover(
            &current_lover(1, json!(1)),
            &scenario,
            &state,
        ));

        let singleton = base_state(1, vec![current_lover(1, json!(0))]);
        assert_eq!(
            current_lover_possible_actual_counts(1, &empty_scenario(), &singleton),
            vec![0],
        );

        let mut singleton_puppet = empty_scenario();
        singleton_puppet.puppet_position = Some(1);
        assert_eq!(
            current_lover_possible_actual_counts(1, &singleton_puppet, &singleton),
            vec![2],
        );

        let mut wretch_state = base_state(2, vec![current_lover(1, json!(2))]);
        wretch_state.deck.outcasts = vec!["Wretch".to_string()];
        wretch_state.board_outcast_count = Some(1);
        wretch_state.board_count_provenance =
            crate::types::BoardCountProvenance::TrustedPreStart;
        assert_eq!(
            current_lover_possible_actual_counts(
                1,
                &empty_scenario(),
                &wretch_state,
            ),
            vec![2],
        );
    }

    #[test]
    fn current_knitter_schema_is_closed_text_bound_and_fail_closed() {
        let state = base_state(3, vec![]);
        let scenario = empty_scenario();

        for (claimed, text) in [
            (0, "Evils are not adjacent to eachother"),
            (1, "There is only 1 pair of Evil"),
            (2, "There are 2 pairs of Evil"),
            (3, "There are 3 pairs of Evil"),
        ] {
            let direct = current_knitter(1, json!(claimed));
            assert_eq!(direct.info_text, text);
            assert_eq!(
                parse_current_knitter_claim(&direct, CurrentPassivePayloadSource::Direct, &state),
                Some(claimed),
            );

            let poet = current_poet("Knitter", json!({"evil_pairs": claimed}));
            assert_eq!(poet.info_text, text);
            assert!(validate_current_poet_payload(&poet, &state, "Knitter"));
        }

        for text in [
            "Evil are not adjacent to eachother",
            "Evils are not adjacent to each other",
            "There is only one pair of Evil",
            "There are 2 pairs of evil",
            "",
        ] {
            let mut wrong_text = current_knitter(1, json!(0));
            wrong_text.info_text = text.to_string();
            assert!(!validate_knitter(&wrong_text, &scenario, &state));
        }

        for claimed in [json!(-1), json!(4), json!(true), json!("1"), json!(1.5)] {
            assert!(!validate_knitter(
                &current_knitter(1, claimed),
                &scenario,
                &state,
            ));
        }

        let mut extra = current_knitter(1, json!(0));
        extra.info_parsed.insert("targets".to_string(), json!([]));
        assert!(!validate_knitter(&extra, &scenario, &state));

        for position in [0, 4] {
            assert!(!validate_knitter(
                &current_knitter(position, json!(0)),
                &scenario,
                &state,
            ));
        }

        let mut noncanonical = current_knitter(1, json!(0));
        noncanonical.apparent_role = "knitter".to_string();
        assert!(!validate_knitter(&noncanonical, &scenario, &state));

        for info in [
            json!({"knitter_variant": "future", "evil_pairs": 0}),
            json!({"knitter_variant": 7, "evil_pairs": 0}),
            json!({
                "knitter_variant": "public_current",
                "poet_variant": "public_current",
                "evil_pairs": 0,
            }),
            json!({"medium_variant": "public_current", "evil_pairs": 0}),
        ] {
            let mut malformed = make_card(1, "Knitter", info);
            malformed.info_text = current_knitter_claim_text(0).unwrap();
            assert!(!validate_knitter(&malformed, &scenario, &state));
        }
    }

    #[test]
    fn current_knitter_uses_native_singleton_double_edge_and_circle_geometry() {
        assert_eq!(current_knitter_pair_count(1, &HashSet::new()), 0);
        assert_eq!(current_knitter_pair_count(1, &HashSet::from([1])), 1);
        assert_eq!(current_knitter_pair_count(2, &HashSet::from([1])), 0);
        assert_eq!(current_knitter_pair_count(2, &HashSet::from([1, 2])), 2);
        assert_eq!(current_knitter_pair_count(3, &HashSet::from([1, 2])), 1);
        assert_eq!(current_knitter_pair_count(3, &HashSet::from([1, 3])), 1);
        assert_eq!(current_knitter_pair_count(3, &HashSet::from([1, 2, 3])), 3);
        assert_eq!(current_knitter_pair_count(4, &HashSet::from([1, 2, 3])), 2);

        let singleton = base_state(1, vec![]);
        let mut singleton_puppet = empty_scenario();
        singleton_puppet.puppet_position = Some(1);
        assert!(validate_knitter(
            &current_knitter(1, json!(1)),
            &singleton_puppet,
            &singleton,
        ));
        assert!(!validate_knitter(
            &current_knitter(1, json!(0)),
            &singleton_puppet,
            &singleton,
        ));

        let two = base_state(2, vec![]);
        let mut two_evil = empty_scenario();
        two_evil.puppet_position = Some(1);
        two_evil.evil_positions.insert(2, "Pooka".to_string());
        assert!(validate_knitter(
            &current_knitter(1, json!(2)),
            &two_evil,
            &two,
        ));

        let three = base_state(3, vec![]);
        let mut adjacent = empty_scenario();
        adjacent.evil_positions.insert(2, "Witch".to_string());
        adjacent.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_knitter(
            &current_knitter(1, json!(1)),
            &adjacent,
            &three,
        ));
    }

    #[test]
    fn current_knitter_includes_actor_dead_hidden_executed_and_register_as_surfaces() {
        let mut state = base_state(4, vec![]);
        state.executed = vec![2];
        state.night_kills = vec![3];
        let mut hidden_dead = empty_scenario();
        hidden_dead.evil_positions.insert(2, "Witch".to_string());
        hidden_dead.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_knitter(
            &current_knitter(1, json!(1)),
            &hidden_dead,
            &state,
        ));

        let mut spy = empty_scenario();
        spy.evil_positions.insert(2, "Spy".to_string());
        spy.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_knitter(
            &current_knitter(1, json!(0)),
            &spy,
            &state,
        ));
        assert!(!validate_knitter(
            &current_knitter(1, json!(1)),
            &spy,
            &state,
        ));

        state.cards = vec![make_card(2, "Wretch", json!({}))];
        state.deck.outcasts = vec!["Wretch".to_string()];
        let mut explicit_wretch = empty_scenario();
        explicit_wretch
            .evil_positions
            .insert(3, "Pooka".to_string());
        assert!(validate_knitter(
            &current_knitter(1, json!(1)),
            &explicit_wretch,
            &state,
        ));

        let mut moved_state = base_state(
            4,
            vec![
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Wretch", json!({})),
            ],
        );
        moved_state.deck.outcasts = vec!["Wretch".to_string()];
        let mut moved_wretch = empty_scenario();
        moved_wretch
            .evil_positions
            .insert(2, "Twin Minion".to_string());
        moved_wretch
            .evil_positions
            .insert(4, "Pooka".to_string());
        moved_wretch.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 2,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 4,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 3,
                neighbor_pre_swap_role: "Wretch".to_string(),
            },
        });
        assert!(validate_knitter(
            &current_knitter(1, json!(0)),
            &moved_wretch,
            &moved_state,
        ));

        state.cards = vec![make_card(2, "Scout", json!({}))];
        let mut copied_wretch = explicit_wretch.clone();
        copied_wretch.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 4,
            target_position: 2,
            copied_role: "Wretch".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });
        assert!(validate_knitter(
            &current_knitter(1, json!(2)),
            &copied_wretch,
            &state,
        ));

        let mut copied_spy = hidden_dead;
        copied_spy.evil_positions.remove(&2);
        copied_spy.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 4,
            target_position: 2,
            copied_role: "Spy".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });
        assert!(validate_knitter(
            &current_knitter(1, json!(0)),
            &copied_spy,
            &state,
        ));
    }

    #[test]
    fn current_knitter_projects_baker_spy_reset_at_each_observation() {
        let mut immediate_poet = current_poet("Knitter", json!({"evil_pairs": 0}));
        immediate_poet.position = 5;
        let mut state = base_state(
            6,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                make_card(3, "Pooka", json!({})),
                current_knitter(4, json!(0)),
                immediate_poet,
                current_knitter(6, json!(1)),
            ],
        );
        state.deck.villagers = vec![
            "Baker".to_string(),
            "Knitter".to_string(),
            "Poet".to_string(),
            "Knitter".to_string(),
        ];
        state.deck.minions = vec!["Spy".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.n_evil = 2;
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![4, 1, 5, 6, 2, 3];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Spy".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());

        // #4 is before conversion. The immediately following Poet at #5 may
        // still see Spy's stale Good registerAs at +0.2s, while #6 can see the
        // delayed Baker-null reset. One monotonic reset boundary supports all.
        assert!(check_scenario(&scenario, &state));

        // A reset cannot reverse: post-reset Evil at #5 followed by stale Good
        // at #6 has no single Baker timing witness.
        state.cards[4] = {
            let mut poet = current_poet("Knitter", json!({"evil_pairs": 1}));
            poet.position = 5;
            poet
        };
        state.cards[5] = current_knitter(6, json!(0));
        assert!(!check_scenario(&scenario, &state));
    }

    #[test]
    fn baker_spy_same_event_has_baker_data_stale_register_as_and_no_raw_bluff() {
        let mut state = base_state(
            3,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        state.deck.villagers = vec!["Baker".to_string(), "Scout".to_string()];
        state.deck.minions = vec!["Spy".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Spy".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());

        let timelines = baker_spy_conversion_timelines(&scenario, &state);
        assert!(!timelines.is_empty());
        for timeline in timelines {
            assert_eq!(
                baker_spy_observation_phase(2, 1, &timeline, &state),
                Some(BakerSpyObservationPhase::PendingRegisterAsReset)
            );
            assert_eq!(
                current_data_role_at_observation(2, 1, &timeline, &scenario, &state).as_deref(),
                Some("Baker")
            );
            assert_eq!(
                registered_alignment_at_observation(2, 1, &timeline, &scenario, &state),
                Some(EffectiveAlignment::Good)
            );
            assert_eq!(
                current_medium_raw_bluff_holder_at(2, 1, &timeline, &scenario, &state),
                CurrentMediumRawBluffHolder::Impossible
            );
        }
    }

    #[test]
    fn current_medium_cannot_select_a_baker_cleared_spy_raw_bluff() {
        let medium = current_medium(2, json!(3), json!("Scout"));
        let mut state = base_state(
            4,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                medium.clone(),
                make_card(3, "Baker", json!({"original_role": "Spy"})),
                make_card(4, "Pooka", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Baker".to_string(),
            "Medium".to_string(),
            "Scout".to_string(),
        ];
        state.deck.minions = vec!["Spy".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(3, "Spy".to_string());
        scenario.evil_positions.insert(4, "Pooka".to_string());
        scenario.corrupted.insert(2);

        assert!(validate_baker_history(&scenario, &state));
        assert!(!validate_medium(&medium, &scenario, &state));
    }

    #[test]
    fn lover_and_bounty_hunter_share_one_monotonic_baker_spy_reset() {
        let lover = current_lover(5, json!(1));
        let mut bounty = current_poet("Bounty Hunter", json!({"evil_position": 4}));
        bounty.position = 3;
        let mut state = base_state(
            8,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Scout", json!({})),
                bounty.clone(),
                make_card(4, "Baker", json!({"original_role": "Spy"})),
                lover.clone(),
                make_card(8, "Pooka", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Baker".to_string(),
            "Scout".to_string(),
            "Poet".to_string(),
            "Lover".to_string(),
        ];
        state.deck.minions = vec!["Spy".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.n_evil = 2;
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Spy".to_string());
        scenario.evil_positions.insert(8, "Pooka".to_string());
        scenario.corrupted.insert(3);

        // Each clue is independently reachable: Lover can choose an early
        // reset, while lying Bounty Hunter can choose a late reset. In this
        // order those witnesses contradict because stale registerAs cannot
        // return after Lover already counted the Spy as Evil.
        state.reveal_order = vec![1, 5, 2, 3, 4, 8];
        assert!(validate_lover(&lover, &scenario, &state));
        assert!(validate_poet(&bounty, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state
        ));

        // Observing the stale-Good Bounty target first and the reset-Evil
        // Lover adjacency later has one monotonic boundary.
        state.reveal_order = vec![1, 3, 2, 5, 4, 8];
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state
        ));
    }

    #[test]
    fn current_knitter_bluff_uses_authored_slots_native_endpoints_and_collision_removal() {
        let mut state = base_state(5, vec![]);
        state.n_evil = 3;
        let mut lying = empty_scenario();
        lying.corrupted.insert(1);
        lying.evil_positions.insert(2, "Witch".to_string());
        lying.evil_positions.insert(3, "Pooka".to_string());

        assert!(validate_knitter(
            &current_knitter(1, json!(0)),
            &lying,
            &state,
        ));
        assert!(!validate_knitter(
            &current_knitter(1, json!(1)),
            &lying,
            &state,
        ));
        assert!(validate_knitter(
            &current_knitter(1, json!(2)),
            &lying,
            &state,
        ));
        assert!(!validate_knitter(
            &current_knitter(1, json!(3)),
            &lying,
            &state,
        ));

        state.n_evil = 2;
        state.cards = vec![make_card(4, "Wretch", json!({}))];
        state.deck.outcasts = vec!["Wretch".to_string()];
        assert!(validate_knitter(
            &current_knitter(1, json!(0)),
            &lying,
            &state,
        ));
        assert!(validate_knitter(
            &current_knitter(1, json!(1)),
            &lying,
            &state,
        ));
        assert!(!validate_knitter(
            &current_knitter(1, json!(2)),
            &lying,
            &state,
        ));

        state.n_evil = 3;
        state.cards.clear();
        state.deck.outcasts.clear();
        state.deck.minions = vec!["Witch".to_string(), "Poisoner".to_string()];
        state.deck.demons = vec!["Pooka".to_string(), "Lilis".to_string()];
        let mut generated_puppet = empty_scenario();
        generated_puppet.corrupted.insert(1);
        generated_puppet.puppet_position = Some(5);
        assert_eq!(current_authored_evil_slots(&generated_puppet, &state), Some(2));
        assert!(validate_knitter(
            &current_knitter(1, json!(1)),
            &generated_puppet,
            &state,
        ));
        assert!(!validate_knitter(
            &current_knitter(1, json!(2)),
            &generated_puppet,
            &state,
        ));

        generated_puppet.puppet_position = None;
        state.n_evil = 1;
        assert_eq!(current_authored_evil_slots(&generated_puppet, &state), Some(1));
        assert!(validate_knitter(
            &current_knitter(1, json!(1)),
            &generated_puppet,
            &state,
        ));
        assert!(!validate_knitter(
            &current_knitter(1, json!(2)),
            &generated_puppet,
            &state,
        ));

        state.n_evil = 0;
        let mut zero_authored_slots = empty_scenario();
        zero_authored_slots.corrupted.insert(1);
        assert_eq!(current_authored_evil_slots(&zero_authored_slots, &state), Some(0));
        assert!(!validate_knitter(
            &current_knitter(1, json!(0)),
            &zero_authored_slots,
            &state,
        ));
        assert!(validate_knitter(
            &current_knitter(1, json!(1)),
            &zero_authored_slots,
            &state,
        ));
        assert!(!validate_knitter(
            &current_knitter(1, json!(2)),
            &zero_authored_slots,
            &state,
        ));
    }

    #[test]
    fn legacy_generated_puppet_count_restores_current_authored_evil_slots() {
        let mut state = base_state(5, vec![current_knitter(1, json!(2))]);
        state.n_evil = 3;
        state.board_count_provenance = BoardCountProvenance::LegacyUnknown;
        let mut scenario = empty_scenario();
        scenario.evil_positions = HashMap::from([
            (2, "Puppeteer".to_string()),
            (3, "Pooka".to_string()),
            (4, "Witch".to_string()),
            (5, "Puppet".to_string()),
        ]);
        scenario.puppet_position = Some(5);
        scenario.corrupted.insert(1);

        assert_eq!(current_authored_evil_slots(&scenario, &state), Some(3));
        assert!(validate_knitter(
            &current_knitter(1, json!(2)),
            &scenario,
            &state,
        ));

        state.board_count_provenance = BoardCountProvenance::TrustedPreStart;
        assert_eq!(current_authored_evil_slots(&scenario, &state), Some(2));
        assert!(!validate_knitter(
            &current_knitter(1, json!(2)),
            &scenario,
            &state,
        ));

        scenario.evil_positions = HashMap::from([
            (2, "Puppeteer".to_string()),
            (3, "Twin Minion".to_string()),
            (4, "Pooka".to_string()),
        ]);
        scenario.puppet_position = Some(3);
        assert_eq!(current_authored_evil_slots(&scenario, &state), Some(3));
    }

    #[test]
    fn current_knitter_anonymous_wretch_worlds_are_exact_and_joint() {
        let direct = current_knitter(1, json!(0));
        let mut poet = current_poet("Knitter", json!({"evil_pairs": 1}));
        poet.position = 3;
        let mut state = base_state(3, vec![direct.clone(), poet.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        let mut scenario = empty_scenario();
        scenario.puppet_position = Some(3);

        assert_eq!(anonymous_natural_wretch_candidates(&scenario, &state), vec![2]);
        assert!(validate_knitter(&direct, &scenario, &state));
        assert!(validate_poet(&poet, &scenario, &state));
        assert!(!validate_current_knitter_consistency(&scenario, &state));

        poet.info_parsed.insert("evil_pairs".to_string(), json!(0));
        poet.info_text = current_knitter_claim_text(0).unwrap();
        state.cards[1] = poet.clone();
        assert!(validate_poet(&poet, &scenario, &state));
        assert!(validate_current_knitter_consistency(&scenario, &state));
    }

    #[test]
    fn current_hidden_surfaces_share_one_anonymous_wretch_world_across_providers() {
        let mut bounty = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        bounty.position = 2;
        let knitter = current_knitter(1, json!(0));
        let mut state = base_state(3, vec![knitter.clone(), bounty.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        let mut scenario = empty_scenario();
        scenario.puppet_position = Some(2);

        assert!(validate_knitter(&knitter, &scenario, &state));
        assert!(validate_poet(&bounty, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(&scenario, &state));

        state.cards[0] = current_knitter(1, json!(1));
        assert!(validate_current_hidden_surface_consistency(&scenario, &state));

        let scout = current_scout(1, json!({"one_evil": true}));
        state.cards[0] = scout.clone();
        assert!(validate_scout(&scout, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(&scenario, &state));

        let medium = current_medium(1, json!(3), json!("Scout"));
        scenario.corrupted.insert(1);
        state.cards[0] = medium.clone();
        assert!(validate_medium(&medium, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(&scenario, &state));

        let mut hunter_bounty = current_poet("Bounty Hunter", json!({"evil_position": 2}));
        hunter_bounty.position = 3;
        let hunter = current_hunter(1, json!(2));
        state = base_state(
            4,
            vec![
                hunter.clone(),
                hunter_bounty.clone(),
                make_card(4, "Scout", json!({})),
            ],
        );
        state.deck.outcasts = vec!["Wretch".to_string()];
        scenario = empty_scenario();
        scenario.puppet_position = Some(3);
        assert!(validate_hunter(&hunter, &scenario, &state));
        assert!(validate_poet(&hunter_bounty, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(&scenario, &state));

        let lover = current_lover(1, json!(0));
        state.cards[0] = lover.clone();
        assert!(validate_lover(&lover, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(&scenario, &state));
    }

    #[test]
    fn current_oracles_share_the_anonymous_wretch_register_as_draw() {
        let direct = current_oracle(
            1,
            json!({"targets": [3, 4], "minion_role": "Witch"}),
        );
        let mut poet = current_poet(
            "Oracle",
            json!({"targets": [3, 4], "minion_role": "Twin Minion"}),
        );
        poet.position = 2;
        let mut state = base_state(
            4,
            vec![direct.clone(), poet.clone(), make_card(4, "Scout", json!({}))],
        );
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.minions = vec!["Witch".to_string(), "Twin Minion".to_string()];
        let mut scenario = empty_scenario();
        scenario.puppet_position = Some(2);

        assert!(validate_oracle(&direct, &scenario, &state));
        assert!(validate_poet(&poet, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(&scenario, &state));

        poet = current_poet(
            "Oracle",
            json!({"targets": [3, 4], "minion_role": "Witch"}),
        );
        poet.position = 2;
        state.cards[1] = poet;
        assert!(validate_current_hidden_surface_consistency(&scenario, &state));
    }

    #[test]
    fn current_knitter_direct_poet_parity_and_unmarked_legacy_preservation() {
        let direct = current_knitter(1, json!(1));
        let poet = current_poet("Knitter", json!({"evil_pairs": 1}));
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Witch".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        let mut state = base_state(4, vec![direct.clone()]);
        state.n_evil = 2;

        assert!(validate_knitter(&direct, &scenario, &state));
        assert!(validate_poet(&poet, &scenario, &state));

        scenario.corrupted.insert(1);
        let direct_lie = current_knitter(1, json!(0));
        let poet_lie = current_poet("Knitter", json!({"evil_pairs": 0}));
        assert!(validate_knitter(&direct_lie, &scenario, &state));
        assert!(validate_poet(&poet_lie, &scenario, &state));
        assert!(!validate_knitter(&direct, &scenario, &state));
        assert!(!validate_poet(&poet, &scenario, &state));

        scenario.corrupted.clear();
        let legacy_direct = make_card(1, "Knitter", json!({"evil_pairs": 1}));
        let legacy_poet = make_card(
            1,
            "Poet",
            json!({"copied_role": "Knitter", "evil_pairs": 1}),
        );
        assert!(legacy_direct.info_text.is_empty());
        assert!(validate_knitter(&legacy_direct, &scenario, &state));
        assert!(validate_poet(&legacy_poet, &scenario, &state));
        assert!(validate_knitter(
            &make_card(1, "Knitter", json!({})),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_poet_scout_requires_a_selectable_named_target() {
        let poet = current_poet(
            "Scout",
            json!({"evil_role": "Pooka", "distance": 2}),
        );
        let state = base_state(5, vec![poet.clone()]);

        let mut named_role_absent = empty_scenario();
        named_role_absent
            .evil_positions
            .insert(4, "Witch".to_string());
        assert!(!validate_poet(&poet, &named_role_absent, &state));
        named_role_absent.corrupted.insert(1);
        assert!(!validate_poet(&poet, &named_role_absent, &state));

        let mut only_named_evil = empty_scenario();
        only_named_evil
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(!validate_poet(&poet, &only_named_evil, &state));
        only_named_evil.corrupted.insert(1);
        assert!(validate_poet(&poet, &only_named_evil, &state));

        let mut exact_distance = empty_scenario();
        exact_distance
            .evil_positions
            .insert(2, "Pooka".to_string());
        exact_distance
            .evil_positions
            .insert(4, "Witch".to_string());
        assert!(validate_poet(&poet, &exact_distance, &state));
        exact_distance.corrupted.insert(1);
        assert!(!validate_poet(&poet, &exact_distance, &state));

        let mut unresolved_role = empty_scenario();
        unresolved_role
            .evil_positions
            .insert(3, "Unknown".to_string());
        assert!(!validate_poet(&poet, &unresolved_role, &state));
        unresolved_role.corrupted.insert(1);
        assert!(!validate_poet(&poet, &unresolved_role, &state));
    }

    #[test]
    fn current_direct_scout_payload_is_exact_and_fail_closed() {
        let state = base_state(6, vec![]);

        let numeric = current_scout(
            1,
            json!({"evil_role": "Scout", "distance": 2}),
        );
        assert!(matches!(
            parse_current_scout_claim(
                &numeric.info_parsed,
                CurrentPassivePayloadSource::Direct,
                state.n_cards,
            ),
            Some(CurrentScoutClaim::Numeric {
                evil_role: "Scout",
                distance: 2,
            })
        ));

        let sentinel = current_scout(1, json!({"one_evil": true}));
        assert_eq!(
            parse_current_scout_claim(
                &sentinel.info_parsed,
                CurrentPassivePayloadSource::Direct,
                state.n_cards,
            ),
            Some(CurrentScoutClaim::OneEvil),
        );

        for payload in [
            json!({"evil_role": "pooka", "distance": 2}),
            json!({"evil_role": "Pooka", "distance": 0}),
            json!({"evil_role": "Pooka", "distance": 4}),
            json!({"one_evil": false}),
            json!({"one_evil": true, "distance": 1}),
            json!({"evil_role": "Pooka"}),
        ] {
            let malformed = current_scout(1, payload);
            assert!(parse_current_scout_claim(
                &malformed.info_parsed,
                CurrentPassivePayloadSource::Direct,
                state.n_cards,
            )
            .is_none());
        }

        let scenario = empty_scenario();
        for info in [
            json!({
                "scout_variant": "future",
                "evil_role": "Pooka",
                "distance": 2,
            }),
            json!({
                "scout_variant": 7,
                "evil_role": "Pooka",
                "distance": 2,
            }),
            json!({
                "scout_variant": "public_current",
                "poet_variant": "public_current",
                "evil_role": "Pooka",
                "distance": 2,
            }),
            json!({
                "hunter_variant": "public_current",
                "evil_role": "Pooka",
                "distance": 2,
            }),
        ] {
            assert!(!validate_scout(
                &make_card(1, "Scout", info),
                &scenario,
                &state,
            ));
        }

        // Marker absence preserves the archived permissive fallback.
        assert!(validate_scout(
            &make_card(1, "Scout", json!({})),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_scout_uses_existential_duplicate_targets_and_registered_distance() {
        let scout = current_scout(
            1,
            json!({"evil_role": "Pooka", "distance": 3}),
        );
        let state = base_state(10, vec![scout.clone()]);
        let mut duplicates = empty_scenario();
        duplicates.evil_positions = HashMap::from([
            (2, "Pooka".to_string()),
            (5, "Pooka".to_string()),
            (6, "Witch".to_string()),
        ]);

        // #2 has public distance 3 while the duplicate Pooka at #5 has
        // distance 1. Native random target selection makes the observation
        // existential across both same-name candidates.
        assert!(validate_scout(&scout, &duplicates, &state));
        let no_duplicate_support = current_scout(
            1,
            json!({"evil_role": "Pooka", "distance": 2}),
        );
        assert!(!validate_scout(
            &no_duplicate_support,
            &duplicates,
            &state,
        ));

        let registered_scout = current_scout(
            1,
            json!({"evil_role": "Pooka", "distance": 2}),
        );
        let mut registered_state = base_state(
            10,
            vec![registered_scout.clone(), make_card(4, "Wretch", json!({}))],
        );
        registered_state.executed = vec![2];
        let mut registered = empty_scenario();
        registered.evil_positions = HashMap::from([
            (2, "Pooka".to_string()),
            (8, "Witch".to_string()),
        ]);

        // The dead runtime-Evil target remains selectable support, and the
        // runtime-Good Wretch still participates as another registered Evil.
        assert!(validate_scout(
            &registered_scout,
            &registered,
            &registered_state,
        ));
    }

    #[test]
    fn current_scout_names_moved_data_and_wretch_register_as_by_truth_path() {
        let moved_scout = current_scout(
            1,
            json!({"evil_role": "Scout", "distance": 2}),
        );
        let mut moved_state = base_state(6, vec![moved_scout.clone()]);
        moved_state.executed = vec![2];
        let mut moved = empty_scenario();
        moved.evil_positions = HashMap::from([
            (2, "Twin Minion".to_string()),
            (6, "Pooka".to_string()),
        ]);
        moved.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 2,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 6,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 3,
                neighbor_pre_swap_role: "Scout".to_string(),
            },
        });
        assert!(validate_scout(&moved_scout, &moved, &moved_state));

        let lying_actor = current_scout(
            2,
            json!({"evil_role": "Scout", "distance": 1}),
        );
        moved_state.cards = vec![lying_actor.clone()];
        moved_state.executed.clear();
        assert!(validate_scout(&lying_actor, &moved, &moved_state));

        let truthful_wretch = current_scout(
            1,
            json!({"evil_role": "Witch", "distance": 2}),
        );
        let mut wretch_state = base_state(6, vec![truthful_wretch.clone()]);
        wretch_state.deck.minions = vec!["Witch".to_string(), "Twin_Minion".to_string()];
        let mut wretch = empty_scenario();
        wretch.evil_positions = HashMap::from([
            (2, "Twin Minion".to_string()),
            (6, "Pooka".to_string()),
        ]);
        wretch.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 2,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 6,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 3,
                neighbor_pre_swap_role: "Wretch".to_string(),
            },
        });

        // Truthful Scout names Wretch's sampled Minion register-as record.
        assert!(validate_scout(
            &truthful_wretch,
            &wretch,
            &wretch_state,
        ));
        let normalized_register_as = current_scout(
            1,
            json!({"evil_role": "Twin Minion", "distance": 2}),
        );
        assert!(validate_scout(
            &normalized_register_as,
            &wretch,
            &wretch_state,
        ));
        let truthful_data_ref_name = current_scout(
            1,
            json!({"evil_role": "Wretch", "distance": 2}),
        );
        assert!(!validate_scout(
            &truthful_data_ref_name,
            &wretch,
            &wretch_state,
        ));

        // Bluff Scout instead names the direct current dataRef.
        wretch.corrupted.insert(1);
        let lying_data_ref_name = current_scout(
            1,
            json!({"evil_role": "Wretch", "distance": 1}),
        );
        assert!(validate_scout(
            &lying_data_ref_name,
            &wretch,
            &wretch_state,
        ));
        let lying_register_as_name = current_scout(
            1,
            json!({"evil_role": "Witch", "distance": 1}),
        );
        assert!(!validate_scout(
            &lying_register_as_name,
            &wretch,
            &wretch_state,
        ));
    }

    #[test]
    fn current_scout_sentinel_and_bluff_domain_follow_native_support() {
        let sentinel = current_scout(1, json!({"one_evil": true}));
        let mut state = base_state(8, vec![sentinel.clone()]);
        let mut one_evil = empty_scenario();
        one_evil.evil_positions.insert(4, "Pooka".to_string());
        assert!(validate_scout(&sentinel, &one_evil, &state));
        let poet_sentinel = current_poet("Scout", json!({"one_evil": true}));
        let poet_state = base_state(8, vec![poet_sentinel.clone()]);
        assert!(validate_poet(&poet_sentinel, &one_evil, &poet_state));

        state.cards.push(make_card(6, "Wretch", json!({})));
        assert!(!validate_scout(&sentinel, &one_evil, &state));
        state.cards.pop();

        one_evil.corrupted.insert(1);
        assert!(!validate_scout(&sentinel, &one_evil, &state));

        let lying_numeric = current_scout(
            1,
            json!({"evil_role": "Pooka", "distance": 1}),
        );
        assert!(validate_scout(&lying_numeric, &one_evil, &state));
        let absent_name = current_scout(
            1,
            json!({"evil_role": "Witch", "distance": 1}),
        );
        assert!(!validate_scout(&absent_name, &one_evil, &state));
        let outside_bluff_domain = current_scout(
            1,
            json!({"evil_role": "Pooka", "distance": 4}),
        );
        assert!(!validate_scout(
            &outside_bluff_domain,
            &one_evil,
            &state,
        ));
    }

    #[test]
    fn current_hunter_schema_is_exact_and_matches_native_union() {
        for (n_cards, accepted, rejected) in [
            (1, vec![0], vec![1]),
            (2, vec![1], vec![0, 2]),
            (5, vec![1, 2, 4], vec![0, 3, 5]),
            (6, vec![1, 2, 3, 5], vec![0, 4, 6]),
        ] {
            let state = base_state(n_cards, vec![]);
            for distance in accepted {
                let direct = current_hunter(1, json!(distance));
                assert_eq!(
                    parse_current_hunter_distance(
                        &direct.info_parsed,
                        CurrentPassivePayloadSource::Direct,
                        state.n_cards,
                    ),
                    Some(distance),
                );
                let poet = current_poet("Hunter", json!({"distance": distance}));
                assert!(validate_current_poet_payload(
                    &poet,
                    &state,
                    "Hunter",
                ));
            }
            for distance in rejected {
                let direct = current_hunter(1, json!(distance));
                assert!(parse_current_hunter_distance(
                    &direct.info_parsed,
                    CurrentPassivePayloadSource::Direct,
                    state.n_cards,
                )
                .is_none());
            }
        }

        let state = base_state(5, vec![]);
        let mut extra = current_hunter(1, json!(2));
        extra.info_parsed.insert("unexpected".to_string(), json!(true));
        assert!(!validate_hunter(&extra, &empty_scenario(), &state));
        for variant in [json!("future"), json!(7)] {
            let malformed = make_card(
                1,
                "Hunter",
                json!({"hunter_variant": variant, "distance": 2}),
            );
            assert!(!validate_hunter(&malformed, &empty_scenario(), &state));
        }
        let wrong_role_marker = make_card(
            1,
            "Hunter",
            json!({"scout_variant": "public_current", "distance": 2}),
        );
        assert!(!validate_hunter(
            &wrong_role_marker,
            &empty_scenario(),
            &state,
        ));
    }

    #[test]
    fn current_hunter_uses_registered_evil_and_no_candidate_distance() {
        let hunter = current_hunter(1, json!(2));
        let state = base_state(
            7,
            vec![hunter.clone(), make_card(3, "Wretch", json!({}))],
        );
        let mut registered = empty_scenario();
        registered.evil_positions.insert(5, "Pooka".to_string());
        assert!(validate_hunter(&hunter, &registered, &state));
        assert!(!validate_hunter(
            &current_hunter(1, json!(1)),
            &registered,
            &state,
        ));

        let no_candidate = current_hunter(1, json!(6));
        let empty_state = base_state(7, vec![no_candidate.clone()]);
        assert!(validate_hunter(
            &no_candidate,
            &empty_scenario(),
            &empty_state,
        ));
        let poet_no_candidate = current_poet("Hunter", json!({"distance": 6}));
        let poet_state = base_state(7, vec![poet_no_candidate.clone()]);
        assert!(validate_poet(
            &poet_no_candidate,
            &empty_scenario(),
            &poet_state,
        ));

        let singleton = current_hunter(1, json!(0));
        let singleton_state = base_state(1, vec![singleton.clone()]);
        assert!(validate_hunter(
            &singleton,
            &empty_scenario(),
            &singleton_state,
        ));
    }

    #[test]
    fn current_hunter_bluff_requires_nonempty_native_domain_and_false_value() {
        let state = base_state(7, vec![]);
        let mut lying = empty_scenario();
        lying.corrupted.insert(1);
        lying.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_hunter(
            &current_hunter(1, json!(1)),
            &lying,
            &state,
        ));
        assert!(!validate_hunter(
            &current_hunter(1, json!(2)),
            &lying,
            &state,
        ));
        assert!(validate_hunter(
            &current_hunter(1, json!(3)),
            &lying,
            &state,
        ));

        let two_card_state = base_state(2, vec![]);
        let mut two_card_lying = empty_scenario();
        two_card_lying.corrupted.insert(1);
        assert!(!validate_hunter(
            &current_hunter(1, json!(1)),
            &two_card_lying,
            &two_card_state,
        ));
        two_card_lying
            .evil_positions
            .insert(2, "Pooka".to_string());
        assert!(!validate_hunter(
            &current_hunter(1, json!(1)),
            &two_card_lying,
            &two_card_state,
        ));

        let singleton_state = base_state(1, vec![]);
        let mut singleton_lying = empty_scenario();
        singleton_lying.corrupted.insert(1);
        assert!(!validate_hunter(
            &current_hunter(1, json!(0)),
            &singleton_lying,
            &singleton_state,
        ));
    }

    #[test]
    fn anonymous_wretch_constraints_reuse_exact_outcast_pool_and_header_budget() {
        let mut state = base_state(4, vec![current_scout(1, json!({"one_evil": true}))]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Pooka".to_string());

        assert!(crate::scenario::scenario_allows_anonymous_natural_outcast_role_assignments(
            &HashSet::from([3]),
            "Wretch",
            &HashSet::from([2]),
            &scenario,
            &state,
        ));
        assert!(!crate::scenario::scenario_allows_anonymous_natural_outcast_role_assignments(
            &HashSet::new(),
            "Wretch",
            &HashSet::from([2, 3]),
            &scenario,
            &state,
        ));
        assert!(!crate::scenario::scenario_allows_anonymous_natural_outcast_role_assignments(
            &HashSet::from([3]),
            "Wretch",
            &HashSet::from([3]),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_scout_and_hunter_include_exact_anonymous_wretch_distance_support() {
        let scout = current_scout(
            1,
            json!({"evil_role": "Pooka", "distance": 1}),
        );
        let hunter = current_hunter(7, json!(1));
        let mut state = base_state(8, vec![scout.clone(), hunter.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Pooka".to_string());

        assert!(validate_scout(&scout, &scenario, &state));
        assert!(validate_hunter(&hunter, &scenario, &state));
        // Scout needs the sole Wretch at #3/#5 while Hunter needs it at
        // #6/#8. Each clue is reachable alone, but no physical assignment
        // supports both observations.
        assert!(!validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));

        let compatible_hunter = current_hunter(2, json!(1));
        let mut compatible_state = base_state(8, vec![scout.clone(), compatible_hunter.clone()]);
        compatible_state.deck.outcasts = vec!["Wretch".to_string()];
        compatible_state.board_outcast_count = Some(1);
        compatible_state.board_count_provenance =
            crate::types::BoardCountProvenance::TrustedPreStart;
        assert!(validate_hunter(
            &compatible_hunter,
            &scenario,
            &compatible_state,
        ));
        // Both providers can share the one Wretch at #3.
        assert!(validate_current_hidden_surface_consistency(
            &scenario,
            &compatible_state,
        ));

        let sentinel = current_scout(1, json!({"one_evil": true}));
        assert!(!validate_scout(&sentinel, &scenario, &state));

        state.board_outcast_count = Some(0);
        assert!(!validate_scout(&scout, &scenario, &state));
        assert!(!validate_hunter(&hunter, &scenario, &state));
        assert!(validate_scout(&sentinel, &scenario, &state));
    }

    #[test]
    fn current_lover_and_bounty_hunter_share_one_anonymous_wretch_assignment() {
        let lover = current_lover(1, json!(1));
        let mut bounty_hunter = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        bounty_hunter.position = 2;
        let mut state = base_state(5, vec![lover.clone(), bounty_hunter.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance =
            crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Pooka".to_string());

        // Lover's only anonymous adjacent seat is #5, while Bounty Hunter
        // requires the named anonymous target #3 to be the same sole Wretch.
        assert!(validate_lover(&lover, &scenario, &state));
        assert!(validate_poet(&bounty_hunter, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));

        bounty_hunter.info_parsed.insert(
            "evil_position".to_string(),
            serde_json::Value::from(5),
        );
        bounty_hunter.info_text = "#5\nis Evil".to_string();
        state.cards[1] = bounty_hunter;
        assert!(validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_medium_raw_bluff_and_bounty_hunter_join_wretch_identity() {
        let bounty_hunter = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        let medium = current_medium(2, json!(3), json!("Judge"));
        let mut state = base_state(5, vec![bounty_hunter.clone(), medium.clone()]);
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance =
            crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(5, "Pooka".to_string());
        scenario.corrupted.insert(2);

        // Bounty Hunter truth requires anonymous #3 to be Wretch. A lying
        // Medium may independently select #3 as an unmodeled raw-bluff holder
        // only in a world where base-null natural Wretch is forbidden there.
        assert!(validate_poet(&bounty_hunter, &scenario, &state));
        assert!(validate_medium(&medium, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));

        let compatible_medium = current_medium(2, json!(4), json!("Judge"));
        state.cards[1] = compatible_medium.clone();
        assert!(validate_medium(&compatible_medium, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_scout_observations_share_one_explicit_wretch_register_as_draw() {
        let direct = current_scout(
            1,
            json!({"evil_role": "Witch", "distance": 4}),
        );
        let mut poet = current_poet(
            "Scout",
            json!({"evil_role": "Twin Minion", "distance": 4}),
        );
        poet.position = 2;
        let mut state = base_state(8, vec![direct.clone(), poet.clone()]);
        state.deck.minions = vec!["Witch".to_string(), "Twin_Minion".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions = HashMap::from([
            (4, "Twin Minion".to_string()),
            (8, "Pooka".to_string()),
        ]);
        scenario.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 4,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 8,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 5,
                neighbor_pre_swap_role: "Wretch".to_string(),
            },
        });

        assert!(validate_scout(&direct, &scenario, &state));
        assert!(validate_poet(&poet, &scenario, &state));
        assert!(!validate_current_register_as_consistency(
            &scenario,
            &state,
        ));

        let mut matching_poet = current_poet(
            "Scout",
            json!({"evil_role": "Witch", "distance": 4}),
        );
        matching_poet.position = 2;
        state.cards[1] = matching_poet;
        assert!(validate_current_register_as_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn current_scout_register_as_consistency_skips_untyped_executed_evils() {
        let malformed = current_scout(1, json!({"one_evil": false}));
        let mut state = base_state(4, vec![malformed]);
        state.executed = vec![1];
        state.confirmed_evil = vec![1];
        let scenario = empty_scenario();

        assert!(validate_current_register_as_consistency(
            &scenario,
            &state,
        ));

        state
            .executed_evil_roles
            .insert(1, "Pooka".to_string());
        // A singleton has no cross-observation register-as conflict; its
        // ordinary validator still owns and rejects the malformed schema.
        assert!(validate_current_register_as_consistency(
            &scenario,
            &state,
        ));
        assert!(!validate_scout(&state.cards[0], &scenario, &state));
    }

    #[test]
    fn current_oracle_schema_is_exact_text_bound_and_non_decreasing() {
        let state = base_state(6, vec![]);
        let positive = current_oracle(
            1,
            json!({"targets": [2, 3], "minion_role": "Witch"}),
        );
        assert!(matches!(
            parse_current_oracle_claim(
                &positive,
                CurrentPassivePayloadSource::Direct,
                state.n_cards,
            ),
            Some(CurrentOracleClaim::Positive {
                targets: [2, 3],
                minion_role: "Witch",
            })
        ));

        let duplicate = current_oracle(
            1,
            json!({"targets": [2, 2], "minion_role": "Twin Minion"}),
        );
        assert!(parse_current_oracle_claim(
            &duplicate,
            CurrentPassivePayloadSource::Direct,
            state.n_cards,
        )
        .is_some());

        let sentinel = current_oracle(1, json!({"no_minions": true}));
        assert_eq!(
            parse_current_oracle_claim(
                &sentinel,
                CurrentPassivePayloadSource::Direct,
                state.n_cards,
            ),
            Some(CurrentOracleClaim::NoMinions),
        );

        let mut wrong_text = positive.clone();
        wrong_text.info_text = "#2 or #3 is Witch".to_string();
        assert!(parse_current_oracle_claim(
            &wrong_text,
            CurrentPassivePayloadSource::Direct,
            state.n_cards,
        )
        .is_none());

        let mut wrong_sentinel_text = sentinel.clone();
        wrong_sentinel_text.info_text = "There are no Minions".to_string();
        assert!(parse_current_oracle_claim(
            &wrong_sentinel_text,
            CurrentPassivePayloadSource::Direct,
            state.n_cards,
        )
        .is_none());

        let mut extra = positive.clone();
        extra.info_parsed.insert("unexpected".to_string(), json!(true));
        assert!(parse_current_oracle_claim(
            &extra,
            CurrentPassivePayloadSource::Direct,
            state.n_cards,
        )
        .is_none());

        for payload in [
            json!({"targets": [3, 2], "minion_role": "Witch"}),
            json!({"targets": [2], "minion_role": "Witch"}),
            json!({"targets": [2, 3, 4], "minion_role": "Witch"}),
            json!({"targets": [0, 2], "minion_role": "Witch"}),
            json!({"targets": [2, 7], "minion_role": "Witch"}),
            json!({"targets": [2, 256], "minion_role": "Witch"}),
            json!({"targets": [2, -1], "minion_role": "Witch"}),
            json!({"targets": [2, true], "minion_role": "Witch"}),
            json!({"targets": [2, "3"], "minion_role": "Witch"}),
            json!({"targets": [2, 3], "minion_role": "witch"}),
            json!({"targets": [2, 3], "minion_role": "Pooka"}),
            json!({"no_minions": false}),
            json!({"no_minions": true, "targets": [2, 3]}),
        ] {
            let malformed = current_oracle(1, payload);
            assert!(
                parse_current_oracle_claim(
                    &malformed,
                    CurrentPassivePayloadSource::Direct,
                    state.n_cards,
                )
                .is_none(),
                "malformed current Oracle payload unexpectedly parsed: {:?}",
                malformed.info_parsed,
            );
        }

        for position in [0, 7] {
            let mut wrapped = positive.clone();
            wrapped.position = position;
            assert!(!validate_current_oracle(
                &wrapped,
                &empty_scenario(),
                &state,
                CurrentPassivePayloadSource::Direct,
            ));
        }
    }

    #[test]
    fn current_oracle_truth_supports_self_orientation_and_only_truth_duplicates() {
        let oracle = current_oracle(
            1,
            json!({"targets": [1, 3], "minion_role": "Witch"}),
        );
        let wretch = make_card(3, "Wretch", json!({}));
        let mut state = base_state(3, vec![oracle.clone(), wretch]);
        state.deck.minions = vec!["Witch".to_string()];
        let scenario = empty_scenario();

        assert_eq!(
            registered_alignment_at(3, &scenario, &state),
            EffectiveAlignment::Evil,
        );
        assert!(validate_oracle(&oracle, &scenario, &state));
        let duplicate_wretch = current_oracle(
            1,
            json!({"targets": [3, 3], "minion_role": "Witch"}),
        );
        assert!(!validate_oracle(&duplicate_wretch, &scenario, &state));

        let duplicate_twin = current_oracle(
            1,
            json!({"targets": [5, 5], "minion_role": "Twin Minion"}),
        );
        let known_good = make_card(6, "Scout", json!({}));
        let mut twin_state = base_state(8, vec![duplicate_twin.clone(), known_good]);
        twin_state.deck.minions = vec!["Twin_Minion".to_string()];
        let mut twin = empty_scenario();
        twin.evil_positions = HashMap::from([
            (4, "Twin Minion".to_string()),
            (8, "Pooka".to_string()),
        ]);
        twin.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 4,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 8,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 5,
                neighbor_pre_swap_role: "Wretch".to_string(),
            },
        });

        assert_eq!(
            registered_alignment_at(5, &twin, &twin_state),
            EffectiveAlignment::Good,
        );
        assert!(validate_oracle(&duplicate_twin, &twin, &twin_state));

        twin.corrupted.insert(1);
        assert!(!validate_oracle(&duplicate_twin, &twin, &twin_state));
        let true_sentence_from_bluff_generation = current_oracle(
            1,
            json!({"targets": [5, 6], "minion_role": "Twin Minion"}),
        );
        assert!(validate_oracle(
            &true_sentence_from_bluff_generation,
            &twin,
            &twin_state,
        ));
    }

    #[test]
    fn current_oracle_wretch_and_bluff_labels_use_authored_pool_or_fallback() {
        let direct = current_oracle(
            1,
            json!({"targets": [2, 3], "minion_role": "Twin Minion"}),
        );
        let good = make_card(2, "Scout", json!({}));
        let wretch = make_card(3, "Wretch", json!({}));
        let mut state = base_state(4, vec![direct.clone(), good.clone(), wretch.clone()]);
        state.deck.minions = vec!["Twin_Minion".to_string()];
        let scenario = empty_scenario();
        assert!(validate_oracle(&direct, &scenario, &state));

        let wrong_authored_label = current_oracle(
            1,
            json!({"targets": [2, 3], "minion_role": "Witch"}),
        );
        assert!(!validate_oracle(
            &wrong_authored_label,
            &scenario,
            &state,
        ));

        let poet = current_poet(
            "Oracle",
            json!({"targets": [2, 3], "minion_role": "Twin Minion"}),
        );
        let poet_state = base_state(4, vec![poet.clone(), good.clone(), wretch.clone()]);
        let poet_state = GameState {
            deck: state.deck.clone(),
            ..poet_state
        };
        assert_eq!(
            validate_oracle(&direct, &scenario, &state),
            validate_poet(&poet, &scenario, &poet_state),
        );

        state.deck.minions.clear();
        let fallback = current_oracle(
            1,
            json!({"targets": [2, 3], "minion_role": "Witch"}),
        );
        assert!(validate_oracle(&fallback, &scenario, &state));

        let bluff = current_oracle(
            1,
            json!({"targets": [2, 4], "minion_role": "Twin Minion"}),
        );
        let other_good = make_card(4, "Hunter", json!({}));
        let mut bluff_state = base_state(4, vec![bluff.clone(), good, wretch, other_good]);
        bluff_state.deck.minions = vec!["Twin_Minion".to_string()];
        let mut lying = empty_scenario();
        lying.corrupted.insert(1);
        assert!(validate_oracle(&bluff, &lying, &bluff_state));
        assert!(!validate_oracle(
            &current_oracle(
                1,
                json!({"targets": [2, 4], "minion_role": "Witch"}),
            ),
            &lying,
            &bluff_state,
        ));
        assert!(!validate_oracle(
            &current_oracle(
                1,
                json!({"targets": [2, 3], "minion_role": "Twin Minion"}),
            ),
            &lying,
            &bluff_state,
        ));
        bluff_state.deck.minions.clear();
        assert!(validate_oracle(
            &current_oracle(
                1,
                json!({"targets": [2, 4], "minion_role": "Witch"}),
            ),
            &lying,
            &bluff_state,
        ));
    }

    #[test]
    fn current_oracle_uses_exact_anonymous_wretch_and_sentinel_assignments() {
        let positive = current_oracle(
            1,
            json!({"targets": [2, 3], "minion_role": "Witch"}),
        );
        let known_good = make_card(2, "Scout", json!({}));
        let mut state = base_state(4, vec![positive.clone(), known_good]);
        state.deck.minions = vec!["Witch".to_string()];
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Pooka".to_string());
        let sentinel = current_oracle(1, json!({"no_minions": true}));

        assert!(validate_oracle(&positive, &scenario, &state));
        assert!(!validate_oracle(&sentinel, &scenario, &state));

        let mut lying = scenario.clone();
        lying.corrupted.insert(1);
        assert!(!validate_oracle(&positive, &lying, &state));
        assert!(!validate_oracle(&sentinel, &lying, &state));

        state.board_outcast_count = Some(0);
        assert!(!validate_oracle(&positive, &scenario, &state));
        assert!(validate_oracle(&sentinel, &scenario, &state));
        assert!(validate_oracle(&positive, &lying, &state));

        let known_minion = make_card(3, "Twin Minion", json!({}));
        state.cards.push(known_minion);
        assert!(!validate_oracle(&sentinel, &scenario, &state));
    }

    #[test]
    fn current_scout_and_oracle_share_one_explicit_wretch_register_as_draw() {
        let scout = current_scout(
            1,
            json!({"evil_role": "Witch", "distance": 4}),
        );
        let oracle = current_oracle(
            2,
            json!({"targets": [4, 5], "minion_role": "Twin Minion"}),
        );
        let mut state = base_state(8, vec![scout.clone(), oracle.clone()]);
        state.deck.minions = vec!["Witch".to_string(), "Twin_Minion".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions = HashMap::from([
            (4, "Twin Minion".to_string()),
            (8, "Pooka".to_string()),
        ]);
        scenario.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 4,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 8,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 5,
                neighbor_pre_swap_role: "Wretch".to_string(),
            },
        });

        assert!(validate_scout(&scout, &scenario, &state));
        assert!(validate_oracle(&oracle, &scenario, &state));
        assert!(!validate_current_register_as_consistency(
            &scenario,
            &state,
        ));

        let matching_oracle = current_oracle(
            2,
            json!({"targets": [4, 5], "minion_role": "Witch"}),
        );
        state.cards[1] = matching_oracle;
        assert!(validate_current_register_as_consistency(
            &scenario,
            &state,
        ));
    }

    #[test]
    fn legacy_scout_zero_distance_preserves_the_one_evil_sentinel() {
        let scout = make_card(
            4,
            "Scout",
            json!({"evil_role": "Pooka", "distance": 0}),
        );
        let state = base_state(7, vec![scout.clone()]);
        let mut one_evil = empty_scenario();
        one_evil.evil_positions.insert(2, "Pooka".to_string());

        assert!(validate_scout(&scout, &one_evil, &state));
        one_evil.corrupted.insert(4);
        assert!(!validate_scout(&scout, &one_evil, &state));
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
    fn judge_uses_shaman_copied_confessor_status_on_both_endpoints() {
        let says_truthful_source = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": false}),
        );
        let says_lying_source = make_card(
            1,
            "Judge",
            json!({"target": 2, "is_lying": true}),
        );
        let says_truthful_target = make_card(
            1,
            "Judge",
            json!({"target": 3, "is_lying": false}),
        );
        let says_lying_target = make_card(
            1,
            "Judge",
            json!({"target": 3, "is_lying": true}),
        );
        let says_truthful_control = make_card(
            1,
            "Judge",
            json!({"target": 4, "is_lying": false}),
        );
        let says_lying_control = make_card(
            1,
            "Judge",
            json!({"target": 4, "is_lying": true}),
        );
        let state = base_state(
            4,
            vec![
                says_truthful_source.clone(),
                make_card(2, "Baker", json!({})),
                make_card(3, "Scout", json!({})),
                make_card(4, "Witness", json!({})),
            ],
        );
        let mut scenario = empty_scenario();
        scenario.corrupted = HashSet::from([2, 3, 4]);
        scenario.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 3,
            copied_role: "Confessor".to_string(),
            target_previous_roles: vec!["Scout".to_string()],
        });

        assert!(validate_judge(&says_truthful_source, &scenario, &state));
        assert!(!validate_judge(&says_lying_source, &scenario, &state));
        assert!(validate_judge(&says_truthful_target, &scenario, &state));
        assert!(!validate_judge(&says_lying_target, &scenario, &state));
        assert!(!validate_judge(&says_truthful_control, &scenario, &state));
        assert!(validate_judge(&says_lying_control, &scenario, &state));

        scenario.corrupted.insert(1);
        assert!(!validate_judge(&says_truthful_source, &scenario, &state));
        assert!(validate_judge(&says_lying_source, &scenario, &state));
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

    fn current_jester_registered_count_state(jester: CardInfo) -> (GameState, Scenario) {
        let mut state = base_state(
            4,
            vec![
                jester,
                make_card(2, "Wretch", json!({})),
                make_card(3, "Bard", json!({})),
                make_card(4, "Puppet", json!({})),
            ],
        );
        state.deck.villagers = vec!["Jester".to_string(), "Bard".to_string()];
        state.deck.outcasts = vec!["Wretch".to_string()];
        state.deck.minions = vec!["Spy".to_string(), "Puppeteer".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(3, "Spy".to_string());
        scenario.puppet_position = Some(4);
        (state, scenario)
    }

    #[test]
    fn current_jester_accepts_only_exact_unused_or_ordered_shapes() {
        let unused = make_card(1, "Jester", json!({"jester_variant": POET_CURRENT_VARIANT}));
        let (unused_state, unused_scenario) = current_jester_registered_count_state(unused.clone());
        assert!(validate_jester(&unused, &unused_scenario, &unused_state));
        assert!(!card_has_normal_clue(&unused, true));

        let marked_scalar = make_card(
            1,
            "Jester",
            json!({
                "jester_variant": POET_CURRENT_VARIANT,
                "targets": [4, 2, 3],
                "evil_count": 2,
            }),
        );
        let (scalar_state, scalar_scenario) =
            current_jester_registered_count_state(marked_scalar.clone());
        assert!(!validate_jester(
            &marked_scalar,
            &scalar_scenario,
            &scalar_state,
        ));

        let event = current_jester_result_event(
            1,
            0,
            "either",
            [4, 2, 3],
            2,
            1,
            0,
            "single_callback_suffix",
        );
        let current = current_jester_ledger(1, vec![event]);
        let (state, scenario) = current_jester_registered_count_state(current.clone());
        assert_eq!(current.info_text, "Among:\n#2, #3, #4:\nThere are 2 Evils");
        assert!(validate_jester(&current, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        let mut sorted_references = current.clone();
        sorted_references.info_parsed["callback_events"][0]["references"] = json!([2, 3, 4]);
        assert!(validate_jester(&sorted_references, &scenario, &state));

        let duplicate_display_ids = current_jester_ledger(
            1,
            vec![current_jester_result_event_with_references(
                1,
                0,
                "either",
                [4, 2, 3],
                [2, 2, 4],
                2,
                1,
                0,
                "single_callback_suffix",
            )],
        );
        let (duplicate_state, duplicate_scenario) =
            current_jester_registered_count_state(duplicate_display_ids.clone());
        assert_eq!(
            duplicate_display_ids.info_text,
            "Among:\n#2, #2, #4:\nThere are 2 Evils"
        );
        assert!(validate_jester(
            &duplicate_display_ids,
            &duplicate_scenario,
            &duplicate_state,
        ));

        let mut out_of_board_reference = duplicate_display_ids.clone();
        out_of_board_reference.info_parsed["callback_events"][0]["references"] =
            json!([2, 2, 5]);
        assert!(!validate_jester(
            &out_of_board_reference,
            &duplicate_scenario,
            &duplicate_state,
        ));

        let mut malformed_latest = current.clone();
        malformed_latest.info_parsed["evil_count"] = json!(1);
        assert!(!validate_jester(&malformed_latest, &scenario, &state));

        let mut duplicate_target = current.clone();
        duplicate_target.info_parsed["callback_events"][0]["targets"] = json!([4, 2, 2]);
        assert!(!validate_jester(&duplicate_target, &scenario, &state));

        let current_poet = make_card(
            1,
            "Poet",
            json!({
                "poet_variant": POET_CURRENT_VARIANT,
                "copied_role": "Jester",
                "targets": [2, 3, 4],
                "evil_count": 2,
            }),
        );
        assert!(!validate_poet(&current_poet, &scenario, &state));
    }

    #[test]
    fn current_jester_uses_register_as_first_and_false_count_complement() {
        let truthful = current_jester_ledger(
            1,
            vec![current_jester_result_event(
                1,
                0,
                "either",
                [4, 2, 3],
                2,
                1,
                0,
                "single_callback_suffix",
            )],
        );
        let (mut state, scenario) = current_jester_registered_count_state(truthful.clone());
        assert!(validate_jester(&truthful, &scenario, &state));

        let false_truth = current_jester_ledger(
            1,
            vec![current_jester_result_event(
                1,
                0,
                "either",
                [4, 2, 3],
                3,
                1,
                0,
                "single_callback_suffix",
            )],
        );
        state.cards[0] = false_truth.clone();
        assert!(!validate_jester(&false_truth, &scenario, &state));

        let mut corrupted = scenario;
        corrupted.corrupted.insert(1);
        assert!(validate_jester(&false_truth, &corrupted, &state));
        assert!(!validate_jester(&truthful, &corrupted, &state));
    }

    #[test]
    fn current_jester_activation_evidence_matches_bridge_reachability() {
        let single = |evidence: &str, generation: usize| {
            current_jester_ledger(
                1,
                vec![current_jester_result_event(
                    1,
                    0,
                    "either",
                    [1, 2, 3],
                    0,
                    1,
                    generation,
                    evidence,
                )],
            )
        };
        let mut state = base_state(3, vec![]);
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3];
        assert!(
            parse_current_jester_payload(&single("single_callback_suffix", 0), &state,).is_some()
        );
        assert!(
            parse_current_jester_payload(&single("single_callback_suffix", 1), &state,).is_none()
        );
        assert!(
            parse_current_jester_payload(&single("session_reset_generation", 1), &state,).is_some()
        );
        assert!(
            parse_current_jester_payload(&single("session_reset_generation", 0), &state,).is_none()
        );

        let dual = current_jester_ledger(
            1,
            vec![
                current_jester_result_event(
                    1,
                    0,
                    "real",
                    [1, 2, 3],
                    0,
                    1,
                    0,
                    "same_activation_extension",
                ),
                current_jester_result_event(
                    1,
                    1,
                    "raw",
                    [1, 2, 3],
                    1,
                    1,
                    0,
                    "same_activation_extension",
                ),
            ],
        );
        assert!(parse_current_jester_payload(&dual, &state).is_some());
        let mut bad_targets = dual;
        bad_targets.info_parsed["callback_events"][1]["targets"] = json!([3, 2, 1]);
        assert!(parse_current_jester_payload(&bad_targets, &state).is_none());
    }

    fn current_jester_mixed_rambler_world(
        jester: CardInfo,
        second_is_evil_fake: bool,
        fifth_is_evil_fake_rambler: bool,
    ) -> (GameState, Scenario) {
        let interruptions = jester.info_parsed["callback_events"]
            .as_array()
            .unwrap()
            .iter()
            .filter_map(|event| {
                (event["event_kind"] == "rambler_interruption").then(|| {
                    crate::types::RamblerShutUpObservation {
                        speaker_position: 1,
                        shut_up_target: event["shut_up_target"].as_u64().unwrap() as u8,
                    }
                })
            })
            .collect();
        let mut state = base_state(
            5,
            vec![
                jester,
                make_card(2, "Rambler", json!({})),
                make_card(3, "Bard", json!({})),
                make_card(4, "Jester", json!({"jester_variant": POET_CURRENT_VARIANT})),
                make_card(
                    5,
                    if fifth_is_evil_fake_rambler {
                        "Rambler"
                    } else {
                        "Shaman"
                    },
                    json!({}),
                ),
            ],
        );
        state.deck.villagers = vec![
            "Jester".to_string(),
            "Jester".to_string(),
            "Bard".to_string(),
        ];
        state.deck.outcasts = vec!["Rambler".to_string()];
        state.deck.minions = vec!["Minion".to_string(), "Shaman".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4, 5];
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        state.rambler_shut_up_observations = interruptions;

        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(1, "Pooka".to_string());
        scenario.evil_positions.insert(5, "Shaman".to_string());
        if second_is_evil_fake {
            scenario.evil_positions.insert(2, "Minion".to_string());
        }
        scenario.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 4,
            target_position: 1,
            copied_role: "Jester".to_string(),
            target_previous_roles: vec!["Pooka".to_string()],
        });
        (state, scenario)
    }

    #[test]
    fn current_jester_rambler_replacement_is_callback_local() {
        // Runtime-Evil copied Jester dispatches truthful real Jester first and
        // lying raw Jester second. A lying-mode fake Rambler can therefore
        // replace only the raw event while the real event remains normal.
        let normal_then_interrupted = current_jester_ledger(
            1,
            vec![
                current_jester_result_event(1, 0, "real", [1, 3, 5], 2, 1, 0, "auto_use_click"),
                current_druid_interruption_event(1, 1, "raw", 2, 1, 0, "auto_use_click"),
            ],
        );
        let (state, scenario) =
            current_jester_mixed_rambler_world(normal_then_interrupted.clone(), true, true);
        assert!(validate_jester(&normal_then_interrupted, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        // The inverse ordering is independently reachable with a natural
        // truthful Rambler: it replaces real truth but not the lying raw clue.
        let interrupted_then_normal = current_jester_ledger(
            1,
            vec![
                current_druid_interruption_event(1, 0, "real", 2, 1, 0, "auto_use_click"),
                current_jester_result_event(1, 1, "raw", [1, 3, 5], 1, 1, 0, "auto_use_click"),
            ],
        );
        let (state, scenario) =
            current_jester_mixed_rambler_world(interrupted_then_normal.clone(), false, false);
        assert!(validate_jester(&interrupted_then_normal, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        // Both callbacks may also be replaced by different persistent sources:
        // natural truthful Rambler #2 for real, fake lying Rambler #5 for raw.
        let two_sources = current_jester_ledger(
            1,
            vec![
                current_druid_interruption_event(1, 0, "real", 2, 1, 0, "auto_use_click"),
                current_druid_interruption_event(1, 1, "raw", 5, 1, 0, "auto_use_click"),
            ],
        );
        let (state, scenario) =
            current_jester_mixed_rambler_world(two_sources.clone(), false, true);
        assert!(validate_jester(&two_sources, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
    }

    #[test]
    fn current_jester_opaque_real_requires_one_nonopaque_raw_callback() {
        let opaque = json!({
            "activation_id": 1,
            "callback_index": 0,
            "dispatch_path": "real",
            "event_kind": "opaque_real",
            "text": "There are NO Corrupted around me",
            "references": null,
            "settled_reveal_count": 1,
            "reset_generation": 0,
            "activation_evidence": "auto_use_click",
        });
        let raw = current_jester_result_event(1, 1, "raw", [1, 2, 3], 1, 1, 0, "auto_use_click");
        let jester = current_jester_ledger(1, vec![opaque.clone(), raw.clone()]);
        let mut state = base_state(
            4,
            vec![
                jester.clone(),
                make_card(2, "Bard", json!({})),
                make_card(3, "Shaman", json!({})),
                make_card(4, "Medium", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Jester".to_string(),
            "Medium".to_string(),
            "Bard".to_string(),
        ];
        state.deck.minions = vec!["Shaman".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(1, "Pooka".to_string());
        scenario.evil_positions.insert(3, "Shaman".to_string());
        scenario.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 4,
            target_position: 1,
            copied_role: "Medium".to_string(),
            target_previous_roles: vec!["Pooka".to_string()],
        });
        assert!(validate_jester(&jester, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        let mut opaque_last = jester.clone();
        opaque_last.info_parsed["callback_events"] = json!([raw, opaque]);
        assert!(!validate_jester(&opaque_last, &scenario, &state));

        let mut launders_jester = jester.clone();
        launders_jester.info_parsed["callback_events"][0]["text"] =
            json!("Among:\n#1, #2, #3:\nThere are 2 Evils");
        assert!(!validate_jester(&launders_jester, &scenario, &state));
        let mut launders_rambler = jester;
        launders_rambler.info_parsed["callback_events"][0]["text"] = json!("#2 SHUT UP");
        assert!(!validate_jester(&launders_rambler, &scenario, &state));
    }

    #[test]
    fn current_jester_single_raw_requires_real_role_with_no_day_output() {
        let raw_only = current_jester_ledger(
            1,
            vec![current_jester_result_event(
                1,
                0,
                "either",
                [1, 2, 3],
                1,
                1,
                0,
                "single_callback_suffix",
            )],
        );
        let mut state = base_state(
            3,
            vec![
                raw_only.clone(),
                make_card(2, "Bard", json!({})),
                make_card(3, "Bard", json!({})),
            ],
        );
        state.deck.villagers = vec!["Jester".to_string(), "Medium".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3];

        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(1);
        assert!(validate_jester(&raw_only, &drunk, &state));

        let mut medium = empty_scenario();
        medium.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Medium".to_string(),
            target_previous_roles: vec!["Drunk".to_string()],
        });
        medium.drunk_position = Some(1);
        assert!(!validate_jester(&raw_only, &medium, &state));
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
    fn rambler_uses_shaman_copied_confessor_appearance_on_nonconfessor_speaker() {
        let state = base_state(
            5,
            vec![
                make_card(1, "Rambler", json!({"quote_observed": true})),
                make_card(2, "Scout", json!({"shut_up_target": 1})),
                make_card(3, "Baker", json!({})),
            ],
        );
        let mut copied_confessor = empty_scenario();
        copied_confessor.corrupted.insert(2);
        copied_confessor.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 3,
            copied_role: "Confessor".to_string(),
            target_previous_roles: vec!["Baker".to_string()],
        });

        assert_eq!(
            truth_status(2, &copied_confessor, &state),
            TruthStatus::Lying,
        );
        assert_eq!(
            truth_appearance_status(2, &copied_confessor, &state),
            TruthStatus::Truthful,
        );
        assert!(validate_rambler_shut_ups(&copied_confessor, &state));

        copied_confessor.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 3,
            target_position: 4,
            copied_role: "Confessor".to_string(),
            target_previous_roles: vec!["Baker".to_string()],
        });
        assert!(!validate_rambler_shut_ups(&copied_confessor, &state));
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
    fn slayer_twin_body_puppet_overlay_is_exact_for_actor_and_target() {
        let stable_roles = HashMap::from([
            (1, "Puppeteer".to_string()),
            (2, "Twin Minion".to_string()),
            (5, "Pooka".to_string()),
        ]);

        let mut target_state = base_state(5, vec![make_card(4, "Slayer", json!({}))]);
        target_state.deck.minions = vec!["Puppeteer".to_string(), "Twin Minion".to_string()];
        target_state.deck.demons = vec!["Pooka".to_string()];
        target_state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 4,
            target_pos: 2,
            killed: true,
            revealed_role: Some("Puppet".to_string()),
            was_evil: Some(true),
        });
        let mut no_overlay = empty_scenario();
        no_overlay.evil_positions = stable_roles.clone();
        assert!(!validate_slayer_results(&no_overlay, &target_state));

        let mut overlay = no_overlay.clone();
        overlay.puppet_position = Some(2);
        assert!(validate_slayer_results(&overlay, &target_state));

        let mut actor_state = base_state(5, vec![make_card(2, "Slayer", json!({}))]);
        actor_state.deck = target_state.deck.clone();
        actor_state.slayer_results.push(crate::types::SlayerResult {
            slayer_pos: 2,
            target_pos: 5,
            killed: true,
            revealed_role: Some("Pooka".to_string()),
            was_evil: Some(true),
        });
        assert!(validate_slayer_results(&overlay, &actor_state));
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
    fn exact_twin_moves_sole_lilis_protection_to_the_current_data_actor() {
        let mut state = base_state(3, vec![]);
        state.n_evil = 3;
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Lilis".to_string(), "Pooka".to_string()];
        state.night_kills = vec![1];
        state.night_kill_evil_count = 1;

        let mut scenario = empty_scenario();
        scenario
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        scenario.evil_positions.insert(2, "Lilis".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Lilis".to_string(),
            },
        });

        assert_eq!(
            current_data_role_at(1, &scenario, &state).as_deref(),
            Some("Lilis"),
        );
        assert_eq!(
            current_data_role_at(2, &scenario, &state).as_deref(),
            Some("Twin Minion"),
        );

        // The sole later Lilis actor installs protection on the runtime Twin
        // body that now owns its data, so a successful death there is invalid.
        assert!(!validate_lilis_night_kills(&scenario, &state));

        // The former stable-Lilis body now owns Twin data and does not retain
        // role-based Lilis protection merely because its runtime origin did.
        state.night_kills = vec![2];
        assert!(validate_lilis_night_kills(&scenario, &state));
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
    fn current_druid_schema_text_click_order_and_direct_provenance_are_exact() {
        let druid = current_druid(2, [4, 2, 3], Some("Bombardier"));
        let mut state = base_state(
            4,
            vec![
                druid.clone(),
                make_card(3, "Bombardier", json!({})),
                make_card(4, "Scout", json!({})),
            ],
        );
        state.deck.outcasts = vec!["Bombardier".to_string()];
        assert_eq!(
            druid.info_text,
            "Among #2, #3, #4\nthere is: Bombardier"
        );
        assert!(validate_druid(&druid, &empty_scenario(), &state));

        let mut malformed = druid.clone();
        malformed.info_text.push('.');
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        malformed = druid.clone();
        malformed.info_parsed.insert("extra".to_string(), json!(true));
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        malformed = druid.clone();
        malformed.info_parsed.insert("targets".to_string(), json!([2, 2, 3]));
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        malformed = druid.clone();
        malformed.info_parsed.insert("targets".to_string(), json!([2, 3, 5]));
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        let tiny = current_druid(1, [1, 2, 2], None);
        assert!(!validate_druid(
            &tiny,
            &empty_scenario(),
            &base_state(2, vec![tiny.clone()]),
        ));
        malformed = druid.clone();
        malformed
            .info_parsed
            .insert("found_outcast".to_string(), json!("Plague Doctor"));
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        malformed = druid.clone();
        malformed
            .info_parsed
            .insert("found_outcast".to_string(), json!("plague_doctor"));
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        malformed = druid.clone();
        malformed.info_parsed.insert(
            DRUID_CURRENT_VARIANT_FIELD.to_string(),
            json!("future"),
        );
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        malformed = druid.clone();
        malformed
            .info_parsed
            .insert("poet_variant".to_string(), json!(POET_CURRENT_VARIANT));
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));

        let poet = current_poet(
            "Druid",
            json!({"targets": [2, 3, 4], "found_outcast": null}),
        );
        assert!(!validate_poet(&poet, &empty_scenario(), &state));
    }

    #[test]
    fn current_druid_history_validates_every_event_and_exact_latest_alias() {
        let druid = current_druid_history(1, &[([2, 3, 4], None, 1), ([5, 1, 4], None, 5)]);
        let mut state = base_state(
            5,
            vec![
                druid.clone(),
                make_card(2, "Scout", json!({})),
                make_card(3, "Bard", json!({})),
                make_card(4, "Lover", json!({})),
                make_card(5, "Knitter", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Druid".to_string(),
            "Scout".to_string(),
            "Bard".to_string(),
            "Lover".to_string(),
            "Knitter".to_string(),
        ];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4, 5];
        assert!(validate_druid(&druid, &empty_scenario(), &state));

        let mut malformed = druid.clone();
        malformed
            .info_parsed
            .insert("targets".to_string(), json!([2, 3, 4]));
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));

        malformed = druid.clone();
        malformed.info_text = "Among #1, #4, #5\nthere are NO Outcasts.".to_string();
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));

        malformed = druid.clone();
        malformed.info_parsed["callback_events"][0]["text"] =
            json!("Among #2, #3, #4\nthere are no Outcasts");
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
    }

    #[test]
    fn current_druid_history_rejects_malformed_or_unsettled_boundaries() {
        let valid = current_druid_history(2, &[([1, 2, 3], None, 2), ([2, 3, 4], None, 4)]);
        let mut state = base_state(4, vec![valid.clone()]);
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4];
        assert!(validate_druid(&valid, &empty_scenario(), &state));

        for malformed in [
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][1]["settled_reveal_count"] = json!(1);
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][1]["settled_reveal_count"] = json!(5);
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("settled_reveal_count");
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][0]["extra"] = json!(true);
                card
            },
        ] {
            assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        }

        let mut actor_not_revealed = state.clone();
        actor_not_revealed.reveal_order = vec![1, 3, 2, 4];
        assert!(!validate_druid(
            &valid,
            &empty_scenario(),
            &actor_not_revealed,
        ));

        let mut unproven_order = state.clone();
        unproven_order.baker_rule_version = None;
        assert!(!validate_druid(&valid, &empty_scenario(), &unproven_order,));
    }

    #[test]
    fn current_druid_history_joins_hidden_outcasts_in_one_world() {
        let disjoint = current_druid_history(
            1,
            &[
                ([2, 3, 4], Some("Bombardier"), 4),
                ([5, 6, 7], Some("Bombardier"), 7),
            ],
        );
        let mut state = base_state(7, vec![disjoint.clone()]);
        state.deck.villagers = vec!["Druid".to_string()];
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.board_villager_count = Some(6);
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = (1..=7).collect();

        assert!(validate_druid(
            &current_druid(1, [2, 3, 4], Some("Bombardier")),
            &empty_scenario(),
            &state,
        ));
        assert!(validate_druid(
            &current_druid(1, [5, 6, 7], Some("Bombardier")),
            &empty_scenario(),
            &state,
        ));
        assert!(!validate_druid(&disjoint, &empty_scenario(), &state));

        let overlapping = current_druid_history(
            1,
            &[
                ([2, 3, 4], Some("Bombardier"), 4),
                ([4, 5, 6], Some("Bombardier"), 7),
            ],
        );
        assert!(validate_druid(&overlapping, &empty_scenario(), &state,));
    }

    #[test]
    fn current_druid_interruption_and_mixed_history_are_exact() {
        let interruption = current_druid_interruption_event(
            1,
            0,
            "either",
            2,
            1,
            0,
            "single_callback_suffix",
        );
        let interrupted = current_druid_ledger(1, vec![interruption.clone()]);
        let mut state = base_state(3, vec![interrupted.clone()]);
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3];
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        state.rambler_shut_up_observations = vec![crate::types::RamblerShutUpObservation {
            speaker_position: 1,
            shut_up_target: 2,
        }];
        assert!(validate_druid(&interrupted, &empty_scenario(), &state,));
        assert!(!card_has_normal_clue(&interrupted, true));

        let mut malformed = interrupted.clone();
        malformed.info_text = "#2 shut up!".to_string();
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));
        malformed = interrupted.clone();
        malformed.info_parsed["shut_up_target"] = json!(3);
        assert!(!validate_druid(&malformed, &empty_scenario(), &state));

        let normal = current_druid_history(1, &[([1, 2, 3], None, 1)]);
        let first = normal.info_parsed["callback_events"][0].clone();
        let second = current_druid_interruption_event(
            2,
            0,
            "either",
            2,
            3,
            1,
            "session_reset_generation",
        );
        let mixed = current_druid_ledger(1, vec![first, second]);
        state.cards[0] = mixed.clone();
        assert!(validate_druid(&mixed, &empty_scenario(), &state));
        assert!(card_has_normal_clue(&mixed, true));

        state.cards[0] = normal.clone();
        assert!(!validate_druid(&normal, &empty_scenario(), &state));

        let mut scalar_interruption = make_card(
            1,
            "Druid",
            json!({
                "druid_variant": "public_current",
                "shut_up_target": 2,
            }),
        );
        scalar_interruption.info_text = "#2\nshut up!".to_string();
        assert!(!validate_druid(
            &scalar_interruption,
            &empty_scenario(),
            &state,
        ));
    }

    #[test]
    fn current_druid_rambler_closure_persists_and_mixed_temporal_history_fails_closed() {
        let normal = |activation_id, boundary, generation| {
            current_druid_result_event(
                activation_id,
                0,
                "either",
                [1, 2, 4],
                Some("Drunk"),
                boundary,
                generation,
                if activation_id == 1 && generation == 0 {
                    "single_callback_suffix"
                } else {
                    "session_reset_generation"
                },
            )
        };
        let interruption = |activation_id, boundary, generation| {
            current_druid_interruption_event(
                activation_id,
                0,
                "either",
                2,
                boundary,
                generation,
                if activation_id == 1 && generation == 0 {
                    "single_callback_suffix"
                } else {
                    "session_reset_generation"
                },
            )
        };

        // Stable Spy #2 installs a lying Rambler-bluff closure at initial
        // AfterRoundStart. Baker later clears its raw pointer and changes its
        // visible/current role, but the installed closure must still replace a
        // later lying Druid result.
        let after_conversion = current_druid_ledger(3, vec![interruption(1, 2, 0)]);
        let (state, scenario) = current_druid_rambler_timeline_state(after_conversion.clone());
        assert!(validate_baker_history(&scenario, &state));
        assert!(validate_druid(&after_conversion, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        // The represented Baker/Spy chronology never changes the physical
        // Druid's CheckLyingAppearance. One persistent installed matcher cannot
        // both miss and hit it, in either chronological direction.
        for events in [
            vec![normal(1, 1, 0), interruption(2, 2, 1)],
            vec![interruption(1, 1, 0), normal(2, 2, 1)],
            vec![normal(1, 2, 0), interruption(2, 2, 1)],
        ] {
            let mixed = current_druid_ledger(3, events);
            let (state, scenario) = current_druid_rambler_timeline_state(mixed.clone());
            assert!(validate_druid(&mixed, &scenario, &state));
            assert!(!validate_current_hidden_surface_consistency(
                &scenario, &state,
            ));
        }
    }

    #[test]
    fn current_druid_rambler_identity_joins_stable_spy_cached_register_as() {
        let interruption = current_druid_ledger(
            1,
            vec![current_druid_interruption_event(
                1,
                0,
                "either",
                2,
                4,
                0,
                "single_callback_suffix",
            )],
        );
        let medium = current_medium(4, json!(2), json!("Scout"));
        let mut interrupted_state = base_state(
            4,
            vec![
                interruption.clone(),
                make_card(2, "Rambler", json!({})),
                make_card(3, "Pooka", json!({})),
                medium.clone(),
            ],
        );
        interrupted_state.deck.villagers = vec![
            "Druid".to_string(),
            "Medium".to_string(),
            "Scout".to_string(),
        ];
        interrupted_state.deck.outcasts = vec!["Rambler".to_string()];
        interrupted_state.deck.minions = vec!["Spy".to_string()];
        interrupted_state.deck.demons = vec!["Pooka".to_string()];
        interrupted_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        interrupted_state.reveal_order = vec![2, 4, 1, 3];
        interrupted_state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        interrupted_state.rambler_shut_up_observations = vec![
            crate::types::RamblerShutUpObservation {
                speaker_position: 1,
                shut_up_target: 2,
            },
        ];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Spy".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.corrupted.insert(1);

        assert!(validate_druid(&interruption, &scenario, &interrupted_state));
        assert!(validate_medium(&medium, &scenario, &interrupted_state));
        // Spy's one cache cannot register as Scout for Medium while its raw
        // bluff simultaneously installs a Rambler interruption closure.
        assert!(!validate_current_hidden_surface_consistency(
            &scenario,
            &interrupted_state,
        ));

        // Exercise the reverse implication at the temporal join boundary too:
        // a cached Rambler registration forces the raw identity, so a lying
        // Spy must interrupt the adjacent corrupted Druid instead of allowing
        // its ordinary result through. This input is kept at the join level
        // because the current authored Rambler is an Outcast, while historical
        // serialized pools may classify it differently.
        let normal = current_druid_ledger(
            1,
            vec![current_druid_result_event(
                1,
                0,
                "either",
                [1, 2, 3],
                Some("Rambler"),
                3,
                0,
                "single_callback_suffix",
            )],
        );
        let mut normal_state = base_state(
            3,
            vec![
                normal,
                make_card(2, "Scout", json!({})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        normal_state.deck.villagers = vec!["Druid".to_string(), "Scout".to_string()];
        normal_state.deck.outcasts = vec!["Rambler".to_string()];
        normal_state.deck.minions = vec!["Spy".to_string()];
        normal_state.deck.demons = vec!["Pooka".to_string()];
        normal_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        normal_state.reveal_order = vec![1, 2, 3];
        normal_state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        let timeline = baker_spy_conversion_timelines(&scenario, &normal_state)
            .into_iter()
            .next()
            .expect("the stable-Spy world has one no-conversion timeline");
        assert!(current_rambler_timeline_jointly_possible(
            &timeline,
            &HashMap::from([(2, normalize_role("Scout"))]),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &[],
            &scenario,
            &normal_state,
        ));
        assert!(!current_rambler_timeline_jointly_possible(
            &timeline,
            &HashMap::from([(2, normalize_role("Rambler"))]),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &[],
            &scenario,
            &normal_state,
        ));

        // A later Twin writer breaks that equality: physical Spy #2 retains
        // its old Scout registerAs, while incoming Twin data has an independent
        // raw selector that can install Rambler. Physical evil origin alone is
        // therefore not enough to classify the live cache as Spy-owned.
        let twin_interruption = current_druid_ledger(
            1,
            vec![current_druid_interruption_event(
                1,
                0,
                "either",
                2,
                5,
                0,
                "single_callback_suffix",
            )],
        );
        let twin_medium = current_medium(5, json!(2), json!("Scout"));
        let mut twin_state = base_state(
            5,
            vec![
                twin_interruption.clone(),
                make_card(2, "Rambler", json!({})),
                make_card(3, "Pooka", json!({})),
                make_card(4, "Twin Minion", json!({})),
                twin_medium.clone(),
            ],
        );
        twin_state.deck.villagers = vec![
            "Druid".to_string(),
            "Medium".to_string(),
            "Scout".to_string(),
        ];
        twin_state.deck.outcasts = vec!["Rambler".to_string()];
        twin_state.deck.minions = vec!["Spy".to_string(), "Twin Minion".to_string()];
        twin_state.deck.demons = vec!["Pooka".to_string()];
        twin_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        twin_state.reveal_order = vec![1, 2, 3, 4, 5];
        twin_state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        twin_state.rambler_shut_up_observations = vec![
            crate::types::RamblerShutUpObservation {
                speaker_position: 1,
                shut_up_target: 2,
            },
        ];
        let mut twin = scenario;
        twin.evil_positions.insert(4, "Twin Minion".to_string());
        twin.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 4,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Previous,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Spy".to_string(),
            },
        });
        let timeline = baker_spy_conversion_timelines(&twin, &twin_state)
            .into_iter()
            .next()
            .expect("the Twin-overwritten Spy world has one no-conversion timeline");
        assert_eq!(
            current_data_role_at_druid_observation(
                2,
                CurrentDruidObservationBoundary::SettledRevealCount(0),
                &timeline,
                &twin,
                &twin_state,
            )
            .as_deref(),
            Some("Twin Minion"),
        );
        assert!(validate_druid(&twin_interruption, &twin, &twin_state));
        assert!(validate_medium(&twin_medium, &twin, &twin_state));
        assert!(current_rambler_timeline_jointly_possible(
            &timeline,
            &HashMap::from([(2, normalize_role("Scout"))]),
            &HashMap::new(),
            &HashMap::new(),
            &HashMap::new(),
            &[],
            &twin,
            &twin_state,
        ));
        assert!(validate_current_hidden_surface_consistency(
            &twin,
            &twin_state,
        ));
    }

    #[test]
    fn current_druid_temporal_rambler_join_keeps_unrelated_positive_and_negative_facts() {
        let druid = current_druid_history(1, &[([1, 2, 5], None, 1)]);
        let scout = make_card(4, "Scout", json!({"legacy_clue": true}));
        let mut state = base_state(
            5,
            vec![
                druid.clone(),
                make_card(2, "Bard", json!({})),
                make_card(3, "Rambler", json!({})),
                scout,
                make_card(5, "Lover", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Druid".to_string(),
            "Bard".to_string(),
            "Scout".to_string(),
            "Lover".to_string(),
        ];
        state.deck.outcasts = vec!["Rambler".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4, 5];
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        let scenario = empty_scenario();
        assert!(validate_druid(&druid, &scenario, &state));

        // Adding an ordered Druid ledger moves Rambler validation into the
        // temporal hidden-surface join. It must not discard unrelated normal
        // clue evidence: real Rambler #3 would have replaced truthful #4.
        assert!(validate_rambler_shut_ups(&scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        state.cards[3].info_parsed = json!({"shut_up_target": 3})
            .as_object()
            .unwrap()
            .clone();
        state.rambler_shut_up_observations = vec![crate::types::RamblerShutUpObservation {
            speaker_position: 4,
            shut_up_target: 3,
        }];
        let timelines = baker_spy_conversion_timelines(&scenario, &state);
        assert!(timelines.iter().any(|timeline| {
            current_rambler_installed_matchers(
                3,
                CurrentRamblerIdentity::Existing,
                timeline,
                &scenario,
                &state,
            ) == Some(RAMBLER_MATCHES_TRUTHFUL)
        }));
        assert!(timelines.iter().any(|timeline| {
            current_rambler_timeline_jointly_possible(
                timeline,
                &HashMap::new(),
                &HashMap::new(),
                &HashMap::new(),
                &HashMap::new(),
                &[],
                &scenario,
                &state,
            )
        }));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
    }

    #[test]
    fn current_druid_multiple_ordered_cards_share_one_baker_spy_timeline() {
        let first = current_druid_history(4, &[([1, 2, 4], None, 1)]);
        let second = current_druid_history(5, &[([1, 2, 5], None, 3)]);
        let mut state = base_state(
            5,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                make_card(3, "Pooka", json!({})),
                first.clone(),
                second.clone(),
            ],
        );
        state.deck.villagers = vec![
            "Baker".to_string(),
            "Druid".to_string(),
            "Druid".to_string(),
        ];
        state.deck.minions = vec!["Spy".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![4, 1, 5, 2, 3];
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Spy".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());

        assert!(validate_baker_history(&scenario, &state));
        assert!(validate_druid(&first, &scenario, &state));
        assert!(validate_druid(&second, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
    }

    #[test]
    fn current_druid_temporal_rambler_fails_closed_on_preserved_confessor_status() {
        let opaque = json!({
            "activation_id": 1,
            "callback_index": 0,
            "dispatch_path": "real",
            "event_kind": "opaque_real",
            "text": "I am dizzy",
            "references": null,
            "settled_reveal_count": 4,
            "reset_generation": 0,
            "activation_evidence": "auto_use_click",
        });
        let raw = current_druid_result_event(
            1,
            1,
            "raw",
            [2, 3, 4],
            Some("Drunk"),
            4,
            0,
            "auto_use_click",
        );
        let druid = current_druid_ledger(1, vec![opaque, raw]);
        let mut state = base_state(
            4,
            vec![
                druid.clone(),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Pooka", json!({})),
                make_card(4, "Scout", json!({})),
            ],
        );
        state.deck.villagers = vec!["Confessor".to_string(), "Scout".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4];
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        let mut scenario = empty_scenario();
        scenario
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Confessor".to_string(),
            },
        });

        // The real Confessor callback plus raw Druid callback is structurally
        // valid. Confessor's Init, however, writes a physical
        // AppearTruthfull status that Twin/InitWithNoReset preserves while the
        // current public provider is Druid. Scenario has no exact status or
        // resistance history for that surface, so temporal Rambler validation
        // must not infer appearance from either role name.
        assert!(validate_druid(&druid, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
    }

    #[test]
    fn current_rambler_keeps_exact_copied_confessor_status_after_baker_writer() {
        let mut state = base_state(
            4,
            vec![
                make_card(1, "Baker", json!({"original_role": "Confessor"})),
                make_card(2, "Confessor", json!({})),
                make_card(3, "Baker", json!({"original_role": "original"})),
                make_card(4, "Shaman", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Baker".to_string(),
            "Confessor".to_string(),
            "Druid".to_string(),
        ];
        state.deck.minions = vec!["Shaman".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![2, 3, 1, 4];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Shaman".to_string());
        scenario.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Confessor".to_string(),
            target_previous_roles: vec!["Druid".to_string()],
        });

        assert_eq!(
            current_data_role_at(1, &scenario, &state).as_deref(),
            Some("Baker"),
        );
        assert_eq!(
            current_rambler_speaker_matcher_at(
                1,
                CurrentRamblerBoundary::Final,
                &BakerSpyTimeline::default(),
                &scenario,
                &state,
            ),
            Some(RAMBLER_MATCHES_TRUTHFUL),
        );
    }

    #[test]
    fn current_rambler_confessor_status_fails_closed_across_shaman_baker_and_later_twin_writers() {
        let mut shaman_state = base_state(
            3,
            vec![
                make_card(1, "Druid", json!({})),
                make_card(2, "Druid", json!({})),
                make_card(3, "Shaman", json!({})),
            ],
        );
        shaman_state.deck.villagers = vec!["Confessor".to_string(), "Druid".to_string()];
        shaman_state.deck.minions = vec!["Shaman".to_string()];
        shaman_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        shaman_state.reveal_order = vec![1, 2, 3];
        let timeline = BakerSpyTimeline::default();
        assert_eq!(
            current_rambler_speaker_matcher_at(
                1,
                CurrentRamblerBoundary::Final,
                &timeline,
                &empty_scenario(),
                &shaman_state,
            ),
            Some(RAMBLER_MATCHES_TRUTHFUL),
        );
        let mut shaman = empty_scenario();
        shaman.evil_positions.insert(3, "Shaman".to_string());
        shaman.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Druid".to_string(),
            target_previous_roles: vec!["Confessor".to_string()],
        });
        assert_eq!(current_data_role_at(1, &shaman, &shaman_state).as_deref(), Some("Druid"));
        assert_eq!(
            current_rambler_speaker_matcher_at(
                1,
                CurrentRamblerBoundary::Final,
                &timeline,
                &shaman,
                &shaman_state,
            ),
            None,
        );

        let stale_twin_state = base_state(
            3,
            vec![
                make_card(1, "Druid", json!({})),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        let mut stale_twin = empty_scenario();
        stale_twin
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        stale_twin
            .evil_positions
            .insert(3, "Pooka".to_string());
        stale_twin.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Druid".to_string(),
            },
        });
        let selected_druid = HashMap::from([(1, "druid".to_string())]);
        assert_eq!(
            current_rambler_speaker_matcher_at_with_hidden_labels(
                1,
                CurrentRamblerBoundary::Final,
                &timeline,
                &HashMap::new(),
                &selected_druid,
                &stale_twin,
                &stale_twin_state,
            ),
            Some(RAMBLER_MATCHES_LYING),
        );

        let mut baker_state = base_state(
            2,
            vec![
                make_card(1, "Baker", json!({"original_role": "Confessor"})),
                make_card(2, "Baker", json!({"original_role": "original"})),
            ],
        );
        baker_state.deck.villagers = vec!["Baker".to_string(), "Confessor".to_string()];
        baker_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        baker_state.reveal_order = vec![2, 1];
        assert!(validate_baker_history(&empty_scenario(), &baker_state));
        assert!(baker_history_supports_pre_day_role(
            &empty_scenario(),
            &baker_state,
            1,
            "Confessor",
        ));
        assert_eq!(
            current_rambler_speaker_matcher_at(
                1,
                CurrentRamblerBoundary::Final,
                &timeline,
                &empty_scenario(),
                &baker_state,
            ),
            None,
        );

        let mut twin_state = base_state(
            5,
            vec![
                make_card(1, "Twin Minion", json!({})),
                make_card(2, "Druid", json!({})),
                make_card(3, "Pooka", json!({})),
                make_card(4, "Druid", json!({})),
                make_card(5, "Shaman", json!({})),
            ],
        );
        twin_state.deck.villagers = vec!["Confessor".to_string(), "Druid".to_string()];
        twin_state.deck.minions = vec!["Twin Minion".to_string(), "Shaman".to_string()];
        twin_state.deck.demons = vec!["Pooka".to_string()];
        twin_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        twin_state.reveal_order = vec![1, 2, 3, 4, 5];
        let mut twin_then_shaman = empty_scenario();
        twin_then_shaman
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        twin_then_shaman
            .evil_positions
            .insert(3, "Pooka".to_string());
        twin_then_shaman
            .evil_positions
            .insert(5, "Shaman".to_string());
        twin_then_shaman.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Confessor".to_string(),
            },
        });
        twin_then_shaman.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 4,
            target_position: 2,
            copied_role: "Druid".to_string(),
            target_previous_roles: vec!["Twin Minion".to_string()],
        });
        assert_eq!(
            current_data_role_at(2, &twin_then_shaman, &twin_state).as_deref(),
            Some("Druid"),
        );
        assert_eq!(
            current_rambler_speaker_matcher_at(
                2,
                CurrentRamblerBoundary::Final,
                &timeline,
                &twin_then_shaman,
                &twin_state,
            ),
            None,
        );
    }

    #[test]
    fn current_rambler_confessor_status_fails_closed_on_pre_writer_raw_bluffs() {
        let mut baker_state = base_state(
            3,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        baker_state.deck.villagers = vec!["Baker".to_string(), "Confessor".to_string()];
        baker_state.deck.minions = vec!["Spy".to_string()];
        baker_state.deck.demons = vec!["Pooka".to_string()];
        baker_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        baker_state.reveal_order = vec![1, 2, 3];
        let mut baker = empty_scenario();
        baker.evil_positions.insert(2, "Spy".to_string());
        baker.evil_positions.insert(3, "Pooka".to_string());
        let baker_timeline = baker_spy_conversion_timelines(&baker, &baker_state)
            .into_iter()
            .find(|timeline| timeline.contains_position(2))
            .expect("the exact Baker history must include the converted Spy");
        assert_eq!(
            current_rambler_speaker_matcher_at(
                2,
                CurrentRamblerBoundary::Final,
                &baker_timeline,
                &baker,
                &baker_state,
            ),
            None,
        );
        baker_state.deck.villagers.push("Scout".to_string());
        let selected_spy_cache = HashMap::from([(2, "scout".to_string())]);
        assert_eq!(
            current_rambler_speaker_matcher_at_with_hidden_labels(
                2,
                CurrentRamblerBoundary::Final,
                &baker_timeline,
                &selected_spy_cache,
                &HashMap::new(),
                &baker,
                &baker_state,
            ),
            Some(RAMBLER_MATCHES_LYING),
        );

        let mut twin_state = base_state(
            3,
            vec![
                make_card(1, "Spy", json!({})),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        twin_state.deck.villagers = vec!["Scout".to_string()];
        twin_state.deck.minions = vec!["Twin Minion".to_string(), "Spy".to_string()];
        twin_state.deck.demons = vec!["Pooka".to_string()];
        let mut twin = empty_scenario();
        twin.evil_positions.insert(1, "Twin Minion".to_string());
        twin.evil_positions.insert(2, "Spy".to_string());
        twin.evil_positions.insert(3, "Pooka".to_string());
        twin.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Spy".to_string(),
            },
        });
        let timeline = BakerSpyTimeline::default();
        assert_eq!(
            current_rambler_speaker_matcher_at(
                1,
                CurrentRamblerBoundary::Final,
                &timeline,
                &twin,
                &twin_state,
            ),
            None,
        );
        assert_eq!(
            current_rambler_speaker_matcher_at(
                2,
                CurrentRamblerBoundary::Final,
                &timeline,
                &twin,
                &twin_state,
            ),
            Some(RAMBLER_MATCHES_LYING),
        );
        twin_state.deck.villagers.push("Confessor".to_string());
        assert_eq!(
            current_rambler_speaker_matcher_at(
                2,
                CurrentRamblerBoundary::Final,
                &timeline,
                &twin,
                &twin_state,
            ),
            None,
        );
        assert_eq!(
            current_rambler_speaker_matcher_at_with_hidden_labels(
                2,
                CurrentRamblerBoundary::Final,
                &timeline,
                &selected_spy_cache,
                &HashMap::new(),
                &twin,
                &twin_state,
            ),
            Some(RAMBLER_MATCHES_LYING),
        );

        let mut shaman_state = base_state(
            4,
            vec![
                make_card(1, "Druid", json!({})),
                make_card(2, "Druid", json!({})),
                make_card(3, "Shaman", json!({})),
                make_card(4, "Scout", json!({})),
            ],
        );
        shaman_state.deck.villagers = vec!["Druid".to_string(), "Scout".to_string()];
        shaman_state.deck.minions = vec!["Spy".to_string(), "Shaman".to_string()];
        let mut shaman = empty_scenario();
        shaman.evil_positions.insert(1, "Spy".to_string());
        shaman.evil_positions.insert(3, "Shaman".to_string());
        shaman.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 2,
            target_position: 1,
            copied_role: "Druid".to_string(),
            target_previous_roles: vec!["Spy".to_string()],
        });
        assert_eq!(
            current_rambler_speaker_matcher_at(
                1,
                CurrentRamblerBoundary::Final,
                &timeline,
                &shaman,
                &shaman_state,
            ),
            Some(RAMBLER_MATCHES_LYING),
        );
        shaman_state.deck.villagers.push("Confessor".to_string());
        let selected_druid = HashMap::from([(1, "druid".to_string())]);
        assert_eq!(
            current_rambler_speaker_matcher_at(
                1,
                CurrentRamblerBoundary::Final,
                &timeline,
                &shaman,
                &shaman_state,
            ),
            None,
        );
        assert_eq!(
            current_rambler_speaker_matcher_at_with_hidden_labels(
                1,
                CurrentRamblerBoundary::Final,
                &timeline,
                &HashMap::new(),
                &selected_druid,
                &shaman,
                &shaman_state,
            ),
            Some(RAMBLER_MATCHES_LYING),
        );
    }

    #[test]
    fn current_druid_ordered_callback_schema_and_activation_evidence_are_exact() {
        let valid = current_druid_history(
            1,
            &[([1, 2, 3], None, 1), ([1, 3, 4], None, 4)],
        );
        let mut state = base_state(4, vec![valid.clone()]);
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4];
        assert!(validate_druid(&valid, &empty_scenario(), &state));

        let malformed = [
            {
                let mut card = valid.clone();
                card.info_parsed["callback_ledger_variant"] = json!("future");
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][0]["activation_evidence"] =
                    json!("manual_single");
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][0]
                    .as_object_mut()
                    .unwrap()
                    .remove("reset_generation");
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][0]["native_use_count"] = json!(1);
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][1]["activation_id"] = json!(3);
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][1]["reset_generation"] = json!(0);
                card.info_parsed["callback_events"][0]["reset_generation"] = json!(1);
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][0]["activation_evidence"] =
                    json!("same_activation_extension");
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][1]["activation_evidence"] =
                    json!("single_callback_suffix");
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][0]["reset_generation"] = json!(1);
                card.info_parsed["callback_events"][1]["reset_generation"] = json!(2);
                card
            },
            {
                let mut card = valid.clone();
                card.info_parsed["callback_events"][0]["activation_evidence"] =
                    json!("session_reset_generation");
                card
            },
        ];
        for card in malformed {
            assert!(!validate_druid(&card, &empty_scenario(), &state));
        }

        let initial_session_single = current_druid_ledger(
            1,
            vec![current_druid_result_event(
                1,
                0,
                "either",
                [1, 2, 3],
                None,
                4,
                1,
                "session_reset_generation",
            )],
        );
        assert!(parse_current_druid_payload(&initial_session_single, &state).is_some());

        let skipped_generation_single = current_druid_ledger(
            1,
            vec![
                current_druid_result_event(
                    1,
                    0,
                    "either",
                    [1, 2, 3],
                    None,
                    1,
                    0,
                    "single_callback_suffix",
                ),
                current_druid_result_event(
                    2,
                    0,
                    "either",
                    [1, 3, 4],
                    None,
                    4,
                    3,
                    "session_reset_generation",
                ),
            ],
        );
        assert!(parse_current_druid_payload(&skipped_generation_single, &state).is_some());

        let next_generation_dual = current_druid_ledger(
            1,
            vec![
                current_druid_result_event(
                    1,
                    0,
                    "either",
                    [1, 2, 3],
                    None,
                    1,
                    0,
                    "single_callback_suffix",
                ),
                current_druid_result_event(
                    2,
                    0,
                    "real",
                    [1, 3, 4],
                    None,
                    4,
                    1,
                    "session_reset_generation",
                ),
                current_druid_result_event(
                    2,
                    1,
                    "raw",
                    [1, 3, 4],
                    None,
                    4,
                    1,
                    "session_reset_generation",
                ),
            ],
        );
        assert!(parse_current_druid_payload(&next_generation_dual, &state).is_some());

        let initial_session_dual = current_druid_ledger(
            1,
            vec![
                current_druid_result_event(
                    1,
                    0,
                    "real",
                    [1, 2, 3],
                    None,
                    4,
                    1,
                    "session_reset_generation",
                ),
                current_druid_result_event(
                    1,
                    1,
                    "raw",
                    [1, 2, 3],
                    None,
                    4,
                    1,
                    "session_reset_generation",
                ),
            ],
        );
        assert!(parse_current_druid_payload(&initial_session_dual, &state).is_none());

        let skipped_generation_dual = current_druid_ledger(
            1,
            vec![
                current_druid_result_event(
                    1,
                    0,
                    "either",
                    [1, 2, 3],
                    None,
                    1,
                    0,
                    "single_callback_suffix",
                ),
                current_druid_result_event(
                    2,
                    0,
                    "real",
                    [1, 3, 4],
                    None,
                    4,
                    2,
                    "session_reset_generation",
                ),
                current_druid_result_event(
                    2,
                    1,
                    "raw",
                    [1, 3, 4],
                    None,
                    4,
                    2,
                    "session_reset_generation",
                ),
            ],
        );
        assert!(parse_current_druid_payload(&skipped_generation_dual, &state).is_none());

        let opaque_extension = json!({
            "activation_id": 1,
            "callback_index": 0,
            "dispatch_path": "real",
            "event_kind": "opaque_real",
            "text": "foreign callback",
            "references": null,
            "settled_reveal_count": 4,
            "reset_generation": 0,
            "activation_evidence": "same_activation_extension",
        });
        let opaque_extension = current_druid_ledger(
            1,
            vec![
                opaque_extension,
                current_druid_result_event(
                    1,
                    1,
                    "raw",
                    [1, 2, 3],
                    None,
                    4,
                    0,
                    "same_activation_extension",
                ),
            ],
        );
        assert!(parse_current_druid_payload(&opaque_extension, &state).is_none());

        let old_v0 = make_card(
            1,
            "Druid",
            json!({
                "druid_variant": "public_current",
                "targets": [1, 2, 3],
                "found_outcast": null,
                "observations": [{
                    "targets": [1, 2, 3],
                    "found_outcast": null,
                    "text": "Among #1, #2, #3\nthere are NO Outcasts",
                    "settled_reveal_count": 1,
                }],
            }),
        );
        assert!(!validate_druid(&old_v0, &empty_scenario(), &state));
    }

    #[test]
    fn current_druid_dispatch_groups_require_available_real_then_raw_callbacks() {
        let real = current_druid_result_event(
            1,
            0,
            "real",
            [1, 2, 3],
            None,
            3,
            0,
            "auto_use_click",
        );
        let raw = current_druid_result_event(
            1,
            1,
            "raw",
            [1, 2, 3],
            Some("Drunk"),
            3,
            0,
            "auto_use_click",
        );
        let dual = current_druid_ledger(1, vec![real.clone(), raw.clone()]);
        let mut state = base_state(
            3,
            vec![
                dual.clone(),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Pooka", json!({})),
            ],
        );
        state.deck.villagers = vec!["Druid".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3];
        let mut moved = empty_scenario();
        moved
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        moved.evil_positions.insert(3, "Pooka".to_string());
        moved.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Druid".to_string(),
            },
        });
        assert!(validate_druid(&dual, &moved, &state));

        let mut transient_event = real.clone();
        transient_event["dispatch_path"] = json!("either");
        transient_event["settled_reveal_count"] = json!(4);
        let transient = current_druid_ledger(1, vec![transient_event]);
        let mut medium = current_medium(4, json!(1), json!("Druid"));
        let mut transient_state = base_state(
            4,
            vec![
                transient.clone(),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Pooka", json!({})),
                medium.clone(),
            ],
        );
        transient_state.deck.villagers = vec!["Druid".to_string(), "Medium".to_string()];
        transient_state.deck.outcasts = vec!["Drunk".to_string()];
        transient_state.deck.minions = vec!["Twin Minion".to_string()];
        transient_state.deck.demons = vec!["Pooka".to_string()];
        transient_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        transient_state.reveal_order = vec![1, 2, 3, 4];
        let mut transient_world = moved.clone();
        transient_world.corrupted.insert(4);
        assert!(validate_druid(
            &transient,
            &transient_world,
            &transient_state,
        ));
        assert!(validate_medium(
            &medium,
            &transient_world,
            &transient_state,
        ));
        transient_state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        assert!(validate_current_hidden_surface_consistency(
            &transient_world,
            &transient_state,
        ));

        let interrupted = current_druid_ledger(
            1,
            vec![current_druid_interruption_event(
                1,
                0,
                "either",
                2,
                4,
                0,
                "auto_use_click",
            )],
        );
        let mut interrupted_state = transient_state.clone();
        interrupted_state.cards[0] = interrupted.clone();
        interrupted_state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        interrupted_state.rambler_shut_up_observations =
            vec![crate::types::RamblerShutUpObservation {
                speaker_position: 1,
                shut_up_target: 2,
            }];
        let CurrentDruidPayload::Ledger(interrupted_events) =
            parse_current_druid_payload(&interrupted, &interrupted_state).unwrap()
        else {
            panic!("interruption helper must produce an ordered ledger");
        };
        let interrupted_supports = current_druid_group_supports(
            &interrupted,
            &interrupted_events,
            true,
            &transient_world,
            &interrupted_state,
        );
        assert!(interrupted_supports.iter().any(|support| {
            support
                .raw_bluff
                .as_ref()
                .is_some_and(|(position, role)| *position == 1 && roles_equal(role, "Druid"))
        }));
        let historical_supports = current_druid_group_supports(
            &interrupted,
            &interrupted_events,
            false,
            &transient_world,
            &interrupted_state,
        );
        assert!(!historical_supports.iter().any(|support| {
            support
                .raw_bluff
                .as_ref()
                .is_some_and(|(position, role)| *position == 1 && roles_equal(role, "Druid"))
        }));

        // Once the zero-delay raw event appends, its Druid identity is exact
        // and must still join the same Medium witness.
        let upgraded = current_druid_ledger(
            1,
            vec![
                current_druid_result_event(
                    1,
                    0,
                    "real",
                    [1, 2, 3],
                    None,
                    4,
                    0,
                    "same_activation_extension",
                ),
                current_druid_result_event(
                    1,
                    1,
                    "raw",
                    [1, 2, 3],
                    Some("Drunk"),
                    4,
                    0,
                    "same_activation_extension",
                ),
            ],
        );
        transient_state.cards[0] = upgraded.clone();
        assert!(validate_current_hidden_surface_consistency(
            &transient_world,
            &transient_state,
        ));
        medium = current_medium(4, json!(1), json!("Judge"));
        transient_state.cards[3] = medium.clone();
        assert!(validate_medium(
            &medium,
            &transient_world,
            &transient_state,
        ));
        assert!(!validate_current_hidden_surface_consistency(
            &transient_world,
            &transient_state,
        ));

        // A raw-shaped single result cannot omit the real Druid callback.
        let raw_only = current_druid_ledger(
            1,
            vec![current_druid_result_event(
                1,
                0,
                "either",
                [1, 2, 3],
                Some("Drunk"),
                3,
                0,
                "single_callback_suffix",
            )],
        );
        assert!(!validate_druid(&raw_only, &moved, &state));

        let mut raw_raw = dual.clone();
        raw_raw.info_parsed["callback_events"][0]["dispatch_path"] = json!("raw");
        assert!(!validate_druid(&raw_raw, &moved, &state));
        let mut mismatched_group = dual.clone();
        mismatched_group.info_parsed["callback_events"][1]["settled_reveal_count"] = json!(2);
        assert!(!validate_druid(&mismatched_group, &moved, &state));
        let mut mismatched_evidence = dual.clone();
        mismatched_evidence.info_parsed["callback_events"][1]["activation_evidence"] =
            json!("session_reset_generation");
        assert!(!validate_druid(&mismatched_evidence, &moved, &state));
    }

    #[test]
    fn current_druid_dual_groups_preserve_uniform_rambler_mutation_and_opaque_order() {
        let opaque = json!({
            "activation_id": 1,
            "callback_index": 0,
            "dispatch_path": "real",
            "event_kind": "opaque_real",
            "text": "foreign callback",
            "references": null,
            "settled_reveal_count": 4,
            "reset_generation": 0,
            "activation_evidence": "auto_use_click",
        });
        let raw = current_druid_result_event(
            1,
            1,
            "raw",
            [2, 3, 4],
            Some("Drunk"),
            4,
            0,
            "auto_use_click",
        );
        let opaque_then_raw = current_druid_ledger(1, vec![opaque.clone(), raw.clone()]);
        let mut no_output_state = base_state(4, vec![opaque_then_raw.clone()]);
        no_output_state.deck.villagers = vec!["Druid".to_string()];
        no_output_state.deck.outcasts = vec!["Drunk".to_string()];
        no_output_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        no_output_state.reveal_order = vec![1, 2, 3, 4];
        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(1);
        assert!(!validate_druid(
            &opaque_then_raw,
            &drunk,
            &no_output_state,
        ));
        let raw_only = current_druid_ledger(
            1,
            vec![current_druid_result_event(
                1,
                0,
                "either",
                [2, 3, 4],
                Some("Drunk"),
                4,
                0,
                "single_callback_suffix",
            )],
        );
        no_output_state.cards[0] = raw_only.clone();
        assert!(validate_druid(&raw_only, &drunk, &no_output_state));

        let mut state = base_state(
            4,
            vec![
                opaque_then_raw.clone(),
                make_card(2, "Twin Minion", json!({})),
                make_card(3, "Pooka", json!({})),
                make_card(4, "Scout", json!({})),
            ],
        );
        state.deck.villagers = vec!["Confessor".to_string(), "Scout".to_string()];
        state.deck.outcasts = vec!["Drunk".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4];
        let mut callback_capable = empty_scenario();
        callback_capable
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        callback_capable
            .evil_positions
            .insert(3, "Pooka".to_string());
        callback_capable.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Confessor".to_string(),
            },
        });
        assert!(validate_druid(
            &opaque_then_raw,
            &callback_capable,
            &state,
        ));
        let mut medium_state = state.clone();
        medium_state.cards[0] = raw_only.clone();
        medium_state.deck.villagers = vec!["Medium".to_string(), "Scout".to_string()];
        let mut medium_capable = callback_capable.clone();
        let Some(crate::types::TwinTrace {
            outcome:
                crate::types::TwinStartOutcome::Swap {
                    neighbor_pre_swap_role,
                    ..
                },
            ..
        }) = medium_capable.twin_trace.as_mut()
        else {
            panic!("callback-capable fixture must have an exact Twin swap");
        };
        *neighbor_pre_swap_role = "Medium".to_string();
        assert!(!validate_druid(
            &raw_only,
            &medium_capable,
            &medium_state,
        ));
        let raw_interruption = current_druid_ledger(
            1,
            vec![current_druid_interruption_event(
                1,
                0,
                "either",
                2,
                4,
                0,
                "single_callback_suffix",
            )],
        );
        no_output_state.cards[0] = raw_interruption.clone();
        no_output_state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        no_output_state.rambler_shut_up_observations =
            vec![crate::types::RamblerShutUpObservation {
                speaker_position: 1,
                shut_up_target: 2,
            }];
        let CurrentDruidPayload::Ledger(raw_interruption_events) =
            parse_current_druid_payload(&raw_interruption, &no_output_state).unwrap()
        else {
            panic!("raw interruption fixture must produce an ordered ledger");
        };
        let drunk_interruption_supports = current_druid_group_supports(
            &raw_interruption,
            &raw_interruption_events,
            false,
            &drunk,
            &no_output_state,
        );
        assert!(drunk_interruption_supports.iter().any(|support| {
            support
                .callbacks
                .first()
                .is_some_and(|callback| callback.path == CurrentDruidResolvedPath::Raw)
        }));
        medium_state.cards[0] = raw_interruption.clone();
        medium_state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        medium_state.rambler_shut_up_observations =
            no_output_state.rambler_shut_up_observations.clone();
        let medium_interruption_supports = current_druid_group_supports(
            &raw_interruption,
            &raw_interruption_events,
            false,
            &medium_capable,
            &medium_state,
        );
        assert!(!medium_interruption_supports.iter().any(|support| {
            support
                .callbacks
                .first()
                .is_some_and(|callback| callback.path == CurrentDruidResolvedPath::Raw)
        }));

        // A foreign three-target clue is still opaque unless its clause is in
        // Druid's result family. The bridge authenticates this as a real
        // callback followed by raw Druid in one activation.
        let mut foreign_evils = opaque_then_raw.clone();
        foreign_evils.info_parsed["callback_events"][0]["text"] =
            json!("Among #1, #2, #3 there are 2 Evils");
        foreign_evils.info_parsed["callback_events"][0]["references"] = json!([1, 2, 3]);
        assert!(validate_druid(&foreign_evils, &callback_capable, &state));

        let mut later_line_shut = opaque_then_raw.clone();
        later_line_shut.info_parsed["callback_events"][0]["text"] =
            json!("#1\nunrelated shut callback");
        assert!(validate_druid(
            &later_line_shut,
            &callback_capable,
            &state,
        ));

        let mut opaque_last = current_druid_ledger(1, vec![opaque.clone(), raw.clone()]);
        opaque_last.info_parsed["callback_events"].as_array_mut().unwrap().swap(0, 1);
        assert!(!validate_druid(
            &opaque_last,
            &callback_capable,
            &state,
        ));
        let mut malformed_druid_opaque = opaque_then_raw.clone();
        malformed_druid_opaque.info_parsed["callback_events"][0]["text"] =
            json!("Among #2, #3, #4\nthere is: unknown");
        assert!(!validate_druid(
            &malformed_druid_opaque,
            &callback_capable,
            &state,
        ));
        let mut unicode_digit_druid_opaque = opaque_then_raw.clone();
        unicode_digit_druid_opaque.info_parsed["callback_events"][0]["text"] =
            json!("Among #١, #٢, #٣ there is: unknown");
        assert!(!validate_druid(
            &unicode_digit_druid_opaque,
            &callback_capable,
            &state,
        ));
        let mut long_s_druid_opaque = opaque_then_raw.clone();
        long_s_druid_opaque.info_parsed["callback_events"][0]["text"] =
            json!("Among #1, #2, #3 there iſ: unknown");
        assert!(!validate_druid(
            &long_s_druid_opaque,
            &callback_capable,
            &state,
        ));
        let mut python_c0_space_druid_opaque = opaque_then_raw.clone();
        python_c0_space_druid_opaque.info_parsed["callback_events"][0]["text"] =
            json!("\u{001C}Among\u{001D}#\u{001E}1, #2, #3 there\u{001F}is\u{001C}: unknown");
        assert!(!validate_druid(
            &python_c0_space_druid_opaque,
            &callback_capable,
            &state,
        ));
        let mut non_decimal_numeric_opaque = opaque_then_raw.clone();
        non_decimal_numeric_opaque.info_parsed["callback_events"][0]["text"] =
            json!("Among #², #², #² there is: unknown");
        assert!(validate_druid(
            &non_decimal_numeric_opaque,
            &callback_capable,
            &state,
        ));
        for near_miss in [
            "Among #1, #2, #3 there was: Wretch",
            "Among #1, #2, #3 there were zero Outcasts",
        ] {
            let mut malformed = opaque_then_raw.clone();
            malformed.info_parsed["callback_events"][0]["text"] = json!(near_miss);
            malformed.info_parsed["callback_events"][0]["references"] = json!([1, 2, 3]);
            assert!(!validate_druid(&malformed, &callback_capable, &state));
        }
        let mut malformed_shut_up_opaque = opaque_then_raw.clone();
        malformed_shut_up_opaque.info_parsed["callback_events"][0]["text"] =
            json!("#2 shut up!");
        assert!(!validate_druid(
            &malformed_shut_up_opaque,
            &callback_capable,
            &state,
        ));
        let mut unicode_digit_shut_up_opaque = opaque_then_raw.clone();
        unicode_digit_shut_up_opaque.info_parsed["callback_events"][0]["text"] =
            json!("#١ shut up!");
        assert!(!validate_druid(
            &unicode_digit_shut_up_opaque,
            &callback_capable,
            &state,
        ));
        let mut long_s_shut_up_opaque = opaque_then_raw.clone();
        long_s_shut_up_opaque.info_parsed["callback_events"][0]["text"] =
            json!("#1 ſhut up!");
        assert!(!validate_druid(
            &long_s_shut_up_opaque,
            &callback_capable,
            &state,
        ));
        let mut python_c0_space_shut_opaque = opaque_then_raw.clone();
        python_c0_space_shut_opaque.info_parsed["callback_events"][0]["text"] =
            json!("\u{001C}#\u{001D}1 anything shut");
        assert!(!validate_druid(
            &python_c0_space_shut_opaque,
            &callback_capable,
            &state,
        ));
        let mut non_decimal_numeric_shut_opaque = opaque_then_raw.clone();
        non_decimal_numeric_shut_opaque.info_parsed["callback_events"][0]["text"] =
            json!("#² shut up!");
        assert!(validate_druid(
            &non_decimal_numeric_shut_opaque,
            &callback_capable,
            &state,
        ));

        let first_interruption = current_druid_interruption_event(
            1,
            0,
            "real",
            2,
            4,
            0,
            "same_activation_extension",
        );
        let second_interruption = current_druid_interruption_event(
            1,
            1,
            "raw",
            2,
            4,
            0,
            "same_activation_extension",
        );
        let interrupted = current_druid_ledger(
            1,
            vec![first_interruption.clone(), second_interruption],
        );
        state.rambler_rule_version = Some(RAMBLER_CURRENT_RULE.to_string());
        state.rambler_shut_up_observations = vec![
            crate::types::RamblerShutUpObservation {
                speaker_position: 1,
                shut_up_target: 2,
            },
            crate::types::RamblerShutUpObservation {
                speaker_position: 1,
                shut_up_target: 2,
            },
        ];
        assert!(validate_druid(
            &interrupted,
            &callback_capable,
            &state,
        ));
        assert!(!validate_druid(
            &interrupted,
            &drunk,
            &no_output_state,
        ));

        let mixed = current_druid_ledger(1, vec![first_interruption, raw]);
        state.rambler_shut_up_observations.truncate(1);
        assert!(!validate_druid(&mixed, &callback_capable, &state));
    }

    #[test]
    fn current_druid_truth_uses_registered_outcast_identity_and_keeps_lifecycle_seats() {
        let positive = current_druid(1, [2, 3, 4], Some("Bombardier"));
        let mut state = base_state(
            6,
            vec![
                positive.clone(),
                make_card(2, "Bombardier", json!({})),
                make_card(3, "Plague Doctor", json!({})),
                make_card(4, "Scout", json!({})),
                make_card(5, "Wretch", json!({})),
                make_card(6, "Scout", json!({})),
            ],
        );
        state.deck.outcasts = vec![
            "Bombardier".to_string(),
            "Plague Doctor".to_string(),
            "Wretch".to_string(),
        ];
        state.deck.minions = vec!["Spy".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(6, "Spy".to_string());

        assert!(validate_druid(&positive, &scenario, &state));
        let plague_doctor = current_druid(1, [2, 3, 4], Some("Plague_Doctor"));
        assert_eq!(
            plague_doctor.info_text,
            "Among #2, #3, #4\nthere is: Plague Doctor"
        );
        assert!(validate_druid(&plague_doctor, &scenario, &state));
        assert!(!validate_druid(
            &current_druid(1, [2, 3, 4], Some("Drunk")),
            &scenario,
            &state,
        ));
        assert!(!validate_druid(
            &current_druid(1, [2, 3, 4], None),
            &scenario,
            &state,
        ));

        // Death/hidden-state bookkeeping does not remove a physical target
        // from CurrentCharacters or Druid's active picker result.
        state.executed = vec![2];
        state.night_kills = vec![3];
        assert!(validate_druid(&positive, &scenario, &state));

        // Wretch projects a Minion register-as and Spy a Villager register-as;
        // neither belongs to the selected Outcast occurrence pool.
        assert!(validate_druid(
            &current_druid(1, [4, 5, 6], None),
            &scenario,
            &state,
        ));
        assert!(!validate_druid(
            &current_druid(1, [4, 5, 6], Some("Wretch")),
            &scenario,
            &state,
        ));

        // Ordinary Doppelganger has no explicit register-as override, so the
        // fallback is its real Outcast data rather than its visible bluff.
        let mut doppelganger = empty_scenario();
        doppelganger.doppelganger_position = Some(6);
        assert!(validate_druid(
            &current_druid(1, [4, 5, 6], Some("Doppelganger")),
            &doppelganger,
            &state,
        ));
    }

    #[test]
    fn current_druid_bluff_is_the_exact_complement_and_false_role_ladder() {
        let mut state = base_state(
            5,
            vec![
                current_druid(1, [2, 3, 4], Some("Drunk")),
                make_card(2, "Scout", json!({})),
                make_card(3, "Bard", json!({})),
                make_card(4, "Lover", json!({})),
                make_card(5, "Bombardier", json!({})),
            ],
        );
        state.deck.outcasts = vec![
            "Doppelganger".to_string(),
            "Drunk".to_string(),
            "Wretch".to_string(),
            "Bombardier".to_string(),
        ];
        let mut lying = empty_scenario();
        lying.corrupted.insert(1);

        for role in ["Doppelganger", "Drunk", "Wretch"] {
            assert!(validate_druid(
                &current_druid(1, [2, 3, 4], Some(role)),
                &lying,
                &state,
            ));
        }
        assert!(!validate_druid(
            &current_druid(1, [2, 3, 4], Some("Bombardier")),
            &lying,
            &state,
        ));
        assert!(!validate_druid(
            &current_druid(1, [2, 3, 4], None),
            &lying,
            &state,
        ));

        assert!(validate_druid(
            &current_druid(1, [3, 4, 5], None),
            &lying,
            &state,
        ));
        assert!(!validate_druid(
            &current_druid(1, [3, 4, 5], Some("Drunk")),
            &lying,
            &state,
        ));

        // Drunk reaches the raw Druid BluffAct path; clean Doppelganger's
        // HealthyBluff reaches the truthful raw Druid Act path.
        let mut drunk = empty_scenario();
        drunk.drunk_position = Some(1);
        assert!(validate_druid(
            &current_druid(1, [2, 3, 4], Some("Drunk")),
            &drunk,
            &state,
        ));
        let mut doppel = empty_scenario();
        doppel.doppelganger_position = Some(1);
        assert!(validate_druid(
            &current_druid(1, [3, 4, 5], Some("Bombardier")),
            &doppel,
            &state,
        ));

        let mut puppet = empty_scenario();
        puppet.puppet_position = Some(1);
        assert!(validate_druid(
            &current_druid(1, [3, 4, 5], Some("Bombardier")),
            &puppet,
            &state,
        ));
        puppet.corrupted.insert(1);
        assert!(validate_druid(
            &current_druid(1, [3, 4, 5], None),
            &puppet,
            &state,
        ));
    }

    #[test]
    fn current_druid_history_projects_each_settled_baker_boundary() {
        // Neither result targets physical Spy #2. Its delayed clear still has
        // to be settled globally before the certified prefix can be used.
        let history = current_druid_history(4, &[([1, 3, 4], None, 1), ([1, 3, 4], None, 2)]);
        let mut state = base_state(
            4,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                make_card(3, "Pooka", json!({})),
                history.clone(),
            ],
        );
        state.deck.villagers = vec!["Baker".to_string(), "Druid".to_string()];
        state.deck.minions = vec!["Spy".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![4, 1, 2, 3];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(2, "Spy".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_baker_history(&scenario, &state));
        assert!(validate_druid(&history, &scenario, &state));

        let timelines = baker_spy_conversion_timelines(&scenario, &state);
        let timeline = timelines
            .iter()
            .cloned()
            .into_iter()
            .find(|timeline| {
                timeline.contains_position(2)
                    && timeline.supports_settled_reveal_count(2, &state)
            })
            .expect("the exact Baker history converts physical Spy #2");
        assert_eq!(
            current_data_role_at_druid_observation(
                2,
                CurrentDruidObservationBoundary::SettledRevealCount(1),
                &timeline,
                &scenario,
                &state,
            )
            .as_deref(),
            Some("Spy"),
        );
        assert_eq!(
            current_data_role_at_druid_observation(
                2,
                CurrentDruidObservationBoundary::SettledRevealCount(2),
                &timeline,
                &scenario,
                &state,
            )
            .as_deref(),
            Some("Baker"),
        );

        let late_clear = timelines
            .iter()
            .find(|timeline| !timeline.supports_settled_reveal_count(2, &state))
            .expect("native delay admits a clear after the second prefix");
        assert_eq!(
            current_data_role_at_druid_observation(
                1,
                CurrentDruidObservationBoundary::SettledRevealCount(2),
                late_clear,
                &scenario,
                &state,
            ),
            None,
            "an unrelated read must still reject the whole unsettled timeline",
        );
        let CurrentDruidPayload::Ledger(events) =
            parse_current_druid_payload(&history, &state).unwrap()
        else {
            panic!("history helper must produce the ordered ledger");
        };
        let supports = current_druid_group_supports(
            &history,
            &events[1..2],
            true,
            &scenario,
            &state,
        );
        assert!(!supports.is_empty());
        assert!(supports.iter().all(|support| support
            .baker_spy_timeline
            .supports_settled_reveal_count(2, &state)));
        assert!(!supports
            .iter()
            .any(|support| &support.baker_spy_timeline == late_clear));
    }

    #[test]
    fn current_druid_projects_twin_shaman_puppet_and_settled_baker_data() {
        let druid = current_druid(1, [2, 4, 5], Some("Bombardier"));
        let mut state = base_state(
            5,
            vec![
                druid.clone(),
                make_card(2, "Bombardier", json!({})),
                make_card(3, "Pooka", json!({})),
                make_card(4, "Scout", json!({})),
                make_card(5, "Twin Minion", json!({})),
            ],
        );
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];
        let mut moved = empty_scenario();
        moved.evil_positions.insert(3, "Pooka".to_string());
        moved.evil_positions.insert(5, "Twin Minion".to_string());
        moved.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 5,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Bombardier".to_string(),
            },
        });
        assert!(validate_druid(&druid, &moved, &state));
        assert!(!validate_druid(
            &current_druid(1, [2, 4, 5], None),
            &moved,
            &state,
        ));

        // A Twin actor that received Villager data remains runtime Evil and
        // can retain an Outcast raw bluff. Shaman sees its current Villager
        // type, copies that raw Plague Doctor identity, and writes it to #4.
        let copied_druid = current_druid(1, [2, 3, 4], Some("Plague_Doctor"));
        let mut copied_state = base_state(
            5,
            vec![
                copied_druid.clone(),
                make_card(2, "Scout", json!({})),
                make_card(3, "Pooka", json!({})),
                make_card(4, "Bard", json!({})),
                make_card(5, "Twin Minion", json!({})),
            ],
        );
        copied_state.deck.villagers = vec![
            "Druid".to_string(),
            "Scout".to_string(),
            "Bard".to_string(),
        ];
        copied_state.deck.outcasts = vec!["Plague Doctor".to_string()];
        copied_state.deck.minions = vec!["Twin Minion".to_string()];
        copied_state.deck.demons = vec!["Pooka".to_string()];
        let mut copied = empty_scenario();
        copied.evil_positions.insert(3, "Pooka".to_string());
        copied.evil_positions.insert(5, "Twin Minion".to_string());
        copied.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 5,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Scout".to_string(),
            },
        });
        copied.shaman_trace = Some(crate::types::ShamanTrace {
            source_position: 5,
            target_position: 4,
            copied_role: "Plague Doctor".to_string(),
            target_previous_roles: vec!["Bard".to_string()],
        });
        assert!(validate_druid(&copied_druid, &copied, &copied_state));

        let puppet_positive = current_druid(1, [2, 4, 5], Some("Bombardier"));
        let mut puppet_state = base_state(
            5,
            vec![
                puppet_positive.clone(),
                make_card(2, "Scout", json!({})),
                make_card(3, "Puppeteer", json!({})),
                make_card(4, "Bard", json!({})),
                // Public presentation can be the generated Puppet's bluff;
                // exact current data remains Puppet.
                make_card(5, "Bombardier", json!({})),
            ],
        );
        puppet_state.deck.minions = vec!["Puppeteer".to_string()];
        puppet_state.deck.outcasts = vec!["Bombardier".to_string()];
        let mut puppet = empty_scenario();
        puppet.evil_positions.insert(3, "Puppeteer".to_string());
        puppet.puppet_position = Some(5);
        assert!(!validate_druid(
            &puppet_positive,
            &puppet,
            &puppet_state,
        ));
        assert!(validate_druid(
            &current_druid(1, [2, 4, 5], None),
            &puppet,
            &puppet_state,
        ));

        let active = current_druid(4, [2, 3, 4], None);
        let mut baker_state = base_state(
            4,
            vec![
                make_card(1, "Baker", json!({"original_role": "original"})),
                make_card(2, "Baker", json!({"original_role": "Spy"})),
                make_card(3, "Pooka", json!({})),
                active.clone(),
            ],
        );
        baker_state.deck.villagers = vec!["Baker".to_string(), "Druid".to_string()];
        baker_state.deck.minions = vec!["Spy".to_string()];
        baker_state.deck.demons = vec!["Pooka".to_string()];
        baker_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        baker_state.reveal_order = vec![1, 4, 2, 3];
        let mut baker = empty_scenario();
        baker.evil_positions.insert(2, "Spy".to_string());
        baker.evil_positions.insert(3, "Pooka".to_string());
        assert!(validate_baker_history(&baker, &baker_state));
        assert!(validate_druid(&active, &baker, &baker_state));
        assert!(!validate_druid(
            &current_druid(2, [1, 2, 4], Some("Drunk")),
            &baker,
            &baker_state,
        ));
    }

    #[test]
    fn current_druid_hidden_outcast_occurrences_are_joined_across_observations() {
        let first = current_druid(1, [3, 4, 5], Some("Bombardier"));
        let second = current_druid(2, [4, 5, 6], Some("Bombardier"));
        let mut state = base_state(6, vec![first.clone(), second.clone()]);
        state.deck.villagers = vec!["Druid".to_string(), "Druid".to_string()];
        state.deck.outcasts = vec!["Bombardier".to_string()];
        state.board_villager_count = Some(5);
        state.board_outcast_count = Some(1);
        state.board_count_provenance =
            crate::types::BoardCountProvenance::TrustedPreStart;
        let scenario = empty_scenario();

        assert!(validate_druid(&first, &scenario, &state));
        assert!(validate_druid(&second, &scenario, &state));
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        // These disjoint triples require the sole Bombardier at two physical
        // seats. Each observation is independently possible, but not jointly.
        let disjoint = current_druid(2, [1, 2, 6], Some("Bombardier"));
        state.cards[1] = disjoint.clone();
        assert!(validate_druid(&disjoint, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
    }

    #[test]
    fn current_druid_joins_wretch_and_raw_callback_labels_globally() {
        let druid = current_druid_history(
            1,
            &[
                ([3, 4, 5], Some("Bombardier"), 3),
                ([3, 4, 5], Some("Bombardier"), 5),
            ],
        );
        let mut bounty = current_poet("Bounty Hunter", json!({"evil_position": 3}));
        bounty.position = 2;
        let mut state = base_state(
            5,
            vec![
                druid.clone(),
                bounty.clone(),
                make_card(4, "Scout", json!({})),
                make_card(5, "Bard", json!({})),
            ],
        );
        state.deck.villagers = vec![
            "Druid".to_string(),
            "Poet".to_string(),
            "Scout".to_string(),
            "Bard".to_string(),
        ];
        state.deck.outcasts = vec!["Bombardier".to_string(), "Wretch".to_string()];
        state.board_villager_count = Some(4);
        state.board_outcast_count = Some(1);
        state.board_count_provenance = crate::types::BoardCountProvenance::TrustedPreStart;
        state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        state.reveal_order = vec![1, 2, 3, 4, 5];
        let mut scenario = empty_scenario();
        assert!(validate_druid(&druid, &scenario, &state));
        assert!(validate_poet(&bounty, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));
        scenario.corrupted.insert(2);
        assert!(validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        let raw_druid = current_druid_history(
            1,
            &[([2, 3, 4], Some("Drunk"), 1), ([2, 3, 4], Some("Drunk"), 5)],
        );
        let mut medium = current_medium(5, json!(1), json!("Judge"));
        let mut raw_state = base_state(
            5,
            vec![
                raw_druid.clone(),
                make_card(2, "Scout", json!({})),
                make_card(3, "Bard", json!({})),
                make_card(4, "Lover", json!({})),
                medium.clone(),
            ],
        );
        raw_state.deck.villagers = vec![
            "Druid".to_string(),
            "Medium".to_string(),
            "Judge".to_string(),
        ];
        raw_state.deck.outcasts = vec!["Drunk".to_string()];
        raw_state.deck.demons = vec!["Pooka".to_string()];
        raw_state.baker_rule_version = Some(BAKER_CURRENT_RULE.to_string());
        raw_state.reveal_order = vec![1, 2, 3, 4, 5];
        let mut raw = empty_scenario();
        raw.evil_positions.insert(1, "Pooka".to_string());
        raw.corrupted.insert(5);
        assert!(validate_druid(&raw_druid, &raw, &raw_state));
        assert!(validate_medium(&medium, &raw, &raw_state));
        assert!(!validate_current_hidden_surface_consistency(
            &raw, &raw_state,
        ));
        medium = current_medium(5, json!(1), json!("Druid"));
        raw_state.cards[4] = medium.clone();
        assert!(validate_medium(&medium, &raw, &raw_state));
        assert!(validate_current_hidden_surface_consistency(
            &raw, &raw_state
        ));
    }

    #[test]
    fn current_druid_fails_closed_on_unknown_start_identity_but_legacy_is_unchanged() {
        let current = current_druid(1, [2, 3, 4], None);
        let mut state = base_state(4, vec![current.clone()]);
        state.deck.villagers = vec!["Druid".to_string()];
        let mut scenario = empty_scenario();
        scenario.evil_positions.insert(4, "Unknown".to_string());
        assert!(!validate_druid(&current, &scenario, &state));
        assert!(!validate_current_hidden_surface_consistency(
            &scenario, &state,
        ));

        let legacy = make_card(
            1,
            "Druid",
            json!({"targets": [2, 3, 4], "found_outcast": null}),
        );
        state.cards[0] = legacy.clone();
        assert!(validate_druid(&legacy, &scenario, &state));
        assert!(validate_druid(
            &make_card(1, "Druid", json!({})),
            &scenario,
            &state,
        ));
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

        // Puppeteer acts after Twin, but that generated full Init is now an
        // exact overlay rather than part of the coarse swap-endpoint waiver.
        state.deck.minions.push("Puppeteer".to_string());
        assert!(!matches_executed_good_role(
            &stable_twin,
            &state,
            2,
            "Puppet",
        ));
        stable_twin.puppet_position = Some(2);
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
    fn legacy_role_labels_project_puppet_over_stable_twin() {
        let mut state = base_state(3, vec![]);
        state.deck.villagers = vec![
            "Scout".to_string(),
            "Oracle".to_string(),
            "Dreamer".to_string(),
        ];
        state.deck.minions = vec!["Puppeteer".to_string(), "Twin Minion".to_string()];
        let mut scenario = empty_scenario();
        scenario
            .evil_positions
            .insert(1, "Puppeteer".to_string());
        scenario
            .evil_positions
            .insert(2, "Twin Minion".to_string());
        scenario.puppet_position = Some(2);

        assert!(validate_scout(
            &make_card(3, "Scout", json!({"evil_role": "Puppet", "distance": 1})),
            &scenario,
            &state,
        ));
        assert!(validate_oracle(
            &make_card(3, "Oracle", json!({"targets": [2], "minion_role": "Puppet"})),
            &scenario,
            &state,
        ));
        assert!(validate_dreamer(
            &make_card(3, "Dreamer", json!({"target": 2, "evil_role": "Puppet"})),
            &scenario,
            &state,
        ));
    }

    #[test]
    fn traced_twin_current_roles_are_exact_and_receive_no_opaque_waiver() {
        let mut state = base_state(3, vec![]);
        state.deck.villagers = vec!["Baker".to_string(), "Knight".to_string()];
        state.deck.minions = vec!["Twin Minion".to_string()];
        state.deck.demons = vec!["Pooka".to_string()];

        let mut scenario = empty_scenario();
        scenario
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        scenario.evil_positions.insert(3, "Pooka".to_string());
        scenario.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Baker".to_string(),
            },
        });

        // The original runtime-Evil Twin body now holds Baker data, while the
        // runtime-Good neighbor holds Twin data.
        assert!(matches_executed_good_role(
            &scenario, &state, 1, "Baker",
        ));
        assert!(matches_executed_good_role(
            &scenario,
            &state,
            2,
            "Twin Minion",
        ));
        assert!(!matches_executed_good_role(
            &scenario, &state, 1, "Knight",
        ));
        assert!(!matches_executed_good_role(
            &scenario, &state, 2, "Baker",
        ));

        assert!(!twin_can_explain_current_role_mismatch(
            1, "Knight", &scenario, &state,
        ));
        assert!(!twin_can_explain_current_role_mismatch(
            2, "Baker", &scenario, &state,
        ));
        assert!(!twin_may_have_replaced_current_data_at(
            1, &scenario, &state,
        ));
        assert!(!twin_may_have_replaced_current_data_at(
            2, &scenario, &state,
        ));

        // Removing the trace restores the pre-trace geometry waiver without
        // changing the stable role assignment or authored deck.
        scenario.twin_trace = None;
        assert!(twin_can_explain_current_role_mismatch(
            1, "Baker", &scenario, &state,
        ));
        assert!(twin_can_explain_current_role_mismatch(
            2,
            "Twin Minion",
            &scenario,
            &state,
        ));
        assert!(twin_may_have_replaced_current_data_at(
            2, &scenario, &state,
        ));
    }

    #[test]
    fn exact_no_demon_and_self_swap_disable_coarse_twin_waivers() {
        let mut no_demon_state = base_state(3, vec![]);
        no_demon_state.deck.villagers = vec!["Baker".to_string()];
        no_demon_state.deck.minions = vec!["Twin Minion".to_string()];

        let mut no_demon = empty_scenario();
        no_demon
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        no_demon.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::NoDemon,
        });

        assert!(matches_executed_good_role(
            &no_demon,
            &no_demon_state,
            1,
            "Twin Minion",
        ));
        assert!(!matches_executed_good_role(
            &no_demon,
            &no_demon_state,
            1,
            "Baker",
        ));
        assert!(!twin_can_explain_current_role_mismatch(
            1,
            "Baker",
            &no_demon,
            &no_demon_state,
        ));

        let mut self_swap_state = no_demon_state.clone();
        self_swap_state.deck.demons = vec!["Pooka".to_string()];
        let mut self_swap = empty_scenario();
        self_swap
            .evil_positions
            .insert(1, "Twin Minion".to_string());
        self_swap
            .evil_positions
            .insert(2, "Pooka".to_string());
        self_swap.twin_trace = Some(crate::types::TwinTrace {
            actor_position: 1,
            outcome: crate::types::TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 2,
                neighbor_side: crate::types::TwinNeighborSide::Next,
                neighbor_position: 1,
                neighbor_pre_swap_role: "Twin Minion".to_string(),
            },
        });

        assert!(matches_executed_good_role(
            &self_swap,
            &self_swap_state,
            1,
            "Twin Minion",
        ));
        assert!(!matches_executed_good_role(
            &self_swap,
            &self_swap_state,
            1,
            "Baker",
        ));
        assert!(!twin_can_explain_current_role_mismatch(
            1,
            "Baker",
            &self_swap,
            &self_swap_state,
        ));
        assert!(!twin_may_have_replaced_current_data_at(
            3,
            &self_swap,
            &self_swap_state,
        ));

        // The same stable world without its exact self-swap still admits the
        // coarse alternative where the other adjacent endpoint supplied data.
        self_swap.twin_trace = None;
        assert!(twin_can_explain_current_role_mismatch(
            1,
            "Baker",
            &self_swap,
            &self_swap_state,
        ));
        assert!(twin_may_have_replaced_current_data_at(
            3,
            &self_swap,
            &self_swap_state,
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
