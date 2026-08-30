/// Scenario generation: enumerate all possible evil placements and build
/// full scenarios with corruption variants.

use std::collections::{HashMap, HashSet};
use crate::corruption::{enumerate_start_corruption, StartCorruptionContext};
use crate::geometry::adjacent_positions;
use crate::knowledge_base::{self, get_card, is_villager_role, normalize_role, Faction};
use crate::types::{GameState, Scenario};

/// Generate all candidate scenarios for the current game state.
pub fn build_scenarios(state: &GameState) -> Vec<Scenario> {
    let placements = generate_evil_placements(state);
    let mut scenarios = Vec::new();
    let n = state.n_cards;

    let pd_in_deck = state.deck.outcasts.iter().any(|o| knowledge_base::is_plague_doctor(o));
    let (pd_can_be_on_board, pd_can_be_absent) = hidden_outcast_presence_flags("Plague_Doctor", state);

    for placement in &placements {
        if !apply_placement_constraints(placement, state) {
            continue;
        }

        // Find Puppet position
        let mut puppet_pos: Option<u8> = placement.iter()
            .find(|(_, r)| r.as_str() == "Puppet")
            .map(|(&p, _)| p);
        if puppet_pos.is_none() {
            puppet_pos = state.executed_evil_roles.iter()
                .find(|(_, r)| r.as_str() == "Puppet")
                .map(|(&p, _)| p);
        }

        // Build full evil set including executed evils
        let mut full_evil: HashMap<u8, String> = placement.clone();
        for (&pos, role) in &state.executed_evil_roles {
            full_evil.entry(pos).or_insert_with(|| role.clone());
        }
        for &pos in &state.confirmed_evil {
            if state.executed.contains(&pos) && !full_evil.contains_key(&pos) {
                full_evil.insert(pos, "Unknown".to_string());
            }
        }

        // Doppelganger candidates
        let has_doppelganger = state.deck.outcasts.iter().any(|o| o == "Doppelganger");
        let (can_have_dopp, can_skip_dopp) = hidden_outcast_presence_flags("Doppelganger", state);
        let mut dopp_candidates: Vec<Option<u8>> = if can_skip_dopp { vec![None] } else { vec![] };
        if has_doppelganger && can_have_dopp {
            for p in 1..=n {
                if full_evil.contains_key(&p) || puppet_pos == Some(p) { continue; }
                if let Some(card) = state.card_at(p) {
                    if is_villager_role(&card.apparent_role) {
                        dopp_candidates.push(Some(p));
                    }
                } else {
                    dopp_candidates.push(Some(p)); // Unrevealed
                }
            }
        }

        // Drunk candidates
        let has_drunk = state.deck.outcasts.iter().any(|o| o == "Drunk");
        let drunk_already_known = state.cards.iter().any(|c| c.apparent_role == "Drunk");
        let (can_have_drunk, can_skip_drunk) = hidden_outcast_presence_flags("Drunk", state);
        let mut drunk_candidates: Vec<Option<u8>> = if drunk_already_known || can_skip_drunk { vec![None] } else { vec![] };
        if has_drunk && !drunk_already_known && can_have_drunk {
            for p in 1..=n {
                if full_evil.contains_key(&p) || puppet_pos == Some(p) { continue; }
                if let Some(card) = state.card_at(p) {
                    if is_villager_role(&card.apparent_role) {
                        drunk_candidates.push(Some(p));
                    }
                } else {
                    drunk_candidates.push(Some(p));
                }
            }
        }

        // `full_evil` describes the board after Chancellor has moved its role
        // identity beside a real Outcast. The separate conversion hypothesis is
        // built below, after hidden-Outcast identities are known.
        let chancellor_pos_opt = full_evil.iter()
            .filter(|(_, role)| role.as_str() == "Chancellor")
            .map(|(&position, _)| position)
            .max();

        // Identity hypotheses feed one ordered Start simulator. Random
        // Poisoner and Plague Doctor choices branch inside that simulator so
        // each actor sees the live statuses left by earlier actors.
        let mut seen: HashSet<(
            Vec<u8>, Option<u8>, Vec<(u8, u8)>, Option<u8>, Option<u8>, Option<u8>,
        )> = HashSet::new();

        for &dopp_pos_opt in &dopp_candidates {
            for &drunk_pos_opt in &drunk_candidates {
                if drunk_pos_opt.is_some() && drunk_pos_opt == dopp_pos_opt {
                    continue;
                }

                let chancellor_conv_cands = chancellor_conversion_candidates(
                    state,
                    &full_evil,
                    puppet_pos,
                    dopp_pos_opt,
                    drunk_pos_opt,
                    chancellor_pos_opt,
                );
                for &chan_conv in &chancellor_conv_cands {

                    let pd_act_variants = plague_doctor_act_variants(
                        state, &full_evil, dopp_pos_opt, drunk_pos_opt, puppet_pos,
                        pd_in_deck, pd_can_be_on_board, pd_can_be_absent,
                    );
                    let nk_alch_variants = night_killed_alchemist_variants(
                        state, &full_evil, dopp_pos_opt, drunk_pos_opt, puppet_pos, chan_conv,
                    );

                    for nk_alchemists in &nk_alch_variants {
                        for &plague_doctor_acts in &pd_act_variants {
                            let context = build_start_corruption_context(
                                state, &full_evil, dopp_pos_opt, drunk_pos_opt, puppet_pos,
                                chan_conv, nk_alchemists, plague_doctor_acts,
                            );
                            let outcomes = enumerate_start_corruption(
                                n, &full_evil, &context, state.pd_corruption_target,
                            );

                            for outcome in outcomes {
                                let mut corr_key: Vec<u8> = outcome.corrupted.iter().copied().collect();
                                corr_key.sort_unstable();
                                let mut alch_key: Vec<(u8, u8)> = outcome.alchemist_counts.iter()
                                    .map(|(&position, &count)| (position, count)).collect();
                                alch_key.sort_unstable();
                                let key = (
                                    corr_key, outcome.pd_target, alch_key,
                                    dopp_pos_opt, drunk_pos_opt, chan_conv,
                                );
                                if !seen.insert(key) {
                                    continue;
                                }

                                scenarios.push(Scenario {
                                    evil_positions: full_evil.clone(),
                                    puppet_position: puppet_pos,
                                    corrupted: outcome.corrupted,
                                    pd_corrupted: outcome.pd_target,
                                    doppelganger_position: dopp_pos_opt,
                                    drunk_position: drunk_pos_opt,
                                    alchemist_cures: outcome.alchemist_counts,
                                    chancellor_conversion: chan_conv,
                                });
                            }
                        }
                    }
                }
            }
        }
    }

    scenarios
}

/// Enumerate the final position of the Outcast identity Chancellor added at
/// Start. Native code first replaces a real Villager anywhere on the board,
/// then moves Chancellor beside a real Outcast before delayed Reveal populates
/// register-as values. Consequently the added
/// identity is observed after Reveal either as a visible Outcast, as a hidden
/// Drunk/Doppelganger, or on an unrevealed good card; it is not an arbitrary
/// adjacent apparent Villager.
///
/// The rare path where Chancellor swaps through the just-converted physical
/// card moves the added Outcast data to Chancellor's original card. At the
/// scenario layer we track that final Outcast home, which is the identity fact
/// consumed by type/count validators and keeps the state space bounded.
#[allow(clippy::too_many_arguments)]
fn chancellor_conversion_candidates(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    puppet_position: Option<u8>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    chancellor_position: Option<u8>,
) -> Vec<Option<u8>> {
    let Some(chancellor_position) = chancellor_position else {
        return vec![None];
    };

    let is_final_outcast = |position: u8| {
        if full_evil.contains_key(&position) || puppet_position == Some(position) {
            return false;
        }
        if doppelganger_position == Some(position) || drunk_position == Some(position) {
            return true;
        }
        match state.card_at(position) {
            Some(card) => get_card(&card.apparent_role)
                .map(|role| role.faction == Faction::Outcast)
                .unwrap_or(false),
            // A killed or Witch-blocked card can hold the generated Outcast.
            None => true,
        }
    };

    let has_outcast_neighbor = adjacent_positions(chancellor_position, state.n_cards)
        .into_iter()
        .any(&is_final_outcast);
    if !has_outcast_neighbor {
        return Vec::new();
    }

    let mut candidates: Vec<Option<u8>> = (1..=state.n_cards)
        .filter(|&position| is_final_outcast(position))
        .map(Some)
        .collect();
    candidates.sort_unstable();
    candidates.dedup();

    // "Add one Outcast if able." A board with no surviving candidate can only
    // represent the unable branch; normally the generated identity itself
    // guarantees at least one candidate.
    if candidates.is_empty() {
        vec![None]
    } else {
        candidates
    }
}

#[allow(clippy::too_many_arguments)]
fn build_start_corruption_context(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    chancellor_conversion: Option<u8>,
    extra_true_alchemists: &[u8],
    plague_doctor_acts: bool,
) -> StartCorruptionContext {
    let extra_alchemists: HashSet<u8> = extra_true_alchemists.iter().copied().collect();
    let mut real_villagers_before_puppet = HashSet::new();

    for position in 1..=state.n_cards {
        let is_real_villager = if Some(position) == chancellor_conversion {
            false // Chancellor has already replaced this data with an Outcast.
        } else if Some(position) == puppet_position {
            true // Puppeteer selects a real Villager and replaces it later.
        } else if full_evil.contains_key(&position)
            || Some(position) == doppelganger_position
            || Some(position) == drunk_position
        {
            false
        } else if extra_alchemists.contains(&position) {
            true
        } else if let Some(card) = state.card_at(position) {
            is_villager_role(&card.apparent_role)
        } else {
            unrevealed_must_be_villager(
                position, full_evil, state, doppelganger_position, drunk_position,
            )
        };
        if is_real_villager {
            real_villagers_before_puppet.insert(position);
        }
    }

    let mut corruption_resistant_at_init = extra_alchemists.clone();
    let mut true_alchemist_positions: Vec<u8> = extra_true_alchemists.to_vec();
    for card in &state.cards {
        if normalize_role(&card.apparent_role) != "alchemist"
            || Some(card.position) == doppelganger_position
            || Some(card.position) == drunk_position
        {
            continue;
        }

        // A future Puppet or Chancellor target already ran its original
        // Alchemist Init hook, so resistance survives the role replacement.
        if !full_evil.contains_key(&card.position) || Some(card.position) == puppet_position {
            corruption_resistant_at_init.insert(card.position);
        }
        if !full_evil.contains_key(&card.position)
            && Some(card.position) != puppet_position
            && Some(card.position) != chancellor_conversion
        {
            true_alchemist_positions.push(card.position);
        }
    }
    true_alchemist_positions.sort_unstable();
    true_alchemist_positions.dedup();

    // Init clears registerAs, and delayed internal Reveal has not resumed yet.
    // PD's registered-type predicate therefore matches post-conversion dataRef.
    let mut registered_villagers_at_pd_call = real_villagers_before_puppet.clone();
    if let Some(puppet_position) = puppet_position {
        registered_villagers_at_pd_call.remove(&puppet_position);
    }

    StartCorruptionContext {
        real_villagers_before_puppet,
        registered_villagers_at_pd_call,
        corruption_resistant_at_init,
        true_alchemist_positions,
        drunk_position,
        puppet_position,
        plague_doctor_acts,
    }
}

#[allow(clippy::too_many_arguments)]
fn plague_doctor_act_variants(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    pd_in_deck: bool,
    pd_can_be_on_board: bool,
    pd_can_be_absent: bool,
) -> Vec<bool> {
    let identity_replaced = |position: u8| {
        full_evil.contains_key(&position)
            || Some(position) == doppelganger_position
            || Some(position) == drunk_position
            || Some(position) == puppet_position
    };
    // Chancellor runs before Plague Doctor. If its generated Outcast identity
    // is the visible PD, that newly initialized role is a real actor here; the
    // final Outcast home must therefore not be classified as a fake identity.
    let known_true_pd = state.cards.iter().any(|card| {
        knowledge_base::is_plague_doctor(&card.apparent_role)
            && !identity_replaced(card.position)
    });
    let revealed: HashSet<u8> = state.cards.iter().map(|card| card.position).collect();
    let hidden_slot_exists = (1..=state.n_cards)
        .any(|position| !revealed.contains(&position) && !identity_replaced(position));
    let hidden_pd_possible = !known_true_pd && pd_in_deck
        && pd_can_be_on_board && hidden_slot_exists;

    let mut variants = Vec::new();
    if known_true_pd || hidden_pd_possible { variants.push(true); }
    if !known_true_pd && pd_can_be_absent { variants.push(false); }
    if state.pd_corruption_target.is_some() { variants.retain(|acts| *acts); }
    variants.sort_unstable();
    variants.dedup();
    variants
}

#[allow(clippy::too_many_arguments)]
fn night_killed_alchemist_variants(
    state: &GameState,
    full_evil: &HashMap<u8, String>,
    doppelganger_position: Option<u8>,
    drunk_position: Option<u8>,
    puppet_position: Option<u8>,
    chancellor_conversion: Option<u8>,
) -> Vec<Vec<u8>> {
    let deck_alchemists = state.deck.villagers.iter()
        .filter(|role| normalize_role(role) == "alchemist").count();
    if state.night_kills.is_empty() || deck_alchemists == 0 {
        return vec![Vec::new()];
    }

    let known_alchemists = state.cards.iter().filter(|card| {
        normalize_role(&card.apparent_role) == "alchemist"
            && Some(card.position) != doppelganger_position
            && Some(card.position) != drunk_position
            && (!full_evil.contains_key(&card.position)
                || Some(card.position) == puppet_position)
    }).count();
    let missing_alchemists = deck_alchemists.saturating_sub(known_alchemists);
    if missing_alchemists == 0 { return vec![Vec::new()]; }

    let revealed: HashSet<u8> = state.cards.iter().map(|card| card.position).collect();
    let hidden_night_kills: Vec<u8> = state.night_kills.iter().copied()
        .filter(|position| {
            !revealed.contains(position)
                && !full_evil.contains_key(position)
                && Some(*position) != doppelganger_position
                && Some(*position) != drunk_position
                && Some(*position) != puppet_position
                && Some(*position) != chancellor_conversion
        }).collect();

    let mut variants = vec![Vec::new()];
    for count in 1..=missing_alchemists.min(hidden_night_kills.len()) {
        variants.extend(combinations_of(&hidden_night_kills, count));
    }
    variants
}

// ── Placement generation ──

fn generate_evil_placements(state: &GameState) -> Vec<HashMap<u8, String>> {
    let n = state.n_cards;
    let mut evil_roles: Vec<String> = state.deck.evil_roles();

    let puppet_in_deck = evil_roles.iter().any(|r| r == "Puppeteer");
    if puppet_in_deck {
        if let Some(idx) = evil_roles.iter().position(|r| r == "Puppet") {
            evil_roles.remove(idx);
        }
    }

    // Remove executed evil roles
    let mut remaining = evil_roles.clone();
    for (_pos, role) in &state.executed_evil_roles {
        let norm = normalize_role(role);
        if let Some(idx) = remaining.iter().position(|r| normalize_role(r) == norm) {
            remaining.remove(idx);
        }
        // Puppet isn't in evil_roles — silently skip
    }

    // Unknown executed evils
    let executed_evil_without_role: Vec<u8> = state.confirmed_evil.iter()
        .filter(|&&p| state.executed.contains(&p) && !state.executed_evil_roles.contains_key(&p))
        .copied().collect();

    let n_to_remove = executed_evil_without_role.len().min(remaining.len());
    let possible_remaining_lists = if n_to_remove > 0 {
        let mut seen_keys: HashSet<Vec<String>> = HashSet::new();
        let mut lists = Vec::new();
        for removal in combinations_indices(remaining.len(), n_to_remove) {
            let kept: Vec<String> = remaining.iter().enumerate()
                .filter(|(i, _)| !removal.contains(i))
                .map(|(_, r)| r.clone()).collect();
            let mut key = kept.clone();
            key.sort();
            if seen_keys.insert(key) {
                lists.push(kept);
            }
        }
        lists
    } else {
        vec![remaining]
    };

    let n_executed_evil = state.executed_evil_roles.len() + executed_evil_without_role.len();
    let expected_remaining = state.n_evil as i32 - n_executed_evil as i32;

    let valid_sizes: HashSet<usize> = if puppet_in_deck {
        [expected_remaining.max(0) as usize, (expected_remaining + 1).max(0) as usize].into()
    } else {
        [expected_remaining.max(0) as usize].into()
    };

    let night_kills_set: HashSet<u8> = state.night_kills.iter().copied().collect();
    let player_executed: HashSet<u8> = state.executed.iter()
        .filter(|p| !night_kills_set.contains(p))
        .copied().collect();
    let confirmed_good_set: HashSet<u8> = state.confirmed_good.iter().copied().collect();
    let available: Vec<u8> = (1..=n)
        .filter(|p| !player_executed.contains(p) && !confirmed_good_set.contains(p))
        .collect();

    // Check Puppeteer/Puppet execution status
    let puppeteer_executed_pos: Option<u8> = state.executed_evil_roles.iter()
        .find(|(_, r)| r.as_str() == "Puppeteer").map(|(&p, _)| p);
    let puppet_executed_pos: Option<u8> = state.executed_evil_roles.iter()
        .find(|(_, r)| r.as_str() == "Puppet").map(|(&p, _)| p);

    let mut all_placements: Vec<HashMap<u8, String>> = Vec::new();
    let mut seen_placements: HashSet<Vec<(u8, String)>> = HashSet::new();

    let mut add_placement = |p: &HashMap<u8, String>| {
        if valid_sizes.contains(&p.len()) {
            let mut key: Vec<(u8, String)> = p.iter().map(|(&k, v)| (k, v.clone())).collect();
            key.sort_by_key(|(k, _)| *k);
            if seen_placements.insert(key) {
                all_placements.push(p.clone());
            }
        }
    };

    for evil_roles in &possible_remaining_lists {
        let has_puppeteer = evil_roles.iter().any(|r| r == "Puppeteer");

        // Case: Puppeteer executed, Puppet still alive
        let puppet_still_alive = puppeteer_executed_pos.is_some()
            && !has_puppeteer
            && !state.executed_evil_roles.values().any(|r| r == "Puppet");

        if puppet_still_alive {
            let pep = puppeteer_executed_pos.unwrap();
            let puppet_cands: Vec<u8> = adjacent_positions(pep, n).into_iter()
                .filter(|a| available.contains(a)).collect();
            // Case 1: Puppet at adjacent
            for &pp in &puppet_cands {
                let remaining_avail: Vec<u8> = available.iter().filter(|&&p| p != pp).copied().collect();
                if evil_roles.is_empty() {
                    let mut p = HashMap::new();
                    p.insert(pp, "Puppet".to_string());
                    add_placement(&p);
                } else {
                    for combo in combinations_of(&remaining_avail, evil_roles.len()) {
                        for perm in permutations_of(evil_roles) {
                            let mut p = HashMap::new();
                            p.insert(pp, "Puppet".to_string());
                            for (i, &pos) in combo.iter().enumerate() {
                                p.insert(pos, perm[i].clone());
                            }
                            add_placement(&p);
                        }
                    }
                }
            }
            // Case 2: No Puppet
            for combo in combinations_of(&available, evil_roles.len()) {
                for perm in permutations_of(evil_roles) {
                    let mut p = HashMap::new();
                    for (i, &pos) in combo.iter().enumerate() {
                        p.insert(pos, perm[i].clone());
                    }
                    add_placement(&p);
                }
            }
            continue;
        }

        // Case: Puppet executed, Puppeteer must be adjacent
        if puppet_executed_pos.is_some() && has_puppeteer {
            let pxp = puppet_executed_pos.unwrap();
            let base_evil: Vec<String> = evil_roles.iter().filter(|r| r.as_str() != "Puppeteer").cloned().collect();
            let puppeteer_cands: Vec<u8> = adjacent_positions(pxp, n).into_iter()
                .filter(|a| available.contains(a)).collect();
            for &pp_pos in &puppeteer_cands {
                let rem: Vec<u8> = available.iter().filter(|&&p| p != pp_pos).copied().collect();
                if base_evil.is_empty() {
                    let mut p = HashMap::new();
                    p.insert(pp_pos, "Puppeteer".to_string());
                    add_placement(&p);
                } else {
                    for combo in combinations_of(&rem, base_evil.len()) {
                        for perm in permutations_of(&base_evil) {
                            let mut p = HashMap::new();
                            p.insert(pp_pos, "Puppeteer".to_string());
                            for (i, &pos) in combo.iter().enumerate() {
                                p.insert(pos, perm[i].clone());
                            }
                            add_placement(&p);
                        }
                    }
                }
            }
            continue;
        }

        // Case: Puppeteer present (needs Puppet slot)
        if has_puppeteer {
            let base_evil: Vec<String> = evil_roles.iter().filter(|r| r.as_str() != "Puppeteer").cloned().collect();
            for &puppeteer_pos in &available {
                let adj = adjacent_positions(puppeteer_pos, n);
                let puppet_cands: Vec<u8> = adj.iter()
                    .filter(|&&a| available.contains(&a) && a != puppeteer_pos)
                    .copied().collect();

                // Filter to Villager-or-unknown
                let mut villager_or_unknown = Vec::new();
                let mut has_known_villager = false;
                for &pc in &puppet_cands {
                    if let Some(card) = state.card_at(pc) {
                        if is_villager_role(&card.apparent_role) {
                            villager_or_unknown.push(pc);
                            has_known_villager = true;
                        }
                    } else {
                        villager_or_unknown.push(pc);
                    }
                }
                let actual_puppet_cands = if has_known_villager { &villager_or_unknown } else { &puppet_cands };

                // Case 1: Puppet created
                for &puppet_p in actual_puppet_cands {
                    let rem: Vec<u8> = available.iter()
                        .filter(|&&p| p != puppeteer_pos && p != puppet_p)
                        .copied().collect();
                    if base_evil.is_empty() {
                        let mut p = HashMap::new();
                        p.insert(puppeteer_pos, "Puppeteer".to_string());
                        p.insert(puppet_p, "Puppet".to_string());
                        add_placement(&p);
                    } else {
                        for combo in combinations_of(&rem, base_evil.len()) {
                            for perm in permutations_of(&base_evil) {
                                let mut p = HashMap::new();
                                p.insert(puppeteer_pos, "Puppeteer".to_string());
                                p.insert(puppet_p, "Puppet".to_string());
                                for (i, &pos) in combo.iter().enumerate() {
                                    p.insert(pos, perm[i].clone());
                                }
                                add_placement(&p);
                            }
                        }
                    }
                }
                // Case 2: Puppet NOT created
                let rem: Vec<u8> = available.iter()
                    .filter(|&&p| p != puppeteer_pos).copied().collect();
                if base_evil.is_empty() {
                    let mut p = HashMap::new();
                    p.insert(puppeteer_pos, "Puppeteer".to_string());
                    add_placement(&p);
                } else {
                    for combo in combinations_of(&rem, base_evil.len()) {
                        for perm in permutations_of(&base_evil) {
                            let mut p = HashMap::new();
                            p.insert(puppeteer_pos, "Puppeteer".to_string());
                            for (i, &pos) in combo.iter().enumerate() {
                                p.insert(pos, perm[i].clone());
                            }
                            add_placement(&p);
                        }
                    }
                }
            }
            continue;
        }

        // No Puppeteer — straightforward combinations
        let n_evil_count = evil_roles.len() as i32;
        let evil_role_subsets: Vec<Vec<String>> = if n_evil_count > expected_remaining && expected_remaining > 0 {
            evil_role_subsets_fn(evil_roles, state, expected_remaining as usize)
        } else if n_evil_count == expected_remaining {
            vec![evil_roles.clone()]
        } else if !puppet_in_deck && n_evil_count != expected_remaining {
            continue; // Skip invalid branch
        } else {
            if evil_roles.is_empty() { vec![] } else { vec![evil_roles.clone()] }
        };

        if expected_remaining > 0 {
            for role_set in &evil_role_subsets {
                for combo in combinations_of(&available, role_set.len()) {
                    for perm in permutations_of(role_set) {
                        let mut p = HashMap::new();
                        for (i, &pos) in combo.iter().enumerate() {
                            p.insert(pos, perm[i].clone());
                        }
                        add_placement(&p);
                    }
                }
            }
        } else {
            add_placement(&HashMap::new());
        }
    }

    all_placements
}

// ── Helper functions ──

fn apply_placement_constraints(placement: &HashMap<u8, String>, state: &GameState) -> bool {
    let n = state.n_cards;
    for (&pos, role) in placement {
        if role == "Chancellor" {
            let adj = adjacent_positions(pos, n);
            if !adj.iter().any(|a| !placement.contains_key(a)) {
                return false;
            }
        }
    }
    for &pos in &state.confirmed_evil {
        if !state.executed.contains(&pos) && !placement.contains_key(&pos) {
            return false;
        }
    }
    true
}

fn unrevealed_must_be_villager(
    pos: u8, evil_positions: &HashMap<u8, String>, state: &GameState,
    doppelganger_pos: Option<u8>, drunk_pos: Option<u8>,
) -> bool {
    let mut max_outcasts = state.board_outcast_count
        .unwrap_or(state.deck.outcasts.len() as u8) as i32;
    if state.deck.minions.iter().any(|m| m == "Chancellor") {
        max_outcasts += 1;
    }
    let mut occupied = 0i32;
    for card in &state.cards {
        if evil_positions.contains_key(&card.position) || card.position == pos { continue; }
        if let Some(cd) = get_card(&card.apparent_role) {
            if cd.faction == Faction::Outcast {
                occupied += 1;
            }
        }
    }
    if let Some(dp) = doppelganger_pos {
        if dp != pos && !evil_positions.contains_key(&dp) { occupied += 1; }
    }
    if let Some(dp) = drunk_pos {
        if dp != pos && !evil_positions.contains_key(&dp) { occupied += 1; }
    }
    occupied >= max_outcasts
}

fn hidden_outcast_presence_flags(role_name: &str, state: &GameState) -> (bool, bool) {
    let normalized_role = normalize_role(role_name);
    if !state.deck.outcasts.iter().any(|role| normalize_role(role) == normalized_role) {
        return (false, true);
    }
    let mut slots = match state.board_outcast_count {
        Some(s) => s as i32,
        None => return (true, true),
    };
    // The identity hypotheses describe the post-conversion board. Chancellor
    // adds one Outcast identity, represented by allowing its conversion target
    // to overlap a hidden-Outcast position such as Drunk.
    if state.deck.minions.iter().any(|m| m == "Chancellor") {
        slots += 1;
    }
    let other_outcasts = state.deck.outcasts.len() as i32 - 1;
    (slots > 0, other_outcasts >= slots)
}

fn evil_role_subsets_fn(evil_roles: &[String], state: &GameState, expected_remaining: usize) -> Vec<Vec<String>> {
    let minion_pool: Vec<String> = evil_roles.iter()
        .filter(|r| get_card(r).map_or(false, |c| c.faction == Faction::Minion))
        .cloned().collect();
    let demon_pool: Vec<String> = evil_roles.iter()
        .filter(|r| get_card(r).map_or(false, |c| c.faction == Faction::Demon))
        .cloned().collect();

    let mut bm = state.board_minion_count.map(|x| x as i32);
    let mut bd = state.board_demon_count.map(|x| x as i32);

    let mut mp = minion_pool.clone();
    let mut dp = demon_pool.clone();
    for (_pos, role) in &state.executed_evil_roles {
        let norm = normalize_role(role);
        if let Some(idx) = mp.iter().position(|r| normalize_role(r) == norm) {
            if let Some(ref mut b) = bm { *b -= 1; }
            mp.remove(idx);
        } else if let Some(idx) = dp.iter().position(|r| normalize_role(r) == norm) {
            if let Some(ref mut b) = bd { *b -= 1; }
            dp.remove(idx);
        }
    }

    if let (Some(bm_v), Some(bd_v)) = (bm, bd) {
        let bm_pick = bm_v.max(0) as usize;
        let bd_pick = bd_v.max(0) as usize;
        let mut subsets = Vec::new();
        for m_combo in combinations_of_strings(&mp, bm_pick) {
            for d_combo in combinations_of_strings(&dp, bd_pick) {
                let mut subset: Vec<String> = m_combo.clone();
                subset.extend(d_combo);
                if subset.len() == expected_remaining {
                    subsets.push(subset);
                }
            }
        }
        if subsets.is_empty() {
            vec![evil_roles[..expected_remaining.min(evil_roles.len())].to_vec()]
        } else {
            subsets
        }
    } else {
        combinations_of_strings(evil_roles, expected_remaining)
    }
}

// ── Combinatorial helpers ──

fn combinations_of(items: &[u8], k: usize) -> Vec<Vec<u8>> {
    if k == 0 { return vec![vec![]]; }
    if k > items.len() { return vec![]; }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        for mut rest in combinations_of(&items[i + 1..], k - 1) {
            rest.insert(0, item);
            result.push(rest);
        }
    }
    result
}

fn combinations_of_strings(items: &[String], k: usize) -> Vec<Vec<String>> {
    if k == 0 { return vec![vec![]]; }
    if k > items.len() { return vec![]; }
    let mut result = Vec::new();
    for (i, item) in items.iter().enumerate() {
        for mut rest in combinations_of_strings(&items[i + 1..], k - 1) {
            rest.insert(0, item.clone());
            result.push(rest);
        }
    }
    result
}

fn combinations_indices(n: usize, k: usize) -> Vec<Vec<usize>> {
    let items: Vec<usize> = (0..n).collect();
    if k == 0 { return vec![vec![]]; }
    if k > n { return vec![]; }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        for mut rest in combinations_indices_inner(&items[i + 1..], k - 1) {
            rest.insert(0, item);
            result.push(rest);
        }
    }
    result
}

fn combinations_indices_inner(items: &[usize], k: usize) -> Vec<Vec<usize>> {
    if k == 0 { return vec![vec![]]; }
    if k > items.len() { return vec![]; }
    let mut result = Vec::new();
    for (i, &item) in items.iter().enumerate() {
        for mut rest in combinations_indices_inner(&items[i + 1..], k - 1) {
            rest.insert(0, item);
            result.push(rest);
        }
    }
    result
}

/// Distinct permutations (handles duplicates).
fn permutations_of(roles: &[String]) -> Vec<Vec<String>> {
    if roles.is_empty() { return vec![vec![]]; }
    let mut seen: HashSet<Vec<String>> = HashSet::new();
    let mut result = Vec::new();
    for (i, role) in roles.iter().enumerate() {
        let rest: Vec<String> = roles.iter().enumerate()
            .filter(|(j, _)| *j != i)
            .map(|(_, r)| r.clone()).collect();
        for perm in permutations_of(&rest) {
            let mut full = vec![role.clone()];
            full.extend(perm);
            if seen.insert(full.clone()) {
                result.push(full);
            }
        }
    }
    result
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CardInfo;

    fn card(position: u8, apparent_role: &str) -> CardInfo {
        CardInfo {
            position,
            apparent_role: apparent_role.to_string(),
            ..CardInfo::default()
        }
    }

    fn chancellor_at(position: u8) -> HashMap<u8, String> {
        HashMap::from([(position, "Chancellor".to_string())])
    }

    #[test]
    fn chancellor_added_outcast_is_not_restricted_to_adjacent_cards() {
        let mut state = GameState::default();
        state.n_cards = 6;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Plague Doctor"),
            card(3, "Baker"),
            card(4, "Bombardier"),
            card(5, "Baker"),
            card(6, "Baker"),
        ];

        let candidates = chancellor_conversion_candidates(
            &state,
            &chancellor_at(1),
            None,
            None,
            None,
            Some(1),
        );

        assert!(candidates.contains(&Some(2)));
        assert!(candidates.contains(&Some(4)));
        assert!(!candidates.contains(&Some(3)));
    }

    #[test]
    fn hidden_doppelganger_can_be_the_added_outcast_identity() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Plague Doctor"),
            card(3, "Baker"),
            card(4, "Architect"),
            card(5, "Baker"),
        ];

        let candidates = chancellor_conversion_candidates(
            &state,
            &chancellor_at(1),
            None,
            Some(4),
            None,
            Some(1),
        );

        assert!(candidates.contains(&Some(4)));
    }

    #[test]
    fn final_chancellor_requires_a_real_outcast_neighbor() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Baker"),
            card(3, "Bombardier"),
            card(4, "Baker"),
            card(5, "Baker"),
        ];

        let candidates = chancellor_conversion_candidates(
            &state,
            &chancellor_at(1),
            None,
            None,
            None,
            Some(1),
        );

        assert!(candidates.is_empty());
    }

    #[test]
    fn chancellor_generated_plague_doctor_still_acts_later_in_start() {
        let mut state = GameState::default();
        state.n_cards = 4;
        state.n_evil = 1;
        state.cards = vec![
            card(1, "Baker"),
            card(2, "Plague Doctor"),
            card(3, "Baker"),
            card(4, "Architect"),
        ];
        state.deck.villagers = vec!["Baker".to_string(), "Architect".to_string()];
        state.deck.outcasts = vec!["Plague Doctor".to_string()];
        state.deck.minions = vec!["Chancellor".to_string()];

        let generated_pd_worlds: Vec<Scenario> = build_scenarios(&state)
            .into_iter()
            .filter(|scenario| {
                scenario.evil_positions.get(&1).map(String::as_str) == Some("Chancellor")
                    && scenario.chancellor_conversion == Some(2)
            })
            .collect();

        assert!(!generated_pd_worlds.is_empty());
        assert!(generated_pd_worlds
            .iter()
            .all(|scenario| scenario.pd_corrupted.is_some()));
    }
}
