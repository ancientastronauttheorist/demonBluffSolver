/// Scenario generation: enumerate all possible evil placements and build
/// full scenarios with corruption variants.

use std::collections::{HashMap, HashSet};
use crate::corruption::{compute_corruption, apply_post_corruption};
use crate::geometry::adjacent_positions;
use crate::knowledge_base::{self, get_card, is_villager_role, normalize_role, Faction};
use crate::types::{GameState, Scenario};

/// Generate all candidate scenarios for the current game state.
pub fn build_scenarios(state: &GameState) -> Vec<Scenario> {
    let placements = generate_evil_placements(state);
    let mut scenarios = Vec::new();
    let n = state.n_cards;

    // Find all PD positions
    let all_pd_positions: Vec<u8> = state.cards.iter()
        .filter(|c| knowledge_base::is_plague_doctor(&c.apparent_role))
        .map(|c| c.position)
        .collect();

    let pd_in_deck = state.deck.outcasts.iter().any(|o| knowledge_base::is_plague_doctor(o));
    let (pd_can_be_on_board, pd_can_be_absent) = hidden_outcast_presence_flags("Plague_Doctor", state);
    let pd_hidden_but_possible = pd_in_deck && all_pd_positions.is_empty() && pd_can_be_on_board;

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

        // Determine PD corruption targets
        let pd_pos = all_pd_positions.iter()
            .find(|&&p| !full_evil.contains_key(&p))
            .copied();

        let mut pd_targets: Vec<Option<u8>> = vec![None];
        let mut pd_on_board = pd_pos.is_some();
        if !pd_on_board && pd_hidden_but_possible {
            let revealed: HashSet<u8> = state.cards.iter().map(|c| c.position).collect();
            pd_on_board = (1..=n).any(|p| !revealed.contains(&p) && !full_evil.contains_key(&p));
        }
        if pd_on_board && pd_in_deck {
            if let Some(known) = state.pd_corruption_target {
                pd_targets = vec![Some(known)];
            } else {
                let mut cands: Vec<Option<u8>> = Vec::new();
                for p in 1..=n {
                    if Some(p) == pd_pos || full_evil.contains_key(&p) { continue; }
                    if let Some(card) = state.card_at(p) {
                        if is_villager_role(&card.apparent_role) {
                            cands.push(Some(p));
                        }
                    } else if unrevealed_must_be_villager(p, &full_evil, state, None, None) {
                        cands.push(Some(p));
                    }
                }
                if !cands.is_empty() {
                    pd_targets = cands;
                }
            }
        }
        if pd_can_be_absent && !pd_targets.contains(&None) {
            pd_targets.push(None);
        }

        // Poisoner detection
        let _has_poisoner = full_evil.values().any(|r| r == "Poisoner");
        let poisoner_pos = full_evil.iter()
            .find(|(_, r)| r.as_str() == "Poisoner")
            .map(|(&p, _)| p);

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

        // Chancellor conversion candidates
        let mut chancellor_conv_cands: Vec<Option<u8>> = vec![None];
        let chancellor_pos_opt = full_evil.iter()
            .find(|(_, r)| r.as_str() == "Chancellor")
            .map(|(&p, _)| p);
        if let Some(ch_pos) = chancellor_pos_opt {
            let mut conv_cands = Vec::new();
            for adj in adjacent_positions(ch_pos, n) {
                if full_evil.contains_key(&adj) { continue; }
                if let Some(card) = state.card_at(adj) {
                    if is_villager_role(&card.apparent_role) {
                        conv_cands.push(adj);
                    }
                } else if unrevealed_must_be_villager(adj, &full_evil, state, None, None) {
                    conv_cands.push(adj);
                }
            }
            if !conv_cands.is_empty() {
                chancellor_conv_cands = conv_cands.into_iter().map(Some).collect();
            }
        }

        // Nested loops: chancellor × pd × dopp × drunk × poisoner × nk_alch
        for &chan_conv in &chancellor_conv_cands {
            for &pd_t in &pd_targets {
                let mut seen: HashSet<(Vec<u8>, Option<u8>, Option<u8>, Option<u8>)> = HashSet::new();

                for &dopp_pos_opt in &dopp_candidates {
                    // Doppelganger is an Outcast — PD only corrupts Villagers
                    if pd_t.is_some() && pd_t == dopp_pos_opt {
                        continue;
                    }
                    for &drunk_pos_opt in &drunk_candidates {
                        if drunk_pos_opt.is_some() && drunk_pos_opt == dopp_pos_opt {
                            continue;
                        }
                        // Drunk is an Outcast — PD only corrupts Villagers
                        if pd_t.is_some() && pd_t == drunk_pos_opt {
                            continue;
                        }

                        // Poisoner targets (computed inside dopp/drunk loop)
                        let mut pois_targets: Vec<Option<u8>> = vec![None];
                        if let Some(pp) = poisoner_pos {
                            let mut cands = Vec::new();
                            for adj in adjacent_positions(pp, n) {
                                if full_evil.contains_key(&adj) { continue; }
                                if let Some(card) = state.card_at(adj) {
                                    if is_villager_role(&card.apparent_role) {
                                        cands.push(adj);
                                    }
                                } else if unrevealed_must_be_villager(adj, &full_evil, state, dopp_pos_opt, drunk_pos_opt) {
                                    cands.push(adj);
                                }
                            }
                            if !cands.is_empty() {
                                pois_targets = cands.into_iter().map(Some).collect();
                            }
                        }

                        // Night-killed Alchemist variants
                        let mut nk_alch_variants: Vec<Vec<u8>> = vec![vec![]];
                        if !state.night_kills.is_empty() && state.deck.villagers.iter().any(|v| v == "Alchemist") {
                            let revealed: HashSet<u8> = state.cards.iter().map(|c| c.position).collect();
                            let known_alch = state.cards.iter()
                                .filter(|c| c.apparent_role == "Alchemist" && !full_evil.contains_key(&c.position))
                                .count();
                            let deck_alch = state.deck.villagers.iter().filter(|v| v.as_str() == "Alchemist").count();
                            if known_alch < deck_alch {
                                for &nk in &state.night_kills {
                                    if !full_evil.contains_key(&nk) && !revealed.contains(&nk) {
                                        nk_alch_variants.push(vec![nk]);
                                    }
                                }
                            }
                        }

                        for &pois_t in &pois_targets {
                            let raw_corrupted = compute_corruption(
                                &full_evil, state, pd_t, dopp_pos_opt, pois_t, drunk_pos_opt,
                            );
                            for nk_alch in &nk_alch_variants {
                                let extra = if nk_alch.is_empty() { &[][..] } else { nk_alch.as_slice() };
                                let (final_corrupted, alch_cures) = apply_post_corruption(
                                    raw_corrupted.clone(), &full_evil, state,
                                    puppet_pos, drunk_pos_opt, extra,
                                );

                                // Dedup key
                                let mut corr_key: Vec<u8> = final_corrupted.iter().copied().collect();
                                corr_key.sort_unstable();
                                let key = (corr_key, dopp_pos_opt, drunk_pos_opt, chan_conv);
                                if seen.contains(&key) {
                                    continue;
                                }
                                seen.insert(key);

                                scenarios.push(Scenario {
                                    evil_positions: full_evil.clone(),
                                    puppet_position: puppet_pos,
                                    corrupted: final_corrupted,
                                    pd_corrupted: pd_t,
                                    doppelganger_position: dopp_pos_opt,
                                    drunk_position: drunk_pos_opt,
                                    alchemist_cures: alch_cures,
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
    if !state.deck.outcasts.iter().any(|o| o == role_name) {
        return (false, true);
    }
    let mut slots = match state.board_outcast_count {
        Some(s) => s as i32,
        None => return (true, true),
    };
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
