/// Corruption computation: Pooka, Poisoner, PD -> Puppet cure -> Alchemist cures.

use std::collections::{HashMap, HashSet};
use crate::geometry::{adjacent_positions, positions_in_range};
use crate::knowledge_base::is_villager_role;
use crate::types::GameState;

/// Compute the raw corruption set for a scenario, BEFORE Puppet cure and Alchemist cures.
///
/// Sources: Drunk (intrinsic status), Pooka (adjacent), Poisoner (1 adjacent),
/// PD (1 villager).
///
/// Drunk is special in the current build: corruption-status clues such as Bard
/// count Drunk as Corrupted, but Plague Doctor reports Drunk as Not Corrupted.
/// Doppelganger is immune to Pooka/Poisoner (Outcast, not Villager).
/// Alchemist is immune to all corruption sources.
pub fn compute_corruption(
    full_evil: &HashMap<u8, String>,
    state: &GameState,
    pd_target: Option<u8>,
    doppelganger_pos: Option<u8>,
    poisoner_target: Option<u8>,
    drunk_pos: Option<u8>,
) -> HashSet<u8> {
    let n = state.n_cards;
    let mut corrupted = HashSet::new();

    if let Some(dp) = drunk_pos {
        corrupted.insert(dp);
    }

    let is_alchemist = |p: u8| -> bool {
        state.card_at(p)
            .map(|c| c.apparent_role == "Alchemist" || c.apparent_role == "alchemist")
            .unwrap_or(false)
    };

    // Evil-role corruption
    for (&pos, role) in full_evil {
        match role.as_str() {
            "Pooka" => {
                for adj in adjacent_positions(pos, n) {
                    if full_evil.contains_key(&adj) {
                        continue;
                    }
                    if Some(adj) == doppelganger_pos {
                        continue; // Doppelganger immune
                    }
                    if is_alchemist(adj) {
                        continue; // Alchemist immune
                    }
                    // Must be a Villager or unrevealed
                    if let Some(card) = state.card_at(adj) {
                        if is_villager_role(&card.apparent_role) {
                            corrupted.insert(adj);
                        }
                    } else {
                        // Unrevealed — could be a Villager
                        corrupted.insert(adj);
                    }
                }
            }
            "Poisoner" => {
                if let Some(pt) = poisoner_target {
                    if !full_evil.contains_key(&pt)
                        && Some(pt) != doppelganger_pos
                        && !is_alchemist(pt)
                    {
                        corrupted.insert(pt);
                    }
                }
            }
            _ => {}
        }
    }

    // PD corruption (Good ability, not from evil — but still a corruption source)
    if let Some(pd_t) = pd_target {
        if !full_evil.contains_key(&pd_t) && !is_alchemist(pd_t) {
            corrupted.insert(pd_t);
        }
    }

    corrupted
}

/// Apply post-corruption processing: Puppet cure then Alchemist cures.
///
/// Returns (final corrupted set, alch_cures map). The map's value is the count
/// each Alchemist reports — post-patch this is the number of Corrupted characters
/// in Range 2 at the start of the Round (i.e., AFTER Puppet cure but BEFORE any
/// Alchemist cure). The cure mechanic itself is unchanged (Alchemist removes
/// corruption from cureable neighbors).
pub fn apply_post_corruption(
    mut corrupted: HashSet<u8>,
    full_evil: &HashMap<u8, String>,
    state: &GameState,
    puppet_pos: Option<u8>,
    drunk_pos: Option<u8>,
    extra_alch_positions: &[u8],
) -> (HashSet<u8>, HashMap<u8, u8>) {
    let n = state.n_cards;
    let mut alch_cures: HashMap<u8, u8> = HashMap::new();

    // Step 1: Puppet cure — remove corruption from Puppet
    if let Some(pp) = puppet_pos {
        corrupted.remove(&pp);
    }

    // Snapshot the start-of-round corrupted set for Alchemist's clue.
    // All Alchemists report from this same snapshot — order-independent.
    let start_of_round = corrupted.clone();

    // Step 2: Alchemist cures in reverse seat order
    let mut alchemist_positions: Vec<u8> = Vec::new();
    for card in &state.cards {
        if card.apparent_role == "Alchemist" || card.apparent_role == "alchemist" {
            alchemist_positions.push(card.position);
        }
    }
    // Add extra Alchemist positions (night-killed positions that might be Alchemist)
    alchemist_positions.extend_from_slice(extra_alch_positions);
    // Reverse seat order: highest position first
    alchemist_positions.sort_unstable();
    alchemist_positions.reverse();

    for &alch_pos in &alchemist_positions {
        // Count = corrupted-in-range from start-of-round snapshot (post-patch).
        // Excludes the Alchemist itself (immune, never in the corrupted set).
        let mut count = 0u8;
        for p in positions_in_range(alch_pos, 2, n) {
            if start_of_round.contains(&p) {
                count += 1;
            }
        }
        // Evil Alchemist still reports a count (will be inverted by truth_status
        // in the validator since they lie). But evil Alchemist also can't cure.
        let can_cure = !full_evil.contains_key(&alch_pos);
        alch_cures.insert(alch_pos, count);
        if !can_cure {
            continue;
        }
        // Apply cures to the live corrupted set.
        for p in positions_in_range(alch_pos, 2, n) {
            if !corrupted.contains(&p) {
                continue;
            }
            if Some(p) != drunk_pos {
                corrupted.remove(&p);
            }
        }
    }

    (corrupted, alch_cures)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::CardInfo;

    #[test]
    fn drunk_is_intrinsic_corruption_status_and_not_alchemist_cured() {
        let mut state = GameState::default();
        state.n_cards = 5;
        state.cards = vec![
            CardInfo {
                position: 1,
                apparent_role: "Scout".to_string(),
                ..CardInfo::default()
            },
            CardInfo {
                position: 2,
                apparent_role: "Alchemist".to_string(),
                ..CardInfo::default()
            },
        ];

        let raw = compute_corruption(&HashMap::new(), &state, None, None, None, Some(1));
        assert!(raw.contains(&1));

        let (final_corrupted, alch_counts) =
            apply_post_corruption(raw, &HashMap::new(), &state, None, Some(1), &[]);
        assert!(final_corrupted.contains(&1));
        assert_eq!(alch_counts.get(&2), Some(&1));
    }
}
