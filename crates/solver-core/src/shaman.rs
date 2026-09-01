//! Pure native Shaman Start enumeration and current-role replay.
//!
//! The caller supplies the complete current `CharacterData` map at Shaman's
//! ordered Start slot. In ordinary pre-Reveal setup `registerAs` is still null,
//! so the native Villager filter sees this exact data map.

use std::collections::{HashMap, HashSet};

use crate::knowledge_base::{get_card, normalize_role, Faction};
use crate::types::ShamanTrace;

fn is_real_villager(role: &str) -> Option<bool> {
    // Public Saint embeds SaintVillager but is not in the normal Rust card
    // table. Public Bombardier's unrelated managed Saint role arrives by its
    // display name and remains an Outcast.
    if normalize_role(role) == "saint" {
        return Some(true);
    }
    get_card(role).map(|card| card.faction == Faction::Villager)
}

/// Enumerate Shaman's ordinary ordered source/destination draws.
///
/// `None` means the exact input is incomplete or malformed and its caller must
/// fall back atomically. `Some(empty)` is an exact native failure surface:
/// fewer than two Villager occurrences survive to the second unguarded draw.
/// A complete ordinary board has one unique occurrence per physical position.
pub fn enumerate_shaman_traces(
    current_roles: &HashMap<u8, String>,
    current_order: &[u8],
    board_size: u8,
) -> Option<Vec<ShamanTrace>> {
    if board_size == 0
        || current_roles.len() != board_size as usize
        || current_order.len() != board_size as usize
    {
        return None;
    }

    let order_set: HashSet<u8> = current_order.iter().copied().collect();
    if order_set.len() != board_size as usize
        || (1..=board_size).any(|position| {
            !order_set.contains(&position) || !current_roles.contains_key(&position)
        })
    {
        return None;
    }

    let shaman_count = current_order
        .iter()
        .filter(|position| {
            current_roles
                .get(position)
                .is_some_and(|role| normalize_role(role) == "shaman")
        })
        .count();
    if shaman_count != 1 {
        return None;
    }

    let mut villagers = Vec::new();
    for &position in current_order {
        let role = current_roles.get(&position)?;
        match is_real_villager(role) {
            Some(true) => villagers.push(position),
            Some(false) => {}
            None => return None,
        }
    }
    if villagers.len() < 2 {
        return Some(Vec::new());
    }

    let mut traces = Vec::new();
    for &source_position in &villagers {
        for &target_position in &villagers {
            if source_position == target_position {
                continue;
            }
            traces.push(ShamanTrace {
                source_position,
                target_position,
                copied_role: current_roles.get(&source_position)?.clone(),
                target_previous_roles: vec![current_roles.get(&target_position)?.clone()],
            });
        }
    }
    Some(traces)
}

/// Apply one exact Shaman overwrite to a single current-data position.
pub fn role_after_shaman(
    position: u8,
    current_roles: &HashMap<u8, String>,
    trace: &ShamanTrace,
) -> Option<String> {
    if trace.source_position == trace.target_position
        || !current_roles
            .get(&trace.source_position)
            .is_some_and(|role| normalize_role(role) == normalize_role(&trace.copied_role))
        || trace.target_previous_roles.len() != 1
        || !current_roles
            .get(&trace.target_position)
            .is_some_and(|role| {
                normalize_role(role) == normalize_role(&trace.target_previous_roles[0])
            })
    {
        return None;
    }
    if position == trace.target_position {
        return Some(trace.copied_role.clone());
    }
    current_roles.get(&position).cloned()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn roles(entries: &[(u8, &str)]) -> HashMap<u8, String> {
        entries
            .iter()
            .map(|(position, role)| (*position, (*role).to_string()))
            .collect()
    }

    #[test]
    fn enumerates_every_ordered_villager_pair_from_current_data() {
        let current = roles(&[
            (1, "Puppeteer"),
            (2, "Puppet"),
            (3, "Shaman"),
            (4, "Scout"),
            (5, "Witness"),
            (6, "Lilis"),
        ]);

        let traces = enumerate_shaman_traces(&current, &[6, 5, 4, 3, 2, 1], 6)
            .expect("the complete current map is exact");

        assert_eq!(traces.len(), 2);
        assert_eq!(
            traces
                .iter()
                .map(|trace| (
                    trace.source_position,
                    trace.target_position,
                    normalize_role(&trace.copied_role),
                    normalize_role(&trace.target_previous_roles[0]),
                ))
                .collect::<HashSet<_>>(),
            HashSet::from([
                (4, 5, "scout".to_string(), "witness".to_string()),
                (5, 4, "witness".to_string(), "scout".to_string()),
            ])
        );
    }

    #[test]
    fn fewer_than_two_villagers_is_an_exact_native_failure() {
        let current = roles(&[(1, "Puppeteer"), (2, "Puppet"), (3, "Shaman")]);
        assert_eq!(
            enumerate_shaman_traces(&current, &[3, 2, 1], 3),
            Some(Vec::new())
        );
    }

    #[test]
    fn incomplete_duplicate_or_actorless_inputs_are_malformed() {
        let current = roles(&[(1, "Puppeteer"), (2, "Scout"), (3, "Shaman")]);
        assert!(enumerate_shaman_traces(&current, &[3, 2], 3).is_none());
        assert!(enumerate_shaman_traces(&current, &[3, 2, 2], 3).is_none());

        let no_actor = roles(&[(1, "Puppeteer"), (2, "Scout"), (3, "Witness")]);
        assert!(enumerate_shaman_traces(&no_actor, &[3, 2, 1], 3).is_none());
    }

    #[test]
    fn replay_overwrites_only_the_destination_and_validates_provenance() {
        let current = roles(&[(1, "Shaman"), (2, "Scout"), (3, "Witness")]);
        let trace = ShamanTrace {
            source_position: 2,
            target_position: 3,
            copied_role: "Scout".to_string(),
            target_previous_roles: vec!["Witness".to_string()],
        };

        assert_eq!(
            role_after_shaman(1, &current, &trace).as_deref(),
            Some("Shaman")
        );
        assert_eq!(
            role_after_shaman(2, &current, &trace).as_deref(),
            Some("Scout")
        );
        assert_eq!(
            role_after_shaman(3, &current, &trace).as_deref(),
            Some("Scout")
        );

        let mut stale = trace;
        stale.target_previous_roles = vec!["Judge".to_string()];
        assert!(role_after_shaman(3, &current, &stale).is_none());
    }
}
