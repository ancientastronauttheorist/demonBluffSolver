//! Pure native Puppeteer Start enumeration and current-role replay.
//!
//! This module deliberately does not inspect stable Evil origins or apparent
//! card presentations. Its caller supplies the exact current `CharacterData`
//! role map after Twin Minion and the native current-character scan order.

use std::collections::HashMap;

use crate::geometry::adjacent_positions;
use crate::knowledge_base::{get_card, normalize_role, Faction};
use crate::types::{PuppeteerNeighborSide, PuppeteerStartOutcome, PuppeteerTrace};

fn is_puppeteer(role: &str) -> bool {
    normalize_role(role) == "puppeteer"
}

fn is_saint_villager(role: &str) -> bool {
    // Public Saint embeds managed SaintVillager. Public Bombardier embeds the
    // unrelated managed Saint role and therefore arrives here as Bombardier.
    normalize_role(role) == "saint"
}

fn is_real_villager(role: &str) -> Option<bool> {
    if is_saint_villager(role) {
        return Some(true);
    }
    get_card(role).map(|card| card.faction == Faction::Villager)
}

#[derive(Debug, Clone)]
struct CandidateOccurrence {
    side: PuppeteerNeighborSide,
    position: u8,
    erased_role: String,
}

/// Enumerate the ordinary shipped Puppeteer Start outcomes.
///
/// `current_roles` is the complete current `CharacterData` role map after
/// Twin Minion acts. `current_order` is the shallow
/// `Gameplay.CurrentCharacters` order used by the ordinary first-match Start
/// scanner. Physical adjacency is the circular displayed-position ring of
/// `board_size` cards.
///
/// The first current exact Puppeteer occurrence is the only actor. Its
/// previous then next physical neighbours are filtered by real Villager data.
/// The first retained public Saint occurrence is removed exactly once. An
/// empty result emits one [`PuppeteerStartOutcome::NoCandidate`]; otherwise
/// conversion is mandatory and one trace is emitted for every remaining
/// occurrence, without deduplicating equal physical targets.
///
/// A missing actor, zero-sized board, out-of-range actor, or incomplete exact
/// role map is malformed input and produces no traces.
pub fn enumerate_puppeteer_traces(
    current_roles: &HashMap<u8, String>,
    current_order: &[u8],
    board_size: u8,
) -> Vec<PuppeteerTrace> {
    if board_size == 0 || (1..=board_size).any(|position| !current_roles.contains_key(&position)) {
        return Vec::new();
    }

    let Some(actor_position) = current_order.iter().copied().find(|position| {
        current_roles
            .get(position)
            .is_some_and(|role| is_puppeteer(role))
    }) else {
        return Vec::new();
    };
    if actor_position == 0 || actor_position > board_size {
        return Vec::new();
    }

    // Removing Puppeteer from a one-card board leaves no neighbour surface.
    let mut candidates = Vec::new();
    if board_size > 1 {
        let [previous, next] = adjacent_positions(actor_position, board_size);
        for (side, position) in [
            (PuppeteerNeighborSide::Previous, previous),
            (PuppeteerNeighborSide::Next, next),
        ] {
            let role = current_roles
                .get(&position)
                .expect("complete current-role map checked above");
            let Some(is_villager) = is_real_villager(role) else {
                // An exact role map cannot contain an unclassified role at a
                // native type-test boundary.
                return Vec::new();
            };
            if is_villager {
                candidates.push(CandidateOccurrence {
                    side,
                    position,
                    erased_role: role.clone(),
                });
            }
        }
    }

    if let Some(first_saint) = candidates
        .iter()
        .position(|candidate| is_saint_villager(&candidate.erased_role))
    {
        candidates.remove(first_saint);
    }

    if candidates.is_empty() {
        return vec![PuppeteerTrace {
            actor_position,
            outcome: PuppeteerStartOutcome::NoCandidate,
        }];
    }

    candidates
        .into_iter()
        .enumerate()
        .filter_map(|(candidate_occurrence_index, candidate)| {
            let candidate_occurrence_index = u8::try_from(candidate_occurrence_index).ok()?;
            Some(PuppeteerTrace {
                actor_position,
                outcome: PuppeteerStartOutcome::Converted {
                    candidate_occurrence_index,
                    neighbor_side: candidate.side,
                    target_position: candidate.position,
                    erased_villager_role: candidate.erased_role,
                },
            })
        })
        .collect()
}

/// Apply one exact Puppeteer event to a single current-data role.
///
/// Conversion performs full `Init` with Puppet data at the selected physical
/// target. The saved Villager role remains in the trace as copied/bluff
/// provenance and is not the target's current data after this writer.
pub fn current_data_after_puppeteer_at(
    position: u8,
    before: Option<&str>,
    trace: &PuppeteerTrace,
) -> Option<String> {
    if matches!(
        trace.outcome,
        PuppeteerStartOutcome::Converted {
            target_position,
            ..
        } if target_position == position
    ) {
        return Some("Puppet".to_string());
    }
    before.map(str::to_string)
}

/// Return the current role at `position` after replaying one Puppeteer trace
/// over an explicit complete post-Twin role map.
pub fn role_after_puppeteer(
    position: u8,
    current_roles: &HashMap<u8, String>,
    trace: &PuppeteerTrace,
) -> Option<String> {
    current_data_after_puppeteer_at(
        position,
        current_roles.get(&position).map(String::as_str),
        trace,
    )
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
    fn filters_real_villagers_and_makes_conversion_mandatory() {
        let current = roles(&[
            (1, "Pooka"),
            (2, "Scout"),
            (3, "Puppeteer"),
            (4, "Witch"),
            (5, "Baker"),
        ]);
        let traces = enumerate_puppeteer_traces(&current, &[5, 4, 3, 2, 1], 5);

        assert_eq!(traces.len(), 1);
        assert_eq!(
            traces[0],
            PuppeteerTrace {
                actor_position: 3,
                outcome: PuppeteerStartOutcome::Converted {
                    candidate_occurrence_index: 0,
                    neighbor_side: PuppeteerNeighborSide::Previous,
                    target_position: 2,
                    erased_villager_role: "Scout".to_string(),
                },
            }
        );
        assert!(!traces
            .iter()
            .any(|trace| matches!(trace.outcome, PuppeteerStartOutcome::NoCandidate)));
    }

    #[test]
    fn no_real_villager_candidate_is_an_explicit_no_op() {
        let current = roles(&[(1, "Witch"), (2, "Puppeteer"), (3, "Bombardier")]);
        let traces = enumerate_puppeteer_traces(&current, &[3, 2, 1], 3);

        assert_eq!(
            traces,
            vec![PuppeteerTrace {
                actor_position: 2,
                outcome: PuppeteerStartOutcome::NoCandidate,
            }]
        );
    }

    #[test]
    fn current_order_selects_only_the_first_current_puppeteer() {
        let current = roles(&[
            (1, "Puppeteer"),
            (2, "Scout"),
            (3, "Baker"),
            (4, "Puppeteer"),
            (5, "Pooka"),
        ]);
        let traces = enumerate_puppeteer_traces(&current, &[5, 4, 3, 2, 1], 5);

        assert_eq!(traces.len(), 1);
        assert!(traces.iter().all(|trace| trace.actor_position == 4));
        assert!(matches!(
            traces[0].outcome,
            PuppeteerStartOutcome::Converted {
                target_position: 3,
                ..
            }
        ));
    }

    #[test]
    fn post_twin_map_can_relocate_the_puppeteer_actor() {
        // Stable-origin placement is deliberately absent. This is the exact
        // post-Twin current-data map, where Puppeteer has moved to #2.
        let current = roles(&[
            (1, "Scout"),
            (2, "Puppeteer"),
            (3, "Baker"),
            (4, "Twin Minion"),
            (5, "Pooka"),
        ]);
        let traces = enumerate_puppeteer_traces(&current, &[5, 4, 3, 2, 1], 5);

        assert_eq!(traces.len(), 2);
        assert!(traces.iter().all(|trace| trace.actor_position == 2));
        assert!(matches!(
            traces[0].outcome,
            PuppeteerStartOutcome::Converted {
                neighbor_side: PuppeteerNeighborSide::Previous,
                target_position: 1,
                ..
            }
        ));
        assert!(matches!(
            traces[1].outcome,
            PuppeteerStartOutcome::Converted {
                neighbor_side: PuppeteerNeighborSide::Next,
                target_position: 3,
                ..
            }
        ));
    }

    #[test]
    fn removes_only_the_first_saint_occurrence() {
        let current = roles(&[(1, "Saint"), (2, "Puppeteer"), (3, "Saint")]);
        let traces = enumerate_puppeteer_traces(&current, &[3, 2, 1], 3);

        assert_eq!(traces.len(), 1);
        assert_eq!(
            traces[0].outcome,
            PuppeteerStartOutcome::Converted {
                candidate_occurrence_index: 0,
                neighbor_side: PuppeteerNeighborSide::Next,
                target_position: 3,
                erased_villager_role: "Saint".to_string(),
            }
        );
    }

    #[test]
    fn sole_saint_candidate_produces_no_candidate() {
        let current = roles(&[(1, "Pooka"), (2, "Puppeteer"), (3, "Saint")]);
        let traces = enumerate_puppeteer_traces(&current, &[3, 2, 1], 3);

        assert!(matches!(
            traces.as_slice(),
            [PuppeteerTrace {
                actor_position: 2,
                outcome: PuppeteerStartOutcome::NoCandidate,
            }]
        ));
    }

    #[test]
    fn two_card_non_saint_preserves_both_neighbor_occurrences() {
        let current = roles(&[(1, "Puppeteer"), (2, "Scout")]);
        let traces = enumerate_puppeteer_traces(&current, &[2, 1], 2);

        assert_eq!(traces.len(), 2);
        assert_eq!(
            traces
                .iter()
                .map(|trace| match trace.outcome {
                    PuppeteerStartOutcome::Converted {
                        candidate_occurrence_index,
                        neighbor_side,
                        target_position,
                        ..
                    } => (candidate_occurrence_index, neighbor_side, target_position),
                    PuppeteerStartOutcome::NoCandidate => unreachable!(),
                })
                .collect::<Vec<_>>(),
            vec![
                (0, PuppeteerNeighborSide::Previous, 2),
                (1, PuppeteerNeighborSide::Next, 2),
            ]
        );
    }

    #[test]
    fn two_card_saint_removes_one_duplicate_and_keeps_the_other() {
        let current = roles(&[(1, "Puppeteer"), (2, "Saint")]);
        let traces = enumerate_puppeteer_traces(&current, &[2, 1], 2);

        assert_eq!(traces.len(), 1);
        assert_eq!(
            traces[0].outcome,
            PuppeteerStartOutcome::Converted {
                candidate_occurrence_index: 0,
                neighbor_side: PuppeteerNeighborSide::Next,
                target_position: 2,
                erased_villager_role: "Saint".to_string(),
            }
        );
    }

    #[test]
    fn replay_writes_puppet_only_at_the_selected_target() {
        let current = roles(&[(1, "Puppeteer"), (2, "Scout"), (3, "Pooka")]);
        let trace = PuppeteerTrace {
            actor_position: 1,
            outcome: PuppeteerStartOutcome::Converted {
                candidate_occurrence_index: 0,
                neighbor_side: PuppeteerNeighborSide::Next,
                target_position: 2,
                erased_villager_role: "Scout".to_string(),
            },
        };

        assert_eq!(
            role_after_puppeteer(1, &current, &trace).as_deref(),
            Some("Puppeteer")
        );
        assert_eq!(
            role_after_puppeteer(2, &current, &trace).as_deref(),
            Some("Puppet")
        );
        assert_eq!(
            current_data_after_puppeteer_at(2, None, &trace).as_deref(),
            Some("Puppet")
        );
        assert_eq!(
            role_after_puppeteer(3, &current, &trace).as_deref(),
            Some("Pooka")
        );
    }

    #[test]
    fn malformed_partial_map_or_missing_actor_has_no_complete_trace() {
        let partial = roles(&[(1, "Puppeteer"), (2, "Scout")]);
        assert!(enumerate_puppeteer_traces(&partial, &[2, 1], 3).is_empty());

        let complete = roles(&[(1, "Witch"), (2, "Scout")]);
        assert!(enumerate_puppeteer_traces(&complete, &[2, 1], 2).is_empty());

        let unknown_neighbor = roles(&[(1, "Puppeteer"), (2, "Unknown")]);
        assert!(enumerate_puppeteer_traces(&unknown_neighbor, &[2, 1], 2).is_empty());
    }

    #[test]
    fn trace_serde_preserves_selected_occurrence_side_and_erased_role() {
        let trace = PuppeteerTrace {
            actor_position: 7,
            outcome: PuppeteerStartOutcome::Converted {
                candidate_occurrence_index: 1,
                neighbor_side: PuppeteerNeighborSide::Next,
                target_position: 8,
                erased_villager_role: "Fortune Teller".to_string(),
            },
        };

        let json = serde_json::to_value(&trace).unwrap();
        assert_eq!(json["outcome"]["kind"], "converted");
        assert_eq!(json["outcome"]["candidate_occurrence_index"], 1);
        assert_eq!(json["outcome"]["neighbor_side"], "next");
        assert_eq!(json["outcome"]["erased_villager_role"], "Fortune Teller");
        assert_eq!(
            serde_json::from_value::<PuppeteerTrace>(json).unwrap(),
            trace
        );
    }
}
