//! Pure native Twin Minion Start enumeration and current-role replay.
//!
//! This module deliberately does not read or mutate [`crate::types::Scenario`].
//! Its caller supplies the exact post-Chancellor current-data map and the two
//! native collection orders. Scenario generation will adopt this boundary in a
//! later slice, after every later Start consumer can use the moved identity.

use std::collections::HashMap;

use crate::knowledge_base::{get_card, normalize_role, Faction};
use crate::types::{GameState, TwinNeighborSide, TwinStartOutcome, TwinTrace};

fn is_twin(role: &str) -> bool {
    normalize_role(role) == "twinminion"
}

fn is_demon(role: &str) -> bool {
    get_card(role).is_some_and(|card| card.faction == Faction::Demon)
}

/// Whether a distinct Twin swap needs public-action provenance that the
/// role-only replay does not yet carry.
///
/// Native keeps runtime alignment, current CharacterData dispatch, and the
/// delayed Minion bluff on separate layers after a swap. Until those layers
/// have their own trace, any captured reveal/action history can observe an
/// unsupported combination. Ordinary execution/current-role evidence is
/// deliberately absent from this gate: it exposes the final current dataRef,
/// which the exact replay derives directly.
pub fn distinct_swap_has_unsupported_public_action_evidence(
    state: &GameState,
    trace: &TwinTrace,
) -> bool {
    let TwinStartOutcome::Swap {
        neighbor_position, ..
    } = trace.outcome
    else {
        return false;
    };
    if neighbor_position == trace.actor_position {
        return false;
    }

    !state.cards.is_empty()
        || !state.slayer_results.is_empty()
        || !state.pd_ability_results.is_empty()
        || !state.blocked_positions.is_empty()
        || !state.night_kills.is_empty()
        || state.night_kill_evil_count != 0
        || !state.reveal_order.is_empty()
        || !state.used_abilities.is_empty()
        || !state.rambler_shut_up_observations.is_empty()
        || state.terminal_loss_role.is_some()
}

/// Enumerate the ordinary shipped Twin Minion Start outcomes.
///
/// `current_roles` is the exact current `CharacterData` role at each physical
/// position after Chancellor and before Twin. `current_order` is the shallow
/// `Gameplay.CurrentCharacters` order used to build the Demon pool, and may
/// contain repeated position references. `alive_order` is the physical circle
/// after native alive filtering.
///
/// The ordinary actor scan chooses the first exact Twin occurrence in
/// `current_order`. Normal construction orders that collection by descending
/// displayed ID, making the first match the highest-ID Twin as a consequence.
/// Demon-pool and previous/next occurrences are intentionally not deduplicated.
/// Invalid partial inputs (no Twin, a selected Demon absent from `alive_order`,
/// or a neighbor missing from `current_roles`) produce no branch for the
/// malformed path; authored setup supplies complete collections.
pub fn enumerate_twin_traces(
    current_roles: &HashMap<u8, String>,
    current_order: &[u8],
    alive_order: &[u8],
) -> Vec<TwinTrace> {
    let Some(actor_position) = current_order.iter().copied().find(|position| {
        current_roles
            .get(position)
            .is_some_and(|role| is_twin(role))
    }) else {
        return Vec::new();
    };

    let demon_occurrences: Vec<u8> = current_order
        .iter()
        .copied()
        .filter(|position| {
            current_roles
                .get(position)
                .is_some_and(|role| is_demon(role))
        })
        .collect();

    if demon_occurrences.is_empty() {
        return vec![TwinTrace {
            actor_position,
            outcome: TwinStartOutcome::NoDemon,
        }];
    }

    if alive_order.is_empty() {
        return Vec::new();
    }

    let mut traces = Vec::new();
    for (demon_occurrence_index, demon_anchor_position) in demon_occurrences.into_iter().enumerate()
    {
        let Some(anchor_index) = alive_order
            .iter()
            .position(|position| *position == demon_anchor_position)
        else {
            continue;
        };
        let previous_index = (anchor_index + alive_order.len() - 1) % alive_order.len();
        let next_index = (anchor_index + 1) % alive_order.len();

        for (neighbor_side, neighbor_index) in [
            (TwinNeighborSide::Previous, previous_index),
            (TwinNeighborSide::Next, next_index),
        ] {
            let neighbor_position = alive_order[neighbor_index];
            let Some(neighbor_pre_swap_role) = current_roles.get(&neighbor_position) else {
                continue;
            };
            let Ok(demon_occurrence_index) = u8::try_from(demon_occurrence_index) else {
                continue;
            };
            traces.push(TwinTrace {
                actor_position,
                outcome: TwinStartOutcome::Swap {
                    demon_occurrence_index,
                    demon_anchor_position,
                    neighbor_side,
                    neighbor_position,
                    neighbor_pre_swap_role: neighbor_pre_swap_role.clone(),
                },
            });
        }
    }
    traces
}

/// Apply one exact Twin event to a single pre-Twin current-data role.
///
/// A distinct actor receives the stored former neighbor data and the neighbor
/// receives Twin data. Those endpoint results are exact even when presentation
/// cannot supply `before`. Self-swap still performs both native Init calls but
/// leaves the data mapping unchanged.
pub fn current_data_after_twin_at(
    position: u8,
    before: Option<&str>,
    trace: &TwinTrace,
) -> Option<String> {
    let TwinStartOutcome::Swap {
        neighbor_position,
        neighbor_pre_swap_role,
        ..
    } = &trace.outcome
    else {
        return before.map(str::to_string);
    };

    if trace.actor_position == *neighbor_position {
        return before.map(str::to_string);
    }
    if position == trace.actor_position {
        return Some(neighbor_pre_swap_role.clone());
    }
    if position == *neighbor_position {
        return Some("Twin Minion".to_string());
    }
    before.map(str::to_string)
}

/// Return the current role at `position` after replaying one Twin trace over an
/// explicit complete pre-Twin role map.
pub fn role_after_twin(
    position: u8,
    current_roles: &HashMap<u8, String>,
    trace: &TwinTrace,
) -> Option<String> {
    current_data_after_twin_at(
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
    fn enumerates_anchor_then_previous_and_next_and_replays_swap() {
        let current = roles(&[(1, "Twin Minion"), (2, "Scout"), (3, "Pooka"), (4, "Witch")]);
        let traces = enumerate_twin_traces(&current, &[4, 3, 2, 1], &[1, 2, 3, 4]);

        assert_eq!(traces.len(), 2);
        assert_eq!(
            traces[0],
            TwinTrace {
                actor_position: 1,
                outcome: TwinStartOutcome::Swap {
                    demon_occurrence_index: 0,
                    demon_anchor_position: 3,
                    neighbor_side: TwinNeighborSide::Previous,
                    neighbor_position: 2,
                    neighbor_pre_swap_role: "Scout".to_string(),
                },
            }
        );
        assert!(matches!(
            traces[1].outcome,
            TwinStartOutcome::Swap {
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 4,
                ..
            }
        ));
        assert_eq!(
            role_after_twin(1, &current, &traces[0]).as_deref(),
            Some("Scout")
        );
        assert_eq!(
            role_after_twin(2, &current, &traces[0]).as_deref(),
            Some("Twin Minion")
        );
        assert_eq!(
            role_after_twin(3, &current, &traces[0]).as_deref(),
            Some("Pooka")
        );
    }

    #[test]
    fn no_demon_is_an_explicit_complete_outcome() {
        let current = roles(&[(1, "Twin Minion"), (2, "Scout")]);
        let traces = enumerate_twin_traces(&current, &[2, 1], &[1, 2]);

        assert_eq!(
            traces,
            vec![TwinTrace {
                actor_position: 1,
                outcome: TwinStartOutcome::NoDemon,
            }]
        );
        assert_eq!(
            role_after_twin(1, &current, &traces[0]).as_deref(),
            Some("Twin Minion")
        );
        assert_eq!(
            role_after_twin(2, &current, &traces[0]).as_deref(),
            Some("Scout")
        );
    }

    #[test]
    fn two_card_board_preserves_both_neighbor_occurrences_and_self_swap() {
        let current = roles(&[(1, "Twin Minion"), (2, "Pooka")]);
        let traces = enumerate_twin_traces(&current, &[2, 1], &[1, 2]);

        assert_eq!(traces.len(), 2);
        assert!(matches!(
            traces[0].outcome,
            TwinStartOutcome::Swap {
                neighbor_side: TwinNeighborSide::Previous,
                neighbor_position: 1,
                ..
            }
        ));
        assert!(matches!(
            traces[1].outcome,
            TwinStartOutcome::Swap {
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 1,
                ..
            }
        ));
        for trace in &traces {
            assert_eq!(
                role_after_twin(1, &current, trace).as_deref(),
                Some("Twin Minion")
            );
            assert_eq!(
                role_after_twin(2, &current, trace).as_deref(),
                Some("Pooka")
            );
        }
    }

    #[test]
    fn normal_descending_order_makes_highest_id_twin_the_only_actor() {
        let current = roles(&[
            (1, "Twin Minion"),
            (2, "Pooka"),
            (3, "Scout"),
            (4, "Twin Minion"),
        ]);
        let traces = enumerate_twin_traces(&current, &[4, 3, 2, 1], &[1, 2, 3, 4]);

        assert_eq!(traces.len(), 2);
        assert!(traces.iter().all(|trace| trace.actor_position == 4));
    }

    #[test]
    fn actor_selection_follows_current_collection_order_exactly() {
        let current = roles(&[
            (1, "Twin Minion"),
            (2, "Pooka"),
            (3, "Scout"),
            (4, "Twin Minion"),
        ]);
        let reordered = enumerate_twin_traces(&current, &[1, 2, 3, 4], &[1, 2, 3, 4]);
        assert_eq!(reordered.len(), 2);
        assert!(reordered.iter().all(|trace| trace.actor_position == 1));

        let actor_absent = enumerate_twin_traces(&current, &[3, 2], &[1, 2, 3, 4]);
        assert!(actor_absent.is_empty());
    }

    #[test]
    fn repeated_identical_demon_references_have_distinct_pool_indices() {
        let current = roles(&[(1, "Twin Minion"), (2, "Pooka"), (3, "Scout")]);
        let traces = enumerate_twin_traces(&current, &[3, 2, 2, 1], &[1, 2, 3]);

        assert_eq!(traces.len(), 4);
        let occurrence_indices: Vec<u8> = traces
            .iter()
            .map(|trace| match trace.outcome {
                TwinStartOutcome::Swap {
                    demon_occurrence_index,
                    ..
                } => demon_occurrence_index,
                TwinStartOutcome::NoDemon => unreachable!(),
            })
            .collect();
        assert_eq!(occurrence_indices, vec![0, 0, 1, 1]);
        assert_ne!(traces[0], traces[2]);
    }

    #[test]
    fn explicit_post_chancellor_map_controls_the_demon_anchor() {
        // The Demon data has already moved to physical position 2. Stable
        // origin information is intentionally outside this pure boundary.
        let current = roles(&[
            (1, "Twin Minion"),
            (2, "Lilis"),
            (3, "Chancellor"),
            (4, "Scout"),
        ]);
        let traces = enumerate_twin_traces(&current, &[4, 3, 2, 1], &[1, 2, 3, 4]);

        assert_eq!(traces.len(), 2);
        assert!(traces.iter().all(|trace| matches!(
            trace.outcome,
            TwinStartOutcome::Swap {
                demon_anchor_position: 2,
                ..
            }
        )));
    }

    #[test]
    fn exact_swap_endpoints_do_not_require_presentation_baselines() {
        let trace = TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 3,
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 2,
                neighbor_pre_swap_role: "Scout".to_string(),
            },
        };

        assert_eq!(
            current_data_after_twin_at(1, None, &trace).as_deref(),
            Some("Scout")
        );
        assert_eq!(
            current_data_after_twin_at(2, None, &trace).as_deref(),
            Some("Twin Minion")
        );
        assert_eq!(current_data_after_twin_at(4, None, &trace), None);
    }

    #[test]
    fn no_demon_and_self_swap_preserve_the_supplied_baseline() {
        let no_demon = TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::NoDemon,
        };
        assert_eq!(
            current_data_after_twin_at(1, Some("Twin Minion"), &no_demon).as_deref(),
            Some("Twin Minion")
        );

        let self_swap = TwinTrace {
            actor_position: 1,
            outcome: TwinStartOutcome::Swap {
                demon_occurrence_index: 0,
                demon_anchor_position: 2,
                neighbor_side: TwinNeighborSide::Previous,
                neighbor_position: 1,
                neighbor_pre_swap_role: "Twin Minion".to_string(),
            },
        };
        assert_eq!(
            current_data_after_twin_at(1, Some("Twin Minion"), &self_swap).as_deref(),
            Some("Twin Minion")
        );
        assert_eq!(current_data_after_twin_at(1, None, &self_swap), None);
    }

    #[test]
    fn trace_serde_preserves_occurrence_and_side() {
        let trace = TwinTrace {
            actor_position: 7,
            outcome: TwinStartOutcome::Swap {
                demon_occurrence_index: 2,
                demon_anchor_position: 3,
                neighbor_side: TwinNeighborSide::Next,
                neighbor_position: 4,
                neighbor_pre_swap_role: "Bombardier".to_string(),
            },
        };

        let json = serde_json::to_value(&trace).unwrap();
        assert_eq!(json["outcome"]["kind"], "swap");
        assert_eq!(json["outcome"]["demon_occurrence_index"], 2);
        assert_eq!(json["outcome"]["neighbor_side"], "next");
        assert_eq!(serde_json::from_value::<TwinTrace>(json).unwrap(), trace);
    }
}
