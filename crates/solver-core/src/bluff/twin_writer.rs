//! Offline projection of one entered Marionette.Act(Start) body.
//!
//! The caller establishes dispatch; this is not Character.Act and does not set
//! its latch. Stops after the two InitWithNoReset calls and immediate coroutine
//! role clones. No pending Reveal is resumed or ordered here. Native objects
//! must be valid, with no state-change subscribers or presentation-side writers.
use super::ledger::{LedgerError, Probability};
use super::reveal::{
    replay_reveal_callbacks, BluffReference, CallbackRole, DataRole, RevealContext, RoleSlot,
    REVEAL_CALLBACKS_START_NATIVE_V3,
};
use crate::knowledge_base::{get_card, Faction};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};

pub const TWIN_WRITER_NATIVE_V1: &str = "twin_start_writer_native_v1";
const DEAD: i32 = 20;
const HIDDEN: i32 = 5;

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct BodyState {
    pub state: i32,
    pub previous_state: i32,
    pub revealed: bool,
    pub killed_by_demon: bool,
    pub pickable_uses: i32,
    pub acted_info_count: u32,
    pub created_dead_presentation: bool,
    pub on_state_change_subscribed: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub struct TwinWriterContext {
    pub rule_version: String,
    /// V3 input, with explicit latches and no scheduled resume events.
    pub reveal: RevealContext,
    /// Exact global CurrentCharacters order, preserving repeated references.
    pub current_order: Vec<u8>,
    pub bodies: BTreeMap<u8, BodyState>,
    pub position: u8,
    /// The already-entered concrete Twin action may occupy either slot.
    pub copied_slot: bool,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct ReplacementTrace {
    pub position: u8,
    pub old_data: DataRole,
    pub new_data: DataRole,
    pub continuations_after: u16,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize)]
pub struct TwinWriterPath {
    pub probability: Probability,
    pub context: TwinWriterContext,
    pub slot: RoleSlot,
    pub demon_occurrence: Option<usize>,
    pub demon_position: Option<u8>,
    /// 0 = previous, 1 = next in the alive ring.
    pub neighbor_occurrence: Option<u8>,
    pub rng_draw_count: u8,
    pub replacements: Vec<ReplacementTrace>,
}

fn clone_role(data: DataRole) -> CallbackRole {
    match data {
        DataRole::Lilis => CallbackRole::Lilis,
        DataRole::TwinMinion => CallbackRole::TwinMinion,
        DataRole::Drunk => CallbackRole::Drunk,
        DataRole::Spy { .. } => CallbackRole::Spy,
    }
}

fn replace(
    context: &mut TwinWriterContext,
    position: u8,
    data: DataRole,
) -> Result<ReplacementTrace, LedgerError> {
    let actor = context
        .reveal
        .actors
        .iter_mut()
        .find(|a| a.position == position)
        .ok_or(LedgerError::InvalidContext)?;
    let old_data = actor.data_role;
    actor.character_start_acted = Some(false);
    actor.bluff = BluffReference::Null;
    actor.data_role = data;
    actor.action_role = clone_role(data);
    actor.remaining_continuations = actor
        .remaining_continuations
        .checked_add(1)
        .ok_or(LedgerError::Capacity)?;
    let body = context
        .bodies
        .get_mut(&position)
        .ok_or(LedgerError::InvalidContext)?;
    body.acted_info_count = 0;
    body.created_dead_presentation = false;
    body.revealed = false;
    body.killed_by_demon = false;
    body.pickable_uses = 1;
    body.previous_state = body.state;
    body.state = HIDDEN;
    Ok(ReplacementTrace {
        position,
        old_data,
        new_data: data,
        continuations_after: actor.remaining_continuations,
    })
}

pub(super) fn validate_board(context: &TwinWriterContext) -> Result<(), LedgerError> {
    if context.rule_version != TWIN_WRITER_NATIVE_V1
        || context.reveal.rule_version != REVEAL_CALLBACKS_START_NATIVE_V3
        || !context.reveal.resumes.is_empty()
        || context.current_order.len() > 256
        || context
            .bodies
            .values()
            .any(|b| b.on_state_change_subscribed)
    {
        return Err(LedgerError::InvalidContext);
    }
    // Shared validation of assets, explicit cache identity, statuses and latches.
    replay_reveal_callbacks(&context.reveal)?;
    let positions: BTreeSet<_> = context.reveal.actors.iter().map(|a| a.position).collect();
    if positions != context.bodies.keys().copied().collect()
        || positions != context.current_order.iter().copied().collect()
        || !positions.contains(&context.position)
    {
        return Err(LedgerError::InvalidContext);
    }
    Ok(())
}

pub(super) fn retained_entries(context: &TwinWriterContext) -> usize {
    let pools = &context.reveal.pools;
    context.current_order.len()
        + context.bodies.len() * 10
        + context.reveal.spy_caches.len()
        + [
            &pools.unique,
            &pools.duplicate,
            &pools.must_include,
            &pools.script.villagers,
            &pools.script.outcasts,
            &pools.script.minions,
            &pools.script.demons,
        ]
        .iter()
        .map(|p| p.len())
        .sum::<usize>()
        + context
            .reveal
            .actors
            .iter()
            .map(|a| 16 + a.statuses.values.len() + a.statuses.resistance.len())
            .sum::<usize>()
        + 2
}

pub fn replay_twin_start(context: &TwinWriterContext) -> Result<Vec<TwinWriterPath>, LedgerError> {
    validate_board(context)?;
    let source = context
        .reveal
        .actors
        .iter()
        .find(|a| a.position == context.position)
        .ok_or(LedgerError::InvalidContext)?;
    if (if context.copied_slot {
        source.bluff_role
    } else {
        Some(source.action_role)
    }) != Some(CallbackRole::TwinMinion)
    {
        return Err(LedgerError::InvalidContext);
    }
    let demons: Vec<_> = context
        .current_order
        .iter()
        .copied()
        .enumerate()
        .filter(|(_, position)| {
            let actor = context
                .reveal
                .actors
                .iter()
                .find(|a| a.position == *position)
                .unwrap();
            actor
                .register_as
                .as_ref()
                .map_or(actor.data_role == DataRole::Lilis, |role| {
                    get_card(role).unwrap().faction == Faction::Demon
                })
        })
        .collect();
    let entries = retained_entries(context);
    if entries
        .checked_mul((demons.len() * 2).max(1))
        .is_none_or(|n| n > 1_048_576)
    {
        return Err(LedgerError::Capacity);
    }
    let base = TwinWriterPath {
        probability: Probability {
            numerator: 1,
            denominator: 1,
        },
        context: context.clone(),
        slot: if context.copied_slot {
            RoleSlot::Bluff
        } else {
            RoleSlot::Real
        },
        demon_occurrence: None,
        demon_position: None,
        neighbor_occurrence: None,
        rng_draw_count: 0,
        replacements: vec![],
    };
    if demons.is_empty() {
        return Ok(vec![base]);
    }
    let alive: Vec<_> = context
        .current_order
        .iter()
        .copied()
        .filter(|p| context.bodies[p].state != DEAD)
        .collect();
    let mut paths = Vec::new();
    for (occurrence, demon) in &demons {
        // Native uses the first matching physical reference in the alive list.
        // A dead Demon has positive draw mass but no neighbor support: reject
        // the whole invocation, never condition on surviving branches.
        let index = alive
            .iter()
            .position(|p| p == demon)
            .ok_or(LedgerError::EmptySupport)?;
        let neighbors = [
            alive[(index + alive.len() - 1) % alive.len()],
            alive[(index + 1) % alive.len()],
        ];
        for (side, neighbor) in neighbors.into_iter().enumerate() {
            let mut path = base.clone();
            path.probability = path.probability.multiply(1, (demons.len() * 2) as u64)?;
            path.demon_occurrence = Some(*occurrence);
            path.demon_position = Some(*demon);
            path.neighbor_occurrence = Some(side as u8);
            path.rng_draw_count = 2;
            let saved = context
                .reveal
                .actors
                .iter()
                .find(|a| a.position == neighbor)
                .unwrap()
                .data_role;
            path.replacements
                .push(replace(&mut path.context, neighbor, source.data_role)?);
            path.replacements
                .push(replace(&mut path.context, context.position, saved)?);
            paths.push(path);
        }
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bluff::character_start::{
        replay_character_start, CharacterStartContext, CHARACTER_START_NATIVE_V1,
    };
    use crate::bluff::continuation_registry::{
        advance_ready_batch, ContinuationState, CONTINUATION_REGISTRY_NATIVE_V1,
    };
    use crate::bluff::ledger::{ScriptLists, SelectorPools};
    use crate::bluff::ready_batch::{
        replay_ready_batch, ReadyBatchContext, ReadyContinuation, READY_BATCH_NATIVE_V1,
    };
    use crate::bluff::reveal::{BluffRole, ResumeEvent, RevealActor, StatusState};
    use crate::bluff::reveal_writer::{
        replay_reveal_writers, RevealWriterContext, REVEAL_WRITER_NATIVE_V1,
    };

    fn input() -> TwinWriterContext {
        let actors = [DataRole::TwinMinion, DataRole::Lilis, DataRole::Drunk]
            .into_iter()
            .enumerate()
            .map(|(i, data)| RevealActor {
                position: (i + 1) as u8,
                data_role: data,
                action_role: clone_role(data),
                runtime_evil: i < 2,
                bluff: BluffReference::Live {
                    role: BluffRole::Witness,
                },
                bluff_role: Some(CallbackRole::Confessor),
                register_as: None,
                statuses: StatusState {
                    values: vec![30],
                    resistance: vec![10],
                    target_position: Some(2),
                },
                remaining_continuations: 1,
                on_trigger_subscribed: false,
                character_start_acted: Some(true),
            })
            .collect();
        TwinWriterContext {
            rule_version: TWIN_WRITER_NATIVE_V1.into(),
            position: 1,
            copied_slot: false,
            current_order: vec![1, 2, 3],
            bodies: (1..=3)
                .map(|p| {
                    (
                        p,
                        BodyState {
                            state: 10,
                            previous_state: 5,
                            revealed: true,
                            killed_by_demon: true,
                            pickable_uses: 4,
                            acted_info_count: 3,
                            created_dead_presentation: true,
                            on_state_change_subscribed: false,
                        },
                    )
                })
                .collect(),
            reveal: RevealContext {
                rule_version: REVEAL_CALLBACKS_START_NATIVE_V3.into(),
                board_size: 3,
                trailer_mode: false,
                actors,
                resumes: vec![],
                spy_caches: BTreeMap::new(),
                pools: SelectorPools {
                    unique: vec!["Witness".into()],
                    duplicate: vec!["Scout".into()],
                    must_include: vec![],
                    script: ScriptLists {
                        villagers: vec!["Witness".into()],
                        outcasts: vec!["Drunk".into()],
                        minions: vec!["Twin Minion".into()],
                        demons: vec!["Lilis".into()],
                    },
                },
            },
        }
    }

    fn start_input() -> CharacterStartContext {
        let mut board = input();
        board.reveal.actors[0].character_start_acted = Some(false);
        CharacterStartContext {
            rule_version: CHARACTER_START_NATIVE_V1.into(),
            board,
        }
    }

    fn writer_input() -> RevealWriterContext {
        RevealWriterContext {
            rule_version: REVEAL_WRITER_NATIVE_V1.into(),
            board: start_input().board,
            resumes: vec![ResumeEvent {
                position: 1,
                resume_ordinal: 0,
                acquisition_ordinal: None,
            }],
            ui: BTreeMap::new(),
        }
    }

    fn view_writer_input() -> RevealWriterContext {
        use crate::bluff::reveal_writer::{ViewUiState, REVEAL_WRITER_VIEW_NATIVE_V2};
        let mut context = writer_input();
        context.rule_version = REVEAL_WRITER_VIEW_NATIVE_V2.into();
        context.ui = (1..=3)
            .map(|p| {
                (
                    p,
                    ViewUiState {
                        pickable_active: true,
                        rip_active: true,
                        disguise_icon_active: Some(true),
                    },
                )
            })
            .collect();
        context
    }

    fn ready_input(positions: &[u8]) -> ReadyBatchContext {
        let mut initial = view_writer_input();
        initial.resumes.clear();
        ReadyBatchContext {
            rule_version: READY_BATCH_NATIVE_V1.into(),
            initial,
            ready: positions
                .iter()
                .enumerate()
                .map(|(i, p)| ReadyContinuation {
                    id: i as u16,
                    position: *p,
                })
                .collect(),
        }
    }

    fn registry_input() -> ContinuationState {
        ContinuationState {
            rule_version: CONTINUATION_REGISTRY_NATIVE_V1.into(),
            initial: ready_input(&[]).initial,
            pending: BTreeMap::from([(10, 1), (20, 2), (30, 3)]),
            next_id: 100,
            batch_ordinal: 0,
        }
    }

    #[test]
    fn registry_labels_ordered_writer_creations_and_preserves_unready_instances() {
        let result = advance_ready_batch(&registry_input(), &[10]).unwrap();
        assert_eq!(result[0].order, vec![10]);
        let path = &result[0].paths[1];
        assert_eq!(
            path.state.pending,
            BTreeMap::from([(20, 2), (30, 3), (100, 3), (101, 1)])
        );
        assert_eq!(
            path.created
                .iter()
                .map(|c| (c.logical_id, c.position, c.replacement_index))
                .collect::<Vec<_>>(),
            vec![(100, 3, 0), (101, 1, 1)]
        );
        assert_eq!(path.state.next_id, 102);
        assert_eq!(path.state.batch_ordinal, 1);
        assert_eq!(
            path.probability,
            Probability {
                numerator: 1,
                denominator: 2
            }
        );
    }

    #[test]
    fn registry_new_instance_can_resume_in_later_explicit_batch_only_once() {
        let first = advance_ready_batch(&registry_input(), &[10]).unwrap();
        let second = advance_ready_batch(&first[0].paths[1].state, &[101]).unwrap();
        assert_eq!(second[0].paths.len(), 1);
        let path = &second[0].paths[0];
        assert_eq!(
            path.state.pending,
            BTreeMap::from([(20, 2), (30, 3), (100, 3)])
        );
        assert_eq!(path.state.next_id, 102);
        assert_eq!(path.state.batch_ordinal, 2);
        assert!(path.created.is_empty());
        assert_eq!(path.trace[0].acquisition.event.acquisition_ordinal, Some(0));
        assert_eq!(
            path.probability,
            Probability {
                numerator: 1,
                denominator: 1
            }
        );
        assert_eq!(
            advance_ready_batch(&path.state, &[101]),
            Err(LedgerError::InvalidContext)
        );
        assert_eq!(
            advance_ready_batch(&path.state, &[10]),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn registry_self_swap_keeps_distinct_ids_on_same_body() {
        let first = advance_ready_batch(&registry_input(), &[10]).unwrap();
        let state = &first[0].paths[0].state;
        assert_eq!(state.pending[&100], 1);
        assert_eq!(state.pending[&101], 1);
        let second = advance_ready_batch(state, &[100, 101]).unwrap();
        assert_eq!(second.len(), 2);
        assert_eq!(second[0].order, vec![100, 101]);
        assert_eq!(second[1].order, vec![101, 100]);
        for schedule in second {
            for path in schedule.paths {
                assert!(!path.state.pending.contains_key(&100));
                assert!(!path.state.pending.contains_key(&101));
                assert!(path.created.iter().all(|c| c.logical_id >= 102));
            }
        }
    }

    #[test]
    fn registry_rejects_incomplete_counts_and_unknown_or_duplicate_ready_ids() {
        let mut state = registry_input();
        state.pending.remove(&30);
        assert_eq!(
            advance_ready_batch(&state, &[10]),
            Err(LedgerError::InvalidContext)
        );
        state = registry_input();
        state.pending.insert(40, 1);
        assert_eq!(
            advance_ready_batch(&state, &[]),
            Err(LedgerError::InvalidContext)
        );
        state = registry_input();
        state.next_id = 30;
        assert_eq!(
            advance_ready_batch(&state, &[]),
            Err(LedgerError::InvalidContext)
        );
        assert_eq!(
            advance_ready_batch(&registry_input(), &[99]),
            Err(LedgerError::InvalidContext)
        );
        assert_eq!(
            advance_ready_batch(&registry_input(), &[10, 10]),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn registry_allocation_and_batch_overflow_fail_without_partial_output() {
        let mut state = registry_input();
        state.next_id = u64::MAX;
        assert_eq!(
            advance_ready_batch(&state, &[10]),
            Err(LedgerError::Capacity)
        );
        state = registry_input();
        state.batch_ordinal = u16::MAX;
        assert_eq!(advance_ready_batch(&state, &[]), Err(LedgerError::Capacity));
    }

    #[test]
    fn registry_empty_batch_preserves_state_and_roundtrips_for_next_handoff() {
        let state = registry_input();
        let result = advance_ready_batch(&state, &[]).unwrap();
        let path = &result[0].paths[0];
        assert_eq!(path.state.pending, state.pending);
        assert_eq!(path.state.initial, state.initial);
        assert_eq!(path.state.next_id, state.next_id);
        assert!(path.created.is_empty());
        assert_eq!(path.state.batch_ordinal, 1);
        let json = serde_json::to_value(&path.state).unwrap();
        assert_eq!(
            serde_json::from_value::<ContinuationState>(json.clone()).unwrap(),
            path.state
        );
        let mut invalid = json;
        invalid["native_handle_guessed"] = true.into();
        assert!(serde_json::from_value::<ContinuationState>(invalid).is_err());
    }

    #[test]
    fn ready_batch_keeps_schedule_choices_separate_from_rng_weights() {
        let mut input = ready_input(&[1, 2]);
        input.initial.board.reveal.actors[0].character_start_acted = Some(true);
        let schedules = replay_ready_batch(&input).unwrap();
        assert_eq!(schedules.len(), 2);
        assert_eq!(schedules[0].order, vec![0, 1]);
        assert_eq!(schedules[1].order, vec![1, 0]);
        for schedule in &schedules {
            assert_eq!(schedule.paths.len(), 1);
            assert_eq!(
                schedule.paths[0].replay.probability,
                Probability {
                    numerator: 1,
                    denominator: 1
                }
            );
            assert!(schedule.paths[0].new_continuations.is_empty());
        }
        let json = serde_json::to_value(schedules).unwrap();
        assert!(json[0].get("probability").is_none());
    }

    #[test]
    fn ready_batch_defers_writer_created_continuations() {
        let schedules = replay_ready_batch(&ready_input(&[1])).unwrap();
        assert_eq!(schedules.len(), 1);
        assert_eq!(schedules[0].paths.len(), 2);
        let self_swap = &schedules[0].paths[0];
        assert_eq!(self_swap.new_continuations, BTreeMap::from([(1, 2)]));
        let distinct = &schedules[0].paths[1];
        assert_eq!(distinct.new_continuations, BTreeMap::from([(1, 1), (3, 1)]));
        for path in &schedules[0].paths {
            assert_eq!(path.replay.trace.len(), 1);
            assert_eq!(
                path.replay.probability,
                Probability {
                    numerator: 1,
                    denominator: 2
                }
            );
        }
    }

    #[test]
    fn ready_batch_derives_branch_local_acquisitions_without_discarding_paths() {
        let mut input = ready_input(&[1, 1, 1]);
        input.initial.board.reveal.actors[0].remaining_continuations = 3;
        let schedules = replay_ready_batch(&input).unwrap();
        assert_eq!(schedules.len(), 6);
        for schedule in schedules {
            assert!(schedule.paths.iter().any(|p| p.replay.trace[2]
                .acquisition
                .event
                .acquisition_ordinal
                .is_none()));
            assert!(schedule.paths.iter().any(|p| p.replay.trace[2]
                .acquisition
                .event
                .acquisition_ordinal
                == Some(2)));
            // All weights have denominators dividing 200 in this fixture.
            assert_eq!(
                schedule
                    .paths
                    .iter()
                    .map(|p| {
                        assert_eq!(200 % p.replay.probability.denominator, 0);
                        p.replay.probability.numerator * (200 / p.replay.probability.denominator)
                    })
                    .sum::<u64>(),
                200
            );
        }
    }

    #[test]
    fn ready_batch_matches_explicit_ordered_replay_when_flags_are_shared() {
        let input = ready_input(&[1]);
        let schedules = replay_ready_batch(&input).unwrap();
        let direct = replay_reveal_writers(&view_writer_input()).unwrap();
        assert_eq!(
            schedules[0]
                .paths
                .iter()
                .map(|p| p.replay.clone())
                .collect::<Vec<_>>(),
            direct
        );
    }

    #[test]
    fn ready_batch_rejects_missing_duplicate_and_excess_ready_provenance() {
        let mut input = ready_input(&[1, 1]);
        assert_eq!(replay_ready_batch(&input), Err(LedgerError::InvalidContext));
        input.initial.board.reveal.actors[0].remaining_continuations = 2;
        input.ready[1].id = input.ready[0].id;
        assert_eq!(replay_ready_batch(&input), Err(LedgerError::InvalidContext));
        input = ready_input(&[4]);
        assert_eq!(replay_ready_batch(&input), Err(LedgerError::InvalidContext));
        input = ready_input(&[1; 7]);
        assert_eq!(replay_ready_batch(&input), Err(LedgerError::InvalidContext));
        let mut json = serde_json::to_value(ready_input(&[])).unwrap();
        json["uniform_scheduler"] = true.into();
        assert!(serde_json::from_value::<ReadyBatchContext>(json).is_err());
    }

    #[test]
    fn ready_batch_empty_is_identity_and_failed_rng_branch_fails_exploration() {
        let input = ready_input(&[]);
        let schedules = replay_ready_batch(&input).unwrap();
        assert_eq!(schedules.len(), 1);
        assert!(schedules[0].order.is_empty());
        assert_eq!(schedules[0].paths[0].replay.board, input.initial.board);
        let mut input = ready_input(&[1]);
        input.initial.board.bodies.get_mut(&2).unwrap().state = 20;
        assert_eq!(replay_ready_batch(&input), Err(LedgerError::EmptySupport));
    }

    #[test]
    fn view_writer_swaps_update_both_endpoints_and_use_post_start_identity() {
        use crate::bluff::reveal_view::VisualSource;
        let input = view_writer_input();
        let paths = replay_reveal_writers(&input).unwrap();
        let path = &paths[1];
        for p in [1, 3] {
            assert!(!path.ui[&p].rip_active);
            assert_eq!(path.ui[&p].disguise_icon_active, Some(false));
            assert!(path.ui[&p].pickable_active);
            assert!(!path.board.bodies[&p].created_dead_presentation);
        }
        assert_eq!(path.ui[&2], input.ui[&2]);
        assert_eq!(
            path.trace[0]
                .replacement_views
                .iter()
                .map(|v| v.position)
                .collect::<Vec<_>>(),
            vec![3, 1]
        );
        let tail = path.trace[0].view.as_ref().unwrap();
        assert_eq!(tail.name_art_source, VisualSource::CurrentData);
        assert_eq!(tail.writes.len(), 4);
        assert_eq!(
            path.probability,
            Probability {
                numerator: 1,
                denominator: 2
            }
        );
    }

    #[test]
    fn view_writer_self_swap_destroys_death_presentation_only_once() {
        let path = replay_reveal_writers(&view_writer_input())
            .unwrap()
            .remove(0);
        let writes = &path.trace[0].replacement_views;
        assert_eq!(writes.len(), 2);
        assert_eq!(writes[0].rip_write, Some(false));
        assert_eq!(writes[1].rip_write, None);
        assert_eq!(path.board.reveal.actors[0].remaining_continuations, 2);
        assert_eq!(path.board.bodies[&1].previous_state, 5);
    }

    #[test]
    fn view_writer_retains_new_death_presentation_across_resumes() {
        use crate::bluff::reveal_view::ViewWrite;
        let mut input = view_writer_input();
        input.board.bodies.get_mut(&1).unwrap().state = 20;
        input
            .board
            .bodies
            .get_mut(&1)
            .unwrap()
            .created_dead_presentation = false;
        input.board.reveal.actors[0].character_start_acted = Some(true);
        input.board.reveal.actors[0].remaining_continuations = 2;
        input.ui.get_mut(&1).unwrap().rip_active = false;
        input.resumes.push(ResumeEvent {
            position: 1,
            resume_ordinal: 1,
            acquisition_ordinal: None,
        });
        let path = replay_reveal_writers(&input).unwrap().remove(0);
        assert!(path.ui[&1].rip_active);
        assert!(path.board.bodies[&1].created_dead_presentation);
        let created = |i: usize| {
            path.trace[i]
                .view
                .as_ref()
                .unwrap()
                .writes
                .iter()
                .filter(|w| {
                    matches!(
                        w,
                        ViewWrite::Refresh {
                            created_dead: true,
                            ..
                        }
                    )
                })
                .count()
        };
        assert_eq!(created(0), 1);
        assert_eq!(created(1), 0);
        assert_eq!(path.board.reveal.actors[0].remaining_continuations, 0);
    }

    #[test]
    fn view_writer_later_reacquisition_changes_tail_from_real_to_bluff() {
        use crate::bluff::reveal_view::VisualSource;
        let mut input = view_writer_input();
        input.resumes.push(ResumeEvent {
            position: 1,
            resume_ordinal: 1,
            acquisition_ordinal: Some(0),
        });
        let paths = replay_reveal_writers(&input).unwrap();
        let path = paths
            .iter()
            .find(|p| p.trace[1].acquisition.selector_status.is_some())
            .unwrap();
        assert_eq!(
            path.trace[0].view.as_ref().unwrap().name_art_source,
            VisualSource::CurrentData
        );
        assert_eq!(
            path.trace[1].view.as_ref().unwrap().name_art_source,
            VisualSource::RawBluff
        );
        assert_eq!(path.trace[1].view.as_ref().unwrap().writes.len(), 5);
        assert!(!path.ui[&1].rip_active);
        assert!(!path.ui[&3].rip_active);
        assert_eq!(paths.len(), 5);
    }

    #[test]
    fn view_writer_replacement_preserves_rip_without_live_death_object_and_absent_icon() {
        let mut input = view_writer_input();
        input
            .board
            .bodies
            .get_mut(&1)
            .unwrap()
            .created_dead_presentation = false;
        input.ui.get_mut(&1).unwrap().disguise_icon_active = None;
        let paths = replay_reveal_writers(&input).unwrap();
        for path in paths {
            assert!(path.ui[&1].rip_active);
            assert_eq!(path.ui[&1].disguise_icon_active, None);
            assert!(path.trace[0]
                .replacement_views
                .iter()
                .filter(|v| v.position == 1)
                .all(|v| v.rip_write.is_none() && v.disguise_write.is_none()));
        }
    }

    #[test]
    fn view_writer_requires_exact_ui_provenance_and_preserves_v1_shape() {
        use crate::bluff::reveal_writer::ViewUiState;
        let mut input = view_writer_input();
        input.ui.remove(&3);
        assert_eq!(
            replay_reveal_writers(&input),
            Err(LedgerError::InvalidContext)
        );
        input = view_writer_input();
        input.ui.insert(4, input.ui[&1].clone());
        assert_eq!(
            replay_reveal_writers(&input),
            Err(LedgerError::InvalidContext)
        );
        input = view_writer_input();
        input.rule_version = REVEAL_WRITER_NATIVE_V1.into();
        assert_eq!(
            replay_reveal_writers(&input),
            Err(LedgerError::InvalidContext)
        );
        let old = serde_json::to_value(replay_reveal_writers(&writer_input()).unwrap()).unwrap();
        assert!(old[0].get("ui").is_none());
        assert!(old[0]["trace"][0].get("view").is_none());
        assert!(old[0]["trace"][0].get("replacement_views").is_none());
        assert!(serde_json::from_str::<ViewUiState>(
            r#"{"pickable_active":false,"rip_active":false}"#
        )
        .is_err());
        assert!(serde_json::from_str::<ViewUiState>(
            r#"{"pickable_active":false,"rip_active":false,"disguise_icon_active":null}"#
        )
        .is_ok());
    }

    #[test]
    fn reveal_writer_acquires_once_before_swap_then_dispatches_new_role() {
        let mut input = writer_input();
        input.board.reveal.actors[0].bluff = BluffReference::Null;
        input.resumes[0].acquisition_ordinal = Some(0);
        let paths = replay_reveal_writers(&input).unwrap();
        assert_eq!(paths.len(), 4);
        for path in &paths {
            let trace = &path.trace[0];
            assert!(trace.acquisition.acquisition.is_some());
            assert!(trace.acquisition.callbacks.is_empty());
            assert_eq!(path.board.reveal.actors[0].bluff, BluffReference::Null);
            assert_eq!(
                path.board.reveal.actors[0].character_start_acted,
                Some(false)
            );
            assert_eq!(
                trace.callbacks[0].role,
                path.board.reveal.actors[0].action_role
            );
            assert!(trace
                .callbacks
                .iter()
                .all(|c| c.trigger != crate::bluff::reveal::Trigger::Start));
            assert_eq!(trace.start.as_ref().unwrap().callbacks.len(), 2);
        }
        assert_eq!(
            paths.iter().map(|p| p.probability).collect::<Vec<_>>(),
            vec![
                Probability {
                    numerator: 1,
                    denominator: 5
                },
                Probability {
                    numerator: 1,
                    denominator: 5
                },
                Probability {
                    numerator: 3,
                    denominator: 10
                },
                Probability {
                    numerator: 3,
                    denominator: 10
                }
            ]
        );
    }

    #[test]
    fn reveal_writer_later_resume_uses_new_data_and_created_continuation() {
        let mut input = writer_input();
        input.resumes.push(ResumeEvent {
            position: 1,
            resume_ordinal: 1,
            acquisition_ordinal: Some(0),
        });
        let paths = replay_reveal_writers(&input).unwrap();
        assert_eq!(paths.len(), 5);
        let drunk = paths
            .iter()
            .find(|p| p.trace[1].acquisition.selector_status.is_some())
            .unwrap();
        assert_eq!(
            drunk.probability,
            Probability {
                numerator: 1,
                denominator: 2
            }
        );
        assert_eq!(drunk.board.reveal.actors[0].data_role, DataRole::Drunk);
        assert_eq!(drunk.board.reveal.actors[0].remaining_continuations, 0);
        assert_eq!(
            drunk.board.reveal.actors[0].character_start_acted,
            Some(true)
        );
        assert_eq!(drunk.board.reveal.actors[2].remaining_continuations, 2);
        assert!(
            !drunk.trace[1]
                .acquisition
                .selector_status
                .as_ref()
                .unwrap()
                .accepted
        );
        assert_eq!(
            paths
                .iter()
                .map(|p| p.probability.numerator * (20 / p.probability.denominator))
                .sum::<u64>(),
            20
        );
    }

    #[test]
    fn reveal_writer_init_confessor_runs_after_swap_and_clears_target() {
        let path = replay_reveal_writers(&writer_input()).unwrap().remove(1);
        assert_eq!(path.trace[0].callbacks[0].role, CallbackRole::Drunk);
        assert_eq!(path.trace[0].callbacks[1].role, CallbackRole::Confessor);
        assert!(
            path.trace[0].callbacks[1]
                .status_application
                .as_ref()
                .unwrap()
                .inserted
        );
        assert_eq!(path.board.reveal.actors[0].statuses.target_position, None);
        assert_eq!(
            path.board.reveal.actors[2].statuses.target_position,
            Some(2)
        );
        assert_eq!(path.board.reveal.actors[0].remaining_continuations, 1);
    }

    #[test]
    fn reveal_writer_no_writer_matches_original_v3_callback_projection() {
        let mut input = writer_input();
        input.board.reveal.actors[0].action_role = CallbackRole::Drunk;
        input.board.reveal.actors[0].statuses.resistance.clear();
        let mut old_input = input.board.reveal.clone();
        old_input.resumes = input.resumes.clone();
        let old = replay_reveal_callbacks(&old_input).unwrap().remove(0);
        let new = replay_reveal_writers(&input).unwrap().remove(0);
        assert_eq!(new.board.reveal.actors, old.actors);
        assert_eq!(new.board.reveal.pools, old.pools);
        assert_eq!(new.probability, old.probability);
        let start = new.trace[0].start.as_ref().unwrap();
        assert_eq!(
            start.callbacks.len() + new.trace[0].callbacks.len(),
            old.trace[0].callbacks.len()
        );
        assert_eq!(new.trace[0].callbacks, old.trace[0].callbacks[2..]);
    }

    #[test]
    fn reveal_writer_spy_acquisition_replaces_unsupported_copied_slot_before_guard() {
        let mut input = writer_input();
        let actor = &mut input.board.reveal.actors[0];
        actor.data_role = DataRole::Spy { cache_key: 7 };
        actor.action_role = CallbackRole::Scout;
        actor.bluff_role = Some(CallbackRole::Spy);
        actor.bluff = BluffReference::Null;
        input
            .board
            .reveal
            .spy_caches
            .insert(7, BluffReference::Null);
        input.resumes[0].acquisition_ordinal = Some(0);
        let result = replay_reveal_writers(&input).unwrap();
        assert_eq!(result.len(), 1);
        let path = &result[0];
        assert!(path.trace[0].acquisition.spy_register_as.is_some());
        assert!(path.trace[0].acquisition.spy_acquisition.is_some());
        assert_eq!(
            path.trace[0].start.as_ref().unwrap().callbacks[1].role,
            CallbackRole::Witness
        );
        assert_eq!(
            path.board.reveal.actors[0].register_as.as_deref(),
            Some("Witness")
        );
    }

    #[test]
    fn reveal_writer_distinguishes_absent_start_from_latched_start() {
        let mut input = writer_input();
        input.board.reveal.actors[0].statuses.values.clear();
        let absent = replay_reveal_writers(&input).unwrap().remove(0);
        assert!(absent.trace[0].start.is_none());
        assert_eq!(
            absent.board.reveal.actors[0].character_start_acted,
            Some(false)
        );
        input.board.reveal.actors[0].statuses.values.push(30);
        input.board.reveal.actors[0].character_start_acted = Some(true);
        let latched = replay_reveal_writers(&input).unwrap().remove(0);
        let start = latched.trace[0].start.as_ref().unwrap();
        assert_eq!(start.initial_lying, None);
        assert!(start.callbacks.is_empty());
        assert_eq!(latched.trace[0].callbacks.len(), 4);
    }

    #[test]
    fn reveal_writer_branch_dependent_acquisition_provenance_fails_whole_replay() {
        let mut input = writer_input();
        input.resumes.push(ResumeEvent {
            position: 1,
            resume_ordinal: 1,
            acquisition_ordinal: Some(0),
        });
        // Distinct first swap -> Drunk leaves live bluff; self-swap branches
        // clear it again. Neither single acquisition assertion fits all paths.
        for acquisition in [None, Some(1)] {
            let mut fork = input.clone();
            fork.resumes.push(ResumeEvent {
                position: 1,
                resume_ordinal: 2,
                acquisition_ordinal: acquisition,
            });
            assert_eq!(
                replay_reveal_writers(&fork),
                Err(LedgerError::InvalidContext)
            );
        }
    }

    #[test]
    fn reveal_writer_validates_global_ordinals_and_pending_provenance() {
        let mut input = writer_input();
        input.resumes.push(input.resumes[0].clone());
        assert_eq!(
            replay_reveal_writers(&input),
            Err(LedgerError::InvalidContext)
        );
        input = writer_input();
        input.board.reveal.actors[0].remaining_continuations = 0;
        assert_eq!(
            replay_reveal_writers(&input),
            Err(LedgerError::InvalidContext)
        );
        input = writer_input();
        input.board.reveal.actors[0].bluff = BluffReference::Null;
        input.resumes = vec![
            ResumeEvent {
                position: 1,
                resume_ordinal: 0,
                acquisition_ordinal: Some(3),
            },
            ResumeEvent {
                position: 1,
                resume_ordinal: 1,
                acquisition_ordinal: Some(2),
            },
        ];
        assert_eq!(
            replay_reveal_writers(&input),
            Err(LedgerError::InvalidContext)
        );
        let mut json = serde_json::to_value(writer_input()).unwrap();
        json["scheduler_guessed"] = true.into();
        assert!(serde_json::from_value::<RevealWriterContext>(json).is_err());
    }

    #[test]
    fn character_start_latched_twin_skips_rng_and_subscribers_still_fail() {
        let mut input = start_input();
        input.board.reveal.actors[0].character_start_acted = Some(true);
        let result = replay_character_start(&input).unwrap();
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].board, input.board);
        assert_eq!(result[0].initial_lying, None);
        assert!(result[0].callbacks.is_empty());
        input.board.reveal.actors[0].on_trigger_subscribed = true;
        assert_eq!(
            replay_character_start(&input),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn character_start_swap_keeps_reset_latch_and_runs_stale_copied_role() {
        let result = replay_character_start(&start_input()).unwrap();
        assert_eq!(result.len(), 2);
        let branch = &result[1];
        assert_eq!(
            branch.board.reveal.actors[0].action_role,
            CallbackRole::Drunk
        );
        assert_eq!(
            branch.board.reveal.actors[0].character_start_acted,
            Some(false)
        );
        assert_eq!(branch.board.reveal.actors[0].bluff, BluffReference::Null);
        assert_eq!(
            branch.callbacks.iter().map(|c| c.role).collect::<Vec<_>>(),
            vec![CallbackRole::TwinMinion, CallbackRole::Confessor]
        );
        assert!(branch.callbacks[1].status_application.is_none()); // Confessor Start is inert.
        assert_eq!(branch.board.reveal.actors[0].remaining_continuations, 2);
    }

    #[test]
    fn character_start_both_twin_slots_swap_with_four_weighted_paths() {
        let mut input = start_input();
        input.board.reveal.actors[0].bluff_role = Some(CallbackRole::TwinMinion);
        let paths = replay_character_start(&input).unwrap();
        assert_eq!(paths.len(), 4);
        for path in &paths {
            assert_eq!(
                path.probability,
                Probability {
                    numerator: 1,
                    denominator: 4
                }
            );
            assert_eq!(path.callbacks.len(), 2);
            assert!(path
                .callbacks
                .iter()
                .all(|c| c.twin.as_ref().unwrap().rng_draw_count == 2));
            assert_eq!(
                path.board
                    .reveal
                    .actors
                    .iter()
                    .map(|a| u32::from(a.remaining_continuations))
                    .sum::<u32>(),
                7
            );
            assert_eq!(
                path.board.reveal.actors[0].character_start_acted,
                Some(false)
            );
        }
        // The second Twin callback swaps the source's NEW data, not the role
        // identity on the copied Twin object that is currently executing.
        let twice_distinct = &paths[3];
        assert_eq!(
            twice_distinct.callbacks[1]
                .twin
                .as_ref()
                .unwrap()
                .replacements[0]
                .new_data,
            DataRole::Drunk
        );
        assert_eq!(
            twice_distinct.board.reveal.actors[0].data_role,
            DataRole::TwinMinion
        );
        assert_eq!(paths[0].board.reveal.actors[0].remaining_continuations, 5);
    }

    #[test]
    fn character_start_truth_decision_survives_status_mutation_before_copied_twin() {
        let mut input = start_input();
        input.board.reveal.actors[0].action_role = CallbackRole::Drunk;
        input.board.reveal.actors[0].bluff_role = Some(CallbackRole::TwinMinion);
        input.board.reveal.actors[0].statuses.resistance.clear();
        let paths = replay_character_start(&input).unwrap();
        for path in paths {
            assert_eq!(path.initial_lying, Some(false));
            assert_eq!(
                path.callbacks[0].dispatch,
                crate::bluff::reveal::Dispatch::Act
            );
            assert_eq!(
                path.callbacks[1].dispatch,
                crate::bluff::reveal::Dispatch::Act
            );
            assert!(path.board.reveal.actors[0].is_lying());
            assert_eq!(
                path.callbacks[0]
                    .status_application
                    .as_ref()
                    .unwrap()
                    .target_after,
                Some(1)
            );
        }
    }

    #[test]
    fn character_start_lying_real_gate_uses_runtime_alignment_and_copied_pointer() {
        use crate::bluff::reveal::Dispatch;
        for evil in [false, true] {
            for copied in [false, true] {
                let mut input = start_input();
                input.board.reveal.actors[0].statuses.values = vec![10];
                input.board.reveal.actors[0].runtime_evil = evil;
                input.board.reveal.actors[0].bluff_role = copied.then_some(CallbackRole::Lilis);
                let paths = replay_character_start(&input).unwrap();
                for path in paths {
                    assert_eq!(
                        path.callbacks[0].dispatch,
                        if evil && copied {
                            Dispatch::Act
                        } else {
                            Dispatch::BluffAct
                        }
                    );
                    assert!(path.callbacks[0].twin.is_some()); // Base BluffAct forwards.
                    assert_eq!(path.callbacks.len(), if copied { 2 } else { 1 });
                    if copied {
                        assert_eq!(path.callbacks[1].dispatch, Dispatch::BluffAct);
                        assert_eq!(
                            path.callbacks[1]
                                .status_application
                                .as_ref()
                                .unwrap()
                                .status,
                            60
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn character_start_without_healthy_flag_still_runs_and_no_demon_keeps_latch() {
        let mut input = start_input();
        input.board.reveal.actors[0].statuses.values.clear();
        input.board.reveal.actors[1].register_as = Some("Scout".into());
        let path = replay_character_start(&input).unwrap().remove(0);
        assert_eq!(path.callbacks[0].twin.as_ref().unwrap().rng_draw_count, 0);
        assert_eq!(
            path.board.reveal.actors[0].character_start_acted,
            Some(true)
        );
        assert_eq!(path.board.reveal.actors[0].remaining_continuations, 1);
        // A second explicit Character.Act respects the latch left by the first.
        input.board = path.board;
        assert!(replay_character_start(&input).unwrap()[0]
            .callbacks
            .is_empty());
    }

    #[test]
    fn character_start_rejects_unsupported_second_slot_without_partial_branches() {
        let mut input = start_input();
        input.board.reveal.actors[0].bluff_role = Some(CallbackRole::Spy);
        assert_eq!(
            replay_character_start(&input),
            Err(LedgerError::InvalidContext)
        );
        input.board.reveal.actors[0].character_start_acted = Some(true);
        assert!(replay_character_start(&input).is_ok());
        input.board.copied_slot = true;
        assert_eq!(
            replay_character_start(&input),
            Err(LedgerError::InvalidContext)
        );
        input = start_input();
        input.rule_version = "unversioned".into();
        assert_eq!(
            replay_character_start(&input),
            Err(LedgerError::InvalidContext)
        );
    }

    #[test]
    fn character_start_second_swap_failure_rejects_whole_call() {
        let mut input = start_input();
        input.board.reveal.actors[0].bluff_role = Some(CallbackRole::TwinMinion);
        input.board.reveal.actors[0].remaining_continuations = u16::MAX - 2;
        assert_eq!(replay_character_start(&input), Err(LedgerError::Capacity));
        let mut json = serde_json::to_value(start_input()).unwrap();
        json["inferred_order"] = true.into();
        assert!(serde_json::from_value::<CharacterStartContext>(json).is_err());
    }

    #[test]
    fn distinct_swap_resets_only_reinitialized_storage() {
        let original = input();
        let paths = replay_twin_start(&original).unwrap();
        assert_eq!(paths.len(), 2);
        let path = &paths[1];
        assert_eq!(
            path.probability,
            Probability {
                numerator: 1,
                denominator: 2
            }
        );
        assert_eq!(
            path.replacements
                .iter()
                .map(|r| r.position)
                .collect::<Vec<_>>(),
            vec![3, 1]
        );
        assert_eq!(path.context.reveal.actors[0].data_role, DataRole::Drunk);
        assert_eq!(
            path.context.reveal.actors[2].action_role,
            CallbackRole::TwinMinion
        );
        for index in [0, 2] {
            let a = &path.context.reveal.actors[index];
            assert_eq!(a.runtime_evil, original.reveal.actors[index].runtime_evil);
            assert_eq!(a.statuses, original.reveal.actors[index].statuses);
            assert_eq!(a.bluff, BluffReference::Null);
            assert_eq!(a.bluff_role, Some(CallbackRole::Confessor));
            assert_eq!(a.character_start_acted, Some(false));
            assert_eq!(a.remaining_continuations, 2);
            let b = &path.context.bodies[&a.position];
            assert_eq!(
                (
                    b.state,
                    b.previous_state,
                    b.pickable_uses,
                    b.acted_info_count
                ),
                (5, 10, 1, 0)
            );
            assert!(!b.revealed && !b.killed_by_demon && !b.created_dead_presentation);
        }
        assert_eq!(path.context.reveal.pools, original.reveal.pools);
        assert_eq!(path.context.reveal.actors[1], original.reveal.actors[1]);
    }

    #[test]
    fn self_swap_runs_both_writes_and_retains_stale_registration() {
        let mut original = input();
        original.reveal.actors[0].register_as = Some("Scout".into());
        let path = replay_twin_start(&original).unwrap().remove(0);
        assert_eq!(
            path.replacements
                .iter()
                .map(|r| r.continuations_after)
                .collect::<Vec<_>>(),
            vec![2, 3]
        );
        assert_eq!(path.context.bodies[&1].previous_state, HIDDEN);
        assert_eq!(
            path.context.reveal.actors[0].register_as.as_deref(),
            Some("Scout")
        );
        assert_eq!(
            path.context.reveal.actors[0].data_role,
            DataRole::TwinMinion
        );
        assert_eq!(path.rng_draw_count, 2);
    }

    #[test]
    fn registered_demon_overrides_data_and_dead_mass_is_not_pruned() {
        let mut original = input();
        original.reveal.actors[1].register_as = Some("Scout".into());
        let noop = replay_twin_start(&original).unwrap().remove(0);
        assert_eq!(noop.context, original);
        assert_eq!(noop.rng_draw_count, 0);
        original.reveal.actors[2].register_as = Some("Lilis".into());
        assert!(replay_twin_start(&original)
            .unwrap()
            .iter()
            .all(|p| p.demon_position == Some(3)));
        original.reveal.actors[1].register_as = None;
        original.bodies.get_mut(&3).unwrap().state = DEAD;
        assert_eq!(replay_twin_start(&original), Err(LedgerError::EmptySupport));
    }

    #[test]
    fn duplicate_global_references_use_first_alive_index_and_keep_draw_mass() {
        let mut original = input();
        original.current_order = vec![2, 1, 2, 3];
        let paths = replay_twin_start(&original).unwrap();
        assert_eq!(paths.len(), 4);
        assert_eq!(
            paths
                .iter()
                .map(|p| p.replacements[0].position)
                .collect::<Vec<_>>(),
            vec![3, 1, 3, 1]
        );
        assert_eq!(
            paths.iter().map(|p| p.demon_occurrence).collect::<Vec<_>>(),
            vec![Some(0), Some(0), Some(2), Some(2)]
        );
        assert!(paths.iter().all(|p| p.probability
            == Probability {
                numerator: 1,
                denominator: 4
            }));
    }

    #[test]
    fn one_alive_card_keeps_two_identical_neighbor_occurrences() {
        let mut original = input();
        original.bodies.get_mut(&1).unwrap().state = DEAD;
        original.bodies.get_mut(&3).unwrap().state = DEAD;
        let paths = replay_twin_start(&original).unwrap();
        assert_eq!(paths.len(), 2);
        assert!(paths.iter().all(|p| p.replacements[0].position == 2));
        assert_eq!(paths[0].context.bodies[&1].previous_state, DEAD);
        assert_eq!(paths[0].context.bodies[&1].state, HIDDEN);
    }

    #[test]
    fn copied_twin_moves_actual_data_and_spy_cache_identity() {
        let mut original = input();
        original.copied_slot = true;
        original.reveal.actors[0].data_role = DataRole::Spy { cache_key: 9 };
        original.reveal.actors[0].action_role = CallbackRole::Scout;
        original.reveal.actors[0].bluff_role = Some(CallbackRole::TwinMinion);
        original.reveal.spy_caches.insert(
            9,
            BluffReference::Live {
                role: BluffRole::Scout,
            },
        );
        let path = replay_twin_start(&original).unwrap().remove(1);
        assert_eq!(
            path.context.reveal.actors[2].data_role,
            DataRole::Spy { cache_key: 9 }
        );
        assert_eq!(path.context.reveal.actors[2].action_role, CallbackRole::Spy);
        assert_eq!(path.context.reveal.spy_caches, original.reveal.spy_caches);
    }

    #[test]
    fn moved_drunk_can_resume_from_new_data_with_preserved_resistance() {
        let mut reveal = replay_twin_start(&input())
            .unwrap()
            .remove(1)
            .context
            .reveal;
        reveal.resumes = vec![ResumeEvent {
            position: 1,
            resume_ordinal: 0,
            acquisition_ordinal: Some(0),
        }];
        let path = replay_reveal_callbacks(&reveal).unwrap().remove(0);
        assert_eq!(path.actors[0].remaining_continuations, 1);
        assert_eq!(path.actors[0].character_start_acted, Some(true));
        assert!(!path.actors[0].statuses.values.contains(&10));
        assert_eq!(path.actors[2].remaining_continuations, 2);
    }

    #[test]
    fn provenance_and_capacity_fail_atomically() {
        let mut original = input();
        original.reveal.actors[0].remaining_continuations = u16::MAX - 1;
        assert_eq!(replay_twin_start(&original), Err(LedgerError::Capacity));
        original = input();
        original.current_order.pop();
        assert_eq!(
            replay_twin_start(&original),
            Err(LedgerError::InvalidContext)
        );
        original = input();
        original
            .bodies
            .get_mut(&1)
            .unwrap()
            .on_state_change_subscribed = true;
        assert_eq!(
            replay_twin_start(&original),
            Err(LedgerError::InvalidContext)
        );
        original = input();
        original.current_order = vec![2; 257];
        assert_eq!(
            replay_twin_start(&original),
            Err(LedgerError::InvalidContext)
        );
        let mut json = serde_json::to_value(input()).unwrap();
        json["unknown"] = true.into();
        assert!(serde_json::from_value::<TwinWriterContext>(json).is_err());
    }
}
