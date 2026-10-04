use super::*;
use crate::bluff::continuation_registry::CONTINUATION_REGISTRY_NATIVE_V1;
use crate::bluff::reveal::setup_reveal_tests;
use crate::bluff::reveal_writer::{RevealWriterContext, ViewUiState, REVEAL_WRITER_VIEW_NATIVE_V2};
use crate::bluff::twin_writer::{BodyState, TwinWriterContext, TWIN_WRITER_NATIVE_V1};
use crate::bluff::wait_eligibility::{
    make_wait_for_seconds, WaitForSecondsContext, UNITY_WAIT_ELIGIBILITY_NATIVE_V1,
};
use crate::bluff::wait_queue::{WaitQueueEntry, UNITY_WAIT_QUEUE_NATIVE_V1};

/// Authored supplied-clock case; not an original engine lifecycle schedule.
fn input() -> ScheduledRevealContext {
    let mut reveal = setup_reveal_tests::context();
    reveal.resumes.clear();
    let bodies = (1..=5)
        .map(|position| {
            (
                position,
                BodyState {
                    state: 5,
                    previous_state: 20,
                    revealed: false,
                    killed_by_demon: false,
                    pickable_uses: 1,
                    acted_info_count: 0,
                    created_dead_presentation: false,
                    on_state_change_subscribed: false,
                },
            )
        })
        .collect();
    let continuations = ContinuationState {
        rule_version: CONTINUATION_REGISTRY_NATIVE_V1.into(),
        initial: RevealWriterContext {
            rule_version: REVEAL_WRITER_VIEW_NATIVE_V2.into(),
            board: TwinWriterContext {
                rule_version: TWIN_WRITER_NATIVE_V1.into(),
                reveal,
                current_order: (1..=5).collect(),
                bodies,
                position: 1,
                copied_slot: false,
            },
            ui: (1..=5)
                .map(|position| {
                    (
                        position,
                        ViewUiState {
                            pickable_active: false,
                            rip_active: false,
                            disguise_icon_active: Some(false),
                        },
                    )
                })
                .collect(),
            resumes: vec![],
        },
        pending: (1..=5).map(|id| (id, id as u8)).collect(),
        next_id: 8,
        batch_ordinal: 0,
    };
    let card_time = 1.0 + f64::from(0.3_f32);
    ScheduledRevealContext {
        rule_version: SCHEDULED_SETUP_REVEAL_NATIVE_V2.into(),
        initial: ScheduledRevealState {
            rule_version: SCHEDULED_SETUP_REVEAL_NATIVE_V2.into(),
            continuations,
            deferred_waits: BTreeMap::from([
                (6, DeferredSetupWait::Audio),
                (7, DeferredSetupWait::Shuffle),
            ]),
            queue: WaitQueueState {
                rule_version: UNITY_WAIT_QUEUE_NATIVE_V1.into(),
                generation: 5,
                next_id: 8,
                entries: (1..=7)
                    .map(|logical_id| WaitQueueEntry {
                        logical_id,
                        timing: make_wait_for_seconds(&WaitForSecondsContext {
                            rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
                            duration: match logical_id {
                                6 => 0.4,
                                7 => 0.5,
                                _ => 0.3,
                            },
                            producer_time: 1.0,
                            producer_frame_counter: 7,
                            insertion_generation: 0,
                        })
                        .unwrap(),
                        release_present: true,
                    })
                    .collect(),
            },
        },
        dispatch: WaitDispatchContext {
            rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
            sampled_time: card_time,
            sampled_frame_counter: 13,
            phase_mask: 2,
            generation_before: 5,
        },
        callbacks: (1..=5)
            .map(|id| {
                (
                    id,
                    RevealCallbackBoundary {
                        same_live_owner: true,
                        callback_result: 1,
                        producer_time: card_time,
                        producer_frame_counter: 13,
                    },
                )
            })
            .collect(),
    }
}

#[test]
fn five_hidden_acquisitions_complete_and_both_auxiliary_waits_survive() {
    let context = input();
    let paths = replay_scheduled_reveal(&context).unwrap();
    assert_eq!(paths.len(), 6);
    for path in paths {
        assert_eq!(
            path.callbacks
                .iter()
                .map(|c| c.logical_id)
                .collect::<Vec<_>>(),
            [1, 2, 3, 4, 5]
        );
        assert!(path.state.continuations.pending.is_empty());
        assert_eq!(path.state.continuations.next_id, 8);
        assert_eq!(path.state.continuations.batch_ordinal, 5);
        assert_eq!(path.state.rule_version, SCHEDULED_SETUP_REVEAL_NATIVE_V2);
        assert_eq!(path.state.deferred_waits, context.initial.deferred_waits);
        assert_eq!(path.state.queue.entries, context.initial.queue.entries[5..]);
        assert_eq!(path.state.queue.generation, 6);
        assert_eq!(path.state.queue.next_id, 8);
        assert_eq!(
            path.state.continuations.initial.board.bodies,
            context.initial.continuations.initial.board.bodies
        );
        assert!(path
            .state
            .continuations
            .initial
            .board
            .reveal
            .actors
            .iter()
            .all(|a| a.remaining_continuations == 0));
        assert_eq!(
            path.queue_trace
                .iter()
                .filter(|e| matches!(e, WaitQueueEvent::Release { .. }))
                .count(),
            5
        );
        assert!(path
            .queue_trace
            .iter()
            .all(|e| !matches!(e, WaitQueueEvent::Callback { logical_id: 6 | 7 })));
    }
}

#[test]
fn crossing_audio_after_card_callbacks_rejects_the_whole_drain() {
    let mut context = input();
    context.dispatch.sampled_time = context.initial.queue.entries[5].timing.deadline;
    let before = context.clone();
    assert_eq!(
        replay_scheduled_reveal(&context),
        Err(LedgerError::InvalidContext)
    );
    assert_eq!(context, before);
    let successful = replay_scheduled_reveal(&input()).unwrap().remove(0);
    context.initial = successful.state;
    context.dispatch.generation_before = context.initial.queue.generation;
    context.callbacks.clear();
    assert_eq!(
        replay_scheduled_reveal(&context),
        Err(LedgerError::InvalidContext)
    );
}

#[test]
fn complete_union_owner_release_phase_and_typed_wait_guards_fail_closed() {
    for mutation in 0..9 {
        let mut context = input();
        match mutation {
            0 => {
                context.initial.queue.entries.pop();
            }
            1 => {
                context
                    .initial
                    .deferred_waits
                    .insert(7, DeferredSetupWait::Audio);
            }
            2 => {
                context
                    .initial
                    .deferred_waits
                    .insert(1, DeferredSetupWait::Audio);
            }
            3 => context.initial.queue.entries[5].release_present = false,
            4 => context.initial.queue.entries[6].timing.phase_mask = 2,
            5 => {
                context
                    .initial
                    .continuations
                    .initial
                    .board
                    .bodies
                    .get_mut(&1)
                    .unwrap()
                    .state = 30
            }
            6 => {
                context
                    .initial
                    .continuations
                    .initial
                    .board
                    .bodies
                    .get_mut(&1)
                    .unwrap()
                    .revealed = true
            }
            7 => context.callbacks.get_mut(&1).unwrap().same_live_owner = false,
            8 => context.callbacks.get_mut(&1).unwrap().callback_result = 0,
            _ => unreachable!(),
        }
        let before = context.clone();
        assert_eq!(
            replay_scheduled_reveal(&context),
            Err(LedgerError::InvalidContext)
        );
        assert_eq!(context, before);
    }
}

#[test]
fn pre_deadline_or_wrong_phase_drain_preserves_every_wait_without_callbacks() {
    for phase_gate in [false, true] {
        let mut context = input();
        if phase_gate {
            context.dispatch.phase_mask = 1;
        } else {
            context.dispatch.sampled_time = 1.3;
        }
        context.callbacks.clear();
        let paths = replay_scheduled_reveal(&context).unwrap();
        assert_eq!(paths.len(), 1);
        assert!(paths[0].callbacks.is_empty());
        assert_eq!(paths[0].state.queue.entries, context.initial.queue.entries);
        assert_eq!(paths[0].state.continuations, context.initial.continuations);
        assert_eq!(
            paths[0].state.deferred_waits,
            context.initial.deferred_waits
        );
    }
}

#[test]
fn v1_keeps_complete_delay_reveal_only_contract_and_old_serialization_shape() {
    let mut context = input();
    context.rule_version = SCHEDULED_REVEAL_NATIVE_V1.into();
    context.initial.rule_version = SCHEDULED_REVEAL_NATIVE_V1.into();
    assert_eq!(
        replay_scheduled_reveal(&context),
        Err(LedgerError::InvalidContext)
    );
    context.initial.deferred_waits.clear();
    context.initial.queue.entries.truncate(5);
    assert_eq!(
        replay_scheduled_reveal(&context),
        Err(LedgerError::InvalidContext)
    );
    let serialized = serde_json::to_value(&context.initial).unwrap();
    assert!(serialized.get("deferred_waits").is_none());
    assert_eq!(
        serde_json::from_value::<ScheduledRevealState>(serialized).unwrap(),
        context.initial
    );
}

#[test]
fn shuffle_reached_while_audio_is_frame_skipped_still_rejects_the_whole_drain() {
    let mut context = input();
    context.dispatch.sampled_time = context.initial.queue.entries[6].timing.deadline;
    context.initial.queue.entries[5].timing.frame_threshold = 14;
    let before = context.clone();
    assert_eq!(
        replay_scheduled_reveal(&context),
        Err(LedgerError::InvalidContext)
    );
    assert_eq!(context, before);
}
