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

#[test]
fn retained_original_n5_acquisition_matches_independent_native_projection() {
    use crate::bluff::reveal::BluffReference;
    use crate::bluff::setup_action_bridge;
    use crate::bluff::wait_queue::{replay_wait_queue, WaitQueueContext};
    use crate::types::BluffAcquisitionSource;

    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/fixtures/synthetic/first_village_retained_acquisition_v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(fixture["build_id"], "f530404b0f3f_807de4a83df4");
    assert_eq!(
        fixture["native_report_sha256"],
        "f74f10f6f9eccd5ae1af512871dc01e5c173e590290a414b59a11b74122ce9f3"
    );
    assert_eq!(
        fixture["native_source_sha256"],
        "cdcc5ef15c2d36a1b961263c483b27c264e520a495f97985ad640d68df4c68cd"
    );

    let setup: setup_action_bridge::Context =
        serde_json::from_value(fixture["setup_context"].clone()).unwrap();
    let setup_paths = setup_action_bridge::replay(&setup).unwrap();
    assert_eq!(setup_paths.len(), 1);
    let setup_path = &setup_paths[0];
    let setup_board = &setup_path.state.initial.board;
    let expected_setup = &fixture["expected_setup"];
    for (actual, field) in [
        (
            serde_json::to_value(&setup_board.reveal.actors).unwrap(),
            "actors",
        ),
        (serde_json::to_value(&setup_board.bodies).unwrap(), "bodies"),
        (
            serde_json::to_value(&setup_board.reveal.pools).unwrap(),
            "pools",
        ),
        (
            serde_json::to_value(&setup_board.current_order).unwrap(),
            "current_order",
        ),
        (
            serde_json::to_value(&setup_path.current_data).unwrap(),
            "current_data",
        ),
        (
            serde_json::to_value(&setup_path.state.pending).unwrap(),
            "pending",
        ),
    ] {
        assert_eq!(&actual, &expected_setup[field], "native setup {field}");
    }

    let context: ScheduledRevealContext =
        serde_json::from_value(fixture["context"].clone()).unwrap();
    let registry = &context.initial.continuations;
    let board = &registry.initial.board;
    assert_eq!(board.reveal.actors, setup_board.reveal.actors);
    assert_eq!(board.reveal.pools, setup_board.reveal.pools);
    assert_eq!(board.bodies, setup_board.bodies);
    assert_eq!(board.current_order, setup_board.current_order);
    assert_eq!(registry.initial.ui, setup_path.state.initial.ui);
    assert_eq!(board.reveal.rule_version, SETUP_REVEAL_CALLBACKS_NATIVE_V5);
    assert_eq!(context.initial.queue.generation, 8);
    assert_eq!(context.initial.queue.next_id, 12);
    assert_eq!(registry.next_id, 12);
    assert_eq!(registry.batch_ordinal, 0);
    let mapping = fixture["native_logical_map"].as_array().unwrap();
    assert_eq!(mapping.len(), 5);
    let initialized =
        crate::bluff::setup_initialization_batch::replay(&setup.initialization).unwrap();
    for row in mapping {
        let id = row["logical_id"].as_u64().unwrap();
        let position = row["position"].as_u64().unwrap() as u8;
        let native_iterator = row["native_iterator"].as_u64().unwrap();
        assert_eq!(registry.pending[&id], position);
        assert_eq!(setup_path.state.pending[&native_iterator], position);
        let publication = initialized
            .publications
            .iter()
            .find(|publication| publication.continuation_identity == native_iterator)
            .unwrap();
        assert_eq!(publication.position, position);
        assert_eq!(row["native_display_id"], publication.display_id);
    }

    // The callback effects here are supplied native-derived queue responses.
    // The full animation/tween graph remains the native witness's domain.
    let drains = fixture["native_only"]["preceding_drains"]
        .as_array()
        .unwrap();
    assert_eq!(drains.len(), 9);
    let mut previous = None;
    for drain in &drains[..8] {
        let initial: WaitQueueState = serde_json::from_value(drain["before"].clone()).unwrap();
        if let Some(after) = previous.take() {
            assert_eq!(initial, after);
        }
        let dispatch = WaitDispatchContext {
            rule_version: UNITY_WAIT_ELIGIBILITY_NATIVE_V1.into(),
            sampled_time: drain["input"]["time"].as_f64().unwrap(),
            sampled_frame_counter: drain["input"]["frame"].as_i64().unwrap(),
            phase_mask: drain["input"]["phase"].as_u64().unwrap() as u32,
            generation_before: drain["input"]["generation_before"].as_u64().unwrap() as u32,
        };
        let result = replay_wait_queue(&WaitQueueContext {
            rule_version: UNITY_WAIT_QUEUE_NATIVE_V1.into(),
            initial,
            dispatch,
            responses: serde_json::from_value(drain["responses"].clone()).unwrap(),
        })
        .unwrap();
        assert_eq!(serde_json::to_value(&result.state).unwrap(), drain["after"]);
        for (actual, field) in [
            (
                result
                    .trace
                    .iter()
                    .filter_map(|e| match e {
                        WaitQueueEvent::Visit { logical_id, .. } => Some(*logical_id),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                "visited_ids",
            ),
            (
                result
                    .trace
                    .iter()
                    .filter_map(|e| match e {
                        WaitQueueEvent::Callback { logical_id } => Some(*logical_id),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                "callback_ids",
            ),
            (
                result
                    .trace
                    .iter()
                    .filter_map(|e| match e {
                        WaitQueueEvent::Insert { entry } => Some(entry.logical_id),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                "inserted_ids",
            ),
            (
                result
                    .trace
                    .iter()
                    .filter_map(|e| match e {
                        WaitQueueEvent::Release { logical_id } => Some(*logical_id),
                        _ => None,
                    })
                    .collect::<Vec<_>>(),
                "released_ids",
            ),
        ] {
            assert_eq!(
                serde_json::to_value(actual).unwrap(),
                drain[field],
                "native animation {field}"
            );
        }
        previous = Some(result.state);
    }
    assert_eq!(previous.unwrap(), context.initial.queue);

    let paths = replay_scheduled_reveal(&context).unwrap();
    assert_eq!(paths.len(), 6);
    let support: BTreeMap<_, _> = paths
        .iter()
        .map(|path| {
            let role = &path.callbacks[0].replay.trace[0]
                .acquisition
                .acquisition
                .as_ref()
                .unwrap()
                .bluff_role;
            (
                role.clone(),
                (path.probability.numerator, path.probability.denominator),
            )
        })
        .collect();
    assert_eq!(
        support,
        BTreeMap::from([
            ("Confessor".into(), (1, 10)),
            ("Lover".into(), (1, 10)),
            ("Hunter".into(), (1, 10)),
            ("Enlightened".into(), (1, 10)),
            ("Gemcrafter".into(), (3, 10)),
            ("Alchemist".into(), (3, 10)),
        ])
    );
    let path = paths
        .iter()
        .find(|path| {
            path.state.continuations.initial.board.reveal.actors[0].bluff
                == (BluffReference::Live {
                    role: crate::bluff::reveal::BluffRole::Confessor,
                })
        })
        .unwrap();
    let expected = &fixture["expected"];
    let final_board = &path.state.continuations.initial.board;
    for (actual, field) in [
        (
            serde_json::to_value(&final_board.reveal.actors).unwrap(),
            "actors",
        ),
        (serde_json::to_value(&final_board.bodies).unwrap(), "bodies"),
        (
            serde_json::to_value(&final_board.reveal.pools).unwrap(),
            "pools",
        ),
        (
            serde_json::to_value(&final_board.current_order).unwrap(),
            "current_order",
        ),
        (
            serde_json::to_value(&path.state.continuations.initial.ui).unwrap(),
            "ui",
        ),
        (
            serde_json::to_value(&path.state.continuations.pending).unwrap(),
            "pending",
        ),
        (serde_json::to_value(&path.state.queue).unwrap(), "queue"),
        (
            serde_json::to_value(&path.state.deferred_waits).unwrap(),
            "deferred_waits",
        ),
    ] {
        assert_eq!(&actual, &expected[field], "native acquisition {field}");
    }
    assert_eq!(expected["next_id"], path.state.continuations.next_id);
    assert_eq!(
        expected["batch_ordinal"],
        path.state.continuations.batch_ordinal
    );
    let callback_order: Vec<_> = path.callbacks.iter().map(|c| c.logical_id).collect();
    assert_eq!(
        serde_json::to_value(&callback_order).unwrap(),
        expected["callback_order"]
    );
    for (index, callback) in path.callbacks.iter().enumerate() {
        assert_eq!(callback.replay.trace.len(), 1);
        assert!(callback.replay.created.is_empty());
        let trace = &callback.replay.trace[0];
        let native = &expected["callbacks"][index];
        assert!(trace.start.is_none());
        assert_eq!(native["logical_id"], callback.logical_id);
        assert_eq!(native["position"], trace.acquisition.event.position);
        assert_eq!(
            serde_json::to_value(&trace.callbacks).unwrap(),
            native["callbacks"]
        );
        assert_eq!(trace.acquisition.event.acquisition_ordinal, Some(0));
        if index == 0 {
            let draw = trace.acquisition.acquisition.as_ref().unwrap();
            assert_eq!(draw.rng_draw_count, 2);
            assert_eq!(
                draw.source,
                BluffAcquisitionSource::DuplicatePool {
                    occurrence_index: 0
                }
            );
            assert!(!draw.script_added);
        } else {
            assert!(trace.acquisition.acquisition.is_none());
        }
        let position = trace.acquisition.event.position;
        let versions = &expected["status_versions"][position.to_string()];
        let (identity, _) = initialized
            .positions
            .iter()
            .find(|(_, p)| **p == position)
            .unwrap();
        let setup_inserted = setup_path
            .calls
            .iter()
            .filter(|call| call.position == position)
            .flat_map(|call| &call.init_callbacks)
            .filter(|call| {
                call.status_application
                    .as_ref()
                    .is_some_and(|effect| effect.inserted)
            })
            .count() as u64;
        let before_version =
            u64::from(initialized.actors[identity].statuses.version) + setup_inserted;
        assert_eq!(versions["before"], before_version);
        assert_eq!(
            expected_setup["status_versions"][position.to_string()]["after"],
            before_version
        );
        let inserted = trace
            .callbacks
            .iter()
            .filter(|c| {
                c.status_application
                    .as_ref()
                    .is_some_and(|effect| effect.inserted)
            })
            .count() as u64;
        assert_eq!(
            versions["after"].as_u64().unwrap(),
            versions["before"].as_u64().unwrap() + inserted
        );
    }
    for (actual, field) in [
        (
            path.queue_trace
                .iter()
                .filter_map(|e| match e {
                    WaitQueueEvent::Visit { logical_id, .. } => Some(*logical_id),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            "queue_visit_order",
        ),
        (
            path.queue_trace
                .iter()
                .filter_map(|e| match e {
                    WaitQueueEvent::Erase { logical_id } => Some(*logical_id),
                    _ => None,
                })
                .collect::<Vec<_>>(),
            "queue_erase_order",
        ),
    ] {
        assert_eq!(serde_json::to_value(actual).unwrap(), expected[field]);
    }
    let released: Vec<_> = path
        .queue_trace
        .iter()
        .filter_map(|e| match e {
            WaitQueueEvent::Release { logical_id } => Some(*logical_id),
            _ => None,
        })
        .collect();
    assert_eq!(released, callback_order);
}
