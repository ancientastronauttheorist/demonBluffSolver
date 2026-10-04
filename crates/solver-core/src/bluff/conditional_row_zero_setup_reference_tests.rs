//! Independent support reference for one explicitly supplied row-zero input.
//!
//! Only fixture inputs are deserialized. Expected transitions below are authored
//! from Character.Init, ordinary Minion selection/registration and the guarded
//! non-Day callbacks; no production transition helper constructs expectations.
//! This is not a generative world set, player-history domain, RNG prior or UI
//! capture certificate. The five unrecorded copy outcomes are authored support.

use crate::bluff::{
    character_initialization::{Actor, Continuation, Event},
    continuation_registry::{ContinuationState, CONTINUATION_REGISTRY_NATIVE_V1},
    ledger::{LedgerError, Selector, SelectorEvent, SelectorTrace},
    reveal::{
        BluffReference, BluffRole, CallbackRole, CallbackTrace, DataRole, Dispatch, ResumeEvent,
        ResumeTrace, RevealActor, RevealContext, RoleSlot, StatusApplication, StatusState, Trigger,
        SETUP_CALLBACKS_NATIVE_V4, SETUP_REVEAL_CALLBACKS_NATIVE_V5,
    },
    reveal_view::{ViewWrite, VisualSource},
    reveal_writer::{
        RevealTailTrace, RevealWriterContext, WriterResumeTrace, REVEAL_WRITER_VIEW_NATIVE_V2,
    },
    scheduled_reveal::{
        self, DeferredSetupWait, ScheduledRevealContext, ScheduledRevealPath, ScheduledRevealState,
        SCHEDULED_SETUP_REVEAL_NATIVE_V2,
    },
    setup_action_bridge::{self, ActionCall, SETUP_ACTION_BRIDGE_NATIVE_V2},
    setup_initialization_batch::{self, Publication, Replay as Initialized},
    twin_writer::{BodyState, TwinWriterContext, TWIN_WRITER_NATIVE_V1},
    wait_eligibility::WaitEligibility,
    wait_queue::WaitQueueEvent,
};
use crate::types::BluffAcquisitionSource;
use serde::Deserialize;
use serde_json::{json, Value};
use std::collections::{BTreeMap, BTreeSet};

#[derive(Deserialize)]
struct Inputs {
    schema_version: u32,
    build_id: String,
    native_report_sha256: String,
    native_source_sha256: String,
    setup_context: setup_action_bridge::Context,
    context: ScheduledRevealContext,
    native_logical_map: Vec<Mapping>,
    // Intentionally no expected, expected_setup, or native_only output fields.
}

#[derive(Deserialize)]
struct Mapping {
    logical_id: u64,
    native_iterator: u64,
    native_actor: u64,
    position: u8,
    native_display_id: i32,
}

const ROW: [(DataRole, CallbackRole, &str, &str); 5] = [
    (
        DataRole::Minion,
        CallbackRole::Minion,
        "Minion",
        "data:21596",
    ),
    (
        DataRole::Confessor,
        CallbackRole::Confessor,
        "Confessor",
        "data:21614",
    ),
    (DataRole::Lover, CallbackRole::Lover, "Empath", "data:21626"),
    (
        DataRole::Hunter,
        CallbackRole::Hunter,
        "Tracker",
        "data:21621",
    ),
    (
        DataRole::Enlightened,
        CallbackRole::Enlightened,
        "Shugenja",
        "data:21618",
    ),
];

const COPIES: [(BluffRole, CallbackRole, &str); 6] = [
    (BluffRole::Confessor, CallbackRole::Confessor, "Confessor"),
    (BluffRole::Lover, CallbackRole::Lover, "Lover"),
    (BluffRole::Hunter, CallbackRole::Hunter, "Hunter"),
    (
        BluffRole::Enlightened,
        CallbackRole::Enlightened,
        "Enlightened",
    ),
    (
        BluffRole::Gemcrafter,
        CallbackRole::Gemcrafter,
        "Gemcrafter",
    ),
    (BluffRole::Alchemist, CallbackRole::Alchemist, "Alchemist"),
];

fn inputs() -> Inputs {
    let input: Inputs = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/fixtures/synthetic/first_village_retained_acquisition_v1.json"
    )))
    .unwrap();
    assert_eq!(input.schema_version, 1);
    assert_eq!(input.build_id, "f530404b0f3f_807de4a83df4");
    assert_eq!(
        input.native_report_sha256,
        "f74f10f6f9eccd5ae1af512871dc01e5c173e590290a414b59a11b74122ce9f3"
    );
    assert_eq!(
        input.native_source_sha256,
        "cdcc5ef15c2d36a1b961263c483b27c264e520a495f97985ad640d68df4c68cd"
    );
    input
}

fn names(values: &[&str]) -> Vec<String> {
    values.iter().map(|s| (*s).into()).collect()
}

fn assert_fixed_input(input: &Inputs) {
    let setup = &input.setup_context;
    let init = &setup.initialization;
    let caller = &init.caller;
    assert_eq!(setup.version, SETUP_ACTION_BRIDGE_NATIVE_V2);
    assert_eq!(
        caller.roster,
        Some(ROW.map(|r| Some(r.3.to_string())).to_vec())
    );
    let roster = caller.roster.as_ref().unwrap();
    let board = caller.state.board.as_ref().unwrap();
    let board_list = &caller.state.lists[board];
    assert_eq!(board_list.count, 5);
    assert_eq!(board_list.items.len(), 5);
    assert_eq!(init.actors.len(), 5);
    assert!(init.continuations.is_empty());
    assert!(init.raw_bluff_liveness.is_empty());
    assert_eq!(init.allocations.len(), 5);
    assert_eq!(input.native_logical_map.len(), 5);
    assert_eq!(
        setup.pools.duplicate,
        names(&["Confessor", "Lover", "Hunter", "Enlightened"])
    );
    assert_eq!(setup.pools.unique, names(&["Gemcrafter", "Alchemist"]));
    assert!(setup.pools.must_include.is_empty());
    assert_eq!(setup.pools.script.villagers, setup.pools.duplicate);
    assert_eq!(setup.pools.script.minions, names(&["Minion"]));
    assert!(setup.pools.script.outcasts.is_empty());
    assert!(setup.pools.script.demons.is_empty());
    assert!(setup.spy_caches.is_empty());
    assert!(setup.action_classes_and_caches_verified && setup.on_trigger_absent);
    assert!(setup.final_services_inert && setup.post_initialization_ui_verified);
    assert!(init.required_objects_and_lists_valid && init.callbacks_and_ui_inert);
    assert!(init.clone_results_verified && init.synchronous_first_yield_verified);
    assert!(caller.failure.is_none() && caller.after.is_empty());

    // This is the actual supplied 15-entry array, not None or an authored empty
    // array. None remains an unsupported full-bridge input.
    let order_name = caller.state.order.as_ref().unwrap();
    assert_eq!(order_name, "original_start_order");
    let order = &caller.state.arrays[order_name];
    let start_aliases = [
        "data:21594",
        "data:21593",
        "data:21597",
        "data:21605",
        "data:21602",
        "data:21595",
        "data:21599",
        "data:21590",
        "data:21606",
        "data:21600",
        "data:21609",
        "data:21634",
        "data:21598",
        "data:21607",
        "data:21591",
    ];
    assert_eq!(order.length, 15);
    assert_eq!(order.items, start_aliases.map(|s| Some(s.into())).to_vec());
    assert!(start_aliases
        .iter()
        .all(|a| !roster.iter().any(|r| r.as_deref() == Some(*a))));

    for (index, (actor_alias, role)) in board_list.items.iter().zip(ROW).enumerate() {
        let actor_id = init.object_identities[actor_alias.as_ref().unwrap()];
        let actor = &init.actors[&actor_id];
        let data_id = init.object_identities[role.3];
        let source = init.data[&data_id].source_role.unwrap();
        let clone = init.allocations[index].clone.as_ref().unwrap();
        assert_eq!(init.positions[&actor_id], index as u8 + 1);
        assert_eq!(setup.data_roles[&data_id], role.0);
        assert_eq!(clone.source_role, source);
        assert_ne!(clone.identity, source);
        assert_eq!(clone.managed_class, role.2);
        assert_eq!(setup.action_classes[&clone.identity], role.2);
        let source_alias = caller.state.data_roles[role.3].as_ref().unwrap();
        assert_eq!(init.object_identities[source_alias], source);
        assert_eq!(caller.roles[source_alias], role.2);
        assert_eq!(
            init.data[&data_id].starting_alignment,
            if index == 0 { 20 } else { 10 }
        );
        assert_eq!(actor.state, 20);
        assert!(!actor.revealed && !actor.started && !actor.killed_hidden && !actor.killed_demon);
        assert!(actor.dead_prefab.is_none() && actor.state_callback.is_none());
        assert!(actor.bluff_role.is_none() && actor.statuses.active.is_empty());
        assert!(actor.statuses.resistances.is_empty() && actor.statuses.target.is_none());
        assert_eq!(actor.statuses.version, 0);
        assert_eq!(actor.info_version, 0);
        let mapping = &input.native_logical_map[index];
        assert_eq!(mapping.logical_id, index as u64);
        assert_eq!(mapping.position, index as u8 + 1);
        assert_eq!(mapping.native_actor, actor_id);
        assert_eq!(
            mapping.native_iterator,
            init.allocations[index].continuation_identity
        );
        assert_eq!(mapping.native_display_id, 5 - index as i32);
    }
    let scheduled = &input.context;
    assert_eq!(scheduled.rule_version, SCHEDULED_SETUP_REVEAL_NATIVE_V2);
    assert_eq!(
        scheduled.initial.rule_version,
        SCHEDULED_SETUP_REVEAL_NATIVE_V2
    );
    assert_eq!(scheduled.initial.queue.generation, 8);
    assert_eq!(scheduled.initial.queue.next_id, 12);
    assert_eq!(scheduled.initial.queue.entries.len(), 7);
    for (entry, (id, deadline)) in scheduled.initial.queue.entries.iter().zip([
        (0, 1.300000011920929),
        (1, 1.300000011920929),
        (2, 1.300000011920929),
        (3, 1.300000011920929),
        (4, 1.300000011920929),
        (5, 1.4000000059604645),
        (7, 1.5),
    ]) {
        assert_eq!(entry.logical_id, id);
        assert_eq!(entry.timing.deadline, deadline);
        assert_eq!(entry.timing.frame_threshold, 8);
        assert_eq!(entry.timing.phase_mask, 0xA);
        assert_eq!(entry.timing.insertion_generation, 0);
        assert!(entry.release_present);
    }
    assert_eq!(scheduled.initial.continuations.next_id, 12);
    assert_eq!(scheduled.initial.continuations.batch_ordinal, 0);
    assert_eq!(scheduled.dispatch.generation_before, 8);
    assert_eq!(scheduled.dispatch.sampled_time, 1.300000011920929);
    assert_eq!(scheduled.dispatch.sampled_frame_counter, 13);
    assert_eq!(scheduled.dispatch.phase_mask, 2);
    assert_eq!(
        scheduled.initial.deferred_waits,
        BTreeMap::from([
            (5, DeferredSetupWait::Audio),
            (7, DeferredSetupWait::Shuffle)
        ])
    );
    assert_eq!(
        scheduled.callbacks.keys().copied().collect::<Vec<_>>(),
        vec![0, 1, 2, 3, 4]
    );
    for boundary in scheduled.callbacks.values() {
        assert!(boundary.same_live_owner);
        assert_eq!(boundary.callback_result, 1);
        assert_eq!(boundary.producer_time, scheduled.dispatch.sampled_time);
        assert_eq!(
            boundary.producer_frame_counter,
            scheduled.dispatch.sampled_frame_counter
        );
    }
}

fn initialized_reference(input: &Inputs) -> Initialized {
    let c = &input.setup_context.initialization;
    let board = &c.caller.state.lists[c.caller.state.board.as_ref().unwrap()];
    let mut actors = c.actors.clone();
    let mut continuations = c.continuations.clone();
    let mut publications = Vec::new();
    let mut current_order = Vec::new();
    for (index, alias) in board.items.iter().enumerate() {
        let id = c.object_identities[alias.as_ref().unwrap()];
        let data = c.object_identities[ROW[index].3];
        let allocation = &c.allocations[index];
        let actor = actors.get_mut(&id).unwrap();
        actor.trailer = None;
        actor.infos.clear();
        actor.info_version = actor.info_version.wrapping_add(1);
        actor.runtime = None;
        actor.started = false;
        actor.bluff = None;
        actor.data = Some(data);
        actor.register_as = None;
        actor.revealed = false;
        actor.killed_demon = false;
        actor.uses = 1;
        actor.alignment = c.data[&data].starting_alignment;
        actor.id = 5 - index as i32;
        actor.previous = actor.state;
        actor.state = 5;
        actor.statuses.active.clear();
        actor.statuses.version = actor.statuses.version.wrapping_add(1);
        actor.role = allocation.clone.as_ref().map(|clone| clone.identity);
        // No reset of resistance/shared target, saved text, killedHidden, copied
        // role or other retained fields is inferred beyond the native stores.
        continuations.push(Continuation {
            identity: allocation.continuation_identity,
            actor: id,
            state: 1,
            current: Some(allocation.wait_identity),
        });
        current_order.push(id);
        publications.push(Publication {
            init_index: index,
            physical_actor: id,
            position: c.positions[&id],
            data,
            source_role: c.data[&data].source_role.unwrap(),
            clone: allocation.clone.clone(),
            display_id: 5 - index as i32,
            continuation_identity: allocation.continuation_identity,
            wait_identity: allocation.wait_identity,
            events: vec![
                Event::ClearTrailer,
                Event::HideActed,
                Event::ClearInfos,
                Event::ClearRuntime,
                Event::ClearStartGuard,
                Event::ClearBluff,
                Event::StoreData,
                Event::DiagnosticLog,
                Event::ClearRegisterAs,
                Event::ClearRevealed,
                Event::ResetKilledDemonAndUses,
                Event::StoreStartingAlignment,
                Event::SetNumberText,
                Event::StoreId,
                Event::SetHidden,
                Event::ClearStatuses,
                Event::RefreshCharacter,
                Event::RefreshView,
                Event::RegisterContinuation,
                Event::PublishRole,
                Event::FirstYield,
            ],
        });
    }
    Initialized {
        actors,
        raw_bluff_liveness: c.raw_bluff_liveness.clone(),
        positions: c.positions.clone(),
        current_order,
        continuations,
        publications,
        initialization_complete: true,
        prefix_error: None,
    }
}

fn callback(
    trigger: Trigger,
    slot: RoleSlot,
    role: CallbackRole,
    dispatch: Dispatch,
    inserted: bool,
) -> CallbackTrace {
    CallbackTrace {
        trigger,
        slot,
        role,
        dispatch,
        status_application: (trigger == Trigger::Init && role == CallbackRole::Confessor)
            .then_some(StatusApplication {
                status: 25,
                accepted: true,
                inserted,
                target_after: None,
            }),
    }
}

fn setup_reference(
    input: &Inputs,
    initialized: &Initialized,
) -> (ContinuationState, BTreeMap<u8, u64>, Vec<ActionCall>) {
    let mut actors = Vec::new();
    let mut bodies = BTreeMap::new();
    let mut current_data = BTreeMap::new();
    let mut calls = Vec::new();
    for (index, id) in initialized.current_order.iter().enumerate() {
        let a: &Actor = &initialized.actors[id];
        let position = index as u8 + 1;
        let real = ROW[index].1;
        let confessor = index == 1;
        actors.push(RevealActor {
            position,
            data_role: ROW[index].0,
            action_role: real,
            runtime_evil: index == 0,
            bluff: BluffReference::Null,
            bluff_role: None,
            register_as: None,
            statuses: StatusState {
                values: if confessor { vec![25] } else { vec![] },
                resistance: a.statuses.resistances.clone(),
                target_position: None,
            },
            remaining_continuations: 1,
            on_trigger_subscribed: false,
            character_start_acted: Some(false),
        });
        bodies.insert(
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
        );
        current_data.insert(position, a.data.unwrap());
        calls.push(ActionCall {
            position,
            trigger: Trigger::Init,
            init_callbacks: vec![callback(
                Trigger::Init,
                RoleSlot::Real,
                real,
                if index == 0 {
                    Dispatch::BluffAct
                } else {
                    Dispatch::Act
                },
                confessor,
            )],
            start_callbacks: vec![],
            initial_lying: Some(index == 0),
        });
    }
    let state = ContinuationState {
        rule_version: CONTINUATION_REGISTRY_NATIVE_V1.into(),
        initial: RevealWriterContext {
            rule_version: REVEAL_WRITER_VIEW_NATIVE_V2.into(),
            board: TwinWriterContext {
                rule_version: TWIN_WRITER_NATIVE_V1.into(),
                reveal: RevealContext {
                    rule_version: SETUP_CALLBACKS_NATIVE_V4.into(),
                    board_size: 5,
                    trailer_mode: false,
                    pools: input.setup_context.pools.clone(),
                    actors,
                    resumes: vec![],
                    spy_caches: BTreeMap::new(),
                },
                current_order: vec![1, 2, 3, 4, 5],
                position: 1,
                copied_slot: false,
                bodies,
            },
            resumes: vec![],
            ui: input.setup_context.ui.clone(),
        },
        pending: initialized
            .continuations
            .iter()
            .map(|c| (c.identity, initialized.positions[&c.actor]))
            .collect(),
        next_id: input.setup_context.next_logical_id,
        batch_ordinal: 0,
    };
    (state, current_data, calls)
}

fn logical_setup(input: &Inputs, setup: &ContinuationState) -> ContinuationState {
    let mut logical = setup.clone();
    logical.initial.board.reveal.rule_version = SETUP_REVEAL_CALLBACKS_NATIVE_V5.into();
    logical.pending = input
        .native_logical_map
        .iter()
        .map(|m| {
            assert_eq!(setup.pending[&m.native_iterator], m.position);
            (m.logical_id, m.position)
        })
        .collect();
    logical.next_id = 12;
    logical
}

fn callback_evidence(path: &ScheduledRevealPath) -> Value {
    // Deliberately exclude the two declared probability fields only. Full state,
    // trace and created-record evidence remain in the comparison.
    Value::Array(
        path.callbacks
            .iter()
            .map(|c| {
                json!({
                    "logical_id": c.logical_id, "state": c.replay.state,
                    "trace": c.replay.trace, "created": c.replay.created,
                })
            })
            .collect(),
    )
}

fn tail(live: bool) -> RevealTailTrace {
    let source = if live {
        VisualSource::RawBluff
    } else {
        VisualSource::CurrentData
    };
    let refresh = ViewWrite::Refresh {
        created_dead: false,
        pickable_write: None,
        rip_write: None,
        disguise_write: None,
    };
    let mut writes = vec![ViewWrite::Colors { source }, refresh.clone()];
    if live {
        writes.push(refresh.clone());
    }
    writes.extend([ViewWrite::Colors { source }, refresh]);
    RevealTailTrace {
        name_art_source: source,
        final_color_source: source,
        writes,
    }
}

fn acquisition_reference(
    input: &Inputs,
    logical: &ContinuationState,
    choice: usize,
) -> (
    ScheduledRevealState,
    Value,
    Vec<WaitQueueEvent>,
    BTreeMap<u8, u32>,
) {
    let (bluff, copied, name) = COPIES[choice];
    let mut registry = logical.clone();
    let mut queue = input.context.initial.queue.clone();
    queue.generation = queue.generation.wrapping_add(1);
    let mut trace = Vec::new();
    let mut evidence = Vec::new();
    let mut versions = BTreeMap::from([(1, 1), (2, 2), (3, 1), (4, 1), (5, 1)]);
    for id in 0..5 {
        let position = id as u8 + 1;
        let live = id == 0;
        let actor = &mut registry.initial.board.reveal.actors[id as usize];
        let mut selector = None;
        if live {
            actor.bluff = BluffReference::Live { role: bluff };
            actor.bluff_role = Some(copied);
            if copied == CallbackRole::Confessor {
                actor.statuses.values.push(25);
                versions.insert(1, 2);
            }
            // Copying an already registered duplicate creates a runtime object,
            // not a second deck occurrence. Only an absent unique is appended.
            if choice >= 4 {
                registry
                    .initial
                    .board
                    .reveal
                    .pools
                    .script
                    .villagers
                    .push(name.into());
            }
            selector = Some(SelectorTrace {
                event: SelectorEvent {
                    position,
                    acquisition_ordinal: 0,
                    selector: Selector::Minion,
                },
                bluff_role: name.into(),
                source: if choice < 4 {
                    BluffAcquisitionSource::DuplicatePool {
                        occurrence_index: choice as u16,
                    }
                } else {
                    BluffAcquisitionSource::UniquePool {
                        occurrence_index: (choice - 4) as u16,
                    }
                },
                rng_draw_count: 2,
                script_added: choice >= 4,
                corruption_attempt: None,
            });
        }
        actor.remaining_continuations = 0;
        let real = ROW[id as usize].1;
        let mut callbacks = Vec::new();
        for trigger in [Trigger::Init, Trigger::AfterRoundStart] {
            // Evil with an installed copied role calls real.Act and copy.BluffAct.
            // All four Good actors have null selectors and call real.Act.
            callbacks.push(callback(
                trigger,
                RoleSlot::Real,
                real,
                Dispatch::Act,
                false,
            ));
            if live {
                callbacks.push(callback(
                    trigger,
                    RoleSlot::Bluff,
                    copied,
                    Dispatch::BluffAct,
                    copied == CallbackRole::Confessor,
                ));
            }
        }
        registry.pending.remove(&id);
        registry.batch_ordinal += 1;
        registry.initial.board.position = position;
        let resume = WriterResumeTrace {
            acquisition: ResumeTrace {
                event: ResumeEvent {
                    position,
                    resume_ordinal: 0,
                    acquisition_ordinal: Some(0),
                },
                previous_register_as: None,
                acquisition: selector,
                selector_status: None,
                callbacks: vec![],
                spy_register_as: None,
                spy_acquisition: None,
            },
            start: None,
            callbacks,
            replacement_views: vec![],
            view: Some(tail(live)),
        };
        evidence.push(json!({"logical_id":id,"state":registry,"trace":[resume],"created":[]}));
        trace.extend([
            WaitQueueEvent::Visit {
                logical_id: id,
                eligibility: WaitEligibility::Eligible,
            },
            WaitQueueEvent::Erase { logical_id: id },
            WaitQueueEvent::Callback { logical_id: id },
            WaitQueueEvent::CallbackResult {
                logical_id: id,
                value: 1,
            },
            WaitQueueEvent::Release { logical_id: id },
        ]);
    }
    // Native traversal stops on the first future deadline, without visiting
    // the later Shuffle record. Both complete record snapshots are preserved.
    trace.push(WaitQueueEvent::Visit {
        logical_id: 5,
        eligibility: WaitEligibility::StopAtFutureDeadline,
    });
    queue
        .entries
        .retain(|e| e.logical_id == 5 || e.logical_id == 7);
    (
        ScheduledRevealState {
            rule_version: SCHEDULED_SETUP_REVEAL_NATIVE_V2.into(),
            continuations: registry,
            queue,
            deferred_waits: input.context.initial.deferred_waits.clone(),
        },
        Value::Array(evidence),
        trace,
        versions,
    )
}

fn actual_versions(
    initialized: &Initialized,
    setup: &setup_action_bridge::Path,
    path: &ScheduledRevealPath,
) -> BTreeMap<u8, u32> {
    let mut versions = initialized
        .actors
        .iter()
        .map(|(id, a)| (initialized.positions[id], a.statuses.version))
        .collect::<BTreeMap<_, _>>();
    for call in &setup.calls {
        for effect in call
            .init_callbacks
            .iter()
            .filter_map(|c| c.status_application.as_ref())
        {
            if effect.inserted {
                *versions.get_mut(&call.position).unwrap() += 1;
            }
        }
    }
    for callback in &path.callbacks {
        for resume in &callback.replay.trace {
            for effect in resume
                .callbacks
                .iter()
                .filter_map(|c| c.status_application.as_ref())
            {
                if effect.inserted {
                    *versions
                        .get_mut(&resume.acquisition.event.position)
                        .unwrap() += 1;
                }
            }
        }
    }
    versions
}

#[test]
fn conditional_row_zero_six_outcomes_match_independent_full_state_reference() {
    let input = inputs();
    assert_fixed_input(&input);
    let expected_init = initialized_reference(&input);
    let actual_init =
        setup_initialization_batch::replay(&input.setup_context.initialization).unwrap();
    assert_eq!(actual_init, expected_init);
    let (expected_setup, current_data, calls) = setup_reference(&input, &expected_init);
    let paths = setup_action_bridge::replay(&input.setup_context).unwrap();
    assert_eq!(paths.len(), 1);
    let setup = &paths[0];
    assert_eq!(setup.state, expected_setup);
    assert_eq!(setup.current_data, current_data);
    assert_eq!(setup.calls, calls);
    assert!(setup.created.is_empty());
    let logical = logical_setup(&input, &expected_setup);
    assert_eq!(input.context.initial.continuations, logical);
    // Supplied UI absence is part of the boundary, not an inferred asset fact.
    assert!(logical
        .initial
        .ui
        .values()
        .all(|ui| ui.disguise_icon_active.is_none()));
    let actual = scheduled_reveal::replay_scheduled_reveal(&input.context).unwrap();
    assert_eq!(actual.len(), 6);
    let mut actual_support = BTreeSet::new();
    let mut expected_support = BTreeSet::new();
    for (choice, (_, role, name)) in COPIES.iter().enumerate() {
        let matching = actual
            .iter()
            .filter(|path| {
                path.state.continuations.initial.board.reveal.actors[0].bluff_role == Some(*role)
            })
            .collect::<Vec<_>>();
        assert_eq!(matching.len(), 1, "copy {name}");
        let path = matching[0];
        let (state, callbacks, queue_trace, versions) =
            acquisition_reference(&input, &logical, choice);
        assert_eq!(path.state, state, "complete state for {name}");
        assert_eq!(
            callback_evidence(path),
            callbacks,
            "complete callback states/traces for {name}"
        );
        assert_eq!(path.queue_trace, queue_trace, "chronology for {name}");
        let observed_versions = actual_versions(&actual_init, setup, path);
        assert_eq!(observed_versions, versions, "status versions for {name}");
        // Current-data identity has no writer in this declared callback family;
        // retain it, and the independently accumulated versions, in the key.
        actual_support.insert(
            serde_json::to_string(&json!({
                "state": path.state, "current_data": setup.current_data,
                "status_versions": observed_versions,
            }))
            .unwrap(),
        );
        expected_support.insert(
            serde_json::to_string(&json!({
                "state": state, "current_data": current_data,
                "status_versions": versions,
            }))
            .unwrap(),
        );
    }
    assert_eq!(actual_support.len(), 6);
    assert_eq!(actual_support, expected_support);
}

fn rejects_without_mutation(context: &ScheduledRevealContext) {
    let before = context.clone();
    assert_eq!(
        scheduled_reveal::replay_scheduled_reveal(context),
        Err(LedgerError::InvalidContext)
    );
    assert_eq!(*context, before);
}

#[test]
fn conditional_row_zero_atomic_joins_and_deferred_boundaries_reject() {
    let input = inputs();
    assert_fixed_input(&input);
    let mut missing_start = input.setup_context.clone();
    missing_start.initialization.caller.state.order = None;
    let before = missing_start.clone();
    assert_eq!(
        setup_action_bridge::replay(&missing_start),
        Err(LedgerError::InvalidContext)
    );
    assert_eq!(missing_start, before);

    let mut binding = input.setup_context.clone();
    binding.initialization.allocations[0]
        .clone
        .as_mut()
        .unwrap()
        .managed_class = "Tracker".into();
    let before = binding.clone();
    assert_eq!(
        setup_action_bridge::replay(&binding),
        Err(LedgerError::InvalidContext)
    );
    assert_eq!(binding, before);

    let mut context = input.context.clone();
    context.initial.continuations.pending.remove(&4);
    rejects_without_mutation(&context);
    let mut context = input.context.clone();
    context.initial.queue.entries[4].logical_id = 5;
    rejects_without_mutation(&context);
    let mut context = input.context.clone();
    context.initial.deferred_waits.remove(&7);
    rejects_without_mutation(&context);
    let mut context = input.context.clone();
    context.initial.queue.entries[4].release_present = false;
    rejects_without_mutation(&context);
    let mut context = input.context.clone();
    context.callbacks.get_mut(&4).unwrap().same_live_owner = false;
    rejects_without_mutation(&context);
    let mut context = input.context.clone();
    context.callbacks.get_mut(&4).unwrap().callback_result = 0;
    rejects_without_mutation(&context);

    // Failure occurs after the five otherwise valid card callbacks; no partial
    // successful support escapes the API when a deferred callback becomes due.
    for (due_time, deferred_id) in [(1.4000000059604645, 5), (1.5, 7)] {
        let mut context = input.context.clone();
        context.dispatch.sampled_time = due_time;
        if deferred_id == 7 {
            context
                .initial
                .queue
                .entries
                .iter_mut()
                .find(|e| e.logical_id == 5)
                .unwrap()
                .timing
                .frame_threshold = context.dispatch.sampled_frame_counter + 1;
        }
        rejects_without_mutation(&context);
    }
    let mut context = input.context.clone();
    context
        .initial
        .continuations
        .initial
        .board
        .reveal
        .pools
        .unique
        .push("Bombardier".into());
    rejects_without_mutation(&context);
}
