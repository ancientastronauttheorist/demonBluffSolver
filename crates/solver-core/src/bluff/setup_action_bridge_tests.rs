use super::super::ledger::ScriptLists;
use super::super::{manage_setup_caller as caller, setup_initialization_batch as batch};
use super::*;
fn context() -> Context {
    let mut initialization = batch::tests::context();
    initialization.positions = BTreeMap::from([(1, 1), (2, 2), (3, 3)]);
    initialization
        .caller
        .state
        .lists
        .get_mut("board")
        .unwrap()
        .items = vec![Some("a".into()), Some("b".into()), Some("c".into())];
    initialization.caller.roster = Some(vec![
        Some("d1".into()),
        Some("d0".into()),
        Some("d2".into()),
    ]);
    initialization
        .caller
        .state
        .data_roles
        .insert("d2".into(), Some("r2".into()));
    initialization.object_identities.insert("d2".into(), 12);
    initialization.object_identities.insert("r2".into(), 22);
    initialization.data.insert(
        12,
        batch::DataBinding {
            starting_alignment: 10,
            source_role: Some(22),
        },
    );
    initialization.data.get_mut(&11).unwrap().starting_alignment = 20;
    initialization.caller.roles = BTreeMap::from([
        ("r0".into(), "Striga".into()),
        ("r1".into(), "Marionette".into()),
        ("r2".into(), "Drunk".into()),
    ]);
    initialization.caller.classes = BTreeMap::from([
        ("Striga".into(), vec!["Striga".into()]),
        ("Marionette".into(), vec!["Marionette".into()]),
        ("Drunk".into(), vec!["Drunk".into()]),
    ]);
    for (i, (source, class)) in [(21, "Marionette"), (20, "Striga"), (22, "Drunk")]
        .into_iter()
        .enumerate()
    {
        let clone = initialization.allocations[i].clone.as_mut().unwrap();
        clone.source_role = source;
        clone.managed_class = class.into();
    }
    initialization.caller.state.arrays.insert(
        "order".into(),
        caller::Array {
            items: vec![Some("d1".into()), Some("d0".into()), Some("d2".into())],
            length: 3,
        },
    );
    for a in initialization.actors.values_mut() {
        a.statuses.resistances.clear();
    }
    Context {
        version: SETUP_ACTION_BRIDGE_NATIVE_V1.into(),
        initialization,
        data_roles: BTreeMap::from([
            (10, DataRole::Lilis),
            (11, DataRole::TwinMinion),
            (12, DataRole::Drunk),
        ]),
        action_classes: BTreeMap::from([
            (1000, "Marionette".into()),
            (1010, "Striga".into()),
            (1020, "Drunk".into()),
            (906, "Confessor".into()),
        ]),
        action_classes_and_caches_verified: true,
        on_trigger_absent: true,
        final_services_inert: true,
        post_initialization_ui_verified: true,
        pools: SelectorPools {
            unique: vec![],
            duplicate: vec![],
            must_include: vec![],
            script: ScriptLists {
                villagers: vec![],
                outcasts: vec![],
                minions: vec![],
                demons: vec![],
            },
        },
        spy_caches: BTreeMap::new(),
        ui: (1..=3)
            .map(|p| {
                (
                    p,
                    ViewUiState {
                        pickable_active: false,
                        rip_active: true,
                        disguise_icon_active: Some(true),
                    },
                )
            })
            .collect(),
        next_logical_id: 2000,
    }
}
#[test]
fn initializer_clones_and_stale_copied_hooks_precede_dynamic_start_scan() {
    let c = context();
    let paths = replay(&c).unwrap();
    assert_eq!(paths.len(), 2);
    let mut last_positions = BTreeSet::new();
    for path in paths {
        assert_eq!(
            path.probability,
            Probability {
                numerator: 1,
                denominator: 2
            }
        );
        assert_eq!(
            path.calls.iter().map(|c| c.trigger).collect::<Vec<_>>(),
            vec![
                Trigger::Init,
                Trigger::Init,
                Trigger::Init,
                Trigger::Start,
                Trigger::Start,
                Trigger::Start
            ]
        );
        for call in &path.calls[..3] {
            assert_eq!(call.init_callbacks.len(), 2);
            assert_eq!(call.init_callbacks[1].role, CallbackRole::Confessor);
            assert_eq!(
                call.init_callbacks[1]
                    .status_application
                    .as_ref()
                    .unwrap()
                    .status,
                25
            );
        }
        last_positions.insert(path.calls.last().unwrap().position);
        assert_eq!(path.state.pending.len(), 6);
        assert_eq!(path.created.len(), 2);
        assert_eq!(path.state.next_id, 2002);
        assert!(path.state.pending.contains_key(&500));
        assert!(path.state.pending.contains_key(&1001));
        let mut counts = BTreeMap::<u8, u16>::new();
        for p in path.state.pending.values() {
            *counts.entry(*p).or_default() += 1;
        }
        for actor in &path.state.initial.board.reveal.actors {
            assert_eq!(actor.remaining_continuations, counts[&actor.position]);
        }
        assert_eq!(path.state.batch_ordinal, 0);
        assert!(path.state.initial.resumes.is_empty());
    }
    // Twin can move Drunk from physical3 to physical1. Precomputed supplied
    // Manage ActStart events would incorrectly keep requesting physical3.
    assert_eq!(last_positions, BTreeSet::from([1, 3]));
}
#[test]
fn all_initializers_complete_before_first_init_hook() {
    let c = context();
    let mut path = project(&c).unwrap();
    assert!(path
        .state
        .initial
        .board
        .reveal
        .actors
        .iter()
        .all(|a| a.statuses.values.is_empty()));
    init_pass(&mut path);
    assert!(path
        .state
        .initial
        .board
        .reveal
        .actors
        .iter()
        .all(|a| a.statuses.values == vec![25]
            && a.statuses.target_position.is_none()
            && a.character_start_acted == Some(false)));
}
#[test]
fn duplicate_order_consumes_first_match_even_when_latched() {
    let mut c = context();
    c.initialization
        .caller
        .state
        .arrays
        .get_mut("order")
        .unwrap()
        .items = vec![Some("d0".into()), Some("d0".into())];
    c.initialization
        .caller
        .state
        .arrays
        .get_mut("order")
        .unwrap()
        .length = 2;
    let paths = replay(&c).unwrap();
    assert_eq!(paths.len(), 1);
    let path = &paths[0];
    assert_eq!(path.calls[3].position, 2);
    assert_eq!(path.calls[4].position, 2);
    assert!(!path.calls[3].start_callbacks.is_empty());
    assert!(path.calls[4].start_callbacks.is_empty());
    assert_eq!(path.calls[4].initial_lying, None);
    assert!(path.created.is_empty());
}
#[test]
fn repeated_physical_occurrences_keep_init_hooks_and_continuations() {
    let mut c = context();
    c.initialization
        .caller
        .state
        .lists
        .get_mut("board")
        .unwrap()
        .items = vec![
        Some("a".into()),
        Some("a".into()),
        Some("b".into()),
        Some("c".into()),
    ];
    c.initialization
        .caller
        .state
        .lists
        .get_mut("board")
        .unwrap()
        .count = 4;
    c.initialization.caller.roster = Some(vec![
        Some("d1".into()),
        Some("d1".into()),
        Some("d0".into()),
        Some("d2".into()),
    ]);
    c.initialization.allocations = [
        (21, "Marionette"),
        (21, "Marionette"),
        (20, "Striga"),
        (22, "Drunk"),
    ]
    .into_iter()
    .enumerate()
    .map(|(i, (source, name))| batch::Allocation {
        init_index: i,
        clone: Some(batch::CloneBinding {
            identity: 1000 + i as u64 * 10,
            source_role: source,
            managed_class: name.into(),
        }),
        continuation_identity: 1001 + i as u64 * 10,
        wait_identity: 1002 + i as u64 * 10,
    })
    .collect();
    c.action_classes = BTreeMap::from([
        (1010, "Marionette".into()),
        (1020, "Striga".into()),
        (1030, "Drunk".into()),
        (906, "Confessor".into()),
    ]);
    c.initialization
        .caller
        .state
        .arrays
        .get_mut("order")
        .unwrap()
        .length = 0;
    let paths = replay(&c).unwrap();
    let path = &paths[0];
    assert_eq!(
        path.calls.iter().map(|c| c.position).collect::<Vec<_>>(),
        vec![1, 1, 2, 3]
    );
    assert_eq!(path.state.pending.values().filter(|p| **p == 1).count(), 3);
}
#[test]
fn missing_metadata_unverified_state_and_unready_continuations_reject_atomically() {
    for change in 0..8 {
        let mut c = context();
        match change {
            0 => c.on_trigger_absent = false,
            1 => {
                c.action_classes.insert(1000, "Drunk".into());
            }
            2 => {
                c.data_roles.insert(11, DataRole::Lilis);
            }
            3 => c.initialization.continuations[0].state = 0,
            4 => c.next_logical_id = 1001,
            5 => c.initialization.actors.get_mut(&1).unwrap().state_callback = Some(999),
            6 => {
                c.action_classes.insert(906, "Unsupported".into());
            }
            7 => {
                c.initialization.caller.failure = Some(caller::Failure {
                    gateway: caller::Gateway::Publish,
                    occurrence: 1,
                })
            }
            _ => unreachable!(),
        }
        assert_eq!(
            replay(&c),
            Err(LedgerError::InvalidContext),
            "change {change}"
        );
    }
}
#[test]
fn known_source_class_and_spy_cache_identity_are_separate_from_clones() {
    let mut c = context();
    c.data_roles.insert(12, DataRole::Spy { cache_key: 17 });
    c.initialization
        .caller
        .roles
        .insert("r2".into(), "Spy".into());
    c.initialization.caller.classes.remove("Drunk");
    c.initialization
        .caller
        .classes
        .insert("Spy".into(), vec!["Spy".into()]);
    c.initialization.allocations[2]
        .clone
        .as_mut()
        .unwrap()
        .managed_class = "Spy".into();
    c.action_classes.insert(1020, "Spy".into());
    c.spy_caches.insert(17, BluffReference::Null);
    c.initialization
        .caller
        .state
        .arrays
        .get_mut("order")
        .unwrap()
        .length = 0;
    let paths = replay(&c).unwrap();
    assert_eq!(paths[0].state.initial.board.reveal.spy_caches, c.spy_caches);
    c.spy_caches.clear();
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
}

fn original_n5_context() -> Context {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/fixtures/synthetic/first_village_initialization_v1.json"
    )))
    .unwrap();
    let mut initialization: batch::Context =
        serde_json::from_value(fixture["cases"][4]["context"].clone()).unwrap();
    initialization.caller.failure = None;
    let mut data_roles = BTreeMap::new();
    let mut action_classes = BTreeMap::new();
    for allocation in &initialization.allocations {
        let clone = allocation.clone.as_ref().unwrap();
        let data = *initialization
            .data
            .iter()
            .find(|(_, binding)| binding.source_role == Some(clone.source_role))
            .unwrap()
            .0;
        let role = match clone.managed_class.as_str() {
            "Minion" => DataRole::Minion,
            "Confessor" => DataRole::Confessor,
            "Empath" => DataRole::Lover,
            "Tracker" => DataRole::Hunter,
            "Shugenja" => DataRole::Enlightened,
            unexpected => panic!("unreviewed source {unexpected}"),
        };
        data_roles.insert(data, role);
        action_classes.insert(clone.identity, clone.managed_class.clone());
    }
    let next_logical_id = initialization
        .allocations
        .iter()
        .map(|a| a.continuation_identity)
        .max()
        .unwrap()
        + 1;
    let ui = initialization
        .positions
        .values()
        .map(|p| {
            (
                *p,
                ViewUiState {
                    pickable_active: false,
                    rip_active: false,
                    disguise_icon_active: None,
                },
            )
        })
        .collect();
    Context {
        version: SETUP_ACTION_BRIDGE_NATIVE_V2.into(),
        initialization,
        data_roles,
        action_classes,
        action_classes_and_caches_verified: true,
        on_trigger_absent: true,
        final_services_inert: true,
        post_initialization_ui_verified: true,
        pools: SelectorPools {
            unique: vec!["Gemcrafter".into(), "Alchemist".into()],
            duplicate: ["Confessor", "Lover", "Hunter", "Enlightened"]
                .map(String::from)
                .to_vec(),
            must_include: vec![],
            script: ScriptLists {
                villagers: ["Confessor", "Lover", "Hunter", "Enlightened"]
                    .map(String::from)
                    .to_vec(),
                outcasts: vec![],
                minions: vec!["Minion".into()],
                demons: vec![],
            },
        },
        spy_caches: BTreeMap::new(),
        ui,
        next_logical_id,
    }
}

#[test]
fn init_prefix_stops_before_ordered_start_and_preserves_original_n5_waits() {
    let c = context();
    let prefix = replay_init_prefix(&c).unwrap();
    assert_eq!(prefix.calls.len(), 3);
    assert!(prefix
        .calls
        .iter()
        .all(|call| call.trigger == Trigger::Init));
    assert!(prefix.created.is_empty());
    assert_eq!(prefix.state.pending.len(), 4);
    assert_eq!(replay(&c).unwrap()[0].calls.len(), 6);

    let c = original_n5_context();
    let prefix = replay_init_prefix(&c).unwrap();
    assert_eq!(prefix.calls.len(), 5);
    assert_eq!(prefix.state.pending.len(), 5);
    assert!(prefix.created.is_empty());
    assert_eq!(prefix.state.initial.board.reveal.pools, c.pools);
    for actor in &prefix.state.initial.board.reveal.actors {
        assert_eq!(actor.character_start_acted, Some(false));
        assert_eq!(actor.remaining_continuations, 1);
        assert_eq!(
            actor.statuses.values,
            if actor.position == 2 {
                vec![25]
            } else {
                vec![]
            }
        );
    }
    assert_eq!(prefix.calls[0].initial_lying, Some(true));
    assert_eq!(
        prefix.calls[0].init_callbacks[0].dispatch,
        Dispatch::BluffAct
    );
    assert_eq!(prefix.calls[1].init_callbacks[0].dispatch, Dispatch::Act);
    let mut legacy = c.clone();
    legacy.version = SETUP_ACTION_BRIDGE_NATIVE_V1.into();
    assert_eq!(
        replay_init_prefix(&legacy),
        Err(LedgerError::InvalidContext)
    );
}

#[test]
fn setup_only_domain_rejects_acquisition_and_copied_callback_bypasses() {
    use super::super::reveal::{
        replay_reveal_callbacks, ResumeEvent, REVEAL_CALLBACKS_NATIVE_V1,
        REVEAL_CALLBACKS_SPY_NATIVE_V2,
    };
    let prefix = replay_init_prefix(&original_n5_context()).unwrap();
    for missing_provenance in 0..2 {
        let mut input = prefix.state.initial.board.reveal.clone();
        if missing_provenance == 0 {
            input.actors[0].character_start_acted = None;
        } else {
            input.pools.duplicate.push("hunter".into());
        }
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }
    for acquisition_ordinal in [None, Some(0)] {
        let mut input = prefix.state.initial.board.reveal.clone();
        if acquisition_ordinal.is_none() {
            input.actors[0].bluff = BluffReference::Live {
                role: super::super::reveal::BluffRole::Scout,
            };
        }
        input.resumes.push(ResumeEvent {
            position: 1,
            resume_ordinal: 0,
            acquisition_ordinal,
        });
        assert_eq!(
            replay_reveal_callbacks(&input),
            Err(LedgerError::InvalidContext)
        );
    }
    let old = replay_init_prefix(&context()).unwrap();
    for version in [
        REVEAL_CALLBACKS_NATIVE_V1,
        REVEAL_CALLBACKS_SPY_NATIVE_V2,
        REVEAL_CALLBACKS_START_NATIVE_V3,
    ] {
        for role in [
            CallbackRole::Minion,
            CallbackRole::Lover,
            CallbackRole::Hunter,
            CallbackRole::Enlightened,
        ] {
            let mut input = old.state.initial.board.reveal.clone();
            input.rule_version = version.into();
            for actor in &mut input.actors {
                actor.character_start_acted =
                    (version == REVEAL_CALLBACKS_START_NATIVE_V3).then_some(false);
            }
            input.actors[0].bluff_role = Some(role);
            assert_eq!(
                replay_reveal_callbacks(&input),
                Err(LedgerError::InvalidContext)
            );
            input.actors[0].bluff_role = None;
            input.actors[0].action_role = role;
            assert_eq!(
                replay_reveal_callbacks(&input),
                Err(LedgerError::InvalidContext)
            );
        }
        for role in [
            DataRole::Minion,
            DataRole::Confessor,
            DataRole::Lover,
            DataRole::Hunter,
            DataRole::Enlightened,
        ] {
            let mut input = old.state.initial.board.reveal.clone();
            input.rule_version = version.into();
            for actor in &mut input.actors {
                actor.character_start_acted =
                    (version == REVEAL_CALLBACKS_START_NATIVE_V3).then_some(false);
            }
            input.actors[0].data_role = role;
            assert_eq!(
                replay_reveal_callbacks(&input),
                Err(LedgerError::InvalidContext)
            );
        }
    }
    let mut input = old.state.initial.board.clone();
    input.reveal.rule_version = SETUP_CALLBACKS_NATIVE_V4.into();
    assert_eq!(
        twin_writer::replay_twin_start(&input),
        Err(LedgerError::InvalidContext)
    );
}

#[test]
fn original_n5_separate_start_dispatch_preserves_init_status_and_latches_once() {
    let prefix = replay_init_prefix(&original_n5_context()).unwrap();
    for position in 1..=5 {
        let mut board = prefix.state.initial.board.clone();
        board.position = position;
        let started = character_start::replay_character_start(&CharacterStartContext {
            rule_version: CHARACTER_START_NATIVE_V1.into(),
            board,
        })
        .unwrap();
        assert_eq!(started.len(), 1);
        assert_eq!(started[0].callbacks.len(), 1);
        assert_eq!(started[0].callbacks[0].status_application, None);
        let actor = started[0]
            .board
            .reveal
            .actors
            .iter()
            .find(|a| a.position == position)
            .unwrap();
        assert_eq!(actor.character_start_acted, Some(true));
        assert_eq!(
            actor.statuses.values,
            if position == 2 { vec![25] } else { vec![] }
        );
        let repeated = character_start::replay_character_start(&CharacterStartContext {
            rule_version: CHARACTER_START_NATIVE_V1.into(),
            board: started[0].board.clone(),
        })
        .unwrap();
        assert_eq!(repeated.len(), 1);
        assert_eq!(repeated[0].board, started[0].board);
        assert_eq!(repeated[0].initial_lying, None);
        assert!(repeated[0].callbacks.is_empty());
    }
}

#[test]
fn retained_original_n5_publication_init_matches_native_semantic_checkpoint() {
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/fixtures/synthetic/first_village_publication_init_v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(fixture["build_id"], "f530404b0f3f_807de4a83df4");
    assert_eq!(
        fixture["native_report_sha256"],
        "31929f003a87c6e0ae4242ea9dc3eb090fba00bb641eedad0526c1afc651b48e"
    );
    let c: Context = serde_json::from_value(fixture["context"].clone()).unwrap();
    let expected = &fixture["expected"];
    let r = replay_init_prefix(&c).unwrap();
    let board = &r.state.initial.board;
    assert_eq!(
        &serde_json::to_value(&board.reveal.actors).unwrap(),
        &expected["actors"]
    );
    assert_eq!(
        &serde_json::to_value(&board.bodies).unwrap(),
        &expected["bodies"]
    );
    assert_eq!(&serde_json::to_value(&r.calls).unwrap(), &expected["calls"]);
    assert_eq!(
        &serde_json::to_value(&r.current_data).unwrap(),
        &expected["current_data"]
    );
    assert_eq!(
        &serde_json::to_value(&board.current_order).unwrap(),
        &expected["current_order"]
    );
    assert_eq!(
        &serde_json::to_value(&board.reveal.pools).unwrap(),
        &expected["pools"]
    );
    assert_eq!(
        &serde_json::to_value(&r.state.pending).unwrap(),
        &expected["pending"]
    );
    assert!(r.created.is_empty());
    assert_eq!(r.state.next_id, c.next_logical_id);
    assert_eq!(r.state.batch_ordinal, 0);
    assert!(r.state.initial.resumes.is_empty());
    assert!(r.calls.iter().all(|call| call.trigger == Trigger::Init));

    // The semantic state omits physical list versions. Compare the native
    // insertion delta explicitly, without claiming reconstructed storage.
    let initialized = batch::replay(&c.initialization).unwrap();
    for (identity, position) in &initialized.positions {
        let native = &expected["status_versions"][position.to_string()];
        let before = initialized.actors[identity].statuses.version;
        let inserted = r
            .calls
            .iter()
            .filter(|call| call.position == *position)
            .flat_map(|call| &call.init_callbacks)
            .filter(|call| call.status_application.as_ref().is_some_and(|s| s.inserted))
            .count() as u32;
        assert_eq!(native["before"], before);
        assert_eq!(native["after"], before + inserted);
    }
}
