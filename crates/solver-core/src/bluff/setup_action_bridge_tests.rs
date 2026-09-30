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
