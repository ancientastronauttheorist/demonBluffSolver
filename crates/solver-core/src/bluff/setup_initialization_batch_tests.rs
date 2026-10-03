use super::*;
use serde_json::json;

fn actor(identity: u64, data: u64) -> Actor {
    Actor {
        identity,
        data: Some(data),
        bluff: Some(900),
        register_as: Some(901),
        trailer: Some(902),
        runtime: Some(903),
        dead_prefab: Some(init::ObjectReference {
            identity: 904,
            live: false,
        }),
        revealed: true,
        uses: 4,
        previous: 10,
        state: 20,
        killed_hidden: true,
        killed_demon: true,
        alignment: 10,
        id: 99,
        started: true,
        role: Some(905),
        bluff_role: Some(906),
        saved_act: Some(907),
        infos: vec![Some(908)],
        info_version: 7,
        statuses: init::Statuses {
            active: vec![10],
            version: 8,
            resistances: vec![10],
            target: Some(2),
        },
        state_callback: None,
    }
}

pub(crate) fn context() -> Context {
    let caller: caller::Context = serde_json::from_value(json!({
        "version": caller::MANAGE_SETUP_CALLER_NATIVE_V1,
        "pinned_class_hierarchies": true, "stable_occurrence_services": true,
        "reference_equality": true, "supplied_init_data_write_only": true,
        "supplied_gateway_effects_only": true, "caller_metadata_initialized": true,
        "shuffle_metadata_initialized": true, "math_initialized": true,
        "gameplay_initialized": true, "object_initialized": true,
        "state": {
            "board": "board", "order": "order", "callback": null,
            "lists": {"board": {"items": ["a", "a", "b"], "count": 3}},
            "arrays": {"order": {"items": [], "length": 0}},
            "identities": {"a": "d0", "b": "d1", "c": "d0"},
            "data_roles": {"d0": "r0", "d1": "r1"}, "iterator_state": 0
        },
        "roster": ["d0", "d1", "d0"],
        "roles": {"r0": "Lilis", "r1": "Marionette"},
        "classes": {"Role": ["Role"], "Demon": ["Role", "Demon"], "Minion": ["Role", "Minion"],
            "Lilis": ["Role", "Demon", "Lilis"], "Marionette": ["Role", "Minion", "Marionette"]},
        "callbacks": [], "failure": null, "after": []
    }))
    .unwrap();
    Context {
        version: SETUP_INITIALIZATION_BATCH_NATIVE_V1.into(),
        caller,
        object_identities: BTreeMap::from([
            ("a".into(), 1),
            ("b".into(), 2),
            ("c".into(), 3),
            ("d0".into(), 10),
            ("d1".into(), 11),
            ("r0".into(), 20),
            ("r1".into(), 21),
        ]),
        actors: BTreeMap::from([(1, actor(1, 10)), (2, actor(2, 11)), (3, actor(3, 10))]),
        raw_bluff_liveness: BTreeMap::from([(900, false)]),
        positions: BTreeMap::from([(1, 9), (2, 4), (3, 7)]),
        data: BTreeMap::from([
            (
                10,
                DataBinding {
                    starting_alignment: 20,
                    source_role: Some(20),
                },
            ),
            (
                11,
                DataBinding {
                    starting_alignment: 10,
                    source_role: Some(21),
                },
            ),
        ]),
        continuations: vec![Continuation {
            identity: 500,
            actor: 1,
            state: 1,
            current: Some(501),
        }],
        allocations: [(20, "Lilis"), (21, "Marionette"), (20, "Lilis")]
            .into_iter()
            .enumerate()
            .map(|(i, (source_role, class))| {
                let base = 1000 + i as u64 * 10;
                Allocation {
                    init_index: i,
                    clone: Some(CloneBinding {
                        identity: base,
                        source_role,
                        managed_class: class.into(),
                    }),
                    continuation_identity: base + 1,
                    wait_identity: base + 2,
                }
            })
            .collect(),
        required_objects_and_lists_valid: true,
        callbacks_and_ui_inert: true,
        clone_results_verified: true,
        synchronous_first_yield_verified: true,
    }
}

#[test]
fn repeated_physical_actor_retains_registrations_and_last_role_publication() {
    let c = context();
    let r = replay(&c).unwrap();
    assert!(r.initialization_complete);
    assert_eq!(r.prefix_error, None);
    assert_eq!(r.current_order, vec![1, 1, 2]);
    assert_eq!(
        r.continuations
            .iter()
            .map(|p| (p.identity, p.actor))
            .collect::<Vec<_>>(),
        vec![(500, 1), (1001, 1), (1011, 1), (1021, 2)]
    );
    assert_eq!(r.continuations[0], c.continuations[0]);
    assert_eq!(r.actors[&1].role, Some(1010));
    assert_eq!(r.actors[&1].data, Some(11));
    assert_eq!(r.actors[&1].id, 2);
    assert_eq!(r.actors[&1].previous, 5);
    assert_eq!(r.positions[&1], 9);
    assert_eq!(r.actors[&1].info_version, 9);
    assert_eq!(r.actors[&1].statuses.version, 10);
    assert_eq!(r.actors[&1].statuses.target, Some(2));
    assert_eq!(r.actors[&1].statuses.resistances, vec![10]);
    assert_eq!(r.actors[&1].bluff_role, Some(906));
    assert_eq!(r.actors[&3], c.actors[&3]);
    assert_eq!(r.actors[&3].bluff, Some(900));
    assert!(!r.raw_bluff_liveness[&900]);
    assert_eq!(r.actors[&1].bluff, None);
}

#[test]
fn failed_init_gateway_does_not_initialize_or_register_its_attempt() {
    let mut c = context();
    c.caller.failure = Some(caller::Failure {
        gateway: Gateway::Init,
        occurrence: 2,
    });
    c.allocations.truncate(1);
    let r = replay(&c).unwrap();
    assert!(!r.initialization_complete);
    assert_eq!(r.prefix_error.as_deref(), Some("init"));
    assert_eq!(r.publications.len(), 1);
    assert_eq!(r.actors[&1].role, Some(1000));
    assert_eq!(r.actors[&1].data, Some(10));
    assert_eq!(r.actors[&1].id, 3);
    assert_eq!(r.actors[&2], c.actors[&2]);
    assert_eq!(r.continuations.len(), 2);
    c.allocations.push(Allocation {
        init_index: 1,
        clone: None,
        continuation_identity: 2001,
        wait_identity: 2002,
    });
    assert!(matches!(replay(&c), Err(LedgerError::InvalidContext)));
}

#[test]
fn publication_failure_is_outside_the_completed_initializer_prefix() {
    let mut c = context();
    c.caller.failure = Some(caller::Failure {
        gateway: Gateway::Publish,
        occurrence: 1,
    });
    let r = replay(&c).unwrap();
    assert!(r.initialization_complete);
    assert_eq!(r.prefix_error, None);
    assert_eq!(r.publications.len(), 3);
}

#[test]
fn rejects_class_source_alias_liveness_and_allocation_mismatches() {
    let c = context();
    let mut bad = c.clone();
    bad.allocations[1].clone.as_mut().unwrap().managed_class = "Lilis".into();
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    bad = c.clone();
    bad.allocations[1].clone.as_mut().unwrap().source_role = 20;
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    bad = c.clone();
    bad.object_identities.insert("b".into(), 1);
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    bad = c.clone();
    bad.raw_bluff_liveness.clear();
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    bad = c.clone();
    bad.allocations[1].wait_identity = 1001;
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    bad = c;
    bad.allocations[1].clone.as_mut().unwrap().identity = 501;
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
}

#[test]
fn null_clone_is_preserved_without_inventing_an_action_class() {
    let mut c = context();
    c.allocations[1].clone = None;
    let r = replay(&c).unwrap();
    assert_eq!(r.actors[&1].role, None);
    assert_eq!(r.publications[1].clone, None);
    assert_eq!(r.continuations.len(), 4);
}

#[test]
fn stopped_before_any_init_keeps_actor_and_continuation_state() {
    let mut c = context();
    c.caller.failure = Some(caller::Failure {
        gateway: Gateway::Unique,
        occurrence: 1,
    });
    c.allocations.clear();
    let r = replay(&c).unwrap();
    assert_eq!(r.actors, c.actors);
    assert_eq!(r.continuations, c.continuations);
    assert_eq!(r.prefix_error.as_deref(), Some("unique"));
}

#[test]
fn zero_init_prefix_still_rejects_incompatible_pending_object_aliases() {
    let mut c = context();
    c.caller.failure = Some(caller::Failure {
        gateway: Gateway::Unique,
        occurrence: 1,
    });
    c.allocations.clear();
    for identity in [1, 10, 20, 501] {
        let mut bad = c.clone();
        bad.continuations[0].identity = identity;
        assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    }
    for current in [0, 1, 10, 20, 500] {
        let mut bad = c.clone();
        bad.continuations[0].current = Some(current);
        assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    }
    c.continuations.push(Continuation {
        identity: 502,
        actor: 2,
        state: 1,
        current: Some(501),
    });
    assert_eq!(replay(&c).unwrap().continuations, c.continuations);
}

fn native_actor(value: &serde_json::Value) -> Actor {
    let n = |key: &str| value[key].as_u64().unwrap();
    let reference = |key: &str| (n(key) != 0).then_some(n(key));
    let mut a = actor(n("actor"), n("data"));
    a.role = reference("role");
    a.bluff_role = reference("bluff_role");
    a.bluff = reference("bluff");
    a.runtime = reference("runtime");
    a.register_as = reference("register_as");
    a.alignment = n("alignment") as i32;
    a.id = n("id") as i32;
    a.state = n("state") as i32;
    a.previous = n("previous") as i32;
    a.info_version = n("info_version") as u32;
    a.infos = vec![None, None];
    a.dead_prefab = None;
    a.statuses.active = vec![10; n("status_count") as usize];
    a.statuses.version = n("status_version") as u32;
    a.statuses.target = reference("status_target");
    a
}

fn compare_native_actor(a: &Actor, expected: &serde_json::Value) {
    let actual = json!({"actor": a.identity, "data": a.data.unwrap_or(0), "role": a.role.unwrap_or(0),
        "bluff_role": a.bluff_role.unwrap_or(0), "bluff": a.bluff.unwrap_or(0),
        "runtime": a.runtime.unwrap_or(0), "register_as": a.register_as.unwrap_or(0),
        "alignment": a.alignment, "id": a.id, "state": a.state, "previous": a.previous,
        "info_version": a.info_version, "status_count": a.statuses.active.len(),
        "status_version": a.statuses.version, "status_target": a.statuses.target.unwrap_or(0)});
    for (key, value) in actual.as_object().unwrap() {
        assert_eq!(value, &expected[key], "{key}");
    }
}

#[test]
fn actual_native_repeated_body_sequences_match_primitive_and_batch() {
    let report: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_init_sequence.json"
    )))
    .unwrap();
    assert_eq!(report["sequence_count"], 4);
    assert_eq!(report["initializer_calls"], 12);
    for (sequence_index, sequence) in report["cases"].as_array().unwrap().iter().enumerate() {
        let initial: Vec<Actor> = sequence["initial"]
            .as_array()
            .unwrap()
            .iter()
            .map(native_actor)
            .collect();
        let mut actors: BTreeMap<_, _> = initial.iter().map(|a| (a.identity, a.clone())).collect();
        let mut continuations =
            vec![
                serde_json::from_value::<Continuation>(sequence["continuations"][0].clone())
                    .unwrap(),
            ];
        for call in sequence["calls"].as_array().unwrap() {
            let identity = call["before"]["actor"].as_u64().unwrap();
            compare_native_actor(&actors[&identity], &call["before"]);
            let result = init::replay(&init::Context {
                version: init::CHARACTER_INITIALIZATION_NATIVE_V1.into(),
                method: if call["method"] == "Init" {
                    init::Method::Init
                } else {
                    init::Method::InitWithNoReset
                },
                actor: actors[&identity].clone(),
                data: call["after"]["data"].as_u64().unwrap(),
                starting_alignment: if call["data_index"] == 0 { 20 } else { 10 },
                id: call["display_id"].as_i64().unwrap() as i32,
                required_objects_and_lists_valid: true,
                callbacks_and_ui_inert: true,
                clone_result_verified: true,
                synchronous_first_yield_verified: true,
                clone_result: Some(call["clone_calls"][0]["result"].as_u64().unwrap()),
                continuation_identity: call["iterator"].as_u64().unwrap(),
                wait_identity: call["wait"].as_u64().unwrap(),
                continuations,
            })
            .unwrap();
            compare_native_actor(&result.actor, &call["after"]);
            actors.insert(identity, result.actor);
            continuations = result.continuations;
        }
        assert_eq!(
            serde_json::to_value(&continuations).unwrap(),
            sequence["continuations"]
        );
        if sequence_index != 0 {
            continue;
        }
        // Feed the exact all-Init native sequence through the Manage prefix join.
        let mut c = context();
        c.caller.state.identities.remove("c");
        let a = initial[0].identity;
        let b = initial[1].identity;
        let d0 = initial[0].data.unwrap();
        let d1 = initial[1].data.unwrap();
        let r0 = sequence["calls"][0]["clone_calls"][0]["source"]
            .as_u64()
            .unwrap();
        let r1 = sequence["calls"][1]["clone_calls"][0]["source"]
            .as_u64()
            .unwrap();
        c.object_identities = BTreeMap::from([
            ("a".into(), a),
            ("b".into(), b),
            ("d0".into(), d0),
            ("d1".into(), d1),
            ("r0".into(), r0),
            ("r1".into(), r1),
        ]);
        c.actors = initial.iter().map(|a| (a.identity, a.clone())).collect();
        c.positions = BTreeMap::from([(a, 1), (b, 2)]);
        c.raw_bluff_liveness = BTreeMap::from([(initial[0].bluff.unwrap(), false)]);
        c.data = BTreeMap::from([
            (
                d0,
                DataBinding {
                    starting_alignment: 20,
                    source_role: Some(r0),
                },
            ),
            (
                d1,
                DataBinding {
                    starting_alignment: 10,
                    source_role: Some(r1),
                },
            ),
        ]);
        c.continuations =
            vec![serde_json::from_value(sequence["continuations"][0].clone()).unwrap()];
        c.allocations = sequence["calls"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
            .map(|(i, call)| Allocation {
                init_index: i,
                clone: Some(CloneBinding {
                    identity: call["clone_calls"][0]["result"].as_u64().unwrap(),
                    source_role: call["clone_calls"][0]["source"].as_u64().unwrap(),
                    managed_class: if i == 1 {
                        "Marionette".into()
                    } else {
                        "Lilis".into()
                    },
                }),
                continuation_identity: call["iterator"].as_u64().unwrap(),
                wait_identity: call["wait"].as_u64().unwrap(),
            })
            .collect();
        let result = replay(&c).unwrap();
        assert_eq!(result.actors, actors);
        assert_eq!(result.continuations, continuations);
    }
}

#[test]
fn original_generated_n5_completed_init_prefixes_match_native_actors_and_waits() {
    // These selectors delimit successful native prefixes. They do not replay
    // failed native service mutations or prove publication/action completion.
    let fixture: serde_json::Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/fixtures/synthetic/first_village_initialization_v1.json"
    )))
    .unwrap();
    assert_eq!(fixture["schema_version"], 1);
    assert_eq!(fixture["build_id"], "f530404b0f3f_807de4a83df4");
    assert_eq!(
        fixture["native_report_sha256"],
        "855f8976081da9541161bcdf15e0a130cbd6f0115f2351c027708b6c3fb08eb1"
    );
    let cases = fixture["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 5);
    for (index, case) in cases.iter().enumerate() {
        let completed = index + 1;
        assert_eq!(case["completed_inits"], completed);
        let c: Context = serde_json::from_value(case["context"].clone()).unwrap();
        let expected_actors: BTreeMap<Identity, Actor> =
            serde_json::from_value(case["expected_actors"].clone()).unwrap();
        let expected_continuations: Vec<Continuation> =
            serde_json::from_value(case["expected_continuations"].clone()).unwrap();
        let r = replay(&c).unwrap();
        assert_eq!(r.actors, expected_actors, "prefix {completed}");
        assert_eq!(
            r.continuations, expected_continuations,
            "prefix {completed}"
        );
        assert_eq!(r.publications.len(), completed);
        assert_eq!(r.initialization_complete, completed == 5);
        assert_eq!(r.prefix_error.as_deref(), (completed < 5).then_some("init"));
        assert_eq!(r.positions, c.positions);
        for (i, publication) in r.publications.iter().enumerate() {
            assert_eq!(publication.clone, c.allocations[i].clone);
        }
    }
}
