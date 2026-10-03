use super::*;
use serde_json::{json, Value};

const ARENA: u64 = 0x2_0000_0000;
const OLD: u64 = 0xa5a5_a5a5_a5a5_a5a5;

fn context() -> Context {
    Context {
        version: CHARACTER_INITIALIZATION_NATIVE_V1.into(),
        method: Method::Init,
        actor: Actor {
            identity: ARENA + 0x1000,
            data: Some(OLD),
            bluff: Some(OLD),
            register_as: Some(OLD),
            trailer: Some(OLD),
            runtime: Some(OLD),
            dead_prefab: None,
            revealed: true,
            uses: 7,
            previous: 10,
            state: 20,
            killed_hidden: true,
            killed_demon: true,
            alignment: 10,
            id: 73,
            started: true,
            role: Some(OLD),
            bluff_role: Some(OLD),
            saved_act: Some(OLD),
            infos: vec![Some(11), None],
            info_version: 17,
            statuses: Statuses {
                active: vec![10, 30, 50],
                version: 23,
                resistances: vec![10],
                target: Some(ARENA + 0x9000),
            },
            state_callback: Some(ARENA + 0xc000),
        },
        data: ARENA + 0x2000,
        starting_alignment: 20,
        id: -100,
        required_objects_and_lists_valid: true,
        callbacks_and_ui_inert: true,
        clone_result_verified: true,
        synchronous_first_yield_verified: true,
        clone_result: Some(ARENA + 0x1c300),
        continuation_identity: ARENA + 0xd000,
        wait_identity: ARENA + 0x1c200,
        continuations: vec![],
    }
}

fn snapshot(a: &Actor) -> Value {
    json!({
        "data": a.data.unwrap_or(0), "bluff": a.bluff.unwrap_or(0),
        "register_as": a.register_as.unwrap_or(0), "trailer": a.trailer.unwrap_or(0),
        "runtime": a.runtime.unwrap_or(0), "dead_prefab": a.dead_prefab.as_ref().map_or(0, |p| p.identity),
        "revealed": u8::from(a.revealed), "uses": a.uses, "previous": a.previous,
        "state": a.state, "killed_hidden": u8::from(a.killed_hidden),
        "killed_demon": u8::from(a.killed_demon), "alignment": a.alignment,
        "id": a.id, "started": u8::from(a.started), "role": a.role.unwrap_or(0),
        "bluff_role": a.bluff_role.unwrap_or(0), "saved_act": a.saved_act.unwrap_or(0),
        "info_count": a.infos.len(), "info_version": a.info_version,
        "status_count": a.statuses.active.len(), "status_version": a.statuses.version,
    })
}

fn assert_snapshot(a: &Actor, native: &Value) {
    for (field, value) in snapshot(a).as_object().unwrap() {
        assert_eq!(native[field], *value, "{field}");
    }
}

#[test]
fn agrees_with_successful_joined_native_corpus() {
    let report: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_init.json"
    )))
    .unwrap();
    assert_eq!(report["case_count"], 475);
    let mut compared = 0;
    for case in report["cases"].as_array().unwrap() {
        let input = &case["input"];
        if input["joined"] != true
            || !case["error"].is_null()
            || input["swap_status"] == true
            || !input["callback_state"].is_null()
            || !input["null"].is_null()
            || !input["fail"].is_null()
        {
            continue;
        }
        let mut c = context();
        c.method = if case["method"] == "Init" {
            Method::Init
        } else {
            Method::InitWithNoReset
        };
        c.id = input["id"].as_i64().unwrap() as i32;
        c.actor.infos = vec![None; input["info_count"].as_u64().unwrap() as usize];
        c.actor.state_callback = if input["callback"] == true {
            Some(ARENA + 0xc000)
        } else {
            None
        };
        c.actor.dead_prefab = match input["dead"].as_str().unwrap() {
            "absent" => None,
            value => Some(ObjectReference {
                identity: ARENA + 0x1d000,
                live: value == "live",
            }),
        };
        if input["clone_null"] == true {
            c.clone_result = None;
        }
        let result = replay(&c).unwrap();
        assert_snapshot(&result.actor, &case["final"]);
        assert_eq!(
            result.continuations.last().unwrap().state as u64,
            case["final"]["iterator_state"].as_u64().unwrap()
        );
        assert_eq!(
            result.continuations.last().unwrap().current.unwrap(),
            case["final"]["iterator_current"].as_u64().unwrap()
        );
        let mut cumulative = serde_json::Map::new();
        for event in case["events"].as_array().unwrap() {
            cumulative.extend(event["actor_changes"].as_object().unwrap().clone());
            if event["kind"] == "state_callback" {
                assert_snapshot(
                    result.callback_observation.as_ref().unwrap(),
                    &Value::Object(cumulative.clone()),
                );
            }
        }
        assert_eq!(result.wait_seconds_f32_bits, 0x3e99_999a);
        compared += 1;
    }
    assert_eq!(compared, 162);
}

#[test]
fn retained_manage_pool_init_join_matches_complete_first_yield_projections() {
    let report: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_manage_initialization_join.json"
    )))
    .unwrap();
    assert_eq!(report["schema"], "manage_initialization_join_v1");
    assert_eq!(report["case_count"], 8);
    assert_eq!(report["initializer_calls"], 16);
    assert_eq!(report["completed_initializers"], 12);
    let mut compared_calls = 0;
    for case in report["cases"].as_array().unwrap() {
        // State-zero scheduling and callbacks that stop mid-body are outside
        // this replay's admitted domain. Native assertions retain those cases.
        if case["stage"] != "first_yield" || !case["error"].is_null() {
            continue;
        }
        let initial = &case["initial_actors"]["actors"]["character"];
        let n = |key: &str| initial[key].as_u64().unwrap();
        let pointer = |key: &str| (n(key) != 0).then_some(n(key));
        let mut a = context().actor;
        a.identity = n("actor");
        a.data = pointer("data_pointer");
        a.bluff = pointer("bluff");
        a.register_as = pointer("register_as");
        a.trailer = pointer("trailer");
        a.runtime = pointer("runtime");
        a.role = pointer("role");
        a.bluff_role = pointer("bluff_role");
        a.saved_act = pointer("saved_act");
        a.dead_prefab = None;
        a.state_callback = pointer("state_callback");
        a.statuses.target = pointer("status_target");
        a.alignment = n("alignment") as i32;
        a.id = n("id") as i32;
        a.previous = n("previous") as i32;
        a.state = n("state") as i32;
        a.info_version = n("info_version") as u32;
        a.statuses.version = n("status_version") as u32;
        a.infos = initial["info_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_u64().filter(|p| *p != 0))
            .collect();
        a.statuses.active = initial["status_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        a.statuses.resistances = initial["resistance_values"]
            .as_array()
            .unwrap()
            .iter()
            .map(|v| v.as_i64().unwrap() as i32)
            .collect();
        a.revealed = n("revealed") != 0;
        a.started = n("started") != 0;
        a.killed_hidden = n("killed_hidden") != 0;
        a.killed_demon = n("killed_demon") != 0;
        a.uses = n("uses") as i32;
        let assert_join_actor = |actor: &Actor, native: &Value| {
            let mut expected = native.clone();
            expected["data"] = native["data_pointer"].clone();
            assert_snapshot(actor, &expected);
            assert_eq!(actor.identity, native["actor"].as_u64().unwrap());
            assert_eq!(
                actor.statuses.target.unwrap(),
                native["status_target"].as_u64().unwrap()
            );
            assert_eq!(
                serde_json::to_value(&actor.statuses.resistances).unwrap(),
                native["resistance_values"]
            );
            assert_eq!(
                serde_json::to_value(&actor.statuses.active).unwrap(),
                native["status_values"]
            );
            assert_eq!(
                serde_json::to_value(
                    actor
                        .infos
                        .iter()
                        .map(|p| p.unwrap_or(0))
                        .collect::<Vec<_>>()
                )
                .unwrap(),
                native["info_values"]
            );
            assert_eq!(
                actor.state_callback.unwrap_or(0),
                native["state_callback"].as_u64().unwrap()
            );
        };
        assert_join_actor(&a, initial);
        let mut continuations = vec![];
        for (index, call) in case["initializer_calls"]
            .as_array()
            .unwrap()
            .iter()
            .enumerate()
        {
            assert_eq!(call["completed"], true);
            assert_join_actor(&a, &call["before"]);
            let result = replay(&Context {
                version: CHARACTER_INITIALIZATION_NATIVE_V1.into(),
                method: Method::Init,
                actor: a,
                data: call["after"]["data_pointer"].as_u64().unwrap(),
                // Both authored assets 7 and 1 have startingAlignment=10.
                starting_alignment: 10,
                id: call["display_id"].as_i64().unwrap() as i32,
                required_objects_and_lists_valid: true,
                callbacks_and_ui_inert: true,
                clone_result_verified: true,
                synchronous_first_yield_verified: true,
                clone_result: Some(call["clone"].as_u64().unwrap()),
                continuation_identity: call["iterator"].as_u64().unwrap(),
                wait_identity: case["final_actors"]["continuations"][index]["current"]
                    .as_u64()
                    .unwrap(),
                continuations,
            })
            .unwrap();
            let observed = case["initializer_events"]
                .as_array()
                .unwrap()
                .iter()
                .find(|event| event["kind"] == "state_callback" && event["init_index"] == index)
                .unwrap();
            assert_join_actor(
                result.callback_observation.as_ref().unwrap(),
                &observed["snapshot"]["actors"]["character"],
            );
            assert_join_actor(&result.actor, &call["after"]);
            a = result.actor;
            continuations = result.continuations;
            compared_calls += 1;
        }
        let expected: Vec<_> = case["final_actors"]["continuations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|c| Continuation {
                identity: c["identity"].as_u64().unwrap(),
                actor: a.identity,
                state: c["state"].as_i64().unwrap() as i32,
                current: Some(c["current"].as_u64().unwrap()),
            })
            .collect();
        assert_eq!(continuations, expected);
        assert_join_actor(&a, &case["final_actors"]["actors"]["character"]);
    }
    assert_eq!(compared_calls, 4);
}

#[test]
fn reset_preserves_resistance_target_copied_role_and_prior_continuations() {
    let mut c = context();
    let prior = Continuation {
        identity: 77,
        actor: c.actor.identity,
        state: 1,
        current: Some(88),
    };
    c.continuations.push(prior.clone());
    c.actor.info_version = u32::MAX;
    c.actor.statuses.version = u32::MAX;
    let r = replay(&c).unwrap();
    assert_eq!(r.actor.info_version, 0);
    assert_eq!(r.actor.statuses.version, 0);
    assert_eq!(r.actor.statuses.resistances, c.actor.statuses.resistances);
    assert_eq!(r.actor.statuses.target, c.actor.statuses.target);
    assert_eq!(r.actor.bluff_role, c.actor.bluff_role);
    assert_eq!(r.actor.saved_act, c.actor.saved_act);
    assert_eq!(r.continuations[0], prior);
    assert_eq!(r.continuations.len(), 2);
    let seen = r.callback_observation.unwrap();
    assert_eq!(seen.statuses.active, c.actor.statuses.active);
    assert_eq!(seen.state, 5);
    assert_eq!(seen.role, c.actor.role);
    assert!(r.actor.statuses.active.is_empty());
}

#[test]
fn no_reset_retains_alignment_runtime_and_destroyed_reference() {
    let mut c = context();
    c.method = Method::InitWithNoReset;
    c.actor.dead_prefab = Some(ObjectReference {
        identity: 123,
        live: false,
    });
    c.id = 0;
    c.clone_result = None;
    let r = replay(&c).unwrap();
    assert_eq!(r.actor.statuses, c.actor.statuses);
    assert_eq!(r.actor.alignment, 10);
    assert_eq!(r.actor.runtime, c.actor.runtime);
    assert_eq!(r.actor.register_as, c.actor.register_as);
    assert_eq!(r.actor.trailer, c.actor.trailer);
    assert_eq!(r.actor.dead_prefab, c.actor.dead_prefab);
    assert_eq!(r.actor.id, 0);
    assert_eq!(r.actor.role, None);
    assert!(!r.events.contains(&Event::HideRip));
    assert!(
        r.events
            .iter()
            .position(|e| *e == Event::ClearRevealed)
            .unwrap()
            < r.events
                .iter()
                .position(|e| *e == Event::StoreData)
                .unwrap()
    );
}

#[test]
fn rejects_unverified_boundaries_duplicate_allocations_and_capacity() {
    let c = context();
    for field in [
        "required_objects_and_lists_valid",
        "callbacks_and_ui_inert",
        "clone_result_verified",
        "synchronous_first_yield_verified",
    ] {
        let mut value = serde_json::to_value(&c).unwrap();
        value[field] = json!(false);
        assert!(matches!(
            replay(&serde_json::from_value(value).unwrap()),
            Err(LedgerError::InvalidContext)
        ));
    }
    let mut bad = c.clone();
    bad.version = "future".into();
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    bad = c.clone();
    bad.continuations.push(Continuation {
        identity: c.continuation_identity,
        actor: c.actor.identity,
        state: 1,
        current: Some(99),
    });
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    bad = c;
    bad.actor.statuses.active.resize(4097, 10);
    assert!(matches!(replay(&bad), Err(LedgerError::Capacity)));
}

#[test]
fn allocations_cannot_alias_retained_objects_or_waits() {
    let c = context();
    for identity in [
        OLD,
        c.actor.statuses.target.unwrap(),
        c.actor.state_callback.unwrap(),
        c.clone_result.unwrap(),
    ] {
        let mut bad = c.clone();
        bad.continuation_identity = identity;
        assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
        bad = c.clone();
        bad.wait_identity = identity;
        assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    }
    let mut bad = c.clone();
    bad.continuations.push(Continuation {
        identity: 77,
        actor: c.actor.identity,
        state: 1,
        current: Some(c.wait_identity),
    });
    assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
}

#[test]
fn prior_iterators_and_waits_cannot_alias_known_incompatible_objects() {
    let c = context();
    for identity in [c.actor.identity, c.data, c.actor.role.unwrap(), 1234] {
        let mut bad = c.clone();
        bad.continuations.push(Continuation {
            identity,
            actor: 1234,
            state: 1,
            current: Some(88),
        });
        assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    }
    for current in [0, c.actor.identity, c.data, 77] {
        let mut bad = c.clone();
        bad.continuations.push(Continuation {
            identity: 77,
            actor: c.actor.identity,
            state: 1,
            current: Some(current),
        });
        assert!(matches!(replay(&bad), Err(LedgerError::InvalidContext)));
    }
    let mut shared = c;
    shared.continuations = vec![
        Continuation {
            identity: 77,
            actor: shared.actor.identity,
            state: 1,
            current: Some(88),
        },
        Continuation {
            identity: 78,
            actor: 1234,
            state: 1,
            current: Some(88),
        },
    ];
    assert_eq!(
        replay(&shared).unwrap().continuations[..2],
        shared.continuations
    );
}
