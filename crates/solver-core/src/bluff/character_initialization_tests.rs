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
