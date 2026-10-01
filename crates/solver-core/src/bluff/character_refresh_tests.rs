use super::super::character_initialization::Statuses;
use super::*;
use serde_json::{json, Value};

const ARENA: u64 = 0x2_0000_0000;

fn context() -> Context {
    Context {
        version: CHARACTER_REFRESH_NATIVE_V1.into(),
        actor: Actor {
            identity: ARENA + 0x1000,
            data: Some(ARENA + 0x2000),
            bluff: Some(ARENA + 0x3000),
            register_as: Some(0xAA01),
            trailer: Some(0xAA02),
            runtime: Some(0xAA03),
            dead_prefab: Some(ObjectReference {
                identity: 0xAA04,
                live: false,
            }),
            revealed: false,
            uses: 7,
            previous: 30,
            state: 10,
            killed_hidden: true,
            killed_demon: true,
            alignment: 20,
            id: -100,
            started: true,
            role: Some(0xAA05),
            bluff_role: Some(0xAA06),
            saved_act: Some(0xAA07),
            infos: vec![Some(0xAA08), None],
            info_version: 0xFFFF_FFFF,
            statuses: Statuses {
                active: vec![10, 30],
                version: 123,
                resistances: vec![40, 50],
                target: Some(ARENA + 0x1000),
            },
            state_callback: Some(0xAA09),
        },
        raw_bluff: Some(ObjectReference {
            identity: ARENA + 0x3000,
            live: true,
        }),
        gameplay_previous_state: 20,
        gameplay_current_state: 10,
        data: [0, 10, 10]
            .into_iter()
            .enumerate()
            .map(|(index, usage)| AbilityData {
                identity: ARENA + 0x2000 + index as u64 * 0x1000,
                ability_usage: usage,
                picking: true,
            })
            .collect(),
        picked_array_identity: ARENA + 0x5000,
        picked_objects: vec![ARENA + 0x8000, ARENA + 0x9000],
        pickable_object: ARENA + 0x7000,
        ui_objects: [0x7000, 0x8000, 0x9000, 0xA000]
            .into_iter()
            .map(|offset| UiObject {
                identity: ARENA + offset,
                active: offset != 0x7000,
            })
            .collect(),
        native_runtime_provenance_verified: true,
        unity_liveness_verified: true,
        callbacks_and_ui_inert: true,
        normal_completion_verified: true,
    }
}

fn active(c: &Context, identity: u64) -> bool {
    c.ui_objects
        .iter()
        .find(|o| o.identity == identity)
        .unwrap()
        .active
}

fn snapshot(c: &Context) -> Value {
    json!({"uses": c.actor.uses as u32, "state": c.actor.state as u32,
           "revealed": u8::from(c.actor.revealed), "data": c.actor.data.unwrap_or(0),
           "bluff": c.actor.bluff.unwrap_or(0), "pickable_active": active(c, ARENA + 0x7000),
           "picked_active": [active(c, ARENA + 0x8000), active(c, ARENA + 0x9000), active(c, ARENA + 0xA000)]})
}

#[test]
fn agrees_with_supported_complete_native_caller_fixtures() {
    let report: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_refresh.json"
    )))
    .unwrap();
    assert_eq!(report["case_count"], 2094);
    let mut compared = 0;
    for native in report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report["failure_baselines"].as_array().unwrap())
    {
        let options = &native["options"];
        let count = options["picked_count"].as_u64().unwrap_or(2);
        if native["returned"] != true || !options["null"].is_null() || count > 4096 {
            continue;
        }
        let mut c = context();
        c.actor.uses = options["uses"].as_i64().unwrap_or(0) as i32;
        c.actor.state = options["state"].as_i64().unwrap_or(10) as i32;
        c.actor.revealed = options["revealed"].as_u64().unwrap_or(0) != 0;
        c.gameplay_previous_state = options["previous_phase"].as_i64().unwrap_or(20) as i32;
        c.gameplay_current_state = options["current_phase"].as_i64().unwrap_or(20) as i32;
        let liveness = options["bluff_liveness"].as_str().unwrap_or("live");
        if liveness == "absent" {
            c.actor.bluff = None;
            c.raw_bluff = None;
        } else {
            c.raw_bluff.as_mut().unwrap().live = liveness == "live";
        }
        for (index, d) in c.data.iter_mut().enumerate() {
            d.ability_usage = options["usages"][index]
                .as_i64()
                .unwrap_or([0, 10, 10][index]) as i32;
            d.picking = options["picking"][index].as_u64().unwrap_or(1) != 0;
        }
        c.picked_objects =
            [ARENA + 0x8000, ARENA + 0x9000, ARENA + 0xA000][..count as usize].to_vec();
        c.ui_objects[0].active = options["pickable_active"].as_bool().unwrap_or(false);
        assert_eq!(snapshot(&c), native["before"]);
        let original = c.clone();
        let result = replay(&c).unwrap();
        assert_eq!(snapshot(&result.context), native["final"]);
        let mut expected_actor = original.actor.clone();
        expected_actor.uses = native["final"]["uses"].as_u64().unwrap() as u32 as i32;
        assert_eq!(result.context.actor, expected_actor);
        assert_eq!(result.context.data, original.data);
        assert_eq!(result.context.raw_bluff, original.raw_bluff);
        assert_eq!(result.context.picked_objects, original.picked_objects);
        assert_eq!(
            result.context.picked_array_identity,
            original.picked_array_identity
        );
        let native_set_count = native
            .get("event_kinds")
            .map(|kinds| {
                kinds
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|kind| **kind == "set_active")
                    .count()
            })
            .unwrap_or_else(|| {
                native["events"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .filter(|e| e["kind"] == "set_active")
                    .count()
            });
        assert_eq!(
            result
                .events
                .iter()
                .filter(|e| matches!(e, Event::HidePicked { .. } | Event::ActivatePickable { .. }))
                .count(),
            native_set_count
        );
        assert_eq!(c, original);
        compared += 1;
    }
    assert_eq!(compared, 1975);
}

#[test]
fn ordered_reads_and_writes_preserve_the_complete_actor() {
    let original = context();
    let result = replay(&original).unwrap();
    assert_eq!(
        result.events,
        vec![
            Event::HidePicked {
                array_index: 0,
                object: ARENA + 0x8000
            },
            Event::HidePicked {
                array_index: 1,
                object: ARENA + 0x9000
            },
            Event::SelectData {
                pass: SelectionPass::AbilityUsage,
                source: DataSource::Bluff,
                identity: ARENA + 0x3000
            },
            Event::ResetUses {
                previous: 7,
                current: 1
            },
            Event::SelectData {
                pass: SelectionPass::Picking,
                source: DataSource::Bluff,
                identity: ARENA + 0x3000
            },
            Event::ActivatePickable {
                object: ARENA + 0x7000
            },
        ]
    );
    let mut expected = original.actor.clone();
    expected.uses = 1;
    assert_eq!(result.context.actor, expected);
    for state in [5, 20] {
        let mut c = original.clone();
        c.actor.state = state;
        c.data[0].ability_usage = 10;
        let result = replay(&c).unwrap();
        assert_eq!(result.context.actor.uses, 1);
        assert!(matches!(
            result.events.last(),
            Some(Event::ResetUses { .. })
        ));
        assert!(!active(&result.context, c.pickable_object));
    }
}

#[test]
fn repeated_and_aliased_ui_occurrences_retain_physical_identity() {
    let mut c = context();
    c.picked_objects = vec![c.pickable_object, c.pickable_object];
    c.ui_objects[0].active = true;
    let result = replay(&c).unwrap();
    assert!(active(&result.context, c.pickable_object));
    assert_eq!(
        result.events[..2],
        [
            Event::HidePicked {
                array_index: 0,
                object: c.pickable_object
            },
            Event::HidePicked {
                array_index: 1,
                object: c.pickable_object
            },
        ]
    );
    let mut second = result.context.clone();
    second.gameplay_previous_state = 0;
    let next = replay(&second).unwrap();
    assert!(!active(&next.context, second.pickable_object));
    assert_eq!(next.events.len(), 2);
    assert_eq!(next.context.actor, second.actor);
}

#[test]
fn destroyed_data_and_non_reset_usage_preserve_existing_active_control() {
    let mut c = context();
    c.raw_bluff.as_mut().unwrap().live = false;
    c.ui_objects[0].active = true;
    let result = replay(&c).unwrap();
    assert_eq!(result.context.actor.uses, 7);
    assert!(active(&result.context, c.pickable_object));
    assert_eq!(result.context.actor.bluff, c.actor.bluff);
    assert!(matches!(
        result.events.last(),
        Some(Event::SelectData {
            source: DataSource::Real,
            ..
        })
    ));
    c.raw_bluff.as_mut().unwrap().live = true;
    c.data[1].ability_usage = 50;
    assert_eq!(replay(&c).unwrap().context.actor, c.actor);
    // Equal data references are valid; each selector retains that same physical record.
    c.actor.bluff = c.actor.data;
    c.raw_bluff.as_mut().unwrap().identity = c.actor.data.unwrap();
    c.data[0].ability_usage = 10;
    let aliased = replay(&c).unwrap();
    assert!(aliased.events.iter().filter(|e| matches!(e, Event::SelectData { .. }))
        .all(|e| matches!(e, Event::SelectData { identity, .. } if *identity == c.actor.data.unwrap())));
}

#[test]
fn unsupported_provenance_and_typed_identity_collisions_reject_atomically() {
    let original = context();
    for field in [
        "native_runtime_provenance_verified",
        "unity_liveness_verified",
        "callbacks_and_ui_inert",
        "normal_completion_verified",
    ] {
        let mut value = serde_json::to_value(&original).unwrap();
        value[field] = json!(false);
        let c: Context = serde_json::from_value(value).unwrap();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    let mut invalid = vec![];
    let mut c = original.clone();
    c.actor.data = None;
    invalid.push(c);
    let mut c = original.clone();
    c.raw_bluff = None;
    invalid.push(c);
    let mut c = original.clone();
    c.actor.bluff = Some(0);
    invalid.push(c);
    let mut c = original.clone();
    c.pickable_object = 0;
    invalid.push(c);
    let mut c = original.clone();
    c.picked_objects.push(0);
    invalid.push(c);
    let mut c = original.clone();
    c.ui_objects.push(c.ui_objects[0].clone());
    invalid.push(c);
    let mut c = original.clone();
    c.data.push(c.data[0].clone());
    invalid.push(c);
    let mut c = original.clone();
    c.picked_array_identity = c.actor.identity;
    invalid.push(c);
    let mut c = original.clone();
    c.data[0].identity = c.ui_objects[0].identity;
    invalid.push(c);
    let mut c = original.clone();
    c.data.clear();
    invalid.push(c);
    let mut c = original.clone();
    c.version = "unknown".into();
    invalid.push(c);
    for c in invalid {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
    for extra in ["failure", "equality_effects", "inferred_liveness"] {
        let mut value = serde_json::to_value(&original).unwrap();
        value[extra] = json!({});
        assert!(serde_json::from_value::<Context>(value).is_err());
    }
}

#[test]
fn capacity_limits_apply_before_cloning_and_keep_input_unchanged() {
    let mut c = context();
    c.picked_objects = vec![c.pickable_object; 4096];
    assert!(replay(&c).is_ok());
    c.picked_objects.push(c.pickable_object);
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    c = context();
    c.actor.statuses.resistances = vec![10; 4097];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}
