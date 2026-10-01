use super::super::character_initialization::Statuses;
use super::*;
use serde_json::{json, Value};

const ARENA: u64 = 0x2_0000_0000;
const POSITION: VectorBits = [0x3F800000, 0xC0000000, 0x40400000];
const SENTINEL: VectorBits = [0xA5A5A5A5; 3];

fn context() -> Context {
    Context {
        version: CHARACTER_REFRESH_VIEW_NATIVE_V1.into(),
        actor: Actor {
            identity: ARENA + 0x1000,
            data: Some(0xAA01),
            bluff: Some(ARENA + 0x2000),
            register_as: Some(0xAA02),
            trailer: Some(0xAA03),
            runtime: Some(0xAA04),
            dead_prefab: None,
            revealed: false,
            uses: 0,
            previous: 30,
            state: 20,
            killed_hidden: true,
            killed_demon: false,
            alignment: 20,
            id: -100,
            started: true,
            role: Some(0xAA05),
            bluff_role: Some(0xAA06),
            saved_act: Some(0xAA07),
            infos: vec![None, Some(0xAA08)],
            info_version: 0xFFFFFFFF,
            statuses: Statuses {
                active: vec![10, 30],
                version: 73,
                resistances: vec![40, 50],
                target: Some(ARENA + 0x1000),
            },
            state_callback: Some(0xAA09),
        },
        raw_bluff: Some(ObjectReference {
            identity: ARENA + 0x2000,
            live: true,
        }),
        icon_component: ARENA + 0x6000,
        actor_transform: ARENA + 0x7000,
        icon_transform: ARENA + 0x8000,
        dead_prefab_template: ARENA + 0x2000,
        pickable: ARENA + 0xB000,
        rip: ARENA + 0xC000,
        disguise: Some(ObjectReference {
            identity: ARENA + 0xD000,
            live: true,
        }),
        ui_objects: [0xB000, 0xC000, 0xD000, 0xE000]
            .into_iter()
            .map(|offset| UiObject {
                identity: ARENA + offset,
                active: matches!(offset, 0xB000 | 0xD000),
            })
            .collect(),
        transforms: [0x6000, 0x7000, 0x8000, 0x9000, 0xA000]
            .into_iter()
            .map(|offset| Transform {
                identity: ARENA + offset,
                position: if offset == 0x8000 { POSITION } else { SENTINEL },
                euler_angles: SENTINEL,
            })
            .collect(),
        vector3_zero: [0; 3],
        creation: Some(Creation {
            instance: ObjectReference {
                identity: ARENA + 0x4000,
                live: true,
            },
            first_transform: ARENA + 0x9000,
            second_transform: ARENA + 0x9000,
        }),
        native_runtime_and_bindings_verified: true,
        unity_liveness_verified: true,
        callbacks_and_services_inert: true,
        normal_completion_verified: true,
    }
}

fn active(c: &Context, id: u64) -> bool {
    c.ui_objects
        .iter()
        .find(|o| o.identity == id)
        .unwrap()
        .active
}

fn bits(value: &Value, default: VectorBits) -> VectorBits {
    value.as_array().map_or(default, |items| {
        assert_eq!(items.len(), 3);
        std::array::from_fn(|i| items[i].as_u64().unwrap() as u32)
    })
}

fn snapshot(c: &Context) -> Value {
    json!({"created": c.actor.dead_prefab.as_ref().map_or(0, |d| d.identity),
        "uses": c.actor.uses as u32, "state": c.actor.state as u32,
        "revealed": u8::from(c.actor.revealed), "killed_by_demon": u8::from(c.actor.killed_demon),
        "prefab": c.dead_prefab_template, "disguise": c.disguise.as_ref().map_or(0, |d| d.identity),
        "pickable_active": active(c, ARENA + 0xB000), "rip_active": active(c, ARENA + 0xC000),
        "disguise_active": active(c, ARENA + 0xD000), "alternate_disguise_active": active(c, ARENA + 0xE000)})
}

fn api_name(event: &Event) -> Option<&'static str> {
    Some(match event {
        Event::SetActive { .. } => "set_active",
        Event::UnityNull { .. } => "unity_null",
        Event::UnityLive { .. } => "unity_live",
        Event::ComponentTransform { .. } => "component_transform",
        Event::InstantiateGameObject { .. } => "instantiate",
        Event::StoreCreated { .. } => return None,
        Event::BarrierCreated { .. } => "barrier",
        Event::GameObjectTransform { .. } => "game_object_transform",
        Event::GetPosition { .. } => "get_position",
        Event::SetPosition { .. } => "set_position",
        Event::SetEulerAngles { .. } => "set_euler_angles",
    })
}

fn native_input(native: &Value) -> Context {
    let options = &native["options"];
    let mut c = context();
    c.actor.uses = options["uses"].as_i64().unwrap_or(1) as i32;
    c.actor.state = options["state"].as_i64().unwrap_or(20) as i32;
    c.actor.revealed = options["revealed"].as_u64().unwrap_or(0) != 0;
    c.actor.killed_demon = options["killed_by_demon"].as_u64().unwrap_or(0) != 0;
    let created = native["before"]["created"].as_u64().unwrap();
    c.actor.dead_prefab = (created != 0).then(|| ObjectReference {
        identity: created,
        live: created == ARENA + 0x4000 || options["created_liveness"] == "live",
    });
    if options["bluff_liveness"] == "absent" {
        c.actor.bluff = None;
        c.raw_bluff = None;
    } else {
        c.raw_bluff.as_mut().unwrap().live = options["bluff_liveness"] != "destroyed";
    }
    if options["disguise_liveness"] == "absent" {
        c.disguise = None;
    } else {
        c.disguise.as_mut().unwrap().live = options["disguise_liveness"] != "destroyed";
    }
    c.vector3_zero = bits(&options["zero_bits"], [0; 3]);
    c.transforms[2].position = bits(&options["position_bits"], POSITION);
    if options["different_second_transform"] == true {
        c.creation.as_mut().unwrap().second_transform = ARENA + 0xA000;
    }
    if !creates(&c) {
        c.creation = None;
    }
    for (o, field) in c.ui_objects.iter_mut().zip([
        "pickable_active",
        "rip_active",
        "disguise_active",
        "alternate_disguise_active",
    ]) {
        o.active = native["before"][field].as_bool().unwrap();
    }
    for (field, euler) in [("position_writes", false), ("euler_writes", true)] {
        for write in native["before"][field].as_array().unwrap() {
            let id = write["transform"].as_u64().unwrap();
            let t = c.transforms.iter_mut().find(|t| t.identity == id).unwrap();
            if euler {
                t.euler_angles = bits(&write["bits"], [0; 3]);
            } else {
                t.position = bits(&write["bits"], [0; 3]);
            }
        }
    }
    c
}

fn assert_native(c: &Context, native: &Value) -> Replay {
    let before = c.clone();
    let result = replay(c).unwrap();
    for (key, value) in snapshot(&result.context).as_object().unwrap() {
        assert_eq!(&native["final"][key], value, "{key}");
    }
    let mut expected_actor = before.actor.clone();
    expected_actor.dead_prefab = result.context.actor.dead_prefab.clone();
    assert_eq!(result.context.actor, expected_actor);
    assert_eq!(result.context.raw_bluff, before.raw_bluff);
    assert_eq!(result.context.disguise, before.disguise);
    assert_eq!(result.context.vector3_zero, before.vector3_zero);
    let mut expected_transforms = before.transforms.clone();
    for (field, euler) in [("position_writes", false), ("euler_writes", true)] {
        for write in native["final"][field].as_array().unwrap() {
            let id = write["transform"].as_u64().unwrap();
            let t = expected_transforms
                .iter_mut()
                .find(|t| t.identity == id)
                .unwrap();
            if euler {
                t.euler_angles = bits(&write["bits"], [0; 3]);
            } else {
                t.position = bits(&write["bits"], [0; 3]);
            }
        }
    }
    assert_eq!(result.context.transforms, expected_transforms);
    let kinds: Vec<&str> = native.get("event_kinds").map_or_else(
        || {
            native["events"]
                .as_array()
                .unwrap()
                .iter()
                .map(|e| e["kind"].as_str().unwrap())
                .collect()
        },
        |kinds| {
            kinds
                .as_array()
                .unwrap()
                .iter()
                .map(|k| k.as_str().unwrap())
                .collect()
        },
    );
    let native_api: Vec<_> = kinds
        .into_iter()
        .filter(|k| !matches!(*k, "metadata" | "class_init"))
        .collect();
    assert_eq!(
        result
            .events
            .iter()
            .filter_map(api_name)
            .collect::<Vec<_>>(),
        native_api
    );
    assert_eq!(c, &before);
    result
}

#[test]
fn agrees_with_normal_native_caller_and_repeated_retention_corpus() {
    let report: Value = serde_json::from_str(include_str!(concat!(
        env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_refresh_view.json"
    )))
    .unwrap();
    assert_eq!(report["case_count"], 1645);
    let mut compared = 0;
    for native in report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report["failure_baselines"].as_array().unwrap())
    {
        let options = &native["options"];
        if native["returned"] != true || !options["null"].is_null() || options["clone_null"] == true
        {
            continue;
        }
        assert_native(&native_input(native), native);
        compared += 1;
    }
    assert_eq!(compared, 1639);
    for sequence in report["sequences"].as_array().unwrap() {
        let first = assert_native(&native_input(&sequence["first"]), &sequence["first"]);
        let second = assert_native(&first.context, &sequence["second"]);
        assert_eq!(second.context.actor, first.context.actor);
        assert!(!second
            .events
            .iter()
            .any(|e| matches!(e, Event::InstantiateGameObject { .. })));
    }
}

#[test]
fn api_order_stores_identity_before_barrier_and_keeps_both_transform_getters() {
    let original = context();
    let result = replay(&original).unwrap();
    assert_eq!(
        result.events,
        vec![
            Event::SetActive {
                object: original.pickable,
                active: false
            },
            Event::UnityNull {
                object: None,
                result: true
            },
            Event::ComponentTransform {
                component: original.actor.identity,
                result: original.actor_transform
            },
            Event::InstantiateGameObject {
                template: original.dead_prefab_template,
                parent: original.actor_transform,
                result: ARENA + 0x4000
            },
            Event::StoreCreated {
                previous: None,
                current: ObjectReference {
                    identity: ARENA + 0x4000,
                    live: true
                }
            },
            Event::BarrierCreated {
                actor: original.actor.identity,
                value: ARENA + 0x4000
            },
            Event::GameObjectTransform {
                object: ARENA + 0x4000,
                result: ARENA + 0x9000
            },
            Event::ComponentTransform {
                component: original.icon_component,
                result: original.icon_transform
            },
            Event::GetPosition {
                transform: original.icon_transform,
                bits: POSITION
            },
            Event::SetPosition {
                transform: ARENA + 0x9000,
                bits: POSITION
            },
            Event::GameObjectTransform {
                object: ARENA + 0x4000,
                result: ARENA + 0x9000
            },
            Event::SetEulerAngles {
                transform: ARENA + 0x9000,
                bits: [0; 3]
            },
            Event::SetActive {
                object: original.rip,
                active: true
            },
            Event::UnityLive {
                object: original.disguise.as_ref().map(|d| d.identity),
                result: true
            },
            Event::UnityNull {
                object: original.actor.bluff,
                result: false
            },
            Event::SetActive {
                object: original.disguise.as_ref().unwrap().identity,
                active: true
            },
        ]
    );
    assert!(result.context.creation.is_none());
}

#[test]
fn aliases_are_physical_writes_and_float_bits_are_not_numbers() {
    let mut c = context();
    c.rip = c.pickable;
    c.disguise.as_mut().unwrap().identity = c.pickable;
    c.transforms[2].position = [0x80000000, 1, 0x7FC01234];
    c.vector3_zero = [0x7F800000, 0xFF800000, 0x00800000];
    c.creation.as_mut().unwrap().second_transform = ARENA + 0xA000;
    let result = replay(&c).unwrap();
    assert!(active(&result.context, c.pickable));
    let writes: Vec<_> = result
        .events
        .iter()
        .filter_map(|e| match e {
            Event::SetActive { object, active } if *object == c.pickable => Some(*active),
            _ => None,
        })
        .collect();
    assert_eq!(writes, [false, true, true]);
    assert_eq!(
        result.context.transforms[3].position,
        [0x80000000, 1, 0x7FC01234]
    );
    assert_eq!(result.context.transforms[3].euler_angles, SENTINEL);
    assert_eq!(result.context.transforms[4].euler_angles, c.vector3_zero);
    assert_eq!(result.context.transforms[4].position, SENTINEL);
}

#[test]
fn killed_and_destroyed_icons_preserve_state_without_selecting_roles() {
    for killed in [false, true] {
        for icon_live in [false, true] {
            let mut c = context();
            c.actor.state = 30;
            c.actor.revealed = true;
            c.actor.dead_prefab = Some(ObjectReference {
                identity: ARENA + 0x3000,
                live: false,
            });
            c.creation = None;
            c.actor.killed_demon = killed;
            c.disguise.as_mut().unwrap().live = icon_live;
            c.ui_objects[2].active = true;
            c.raw_bluff.as_mut().unwrap().live = false;
            let result = replay(&c).unwrap();
            assert_eq!(result.context.actor, c.actor);
            assert_eq!(
                active(&result.context, ARENA + 0xD000),
                killed || !icon_live
            );
            assert_eq!(result.context.transforms, c.transforms);
            assert_eq!(
                result
                    .events
                    .iter()
                    .filter(|e| matches!(e, Event::UnityNull { .. }))
                    .count(),
                usize::from(!killed && icon_live)
            );
        }
    }
}

#[test]
fn missing_provenance_nulls_collisions_and_consumed_allocations_reject_atomically() {
    let original = context();
    for field in [
        "native_runtime_and_bindings_verified",
        "unity_liveness_verified",
        "callbacks_and_services_inert",
        "normal_completion_verified",
    ] {
        let mut value = serde_json::to_value(&original).unwrap();
        value[field] = json!(false);
        let c: Context = serde_json::from_value(value).unwrap();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    let mut invalid = vec![];
    let mut c = original.clone();
    c.pickable = 0;
    invalid.push(c);
    let mut c = original.clone();
    c.rip = 0;
    invalid.push(c);
    let mut c = original.clone();
    c.icon_component = 0;
    invalid.push(c);
    let mut c = original.clone();
    c.actor_transform = 0;
    invalid.push(c);
    let mut c = original.clone();
    c.dead_prefab_template = 0;
    invalid.push(c);
    let mut c = original.clone();
    c.creation = None;
    invalid.push(c);
    let mut c = original.clone();
    c.raw_bluff = None;
    invalid.push(c);
    let mut c = original.clone();
    c.creation.as_mut().unwrap().instance.identity = 0;
    invalid.push(c);
    let mut c = original.clone();
    c.creation.as_mut().unwrap().instance.live = false;
    invalid.push(c);
    let mut c = original.clone();
    c.creation.as_mut().unwrap().instance.identity = c.actor.role.unwrap();
    invalid.push(c);
    let mut c = original.clone();
    c.creation.as_mut().unwrap().first_transform = 0;
    invalid.push(c);
    let mut c = original.clone();
    c.transforms.push(c.transforms[0].clone());
    invalid.push(c);
    let mut c = original.clone();
    c.ui_objects.push(c.ui_objects[0].clone());
    invalid.push(c);
    let mut c = original.clone();
    c.transforms[0].identity = c.pickable;
    invalid.push(c);
    let mut c = original.clone();
    c.version = "unknown".into();
    invalid.push(c);
    let mut c = original.clone();
    c.actor.state = 10;
    invalid.push(c);
    for c in invalid {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
    let result = replay(&original).unwrap();
    let mut consumed = result.context.clone();
    consumed.actor.dead_prefab = None;
    consumed.creation = original.creation.clone();
    // The existing transform/UI context does not represent the old created
    // object, so explicitly retain its identity in an opaque actor reference.
    consumed.actor.runtime = result
        .context
        .actor
        .dead_prefab
        .as_ref()
        .map(|d| d.identity);
    assert_eq!(replay(&consumed), Err(LedgerError::InvalidContext));
    for field in ["effects", "failure", "null"] {
        let mut value = serde_json::to_value(&original).unwrap();
        value[field] = json!({});
        assert!(serde_json::from_value::<Context>(value).is_err());
    }
}

#[test]
fn physical_aliases_require_consistent_verified_liveness() {
    for live in [false, true] {
        let mut c = context();
        let identity = c.disguise.as_ref().unwrap().identity;
        let reference = ObjectReference { identity, live };
        c.actor.bluff = Some(identity);
        c.raw_bluff = Some(reference.clone());
        c.actor.dead_prefab = Some(reference.clone());
        c.disguise = Some(reference);
        if live {
            c.creation = None;
        }
        // One stable physical object may occupy all three slots.
        assert!(replay(&c).is_ok());
        for field in 0..3 {
            let mut invalid = c.clone();
            match field {
                0 => invalid.raw_bluff.as_mut().unwrap().live = !live,
                1 => {
                    invalid.actor.dead_prefab.as_mut().unwrap().live = !live;
                    invalid.creation = if live { context().creation } else { None };
                }
                _ => invalid.disguise.as_mut().unwrap().live = !live,
            }
            let before = invalid.clone();
            assert_eq!(replay(&invalid), Err(LedgerError::InvalidContext));
            assert_eq!(invalid, before);
        }
    }
}

#[test]
fn retained_storage_capacity_rejects_before_cloning() {
    let mut c = context();
    c.actor.infos = vec![None; 4096];
    assert!(replay(&c).is_ok());
    c.actor.infos.push(None);
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    c = context();
    c.transforms = (0..4097)
        .map(|i| Transform {
            identity: 0x9_0000_0000 + i,
            position: [0; 3],
            euler_angles: [0; 3],
        })
        .collect();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}
