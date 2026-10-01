use super::super::character_initialization::Statuses;
use super::*;
use serde_json::{json, Value};

const ARENA: u64 = 0x2_0000_0000;
const OPAQUE: u64 = 0xA5A5_A5A5_A5A5_A5A5;

fn fixture(options: &Value) -> Context {
    let death = options["death"].as_str().unwrap_or("absent");
    let disguise = options["disguise"].as_str().unwrap_or("live");
    let callback = options["callback"] != false;
    let picked_count = options["picked_count"].as_u64().unwrap_or(2) as usize;
    let methods: Vec<_> = options["sequence"]
        .as_array()
        .map_or(vec![Method::Init], |rows| {
            rows.iter()
                .map(|r| {
                    if r == "Init" {
                        Method::Init
                    } else {
                        Method::InitWithNoReset
                    }
                })
                .collect()
        });
    let ui_ids = [0x16800, 0x26000, 0x26100, 0x26200, 0x26300, 0x26400];
    Context {
        version: CHARACTER_CONSTRUCTOR_INIT_NATIVE_V1.into(),
        actor: Actor {
            identity: ARENA + 0x1000,
            data: Some(OPAQUE),
            bluff: Some(OPAQUE),
            register_as: Some(OPAQUE),
            trailer: Some(OPAQUE),
            runtime: Some(OPAQUE),
            dead_prefab: (death != "absent").then_some(ObjectReference {
                identity: ARENA + 0x26500,
                live: death == "live",
            }),
            revealed: true,
            uses: 7,
            previous: 10,
            state: 20,
            killed_hidden: true,
            killed_demon: true,
            alignment: 10,
            id: 73,
            started: true,
            role: Some(OPAQUE),
            bluff_role: Some(OPAQUE),
            saved_act: Some(OPAQUE),
            infos: vec![],
            info_version: 0,
            statuses: Statuses {
                active: vec![10, 30, 50],
                version: 23,
                resistances: vec![],
                target: Some(ARENA + 0x23000),
            },
            state_callback: callback.then_some(ARENA + 0x22000),
        },
        act: false,
        initial_acted_list: None,
        initial_hover_list: None,
        retained_lists: vec![],
        constructor: Constructor {
            acted_list: ARENA + 0x4000,
            hover_list: ARENA + 0x5000,
            empty_array: ARENA + 0x10000,
            empty_string: ARENA + 0x6000,
        },
        data: (0..2)
            .map(|i| DataBinding {
                identity: ARENA + 0x12000 + i * 0x1000,
                starting_alignment: [20, 30][i as usize],
                ability_usage: options["ability_usage"].as_i64().unwrap_or(10) as i32,
                picking: false,
                source_role: ARENA + 0x14000 + i * 0x1000,
            })
            .collect(),
        components: Components {
            acted_component: ARENA + 0x16000,
            acted_game_object: ARENA + 0x16800,
            number_component: ARENA + 0x17000,
            statuses_component: ARENA + 0x19000,
            active_status_list: ARENA + 0x1A000,
            active_status_array: ARENA + 0x1A100,
            active_status_slots: vec![10, 30, 50],
            resistance_list: ARENA + 0x1B000,
            picked_array: ARENA + 0x24000,
            picked_objects: (0..picked_count)
                .map(|i| ARENA + 0x26000 + i as u64 * 0x100)
                .collect(),
            pickable: ARENA + 0x26200,
            rip: ARENA + 0x26300,
            disguise: (disguise != "absent").then_some(ObjectReference {
                identity: ARENA + 0x26400,
                live: disguise == "live",
            }),
            icon_transform: ARENA + 0x50000,
            actor_transform: ARENA + 0x50100,
            dead_template: ARENA + 0x50200,
            ui_objects: ui_ids
                .into_iter()
                .map(|offset| UiObject {
                    identity: ARENA + offset,
                    active: true,
                })
                .collect(),
            transforms: [0x50000, 0x50100]
                .into_iter()
                .map(|offset| Transform {
                    identity: ARENA + offset,
                    position: [0x80000000, 1, 0x7FC01234],
                    euler_angles: [0xA5A5A5A5; 3],
                })
                .collect(),
            vector3_zero: [0x80000000, 0, 0],
        },
        gameplay_previous_state: options["previous_phase"].as_i64().unwrap_or(20) as i32,
        gameplay_current_state: 0,
        continuations: vec![],
        occurrences: methods
            .into_iter()
            .enumerate()
            .map(|(i, method)| Occurrence {
                method,
                data: ARENA + 0x12000 + (i % 2) as u64 * 0x1000,
                id: options["id"].as_i64().unwrap_or(-100) as i32,
                clone_source: ARENA + 0x14000 + (i % 2) as u64 * 0x1000,
                clone_result: (options["clone_null"] != true)
                    .then_some(ARENA + 0x38100 + i as u64 * 0x100),
                continuation: ARENA + 0x30100 + i as u64 * 0x100,
                wait: ARENA + 0x34100 + i as u64 * 0x100,
            })
            .collect(),
        services: Services {
            native_runtime_and_bindings_verified: true,
            constructor_allocations_and_empty_storage_verified: true,
            empty_literal_verified: true,
            base_and_all_callbacks_inert: true,
            ui_and_other_services_inert: true,
            unity_liveness_verified: true,
            required_initializer_objects_valid: true,
            clone_results_verified: true,
            synchronous_first_yield_verified: true,
            normal_completion_verified: true,
        },
    }
}

fn native_actor(actor: &Actor, components: &Components, acted: u64, hover: u64) -> Value {
    json!({
        "data": actor.data.unwrap_or(0), "bluff": actor.bluff.unwrap_or(0), "register_as": actor.register_as.unwrap_or(0),
        "trailer": actor.trailer.unwrap_or(0), "runtime": actor.runtime.unwrap_or(0),
        "dead_prefab": actor.dead_prefab.as_ref().map_or(0, |d| d.identity), "revealed": u32::from(actor.revealed),
        "uses": actor.uses as u32, "previous": actor.previous as u32, "state": actor.state as u32,
        "killed_hidden": u32::from(actor.killed_hidden), "killed_demon": u32::from(actor.killed_demon),
        "alignment": actor.alignment as u32, "id": actor.id as u32, "started": u32::from(actor.started),
        "acted_infos": acted, "hover_infos": hover, "role": actor.role.unwrap_or(0),
        "bluff_role": actor.bluff_role.unwrap_or(0), "saved_act": actor.saved_act.unwrap_or(0),
        "act": 1, "statuses": components.statuses_component,
    })
}

fn assert_native_snapshot(
    actor: &Actor,
    components: &Components,
    lists: &[ReferenceList],
    continuations: &[Continuation],
    snapshot: &Value,
    c: &Context,
) {
    assert_eq!(
        native_actor(
            actor,
            components,
            c.constructor.acted_list,
            c.constructor.hover_list
        ),
        snapshot["actor"]
    );
    for (list, native) in lists.iter().zip(snapshot["lists"].as_array().unwrap()) {
        assert_eq!(list.identity, native["identity"].as_u64().unwrap());
        assert_eq!(list.backing_array, native["backing"].as_u64().unwrap());
        assert_eq!(list.count as u64, native["count"].as_u64().unwrap());
        assert_eq!(list.version as u64, native["version"].as_u64().unwrap());
    }
    assert_eq!(lists.len(), snapshot["lists"].as_array().unwrap().len());
    assert_eq!(
        actor.statuses.active.len() as u64,
        snapshot["statuses"][0]["count"]
    );
    assert_eq!(
        actor.statuses.version as u64,
        snapshot["statuses"][0]["version"]
    );
    assert_eq!(
        serde_json::to_value(&components.active_status_slots).unwrap(),
        snapshot["statuses"][0]["backing_values"]
    );
    for object in &components.ui_objects {
        let native = snapshot["controls"]
            .as_array()
            .unwrap()
            .iter()
            .find(|r| r["identity"] == object.identity)
            .unwrap();
        assert_eq!(native["active"], object.active);
    }
    assert_eq!(
        components.ui_objects.len(),
        snapshot["controls"].as_array().unwrap().len()
    );
    let projected: Vec<_> = continuations.iter().map(|i| json!({"identity":i.identity,"actor":i.actor,"state":i.state as u32,"current":i.current.unwrap_or(0)})).collect();
    assert_eq!(
        serde_json::to_value(projected).unwrap(),
        snapshot["continuations"]
    );
}

fn ui_order(out: &Replay, c: &Context) -> Vec<(String, u64, bool)> {
    let mut order = Vec::new();
    for (i, publication) in out.publications.iter().enumerate() {
        let owner = if c.occurrences[i].method == Method::Init {
            "Init"
        } else {
            "InitWithNoReset"
        };
        for event in &publication.initialization.events {
            match event {
                init::Event::HideActed => order.push((
                    format!("{owner}#{}", i + 1),
                    c.components.acted_game_object,
                    false,
                )),
                init::Event::HideRip => {
                    order.push((format!("{owner}#{}", i + 1), c.components.rip, false))
                }
                init::Event::RefreshCharacter => {
                    for event in &publication.refresh_character.events {
                        if let refresh::Event::HidePicked { object, .. } = event {
                            order.push((format!("RefreshCharacter#{}", i + 1), *object, false));
                        }
                    }
                }
                init::Event::RefreshView => {
                    for event in &publication.refresh_view.events {
                        if let view::Event::SetActive { object, active } = event {
                            order.push((format!("RefreshView#{}", i + 1), *object, *active));
                        }
                    }
                }
                _ => {}
            }
        }
    }
    order
}

#[test]
fn matches_all_inert_native_producer_fixtures_and_repeated_reuse() {
    let report: Value = serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_constructor_init.json"))).unwrap();
    assert_eq!(report["case_count"], 148);
    assert_eq!(report["failure_case_count"], 119);
    let mut checked = 0;
    for native in report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report["repeated_sequences"].as_array().unwrap())
        .chain(report["failure_baselines"].as_array().unwrap())
    {
        let c = fixture(&native["input"]);
        let out = replay(&c).unwrap();
        let constructed = &native["stages"][0]["snapshot"];
        assert_native_snapshot(
            &out.constructed.actor,
            &c.components,
            &out.constructed.lists,
            &c.continuations,
            constructed,
            &c,
        );
        for (index, publication) in out.publications.iter().enumerate() {
            let stage = &native["stages"][index + 1]["snapshot"];
            let mut stage_lists = out.constructed.lists.clone();
            stage_lists
                .iter_mut()
                .find(|l| l.identity == c.constructor.acted_list)
                .unwrap()
                .version = publication.list_version_after;
            assert_native_snapshot(
                &publication.initialization.actor,
                &Components {
                    ui_objects: publication.refresh_view.context.ui_objects.clone(),
                    ..c.components.clone()
                },
                &stage_lists,
                &publication.initialization.continuations,
                stage,
                &c,
            );
            assert_eq!(
                publication.refresh_character.context.actor,
                publication.before_refresh
            );
            assert_eq!(
                publication.refresh_view.context.actor,
                publication.before_refresh
            );
            assert_eq!(publication.initialization.wait_seconds_f32_bits, 0x3E99999A);
            let events = &publication.initialization.events;
            let first = events
                .iter()
                .position(|e| *e == init::Event::RefreshCharacter)
                .unwrap();
            assert_eq!(events[first + 1], init::Event::RefreshView);
            assert_eq!(events[first + 2], init::Event::RegisterContinuation);
            assert_eq!(events[first + 3], init::Event::PublishRole);
            assert_eq!(events[first + 4], init::Event::FirstYield);
            if let Some(native_events) = native["events"].as_array() {
                let native_refresh = native_events
                    .iter()
                    .find(|e| {
                        e["phase"] == format!("RefreshCharacter#{}", index + 1)
                            && e["kind"] == "native_refresh_entry"
                    })
                    .unwrap();
                assert_eq!(
                    native_actor(
                        &publication.before_refresh,
                        &c.components,
                        c.constructor.acted_list,
                        c.constructor.hover_list
                    ),
                    native_refresh["snapshot"]["actor"]
                );
                assert_eq!(
                    publication.before_refresh.statuses.active.len() as u64,
                    native_refresh["snapshot"]["statuses"][0]["count"]
                );
                assert_eq!(
                    publication.before_refresh.statuses.version as u64,
                    native_refresh["snapshot"]["statuses"][0]["version"]
                );
                if let Some(observed) = &publication.initialization.callback_observation {
                    let owner = if c.occurrences[index].method == Method::Init {
                        "Init"
                    } else {
                        "InitWithNoReset"
                    };
                    let event = native_events
                        .iter()
                        .find(|e| {
                            e["phase"] == format!("{owner}#{}", index + 1)
                                && e["kind"] == "state_callback"
                        })
                        .unwrap();
                    assert_eq!(
                        native_actor(
                            observed,
                            &c.components,
                            c.constructor.acted_list,
                            c.constructor.hover_list
                        ),
                        event["snapshot"]["actor"]
                    );
                    assert_eq!(
                        observed.statuses.active.len() as u64,
                        event["snapshot"]["statuses"][0]["count"]
                    );
                }
            }
        }
        assert_native_snapshot(
            &out.actor,
            &out.components,
            &out.lists,
            &out.continuations,
            &native["final"],
            &c,
        );
        assert_eq!(out.components.transforms, c.components.transforms);
        if let Some(events) = native["events"].as_array() {
            let expected: Vec<_> = events
                .iter()
                .filter(|e| e["kind"] == "set_active")
                .map(|e| {
                    (
                        e["phase"].as_str().unwrap().to_owned(),
                        e["object"].as_u64().unwrap(),
                        e["active"].as_bool().unwrap(),
                    )
                })
                .collect();
            assert_eq!(ui_order(&out, &c), expected);
        } else {
            let expected: Vec<_> = native["api_order"]
                .as_array()
                .unwrap()
                .iter()
                .filter(|e| e["kind"] == "set_active")
                .map(|e| e["phase"].as_str().unwrap().to_owned())
                .collect();
            assert_eq!(
                ui_order(&out, &c)
                    .into_iter()
                    .map(|e| e.0)
                    .collect::<Vec<_>>(),
                expected
            );
        }
        let native_events = native["events"]
            .as_array()
            .or_else(|| native["api_order"].as_array())
            .unwrap();
        let constructor_apis: Vec<_> = native_events
            .iter()
            .filter(|e| e["phase"] == "Construct" && e["kind"] != "metadata")
            .map(|e| e["kind"].as_str().unwrap())
            .collect();
        let our_constructor_apis: Vec<_> = out
            .constructor_events
            .iter()
            .filter_map(|e| match e {
                ConstructorEvent::AllocateList { .. } => Some("allocate"),
                ConstructorEvent::ConstructEmptyList { .. } => Some("list_constructor"),
                ConstructorEvent::Barrier { .. } => Some("barrier"),
                ConstructorEvent::BaseConstructor { .. } => Some("base_constructor"),
                _ => None,
            })
            .collect();
        assert_eq!(our_constructor_apis, constructor_apis);
        for (i, publication) in out.publications.iter().enumerate() {
            let phase = format!("RefreshView#{}", i + 1);
            let expected: Vec<_> = native_events
                .iter()
                .filter(|e| e["phase"] == phase)
                .map(|e| e["kind"].as_str().unwrap())
                .filter(|kind| !["metadata", "class_init"].contains(kind))
                .collect();
            let mut actual = vec!["native_refresh_entry"];
            actual.extend(publication.refresh_view.events.iter().map(|e| match e {
                view::Event::UnityLive { .. } => "unity_live",
                view::Event::SetActive { .. } => "set_active",
                _ => panic!("unsupported Hidden RefreshView event"),
            }));
            assert_eq!(actual, expected);
        }
        checked += 1;
    }
    assert_eq!(checked, 153);
}

#[test]
fn constructor_resets_sources_in_native_order_and_retains_old_physical_lists() {
    let mut c = fixture(&json!({"sequence":["InitWithNoReset"]}));
    let old = ReferenceList {
        identity: 0xAA01,
        backing_array: 0xAA02,
        count: 1,
        version: u32::MAX,
        slots: vec![Some(0xAA03), None, Some(0xAA04)],
    };
    c.actor.infos = old.slots[..old.count].to_vec();
    c.actor.info_version = old.version;
    c.initial_acted_list = Some(old.identity);
    c.initial_hover_list = Some(old.identity);
    c.retained_lists = vec![old.clone()];
    let out = replay(&c).unwrap();
    assert_eq!(out.lists[0], old);
    assert_eq!(out.constructed.lists[0], old);
    assert_eq!(out.constructed.actor.info_version, 0);
    assert!(out.constructed.actor.infos.is_empty());
    assert_eq!(out.actor.info_version, 1);
    assert_eq!(out.actor.register_as, c.actor.register_as);
    assert_eq!(out.actor.runtime, c.actor.runtime);
    assert_eq!(out.actor.trailer, c.actor.trailer);
    assert_eq!(
        out.constructor_events,
        vec![
            ConstructorEvent::Uses { value: 1 },
            ConstructorEvent::AllocateList {
                list: c.constructor.acted_list
            },
            ConstructorEvent::ConstructEmptyList {
                list: c.constructor.acted_list,
                backing: c.constructor.empty_array
            },
            ConstructorEvent::StoreActed {
                list: c.constructor.acted_list
            },
            ConstructorEvent::Barrier {
                owner: c.actor.identity,
                field: "acted_infos".into(),
                value: c.constructor.acted_list
            },
            ConstructorEvent::AllocateList {
                list: c.constructor.hover_list
            },
            ConstructorEvent::ConstructEmptyList {
                list: c.constructor.hover_list,
                backing: c.constructor.empty_array
            },
            ConstructorEvent::StoreHover {
                list: c.constructor.hover_list
            },
            ConstructorEvent::Barrier {
                owner: c.actor.identity,
                field: "hover_infos".into(),
                value: c.constructor.hover_list
            },
            ConstructorEvent::StoreSavedAct {
                text: c.constructor.empty_string
            },
            ConstructorEvent::Barrier {
                owner: c.actor.identity,
                field: "saved_act".into(),
                value: c.constructor.empty_string
            },
            ConstructorEvent::Act { value: true },
            ConstructorEvent::BaseConstructor {
                actor: c.actor.identity
            },
        ]
    );
    // A pinned empty literal may already occupy the old saved-speech field.
    c.actor.saved_act = Some(c.constructor.empty_string);
    assert!(replay(&c).is_ok());
}

#[test]
fn repeated_alias_controls_and_prior_continuations_survive() {
    let mut c = fixture(&json!({"sequence":["Init","InitWithNoReset","Init"]}));
    c.components.disguise.as_mut().unwrap().identity = c.components.pickable;
    c.components.picked_objects = vec![c.components.acted_game_object; 2];
    c.continuations.push(Continuation {
        identity: 0xBB01,
        actor: c.actor.identity,
        state: 1,
        current: Some(0xBB02),
    });
    let out = replay(&c).unwrap();
    assert_eq!(out.continuations[0], c.continuations[0]);
    assert_eq!(out.continuations.len(), 4);
    assert_eq!(out.actor.info_version, 3);
    assert_eq!(out.lists[1].version, 0);
    assert_eq!(out.lists[0].backing_array, out.lists[1].backing_array);
    assert_eq!(
        out.components.active_status_slots,
        c.components.active_status_slots
    );
    assert!(
        !out.components
            .ui_objects
            .iter()
            .find(|o| o.identity == c.components.pickable)
            .unwrap()
            .active
    );
    for publication in &out.publications {
        assert_eq!(
            publication.before_refresh.role,
            if publication.occurrence == 0 {
                c.actor.role
            } else {
                c.occurrences[publication.occurrence - 1].clone_result
            }
        );
    }
}

#[test]
fn rejects_unverified_mutations_failures_and_physical_collisions_atomically() {
    let base = fixture(&json!({}));
    for name in [
        "native_runtime_and_bindings_verified",
        "constructor_allocations_and_empty_storage_verified",
        "empty_literal_verified",
        "base_and_all_callbacks_inert",
        "ui_and_other_services_inert",
        "unity_liveness_verified",
        "required_initializer_objects_valid",
        "clone_results_verified",
        "synchronous_first_yield_verified",
        "normal_completion_verified",
    ] {
        let mut value = serde_json::to_value(&base).unwrap();
        value["services"][name] = json!(false);
        let c: Context = serde_json::from_value(value).unwrap();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext), "{name}");
    }
    let mut invalid = Vec::new();
    let mut c = base.clone();
    c.constructor.hover_list = c.constructor.acted_list;
    invalid.push(c);
    let mut c = base.clone();
    c.constructor.acted_list = c.components.pickable;
    invalid.push(c);
    let mut c = base.clone();
    c.constructor.empty_array = c.components.active_status_array;
    invalid.push(c);
    let mut c = base.clone();
    c.constructor.empty_string = c.data[0].identity;
    c.actor.saved_act = Some(c.constructor.empty_string);
    invalid.push(c);
    let mut c = base.clone();
    c.constructor.empty_string = c.data[0].source_role;
    c.actor.saved_act = Some(c.constructor.empty_string);
    invalid.push(c);
    let mut c = base.clone();
    c.continuations.push(Continuation {
        identity: 0xDD01,
        actor: c.actor.identity,
        state: 1,
        current: Some(c.constructor.empty_string),
    });
    c.actor.saved_act = Some(c.constructor.empty_string);
    invalid.push(c);
    let mut c = base.clone();
    c.occurrences[0].clone_result = Some(c.constructor.empty_string);
    invalid.push(c);
    let mut c = base.clone();
    c.occurrences[0].wait = c.occurrences[0].continuation;
    invalid.push(c);
    let mut c = base.clone();
    c.occurrences[0].clone_source = c.data[1].source_role;
    invalid.push(c);
    let mut c = base.clone();
    c.components.actor_transform = 0;
    invalid.push(c);
    let mut c = base.clone();
    c.components
        .ui_objects
        .push(c.components.ui_objects[0].clone());
    invalid.push(c);
    let mut c = base.clone();
    c.actor.dead_prefab = Some(ObjectReference {
        identity: c.components.disguise.as_ref().unwrap().identity,
        live: true,
    });
    invalid.push(c);
    let mut c = base.clone();
    c.components.active_status_slots[0] = 99;
    invalid.push(c);
    let mut c = base.clone();
    c.actor.infos.push(None);
    invalid.push(c);
    let mut c = base.clone();
    c.initial_hover_list = Some(999);
    invalid.push(c);
    for callback in [
        base.actor.identity,
        base.components.pickable,
        base.components.active_status_list,
        base.components.active_status_array,
        base.components.icon_transform,
        base.data[0].identity,
        base.data[0].source_role,
    ] {
        let mut c = base.clone();
        c.actor.state_callback = Some(callback);
        invalid.push(c);
    }
    let mut c = base.clone();
    c.retained_lists.push(ReferenceList {
        identity: 0xEE01,
        backing_array: 0xEE02,
        count: 0,
        version: 0,
        slots: vec![Some(0xEE03)],
    });
    c.actor.state_callback = Some(0xEE03);
    invalid.push(c);
    for c in invalid {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
    for field in [
        "effects",
        "failure",
        "callback_state",
        "second_resume",
        "engine_admission",
    ] {
        let mut value = serde_json::to_value(&base).unwrap();
        value[field] = json!({});
        assert!(serde_json::from_value::<Context>(value).is_err());
    }
}

#[test]
fn aggregate_whole_producer_reservation_rejects_before_cloning() {
    let mut c = fixture(&json!({}));
    let repeat = c.occurrences[0].clone();
    c.occurrences = (0..32)
        .map(|i| Occurrence {
            continuation: 0xCC00 + i * 3,
            wait: 0xCC01 + i * 3,
            clone_result: Some(0xCC02 + i * 3),
            ..repeat.clone()
        })
        .collect();
    assert!(replay(&c).is_ok());
    c.components.active_status_slots.resize(4096, 0);
    c.actor.statuses.resistances = vec![1; 4096];
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = fixture(&json!({}));
    c.data.resize(4097, c.data[0].clone());
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}
