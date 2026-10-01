use super::super::character_initialization::{ObjectReference, Statuses};
use super::*;
use serde_json::{json, Value};

const ARENA: u64 = 0x3_0000_0000;

fn method(name: &str) -> Method {
    match name {
        "DelayReveal" => Method::DelayReveal,
        "DelayedDemonKill" => Method::DelayedDemonKill,
        "ShowActed" => Method::ShowActed,
        "ShowInfoDelayed" => Method::ShowInfoDelayed,
        "ShowTrailerAct" => Method::ShowTrailerAct,
        "HideActed" => Method::HideActed,
        _ => panic!("method"),
    }
}

fn id(label: &str) -> u64 {
    ARENA
        + match label {
            "actor" => 0x11000,
            "role" => 0x12000,
            "info" => 0x14000,
            "prior_info" => 0x64000,
            "acted" => 0x65000,
            "version" => 0x66000,
            "rect" => 0x6A000,
            "game" => 0x6B000,
            "data" => 0x6D000,
            "original" => 0xA0000,
            "empty" => 0xA0200,
            "old" => 0xA0300,
            _ => {
                let prefix = ["reveal_factory", "kill_factory", "iterator", "speech"]
                    .into_iter()
                    .find(|p| label.starts_with(p))
                    .unwrap();
                0x30000 + label[prefix.len()..].parse::<u64>().unwrap() * 0x100
            }
        }
}

fn label(pointer: Option<u64>, calls: &[Call]) -> Value {
    let Some(pointer) = pointer else {
        return Value::Null;
    };
    for name in [
        "actor",
        "role",
        "info",
        "prior_info",
        "acted",
        "version",
        "rect",
        "game",
        "data",
        "original",
        "empty",
        "old",
    ] {
        if pointer == id(name) {
            return json!(name);
        }
    }
    let (index, call) = calls
        .iter()
        .filter(|c| c.fresh_iterator.is_some())
        .enumerate()
        .find(|(_, c)| c.fresh_iterator == Some(pointer))
        .unwrap();
    json!(format!(
        "{}{}",
        match call.method {
            Method::DelayReveal => "reveal_factory",
            Method::DelayedDemonKill => "kill_factory",
            Method::ShowActed => "iterator",
            Method::ShowInfoDelayed => "speech",
            _ => panic!("iterator"),
        },
        index
    ))
}

fn fixture(rows: &[&Value]) -> Context {
    let first = &rows[0]["options"];
    let actor = Actor {
        identity: id("actor"),
        data: Some(id("data")),
        bluff: None,
        register_as: None,
        trailer: None,
        runtime: None,
        dead_prefab: None,
        revealed: false,
        uses: 1,
        previous: 0,
        state: 10,
        killed_hidden: false,
        killed_demon: false,
        alignment: 0,
        id: 2,
        started: false,
        role: None,
        bluff_role: None,
        saved_act: Some(id("old")),
        infos: vec![Some(id("prior_info"))],
        info_version: 9,
        statuses: Statuses {
            active: vec![],
            version: 0,
            resistances: vec![],
            target: None,
        },
        state_callback: None,
    };
    let calls = rows
        .iter()
        .enumerate()
        .map(|(index, row)| {
            let m = method(row["method"].as_str().unwrap());
            let o = &row["options"];
            let arg = if o["null_argument"] == true || m.argument_kind().is_none() {
                None
            } else {
                Some(id(match m {
                    Method::DelayedDemonKill => "role",
                    Method::ShowActed => "info",
                    _ => {
                        if o["empty_argument"] == true {
                            "empty"
                        } else {
                            "original"
                        }
                    }
                }))
            };
            Call {
                method: m,
                owner: if o["null_actor"] == true {
                    None
                } else {
                    Some(id("actor"))
                },
                argument: arg,
                trigger_bits: if m == Method::ShowActed {
                    o["trigger"].as_u64().unwrap_or(30) as u32
                } else {
                    0
                },
                delay_bits: if m == Method::ShowActed {
                    o["delay_bits"].as_u64().unwrap_or(0) as u32
                } else {
                    0
                },
                fresh_iterator: m
                    .factory()
                    .then_some(ARENA + 0x30000 + index as u64 * 0x100),
                registration_result: (m == Method::ShowActed)
                    .then_some(ARENA + 0xB0000 + index as u64 * 0x100),
            }
        })
        .collect();
    Context {
        version: CHARACTER_PUBLICATION_ENTRIES_NATIVE_V1.into(),
        state: State {
            actor,
            ui_objects: vec![UiObject {
                identity: id("game"),
                active: false,
            }],
            iterators: vec![],
            registrations: vec![],
            shown: vec![],
            layout_rebuilds: vec![],
        },
        components: Components {
            acted_component: (first["null_acteds"] != true).then_some(id("acted")),
            game_object: Some(id("game")),
            version: (first["null_version"] != true).then_some(id("version")),
            layout_array: (first["null_layouts"] != true).then_some(ARENA + 0x69000),
            layouts: if first["null_layouts"] == true {
                vec![]
            } else {
                vec![id("rect"); 2]
            },
        },
        arguments: [
            ("role", ArgumentKind::Character),
            ("info", ArgumentKind::ActedInfo),
            ("prior_info", ArgumentKind::ActedInfo),
            ("original", ArgumentKind::String),
            ("empty", ArgumentKind::String),
            ("old", ArgumentKind::String),
        ]
        .into_iter()
        .map(|(name, kind)| Argument {
            identity: id(name),
            kind,
        })
        .collect(),
        calls,
        services: Services {
            native_runtime_and_bindings_verified: true,
            metadata_and_class_state_verified: true,
            zeroed_fresh_allocations_verified: true,
            empty_base_body_verified: true,
            arguments_and_ui_bindings_verified: true,
            gc_and_other_services_inert: true,
            registration_without_resume_verified: true,
            fresh_registration_results_verified: true,
            ui_services_and_callbacks_inert: true,
            normal_completion_verified: true,
        },
    }
}

fn iterator_value(it: &Iterator, calls: &[Call]) -> Value {
    if it.method == Method::ShowActed {
        json!({"id":label(Some(it.identity),calls),"state":it.state as u32,"current":label(it.current,calls),
            "delay_bits":it.delay_bits,"owner":label(it.owner,calls),"info":label(it.argument,calls),"trigger":it.trigger_bits})
    } else {
        let mut value = json!({"id":label(Some(it.identity),calls),"state":it.state as u32,"current":label(it.current,calls),"owner":label(it.owner,calls)});
        if it.method != Method::DelayReveal {
            value["argument"] = label(it.argument, calls);
        }
        value
    }
}

fn assert_snapshot(state: &State, native: &Value, c: &Context) {
    assert_eq!(state.actor, c.state.actor);
    let it: Vec<_> = state
        .iterators
        .iter()
        .map(|it| iterator_value(it, &c.calls))
        .collect();
    assert_eq!(json!(it), native["factory_objects"]);
    let registered: Vec<_> = state
        .registrations
        .iter()
        .map(|reg| {
            iterator_value(
                state
                    .iterators
                    .iter()
                    .find(|it| it.identity == reg.iterator)
                    .unwrap(),
                &c.calls,
            )
        })
        .collect();
    assert_eq!(json!(registered), native["entry_registered"]);
    assert_eq!(state.actor.uses as u32, native["uses_bits"]);
    assert_eq!(state.actor.info_version, native["history_version"]);
    assert_eq!(
        label(state.actor.saved_act, &c.calls),
        native["saved_speech"]
    );
    assert_eq!(
        json!(state
            .actor
            .infos
            .iter()
            .map(|p| label(*p, &c.calls))
            .collect::<Vec<_>>()),
        native["history"]
    );
    assert_eq!(
        json!(state
            .shown
            .iter()
            .map(|s| label(s.text, &c.calls))
            .collect::<Vec<_>>()),
        native["shown"]
    );
    assert_eq!(
        label(c.components.acted_component, &c.calls),
        native["acted_reference"]
    );
    for (name, active) in native["active"].as_object().unwrap() {
        assert_eq!(
            state
                .ui_objects
                .iter()
                .find(|o| o.identity == id(name))
                .unwrap()
                .active,
            active.as_bool().unwrap()
        );
    }
}

fn api(step: &Step, c: &Context) -> Option<Value> {
    let (kind, args) = match &step.event {
        Event::Allocate { method, .. } => (
            "allocate_service",
            json!([match method {
                Method::DelayReveal => "reveal_factory",
                Method::DelayedDemonKill => "kill_factory",
                Method::ShowActed => "iterator",
                Method::ShowInfoDelayed => "speech",
                _ => panic!(),
            }]),
        ),
        Event::Barrier {
            iterator,
            offset,
            value,
        } => (
            "barrier_service",
            json!([
                iterator - ARENA + u64::from(*offset),
                label(*value, &c.calls)
            ]),
        ),
        Event::Register { iterator, .. } => (
            "entry_coroutine_registration_service",
            json!([iterator_value(
                step.state
                    .iterators
                    .iter()
                    .find(|it| it.identity == *iterator)
                    .unwrap(),
                &c.calls
            )]),
        ),
        Event::GetGameObject { .. } => ("game_object_service", json!([])),
        Event::SetActive { object, active } => (
            "set_active_service",
            json!([label(Some(*object), &c.calls), active]),
        ),
        Event::ShowVersion { text, .. } => {
            ("show_version_service", json!([label(*text, &c.calls)]))
        }
        Event::RebuildLayout { object } => {
            ("layout_service", json!([label(Some(*object), &c.calls)]))
        }
        _ => return None,
    };
    Some(json!({"kind":kind,"args":args}))
}

#[test]
fn matches_supported_normal_native_profiles_service_snapshots_and_repeated_calls() {
    let r:Value=serde_json::from_str(include_str!(concat!(env!("CARGO_MANIFEST_DIR"),
        "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_character_publication_entries.json"))).unwrap();
    assert_eq!(r["case_count"], 112);
    assert_eq!(r["failure_case_count"], 25);
    let mut profiles: Vec<Vec<&Value>> = r["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(r["failure_baselines"].as_array().unwrap())
        .filter(|c| c["returned"] == true)
        .map(|c| vec![c])
        .collect();
    profiles.extend(
        r["retained_sequences"]
            .as_array()
            .unwrap()
            .iter()
            .map(|s| s["calls"].as_array().unwrap().iter().collect::<Vec<_>>()),
    );
    assert_eq!(profiles.len(), 116);
    for rows in profiles {
        let c = fixture(&rows);
        let before = c.clone();
        let out = replay(&c).unwrap();
        assert_eq!(c, before);
        for (index, (call, native)) in out.calls.iter().zip(&rows).enumerate() {
            assert_snapshot(&call.state, &native["final"], &c);
            assert_eq!(label(call.returned_iterator, &c.calls), native["result"]);
            let mut start = 0;
            if index > 0 {
                start = rows[index - 1]["events"].as_array().unwrap().len();
            }
            let expected: Vec<_> = native["events"].as_array().unwrap()[start..]
                .iter()
                .filter(|e| {
                    !["metadata_service", "class_initialization_service"]
                        .contains(&e["kind"].as_str().unwrap())
                })
                .collect();
            let ours: Vec<_> = out
                .steps
                .iter()
                .filter(|s| s.call == index)
                .filter_map(|s| api(s, &c).map(|e| (s, e)))
                .collect();
            assert_eq!(ours.len(), expected.len());
            for ((step, event), native) in ours.iter().zip(expected) {
                assert_eq!(event, &json!({"kind":native["kind"],"args":native["args"]}));
                assert_snapshot(&step.state, &native["snapshot"], &c);
            }
            assert!(call
                .state
                .iterators
                .iter()
                .all(|it| it.state == 0 && it.current.is_none()));
        }
        assert_eq!(
            out.state
                .layout_rebuilds
                .iter()
                .filter(|pointer| **pointer == id("rect"))
                .count(),
            rows.iter()
                .filter(|r| r["method"] == "ShowTrailerAct")
                .count()
                * 2
        );
    }
}

fn sample() -> Context {
    fixture(&[&json!({"method":"ShowActed","options":{}})])
}

#[test]
fn capture_only_nulls_self_character_empty_text_and_exact_float_bits() {
    let mut c = sample();
    c.calls = vec![
        Call {
            method: Method::DelayedDemonKill,
            owner: None,
            argument: Some(c.state.actor.identity),
            trigger_bits: 0,
            delay_bits: 0,
            fresh_iterator: Some(0xCC01),
            registration_result: None,
        },
        Call {
            method: Method::ShowInfoDelayed,
            owner: None,
            argument: Some(id("empty")),
            trigger_bits: 0,
            delay_bits: 0,
            fresh_iterator: Some(0xCC02),
            registration_result: None,
        },
    ];
    let out = replay(&c).unwrap();
    assert_eq!(
        out.state.iterators[0].argument,
        Some(c.state.actor.identity)
    );
    assert_eq!(out.state.iterators[1].argument, Some(id("empty")));
    assert!(out.state.iterators.iter().all(|it| it.owner.is_none()));
    for bits in [
        0,
        0x80000000,
        0x3E99999A,
        0x3ECCCCCD,
        0x7FC01234,
        0x7F800000,
        u32::MAX,
    ] {
        let mut c = sample();
        c.calls[0].delay_bits = bits;
        c.calls[0].trigger_bits = bits;
        let out = replay(&c).unwrap();
        assert_eq!(out.state.iterators[0].delay_bits, bits);
        assert_eq!(out.state.iterators[0].trigger_bits, bits);
        let argument = out
            .steps
            .iter()
            .find(|s| matches!(s.event, Event::Barrier { offset: 0x30, .. }))
            .unwrap();
        assert_eq!(argument.state.iterators[0].delay_bits, bits);
        assert_eq!(argument.state.iterators[0].trigger_bits, 0);
    }
}

#[test]
fn retains_full_actor_existing_iterators_and_alias_layout_occurrences() {
    let mut c = fixture(&[
        &json!({"method":"ShowTrailerAct","options":{}}),
        &json!({"method":"HideActed","options":{}}),
    ]);
    c.state.actor.dead_prefab = Some(ObjectReference {
        identity: 0xDD01,
        live: false,
    });
    c.state.actor.statuses.active = vec![1, 2, 3];
    c.state.actor.statuses.version = u32::MAX;
    c.state.actor.statuses.resistances = vec![10, 20];
    c.state.actor.statuses.target = Some(id("role"));
    c.state.actor.runtime = Some(0xDD02);
    c.state.actor.started = true;
    c.state.actor.revealed = true;
    c.state.actor.state_callback = Some(0xDD03);
    c.state.actor.uses = i32::MIN;
    let old = Iterator {
        identity: 0xDD04,
        method: Method::ShowActed,
        state: 1,
        current: Some(0xDD05),
        owner: Some(id("actor")),
        argument: Some(id("info")),
        delay_bits: 0x7FC01234,
        trigger_bits: u32::MAX,
    };
    c.state.iterators.push(old.clone());
    c.state.registrations.push(Registration {
        iterator: old.identity,
        result: 0xDD06,
    });
    let out = replay(&c).unwrap();
    assert_eq!(out.state.actor, c.state.actor);
    assert_eq!(out.state.iterators, vec![old]);
    assert_eq!(out.state.registrations, c.state.registrations);
    assert_eq!(out.state.layout_rebuilds, vec![id("rect"); 2]);
    assert!(!out.state.ui_objects[0].active);
}

#[test]
fn rejects_wrong_types_fresh_aliases_missing_ui_and_unsupported_provenance_atomically() {
    let base = sample();
    for flag in [
        "native_runtime_and_bindings_verified",
        "metadata_and_class_state_verified",
        "zeroed_fresh_allocations_verified",
        "empty_base_body_verified",
        "arguments_and_ui_bindings_verified",
        "gc_and_other_services_inert",
        "registration_without_resume_verified",
        "fresh_registration_results_verified",
        "ui_services_and_callbacks_inert",
        "normal_completion_verified",
    ] {
        let mut value = serde_json::to_value(&base).unwrap();
        value["services"][flag] = json!(false);
        let c: Context = serde_json::from_value(value).unwrap();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    let mut cases = vec![];
    for pointer in [
        id("actor"),
        id("info"),
        id("old"),
        id("game"),
        id("rect"),
        base.components.layout_array.unwrap(),
        base.calls[0].registration_result.unwrap(),
    ] {
        let mut c = base.clone();
        c.calls[0].fresh_iterator = Some(pointer);
        cases.push(c);
    }
    let mut c = base.clone();
    c.calls[0].registration_result = c.calls[0].fresh_iterator;
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].argument = Some(id("original"));
    cases.push(c);
    for (name, wrong) in [
        ("DelayedDemonKill", "info"),
        ("ShowInfoDelayed", "role"),
        ("ShowTrailerAct", "info"),
    ] {
        let mut c = fixture(&[&json!({"method":name,"options":{}})]);
        c.calls[0].argument = Some(id(wrong));
        cases.push(c);
    }
    let mut c = base.clone();
    c.calls[0].owner = None;
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].owner = Some(id("role"));
    cases.push(c);
    let mut c = base.clone();
    c.arguments.push(Argument {
        identity: id("game"),
        kind: ArgumentKind::String,
    });
    cases.push(c);
    let mut c = base.clone();
    c.state.actor.runtime = Some(0xFF01);
    c.calls[0].fresh_iterator = Some(0xFF01);
    cases.push(c);
    let mut c = base.clone();
    c.calls.push(c.calls[0].clone());
    cases.push(c);
    let mut c = base.clone();
    c.state.iterators.push(Iterator {
        identity: 0xFF11,
        method: Method::ShowActed,
        state: 1,
        current: Some(0xFF12),
        owner: Some(id("actor")),
        argument: Some(id("info")),
        delay_bits: 0,
        trigger_bits: 0,
    });
    c.calls[0].registration_result = Some(0xFF12);
    cases.push(c);
    let mut c = fixture(&[&json!({"method":"ShowTrailerAct","options":{}})]);
    c.components.version = None;
    cases.push(c);
    let mut c = fixture(&[&json!({"method":"HideActed","options":{}})]);
    c.components.game_object = None;
    cases.push(c);
    for c in cases {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
    for field in [
        "effects",
        "failure",
        "callback",
        "second_resume",
        "scheduling",
        "multicast",
    ] {
        let mut value = serde_json::to_value(&base).unwrap();
        value[field] = json!({});
        assert!(serde_json::from_value::<Context>(value).is_err());
    }
}

#[test]
fn reserves_aggregate_snapshots_before_cloning_and_accepts_unused_hide_bindings() {
    let mut c = fixture(&[&json!({"method":"HideActed","options":{}})]);
    c.components.version = None;
    c.components.layout_array = None;
    c.components.layouts.clear();
    assert!(replay(&c).is_ok());
    let mut c = sample();
    let one = c.calls[0].clone();
    c.calls = (0..32)
        .map(|i| Call {
            fresh_iterator: Some(0xABC00 + i * 2),
            registration_result: Some(0xABC01 + i * 2),
            ..one.clone()
        })
        .collect();
    assert!(replay(&c).is_ok());
    c.state.actor.statuses.resistances = vec![1; MAX_RETAINED / 2];
    let before = c.clone();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c, before);
    let mut c = fixture(&[&json!({"method":"ShowTrailerAct","options":{}})]);
    c.components.layouts = vec![id("rect"); 4096];
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}

#[test]
fn rejects_incompatible_nominal_types_in_all_retained_actor_reference_fields() {
    let base = sample();
    let typed_ui = [
        id("actor"),
        id("game"),
        id("acted"),
        id("version"),
        base.components.layout_array.unwrap(),
        id("rect"),
    ];
    for field in [
        "data",
        "bluff",
        "register_as",
        "trailer",
        "runtime",
        "role",
        "bluff_role",
        "saved_act",
        "state_callback",
        "infos",
        "target",
        "dead_prefab",
    ] {
        for pointer in typed_ui {
            if (field == "target" && pointer == id("actor"))
                || (field == "dead_prefab" && pointer == id("game"))
            {
                continue;
            }
            let mut c = base.clone();
            match field {
                "data" => c.state.actor.data = Some(pointer),
                "bluff" => c.state.actor.bluff = Some(pointer),
                "register_as" => c.state.actor.register_as = Some(pointer),
                "trailer" => c.state.actor.trailer = Some(pointer),
                "runtime" => c.state.actor.runtime = Some(pointer),
                "role" => c.state.actor.role = Some(pointer),
                "bluff_role" => c.state.actor.bluff_role = Some(pointer),
                "saved_act" => c.state.actor.saved_act = Some(pointer),
                "state_callback" => c.state.actor.state_callback = Some(pointer),
                "infos" => c.state.actor.infos[0] = Some(pointer),
                "target" => c.state.actor.statuses.target = Some(pointer),
                "dead_prefab" => {
                    c.state.actor.dead_prefab = Some(ObjectReference {
                        identity: pointer,
                        live: true,
                    })
                }
                _ => unreachable!(),
            }
            let before = c.clone();
            assert_eq!(
                replay(&c),
                Err(LedgerError::InvalidContext),
                "{field} {pointer:x}"
            );
            assert_eq!(c, before);
        }
    }
    // Existing Actor fields also declare nominal types without an Argument record.
    let mut c = base.clone();
    c.state.actor.trailer = c.state.actor.data;
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    let mut c = base.clone();
    c.state.actor.runtime = c.state.actor.saved_act;
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    let mut c = base;
    c.state.actor.state_callback = c.state.actor.role.or(c.state.actor.data);
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
}

#[test]
fn preserves_compatible_assets_game_objects_character_targets_and_repeated_layouts() {
    let mut c = fixture(&[&json!({"method":"ShowTrailerAct","options":{}})]);
    c.state.actor.bluff = c.state.actor.data;
    c.state.actor.register_as = c.state.actor.data;
    c.state.actor.role = Some(0xDD21);
    c.state.actor.bluff_role = c.state.actor.role;
    c.state.actor.statuses.target = Some(id("actor"));
    c.state.actor.dead_prefab = Some(ObjectReference {
        identity: id("game"),
        live: true,
    });
    let out = replay(&c).unwrap();
    assert_eq!(out.state.actor, c.state.actor);
    assert_eq!(out.state.layout_rebuilds, vec![id("rect"); 2]);
    c.state.actor.statuses.target = Some(id("role"));
    assert!(replay(&c).is_ok());
}
