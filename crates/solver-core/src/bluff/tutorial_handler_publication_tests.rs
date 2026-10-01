use super::*;
use serde_json::{json, Value};

fn sample(method: Method) -> Context {
    Context {
        version: TUTORIAL_HANDLER_PUBLICATION_NATIVE_V1.into(),
        controller: Some(10),
        character: Some(11),
        gameplay_class: 12,
        gameplay_static: 13,
        gameplay_instance: Some(14),
        state: State {
            routines: vec![],
            publications: vec![],
            retained: vec![],
            metadata_bytes: [0; 10],
            class_initialized_word: 0,
            current_level: 1,
        },
        services: Services {
            native_bindings_and_layouts_verified: true,
            metadata_and_class_state_verified: true,
            zeroed_fresh_allocations_and_base_verified: true,
            gc_and_retained_storage_inert: true,
            class_initializer_sets_one_and_preserves_level: true,
            registration_acceptance_without_resume_verified: true,
            callbacks_and_gameplay_inert: true,
            normal_completion_verified: true,
        },
        calls: vec![Call {
            method,
            fresh_routines: (0..kinds(method, 1).len())
                .map(|i| 100 + i as u64)
                .collect(),
        }],
    }
}

fn pointer(v: &Value) -> Option<u64> {
    v.as_u64().filter(|v| *v != 0)
}

fn native_events(out: &Replay) -> Vec<Value> {
    out.steps.iter().map(|step|match &step.event {
        Event::Metadata {name}=>json!({"kind":"publication_metadata_service","args":[name]}),
        Event::ClassInitialize {class}=>json!({"kind":"gameplay_class_initialize_service","args":[class]}),
        Event::Allocate {routine_kind,..}=>json!({"kind":"publication_allocate_service","args":[routine_kind.name()]}),
        Event::Barrier {..}=>json!({"kind":"write_barrier_service","args":["stored_reference"]}),
        Event::Register {controller,routine,routine_kind}=>json!({"kind":"publication_start_coroutine_service","args":[controller.unwrap_or(0),routine,routine_kind.name()]}),
    }).collect()
}

#[test]
fn compares_all_48_supported_normal_native_cases() {
    let report:Value=serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_tutorial_handler_publication.json")).unwrap();
    let mut checked = 0;
    for case in report["cases"].as_array().unwrap() {
        let options = &case["options"];
        if !case["returned"].as_bool().unwrap()
            || options.get("publication_actions").is_some()
            || options.get("init_level").is_some()
        {
            continue;
        }
        let method = match case["method"].as_str().unwrap() {
            "start" => Method::Start,
            "info" => Method::Info,
            "killed" => Method::Killed,
            "level" => Method::Level,
            _ => unreachable!(),
        };
        let mut c = sample(method);
        let p = &case["final"]["publication"];
        c.controller = pointer(&p["incoming_controller"]);
        c.character = pointer(&p["incoming_character"]);
        c.gameplay_instance = pointer(&p["gameplay_instance"]);
        c.state.current_level = options["level"].as_i64().unwrap_or(0) as i32;
        c.state.class_initialized_word =
            u32::from(options["class_initialized"].as_bool().unwrap_or(true));
        c.state.metadata_bytes = [u8::from(options["warm"].as_bool().unwrap_or(false)); 10];
        if let Some(event) = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .find(|e| e["kind"] == "gameplay_class_initialize_service")
        {
            c.gameplay_class = event["args"][0].as_u64().unwrap();
        }
        c.calls[0].fresh_routines = p["routines"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| r["identity"].as_u64().unwrap())
            .collect();
        let out = replay(&c).unwrap();
        let expected_events: Vec<_> = case["events"]
            .as_array()
            .unwrap()
            .iter()
            .map(|e| json!({"kind":e["kind"],"args":e["args"]}))
            .collect();
        assert_eq!(
            native_events(&out),
            expected_events,
            "{method:?}: {options}"
        );
        let routines:Vec<_>=out.state.routines.iter().map(|r|json!({"identity":r.identity,"kind":r.kind.name(),"state":r.state as u32,
            "current":r.current.unwrap_or(0),"controller":r.controller.unwrap_or(0),
            "character":r.kind.character_offset().map(|_|r.character.unwrap_or(0)),
            "controller_offset":r.kind.controller_offset(),"character_offset":r.kind.character_offset(),
            "closure":if r.kind.has_closure(){Some(r.closure.unwrap_or(0))}else{None}})).collect();
        assert_eq!(routines, *p["routines"].as_array().unwrap());
        assert_eq!(
            out.state.publications,
            p["calls"]
                .as_array()
                .unwrap()
                .iter()
                .map(|r| r["identity"].as_u64().unwrap())
                .collect::<Vec<_>>()
        );
        assert_eq!(
            out.state.class_initialized_word,
            p["class_initialized"].as_u64().unwrap() as u32
        );
        for (index, rva) in METADATA_RVAS.iter().enumerate() {
            assert_eq!(
                out.state.metadata_bytes[index] != 0,
                p["metadata_initialized"][rva].as_bool().unwrap()
            );
        }
        assert_eq!(
            out.state.current_level as u32,
            p["current_level"].as_u64().unwrap() as u32
        );
        checked += 1;
    }
    assert_eq!(checked, 48);
}

#[test]
fn poison_reverses_offsets_but_keeps_controller_store_first() {
    let c = sample(Method::Killed);
    let out = replay(&c).unwrap();
    let barriers: Vec<_> = out
        .steps
        .iter()
        .filter_map(|s| match s.event {
            Event::Barrier {
                routine,
                offset,
                value,
            } => Some((
                routine,
                offset,
                value,
                s.state.routines.last().unwrap().clone(),
            )),
            _ => None,
        })
        .collect();
    assert_eq!(
        barriers.iter().map(|x| (x.0, x.1, x.2)).collect::<Vec<_>>(),
        vec![
            (100, 0x20, Some(10)),
            (100, 0x28, Some(11)),
            (101, 0x28, Some(10)),
            (101, 0x20, Some(11))
        ]
    );
    assert_eq!(barriers[2].3.character, None);
    assert_eq!(barriers[2].3.controller, Some(10));
    let registration = out
        .steps
        .iter()
        .find(|s| matches!(s.event, Event::Register { routine: 101, .. }))
        .unwrap();
    assert_eq!(registration.state.publications, vec![100]);
}

#[test]
fn repeated_calls_retain_storage_and_existing_routine_states() {
    let mut c = sample(Method::Info);
    c.state.retained = vec![RetainedStorage {
        identity: 10,
        offset: 0,
        bytes: vec![0xA5; 64],
    }];
    c.state.routines = vec![Routine {
        identity: 200,
        kind: Kind::Reveal,
        state: -1,
        current: Some(300),
        controller: Some(10),
        character: None,
        closure: None,
    }];
    c.state.publications = vec![200];
    c.state.metadata_bytes = [0x80; 10];
    c.state.class_initialized_word = 0xDEADBEEF;
    c.calls.push(Call {
        method: Method::Killed,
        fresh_routines: vec![101, 102],
    });
    let out = replay(&c).unwrap();
    assert_eq!(out.state.retained, c.state.retained);
    assert_eq!(out.state.routines[0], c.state.routines[0]);
    assert_eq!(out.state.metadata_bytes, [0x80; 10]);
    assert_eq!(out.state.class_initialized_word, 0xDEADBEEF);
    assert_eq!(out.state.publications, vec![200, 100, 101, 102]);
    assert!(!out.steps.iter().any(|s| matches!(
        s.event,
        Event::Metadata { .. } | Event::ClassInitialize { .. }
    )));
    c.controller = None;
    c.character = None;
    let null = replay(&c).unwrap();
    assert!(null.state.routines[1..]
        .iter()
        .all(|r| r.controller.is_none() && r.character.is_none()));
}

#[test]
fn rejects_wrong_provenance_and_physical_collisions_atomically() {
    let base = sample(Method::Info);
    for id in [0, 10, 11, 12, 13, 14] {
        let mut c = base.clone();
        c.calls[0].fresh_routines[0] = id;
        assert!(replay(&c).is_err());
    }
    let mut c = base.clone();
    c.services.callbacks_and_gameplay_inert = false;
    assert!(replay(&c).is_err());
    c = base.clone();
    c.version = "future".into();
    assert!(replay(&c).is_err());
    c = base.clone();
    c.character = c.controller;
    assert!(replay(&c).is_err());
    c = sample(Method::Level);
    c.gameplay_instance = None;
    assert!(replay(&c).is_err());
    c = base.clone();
    c.calls.push(c.calls[0].clone());
    assert!(replay(&c).is_err());
    c = base.clone();
    c.state.routines.push(Routine {
        identity: 200,
        kind: Kind::CharacterInfo,
        state: 0,
        current: Some(100),
        controller: Some(10),
        character: Some(11),
        closure: None,
    });
    assert!(replay(&c).is_err());
    c = base.clone();
    c.state.routines.push(Routine {
        identity: 200,
        kind: Kind::CharacterInfo,
        state: 0,
        current: None,
        controller: Some(11),
        character: Some(10),
        closure: None,
    });
    assert!(replay(&c).is_err());
    for (identity, start, end) in [
        (12, 0xE0, 0xE4),
        (12, 0xB8, 0xC0),
        (13, 0x10, 0x18),
        (14, 0x78, 0x7C),
    ] {
        for offset in [start - 1, start, end - 1] {
            c = base.clone();
            c.state.retained.push(RetainedStorage {
                identity,
                offset,
                bytes: vec![0; 2],
            });
            assert!(replay(&c).is_err());
        }
        for offset in [start - 2, end] {
            c = base.clone();
            c.state.retained.push(RetainedStorage {
                identity,
                offset,
                bytes: vec![0; 2],
            });
            assert!(replay(&c).is_ok());
        }
    }
    assert_eq!(base.state.routines.len(), 0);
}

#[test]
fn aggregate_snapshot_budget_is_reserved_before_copying_storage() {
    let mut c = sample(Method::Start);
    c.state.retained = vec![RetainedStorage {
        identity: 900,
        offset: 0,
        bytes: vec![0xFE; 1000],
    }];
    assert!(replay(&c).is_ok());
    c.calls = (0..20)
        .map(|i| Call {
            method: Method::Start,
            fresh_routines: (0..4).map(|j| 1000 + i * 4 + j).collect(),
        })
        .collect();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    assert_eq!(c.state.retained[0].bytes, vec![0xFE; 1000]);
}
