use super::*;
use serde_json::{json, Value};

const ARENA: u64 = 0x3_0000_0000;
const FOREIGN_HEADER: u64 = ARENA + 0x182000;

fn flavor(name: &str) -> Flavor {
    [
        Flavor::Action,
        Flavor::Character,
        Flavor::ShowTutorial,
        Flavor::CloseTutorial,
    ]
    .into_iter()
    .find(|f| f.metadata_name() == name)
    .unwrap()
}

fn operation(name: &str) -> Operation {
    match name {
        "enable" => Operation::Enable,
        "disable" => Operation::Disable,
        _ => panic!("unknown entry"),
    }
}

fn info_id(name: &str) -> u64 {
    ORDER
        .iter()
        .position(|slot| slot.handler() == name)
        .map_or_else(
            || match name {
                "authored_prior_handler" => 0xEFFE0001,
                "authored_trailing_handler" => 0xEFFE0002,
                _ => panic!("unknown authored handler {name}"),
            },
            |index| ARENA + 0x190000 + index as u64 * 0x80,
        )
}

fn runtime(controller: u64) -> Runtime {
    Runtime {
        controller,
        gameplay_class: ARENA + 0x180000,
        static_fields: ARENA + 0x181000,
        headers: [
            Flavor::Action,
            Flavor::Character,
            Flavor::ShowTutorial,
            Flavor::CloseTutorial,
        ]
        .into_iter()
        .enumerate()
        .map(|(i, flavor)| HeaderBinding {
            flavor,
            identity: ARENA + 0x183000 + i as u64 * 0x1000,
        })
        .collect(),
        other_class_headers: vec![FOREIGN_HEADER],
        methods: ORDER
            .into_iter()
            .map(|slot| MethodBinding {
                slot,
                method_info: info_id(slot.handler()),
            })
            .collect(),
    }
}

fn services(normal: bool) -> Services {
    Services {
        runtime_metadata_and_bindings_verified: true,
        allocation_and_constructor_verified: true,
        combine_remove_outcomes_verified: true,
        pointer_cast_outcomes_verified: true,
        bindings_headers_and_other_services_stable: true,
        normal_completion_verified: normal,
        native_cast_failure_boundary_verified: !normal,
    }
}

fn native_delegate(token: u64, native: &Value, options: &Value, r: &Runtime) -> Delegate {
    let flavor = flavor(native["type"].as_str().unwrap());
    let foreign = (options["force_cast"] == true && flavor != Flavor::Action)
        || (options["wrong_plain_type"] == true && flavor == Flavor::Action);
    Delegate {
        identity: token,
        flavor,
        class_header: if foreign {
            FOREIGN_HEADER
        } else {
            header(r, flavor)
        },
        invocations: native["invocations"]
            .as_array()
            .unwrap()
            .iter()
            .map(|i| Invocation {
                target: i["target"].as_u64().unwrap(),
                method_info: info_id(i["method"].as_str().unwrap()),
            })
            .collect(),
    }
}

fn native_context(native: &Value, fields: &Value) -> Context {
    let events = native["events"].as_array().unwrap();
    let constructor = events
        .iter()
        .find(|e| e["kind"] == "registration_delegate_constructor_service")
        .unwrap();
    let r = runtime(constructor["args"][1].as_u64().unwrap());
    let registration = &native["final"]["registration"];
    let nodes = registration["delegates"].as_object().unwrap();
    let mut slots = [None; 29];
    for field in fields.as_array().unwrap() {
        let offset = u32::from_str_radix(
            field["offset"].as_str().unwrap().trim_start_matches("0x"),
            16,
        )
        .unwrap();
        let pointer = registration["fields"][field["name"].as_str().unwrap()]
            .as_u64()
            .unwrap();
        slots[offset as usize / 8] = (pointer != 0).then_some(pointer);
    }
    // First Combine/Remove left input for each field is its actual retained
    // initial pointer; later calls must read the prior published output.
    let mut seen = BTreeSet::new();
    let mut current_slot = 0;
    for event in events {
        match event["kind"].as_str().unwrap() {
            "registration_delegate_constructor_service" => {
                current_slot = ORDER
                    .iter()
                    .position(|slot| slot.handler() == event["args"][3]["method"].as_str().unwrap())
                    .unwrap();
            }
            "registration_combine_service" | "registration_remove_service" => {
                if seen.insert(ORDER[current_slot]) {
                    let left = event["args"][0].as_u64().unwrap();
                    slots[ORDER[current_slot].index()] = (left != 0).then_some(left);
                }
            }
            _ => {}
        }
    }
    // Failure may stop before later fields; their final pointers are initial.
    let initial_ids: BTreeSet<_> = ORDER.into_iter().filter_map(|s| slots[s.index()]).collect();
    let delegates: Vec<_> = initial_ids
        .iter()
        .map(|id| native_delegate(*id, &nodes[&id.to_string()], &native["options"], &r))
        .collect();
    let warm = native["options"]["warm"] == true;
    let state = State {
        slots,
        delegates,
        allocation_order: vec![],
        enable_metadata_byte: u8::from(warm),
        disable_metadata_byte: u8::from(warm),
        class_initialized_word: u32::from(native["options"]["class_cold"] != true),
    };
    let mut known: BTreeSet<_> = state.delegates.iter().map(|d| d.identity).collect();
    let sequence = native["options"]["wiring_sequence"].as_array().unwrap();
    let mut calls = Vec::new();
    let mut cursor = 0;
    for entry in sequence {
        let operation = operation(entry.as_str().unwrap());
        let mut registrations = Vec::new();
        for slot in ORDER {
            while cursor < events.len() && events[cursor]["kind"] == "registration_metadata_service"
            {
                cursor += 1;
            }
            if cursor == events.len() {
                break;
            }
            assert_eq!(events[cursor]["kind"], "registration_allocate_service");
            cursor += 1;
            let ctor = &events[cursor];
            cursor += 1;
            assert_eq!(ctor["kind"], "registration_delegate_constructor_service");
            assert_eq!(ctor["args"][3]["method"], slot.handler());
            assert_eq!(ctor["args"][2].as_u64().unwrap(), method(&r, slot));
            let own = ctor["args"][0].as_u64().unwrap();
            let mut allocation =
                native_delegate(own, &nodes[&own.to_string()], &native["options"], &r);
            allocation.invocations.clear();
            known.insert(own);
            assert_eq!(
                events[cursor]["kind"],
                if operation == Operation::Enable {
                    "registration_combine_service"
                } else {
                    "registration_remove_service"
                }
            );
            cursor += 1;
            let first_kind = events[cursor]["kind"].as_str().unwrap();
            let raw = match first_kind {
                "registration_cast_service" | "registration_cast_failure" => {
                    events[cursor]["args"][0].as_u64().unwrap()
                }
                "registration_barrier_service" => events[cursor]["args"][1].as_u64().unwrap(),
                _ => panic!("unexpected post-combine event {first_kind}"),
            };
            let pointer = (raw != 0).then_some(raw);
            let created = pointer
                .filter(|id| !known.contains(id))
                .map(|id| native_delegate(id, &nodes[&id.to_string()], &native["options"], &r));
            if let Some(d) = &created {
                known.insert(d.identity);
            }
            let mut first_cast = CastOutcome::NotCalled;
            let mut second_cast = CastOutcome::NotCalled;
            let mut stopped = false;
            if events[cursor]["kind"] == "registration_cast_service" {
                cursor += 1;
                if events[cursor]["kind"] == "registration_cast_failure" {
                    first_cast = CastOutcome::Return { pointer: None };
                    cursor += 1;
                    stopped = true;
                } else {
                    first_cast = CastOutcome::Return { pointer };
                    assert_eq!(events[cursor]["kind"], "registration_cast_service");
                    cursor += 1;
                    if events[cursor]["kind"] == "registration_cast_failure" {
                        second_cast = CastOutcome::Return { pointer: None };
                        cursor += 1;
                        stopped = true;
                    } else {
                        second_cast = CastOutcome::Return { pointer };
                    }
                }
            } else if events[cursor]["kind"] == "registration_cast_failure" {
                cursor += 1;
                stopped = true;
            }
            if !stopped {
                assert_eq!(events[cursor]["kind"], "registration_barrier_service");
                cursor += 1;
            }
            registrations.push(Registration {
                slot,
                allocation,
                outcome: ServiceOutcome { pointer, created },
                first_cast,
                second_cast,
            });
            if stopped {
                break;
            }
        }
        calls.push(Call {
            operation,
            registrations,
        });
        if cursor == events.len() {
            break;
        }
    }
    assert_eq!(cursor, events.len());
    let normal = native["returned"] == true;
    Context {
        version: TUTORIAL_EVENT_WIRING_NATIVE_V1.into(),
        runtime: r,
        state,
        calls,
        completion: if normal {
            Completion::Normal
        } else {
            Completion::NativeCastFailure
        },
        services: services(normal),
    }
}

fn registration_value(state: &State, fields: &Value) -> Value {
    let fields: serde_json::Map<String, Value> = fields
        .as_array()
        .unwrap()
        .iter()
        .map(|f| {
            let offset =
                u32::from_str_radix(f["offset"].as_str().unwrap().trim_start_matches("0x"), 16)
                    .unwrap();
            (
                f["name"].as_str().unwrap().into(),
                json!(state.slots[offset as usize / 8].unwrap_or(0)),
            )
        })
        .collect();
    let delegates: serde_json::Map<String, Value> = state
        .delegates
        .iter()
        .map(|d| {
            let invocations: Vec<_> = d
                .invocations
                .iter()
                .map(|i| {
                    let method = ORDER
                        .iter()
                        .find(|slot| info_id(slot.handler()) == i.method_info)
                        .map_or_else(
                            || match i.method_info {
                                0xEFFE0001 => "authored_prior_handler",
                                0xEFFE0002 => "authored_trailing_handler",
                                _ => panic!("unknown method token"),
                            },
                            |slot| slot.handler(),
                        );
                    json!({"target":i.target,"method":method})
                })
                .collect();
            (
                d.identity.to_string(),
                json!({"type":d.flavor.metadata_name(),"invocations":invocations}),
            )
        })
        .collect();
    json!({"fields":fields,"delegates":delegates,
        "metadata_initialized":{"enable":state.enable_metadata_byte!=0,"disable":state.disable_metadata_byte!=0},
        "class_initialized":state.class_initialized_word!=0,"allocation_order":state.allocation_order})
}

fn native_event(event: &Event) -> Option<Value> {
    let (kind, args) = match event {
        Event::Metadata { name } => ("registration_metadata_service", json!([name])),
        Event::Allocate { flavor, .. } => (
            "registration_allocate_service",
            json!([flavor.metadata_name()]),
        ),
        Event::Construct {
            token,
            receiver,
            method_info,
            flavor,
        } => {
            let slot = ORDER
                .into_iter()
                .find(|slot| info_id(slot.handler()) == *method_info)
                .unwrap();
            let slot_rva = match slot {
                Slot::GameStart => 0x271a520,
                Slot::CharacterRevealed => 0x271a410,
                Slot::CharacterInfoRevealed => 0x271a168,
                Slot::CharacterKilled => 0x271a1f0,
                Slot::ShowTutorial => 0x271a300,
                Slot::CloseTutorial => 0x271a278,
                Slot::StartNewLevel => 0x271a388,
            };
            (
                "registration_delegate_constructor_service",
                json!([token,receiver,method_info,
                {"method":slot.handler(),"type":flavor.metadata_name(),"slot":slot_rva}]),
            )
        }
        Event::CombineRemove {
            operation,
            left,
            right,
            ..
        } => (
            if *operation == Operation::Enable {
                "registration_combine_service"
            } else {
                "registration_remove_service"
            },
            json!([left.unwrap_or(0), right]),
        ),
        Event::Cast {
            token, expected, ..
        } => ("registration_cast_service", json!([token, expected])),
        Event::Barrier { slot, pointer } => (
            "registration_barrier_service",
            json!([slot.name(), pointer.unwrap_or(0)]),
        ),
        Event::NativeCastFailure {
            token, expected, ..
        } => ("registration_cast_failure", json!([token, expected])),
        Event::PublishMetadata { .. } | Event::Publish { .. } | Event::HeaderGate { .. } => {
            return None
        }
    };
    Some(json!({"kind":kind,"args":args}))
}

fn native_report() -> &'static Value {
    static REPORT: std::sync::OnceLock<Value> = std::sync::OnceLock::new();
    REPORT.get_or_init(|| {
        let raw = std::fs::read_to_string(concat!(
            env!("CARGO_MANIFEST_DIR"),
            "/../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_tutorial_event_wiring.json"
        )).unwrap();
        serde_json::from_str(&raw).unwrap()
    })
}

#[test]
fn matches_all_normal_and_cast_failure_native_cases_and_retained_steps() {
    let report = native_report();
    assert_eq!(report["case_count"], 228);
    assert_eq!(report["failure_case_count"], 100);
    let fields = &report["event_fields"];
    let mut normal = 0;
    let mut stopped = 0;
    for native in report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .chain(report["failure_baselines"].as_array().unwrap())
    {
        let c = native_context(native, fields);
        let before = c.clone();
        let out = replay(&c).unwrap();
        assert_eq!(c, before);
        assert_eq!(
            registration_value(&out.state, fields),
            native["final"]["registration"]
        );
        assert_eq!(out.calls.len(), native["steps"].as_array().unwrap().len());
        for (call, native) in out.calls.iter().zip(native["steps"].as_array().unwrap()) {
            assert_eq!(call.returned, native["returned"]);
            assert_eq!(registration_value(&call.state, fields), native["state"]);
        }
        let actual: Vec<_> = out
            .steps
            .iter()
            .filter_map(|step| native_event(&step.event))
            .collect();
        assert_eq!(actual, *native["events"].as_array().unwrap());
        if native["returned"] == true {
            normal += 1;
            assert!(out.stopped.is_none());
        } else {
            stopped += 1;
            assert!(out.stopped.is_some());
        }
        assert_eq!(
            out.state.class_initialized_word,
            c.state.class_initialized_word
        );
        for index in 0..29 {
            if !ORDER.iter().any(|s| s.index() == index) {
                assert_eq!(out.state.slots[index], c.state.slots[index]);
            }
        }
    }
    assert_eq!((normal, stopped), (226, 4));
}

fn sample() -> Context {
    let r = native_report();
    native_context(&r["cases"][0], &r["event_fields"])
}

#[test]
fn cold_bytes_exact_headers_and_pointer_width_are_preserved() {
    let mut c = sample();
    c.state.enable_metadata_byte = 0x80;
    c.state.disable_metadata_byte = 0x7F;
    c.state.class_initialized_word = 0xDEADBEEF;
    let out = replay(&c).unwrap();
    assert_eq!(out.state.enable_metadata_byte, 0x80);
    assert_eq!(out.state.disable_metadata_byte, 0x7F);
    assert_eq!(out.state.class_initialized_word, 0xDEADBEEF);
    assert!(!out
        .steps
        .iter()
        .any(|s| matches!(s.event, Event::Metadata { .. })));
    let raw = c.calls[0].registrations[0].outcome.pointer.unwrap();
    assert!(raw > u32::MAX as u64);
    assert_eq!(out.state.slots[Slot::GameStart.index()], Some(raw));
    let header = header(&c.runtime, Flavor::Action);
    let foreign = header.wrapping_add(1u64 << 32);
    c.runtime.other_class_headers.push(foreign);
    c.calls[0].registrations[0].allocation.class_header = foreign;
    c.calls[0].registrations.truncate(1);
    c.completion = Completion::NativeCastFailure;
    c.services = services(false);
    let out = replay(&c).unwrap();
    assert_eq!(
        out.stopped,
        Some((Slot::GameStart, FailurePhase::PlainHeader))
    );
    assert_eq!(
        out.state.slots[Slot::GameStart.index()],
        c.state.slots[Slot::GameStart.index()]
    );
    assert_eq!(out.state.allocation_order, vec![raw]);
    assert_eq!(out.state.delegates[0].invocations.len(), 1);
}

#[test]
fn retained_repeated_enables_and_absent_removes_preserve_supplied_tokens() {
    let report = native_report();
    let cases = report["cases"].as_array().unwrap();
    let native = cases
        .iter()
        .find(|n| {
            n["options"]["wiring_sequence"] == json!(["enable", "enable", "disable"])
                && n["options"]["prior"] == true
                && n["options"]["preexisting_own"] == true
                && n["options"]["trailing_prior"] == true
        })
        .unwrap();
    let c = native_context(native, &report["event_fields"]);
    let out = replay(&c).unwrap();
    for slot in ORDER {
        let node = out
            .state
            .delegates
            .iter()
            .find(|d| Some(d.identity) == out.state.slots[slot.index()])
            .unwrap();
        assert_eq!(
            node.invocations
                .iter()
                .map(|i| i.target)
                .collect::<Vec<_>>(),
            vec![
                0xAFFE0001,
                c.runtime.controller,
                0xAFFE0002,
                c.runtime.controller
            ]
        );
        assert_eq!(
            node.invocations
                .iter()
                .filter(|i| i.method_info == method(&c.runtime, slot))
                .count(),
            2
        );
    }
    assert_eq!(out.state.allocation_order.len(), 21);
    let native = cases
        .iter()
        .find(|n| {
            n["options"]["wiring_sequence"] == json!(["disable"])
                && n["options"]["prior"] == true
                && n["options"]["preexisting_own"] == false
        })
        .unwrap();
    let c = native_context(native, &report["event_fields"]);
    let out = replay(&c).unwrap();
    assert_eq!(out.state.slots, c.state.slots);
    assert_eq!(out.state.allocation_order.len(), 7);
    for initial in &c.state.delegates {
        assert_eq!(
            out.state
                .delegates
                .iter()
                .find(|d| d.identity == initial.identity)
                .unwrap(),
            initial
        );
    }
}

#[test]
fn cast_failure_requires_separate_guard_and_retains_second_cast_publication() {
    let report = native_report();
    let native = report["cases"]
        .as_array()
        .unwrap()
        .iter()
        .find(|n| n["options"]["cast_fail_at"] == 2)
        .unwrap();
    let c = native_context(native, &report["event_fields"]);
    let out = replay(&c).unwrap();
    assert_eq!(
        out.stopped,
        Some((Slot::CharacterRevealed, FailurePhase::SecondGenericCast))
    );
    assert!(out.state.slots[Slot::GameStart.index()].is_some());
    assert!(out.state.slots[Slot::CharacterRevealed.index()].is_some());
    assert!(!out.steps.iter().any(|s| matches!(
        s.event,
        Event::Barrier {
            slot: Slot::CharacterRevealed,
            ..
        }
    )));
    assert!(!out.calls[0].returned);
    let mut bad = c.clone();
    bad.services.native_cast_failure_boundary_verified = false;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c;
    bad.services.normal_completion_verified = true;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
}

#[test]
fn rejects_unknown_services_storage_aliases_cast_tokens_and_aggregate_work() {
    let base = sample();
    for field in [
        "runtime_metadata_and_bindings_verified",
        "allocation_and_constructor_verified",
        "combine_remove_outcomes_verified",
        "pointer_cast_outcomes_verified",
        "bindings_headers_and_other_services_stable",
        "normal_completion_verified",
    ] {
        let mut value = serde_json::to_value(&base).unwrap();
        value["services"][field] = json!(false);
        let c: Context = serde_json::from_value(value).unwrap();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext), "{field}");
    }
    let mut cases = Vec::new();
    let mut c = base.clone();
    c.calls[0].registrations[0].allocation.identity = c.runtime.controller;
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].registrations[1].allocation.identity =
        c.calls[0].registrations[0].allocation.identity;
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].registrations[1].first_cast = CastOutcome::Return {
        pointer: Some(c.runtime.controller),
    };
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].registrations[1].second_cast = CastOutcome::Return { pointer: Some(1) };
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].registrations[1].first_cast = CastOutcome::Return {
        pointer: c.calls[0].registrations[1]
            .outcome
            .pointer
            .map(|p| p & u32::MAX as u64),
    };
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].registrations.swap(0, 1);
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].registrations.pop();
    cases.push(c);
    let mut c = base.clone();
    c.calls[0].registrations[0]
        .allocation
        .invocations
        .push(Invocation {
            target: 1,
            method_info: 1,
        });
    cases.push(c);
    let mut c = base.clone();
    c.runtime.headers[0].identity = c.runtime.static_fields;
    cases.push(c);
    let mut c = base.clone();
    c.state.slots[1] = Some(c.calls[0].registrations[0].allocation.identity);
    cases.push(c);
    for c in cases {
        let before = c.clone();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
        assert_eq!(c, before);
    }
    let mut c = base.clone();
    let node = Delegate {
        identity: 0xBA000000,
        class_header: header(&c.runtime, Flavor::Action),
        flavor: Flavor::Action,
        invocations: vec![
            Invocation {
                target: 1,
                method_info: 1
            };
            MAX_RETAINED / 2
        ],
    };
    c.state.delegates.push(node);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut value = serde_json::to_value(base).unwrap();
    value["clr_multicast_algorithm_verified"] = json!(true);
    assert!(serde_json::from_value::<Context>(value).is_err());
}
