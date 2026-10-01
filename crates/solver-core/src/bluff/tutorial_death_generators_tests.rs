use super::*;
use serde_json::Value;

fn make(id: u64, kind: StorageKind, size: usize, fill: u8) -> Storage {
    Storage {
        identity: id,
        kind,
        bytes: vec![fill; size],
    }
}
fn fixture(case: &Value) -> Context {
    let rows = case["death"]["publications"].as_array().unwrap();
    let kill = rows.iter().find(|v| v["kind"] == "kill").unwrap();
    let poison = rows.iter().find(|v| v["kind"] == "poison").unwrap();
    let actor = kill["character"].as_u64().unwrap();
    let controller = kill["controller"].as_u64().unwrap();
    let kill_id = kill["identity"].as_u64().unwrap();
    let poison_id = poison["identity"].as_u64().unwrap();
    let statuses = case["input"]["statuses"].as_array().unwrap();
    let list_id = case["death"]["statuses"]["identity"].as_u64().unwrap();
    let storage: Vec<Storage> = serde_json::from_value(case["initial_storage"].clone()).unwrap();
    let actor_record = storage.iter().find(|o| o.identity == actor).unwrap();
    let icon = pointer(actor_record, 0x20);
    let pivot = case["transform_calls"]
        .as_array()
        .unwrap()
        .first()
        .map(|v| v[1].as_u64().unwrap())
        .unwrap_or(0);
    let gameplay_class = case["gameplay_class"].as_u64().unwrap();
    let status_contains_method = case["status_contains_method"].as_u64().unwrap();
    let mut resumes = Vec::new();
    let gameplay = case["input"]["gameplay_state"].as_i64().unwrap() as i32;
    let mut initialized = case["input"]["class_initialized"].as_bool().unwrap();
    for (i, row) in case["death"]["manual_steps"]
        .as_array()
        .unwrap()
        .iter()
        .enumerate()
    {
        let kind = if row["kind"] == "kill" {
            Kind::Kill
        } else {
            Kind::Poison
        };
        let mut r = Resume {
            routine: row["identity"].as_u64().unwrap(),
            fresh_wait: None,
            class_initializer: None,
            contains: None,
            transform: None,
            show: None,
        };
        if i < 2 {
            let id = row["current"].as_u64().unwrap();
            let mut wait: Storage = serde_json::from_value(
                case["wait_storage"]
                    .as_array()
                    .unwrap()
                    .iter()
                    .find(|w| w["identity"].as_u64() == Some(id))
                    .unwrap()
                    .clone(),
            )
            .unwrap();
            put_word(&mut wait, 0x10, 0);
            r.fresh_wait = Some(wait);
        } else {
            if !initialized {
                r.class_initializer = Some(ClassOutcome {
                    receiver: gameplay_class,
                    initialized_word: 1,
                    retained_gameplay_state: gameplay,
                });
                initialized = true;
            }
            let corrupted = statuses.iter().any(|v| v == 10);
            if gameplay != 50 {
                if kind == Kind::Poison {
                    r.contains = Some(ContainsOutcome {
                        receiver: list_id,
                        status: 10,
                        method_info: status_contains_method,
                        returned: corrupted,
                    });
                }
                if kind == Kind::Kill || corrupted {
                    r.transform = Some(TransformOutcome {
                        receiver: icon,
                        method_info: 0,
                        returned: pivot,
                    });
                    r.show = Some(ShowOutcome {
                        controller,
                        tutorial_type: if kind == Kind::Kill { 100 } else { 45 },
                        pivot,
                        method_info: 0,
                        accepted_normally_with_caller_storage_retained: true,
                    });
                }
            }
        }
        resumes.push(r);
    }
    Context {
        version: TUTORIAL_DEATH_GENERATORS_NATIVE_V1.into(),
        gameplay_class,
        wait_class: pointer(resumes[0].fresh_wait.as_ref().unwrap(), 0),
        status_contains_method,
        state: State {
            storage,
            routines: vec![
                Routine {
                    identity: kill_id,
                    kind: Kind::Kill,
                },
                Routine {
                    identity: poison_id,
                    kind: Kind::Poison,
                },
            ],
            metadata: [u8::from(case["input"]["warm"].as_bool().unwrap()); 3],
        },
        services: Services {
            native_layout_and_metadata_verified: true,
            native_wait_and_base_constructor_verified: true,
            metadata_and_gc_preserve_caller_storage: true,
            no_external_mutation_failure_or_implicit_resume: true,
        },
        resumes,
    }
}
fn corpus() -> Value {
    serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_tutorial_death_show_requests.json")).unwrap()
}

#[test]
fn matches_48_native_normal_profiles_and_exact_show_entries() {
    let corpus = corpus();
    let cases = corpus["cases"].as_array().unwrap();
    assert_eq!(cases.len(), 48);
    for case in cases {
        let c = fixture(case);
        let before = c.state.clone();
        let out = replay(&c).unwrap();
        for (actual, expected) in out
            .steps
            .iter()
            .zip(case["death"]["manual_steps"].as_array().unwrap())
        {
            let o = object(&actual.state, actual.routine, StorageKind::Routine).unwrap();
            assert_eq!(word(o, 0x10) as u64, expected["state"].as_u64().unwrap());
            assert_eq!(pointer(o, 0x18), expected["current"].as_u64().unwrap());
            assert_eq!(
                actual.returned,
                u64::from(expected["returned_bool"].as_u64().unwrap() != 0) != 0
            );
            let kind = actual
                .state
                .routines
                .iter()
                .find(|r| r.identity == actual.routine)
                .unwrap()
                .kind;
            let (co, ch) = kind.offsets();
            assert_eq!(pointer(o, co), expected["controller"].as_u64().unwrap());
            assert_eq!(pointer(o, ch), expected["character"].as_u64().unwrap());
        }
        let requests: Vec<_> = out.steps.iter().filter_map(|v| v.show.as_ref()).collect();
        let native = case["show_entries"].as_array().unwrap();
        assert_eq!(requests.len(), native.len());
        for (r, n) in requests.iter().zip(native) {
            assert_eq!(r.controller, n["controller"].as_u64().unwrap());
            assert_eq!(r.tutorial_type as i64, n["type"].as_i64().unwrap());
            assert_eq!(r.pivot, n["pivot"].as_u64().unwrap());
            assert_eq!(r.method_info, n["method_info"].as_u64().unwrap());
            let caller_step = out
                .steps
                .iter()
                .find(|step| step.show.as_ref() == Some(*r))
                .unwrap();
            for row in n["routines"].as_array().unwrap() {
                let id = row["identity"].as_u64().unwrap();
                let o = object(&caller_step.state, id, StorageKind::Routine).unwrap();
                assert_eq!(word(o, 0x10) as u64, row["state"].as_u64().unwrap());
                assert_eq!(pointer(o, 0x18), row["current"].as_u64().unwrap());
            }
        }
        for w in case["waits"].as_array().unwrap() {
            assert_eq!(
                word(
                    object(
                        &out.final_state,
                        w["identity"].as_u64().unwrap(),
                        StorageKind::Wait
                    )
                    .unwrap(),
                    0x10
                ) as u64,
                w["seconds_bits"].as_u64().unwrap()
            );
        }
        assert_eq!(
            word(
                object(&out.final_state, c.gameplay_class, StorageKind::Class).unwrap(),
                0xe0
            ) as u64,
            case["death"]["class_initialized"].as_u64().unwrap()
        );
        let expected_storage: Vec<Storage> =
            serde_json::from_value(case["final_storage"].clone()).unwrap();
        for o in &expected_storage {
            assert_eq!(
                out.final_state
                    .storage
                    .iter()
                    .find(|v| v.identity == o.identity)
                    .unwrap(),
                o
            );
        }
        for o in &before.storage {
            if !matches!(o.kind, StorageKind::Routine | StorageKind::Class) {
                assert_eq!(
                    out.final_state
                        .storage
                        .iter()
                        .find(|v| v.identity == o.identity)
                        .unwrap(),
                    o
                );
            }
        }
        for w in case["wait_storage"].as_array().unwrap() {
            let expected: Storage = serde_json::from_value(w.clone()).unwrap();
            assert_eq!(
                out.final_state
                    .storage
                    .iter()
                    .find(|v| v.identity == expected.identity)
                    .unwrap(),
                &expected
            );
        }
        for (i, key) in ["0x288c31a", "0x288c31f", "0x288c199"].iter().enumerate() {
            assert_eq!(
                out.final_state.metadata[i] != 0,
                case["death"]["metadata_initialized"][key]
                    .as_bool()
                    .unwrap()
            );
        }
    }
}
#[test]
fn completed_state_retains_current_and_captures() {
    let v = corpus();
    let mut c = fixture(&v["cases"][0]);
    let mut call = c.resumes[2].clone();
    call.class_initializer = None;
    call.transform = None;
    call.show = None;
    c.resumes.push(call);
    let out = replay(&c).unwrap();
    assert!(!out.steps[4].returned);
    assert_eq!(out.steps[3].state, out.steps[4].state);
}
#[test]
fn rejects_aliases_wrong_abi_and_unsupplied_outcomes() {
    let v = corpus();
    let c = fixture(&v["cases"][0]);
    let mut bad = c.clone();
    bad.resumes[0].fresh_wait.as_mut().unwrap().identity = bad.state.storage[0].identity;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.resumes[2].show.as_mut().unwrap().method_info = 1;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.resumes[2]
        .class_initializer
        .as_mut()
        .unwrap()
        .retained_gameplay_state = 50;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.resumes[3].contains.as_mut().unwrap().returned = true;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.services.no_external_mutation_failure_or_implicit_resume = false;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.state.storage[0].bytes.pop();
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
}
#[test]
fn rejects_aggregate_snapshot_work_before_replay() {
    let v = corpus();
    let mut c = fixture(&v["cases"][0]);
    for id in 200..220 {
        let mut a = make(id, StorageKind::Array, 0x20 + 1024 * 4, 0);
        put_pointer(&mut a, 0x18, 1024);
        c.state.storage.push(a);
    }
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}

#[test]
fn rejects_nominal_collisions_negative_list_size_and_false_show_acceptance() {
    let v = corpus();
    let c = fixture(&v["cases"][0]);
    let mut bad = c.clone();
    bad.status_contains_method = bad.state.storage[0].identity;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.wait_class = bad.gameplay_class;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    let list = bad
        .state
        .storage
        .iter_mut()
        .find(|o| o.kind == StorageKind::List)
        .unwrap();
    put_word(list, 0x18, u32::MAX);
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.resumes[2]
        .show
        .as_mut()
        .unwrap()
        .accepted_normally_with_caller_storage_retained = false;
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
    let mut bad = c.clone();
    bad.resumes[0].transform = c.resumes[2].transform.clone();
    assert_eq!(replay(&bad), Err(LedgerError::InvalidContext));
}

#[test]
fn reserves_snapshot_bytes_even_without_excess_backing_slots() {
    let v = corpus();
    let mut c = fixture(&v["cases"][0]);
    for id in 200..300 {
        c.state
            .storage
            .push(make(id, StorageKind::Actor, 0x1b8, 0xa5));
    }
    while c.resumes.len() < 32 {
        let mut call = c.resumes[2].clone();
        call.class_initializer = None;
        call.transform = None;
        call.show = None;
        c.resumes.push(call);
    }
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}

#[test]
fn preserves_nonzero_metadata_bytes_and_class_words() {
    let v = corpus();
    let mut c = fixture(&v["cases"][0]);
    c.state.metadata = [0xfe; 3];
    put_word(object_mut(&mut c.state, c.gameplay_class), 0xe0, 0x80000001);
    c.resumes[2].class_initializer = None;
    let out = replay(&c).unwrap();
    assert_eq!(out.final_state.metadata, [0xfe; 3]);
    assert_eq!(
        word(
            object(&out.final_state, c.gameplay_class, StorageKind::Class).unwrap(),
            0xe0
        ),
        0x80000001
    );
}
