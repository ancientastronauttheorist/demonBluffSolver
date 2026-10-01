use super::*;
use serde_json::{json, Value};

fn native_list(row: &Value) -> List {
    List {
        identity: row["identity"].as_u64().unwrap() + 1,
        count: row["count"].as_u64().unwrap() as usize,
        version: row["version"].as_u64().unwrap() as u32,
        backing: serde_json::from_value(row["backing_values"].clone()).unwrap(),
    }
}

fn native_state(row: &Value) -> State {
    State {
        key: serde_json::from_value(row["key"].clone()).unwrap(),
        completed_tutorials: row["completedTutorials"]["identity"]
            .as_u64()
            .map(|id| id + 1),
        unlocked_characters: row["unlockedCharactersId"]["identity"]
            .as_u64()
            .map(|id| id + 1),
        lists: row["allocated_lists"]
            .as_array()
            .unwrap()
            .iter()
            .map(native_list)
            .collect(),
    }
}

fn services() -> Services {
    Services {
        runtime_and_metadata_verified: true,
        initialization_and_callbacks_inert: true,
        nullable_text_contains_verified: true,
        backing_storage_unaliased_verified: true,
        array_clear_zeroes_requested_slots: true,
        constructors_publish_empty_lists: true,
        normal_completion_verified: true,
        resized_capacity: None,
        constructor_list_identities: Vec::new(),
    }
}

fn context_from_native(row: &Value) -> Context {
    let operation = match row["method"].as_str().unwrap() {
        "AddTutorial" => Operation::AddTutorial,
        "AddCharacter" => Operation::AddCharacter,
        "ClearTutorials" => Operation::ClearTutorials,
        "ClearUnlockedCharacters" => Operation::ClearUnlockedCharacters,
        ".ctor" => Operation::Construct,
        _ => panic!("unsupported audited method"),
    };
    let mut state = State {
        key: serde_json::from_value(row["input_state"]["key"].clone()).unwrap(),
        completed_tutorials: None,
        unlocked_characters: None,
        lists: Vec::new(),
    };
    for (ordinal, field) in ["completedTutorials", "unlockedCharactersId"]
        .iter()
        .enumerate()
    {
        if row["input_state"][field].is_null() {
            continue;
        }
        let mut backing: Vec<Option<String>> =
            serde_json::from_value(row["input_state"][field].clone()).unwrap();
        let count = backing.len();
        let capacity = row["options"]["capacity"]
            .as_u64()
            .map(|n| n as usize)
            .unwrap_or(count);
        backing.resize(capacity, None);
        let identity = state.lists.len() as u64 + 1;
        state.lists.push(List {
            identity,
            count,
            version: row["options"]["version"].as_u64().unwrap_or(17) as u32,
            backing: Some(backing),
        });
        if ordinal == 0 {
            state.completed_tutorials = Some(identity);
        } else {
            state.unlocked_characters = Some(identity);
        }
    }
    let mut contract = services();
    if operation == Operation::Construct {
        contract.constructor_list_identities =
            vec![state.lists.len() as u64 + 1, state.lists.len() as u64 + 2];
    }
    if row["events"]
        .as_array()
        .unwrap()
        .iter()
        .any(|e| e["kind"] == "resize_append_service")
    {
        let field = if operation == Operation::AddTutorial {
            "completedTutorials"
        } else {
            "unlockedCharactersId"
        };
        contract.resized_capacity =
            Some(row["final"][field]["capacity"].as_u64().unwrap() as usize);
    }
    Context {
        version: SAVED_GAME_INFO_NATIVE_V1.into(),
        state,
        operation,
        argument: serde_json::from_value(row["argument"].clone()).unwrap(),
        services: contract,
    }
}

fn native_steps(row: &Value) -> Vec<Step> {
    row["events"]
        .as_array()
        .unwrap()
        .iter()
        .filter_map(|e| {
            let args = &e["args"];
            let event = match e["kind"].as_str().unwrap() {
                "metadata_initialize_service" => return None,
                "contains_service" => Event::Contains {
                    list: args[0].as_u64().unwrap() + 1,
                    value: serde_json::from_value(args[1].clone()).unwrap(),
                },
                "resize_append_service" => Event::ResizeAppend {
                    list: args[0].as_u64().unwrap() + 1,
                    value: serde_json::from_value(args[1].clone()).unwrap(),
                },
                "write_barrier_service" => Event::WriteBarrier,
                "array_clear_service" => Event::ArrayClear {
                    count: args[0].as_u64().unwrap() as usize,
                },
                "list_allocate_service" => Event::ListAllocate,
                "list_constructor_service" => Event::ListConstructor {
                    list: args[0].as_u64().unwrap() + 1,
                },
                _ => panic!("unsupported native event"),
            };
            Some(Step {
                event,
                state: native_state(&e["snapshot"]),
            })
        })
        .collect()
}

#[test]
fn normal_native_methods_and_json_callers_match_complete_list_snapshots() {
    let report: Value = serde_json::from_str(include_str!("../../../../reverse_engineering/reports/f530404b0f3f_807de4a83df4_saved_game_info_methods.json")).unwrap();
    let mut compared = 0;
    for row in report["cases"].as_array().unwrap().iter().chain(
        report["joined_cases"]
            .as_array()
            .unwrap()
            .iter()
            .map(|r| &r["native_caller"]),
    ) {
        let c = context_from_native(row);
        if !row["returned"].as_bool().unwrap() {
            assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
            continue;
        }
        let replay = replay(&c).unwrap();
        assert_eq!(
            replay.state,
            native_state(&row["final"]),
            "{} {:?}",
            row["method"],
            row["options"]
        );
        assert_eq!(
            replay.steps,
            native_steps(row),
            "{} {:?}",
            row["method"],
            row["options"]
        );
        compared += 1;
    }
    assert_eq!(compared, 134);
}

fn sample() -> Context {
    Context {
        version: SAVED_GAME_INFO_NATIVE_V1.into(),
        state: State {
            key: Some("retained".into()),
            completed_tutorials: Some(1),
            unlocked_characters: Some(2),
            lists: vec![
                List {
                    identity: 1,
                    count: 1,
                    version: u32::MAX,
                    backing: Some(vec![Some("old".into()), Some("unused".into())]),
                },
                List {
                    identity: 2,
                    count: 0,
                    version: 7,
                    backing: Some(vec![]),
                },
            ],
        },
        operation: Operation::AddTutorial,
        argument: Some("new".into()),
        services: services(),
    }
}

#[test]
fn inline_append_wraps_before_barrier_and_clear_retains_unused_slots() {
    let mut c = sample();
    let result = replay(&c).unwrap();
    assert_eq!(result.state.lists[0].version, 0);
    assert_eq!(
        result.state.lists[0].backing.as_ref().unwrap(),
        &vec![Some("old".into()), Some("new".into())]
    );
    assert_eq!(result.steps.last().unwrap().state, result.state);
    assert_eq!(result.state.lists[1], c.state.lists[1]);
    c.operation = Operation::ClearTutorials;
    let result = replay(&c).unwrap();
    assert_eq!(result.steps[0].event, Event::ArrayClear { count: 1 });
    assert_eq!(result.steps[0].state.lists[0].count, 0);
    assert_eq!(result.steps[0].state.lists[0].version, 0);
    assert_eq!(
        result.steps[0].state.lists[0].backing,
        c.state.lists[0].backing
    );
    assert_eq!(
        result.state.lists[0].backing,
        Some(vec![None, Some("unused".into())])
    );
}

#[test]
fn growth_is_supplied_and_nullable_duplicates_preserve_version() {
    let mut c = sample();
    c.state.lists[0].backing = Some(vec![None]);
    c.argument = None;
    let unchanged = replay(&c).unwrap();
    assert_eq!(unchanged.state, c.state);
    assert_eq!(unchanged.steps.len(), 1);
    c.argument = Some("new".into());
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    c.services.resized_capacity = Some(7);
    let result = replay(&c).unwrap();
    assert_eq!(result.state.lists[0].backing.as_ref().unwrap().len(), 7);
    assert_eq!(
        result.steps[1].event,
        Event::ResizeAppend {
            list: 1,
            value: c.argument.clone()
        }
    );
    assert_eq!(result.steps[1].state.lists[0].count, 1);
    assert_eq!(result.steps[1].state.lists[0].version, 0);
}

#[test]
fn constructor_publishes_fresh_lists_in_order_and_retains_old_records() {
    let mut c = sample();
    c.operation = Operation::Construct;
    c.services.constructor_list_identities = vec![3, 4];
    let result = replay(&c).unwrap();
    assert_eq!(result.state.key.as_deref(), Some("Tutorials"));
    assert_eq!(&result.state.lists[..2], &c.state.lists);
    assert_eq!(result.state.completed_tutorials, Some(3));
    assert_eq!(result.state.unlocked_characters, Some(4));
    assert_eq!(result.steps.len(), 7);
    assert_eq!(result.steps[2].state.lists[2].backing, None);
    assert_eq!(result.steps[2].state.completed_tutorials, Some(1));
    assert_eq!(result.steps[3].state.completed_tutorials, Some(3));
    assert_eq!(result.steps[3].state.unlocked_characters, Some(2));
    assert_eq!(result.steps[5].state.lists[3].backing, None);
}

#[test]
fn provenance_shapes_capacity_and_unknown_fields_fail_closed() {
    for flag in [
        "runtime_and_metadata_verified",
        "initialization_and_callbacks_inert",
        "nullable_text_contains_verified",
        "backing_storage_unaliased_verified",
        "array_clear_zeroes_requested_slots",
        "constructors_publish_empty_lists",
        "normal_completion_verified",
    ] {
        let mut value = serde_json::to_value(sample()).unwrap();
        value["services"][flag] = json!(false);
        let c: Context = serde_json::from_value(value).unwrap();
        assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    }
    let mut c = sample();
    c.state.lists[0].count = 3;
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    c = sample();
    c.state.lists[1].identity = 1;
    assert_eq!(replay(&c), Err(LedgerError::InvalidContext));
    c = sample();
    c.services.resized_capacity = Some(257);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    c = sample();
    c.state.lists[0].backing = Some(vec![None; 257]);
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    let mut value = serde_json::to_value(sample()).unwrap();
    value["unexpected"] = json!(true);
    assert!(serde_json::from_value::<Context>(value).is_err());
}

#[test]
fn aggregate_budgets_cover_retained_text_and_supplied_growth() {
    let mut c = sample();
    c.state.lists = (1..=17)
        .map(|identity| List {
            identity,
            count: 0,
            version: 0,
            backing: Some(vec![None; 256]),
        })
        .collect();
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    c.state.lists.pop();
    c.state.lists[0].count = 256;
    c.state.lists[0].backing.as_mut().unwrap().truncate(255);
    c.state.lists[0].count = 255;
    c.services.resized_capacity = Some(256);
    // Exactly 4096 slots after supplied growth is permitted.
    assert!(replay(&c).is_ok());
    c.state.lists.push(List {
        identity: 17,
        count: 0,
        version: 0,
        backing: Some(vec![None]),
    });
    assert_eq!(replay(&c), Err(LedgerError::Capacity));

    c = sample();
    c.state.key = None;
    c.argument = None;
    c.state.lists[0].backing = Some(vec![Some("x".repeat(4096)); 256]);
    assert!(replay(&c).is_ok());
    c.argument = Some("x".into());
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    c.argument = None;
    c.operation = Operation::Construct;
    c.services.constructor_list_identities = vec![3, 4];
    // The constructor's fixed key must fit even when the old key was null.
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
    c.state.lists[0].backing.as_mut().unwrap()[255] = Some("x".repeat(4096 - "Tutorials".len()));
    let result = replay(&c).unwrap();
    let retained_text: usize = result
        .state
        .lists
        .iter()
        .flat_map(|l| l.backing.as_ref().unwrap().iter().flatten())
        .map(String::len)
        .sum();
    assert_eq!(
        retained_text + result.state.key.unwrap().len(),
        MAX_TOTAL_TEXT_BYTES
    );
    c.state.lists[0].backing.as_mut().unwrap()[255]
        .as_mut()
        .unwrap()
        .push('x');
    assert_eq!(replay(&c), Err(LedgerError::Capacity));
}
